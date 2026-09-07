#!/usr/bin/env python3
"""EXP-MARGUTIL01 evaluation: two separate, never-conflated measurements.

A. Teacher-forced utility prediction (given the oracle prefix, did the model
   learn the utility function itself?): Spearman/Pearson between u_hat and
   the raw dense oracle utility u_teacher=-A_weighted(S*_{t-1}+{i}), and
   selected-candidate regret, computed over the FULL candidate population
   but a QUERY subsample (--utility_eval_queries) for tractability -- never
   a candidate subsample; the full-memory invariant is about candidates, not
   about how many queries a correlation diagnostic averages over.

B. Free-running sequential retrieval (did the learned utility translate into
   a good Top-K on its own, without oracle help?): the model's own argmax
   picks, at FULL test-set scale, injected into B0's UNCHANGED Stage-2 via
   `RelationStage2.set_forced_selection` -- the same production mechanism
   EXP-1/EXP-2/EXP-SEQFULL01/EXP-SEQDIAG01 already use. Only Top-K
   membership differs; B0's own weighting/gate/fusion/base-forecaster are
   untouched.

Also runs B0's own (unforced) selection through the identical Stage-2 loop
first, as the fingerprint check this project's protocol requires.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp.exp_stage1_relation import Exp_Stage1_Relation
from models.DenseUtilityRetriever import AsymmetricUtilityHead, UtilityHead
from models.SequentialSetRetriever import EmptySetToken, SetConditioner
from scripts.train_margutil01 import run_sequence_dense
from utils.dense_utility import candidate_weights, dense_utility
from utils.retrieval_diagnostics import load_stage2


def load_trained_selector(seq_ckpt_path, device):
    ckpt = torch.load(seq_ckpt_path, map_location='cpu')
    args = SimpleNamespace(**ckpt['args'])
    exp = Exp_Stage1_Relation(args)
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(device)
    d_model = int(args.d_model)
    set_conditioner = SetConditioner(d_model).to(device)
    set_conditioner.load_state_dict(ckpt['set_conditioner_state_dict'])
    set_conditioner.eval()
    empty_token = EmptySetToken(d_model).to(device)
    empty_token.load_state_dict(ckpt['empty_token_state_dict'])
    empty_token.eval()
    if ckpt.get('scorer_mode') == 'asymmetric':
        utility_head = AsymmetricUtilityHead(d_model).to(device)
    else:
        utility_head = UtilityHead().to(device)
    utility_head.load_state_dict(ckpt['utility_head_state_dict'])
    utility_head.eval()
    return model, set_conditioner, empty_token, utility_head, args


def encode(model, x, c):
    with torch.no_grad():
        return F.normalize(model.encoder(model._relation_tensor(x, c, c)), dim=-1)


def _spearman(a, b):
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = (ra.norm() * rb.norm()).clamp_min(1e-12)
    return float((ra * rb).sum() / denom)


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    return float((a * b).sum() / denom)


@torch.no_grad()
def evaluate(s2_ckpt, seq_ckpt, teacher_cache_path, k, chunk_size, utility_eval_queries, device):
    b0_exp, b0_args = load_stage2(s2_ckpt)
    b0_exp._ensure_memory()
    b0_exp._build_key_bank()
    b0_model = b0_exp.model.module if hasattr(b0_exp.model, 'module') else b0_exp.model
    memory_x = torch.from_numpy(b0_exp.memory_bank.memory_x).float().to(device)
    memory_y = b0_exp.memory_y.to(device)
    memory_x_last = b0_exp.memory_x_last.to(device)

    seq_model, set_cond, empty_token, utility_head, seq_args = load_trained_selector(seq_ckpt, device)
    if int(seq_args.pred_len) != int(b0_args.pred_len) or seq_args.data != b0_args.data:
        raise ValueError('sequential-selector checkpoint and B0 Stage-2 checkpoint disagree '
                          'on dataset/pred_len -- refusing to evaluate')
    tau = float(seq_args.tau_topk)

    teacher = torch.load(teacher_cache_path, map_location='cpu')
    _, loader = b0_exp._get_data(flag='test', shuffle=False)
    channels = list(b0_model.target_channels())

    def zero_totals():
        return {'final_se': 0.0, 'final_ae': 0.0, 'n': 0.0}

    results = {label: zero_totals() for label in ('b0_unforced', 'sequential_forced')}
    diag = {
        'individual_oracle_mse_sum': 0.0, 'individual_oracle_n': 0,
        'set_oracle_a_weighted_sum': 0.0, 'set_oracle_n': 0,
        'seq_a_weighted_sum': 0.0, 'seq_a_weighted_n': 0,
        'b0_a_weighted_sum': 0.0, 'b0_a_weighted_n': 0,
        'set_recall_sum': 0.0, 'set_recall_n': 0,
        'duplicate_rows': 0, 'invalid_selected_rows': 0, 'total_rows': 0,
        'hard_agg_individual_se': 0.0, 'hard_agg_individual_n': 0,
        'hard_agg_set_oracle_se': 0.0, 'hard_agg_set_oracle_n': 0,
        'hard_agg_seq_se': 0.0, 'hard_agg_seq_n': 0,
        'hard_agg_b0_se': 0.0, 'hard_agg_b0_n': 0,
    }
    utility_rows_seen = 0
    spearman_sum, pearson_sum, corr_n = 0.0, 0.0, 0
    regret_sum = [0.0] * k
    regret_n = [0] * k

    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, counts = b0_exp._candidate_mask(batch_start_idx)
        valid_query = counts.to(device) >= k
        rows = [teacher['splits']['test']['start_to_row'][int(s)] for s in batch_start_idx.tolist()]

        forced = {}
        for c in channels:
            E = encode(seq_model, memory_x, c)
            q = encode(seq_model, batch_x, c)
            teacher_idx_c = teacher['splits']['test']['teacher_idx'][int(c)][rows].to(device)
            vq = valid_query & (teacher_idx_c[:, 0] != -1)

            memory_c, offset_c = b0_model._memory_value(batch_x, memory_y, memory_x_last, c)
            all_tgt = memory_c + offset_c.view(-1, 1, 1)
            q_tgt = batch_y[:, :, c]

            # ---- B. free-running sequential retrieval ----
            _, _, _, picks = run_sequence_dense(
                q, E, cand_mask, set_cond, empty_token, utility_head,
                all_tgt, q_tgt, tau, k, chunk_size, teacher_idx=None)
            forced[(int(c), 0)] = picks

            z_q_ref = b0_model._branch_embedding(batch_x, c, c)
            z_mem_ref = b0_model._branch_memory(b0_exp.key_bank, c, 0, c, z_q_ref.dtype, device)
            cosine_ref = torch.matmul(z_q_ref, z_mem_ref.transpose(0, 1))
            score_fn = b0_model._retrieval_score_fn()
            learned_ref = score_fn(z_q_ref, z_mem_ref) if score_fn is not None else cosine_ref

            dist_individual = (all_tgt - q_tgt.unsqueeze(1)).pow(2).mean(-1)
            dist_individual = dist_individual.masked_fill(~cand_mask, float('inf'))
            best_idx = dist_individual.argmin(dim=-1, keepdim=True)
            individual_oracle_mse = dist_individual.gather(1, best_idx).squeeze(1)

            def a_weighted(idx):
                tgt = all_tgt.gather(1, idx.unsqueeze(-1).expand(-1, -1, all_tgt.size(-1)))
                sc = learned_ref.gather(1, idx)
                al = torch.softmax(sc / tau, dim=-1)
                return ((al.unsqueeze(-1) * tgt).sum(1) - q_tgt).pow(2).mean(-1)

            def hard_agg(idx):
                tgt = all_tgt.gather(1, idx.unsqueeze(-1).expand(-1, -1, all_tgt.size(-1)))
                agg = tgt.mean(1)
                return (agg - q_tgt).pow(2).sum(-1)

            seq_a = a_weighted(picks)
            set_a = a_weighted(teacher_idx_c)
            b0_neg = torch.finfo(learned_ref.dtype).min / 4
            b0_masked = learned_ref.masked_fill(~cand_mask, b0_neg)
            b0_topk_idx = b0_masked.topk(k, dim=-1).indices
            b0_a = a_weighted(b0_topk_idx)

            diag['individual_oracle_mse_sum'] += float(individual_oracle_mse[vq].sum())
            diag['individual_oracle_n'] += int(vq.sum())
            diag['set_oracle_a_weighted_sum'] += float(set_a[vq].sum())
            diag['set_oracle_n'] += int(vq.sum())
            diag['seq_a_weighted_sum'] += float(seq_a[vq].sum())
            diag['seq_a_weighted_n'] += int(vq.sum())
            diag['b0_a_weighted_sum'] += float(b0_a[vq].sum())
            diag['b0_a_weighted_n'] += int(vq.sum())

            h = all_tgt.size(-1)
            diag['hard_agg_individual_se'] += float(dist_individual.gather(1, best_idx).squeeze(1)[vq].sum() * h)
            diag['hard_agg_individual_n'] += int(vq.sum())
            diag['hard_agg_set_oracle_se'] += float(hard_agg(teacher_idx_c)[vq].sum())
            diag['hard_agg_set_oracle_n'] += int(vq.sum())
            diag['hard_agg_seq_se'] += float(hard_agg(picks)[vq].sum())
            diag['hard_agg_seq_n'] += int(vq.sum())
            diag['hard_agg_b0_se'] += float(hard_agg(b0_topk_idx)[vq].sum())
            diag['hard_agg_b0_n'] += int(vq.sum())

            for b in range(picks.size(0)):
                if not vq[b]:
                    continue
                p = picks[b].tolist()
                t = teacher_idx_c[b].tolist()
                diag['set_recall_sum'] += len(set(p) & set(t)) / k
                diag['set_recall_n'] += 1
                diag['total_rows'] += 1
                if len(set(p)) != len(p):
                    diag['duplicate_rows'] += 1
                if not bool(cand_mask[b, p].all()):
                    diag['invalid_selected_rows'] += 1

            # ---- A. teacher-forced utility prediction (query-subsampled) ----
            if utility_rows_seen < utility_eval_queries:
                take = min(int(vq.sum()), utility_eval_queries - utility_rows_seen)
                if take > 0:
                    idx_take = vq.nonzero(as_tuple=True)[0][:take]
                    w = candidate_weights(torch.matmul(q, E.transpose(0, 1)), cand_mask, tau)
                    selected_mask = torch.zeros_like(cand_mask)
                    for t in range(k):
                        if t == 0:
                            m = empty_token(q.size(0), q.device, q.dtype)
                        else:
                            m = E[teacher_idx_c[:, :t].clamp_min(0)].mean(dim=1)
                        h_t = set_cond(q, m)
                        u_hat = utility_head(h_t, E)
                        a_dense = dense_utility(teacher_idx_c[:, :t].clamp_min(0), w, all_tgt, q_tgt,
                                                 chunk_size=chunk_size)
                        u_teacher = -a_dense
                        valid_now = cand_mask & ~selected_mask
                        u_hat_masked = u_hat.masked_fill(~valid_now, float('-inf'))
                        student_pick = u_hat_masked.argmax(dim=-1)  # what the model would pick here
                        for b in idx_take.tolist():
                            valid_b = valid_now[b]
                            uh = u_hat[b][valid_b]
                            ut = u_teacher[b][valid_b]
                            if uh.numel() < 2:
                                continue
                            spearman_sum += _spearman(uh, ut)
                            pearson_sum += _pearson(uh, ut)
                            corr_n += 1
                            best_u = ut.max()
                            # Regret is about the STUDENT's own choice under
                            # the true oracle prefix, scored by the oracle's
                            # own utility -- not the oracle's own pick (that
                            # would be zero by construction).
                            sel_u = u_teacher[b, student_pick[b]]
                            regret_sum[t] += float(best_u - sel_u)
                            regret_n[t] += 1
                        selected_mask = selected_mask.scatter(1, teacher_idx_c[:, t:t + 1].clamp_min(0), True)
                    utility_rows_seen += take

        for label, force_table in (('b0_unforced', None), ('sequential_forced', forced)):
            b0_model.set_forced_selection(force_table)
            y_final, y_base, y_ret, beta, lam, debug = b0_exp.model(
                batch_x=batch_x, memory_y=b0_exp.memory_y, valid_mask=cand_mask,
                key_bank=b0_exp.key_bank, memory_x_last=b0_exp.memory_x_last,
                retrieval_cache=None, target_y=batch_y,
                teacher_key_bank=getattr(b0_exp, 'teacher_key_bank', None),
            )
            b0_model.set_forced_selection(None)
            yf, by = y_final[valid_query], batch_y[valid_query]
            results[label]['final_se'] += float((yf - by).pow(2).sum())
            results[label]['final_ae'] += float((yf - by).abs().sum())
            results[label]['n'] += float(yf.numel())

    out = {}
    for label, tot in results.items():
        out[f'{label}_final_mse'] = tot['final_se'] / max(tot['n'], 1)
        out[f'{label}_final_mae'] = tot['final_ae'] / max(tot['n'], 1)
    out['individual_oracle_mse'] = diag['individual_oracle_mse_sum'] / max(diag['individual_oracle_n'], 1)
    out['set_oracle_a_weighted'] = diag['set_oracle_a_weighted_sum'] / max(diag['set_oracle_n'], 1)
    out['seq_a_weighted'] = diag['seq_a_weighted_sum'] / max(diag['seq_a_weighted_n'], 1)
    out['b0_a_weighted'] = diag['b0_a_weighted_sum'] / max(diag['b0_a_weighted_n'], 1)
    denom = out['b0_a_weighted'] - out['set_oracle_a_weighted']
    out['gap_recovery'] = (
        (out['b0_a_weighted'] - out['seq_a_weighted']) / denom if abs(denom) > 1e-12 else float('nan'))
    out['seq_set_recall_at_k'] = diag['set_recall_sum'] / max(diag['set_recall_n'], 1)
    out['duplicate_rate'] = diag['duplicate_rows'] / max(diag['total_rows'], 1)
    out['invalid_rate'] = diag['invalid_selected_rows'] / max(diag['total_rows'], 1)
    out['n_channels'] = len(channels)
    out['n_test_rows_per_channel'] = diag['total_rows'] // max(len(channels), 1)
    horizon = int(b0_args.pred_len)
    out['hard_aggregate_mse_individual'] = diag['hard_agg_individual_se'] / max(diag['hard_agg_individual_n'], 1) / horizon
    out['hard_aggregate_mse_set_oracle'] = diag['hard_agg_set_oracle_se'] / max(diag['hard_agg_set_oracle_n'], 1) / horizon
    out['hard_aggregate_mse_seq'] = diag['hard_agg_seq_se'] / max(diag['hard_agg_seq_n'], 1) / horizon
    out['hard_aggregate_mse_b0'] = diag['hard_agg_b0_se'] / max(diag['hard_agg_b0_n'], 1) / horizon
    out['utility_spearman_mean'] = spearman_sum / max(corr_n, 1)
    out['utility_pearson_mean'] = pearson_sum / max(corr_n, 1)
    out['utility_eval_rows'] = utility_rows_seen
    out['utility_regret_by_step'] = [regret_sum[t] / max(regret_n[t], 1) for t in range(k)]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage2_checkpoint', required=True)
    ap.add_argument('--sequential_checkpoint', required=True)
    ap.add_argument('--teacher_cache', required=True)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--utility_eval_queries', type=int, default=200)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = evaluate(args.stage2_checkpoint, args.sequential_checkpoint,
                    args.teacher_cache, args.top_k, args.chunk_size,
                    args.utility_eval_queries, device)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=2)
    for key, val in out.items():
        print(f'[margutil01-stage2] {key} = {val}')
    print(f'[margutil01-stage2] written to {args.out}')


if __name__ == '__main__':
    main()
