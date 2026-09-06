#!/usr/bin/env python3
"""EXP-SEQFULL01 Stage-2 evaluation: inject the trained sequential selector's
free-running (argmax, no teacher forcing) Top-10 into B0's UNCHANGED Stage-2
via RelationStage2.set_forced_selection -- the exact mechanism EXP-1/EXP-2's
oracle_intervention already uses and this project has already reviewed.
Nothing about Stage-2's weighting/gate/fusion/base-forecaster changes; only
Top-K membership does.

Also runs B0's own (unforced) selection through the identical loop first, as
the fingerprint check the experiment spec requires: if that does not
reproduce the known B0 Final MSE (~0.37312), something about protocol/support
has silently drifted and the forced-selection numbers would not be trustworthy.
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
from models.SequentialSetRetriever import EmptySetToken, SetConditioner
from scripts.train_seqfull01 import run_sequence
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
    return model, set_conditioner, empty_token, args


def encode(model, x, c):
    return F.normalize(model.encoder(model._relation_tensor(x, c, c)), dim=-1)


@torch.no_grad()
def evaluate(s2_ckpt, seq_ckpt, teacher_cache_path, k, device):
    b0_exp, b0_args = load_stage2(s2_ckpt)
    b0_exp._ensure_memory()
    b0_exp._build_key_bank()
    b0_model = b0_exp.model.module if hasattr(b0_exp.model, 'module') else b0_exp.model
    memory_x = torch.from_numpy(b0_exp.memory_bank.memory_x).float().to(device)
    memory_y = b0_exp.memory_y.to(device)
    memory_x_last = b0_exp.memory_x_last.to(device)

    seq_model, set_cond, empty_token, seq_args = load_trained_selector(seq_ckpt, device)
    if int(seq_args.pred_len) != int(b0_args.pred_len) or seq_args.data != b0_args.data:
        raise ValueError('sequential-selector checkpoint and B0 Stage-2 checkpoint disagree '
                          'on dataset/pred_len -- refusing to evaluate')

    teacher = torch.load(teacher_cache_path, map_location='cpu')

    _, loader = b0_exp._get_data(flag='test', shuffle=False)

    def zero_totals():
        return {'final_se': 0.0, 'final_ae': 0.0, 'n': 0.0}

    results = {}
    for label in ('b0_unforced', 'sequential_forced'):
        results[label] = zero_totals()

    # Stage-1 diagnostics accumulated alongside the sequential arm's pass.
    diag = {
        'individual_oracle_mse_sum': 0.0, 'individual_oracle_n': 0,
        'set_oracle_a_weighted_sum': 0.0, 'set_oracle_n': 0,
        'seq_a_weighted_sum': 0.0, 'seq_a_weighted_n': 0,
        'set_recall_sum': 0.0, 'set_recall_n': 0,
        'duplicate_rows': 0, 'invalid_selected_rows': 0, 'total_rows': 0,
    }

    channels = list(b0_model.target_channels())

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
            _, picks = run_sequence(q, E, cand_mask, set_cond, empty_token, k, teacher_idx=None)
            forced[(int(c), 0)] = picks

            # ---- Stage-1 diagnostics for this channel ----
            memory_c, offset_c = b0_model._memory_value(batch_x, memory_y, memory_x_last, c)
            q_tgt = batch_y[:, :, c]
            teacher_idx_c = teacher['splits']['test']['teacher_idx'][int(c)][rows].to(device)

            z_q_ref = b0_model._branch_embedding(batch_x, c, c)
            z_mem_ref = b0_model._branch_memory(b0_exp.key_bank, c, 0, c, z_q_ref.dtype, device)
            cosine_ref = torch.matmul(z_q_ref, z_mem_ref.transpose(0, 1))
            score_fn = b0_model._retrieval_score_fn()
            learned_ref = score_fn(z_q_ref, z_mem_ref) if score_fn is not None else cosine_ref

            all_tgt = memory_c + offset_c.view(-1, 1, 1)  # [B, N, H], every candidate
            dist_individual = (all_tgt - q_tgt.unsqueeze(1)).pow(2).mean(-1)
            dist_individual = dist_individual.masked_fill(~cand_mask, float('inf'))
            individual_oracle_mse = dist_individual.min(dim=-1).values

            seq_tgt = all_tgt.gather(1, picks.unsqueeze(-1).expand(-1, -1, all_tgt.size(-1)))
            seq_scores = learned_ref.gather(1, picks)
            seq_alpha = torch.softmax(seq_scores / float(b0_args.tau_topk), dim=-1)
            seq_a_weighted = (
                (seq_alpha.unsqueeze(-1) * seq_tgt).sum(1) - q_tgt
            ).pow(2).mean(-1)

            # B0's own natural (unforced) Top-K selection, for gap_recovery's
            # denominator/numerator baseline: A_B0 = the aggregate B0's own
            # retriever forms, in the same A_weighted space as the other two.
            b0_neg = torch.finfo(learned_ref.dtype).min / 4
            b0_masked = learned_ref.masked_fill(~cand_mask, b0_neg)
            b0_topk_idx = b0_masked.topk(k, dim=-1).indices
            b0_tgt = all_tgt.gather(1, b0_topk_idx.unsqueeze(-1).expand(-1, -1, all_tgt.size(-1)))
            b0_scores = learned_ref.gather(1, b0_topk_idx)
            b0_alpha = torch.softmax(b0_scores / float(b0_args.tau_topk), dim=-1)
            b0_a_weighted = (
                (b0_alpha.unsqueeze(-1) * b0_tgt).sum(1) - q_tgt
            ).pow(2).mean(-1)

            # Reuses the cached teacher sequence rather than recomputing
            # select_greedy_weighted_set -- same function, same inputs
            # (this is exactly how that cache was built), so recomputing
            # would only be redundant compute, not an independent check.
            set_oracle_idx = teacher_idx_c
            set_tgt = all_tgt.gather(1, set_oracle_idx.unsqueeze(-1).expand(-1, -1, all_tgt.size(-1)))
            set_scores = learned_ref.gather(1, set_oracle_idx)
            set_alpha = torch.softmax(set_scores / float(b0_args.tau_topk), dim=-1)
            set_a_weighted = (
                (set_alpha.unsqueeze(-1) * set_tgt).sum(1) - q_tgt
            ).pow(2).mean(-1)

            vq = valid_query
            diag['individual_oracle_mse_sum'] += float(individual_oracle_mse[vq].sum())
            diag['individual_oracle_n'] += int(vq.sum())
            diag['set_oracle_a_weighted_sum'] += float(set_a_weighted[vq].sum())
            diag['set_oracle_n'] += int(vq.sum())
            diag['seq_a_weighted_sum'] += float(seq_a_weighted[vq].sum())
            diag['seq_a_weighted_n'] += int(vq.sum())
            diag['b0_a_weighted_sum'] = diag.get('b0_a_weighted_sum', 0.0) + float(b0_a_weighted[vq].sum())
            diag['b0_a_weighted_n'] = diag.get('b0_a_weighted_n', 0) + int(vq.sum())
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
    out['b0_a_weighted'] = diag['b0_a_weighted_sum'] / max(diag.get('b0_a_weighted_n', 0), 1)
    denom = out['b0_a_weighted'] - out['set_oracle_a_weighted']
    out['gap_recovery'] = (
        (out['b0_a_weighted'] - out['seq_a_weighted']) / denom if abs(denom) > 1e-12 else float('nan')
    )
    out['seq_set_recall_at_k'] = diag['set_recall_sum'] / max(diag['set_recall_n'], 1)
    out['duplicate_rate'] = diag['duplicate_rows'] / max(diag['total_rows'], 1)
    out['invalid_rate'] = diag['invalid_selected_rows'] / max(diag['total_rows'], 1)
    out['n_channels'] = len(channels)
    out['n_test_rows_per_channel'] = diag['total_rows'] // max(len(channels), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage2_checkpoint', required=True)
    ap.add_argument('--sequential_checkpoint', required=True)
    ap.add_argument('--teacher_cache', default='cache/seqfull01_teacher/ETTh1_pred96.pt')
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--out', default='results/EXP-SEQFULL01/stage2_eval.json')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = evaluate(args.stage2_checkpoint, args.sequential_checkpoint,
                    args.teacher_cache, args.top_k, device)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(out, fh, indent=2)
    for key, val in out.items():
        print(f'[seqfull01-stage2] {key} = {val}')
    print(f'[seqfull01-stage2] written to {args.out}')


if __name__ == '__main__':
    main()
