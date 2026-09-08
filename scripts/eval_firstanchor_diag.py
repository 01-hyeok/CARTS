#!/usr/bin/env python3
"""EXP-FIRSTANCHOR-DIAG: causal decomposition of EXP-MARGUTIL01's failure.

No new training. Reuses an EXP-MARGUTIL01 checkpoint (frozen B0 encoder,
SetConditioner, UtilityHead) exactly as-is. The only thing this script
changes is WHICH RULE PICKS THE t=1 (FIRST) CANDIDATE; steps t=2..10 always
use the same, unmodified, free-running Dense Marginal Utility selector.

Four arms per query:
  A. Dense-first  -- t=1 also picked by the Dense utility head (== the
     existing EXP-MARGUTIL01 free-running result, reused not recomputed
     where a stage2 json for the cell already exists).
  B. B0-first      -- t=1 picked by B0's own PRODUCTION retrieval score
     (`RelationStage2._retrieval_score_fn()` / cosine, the exact scorer
     Stage-2 already uses -- not a new score).
  C. Oracle-first  -- t=1 picked by argmin singleton future MSE (uses Y_q;
     diagnostic only, never reused past t=1).
  D. B0 baseline   -- B0's own natural (unforced) Top-10, for reference.

t=2..10 for B/C use the identical unmodified Dense selector as A -- no
teacher forcing after t=1, no oracle injection past t=1, no B0 score used
past t=1 either.
"""
import argparse
import csv
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
from models.DenseUtilityRetriever import AsymmetricUtilityHead, StrongResidualPairScorer, UtilityHead
from models.SequentialSetRetriever import EmptySetToken, SetConditioner
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
    scorer_mode = ckpt.get('scorer_mode')
    if scorer_mode == 'asymmetric':
        utility_head = AsymmetricUtilityHead(d_model).to(device)
    elif scorer_mode == 'strong_pair':
        utility_head = StrongResidualPairScorer(d_model).to(device)
    else:
        utility_head = UtilityHead().to(device)
    utility_head.load_state_dict(ckpt['utility_head_state_dict'])
    utility_head.eval()
    return model, set_conditioner, empty_token, utility_head, args


def encode(model, x, c):
    with torch.no_grad():
        return F.normalize(model.encoder(model._relation_tensor(x, c, c)), dim=-1)


def run_arm(q, E, cand_mask, set_cond, empty_token, utility_head, k,
            first_pick=None):
    """K steps. t=1 is `first_pick` if given (an intervention), else the
    Dense model's own argmax (arm A). t=2..K is ALWAYS the unmodified Dense
    selector's free-running argmax -- identical code path regardless of arm,
    so a bug here cannot accidentally make one arm's later steps different
    from another's except through the state each step's own set summary
    inherits from t=1's pick.

    Returns picks [B, K].
    """
    bsz = q.size(0)
    selected_mask = torch.zeros_like(cand_mask)
    picks = []
    for t in range(k):
        if t == 0:
            m = empty_token(bsz, q.device, q.dtype)
        else:
            m = E[torch.stack(picks, dim=1)].mean(dim=1)
        if t == 0 and first_pick is not None:
            nxt = first_pick.unsqueeze(-1)
        else:
            h = set_cond(q, m)
            u_hat = utility_head(h, E)
            valid_now = cand_mask & ~selected_mask
            u_masked = u_hat.masked_fill(~valid_now, float('-inf'))
            nxt = u_masked.argmax(dim=-1, keepdim=True)
        picks.append(nxt.squeeze(-1))
        selected_mask = selected_mask.scatter(1, nxt, True)
    return torch.stack(picks, dim=1)


def a_weighted_prefix(prefix_idx, learned_ref, all_tgt, q_tgt, tau):
    """A_weighted(S) for a GROWING prefix (any length 1..K), using B0's own
    fixed retrieval score for the softmax weights -- same convention as
    every other arm in this project, never the Dense model's own score."""
    sc = learned_ref.gather(1, prefix_idx)
    al = torch.softmax(sc / tau, dim=-1)
    tgt = all_tgt.gather(1, prefix_idx.unsqueeze(-1).expand(-1, -1, all_tgt.size(-1)))
    return ((al.unsqueeze(-1) * tgt).sum(1) - q_tgt).pow(2).mean(-1)


def hard_agg(prefix_idx, all_tgt, q_tgt):
    tgt = all_tgt.gather(1, prefix_idx.unsqueeze(-1).expand(-1, -1, all_tgt.size(-1)))
    agg = tgt.mean(1)
    return (agg - q_tgt).pow(2).sum(-1)


def _ndcg_at_k(rel, k):
    """rel: 1D tensor of relevance scores IN PREDICTED ORDER (already sorted
    by the model's own predicted rank, descending). Standard NDCG."""
    n = min(k, rel.numel())
    if n == 0:
        return 0.0
    gains = rel[:n]
    discounts = torch.log2(torch.arange(2, n + 2, dtype=torch.float32, device=rel.device))
    dcg = (gains / discounts).sum()
    ideal, _ = torch.sort(rel, descending=True)
    ideal = ideal[:n]
    idcg = (ideal / discounts).sum().clamp_min(1e-12)
    return float(dcg / idcg)


@torch.no_grad()
def evaluate(s2_ckpt, seq_ckpt, teacher_cache_path, k, rank_eval_queries, device, split='test'):
    b0_exp, b0_args = load_stage2(s2_ckpt)
    b0_exp._ensure_memory()
    b0_exp._build_key_bank()
    b0_model = b0_exp.model.module if hasattr(b0_exp.model, 'module') else b0_exp.model
    memory_y = b0_exp.memory_y.to(device)
    memory_x_last = b0_exp.memory_x_last.to(device)

    seq_model, set_cond, empty_token, utility_head, seq_args = load_trained_selector(seq_ckpt, device)
    if int(seq_args.pred_len) != int(b0_args.pred_len) or seq_args.data != b0_args.data:
        raise ValueError('sequential-selector checkpoint and B0 Stage-2 checkpoint disagree '
                          'on dataset/pred_len -- refusing to evaluate')
    tau = float(seq_args.tau_topk)

    memory_x = torch.from_numpy(b0_exp.memory_bank.memory_x).float().to(device)
    teacher = torch.load(teacher_cache_path, map_location='cpu')
    _, loader = b0_exp._get_data(flag=split, shuffle=False)
    channels = list(b0_model.target_channels())

    arms = ['dense_first', 'b0_first', 'oracle_first']
    stage2_res = {a: {'final_se': 0.0, 'final_ae': 0.0, 'n': 0.0} for a in ('b0_unforced',) + tuple(arms)}
    agg_diag = {a: {'a_weighted_sum': 0.0, 'hard_agg_se': 0.0, 'n': 0,
                     'set_recall_sum': 0.0, 'dup': 0, 'invalid': 0}
                for a in arms}
    step_traj = {a: [0.0] * k for a in arms}
    step_traj_n = {a: [0] * k for a in arms}
    total_rows = 0

    cand_quality = {  # singleton future MSE of the t=1 pick, per arm
        'dense': [], 'b0': [], 'oracle': [],
    }
    b0_eq_oracle, dense_eq_oracle, n_first = 0, 0, 0
    b0_beats_dense, dense_beats_b0, first_ties = 0, 0, 0

    rank_rows_seen = 0
    oracle_rank_list, oracle_pct_list = [], []
    top1_hits = top5_hits = top10_hits = top50_hits = 0
    spearman_top1pct_sum, spearman_top1pct_n = 0.0, 0
    ndcg10_sum, ndcg50_sum, ndcg_n = 0.0, 0.0, 0

    identity_checked = False
    identity_ok = True

    stepwise_csv_rows = []
    quality_csv_rows = []

    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, counts = b0_exp._candidate_mask(batch_start_idx)
        valid_query = counts.to(device) >= k
        rows = [teacher['splits'][split]['start_to_row'][int(s)] for s in batch_start_idx.tolist()]

        forced = {a: {} for a in arms}
        for c in channels:
            E = encode(seq_model, memory_x, c)
            q = encode(seq_model, batch_x, c)
            teacher_idx_c = teacher['splits'][split]['teacher_idx'][int(c)][rows].to(device)
            vq = valid_query & (teacher_idx_c[:, 0] != -1)

            memory_c, offset_c = b0_model._memory_value(batch_x, memory_y, memory_x_last, c)
            all_tgt = memory_c + offset_c.view(-1, 1, 1)
            q_tgt = batch_y[:, :, c]

            z_q_ref = b0_model._branch_embedding(batch_x, c, c)
            z_mem_ref = b0_model._branch_memory(b0_exp.key_bank, c, 0, c, z_q_ref.dtype, device)
            cosine_ref = torch.matmul(z_q_ref, z_mem_ref.transpose(0, 1))
            score_fn = b0_model._retrieval_score_fn()
            learned_ref = score_fn(z_q_ref, z_mem_ref) if score_fn is not None else cosine_ref

            # ---- t=1 candidates for each intervention ----
            dist_individual = (all_tgt - q_tgt.unsqueeze(1)).pow(2).mean(-1)
            dist_individual = dist_individual.masked_fill(~cand_mask, float('inf'))
            i1_oracle = dist_individual.argmin(dim=-1)

            if not identity_checked:
                # Section 6 Arm C requirement: singleton oracle pick must equal
                # the greedy Weighted Set Oracle's own cached first candidate.
                identity_ok = identity_ok and bool(
                    torch.equal(i1_oracle[vq], teacher_idx_c[vq, 0]))
                identity_checked = True

            b0_neg = torch.finfo(learned_ref.dtype).min / 4
            b0_masked = learned_ref.masked_fill(~cand_mask, b0_neg)
            i1_b0 = b0_masked.argmax(dim=-1)

            # ---- Arm A: pure Dense-first (also gives i1_dense for quality table) ----
            picks_dense = run_arm(q, E, cand_mask, set_cond, empty_token, utility_head, k, first_pick=None)
            i1_dense = picks_dense[:, 0]

            picks_b0first = run_arm(q, E, cand_mask, set_cond, empty_token, utility_head, k, first_pick=i1_b0)
            picks_oraclefirst = run_arm(q, E, cand_mask, set_cond, empty_token, utility_head, k, first_pick=i1_oracle)

            picks_by_arm = {'dense_first': picks_dense, 'b0_first': picks_b0first, 'oracle_first': picks_oraclefirst}
            for a in arms:
                forced[a][(int(c), 0)] = picks_by_arm[a]

            # ---- t=1 candidate quality (singleton future MSE) ----
            mse_dense = dist_individual.gather(1, i1_dense.unsqueeze(-1)).squeeze(1)
            mse_b0 = dist_individual.gather(1, i1_b0.unsqueeze(-1)).squeeze(1)
            mse_oracle = dist_individual.gather(1, i1_oracle.unsqueeze(-1)).squeeze(1)
            for b in vq.nonzero(as_tuple=True)[0].tolist():
                cand_quality['dense'].append(float(mse_dense[b]))
                cand_quality['b0'].append(float(mse_b0[b]))
                cand_quality['oracle'].append(float(mse_oracle[b]))
                n_first += 1
                if int(i1_b0[b]) == int(i1_oracle[b]):
                    b0_eq_oracle += 1
                if int(i1_dense[b]) == int(i1_oracle[b]):
                    dense_eq_oracle += 1
                if float(mse_b0[b]) < float(mse_dense[b]) - 1e-9:
                    b0_beats_dense += 1
                elif float(mse_dense[b]) < float(mse_b0[b]) - 1e-9:
                    dense_beats_b0 += 1
                else:
                    first_ties += 1

            # ---- stepwise A_weighted(S_t) trajectory + hard agg + final diagnostics ----
            for a in arms:
                picks = picks_by_arm[a]
                for t in range(k):
                    prefix = picks[:, :t + 1]
                    a_t = a_weighted_prefix(prefix, learned_ref, all_tgt, q_tgt, tau)
                    step_traj[a][t] += float(a_t[vq].sum())
                    step_traj_n[a][t] += int(vq.sum())
                    stepwise_csv_rows.append({
                        'channel': int(c), 'arm': a, 'step': t + 1,
                        'a_weighted_mean_this_batch': float(a_t[vq].mean()) if int(vq.sum()) else float('nan'),
                        'n_valid': int(vq.sum()),
                    })
                final_a = a_weighted_prefix(picks, learned_ref, all_tgt, q_tgt, tau)
                agg_diag[a]['a_weighted_sum'] += float(final_a[vq].sum())
                agg_diag[a]['hard_agg_se'] += float(hard_agg(picks, all_tgt, q_tgt)[vq].sum())
                agg_diag[a]['n'] += int(vq.sum())
                for b in range(picks.size(0)):
                    if not vq[b]:
                        continue
                    p = picks[b].tolist()
                    t_ = teacher_idx_c[b].tolist()
                    agg_diag[a]['set_recall_sum'] += len(set(p) & set(t_)) / k
                    if len(set(p)) != len(p):
                        agg_diag[a]['dup'] += 1
                    if not bool(cand_mask[b, p].all()):
                        agg_diag[a]['invalid'] += 1
                total_rows += int(vq.sum()) if a == arms[0] else 0

            # ---- t=1 rank / top-tail diagnostics (query-subsampled) ----
            if rank_rows_seen < rank_eval_queries:
                take = min(int(vq.sum()), rank_eval_queries - rank_rows_seen)
                if take > 0:
                    idx_take = vq.nonzero(as_tuple=True)[0][:take]
                    m0 = empty_token(q.size(0), q.device, q.dtype)
                    h0 = set_cond(q, m0)
                    u_hat0 = utility_head(h0, E)
                    u_teacher0 = -dist_individual  # -MSE, matches dense_utility's empty-prefix identity
                    for b in idx_take.tolist():
                        valid_b = cand_mask[b]
                        uh = u_hat0[b][valid_b]
                        ut = u_teacher0[b][valid_b]
                        n_valid = uh.numel()
                        # rank of the oracle-best candidate in the STUDENT's
                        # predicted ordering (0 = predicted best)
                        order = torch.argsort(uh, descending=True)
                        oracle_local = int((valid_b.nonzero(as_tuple=True)[0] == int(i1_oracle[b])).nonzero())
                        rank_of_oracle = int((order == oracle_local).nonzero())
                        pct = rank_of_oracle / max(n_valid - 1, 1)
                        oracle_rank_list.append(rank_of_oracle)
                        oracle_pct_list.append(pct)
                        if rank_of_oracle == 0:
                            top1_hits += 1
                        if rank_of_oracle < 5:
                            top5_hits += 1
                        if rank_of_oracle < 10:
                            top10_hits += 1
                        if rank_of_oracle < 50:
                            top50_hits += 1

                        # top-tail: Spearman within TRUE top 1% by teacher utility
                        top1pct_n = max(int(0.01 * n_valid), 2)
                        true_order = torch.argsort(ut, descending=True)[:top1pct_n]
                        uh_top = uh[true_order]
                        ut_top = ut[true_order]
                        ra = uh_top.argsort().argsort().float() - (top1pct_n - 1) / 2
                        rb = ut_top.argsort().argsort().float() - (top1pct_n - 1) / 2
                        denom = (ra.norm() * rb.norm()).clamp_min(1e-12)
                        spearman_top1pct_sum += float((ra * rb).sum() / denom)
                        spearman_top1pct_n += 1

                        # NDCG@10 / @50: relevance = teacher utility (shifted
                        # positive), ranked by the STUDENT's predicted order
                        rel = ut - ut.min() + 1e-6
                        rel_pred_order = rel[order]
                        ndcg10_sum += _ndcg_at_k(rel_pred_order, 10)
                        ndcg50_sum += _ndcg_at_k(rel_pred_order, 50)
                        ndcg_n += 1
                    rank_rows_seen += take

        for label, force_table in (('b0_unforced', None), ('dense_first', forced['dense_first']),
                                    ('b0_first', forced['b0_first']), ('oracle_first', forced['oracle_first'])):
            b0_model.set_forced_selection(force_table)
            y_final, y_base, y_ret, beta, lam, debug = b0_exp.model(
                batch_x=batch_x, memory_y=b0_exp.memory_y, valid_mask=cand_mask,
                key_bank=b0_exp.key_bank, memory_x_last=b0_exp.memory_x_last,
                retrieval_cache=None, target_y=batch_y,
                teacher_key_bank=getattr(b0_exp, 'teacher_key_bank', None),
            )
            b0_model.set_forced_selection(None)
            yf, by = y_final[valid_query], batch_y[valid_query]
            stage2_res[label]['final_se'] += float((yf - by).pow(2).sum())
            stage2_res[label]['final_ae'] += float((yf - by).abs().sum())
            stage2_res[label]['n'] += float(yf.numel())

    out = {'stage2': {}, 'aggregate': {}, 'stepwise_trajectory': {}, 't1_rank': {},
           't1_candidate_quality': {}, 'identity_check_singleton_oracle_eq_teacher_first': identity_ok}
    for label, tot in stage2_res.items():
        out['stage2'][f'{label}_final_mse'] = tot['final_se'] / max(tot['n'], 1)
        out['stage2'][f'{label}_final_mae'] = tot['final_ae'] / max(tot['n'], 1)
    horizon = int(b0_args.pred_len)
    for a in arms:
        d = agg_diag[a]
        out['aggregate'][a] = {
            'a_weighted': d['a_weighted_sum'] / max(d['n'], 1),
            'hard_aggregate_mse': d['hard_agg_se'] / max(d['n'], 1) / horizon,
            'set_recall_at_k': d['set_recall_sum'] / max(d['n'], 1),
            'duplicate_rate': d['dup'] / max(d['n'], 1),
            'invalid_rate': d['invalid'] / max(d['n'], 1),
        }
        out['stepwise_trajectory'][a] = [
            step_traj[a][t] / max(step_traj_n[a][t], 1) for t in range(k)]

    n = max(len(oracle_rank_list), 1)
    out['t1_rank'] = {
        'rows': len(oracle_rank_list),
        'oracle_rank_mean': sum(oracle_rank_list) / n,
        'oracle_rank_median': sorted(oracle_rank_list)[len(oracle_rank_list) // 2] if oracle_rank_list else float('nan'),
        'oracle_percentile_rank_mean': sum(oracle_pct_list) / n,
        'top1_hit_rate': top1_hits / n,
        'top5_containment': top5_hits / n,
        'top10_containment': top10_hits / n,
        'top50_containment': top50_hits / n,
        'spearman_within_teacher_top1pct': spearman_top1pct_sum / max(spearman_top1pct_n, 1),
        'ndcg10_utility_at_predicted_rank': ndcg10_sum / max(ndcg_n, 1),
        'ndcg50_utility_at_predicted_rank': ndcg50_sum / max(ndcg_n, 1),
    }

    def _stats(xs):
        if not xs:
            return {}
        xs_sorted = sorted(xs)
        n_ = len(xs_sorted)
        return {
            'mean': sum(xs_sorted) / n_,
            'median': xs_sorted[n_ // 2],
            'p25': xs_sorted[int(0.25 * n_)],
            'p75': xs_sorted[int(0.75 * n_)],
        }
    out['t1_candidate_quality'] = {
        'dense': _stats(cand_quality['dense']),
        'b0': _stats(cand_quality['b0']),
        'oracle': _stats(cand_quality['oracle']),
        'n_first': n_first,
        'b0_eq_oracle_fraction': b0_eq_oracle / max(n_first, 1),
        'dense_eq_oracle_fraction': dense_eq_oracle / max(n_first, 1),
        'b0_beats_dense_fraction': b0_beats_dense / max(n_first, 1),
        'dense_beats_b0_fraction': dense_beats_b0 / max(n_first, 1),
        'first_pick_tie_fraction': first_ties / max(n_first, 1),
    }
    out['_stepwise_csv_rows'] = stepwise_csv_rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage2_checkpoint', required=True)
    ap.add_argument('--sequential_checkpoint', required=True)
    ap.add_argument('--teacher_cache', required=True)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--rank_eval_queries', type=int, default=200)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--cell_name', required=True)
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = evaluate(args.stage2_checkpoint, args.sequential_checkpoint,
                    args.teacher_cache, args.top_k, args.rank_eval_queries, device, split=args.split)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stepwise_rows = out.pop('_stepwise_csv_rows')
    with open(out_dir / f'{args.cell_name}_summary.json', 'w') as fh:
        json.dump(out, fh, indent=2)
    with open(out_dir / f'{args.cell_name}_stepwise_raw.csv', 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=['channel', 'arm', 'step', 'a_weighted_mean_this_batch', 'n_valid'])
        writer.writeheader()
        writer.writerows(stepwise_rows)
    print(json.dumps(out, indent=2))
    print(f'[firstanchor] written to {out_dir}/{args.cell_name}_summary.json')


if __name__ == '__main__':
    main()
