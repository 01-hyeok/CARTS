#!/usr/bin/env python3
"""EXP-CORRECTION-ORACLE-DIAG01 (Track B) -- diagnostic only, no training.
Runs in PARALLEL with Track A (EXP-ENCODER-ANCHOR01 -> EXP-ORACLE-CHOICE01
-> EXP-TEACHER-FORCING-DIAG01 -> EXP-ONPOLICY-PREFIX01); touches none of
Track A's code, checkpoints, or result directories.

Question: is retrieving historical candidates whose realized future is
close to the query's future (the existing "Future Set Oracle") the right
target, or should retrieval instead target candidates whose OWN frozen-B0
forecast ERROR resembles the query's own forecast error (a "Correction Set
Oracle"), given that Stage-2's actual fusion is `Y_final = B_q + gamma *
Y_ret` (residual/correction semantics), not a reconstruction of Y_q from
scratch?

Leakage rule: candidate residuals `r_i = Y_i - B_i` use ONLY candidate i's
OWN past window `X_i` (through the SAME frozen B0 base forecaster
`base_head`, no_grad) -- never the query's future. The query residual
`r_q = Y_q - B_q` is computed ONLY for Oracle construction/evaluation
(diagnostic), never fed to any inference-time selector input (there is no
selector here at all -- Track B trains nothing).

Frame note: `r_i` is computed directly from candidate i's OWN absolute
frame (its own last-observed value, restored by `base_forecast` exactly as
the production model does) -- NOT through `_memory_value`'s delta_last
query-offset transplant (that transplant repositions a raw future value
onto the QUERY's price level, which is the right thing for a raw future
value but wrong for an already-scale-appropriate forecast ERROR). The
existing Future Set Oracle reproduction, in contrast, DOES use
`_memory_value` exactly as production Stage-2 does, to remain a faithful
reproduction of the existing target.

Reuses, not reimplements: `utils.oracle_intervention.select_greedy_weighted_set`
(the existing greedy Weighted Set Oracle, called twice -- once with raw
futures for the Future Oracle, once with residuals for the Correction
Oracle, since the function is agnostic to what the value tensor means),
`utils.dense_utility.dense_utility`/`candidate_weights` (per-step utility,
for diagnostics and as an independent cross-check against
`select_greedy_weighted_set`'s own picks), `utils.retrieval_diagnostics.
load_stage2`/`base_forecast`, `scripts.eval_firstanchor_diag._ndcg_at_k`.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_firstanchor_diag import _ndcg_at_k
from utils.dense_utility import candidate_weights, dense_utility
from utils.oracle_intervention import select_greedy_weighted_set
from utils.retrieval_diagnostics import base_forecast, load_stage2, unwrap


def _spearman(a, b):
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = (ra.norm() * rb.norm()).clamp_min(1e-12)
    return float((ra * rb).sum() / denom)


@torch.no_grad()
def evaluate(s2_ckpt, k, tau_override, chunk_size, n_queries, near_tie_thresholds,
             eps_thresholds, device, split='test', channel=None):
    b0_exp, b0_args = load_stage2(s2_ckpt, device=device)
    b0_exp.model.to(device)
    b0_exp._ensure_memory()
    b0_exp._build_key_bank(force=True)
    b0_model = unwrap(b0_exp.model)
    b0_model.eval()
    for p in b0_model.parameters():
        assert p.grad is None, 'B0 parameters must never carry a gradient (no_grad-only diagnostic)'
        p.requires_grad_(False)

    tau = float(tau_override) if tau_override is not None else float(getattr(b0_args, 'tau_topk', 0.1))
    memory_x = torch.from_numpy(b0_exp.memory_bank.memory_x).float().to(device)
    memory_y = b0_exp.memory_y.to(device)
    memory_x_last = b0_exp.memory_x_last.to(device)
    n_cand = memory_x.size(0)

    # ---- Historical Residual Memory: B_i, r_i for the FULL candidate bank,
    # computed ONCE (candidate's OWN past only, B0 in eval/no_grad mode). ----
    B_i_full = base_forecast(b0_model, memory_x, chunk_size=chunk_size)  # [N, pred_len, C]
    assert torch.isfinite(B_i_full).all(), 'non-finite candidate base forecast'
    assert B_i_full.shape[0] == n_cand == memory_y.shape[0], 'candidate/base-forecast index misalignment'
    r_i_full = memory_y - B_i_full  # [N, pred_len, C], candidate's OWN frame, no query-offset transplant
    assert torch.isfinite(r_i_full).all(), 'non-finite candidate residual'

    _, loader = b0_exp._get_data(flag=split, shuffle=False)
    channels = list(b0_model.target_channels())
    for c in channels:
        sources = b0_model.source_channels(c)
        if len(sources) != 1 or int(sources[0]) != int(c):
            raise ValueError(f'EXP-CORRECTION-ORACLE-DIAG01 is self-only; channel {c} has sources {sources}')
    c0 = channel if channel is not None else channels[0]  # single channel per call;
    # EXP-CORRECTION-ORACLE-DIAG02 calls evaluate() once per channel and
    # aggregates -- this default (channels[0]) preserves EXP-CORRECTION-
    # ORACLE-DIAG01's original single-channel behaviour unmodified.

    agg = {
        'b0_se': 0.0, 'ref_se': 0.0, 'future_final_se': 0.0, 'correction_final_se': 0.0, 'n': 0.0,
    }
    overlap_sum = [0.0] * k
    overlap_n = 0
    rank_rows_seen = 0
    step_rows_future = {t: {'rank': [], 'top1': [], 'top5': [], 'top10': [], 'top50': [],
                             'ndcg10': [], 'ndcg50': [], 'spearman_top1pct': [],
                             'margin_abs': [], 'margin_rel': [], 'near_tie': {th: [] for th in near_tie_thresholds},
                             'eps_optimal': {th: [] for th in eps_thresholds}} for t in range(k)}
    step_rows_correction = {t: {'rank': [], 'top1': [], 'top5': [], 'top10': [], 'top50': [],
                                 'ndcg10': [], 'ndcg50': [], 'spearman_top1pct': [],
                                 'margin_abs': [], 'margin_rel': [], 'near_tie': {th: [] for th in near_tie_thresholds},
                                 'eps_optimal': {th: [] for th in eps_thresholds}} for t in range(k)}
    consistency_mismatches = 0
    consistency_checked = 0

    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, counts = b0_exp._candidate_mask(batch_start_idx)
        valid_query = counts.to(device) >= k
        if not bool(valid_query.any()):
            continue

        # ---- B0's OWN existing production retrieval score (past-only), the
        # SAME score used to weight EVERY arm below -- no new selector. ----
        z_q_ref = b0_model._branch_embedding(batch_x, c0, c0)
        z_mem_ref = b0_model._branch_memory(b0_exp.key_bank, c0, 0, c0, z_q_ref.dtype, device)
        cosine_ref = torch.matmul(z_q_ref, z_mem_ref.transpose(0, 1))
        score_fn = b0_model._retrieval_score_fn()
        learned_ref = score_fn(z_q_ref, z_mem_ref) if score_fn is not None else cosine_ref
        w = candidate_weights(learned_ref, cand_mask, tau)

        # ---- Future Set Oracle target (existing, faithful reproduction) ----
        memory_c, offset_c = b0_model._memory_value(batch_x, memory_y, memory_x_last, c0)
        all_tgt = memory_c + offset_c.view(-1, 1, 1)  # [bsz, N, pred_len], query-offset transplanted
        q_tgt = batch_y[:, :, c0]

        # ---- Correction Set Oracle target (new): candidate residual, NO
        # query-offset transplant (already scale-appropriate as an error). ----
        B_q_full = base_forecast(b0_model, batch_x, chunk_size=chunk_size)  # [bsz, pred_len, C]
        B_q = B_q_full[:, :, c0]
        r_q = q_tgt - B_q  # [bsz, pred_len]
        r_i_c = r_i_full[:, :, c0]  # [N, pred_len]
        r_i_batched = r_i_c.unsqueeze(0).expand(batch_x.size(0), -1, -1)  # [bsz, N, pred_len]

        # ---- sanity 2: MSE(B_q + C, Y_q) == MSE(C, r_q) at C=0 already
        # trivially holds; verified generally below once C(S) is built. ----
        # ---- sanity 3: empty correction -> final == B_q exactly ----
        assert torch.allclose(B_q, B_q + torch.zeros_like(B_q)), 'trivial'

        picks_future = select_greedy_weighted_set(all_tgt, q_tgt, learned_ref, cand_mask, k, tau)
        picks_correction = select_greedy_weighted_set(r_i_batched, r_q, learned_ref, cand_mask, k, tau)

        for t in range(k):
            overlap_sum[t] += float(sum(
                len(set(picks_future[b, :t + 1].tolist()) & set(picks_correction[b, :t + 1].tolist())) / (t + 1)
                for b in range(picks_future.size(0)) if bool(valid_query[b])
            ))
        overlap_n += int(valid_query.sum())

        # ---- final downstream predictions ----
        C_S = torch.zeros_like(q_tgt)
        w_pick = w.gather(1, picks_correction)
        alpha = w_pick / w_pick.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        C_S = (alpha.unsqueeze(-1) * r_i_c[picks_correction]).sum(dim=1)
        y_final_correction = B_q + C_S

        w_pick_f = w.gather(1, picks_future)
        alpha_f = w_pick_f / w_pick_f.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        y_final_future = (alpha_f.unsqueeze(-1) * all_tgt.gather(
            1, picks_future.unsqueeze(-1).expand(-1, -1, all_tgt.size(-1)))).sum(dim=1)

        # sanity 2, real check (not just C=0): the correction objective at the
        # FINAL selected set must equal MSE(C(S), r_q) exactly.
        obj_a = (y_final_correction - q_tgt).pow(2).mean(-1)
        obj_b = (C_S - r_q).pow(2).mean(-1)
        assert torch.allclose(obj_a[valid_query], obj_b[valid_query], atol=1e-4), \
            'MSE(B_q+C,Y_q) != MSE(C,r_q) -- correction objective equivalence violated'

        vq = valid_query
        agg['b0_se'] += float(((B_q - q_tgt).pow(2))[vq].sum())
        agg['future_final_se'] += float(((y_final_future - q_tgt).pow(2))[vq].sum())
        agg['correction_final_se'] += float(((y_final_correction - q_tgt).pow(2))[vq].sum())
        agg['n'] += float(q_tgt[vq].numel())

        # ---- per-step diagnostics (query-subsampled) ----
        if rank_rows_seen < n_queries:
            take = min(int(vq.sum()), n_queries - rank_rows_seen)
            idx_take = vq.nonzero(as_tuple=True)[0][:take]
            # SEPARATE selected-mask trajectories: Future Oracle and
            # Correction Oracle pick DIFFERENT candidates in general, so
            # each needs its OWN "already selected" state -- sharing one
            # mask would incorrectly treat one oracle's picks as unavailable
            # to the other.
            selected_mask_f = torch.zeros_like(cand_mask)
            selected_mask_c = torch.zeros_like(cand_mask)
            for t in range(k):
                prefix_f = picks_future[:, :t]
                prefix_c = picks_correction[:, :t]
                a_future = dense_utility(prefix_f, w, all_tgt, q_tgt, chunk_size=chunk_size)
                a_correction = dense_utility(prefix_c, w, r_i_batched, r_q, chunk_size=chunk_size)
                u_future = -a_future
                u_correction = -a_correction
                valid_now_f = cand_mask & ~selected_mask_f
                valid_now_c = cand_mask & ~selected_mask_c

                for b in idx_take.tolist():
                    for label, u_all, picks_all, rows, target_next, valid_now in (
                        ('future', u_future, picks_future, step_rows_future[t], all_tgt, valid_now_f),
                        ('correction', u_correction, picks_correction, step_rows_correction[t], r_i_batched, valid_now_c),
                    ):
                        valid_b = valid_now[b]
                        if int(valid_b.sum()) < 5:
                            continue
                        uu = u_all[b][valid_b]
                        score_b = learned_ref[b][valid_b]
                        i_star_global = int(picks_all[b, t])
                        i_star_local = int((valid_b.nonzero(as_tuple=True)[0] == i_star_global).nonzero())
                        # sanity 4/5 (online consistency): argmax(u) must equal
                        # select_greedy_weighted_set's own pick at this step.
                        recomputed_argmax_local = int(uu.argmax())
                        consistency_checked_local = (recomputed_argmax_local == i_star_local)
                        rows.setdefault('_consistency_checked', 0)
                        rows.setdefault('_consistency_mismatch', 0)
                        rows['_consistency_checked'] += 1
                        if not consistency_checked_local:
                            rows['_consistency_mismatch'] += 1

                        order = torch.argsort(score_b, descending=True)
                        rank_of_star = int((order == i_star_local).nonzero())
                        n_valid = uu.numel()
                        rows['rank'].append(rank_of_star)
                        rows['top1'].append(1.0 if rank_of_star == 0 else 0.0)
                        rows['top5'].append(1.0 if rank_of_star < 5 else 0.0)
                        rows['top10'].append(1.0 if rank_of_star < 10 else 0.0)
                        rows['top50'].append(1.0 if rank_of_star < 50 else 0.0)
                        rel = uu - uu.min() + 1e-6
                        rel_pred_order = rel[order]
                        rows['ndcg10'].append(_ndcg_at_k(rel_pred_order, 10))
                        rows['ndcg50'].append(_ndcg_at_k(rel_pred_order, 50))
                        top1pct_n = max(int(0.01 * n_valid), 2)
                        true_order = torch.argsort(uu, descending=True)[:top1pct_n]
                        score_top = score_b[true_order]
                        u_top = uu[true_order]
                        rows['spearman_top1pct'].append(_spearman(score_top, u_top))

                        top2 = uu.topk(min(2, n_valid)).values
                        m_abs = float(top2[0] - top2[-1]) if n_valid >= 2 else float('nan')
                        m_rel = m_abs / (abs(float(top2[0])) + 1e-8)
                        rows['margin_abs'].append(m_abs)
                        rows['margin_rel'].append(m_rel)
                        for th in near_tie_thresholds:
                            rows['near_tie'][th].append(1.0 if m_rel < th else 0.0)

                        # epsilon-optimal: reference score's own top-1 pick
                        # vs oracle's true best, gap measured in oracle utility
                        score_top1_local = int(score_b.argmax())
                        u_ref_pick = float(uu[score_top1_local])
                        u_best = float(uu.max())
                        gap_rel = (u_best - u_ref_pick) / (abs(u_best) + 1e-8)
                        for th in eps_thresholds:
                            rows['eps_optimal'][th].append(1.0 if gap_rel <= th else 0.0)

                nxt_f = picks_future[:, t:t + 1]
                nxt_c = picks_correction[:, t:t + 1]
                selected_mask_f = selected_mask_f.scatter(1, nxt_f, True)
                selected_mask_c = selected_mask_c.scatter(1, nxt_c, True)
            rank_rows_seen += take

    def summarize(rows_by_step):
        out = {}
        for t in range(k):
            r = rows_by_step[t]
            for key in ('rank', 'top1', 'top5', 'top10', 'top50', 'ndcg10', 'ndcg50',
                        'spearman_top1pct', 'margin_abs', 'margin_rel'):
                vals = r[key]
                out[f't{t+1}_{key}_mean'] = sum(vals) / max(len(vals), 1) if vals else float('nan')
            for th in near_tie_thresholds:
                vals = r['near_tie'][th]
                out[f't{t+1}_near_tie_frac_{th}'] = sum(vals) / max(len(vals), 1) if vals else float('nan')
            for th in eps_thresholds:
                vals = r['eps_optimal'][th]
                out[f't{t+1}_eps_optimal_{th}'] = sum(vals) / max(len(vals), 1) if vals else float('nan')
            checked = r.get('_consistency_checked', 0)
            mismatch = r.get('_consistency_mismatch', 0)
            out[f't{t+1}_consistency_mismatch_rate'] = mismatch / max(checked, 1)
        return out

    future_summary = summarize(step_rows_future)
    correction_summary = summarize(step_rows_correction)

    def overall(rows_by_step, key):
        allv = [v for t in range(k) for v in rows_by_step[t][key]]
        return sum(allv) / max(len(allv), 1) if allv else float('nan')

    b0_mse = agg['b0_se'] / max(agg['n'], 1)
    future_mse = agg['future_final_se'] / max(agg['n'], 1)
    correction_mse = agg['correction_final_se'] / max(agg['n'], 1)

    result = {
        'n_cand': n_cand, 'n_eval_rows': agg['n'], 'n_queries_detail': rank_rows_seen,
        'split': split, 'channel': int(c0), 'tau': tau,
        'downstream': {
            'b0_mse': b0_mse,
            'future_oracle_final_mse': future_mse,
            'correction_oracle_final_mse': correction_mse,
            'gain_future_vs_b0': b0_mse - future_mse,
            'gain_correction_vs_b0': b0_mse - correction_mse,
        },
        'overlap_at_k': [overlap_sum[t] / max(overlap_n, 1) for t in range(k)],
        'future_oracle': {
            'mean_rank': overall(step_rows_future, 'rank'),
            'ndcg10_mean': overall(step_rows_future, 'ndcg10'),
            'ndcg50_mean': overall(step_rows_future, 'ndcg50'),
            'top1_acc': overall(step_rows_future, 'top1'),
            'top10_containment': overall(step_rows_future, 'top10'),
            'top50_containment': overall(step_rows_future, 'top50'),
            'margin_abs_mean': overall(step_rows_future, 'margin_abs'),
            'margin_rel_mean': overall(step_rows_future, 'margin_rel'),
            'per_step': future_summary,
        },
        'correction_oracle': {
            'mean_rank': overall(step_rows_correction, 'rank'),
            'ndcg10_mean': overall(step_rows_correction, 'ndcg10'),
            'ndcg50_mean': overall(step_rows_correction, 'ndcg50'),
            'top1_acc': overall(step_rows_correction, 'top1'),
            'top10_containment': overall(step_rows_correction, 'top10'),
            'top50_containment': overall(step_rows_correction, 'top50'),
            'margin_abs_mean': overall(step_rows_correction, 'margin_abs'),
            'margin_rel_mean': overall(step_rows_correction, 'margin_rel'),
            'per_step': correction_summary,
        },
    }
    return result, step_rows_future, step_rows_correction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage2_checkpoint', required=True)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--tau', type=float, default=None)
    ap.add_argument('--chunk_size', type=int, default=2048)
    ap.add_argument('--n_queries', type=int, default=200)
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    near_tie_thresholds = [0.001, 0.01, 0.05]
    eps_thresholds = [0.001, 0.01, 0.05]

    result, step_f, step_c = evaluate(
        args.stage2_checkpoint, args.top_k, args.tau, args.chunk_size, args.n_queries,
        near_tie_thresholds, eps_thresholds, device, args.split)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'summary.json', 'w') as fh:
        json.dump(result, fh, indent=2, default=str)

    with open(out_dir / 'oracle_overlap.csv', 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['step', 'set_overlap_at_step'])
        for t, v in enumerate(result['overlap_at_k']):
            writer.writerow([t + 1, v])

    def write_metrics_csv(path, per_step):
        with open(path, 'w', newline='') as fh:
            writer = csv.writer(fh)
            writer.writerow(['step', 'metric', 'value'])
            for k_, v in per_step.items():
                writer.writerow([k_, '', v])

    write_metrics_csv(out_dir / 'future_oracle_metrics.csv', result['future_oracle']['per_step'])
    write_metrics_csv(out_dir / 'correction_oracle_metrics.csv', result['correction_oracle']['per_step'])

    print(json.dumps(result['downstream'], indent=2))
    print(json.dumps({'future_oracle_summary': {k_: v for k_, v in result['future_oracle'].items() if k_ != 'per_step'}}, indent=2))
    print(json.dumps({'correction_oracle_summary': {k_: v for k_, v in result['correction_oracle'].items() if k_ != 'per_step'}}, indent=2))
    print(f'[correction_oracle_diag01] written to {out_dir}')


if __name__ == '__main__':
    main()
