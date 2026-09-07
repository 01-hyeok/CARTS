#!/usr/bin/env python3
"""EXP-CONTINUATION-DIAG: exhaustive t=2 continuation diagnostic.

No new training. Reuses EXP-MARGUTIL01 checkpoints and EXP-FIRSTANCHOR-DIAG's
own first-anchor machinery (`run_arm`, `a_weighted_prefix`, B0 production
score, candidate masking) verbatim, via direct import from
`scripts/eval_firstanchor_diag.py` -- not reimplemented.

For a fixed first candidate i1 (one of dense_first / b0_first / oracle_first,
exactly as EXP-FIRSTANCHOR-DIAG defines them), this computes, EXHAUSTIVELY
over every valid remaining candidate, A(S1 + {i}) using the same closed-form
`utils.dense_utility.dense_utility` the Weighted Set Oracle teacher itself
uses -- then compares the Dense selector's own t=2 pick against the true
best (oracle) t=2 pick, both in outcome (A2) and in rank (where does each
land in the other's ordering).
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

from scripts.eval_firstanchor_diag import (a_weighted_prefix, encode,
                                            load_trained_selector, run_arm)
from utils.dense_utility import candidate_weights, dense_utility
from utils.retrieval_diagnostics import load_stage2

EPS = 1e-12


def _spearman(a, b):
    if a.numel() < 2 or torch.allclose(a, a[0]) or torch.allclose(b, b[0]):
        return float('nan')
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = (ra.norm() * rb.norm()).clamp_min(EPS)
    return float((ra * rb).sum() / denom)


def _rank_of(value_idx, order):
    """0-based rank of `value_idx` within `order` (a 1D index tensor sorted
    best-first)."""
    hit = (order == value_idx).nonzero(as_tuple=True)[0]
    return int(hit[0]) if hit.numel() else -1


@torch.no_grad()
def evaluate_cell(s2_ckpt, seq_ckpt, anchor_policy, k, query_budget, chunk_size, device):
    b0_exp, b0_args = load_stage2(s2_ckpt)
    b0_exp._ensure_memory()
    b0_exp._build_key_bank()
    b0_model = b0_exp.model.module if hasattr(b0_exp.model, 'module') else b0_exp.model
    memory_y = b0_exp.memory_y.to(device)
    memory_x_last = b0_exp.memory_x_last.to(device)
    memory_x = torch.from_numpy(b0_exp.memory_bank.memory_x).float().to(device)

    seq_model, set_cond, empty_token, utility_head, seq_args = load_trained_selector(seq_ckpt, device)
    tau = float(seq_args.tau_topk)
    _, loader = b0_exp._get_data(flag='test', shuffle=False)
    channels = list(b0_model.target_channels())

    rows = []
    sanity = {}
    seen = 0

    for batch_x, batch_y, batch_start_idx in loader:
        if seen >= query_budget:
            break
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, counts = b0_exp._candidate_mask(batch_start_idx)
        valid_query = counts.to(device) >= k

        for c in channels:
            if seen >= query_budget:
                break
            E = encode(seq_model, memory_x, c)
            q = encode(seq_model, batch_x, c)

            memory_c, offset_c = b0_model._memory_value(batch_x, memory_y, memory_x_last, c)
            all_tgt = memory_c + offset_c.view(-1, 1, 1)
            q_tgt = batch_y[:, :, c]

            z_q_ref = b0_model._branch_embedding(batch_x, c, c)
            z_mem_ref = b0_model._branch_memory(b0_exp.key_bank, c, 0, c, z_q_ref.dtype, device)
            cosine_ref = torch.matmul(z_q_ref, z_mem_ref.transpose(0, 1))
            score_fn = b0_model._retrieval_score_fn()
            learned_ref = score_fn(z_q_ref, z_mem_ref) if score_fn is not None else cosine_ref

            dist_individual = (all_tgt - q_tgt.unsqueeze(1)).pow(2).mean(-1)
            dist_individual = dist_individual.masked_fill(~cand_mask, float('inf'))
            i1_oracle = dist_individual.argmin(dim=-1)
            b0_neg = torch.finfo(learned_ref.dtype).min / 4
            b0_masked = learned_ref.masked_fill(~cand_mask, b0_neg)
            i1_b0 = b0_masked.argmax(dim=-1)

            first_pick = {'dense_first': None, 'b0_first': i1_b0, 'oracle_first': i1_oracle}[anchor_policy]
            # Reuses run_arm exactly as EXP-FIRSTANCHOR-DIAG does -- i1/i2_dense
            # here are BY CONSTRUCTION identical to that experiment's own
            # trajectory (sanity check #8): same function, same inputs, k=2
            # simply stops earlier at the same point that run.
            picks2 = run_arm(q, E, cand_mask, set_cond, empty_token, utility_head, 2, first_pick=first_pick)
            i1 = picks2[:, 0]
            i2_dense = picks2[:, 1]

            w = candidate_weights(learned_ref, cand_mask, tau)
            prefix1 = i1.unsqueeze(1)
            a2_all = dense_utility(prefix1, w, all_tgt, q_tgt, chunk_size=chunk_size)
            selected_after_1 = torch.zeros_like(cand_mask).scatter(1, prefix1, True)
            remaining_valid = cand_mask & ~selected_after_1
            a2_all_masked = a2_all.masked_fill(~remaining_valid, float('inf'))
            i2_oracle = a2_all_masked.argmin(dim=-1)
            a2_oracle = a2_all_masked.gather(1, i2_oracle.unsqueeze(-1)).squeeze(1)

            a1 = a_weighted_prefix(prefix1, learned_ref, all_tgt, q_tgt, tau)
            a2_dense = a_weighted_prefix(picks2, learned_ref, all_tgt, q_tgt, tau)
            # Sanity: a2_dense must equal a2_all gathered at i2_dense (same
            # formula via two different code paths -- prefix-softmax vs the
            # incremental closed form).
            a2_dense_check = a2_all.gather(1, i2_dense.unsqueeze(-1)).squeeze(1)
            sanity.setdefault('a2_dense_vs_a2_all_max_abs_diff', 0.0)
            sanity['a2_dense_vs_a2_all_max_abs_diff'] = max(
                sanity['a2_dense_vs_a2_all_max_abs_diff'],
                float((a2_dense - a2_dense_check)[valid_query].abs().max()) if valid_query.any() else 0.0)

            prefix2_oracle = torch.stack([i1, i2_oracle], dim=1)
            a2_oracle_check = a_weighted_prefix(prefix2_oracle, learned_ref, all_tgt, q_tgt, tau)
            sanity.setdefault('a2_oracle_prefix_vs_dense_utility_max_abs_diff', 0.0)
            sanity['a2_oracle_prefix_vs_dense_utility_max_abs_diff'] = max(
                sanity['a2_oracle_prefix_vs_dense_utility_max_abs_diff'],
                float((a2_oracle - a2_oracle_check)[valid_query].abs().max()) if valid_query.any() else 0.0)

            delta_dense = a2_dense - a1
            delta_oracle = a2_oracle - a1
            continuation_regret = a2_dense - a2_oracle
            ratio_dense = a2_dense / (a1 + EPS)

            # ---- predicted utility at t=2 (for rank / spearman), Dense's own sign convention: higher = better ----
            m = E[prefix1].mean(dim=1)
            h_t2 = set_cond(q, m)
            u_hat2 = utility_head(h_t2, E)
            true_gain = a1.unsqueeze(-1) - a2_all  # higher = better, matches u_hat's convention

            # ---- alphas / b0 scores ----
            def alpha_of(prefix):
                sc = learned_ref.gather(1, prefix)
                return torch.softmax(sc / tau, dim=-1)
            alpha1_dense = alpha_of(prefix1)[:, 0]
            alpha_d2 = alpha_of(picks2)
            alpha1_d, alpha2_dense = alpha_d2[:, 0], alpha_d2[:, 1]
            alpha_o2 = alpha_of(prefix2_oracle)
            alpha1_oracle, alpha2_oracle = alpha_o2[:, 0], alpha_o2[:, 1]

            b0_score_i1 = learned_ref.gather(1, i1.unsqueeze(-1)).squeeze(1)
            b0_score_i2_dense = learned_ref.gather(1, i2_dense.unsqueeze(-1)).squeeze(1)
            b0_score_i2_oracle = learned_ref.gather(1, i2_oracle.unsqueeze(-1)).squeeze(1)

            first_mse = dist_individual.gather(1, i1.unsqueeze(-1)).squeeze(1)
            dense2_mse = dist_individual.gather(1, i2_dense.unsqueeze(-1)).squeeze(1)
            oracle2_mse = dist_individual.gather(1, i2_oracle.unsqueeze(-1)).squeeze(1)

            valid_idx = valid_query.nonzero(as_tuple=True)[0].tolist()
            for b in valid_idx:
                if seen >= query_budget:
                    break
                rv = remaining_valid[b]
                n_remaining = int(rv.sum())
                if n_remaining < 2:
                    continue
                tg = true_gain[b][rv]
                uh = u_hat2[b][rv]
                candidate_ids = rv.nonzero(as_tuple=True)[0]

                order_true_desc = candidate_ids[torch.argsort(tg, descending=True)]
                order_pred_desc = candidate_ids[torch.argsort(uh, descending=True)]

                r_dense_true = _rank_of(int(i2_dense[b]), order_true_desc)
                r_oracle_pred = _rank_of(int(i2_oracle[b]), order_pred_desc)
                r_dense_true_frac = r_dense_true / max(n_remaining - 1, 1)
                r_oracle_pred_frac = r_oracle_pred / max(n_remaining - 1, 1)

                def _tail_spearman(frac_or_n, is_frac):
                    n_tail = max(int(frac_or_n * n_remaining), 2) if is_frac else min(frac_or_n, n_remaining)
                    top_ids_local = torch.argsort(tg, descending=True)[:n_tail]
                    return _spearman(uh[top_ids_local], tg[top_ids_local])

                sp_all = _spearman(uh, tg)
                sp_top10pct = _tail_spearman(0.10, True)
                sp_top5pct = _tail_spearman(0.05, True)
                sp_top1pct = _tail_spearman(0.01, True)
                sp_top50 = _tail_spearman(50, False)
                sp_top10 = _tail_spearman(10, False)

                rows.append({
                    'query_id': f'{int(batch_start_idx[b])}_{c}',
                    'anchor_policy': anchor_policy, 'channel': int(c),
                    'i1': int(i1[b]), 'i2_dense': int(i2_dense[b]), 'i2_oracle': int(i2_oracle[b]),
                    'A1': float(a1[b]), 'A2_dense': float(a2_dense[b]), 'A2_oracle': float(a2_oracle[b]),
                    'delta_dense': float(delta_dense[b]), 'delta_oracle': float(delta_oracle[b]),
                    'continuation_regret': float(continuation_regret[b]),
                    'dense_second_true_rank': r_dense_true, 'dense_second_true_rank_fraction': r_dense_true_frac,
                    'oracle_second_predicted_rank': r_oracle_pred, 'oracle_second_predicted_rank_fraction': r_oracle_pred_frac,
                    'n_remaining_valid': n_remaining,
                    'first_candidate_future_mse': float(first_mse[b]),
                    'dense_second_candidate_future_mse': float(dense2_mse[b]),
                    'oracle_second_candidate_future_mse': float(oracle2_mse[b]),
                    'alpha1_dense': float(alpha1_d[b]), 'alpha2_dense': float(alpha2_dense[b]),
                    'alpha1_oracle': float(alpha1_oracle[b]), 'alpha2_oracle': float(alpha2_oracle[b]),
                    'b0_score_i1': float(b0_score_i1[b]), 'b0_score_i2_dense': float(b0_score_i2_dense[b]),
                    'b0_score_i2_oracle': float(b0_score_i2_oracle[b]),
                    'spearman_all': sp_all, 'spearman_true_top10pct': sp_top10pct,
                    'spearman_true_top5pct': sp_top5pct, 'spearman_true_top1pct': sp_top1pct,
                    'spearman_true_top50': sp_top50, 'spearman_true_top10': sp_top10,
                    'ratio_dense': float(ratio_dense[b]),
                })
                seen += 1

    return rows, sanity


def summarize(rows):
    import statistics as st
    n = len(rows)
    if n == 0:
        return {}

    def col(name, finite_only=False):
        vals = [r[name] for r in rows]
        if finite_only:
            vals = [v for v in vals if v == v]  # drop NaN
        return vals

    def frac(pred):
        return sum(1 for r in rows if pred(r)) / n

    def rank_stats(field, frac_field):
        vals = col(field)
        return {
            f'{field}_mean': st.mean(vals), f'{field}_median': st.median(vals),
            f'{field}_top1': frac(lambda r: r[field] == 0),
            f'{field}_top5': frac(lambda r: r[field] < 5),
            f'{field}_top10': frac(lambda r: r[field] < 10),
            f'{field}_top50': frac(lambda r: r[field] < 50),
            f'{field}_top1pct': frac(lambda r: r[frac_field] < 0.01),
        }

    out = {
        'n_queries': n,
        'mean_A1': st.mean(col('A1')), 'mean_A2_dense': st.mean(col('A2_dense')),
        'mean_A2_oracle': st.mean(col('A2_oracle')),
        'mean_delta_dense': st.mean(col('delta_dense')), 'median_delta_dense': st.median(col('delta_dense')),
        'mean_delta_oracle': st.mean(col('delta_oracle')), 'median_delta_oracle': st.median(col('delta_oracle')),
        'mean_continuation_regret': st.mean(col('continuation_regret')),
        'median_continuation_regret': st.median(col('continuation_regret')),
        'dense_improve_frac': frac(lambda r: r['delta_dense'] < 0),
        'dense_hurt_frac': frac(lambda r: r['delta_dense'] > 0),
        'oracle_improve_frac': frac(lambda r: r['delta_oracle'] < 0),
        'dense_hurts_but_oracle_can_improve_frac': frac(lambda r: r['delta_dense'] > 0 and r['delta_oracle'] < 0),
    }
    out.update(rank_stats('dense_second_true_rank', 'dense_second_true_rank_fraction'))
    out.update(rank_stats('oracle_second_predicted_rank', 'oracle_second_predicted_rank_fraction'))
    for f in ('spearman_all', 'spearman_true_top10pct', 'spearman_true_top5pct',
              'spearman_true_top1pct', 'spearman_true_top50', 'spearman_true_top10'):
        vals = col(f, finite_only=True)
        out[f'{f}_mean'] = st.mean(vals) if vals else float('nan')
        out[f'{f}_median'] = st.median(vals) if vals else float('nan')
        out[f'{f}_n_valid'] = len(vals)

    hurt_alphas = [r['alpha2_dense'] for r in rows if r['delta_dense'] > 0]
    improve_alphas = [r['alpha2_dense'] for r in rows if r['delta_dense'] < 0]
    out['mean_alpha2_dense_given_hurt'] = st.mean(hurt_alphas) if hurt_alphas else float('nan')
    out['median_alpha2_dense_given_hurt'] = st.median(hurt_alphas) if hurt_alphas else float('nan')
    out['mean_alpha2_dense_given_improve'] = st.mean(improve_alphas) if improve_alphas else float('nan')
    out['median_alpha2_dense_given_improve'] = st.median(improve_alphas) if improve_alphas else float('nan')
    a2 = torch.tensor(col('alpha2_dense'))
    dd = torch.tensor(col('delta_dense'))
    out['corr_alpha2_dense_vs_delta_dense'] = _spearman(a2, dd)

    ratios = col('ratio_dense')
    ratios_sorted = sorted(ratios)
    out['ratio_dense_median'] = st.median(ratios)
    out['ratio_dense_p90'] = ratios_sorted[int(0.90 * (n - 1))]
    out['ratio_dense_p95'] = ratios_sorted[int(0.95 * (n - 1))]
    out['ratio_dense_p99'] = ratios_sorted[int(0.99 * (n - 1))]
    out['ratio_dense_max'] = ratios_sorted[-1]
    for thr in (2, 5, 10, 20):
        out[f'ratio_dense_gt_{thr}_frac'] = frac(lambda r, t=thr: r['ratio_dense'] > t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage2_checkpoint', required=True)
    ap.add_argument('--sequential_checkpoint', required=True)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--query_budget', type=int, default=500)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--cell_name', required=True)
    ap.add_argument('--anchor_policy', required=True, choices=['dense_first', 'b0_first', 'oracle_first'])
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    rows, sanity = evaluate_cell(args.stage2_checkpoint, args.sequential_checkpoint,
                                  args.anchor_policy, args.top_k, args.query_budget,
                                  args.chunk_size, device)
    summary = summarize(rows)
    summary['_sanity'] = sanity

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f'continuation_diag_{args.cell_name}_{args.anchor_policy}.csv'
    if rows:
        with open(csv_path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    with open(out_dir / f'{args.cell_name}_{args.anchor_policy}_summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2, default=str)

    # top-20 catastrophic (highest ratio_dense)
    top20 = sorted(rows, key=lambda r: -r['ratio_dense'])[:20]
    with open(out_dir / f'{args.cell_name}_{args.anchor_policy}_top20_catastrophic.json', 'w') as fh:
        json.dump(top20, fh, indent=2, default=str)

    print(json.dumps(summary, indent=2, default=str))
    print(f'[continuation-diag] {len(rows)} queries written to {csv_path}')


if __name__ == '__main__':
    main()
