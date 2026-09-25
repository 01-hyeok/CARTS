#!/usr/bin/env python3
"""Experiment E, Stage E1.5: Shared-conditioned component utility diagnostic.

Still no neural encoder trained -- pure oracle diagnostic, reusing E1's
component distance functions (`diag_experiment_e_teacher01.py::transform`,
`component_distance_memsafe`) unmodified. Two families of arms, both scored
by the SAME final metric (overall future aggregate MSE, uniform mean of a
hard Top-10):

  Diagnostic A (score fusion): u_{alpha,e} = alpha*z(u_shared) +
    (1-alpha)*z(u_e), alpha in {0.5,0.75,0.9,1.0}, e in
    {local,trend,seasonal,equal}. alpha=1.0 must reduce EXACTLY to the
    shared-only baseline (verified by an assertion, not just claimed).

  Diagnostic B (shared shortlist -> component rerank): shortlist =
    Shared's own Top-M (M in {50,100,200}), then re-sort ONLY within that
    shortlist by a component's z-scored utility, take the shortlist's own
    Top-10.

Per-query, per-channel results for every arm (+ the Shared baseline) are
collected so Diagnostic C (fixed-arm vs per-query-oracle comparison) is
just a reduction over the same table -- no separate computation.
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

from models.RelationStage1 import stable_topk_indices
from scripts.diag_experiment_e_teacher01 import component_distance_memsafe
from scripts.train_margutil01 import build_experiment, memory_value

COMPONENTS = ('local', 'trend', 'seasonal')
ALPHAS = (0.50, 0.75, 0.90, 1.00)
SHORTLIST_M = (50, 100, 200)
EPS = 1e-8


def zscore(u, valid_mask):
    n_valid = valid_mask.sum(-1, keepdim=True).clamp_min(1).float()
    u_masked = u.masked_fill(~valid_mask, 0.0)
    mean = u_masked.sum(-1, keepdim=True) / n_valid
    sq = ((u - mean) ** 2).masked_fill(~valid_mask, 0.0)
    std = (sq.sum(-1, keepdim=True) / n_valid).clamp_min(EPS).sqrt()
    return (u - mean) / std


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True)
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--pred_len', type=int, default=720)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--chunk_size', type=int, default=2048)
    ap.add_argument('--split', required=True, choices=['train', 'val', 'test'])
    ap.add_argument('--period_json', required=True)
    ap.add_argument('--out_dir', default='outputs/experiment_E/diagnostics/E1_5')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--max_queries', type=int, default=0)
    cli = ap.parse_args()

    period = int(json.loads(Path(cli.period_json).read_text())['median_period'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(cli.out_dir) / cli.cell / cli.split
    out_dir.mkdir(parents=True, exist_ok=True)

    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.pred_len, 'batch_size': cli.batch_size, 'seed': 0,
    })
    exp._ensure_memory()
    channels = list(range(int(args.enc_in)))
    memory_y, memory_x_last = exp.memory_y, exp.memory_x_last

    arm_names = ['shared']
    for e in COMPONENTS:
        for a in ALPHAS:
            arm_names.append(f'fusion_{e}_a{a}')
    for a in ALPHAS:
        arm_names.append(f'fusion_equal_a{a}')
    for e in COMPONENTS:
        for m in SHORTLIST_M:
            arm_names.append(f'rerank_{e}_M{m}')
    for m in SHORTLIST_M:
        arm_names.append(f'rerank_equal_M{m}')

    _, loader = exp._get_data(flag=cli.split, shuffle=False)
    n_seen = 0
    rows = []  # (start, channel, {arm: mse})

    for batch_x, batch_y, batch_start_idx in loader:
        if cli.max_queries and n_seen >= cli.max_queries:
            break
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        bsz = batch_x.size(0)
        batch_arm_mse = {name: torch.zeros(bsz, len(channels)) for name in arm_names}

        for c in channels:
            memory_c, offset_c = memory_value(args, batch_x, memory_y, memory_x_last, c)
            query_future = batch_y[:, :, c]

            d_shared = component_distance_memsafe(memory_c, offset_c, query_future, 'shared', period,
                                                   cli.chunk_size).masked_fill(~cand_mask, float('inf'))
            u_shared = -d_shared.masked_fill(~cand_mask, 0.0)
            z_shared = zscore(u_shared, cand_mask)

            d_comp, u_comp, z_comp = {}, {}, {}
            for e in COMPONENTS:
                d_e = component_distance_memsafe(memory_c, offset_c, query_future, e, period,
                                                  cli.chunk_size).masked_fill(~cand_mask, float('inf'))
                d_comp[e] = d_e
                u_e = -d_e.masked_fill(~cand_mask, 0.0)
                u_comp[e] = u_e
                z_comp[e] = zscore(u_e, cand_mask)
            z_equal = sum(z_comp[e] for e in COMPONENTS) / len(COMPONENTS)

            def eval_picks(picks, name):
                y_sel = memory_c[picks] + offset_c.view(-1, 1, 1)
                agg = y_sel.mean(dim=1)
                mse = ((agg - query_future) ** 2).mean(dim=-1)
                batch_arm_mse[name][:, c] = mse.cpu()

            # ---- Shared baseline ----
            picks_shared = stable_topk_indices(z_shared.masked_fill(~cand_mask, float('-inf')),
                                               cli.top_k, largest=True)
            eval_picks(picks_shared, 'shared')

            # ---- Diagnostic A: score fusion ----
            for e in list(COMPONENTS) + ['equal']:
                z_e = z_equal if e == 'equal' else z_comp[e]
                for a in ALPHAS:
                    score = a * z_shared + (1 - a) * z_e
                    score = score.masked_fill(~cand_mask, float('-inf'))
                    picks = stable_topk_indices(score, cli.top_k, largest=True)
                    if a == 1.00:
                        assert torch.equal(picks, picks_shared), 'alpha=1.0 must reduce to shared-only exactly'
                    eval_picks(picks, f'fusion_{e}_a{a}')

            # ---- Diagnostic B: shared shortlist -> component rerank ----
            for m in SHORTLIST_M:
                shortlist = stable_topk_indices(z_shared.masked_fill(~cand_mask, float('-inf')),
                                                m, largest=True)  # [B, M]
                for e in list(COMPONENTS) + ['equal']:
                    z_e = z_equal if e == 'equal' else z_comp[e]
                    z_e_short = z_e.gather(1, shortlist)  # [B, M]
                    local_top = stable_topk_indices(z_e_short, cli.top_k, largest=True)  # indices into shortlist
                    picks = shortlist.gather(1, local_top)  # map back to candidate indices
                    # subset check: every pick must be in the shortlist
                    assert bool((picks.unsqueeze(-1) == shortlist.unsqueeze(-2)).any(-1).all()), \
                        'reranked picks must be a subset of the shortlist'
                    eval_picks(picks, f'rerank_{e}_M{m}')

        for b in range(bsz):
            rows.append((int(batch_start_idx[b]), {name: float(batch_arm_mse[name][b].mean())
                                                    for name in arm_names}))
        n_seen += bsz

    # ---- per-(query,arm) mean-over-channels already folded in; build summary ----
    arm_means = {name: sum(r[1][name] for r in rows) / max(len(rows), 1) for name in arm_names}
    shared_mean = arm_means['shared']
    best_fixed_arm = min(arm_means, key=arm_means.get)
    best_fixed_mean = arm_means[best_fixed_arm]

    oracle_vals = [min(r[1].values()) for r in rows]
    oracle_mean = sum(oracle_vals) / max(len(oracle_vals), 1)

    winner_counts = {name: 0 for name in arm_names}
    for _, vals in rows:
        winner_counts[min(vals, key=vals.get)] += 1
    n_rows = len(rows)
    winner_share = {name: winner_counts[name] / max(n_rows, 1) for name in arm_names}

    with open(out_dir / 'per_query_arm_metrics.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['query_start_idx'] + arm_names + ['winner'])
        for start, vals in rows:
            w.writerow([start] + [vals[n] for n in arm_names] + [min(vals, key=vals.get)])

    summary = {
        'cell': cli.cell, 'split': cli.split, 'period': period, 'top_k': cli.top_k,
        'n_queries': n_rows, 'n_channels': len(channels),
        'arm_mean_overall_mse': arm_means,
        'shared_mean': shared_mean,
        'best_fixed_arm': best_fixed_arm, 'best_fixed_mean': best_fixed_mean,
        'best_fixed_vs_shared_improve_pct': (shared_mean - best_fixed_mean) / shared_mean * 100.0,
        'scale_oracle_mean': oracle_mean,
        'oracle_vs_shared_improve_pct': (shared_mean - oracle_mean) / shared_mean * 100.0,
        'oracle_vs_best_fixed_improve_pct': (best_fixed_mean - oracle_mean) / best_fixed_mean * 100.0,
        'winner_share': winner_share,
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    top5_arms = sorted(arm_means.items(), key=lambda kv: kv[1])[:5]
    print(f"[E1_5] {cli.cell}/{cli.split}: shared={shared_mean:.6f} "
         f"best_fixed={best_fixed_arm}({best_fixed_mean:.6f}, {summary['best_fixed_vs_shared_improve_pct']:.2f}%) "
         f"oracle={oracle_mean:.6f} ({summary['oracle_vs_shared_improve_pct']:.2f}% vs shared) "
         f"top5_arms={top5_arms}")


if __name__ == '__main__':
    main()
