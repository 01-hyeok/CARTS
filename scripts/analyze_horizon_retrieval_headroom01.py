#!/usr/bin/env python3
"""TRACK-A-HORIZON-RETRIEVAL-HEADROOM01 -- analysis: Global vs Block Oracle
headroom, budget controls, overlap/rank-displacement, query-level benefit
distribution, paired bootstrap CIs. Reads the TEST split's query_level CSV.
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

N_BOOTSTRAP = 2000


def read_csv(path):
    with open(path, newline='') as fh:
        return [{k: (float(v) if k not in ('query_id', 'split') else v)
                for k, v in row.items()} for row in csv.DictReader(fh)]


def mean(xs):
    return sum(xs) / max(len(xs), 1)


def pctl(xs, p):
    if not xs:
        return float('nan')
    s = sorted(xs)
    idx = min(int(p * len(s)), len(s) - 1)
    return s[idx]


def paired_bootstrap_ci(diffs, n=N_BOOTSTRAP, seed=0):
    rng = random.Random(seed)
    n_q = len(diffs)
    if n_q == 0:
        return {'mean': float('nan'), 'ci_low': float('nan'), 'ci_high': float('nan')}
    means = []
    for _ in range(n):
        resample = [diffs[rng.randrange(n_q)] for _ in range(n_q)]
        means.append(mean(resample))
    means.sort()
    return {'mean': mean(diffs), 'ci_low': means[int(0.025 * n)],
           'ci_high': means[int(0.975 * n) - 1], 'n_bootstrap': n}


def analyze_cell(cell_dir):
    rows = read_csv(cell_dir / 'query_level_test.csv')
    n = len(rows)

    global10 = [r['global10_mse'] for r in rows]
    global30 = [r['global30_mse'] for r in rows]
    globalU = [r['globalU_mse'] for r in rows]
    block = [r['block_mse'] for r in rows]
    delta = [r['delta_block_vs_global'] for r in rows]

    summary = {
        'cell': cell_dir.name, 'n_test_queries': n,
        'global_top10_mse': mean(global10), 'global_top30_mse': mean(global30),
        'global_budgetmatched_mse': mean(globalU), 'block_oracle_mse': mean(block),
        'absolute_improvement_block_vs_global10': mean(global10) - mean(block),
        'relative_improvement_pct': (mean(global10) - mean(block)) / mean(global10) * 100
                                    if mean(global10) > 1e-12 else float('nan'),
        'block_vs_global30_gap': mean(global30) - mean(block),
        'block_vs_globalU_gap': mean(globalU) - mean(block),
        'mean_delta': mean(delta), 'median_delta': pctl(delta, 0.5),
        'p25_delta': pctl(delta, 0.25), 'p75_delta': pctl(delta, 0.75), 'p90_delta': pctl(delta, 0.90),
        'fraction_delta_gt_0': sum(1 for d in delta if d > 0) / max(n, 1),
        'fraction_delta_gt_1pct': sum(1 for r in rows
                                     if r['delta_block_vs_global'] > 0.01 * r['global10_mse']) / max(n, 1),
        'fraction_delta_gt_5pct': sum(1 for r in rows
                                     if r['delta_block_vs_global'] > 0.05 * r['global10_mse']) / max(n, 1),
        'fraction_delta_gt_10pct': sum(1 for r in rows
                                      if r['delta_block_vs_global'] > 0.10 * r['global10_mse']) / max(n, 1),
        'mean_unique_candidates_block': mean([r['unique_u'] for r in rows]),
    }
    (cell_dir / 'summary.json').write_text(json.dumps(summary, indent=2))

    overlap = {k: mean([r[k] for r in rows]) for k in
              ('overlap_g_b1', 'overlap_g_b2', 'overlap_g_b3',
               'overlap_b1_b2', 'overlap_b1_b3', 'overlap_b2_b3')}
    (cell_dir / 'overlap_matrix.json').write_text(json.dumps(overlap, indent=2))

    rank_disp = {k: mean([r[k] for r in rows]) for k in
                ('rank_b3_of_b1', 'rank_b1_of_b3', 'rank_g_of_b1', 'rank_g_of_b3')}
    (cell_dir / 'rank_displacement.json').write_text(json.dumps(rank_disp, indent=2))

    block_regret = {
        'block1_regret': mean([r['block1_regret'] for r in rows]),
        'block2_regret': mean([r['block2_regret'] for r in rows]),
        'block3_regret': mean([r['block3_regret'] for r in rows]),
    }
    (cell_dir / 'block_regret.json').write_text(json.dumps(block_regret, indent=2))

    budget_controls = {
        'global_top10_mse': mean(global10), 'global_top30_mse': mean(global30),
        'global_budgetmatched_mse': mean(globalU), 'block_oracle_mse': mean(block),
    }
    (cell_dir / 'budget_controls.json').write_text(json.dumps(budget_controls, indent=2))

    diff_10 = [g - b for g, b in zip(global10, block)]
    diff_30 = [g - b for g, b in zip(global30, block)]
    diff_u = [g - b for g, b in zip(globalU, block)]
    bootstrap = {
        'global10_minus_block': paired_bootstrap_ci(diff_10),
        'global30_minus_block': paired_bootstrap_ci(diff_30),
        'globalU_minus_block': paired_bootstrap_ci(diff_u),
    }
    (cell_dir / 'bootstrap.json').write_text(json.dumps(bootstrap, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='results/TRACK-A-HORIZON-RETRIEVAL-HEADROOM01')
    ap.add_argument('--cells', required=True)
    cli = ap.parse_args()

    all_summaries = {}
    for cell in cli.cells.split(','):
        cell_dir = Path(cli.out_dir) / cell
        if not (cell_dir / 'query_level_test.csv').exists():
            print(f'[analyze_horizon_retrieval_headroom01] {cell}: no test CSV yet, skipping')
            continue
        s = analyze_cell(cell_dir)
        all_summaries[cell] = s
        print(f"[analyze_horizon_retrieval_headroom01] {cell}: global10={s['global_top10_mse']:.6f} "
             f"block={s['block_oracle_mse']:.6f} rel_gain={s['relative_improvement_pct']:.2f}% "
             f"globalU={s['global_budgetmatched_mse']:.6f}")

    Path(cli.out_dir).mkdir(parents=True, exist_ok=True)
    Path(Path(cli.out_dir) / 'all_cells_summary.json').write_text(json.dumps(all_summaries, indent=2))


if __name__ == '__main__':
    main()
