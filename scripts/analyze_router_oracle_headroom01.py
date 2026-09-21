#!/usr/bin/env python3
"""ROUTER-ORACLE-HEADROOM01 -- Phase-1 analysis: per-cell summary, winner
fractions, pairwise win matrix, Best-Static (validation-selected) vs Oracle
Router (test-query future-aware upper bound), paired bootstrap CIs.
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

EXPERTS = ('raw', 'delta', 'learned')
N_BOOTSTRAP = 2000


def read_csv(path):
    with open(path, newline='') as fh:
        return [{k: (float(v) if k not in ('query_id', 'split') else v)
                for k, v in row.items()} for row in csv.DictReader(fh)]


def mean(xs):
    return sum(xs) / max(len(xs), 1)


def best_static_expert(val_rows):
    """Selects the single expert with lowest MEAN uniform_mse on VALIDATION
    only -- never reads test labels (spec section 13 / leakage test R5)."""
    means = {e: mean([r[f'{e}_uniform_mse'] for r in val_rows]) for e in EXPERTS}
    return min(means, key=means.get), means


def oracle_row(row):
    errs = {e: row[f'{e}_uniform_mse'] for e in EXPERTS}
    best = min(errs, key=errs.get)
    sorted_errs = sorted(errs.values())
    margin = (sorted_errs[1] - sorted_errs[0]) if len(sorted_errs) > 1 else 0.0
    return best, errs[best], margin


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
    lo = means[int(0.025 * n)]
    hi = means[int(0.975 * n) - 1]
    return {'mean': mean(diffs), 'ci_low': lo, 'ci_high': hi, 'n_bootstrap': n}


def analyze_cell(cell_dir):
    val_rows = read_csv(cell_dir / 'query_utility_val.csv')
    test_rows = read_csv(cell_dir / 'query_utility_test.csv')

    static_expert, val_means = best_static_expert(val_rows)

    winner_counts = {e: 0 for e in EXPERTS}
    oracle_errs, static_errs, margins = [], [], []
    pairwise = {f'{a}_vs_{b}': 0 for a in EXPERTS for b in EXPERTS if a < b}
    static_not_best = 0
    for row in test_rows:
        best, err, margin = oracle_row(row)
        winner_counts[best] += 1
        oracle_errs.append(err)
        static_errs.append(row[f'{static_expert}_uniform_mse'])
        margins.append(margin)
        if best != static_expert:
            static_not_best += 1
        for a in EXPERTS:
            for b in EXPERTS:
                if a < b:
                    key = f'{a}_vs_{b}'
                    if row[f'{a}_uniform_mse'] < row[f'{b}_uniform_mse']:
                        pairwise[key] += 1
    n = len(test_rows)
    winner_fraction = {e: winner_counts[e] / max(n, 1) for e in EXPERTS}
    pairwise_fraction = {k: v / max(n, 1) for k, v in pairwise.items()}

    mean_static = mean(static_errs)
    mean_oracle = mean(oracle_errs)
    headroom_abs = mean_static - mean_oracle
    headroom_rel = headroom_abs / mean_static if mean_static > 1e-12 else float('nan')

    static_minus_oracle = [s - o for s, o in zip(static_errs, oracle_errs)]
    bootstrap_headroom = paired_bootstrap_ci(static_minus_oracle)

    expert_pair_bootstrap = {}
    for a in EXPERTS:
        for b in EXPERTS:
            if a < b:
                diffs = [r[f'{a}_uniform_mse'] - r[f'{b}_uniform_mse'] for r in test_rows]
                expert_pair_bootstrap[f'{a}_minus_{b}'] = paired_bootstrap_ci(diffs)

    mean_error = {e: mean([r[f'{e}_uniform_mse'] for r in test_rows]) for e in EXPERTS}

    summary = {
        'cell': cell_dir.name, 'n_test_queries': n, 'n_val_queries': len(val_rows),
        'mean_error': mean_error,
        'static_expert': static_expert, 'static_expert_val_means': val_means,
        'best_static_test_error': mean_static, 'oracle_router_test_error': mean_oracle,
        'headroom_absolute': headroom_abs, 'headroom_relative': headroom_rel,
        'headroom_bootstrap': bootstrap_headroom,
        'winner_fraction': winner_fraction,
        'fraction_static_not_query_best': static_not_best / max(n, 1),
        'mean_winner_margin': mean(margins),
        'n_unique_winner_experts_present': sum(1 for e in EXPERTS if winner_counts[e] > 0),
    }
    (cell_dir / 'summary.json').write_text(json.dumps(summary, indent=2))
    (cell_dir / 'winner_fraction.json').write_text(json.dumps(winner_fraction, indent=2))
    (cell_dir / 'pairwise_matrix.json').write_text(json.dumps(pairwise_fraction, indent=2))
    (cell_dir / 'bootstrap.json').write_text(json.dumps({
        'headroom_static_minus_oracle': bootstrap_headroom,
        'expert_pairs': expert_pair_bootstrap,
    }, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='results/ROUTER-ORACLE-HEADROOM01')
    ap.add_argument('--cells', required=True, help='comma-separated cell names')
    cli = ap.parse_args()

    all_summaries = {}
    for cell in cli.cells.split(','):
        cell_dir = Path(cli.out_dir) / cell
        if not (cell_dir / 'query_utility_test.csv').exists():
            print(f'[analyze_router_oracle_headroom01] {cell}: no test CSV yet, skipping')
            continue
        s = analyze_cell(cell_dir)
        all_summaries[cell] = s
        print(f"[analyze_router_oracle_headroom01] {cell}: static={s['static_expert']} "
             f"headroom_abs={s['headroom_absolute']:.6f} headroom_rel={s['headroom_relative']:.4f} "
             f"winner_fraction={s['winner_fraction']}")

    Path(cli.out_dir).mkdir(parents=True, exist_ok=True)
    Path(Path(cli.out_dir) / 'all_cells_summary.json').write_text(json.dumps(all_summaries, indent=2))


if __name__ == '__main__':
    main()
