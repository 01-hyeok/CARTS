#!/usr/bin/env python3
"""TRACK-V5-HEAD-OBJECTIVE-AUDIT01 -- CPU-only pandas analysis of the
per-(query,channel) parquet produced by
`diag_v5_head_objective01.py --mode full_pass`. No model/GPU use here.
Produces the PHASE C/D/E/G numeric tables for one cell, as JSON, from
the val+test parquet pair.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HEADS = [1, 2, 3, 4, 5]
PAIRS = list(itertools.combinations(HEADS, 2))
ALL_SUBSETS = [c for r in range(1, 6) for c in itertools.combinations(HEADS, r)]
SUBSET_NAMES = [''.join(str(h) for h in s) for s in ALL_SUBSETS]


def subset_name(s):
    return ''.join(str(h) for h in s)


def phase_c(df):
    rows = []
    for h in HEADS:
        rows.append({'method': f'Head {h}', 'retmse10': df[f'retmse10_H{h}'].mean(),
                    'aggmse10': df[f'aggmse_H{h}'].mean(), 'D': df[f'D_H{h}'].mean(),
                    'C': df[f'C_H{h}'].mean(), 'recall10': df[f'recall10_H{h}'].mean(),
                    'ndcg10': df[f'ndcg10_H{h}'].mean()})
    rows.append({'method': 'Mean-Mixture', 'retmse10': df['retmse10_mix'].mean(),
                'aggmse10': df['aggmse_mix'].mean(), 'D': df['D_mix'].mean(), 'C': df['C_mix'].mean(),
                'recall10': df['recall10_mix'].mean(), 'ndcg10': df['ndcg10_mix'].mean()})
    best_ind = min(rows[:5], key=lambda r: r['retmse10'])['method']
    best_agg = min(rows[:5], key=lambda r: r['aggmse10'])['method']
    identity_check = float((df['C_mix'] - df['C_mix_subtraction']).abs().max())
    return {'table': rows, 'best_individual_head': best_ind, 'best_aggregate_head': best_agg,
           'same_head_wins_both': best_ind == best_agg,
           'DC_identity_max_abs_diff_direct_vs_subtraction': identity_check}


def phase_d(df):
    pair_rows = []
    for m, n in PAIRS:
        name = f'H{m}-H{n}'
        dot = df[f'dot_{name}']
        cos = df[f'cos_{name}']
        overlap = df[f'overlap_{name}']
        neg_frac = float((cos < 0).mean())
        pear = float(np.corrcoef(overlap, cos)[0, 1])
        spear = float(stats.spearmanr(overlap, cos).correlation)
        pair_rows.append({'pair': name, 'mean_overlap': float(overlap.mean()), 'mean_cos': float(cos.mean()),
                          'mean_dot': float(dot.mean()), 'negative_cos_fraction': neg_frac,
                          'overlap_vs_cos_pearson': pear, 'overlap_vs_cos_spearman': spear})
    all_cos = pd.concat([df[f'cos_{f"H{m}-H{n}"}'] for m, n in PAIRS])
    all_overlap = pd.concat([df[f'overlap_{f"H{m}-H{n}"}'] for m, n in PAIRS])
    pooled_corr = float(np.corrcoef(all_overlap, all_cos)[0, 1])
    cross_head_mean_cos = float(all_cos.mean())
    within_mix_c_mean = float(df['C_mix'].mean())
    return {'pairwise': pair_rows, 'pooled_overlap_vs_cos_pearson_WARNING_not_independent_samples': pooled_corr,
           'cross_head_mean_cosine_all_pairs': cross_head_mean_cos,
           'within_mean_mixture_set_C_mean': within_mix_c_mean,
           'note': 'pooled_overlap_vs_cos_pearson treats every (query,channel,pair) row as independent; '
                  'per-pair correlations above are the query/channel-level analysis the spec requires. '
                  'Cross-Head mean cosine (between independent per-head aggregate predictions R_m) is NOT '
                  'the same quantity as Within-Mean-Mixture-Set C (interaction among the 10 candidates actually '
                  'inside S_mix) -- reported side by side deliberately, per spec section 7.3/D3.'}


def phase_e(df_val, df_test):
    subset_cols = [f'subset_aggmse_{s}' for s in SUBSET_NAMES]
    val_means = {s: df_val[f'subset_aggmse_{s}'].mean() for s in SUBSET_NAMES}
    best_fixed_subset = min(val_means, key=val_means.get)
    all5 = '12345'
    single_best = min((s for s in SUBSET_NAMES if len(s) == 1), key=lambda s: val_means[s])

    test_all5 = df_test[f'subset_aggmse_{all5}'].mean()
    test_best_fixed = df_test[f'subset_aggmse_{best_fixed_subset}'].mean()
    test_single_best = df_test[f'subset_aggmse_{single_best}'].mean()
    per_query_oracle = df_test[subset_cols].min(axis=1).mean()
    oracle_subset_argmin = df_test[subset_cols].idxmin(axis=1).value_counts(normalize=True).head(10).to_dict()

    loo = {}
    for h in HEADS:
        rest = subset_name(tuple(x for x in HEADS if x != h))
        loo[f'H{h}'] = float(df_test[f'subset_aggmse_{rest}'].mean() - test_all5)

    gap_head = test_all5 - per_query_oracle
    return {
        'val_selected_best_fixed_subset': best_fixed_subset, 'val_selected_best_single_head': single_best,
        'test_all5_meanmixture_aggmse': float(test_all5), 'test_best_single_head_aggmse': float(test_single_best),
        'test_best_fixed_subset_aggmse_valselected': float(test_best_fixed),
        'test_per_query_oracle_subset_aggmse_UPPERBOUND_NOT_DEPLOYABLE': float(per_query_oracle),
        'gap_head_all5_minus_oracle': float(gap_head),
        'gap_head_pct_of_all5': float(gap_head / test_all5 * 100.0),
        'leave_one_head_out_contribution_delta': loo,
        'oracle_winner_subset_frequency_top10': {k: float(v) for k, v in oracle_subset_argmin.items()},
        'note_loo': 'Delta_h = AggMSE(all heads except h) - AggMSE(all 5). Positive => head h HELPS '
                   '(removing it makes things worse); negative => head h HURTS (removing it helps). '
                   'Conditional on the other 4 heads, not an absolute independent utility.',
    }


def phase_g(df):
    gap = (df['aggmse_mix'] - df['aggmse_soft']).abs()
    corr = float(np.corrcoef(df['aggmse_mix'], df['aggmse_soft'])[0, 1])
    return {'hard_mean_mixture_aggmse': float(df['aggmse_mix'].mean()),
           'soft_full_pool_expectation_aggmse': float(df['aggmse_soft'].mean()),
           'mean_abs_prediction_gap': float(gap.mean()), 'soft_minus_hard_pct':
           float((df['aggmse_soft'].mean() - df['aggmse_mix'].mean()) / df['aggmse_mix'].mean() * 100.0),
           'query_channel_level_correlation_soft_vs_hard_aggmse': corr}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True)
    ap.add_argument('--in_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    cli = ap.parse_args()
    in_dir = Path(cli.in_dir) / cli.cell
    df_val = pd.read_parquet(in_dir / 'full_pass_val.parquet')
    df_test = pd.read_parquet(in_dir / 'full_pass_test.parquet')

    result = {'cell': cli.cell, 'n_val_rows': len(df_val), 'n_test_rows': len(df_test),
             'phase_c_test': phase_c(df_test), 'phase_c_val': phase_c(df_val),
             'phase_d_test': phase_d(df_test), 'phase_e': phase_e(df_val, df_test),
             'phase_g_test': phase_g(df_test)}
    out_dir = Path(cli.out_dir) / cli.cell
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'analysis.json').write_text(json.dumps(result, indent=2, default=str))
    print(f'[analyze_v5] {cli.cell} DONE -> {out_dir / "analysis.json"}')


if __name__ == '__main__':
    main()
