#!/usr/bin/env python3
"""TRACK-A-FACTORIAL-E2E01 -- aggregation and factorial analysis.

Reads whatever cells have completed and writes the summary tables the spec
asks for (S27). Runs on CPU, touches no checkpoint, trains nothing.

Main effects are reported as plain means over the 4 arms on each side of a
factor, per cell. With one seed, no significance is computed or claimed --
only direction, magnitude and cross-cell consistency.
"""
import argparse
import csv
import json
import math
from pathlib import Path

CELLS = ('ETTh1_96', 'ETTh1_720', 'Weather_96', 'Weather_720')
ARMS = ('individual_tf_cosine', 'individual_tf_asymmetric',
        'individual_onpolicy_cosine', 'individual_onpolicy_asymmetric',
        'set_tf_cosine', 'set_tf_asymmetric',
        'set_onpolicy_cosine', 'set_onpolicy_asymmetric')
FACTORS = {'oracle': ('individual', 'greedy_set'),
           'prefix': ('tf', 'onpolicy'),
           'score': ('cosine', 'asymmetric')}


def _read(path):
    try:
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def _mean(xs):
    xs = [x for x in xs if x is not None and x == x]
    return sum(xs) / len(xs) if xs else float('nan')


def _corr(xs, ys, rank=False):
    pairs = [(a, b) for a, b in zip(xs, ys)
             if a is not None and b is not None and a == a and b == b]
    if len(pairs) < 3:
        return float('nan')
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    if rank:
        a = [sorted(a).index(v) for v in a]
        b = [sorted(b).index(v) for v in b]
    ma, mb = _mean(a), _mean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((y - mb) ** 2 for y in b))
    return num / (da * db) if da > 0 and db > 0 else float('nan')


def _write_csv(path, rows):
    if not rows:
        return
    keys, seen = [], set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'[analyze] wrote {path} ({len(rows)} rows)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='results/track_a_factorial_e2e')
    cli = ap.parse_args()
    root = Path(cli.root)
    root.mkdir(parents=True, exist_ok=True)

    all_rows, missing = [], []
    for cell in CELLS:
        for arm in ARMS:
            r1 = _read(root / cell / f'retrieval_metrics_{arm}.json')
            r2 = _read(root / cell / f'stage2_metrics_{arm}.json')
            if r1 is None and r2 is None:
                missing.append(f'{cell}/{arm}')
                continue
            oracle, prefix, score = arm.split('_')[0], arm.split('_')[1], arm.split('_')[-1]
            oracle = 'greedy_set' if oracle == 'set' else 'individual'
            row = {
                'cell': cell, 'arm': arm,
                'oracle': oracle, 'prefix': prefix, 'score': score,
                'best_epoch': (r1 or {}).get('best_epoch'),
                'best_val_fr_agg_mse': (r1 or {}).get('best_val_free_running_aggregate_future_mse'),
                'test_fr_agg_mse': (r1 or {}).get('best_ckpt_test_free_running_aggregate_future_mse'),
                'final_val_fr_agg_mse': (r1 or {}).get('final_val_free_running_aggregate_future_mse'),
                'final_effective_rank': (r1 or {}).get('final_effective_rank'),
                'final_mean_pairwise_cosine': (r1 or {}).get('final_mean_pairwise_cosine'),
                'final_encoder_param_displacement': (r1 or {}).get('final_encoder_param_displacement'),
                'cosine_init_deviation': (r1 or {}).get('cosine_init_deviation'),
                'encoder_init_sha256': (r1 or {}).get('encoder_init_sha256'),
                # PRIMARY reference
                'independent_base_only_mse': (r2 or {}).get('independent_base_only_mse'),
                'independent_base_only_mae': (r2 or {}).get('independent_base_only_mae'),
                'retrieval_augmented_frozen_host_mse': (r2 or {}).get('retrieval_augmented_frozen_host_mse'),
                'retrieval_augmented_frozen_host_mae': (r2 or {}).get('retrieval_augmented_frozen_host_mae'),
                'delta_mse_vs_independent_base': (r2 or {}).get('delta_mse_vs_independent_base'),
                'delta_mae_vs_independent_base': (r2 or {}).get('delta_mae_vs_independent_base'),
                'relative_improvement_vs_independent_base_pct':
                    (r2 or {}).get('relative_improvement_vs_independent_base_pct'),
                # DIAGNOSTIC ONLY -- never a baseline
                'frozen_host_retrieval_ablated_branch_mse':
                    (r2 or {}).get('frozen_host_retrieval_ablated_branch_mse'),
                'diag_delta_mse_vs_host_ablated_branch':
                    (r2 or {}).get('diag_delta_mse_vs_host_ablated_branch'),
                'frozen_host_original_retrieval_mse':
                    (r2 or {}).get('frozen_host_original_retrieval_mse'),
                'diag_delta_mse_vs_host_original_retrieval':
                    (r2 or {}).get('diag_delta_mse_vs_host_original_retrieval'),
                'duplicate_rate': (r2 or {}).get('duplicate_rate'),
                'invalid_rate': (r2 or {}).get('invalid_rate'),
                'y_base_identical_to_reference': (r2 or {}).get('y_base_identical_to_reference'),
            }
            for e in ('epoch1', 'epoch5', 'epoch10'):
                row[f'matched_val_{e}'] = ((r1 or {}).get('matched_epoch_val') or {}).get(e)
                row[f'matched_test_{e}'] = ((r1 or {}).get('matched_epoch_test') or {}).get(e)
            ti = (r1 or {}).get('test_internal') or {}
            for kk in ('oracle_action_acc', 'expert_rank_median', 'expert_regret_mean',
                       'ndcg_at_10', 'spearman', 'selected_individual_future_mse'):
                row[f'test_{kk}'] = ti.get(kk)
            all_rows.append(row)

    if missing:
        print(f'[analyze] NOT YET AVAILABLE ({len(missing)}): ' + ', '.join(missing))
    _write_csv(root / 'all_results.csv', all_rows)

    _write_csv(root / 'retrieval_summary.csv', [
        {k: r[k] for k in ('cell', 'arm', 'oracle', 'prefix', 'score', 'best_epoch',
                           'best_val_fr_agg_mse', 'test_fr_agg_mse',
                           'test_oracle_action_acc', 'test_expert_rank_median',
                           'test_ndcg_at_10', 'test_spearman')} for r in all_rows])

    _write_csv(root / 'stage2_summary.csv', [
        {k: r[k] for k in (
            'cell', 'arm', 'oracle', 'prefix', 'score',
            'independent_base_only_mse', 'retrieval_augmented_frozen_host_mse',
            'delta_mse_vs_independent_base', 'relative_improvement_vs_independent_base_pct',
            'independent_base_only_mae', 'retrieval_augmented_frozen_host_mae',
            'delta_mae_vs_independent_base',
            'frozen_host_retrieval_ablated_branch_mse', 'diag_delta_mse_vs_host_ablated_branch',
            'frozen_host_original_retrieval_mse', 'diag_delta_mse_vs_host_original_retrieval',
            'test_fr_agg_mse', 'y_base_identical_to_reference')} for r in all_rows])

    _write_csv(root / 'best_epoch_summary.csv', [
        {k: r[k] for k in ('cell', 'arm', 'best_epoch', 'best_val_fr_agg_mse',
                           'test_fr_agg_mse', 'final_val_fr_agg_mse',
                           'matched_val_epoch1', 'matched_val_epoch5', 'matched_val_epoch10',
                           'matched_test_epoch1', 'matched_test_epoch5',
                           'matched_test_epoch10')} for r in all_rows])

    _write_csv(root / 'representation_summary.csv', [
        {k: r[k] for k in ('cell', 'arm', 'oracle', 'prefix', 'score',
                           'final_effective_rank', 'final_mean_pairwise_cosine',
                           'final_encoder_param_displacement', 'cosine_init_deviation',
                           'encoder_init_sha256')} for r in all_rows])

    # ---- factorial effects ----
    eff_rows = []
    metrics = ('test_fr_agg_mse', 'delta_mse_vs_independent_base',
               'retrieval_augmented_frozen_host_mse', 'final_effective_rank')
    for cell in CELLS:
        rows = [r for r in all_rows if r['cell'] == cell]
        if len(rows) < 2:
            continue
        for factor, (lo, hi) in FACTORS.items():
            for m in metrics:
                a = _mean([r[m] for r in rows if r[factor] == lo])
                b = _mean([r[m] for r in rows if r[factor] == hi])
                eff_rows.append({'cell': cell, 'type': 'main_effect', 'factor': factor,
                                 'metric': m, f'mean_{lo}': a, f'mean_{hi}': b,
                                 'effect_hi_minus_lo': b - a if a == a and b == b else float('nan'),
                                 'better_level': (lo if a < b else hi) if a == a and b == b else None,
                                 'n_arms': len(rows)})
        pairs = (('oracle', 'prefix'), ('oracle', 'score'), ('prefix', 'score'))
        for f1, f2 in pairs:
            for m in metrics:
                cellvals = {}
                for l1 in FACTORS[f1]:
                    for l2 in FACTORS[f2]:
                        cellvals[f'{l1}|{l2}'] = _mean(
                            [r[m] for r in rows if r[f1] == l1 and r[f2] == l2])
                k = list(cellvals)
                inter = float('nan')
                if all(cellvals[x] == cellvals[x] for x in k) and len(k) == 4:
                    a, b, c, d = (cellvals[f'{FACTORS[f1][0]}|{FACTORS[f2][0]}'],
                                  cellvals[f'{FACTORS[f1][0]}|{FACTORS[f2][1]}'],
                                  cellvals[f'{FACTORS[f1][1]}|{FACTORS[f2][0]}'],
                                  cellvals[f'{FACTORS[f1][1]}|{FACTORS[f2][1]}'])
                    inter = (d - c) - (b - a)
                eff_rows.append({'cell': cell, 'type': 'interaction',
                                 'factor': f'{f1} x {f2}', 'metric': m,
                                 **cellvals, 'interaction': inter, 'n_arms': len(rows)})

    # ---- retrieval quality vs forecasting (spec S22) ----
    fr = [r['test_fr_agg_mse'] for r in all_rows]
    dm = [r['delta_mse_vs_independent_base'] for r in all_rows]
    rm = [r['retrieval_augmented_frozen_host_mse'] for r in all_rows]
    eff_rows.append({'cell': 'ALL', 'type': 'consistency', 'factor': 'fr_agg_vs_delta_mse',
                     'metric': 'pearson', 'value': _corr(fr, dm),
                     'n_arms': len([1 for a, b in zip(fr, dm)
                                    if a is not None and b is not None])})
    eff_rows.append({'cell': 'ALL', 'type': 'consistency', 'factor': 'fr_agg_vs_delta_mse',
                     'metric': 'spearman', 'value': _corr(fr, dm, rank=True)})
    eff_rows.append({'cell': 'ALL', 'type': 'consistency', 'factor': 'fr_agg_vs_stage2_mse',
                     'metric': 'pearson', 'value': _corr(fr, rm)})
    eff_rows.append({'cell': 'ALL', 'type': 'consistency', 'factor': 'fr_agg_vs_stage2_mse',
                     'metric': 'spearman', 'value': _corr(fr, rm, rank=True)})
    for cell in CELLS:
        rows = [r for r in all_rows if r['cell'] == cell]
        eff_rows.append({'cell': cell, 'type': 'consistency', 'factor': 'fr_agg_vs_delta_mse',
                         'metric': 'spearman',
                         'value': _corr([r['test_fr_agg_mse'] for r in rows],
                                        [r['delta_mse_vs_independent_base'] for r in rows],
                                        rank=True)})
    _write_csv(root / 'factorial_effects.csv', eff_rows)

    # ---- headline console summary ----
    print('\n=== Retrieval-Augmented Frozen-Host Stage-2 '
          'vs Independent Base-only Forecaster ===')
    for cell in CELLS:
        rows = [r for r in all_rows if r['cell'] == cell
                and r['delta_mse_vs_independent_base'] is not None]
        if not rows:
            continue
        base = rows[0]['independent_base_only_mse']
        ablated = rows[0]['frozen_host_retrieval_ablated_branch_mse']
        orig = rows[0]['frozen_host_original_retrieval_mse']
        worse = sum(1 for r in rows if r['delta_mse_vs_independent_base'] > 0)
        print(f'  {cell}:')
        print(f'    Independent Base-only Forecaster (BASELINE)   = {base:.5f}')
        print(f'    Frozen Host Retrieval-Ablated Branch (diag)   = {ablated:.5f}')
        print(f'    Frozen Host Original Retrieval       (diag)   = {orig:.5f}')
        print(f'    arms worse than the baseline: {worse}/{len(rows)}')
        for r in sorted(rows, key=lambda x: x['delta_mse_vs_independent_base']):
            print(f"      {r['arm']:<32} ret={r['retrieval_augmented_frozen_host_mse']:.5f} "
                  f"delta={r['delta_mse_vs_independent_base']:+.5f} "
                  f"({r['relative_improvement_vs_independent_base_pct']:+.2f}%) "
                  f"fr_agg={r['test_fr_agg_mse']}")


if __name__ == '__main__':
    main()
