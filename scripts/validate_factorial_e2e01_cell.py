#!/usr/bin/env python3
"""TRACK-A-FACTORIAL-E2E01 -- per-cell gate (spec S6 of the revised plan).

Run after every cell. Exits non-zero on the first failure so the orchestrator
stops instead of moving to the next cell. Read-only.

Checks:
  1. Independent Base-only Forecaster exists, finite, no NaN
  2. all 8 arms produced retrieval_metrics_*.json and stage2_metrics_*.json
  3. best_epoch recorded and in range for every arm
  4. no NaN/Inf anywhere in the reported numbers
  5. encoder-collapse metrics present for every epoch of every arm
  6. all 8 arms share ONE initial encoder SHA256
  7. y_base bit-identity held for arms 2..8
  8. duplicate_rate == invalid_rate == 0
  9. cosine_init_deviation ~ 0 for the asymmetric arms
 10. the Independent Base-only baseline actually reached every stage2 json
"""
import json
import math
import sys
from pathlib import Path

ARMS = ('individual_tf_cosine', 'individual_tf_asymmetric',
        'individual_onpolicy_cosine', 'individual_onpolicy_asymmetric',
        'set_tf_cosine', 'set_tf_asymmetric',
        'set_onpolicy_cosine', 'set_onpolicy_asymmetric')


def bad(x):
    return isinstance(x, float) and (math.isnan(x) or math.isinf(x))


def main():
    if len(sys.argv) < 2:
        print('usage: validate_factorial_e2e01_cell.py <cell> [root]', file=sys.stderr)
        return 2
    cell = sys.argv[1]
    root = Path(sys.argv[2] if len(sys.argv) > 2 else 'results/track_a_factorial_e2e')
    d = root / cell
    fails = []

    # 1. Independent Base-only
    bp = d / 'independent_base_only.json'
    if not bp.exists():
        fails.append('MISSING Independent Base-only Forecaster result')
        base = None
    else:
        base = json.loads(bp.read_text())
        for k in ('independent_base_only_mse', 'independent_base_only_mae'):
            if k not in base or bad(base[k]):
                fails.append(f'Independent Base-only {k} missing/NaN: {base.get(k)}')
        if base.get('best_epoch', -1) < 1:
            fails.append(f'Independent Base-only best_epoch invalid: {base.get("best_epoch")}')

    shas, arms_seen = set(), []
    for arm in ARMS:
        r1p, r2p = d / f'retrieval_metrics_{arm}.json', d / f'stage2_metrics_{arm}.json'
        if not r1p.exists():
            fails.append(f'{arm}: MISSING retrieval_metrics json')
            continue
        if not r2p.exists():
            fails.append(f'{arm}: MISSING stage2_metrics json')
            continue
        arms_seen.append(arm)
        r1, r2 = json.loads(r1p.read_text()), json.loads(r2p.read_text())

        if not (1 <= int(r1.get('best_epoch', -1)) <= int(r1.get('final_epoch', 0))):
            fails.append(f'{arm}: best_epoch={r1.get("best_epoch")} out of range '
                         f'(final_epoch={r1.get("final_epoch")})')
        for k in ('best_val_free_running_aggregate_future_mse',
                  'best_ckpt_test_free_running_aggregate_future_mse',
                  'final_effective_rank', 'final_mean_pairwise_cosine'):
            if k not in r1 or bad(r1[k]):
                fails.append(f'{arm}: {k} missing/NaN -> {r1.get(k)}')
        if 'asymmetric' in arm and float(r1.get('cosine_init_deviation', 1.0)) > 1e-5:
            fails.append(f'{arm}: cosine_init_deviation={r1.get("cosine_init_deviation")}')
        shas.add(r1.get('encoder_init_sha256'))

        ep = d / f'epoch_metrics_{arm}.csv'
        rep = d / f'representation_metrics_{arm}.csv'
        step = d / f'stepwise_metrics_{arm}.csv'
        for f in (ep, rep, step):
            if not f.exists() or f.stat().st_size == 0:
                fails.append(f'{arm}: missing/empty {f.name}')
        if rep.exists():
            import csv as _csv
            with open(rep) as fh:
                rows = list(_csv.DictReader(fh))
            if not rows or 'rep_effective_rank' not in rows[0]:
                fails.append(f'{arm}: representation csv has no effective-rank column')
            for row in rows:
                for k, v in row.items():
                    if k == 'rep_has_nan':
                        if str(v).strip().lower() not in ('false', ''):
                            fails.append(f'{arm}: rep_has_nan={v} at epoch {row.get("epoch")}')
                        continue
                    if v in (None, '', 'None'):
                        continue
                    try:
                        fv = float(v)
                    except ValueError:
                        continue
                    if bad(fv):
                        fails.append(f'{arm}: NaN/Inf in {k} at epoch {row.get("epoch")}')

        for k in ('retrieval_augmented_frozen_host_mse', 'independent_base_only_mse',
                  'delta_mse_vs_independent_base'):
            if r2.get(k) is None or bad(r2.get(k)):
                fails.append(f'{arm}: stage2 {k} missing/NaN -> {r2.get(k)}')
        if float(r2.get('duplicate_rate', 1)) != 0.0:
            fails.append(f'{arm}: duplicate_rate={r2.get("duplicate_rate")}')
        if float(r2.get('invalid_rate', 1)) != 0.0:
            fails.append(f'{arm}: invalid_rate={r2.get("invalid_rate")}')
        if arm != ARMS[0] and r2.get('y_base_identical_to_reference') is not True:
            fails.append(f'{arm}: y_base NOT bit-identical to reference '
                         f'({r2.get("y_base_identical_to_reference")})')
        if base is not None and r2.get('independent_base_only_mse') is not None:
            if abs(r2['independent_base_only_mse'] - base['independent_base_only_mse']) > 1e-12:
                fails.append(f'{arm}: stage2 json baseline disagrees with '
                             f'independent_base_only.json')

    if len(arms_seen) != len(ARMS):
        fails.append(f'only {len(arms_seen)}/{len(ARMS)} arms completed')
    if len(shas) > 1:
        fails.append(f'initial encoder SHA mismatch across arms: {shas}')

    print(f'=== cell gate: {cell} ===')
    if base:
        print(f'  Independent Base-only Forecaster: mse={base["independent_base_only_mse"]:.6f} '
              f'mae={base["independent_base_only_mae"]:.6f} best_epoch={base["best_epoch"]}')
    print(f'  arms complete: {len(arms_seen)}/{len(ARMS)}   init SHA: '
          f'{(list(shas)[0][:16] if len(shas) == 1 else shas)}')
    if fails:
        print(f'  FAILED ({len(fails)}):')
        for f in fails:
            print(f'    - {f}')
        return 1
    print('  ALL CHECKS PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
