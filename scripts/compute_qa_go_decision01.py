#!/usr/bin/env python3
"""TRACK-P100-SOFTTOPK-QUALITY-AWARE01 -- Phase E: VALIDATION-only
QA-vs-NOQA analysis and the GO/STOP decision (spec section 21).

Reads ONLY `checkpoint_selection.json` (best-epoch VALIDATION
forecast MSE) from every (cell, method, arm) directory produced by
`train_softtopk_e2e_{v,hard,joint}01.py` -- never touches any
`test_metrics.json` (which do not exist yet at this point in the
pipeline; TEST is opened only after a LOCKED_CONFIG.json exists, by
`evaluate_locked_test01.py`).

GO criteria, checked exactly as spec section 21 states, on the 12-cell
x 6-method grid's VALIDATION forecast MSE:
  1. QA < NOQA, averaged over all 72 (cell, method) combinations.
  2. QA < NOQA in >= 7/12 cells (cell-level average over methods).
  3-5. ETTh1 / Weather / Solar dataset averages: QA <= NOQA.
  6. Positive headroom between the Dynamic Oracle (if present) and
     NOQA -- checked only if oracle diagnostics were computed
     (optional; this script does not compute the oracle itself, see
     `compute_validation_oracle_headroom01.py`).
"""
import argparse
import json
from pathlib import Path

METHODS = ('V0', 'V1', 'V2', 'V5', 'Hard', 'Joint')
DATASETS = {
    'ETTh1': ['ETTh1_96', 'ETTh1_192', 'ETTh1_336', 'ETTh1_720'],
    'Weather': ['Weather_96', 'Weather_192', 'Weather_336', 'Weather_720'],
    'Solar': ['Solar_96', 'Solar_192', 'Solar_336', 'Solar_720'],
}
ALL_CELLS = [c for cells in DATASETS.values() for c in cells]


def load_val_mse(root, cell, method, arm):
    p = Path(root) / cell / method / arm / 'checkpoint_selection.json'
    if not p.exists():
        return None
    return json.loads(p.read_text())['val_mse']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_root', required=True,
                    help='results/TRACK-P100-SOFTTOPK-QUALITY-AWARE01')
    ap.add_argument('--oracle_headroom_json', default=None,
                    help='optional: output of compute_validation_oracle_headroom01.py')
    ap.add_argument('--out_json', required=True)
    cli = ap.parse_args()

    rows = {}
    missing = []
    for cell in ALL_CELLS:
        for method in METHODS:
            noqa = load_val_mse(cli.results_root, cell, method, 'NOQA')
            qa = load_val_mse(cli.results_root, cell, method, 'QA')
            if noqa is None or qa is None:
                missing.append(f'{cell}/{method} (NOQA={noqa is not None}, QA={qa is not None})')
                continue
            rows[(cell, method)] = {'noqa_val_mse': noqa, 'qa_val_mse': qa, 'delta': qa - noqa}

    if missing:
        print(f'[compute_qa_go_decision] WARNING: {len(missing)} (cell,method) combos missing, '
             f'decision computed on {len(rows)}/{len(ALL_CELLS) * len(METHODS)} available:')
        for m in missing:
            print(f'  missing: {m}')

    if not rows:
        raise SystemExit('[ISSUE][ABORT] no (cell,method) combos available -- cannot compute a decision')

    all_deltas = [v['delta'] for v in rows.values()]
    overall_avg_delta = sum(all_deltas) / len(all_deltas)
    criterion_1 = overall_avg_delta < 0

    cell_avg_delta = {}
    for cell in ALL_CELLS:
        deltas = [rows[(cell, m)]['delta'] for m in METHODS if (cell, m) in rows]
        if deltas:
            cell_avg_delta[cell] = sum(deltas) / len(deltas)
    cells_with_qa_win = sum(1 for d in cell_avg_delta.values() if d < 0)
    criterion_2 = cells_with_qa_win >= 7

    dataset_avg_delta = {}
    dataset_criteria = {}
    for ds, cells in DATASETS.items():
        deltas = [cell_avg_delta[c] for c in cells if c in cell_avg_delta]
        if deltas:
            dataset_avg_delta[ds] = sum(deltas) / len(deltas)
            dataset_criteria[ds] = dataset_avg_delta[ds] <= 0
        else:
            dataset_criteria[ds] = None

    criterion_3_5 = all(v for v in dataset_criteria.values() if v is not None) and \
        all(v is not None for v in dataset_criteria.values())

    criterion_6 = None
    oracle_headroom = None
    if cli.oracle_headroom_json and Path(cli.oracle_headroom_json).exists():
        oracle_headroom = json.loads(Path(cli.oracle_headroom_json).read_text())
        criterion_6 = bool(oracle_headroom.get('positive_headroom_exists', False))

    criteria = {
        '1_overall_avg_qa_lt_noqa': criterion_1,
        '2_cells_qa_wins_ge_7of12': criterion_2,
        '3_5_all_dataset_avgs_qa_le_noqa': criterion_3_5,
        '6_positive_dynamic_oracle_headroom': criterion_6,
    }
    qa_go = criterion_1 and criterion_2 and criterion_3_5 and (criterion_6 is not False)

    out = {
        'QA_GO': qa_go,
        'criteria': criteria,
        'overall_avg_delta': overall_avg_delta,
        'cells_with_qa_win': cells_with_qa_win,
        'cell_avg_delta': cell_avg_delta,
        'dataset_avg_delta': dataset_avg_delta,
        'per_cell_method_rows': {f'{c}/{m}': v for (c, m), v in rows.items()},
        'missing_combos': missing,
        'oracle_headroom': oracle_headroom,
        'note': 'Computed from VALIDATION checkpoint_selection.json ONLY. TEST was never read to produce this decision.',
    }
    Path(cli.out_json).write_text(json.dumps(out, indent=2))
    print(f'[compute_qa_go_decision] QA_GO={qa_go} overall_avg_delta={overall_avg_delta:+.6f} '
         f'cells_with_qa_win={cells_with_qa_win}/12')
    for k, v in criteria.items():
        print(f'  criterion {k}: {v}')


if __name__ == '__main__':
    main()
