#!/usr/bin/env python3
"""TRACK-P100-QA-CAUSAL-VALIDATION01 -- Phase A3 fix of
`compute_qa_go_decision01.py`.

The original script computes `QA_GO = criterion_1 and criterion_2 and
criterion_3_5 and (criterion_6 is not False)`. Two real problems,
confirmed by reading the code and by actually running it on the
original track's own partial data (125/144 complete, 62/72 (cell,
method) combos available):

  1. It never checks that all 72 (cell, method) combinations are
     present -- it silently computes on whatever subset exists and
     still returns a definite `QA_GO: true/false`, not an
     "INCOMPLETE" state. Running it today on the partial grid printed
     `QA_GO=True` computed on only 62/72 combos.
  2. `criterion_6 is not False` is `True` when `criterion_6` is `None`
     (not computed) -- so a never-computed Oracle-headroom diagnostic
     silently does NOT block GO. The same partial run above returned
     `QA_GO=True` with `criterion_6: null`.

Neither is a numerical bug (the per-combo deltas and averages are
computed correctly) -- both are STATE bugs: the script cannot
distinguish "GO" from "not enough evidence to say GO or STOP yet".

Fix: this script requires, before returning any GO/STOP verdict:
  - All 72 (cell, method) combinations present.
  - Criterion 6 (oracle headroom) actually computed (not None).
If either is missing, the result is `"status": "INCOMPLETE"` and
`QA_GO` is `null`, not `true`/`false`. The original script
(`compute_qa_go_decision01.py`) is left completely unmodified -- it
still backs the ORIGINAL track's own (clearly-labeled PRELIMINARY)
interim reporting; this is a separate script for this causal-validation
track's own, stricter, final decision.
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
TOTAL_COMBOS = len(ALL_CELLS) * len(METHODS)  # 72


def load_val_mse(root, cell, method, arm):
    p = Path(root) / cell / method / arm / 'checkpoint_selection.json'
    if not p.exists():
        return None
    return json.loads(p.read_text())['val_mse']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_root', required=True)
    ap.add_argument('--oracle_headroom_json', default=None,
                    help='REQUIRED for a non-INCOMPLETE verdict (unlike the original script).')
    ap.add_argument('--out_json', required=True)
    ap.add_argument('--allow_incomplete_preview', action='store_true',
                    help='If set, still computes and reports preview numbers on partial data, '
                         'but `status` stays INCOMPLETE and `QA_GO` stays null.')
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

    oracle_headroom = None
    criterion_6 = None
    if cli.oracle_headroom_json and Path(cli.oracle_headroom_json).exists():
        oracle_headroom = json.loads(Path(cli.oracle_headroom_json).read_text())
        criterion_6 = bool(oracle_headroom.get('positive_headroom_exists', False))

    is_complete = (len(rows) == TOTAL_COMBOS) and (criterion_6 is not None)

    if not is_complete and not cli.allow_incomplete_preview:
        out = {
            'status': 'INCOMPLETE', 'QA_GO': None,
            'combos_available': len(rows), 'combos_required': TOTAL_COMBOS,
            'missing_combos': missing,
            'oracle_headroom_computed': criterion_6 is not None,
            'note': 'Refusing to issue a GO/STOP verdict: either not all 72 (cell,method) combos are '
                    'present, or the Dynamic-Oracle headroom diagnostic (criterion 6) has not been '
                    'computed. Pass --allow_incomplete_preview to still see preview numbers (status '
                    'stays INCOMPLETE, QA_GO stays null) -- never treat a preview as the final decision.',
        }
        Path(cli.out_json).write_text(json.dumps(out, indent=2))
        print(json.dumps(out, indent=2))
        return

    if not rows:
        raise SystemExit('[ISSUE][ABORT] no (cell,method) combos available -- cannot compute even a preview')

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

    criteria = {
        '1_overall_avg_qa_lt_noqa': criterion_1,
        '2_cells_qa_wins_ge_7of12': criterion_2,
        '3_5_all_dataset_avgs_qa_le_noqa': criterion_3_5,
        '6_positive_dynamic_oracle_headroom': criterion_6,
    }
    qa_go = is_complete and criterion_1 and criterion_2 and criterion_3_5 and criterion_6

    out = {
        'status': 'COMPLETE' if is_complete else 'INCOMPLETE_PREVIEW',
        'QA_GO': qa_go if is_complete else None,
        'criteria': criteria,
        'overall_avg_delta': overall_avg_delta,
        'cells_with_qa_win': cells_with_qa_win,
        'cell_avg_delta': cell_avg_delta,
        'dataset_avg_delta': dataset_avg_delta,
        'per_cell_method_rows': {f'{c}/{m}': v for (c, m), v in rows.items()},
        'missing_combos': missing,
        'oracle_headroom': oracle_headroom,
        'note': 'Computed from VALIDATION checkpoint_selection.json ONLY. TEST was never read to produce this.'
                + ('' if is_complete else ' PREVIEW ONLY -- not all 72 combos / oracle criterion present, '
                                          'QA_GO is null by construction, do not treat this as a decision.'),
    }
    Path(cli.out_json).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
