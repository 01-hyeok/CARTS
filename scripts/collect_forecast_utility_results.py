#!/usr/bin/env python3
"""STEP 5 -- pull Stage-1 and Stage-2 numbers out of the run logs into one table.

The forecast numbers come from the canonical evaluator's own `Stage2 Test Final`
lines, never from a reconstruction. That is the whole point: a previous round of
this work was invalidated because the analysis recomputed the forecast itself.
"""

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.retrieval_diagnostics import append_row  # noqa: E402

STAGE1_KEYS = [
    'utility_gap_recovery_at_10', 'utility_ndcg_at_10', 'utility_retrieved_at_10',
    'utility_random_at_10', 'utility_oracle_at_10', 'utility_positive_rate_at_10',
    'utility_best_available', 'utility_null_probability', 'utility_student_entropy',
    'utility_teacher_entropy',
    'student_oracle_recall_at_10', 'student_retrieved_future_mse_at_10',
    'oracle_future_mse_at_10', 'student_retrieval_regret_at_10',
]
STAGE2_KEYS = ['final_mse', 'final_mae', 'base_mse', 'base_mae', 'lambda_mean',
               'lambda_std', 'ret_mse', 'retrieval_gain']
COLUMNS = ['dataset', 'pred_len', 'arm', 'teacher', 'objective', 'null_mode',
           'stage2_test_mse', 'stage2_test_mae'] + STAGE2_KEYS + STAGE1_KEYS + ['log']

ARM_SPEC = {
    'future_kl_full': ('future', 'kl', 'off'),
    'full_bank_kl': ('future', 'kl', 'off'),
    'future_kl_pool': ('future', 'kl', 'off'),
    'residual_kl': ('residual', 'kl', 'off'),
    'utility_kl': ('utility', 'kl', 'off'),
    'expected_utility': ('utility', 'expected_utility', 'off'),
    'utility_kl_null': ('utility', 'kl', 'query'),
    'expected_utility_null': ('utility', 'expected_utility', 'query'),
}


def parse_metric_line(line):
    """`key: value | key: value` -> dict of floats."""
    out = {}
    for key, value in re.findall(r'([a-zA-Z0-9_]+):\s*(-?[\d.]+(?:e-?\d+)?|nan)', line):
        try:
            out[key] = float(value)
        except ValueError:
            continue
    return out


def parse_log(path):
    text = Path(path).read_text(errors='ignore')
    if '### RUN COMPLETE' not in text:
        return None
    row = {}
    for line in text.splitlines():
        if line.startswith('Stage1 Test |'):
            stage1 = parse_metric_line(line)
            row.update({key: stage1[key] for key in STAGE1_KEYS if key in stage1})
        elif line.startswith('Stage2 Test |'):
            stage2 = parse_metric_line(line)
            row.update({key: stage2[key] for key in STAGE2_KEYS if key in stage2})
    final = re.search(r'Stage2 Test Final\s*\nfinal_mse:\s*([\d.]+)\s*\nfinal_mae:\s*([\d.]+)', text)
    if final:
        row['stage2_test_mse'] = float(final.group(1))
        row['stage2_test_mae'] = float(final.group(2))
    return row


def collect(log_root, arm_filter=None):
    rows = []
    for path in sorted(Path(log_root).rglob('*.log')):
        arm = path.stem
        if arm not in ARM_SPEC or (arm_filter and arm not in arm_filter):
            continue
        parts = path.parts
        try:
            dataset = parts[-3]
            pred_len = int(parts[-2].replace('pred', ''))
        except (IndexError, ValueError):
            continue
        parsed = parse_log(path)
        if parsed is None:
            print(f'[incomplete] {path}')
            continue
        teacher, objective, null_mode = ARM_SPEC[arm]
        canonical = 'future_kl_full' if arm == 'full_bank_kl' else arm
        parsed.update({'dataset': dataset, 'pred_len': pred_len, 'arm': canonical,
                       'teacher': teacher, 'objective': objective,
                       'null_mode': null_mode, 'log': str(path)})
        rows.append(parsed)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log_root', required=True)
    parser.add_argument('--baseline_log_root', default='')
    parser.add_argument('--metric_dir', default='./metrics/forecast_utility_stage1')
    args = parser.parse_args()

    rows = collect(args.log_root)
    if args.baseline_log_root and Path(args.baseline_log_root).is_dir():
        rows += collect(args.baseline_log_root, arm_filter={'full_bank_kl'})

    csv = Path(args.metric_dir) / 'stage2_forecasting.csv'
    if csv.exists():
        csv.unlink()
    for row in sorted(rows, key=lambda r: (r['dataset'], r['pred_len'], r['arm'])):
        append_row(csv, row, COLUMNS)
        print(f"{row['dataset']}/{row['pred_len']:<4} {row['arm']:<22} "
              f"MSE={row.get('stage2_test_mse', float('nan')):.4f} "
              f"MAE={row.get('stage2_test_mae', float('nan')):.4f} "
              f"UtilRec={row.get('utility_gap_recovery_at_10', float('nan')):+.4f} "
              f"R@10={row.get('student_oracle_recall_at_10', float('nan')):.4f}")
    print(f'wrote {csv} ({len(rows)} runs)')


if __name__ == '__main__':
    main()
