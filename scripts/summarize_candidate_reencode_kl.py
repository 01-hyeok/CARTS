#!/usr/bin/env python3
"""Summarize the Candidate-side Gradient Recovery sweep.

Reads the per-arm logs written by run_candidate_reencode_kl_full.sh and emits a
CSV plus a Markdown table that puts the whole chain on one row:

    candidate gradient -> Stage-1 retrieval -> Stage-2 forecasting
"""

import argparse
import csv
import re
from pathlib import Path

ARM_ORDER = ['full_bank_kl', 'selected100_detached_kl', 'selected100_reencode_kl']
ARM_LABEL = {
    'full_bank_kl': 'Full-Bank KL',
    'selected100_detached_kl': 'Selected-100 Detached KL',
    'selected100_reencode_kl': 'Selected-100 Re-encode KL',
}
BASELINE_ARM = 'full_bank_kl'

# Stage-1 metrics lifted from the "Stage1 Test" / last "Epoch N Vali" lines.
STAGE1_KEYS = [
    'student_oracle_recall_at_1',
    'student_oracle_recall_at_5',
    'student_oracle_recall_at_10',
    'student_retrieved_future_mse_at_10',
    'oracle_future_mse_at_10',
    'student_retrieval_regret_at_10',
    'student_ndcg_at_10',
    'student_spearman_score_vs_negative_mse',
    'bank_oracle_recall_at_100',
    'oracle_missing_count_before_injection',
]
TRAIN_ONLY_KEYS = [
    'bank_oracle_recall_at_10',
    'bank_oracle_recall_at_100',
    'oracle_count_in_bank_top_m',
    'oracle_missing_count_before_injection',
    'candidate_unique_encoded',
]
METRIC_PATTERN = re.compile(r'([A-Za-z@0-9_]+): (-?[\d.]+(?:[eE][-+]?\d+)?)')


def parse_metric_line(line):
    return {k: float(v) for k, v in METRIC_PATTERN.findall(line)}


def parse_log(path):
    """Pull the Stage-1 and Stage-2 numbers out of one arm's log."""
    record = {'log': str(path), 'complete': False}
    stage1_test = None
    stage1_vali = None
    stage1_train = None
    stage2 = {}

    with open(path, errors='replace') as handle:
        lines = handle.readlines()

    # Stage-2 prints its own "Epoch N Train/Vali" lines into the same log, so
    # the per-epoch Stage-1 lines have to be read before that boundary or the
    # Stage-2 numbers silently overwrite them.
    in_stage2 = False
    for index, line in enumerate(lines):
        if line.startswith('### STAGE2'):
            in_stage2 = True
        if line.startswith('### RUN COMPLETE'):
            record['complete'] = True
        elif line.startswith('Stage1 Test'):
            stage1_test = parse_metric_line(line)
        elif not in_stage2 and ' Vali | ' in line and line.startswith('Epoch '):
            stage1_vali = parse_metric_line(line)
        elif not in_stage2 and ' Train | ' in line and line.startswith('Epoch '):
            # Candidate mining runs training-only, so its diagnostics never
            # reach the validation line.
            stage1_train = parse_metric_line(line)
        elif line.startswith('Stage2 Test Final'):
            for follow in lines[index + 1:index + 3]:
                stage2.update(parse_metric_line(follow))
        elif line.startswith('[stage1] new best on'):
            record['stage1_best_criterion'] = line.split('new best on')[1].split(':')[0].strip()

    for key in TRAIN_ONLY_KEYS:
        if stage1_train and key in stage1_train:
            record[f'stage1_train_{key}'] = stage1_train[key]
    for key in STAGE1_KEYS:
        if stage1_vali and key in stage1_vali:
            record[f'stage1_val_{key}'] = stage1_vali[key]
        if stage1_test and key in stage1_test:
            record[f'stage1_test_{key}'] = stage1_test[key]

    record['stage2_test_mse'] = stage2.get('final_mse')
    record['stage2_test_mae'] = stage2.get('final_mae')
    return record


def collect(log_root):
    rows = []
    for log_path in sorted(Path(log_root).glob('*/pred*/*.log')):
        arm = log_path.stem
        pred_len = int(log_path.parent.name.replace('pred', ''))
        dataset = log_path.parent.parent.name
        record = parse_log(log_path)
        record.update({'dataset': dataset, 'pred_len': pred_len, 'arm': arm})
        rows.append(record)
    rows.sort(key=lambda r: (
        r['dataset'],
        r['pred_len'],
        ARM_ORDER.index(r['arm']) if r['arm'] in ARM_ORDER else 99,
    ))
    return rows


def add_deltas(rows):
    """Per (dataset, pred_len), the change against the full-bank baseline arm."""
    baseline = {
        (r['dataset'], r['pred_len']): r
        for r in rows
        if r['arm'] == BASELINE_ARM
    }
    for row in rows:
        base = baseline.get((row['dataset'], row['pred_len']))
        if base is None or base is row:
            continue
        recall_key = 'stage1_test_student_oracle_recall_at_10'
        if row.get(recall_key) is not None and base.get(recall_key) is not None:
            row['delta_recall10_vs_original'] = row[recall_key] - base[recall_key]
        if row.get('stage2_test_mse') is not None and base.get('stage2_test_mse') is not None:
            row['delta_forecast_mse_vs_original'] = (
                row['stage2_test_mse'] - base['stage2_test_mse']
            )
    return rows


CSV_COLUMNS = [
    'dataset', 'pred_len', 'arm', 'complete',
    'stage1_val_student_oracle_recall_at_1',
    'stage1_val_student_oracle_recall_at_5',
    'stage1_val_student_oracle_recall_at_10',
    'stage1_test_student_oracle_recall_at_1',
    'stage1_test_student_oracle_recall_at_5',
    'stage1_test_student_oracle_recall_at_10',
    'stage1_test_student_retrieved_future_mse_at_10',
    'stage1_test_oracle_future_mse_at_10',
    'stage1_test_student_retrieval_regret_at_10',
    'stage1_test_student_ndcg_at_10',
    'stage1_test_student_spearman_score_vs_negative_mse',
    'stage1_train_bank_oracle_recall_at_10',
    'stage1_train_bank_oracle_recall_at_100',
    'stage1_train_oracle_missing_count_before_injection',
    'stage2_test_mse', 'stage2_test_mae',
    'delta_recall10_vs_original', 'delta_forecast_mse_vs_original',
    'stage1_best_criterion', 'log',
]

HEADLINE = [
    ('Dataset', 'dataset', '{}'),
    ('Pred', 'pred_len', '{}'),
    ('Method', '_label', '{}'),
    ('R@10 ↑', 'stage1_test_student_oracle_recall_at_10', '{:.4f}'),
    ('Retrieved MSE@10 ↓', 'stage1_test_student_retrieved_future_mse_at_10', '{:.4f}'),
    ('Regret@10 ↓', 'stage1_test_student_retrieval_regret_at_10', '{:.4f}'),
    ('Forecast MSE ↓', 'stage2_test_mse', '{:.4f}'),
    ('MAE ↓', 'stage2_test_mae', '{:.4f}'),
]


def format_cell(row, key, spec):
    value = row.get(key)
    if value is None:
        return '—'
    if isinstance(value, str):
        return value
    return spec.format(value)


def write_markdown(rows, path):
    lines = [
        '# Candidate-side Gradient Recovery',
        '',
        'Candidate gradient → Stage-1 retrieval → Stage-2 forecasting.',
        'A vs B isolates the candidate subset; **B vs C isolates the candidate gradient**.',
        '',
        '| ' + ' | '.join(name for name, _, _ in HEADLINE) + ' |',
        '|' + '|'.join(['---'] * len(HEADLINE)) + '|',
    ]
    for row in rows:
        row = dict(row)
        row['_label'] = ARM_LABEL.get(row['arm'], row['arm'])
        if not row.get('complete'):
            row['_label'] += ' (incomplete)'
        lines.append(
            '| ' + ' | '.join(
                format_cell(row, key, spec) for _, key, spec in HEADLINE
            ) + ' |'
        )

    lines += ['', '## Baseline 대비 변화 (Full-Bank KL 기준)', '']
    lines += [
        '| Dataset | Pred | Method | ΔR@10 | ΔForecast MSE |',
        '|---|---|---|---|---|',
    ]
    for row in rows:
        if row['arm'] == BASELINE_ARM:
            continue
        lines.append(
            f"| {row['dataset']} | {row['pred_len']} | {ARM_LABEL.get(row['arm'], row['arm'])} "
            f"| {format_cell(row, 'delta_recall10_vs_original', '{:+.4f}')} "
            f"| {format_cell(row, 'delta_forecast_mse_vs_original', '{:+.4f}')} |"
        )

    lines += ['', '## Bank mining diagnostic (Oracle injection 이전)', '']
    lines += ['| Dataset | Pred | Method | bank R@100 | missing/query |', '|---|---|---|---|---|']
    for row in rows:
        if row['arm'] == BASELINE_ARM:
            continue
        lines.append(
            f"| {row['dataset']} | {row['pred_len']} | {ARM_LABEL.get(row['arm'], row['arm'])} "
            f"| {format_cell(row, 'stage1_train_bank_oracle_recall_at_100', '{:.4f}')} "
            f"| {format_cell(row, 'stage1_train_oracle_missing_count_before_injection', '{:.2f}')} |"
        )

    Path(path).write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log_root', required=True)
    parser.add_argument('--out_dir', required=True)
    args = parser.parse_args()

    rows = add_deltas(collect(args.log_root))
    if not rows:
        raise SystemExit(f'no arm logs found under {args.log_root}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'candidate_reencode_kl_summary.csv'
    md_path = out_dir / 'candidate_reencode_kl_summary.md'

    with open(csv_path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, md_path)

    print(Path(md_path).read_text())
    print(f'csv:      {csv_path}')
    print(f'markdown: {md_path}')


if __name__ == '__main__':
    main()
