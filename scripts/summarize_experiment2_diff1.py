#!/usr/bin/env python3
"""Summarize Experiment 2 Stage-1 retrieval logs as Markdown and CSV."""

import argparse
import csv
import math
import re
from pathlib import Path


METHODS = (
    ('Diff1 Direct', 'None', '-', 'direct'),
    ('Diff1 Encoder', 'Trained', 'Future MSE', 'future_mse'),
    ('Diff1 Encoder', 'Trained', 'EMA', 'ema'),
)

RETRIEVAL_FIELDS = {
    'recall_at_10': 'student_oracle_recall_at_10',
    'ndcg_at_10': 'student_ndcg_at_10',
    'top10_future_mse': 'student_retrieved_future_mse_at_10',
    'oracle_top10_future_mse': 'oracle_future_mse_at_10',
    'retrieval_regret': 'student_retrieval_regret_at_10',
}

DIAGNOSTIC_FIELDS = {
    'teacher_entropy': 'teacher_entropy',
    'teacher_normalized_entropy': 'teacher_entropy_normalized',
    'teacher_effective_candidates': 'teacher_effective_candidates',
    'student_entropy': 'student_entropy',
    'student_normalized_entropy': 'student_entropy_normalized',
    'student_effective_candidates': 'student_effective_candidates',
    'pairwise_embedding_cosine': 'online_collapse_pairwise_cosine_mean',
    'embedding_variance': 'online_collapse_embedding_variance_mean',
    'effective_rank': 'online_collapse_effective_rank_mean',
    'dead_dim_fraction': 'online_collapse_dead_dimension_fraction_mean',
    'ema_pairwise_embedding_cosine': 'ema_collapse_pairwise_cosine_mean',
    'ema_embedding_variance': 'ema_collapse_embedding_variance_mean',
    'ema_effective_rank': 'ema_collapse_effective_rank_mean',
    'ema_dead_dim_fraction': 'ema_collapse_dead_dimension_fraction_mean',
    'encoder_grad_norm': 'encoder_grad_norm',
}

METRIC_PATTERN = re.compile(
    r'(?:^|\|)\s*([A-Za-z0-9_@.-]+):\s*'
    r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)'
)


def parse_stage1_test(path):
    text = path.read_text(encoding='utf-8', errors='replace')
    lines = [line for line in text.splitlines() if line.startswith('Stage1 Test |')]
    if not lines:
        raise ValueError(f'No "Stage1 Test" metric line found in {path}')
    metrics = {
        key: float(value)
        for key, value in METRIC_PATTERN.findall(lines[-1])
    }
    train_lines = [
        line for line in text.splitlines()
        if re.match(r'^Epoch \d+ Train \|', line)
        and 'encoder_grad_norm:' in line
    ]
    if train_lines:
        train_metrics = {
            key: float(value)
            for key, value in METRIC_PATTERN.findall(train_lines[-1])
        }
        if 'encoder_grad_norm' in train_metrics:
            metrics['encoder_grad_norm'] = train_metrics['encoder_grad_norm']
    return metrics


def fmt(value):
    if value is None or not math.isfinite(value):
        return ''
    return f'{value:.6f}'


def markdown_table(headers, rows):
    lines = [
        '| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join('---' for _ in headers) + ' |',
    ]
    lines.extend('| ' + ' | '.join(row) + ' |' for row in rows)
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=Path, required=True)
    parser.add_argument('--log-suffix', default='')
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()

    parsed = {}
    for _, _, _, tag in METHODS:
        parsed[tag] = parse_stage1_test(
            args.log_dir / f'{tag}{args.log_suffix}.log'
        )

    retrieval_rows = []
    csv_rows = []
    for method, encoder, teacher, tag in METHODS:
        metrics = parsed[tag]
        row = {
            'method': method,
            'encoder': encoder,
            'teacher': teacher,
            **{
                output_name: metrics.get(metric_name)
                for output_name, metric_name in RETRIEVAL_FIELDS.items()
            },
        }
        csv_rows.append(row)
        retrieval_rows.append([
            method,
            encoder,
            teacher,
            fmt(row['recall_at_10']),
            fmt(row['ndcg_at_10']),
            fmt(row['top10_future_mse']),
            fmt(row['oracle_top10_future_mse']),
            fmt(row['retrieval_regret']),
        ])

    diagnostic_rows = []
    for _, _, teacher, tag in METHODS[1:]:
        metrics = parsed[tag]
        diagnostic_rows.append([
            teacher,
            *[
                fmt(metrics.get(metric_name))
                for metric_name in DIAGNOSTIC_FIELDS.values()
            ],
        ])

    report = '\n\n'.join((
        '# Experiment 2 — Diff1 Retrieval Results',
        markdown_table(
            [
                'Method', 'Encoder', 'Teacher', 'Recall@10 ↑', 'NDCG@10 ↑',
                'Top-10 Future MSE ↓', 'Oracle Top-10 Future MSE ↓', 'Regret ↓',
            ],
            retrieval_rows,
        ),
        markdown_table(
            ['Teacher', *DIAGNOSTIC_FIELDS.keys()],
            diagnostic_rows,
        ),
    )) + '\n'

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix('.md').write_text(report, encoding='utf-8')
    with args.output_prefix.with_suffix('.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(report, end='')


if __name__ == '__main__':
    main()
