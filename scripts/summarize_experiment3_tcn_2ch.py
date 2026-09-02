#!/usr/bin/env python3
"""Summarize Experiment 3 Stage-1 retrieval logs as Markdown and CSV.

Experiment 3 is the 2x2 of encoder architecture (MLP, TCN) against encoder input
representation (delta_last only, delta_last + first-order difference), so the
encoder contribution and the input contribution can be read separately instead
of only as the combined arm. Parsing and metric selection are reused from the
Experiment 2 summarizer so both studies report the same fields.
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from summarize_experiment2_diff1 import (  # noqa: E402
    DIAGNOSTIC_FIELDS,
    RETRIEVAL_FIELDS,
    fmt,
    markdown_table,
    parse_stage1_test,
)

# (encoder label, input label, log tag)
METHODS = (
    ('MLP', 'delta_last', 'mlp_delta_last'),
    ('MLP', 'delta_last + diff1', 'mlp_2ch'),
    ('TCN', 'delta_last', 'tcn_delta_last'),
    ('TCN', 'delta_last + diff1', 'tcn_2ch'),
)

BASELINE_TAG = 'mlp_delta_last'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=Path, required=True)
    parser.add_argument('--log-suffix', default='')
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()

    parsed = {
        tag: parse_stage1_test(args.log_dir / f'{tag}{args.log_suffix}.log')
        for _, _, tag in METHODS
    }
    baseline = parsed[BASELINE_TAG]

    retrieval_rows = []
    delta_rows = []
    csv_rows = []
    for encoder, input_space, tag in METHODS:
        metrics = parsed[tag]
        row = {
            'encoder': encoder,
            'input': input_space,
            **{
                output_name: metrics.get(metric_name)
                for output_name, metric_name in RETRIEVAL_FIELDS.items()
            },
        }
        csv_rows.append(row)
        retrieval_rows.append([
            encoder,
            input_space,
            fmt(row['recall_at_10']),
            fmt(row['ndcg_at_10']),
            fmt(row['top10_future_mse']),
            fmt(row['oracle_top10_future_mse']),
            fmt(row['retrieval_regret']),
        ])

        # Signed deltas against the MLP/delta_last baseline: positive is better
        # for recall and NDCG, negative is better for MSE and regret.
        def delta(name):
            current = metrics.get(RETRIEVAL_FIELDS[name])
            reference = baseline.get(RETRIEVAL_FIELDS[name])
            if current is None or reference is None:
                return None
            return current - reference

        delta_rows.append([
            encoder,
            input_space,
            fmt(delta('recall_at_10')),
            fmt(delta('ndcg_at_10')),
            fmt(delta('top10_future_mse')),
            fmt(delta('retrieval_regret')),
        ])

    diagnostic_rows = [
        [
            f'{encoder} / {input_space}',
            *[fmt(parsed[tag].get(name)) for name in DIAGNOSTIC_FIELDS.values()],
        ]
        for encoder, input_space, tag in METHODS
    ]

    report = '\n\n'.join((
        '# Experiment 3 — TCN Encoder and 2-Channel Relation Input',
        markdown_table(
            [
                'Encoder', 'Input', 'Recall@10 ↑', 'NDCG@10 ↑',
                'Top-10 Future MSE ↓', 'Oracle Top-10 Future MSE ↓', 'Regret ↓',
            ],
            retrieval_rows,
        ),
        f'## Delta vs baseline ({BASELINE_TAG})',
        markdown_table(
            ['Encoder', 'Input', 'ΔRecall@10 ↑', 'ΔNDCG@10 ↑',
             'ΔTop-10 Future MSE ↓', 'ΔRegret ↓'],
            delta_rows,
        ),
        markdown_table(['Arm', *DIAGNOSTIC_FIELDS.keys()], diagnostic_rows),
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
