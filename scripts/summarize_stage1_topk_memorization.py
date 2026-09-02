#!/usr/bin/env python3
"""Compare the Stage-1 Oracle Top-K memorization conditions side by side.

Reads the JSON summaries written by --stage1_overfit_summary_path and prints
one table of Train Recall@1/5/10 plus the case-1..4 reading of the result.
"""

import argparse
import json
from pathlib import Path

SUCCESS_THRESHOLD = 0.95

COLUMNS = (
    ('input_space', 'relation_input_space', 11, 's'),
    ('candidates', 'candidate_mode', 26, 's'),
    ('loss', 'topk_coverage_loss', 10, '.4f'),
    ('Recall@1', 'best_train_recall_at_1', 9, '.4f'),
    ('Recall@5', 'best_train_recall_at_5', 9, '.4f'),
    ('Recall@10', 'best_train_recall_at_10', 10, '.4f'),
    ('regret@10', 'final_student_retrieval_regret_at_10', 10, '.4f'),
    ('verdict', 'verdict', 8, 's'),
)


def load_summaries(summary_dir):
    summaries = []
    for path in sorted(Path(summary_dir).glob('*.json')):
        with open(path) as handle:
            record = json.load(handle)
        record['_path'] = str(path)
        record['verdict'] = (
            'PASS'
            if record.get('best_train_recall_at_10', 0.0) >= SUCCESS_THRESHOLD
            else 'FAIL'
        )
        summaries.append(record)
    return summaries


def format_cell(record, key, width, spec):
    value = record.get(key)
    if value is None:
        return 'n/a'.rjust(width)
    if spec == 's':
        return str(value).rjust(width)
    return format(float(value), spec).rjust(width)


def interpret(summaries):
    """Map the four conditions onto the case 1..4 readings."""
    by_condition = {
        (record['relation_input_space'], record['candidate_mode'].startswith('differentiable')):
        record.get('best_train_recall_at_10', float('nan'))
        for record in summaries
    }
    lines = []

    def recall(space, differentiable):
        return by_condition.get((space, differentiable))

    absolute_diff = recall('absolute', True)
    absolute_bank = recall('absolute', False)
    delta_diff = recall('delta_last', True)
    delta_bank = recall('delta_last', False)

    if absolute_diff is not None and absolute_diff >= SUCCESS_THRESHOLD:
        lines.append(
            'Case 1: absolute + fully differentiable reaches Recall@10 '
            f'{absolute_diff:.4f} >= {SUCCESS_THRESHOLD}. The encoder + cosine '
            'structure has the capacity to represent the Oracle Top-K ranking. '
            'The open question moves to why unseen queries do not generalize.'
        )
    elif absolute_diff is not None:
        lines.append(
            'Case 4: absolute + fully differentiable only reaches Recall@10 '
            f'{absolute_diff:.4f}. Before generalization, check whether this '
            'encoder architecture + cosine geometry can express the Oracle '
            'Top-K ranking at all.'
        )

    if (
        absolute_bank is not None
        and absolute_diff is not None
        and absolute_diff - absolute_bank >= 0.1
    ):
        lines.append(
            'Case 2: the differentiable candidate path beats the step-refresh '
            f'key bank on absolute ({absolute_diff:.4f} vs {absolute_bank:.4f}). '
            'This points at the memory bank / gradient flow / stale candidate '
            'structure obstructing optimization, not at encoder capacity.'
        )
    if (
        delta_bank is not None
        and delta_diff is not None
        and delta_diff - delta_bank >= 0.1
    ):
        lines.append(
            'Case 2 (delta_last): the differentiable candidate path beats the '
            f'step-refresh key bank ({delta_diff:.4f} vs {delta_bank:.4f}).'
        )

    for mode, absolute_value, delta_value in (
        ('step-refresh key bank', absolute_bank, delta_bank),
        ('fully differentiable', absolute_diff, delta_diff),
    ):
        if absolute_value is None or delta_value is None:
            continue
        if absolute_value - delta_value >= 0.1:
            lines.append(
                f'Case 3 ({mode}): absolute {absolute_value:.4f} clearly beats '
                f'delta_last {delta_value:.4f}. The delta-last transform may be '
                'discarding past information the Future-MSE Oracle depends on.'
            )

    if not lines:
        lines.append('No case-1..4 pattern matched; read the table directly.')
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--summary_dir',
        default='./metrics/stage1_topk_memorization',
        help='Directory holding the per-condition JSON summaries',
    )
    args = parser.parse_args()

    summaries = load_summaries(args.summary_dir)
    if not summaries:
        raise SystemExit(f'no JSON summaries found under {args.summary_dir}')

    summaries.sort(
        key=lambda record: (
            record['relation_input_space'],
            record['candidate_mode'],
        )
    )
    reference = summaries[0]
    print('Stage-1 Oracle Top-K memorization sanity check')
    print(
        '  fixed: queries={queries} candidates={candidates} K={coverage_top_k} '
        'target_channel={target_channel} self_only={self_only} '
        'similarity={retrieval_similarity} loss={stage1_loss_mode} '
        'oracle_space={relation_teacher_space} seed={seed}'.format(**reference)
    )
    print(f'  success criterion: train Recall@10 >= {SUCCESS_THRESHOLD}')
    print()

    header = ' | '.join(name.rjust(width) for name, _, width, _ in COLUMNS)
    print(header)
    print('-' * len(header))
    for record in summaries:
        print(' | '.join(
            format_cell(record, key, width, spec)
            for _, key, width, spec in COLUMNS
        ))
    print()
    print('Reading:')
    for line in interpret(summaries):
        print(f'  - {line}')


if __name__ == '__main__':
    main()
