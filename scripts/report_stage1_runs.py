"""One reporting format for every Stage-1 sweep, so arms stay comparable.

Three things kept going wrong when each sweep was summarised ad hoc. Numbers from
runs with different baselines were placed side by side. Relative gains were quoted
without the absolute values that make them readable -- "+68% Recall@10" turned out
to mean 0.44 of the Oracle's ten candidates instead of 0.74. And the selected
epoch was left out, which hides the difference between an arm that improved for
ten epochs and one whose first epoch was its best.

So every row carries: the pool it was ranked against, where in that pool the
retrieved candidates actually sit, the improvement over a named baseline, and the
epoch each criterion would have selected. All 112 logged metrics go to the CSV;
the printed table is the subset that answers the current question.

An epoch-1 pick on `loss` while a retrieval criterion keeps improving is the
signature this exists to surface: validation loss turning up while retrieval
quality is still climbing means the two objectives have parted company, and the
selection rule decides which one the checkpoint follows.
"""

import argparse
import csv
import os
import re
from pathlib import Path

# (name, metric key, +1 when lower is better)
CRITERIA = (
    ('loss', 'loss', +1),
    ('recall10', 'student_oracle_recall_at_10', -1),
    ('ndcg10', 'student_ndcg_at_10', -1),
    ('retMSE10', 'student_retrieved_future_mse_at_10', +1),
)

# Printed table. Recall and Spearman lead because they are the current primaries;
# the rest are kept visible so a win on one metric that costs another is obvious.
SHOWN = (
    ('R@1', 'student_oracle_recall_at_1', '{:.4f}'),
    ('R@5', 'student_oracle_recall_at_5', '{:.4f}'),
    ('R@10', 'student_oracle_recall_at_10', '{:.4f}'),
    ('Spearman', 'student_spearman_score_vs_negative_mse', '{:.4f}'),
    ('NDCG@10', 'student_ndcg_at_10', '{:.4f}'),
    ('retMSE@10', 'student_retrieved_future_mse_at_10', '{:.4f}'),
    ('regret@10', 'student_retrieval_regret_at_10', '{:.4f}'),
    ('effRank', 'online_collapse_effective_rank_mean', '{:.1f}'),
)


def parse_log(path):
    """Every validation epoch of one run as a list of metric dicts."""
    text = Path(path).read_text()
    if '### RUN COMPLETE' not in text:
        return None
    epochs = []
    for block in re.findall(r'Epoch \d+ Vali \| (.*)', text):
        row = {}
        for key, value in re.findall(r'([a-z0-9_@]+): ([-0-9.eE]+)', block):
            if key.startswith(('self_', 'cross_')):
                continue
            try:
                row[key] = float(value)
            except ValueError:
                continue
        epochs.append(row)
    return epochs or None


def pick(epochs, key, sign):
    """Index of the epoch a criterion would select; None when it never reported."""
    scored = [(i, e[key]) for i, e in enumerate(epochs) if key in e and e[key] == e[key]]
    if not scored:
        return None
    return min(scored, key=lambda pair: sign * pair[1])[0]


def collect(log_root, dataset, preds, arms):
    runs = []
    for pred in preds:
        for arm in arms:
            path = Path(log_root) / dataset / f'pred{pred}' / f'{arm}.log'
            if not path.exists():
                continue
            epochs = parse_log(path)
            if epochs is None:
                continue
            picks = {name: pick(epochs, key, sign) for name, key, sign in CRITERIA}
            primary = picks.get('retMSE10')
            if primary is None:
                primary = len(epochs) - 1
            runs.append({
                'pred': pred, 'arm': arm, 'ran': len(epochs),
                'picks': picks, 'best': primary, 'metrics': epochs[primary],
                'epochs': epochs,
            })
    return runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_root', required=True)
    parser.add_argument('--dataset', default='ETTh1')
    parser.add_argument('--preds', default='96 192 336 720')
    parser.add_argument('--arms', required=True,
                        help='space separated; the first is the baseline for the delta column')
    parser.add_argument('--csv', default='')
    parser.add_argument('--title', default='')
    args = parser.parse_args()

    preds = [int(p) for p in args.preds.split()]
    arms = args.arms.split()
    runs = collect(args.log_root, args.dataset, preds, arms)
    if not runs:
        print(f'no completed runs under {args.log_root}')
        return

    baseline_arm = arms[0]
    if args.title:
        print(f'\n{args.title}')
    print(f'baseline = {baseline_arm}   |   selected epoch = validation retrieved_future_mse@10\n')

    header = (f"{'pred':>5} {'arm':<16} {'pool':>6} " +
              ' '.join(f'{name:>9}' for name, _, _ in SHOWN) +
              f" {'ΔR@10':>8} {'ΔretMSE':>8}  " +
              ' '.join(f'{name:>9}' for name, _, _ in CRITERIA) + f" {'ran':>4}")
    print(header)
    print('-' * len(header))

    for pred in preds:
        group = [r for r in runs if r['pred'] == pred]
        base = next((r for r in group if r['arm'] == baseline_arm), None)
        for run in group:
            m = run['metrics']
            pool = m.get('valid_candidate_pool', float('nan'))
            cells = [fmt.format(m[key]) if key in m else '--' for _, key, fmt in SHOWN]
            if base is not None and run is not base:
                b = base['metrics']
                dr = m.get('student_oracle_recall_at_10'), b.get('student_oracle_recall_at_10')
                dm = m.get('student_retrieved_future_mse_at_10'), b.get('student_retrieved_future_mse_at_10')
                d_recall = f'{100 * (dr[0] - dr[1]) / dr[1]:+.1f}%' if dr[0] and dr[1] else '--'
                d_mse = f'{100 * (dm[0] - dm[1]) / dm[1]:+.2f}%' if dm[0] and dm[1] else '--'
            else:
                d_recall = d_mse = '·'
            eps = [f'ep{run["picks"][n] + 1}' if run['picks'][n] is not None else '--'
                   for n, _, _ in CRITERIA]
            print(f"{pred:>5} {run['arm']:<16} {pool:>6.0f} " +
                  ' '.join(f'{c:>9}' for c in cells) +
                  f" {d_recall:>8} {d_mse:>8}  " +
                  ' '.join(f'{e:>9}' for e in eps) + f" {run['ran']:>4}")
        print()

    # Where the retrieved candidates sit in the pool, and how far the Oracle is.
    print('position in the pool (lower is better; random would be half the pool)')
    print(f"{'pred':>5} {'arm':<16} {'pool':>6} {'oracle Top-10 mean rank':>24} {'as % of pool':>13}")
    print('-' * 70)
    for pred in preds:
        for run in [r for r in runs if r['pred'] == pred]:
            m = run['metrics']
            rank = m.get('oracle_top10_mean_rank')
            frac = m.get('oracle_top10_rank_fraction')
            pool = m.get('valid_candidate_pool', float('nan'))
            if rank is None:
                continue
            print(f"{pred:>5} {run['arm']:<16} {pool:>6.0f} {rank:>24.1f} {100 * frac:>12.1f}%")
        print()

    # Overfitting signature: `loss` peaks early while retrieval keeps improving.
    early = [r for r in runs
             if r['picks'].get('loss') == 0 and (r['picks'].get('retMSE10') or 0) > 0]
    if early:
        print('validation loss best at epoch 1 while retrieval kept improving:')
        for run in early:
            rm = run['picks']['retMSE10']
            first = run['epochs'][0].get('student_retrieved_future_mse_at_10')
            best = run['epochs'][rm].get('student_retrieved_future_mse_at_10')
            print(f"   pred {run['pred']:>3} {run['arm']:<16} "
                  f"retMSE@10 ep1 {first:.4f} -> ep{rm + 1} {best:.4f} "
                  f"({100 * (best - first) / first:+.2f}%)")
        print('   selecting on loss would have stopped at epoch 1 in each of these.')
    else:
        print('no run had its validation loss best at epoch 1.')

    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({k for r in runs for k in r['metrics']})
        with open(args.csv, 'w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['dataset', 'pred', 'arm', 'ran', 'selected_epoch'] +
                            [f'epoch_by_{n}' for n, _, _ in CRITERIA] + keys)
            for run in runs:
                writer.writerow(
                    [args.dataset, run['pred'], run['arm'], run['ran'], run['best'] + 1] +
                    [(run['picks'][n] + 1) if run['picks'][n] is not None else ''
                     for n, _, _ in CRITERIA] +
                    [run['metrics'].get(k, '') for k in keys])
        print(f'\nall {len(keys)} metrics written to {args.csv}')


if __name__ == '__main__':
    main()
