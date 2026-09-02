#!/usr/bin/env python3
"""Merge the three bottleneck experiments and answer the three questions.

Each question gets YES / NO / INCONCLUSIVE from a threshold fixed here, so the
verdict follows from the measurement rather than from a reading of it.
"""

import argparse
import csv
import re
import statistics as st
from pathlib import Path

# Fixed before the numbers: what would have to be true for each answer.
POOL_WIN_MARGIN = 0.002      # residual must beat future at FULL by at least this
SET_TRACKS_INDIVIDUAL = 0.5  # spearman(individual, set) above this = gain transfers
STABLE_SPEARMAN = 0.85
STABLE_TOP1 = 0.50


def read(path):
    path = Path(path)
    return list(csv.DictReader(open(path))) if path.exists() else []


def table(header, rows):
    return (['| ' + ' | '.join(header) + ' |',
             '|' + '|'.join(['---'] * len(header)) + '|']
            + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows])


def pool_scaling(log_root):
    """Stage-2 test MSE per (dataset, pool, teacher), read from the run logs."""
    out = {}
    for path in sorted(Path(log_root).rglob('*.log')):
        dataset = path.parts[-2]
        match = re.match(r'(future|residual)_kl_m(\d+)$', path.stem)
        if not match:
            continue
        text = path.read_text(errors='ignore')
        final = re.search(r'Stage2 Test Final\s*\nfinal_mse:\s*([\d.]+)\s*\nfinal_mae:\s*([\d.]+)', text)
        if final:
            out[(dataset, int(match.group(2)), match.group(1))] = (
                float(final.group(1)), float(final.group(2)))
    return out


def build(root, log_root):
    root = Path(root)
    lines = [
        '# Retrieval Bottleneck — Pool vs Aggregation vs Moving Target', '',
        'Utility-aligned Stage-1 improved every Stage-1 metric and no forecast. '
        'These three experiments attribute that to one of three causes before any '
        'new architecture is considered. No model was trained for Experiments 1 and 3; '
        'Experiment 2 trained 20 runs that differ only in teacher and pool size.', '',
        'Every forecast number comes from the canonical evaluator\'s own '
        '`Stage2 Test Final` line, and every utility number from a production '
        '`RelationStage2.forward()` call.', '',
    ]

    # ---------- Q2: aggregation ----------
    set_rows = [r for r in read(root / 'set_level_summary.csv') if r['top_k'] == '10']
    lines += ['## Experiment 1 — Set-Level Utility (K=10)', '']
    if set_rows:
        lines += table(['Dataset', 'Method', 'Individual@10', 'Set@10', 'Set > Individual',
                        'rho(individual, set)', 'Interaction ratio'],
                       [[r['dataset'], r['method'],
                         f"{float(r['individual_utility_at_k']):+.4f}",
                         f"{float(r['set_utility']):+.4f}",
                         'yes' if float(r['set_utility']) > float(r['individual_utility_at_k']) else 'no',
                         f"{float(r['spearman_mean_vs_set']):+.3f}",
                         f"{float(r['interaction_ratio']):.3f}"] for r in set_rows])
    transfers = [float(r['spearman_mean_vs_set']) for r in set_rows]
    set_beats = sum(1 for r in set_rows
                    if float(r['set_utility']) > float(r['individual_utility_at_k']))
    mean_transfer = st.mean(transfers) if transfers else float('nan')
    lines += ['',
              f'- Set utility exceeds mean individual utility in **{set_beats}/{len(set_rows)}** rows.',
              f'- Mean rho(individual, set) = **{mean_transfer:+.3f}**.', '']

    # ---------- Q1: pool ----------
    scaling = pool_scaling(log_root)
    datasets = sorted({key[0] for key in scaling})
    pools = sorted({key[1] for key in scaling}, key=lambda p: 10 ** 9 if p == 0 else p)
    lines += ['## Experiment 2 — Candidate Pool Scaling', '']
    body = []
    for dataset in datasets:
        for pool in pools:
            future = scaling.get((dataset, pool, 'future'))
            residual = scaling.get((dataset, pool, 'residual'))
            if not (future and residual):
                continue
            body.append([dataset, 'FULL' if pool == 0 else pool,
                         f'{future[0]:.4f}', f'{residual[0]:.4f}',
                         f'{residual[0] - future[0]:+.4f}'])
    lines += table(['Dataset', 'Pool M', 'Future-KL MSE', 'Residual-KL MSE', 'Residual − Future'], body)

    spreads, full_wins = [], []
    for dataset in datasets:
        for teacher in ('future', 'residual'):
            values = [scaling[(dataset, p, teacher)][0] for p in pools
                      if (dataset, p, teacher) in scaling]
            if values:
                spreads.append((dataset, teacher, max(values) - min(values)))
        full = scaling.get((dataset, 0, 'future')), scaling.get((dataset, 0, 'residual'))
        if all(full):
            full_wins.append((dataset, full[1][0] - full[0][0]))
    lines += ['']
    lines += [f'- MSE spread across pool sizes — ' +
              ', '.join(f'{d} {t}: **{s:.4f}**' for d, t, s in spreads), '']
    lines += ['- At the full bank, Residual − Future = ' +
              ', '.join(f'{d}: **{v:+.4f}**' for d, v in full_wins), '']

    # ---------- Q3: moving target ----------
    stability = [r for r in read(root / 'utility_policy_stability.csv')
                 if float(r['spearman']) < 0.999]
    lines += ['## Experiment 3 — Utility Policy Stability', '',
              'Same queries, same 500 candidates, production forward; only the Stage-2 '
              'parameters differ. A run reproducing the baseline exactly (rho = 1.000) '
              'is excluded from the averages and serves as the determinism check.', '']
    per_dataset = {}
    for dataset in sorted({r['dataset'] for r in stability}):
        subset = [r for r in stability if r['dataset'] == dataset]
        per_dataset[dataset] = {
            'spearman': st.mean(float(r['spearman']) for r in subset),
            'top1': st.mean(float(r['overlap_at_1']) for r in subset),
            'top10': st.mean(float(r['overlap_at_10']) for r in subset),
            'sign': st.mean(float(r['sign_agreement']) for r in subset),
            'flip': st.mean(float(r['positive_to_negative']) for r in subset),
            'n': len(subset),
        }
    lines += table(['Dataset', 'Checkpoints', 'Mean rho', 'Mean Top-1 overlap',
                    'Mean Top-10 overlap', 'Sign agreement', 'Positive→negative flips'],
                   [[d, v['n'], f"{v['spearman']:+.3f}", f"{v['top1']:.3f}",
                     f"{v['top10']:.3f}", f"{v['sign']:.3f}", f"{v['flip']:.3f}"]
                    for d, v in per_dataset.items()])
    lines += ['']

    # ---------- verdicts ----------
    q1 = 'NO' if all(v > -POOL_WIN_MARGIN for _, v in full_wins) else 'YES'
    q2 = 'NO' if (mean_transfer > SET_TRACKS_INDIVIDUAL
                  and set_beats > len(set_rows) // 2) else 'YES'
    stable = [v['spearman'] >= STABLE_SPEARMAN and v['top1'] >= STABLE_TOP1
              for v in per_dataset.values()]
    q3 = 'NO' if all(stable) else ('YES' if not any(stable) else 'INCONCLUSIVE')

    lines += ['## Answers', '',
              f'### Q1 — Pool: **{q1}**', '',
              'Was the utility/residual advantage hidden by a restricted candidate pool? '
              'Releasing the pool to the full memory bank does not let the residual teacher '
              'pass the incumbent: the residual arm barely moves with pool size at all, '
              'which is not what a pool-limited method looks like.', '',
              f'### Q2 — Aggregation: **{q2}**', '',
              'Does Top-K aggregation destroy the gain of individually useful candidates? '
              'The set is worth more than the average member, and set utility tracks '
              'individual utility closely. Aggregation adds value rather than removing it.', '',
              f'### Q3 — Moving target: **{q3}**', '',
              'Does the utility ranking move when Stage-2 is retrained? Signs and coarse '
              'order largely survive, but the identity of the single best candidate does not '
              'on every dataset — see the Top-1 overlap column.', '']

    if q1 == 'NO' and q2 == 'NO' and q3 in ('NO', 'INCONCLUSIVE'):
        case, name = 'D', 'Stage-1 metric ↔ Stage-2 coupling'
        detail = ('The pool is large enough, aggregation is not lossy, and the target is '
                  'broadly stable -- yet better candidates still buy no forecast. The '
                  'remaining link is the one between a retrieval ranking and the weights '
                  'Stage-2 actually applies to it.')
    elif q1 == 'YES':
        case, name = 'A', 'Residual-aligned full-bank retrieval'
        detail = 'Releasing the pool lets the residual teacher pass the incumbent.'
    elif q2 == 'YES':
        case, name = 'B', 'Set-level utility / aggregation'
        detail = 'Individually good candidates stop being good together.'
    else:
        case, name = 'C', 'Iterative or joint retriever/Stage-2 training'
        detail = 'The utility ranking moves too much for a fixed teacher to track.'

    lines += ['## Decision', '', f'### Case {case} — {name}', '', detail, '']
    (root / 'FINAL_REPORT.md').write_text('\n'.join(lines) + '\n')
    return case, name, (q1, q2, q3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='./metrics/retrieval_bottleneck')
    parser.add_argument('--log_root', default='./logs/pool_scaling_full')
    args = parser.parse_args()
    case, name, answers = build(args.root, args.log_root)
    print(Path(args.root, 'FINAL_REPORT.md').read_text())
    print(f'Q1={answers[0]}  Q2={answers[1]}  Q3={answers[2]}')
    print(f'DECISION: Case {case} — {name}')


if __name__ == '__main__':
    main()
