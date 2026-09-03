"""Recompute stored Oracle numbers from the per-query rows and compare.

The summary CSV is an average of averages produced inside a training-shaped
loop; the per-query CSV is the same quantities one row at a time. Agreement
between them is what rules out a reduction taken over the wrong axis, a
broadcast that silently expanded, or a mean-of-means that is not the mean.

Checks, per horizon:
  * every selected set holds exactly K distinct, valid candidates
  * softmax weights sum to one
  * I = A_uniform + V_uniform, and the weighted analogue
  * the summary means equal the per-query means

Reads artifacts only; runs no model.
"""

import argparse
import csv
import glob
import os

TOL = 1e-6


def load(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='logs/presentation_202609/oracle_full/ETTh1')
    ap.add_argument('--k', type=int, default=10)
    ap.add_argument('--samples', type=int, default=5)
    args = ap.parse_args()

    overall_ok = True
    for hdir in sorted(glob.glob(os.path.join(args.dir, 'H*'))):
        horizon = os.path.basename(hdir)[1:]
        summ = glob.glob(os.path.join(hdir, '*_FULL.csv'))
        perq = glob.glob(os.path.join(hdir, '*_FULL_per_query.csv'))
        if not summ or not perq:
            print(f'[{horizon}] missing artifacts (summary={bool(summ)} per_query={bool(perq)})')
            overall_ok = False
            continue
        summary = {r['arm']: r for r in load(summ[0])}
        rows = load(perq[0])
        lines = [f'=== H{horizon} ===',
                 f'summary   : {os.path.basename(summ[0])}',
                 f'per_query : {os.path.basename(perq[0])} ({len(rows)} rows)']

        by_arm = {}
        for r in rows:
            by_arm.setdefault(r['arm'], []).append(r)

        for arm in ('R1', 'R2-W'):
            got = by_arm.get(arm, [])
            if not got:
                lines.append(f'{arm}: NO ROWS'); overall_ok = False; continue

            bad_k = [r for r in got if int(r['n_unique']) != args.k]
            bad_a = [r for r in got if abs(float(r['alpha_sum']) - 1.0) > 1e-5]
            worst_u = max(abs(float(r['I']) - float(r['A_uniform']) - float(r['V_uniform']))
                          for r in got)
            worst_w = max(abs(float(r['I_weighted']) - float(r['A_weighted'])
                              - float(r['V_weighted'])) for r in got)
            mean = lambda f: sum(float(r[f]) for r in got) / len(got)

            lines.append(f'-- {arm}  n={len(got)}')
            lines.append(f'   K distinct == {args.k}          : '
                         f'{"OK" if not bad_k else f"FAIL ({len(bad_k)} rows)"}')
            lines.append(f'   sum(alpha) == 1                : '
                         f'{"OK" if not bad_a else f"FAIL ({len(bad_a)} rows)"}')
            lines.append(f'   I = A_u + V_u   worst residual : {worst_u:.3e} '
                         f'{"OK" if worst_u < TOL else "FAIL"}')
            lines.append(f'   I_w = A_w + V_w worst residual : {worst_w:.3e} '
                         f'{"OK" if worst_w < TOL else "FAIL"}')
            if bad_k or bad_a or worst_u >= TOL or worst_w >= TOL:
                overall_ok = False

            for field in ('I', 'A_uniform', 'A_weighted'):
                stored = float(summary[arm][field])
                manual = mean(field)
                diff = abs(stored - manual)
                flag = 'OK' if diff < TOL else 'FAIL'
                if diff >= TOL:
                    overall_ok = False
                lines.append(f'   {field:<12} stored={stored:.8f} '
                             f'manual={manual:.8f} diff={diff:.3e} {flag}')

            lines.append(f'   first {args.samples} queries (query_start, ch, I, A_u, A_w):')
            for r in got[:args.samples]:
                lines.append(f'     q={r["query_start"]:>6} c={r["target_channel"]} '
                             f'I={float(r["I"]):.6f} A_u={float(r["A_uniform"]):.6f} '
                             f'A_w={float(r["A_weighted"]):.6f}')

        # The comparison the table is built on.
        if 'R1' in summary and 'R2-W' in summary:
            ai, as_ = float(summary['R1']['A_weighted']), float(summary['R2-W']['A_weighted'])
            ii, is_ = float(summary['R1']['I']), float(summary['R2-W']['I'])
            lines.append(f'-- TABLE 2 row: A_w ind={ai:.6f} set={as_:.6f} '
                         f'improvement={100*(ai-as_)/ai:+.2f}%')
            lines.append(f'   I_set > I_ind ? {is_:.6f} > {ii:.6f} -> {is_ > ii}')

        text = '\n'.join(lines)
        print(text + '\n')
        with open(os.path.join(hdir, 'sanity_check.txt'), 'w') as fh:
            fh.write(text + '\n')

    print('ALL CHECKS PASSED' if overall_ok else 'SOME CHECKS FAILED')
    return 0 if overall_ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
