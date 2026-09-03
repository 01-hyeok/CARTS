"""Split the Current -> Oracle gap into support, ranking and set-composition.

EXP-1 showed the current retriever is far from any oracle. That gap is not one
thing, and the three parts call for different fixes, so they are separated by
holding one factor at a time:

    R0(P100)  --ranking-->  R1(P100)  --set-composition-->  R2-W(P100)
        |                       |
        |                       +--support-->  R1(FULL)
        +----------------------- total ------> R2-W(FULL)

  ranking          same support, best-ranked ten vs the ten the retriever picked
  set-composition  same support, best ten together vs best ten individually
  support          same rule, everything in memory vs what cosine offered

Reads the CSVs the intervention writes; runs no model.
"""

import argparse
import csv
import os

ARMS = ('R0', 'R1', 'R2-U', 'R2-W', 'R3')
LABEL = {'R0': 'Current', 'R1': 'Individual Oracle', 'R2-U': 'Uniform Set Oracle',
         'R2-W': 'Weighted Set Oracle', 'R3': 'Good+Diverse'}


def load(directory, data, horizon, support):
    path = os.path.join(directory, f'{data}_H{horizon}_{support}.csv')
    if not os.path.isfile(path):
        return None
    with open(path) as fh:
        return {row['arm']: row for row in csv.DictReader(fh)}


def f(cell, arm, field):
    if cell is None or arm not in cell:
        return float('nan')
    return float(cell[arm][field])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='logs/oracle_intervention')
    ap.add_argument('--out', default='logs/oracle_intervention/decomposition.csv')
    ap.add_argument('--cells', default='ETTh1:96,ETTh1:720,custom:96,custom:720')
    args = ap.parse_args()

    rows = []
    for spec in args.cells.split(','):
        data, horizon = spec.split(':')
        p100 = load(args.dir, data, horizon, 'P100')
        full = load(args.dir, data, horizon, 'FULL')
        if p100 is None:
            print(f'[skip] {spec}: no P100 result')
            continue

        base = f(p100, 'R0', 'base_mse')
        cur = f(p100, 'R0', 'stage2_mse')
        ind = f(p100, 'R1', 'stage2_mse')
        setw = f(p100, 'R2-W', 'stage2_mse')
        ind_full = f(full, 'R1', 'stage2_mse')
        setw_full = f(full, 'R2-W', 'stage2_mse')

        ranking = cur - ind                 # closed by ranking better inside the support
        composition = ind - setw            # closed by choosing a better set
        support = ind - ind_full            # closed by widening the support
        total = cur - setw_full

        rows.append({
            'dataset': data, 'horizon': horizon,
            'base_mse': base, 'current_mse': cur,
            'ind_p100': ind, 'setw_p100': setw,
            'ind_full': ind_full, 'setw_full': setw_full,
            'gap_ranking': ranking,
            'gap_set_composition': composition,
            'gap_support': support,
            'gap_total': total,
            'pct_ranking': 100.0 * ranking / total if total else float('nan'),
            'pct_set_composition': 100.0 * composition / total if total else float('nan'),
            'pct_support': 100.0 * support / total if total else float('nan'),
            'A_w_R0': f(p100, 'R0', 'A_weighted'),
            'A_w_R1': f(p100, 'R1', 'A_weighted'),
            'A_w_R2W': f(p100, 'R2-W', 'A_weighted'),
            'A_w_R1_full': f(full, 'R1', 'A_weighted'),
            'A_w_R2W_full': f(full, 'R2-W', 'A_weighted'),
        })

    if not rows:
        print('no cells to report')
        return

    hdr = (f"{'cell':<14}{'base':>9}{'current':>9}{'Ind@100':>9}{'Set@100':>9}"
           f"{'Ind@FULL':>10}{'Set@FULL':>10}")
    print('\n=== Final MSE by selection and support ===')
    print(hdr)
    for r in rows:
        print(f"{r['dataset']+' H'+r['horizon']:<14}{r['base_mse']:>9.5f}"
              f"{r['current_mse']:>9.5f}{r['ind_p100']:>9.5f}{r['setw_p100']:>9.5f}"
              f"{r['ind_full']:>10.5f}{r['setw_full']:>10.5f}")

    print('\n=== Current -> Oracle gap decomposition (Final MSE) ===')
    print(f"{'cell':<14}{'total':>9}{'ranking':>10}{'set-comp':>10}{'support':>10}"
          f"{'rank%':>8}{'set%':>8}{'supp%':>8}")
    for r in rows:
        print(f"{r['dataset']+' H'+r['horizon']:<14}{r['gap_total']:>9.5f}"
              f"{r['gap_ranking']:>10.5f}{r['gap_set_composition']:>10.5f}"
              f"{r['gap_support']:>10.5f}{r['pct_ranking']:>8.1f}"
              f"{r['pct_set_composition']:>8.1f}{r['pct_support']:>8.1f}")
    print('\nPercentages are shares of the total Current -> Set@FULL gap and need '
          'not sum to 100: the three factors are measured by holding the others '
          'fixed, and they interact.')

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()
