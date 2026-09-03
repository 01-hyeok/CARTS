"""Build TABLE (per dataset/horizon) and the summary table for the
soft_set_mse experiment from the raw Stage-1 / Stage-2 logs. Reads artifacts
only; runs no model.
"""
import argparse
import glob
import os
import re

ARMS = ['S0_wce', 'S1_set_only', 'S2_lam10', 'S3_lam30', 'S4_lam50']
LABEL = {'S0_wce': 'WCE', 'S1_set_only': 'SetMSE only', 'S2_lam10': 'WCE + Set λ10',
         'S3_lam30': 'WCE + Set λ30', 'S4_lam50': 'WCE + Set λ50'}


def last_metrics_line(path, prefix):
    """Parse the last `prefix key: value | key: value | ...` line in a log."""
    if not os.path.isfile(path):
        return None
    best = None
    with open(path, errors='ignore') as fh:
        for line in fh:
            if line.startswith(prefix):
                best = line
    if best is None:
        return None
    out = {}
    for part in best.split('|'):
        part = part.strip()
        if ':' not in part:
            continue
        key, _, val = part.partition(':')
        key, val = key.strip(), val.strip()
        try:
            out[key] = float(val)
        except ValueError:
            pass
    return out


def stage1_test_metrics(log_root, ds, pred, arm):
    return last_metrics_line(
        os.path.join(log_root, ds, f'pred{pred}', f'{arm}.log'), 'Stage1 Test')


def stage2_test_metrics(log_root, ds, pred, arm):
    return last_metrics_line(
        os.path.join(log_root, ds, f'pred{pred}', f'{arm}_stage2.log'), 'Stage2 Test')


def g(d, *keys, default=float('nan')):
    for k in keys:
        if d and k in d:
            return d[k]
    return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log-root', default='logs/soft_set_mse')
    ap.add_argument('--out', default='logs/soft_set_mse/summary.csv')
    ap.add_argument('--cells', default='ETTh1:96,ETTh1:720,Weather:96,Weather:720')
    args = ap.parse_args()

    import csv
    all_rows = []
    summary_rows = []

    for spec in args.cells.split(','):
        ds, pred = spec.split(':')
        print(f'\n=== {ds} H{pred} ===')
        hdr = (f"{'Arm':<14}{'Recall@10':>10}{'RetMSE@10':>11}{'HardAgg@10':>12}"
               f"{'SoftSetMSE':>11}{'N_eff':>9}{'S2 Final':>10}{'RetGain':>9}")
        print(hdr)
        cell_rows = []
        for arm in ARMS:
            s1 = stage1_test_metrics(args.log_root, ds, pred, arm)
            s2 = stage2_test_metrics(args.log_root, ds, pred, arm)
            row = {
                'dataset': ds, 'horizon': pred, 'arm': arm, 'label': LABEL[arm],
                'recall10': g(s1, 'student_oracle_recall_at_10', 'oracle_recall_at_10'),
                'retrieved_mse10': g(s1, 'student_retrieved_future_mse_at_10',
                                     'retrieved_future_mse_at_10'),
                'hard_aggregate_mse10': g(s1, 'hard_aggregate_mse10'),
                'uniform_aggregate_mse10': g(s1, 'uniform_aggregate_mse10'),
                'weighted_individual_mse10': g(s1, 'weighted_individual_mse10'),
                'weighted_candidate_variance10': g(s1, 'weighted_candidate_variance10'),
                'retrieval_regret10': g(s1, 'student_retrieval_regret_at_10',
                                        'retrieval_regret_at_10'),
                'set_soft_mse_raw': g(s1, 'set_soft_mse_raw'),
                'set_soft_mse_normalized': g(s1, 'set_soft_mse_normalized'),
                'set_soft_entropy': g(s1, 'set_soft_entropy'),
                'set_soft_entropy_norm': g(s1, 'set_soft_entropy_norm'),
                'n_eff': g(s1, 'set_soft_effective_candidates'),
                'set_soft_top10_mass': g(s1, 'set_soft_top10_mass'),
                'set_soft_top100_mass': g(s1, 'set_soft_top100_mass'),
                'online_collapse_effective_rank_mean': g(
                    s1, 'online_collapse_effective_rank_mean'),
                'base_mse': g(s2, 'base_mse'),
                'ret_mse': g(s2, 'ret_mse'),
                'final_mse': g(s2, 'final_mse'),
                'final_mae': g(s2, 'final_mae'),
            }
            base, final = row['base_mse'], row['final_mse']
            row['retrieval_gain'] = base - final if base == base and final == final else float('nan')
            row['retrieval_gain_pct'] = (
                100.0 * row['retrieval_gain'] / base
                if base == base and base != 0 else float('nan'))
            cell_rows.append(row)
            all_rows.append(row)
            print(f"{LABEL[arm]:<14}{row['recall10']:>10.4f}{row['retrieved_mse10']:>11.4f}"
                  f"{row['hard_aggregate_mse10']:>12.4f}{row['set_soft_mse_normalized']:>11.4f}"
                  f"{row['n_eff']:>9.1f}{row['final_mse']:>10.4f}{row['retrieval_gain']:>+9.4f}")

        wce = next((r for r in cell_rows if r['arm'] == 'S0_wce'), None)
        best_set = min(
            (r for r in cell_rows if r['arm'] != 'S0_wce'
             and r['final_mse'] == r['final_mse']),
            key=lambda r: r['final_mse'], default=None)
        if wce and best_set:
            summary_rows.append({
                'dataset': ds, 'horizon': pred,
                'best_arm': best_set['label'],
                'wce_final_mse': wce['final_mse'],
                'best_set_final_mse': best_set['final_mse'],
                'delta_final_mse': best_set['final_mse'] - wce['final_mse'],
                'delta_final_mse_pct': (
                    100.0 * (best_set['final_mse'] - wce['final_mse']) / wce['final_mse']
                    if wce['final_mse'] == wce['final_mse'] and wce['final_mse'] != 0
                    else float('nan')),
                'recall_change': best_set['recall10'] - wce['recall10'],
                'hard_agg_change': best_set['hard_aggregate_mse10'] - wce['hard_aggregate_mse10'],
            })

    print('\n=== SUMMARY (best WCE+Set arm vs WCE baseline, by Stage-2 final MSE) ===')
    print(f"{'Dataset':<10}{'H':>5}{'Best arm':<16}{'WCE MSE':>10}{'Best MSE':>10}"
          f"{'ΔMSE':>9}{'Δ%':>8}{'ΔRecall':>9}{'ΔHardAgg':>10}")
    for r in summary_rows:
        print(f"{r['dataset']:<10}{r['horizon']:>5}{r['best_arm']:<16}"
              f"{r['wce_final_mse']:>10.4f}{r['best_set_final_mse']:>10.4f}"
              f"{r['delta_final_mse']:>+9.4f}{r['delta_final_mse_pct']:>+8.2f}"
              f"{r['recall_change']:>+9.4f}{r['hard_agg_change']:>+10.4f}")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0]) if all_rows else [])
        w.writeheader(); w.writerows(all_rows)
    with open(args.out.replace('.csv', '_summary.csv'), 'w', newline='') as fh:
        if summary_rows:
            w = csv.DictWriter(fh, fieldnames=list(summary_rows[0]))
            w.writeheader(); w.writerows(summary_rows)
    print(f'\nwrote {args.out} and its _summary.csv companion')


if __name__ == '__main__':
    main()
