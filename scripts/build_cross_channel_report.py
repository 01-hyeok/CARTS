#!/usr/bin/env python3
"""Tables and verdict for the cross-channel retrieval ablation.

The decision rule is fixed before the numbers are read (spec section 22). What
settles it is not whether cross-channel beats target-only, but whether
cross-channel *selection* beats cross-channel *direct prediction* -- both arms
having been handed the same source channels.
"""

import argparse
import csv
import os
from pathlib import Path

ARM_LABEL = {
    'target_only_resdirect': 'A. Target-only ResDirect',
    'cross_channel_resdirect': 'B. Cross-channel ResDirect',
    'target_only_ressel': 'C. Target-only ResSel',
    'query_cross_channel_ressel': 'D. Query cross-channel ResSel',
    'query_candidate_cross_channel_ressel': 'E. Query+Candidate cross-channel ResSel',
    'query_cross_channel_residual_aware_ressel': 'F. Cross-channel + candidate residual ResSel',
}
ORDER = list(ARM_LABEL)


def read(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def test_rows(rows):
    return {(r['dataset'], int(r['pred_len']), r['arm']): r
            for r in rows if r['split'] == 'test'}


def settings_of(table):
    return sorted({(d, p) for d, p, _ in table}, key=lambda k: (k[0], k[1]))


def fmt(value, digits=4):
    try:
        return f'{float(value):.{digits}f}'
    except (TypeError, ValueError):
        return '--'


def table_a(table, settings):
    lines = ['### Table A — Forecast', '',
             '| Dataset | Pred | Method | Forecast MSE | Forecast MAE | Base MSE |',
             '|---|---|---|---|---|---|']
    for dataset, pred in settings:
        for arm in ORDER:
            row = table.get((dataset, pred, arm))
            if not row:
                continue
            lines.append(
                f"| {dataset} | {pred} | {ARM_LABEL[arm]} | "
                f"{fmt(row['forecast_mse'])} | {fmt(row['forecast_mae'])} | "
                f"{fmt(row['base_mse'])} |"
            )
    return '\n'.join(lines)


def table_b(table, settings):
    lines = ['### Table B — Selection', '',
             '| Dataset | Pred | Method | Positive@1 | Selected Utility | '
             'Selection Regret | Selection Recovery |', '|---|---|---|---|---|---|---|']
    for dataset, pred in settings:
        for arm in ORDER:
            row = table.get((dataset, pred, arm))
            if not row or 'positive_at_1' not in row:
                continue
            lines.append(
                f"| {dataset} | {pred} | {ARM_LABEL[arm]} | "
                f"{fmt(row['positive_at_1'], 3)} | {fmt(row['selected_utility_at_1'])} | "
                f"{fmt(row['utility_regret_at_1'])} | "
                f"{fmt(row['selection_recovery_at_1'], 3)} |"
            )
    return '\n'.join(lines)


def table_c(table, settings):
    lines = ['### Table C — Residual Prediction', '',
             '| Dataset | Pred | Target-only residual MSE | Cross-channel residual MSE | '
             'Corr (target-only → cross) | Gamma |', '|---|---|---|---|---|---|']
    for dataset, pred in settings:
        a = table.get((dataset, pred, 'target_only_resdirect'))
        b = table.get((dataset, pred, 'cross_channel_resdirect'))
        if not (a and b):
            continue
        lines.append(
            f"| {dataset} | {pred} | {fmt(a['residual_pred_mse'])} | "
            f"{fmt(b['residual_pred_mse'])} | "
            f"{fmt(a['residual_pred_corr'], 3)} → {fmt(b['residual_pred_corr'], 3)} | "
            f"{fmt(b.get('gamma'), 4)} |"
        )
    return '\n'.join(lines)


def table_params(table, settings):
    lines = ['### Parameter fairness', '',
             '| Dataset | Pred | Arm | Total params | Trainable |',
             '|---|---|---|---|---|']
    for dataset, pred in settings:
        for arm in ORDER:
            row = table.get((dataset, pred, arm))
            if not row:
                continue
            lines.append(
                f"| {dataset} | {pred} | {ARM_LABEL[arm]} | "
                f"{row.get('total_params', '--')} | {row.get('trainable_params', '--')} |"
            )
    return '\n'.join(lines)


def verdict(table, settings):
    """Section 22, applied literally. Lower MSE is better throughout."""
    counts = {'B<A': 0, 'D<C': 0, 'D<B': 0, 'sel_up': 0, 'n': 0}
    detail = []
    for dataset, pred in settings:
        get = lambda arm: table.get((dataset, pred, arm))
        a, b, c, d = (get('target_only_resdirect'), get('cross_channel_resdirect'),
                      get('target_only_ressel'), get('query_cross_channel_ressel'))
        if not all((a, b, c, d)):
            continue
        counts['n'] += 1
        mse = lambda r: float(r['forecast_mse'])
        b_lt_a = mse(b) < mse(a)
        d_lt_c = mse(d) < mse(c)
        d_lt_b = mse(d) < mse(b)
        selection_up = (
            float(d['positive_at_1']) > float(c['positive_at_1'])
            and float(d['selected_utility_at_1']) > float(c['selected_utility_at_1'])
            and float(d['utility_regret_at_1']) < float(c['utility_regret_at_1'])
        )
        counts['B<A'] += b_lt_a
        counts['D<C'] += d_lt_c
        counts['D<B'] += d_lt_b
        counts['sel_up'] += selection_up
        detail.append({
            'dataset': dataset, 'pred': pred,
            'A': mse(a), 'B': mse(b), 'C': mse(c), 'D': mse(d),
            'B<A': b_lt_a, 'D<C': d_lt_c, 'D<B': d_lt_b, 'sel_up': selection_up,
        })

    n = counts['n']
    if not n:
        return 'INCOMPLETE', 'Phase 1 has no setting with all four arms present.', counts, detail

    all_of = lambda key: counts[key] == n
    if all_of('D<C') and all_of('D<B') and all_of('sel_up'):
        name = 'CROSS_CHANNEL_RETRIEVAL_USEFUL'
        text = (
            'Case A — Cross-channel Retrieval Revival. Source channels supply the '
            'future-regime information the target past lacked, and that information '
            'is worth more inside historical selection than inside direct residual '
            'prediction. The cross-channel retrieval contribution is worth developing.'
        )
    elif all_of('B<A') and all_of('D<C') and counts['D<B'] == 0:
        name = 'CROSS_CHANNEL_USEFUL_RETRIEVAL_REDUNDANT'
        text = (
            'Case B — Cross-channel useful, retrieval redundant. Source channels help '
            'both arms, but once both have them, predicting the correction beats '
            'retrieving one. The direction is cross-channel residual forecasting, not '
            'retrieval.'
        )
    elif counts['D<C'] > 0 and counts['D<B'] == 0:
        name = 'SELECTION_SIGNAL_IMPROVED_BUT_RETRIEVAL_NOT_NEEDED'
        text = (
            'Case C — Cross-channel improves selection but not enough. Selection gets '
            'better with source context, yet never overtakes direct prediction given '
            'the same context. Retrieval stays unnecessary.'
        )
    elif counts['B<A'] == 0 and counts['D<C'] == 0:
        name = 'CROSS_CHANNEL_SIGNAL_WEAK'
        text = (
            'Case D — Cross-channel ineffective. Source context moves neither arm, so '
            'it is not the missing ingredient for the target-only problem.'
        )
    else:
        name = 'MIXED'
        text = (
            'No case matches cleanly: the arms disagree across settings. The '
            'per-setting table below is the result; a single label would hide it.'
        )
    return name, text, counts, detail


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='./metrics/cross_channel_retrieval')
    a = p.parse_args()
    root = Path(a.root)

    direct = read(root / 'resdirect.csv')
    selection = read(root / 'ressel.csv')
    table = {**test_rows(direct), **test_rows(selection)}
    settings = settings_of(table)

    phase1 = [(d, p) for d, p in settings]
    name, text, counts, detail = verdict(table, phase1)

    lines = [
        '# Cross-Channel Context for Utility-aware Historical Correction Selection',
        '',
        '핵심 질문: **target past만으로 식별하기 어려웠던 historical correction utility를 '
        'related source-channel context가 disambiguate하는가, 그리고 그 정보가 direct '
        'residual forecasting보다 historical retrieval selection을 더 유용하게 만드는가?**',
        '',
        '결정적인 비교는 D vs B다. 두 arm 모두 동일한 source channel 정보를 받은 뒤, '
        'selection이 direct prediction을 이기는지만이 retrieval viability를 말해준다.',
        '',
        table_a(table, settings), '',
        table_b(table, settings), '',
        table_c(table, settings), '',
        table_params(table, settings), '',
        '### Per-setting decisions', '',
        '| Dataset | Pred | A | B | C | D | B<A | D<C | D<B | selection ↑ |',
        '|---|---|---|---|---|---|---|---|---|---|',
    ]
    mark = lambda flag: 'YES' if flag else 'no'
    for row in detail:
        lines.append(
            f"| {row['dataset']} | {row['pred']} | {row['A']:.4f} | {row['B']:.4f} | "
            f"{row['C']:.4f} | {row['D']:.4f} | {mark(row['B<A'])} | "
            f"{mark(row['D<C'])} | {mark(row['D<B'])} | {mark(row['sel_up'])} |"
        )

    n = counts['n']
    lines += [
        '', '## 판정', '', f'### {name}', '', text, '',
        f"- B < A (cross-channel helps direct): **{counts['B<A']}/{n}**",
        f"- D < C (cross-channel helps selection): **{counts['D<C']}/{n}**",
        f"- **D < B (selection beats direct, both with context): {counts['D<B']}/{n}**",
        f"- selection metrics improved together (pos@1 ↑, utility ↑, regret ↓): "
        f"**{counts['sel_up']}/{n}**",
        '',
        '## 최종 연구 질문에 대한 답', '',
        '> Cross-channel context가 historical correction selection을 실제로 살릴 수 있는가?',
        '',
        f"### **{'YES' if counts['D<B'] == n and n else 'NO'}**",
        '',
    ]
    if counts['D<B'] != n:
        lines.append(
            'Source channel이 유용하더라도 D < B가 일관되게 관측되지 않으면 retrieval '
            'contribution으로 해석하지 않는다 (spec 최종 조건).'
        )

    report = '\n'.join(lines) + '\n'
    root.mkdir(parents=True, exist_ok=True)
    (root / 'FINAL_REPORT.md').write_text(report)

    # Section 27 asks for these split out by phase and by table.
    def dump(name_, rows_, columns):
        if not rows_:
            return
        with open(root / name_, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(rows_)

    phase2_arms = {'query_candidate_cross_channel_ressel',
                   'query_cross_channel_residual_aware_ressel'}
    forecast_cols = ['dataset', 'pred_len', 'arm', 'split', 'forecast_mse',
                     'forecast_mae', 'base_mse', 'total_params', 'trainable_params']
    selection_cols = ['dataset', 'pred_len', 'arm', 'split', 'positive_at_1',
                      'selected_utility_at_1', 'utility_regret_at_1',
                      'selection_recovery_at_1', 'oracle_pool_utility',
                      'random_utility', 'top1_identity_accuracy']
    all_rows = direct + selection
    dump('phase1_forecast.csv',
         [r for r in all_rows if r['arm'] not in phase2_arms], forecast_cols)
    dump('phase2_forecast.csv',
         [r for r in all_rows if r['arm'] in phase2_arms], forecast_cols)
    dump('phase1_selection.csv',
         [r for r in selection if r['arm'] not in phase2_arms], selection_cols)
    dump('phase2_selection.csv',
         [r for r in selection if r['arm'] in phase2_arms], selection_cols)
    dump('residual_prediction.csv', direct,
         ['dataset', 'pred_len', 'arm', 'split', 'residual_pred_mse',
          'residual_pred_corr', 'residual_pred_cosine', 'residual_norm_error',
          'gamma'])

    print(report)
    print(f'DECISION: {name}')


if __name__ == '__main__':
    main()
