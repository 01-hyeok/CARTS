#!/usr/bin/env python3
"""Report for the observability diagnostics.

The question is not "did the reranker fail" -- it did. The question is whether
it failed because the representation cannot read a signal that is there (H1) or
because the target past does not contain what the selector needs (H2). The
feature ladder answers it: each rung adds information, so where the curve jumps
is where the missing information was.
"""

import argparse
import csv
import os
import statistics as st
from pathlib import Path

LADDER_ORDER = ['original', 'A_past', 'B_cand_residual', 'C_pred_query_residual',
                'D_true_query_residual', 'E_query_future', 'oracle']
LADDER_LABEL = {
    'original': 'Original retriever (no rerank)',
    'A_past': 'A. Past only',
    'B_cand_residual': 'B. + candidate residual',
    'C_pred_query_residual': 'C. + predicted query residual',
    'D_true_query_residual': 'D. + TRUE query residual',
    'E_query_future': 'E. + query future',
    'oracle': 'Oracle rerank (upper bound)',
}
DEPLOYABLE = {'original': True, 'A_past': True, 'B_cand_residual': True,
              'C_pred_query_residual': True, 'D_true_query_residual': False,
              'E_query_future': False, 'oracle': False}


def read(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def f(row, key):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return float('nan')


def fmt(value, digits=4):
    return '--' if value != value else f'{value:.{digits}f}'


def ladder_section(root):
    ladder = read(root / 'feature_ladder.csv')
    forecast = read(root / 'feature_ladder_forecast.csv')
    if not ladder:
        return [], {}

    rank_by = {(r['dataset'], int(r['pool_m']), r['arm']): r for r in ladder}
    fore_by = {(r['dataset'], int(r['pool_m']), r['arm']): r for r in forecast}
    settings = sorted({(r['dataset'], int(r['pool_m'])) for r in ladder})

    lines = ['### Experiment 1 — Oracle feature ladder', '',
             '`deployable=no`인 D/E는 query future 또는 true residual을 읽으므로 '
             '배포 불가능한 진단 전용이다.', '',
             '| Dataset | M | Arm | deployable | Utility Spearman | NDCG@10 | '
             'Gap Recovery@10 | Stage-2 MSE |', '|---|---|---|---|---|---|---|---|']
    for dataset, pool in settings:
        for arm in LADDER_ORDER:
            row = rank_by.get((dataset, pool, arm))
            if not row:
                continue
            fore = fore_by.get((dataset, pool, arm), {})
            lines.append(
                f"| {dataset} | {pool} | {LADDER_LABEL[arm]} | "
                f"{'yes' if DEPLOYABLE[arm] else '**no**'} | "
                f"{fmt(f(row,'spearman'))} | {fmt(f(row,'ndcg_at_10'))} | "
                f"**{fmt(f(row,'gap_recovery_at_10'))}** | {fmt(f(fore,'forecast_mse'))} |")

    # The jumps: each rung against the rung below it, in gap recovery.
    jumps = {}
    for dataset, pool in settings:
        chain = [arm for arm in ('A_past', 'B_cand_residual', 'C_pred_query_residual',
                                 'D_true_query_residual')
                 if (dataset, pool, arm) in rank_by]
        for lower, upper in zip(chain, chain[1:]):
            delta = (f(rank_by[(dataset, pool, upper)], 'gap_recovery_at_10')
                     - f(rank_by[(dataset, pool, lower)], 'gap_recovery_at_10'))
            jumps.setdefault(f'{lower} -> {upper}', []).append(delta)
    if jumps:
        lines += ['', '#### Rung-to-rung change in gap recovery (mean over settings)', '',
                  '| Step | mean Δ gap recovery |', '|---|---|']
        for step, values in jumps.items():
            lines.append(f'| {step} | {st.mean(values):+.4f} |')
    return lines, {'rank_by': rank_by, 'fore_by': fore_by, 'settings': settings,
                   'jumps': jumps}


def stability_section(root):
    rows = read(root / 'similar_past_summary.csv')
    if not rows:
        return [], {}
    lines = ['', '### Experiment 2 — Similar-past oracle instability', '',
             '같은 fixed candidate pool 위에서 두 query의 utility 순위를 비교한다. '
             'past가 가까울수록 순위가 안정되어야 정보가 past 안에 있다는 뜻이다.', '',
             '| Dataset | Metric | Bin | pairs | Utility Spearman | Top-1 match | '
             'Top-10 overlap | Residual MSE |', '|---|---|---|---|---|---|---|---|']
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['similarity']} | {row['bin']} | {row['pairs']} | "
            f"{fmt(f(row,'utility_spearman'))} | {fmt(f(row,'top1_match'),3)} | "
            f"{fmt(f(row,'top10_overlap'),3)} | {fmt(f(row,'residual_mse'))} |")
    return lines, {'rows': rows}


def probe_section(root):
    probe = read(root / 'feature_probe.csv')
    perm = read(root / 'permutation_control.csv')
    if not probe:
        return [], {}
    lines = ['', '### Experiment 3/4 — Observable feature probe and permutation controls', '',
             '| Dataset | M | Feature group | Model | Label | AUROC | PR-AUC | '
             'Precision@10 | prevalence | shuffled AUROC |', '|---|---|---|---|---|---|---|---|---|---|']
    shuffled = {(r['dataset'], int(r['pool_m']), r['feature_group'], r['shuffle']): r
                for r in perm}
    for row in probe:
        control = shuffled.get(
            (row['dataset'], int(row['pool_m']), row['feature_group'], 'pair'))
        note = (f"{fmt(f(control,'auroc'),3)} ± {fmt(f(control,'auroc_std'),3)}"
                if control else '--')
        lines.append(
            f"| {row['dataset']} | {row['pool_m']} | {row['feature_group']} | "
            f"{row['model']} | {row['label']} | {fmt(f(row,'auroc'),3)} | "
            f"{fmt(f(row,'pr_auc'),3)} | {fmt(f(row,'precision_at_10'),3)} | "
            f"{fmt(f(row,'prevalence'),3)} | {note} |")
    return lines, {'probe': probe, 'perm': perm}


def verdict(ladder_state, stability_state):
    """The spec's final decision matrix, read off the ladder first."""
    if not ladder_state:
        return 'INCOMPLETE', 'feature ladder rows are missing.'
    rank_by, settings = ladder_state['rank_by'], ladder_state['settings']

    def gap(arm):
        values = [f(rank_by[(d, m, arm)], 'gap_recovery_at_10')
                  for d, m in settings if (d, m, arm) in rank_by]
        return st.mean(values) if values else float('nan')

    past, candidate = gap('A_past'), gap('B_cand_residual')
    predicted, true = gap('C_pred_query_residual'), gap('D_true_query_residual')
    baseline = gap('original')

    lift = lambda value: value - max(past, baseline)
    true_lift, predicted_lift, candidate_lift = lift(true), lift(predicted), lift(candidate)

    if true_lift > 0.15 and predicted_lift < 0.5 * true_lift:
        name = 'QUERY_PAST_OBSERVABILITY_LIMIT'
        text = ('True query residual을 주면 selection이 크게 개선되지만, 같은 정보를 '
                'query past로부터 예측해서 주면 그 이득이 거의 따라오지 않는다. 좋은 '
                'historical candidate는 shortlist에 존재하나, 그 정체는 target past로부터 '
                '충분히 관측되지 않는 future correction에 달려 있다. encoder를 더 키우는 '
                '방향은 우선 중단한다.')
    elif predicted_lift > 0.5 * true_lift and true_lift > 0.05:
        name = 'REPRESENTATION_BOTTLENECK'
        text = ('예측된 query residual만으로도 상당 부분을 회수한다. 필요한 정보는 past에 '
                '있으나 기존 Stage-1 embedding이 그것을 표현하지 못한다. 다음은 '
                'residual-aware query representation이다.')
    elif candidate_lift > 0.15 and true_lift <= candidate_lift:
        name = 'CANDIDATE_VALUE_BOTTLENECK'
        text = ('candidate historical correction 정보만으로 큰 개선이 나온다. '
                'value-aware retrieval/reranking이 다음 방향이다.')
    elif true_lift <= 0.05:
        name = 'PAIRWISE_UTILITY_FORMULATION_INSUFFICIENT'
        text = ('true residual까지 주어도 oracle headroom을 회수하지 못한다. '
                'candidate 단위 pairwise reranking 형식 자체가 부족하며, set-level '
                'selection이나 conditional distribution modeling을 검토해야 한다.')
    else:
        name = 'MIXED'
        text = 'ladder가 단일 case로 떨어지지 않는다. 위 표가 결과다.'

    detail = [
        '', f'- Original retriever gap recovery: **{fmt(baseline)}**',
        f'- A past only: **{fmt(past)}**',
        f'- B + candidate residual: **{fmt(candidate)}** (Δ {candidate_lift:+.4f})',
        f'- C + predicted query residual: **{fmt(predicted)}** (Δ {predicted_lift:+.4f})',
        f'- D + TRUE query residual: **{fmt(true)}** (Δ {true_lift:+.4f}, 배포 불가)',
    ]
    return name, text + '\n' + '\n'.join(detail)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='./metrics/utility_observability')
    a = p.parse_args()
    root = Path(a.root)

    ladder_lines, ladder_state = ladder_section(root)
    stability_lines, stability_state = stability_section(root)
    probe_lines, _ = probe_section(root)
    name, text = verdict(ladder_state, stability_state)

    lines = [
        '# Utility Observability — 왜 learned reranker는 oracle headroom을 회수하지 못하는가',
        '',
        '질문: **representation이 부족해서인가 (H1), 아니면 target past 자체가 '
        'high-utility candidate identity를 결정할 정보를 담고 있지 않아서인가 (H2)?**',
        '',
        '모든 utility는 production Stage-2 helper로 측정했고, 모든 arm은 동일한 frozen '
        'shortlist·backbone·loss를 쓴다. 따라서 rung 사이의 차이는 정보의 차이다.',
        '',
    ] + ladder_lines + stability_lines + probe_lines + [
        '', '## 최종 판정', '', f'### {name}', '', text, '',
    ]
    report = '\n'.join(lines) + '\n'
    root.mkdir(parents=True, exist_ok=True)
    (root / 'FINAL_REPORT.md').write_text(report)
    print(report)
    print(f'DECISION: {name}')


if __name__ == '__main__':
    main()
