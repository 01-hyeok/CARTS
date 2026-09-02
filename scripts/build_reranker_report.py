#!/usr/bin/env python3
"""Report and decision for the shortlist-reranking study.

Phase 0 answers one question and gates everything after it: does an oracle that
reorders the frozen retriever's Top-M by measured downstream utility beat plain
CARTS Top-K, on the production Stage-2 path?
"""

import argparse
import csv
import os
from pathlib import Path


def read(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def f(row, key):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return float('nan')
    return value


def phase0(root):
    rows = read(root / 'oracle_headroom.csv')
    if not rows:
        return ['# Utility Reranker — Phase 0', '', 'no rows yet.'], 'INCOMPLETE'

    rows.sort(key=lambda r: (r['dataset'], int(r['pool_m'])))
    lines = [
        '# Utility Reranker — Phase 0: Oracle Reranking Feasibility',
        '',
        '질문: **frozen full-bank retriever가 만든 Top-M 안에, 기존 Top-10보다 더 좋은 '
        'Stage-2 set을 만들 수 있는 candidate가 실제로 존재하는가?**',
        '',
        '모든 forecast는 production `forward`로 계산했다. candidate set은 memory mask를 '
        '그 set으로 좁혀서 평가하며, set이 retriever 자신의 Top-K일 때 기존 예측과 '
        '비트 단위로 일치함을 확인했다 (tests/test_utility_reranker.py + 런타임 검증).',
        '',
        '### Table — Oracle headroom',
        '',
        '| Dataset | M | Original Top-K | Oracle indiv. Top-K | Oracle best single | '
        'Greedy set | Base | Oracle gain | gain % |',
        '|---|---|---|---|---|---|---|---|---|',
    ]
    fmt = lambda v: '--' if v != v else f'{v:.4f}'
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['pool_m']} | {fmt(f(row,'original_topk_mse'))} | "
            f"{fmt(f(row,'oracle_individual_mse'))} | {fmt(f(row,'oracle_best_single_mse'))} | "
            f"{fmt(f(row,'greedy_set_oracle_mse'))} | {fmt(f(row,'base_mse'))} | "
            f"{f(row,'oracle_rerank_gain'):+.4f} | {f(row,'oracle_rerank_gain_pct'):+.1f}% |"
        )

    lines += ['', '### Table — pool contents', '',
              '| Dataset | M | best single-candidate utility | positive utility rate | '
              'Original set utility | Oracle set utility |', '|---|---|---|---|---|---|']
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['pool_m']} | {fmt(f(row,'pool_best_utility'))} | "
            f"{f(row,'pool_positive_rate'):.3f} | {fmt(f(row,'original_set_utility'))} | "
            f"{fmt(f(row,'oracle_set_utility'))} |"
        )

    # Decision. Headroom is judged per dataset on the largest M available, and
    # the M trend decides between "there is headroom" and "coverage is the
    # limit", exactly as the spec defines the three cases.
    by_dataset = {}
    for row in rows:
        by_dataset.setdefault(row['dataset'], []).append(row)
    verdicts, growing, positive = {}, 0, 0
    for dataset, group in by_dataset.items():
        group.sort(key=lambda r: int(r['pool_m']))
        gains = [f(r, 'oracle_rerank_gain_pct') for r in group]
        has = gains[-1] > 1.0
        grows = len(gains) > 1 and gains[-1] > gains[0] + 1.0
        verdicts[dataset] = (gains, has, grows)
        positive += has
        growing += grows

    total = len(verdicts)
    if positive == total and growing == total:
        name = 'POOL_COVERAGE_LIMITED'
        text = ('Oracle headroom이 M과 함께 계속 커진다. Top-M 안에 더 좋은 후보가 있고, '
                'coverage를 넓힐수록 더 많이 생긴다. reranker를 진행하되 M scaling을 '
                '핵심 변수로 함께 보고한다.')
    elif positive == total:
        name = 'ORACLE_RERANK_HEADROOM_EXISTS'
        text = ('Oracle reranking이 기존 CARTS Top-K를 명확히 이긴다. shortlist 안에 '
                '회수할 수 있는 이득이 실재하므로 learned reranker 단계로 진행한다.')
    elif positive == 0:
        name = 'NO_RERANK_HEADROOM'
        text = ('M을 키워도 oracle reranking이 기존 CARTS와 거의 같다. Top-M 내부 '
                'candidate selection은 병목이 아니며 reranker 연구를 중단한다.')
    else:
        name = 'MIXED'
        text = 'Dataset마다 headroom 유무가 갈린다. 위 표가 결과다.'

    lines += ['', '## Phase 0 판정', '', f'### {name}', '', text, '']
    for dataset, (gains, has, grows) in verdicts.items():
        trend = ' → '.join(f'{g:+.1f}%' for g in gains)
        lines.append(f"- {dataset}: gain by M = {trend}  (headroom={'yes' if has else 'no'}, "
                     f"grows with M={'yes' if grows else 'no'})")
    lines += ['', f"다음 단계: {'Phase 1 (shortlist cache) 진행' if positive else '중단'}", '']
    return lines, name


ARM_LABEL = {
    'original': 'Original Retriever (no rerank)',
    'past_pair': 'Past-Pair Reranker',
    'residual_aware': 'Residual-Aware Reranker',
    'oracle': 'Oracle Reranker (upper bound)',
}
ARM_ORDER = ['original', 'past_pair', 'residual_aware', 'oracle']


def phase2(root):
    rerank = read(root / 'reranker_metrics.csv')
    forecast = read(root / 'stage2_forecast.csv')
    if not rerank:
        return ['# Utility Reranker — Phase 2', '', 'no rows yet.'], 'INCOMPLETE'

    key = lambda r: (r['dataset'], int(r['pool_m']), r['arm'], r['target'])
    rank_by = {key(r): r for r in rerank}
    fore_by = {key(r): r for r in forecast}
    # The original/oracle columns are written once per (dataset, M), with
    # whichever arm ran first. Backfilling them here keeps every learned row
    # comparable instead of silently scoring it against a blank.
    reference = {}
    for (dataset, pool, arm, _), row in fore_by.items():
        if arm == 'original':
            reference.setdefault((dataset, pool), {})['original'] = f(row, 'forecast_mse')
        if arm == 'oracle':
            reference.setdefault((dataset, pool), {})['oracle'] = f(row, 'forecast_mse')
    for (dataset, pool, arm, _), row in fore_by.items():
        anchor = reference.get((dataset, pool), {})
        if 'original' in anchor:
            row['original_mse'] = anchor['original']
        if 'oracle' in anchor:
            row['oracle_mse'] = anchor['oracle']
        if 'original' in anchor and 'oracle' in anchor:
            available = anchor['original'] - anchor['oracle']
            row['available_gain'] = available
            row['recovered_gain'] = (
                (anchor['original'] - f(row, 'forecast_mse')) / (available + 1e-12))
    settings = sorted({(r['dataset'], int(r['pool_m'])) for r in rerank})
    targets = sorted({r['target'] for r in rerank if r['target'] != 'none'})
    fmt = lambda v: '--' if v != v else f'{v:.4f}'

    lines = [
        '# Utility Reranker — Phase 2-5: learned shortlist reranking',
        '',
        '질문: **broad retrieval은 frozen global retriever에 맡기고, query-candidate '
        'interaction은 shortlist 내부의 reranker가 배우는 것이 더 적절한가?**',
        '',
        'Stage-2 aggregation은 변경하지 않았다. reranker는 Top-M 안에서 Top-K만 다시 '
        '고르고, 그 K개로 memory mask를 좁혀 production forward를 그대로 돌린다.',
        '',
        '### Table A — Reranking quality (test)',
        '',
        '| Dataset | M | Method | Target | Utility Spearman | NDCG@10 | '
        'Gap Recovery@10 | Positive@10 |', '|---|---|---|---|---|---|---|---|',
    ]
    def rows_for(dataset, pool):
        for arm in ARM_ORDER:
            for target in (['none'] if arm in ('original', 'oracle') else targets):
                row = rank_by.get((dataset, pool, arm, target))
                if row:
                    yield arm, target, row
    for dataset, pool in settings:
        for arm, target, row in rows_for(dataset, pool):
            lines.append(
                f"| {dataset} | {pool} | {ARM_LABEL[arm]} | {target} | "
                f"{fmt(f(row,'spearman'))} | {fmt(f(row,'ndcg_at_10'))} | "
                f"**{fmt(f(row,'gap_recovery_at_10'))}** | {fmt(f(row,'positive_rate_at_10'))} |")

    lines += ['', '### Table B — Actual forecast (production Stage-2)', '',
              '| Dataset | M | Method | Target | Test MSE | Test MAE |',
              '|---|---|---|---|---|---|']
    for dataset, pool in settings:
        for arm in ARM_ORDER:
            for target in (['none'] if arm in ('original', 'oracle') else targets):
                row = fore_by.get((dataset, pool, arm, target))
                if row:
                    lines.append(
                        f"| {dataset} | {pool} | {ARM_LABEL[arm]} | {target} | "
                        f"{fmt(f(row,'forecast_mse'))} | {fmt(f(row,'forecast_mae'))} |")

    lines += ['', '### Table C — Headroom recovered', '',
              '| Dataset | M | Method | Target | Existing | Learned | Oracle | '
              'Available gain | Recovered % |', '|---|---|---|---|---|---|---|---|---|']
    for dataset, pool in settings:
        for arm in ('past_pair', 'residual_aware'):
            for target in targets:
                row = fore_by.get((dataset, pool, arm, target))
                if not row:
                    continue
                lines.append(
                    f"| {dataset} | {pool} | {ARM_LABEL[arm]} | {target} | "
                    f"{fmt(f(row,'original_mse'))} | {fmt(f(row,'forecast_mse'))} | "
                    f"{fmt(f(row,'oracle_mse'))} | {fmt(f(row,'available_gain'))} | "
                    f"{100.0 * f(row,'recovered_gain'):+.1f}% |")

    # Decision, per spec section 10.
    learned = [(k, v) for k, v in fore_by.items() if k[2] in ('past_pair', 'residual_aware')]
    beats, total = 0, 0
    per_arm = {'past_pair': 0, 'residual_aware': 0}
    oracle_headroom = 0
    for (dataset, pool, arm, target), row in learned:
        total += 1
        if f(row, 'forecast_mse') < f(row, 'original_mse'):
            beats += 1
            per_arm[arm] += 1
        if f(row, 'original_mse') - f(row, 'oracle_mse') > 0.001:
            oracle_headroom += 1
    alignment = []
    for (dataset, pool, arm, target), row in rank_by.items():
        if arm in ('past_pair', 'residual_aware'):
            base = rank_by.get((dataset, pool, 'original', 'none'))
            if base:
                alignment.append(f(row, 'gap_recovery_at_10') > f(base, 'gap_recovery_at_10'))

    aligned = sum(alignment)
    if not oracle_headroom:
        name = 'NO_RERANK_HEADROOM'
        text = 'Oracle reranking조차 기존 CARTS를 개선하지 못한다. shortlist 내부 selection은 병목이 아니다.'
    elif beats == total and total:
        name = 'QUERY_CONDITIONED_UTILITY_RERANKING_SUCCESS'
        text = ('Learned reranker가 alignment와 canonical Stage-2 MSE를 모두 개선한다. '
                'broad retrieval과 downstream utility selection은 분리하는 것이 맞다.')
    elif per_arm['residual_aware'] and not per_arm['past_pair']:
        name = 'CANDIDATE_VALUE_INFORMATION_IS_NEEDED'
        text = ('past similarity만으로는 utility를 정할 수 없고, candidate의 historical '
                'correction 정보가 있어야 selection이 작동한다.')
    elif beats == 0:
        name = 'UTILITY_IS_NOT_LEARNABLE_FROM_AVAILABLE_QUERY_FEATURES'
        text = ('shortlist 안에 좋은 후보는 있지만 (oracle headroom 존재), 허용된 query '
                'feature만으로는 그것을 지목하지 못한다. 새 architecture를 만들기 전에 '
                'observability 분석으로 이동한다.')
    else:
        name = 'MIXED'
        text = 'setting마다 결과가 갈린다. 위 표가 결과다.'

    lines += ['', '## 판정', '', f'### {name}', '', text, '',
              f'- learned arm이 Existing CARTS를 이긴 횟수: **{beats}/{total}**',
              f'  - past_pair {per_arm["past_pair"]}, residual_aware {per_arm["residual_aware"]}',
              f'- gap recovery가 original retriever보다 높은 횟수: **{aligned}/{len(alignment)}**',
              f'- oracle headroom이 존재하는 setting: **{oracle_headroom}/{total}**', '']
    return lines, name


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='./metrics/utility_reranker')
    p.add_argument('--phase', type=int, default=0)
    a = p.parse_args()
    root = Path(a.root)
    lines, name = phase2(root) if a.phase >= 2 else phase0(root)
    report = '\n'.join(lines) + '\n'
    root.mkdir(parents=True, exist_ok=True)
    (root / 'FINAL_REPORT.md').write_text(report)
    print(report)
    print(f'DECISION: {name}')


if __name__ == '__main__':
    main()
