#!/usr/bin/env python3
"""STEP 4 -- merge the filtering diagnostics and pick a case.

The decision rule is fixed before the numbers exist, so the recommendation is a
consequence of the measurements rather than a reading of them.
"""

import argparse
import csv
from pathlib import Path


def read(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as handle:
        return list(csv.DictReader(handle))


def f(row, key, spec='.4f'):
    try:
        return format(float(row[key]), spec)
    except (KeyError, TypeError, ValueError):
        return '—'


def num(row, key, default=float('nan')):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def table(header, rows):
    return (['| ' + ' | '.join(header) + ' |',
             '|' + '|'.join(['---'] * len(header)) + '|']
            + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows])


def build(root):
    root = Path(root)
    oracle = read(root / 'oracle_filtering.csv')
    classifier = [r for r in read(root / 'classifier_results.csv') if r['split'] == 'test']
    stage2 = read(root / 'stage2_filtering.csv')
    lines = ['# Candidate Utility Filtering — Diagnosis Report', '',
             '핵심 질문: **현재 검색 결과 안에서 해로운 candidate만 제거해도 Forecast가 좋아지는가?**', '']

    # Q1 / Table 1
    lines += ['## Q1. Top-M 안에 U>0 candidate가 충분히 존재하는가?', '']
    if oracle:
        lines += table(['Dataset', 'Pred', 'Retriever', 'M', 'Positive Rate',
                        'Hit Positive', 'Mean #Pos', 'Best Utility'],
                       [[r['dataset'], r['pred_len'], r['retriever'], r['candidate_pool_m'],
                         f(r, 'positive_rate', '.3f'), f(r, 'at_least_one_positive_rate', '.3f'),
                         f(r, 'mean_positive_count', '.1f'), f(r, 'best_available_utility', '.4f')]
                        for r in oracle])
    else:
        lines += ['(결과 없음)']
    lines += ['']

    # Q2 / Table 2
    lines += ['## Q2. Oracle filtering만으로 Forecast가 얼마나 개선되는가?', '']
    if oracle:
        lines += table(['Dataset', 'Pred', 'Retriever', 'M', 'Base', 'Current',
                        'Pos Uniform', 'Pos Weighted', 'Best Single', 'Global Oracle',
                        'Gain', 'Recovery'],
                       [[r['dataset'], r['pred_len'], r['retriever'], r['candidate_pool_m'],
                         f(r, 'base_mse'), f(r, 'current_retrieval_mse'),
                         f(r, 'oracle_positive_uniform_mse'), f(r, 'oracle_positive_weighted_mse'),
                         f(r, 'oracle_best_single_mse'), f(r, 'global_utility_oracle_mse'),
                         f(r, 'filtering_gain'), f(r, 'filtering_recovery', '.3f')]
                        for r in oracle])
    lines += ['']

    # Q3
    lines += ['## Q3. 평균이 좋은가, best-single이 좋은가? (aggregation bottleneck)', '']
    avg_better = sum(
        1 for r in oracle
        if num(r, 'oracle_positive_uniform_mse') < num(r, 'oracle_best_single_mse'))
    lines += [f'- positive-average가 best-single보다 좋은 경우: **{avg_better} / {len(oracle)}**', '']

    # Q4 / Table 3
    lines += ['## Q4. past-only pair로 utility를 학습할 수 있는가?', '']
    if classifier:
        lines += table(['Dataset', 'Pred', 'Delta', 'Prevalence', 'PR-AUC', 'ROC-AUC',
                        'P@10', 'PosRate@10', 'MeanU@10'],
                       [[r['dataset'], r['pred_len'], r['delta'],
                         f(r, 'positive_prevalence', '.3f'), f(r, 'pr_auc', '.4f'),
                         f(r, 'roc_auc', '.4f'), f(r, 'precision_at_10', '.3f'),
                         f(r, 'positive_utility_rate_at_10', '.3f'),
                         f(r, 'mean_utility_at_10', '+.4f')] for r in classifier])
    else:
        lines += ['(STEP 2 미실행 또는 결과 없음)']
    lines += ['']

    # Q5 / Table 4
    lines += ['## Q5. Learned filtering이 실제 Forecast MSE를 개선하는가?', '']
    if stage2:
        lines += table(['Dataset', 'Pred', 'Base', 'Current', 'Soft Filter',
                        'Hard Filter', 'Oracle Filter', 'Global Oracle',
                        'PosRate 전', 'PosRate 후', '남은 후보'],
                       [[r['dataset'], r['pred_len'], f(r, 'base_mse'), f(r, 'current_mse'),
                         f(r, 'soft_filter_mse'), f(r, 'hard_filter_mse'),
                         f(r, 'oracle_filter_mse'), f(r, 'global_utility_oracle_mse'),
                         f(r, 'positive_rate_before', '.3f'), f(r, 'positive_rate_after_hard', '.3f'),
                         f(r, 'retained_candidates', '.1f')] for r in stage2])
    else:
        lines += ['(STEP 3 미실행 또는 결과 없음)']
    lines += ['']

    # ---- decision ----
    max_gain = max([num(r, 'filtering_gain', 0.0) for r in oracle], default=0.0)
    margins = [num(r, 'pr_auc', 0.0) - num(r, 'positive_prevalence', 1.0) for r in classifier]
    max_margin = max(margins, default=0.0)
    learned_helps = sum(
        1 for r in stage2
        if min(num(r, 'soft_filter_mse', 9e9), num(r, 'hard_filter_mse', 9e9))
        < num(r, 'current_mse', 0.0))

    if max_gain < 0.02:
        case, title = 'D', 'Candidate generation 재설계'
        detail = ('Oracle filtering조차 Base를 거의 개선하지 못한다. 현재 retriever의 pool에 '
                  'utility-positive candidate가 충분히 없다. Filtering 방향을 중단하고 '
                  'candidate generation을 다시 설계할 것.')
    elif avg_better <= len(oracle) // 3 and oracle:
        case, title = 'C', 'Aggregation 재설계'
        detail = ('Oracle best-single은 좋지만 positive-average는 나쁘다. Filtering만이 아니라 '
                  '여러 후보를 평균하는 방식 자체가 문제다. selection/mixture로 전환할 것.')
    elif max_margin < 0.05:
        case, title = 'B', 'Feature / scorer 개선'
        detail = ('Filtering target은 맞지만 past-only pair classifier가 utility를 예측하지 못한다. '
                  '다음 연구는 feature 또는 scorer 구조 개선.')
    elif learned_helps > 0:
        case, title = 'A', 'Candidate Utility Filtering 채택'
        detail = ('Oracle filtering이 크게 개선되고, classifier가 prevalence를 넘으며, '
                  'learned filtering이 실제 Forecast를 개선한다. CARTS의 새로운 주요 방향으로 채택.')
    else:
        case, title = 'B', 'Feature / scorer 개선'
        detail = ('Filtering target과 classifier 성능은 확보됐지만 learned filtering이 아직 '
                  'Forecast를 개선하지 못한다. scorer/threshold/aggregation을 다음으로 개선.')

    lines += ['## 최종 판정', '', f'### Case {case} — {title}', '', detail, '',
              '### 근거', '',
              f'- 최대 filtering gain (Base − Oracle filter): **{max_gain:.4f}**',
              f'- positive-average > best-single: **{avg_better}/{len(oracle)}**',
              f'- 최대 PR-AUC 마진 (test): **{max_margin:+.4f}**',
              f'- learned filter가 current를 이긴 setting: **{learned_helps}/{len(stage2)}**', '']

    (root / 'FINAL_REPORT.md').write_text('\n'.join(lines) + '\n')
    return case, title


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='./metrics/candidate_utility_filtering')
    args = parser.parse_args()
    Path(args.root).mkdir(parents=True, exist_ok=True)
    case, title = build(args.root)
    print(Path(args.root, 'FINAL_REPORT.md').read_text())
    print(f'DECISION: Case {case} — {title}')


if __name__ == '__main__':
    main()
