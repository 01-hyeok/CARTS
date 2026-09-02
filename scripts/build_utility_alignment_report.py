#!/usr/bin/env python3
"""Turn the alignment sweep into a verdict.

The decision rule is fixed here before the numbers exist, so the recommendation
follows from the measurement rather than from a reading of it.
"""

import argparse
import csv
from pathlib import Path

SPEARMAN_STRONG, SPEARMAN_WEAK = 0.6, 0.3
OVERLAP_STRONG = 0.5


def read(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as handle:
        return list(csv.DictReader(handle))


def num(row, key, default=float('nan')):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return default


def fmt(row, key, spec='.4f'):
    value = num(row, key)
    return '—' if value != value else format(value, spec)


def table(header, rows):
    return (['| ' + ' | '.join(header) + ' |',
             '|' + '|'.join(['---'] * len(header)) + '|']
            + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows])


def build(root):
    root = Path(root)
    rows = read(root / 'target_alignment.csv')
    by_score = {name: [r for r in rows if r['score'] == name]
                for name in ('future', 'residual')}

    lines = [
        '# Forecast-Utility Alignment — Stage-2 실제 경로 기준 진단', '',
        '핵심 질문: **Future-MSE 유사도가 실제 Stage-2 downstream 유용성을 정렬하는가?**', '',
        '유용성은 residual 수식이 아니라 실제 `RelationStage2.forward()` 호출로 측정했다:', '',
        '```', 'U(q, k, c) = MSE_c(Y_q, no-retrieval) − MSE_c(Y_q, Stage-2 given only k)', '```', '',
        '후보 k 하나만 retrieval branch에 주입하고 production forward를 그대로 실행한다. '
        'offset 복원 · mixer · gate가 모두 모델 안 한 곳에만 존재하므로, 과거 진단을 무효화한 '
        'double-offset 유형의 오차가 재현될 수 없다.', '',
    ]

    lines += ['## 1. 후보 pool 안에 쓸모 있는 후보가 있는가?', '']
    lines += table(['Dataset', 'Pred', 'Pool', 'Queries', 'Base MSE',
                    'Positive Rate', 'Mean U', 'Best U'],
                   [[r['dataset'], r['pred_len'], r['pool_size'], r['queries'],
                     fmt(r, 'base_mse'), fmt(r, 'positive_rate', '.3f'),
                     fmt(r, 'mean_utility', '+.4f'), fmt(r, 'best_utility', '+.4f')]
                    for r in by_score['future']]) + ['']

    for name, title in (('future', 'Future-MSE 유사도'), ('residual', 'Residual 유사도')):
        lines += [f'## 2{"ab"[name == "residual"]}. {title} ↔ 실제 utility 정렬', '']
        lines += table(['Dataset', 'Pred', 'Pearson', 'Spearman', 'Ov@1', 'Ov@5',
                        'Ov@10', 'Ov@50', 'NDCG@10', 'U@top1', 'U@top10'],
                       [[r['dataset'], r['pred_len'], fmt(r, 'pearson', '.3f'),
                         fmt(r, 'spearman', '.3f'), fmt(r, 'overlap_at_1', '.3f'),
                         fmt(r, 'overlap_at_5', '.3f'), fmt(r, 'overlap_at_10', '.3f'),
                         fmt(r, 'overlap_at_50', '.3f'), fmt(r, 'ndcg_at_10', '.3f'),
                         fmt(r, 'utility_at_score_top1', '+.4f'),
                         fmt(r, 'utility_at_score_top10', '+.4f')]
                        for r in by_score[name]]) + ['']

    # Residual vs Future, head to head on the same queries.
    pairs = []
    for f_row in by_score['future']:
        match = next((r for r in by_score['residual']
                      if r['dataset'] == f_row['dataset']
                      and r['pred_len'] == f_row['pred_len']), None)
        if match:
            pairs.append((f_row, match))
    residual_wins = sum(1 for f, r in pairs if num(r, 'spearman') > num(f, 'spearman'))
    lines += ['## 3. 어느 target이 실제 utility에 더 가까운가?', '',
              f'- Residual 유사도의 Spearman이 더 높은 setting: **{residual_wins} / {len(pairs)}**', '']
    if pairs:
        lines += table(['Dataset', 'Pred', 'Spearman (Future)', 'Spearman (Residual)', 'Δ'],
                       [[f['dataset'], f['pred_len'], fmt(f, 'spearman', '.3f'),
                         fmt(r, 'spearman', '.3f'),
                         format(num(r, 'spearman') - num(f, 'spearman'), '+.3f')]
                        for f, r in pairs]) + ['']

    future_spearman = [num(r, 'spearman') for r in by_score['future']]
    future_overlap = [num(r, 'overlap_at_10') for r in by_score['future']]
    mean_spearman = sum(future_spearman) / len(future_spearman) if future_spearman else float('nan')
    mean_overlap = sum(future_overlap) / len(future_overlap) if future_overlap else float('nan')

    if mean_spearman >= SPEARMAN_STRONG and mean_overlap >= OVERLAP_STRONG:
        case, title = 'A', 'Future-MSE target은 타당하다'
        detail = ('Future-MSE 유사도가 실제 downstream utility를 잘 정렬한다. Stage-1의 검색 '
                  'target 자체는 문제가 아니므로, 개선 여지는 encoder/objective 쪽에 있다.')
    elif mean_spearman >= SPEARMAN_WEAK:
        case, title = 'B', 'Future-MSE는 부분적으로만 타당하다'
        detail = ('Future-MSE는 utility와 어느 정도 상관하지만 상위 구간 정렬이 약하다. '
                  'Top-K 구간을 직접 겨냥한 target(residual 또는 utility 자체)으로 교체하면 '
                  '이득이 예상된다.')
    else:
        case, title = 'C', 'Future-MSE target은 잘못된 목표다'
        detail = ('Future-MSE 유사도가 실제 utility와 거의 무관하다. Stage-1을 Future-MSE '
                  'Oracle에 맞추려는 시도 자체가 잘못된 목표를 최적화한 것이며, retrieval '
                  'target을 utility 기반으로 재정의해야 한다.')

    lines += ['## 최종 판정', '', f'### Case {case} — {title}', '', detail, '',
              '### 근거', '',
              f'- Future-MSE 평균 Spearman: **{mean_spearman:.3f}** '
              f'(기준: 강 ≥ {SPEARMAN_STRONG}, 약 ≥ {SPEARMAN_WEAK})',
              f'- Future-MSE 평균 Top-10 overlap: **{mean_overlap:.3f}** (기준 ≥ {OVERLAP_STRONG})',
              f'- Residual이 Future보다 나은 setting: **{residual_wins}/{len(pairs)}**', '',
              '### 범위에 대한 단서', '',
              '여기서 측정한 것은 **단일 후보** utility다. Stage-2가 실제로 쓰는 것은 Top-K '
              '가중 평균이므로, 단일 후보 순위가 잘 맞아도 집계 단계에서 이득이 사라질 수 있다. '
              '이 진단은 target 정의의 타당성만 판정하며, 집계 방식은 별도 문제로 남는다.', '']

    (root / 'FINAL_REPORT.md').write_text('\n'.join(lines) + '\n')
    return case, title


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='./metrics/forecast_utility_alignment')
    args = parser.parse_args()
    Path(args.root).mkdir(parents=True, exist_ok=True)
    case, title = build(args.root)
    print(Path(args.root, 'FINAL_REPORT.md').read_text())
    print(f'DECISION: Case {case} — {title}')


if __name__ == '__main__':
    main()
