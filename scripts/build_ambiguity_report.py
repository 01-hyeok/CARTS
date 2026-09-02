#!/usr/bin/env python3
"""Report for the past-neighborhood ambiguity diagnostic.

The verdict this produces is about *information*, not about any model:

  PAST_IS_INFORMATIVE       residual dispersion collapses as the past
                            neighborhood tightens -- the target past does carry
                            the signal, so earlier failures were model-side
  IRREDUCIBLE_AMBIGUITY     it does not -- near-identical pasts still have
                            unrelated residuals, and no encoder can fix that
  INCONCLUSIVE              the positive control failed, i.e. the neighborhoods
                            were never actually tight

Read against the trained ResDirect arm: if the k-NN floor is no better than what
ResDirect already achieves, the model has saturated the information in the past.
"""

import argparse
import csv
import os
import statistics as st
from pathlib import Path

BUCKETS = ['nearest1', '1%', '5%', '10%', 'all']


def read(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def num(row, key):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return None if value != value else value


def aggregate(rows):
    """Channel-mean per (dataset, pred_len, past_metric, bucket)."""
    grouped = {}
    for row in rows:
        key = (row['dataset'], int(row['pred_len']), row['past_metric'], row['bucket'])
        grouped.setdefault(key, []).append(row)
    out = {}
    for key, group in grouped.items():
        entry = {'channels': len(group)}
        for column in ('past_distance_ratio', 'residual_pair_mse_ratio',
                       'residual_cosine_dispersion', 'future_pair_mse_ratio',
                       'past_tail_pair_mse_ratio', 'shuffled_residual_pair_mse_ratio',
                       'knn_residual_mse', 'residual_power',
                       'residual_explained_fraction', 'best_candidate_entropy'):
            values = [num(r, column) for r in group]
            values = [v for v in values if v is not None]
            entry[column] = st.mean(values) if values else float('nan')
        out[key] = entry
    return out


def fmt(value, digits=3):
    return '--' if value is None or value != value else f'{value:.{digits}f}'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='./metrics/past_neighborhood_ambiguity')
    p.add_argument('--resdirect_csv', default='./metrics/cross_channel_retrieval/resdirect.csv')
    a = p.parse_args()
    root = Path(a.root)
    table = aggregate(read(root / 'ambiguity.csv'))
    if not table:
        print('no rows'); return

    settings = sorted({(d, p, m) for d, p, m, _ in table}, key=lambda k: (k[0], k[1], k[2]))
    resdirect = {
        (r['dataset'], int(r['pred_len'])): float(r['residual_pred_mse'])
        for r in read(a.resdirect_csv)
        if r['split'] == 'test' and r['arm'] == 'target_only_resdirect'
    }

    lines = [
        '# Past-Neighborhood Future Ambiguity',
        '',
        '질문: **target past가 애초에 useful historical correction을 식별할 정보를 담고 있는가?**',
        '',
        'encoder를 전혀 쓰지 않는다. past가 거의 같은 query들을 모아 그들의 residual이 '
        '얼마나 흩어지는지만 본다. 흩어짐이 줄면 정보는 past 안에 있었고 (모델 문제), '
        '줄지 않으면 target past만으로는 원천적으로 식별 불가능하다.',
        '',
        '`ratio`는 모두 전체 후보(bucket=all) 대비 값이다. 1.0이면 "가까운 past 이웃이 '
        '무작위 이웃보다 나을 게 없다"는 뜻이다.',
        '',
        '### Table — dispersion by neighborhood tightness (channel mean)',
        '',
        '| Dataset | Pred | Past metric | Bucket | d_past ratio | residual MSE ratio | '
        'cosine dispersion | tail ratio (control+) | shuffled ratio (control−) | '
        'kNN residual MSE | explained | best-cand entropy |',
        '|---|---|---|---|---|---|---|---|---|---|---|---|',
    ]
    for dataset, pred, metric in settings:
        for bucket in BUCKETS:
            entry = table.get((dataset, pred, metric, bucket))
            if not entry:
                continue
            lines.append(
                f"| {dataset} | {pred} | {metric} | {bucket} | "
                f"{fmt(entry['past_distance_ratio'])} | "
                f"{fmt(entry['residual_pair_mse_ratio'])} | "
                f"{fmt(entry['residual_cosine_dispersion'])} | "
                f"{fmt(entry['past_tail_pair_mse_ratio'])} | "
                f"{fmt(entry['shuffled_residual_pair_mse_ratio'])} | "
                f"{fmt(entry['knn_residual_mse'], 4)} | "
                f"{fmt(entry['residual_explained_fraction'])} | "
                f"{fmt(entry['best_candidate_entropy'])} |"
            )

    lines += ['', '### kNN information floor vs the trained ResDirect arm', '',
              '| Dataset | Pred | Past metric | kNN residual MSE (1%) | '
              'ResDirect residual MSE | residual power (predict 0) |',
              '|---|---|---|---|---|---|']
    for dataset, pred, metric in settings:
        entry = table.get((dataset, pred, metric, '1%'))
        if not entry:
            continue
        lines.append(
            f"| {dataset} | {pred} | {metric} | {fmt(entry['knn_residual_mse'], 4)} | "
            f"{fmt(resdirect.get((dataset, pred)), 4)} | "
            f"{fmt(entry['residual_power'], 4)} |"
        )

    # Verdict, read off the tightest honest bucket.
    verdicts = []
    for dataset, pred, metric in settings:
        tight = table.get((dataset, pred, metric, '1%'))
        nearest = table.get((dataset, pred, metric, 'nearest1'))
        if not tight:
            continue
        control_ok = (nearest or tight)['past_tail_pair_mse_ratio'] < 0.5
        null_ok = 0.8 < tight['shuffled_residual_pair_mse_ratio'] < 1.25
        collapsed = tight['residual_pair_mse_ratio'] < 0.8
        if not control_ok:
            name = 'INCONCLUSIVE'
        elif collapsed:
            name = 'PAST_IS_INFORMATIVE'
        else:
            name = 'IRREDUCIBLE_AMBIGUITY'
        verdicts.append({
            'setting': f'{dataset}/{pred}/{metric}', 'verdict': name,
            'control_ok': control_ok, 'null_ok': null_ok,
            'residual_ratio': tight['residual_pair_mse_ratio'],
            'explained': tight['residual_explained_fraction'],
            'entropy': tight['best_candidate_entropy'],
        })

    lines += ['', '### 판정 (bucket = 1%)', '',
              '| Setting | Verdict | residual ratio | explained | entropy | '
              'control+ ok | control− ok |', '|---|---|---|---|---|---|---|']
    for row in verdicts:
        lines.append(
            f"| {row['setting']} | **{row['verdict']}** | {fmt(row['residual_ratio'])} | "
            f"{fmt(row['explained'])} | {fmt(row['entropy'])} | "
            f"{'yes' if row['control_ok'] else 'NO'} | "
            f"{'yes' if row['null_ok'] else 'NO'} |"
        )

    names = {row['verdict'] for row in verdicts}
    overall = names.pop() if len(names) == 1 else 'MIXED'
    explanation = {
        'PAST_IS_INFORMATIVE': (
            'Target past 안에 residual에 대한 정보가 실제로 있다. 가까운 past 이웃의 '
            'residual은 무작위 이웃보다 뚜렷하게 덜 흩어진다. 따라서 지금까지의 retrieval '
            '실패는 "입력에 정보가 없어서"가 아니라 모델/목적함수 쪽 문제로 좁혀진다. '
            '다만 best-candidate identity entropy가 높게 남는다면, "residual을 예측할 만큼의 '
            '정보"와 "어느 과거 후보가 최적인지 특정할 만큼의 정보"는 서로 다른 문제다.'
        ),
        'IRREDUCIBLE_AMBIGUITY': (
            'Target past가 거의 같아도 residual은 여전히 전역 수준으로 흩어진다. '
            'future correction을 target past만으로 식별하는 것은 원천적으로 불가능하며, '
            'encoder를 아무리 개선해도 이 한계는 움직이지 않는다.'
        ),
        'INCONCLUSIVE': (
            'Positive control이 통과하지 못했다. 이웃이 실제로는 충분히 가깝지 않아 '
            '다른 수치를 해석할 수 없다.'
        ),
        'MIXED': 'Setting마다 판정이 갈린다. 위 표가 결과이며 단일 라벨은 오히려 정보를 숨긴다.',
    }[overall]
    lines += ['', '## 최종 판정', '', f'### {overall}', '', explanation, '']

    report = '\n'.join(lines) + '\n'
    root.mkdir(parents=True, exist_ok=True)
    (root / 'FINAL_REPORT.md').write_text(report)
    print(report)
    print(f'DECISION: {overall}')


if __name__ == '__main__':
    main()
