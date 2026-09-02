#!/usr/bin/env python3
"""STEP 5 -- merge every diagnostic into one report and one recommendation.

Reads whatever STEP CSVs exist (a failed step just leaves its section empty) and
writes FINAL_REPORT.md plus final_summary.csv. The recommendation follows the
decision rule fixed before the numbers were seen, so the conclusion is a
consequence of the measurements rather than a reading of them.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path


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


def fmt(value, spec='.4f'):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    return '—' if value != value else format(value, spec)


def table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '|'.join(['---'] * len(header)) + '|']
    out += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return out


def majority(labels, fallback='NO_DATA'):
    return Counter(labels).most_common(1)[0][0] if labels else fallback


def build(root):
    root = Path(root)
    gate = read(root / 'stage2_gate.csv')
    geometry = read(root / 'stage1_geometry.csv')
    residual = read(root / 'residual_oracle.csv')
    utility = read(root / 'forecast_utility.csv')
    lines = ['# Next Retrieval Direction — Diagnosis Report', '']

    # ---- 1 ----------------------------------------------------------------
    lines += ['## 1. Stage-2는 retrieval을 실제로 사용하고 있는가?', '']
    if gate:
        lines += table(
            ['Dataset', 'Pred', 'learned λ', 'Learned MSE', 'Gate=0', 'Best α',
             'Best α MSE', 'Shuffled', 'Verdict'],
            [[r['dataset'], r['pred_len'], fmt(r['learned_gate_mean']),
              fmt(r['learned_mse']), fmt(r['gate0_mse']), fmt(r['best_alpha'], '.1f'),
              fmt(r['best_alpha_test_mse']), fmt(r['shuffled_mse']), r['verdict']]
             for r in gate])
        gate_verdict = majority([r['verdict'] for r in gate])
        lines += ['', f'**판정: {gate_verdict}**',
                  '', 'shuffled가 learned보다 나쁘면 retrieval이 query별 정보를 담고 있다는 뜻이고, '
                  'gate=0이 learned보다 좋으면 그 정보를 써도 손해라는 뜻이다.']
    else:
        gate_verdict = 'NO_DATA'
        lines += ['(결과 없음)']
    lines += ['']

    # ---- 2 ----------------------------------------------------------------
    lines += ['## 2. Cosine geometry가 문제인가?', '']
    geom_verdict = 'NO_DATA'
    if geometry:
        rows, pairs = [], {}
        for r in geometry:
            key = (r['dataset'], r['pred_len'], r.get('retrieval_similarity', 'cosine'))
            pairs.setdefault(key, {})[r['split']] = r
        for (ds, pl, sim), splits in sorted(pairs.items()):
            rows.append([ds, pl, sim] + [
                fmt(num(splits[s], k)) if s in splits else '—'
                for s in ('train', 'val', 'test')
                for k in ('oracle_recall_at_10',)
            ] + [
                fmt(num(splits['test'], 'retrieved_future_mse_at_10')) if 'test' in splits else '—',
                fmt(num(splits['test'], 'oracle_gap_recovery_at_10'), '.3f') if 'test' in splits else '—',
            ])
        lines += table(['Dataset', 'Pred', 'Similarity', 'Train R@10', 'Val R@10',
                        'Test R@10', 'Test RetrMSE', 'Test GapRec'], rows)
        deltas = []
        for (ds, pl, sim), splits in pairs.items():
            if sim != 'l2' or 'test' not in splits:
                continue
            cos = pairs.get((ds, pl, 'cosine'), {}).get('test')
            if not cos:
                continue
            deltas.append((
                num(splits['test'], 'oracle_gap_recovery_at_10') - num(cos, 'oracle_gap_recovery_at_10'),
                num(cos, 'oracle_recall_at_10'),
                num(splits['test'], 'oracle_recall_at_10'),
            ))
        if deltas:
            gap_gain = sum(d[0] for d in deltas) / len(deltas)
            rel = [(b - a) / a for _, a, b in deltas if a > 0]
            recall_gain = sum(rel) / len(rel) if rel else 0.0
            if gap_gain >= 0.03 or recall_gain >= 0.20:
                geom_verdict = 'L2_CLEARLY_BETTER'
            elif gap_gain >= 0.005:
                geom_verdict = 'GEOMETRY_MINOR_EFFECT'
            else:
                geom_verdict = 'GEOMETRY_NO_EFFECT'
            lines += ['', f'평균 gap-recovery 차이(L2−cosine): {gap_gain:+.4f}, '
                          f'평균 Recall@10 상대 변화: {recall_gain:+.1%}']
        lines += ['', f'**판정: {geom_verdict}**']
    else:
        lines += ['(결과 없음)']
    lines += ['']

    # ---- 3 ----------------------------------------------------------------
    lines += ['## 3. Residual retrieval의 upper bound가 있는가?', '']
    if residual:
        lines += table(
            ['Dataset', 'Pred', 'Base', 'Future Oracle', 'Residual Oracle', 'best α', 'Verdict'],
            [[r['dataset'], r['pred_len'], fmt(r['base_mse']), fmt(r['future_oracle_mse']),
              fmt(r['residual_oracle_best_mse']), fmt(r['residual_oracle_best_alpha'], '.1f'),
              r['verdict']] for r in residual])
        residual_verdict = majority([r['verdict'] for r in residual])
        lines += ['', f'**판정: {residual_verdict}**',
                  '', '두 Oracle 모두 query future로 후보를 *선택*하므로 배포 가능한 모델이 아니라 '
                  '상한이다. 어느 target을 학습할 가치가 있는지만 말해준다.']
    else:
        residual_verdict = 'NO_DATA'
        lines += ['(결과 없음)']
    lines += ['']

    # ---- 4 ----------------------------------------------------------------
    lines += ['## 4. Forecast Utility target이 유망한가?', '']
    if utility:
        lines += table(
            ['Dataset', 'Pred', 'Base', 'Future', 'Residual', 'Utility',
             'U>0 비율', 'past↔U ρ', 'enc↔U ρ', 'Cov@500', 'GapRec(raw)', 'Verdict'],
            [[r['dataset'], r['pred_len'], fmt(r['base_mse']), fmt(r['future_oracle_mse']),
              fmt(r['residual_oracle_mse']), fmt(r['utility_oracle_mse']),
              fmt(r['positive_utility_fraction'], '.3f'),
              fmt(r['past_utility_spearman'], '.3f'),
              fmt(r['encoder_utility_spearman'], '.3f'),
              fmt(r['raw_utility_coverage_at_500'], '.3f'),
              fmt(r['utility_gap_recovery_raw'], '.3f'), r['verdict']] for r in utility])
        utility_verdict = majority([r['verdict'] for r in utility])
        lines += ['', f'**판정: {utility_verdict}**']
    else:
        utility_verdict = 'NO_DATA'
        lines += ['(결과 없음)']
    lines += ['']

    # ---- decision ---------------------------------------------------------
    signal = [num(r, 'past_utility_spearman', 0.0) for r in utility]
    coverage = [num(r, 'raw_utility_coverage_at_500', 0.0) for r in utility]
    mean_signal = sum(signal) / len(signal) if signal else 0.0
    mean_cov = sum(coverage) / len(coverage) if coverage else 0.0

    if utility_verdict == 'UTILITY_TARGET_PROMISING' and mean_signal >= 0.10 and mean_cov >= 0.20:
        option, title = 'A', 'Utility-aligned Stage-1'
        detail = ('Forecast-Utility teacher + residual retrieval value. '
                  'Utility Oracle이 가장 낮고, past에서 utility 순위 신호가 있으며, '
                  'shortlist coverage도 의미 있는 수준이다.')
    elif residual_verdict in ('RESIDUAL_TARGET_PROMISING', 'RESIDUAL_TARGET_COMPARABLE') \
            and residual_verdict != 'NO_DATA':
        option, title = 'B', 'Residual-aligned Stage-1'
        detail = ('Residual similarity teacher + residual retrieval value. '
                  'Residual Oracle 상한은 확실하지만 utility 순위 신호가 약하거나 '
                  'coverage가 부족하다.')
    elif geom_verdict == 'L2_CLEARLY_BETTER':
        option, title = 'C', 'Geometry 변경 우선'
        detail = 'Unnormalized L2 embedding retrieval이 train fitting과 test quality를 모두 개선한다.'
    else:
        option, title = 'D', 'Retrieval Stage 재설계'
        detail = ('Future / Residual / Utility 어느 것도 past에서 충분히 예측되지 않는다. '
                  '현재 Stage-1 bi-encoder retrieval target을 더 확장하지 말 것.')

    lines += ['## 최종 권고', '',
              f'### Option {option} — {title}', '', detail, '',
              '### 판정 근거', '',
              f'- Stage-2 retrieval 사용: `{gate_verdict}`',
              f'- Geometry: `{geom_verdict}`',
              f'- Residual target: `{residual_verdict}`',
              f'- Utility target: `{utility_verdict}`',
              f'- past↔utility Spearman 평균: {mean_signal:.3f}',
              f'- raw utility coverage@500 평균: {mean_cov:.3f}', '']

    (root / 'FINAL_REPORT.md').write_text('\n'.join(lines) + '\n')

    summary_columns = ['section', 'dataset', 'pred_len', 'metric', 'value']
    with open(root / 'final_summary.csv', 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_columns)
        writer.writeheader()
        for name, rows in (('stage2_gate', gate), ('geometry', geometry),
                           ('residual_oracle', residual), ('forecast_utility', utility)):
            for row in rows:
                for key, value in row.items():
                    if key in ('dataset', 'pred_len', 'checkpoint'):
                        continue
                    writer.writerow({
                        'section': name, 'dataset': row.get('dataset', ''),
                        'pred_len': row.get('pred_len', ''), 'metric': key, 'value': value,
                    })
        writer.writerow({'section': 'decision', 'dataset': '', 'pred_len': '',
                         'metric': 'recommended_option', 'value': option})
    return option, title


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='./metrics/next_retrieval_diagnosis')
    args = parser.parse_args()
    Path(args.root).mkdir(parents=True, exist_ok=True)
    option, title = build(args.root)
    print(Path(args.root, 'FINAL_REPORT.md').read_text())
    print(f'RECOMMENDED: Option {option} — {title}')


if __name__ == '__main__':
    main()
