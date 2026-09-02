#!/usr/bin/env python3
"""Report for the ResDirect vs ResSel oracle-hybrid diagnostic."""
import argparse, csv
from pathlib import Path
from collections import Counter


def fmt(v, spec='.4f'):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    return '—' if v != v else format(v, spec)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='./metrics/resdirect_vs_ressel')
    a = p.parse_args()
    root = Path(a.root)
    path = root / 'oracle_hybrid_summary.csv'
    rows = list(csv.DictReader(open(path))) if path.exists() else []
    if not rows:
        print('no results'); return

    L = ['# ResDirect vs ResSel — Oracle Hybrid Diagnostic', '',
         '질문: **ResDirect가 평균적으로 좋은 상황에서도, retrieval이 반드시 필요한 query subset이 존재하는가?**', '',
         '`shuffled`는 ResSel 예측을 query 간에 섞은 대조군이다. oracle min은 구조상 항상 이기므로, ',
         '실제 gain이 이 노이즈 바닥을 넘어야만 의미가 있다.', '',
         '| Dataset | Pred | Base | Current | ResDirect | ResSel | OracleHybrid | Shuffled | Gain% | Shuffled% | Excess% | SelWin | maxCorr | Verdict |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        L.append('| ' + ' | '.join([
            r['dataset'], r['pred_len'], fmt(r['base_mse']), fmt(r['current_carts_mse']),
            fmt(r['resdirect_mse']), fmt(r['ressel_mse']), fmt(r['oracle_hybrid_mse']),
            fmt(r['shuffled_hybrid_mse']),
            fmt(float(r['oracle_hybrid_gain_ratio']) * 100, '.1f'),
            fmt(float(r['shuffled_hybrid_gain_ratio']) * 100, '.1f'),
            fmt(float(r['excess_gain_ratio']) * 100, '.1f'),
            fmt(r['ressel_win_rate'], '.3f'), fmt(r['max_abs_delta_feature_corr'], '.3f'),
            r['verdict']]) + ' |')

    verdicts = [r['verdict'] for r in rows]
    counts = Counter(verdicts)
    majority = counts.most_common(1)[0][0]
    mixed = len(counts) > 1
    excess = [float(r['excess_gain_ratio']) for r in rows]
    corrs = [float(r['max_abs_delta_feature_corr']) for r in rows]

    L += ['', '## Δ(q)와 관측 가능한 past feature의 상관', '',
          '| Dataset | Pred | 최강 feature | 상관 |', '|---|---|---|---|']
    for r in rows:
        L.append(f"| {r['dataset']} | {r['pred_len']} | {r['best_delta_feature']} "
                 f"| {fmt(r['max_abs_delta_feature_corr'], '.3f')} |")

    if majority == 'RETRIEVAL_COMPLEMENTARY' and not mixed:
        case, answer = 'A — RETRIEVAL_COMPLEMENTARY', 'YES'
        detail = ('노이즈 바닥을 넘는 hybrid gain이 있고 Δ(q)가 관측 가능한 feature와 상관된다. '
                  '다음 연구는 "when to retrieve" router.')
    elif majority == 'RETRIEVAL_REDUNDANT' and not mixed:
        case, answer = 'B — RETRIEVAL_REDUNDANT', 'NO'
        detail = ('hybrid gain이 shuffle 대조군과 구분되지 않는다. retrieval이 ResDirect에 '
                  '추가로 주는 complementary 정보가 없다. retrieval branch를 더 확장하지 않는다.')
    elif majority == 'COMPLEMENTARY_BUT_UNIDENTIFIABLE':
        case, answer = 'A\' — COMPLEMENTARY_BUT_UNIDENTIFIABLE', 'NO (현재로서는)'
        detail = ('regime은 존재하지만 관측 가능한 past feature로 구분되지 않는다. router를 만들 '
                  '근거가 아직 없으므로, 먼저 Δ(q)를 예측할 feature를 찾아야 한다.')
    else:
        dep = 'HORIZON_DEPENDENT' if len({r['dataset'] for r in rows}) == 1 else 'MIXED'
        case, answer = f'C — {dep}', 'PARTIAL'
        detail = f'setting마다 판정이 다르다: {dict(counts)}'

    L += ['', '## 판정', '', f'### Case {case}', '', detail, '',
          f'- excess gain ratio 평균: **{sum(excess)/len(excess)*100:+.1f}%** '
          f'(범위 {min(excess)*100:+.1f} ~ {max(excess)*100:+.1f}%)',
          f'- Δ(q) 최대 feature 상관 평균: **{sum(corrs)/len(corrs):.3f}**', '',
          '## 최종 질문에 대한 답', '',
          '> ResDirect가 평균적으로 더 좋은 상황에서도, Historical Retrieval이 반드시 필요한 query subset이 실제로 존재하는가?',
          '', f'### **{answer}**', '']
    (root / 'oracle_hybrid_report.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
