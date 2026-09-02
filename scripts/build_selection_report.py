#!/usr/bin/env python3
"""STEP 6 -- merge every selection arm and answer the research question.

The decision rule is fixed before the numbers exist, so the recommendation
follows from the measurements instead of being read into them.
"""
import argparse, csv
from pathlib import Path

EPS = 1e-8


def read(path):
    path = Path(path)
    return list(csv.DictReader(open(path))) if path.exists() else []


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
    return (['| ' + ' | '.join(header) + ' |',
             '|' + '|'.join(['---'] * len(header)) + '|']
            + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows])


def build(root):
    root = Path(root)
    ref = [r for r in read(root / 'classifier_topr.csv') if r.get('split') == 'test']
    ranker = [r for r in read(root / 'utility_ranker.csv') if r.get('split') == 'test']
    residual_aware = [r for r in ranker if r.get('features', '').endswith('residual')]
    ranker = [r for r in ranker if not r.get('features', '').endswith('residual')]
    selector = [r for r in read(root / 'predicted_residual_selector.csv')
                if r.get('split') == 'test']

    lines = ['# Utility-aware Candidate Selection — Final Report', '',
             '핵심 질문: **Broad pool에서 Base Forecast를 가장 잘 보정할 하나를 고르는 것이, '
             '비슷한 여러 과거를 평균하는 것보다 나은가?**', '']

    # unified per-setting method table
    per_setting = {}
    for r in ref:
        per_setting.setdefault((r['dataset'], int(r['pred_len'])), {})[r['method']] = r
    for r in ranker:
        per_setting.setdefault((r['dataset'], int(r['pred_len'])), {})[
            f"utility_ranker_{r['loss']}_top{r['top_r']}"] = r
    for r in residual_aware:
        per_setting.setdefault((r['dataset'], int(r['pred_len'])), {})[
            f"residual_aware_{r['loss']}_top{r['top_r']}"] = r
    for r in selector:
        key = (r['dataset'], int(r['pred_len']))
        per_setting.setdefault(key, {})['predicted_residual_selector'] = r
        per_setting[key]['predicted_residual_direct'] = {
            **r, 'forecast_mse': r['direct_correction_mse'],
            'forecast_mae': r['direct_correction_mae'],
        }

    lines += ['## 전체 비교 (test)', '']
    rows = []
    for (ds, pl), methods in sorted(per_setting.items()):
        for name, r in methods.items():
            rows.append([ds, pl, name, fmt(r.get('positive_at_1'), '.3f'),
                         fmt(r.get('selected_utility_at_1')),
                         fmt(r.get('selection_recovery_at_1'), '.3f'),
                         fmt(r.get('forecast_mse')), fmt(r.get('forecast_mae'))])
    lines += table(['Dataset', 'Pred', 'Method', 'Positive@1', 'Selected Utility',
                    'Selection Recovery', 'Forecast MSE', 'Forecast MAE'], rows)
    lines += ['']

    def best(methods, *names):
        vals = [num(methods[n], 'forecast_mse') for n in names if n in methods]
        return min(vals) if vals else float('nan')

    # ---- ablations -------------------------------------------------------
    lines += ['## Ablation', '']
    ab = []
    for (ds, pl), m in sorted(per_setting.items()):
        ab.append([ds, pl,
                   fmt(best(m, 'current_topk_avg')), fmt(best(m, 'classifier_top1')),
                   fmt(best(m, 'classifier_top3')), fmt(best(m, 'classifier_soft')),
                   fmt(best(m, 'utility_ranker_ce_top1', 'utility_ranker_kl_top1')),
                   fmt(best(m, 'residual_aware_ce_top1', 'residual_aware_kl_top1')),
                   fmt(best(m, 'predicted_residual_selector')),
                   fmt(best(m, 'predicted_residual_direct')),
                   fmt(best(m, 'oracle_best_single'))])
    lines += table(['Dataset', 'Pred', 'A: TopK avg', 'A: Top-1', 'A: Top-3',
                    'B: Binary soft', 'B: Utility ranker', 'C: +Cand residual',
                    'D: Residual selector', 'E: Direct correction', 'Oracle best-single'], ab)
    lines += ['']

    # ---- decision --------------------------------------------------------
    def wins(a_names, b_names):
        count = 0
        for m in per_setting.values():
            a, b = best(m, *a_names), best(m, *b_names)
            if a == a and b == b and a < b:
                count += 1
        return count, len(per_setting)

    top1_vs_soft = wins(['classifier_top1'], ['classifier_soft'])
    ranker_vs_top1 = wins(['utility_ranker_ce_top1', 'utility_ranker_kl_top1'],
                          ['classifier_top1'])
    resid_vs_pair = wins(['residual_aware_ce_top1', 'residual_aware_kl_top1'],
                         ['utility_ranker_ce_top1', 'utility_ranker_kl_top1'])
    selector_vs_all = wins(['predicted_residual_selector'],
                           ['utility_ranker_ce_top1', 'utility_ranker_kl_top1',
                            'classifier_top1', 'current_topk_avg'])
    direct_vs_selector = wins(['predicted_residual_direct'],
                              ['predicted_residual_selector'])
    best_learned_vs_current = wins(
        ['predicted_residual_selector', 'utility_ranker_ce_top1',
         'utility_ranker_kl_top1', 'residual_aware_ce_top1',
         'residual_aware_kl_top1', 'classifier_top1'], ['current_topk_avg'])

    total = max(len(per_setting), 1)
    if direct_vs_selector[0] > total // 2:
        case, title = 'E', 'Direct residual prediction이 retrieval보다 낫다'
        detail = ('예측 residual을 그대로 보정에 쓰는 편이 historical residual을 고르는 것보다 좋다. '
                  'retrieval 기구의 추가적 필요성이 약하다는 뜻이므로 무리하게 유지하지 않는다.')
        answer = 'NO'
    elif selector_vs_all[0] > total // 2:
        case, title = 'D', 'Predicted Query Residual Selector 채택'
        detail = ('CARTS retrieval을 similarity search가 아니라 Query Error Prediction → '
                  'Historical Error Matching → Best Correction Selection 문제로 재정의한다.')
        answer = 'YES'
    elif resid_vs_pair[0] > total // 2:
        case, title = 'C', 'Candidate residual이 핵심 정보'
        detail = 'Candidate의 historical error pattern이 correction utility를 결정한다.'
        answer = 'YES'
    elif ranker_vs_top1[0] > total // 2:
        case, title = 'B', 'Utility-aware Selection 채택'
        detail = ('binary useful/harmful 분류보다 relative utility selection objective가 '
                  '최종 forecasting과 더 잘 정렬된다.')
        answer = 'YES'
    elif top1_vs_soft[0] > total // 2:
        case, title = 'A', 'Aggregation이 주요 병목'
        detail = ('기존 classifier에도 ranking 정보가 있었고 averaging이 그것을 묻고 있었다.')
        answer = 'YES'
    else:
        case, title = 'F', 'Oracle과의 gap이 여전히 크다'
        detail = ('Candidate pool은 충분하지만 관측 가능한 past 정보만으로 exact utility '
                  'selection이 어렵다. 다음 후보는 uncertainty-aware selection / abstention.')
        answer = 'NO'

    lines += ['## 최종 판정', '', f'### Case {case} — {title}', '', detail, '',
              '### 근거 (이긴 setting 수)', '',
              f'- A: classifier Top-1 > soft filter — **{top1_vs_soft[0]}/{top1_vs_soft[1]}**',
              f'- B: utility ranker > classifier Top-1 — **{ranker_vs_top1[0]}/{ranker_vs_top1[1]}**',
              f'- C: +candidate residual > past-pair only — **{resid_vs_pair[0]}/{resid_vs_pair[1]}**',
              f'- D: residual selector > 다른 학습 selector 전부 — **{selector_vs_all[0]}/{selector_vs_all[1]}**',
              f'- E: direct correction > residual selector — **{direct_vs_selector[0]}/{direct_vs_selector[1]}**',
              f'- 학습 selector 중 하나라도 current CARTS를 이김 — **{best_learned_vs_current[0]}/{best_learned_vs_current[1]}**',
              '',
              '## 최종 연구 질문에 대한 답', '',
              f'> CARTS의 다음 방향은 "여러 과거를 평균하는 RAF"가 아니라 '
              f'"Broad pool에서 하나의 historical correction을 Utility-aware하게 선택하는 RAF"여야 하는가?',
              '', f'### **{answer}**', '']

    (root / 'FINAL_REPORT.md').write_text('\n'.join(lines) + '\n')
    return case, title, answer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='./metrics/utility_candidate_selection')
    a = p.parse_args()
    Path(a.root).mkdir(parents=True, exist_ok=True)
    case, title, answer = build(a.root)
    print(Path(a.root, 'FINAL_REPORT.md').read_text())
    print(f'DECISION: Case {case} — {title}   ANSWER: {answer}')


if __name__ == '__main__':
    main()
