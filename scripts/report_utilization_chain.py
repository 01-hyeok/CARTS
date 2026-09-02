"""Retrieval quality -> aggregate quality -> utilisation -> forecast, in one table.

Each link is measured on its own terms. Recall grades the ranking against the
individual-Future-MSE Oracle. The aggregate error is what Stage-2 actually
receives -- the error of the single weighted mean it builds from the Top-10 --
and differs from the mean of the individual errors by the spread among those
candidates, which is why the two can move in opposite directions.

Utilisation is a counterfactual, not a gate reading. Under residual fusion
y_final = y_base + lambda*y_ret, so a neutral retrieval signal reproduces
y_base exactly; base_mse is therefore the retrieval-off error and no second
inference pass is needed. lambda is reported alongside but never as the primary
evidence: it is a gate weight, and a fusion without one would still have a
utilisation to measure.
"""

import argparse
import re
from pathlib import Path


def read(path, marker, keys):
    if not Path(path).exists():
        return None
    text = Path(path).read_text()
    match = re.search(rf'{marker}(.*)', text)
    if not match:
        return None
    body = match.group(1)
    out = {}
    for key in keys:
        found = re.search(rf'(?<![a-z_])\b{key}: ([-0-9.eE]+)', body)
        if found:
            out[key] = float(found.group(1))
    return out


def step0(path):
    """The aggregate metrics, printed as a block rather than one metric line."""
    if not Path(path).exists():
        return None
    out = {}
    for line in Path(path).read_text().splitlines():
        found = re.match(r'\[step0\] (\w+)\s*=\s*([-0-9.eE]+)', line.strip())
        if found:
            out[found.group(1)] = float(found.group(2))
    return out or None


def spearman(a, b):
    def rank(xs):
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        out = [0] * len(xs)
        for pos, idx in enumerate(order):
            out[idx] = pos
        return out
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    den = (sum((x - ma) ** 2 for x in ra) * sum((x - mb) ** 2 for x in rb)) ** 0.5
    return cov / den if den else float('nan')


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    den = (sum((x - ma) ** 2 for x in a) * sum((x - mb) ** 2 for x in b)) ** 0.5
    return cov / den if den else float('nan')


DATASETS = {
    'ETTh1': (['cosine_kl', 'cosine_wce', 'asymmetric_kl', 'asymmetric_wce',
               'pair2_kl', 'pair2_wce'],
              'logs/stage2_learned_score/ETTh1/pred{p}/{arm}_stage2.log',
              {'cosine_kl': 'logs/e2_loss/ETTh1/pred{p}/cos_kl.log',
               'cosine_wce': 'logs/e2_loss/ETTh1/pred{p}/cos_weighted_topk_ce.log',
               'asymmetric_kl': 'logs/e2_loss/ETTh1/pred{p}/asym_kl.log',
               'asymmetric_wce': 'logs/e2_loss/ETTh1/pred{p}/asym_weighted_topk_ce.log',
               'pair2_kl': 'logs/e2_loss/ETTh1/pred{p}/pair2_kl.log',
               'pair2_wce': 'logs/e2_loss/ETTh1/pred{p}/pair2_weighted_topk_ce.log'}),
    'weather': (['cosine_kl', 'cosine_wce', 'asymmetric_kl', 'asymmetric_wce'],
                'logs/weather_stage2/weather/pred{p}/{arm}_stage2.log',
                {a: 'logs/weather_stage1/pred{p}/' + a + '.log'
                 for a in ['cosine_kl', 'cosine_wce', 'asymmetric_kl', 'asymmetric_wce']}),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--diag_dir', default='logs/utilization_diag')
    args = parser.parse_args()

    rows = []
    for ds, (arms, s2fmt, s1fmt) in DATASETS.items():
        for pred in (96, 192, 336, 720):
            for arm in arms:
                s1 = read(s1fmt[arm].format(p=pred), r'Stage1 Test \|',
                          ['student_oracle_recall_at_10', 'student_ndcg_at_10'])
                s2 = read(s2fmt.format(p=pred, arm=arm), r'Stage2 Test \|',
                          ['final_mse', 'final_mae', 'base_mse', 'lambda_mean'])
                agg = step0(f'{args.diag_dir}/{ds}_pred{pred}_{arm}.log')
                if not (s1 and s2):
                    continue
                row = dict(ds=ds, pred=pred, arm=arm)
                row.update(s1); row.update(s2)
                if agg:
                    row.update(agg)
                row['gain'] = s2['base_mse'] - s2['final_mse']
                row['rel_gain'] = 100 * row['gain'] / s2['base_mse']
                rows.append(row)

    hdr = (f"{'ds':<8}{'pred':>5} {'arm':<16}{'R@10':>7}{'NDCG':>7}"
           f"{'indivMSE':>9}{'aggMSE':>8}{'unifAgg':>8}{'var':>7}"
           f"{'base(off)':>10}{'withRet':>9}{'Gain':>8}{'Gain%':>7}{'lambda':>7}")
    print(hdr); print('-' * len(hdr))
    for ds in DATASETS:
        for pred in (96, 192, 336, 720):
            for r in [x for x in rows if x['ds'] == ds and x['pred'] == pred]:
                f = lambda k, w=8, p=4: (f"{r[k]:{w}.{p}f}" if k in r else f"{'--':>{w}}")
                print(f"{ds:<8}{pred:>5} {r['arm']:<16}"
                      f"{f('student_oracle_recall_at_10',7)}{f('student_ndcg_at_10',7)}"
                      f"{f('weighted_individual_mse10',9)}{f('hard_aggregate_mse10',8)}"
                      f"{f('uniform_aggregate_mse10',8)}{f('weighted_candidate_var10',7)}"
                      f"{f('base_mse',10)}{f('final_mse',9)}{f('gain',8)}"
                      f"{r['rel_gain']:>6.2f}%{f('lambda_mean',7,3)}")
            print()

    print('\n=== horizon별 retrieval 사용량 (arm 평균) ===')
    print(f"{'ds':<8}{'pred':>5}{'Gain':>9}{'Gain%':>8}{'lambda':>8}{'aggMSE':>9}")
    for ds in DATASETS:
        for pred in (96, 192, 336, 720):
            g = [x for x in rows if x['ds'] == ds and x['pred'] == pred]
            if not g:
                continue
            m = lambda k: (sum(x[k] for x in g if k in x) / max(sum(k in x for x in g), 1)
                           if any(k in x for x in g) else float('nan'))
            print(f"{ds:<8}{pred:>5}{m('gain'):>9.4f}{m('rel_gain'):>7.2f}%"
                  f"{m('lambda_mean'):>8.3f}{m('hard_aggregate_mse10'):>9.4f}")
        print()

    print('=== 무엇이 Stage-2 MSE를 더 잘 설명하는가 (Spearman / Pearson) ===')
    print("Recall은 높을수록 좋으므로 음수가, MSE 계열은 낮을수록 좋으므로 양수가 '설명됨'")
    for scope in list(DATASETS) + ['pooled']:
        sel = rows if scope == 'pooled' else [x for x in rows if x['ds'] == scope]
        for name, key in (('Recall@10', 'student_oracle_recall_at_10'),
                          ('IndividualMSE@10', 'weighted_individual_mse10'),
                          ('HardAggregateMSE@10', 'hard_aggregate_mse10')):
            pts = [(x[key], x['final_mse']) for x in sel if key in x]
            if len(pts) < 4:
                continue
            a = [p[0] for p in pts]; b = [p[1] for p in pts]
            print(f"  {scope:<8} corr({name:<20}, Stage2MSE) "
                  f"rho={spearman(a, b):+.3f}  r={pearson(a, b):+.3f}   n={len(pts)}")
        pts = [(x['hard_aggregate_mse10'], x['gain']) for x in sel
               if 'hard_aggregate_mse10' in x]
        if len(pts) >= 4:
            a = [p[0] for p in pts]; b = [p[1] for p in pts]
            print(f"  {scope:<8} corr({'HardAggregateMSE@10':<20}, RetrievalGain) "
                  f"rho={spearman(a, b):+.3f}  r={pearson(a, b):+.3f}   n={len(pts)}")
        print()


if __name__ == '__main__':
    main()
