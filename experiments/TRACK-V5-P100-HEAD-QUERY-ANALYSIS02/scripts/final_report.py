#!/usr/bin/env python3
import sys,json,itertools
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent));import query_analysis as q
def main():
 rows=[];paired=[];mechanisms=[];text=['# P100 query별 head 기여 분석 — seed0','',
 '학습된 V5를 고정하고 기존 All5 λ를 모든 단일 head 및 leave-one-out에 그대로 적용했습니다. Head 제거 뒤 남은 head 확률을 평균하여 Top10을 다시 선택했습니다. 새로운 encoder/head/λ 학습은 없습니다.',
 '각 방법의 전체 test query를 동일 시작 인덱스·채널로 정렬했습니다. All5 retrieval cache 및 Stage2 MSE/MAE 재현을 검증했습니다. 입력 특성은 validation 입력만으로 채널별 삼분위 경계를 고정했습니다.',
 'CI는 query 시작 인덱스 기준 길이 2H의 연속 블록을 1,000회 재표집했습니다. 채널은 query 안에 묶었습니다. 단일 seed의 의존성 고려 기술 통계이며 독립 seed 유의성 검증이 아닙니다. 블록 사이 장기 의존성은 남을 수 있습니다. 특히 ETTh1 H720은 2개, Weather H720은 7개 블록뿐이므로 CI가 좁더라도 강한 추론 근거로 삼지 않습니다. CI는 점별 95% 구간이며 다중비교 보정은 하지 않았습니다. 다수 head·strata 중 사후 선택한 대비는 탐색적 결과입니다.','']
 for cell in q.p.ORDER:
  data={};interpret=[];text+=['## '+cell,'','| 방법 | All5 MSE | MAE | KL MSE | Base MSE | 가장 큰 제거 손실 head | 제거 ΔMSE |','|---|---:|---:|---:|---:|---|---:|']
  for method in q.p.METHODS:
   dest=q.R/'results'/cell/method;meta=q.read(dest/'COMPLETE.json');a=np.load(dest/'query_metrics.npz');data[method]=a
   assert meta['stage2_reproduction'] and meta['kl_stage2_reproduction']
   S=a['stage2'];d=S[:,:,6:,0]-S[:,:,0,None,0];means=d.mean((0,1));j=int(means.argmax());comp=q.read(dest/'COMPLEMENTARITY.json')
   text.append(f"| {method} | {meta['all5_mse']:.6f} | {meta['all5_mae']:.6f} | {meta['kl_mse']:.6f} | {meta['base_mse']:.6f} | {j+1} | {means[j]:+.6f} |")
   import csv
   headrows=list(csv.DictReader((dest/'HEAD_FIXED_LAMBDA.csv').open()));single=min([r for r in headrows if r['subset'].startswith('Head')],key=lambda r:float(r['mse']));loo=min([r for r in headrows if r['subset'].startswith('Without')],key=lambda r:float(r['mse']));qc=q.read(dest/'QUERY_AGGREGATE_COMPLEMENTARITY.json')
   hp=list(csv.DictReader((dest/'HEAD_SUCCESS_FAILURE.csv').open()))
   mechanisms.append(dict(cell=cell,method=method,mean_head_error_cosine=np.mean([float(r['error_cosine']) for r in hp]),mean_head_top10_overlap=np.mean([float(r['top10_overlap']) for r in hp]),input_cosine_vs_future_utility_correlation=comp['quality_mean'][-1],all5_beats_oracle_best_single_query_fraction=qc['all5_beats_best_single_fraction'],best_posthoc_loo=loo['subset'],best_posthoc_loo_mse=float(loo['mse'])))
   interpret.append(f"{method}: 가장 좋은 고정 단일 head는 {single['subset']} (MSE {float(single['mse']):.6f}), All5는 {meta['all5_mse']:.6f}입니다. Test 사후 최선 제거는 {loo['subset']} (ΔMSE {float(loo['delta_mse_vs_All5']):+.6f}, 블록 CI {loo['ci95']})입니다. Query마다 미래 오차로 최선 head를 고르는 진단 oracle보다 All5가 좋은 query는 {qc['all5_beats_best_single_fraction']:.1%}입니다. 이는 고정 단일 head 비교와 다른 기준입니다.")
   sr=list(csv.DictReader((dest/'INPUT_STRATA.csv').open()));contrasts=[]
   for feature in q.FEATURES:
    for h in range(1,6):
     low=next(r for r in sr if r['feature']==feature and r['removed_head']==str(h) and r['stratum']=='0');high=next(r for r in sr if r['feature']==feature and r['removed_head']==str(h) and r['stratum']=='2')
     if low['delta_mse'] and high['delta_mse']:contrasts.append((abs(float(high['delta_mse'])-float(low['delta_mse'])),feature,h,low,high))
   if contrasts:
    _,feature,h,lo,hi=max(contrasts,key=lambda z:z[0]);interpret.append(f"입력 strata의 탐색적 최대 대비: {feature}, Head{h} 제거 ΔMSE는 하위군 {float(lo['delta_mse']):+.6f} (CI {lo['ci95']}), 상위군 {float(hi['delta_mse']):+.6f} (CI {hi['ci95']})입니다. 경계는 validation에서 고정했지만 이 대비의 선택은 test 사후이므로 전문화 확증으로 쓰지 않습니다.")
   for h in range(5):
    ci=q.block_ci(d[:,:,h],a['starts'],2*int(cell.split('_')[1]));rows.append(dict(cell=cell,method=method,removed_head=h+1,delta_mse=ci['mean'],ci95=str(ci['ci95']),blocks=ci['blocks'],replacement=float(a['replacement'][:,:,h].mean()),all5_beats_best_single=comp['all5_beats_best_single_fraction']))
  text+=['','양의 제거 ΔMSE는 해당 head를 제거했을 때 오차가 증가했음을 뜻합니다. 음수는 제거로 개선됐음을 뜻합니다. Test에서 가장 좋은 head 선택은 진단이며 배포 규칙으로 사용하지 않았습니다.','']
  for paragraph in interpret:text.extend([paragraph,''])
  for m1,m2 in itertools.combinations(q.p.METHODS,2):
   a,b=data[m1],data[m2];assert np.array_equal(a['starts'],b['starts']) and np.array_equal(a['features'],b['features'])
   delta=b['stage2'][:,:,0,0]-a['stage2'][:,:,0,0];ci=q.block_ci(delta,a['starts'],2*int(cell.split('_')[1]))
   paired.append(dict(cell=cell,reference=m1,comparison=m2,delta_mse=ci['mean'],ci95=str(ci['ci95']),blocks=ci['blocks'],comparison_better_fraction=float((delta<0).mean())))
  coverage=q.read(q.R/'coverage'/cell/'COMPLETE.json');assert coverage['full_cache_maxdiff']<1e-5
  text+=['Full 후보 누락은 test에서 균등 간격 64개 query × 전체 채널을 세 P100 방법 간 동일하게 맞춰 측정했습니다. Full All5 예측도 기존 캐시에 재현했습니다. Full과 P100의 학습 checkpoint가 다르므로 누락률이나 성능 차이를 순수 후보 선별의 인과효과로 해석할 수 없습니다.','']
  import csv
  cov=list(csv.DictReader((q.R/'coverage'/cell/'COVERAGE_SUMMARY.csv').open()))
  text+=['| 방법 | Full oracle Top10 보존율 | Full mixture Top10 보존율 |','|---|---:|---:|']
  for r in cov:text.append(f"| {r['method']} | {float(r['full_oracle_retention']):.3%} | {float(r['full_mixture_retention']):.3%} |")
  text+=['']
 q.csvout(q.R/'reports/HEAD_REMOVAL_SUMMARY.csv',rows);q.csvout(q.R/'reports/MATCHED_PRESELECTOR_COMPARISON.csv',paired);q.csvout(q.R/'reports/MECHANISM_SUMMARY.csv',mechanisms)
 text+=['## 핵심 관찰','',
 'Weather H96에서는 SARAF All5가 가장 좋은 Stage2 MSE를 보였지만, 64개 matched query의 Full oracle Top10 보존율은 약 3.31%로 Baseline의 8.01%보다 낮았습니다. 이 표본에서 individual oracle 보존율 증가가 Stage2 개선의 필수 조건은 아니었습니다. 전체 test 및 다른 oracle 정의로 일반화하지 않습니다.',
 'Weather H720에서는 고정 λ로 Baseline의 Head1을 제거하면 MSE가 0.385526에서 0.369014로, SARAF의 Head5를 제거하면 0.370556에서 0.367736으로 낮아졌습니다. 모두 Base MSE 0.319430에는 못 미칩니다. 이는 test 사후 선택이고 7개 시간 블록에 의존하는 탐색적 결과입니다.',
 'Weather H720의 평균 head 간 오차 cosine은 세 방법에서 약 0.78~0.81로 양수였습니다. 후보 Top10 차이가 있어도 예측 오차 방향이 상당히 유사하므로, 후보 다양성을 곧바로 오차 상쇄로 해석할 수 없습니다.',
 '지원되는 해석은 head 기여가 query·방법·checkpoint에 따라 달라지고, All5가 항상 각 head 제거보다 좋은 것은 아니라는 것입니다. 안정적인 전문화나 입력만으로 최선 head를 선택할 수 있는지는 아직 확인하지 않았습니다.','']
 text+=['## 해석과 한계','',
 '성공은 같은 query·채널에서 고정 λ 단일 head 예측의 MSE가 Base보다 작은 경우로 정의했습니다. 성공/실패 교차비율, head 오차 cosine, Top10 중복률은 각 설정의 HEAD_SUCCESS_FAILURE.csv에 있습니다.',
 'All5가 최선의 단일 head보다 좋은 query·채널 비율과 악화량은 COMPLEMENTARITY.json에 있습니다. 채널 평균 후 query 단위로 비교한 비율은 QUERY_AGGREGATE_COMPLEMENTARITY.json에 별도로 있습니다. 이는 test future를 사용하는 진단이며 routing이나 후보 선별에 입력하지 않았습니다.',
 '추세·변동성·주기성·급변의 INPUT_STRATA.csv는 validation에서 정한 입력 특성 그룹별 제거 효과입니다. 그룹 간 차이는 상관관계이며 head의 인과적 의미를 증명하지 않습니다.',
 '대안 설명은 학습 checkpoint 차이, 후보 집합 난이도, 비선형 Top10 경계와 고정 λ의 적합도입니다. 다음의 작은 확인은 별도 validation에서 사전 정의한 head 선택 규칙을 고정한 후 새 test에서 검증하는 것입니다. 이번에는 실행하지 않았습니다.',
 'Head 번호는 각 checkpoint 내부 인덱스입니다. 다른 P100 방법의 같은 번호가 동일한 의미나 전문 역할을 학습했다고 가정하지 않습니다.',
 'GPU1에서 다른 분석과 시간이 겹칠 수 있으므로 실행 시간을 독립 속도 벤치마크로 비교하지 않습니다. 큰 query_metrics.npz와 full_top10_ids.json은 서버 로컬에만 보관하며 외부 업로드하지 않습니다.','',
 '## 산출물','',
 '- reports/HEAD_REMOVAL_SUMMARY.csv: 고정 λ head 제거 효과 및 블록 CI',
 '- reports/MATCHED_PRESELECTOR_COMPARISON.csv: 동일 query에서 P100 방법간 Stage2 차이',
 '- results/<cell>/<method>/: 전체 test head 분석, 입력 strata, 성공/실패, 독립 재현 및 provenance',
 '- coverage/<cell>/COVERAGE_SUMMARY.csv: Full oracle 및 각 head·mixture 후보 보존율',
 '- reports/UNIT_TESTS.json, SMOKE.json, CACHE_PREFLIGHT.csv: 검증 근거']
 (q.R/'reports/ANALYSIS_KO.md').write_text('\n'.join(text))
 q.dump(q.R/'reports/DELIVERABLE_CHECK.json',dict(status='PASS',settings=12,full_coverage_cells=4,stage2_reproduction_all=True,query_channel_alignment_all=True,lambda_retrained=False,new_training=False))
 print('FINAL REPORT PASS',flush=True)
if __name__=='__main__':main()

