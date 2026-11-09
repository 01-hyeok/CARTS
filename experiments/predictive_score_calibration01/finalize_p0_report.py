"""Summarize completed CPU replay without implying full P0 completion."""
import csv,json
from pathlib import Path
out=Path('/data/pjh_workspace/CARTS/results/TRACK-PREDICTIVE-SCORE-CALIBRATION01')
status=json.loads((out/'P0_CPU_COMPLETE.json').read_text())
rows=list(csv.DictReader((out/'P0_CANDIDATE_SWAP.csv').open()))
text=['# P0 후보 교체 CPU 분석 중간 보고서','',
      '완료된 원본 10조건의 val/test 저장 증거를 재검증했다. 신규 GPU 학습 및 fresh encoder 재평가는 수행하지 않았다.',
      '', '모든 timestamp/channel, 원본 NPZ hash, 후보 중복·교집합 MSE, S0 평균 및 swap DeltaRetMSE의 대수적 일치 검사를 통과했다.',
      '', '| Support | Cell | Method | Test DeltaRetMSE | Added MSE | Removed MSE | 교체 수/Q×C |',
      '|---|---|---|---:|---:|---:|---:|']
for r in rows:
    if r['split']=='test':
        text.append('| {support} | {cell} | {method} | {d:.8f} | {a:.6f} | {b:.6f} | {s:.3f} |'.format(**r,d=float(r['delta_retmse']),a=float(r['added_mean']),b=float(r['removed_mean']),s=float(r['swaps_per_query_channel'])))
text += ['', '음수 DeltaRetMSE는 개선이다. Added/Removed MSE는 실제 교체 후보를 분모로 집계하며, DeltaRetMSE는 모든 query×channel을 동일 가중한다. 교체 수가 0이면 후보 평균은 정의되지 않는다.',
         '', '원본 RetMSE 변화와 swap utility는 대수적으로 같은 값이므로 독립적인 성공 근거 두 개로 세지 않는다. Worst tail은 channel 평균 timestamp delta의 상위 1/5/10%이며, CSV에 전체 평균에 대한 기여와 tail 내부 평균을 구분해 저장했다.',
         '', '## 남은 Gate', '', 'Local ranking, head score calibration, Train-fixed regime 분석, fresh baseline ordered ID/mask 검증은 미완료다. P1 학습 gate는 닫혀 있다. 기존 Full 나머지 6조건은 원본 작업 완료 후 추가 분석해야 한다.',
         '', 'Test의 사후 설명은 탐색적이다. 후속 구조/feature 선택에는 Validation만 사용한다.',
         '', 'CPU wall time: %.2f seconds; GPU 사용 없음.'%status['seconds']]
with (out/'P0_SCORE_MECHANISM_REPORT.md').open('x') as f:f.write('\n'.join(text)+'\n')
print(json.dumps(status,indent=2))
