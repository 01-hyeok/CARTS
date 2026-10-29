# 5c59b1a + 검증된 Full/P100 결과 통합

Snapshot UTC: 2026-10-29T02:02:56.748078+00:00
Upstream: 5c59b1a33cb080698269ced3499fb2e935e13083. 공개 저장소 01-hyeok/CARTS, origin/main 기반 별도 worktree/branch. 원본 checkout 및 GPU 작업 변경 없음.

## 관찰과 범위
- Full: seed0, 4셀, 44 Stage2 구성이 기존 원시 metric과 검증 기록에 일치한다.
- P100: 20/24개만 완료+독립 Stage2 검증을 통과한 snapshot이다. 미완료 4개를 성능표에 채우지 않는다. 상태 파일은 live status가 아니다.
- 00:11 주간 발표 자료의 18/24는 당시 snapshot으로 보존했다. 최신 수치는 이 디렉터리 P100_VERIFIED_SNAPSHOT.csv가 우선한다.
- 새 커밋은 Phase G 24행 및 Phase H 결론과 분석 스크립트를 추가한다. Top-K v2 구현/학습은 이전 커밋들에 이미 존재하며, 이번 통합은 구현 기본값이나 실험을 바꾸지 않는다.

## 새 근거: v2와 v1

|셀|Arm|v1 MSE / MAE|v2 MSE / MAE|MSE 개선율|
|---|---|---|---|---|
|ETTh1_96|S2|0.371837 / 0.396790|0.371040 / 0.396262|0.214470%|
|ETTh1_96|D2|0.375010 / 0.398641|0.373257 / 0.397473|0.467661%|
|ETTh1_720|S2|0.514618 / 0.508759|0.512595 / 0.508127|0.393242%|
|ETTh1_720|D2|0.506632 / 0.503890|0.503495 / 0.502264|0.619209%|
|Weather_96|S2|0.168375 / 0.212773|0.167749 / 0.212921|0.371964%|
|Weather_96|D2|0.168030 / 0.213573|0.168352 / 0.213249|-0.191650%|
|Weather_720|S2|0.359719 / 0.365258|0.358064 / 0.365465|0.460217%|
|Weather_720|D2|0.354681 / 0.363751|0.350882 / 0.364083|1.071219%|

개선율 분모는 같은 셀/arm v1 MSE. 7/8에서 Stage2 MSE 개선, Weather96 D2는 악화. MAE는 5/8 개선이며 Weather96 S2와 Weather720 S2/D2에서는 악화된다. MSE 개선을 MSE/MAE 동시 개선으로 표현하지 않는다. seed0 관찰이며 통계적 우월성이나 원인을 확정하지 않는다.

## 원문 해석의 정정·한계
- Phase F/H의 "ETTh1 H720에서는 여전히 D0가 최선"은 표와 원시 JSON에 모순된다. D2-v2=0.503495 < D0=0.505956이므로 v2 포함 시 최선 retrieval arm이 바뀐다. 원문은 보존하고 이 정정을 우선한다.
- Phase G 원시값 집계: head cosine 증가 5/8, overlap 증가 6/8. 두 지표가 모두 6/8 증가한다고 표현하면 부정확하다. ETTh1 96 S2 cosine은 소폭 감소하고 overlap은 증가한다.
- D(개별 후보 오차/K) 개선과 성능 방향의 일치는 candidate relevance 설명을 지지하지만, D는 head별 전문화의 직접 지표가 아니다. "head specialization이 입증됐다"는 결론은 아직 NOT ESTABLISHED.
- v2의 수렴 구현 정합성과 성능 개선은 별개 주장이다. Phase D의 16개 테스트 통과는 upstream 기록이며 이번에 GPU 학습/재평가를 하지 않았다. 원문 v1 K번째 sigmoid=0.5는 정규화 전 값이고, 최종 K번째 가중치는 strict gap 한계에서 1/(2K-1)이다(나머지 K-1개는 2/(2K-1)).
- Phase F는 구성마다 lambda를 재학습한다. 고정-lambda 대조가 없으므로 fusion 기여를 배제할 수 없다. 우리 Full Weather96 H4 제거 효과가 고정 lambda에서 사라진다는 결과가 이 구분의 중요성을 보여준다. 다만 Full과 이 P100 loss track은 서로 다른 실험이다.
- Shared/Separate의 encoder 수/용량 차이, seed0, validation/test 기전 점검 2/8만 수행했다는 한계는 유지된다.

## 우리 결과와 연결
- Full 내부 ablation은 기존 Full V5 checkpoint에서 head만 제거한 결과이고, 새 Top-K track은 P100 기반 S/D loss 비교다. 둘을 동일한 head-count scaling 실험으로 합치지 않는다.
- Full ETTh1 H96 All5는 내부 11구성에서 두 지표 최선. ETTh1 H720은 validation으로 선택한 H3 제거가 All5보다 좋다. Weather720은 Base가 모든 Full head 구성보다 좋다. 전체 결론 MIXED / INCONCLUSIVE 유지.
- P100은 preselector만 바꾸는 KL/V5 비교다. 현재 WCE reference checkpoint에서는 args만 상속하고 weight/optimizer warm start는 하지 않는다. train_v_sharedtop100_01.py:272는 res[kl]을 역전파한다. inherited stage1_loss_mode=wce_soft_set_mse는 직접 encoder 경로에서 사용되지 않는다. 따라서 새 Top-K v2를 현재 P100 결과의 loss라고 표기하지 않는다.
- Weather Base는 L=H 채널별 linear. 원본 inference 재현 PASS지만 train -9999/scaler 영향은 별도 한계다. 채널 제외 수치는 진단이며 공식 성능으로 교체하지 않는다. 공식 CSV와 MPI 원본 수치/시간 일치는 확인했고 -9999 공식 코드 정의는 미확인이다.

## 현재 P100 검증 snapshot
|셀|방법|모델|MSE|MAE|
|---|---|---|---|---|
|ETTh1_96|baseline_cosine|V0|0.369243|0.394206|
|ETTh1_96|baseline_cosine|V5|0.369597|0.395469|
|ETTh1_96|paa16_cosine|V0|0.370967|0.394461|
|ETTh1_96|paa16_cosine|V5|0.370898|0.395283|
|ETTh1_96|saraf_time_aligned|V0|0.371283|0.396044|
|ETTh1_96|saraf_time_aligned|V5|0.372044|0.398562|
|ETTh1_720|baseline_cosine|V0|0.521939|0.513005|
|ETTh1_720|baseline_cosine|V5|0.514917|0.508438|
|ETTh1_720|paa16_cosine|V0|0.536487|0.525711|
|ETTh1_720|paa16_cosine|V5|0.527924|0.520175|
|ETTh1_720|saraf_time_aligned|V0|0.526369|0.517363|
|ETTh1_720|saraf_time_aligned|V5|0.509770|0.507301|
|Weather_96|baseline_cosine|V0|0.174460|0.214232|
|Weather_96|baseline_cosine|V5|0.176649|0.213072|
|Weather_96|paa16_cosine|V0|0.171986|0.213534|
|Weather_96|paa16_cosine|V5|0.174263|0.212299|
|Weather_96|saraf_time_aligned|V0|0.169536|0.209730|
|Weather_96|saraf_time_aligned|V5|0.168068|0.208819|
|Weather_720|baseline_cosine|V0|0.372792|0.370115|
|Weather_720|baseline_cosine|V5|0.385526|0.368637|

## 미완료
- {'cell': 'Weather_720', 'method': 'paa16_cosine', 'model': 'V0', 'status': 'RUNNING_AT_SNAPSHOT'}
- {'cell': 'Weather_720', 'method': 'paa16_cosine', 'model': 'V5', 'status': 'NOT_RUN_AT_SNAPSHOT'}
- {'cell': 'Weather_720', 'method': 'saraf_time_aligned', 'model': 'V0', 'status': 'NOT_RUN_AT_SNAPSHOT'}
- {'cell': 'Weather_720', 'method': 'saraf_time_aligned', 'model': 'V5', 'status': 'NOT_RUN_AT_SNAPSHOT'}

## 다음 최소 검증 (실행 안 함)
P100 잔여 4개 완료 후 동일 기준으로 snapshot 갱신. 추가 기전 실험이 승인되면 동일 v1/v2 cache에 validation에서 사전 선택한 공통 lambda를 적용하는 대조가 fusion 기여를 분리하는 작은 다음 단계다. per-head loss 신규 학습/seed 확대는 이번 통합 범위가 아니다.

## 포함·제외
PUBLICATION_ALLOWLIST.txt에 명시한 작은 코드/테스트/CSV/JSON/Markdown만 포함. Checkpoint, cache, dataset, 원시 예측 배열, 로그, SSH키/인증정보, unrelated track 결과는 제외. SOURCE_SNAPSHOT.json에 원본 SHA256 보존. 개별 provenance의 원본 실행 SHA는 과거 실행을 가리키며 이 통합 commit에서 새로 실험했다는 뜻이 아니다.
