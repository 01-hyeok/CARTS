# Solar 137채널 벡터화 적용 보고서

작성: 2026-10-30T05:48:18.631382+00:00

- 완료 34개는 원본 COMPLETE hash가 모두 동일함. Solar H96 V2의 epoch7 checkpoint 및 부분 결과 보존.
- 중단한 V2만 공통 seed0 초기화부터 새 경로에서 재시작. 기존 7epoch와 새 run을 이어붙이지 않음.
- 총 48개 seed0 Stage1만 실행. seed1/2 96개는 사용자 요청으로 취소. Stage2 없음.
- 물리 GPU1 단일 worker, UUID GPU-91418a73-202a-2a9b-57d5-2538b19e07ec.

## 실측 학습 배치 성능

B32 × 137채널 × P100, V5, FP32. 1회 warmup 후 3회 동기화된 forward/backward/Adam step의 중앙값. DataLoader, pool 생성, validation/test, 파일 저장 제외. 전체 run 속도 배율로 해석하지 않음.

|H|기존 채널순차 초/배치|137채널 초/배치|배율|137채널 allocated GiB|reserved GiB|32채널 초/배치|
|---|---:|---:|---:|---:|---:|---:|
|96|0.575119|0.052228|11.01x|4.245|6.164|0.061875|
|192|0.573624|0.054460|10.53x|4.854|6.625|0.062953|
|336|0.573429|0.059754|9.60x|6.275|6.969|0.069290|
|720|0.571820|0.076735|7.45x|10.075|10.414|0.085365|

네 horizon 모두 전체 137채널이 더 빠르고 OOM 없음. 학습은 137채널 사용. OOM 발생 시 optimizer update 전 같은 batch/RNG를 복원하고 큰 chunk로 재시도하며 기록.

## 검증과 수치 차이

- CPU/synthetic/queue 테스트 18 PASS.
- H96/H720 실제 137채널 ×32 query 배치에서 dropout off/on loss, gradient, Adam update 검사 PASS.
- Dropout은 기존 query→candidate→다음 채널 순서와 같은 shape의 mask로 RNG 소비. 최종 CUDA RNG state 동일.
- 단, FP32 batched GEMM 및 gradient 합산 순서는 비트 단위로 같지 않음. Adam 초기 update의 near-zero gradient 민감도 때문에 파라미터 최대 차이는 3.98e-5. 수학적으로 같은 objective 및 평균가중치지만 동일 학습 궤적이라고 주장하지 않음.
- 새 V2/V5와 기존 완료 V0/V1의 구현 버전 차이를 해석에 명시. 초기 encoder/head hash와 epoch별 batch 순서는 기존 공통값과 비교 강제.

|H|Dropout|Loss 절대차|Gradient 상대 L2|Gradient 최대절대차|Adam update 최대절대차|RNG 동일|
|---|---|---:|---:|---:|---:|---|
|96|False|3.307e-08|3.698e-07|2.235e-08|1.967e-05|True|
|96|True|8.223e-08|3.000e-07|2.235e-08|3.036e-06|True|
|720|False|1.671e-07|6.813e-07|9.499e-08|3.980e-05|True|
|720|True|2.349e-08|5.328e-07|7.823e-08|3.573e-05|True|

## 평가 경로 선택

실험적 vector_eval.py는 사용하지 않음. 거의 동률인 후보에서 batched FP32 score 차이로 선택 집합 1개/35620 구성-query 단위가 바뀌었고, smoke RetMSE 차이 1.77e-5로 gate 실패. 허용오차를 완화해 통과시키지 않음.
최종 runtime은 기존 channelwise 평가를 그대로 호출. V0와 V5의 각 65 query ×137채널, 마지막 batch1 포함, 모든 configuration/ID/metric 및 V0 Oracle/Random 기준선이 원래 경로와 정확히 일치. Validation 체크포인트 기준 및 test 평가는 유지.

## 재개와 보존

Controller PID: 3981216
Controller log: `/data/pjh_workspace/CARTS/results/TRACK-P100-HEAD-RETRIEVAL-QUALITY01-LEGACYPOOL02/logs/controller_vectorized03.log`
신규 결과: `results/TRACK-P100-HEAD-RETRIEVAL-QUALITY01-LEGACYPOOL02/runs_vectorized03/`
신규 checkpoint: `checkpoints/track_p100_head_retrieval_quality01_legacy_pool02_vectorized03/`
기존 부분 결과: `results/TRACK-P100-HEAD-RETRIEVAL-QUALITY01-LEGACYPOOL02/vectorized03_prechange/Solar_96_V2_seed0/`

## 근거

`BENCHMARK.json`, `VECTOR_GATES.json`, `unit_tests.xml`, `TRANSITION.json`, `PAUSED.json`, `ACTIVATED.json`, `benchmark.log`, `validation.log`. 실패 시도도 `benchmark_attempt1.log`, `validation_attempt1.log`, `validation_attempt2_vector_eval_rejected.log`에 보존.
원본 공통 코드/데이터/후보 cache/완료 checkpoint는 변경하지 않음. GitHub 업로드 없음.

## 재개 후 실제 첫 epoch 확인

Solar H96 V2: train 36,601 queries ×137채널, validation 5,161 ×137=707,057 단위 전체 완료. 기존 batch order hash 동일. 첫 epoch checkpoint 저장 후 epoch2 진행.

학습+validation epoch 시간: 기존 728.157초 → 신규 118.917초, 6.12배. 최종 detailed test 평가/전처리까지 포함한 전체 run 배율은 아직 미측정. 근거 `RESUMED_FIRST_EPOCH.json`.
