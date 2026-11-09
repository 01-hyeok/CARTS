# 논리 감사 결과

판정: 연구 질문은 타당하며 수정 후 실행 가능하다. 현재는 사용자 teacher 결정을 기다리는 준비 단계다.

## 사용자 결정이 필요한 항목

첨부의 `softmax(-future_MSE/tau_t)`와 기존 `softmax(-zscore(future_MSE)/tau_t)`는 다른 teacher다. 기존 구현 유지 목적에는 z-score를 권고했지만 선택은 미결이다. z-score는 유효 P100 전체에서 행별 평균과 분산을 계산한다. MSE가 작은 후보의 확률이 더 높다는 검사만으로 두 식의 차이를 검증할 수 없다. 거리 계산은 train scaler로 정규화된 데이터 공간이다.

## 확정적으로 보완할 구현

- 공식 checkpoint 선택/평가에는 Mean-Mixture를 사용한다. 기존 trainer의 Round-Robin validation을 그대로 공식값으로 쓰지 않는다.
- 기존 SlotHeads의 고정 `1000+h` RNG를 `1000+10000*seed+h`로 분리한다. seed0은 과거 초기값과 같고, 같은 seed에서 V1/V2/V5의 공통 head prefix도 같다. architecture와 forward는 기존 SlotHeads를 재사용한다.
- 공통 encoder는 같은 seed로 CPU에서 초기화하고 state를 복사한다. reference checkpoint는 args만 읽으며 다른 dataset/horizon의 가중치를 가져오지 않는다.
- loader 전용 generator를 사용하고 epoch별 query 순서와 batch 경계를 hash한다. model/head RNG가 배치 순서에 영향을 주지 않게 한다.
- scalar batch 평균은 표본 수로 가중한다. 기존 validation KL/entropy의 추가 batch-size 나눗셈 문제를 새 집계에 가져오지 않는다. RetMSE와 training objective 자체의 문제와 구분한다.
- 기존 RAFT는 train 비중첩 offline bank 규칙이며 strict-causal이 아니다. Val/test에서는 train 후보 미래 끝 <= query 예측 시작을 검증한다. 전체 split에서 valid candidate가100개보다 작으면 실패로 처리한다.
- Solar timestamp는 필요 없다. SARAF timestamp wrapper를 사용하지 않고 row index를 쓰는 기존 Solar loader를 사용한다.
- Solar에서는 불필요한 전체 채널 gather를 없애고 shared encoder 양쪽 gradient를 유지한다. 후보 chunk마다 softmax/z-score를 따로 계산하면 다른 objective가 된다. Dropout mask와 optimizer-step 수까지 검증해야 한다. 이 runtime 검증은 미실행이다.

## 해석 제한

RetMSE와 Oracle Regret은 공통 Oracle 항을 뺀 같은 비교 정보다. 독립 증거 두 개로 세지 않는다. Individual-MSE Oracle은 RetMSE 최적이며 uniform aggregate MSE/MAE의 이론적 하한은 아니다. Oracle aggregate가 Last-repeat보다 나쁘다고 P100 안의 모든 후보 집합이 나쁘다는 뜻은 아니다.

Last-repeat 실제 미래오차별 strata는 threshold를 validation에서 정해도 test 미래를 사용하는 비배포 난이도 진단이다. 입력-only strata와 분리하고 head/K/epoch 선택에 쓰지 않는다. Solar 상수 구간에서는 Oracle tie와 경계 gap, Last Skill의0분모를 기록한다. Last Skill은 평균 오차의 비율이며 query별 비율 평균이 아니다.

V1→V5 비교는 head 개수에 따른 전체 구조 효과를 검증한다. 추가 파라미터/최적화 경로를 완전히 분리한 순수 head 수 효과는 아니다. 독립 V1과 V5 내부 단일 head도 다르다. 3seed paired 차이와 시간 블록 불확실성을 분리하고 window/channel을 독립 반복으로 취급하지 않는다.

Weather의 기존 -9999/scaler 문제는 이번 실험 중 조용히 정제하지 않는다. 데이터 hash를 고정하고 원본 프로토콜의 한계로 보고한다. Stage1 실험으로 Stage2 개선을 주장하지 않는다.

## 코드 근거

- scripts/train_horizon_retrieval_expert01.py:88 — z-score teacher.
- scripts/train_k_multislot_predictive_retrieval01.py:67 — 기존 head 초기화.
- scripts/train_v_sharedtop100_01.py:191,273,288 — Round-Robin, 학습 및 평가 집계.
- scripts/train_margutil01.py:46 — configuration-only reference.
- utils/relation_memory.py:75 — RAFT eligibility.
- scripts/train_retriever_pool01.py:129 — 전체 채널 gather.
- models/RelationStage1.py:2910 — MLP/dropout.
- data_provider/data_loader.py:254 — Solar split/스케일러.

기존 연구 문서의 오래된 상태와 과거 잘못된 head cosine 주장을 최신 결과처럼 재사용하지 않는다. 원본 연구 문서는 변경하지 않는다.
