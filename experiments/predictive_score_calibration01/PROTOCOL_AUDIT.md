# Predictive score calibration 사전 감사

2026-11-05. 두 첨부는 하나의 P0–P4 단계형 계획이다. 핵심 질문과 cosine-only, capacity/shuffle, shared-teacher 대조는 타당하다. 아래 수정과 실제 데이터 Gate를 통과하기 전 신규 GPU 학습은 실행하지 않는다.

## 확정한 보정

- P2→P3 및 이후 구조/조합/배정 선택은 Validation만 사용한다. Test는 고정된 비교의 최종 평가 및 탐색적 설명용이다. 기존 Test를 이미 본 P0 가설은 탐색적이며 새로운 독립 확증 증거로 표현하지 않는다.
- 유한 sigmoid gate는 정확히 0이 아니다. 새 residual 마지막 층을 0으로 초기화하면 최종 보정은 정확히 0이고 마지막 층은 학습 가능하다. 학습된 residual을 재사용한다면 별도 zero scalar를 곱한다. beta와 residual 마지막 층을 동시에 0으로 초기화해 gradient를 막지 않는다.
- 보정된 head의 균등 gate는 보정된 uniform 모델을 재현한다. 기존 V5 재현은 head residual까지 0인 초기 상태에서만 주장한다.
- 기존 구현은 `0.1*log(P_V5)`와 student temperature 0.1을 사용한다. 순위뿐 아니라 분포·KL·gradient를 검증한다. 무보정 순위는 원래 FP32 확률과 canonical stable tie-break를 그대로 사용한다.
- 자유 선형 출력층의 global beta는 출력층으로 흡수 가능하다. P1-C/D를 표현력 차이라고 주장하지 않는다. 일차 비교는 단일 residual MLP; 별도 beta 실험은 초기화·최적화 ablation으로만 표시한다.
- Full chunk마다 softmax를 하지 않는다. 전 후보에 걸친 global normalization과 teacher normalization을 유지한다. 계산량/메모리를 실제 측정하기 전 Full 실행 예산이나 소요시간을 단정하지 않는다.
- P2-A~E 입력 차원과 파라미터 수, 초기 가중치, batch 순서, optimizer step을 맞춘다. 무사용 feature는 Train 정규화 후 0. Shuffle donor는 고정 seed Train donor만 사용하고 val/test 미래를 참조하지 않는다. Teacher future와 student feature 경로를 분리한다.

## 실제 코드 확인

`learned_dynamics_cross01/prepare_features.py`는 과거 `d.h` 길이에서 AR 특징을 계산한다. `P100/ETTh1_720/IDENTITY.json`과 feature manifest는 seq_len=pred_len=720이다. H96 고정이라는 기존 보고서 설명은 맞지 않으며 이번 비교는 각 H의 실제 입력 길이를 따른다.

기존 scorer는 train-only FixedStandardizer, frozen V5, train FutureMSE teacher, delta-last reconstruction을 사용한다. Teacher는 valid-support zscore negative FutureMSE, temperature=.1이다. 원본 RAFT의 train non-overlap 정책은 오프라인 학습이며 strict-online 가용성으로 과장하지 않는다. 시계열 누출 검증은 candidate ID/mask 재현과 함께 진행해야 하며 synthetic test만으로 통과로 간주하지 않는다.

2026-11-05 07:31 UTC 서버 확인: P100 8조건 완료, Full ETTh1 H96/H720 2조건 완료. Weather H96 FULL worker PID 697332와 recovery controller PID 618544가 실행 중이다. 기존 실행·checkpoint·결과·source는 변경하지 않는다. recovery controller와 supplemental queue는 Solar를 실행하지 않는다.

## 이번 실행 범위

새 독립 경로에서 CPU P0를 시작한다. 완료된 10조건의 val/test 저장 batch SHA256, 모든 timestamp/channel 수, S0 RetMSE 및 D2/C2 candidate swap의 대수적 일치를 재검증한다. 이는 저장 증거 replay이며 fresh encoder inference의 대체가 아니다.

P0 local ranking, head score calibration, Train-fixed regime 분석과 fresh baseline ID/mask 검증은 아직 완료되지 않았다. P1 gate는 닫힌 상태로 남긴다. 원본 scorer scalar cache는 다섯 raw head cosine을 보존하지 않아 cosine-only 학습에 그대로 충분하지 않다. 기존 canonical B32 encoder precision을 유지한 head-score 추출이 추가로 필요하다.

Synthetic tests는 분포·KL·gradient, Full global chunk normalization, sigmoid residual 초기화, uniform head gate, beta 재파라미터화를 검증한다. 실제 학습 결과나 데이터 누출 통과 증거로 사용하지 않는다. P3/P4는 Validation 근거가 있을 때만 진행한다. Solar와 seed1/2는 시작하지 않는다.
