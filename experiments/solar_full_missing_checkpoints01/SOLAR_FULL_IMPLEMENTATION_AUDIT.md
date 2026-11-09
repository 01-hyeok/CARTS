# Solar Full 후순위 8개 실행기

이 실행기는 현재 48개 실험, K별 Oracle 보완 12셀, P100 진단 12셀, Full 진단 12셀 및 진단 요약 PASS 뒤에만 GPU smoke와 학습을 시작한다. 앞선 Full 진단에서 Solar Full-trained 모델 8개는 MISSING으로 유지한다.

## 고정 범위

- Solar H96 → H192 → H336 → H720, 각 V0 → V5, seed0만 8개.
- 물리 GPU1 UUID `GPU-91418a73-202a-2a9b-57d5-2538b19e07ec`, 주요 작업 한 개.
- 전체 train candidate bank와 137채널. P100 축소나 key embedding gradient 차단 없음.
- L=H, batch32, Adam LR0.001, 고정 10epoch, 마지막 불완전 batch도 포함.
- 기존 유효 후보 mask, normalized delta-last value alignment 유지. 기존 mask는 offline train-memory non-overlap이며 strict causal mask라고 주장하지 않는다.
- 기존 valid Full 후보 기준 z-score future-MSE teacher, tau_t=0.1. Student tau_s=0.1.
- V0는 기존 `train_horizon_retrieval_expert01.kl_loss`의 log-softmax KL. V5는 global candidate-axis per-head softmax → mean → 기존 `kl_loss_from_prob`. 후보 chunk별 softmax 금지.
- MLP encoder H→256→128, GELU/dropout0.1. 기존 config checkpoint의 args만 이용하고 encoder 가중치는 seed0 공통 fresh initialization. Head RNG seed0 연동.
- Validation Mean-Mixture RetMSE 최소 epoch 선택. Test는 선택 이후에만 최종 평가. Stage2 및 추가 seed 없음.

## 실행 경로 및 검증

채널 묶음의 loss에 `chunk_channels/137`을 곱해 backward하고, query batch마다 optimizer를 한 번 갱신한다. Query와 Full candidate key 양쪽 encoder에 gradient가 흐른다. Candidate 미래 MSE만 streaming으로 계산한다. Full candidate 수를 줄이지 않는다.

CPU synthetic 검사에는 4개 horizon × V0/V5의 dropout-off forward/loss/gradient/update 대조, global normalization, RNG replay, full bank indexing이 포함된다. 실제 GPU 검사는 선행 단계 종료 후에만 수행한다. H720 V5 실제 batch32, 전체137채널을 먼저 시도하고 OOM이면 채널 묶음을 64/32/16/8로 줄인다. 선정된 묶음 크기를 모든8개에 공통 적용한다. 평가에서는 기존 channel별 canonical 경로를 유지한다.

Grouped dropout의 난수 호출 순서와 합산 순서는 기존 channel별 경로와 다를 수 있다. 동일한 입력·loss·학습량을 유지하지만 과거 Full 학습의 bitwise 재현을 주장하지 않는다. Torch2/TF32-off 실행 조건을 별도 provenance로 기록한다. 기존 체크포인트는 수정하지 않는다.

매 epoch 저장에는 encoder/head/optimizer, Python·NumPy·Torch·CUDA RNG, DataLoader generator state가 포함된다. 재개는 epoch 경계만 허용한다. 완료된8개 뒤 새 Solar Full 진단은 별도 `TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01-SOLAR-FULL-COMPLETION`에 기록한다.

실제 GPU smoke가 실패하면 해당 작업을 중단하고 `IMMEDIATE_ISSUE.json`에 원인을 남긴다. CPU 준비 상태를 실제 학습 완료로 표시하지 않는다. GPU smoke 이후의 wall time과 allocated/reserved peak만 실측값으로 보고한다.
