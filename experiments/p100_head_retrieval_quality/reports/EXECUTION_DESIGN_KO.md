# Stage1 단일 GPU 순차 실행 설계 (아직 실행 안 함)

## 순서와 소유권

seed0 전체48개 → seed1 전체48개 → seed2 전체48개. 각 seed에서 ETTh1 → Weather → Solar, 각 dataset에서 H96 → H192 → H336 → H720, 각 cell에서 V0 → V1 → V2 → V5 순서다. 물리 GPU1 UUID `GPU-91418a73-202a-2a9b-57d5-2538b19e07ec`만 노출하고 worker는1개만 허용한다. 다른 작업은 종료하지 않는다. Stage2 진입 경로를 만들지 않는다.

향후 controller는 lock, run 단위 상태/로그, 명령·코드·데이터·초기화·pool hash, 종료 코드를 기록해야 한다. COMPLETE는 선택 checkpoint의 공식 Mean-Mixture 평가와 예상 표본 수 검증 후에만 기록한다. 실패 시 부분 결과를 보존하고 다음 seed로 건너뛰지 않는다. 현재는 이 controller/학습 launcher가 구현되지 않았다.

## 공통 초기화와 학습 예산

동일 cell/seed의 encoder state를 CPU에서 한 번 만들어 네 arm에 복사하고 hash를 확인한다. Head는 기존 linear projection architecture를 유지하고 seed별 local generator만 바꾼다. 별도 loader generator로 모든 arm의 batch 순서를 맞춘다. 초기 hash와 epoch별 batch hash는 실제 실행 전 생성한다.

10epochs, logical batch32, Adam lr0.001, tau_s0.1, early stopping 없음. Teacher mode만 미결이며 tau_t0.1 외에 어떤 식도 암묵 적용하지 않는다. P100은 cell당 한 번 생성해 모든 seed/model이 공유한다. 각 run에서 query/candidate encoder gradient를 유지한다.

## Solar 및 메모리 gate

seq_len=H720에서 Solar full X/Y window bank의 float32 논리 크기는 약25.98GiB, 기존 B32/P100 전체137채널 gather는 약1.18GiB다. 이는 모델/activation/복사본을 제외한 크기다. 원본 시계열 또는 CPU/mmap bank를 사용하고 필요한 채널/후보만 GPU로 가져오는 경로를 우선 검토한다. 기존 `_ensure_memory`의 eager 전체 bank 할당도 함께 점검한다.

후보100개 전체의 score를 모아 동일 softmax/teacher 정규화를 적용한다. Chunk마다 별도 softmax, key detach, channel마다 optimizer.step, 마지막 작은 microbatch의 동일가중 평균을 금지한다. Logical batch와 channel 평균을 유지하고 optimizer는 원래 batch당1회만 갱신한다. Dropout0.1 때문에 평가 mode 동등성만으로 training gradient 동등성을 주장하지 않는다. RNG를 일치시킨 synthetic gradient/update 비교와 실제1batch가 다음 gate다.

## 평가와 저장

공식은 validation Mean-Mixture RetMSE@10 최소 epoch다. 선택 checkpoint의 scores를 한 번 계산해 head별·31subset·K1/5/10/20 평가에 재사용한다. Validation-selected best single은 RetMSE와 AggMSE 기준을 별도 표기한다. Test oracle은 비배포 진단이다.

Main table에는 Last-repeat, Random20draw, raw cosine, V0/V1/V2/V5, P100 Individual-MSE Oracle을 포함한다. Random draw 변동은3seed 변동과 별개다. Random generator의 상태/정해진 query-channel 순서를 chunk 사이에 보존해야 한다. 원시 query-channel 오차와 후보 ID는 압축·분할 저장하며 전체 P100 미래 tensor를 중복 저장하지 않는다.

12cell P100 int32 ID는 약12.20GiB, V5의31subset val/test Top10 ID를3seed 모두 저장하면 약33.56GiB다. Solar720의 모든 query×channel×P100×future 배열은 약1.78TiB이므로 생성/저장 금지다. 기존 checkpoint 크기에 기반한1440 epoch checkpoint와 지표/ID는 디스크 여유 안에서 관리 가능하지만 실행 전 reserve budget을 다시 확인한다.

과거 경합 환경의 Weather10epoch는 약44~93분/모델이었다. 채널 수에 단순 비례한144run 학습 추정은 약11~24일 규모이나 보장된 ETA가 아니다. 새 chunking, 경합, 선택/진단 비용의 영향을 실제 승인된 smoke/epoch 측정 후 갱신한다. 현재 시간 추정을 위해 실제 epoch를 실행하지 않았다.

## 남은 상태

- Teacher: PENDING_USER_DECISION.
- CPU synthetic: 별도 test report 참조.
- 데이터/참조: STATIC_PASS; actual loader smoke 미실행.
- 실제 초기화/gradient/memory/reproduction: NOT_RUN.
- P100 생성: NOT_RUN.
- Stage1 144runs: NOT_STARTED.
- Stage2 및 GitHub 업로드: 이번 범위에서 실행 안 함.
