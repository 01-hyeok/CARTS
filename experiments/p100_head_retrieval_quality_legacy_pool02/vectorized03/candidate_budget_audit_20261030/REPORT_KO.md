# 후보 수 및 입력 동일성 CPU 감사

확인 완료: 2026-10-30T06:05:44.140917+00:00

결론: 속도 개선은 후보 축소 때문이 아니다. 변경 전후 모두 전체 train bank에서 고정 P100을 사용하며, 학습 KL은 100개 전체를 소비한다. K10은 공식 retrieval 평가의 최종 선택 수이다.

|항목|기존|vectorized03|
|---|---|---|
|Solar H96 full bank|36,601 windows ×137 channels|동일|
|Train query별 valid bank 후보|36,218–36,409|동일 RAFT mask|
|Query/channel당 학습 후보|100|100|
|공식 최종 retrieval K|10|10|
|학습 batch|32 queries|32 queries|
|채널|137 순차 처리|137 동시 처리|
|10 epoch 계획 optimizer steps|11,440|11,440|

## 전체 캐시 ordered ID 검사

각 split의 전 행/전 채널을 실제 기존 CellData.pool과 새 pool_all로 읽고 torch.equal 및 ordered stream SHA256 비교. 전체 714,852,300개 ID 일치. 모든 행 후보100개 중복 없음과 temporal eligibility 재검사 PASS.

|Split|캐시 shape|ID 원소 수|기존/신규 ordered stream SHA256|
|---|---|---:|---|
|train|[36601, 137, 100]|501,433,700|52c69a4672d38df6230b30386927724e54136ad6d31587c958f6cd4916fcf607|
|val|[5161, 137, 100]|70,705,700|c15f5177ebdfda5033f779a488dbbcc4a6704ee8a2200aded4a0f94d5cf7a2da|
|test|[10417, 137, 100]|142,712,900|c5e1d2d29917f757ca240ae9878952a44003414e58459c8bcb27d55f8da05567|

세 원본 .npy SHA256도 이전/신규 run provenance와 완전히 일치. 실제 셔플 seed0 첫 batch에서 query/target/history/delta-future 모두 bitwise equal; 전체 tensor hash는 AUDIT.json 참조.

입력 변경은 [B32,P100,L96,1]을 채널별137번 처리하던 것을 [137×32×100,L96]=[438400,96]로 묶은 것이다. 같은 batch에서 encode하는 candidate instance 수는 양쪽 모두438,400개다. Candidate ID의 중복 재사용을 포함한 연산량이며 bank 고유 window 수와 다르다. Query encoder 대상도 양쪽4,384개다.

## Loss와 실험 예산

기존: (1/C) sum_c [(1/B) sum_b KL_teacher_student(b,c)]. 신규: (1/(C×B)) sum_(c,b) KL_teacher_student(b,c). Chunk fallback이면 C_chunk/C 가중치. KL 내부 후보축100개를 모두 합산한다. Optimizer update는 B32 batch당 한 번으로 동일하다.
기존 z-score future-MSE teacher, temperature, Adam lr=.001,10 epoch, dropout, seed0, validation Mean-Mixture RetMSE 선택 그대로. 실제 완료된 새 epoch마다36,601 train queries와5,161×137 validation 단위 확인; 저장된 공통 batch hash 일치.

## 적용 범위

남은 Solar H96 V2/V5, H192/H336/H720 V0/V1/V2/V5 모두 새 경로로 dispatch된다. 완료된 ETTh1/Weather32개와 Solar H96 V0/V1 2개는 재실행하지 않는다.
ETTh1/Weather에도 수학적으로 같은 채널 batching을 적용할 수 있으나 이번 실측 gate는 Solar용이다. 현재 runtime은 Solar만 허용하며 다른 데이터셋을 검증 없이 활성화하지 않는다. 적은 채널 수에서 같은 속도배율을 보장하지 않는다.

H192/H336/H720의 실제 B32×137×P100 smoke/메모리 검증은 통과했으나 해당 전체 공식 pool은 아직 생성 전이다. 후속 controller가 동일 canonical precompute 및 전체 ID/마스크 검증 후 사용한다. 해당 미생성 cache 전체 동일성을 이미 검증했다고 주장하지 않는다.

## 수치 동등성 한계

Dropout RNG 소비 순서/최종 state는 맞췄으나 FP32 GEMM 및 gradient 합산은 비트단위 동일하지 않다. 완전한 학습 궤적 동일성은 보장하지 않는다. 테스트 loss/gradient/update 차이는 기존 gate에 기록.
평가 벡터화는 near-tie에서 후보가 달라져 탈락했으므로 현재 runtime은 기존 평가를 그대로 사용한다. 후보 수 유지와 학습 후 선택 후보가 같다는 주장은 별개다.

## 코드 근거

- `runtime.py:154–159`: 기존 P100 로더; `vectorized03/vector_ops.py:35–39`: 동일 캐시와 행 인덱스, 채널축 transpose만 수행.
- `runtime.py:141–148`, `vectorized03/vector_ops.py:42–51`: history/future 동일 gather.
- `runtime.py:193–198`, `scripts/train_k_multislot_predictive_retrieval01.py:80–89`: 100후보 teacher/student KL 및 batch mean.
- `runtime.py:513–535`, `vectorized03/vector_ops.py:64–74`:10epoch, 한 batch당optimizer1회, 채널 평균가중치.
- `vectorized03/benchmark_vector.py:87–97`: 동일 x,y,ids를 기존/137/32chunk benchmark에 전달,ids shape(137,32,100) assert.
- `vectorized03/controller_vector.py:20–24`: Solar 모든 pending run dispatch.
- `vectorized03/runtime_vector.py:18–21`: 기존 evaluation 유지.

CPU-only 감사이며 GPU 초기화 없음. 원본/학습 실행/캐시/완료 결과 변경 없음. Git push/업로드 없음.
