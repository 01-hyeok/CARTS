# Weather720 V0 검증 범위 정정 및 후속 복구

기존 ALL_CHANNEL03의 Weather720 전체 val/test PASS는 **V5에 한정**된다.
`validation_queue_v3.py`는 V5 전체 검증을 실행했고,
`validate_metric_existing_protocol.py:23`도 V5를 고정하여 4 query의 지표 산술만 검사했다.
V0 Torch2 FP32 전체 gate가 통과했다는 증거는 없으며 Weather720 V0 score cache도 없었다.
따라서 “기존 셀 전체 PASS가 TAIL05 후 FAIL로 바뀌었다”는 설명은 현재 증거보다 강하다.

확인된 수치 문제는 FP32 batched GEMM의 누적 차이가 near-tie 순위를 바꾸는 것이다.
동일 canonical query/key로도 bmm score 최대 차이 4.76837158203125e-7,
147 query-channel 중 ordered Top10 7개 및 후보 집합 2개의 차이가 저장되어 있다.
해당 canonical 연산 10회 반복은 bitwise 동일했다.
과거 TAIL05 실패 시점에는 147개 중 ordered mismatch 8개,
저장 예측 최대 차이 0.017908070236444473이었다. 이 값은 해당 실패 소스의 관측이며
아직 재검증하지 않은 최신 소스의 결과로 바꾸어 표기하지 않는다.

TAIL05는 짧은 batch의 score를 bmm에서 Lt로 바꾸고 일부 encoder 및 normalize shape를
수정했다. 기존 old03과 현재 tail05를 **동일 Torch2 V0 입력/weight/TF32/shape**로
대조해야 수정 전부터 있던 결함과 새 차이를 최종 구분할 수 있다.
`OLD03_TO_TAIL05_PACKED.diff`와 `PRIOR_GATE_SCOPE_AUDIT.json`에 정확한 소스 hash를 기록한다.

## 안전한 후속 실행

- Weather192→336의 정상 controller/worker는 수정하지 않는다.
- 이 폴더의 CPU waiter는 두 셀의 실제 산출물 gate와 원본 COMPLETE hash,
  기존 controller 종료, 공유 lock, GPU1 무점유를 모두 확인한 뒤에만 진단 GPU 작업을 시작한다.
- 진단은 old03/tail05 encoder·key 및 같은 canonical query/key에 대한 cuBLAS12
  strided/pointer batch·출력 전치 layout을 비교한다. 모두 전체 21채널·35448후보다.
- 별도 한 번의 packed GEMM 대조는 cross-channel 출력도 계산하고 동일 채널 부분만 추출한다.
  추가 메모리/시간을 기록하며, 정확하거나 빠르다고 미리 가정하지 않는다.
- 이 진단 PASS만으로 production을 열지 않는다. 최종 V0/V5 × val/test 전체 gate,
  올바른 runtime/TF32, ordered ID, 기존 예측 허용오차, 지표 및 source hash가 필요하다.
- 미래 정답에 맞춰 후보나 score를 교체하지 않는다. 허용오차 변경·serial fallback 없음.
- Solar8 선행 조건을 해제하지 않는다. 원본 생성 runtime UNKNOWN 표기는 유지한다.

## 미완료

GPU 추가 대조와 수치 경로 수정은 기존 worker가 끝나는 안전한 경계에서 진행한다.
Weather720 전체 재현과 자동 production 재개는 아직 미완료이며 차단을 유지한다.
