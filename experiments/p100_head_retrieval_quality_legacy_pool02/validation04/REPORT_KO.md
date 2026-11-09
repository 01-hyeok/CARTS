# Validation04 검증 및 적용 기록

확인 시각: 2026-10-30T07:37:45.852251+00:00. 상태: 검증 PASS, 이후 Solar 실행에 적용. 현재 학습이나 기존 결과는 다시 시작하거나 덮어쓰지 않았다.

| 검증 조건 | 전체 query × channel | 기존 초 | 새 경로 초 | 속도 | peak allocated MiB | peak reserved MiB |
|---|---:|---:|---:|---:|---:|---:|
| Solar_96 V5 | 5161 × 137 | 64.551121 | 18.332980 | 3.521× | 135.761 | 186.000 |
| Solar_720 V0 | 4537 × 137 | 55.782127 | 18.234756 | 3.059× | 247.938 | 378.000 |

두 조건 모두 전체 validation의 score 및 ordered Top-10 ID hash 불일치 0, RetMSE/AggMSE/AggMAE/Recall 차이 0이다. H720 V0는 메모리·정합성 검증용 기존 reference checkpoint이며 새 실험의 공식 성능으로 사용하지 않았다. 원시 증거: GRAPH_GATES.json, Solar_96_CANONICAL_TRACE.json, Solar_720_CANONICAL_TRACE.json. CPU 검증 23개 PASS: unit_tests.xml.

## 적용한 방법

137채널을 한 GEMM으로 합치는 기존 평가안은 near-tie 순위 변경으로 탈락했다. 이번 방법은 canonical B32/P100 행렬 shape를 유지한 candidate embedding cache와 CUDA Graph 묶음이다. 매 validation 호출마다 cache를 새로 생성하며 첫 실제 batch의 embedding이 동일하지 않은 채널은 기존 연산으로 fallback한다. 짧은 마지막 batch도 기존 경로다. 하나의 graph를 모든 채널에서 재사용하여 Python launch overhead를 줄인다. 임의 허용오차 확대나 후보 수 감소는 없다.

일반 epoch validation만 새 경로다. 상세 validation, test, 최종 체크포인트 선택 규칙(min validation RetMSE), P100 ID/order, batch32, 10epoch, 전체채널, teacher/loss는 보존한다. 이후 Solar H192/H336/H720의 V0/V1/V2/V5가 validation04/runtime_validation.py로 dispatch된다. 새 경로의 각 호출 통계는 해당 run/validation_acceleration/callXX.json에 기록한다.

## 안전한 인계와 시간 해석

기존 Solar96 V5 PID 3985442는 10epoch 저장 후 final test에서 2026-10-30T06:55:45.221661+00:00에 정지되었고, 2026-10-30T07:21:15.955526+00:00에 같은 프로세스로 재개되었다. pause 약 1530.734초는 해당 V5 elapsed_seconds에 포함된다. 따라서 이 elapsed를 순수 연산시간 또는 가속 벤치마크로 사용하면 안 된다. 원본 COMPLETE.json은 수정하지 않았다.

새 controller PID 3995563, 기존 worker를 adoption 후 다음 셀로 진행한다. 기존 35 COMPLETE 파일의 SHA는 ACTIVATED.json에 보존했다. GPU1 UUID: GPU-91418a73-202a-2a9b-57d5-2538b19e07ec. 벤치마크 중 학습 worker는 정지 상태여서 GPU 연산 경합은 없었지만 VRAM context는 유지됐다. 메모리 표는 validator 프로세스의 PyTorch peak이며 장치 전체 사용량이 아니다.
