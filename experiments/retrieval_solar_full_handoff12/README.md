# Solar Full8 CPU 인계 복구

기존 `retrieval_solar_full_diagnostic11/gate.py`의 구형 waiter 확인은
`WAITING_ALL_PRIOR_DIAGNOSTICS`만 허용한다. 실제 구형 CPU waiter 4088382는
2026-10-30 18:10:07 UTC부터 `HELD_CPU_WAITER_FOR_DELIVERABLE_COMPLETENESS` 상태다.
hold 사유는 missed/false score·pooled rank·입력 strata 등 세부 산출물 누락과
H720 block-bootstrap CI 근거 부족이었다. 이 제한은 유지한다.

새 코드는 상태명만 허용하지 않는다. 기존 48개 COMPLETE, K별 Oracle gate,
P100/Full 각 12개 셀의 main/CPU/inference/deliverable gate와 연결된 해시,
고정된 전체 query/채널 수, 모델별 val/test 재현 및 checkpoint hash,
세부 CSV·rank histogram·probe train-only 공개, 불충분한 CI 제외,
최종 summary/final gate가 모두 통과해야만 구형 CPU waiter를 정리한다.
기존 hold 파일은 수정하지 않는다. 검사한 산출물 hash 목록을 별도로 남긴다.
Solar Full-trained V0/V5 8개는 선행 진단에서 MISSING으로 남는다.

현재 GPU worker 70152는 아무 신호도 받지 않는다. CPU parent 54557만
child 대기 상태에서 pause하고, 새 CPU waiter는 worker의 정상 종료와
Solar192 COMPLETE를 기다린다. 이후 기존 lock을 인계받아 원래 GPU 코드로
보완 및 다음 셀을 진행한다. 원본 11/04 코드와 checkpoint/결과는 변경하지 않는다.
모든 24셀 gate 이후에도 기존 Solar04의 독립 prerequisite와 GPU smoke는 유지된다.

회귀 테스트는 incomplete/stale hash/축소 count/누락 지표/부적합 CI/
잘못된 PID/active child/실행 중 worker/상태명만 있는 release를 거부한다.
추가 GPU 검증이나 학습은 이 CPU 복구 과정에서 실행하지 않는다.
