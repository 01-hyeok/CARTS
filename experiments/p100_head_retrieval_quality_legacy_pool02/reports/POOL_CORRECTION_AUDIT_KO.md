# P100 pool 교정 감사

기존 batch128 생성 경로는 legacy P100과 ordered ID 및 일부 집합이 달랐다. 비교 실험 안에서 pool을 공유했다는 사실만으로 기존 candidate ID/순서 유지 요건을 만족하지 않는다.

E96 query-channel 행 기준: train 순서628/59143, 집합12; validation 순서255/19495, 집합2; test 순서216/19495, 집합0. 경계 후보 차이를 작다고 무시하거나 허용오차로 통과시키지 않았다.

기존4셀 E96/E720/W96/W720의 legacy 원본과 P100 preselection baseline은 train/val/test 총12개 파일의 전체 ordered ID 및 query_start가 정확히 일치한다. 이 원본들은 읽기 전용으로 유지하며 교정본은 해당 ID/순서를 정확히 복사한다. 기존 수치 결과를 새 결과로 복사하지 않는다.

신규8셀은 scripts/precompute_candidate_pool01.py::precompute_split을 직접 호출한다. Canonical DataLoader batch32, query_chunk_size256(실제 batch32), full [N,L,C] memory 입력 layout, 기존 normalization 및 stable TopK 순서를 유지한다. 저장된 모든 ID를 canonical 함수 출력과 전체 비교하고 full eligibility/중복 검사 및 hash를 기록한다. 독립적인 batch128 대체 경로는 사용하지 않는다.

이전 controller3866022만 정지했고 현재 child3868266의 정상 완료 후 해당 controller를 종료했다. E96 V0/V1/V2/V5 및 E192 V0/V1/V2 총7개 완료 결과와 모든 cache/checkpoint는 원래 경로에 보존하며 SUPERSEDED_NONCANONICAL_POOL로 공식144개에서 제외한다. 교정본은 새 결과 및 checkpoint 경로에서144개 모두 새로 학습한다.

이전의 빠른 처리 시간은 전체채널 후보 gather/전송을 채널별로 줄이고 학습에 불필요한 full-oracle 진단을 제거한 경로와 관련된다. 10epoch, 실제query수, batch수, P100 support, 양쪽encoder gradient를 줄였다는 증거는 없다. 따라서 속도를 학습량 축소나 연구 성능 개선의 근거로 주장하지 않는다.

기존 완료4개 또는7개를 legacy baseline의 exact reproduction으로 주장하지 않는다. 새 결과도 pool parity와 연산/gradient parity가 통과했다는 것과 과거 최종 성능 전체 재현은 구분한다.

현재 공식 실행은 legacy_pool02 코드/QUALITY01-LEGACYPOOL02 결과 경로만 사용한다. 원본 결과, 데이터, 공통 소스, 외부 작업은 수정/중단하지 않았으며 GitHub 업로드는 하지 않는다.
