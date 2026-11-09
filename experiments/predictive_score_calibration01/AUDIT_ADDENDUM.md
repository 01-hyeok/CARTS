# 식별자 및 해석 보완

두 원문의 P4-A/B 명칭 충돌을 방지하기 위한 일관된 ID:

| ID | 구조 | 초기화 비교 기준 |
|---|---|---|
| P4-GR | global residual의 query beta | residual=0일 때 V5 |
| P4-HR | head별 residual, uniform mixture | 모든 alpha=0일 때 V5 |
| P4-FW | head별 residual, learned fixed mixture weight | HR uniform |
| P4-QG | head별 residual, query mixture gate | HR uniform; V5라고 부르지 않음 |
| P4-FT | frozen encoder, projection-head fine-tuning | 동일 원본 V5 checkpoint |

P3도 cosine-only, cosine+relation, cosine+shuffled relation을 같은 input dimension/MLP/optimizer budget으로 맞춘 뒤 incremental contribution을 주장한다. Spectral-only scorer와 cosine-only scorer의 성능차만으로 특징의 독립 기여를 단정하지 않는다.

P4-FT relation teacher의 미래 특징은 기존 Train-standardized delta-last value/reconstruction 정의를 명시한 뒤 계산해야 한다. Shape는 기존 복원 candidate MSE와 수치 일치해야 한다. Dynamics, magnitude spectrum, drift 등 offset 불변성은 항목별 검증하며 spectral DC 성분은 offset 불변이 아니므로 공간을 임의로 섞지 않는다. 현재 P4 teacher 구현 및 실데이터 누출 검증은 미완료이며 이 문서는 완료 증거가 아니다.

기존 Test를 보고 연구 방향을 선택한 점을 명시하고, 같은 Test에서 후속 수치는 탐색적 결과로 보고한다. Seed0 또는 시간 block bootstrap이 새로운 독립 test/seed 증거를 대신하지 않는다.

현재 CPU 테스트는 synthetic formula test이며 실제 모델의 ordered candidate ID, loss wiring, 첫 optimizer update, 전체 Train donor shuffle 및 feature/mask 경로까지 통과한 것으로 해석하지 않는다. 새 P1 실행은 해당 integration gate와 전체 P0를 완료한 뒤에만 가능하다.
