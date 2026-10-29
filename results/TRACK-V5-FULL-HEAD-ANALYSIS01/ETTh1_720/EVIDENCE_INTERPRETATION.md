# ETTh1_720: completed seed0 evidence

상태: VERIFIED. 신규 encoder/head 학습 없이 기존 Full V5의 head 확률을 다시 평균하고 Top10을 재선택했다. Primary는 구성별 global lambda 재적합, Secondary는 canonical All5 lambda 고정이다.

## Stage2 결과
| Configuration | primary_mse | primary_mae | secondary_mse | secondary_mae |
| --- | --- | --- | --- | --- |
| H1 | 0.510812 | 0.510829 | 0.511073 | 0.511047 |
| H2 | 0.507495 | 0.506218 | 0.507624 | 0.506325 |
| H3 | 0.519291 | 0.519420 | 0.520253 | 0.520181 |
| H4 | 0.511649 | 0.514527 | 0.511890 | 0.514731 |
| H5 | 0.499040 | 0.503562 | 0.499096 | 0.503620 |
| All-5 | 0.503684 | 0.505833 | 0.503684 | 0.505833 |
| Without H1 | 0.503057 | 0.505789 | 0.503049 | 0.505781 |
| Without H2 | 0.504198 | 0.507379 | 0.504232 | 0.507412 |
| Without H3 | 0.500947 | 0.503654 | 0.500944 | 0.503651 |
| Without H4 | 0.504567 | 0.506036 | 0.504560 | 0.506030 |
| Without H5 | 0.505870 | 0.507356 | 0.505868 | 0.507355 |

All5보다 test primary MSE가 낮은 구성: ['H5', 'Without H1', 'Without H3']. MAE가 낮은 구성: ['H5', 'Without H1', 'Without H3']. 이 비교는 동일 V5 내부 ablation이며 독립적으로 학습한 1-head 모델 비교와 다르다.

## 독립 학습 모델 및 Base
| model | test_mse | test_mae | status |
| --- | --- | --- | --- |
| Base | 0.5603537559509277 | 0.5269773602485657 | VERIFIED |
| V0 | 0.5291333198547363 | 0.5159350037574768 | VERIFIED |
| V1 | 0.5068328976631165 | 0.5070529580116272 | VERIFIED |
| V2 | 0.5072166919708252 | 0.5081303119659424 | VERIFIED |
| V5 | 0.5036840438842773 | 0.5058333277702332 | VERIFIED |

위 독립 모델 수치는 기존 원시 결과를 동일 Base/split forward로 재검증한 값이며 새 학습 결과가 아니다. V0→V1에는 query projection 도입 효과가 포함된다.

## Validation 선택과 Oracle
11개 Stage2 구성에서 validation MSE가 가장 낮은 구성은 Without H3이다. 해당 test MSE/MAE는 0.500947/0.503654. 31개 Stage1 subset의 validation AggMSE 선택은 25, test AggMSE는 0.584165. Test future 기반 per-query-channel oracle AggMSE 0.494114는 배포 불가능한 진단이다. 서로 다른 선택 기준을 혼합하지 않는다.

## 발견·선택·예측 기여
평균 head Top10 overlap=0.246154, union size=33.706816. Head 예측 오류 cosine 평균=0.868874, 음수 비율=0.000271. 후보 다양성과 오류 상쇄는 별개의 근거다. Head union의 individual oracle Top10 coverage=0.127593; 최종 mixture Top10 중 head Top10 union 내부 비율=0.967720. Mixture는 전체 후보 확률을 평균하므로 head별 Top10의 단순 union에서만 고르지 않는다.

## 무결성과 측정 범위
31개 subset의 val/test, 11개 Stage2 및 validation epoch 선택을 검증했다. 원본 retriever/Base 해시는 변경되지 않았다. 전체 캐시 비교 상세는 newcache_stored_lambda_verification.json이다. 셀 wall=941.674s, 분석 worker peak allocated/reserved=1439.453/1906.000MiB. 이 peak는 Stage2 자식과 P100을 제외하며 시간은 GPU1 경합 환경이다.

## 발표 결론과 스크립트
단일 seed에서 학습된 V5 내부 head의 조건부 기여를 측정했다. 독립 모델 우월성, 의미적 specialization, 다양한 후보가 개선의 원인이라는 인과 주장은 별도 근거가 필요하다.

이번에는 기존 Full V5를 고정한 채 head를 단독 사용하거나 하나씩 제거했습니다. 각 구성에서 Top10을 다시 고르고, lambda를 재적합한 결과와 고정한 결과를 분리했습니다. 표의 Stage2 MSE와 MAE가 실제 기여 판단의 기준입니다. Validation 선택과 test 사후 비교도 분리했습니다. Head overlap이 낮다는 사실만으로 오류 상쇄를 결론내리지 않고, 독립 학습 모델과의 비교를 함께 해석해야 합니다.
