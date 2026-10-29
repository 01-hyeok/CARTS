# ETTh1_96: completed seed0 evidence

상태: VERIFIED. 신규 encoder/head 학습 없이 기존 Full V5의 head 확률을 다시 평균하고 Top10을 재선택했다. Primary는 구성별 global lambda 재적합, Secondary는 canonical All5 lambda 고정이다.

## Stage2 결과
| Configuration | primary_mse | primary_mae | secondary_mse | secondary_mae |
| --- | --- | --- | --- | --- |
| H1 | 0.381525 | 0.404236 | 0.386186 | 0.409392 |
| H2 | 0.385765 | 0.399773 | 0.413530 | 0.420695 |
| H3 | 0.385769 | 0.400146 | 0.408388 | 0.413943 |
| H4 | 0.380033 | 0.401002 | 0.386848 | 0.409159 |
| H5 | 0.377826 | 0.401293 | 0.380430 | 0.404484 |
| All-5 | 0.374725 | 0.399461 | 0.374725 | 0.399461 |
| Without H1 | 0.376614 | 0.401039 | 0.377302 | 0.401846 |
| Without H2 | 0.374961 | 0.400474 | 0.374950 | 0.400461 |
| Without H3 | 0.375495 | 0.400766 | 0.375629 | 0.400928 |
| Without H4 | 0.375550 | 0.399950 | 0.375778 | 0.400217 |
| Without H5 | 0.379105 | 0.401666 | 0.380224 | 0.402813 |

All5보다 test primary MSE가 낮은 구성: 없음. MAE가 낮은 구성: 없음. 이 비교는 동일 V5 내부 ablation이며 독립적으로 학습한 1-head 모델 비교와 다르다.

## 독립 학습 모델 및 Base
| model | test_mse | test_mae | status |
| --- | --- | --- | --- |
| Base | 0.392423152923584 | 0.397970974445343 | VERIFIED |
| V0 | 0.3736935257911682 | 0.3972400426864624 | VERIFIED |
| V1 | 0.37568992376327515 | 0.39914971590042114 | VERIFIED |
| V2 | 0.37449896335601807 | 0.3985111117362976 | VERIFIED |
| V5 | 0.3747245967388153 | 0.3994608223438263 | VERIFIED |

위 독립 모델 수치는 기존 원시 결과를 동일 Base/split forward로 재검증한 값이며 새 학습 결과가 아니다. V0→V1에는 query projection 도입 효과가 포함된다.

## Validation 선택과 Oracle
11개 Stage2 구성에서 validation MSE가 가장 낮은 구성은 H5이다. 해당 test MSE/MAE는 0.377826/0.401293. 31개 Stage1 subset의 validation AggMSE 선택은 245, test AggMSE는 0.433198. Test future 기반 per-query-channel oracle AggMSE 0.313774는 배포 불가능한 진단이다. 서로 다른 선택 기준을 혼합하지 않는다.

## 발견·선택·예측 기여
평균 head Top10 overlap=0.056906, union size=45.203591. Head 예측 오류 cosine 평균=0.778709, 음수 비율=0.029259. 후보 다양성과 오류 상쇄는 별개의 근거다. Head union의 individual oracle Top10 coverage=0.143488; 최종 mixture Top10 중 head Top10 union 내부 비율=0.821277. Mixture는 전체 후보 확률을 평균하므로 head별 Top10의 단순 union에서만 고르지 않는다.

## 무결성과 측정 범위
31개 subset의 val/test, 11개 Stage2 및 validation epoch 선택을 검증했다. 원본 retriever/Base 해시는 변경되지 않았다. 전체 캐시 비교 상세는 newcache_stored_lambda_verification.json이다. 셀 wall=801.542s, 분석 worker peak allocated/reserved=231.681/292.000MiB. 이 peak는 Stage2 자식과 P100을 제외하며 시간은 GPU1 경합 환경이다.

## 발표 결론과 스크립트
단일 seed에서 학습된 V5 내부 head의 조건부 기여를 측정했다. 독립 모델 우월성, 의미적 specialization, 다양한 후보가 개선의 원인이라는 인과 주장은 별도 근거가 필요하다.

이번에는 기존 Full V5를 고정한 채 head를 단독 사용하거나 하나씩 제거했습니다. 각 구성에서 Top10을 다시 고르고, lambda를 재적합한 결과와 고정한 결과를 분리했습니다. 표의 Stage2 MSE와 MAE가 실제 기여 판단의 기준입니다. Validation 선택과 test 사후 비교도 분리했습니다. Head overlap이 낮다는 사실만으로 오류 상쇄를 결론내리지 않고, 독립 학습 모델과의 비교를 함께 해석해야 합니다.
