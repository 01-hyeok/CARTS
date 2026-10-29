# Weather_96: completed seed0 evidence

상태: VERIFIED. 신규 encoder/head 학습 없이 기존 Full V5의 head 확률을 다시 평균하고 Top10을 재선택했다. Primary는 구성별 global lambda 재적합, Secondary는 canonical All5 lambda 고정이다.

## Stage2 결과
| Configuration | primary_mse | primary_mae | secondary_mse | secondary_mae |
| --- | --- | --- | --- | --- |
| H1 | 0.173630 | 0.228022 | 0.175825 | 0.230902 |
| H2 | 0.169208 | 0.218334 | 0.172670 | 0.222980 |
| H3 | 0.179596 | 0.222939 | 0.198187 | 0.233563 |
| H4 | 0.176478 | 0.233570 | 0.180501 | 0.239626 |
| H5 | 0.181523 | 0.234608 | 0.189078 | 0.243389 |
| All-5 | 0.168176 | 0.215938 | 0.168176 | 0.215938 |
| Without H1 | 0.170431 | 0.218644 | 0.170842 | 0.219160 |
| Without H2 | 0.171895 | 0.219424 | 0.171559 | 0.219033 |
| Without H3 | 0.169236 | 0.218141 | 0.170062 | 0.219235 |
| Without H4 | 0.167752 | 0.217023 | 0.168388 | 0.217866 |
| Without H5 | 0.170090 | 0.224505 | 0.170418 | 0.224955 |

All5보다 test primary MSE가 낮은 구성: ['Without H4']. MAE가 낮은 구성: 없음. 이 비교는 동일 V5 내부 ablation이며 독립적으로 학습한 1-head 모델 비교와 다르다.

## 독립 학습 모델 및 Base
| model | test_mse | test_mae | status |
| --- | --- | --- | --- |
| Base | 0.16936443746089935 | 0.21181175112724304 | VERIFIED |
| V0 | 0.17026792466640472 | 0.21269848942756653 | VERIFIED |
| V1 | 0.16891944408416748 | 0.2158501297235489 | VERIFIED |
| V2 | 0.16871538758277893 | 0.21553124487400055 | VERIFIED |
| V5 | 0.16817615926265717 | 0.21593798696994781 | VERIFIED |

위 독립 모델 수치는 기존 원시 결과를 동일 Base/split forward로 재검증한 값이며 새 학습 결과가 아니다. V0→V1에는 query projection 도입 효과가 포함된다.

## Validation 선택과 Oracle
11개 Stage2 구성에서 validation MSE가 가장 낮은 구성은 Without H2이다. 해당 test MSE/MAE는 0.171895/0.219424. 31개 Stage1 subset의 validation AggMSE 선택은 1345, test AggMSE는 0.211324. Test future 기반 per-query-channel oracle AggMSE 0.094529는 배포 불가능한 진단이다. 서로 다른 선택 기준을 혼합하지 않는다.

## 발견·선택·예측 기여
평균 head Top10 overlap=0.000296, union size=49.971066. Head 예측 오류 cosine 평균=0.404549, 음수 비율=0.232392. 후보 다양성과 오류 상쇄는 별개의 근거다. Head union의 individual oracle Top10 coverage=0.037456; 최종 mixture Top10 중 head Top10 union 내부 비율=0.418718. Mixture는 전체 후보 확률을 평균하므로 head별 Top10의 단순 union에서만 고르지 않는다.

## 무결성과 측정 범위
31개 subset의 val/test, 11개 Stage2 및 validation epoch 선택을 검증했다. 원본 retriever/Base 해시는 변경되지 않았다. 전체 캐시 비교 상세는 newcache_stored_lambda_verification.json이다. 셀 wall=9128.730s, 분석 worker peak allocated/reserved=1211.806/1640.000MiB. 이 peak는 Stage2 자식과 P100을 제외하며 시간은 GPU1 경합 환경이다.

## 발표 결론과 스크립트
단일 seed에서 학습된 V5 내부 head의 조건부 기여를 측정했다. 독립 모델 우월성, 의미적 specialization, 다양한 후보가 개선의 원인이라는 인과 주장은 별도 근거가 필요하다.

이번에는 기존 Full V5를 고정한 채 head를 단독 사용하거나 하나씩 제거했습니다. 각 구성에서 Top10을 다시 고르고, lambda를 재적합한 결과와 고정한 결과를 분리했습니다. 표의 Stage2 MSE와 MAE가 실제 기여 판단의 기준입니다. Validation 선택과 test 사후 비교도 분리했습니다. Head overlap이 낮다는 사실만으로 오류 상쇄를 결론내리지 않고, 독립 학습 모델과의 비교를 함께 해석해야 합니다.
