# 이번 주 CARTS 진행사항 — 검증 결과 정리

확인 시각: 2026-10-29T00:11:22.049062+00:00

Full 4/4셀, Stage2 44/44구성 VERIFIED. P100 18/24구성 VERIFIED. seed0만 수행. 확인 시점 Weather720 Baseline KL 학습 진행 중이며 나머지 5설정은 미실행.

모든 성능 표는 Test MSE / MAE, 소수점 6자리. 굵은 글씨는 명시한 비교 범위에서 MSE와 MAE 각각의 최소값이며 test 관찰 순위다. 모델/구성 선택은 별도 validation 프로토콜을 따른다. 개선율 = 100 × (기준 지표 − 비교 지표) / 기준 지표; 양수는 개선, 음수는 악화. 서로 다른 후보 지원과 학습 모델을 인과적 단일 요인 비교로 섞지 않는다.

## 1. Full 독립 학습 모델 및 Base

Base는 L=H인 채널별 선형 모델: B_c=W_c(x_c−x_last,c)+b_c+x_last,c. V0→V1은 query projection 도입 효과를 포함한다. V1/V2/V5는 독립 학습 모델이며 V5 내부 head 제거와 다르다.

| 모델 | ETTh1_96 | ETTh1_720 | Weather_96 | Weather_720 |
| --- | --- | --- | --- | --- |
| Base | 0.392423 / 0.397971 | 0.560354 / 0.526977 | 0.169364 / **0.211812** | **0.319430** / **0.343818** |
| V0 | **0.373694** / **0.397240** | 0.529133 / 0.515935 | 0.170268 / 0.212698 | 0.369832 / 0.357739 |
| V1 | 0.375690 / 0.399150 | 0.506833 / 0.507053 | 0.168919 / 0.215850 | 0.336784 / 0.348383 |
| V2 | 0.374499 / 0.398511 | 0.507217 / 0.508130 | 0.168715 / 0.215531 | 0.338317 / 0.348751 |
| V5 | 0.374725 / 0.399461 | **0.503684** / **0.505833** | **0.168176** / 0.215938 | 0.342292 / 0.347555 |

한 줄 결론: ETTh1 H720에서 독립 V5가 가장 좋지만, ETTh1 H96은 V0가 우세하고 Weather H720은 Base가 우세하다. Weather H96은 V5의 MSE와 Base의 MAE가 각각 최선이다.

## 2. V5 내부 11개 구성 — Primary Stage2

기존 V5 encoder/head는 고정, subset별 확률 평균 후 Top10 재선택. 동일 Base에 구성별 global lambda만 train에서 학습하고 validation MSE로 epoch 선택. 아래 굵은 값은 각 셀 11구성 안의 test 최저값이며 배포 선택이 아니다.

| 구성 | ETTh1_96 | ETTh1_720 | Weather_96 | Weather_720 |
| --- | --- | --- | --- | --- |
| All-5 | **0.374725** / **0.399461** | 0.503684 / 0.505833 | 0.168176 / **0.215938** | 0.342292 / **0.347555** |
| H1 | 0.381525 / 0.404236 | 0.510812 / 0.510829 | 0.173630 / 0.228022 | 0.337779 / 0.351535 |
| H2 | 0.385765 / 0.399773 | 0.507495 / 0.506218 | 0.169208 / 0.218334 | 0.350254 / 0.356725 |
| H3 | 0.385769 / 0.400146 | 0.519291 / 0.519420 | 0.179596 / 0.222939 | **0.336509** / 0.348963 |
| H4 | 0.380033 / 0.401002 | 0.511649 / 0.514527 | 0.176478 / 0.233570 | 0.341977 / 0.350395 |
| H5 | 0.377826 / 0.401293 | **0.499040** / **0.503562** | 0.181523 / 0.234608 | 0.362763 / 0.371468 |
| -H1 | 0.376614 / 0.401039 | 0.503057 / 0.505789 | 0.170431 / 0.218644 | 0.343051 / 0.347574 |
| -H2 | 0.374961 / 0.400474 | 0.504198 / 0.507379 | 0.171895 / 0.219424 | 0.344662 / 0.349432 |
| -H3 | 0.375495 / 0.400766 | 0.500947 / 0.503654 | 0.169236 / 0.218141 | 0.345318 / 0.350065 |
| -H4 | 0.375550 / 0.399950 | 0.504567 / 0.506036 | **0.167752** / 0.217023 | 0.343529 / 0.348975 |
| -H5 | 0.379105 / 0.401666 | 0.505870 / 0.507356 | 0.170090 / 0.224505 | 0.339012 / 0.348052 |

한 줄 결론: All-5의 일관된 우월성은 성립하지 않는다. ETTh1 H96에서는 11구성 중 All-5가 두 지표 모두 최선이나, 다른 셀에서는 제거 효과와 MSE/MAE 방향이 다르다.

## 3. Validation 선택과 Test 사후 최선 분리

| 셀 | Val MSE 선택(11개) | 선택 후 Test MSE / MAE | Test MSE 사후 최선 | Test MAE 사후 최선 |
| --- | --- | --- | --- | --- |
| ETTh1_96 | H5 | 0.377826 / 0.401293 | All-5 | All-5 |
| ETTh1_720 | -H3 | 0.500947 / 0.503654 | H5 | H5 |
| Weather_96 | -H2 | 0.171895 / 0.219424 | -H4 | All-5 |
| Weather_720 | -H1 | 0.343051 / 0.347574 | H3 | All-5 |

한 줄 결론: Validation으로 선택한 구성은 ETTh1 H720에서만 All-5의 두 지표를 개선한다. Test 최선 head를 배포 선택으로 제시하지 않는다. 31개 Stage1 subset의 validation AggMSE 선택은 별도 CSV로 분리했고, Stage2 validation MSE 선택과 혼합하지 않는다.

## 4. 고정 lambda 핵심 ablation

Secondary는 셀별 canonical All-5 lambda를 모든 구성에 그대로 적용한다. Primary 재적합과 별개다. 구성은 전체 결과를 설명하는 사후 진단용이며 선택 실험의 승자로 주장하지 않는다.

| 셀 | 구성 | Primary MSE / MAE | Fixed-lambda MSE / MAE | 고정 lambda |
| --- | --- | --- | --- | --- |
| ETTh1_96 | All-5 | 0.374725 / 0.399461 | 0.374725 / 0.399461 | 0.511735 |
| ETTh1_96 | H5 | 0.377826 / 0.401293 | 0.380430 / 0.404484 | 0.511735 |
| ETTh1_720 | All-5 | 0.503684 / 0.505833 | 0.503684 / 0.505833 | 0.550962 |
| ETTh1_720 | -H3 | 0.500947 / 0.503654 | 0.500944 / 0.503651 | 0.550962 |
| Weather_96 | All-5 | 0.168176 / 0.215938 | 0.168176 / 0.215938 | 0.479427 |
| Weather_96 | -H4 | 0.167752 / 0.217023 | 0.168388 / 0.217866 | 0.479427 |
| Weather_720 | All-5 | 0.342292 / 0.347555 | 0.342292 / 0.347556 | 0.554234 |
| Weather_720 | H3 | 0.336509 / 0.348963 | 0.341623 / 0.351775 | 0.554234 |

한 줄 결론: Weather H96의 H4 제거 MSE 개선은 lambda 재적합에 의존하고, lambda를 고정하면 사라진다. Head 제거 자체 효과와 gate 재적합 효과를 분리해야 한다.

## 5. P100 — 완료된 18개 Stage2 결과

동일 P100, 동일 train 후보 메모리, 동일 모델별 loss/예산. Original KL=V0. V5는 per-head 확률의 Mean-Mixture KL이며 새로운 aggregate loss를 추가한 실험이 아니다. Stage1은 validation RetMSE, Stage2는 validation MSE 선택. SARAF-style은 time-aligned preselection만 도입했다.

| 셀 | Preselection | KL MSE / MAE | V5 MSE / MAE |
| --- | --- | --- | --- |
| ETTh1_96 | Baseline | **0.369243** / **0.394206** | 0.369597 / 0.395469 |
| ETTh1_96 | PAA-16 | 0.370967 / 0.394461 | 0.370898 / 0.395283 |
| ETTh1_96 | SARAF-style | 0.371283 / 0.396044 | 0.372044 / 0.398562 |
| ETTh1_720 | Baseline | 0.521939 / 0.513005 | 0.514917 / 0.508438 |
| ETTh1_720 | PAA-16 | 0.536487 / 0.525711 | 0.527924 / 0.520175 |
| ETTh1_720 | SARAF-style | 0.526369 / 0.517363 | **0.509770** / **0.507301** |
| Weather_96 | Baseline | 0.174460 / 0.214232 | 0.176649 / 0.213072 |
| Weather_96 | PAA-16 | 0.171986 / 0.213534 | 0.174263 / 0.212299 |
| Weather_96 | SARAF-style | 0.169536 / 0.209730 | **0.168068** / **0.208819** |

굵은 값은 셀별 여섯 설정 전체에서 각 지표의 test 최소값. Weather720 여섯 설정은 완료 수치가 없어 표에서 제외한다.

한 줄 결론: ETTh1 H96은 Baseline KL, ETTh1 H720과 Weather H96은 SARAF-style V5가 관찰상 가장 좋다. PAA-16 또는 SARAF-style의 효과는 데이터·모델에 따라 다르다.

### 동일 모델 Baseline 대비 개선율

| 셀 | 모델 | 방법 | MSE 개선율 | MAE 개선율 |
| --- | --- | --- | --- | --- |
| ETTh1_720 | V5 | SARAF-style | 0.999701% | 0.223477% |
| Weather_96 | V5 | SARAF-style | 4.857905% | 1.996084% |
| Weather_96 | V0 | SARAF-style | 2.822606% | 2.101576% |

분모는 같은 셀·같은 모델의 Baseline P100 지표다. 한 줄 결론: 개선율은 모델/데이터별 기준을 고정할 때만 비교 가능하며, seed0 결과만으로 통계적 우월성을 주장하지 않는다.

## 6. Head 역할 보조지표 (Full Test)

모든 지표는 query-channel 단위 평균. Overlap은 교집합 크기/10. Error cosine은 head의 집계 예측 오류 사이, D/C는 최종 선택 candidate 오류 분해로 단위가 다르다.

| 셀 | Top10 overlap | Head error cosine | Union size | Union oracle coverage | All5 AggMSE |
| --- | --- | --- | --- | --- | --- |
| ETTh1_96 | 0.056906 | 0.778709 | 45.203591 | 0.143488 | 0.423041 |
| ETTh1_720 | 0.246154 | 0.868874 | 33.706816 | 0.127593 | 0.591777 |
| Weather_96 | 0.000296 | 0.404549 | 49.971066 | 0.037456 | 0.201959 |
| Weather_720 | 0.011033 | 0.605609 | 48.941053 | 0.027310 | 0.429282 |

한 줄 결론: 후보 다양성과 head 간 양의 오류 상관이 공존한다. 낮은 overlap만으로 오류 상쇄, 의미적 전문화 또는 최종 예측 기여를 입증할 수 없다. Oracle coverage는 미래 정답을 사용하는 진단용이며 실제 선택에는 사용하지 않는다.

## 7. Weather Base 감사 — 공식 성능과 진단값 구분

| H | 평가 범위 | Base MSE / MAE | Full All5 MSE / MAE |
| --- | --- | --- | --- |
| 96 | 전체 21채널 | 0.169364 / 0.211812 | 0.168176 / 0.215938 |
| 96 | 3채널 제외 진단 | 0.194578 / 0.236784 | 0.193732 / 0.241268 |
| 720 | 전체 21채널 | 0.319430 / 0.343818 | 0.342292 / 0.347556 |
| 720 | 3채널 제외 진단 | 0.367567 / 0.382314 | 0.396858 / 0.393162 |

Base checkpoint+원본 test loader 새 inference와 float64 독립 집계로 수치 재현 PASS. MPI Jena 2020 공식 ZIP과 CSV 52,696행 수치·timestamp 전부 일치하며 OT=CO2(ppm). train -9999는 wv 1개, max.PAR 30개, OT 50개; 모든 test target에는 0개다. -9999의 공식 코드 정의는 미확인이다. train scale은 wv 52.099413, max.PAR 643.352692, OT 383.957098이며 해당 값 제외 시 각각 1.667195, 571.411863, 18.182134다.

한 줄 결론: 낮은 Weather Base 값은 현재 프로토콜에서 재현되지만 scaler 이상값 영향에 유의해야 한다. 3채널 제외는 평가 범위만 바꾼 진단으로 새 공식 성능 또는 정제 재학습 결과가 아니다. H720은 이 진단에서도 Base가 우세하다.

## 8. 발표 결론과 남은 검증

- Full: multi-head 효과는 셀별로 다르며 전체 판정 MIXED / INCONCLUSIVE. 추가 후보 발견 → Mean-Mixture 선택 → Stage2 기여가 항상 연결되지 않는다.
- P100: Weather720 여섯 설정 완료/검증 전에는 네 셀 전체 결론을 내리지 않는다. 확인시점 실행은 유지한다.
- seed0만 수행했으므로 불확실성과 seed 민감도는 미검증. 새 seed 실행은 이번 범위가 아니다.
- Full과 P100은 후보 지원 및 별도 학습 checkpoint가 달라 단순 수치 차이를 preselection만의 인과효과로 해석하지 않는다.
- RAFT mask는 train query의 자기/겹침 window를 제외하며 strict-causal train 검색과 다르다. Val/test 검색은 train 후보만 사용한다.
- Weather: -9999 공식 의미 및 정제 후 재학습 영향은 미확정. 데이터/scaler를 변경하지 않았고 새로운 성능으로 교체하지 않는다.
- H720 Full V0/V1/V2/V5 및 ablation은 L=720. P100도 L=720 공통 설정이며 진행 중 arm 완료 검증은 추후 필요.
- 시간/메모리 측정은 GPU1 경합 환경이다. 독립 속도 우월성으로 해석하지 않는다.

## 근거

- /data/pjh_workspace/CARTS/results/TRACK-V5-FULL-HEAD-ANALYSIS01/HEAD_STAGE2_MSE_MAE.csv
- /data/pjh_workspace/CARTS/results/TRACK-V5-FULL-HEAD-ANALYSIS01/CANONICAL_COMPARATORS_RAW.csv
- /data/pjh_workspace/CARTS/results/TRACK-V5-FULL-HEAD-ANALYSIS01/DELIVERABLE_AUDIT.json
- /data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/reports/verified_results.csv
- /data/pjh_workspace/CARTS/experiments/p100_preselection_comparison/reports/IMPLEMENTATION_AUDIT.md
- /data/pjh_workspace/CARTS/results/BASE-WEATHER-AUDIT01-20261028/FOLLOWUP_REPORT_KO.md
- /data/pjh_workspace/CARTS/results/BASE-WEATHER-AUDIT01-20261028/OFFICIAL_SOURCE_COMPARISON.json

각 P100 완료 run의 COMPLETE.json + stage2/metrics.json, Full 셀별 COMPLETION_VERIFICATION.json 및 구성별 stage2/metrics.json을 직접 대조했다. SOURCE_MANIFEST.json에 모든 읽은 원본의 SHA256/mtime을 저장했다. 원본 파일 변경, 학습, GPU 계산, 외부 업로드, GitHub commit/push 없음.
