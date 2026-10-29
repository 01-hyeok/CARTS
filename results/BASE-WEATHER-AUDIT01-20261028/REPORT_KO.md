# Weather Base Forecaster 독립 감사

판정: 수치 재현 PASS. 데이터 정제·외부 benchmark 비교 타당성은 별도 우려가 있으며 PASS로 간주하지 않는다. 재학습, 원본 결과 수정, GitHub commit/push는 수행하지 않았다.

## 모델 정체
models/RelationStage2.py의 BaseForecastHead(per_channel_linear)이다. 각 채널별 Linear(L,H) 한 개이며 cross-channel mixing, attention, retrieval, moving-average decomposition이 없다. 마지막 관측값을 빼고 선형 변환 후 마지막 값을 더한다: B_c=W_c(x_c-x_last,c)+b_c+x_last,c. 구조상 NLinear 형태의 last-value centering이지만 프로젝트의 실제 클래스 이름으로 표기한다. H96은 21*(96*96+96)=195552개, H720은 21*(720*720+720)=10901520개 파라미터다. Features=M, 채널21, label_len0, seed0. 96/720은 시간 수가 아니라 10분 간격 sample 수로, test에서 16시간/5일 horizon이다.

학습 소스는 scripts/train_r_base_forecaster01.py: train MSE, Adam lr0.001, batch32, 최대10 epoch, patience5, validation MSE로 checkpoint 선택. 두 파일은 epoch7이며 원본 metadata와 일치한다. 전체 역사적 epoch 로그를 이번 감사에서 확보하지 못했으므로 epoch7의 전 epoch 대비 최적성은 코드·저장 metadata 수준 증거이며 재학습으로 재검증하지 않았다.

## 원본 test loader 재추론 및 독립 계산
| Cell | fresh NumPy float64 MSE | fresh MAE | 원본 MSE | 원본 MAE | Persistence MSE/MAE |
|---|---:|---:|---:|---:|---|
| Weather_96 | 0.169364456028 | 0.211811756846 | 0.169364455855 | 0.211811757450 | 0.259133/0.254220 |
| Weather_720 | 0.319429895267 | 0.343818117432 | 0.319429896190 | 0.343818118126 | 0.465187/0.393504 |

모든 test batch에서 원본 CSV→train-only scaler→window 입력/정답을 별도로 재구성하여 loader와 정확 일치를 확인했다. PyTorch 모델 output은 NumPy float64로 전체 원소 합/개수를 계산했다. 시작·중간·마지막 batch 표본은 checkpoint weight를 직접 읽은 NumPy 선형식과 비교했으며 최대 차이 H96 2.13e-7, H720 4.12e-7이다. 마지막 불완전 batch도 포함했고 batch 평균을 동일 가중 평균하지 않았다.

## 데이터·분할·정규화
Dataset: /data/pjh_workspace/Dataset/Time-Series-Library_dataset/weather/weather.csv

SHA256: 34ee981d07313e51da2a50bb600072c8ae4a69cb4b0651f4cb93a069d7a2ba63

52696 rows. Train [0,36887), validation target [36887,42157), test target [42157,52696). Scaler는 train36887개 행으로만 fit한다. 독립 mean/std 및 변환 결과가 원본 loader와 일치했다. 미래 target 구간은 split간 겹치지 않는다. Val/test 입력 history가 직전 split에 걸치는 것은 알려진 과거 context이며 test 정답 사용이 아니다.

Validation target 시간: 2020-09-13 05:20:00~2020-10-19 19:30:00. Test target 시간: 2020-10-19 19:40:00~2021-01-01 00:00:00. H96 test10444 windows/21055104 원소, H720 test9820 windows/148478400 원소. 모든21채널과 전체96/720 step을 동일 가중 집계한다. Sliding window가 겹치므로 같은 시점은 서로 다른 forecast origin/horizon에서 여러 번 평가된다.

표준화 공간 MSE/MAE이며 역변환하지 않았다. target=OT이지만 features=M이므로 OT 한 채널 평가가 아니다. 실제 timestamp는 대부분10분 간격이며 train에 중복 시각1개와100분 gap1개가 있다. 저장 args.freq=h는 잘못된 nominal metadata이나 Base는 timestamp/time feature를 입력받지 않아 이번 Base 수치에는 영향이 없다. 원본 timestamp를 정제하거나 resample하지 않았다.

## 낮은 수치의 데이터 품질 요인
Train의 -9999는 loader에서 숫자로 취급된다. 의심되는 sentinel의 공식 의미는 별도 데이터 명세 확인이 필요하지만, scaler 분산을 크게 늘린다는 수치적 영향은 확인됐다.

| Channel | -9999 train 개수 | train std | test std | -9999 제외 train std(진단만) |
|---|---:|---:|---:|---:|
| wv (m/s) | 1 | 52.099413 | 1.397981 | 1.667195 |
| max. PAR (�mol/m�/s) | 30 | 643.352692 | 181.461060 | 571.411863 |
| OT | 50 | 383.957098 | 16.264182 | 18.182134 |

H96 wv 정규화 MSE0.001664, OT0.001029처럼 일부 채널 오차가 매우 작아진다. 따라서 현재 수치가 맞다는 사실은 정제된 데이터의 정확도나 외부 논문과의 동등한 전처리를 보증하지 않는다. Test 계절 분포와 채널별 분산도 평균에 영향을 준다. 위 제외 std는 원인 진단용이며 새로운 성능 결과가 아니고 scaler/model은 바꾸지 않았다.

## Full/P100 동일성·lambda0
Full 네트워크와 P100 pipeline은 동일 CELLS의 Base checkpoint를 사용하고 원본 train_r_stage2_lambda01.py를 호출한다. P100 adapter는 Base를 교체하지 않는다. Full H96/H720와 현재 존재하는 P100 H96 원본 Stage2 load_tensors를 새로 실행하여 fresh Base 예측과 최대 차이0, lambda0의 metric이 float64 기준 일치함을 확인했다. P100 H96 완료5개 run의 기록된 Base SHA와 모든 가용 test cache ID도 일치한다. P100 H720는 아직 실행 결과/cache가 없어 실제 run 비교는 NOT YET AVAILABLE이며 코드 연결만 확인했다.

원본 Base 폴더에는 metrics.json만 있고 과거 B prediction array가 없어 과거 B byte-level 비교는 UNAVAILABLE이다. 이번에는 fresh_test_predictions.npy를 별도 감사 경로에 저장했다. Stage2 retrieval cache R을 Base B로 혼동하지 않았다.

## Checkpoint provenance
- Weather_96: /data/pjh_workspace/CARTS/checkpoints/track_r_final_method_generalization01/Weather/H96/seed0/base/checkpoint.pth
  SHA256 3ebb33761a0302ef756cce13470c3d3df8612f287e7f409ade4c3fa3f87aff55; epoch 7. 원본 hash 변경 없음.
- Weather_720: /data/pjh_workspace/CARTS/checkpoints/track_r_final_method_generalization01/Weather/H720/seed0/base/checkpoint.pth
  SHA256 7b84d42f465528e7bf9f6f1b3e605418e8664a91b2a0bb23c5d9db60c0490951; epoch 7. 원본 hash 변경 없음.

## 결론·다음 최소 작업
현재 데이터와 프로토콜에서 0.169364/0.211812 및0.319430/0.343818은 재현된다. 이번 범위에서는 test/val 혼동, horizon checkpoint 혼동, scaler test-fit, 일부 채널/horizon 누락 또는 batch 가중치 오류의 증거가 없다. 그러나 -9999를 포함한 scaler는 데이터 품질 우려이며 이를 해결하지 않고 모델이 물리적으로 매우 정확하다고 해석해서는 안 된다. 후속은 데이터 명세로 sentinel 의미를 확정하고 train-only 정제 규칙을 고정하는 것이다. 정제된 프로토콜의 재학습/재평가는 별도 승인 후 모든 비교군에 동일하게 적용해야 하며 이번 감사에서 실행하지 않았다.

## 감사 실행 기록
독립 metric 검증 초기에 한쪽만 float32 subtraction 후 float64 변환하여 매우 작은 arithmetic 차이가 발생했다. 양쪽 모두 float64 subtraction으로 맞춰 수정했으며 허용오차를 늘리지 않았다. 실패 로그는 audit_attempt1_float32_subtraction.log에 보존했다. Base/Stage2 원본 소스는 수정하지 않았다. P100은 중단하지 않았으며 감사는 GPU1에서 순차 실행 후 종료했다.
