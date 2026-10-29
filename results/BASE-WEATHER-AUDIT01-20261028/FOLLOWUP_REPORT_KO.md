# Weather Base 후속 진단

원본 데이터: MPI Jena roof 2020a/b 공식 ZIP과 현재 CSV의 52,696행, 21개 수치 채널 및 timestamp가 전부 정확히 일치했다. OT는 원본 CO2 (ppm)이다.

출처: https://www.bgc-jena.mpg.de/wetter/weather_data.html ; https://www.bgc-jena.mpg.de/wetter/Weatherstation.pdf

공식 PDF와 다운로드 페이지/ZIP에는 -9999의 명시적인 결측 코드 정의를 찾지 못했다. 따라서 공식 의미는 UNCONFIRMED. TensorFlow 공식 Jena tutorial은 wind -9999를 likely erroneous라고 기술하지만 다른 연도 14채널 자료이며 CO2/PAR의 공식 정의를 대신하지 않는다. https://www.tensorflow.org/tutorials/structured_data/time_series

|H|평가 채널|Base MSE|Base MAE|All5 MSE|All5 MAE|
|---|---|---:|---:|---:|---:|
|96|all21|0.169364456|0.211811757|0.168176157|0.215937994|
|96|exclude_train_sentinel_channels18|0.194577639|0.236783897|0.193731774|0.241267719|
|720|all21|0.319429895|0.343818117|0.342292280|0.347555510|
|720|exclude_train_sentinel_channels18|0.367566895|0.382313718|0.396858385|0.393161602|

제외 기준은 train에서 -9999가 존재하는 채널(wv, max.PAR, OT)이다. 모델/scaler/예측을 바꾸지 않은 진단이며 공식 성능 대체값 또는 정제 데이터 재학습 성능이 아니다. 채널 제외는 평가 대상 변경 효과를 측정할 뿐 scaler 이상값의 인과적 효과량을 식별하지 않는다.

모든 21채널 test raw row와 반복 window target에서 -9999는 0개다. 따라서 해당 target의 직접 MSE/MAE 기여는 0. train scale에는 영향이 남는다. wv 52.0994 vs -9999 제외시 1.66719, OT 383.9571 vs 18.18213, max.PAR 643.3527 vs 571.41186.

H720: Base MSE 우위 15/21채널, MAE 우위 13/21. p/rho/rh가 양의 Base MSE 우위 합의 65.53%를 설명한다. 문제 3채널을 제외해도 Base 우위가 유지된다. H96: Base MSE 우위는 15/21이나 일부 채널의 All5 이득이 더 커 전체 MSE는 All5가 조금 좋다. 전체 MAE는 Base가 좋다.

H720 입력: 독립 V0/V1/V2/V5 encoder 첫 weight 입력축720을 직접 확인, Base weight720x720. Full 11구성은 동일 V5/loader를 공유하며 full_analysis.py:69 seq_len=h 및 :237 Stage2 전달. P100 모든 arm은 pipeline.py:225 공통 CELLS seq_len=720을 전달. Weather720 P100 실행 디렉터리는 이번 확인 시 존재하지 않아 실제 완료 arm 검증은 NOT RUN; 설정 일치만 확인.

Full All5 재구성: 저장 R, 기존 lambda, 직전 fresh Base를 사용. canonical MSE 차이 H96 1.82e-9/H720 1.80e-7. CPU float32 fusion 후 float64 element aggregation. window ID 동일성 검증 PASS. GPU/새 학습/원본 수정/GitHub 업로드 없음.

채널별 전체 수치와 오차 기여: FOLLOWUP_CHANNEL_DIAGNOSTICS.csv. 원본 ZIP SHA와 일치 검증: OFFICIAL_SOURCE_COMPARISON.json. 실행 코드/로그: followup.py/followup.log. 입력 길이 증거: H720_SEQUENCE_LENGTH_EVIDENCE.json.
