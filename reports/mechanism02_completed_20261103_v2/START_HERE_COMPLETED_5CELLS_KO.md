# Mechanism02 완료 5조건 분석용 자료


추출 확인: 2026-11-03T05:58:40.323440+00:00. ETTh1 96/192/336/720 및 Weather96만 포함합니다. Weather192/336/720은 종합 미완료이므로 결과를 채우지 않았습니다.


각 셀은 `CORE_AND_CPU_ANALYSES_PASS_STUDY_REVIEW_PENDING`입니다. 전체 연구 완료 또는 최종 GO가 아닙니다. 완료 30개 checkpoint의 validation/test 재현 gate, 실제 checkpoint SHA256 및 셀 산출물 hash를 재확인했습니다. 대용량 query 예측 배열은 이번 보고 추출에서 다시 읽지 않았으며 기존 검증 manifest를 근거로 합니다.


## 지표·지원 후보군·출처


모든 수치는 Stage1 retrieval입니다. Stage2 forecasting 수치가 아닙니다. `training_support`는 가중치를 학습한 후보군, `support`/`variant`는 평가 후보군입니다. Native 표의 P100/FULL 모델은 서로 다른 checkpoint이므로 후보군만 바꾼 인과 대조가 아닙니다. 동일 checkpoint 개입은 CROSS_SUPPORT 표에서 SHA256을 고정해 봅니다.


Train에서만 fit한 StandardScaler 공간이며 역변환하지 않습니다. 후보 미래는 `v_i(t)=candidate_future_i(t)-candidate_history_last_i+query_history_last`로 정렬합니다. L=H입니다. 먼저 horizon 평균을 낸 뒤 모든 query×channel에 동일 가중치를 줍니다. 누적합/분모는 float64입니다.


- `RetMSE@10 = mean_(q,c) mean_(i in Top10) mean_t (v_i(t)-y_qc(t))²`.

- `AggMSE@10 = mean_(q,c,t) (mean_(i in Top10) v_i(t)-y_qc(t))²`; AggMAE는 같은 평균 예측의 절대오차입니다.

- `Recall@10 = mean_(q,c) |S10 ∩ O10_same_support|/10`. 후보군이 다르면 Oracle 목표 집합도 다르므로 Recall 원시값의 단순 우열 해석을 피합니다.

- Oracle은 같은 support 안에서 개별 미래 MSE가 가장 작은 10개를 선택합니다. RetMSE의 최적 기준이며 AggMSE/MAE 최적 집합이라는 뜻은 아닙니다. Last-repeat는 마지막 관측을 반복하므로 RetMSE/Recall이 정의되지 않습니다.

- Pool regret=R_P100_oracle−R_Full_oracle, Ranking regret=R_model_P100−R_P100_oracle, Total=두 항의 합. 비중은 집계 regret의 비율입니다. Full에서 선택한 모델에 R_P100_oracle≤R_model을 강요하지 않습니다.

- Exchange regret=(false selected의 MSE 합−missed Oracle의 MSE 합)/10이며 동일 support Ranking regret과 같습니다.

- Raw Cosine은 정규화된 과거 입력의 직접 cosine입니다. Delta-last Cosine·Pearson은 서로 별도 방법입니다. Random/Oracle/Raw 등 baseline은 기존 Attribution01의 검증된 동일 population 결과를 명시적으로 재사용했습니다. 새 계산으로 가장하지 않습니다.

- 모델 checkpoint는 원래 validation 선택을 고정했으며 test로 재선택하지 않았습니다. 모델별 실제 선택 규칙/epoch는 아래 inventory 원문에 있습니다. Full V1 일부는 early stopping이므로 V1/V5를 순수 head 수 인과효과로 단정하지 않습니다.


## 표 읽는 법과 한계


표는 소수 6자리, 아래 CSV 원문은 원시 정밀도입니다. 개선율은 `(비교 기준−대상)/비교 기준×100`이며 RetMSE/AggMSE/MAE의 최선이 다를 수 있습니다. 원래 checkpoint 생성 런타임은 UNKNOWN이며 현재 전체 ordered-ID/예측 재현 PASS와 구분합니다.


Core는 전체 query×channel입니다. 고비용 global correlation은 사전 고정 최대256시점×전체채널이며, 기존 Ridge probe도 train256시점×전체채널×100 future-free 후보로 학습한 모델을 검증 후 재사용했습니다. seed0만이므로 seed 간 불확실성은 미측정입니다. Bootstrap은 전체채널을 묶은 시간블록이며 H/2H/4H 사전 고정, 주 블록 유효 개수4 미만은 유의성 근거가 아닙니다. 다중비교는 탐색적입니다.


Geometry는 null/no-cluster를 허용하고 validation 선택을 고정했습니다. 별도 disjoint future-MSE-matched control은 미구현이므로 multimodality 인과 결론은 INCONCLUSIVE입니다. Weather −9999/scaler 영향은 공식 전체채널 결과에 남아 있고 채널 제외 진단은 별도입니다.


대용량 query별 ranking 원본 CSV 두 alias는 서버에 보존하고 hash와 경로를 기록했습니다. 첨부에는 그 원본의 동일 가중 평균·유효 개수·표본 시점/채널 수를 다시 계산한 RANKING_SAMPLED_AGGREGATE_EXPORT.csv를 제공합니다. 이 파일은 새 inference 결과가 아닌 보고용 집계입니다.


## 확인된 coverage

| cell | validation_query_count | test_query_count | channels | cell_files_fresh_hash_pass | checkpoint_fresh_hash_pass |
| --- | --- | --- | --- | --- | --- |
| ETTh1_96 | 2785 | 2785 | 7 | 50 | 6 |
| ETTh1_192 | 2689 | 2689 | 7 | 50 | 6 |
| ETTh1_336 | 2545 | 2545 | 7 | 50 | 6 |
| ETTh1_720 | 2161 | 2161 | 7 | 50 | 6 |
| Weather_96 | 5175 | 10444 | 21 | 50 | 6 |


## ETTh1_96


### Native P100 — 전체 test

| method | retmse | agg_mse | agg_mae | recall10 |
| --- | --- | --- | --- | --- |
| Last-repeat | — | 1.294371 | 0.713181 | — |
| Random | 0.981790 | 0.540281 | 0.512401 | 0.100518 |
| Raw Cosine | 0.891958 | 0.543834 | 0.505371 | 0.130141 |
| Delta-last Cosine | 0.865244 | 0.510790 | 0.495580 | 0.136307 |
| Pearson | 0.884936 | 0.515246 | 0.497592 | 0.133008 |
| PAA-16 Cosine | 0.874801 | 0.521908 | 0.499562 | 0.120621 |
| V0 | 0.682849 | 0.408743 | 0.426343 | 0.238984 |
| V1 | 0.662553 | 0.412714 | 0.428117 | 0.258574 |
| V5 | 0.663855 | 0.409808 | 0.427272 | 0.257522 |
| IndividualMSE_Oracle | 0.352099 | 0.197742 | 0.306533 | 1.000000 |


### Native FULL — 전체 test

| method | retmse | agg_mse | agg_mae | recall10 |
| --- | --- | --- | --- | --- |
| Last-repeat | — | 1.294371 | 0.713181 | — |
| Random | 2.166891 | 1.385408 | 0.758066 | 0.001272 |
| Raw Cosine | 1.068824 | 0.651255 | 0.554335 | 0.019220 |
| Delta-last Cosine | 0.865244 | 0.510790 | 0.495580 | 0.033070 |
| Pearson | 0.977061 | 0.564908 | 0.525491 | 0.024016 |
| PAA-16 Cosine | 0.876088 | 0.522155 | 0.499048 | 0.021800 |
| V0 | 0.680699 | 0.413918 | 0.428435 | 0.053516 |
| V1 | 0.632498 | 0.422343 | 0.433580 | 0.054804 |
| V5 | 0.623351 | 0.423041 | 0.433557 | 0.059113 |
| IndividualMSE_Oracle | 0.246551 | 0.138068 | 0.254550 | 1.000000 |


### 동일 checkpoint의 후보군 개입

P100_CONTROLLED는 같은 Full logits를 P100 ID로 제한한 비교이고 P100_NATIVE는 기존 native 연산 재현입니다. 분모 대조는 같은 logits의 LOCAL_NORMALIZER/FULL_NORMALIZER끼리 비교합니다. 서로 바꾸어 쓰지 않습니다.

| model | training_support | variant | retmse | agg_mse | agg_mae | recall10 |
| --- | --- | --- | --- | --- | --- | --- |
| V0 | P100 | FULL | 0.679067 | 0.407964 | 0.421806 | 0.057081 |
| V0 | P100 | P100_CONTROLLED | 0.682850 | 0.408744 | 0.426344 | 0.238979 |
| V0 | P100 | P100_NATIVE | 0.682849 | 0.408743 | 0.426343 | 0.238984 |
| V1 | P100 | FULL | 0.659046 | 0.419101 | 0.429744 | 0.057507 |
| V1 | P100 | P100_CONTROLLED | 0.662553 | 0.412714 | 0.428117 | 0.258574 |
| V1 | P100 | P100_NATIVE | 0.662553 | 0.412714 | 0.428117 | 0.258574 |
| V5 | P100 | FULL | 0.684983 | 0.422736 | 0.433368 | 0.058025 |
| V5 | P100 | P100_CONTROLLED | 0.663855 | 0.409807 | 0.427271 | 0.257528 |
| V5 | P100 | P100_NATIVE | 0.663855 | 0.409808 | 0.427272 | 0.257522 |
| V0 | FULL | FULL | 0.680699 | 0.413918 | 0.428435 | 0.053516 |
| V0 | FULL | P100_CONTROLLED | 0.689919 | 0.416127 | 0.432701 | 0.228064 |
| V1 | FULL | FULL | 0.632498 | 0.422343 | 0.433580 | 0.054804 |
| V1 | FULL | P100_CONTROLLED | 0.669044 | 0.424183 | 0.437308 | 0.253044 |
| V5 | FULL | FULL | 0.623351 | 0.423041 | 0.433557 | 0.059113 |
| V5 | FULL | P100_CONTROLLED | 0.675430 | 0.432996 | 0.439539 | 0.258620 |


### P100 native의 regret 분해

| model | pool | ranking | total | pool_share | ranking_share | full_oracle_coverage100 |
| --- | --- | --- | --- | --- | --- | --- |
| V0 | 0.105547 | 0.330750 | 0.436297 | 0.241916 | 0.758084 | 0.188997 |
| V1 | 0.105547 | 0.310455 | 0.416002 | 0.253719 | 0.746281 | 0.188997 |
| V5 | 0.105547 | 0.311757 | 0.417304 | 0.252927 | 0.747073 | 0.188997 |


### V5 내부 head 결합 (validation 선택 single)

| training_support | variant | method | validation_selected_head | retmse | agg_mse | recall10 |
| --- | --- | --- | --- | --- | --- | --- |
| FULL | FULL | All5 | 4 | 0.623351 | 0.423041 | 0.059113 |
| FULL | FULL | ValidationBestSingle | 4 | 0.636201 | 0.458325 | 0.042903 |


Geometry·normalizer·missed/exchange·probe·channel/strata 원문은 다음 부록에 보존했습니다. 전체 결론은 남은 Weather 3조건 완료 후 판단합니다.


## ETTh1_192


### Native P100 — 전체 test

| method | retmse | agg_mse | agg_mae | recall10 |
| --- | --- | --- | --- | --- |
| Last-repeat | — | 1.324880 | 0.733101 | — |
| Random | 1.121939 | 0.608280 | 0.558087 | 0.100861 |
| Raw Cosine | 1.076589 | 0.673831 | 0.579520 | 0.122717 |
| Delta-last Cosine | 0.964302 | 0.574021 | 0.540252 | 0.141099 |
| Pearson | 0.975968 | 0.571090 | 0.536824 | 0.146008 |
| PAA-16 Cosine | 1.038576 | 0.630729 | 0.566265 | 0.103565 |
| V0 | 0.769888 | 0.475023 | 0.472827 | 0.236572 |
| V1 | 0.728216 | 0.471402 | 0.471341 | 0.277597 |
| V5 | 0.745257 | 0.470101 | 0.473508 | 0.258939 |
| IndividualMSE_Oracle | 0.446989 | 0.257865 | 0.353022 | 1.000000 |


### Native FULL — 전체 test

| method | retmse | agg_mse | agg_mae | recall10 |
| --- | --- | --- | --- | --- |
| Last-repeat | — | 1.324880 | 0.733101 | — |
| Random | 2.273585 | 1.420301 | 0.780011 | 0.001142 |
| Raw Cosine | 1.327779 | 0.831234 | 0.642861 | 0.019912 |
| Delta-last Cosine | 0.964302 | 0.574021 | 0.540252 | 0.035988 |
| Pearson | 1.143254 | 0.660132 | 0.580344 | 0.024954 |
| PAA-16 Cosine | 1.121198 | 0.690302 | 0.588437 | 0.014339 |
| V0 | 0.757165 | 0.479668 | 0.475745 | 0.046603 |
| V1 | 0.657726 | 0.485911 | 0.480691 | 0.064655 |
| V5 | 0.653009 | 0.472954 | 0.473124 | 0.065909 |
| IndividualMSE_Oracle | 0.328902 | 0.185861 | 0.298017 | 1.000000 |


### 동일 checkpoint의 후보군 개입

P100_CONTROLLED는 같은 Full logits를 P100 ID로 제한한 비교이고 P100_NATIVE는 기존 native 연산 재현입니다. 분모 대조는 같은 logits의 LOCAL_NORMALIZER/FULL_NORMALIZER끼리 비교합니다. 서로 바꾸어 쓰지 않습니다.

| model | training_support | variant | retmse | agg_mse | agg_mae | recall10 |
| --- | --- | --- | --- | --- | --- | --- |
| V0 | P100 | FULL | 0.782877 | 0.481136 | 0.471591 | 0.050656 |
| V0 | P100 | P100_CONTROLLED | 0.769888 | 0.475023 | 0.472827 | 0.236572 |
| V0 | P100 | P100_NATIVE | 0.769888 | 0.475023 | 0.472827 | 0.236572 |
| V1 | P100 | FULL | 0.721520 | 0.486735 | 0.476886 | 0.060692 |
| V1 | P100 | P100_CONTROLLED | 0.728214 | 0.471402 | 0.471340 | 0.277597 |
| V1 | P100 | P100_NATIVE | 0.728216 | 0.471402 | 0.471341 | 0.277597 |
| V5 | P100 | FULL | 0.750458 | 0.473627 | 0.474448 | 0.058046 |
| V5 | P100 | P100_CONTROLLED | 0.745257 | 0.470101 | 0.473508 | 0.258939 |
| V5 | P100 | P100_NATIVE | 0.745257 | 0.470101 | 0.473508 | 0.258939 |
| V0 | FULL | FULL | 0.757165 | 0.479668 | 0.475745 | 0.046603 |
| V0 | FULL | P100_CONTROLLED | 0.763209 | 0.477313 | 0.475445 | 0.225304 |
| V1 | FULL | FULL | 0.657726 | 0.485911 | 0.480691 | 0.064655 |
| V1 | FULL | P100_CONTROLLED | 0.714215 | 0.475774 | 0.476045 | 0.279684 |
| V5 | FULL | FULL | 0.653009 | 0.472954 | 0.473124 | 0.065909 |
| V5 | FULL | P100_CONTROLLED | 0.724512 | 0.482121 | 0.478783 | 0.272013 |


### P100 native의 regret 분해

| model | pool | ranking | total | pool_share | ranking_share | full_oracle_coverage100 |
| --- | --- | --- | --- | --- | --- | --- |
| V0 | 0.118087 | 0.322899 | 0.440986 | 0.267780 | 0.732220 | 0.186437 |
| V1 | 0.118087 | 0.281227 | 0.399314 | 0.295725 | 0.704275 | 0.186437 |
| V5 | 0.118087 | 0.298268 | 0.416355 | 0.283621 | 0.716379 | 0.186437 |


### V5 내부 head 결합 (validation 선택 single)

| training_support | variant | method | validation_selected_head | retmse | agg_mse | recall10 |
| --- | --- | --- | --- | --- | --- | --- |
| FULL | FULL | All5 | 2 | 0.653009 | 0.472954 | 0.065909 |
| FULL | FULL | ValidationBestSingle | 2 | 0.682420 | 0.518790 | 0.054827 |


Geometry·normalizer·missed/exchange·probe·channel/strata 원문은 다음 부록에 보존했습니다. 전체 결론은 남은 Weather 3조건 완료 후 판단합니다.


## ETTh1_336


### Native P100 — 전체 test

| method | retmse | agg_mse | agg_mae | recall10 |
| --- | --- | --- | --- | --- |
| Last-repeat | — | 1.329927 | 0.745972 | — |
| Random | 1.237366 | 0.673143 | 0.590540 | 0.100208 |
| Raw Cosine | 1.241052 | 0.792405 | 0.639306 | 0.128134 |
| Delta-last Cosine | 1.110472 | 0.670900 | 0.596903 | 0.125664 |
| Pearson | 1.089501 | 0.642467 | 0.584527 | 0.138131 |
| PAA-16 Cosine | 1.254000 | 0.788841 | 0.645271 | 0.085445 |
| V0 | 0.850921 | 0.527890 | 0.510673 | 0.236228 |
| V1 | 0.795184 | 0.509524 | 0.501837 | 0.277631 |
| V5 | 0.790100 | 0.505072 | 0.499984 | 0.285086 |
| IndividualMSE_Oracle | 0.529052 | 0.307981 | 0.387337 | 1.000000 |


### Native FULL — 전체 test

| method | retmse | agg_mse | agg_mae | recall10 |
| --- | --- | --- | --- | --- |
| Last-repeat | — | 1.329927 | 0.745972 | — |
| Random | 2.339841 | 1.431686 | 0.795796 | 0.001297 |
| Raw Cosine | 1.450384 | 0.879760 | 0.671621 | 0.014538 |
| Delta-last Cosine | 1.110472 | 0.670900 | 0.596903 | 0.029346 |
| Pearson | 1.319024 | 0.772309 | 0.644217 | 0.017923 |
| PAA-16 Cosine | 1.356950 | 0.897087 | 0.687600 | 0.006023 |
| V0 | 0.824241 | 0.535382 | 0.514363 | 0.040679 |
| V1 | 0.684627 | 0.515986 | 0.508754 | 0.078496 |
| V5 | 0.764649 | 0.527178 | 0.508424 | 0.074426 |
| IndividualMSE_Oracle | 0.394925 | 0.232953 | 0.333509 | 1.000000 |


### 동일 checkpoint의 후보군 개입

P100_CONTROLLED는 같은 Full logits를 P100 ID로 제한한 비교이고 P100_NATIVE는 기존 native 연산 재현입니다. 분모 대조는 같은 logits의 LOCAL_NORMALIZER/FULL_NORMALIZER끼리 비교합니다. 서로 바꾸어 쓰지 않습니다.

| model | training_support | variant | retmse | agg_mse | agg_mae | recall10 |
| --- | --- | --- | --- | --- | --- | --- |
| V0 | P100 | FULL | 0.876550 | 0.531055 | 0.510395 | 0.042184 |
| V0 | P100 | P100_CONTROLLED | 0.850915 | 0.527887 | 0.510671 | 0.236234 |
| V0 | P100 | P100_NATIVE | 0.850921 | 0.527890 | 0.510673 | 0.236228 |
| V1 | P100 | FULL | 0.800535 | 0.555242 | 0.525592 | 0.061953 |
| V1 | P100 | P100_CONTROLLED | 0.795184 | 0.509524 | 0.501837 | 0.277631 |
| V1 | P100 | P100_NATIVE | 0.795184 | 0.509524 | 0.501837 | 0.277631 |
| V5 | P100 | FULL | 0.785745 | 0.547679 | 0.523489 | 0.067741 |
| V5 | P100 | P100_CONTROLLED | 0.790100 | 0.505072 | 0.499984 | 0.285086 |
| V5 | P100 | P100_NATIVE | 0.790100 | 0.505072 | 0.499984 | 0.285086 |
| V0 | FULL | FULL | 0.824241 | 0.535382 | 0.514363 | 0.040679 |
| V0 | FULL | P100_CONTROLLED | 0.847605 | 0.538057 | 0.518133 | 0.225305 |
| V1 | FULL | FULL | 0.684627 | 0.515986 | 0.508754 | 0.078496 |
| V1 | FULL | P100_CONTROLLED | 0.779596 | 0.511819 | 0.504685 | 0.277429 |
| V5 | FULL | FULL | 0.764649 | 0.527178 | 0.508424 | 0.074426 |
| V5 | FULL | P100_CONTROLLED | 0.799799 | 0.513964 | 0.506174 | 0.252607 |


### P100 native의 regret 분해

| model | pool | ranking | total | pool_share | ranking_share | full_oracle_coverage100 |
| --- | --- | --- | --- | --- | --- | --- |
| V0 | 0.134127 | 0.321869 | 0.455996 | 0.294141 | 0.705859 | 0.169627 |
| V1 | 0.134127 | 0.266132 | 0.400260 | 0.335101 | 0.664899 | 0.169627 |
| V5 | 0.134127 | 0.261048 | 0.395176 | 0.339412 | 0.660588 | 0.169627 |


### V5 내부 head 결합 (validation 선택 single)

| training_support | variant | method | validation_selected_head | retmse | agg_mse | recall10 |
| --- | --- | --- | --- | --- | --- | --- |
| FULL | FULL | All5 | 3 | 0.764649 | 0.527178 | 0.074426 |
| FULL | FULL | ValidationBestSingle | 3 | 0.708939 | 0.513314 | 0.065203 |


Geometry·normalizer·missed/exchange·probe·channel/strata 원문은 다음 부록에 보존했습니다. 전체 결론은 남은 Weather 3조건 완료 후 판단합니다.


## ETTh1_720


### Native P100 — 전체 test

| method | retmse | agg_mse | agg_mae | recall10 |
| --- | --- | --- | --- | --- |
| Last-repeat | — | 1.335121 | 0.755045 | — |
| Random | 1.485056 | 0.857598 | 0.669614 | 0.100053 |
| Raw Cosine | 1.488867 | 0.983556 | 0.729742 | 0.150446 |
| Delta-last Cosine | 1.359586 | 0.872563 | 0.686966 | 0.104152 |
| Pearson | 1.277142 | 0.767495 | 0.643364 | 0.145753 |
| PAA-16 Cosine | 1.547397 | 1.044463 | 0.751084 | 0.092226 |
| V0 | 1.020577 | 0.640548 | 0.584571 | 0.222701 |
| V1 | 0.958376 | 0.633240 | 0.581701 | 0.306148 |
| V5 | 0.942633 | 0.620570 | 0.573133 | 0.321286 |
| IndividualMSE_Oracle | 0.690530 | 0.414016 | 0.451151 | 1.000000 |


### Native FULL — 전체 test

| method | retmse | agg_mse | agg_mae | recall10 |
| --- | --- | --- | --- | --- |
| Last-repeat | — | 1.335121 | 0.755045 | — |
| Random | 2.460671 | 1.458333 | 0.809709 | 0.001276 |
| Raw Cosine | 1.697721 | 1.055446 | 0.754771 | 0.005533 |
| Delta-last Cosine | 1.359586 | 0.872563 | 0.686966 | 0.012005 |
| Pearson | 1.561988 | 0.898846 | 0.704580 | 0.005222 |
| PAA-16 Cosine | 1.548250 | 1.100313 | 0.776685 | 0.002856 |
| V0 | 0.955397 | 0.644375 | 0.579360 | 0.022377 |
| V1 | 0.825363 | 0.592796 | 0.562902 | 0.057183 |
| V5 | 0.815278 | 0.591777 | 0.562189 | 0.064983 |
| IndividualMSE_Oracle | 0.497833 | 0.296855 | 0.371533 | 1.000000 |


### 동일 checkpoint의 후보군 개입

P100_CONTROLLED는 같은 Full logits를 P100 ID로 제한한 비교이고 P100_NATIVE는 기존 native 연산 재현입니다. 분모 대조는 같은 logits의 LOCAL_NORMALIZER/FULL_NORMALIZER끼리 비교합니다. 서로 바꾸어 쓰지 않습니다.

| model | training_support | variant | retmse | agg_mse | agg_mae | recall10 |
| --- | --- | --- | --- | --- | --- | --- |
| V0 | P100 | FULL | 0.975029 | 0.604506 | 0.567799 | 0.021002 |
| V0 | P100 | P100_CONTROLLED | 1.020579 | 0.640549 | 0.584572 | 0.222701 |
| V0 | P100 | P100_NATIVE | 1.020577 | 0.640548 | 0.584571 | 0.222701 |
| V1 | P100 | FULL | 0.919609 | 0.653224 | 0.597422 | 0.050684 |
| V1 | P100 | P100_CONTROLLED | 0.958376 | 0.633240 | 0.581701 | 0.306148 |
| V1 | P100 | P100_NATIVE | 0.958376 | 0.633240 | 0.581701 | 0.306148 |
| V5 | P100 | FULL | 0.975693 | 0.706746 | 0.609344 | 0.055926 |
| V5 | P100 | P100_CONTROLLED | 0.942633 | 0.620570 | 0.573133 | 0.321286 |
| V5 | P100 | P100_NATIVE | 0.942633 | 0.620570 | 0.573133 | 0.321286 |
| V0 | FULL | FULL | 0.955397 | 0.644375 | 0.579360 | 0.022377 |
| V0 | FULL | P100_CONTROLLED | 1.016519 | 0.675784 | 0.595608 | 0.245534 |
| V1 | FULL | FULL | 0.825363 | 0.592796 | 0.562902 | 0.057183 |
| V1 | FULL | P100_CONTROLLED | 0.958612 | 0.641595 | 0.585839 | 0.304079 |
| V5 | FULL | FULL | 0.815278 | 0.591777 | 0.562189 | 0.064983 |
| V5 | FULL | P100_CONTROLLED | 0.946754 | 0.631529 | 0.582646 | 0.310795 |


### P100 native의 regret 분해

| model | pool | ranking | total | pool_share | ranking_share | full_oracle_coverage100 |
| --- | --- | --- | --- | --- | --- | --- |
| V0 | 0.192697 | 0.330047 | 0.522744 | 0.368626 | 0.631374 | 0.106393 |
| V1 | 0.192697 | 0.267846 | 0.460543 | 0.418413 | 0.581587 | 0.106393 |
| V5 | 0.192697 | 0.252103 | 0.444800 | 0.433221 | 0.566779 | 0.106393 |


### V5 내부 head 결합 (validation 선택 single)

| training_support | variant | method | validation_selected_head | retmse | agg_mse | recall10 |
| --- | --- | --- | --- | --- | --- | --- |
| FULL | FULL | All5 | 5 | 0.815278 | 0.591777 | 0.064983 |
| FULL | FULL | ValidationBestSingle | 5 | 0.813604 | 0.584348 | 0.061479 |


Geometry·normalizer·missed/exchange·probe·channel/strata 원문은 다음 부록에 보존했습니다. 전체 결론은 남은 Weather 3조건 완료 후 판단합니다.


## Weather_96


### Native P100 — 전체 test

| method | retmse | agg_mse | agg_mae | recall10 |
| --- | --- | --- | --- | --- |
| Last-repeat | — | 0.259133 | 0.254220 | — |
| Random | 1.485412 | 0.413306 | 0.323663 | 0.100380 |
| Raw Cosine | 0.987011 | 0.406627 | 0.283608 | 0.240126 |
| Delta-last Cosine | 2.369283 | 0.509159 | 0.317922 | 0.178914 |
| Pearson | 0.670132 | 0.326724 | 0.308286 | 0.180391 |
| PAA-16 Cosine | 1.366400 | 0.409568 | 0.310377 | 0.176096 |
| V0 | 0.430874 | 0.241648 | 0.245597 | 0.250195 |
| V1 | 0.458429 | 0.264562 | 0.242770 | 0.262657 |
| V5 | 0.455618 | 0.264330 | 0.242401 | 0.266581 |
| IndividualMSE_Oracle | 0.114812 | 0.086541 | 0.117010 | 1.000000 |


### Native FULL — 전체 test

| method | retmse | agg_mse | agg_mae | recall10 |
| --- | --- | --- | --- | --- |
| Last-repeat | — | 0.259133 | 0.254220 | — |
| Random | 1.310615 | 0.364673 | 0.342936 | 0.000279 |
| Raw Cosine | 0.888510 | 0.541116 | 0.290109 | 0.056504 |
| Delta-last Cosine | 2.369263 | 0.509153 | 0.317920 | 0.054889 |
| Pearson | 0.636991 | 0.323288 | 0.307388 | 0.054437 |
| PAA-16 Cosine | 1.139479 | 0.381028 | 0.307976 | 0.053968 |
| V0 | 0.417865 | 0.225874 | 0.237080 | 0.057687 |
| V1 | 0.287642 | 0.198490 | 0.241027 | 0.003180 |
| V5 | 0.299062 | 0.201959 | 0.246373 | 0.004936 |
| IndividualMSE_Oracle | 0.042052 | 0.028542 | 0.054168 | 1.000000 |


### 동일 checkpoint의 후보군 개입

P100_CONTROLLED는 같은 Full logits를 P100 ID로 제한한 비교이고 P100_NATIVE는 기존 native 연산 재현입니다. 분모 대조는 같은 logits의 LOCAL_NORMALIZER/FULL_NORMALIZER끼리 비교합니다. 서로 바꾸어 쓰지 않습니다.

| model | training_support | variant | retmse | agg_mse | agg_mae | recall10 |
| --- | --- | --- | --- | --- | --- | --- |
| V0 | P100 | FULL | 0.444873 | 0.233334 | 0.240251 | 0.057807 |
| V0 | P100 | P100_CONTROLLED | 0.430876 | 0.241649 | 0.245599 | 0.250193 |
| V0 | P100 | P100_NATIVE | 0.430874 | 0.241648 | 0.245597 | 0.250195 |
| V1 | P100 | FULL | 0.418309 | 0.231116 | 0.271435 | 0.004112 |
| V1 | P100 | P100_CONTROLLED | 0.458429 | 0.264562 | 0.242770 | 0.262658 |
| V1 | P100 | P100_NATIVE | 0.458429 | 0.264562 | 0.242770 | 0.262657 |
| V5 | P100 | FULL | 0.521140 | 0.266707 | 0.290598 | 0.005340 |
| V5 | P100 | P100_CONTROLLED | 0.455619 | 0.264331 | 0.242402 | 0.266581 |
| V5 | P100 | P100_NATIVE | 0.455618 | 0.264330 | 0.242401 | 0.266581 |
| V0 | FULL | FULL | 0.417865 | 0.225874 | 0.237080 | 0.057687 |
| V0 | FULL | P100_CONTROLLED | 0.435626 | 0.236405 | 0.245757 | 0.247718 |
| V1 | FULL | FULL | 0.287642 | 0.198490 | 0.241027 | 0.003180 |
| V1 | FULL | P100_CONTROLLED | 0.396752 | 0.216871 | 0.244538 | 0.236139 |
| V5 | FULL | FULL | 0.299062 | 0.201959 | 0.246373 | 0.004936 |
| V5 | FULL | P100_CONTROLLED | 0.669230 | 0.332061 | 0.254678 | 0.249785 |


### P100 native의 regret 분해

| model | pool | ranking | total | pool_share | ranking_share | full_oracle_coverage100 |
| --- | --- | --- | --- | --- | --- | --- |
| V0 | 0.072760 | 0.316062 | 0.388822 | 0.187130 | 0.812870 | 0.078880 |
| V1 | 0.072760 | 0.343616 | 0.416377 | 0.174746 | 0.825254 | 0.078880 |
| V5 | 0.072760 | 0.340805 | 0.413566 | 0.175934 | 0.824066 | 0.078880 |


### V5 내부 head 결합 (validation 선택 single)

| training_support | variant | method | validation_selected_head | retmse | agg_mse | recall10 |
| --- | --- | --- | --- | --- | --- | --- |
| FULL | FULL | All5 | 1 | 0.299062 | 0.201959 | 0.004936 |
| FULL | FULL | ValidationBestSingle | 1 | 0.367890 | 0.231783 | 0.001026 |


Geometry·normalizer·missed/exchange·probe·channel/strata 원문은 다음 부록에 보존했습니다. 전체 결론은 남은 Weather 3조건 완료 후 판단합니다.


## 체크포인트 provenance 원문

```csv

cell,model,training_support,checkpoint,checkpoint_sha256,selected_epoch,epochs_completed,selection,val_queries,test_queries,channels,full_bank_count,reproduction_status,original_runtime,new_retriever_training
ETTh1_96,V0,P100,/data/pjh_workspace/CARTS/checkpoints/track_p100_head_retrieval_quality01_legacy_pool02/ETTh1_96/seed0/V0/epoch10.pth,0ef270a99a38a6a16f32d878827ea7dc8d9a85cea65c7b64af970babd7ce020b,10,10,minimum validation Mean-Mixture RetMSE@10; first epoch on exact tie,2785,2785,7,8449,PASS,UNKNOWN,False
ETTh1_96,V1,P100,/data/pjh_workspace/CARTS/checkpoints/track_p100_head_retrieval_quality01_legacy_pool02/ETTh1_96/seed0/V1/epoch09.pth,b756a7ee8bd81a997cc995f3ca8484dbc32e618662e61bc3519e2f04ce70411c,9,10,minimum validation Mean-Mixture RetMSE@10; first epoch on exact tie,2785,2785,7,8449,PASS,UNKNOWN,False
ETTh1_96,V5,P100,/data/pjh_workspace/CARTS/checkpoints/track_p100_head_retrieval_quality01_legacy_pool02/ETTh1_96/seed0/V5/epoch10.pth,b7460606bc8dd97e373a93978671bee181da2aa7d0d84338e900b4efa335efbf,10,10,minimum validation Mean-Mixture RetMSE@10; first epoch on exact tie,2785,2785,7,8449,PASS,UNKNOWN,False
ETTh1_96,V0,FULL,/data/pjh_workspace/CARTS/checkpoints/track_v_multiquery_generalization01/ETTh1/H96/seed0/V0/stage1/ETTh1_96/checkpoint.pth,3360bbc44c34c817ef6cd04b518ac5a53ff80c045ac33f06697bcccbbb8af62e,9,,min val model_top10_individual_mse,2785,2785,7,8449,PASS,UNKNOWN,False
ETTh1_96,V1,FULL,/data/pjh_workspace/CARTS/checkpoints/track_v_multiquery_generalization01/ETTh1/H96/seed0/V1/stage1/checkpoint.pth,9e4a5f8a8e90acf8b7032e39ebdbd548bb9c70da7ff7a6582524ca943b4f4930,9,10,original min validation single-head raw-score Top10 RetMSE; no reselection,2785,2785,7,8449,PASS,UNKNOWN,False
ETTh1_96,V5,FULL,/data/pjh_workspace/CARTS/checkpoints/track_v_multiquery_generalization01/ETTh1/H96/seed0/V5/stage1/checkpoint_epoch10.pth,34024fb6b65980cec38f7341173a7faf98639eeab62e88f2a04bf320d7050ecd,10,,min validation Mean-Mixture RetMSE@10 (test split NEVER used),2785,2785,7,8449,PASS,UNKNOWN,False
ETTh1_192,V0,P100,/data/pjh_workspace/CARTS/checkpoints/track_p100_head_retrieval_quality01_legacy_pool02/ETTh1_192/seed0/V0/epoch08.pth,471025ff80f7bb5af9b627a608dcbbfa1141b3cd1c7952a2fc628e0deb536191,8,10,minimum validation Mean-Mixture RetMSE@10; first epoch on exact tie,2689,2689,7,8257,PASS,UNKNOWN,False
ETTh1_192,V1,P100,/data/pjh_workspace/CARTS/checkpoints/track_p100_head_retrieval_quality01_legacy_pool02/ETTh1_192/seed0/V1/epoch07.pth,dd2e3cc55d4f3cba02236872de9fc9647d039773a10f6ca51e97ee41c85b4d46,7,10,minimum validation Mean-Mixture RetMSE@10; first epoch on exact tie,2689,2689,7,8257,PASS,UNKNOWN,False
ETTh1_192,V5,P100,/data/pjh_workspace/CARTS/checkpoints/track_p100_head_retrieval_quality01_legacy_pool02/ETTh1_192/seed0/V5/epoch04.pth,30350cb32b400794fa54ccb129f1a1921321d1c9f1612dca4771bb21e89638f9,4,10,minimum validation Mean-Mixture RetMSE@10; first epoch on exact tie,2689,2689,7,8257,PASS,UNKNOWN,False
ETTh1_192,V0,FULL,/data/pjh_workspace/CARTS/checkpoints/track_v_multiquery_generalization01/ETTh1/H192/seed0/V0/stage1/ETTh1_192/checkpoint.pth,8464084a2dec6f9235a1b11d6a8bb028989ac99610aacd2d4f058600f34dbb2d,9,,min val model_top10_individual_mse,2689,2689,7,8257,PASS,UNKNOWN,False
ETTh1_192,V1,FULL,/data/pjh_workspace/CARTS/checkpoints/track_v_multiquery_generalization01/ETTh1/H192/seed0/V1/stage1/checkpoint.pth,6bd87b186029d04337b64220f7a4fce4abf7970a845dcf3c62e521db510df8ae,8,10,original min validation single-head raw-score Top10 RetMSE; no reselection,2689,2689,7,8257,PASS,UNKNOWN,False
ETTh1_192,V5,FULL,/data/pjh_workspace/CARTS/checkpoints/track_v_multiquery_generalization01/ETTh1/H192/seed0/V5/stage1/checkpoint_epoch6.pth,967948da8e0d1e952152bdefa3aebd90d01175507b181f025b9bd42a6a99e204,6,,min validation Mean-Mixture RetMSE@10 (test split NEVER used),2689,2689,7,8257,PASS,UNKNOWN,False
ETTh1_336,V0,P100,/data/pjh_workspace/CARTS/checkpoints/track_p100_head_retrieval_quality01_legacy_pool02/ETTh1_336/seed0/V0/epoch07.pth,ffa41cb90488081389edc0188872a6ad38769d2b9910e9adcbbe07abe3a107ca,7,10,minimum validation Mean-Mixture RetMSE@10; first epoch on exact tie,2545,2545,7,7969,PASS,UNKNOWN,False
ETTh1_336,V1,P100,/data/pjh_workspace/CARTS/checkpoints/track_p100_head_retrieval_quality01_legacy_pool02/ETTh1_336/seed0/V1/epoch04.pth,26f1f6e794c56857a4aecd21c6817063ecd309538977183bdc299fc6013e5985,4,10,minimum validation Mean-Mixture RetMSE@10; first epoch on exact tie,2545,2545,7,7969,PASS,UNKNOWN,False
ETTh1_336,V5,P100,/data/pjh_workspace/CARTS/checkpoints/track_p100_head_retrieval_quality01_legacy_pool02/ETTh1_336/seed0/V5/epoch04.pth,cda2b579f7d3f491c250712842ea4b067a48e220baa66f3d759144b442e45ebd,4,10,minimum validation Mean-Mixture RetMSE@10; first epoch on exact tie,2545,2545,7,7969,PASS,UNKNOWN,False
ETTh1_336,V0,FULL,/data/pjh_workspace/CARTS/checkpoints/track_v_multiquery_generalization01/ETTh1/H336/seed0/V0/stage1/ETTh1_336/checkpoint.pth,524e31450f26fa89b2a3f16139d17648e8cfa7b1ec351f6cd1a075e7f6ab3c89,6,,min val model_top10_individual_mse,2545,2545,7,7969,PASS,UNKNOWN,False
ETTh1_336,V1,FULL,/data/pjh_workspace/CARTS/checkpoints/track_v_multiquery_generalization01/ETTh1/H336/seed0/V1/stage1/checkpoint.pth,1f8ecef1604e122a85a34e2bcbff000e37914ea3234426d68334e7bb57ffc19d,4,9,original min validation single-head raw-score Top10 RetMSE; no reselection,2545,2545,7,7969,PASS,UNKNOWN,False
ETTh1_336,V5,FULL,/data/pjh_workspace/CARTS/checkpoints/track_v_meanmix_checkpoint_correction01/ETTh1/H336/seed0/V5/stage1_retrain/checkpoint_epoch2.pth,2c66d87d8e24450b8806727421790f4627caa5f5c5e38c8f5310f47926a75b74,2,,min validation Mean-Mixture RetMSE@10 (test split NEVER used),2545,2545,7,7969,PASS,UNKNOWN,False
ETTh1_720,V0,P100,/data/pjh_workspace/CARTS/checkpoints/track_p100_head_retrieval_quality01_legacy_pool02/ETTh1_720/seed0/V0/epoch03.pth,b6ab32cce06bde1accdafc0fa0c9156f599ecccaf0cb6c6d264b8d29a586bc33,3,10,minimum validation Mean-Mixture RetMSE@10; first epoch on exact tie,2161,2161,7,7201,PASS,UNKNOWN,False
ETTh1_720,V1,P100,/data/pjh_workspace/CARTS/checkpoints/track_p100_head_retrieval_quality01_legacy_pool02/ETTh1_720/seed0/V1/epoch09.pth,24d580a769b263f8793104005d54ec3efa27dbfdaa5e8d71fdd31b2fc4d71fa4,9,10,minimum validation Mean-Mixture RetMSE@10; first epoch on exact tie,2161,2161,7,7201,PASS,UNKNOWN,False
ETTh1_720,V5,P100,/data/pjh_workspace/CARTS/checkpoints/track_p100_head_retrieval_quality01_legacy_pool02/ETTh1_720/seed0/V5/epoch09.pth,dc8ae6a4c872974ff5ca36f5c7343e323f7394a7f89d6a56ffeb5aaf580f16e1,9,10,minimum validation Mean-Mixture RetMSE@10; first epoch on exact tie,2161,2161,7,7201,PASS,UNKNOWN,False
ETTh1_720,V0,FULL,/data/pjh_workspace/CARTS/checkpoints/track_v_multiquery_generalization01/ETTh1/H720/seed0/V0/stage1/ETTh1_720/checkpoint.pth,afa04b52930a1e62f94612a710019960412ead4cf4b8e2d3bb685f46e6b3bb68,10,,min val model_top10_individual_mse,2161,2161,7,7201,PASS,UNKNOWN,False
ETTh1_720,V1,FULL,/data/pjh_workspace/CARTS/checkpoints/track_v_multiquery_generalization01/ETTh1/H720/seed0/V1/stage1/checkpoint.pth,43cfe8da9ce67bea75d8733267adb325597f4eece512f9bde69d74f74964fb8e,6,10,original min validation single-head raw-score Top10 RetMSE; no reselection,2161,2161,7,7201,PASS,UNKNOWN,False
ETTh1_720,V5,FULL,/data/pjh_workspace/CARTS/checkpoints/track_v_multiquery_generalization01/ETTh1/H720/seed0/V5/stage1/checkpoint_epoch6.pth,3d92d37d21059fbde85ad76ce57e11b16b2c0ca951cbaf74d90508fc8486c55a,6,,min validation Mean-Mixture RetMSE@10 (test split NEVER used),2161,2161,7,7201,PASS,UNKNOWN,False
Weather_96,V0,P100,/data/pjh_workspace/CARTS/checkpoints/track_p100_head_retrieval_quality01_legacy_pool02/Weather_96/seed0/V0/epoch09.pth,20b0f3ca609f4358cb1ad0191dab138998548c537fdca5e7956eb2459efea677,9,10,minimum validation Mean-Mixture RetMSE@10; first epoch on exact tie,5175,10444,21,36696,PASS,UNKNOWN,False
Weather_96,V1,P100,/data/pjh_workspace/CARTS/checkpoints/track_p100_head_retrieval_quality01_legacy_pool02/Weather_96/seed0/V1/epoch08.pth,2ce4d308d75c10a8784c305aa02d04fac5ca132d42076f4b434196ce5f740520,8,10,minimum validation Mean-Mixture RetMSE@10; first epoch on exact tie,5175,10444,21,36696,PASS,UNKNOWN,False
Weather_96,V5,P100,/data/pjh_workspace/CARTS/checkpoints/track_p100_head_retrieval_quality01_legacy_pool02/Weather_96/seed0/V5/epoch04.pth,fa903d09c4340f6288fc034249f088fb41a04625db23b3a44e077b1e3227d5ae,4,10,minimum validation Mean-Mixture RetMSE@10; first epoch on exact tie,5175,10444,21,36696,PASS,UNKNOWN,False
Weather_96,V0,FULL,/data/pjh_workspace/CARTS/checkpoints/track_v_multiquery_generalization01/Weather/H96/seed0/V0/stage1/Weather_96/checkpoint.pth,3f3605fa4e77c906a1bb8d5a767f8f8660177be5945ea8c9900dd2c566ef9c07,8,,min val model_top10_individual_mse,5175,10444,21,36696,PASS,UNKNOWN,False
Weather_96,V1,FULL,/data/pjh_workspace/CARTS/checkpoints/track_v_multiquery_generalization01/Weather/H96/seed0/V1/stage1/checkpoint.pth,d6d4a6ac3764be6aeac2ee8f79f0a1298b9bb9f3b1eead9dc37fab6e9c9ab6a9,9,10,original min validation single-head raw-score Top10 RetMSE; no reselection,5175,10444,21,36696,PASS,UNKNOWN,False
Weather_96,V5,FULL,/data/pjh_workspace/CARTS/checkpoints/track_v_multiquery_generalization01/Weather/H96/seed0/V5/stage1/checkpoint_epoch10.pth,05abcf3e1eb5f06850f84266e2534281cf53c6e70d391b82ecb5a0dba4874184,10,,min validation Mean-Mixture RetMSE@10 (test split NEVER used),5175,10444,21,36696,PASS,UNKNOWN,False

```


## 사전 규약 원문

```json

{
  "version": "MECHANISM02_v2_parallel_authorized_20261102_064812",
  "source_library_id": "libfile_c73041716eac8191a0faae25ecdb7209",
  "source_lines": 1311,
  "authorization": "2026-11-02: existing checkpoints frozen; train V1 ONLY if valid independent V1 unavailable",
  "cells": [
    "ETTh1_96",
    "ETTh1_192",
    "ETTh1_336",
    "ETTh1_720",
    "Weather_96",
    "Weather_192",
    "Weather_336",
    "Weather_720"
  ],
  "models": [
    "V0",
    "V1",
    "V5"
  ],
  "seed": 0,
  "k": 10,
  "physical_gpu": 1,
  "gpu_uuid": "GPU-91418a73-202a-2a9b-57d5-2538b19e07ec",
  "execution_order": "2026-11-02 06:48:12 user supersedes Solar-completion wait: one new diagnostic may coexist with existing Solar on physical GPU1 after CPU tests, actual free-memory and coexistence smoke; each cell P100+FULL before next cell",
  "concurrency": {
    "maximum_new_gpu_workers": 1,
    "allowed_other_worker": "existing authorized Solar Full8 or its model-gap diagnostic",
    "minimum_free_before_start_gib": 8,
    "new_worker_allocated_cap_fraction": 0.18,
    "timing": "shared GPU observations; not standalone throughput"
  },
  "channel_batch": "ALL channels in each encoder/score/distance GPU call; no serial fallback",
  "chunking": "query/candidate axes only; global masked softmax across entire support",
  "core_population": "all validation/test queries and all channels, including tail batches",
  "global_correlation_sampling": {
    "max_query_timestamps": 256,
    "rule": "unique linspace integer row indices across entire split",
    "all_methods_same": true,
    "all_channels": true,
    "selection_uses_future": false
  },
  "sampling_exclusions": "No sampling for RetMSE/AggMSE/AggMAE/Recall/regret/exchange/IDs/head utility/coverage/near-optimal/multimodality",
  "ndcg": {
    "name": "NDCG_inverse_MSE_v1",
    "gain": "1/(1+d)",
    "discount": "1/log2(rank+1)",
    "exponentiation": false,
    "idcg": "best k over ENTIRE valid support",
    "legacy": "separately named; never substituted"
  },
  "recall": {
    "Recall@k_vs_Oracle@k": "intersection(S_k,O_k)/k, k=10,50,100",
    "Oracle10Coverage@k": "intersection(S_k,O_10)/10; separate name"
  },
  "tie_policy": "stable inherited support order; P100 stored ordered global IDs, Full ascending canonical global ID",
  "aggregation": "horizon within error; equal query-channel weight; float64 accumulated sums/counts",
  "normalization": "unchanged train-fitted StandardScaler; delta-last reconstruction; no inverse transform",
  "regret": {
    "identity_unit": "each query-channel",
    "negative_clamp": false,
    "share": "ratio of aggregate sums; abs(total_sum)<=1e-12 => undefined",
    "less_than_10_valid": "INVALID, stop; no silent smaller K",
    "worst_query": "mean channels per timestamp, rank timestamps by regret; ceil(fraction*Q), fractions=.01,.05,.10"
  },
  "support_intervention": "same frozen weights, exact same Full logits; P100 is verified ordered subset of masked Full; native canonical scores separately reproduced",
  "normalization_only": {
    "identity": "pFull_restricted=mean_h(exp(logZp-logZf)*pP_h)",
    "implicit_head_weight": "m_h/sum_h(m_h), m_h=Zp/Zf",
    "arithmetic": "logsumexp; also retain log probability for single-head rank invariance",
    "single_head": "same P100 relative order MUST remain identical"
  },
  "union": {
    "deployable": "deduplicate global IDs of 5 head Top10; choose K10 by original ENTIRE-support mixture score; no union re-softmax",
    "oracle": "future-selected union Top10 is diagnostic only",
    "budget_note": "equal K does not imply equal compute"
  },
  "best_single": "one head per cell/training-support/evaluation-support selected by aggregate validation RetMSE; tie lower head ID; fixed for test",
  "posthoc_single": "POSTHOC_ORACLE_HEAD, separate, never official",
  "near_optimal": {
    "epsilon_per_channel": "max(1e-8, 0.01*median_validation_oracle_kth_mse))",
    "threshold": "test d_i <= test d_oracle10 + frozen validation epsilon_c",
    "unique_relevant": "head Top10 candidate not in other heads Top10 and passes this near-optimal threshold",
    "exclusive_oracle": "head-exclusive candidate in exact Oracle10; separate from near-optimal"
  },
  "geometry": {
    "positive_sizes": [
      10,
      30
    ],
    "spaces": [
      "raw",
      "delta_last",
      "pearson_normalized",
      "V0_candidate",
      "V1_candidate",
      "V5_shared_candidate"
    ],
    "cluster_grid": [
      1,
      2,
      3
    ],
    "minimum_cluster_size": 3,
    "k1": "null/no-cluster",
    "validation_choice": "per cell/support/space/set-size, choose k among 2/3 only if median silhouette>=.25, median subsample ARI>=.8, median silhouette exceeds matched-size random control by>=.10; otherwise k1; freeze for test",
    "stability": "20 deterministic 80% candidate subsamples, compare shared-ID assignments; seed0",
    "controls": [
      "same_size_random",
      "same_size_past_nearest",
      "future_mse_matched_if_feasible"
    ],
    "shared_ids": "report set overlap; never call overlap-induced similarity independent evidence",
    "verdict": "test failure/degenerate geometry => descriptive/INCONCLUSIVE, no forced multimodality"
  },
  "statistics": {
    "seed": 0,
    "repeats": 2000,
    "unit": "time blocks with all channels jointly",
    "block_lengths": "primary L+H, sensitivities H and 2*(L+H); fixed for all comparisons",
    "insufficient": "fewer than four effective primary blocks => descriptive CI only, no significance claim",
    "multiple_comparisons": "exploratory, no unadjusted significance claims",
    "training_seed_uncertainty": "not estimated"
  },
  "probe": {
    "reuse": "existing frozen Ridge only when all provenance/feature/label gates match",
    "fit": "train-only labels with complete future within train boundary",
    "choose": "validation only",
    "features": "past only; no query/candidate IDs"
  },
  "new_v1_if_needed": {
    "architecture": "same independently trained single SlotHeads projection; no V5-head extraction",
    "seed": 0,
    "epochs": 10,
    "batch": 32,
    "adam_lr": 0.001,
    "teacher": "existing valid-support z-score FutureMSE, negative temperature .1",
    "student_temperature": 0.1,
    "selection": "min validation RetMSE@10, earlier epoch on tie; test never used",
    "note": "fresh runs separately identified; do not claim fixed-budget causal parity with historical early-stopped Full models"
  },
  "forbidden": [
    "V0/V5 retraining",
    "Stage2",
    "new loss",
    "new architecture",
    "seed1/2",
    "GPU0",
    "channel reduction",
    "more than one new Mechanism02 GPU worker",
    "push/upload",
    "original result overwrite"
  ]
}

```



## 상세 증거

전체 원시 CSV는 cells/에 있고 100행 통합 비교는 UNIFIED_NATIVE_TEST_COMPARISON.csv에 있습니다. checkpoint·gate·baseline 재사용 원본과 SHA256은 SOURCE_MANIFEST.json 및 checkpoint_evidence/에 포함했습니다. 대용량 개별 ranking CSV 원본은 LARGE_FILES_RETAINED_ON_SERVER.json에 경로와 hash가 기록되어 있고, 첨부에는 정확히 같은 표본의 집계만 담았습니다. 전체 query 예측 배열과 checkpoint 파일 자체는 첨부하지 않았습니다.
