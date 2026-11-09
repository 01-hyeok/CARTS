# CARTS — Comprehensive Retrieval Failure & Multi-Head Mechanism Analysis

## 0. 역할과 연구 목표

당신은 시계열 검색 연구 CARTS의 연구 엔지니어이자 엄격한 검증자다.

Repository:
https://github.com/01-hyeok/CARTS

이번 실험의 목적은 다음과 같다.

**Query의 과거 시계열만으로 미래가 가장 유사한 Historical Candidate Top-10을 검색하는 과정에서, 현재 V0/V1/V5 Retriever가 왜 실패하는지 분석한다.**

특히 다음 사항을 규명해야 한다.

1. P100 후보군에 유용한 후보가 존재하는데도 검색하지 못하는가?
2. 실제로 놓친 Oracle 후보는 Student Ranking의 어느 위치에 존재하는가?
3. 검색 오류가 Top-10 경계에서 발생하는가, 전체 Embedding Ranking이 잘못된 것인가?
4. V5는 왜 ETTh1에서는 유리하지만 Weather P100에서는 악화되는가?
5. V5의 Full/P100 차이는 후보군 변화 때문인가, 별도 학습된 Checkpoint 차이 때문인가?
6. 다섯 Head가 서로 다른 유용한 후보를 검색하는가, 단순히 중복된 검색을 수행하는가?
7. 미래가 유사한 Oracle 후보들이 과거 Embedding 공간에서 여러 집단으로 나타나는가?
8. 어떤 Retrieval Loss 또는 Multi-Head 학습 구조가 다음 연구로 가장 타당한가?

**이번 실험에서는 새로운 Retriever를 학습하지 않는다.**

기존 Checkpoint를 사용한 Diagnostic-only 실험을 수행한다.

Stage2 Forecasting 실험은 제외한다.

Primary Metric은 RetMSE@10으로 고정한다.

기존 파일과 결과를 덮어쓰지 않는다.

---

# 1. 실험 범위

## 1.1 Dataset / Horizon

| Dataset | Seq Len | Pred Len |
|---|---|---|
| ETTh1 | 96 | 96 |
| ETTh1 | 192 | 192 |
| ETTh1 | 336 | 336 |
| ETTh1 | 720 | 720 |
| Weather | 96 | 96 |
| Weather | 192 | 192 |
| Weather | 336 | 336 |
| Weather | 720 | 720 |

총 8개 Dataset × Horizon 조건.

## 1.2 Candidate Support

각 조건에 대해 반드시 두 Support를 모두 진단한다.

- P100: 기존 검증된 Shared Top-100 Candidate Pool
- Full: 전체 유효 Candidate Bank

총 16개 Dataset × Horizon × Support 조건.

## 1.3 평가 모델

Primary:

- V0: Original KL Retriever
- V5: 5-Head Mean-Mixture Retriever

Secondary:

- V1: 독립적으로 학습된 Single-Projection Retriever
- V5 내부 개별 Head H1~H5

V1과 V5 내부 Head는 서로 다른 모델이므로 반드시 구분한다.

Full V1 Checkpoint가 존재하지 않거나 평가 호환성이 검증되지 않았다면 MISSING으로 기록하고, 임의 생성하거나 새로 학습하지 않는다.

## 1.4 고정 설정

- Seed = 0
- Physical GPU = 1
- K = 10
- 기존 Train/Validation/Test Split 유지
- 기존 Candidate Mask 유지
- 동일 Query/Channel 평가
- 기존 Delta-last Future Reconstruction 유지
- 기존 Normalization 유지
- Checkpoint 재선택 및 재학습 금지
- 기존 결과 파일 수정 금지

---

# 2. Phase 0 — 기존 결과 및 구현 감사

실험을 시작하기 전에 다음 로컬 결과를 우선 확인한다.

- `results/TRACK-P100-HEAD-RETRIEVAL-QUALITY01-LEGACYPOOL02/`
- `results/TRACK-RETRIEVAL-FAILURE-ATTRIBUTION01/`
- `results/TRACK-RETRIEVAL-DIAGNOSTIC-SUPPLEMENT01/`
- `results/TRACK-V5-FULL-HEAD-ANALYSIS01/`
- `results/TRACK-HEAD-CONDITIONED-CANDIDATE-SUPPORT01/`
- `results/TRACK-V5-HEAD-OBJECTIVE-COSINE-FIX01/`

위 경로 중 일부는 서버에만 존재할 수 있으므로 실제 파일 존재 여부를 확인한다.

다음 코드의 현재 구현도 감사한다.

- `utils/mean_mixture_selection.py`
- `utils/candidate_pool.py`
- `utils/full_candidate_bank.py`
- `utils/rank_losses.py`
- `scripts/train_v_sharedtop100_01.py`
- `scripts/eval_v5_meanmix_checkpoint_selection_pool01.py`
- `scripts/diag_head_conditioned_support01.py`
- `experiments/full_head_analysis01/`

## 2.1 Checkpoint 감사

8개 조건 각각에서 다음 정보를 확인한다.

- V0 P100 Checkpoint
- V5 P100 Checkpoint
- V0 Full Checkpoint
- V5 Full Checkpoint
- V1 P100 Checkpoint (존재하는 경우)
- V1 Full Checkpoint (존재하는 경우)
- Validation-selected Epoch
- Checkpoint SHA256
- Training Loss
- Candidate Pool 방식
- Model Architecture
- Query/Key Encoder 설정
- Mean-Mixture 적용 여부

과거 Teacher-sign 오류 및 Round-Robin 오류가 확인된 무효화 결과를 사용하지 않는다.

V5 추론은 반드시 다음 공식을 따른다.

\[
p_h(i)=\operatorname{softmax}(s_h(i)/\tau_s)
\]

\[
p_{\mathrm{mix}}(i)=\frac15\sum_{h=1}^{5}p_h(i)
\]

\[
S_{10}=\operatorname{Top10}(p_{\mathrm{mix}})
\]

Head별 Raw Cosine 점수의 단순 평균이나 Round-Robin을 V5 Mean-Mixture로 대체하지 않는다.

## 2.2 기존 지표 재현

각 조건에서 기존 Test 지표를 재현한다.

- RetMSE@10
- AggMSE@10
- AggMAE@10
- Recall@10
- Selected Candidate IDs

기존 검증된 원시 결과와 비교한다.

참고용 수치:

| 조건 | V0 RetMSE | V5 RetMSE |
|---|---:|---:|
| ETTh1 H96 P100 | 0.682849 | 0.663855 |
| ETTh1 H96 Full | 0.680699 | 0.623351 |
| Weather H96 P100 | 0.430874 | 0.455618 |
| Weather H96 Full | 0.417865 | 0.299062 |

위 수치는 기존 보고서의 반올림 값이므로 실제 검증에는 원시 결과를 사용한다.

검증에 실패하면 원인을 수정하지 않은 채 결과만 강제로 일치시키지 않는다.

산출물:

- `CHECKPOINT_AUDIT.csv`
- `METRIC_REPRODUCTION.csv`
- `IMPLEMENTATION_AUDIT.md`

---

# 3. Phase 1 — Retrieval Baseline Comparison

기존 Retriever가 단순 시계열 유사도보다 실제로 좋은 후보를 검색하는지 확인한다.

## 비교 대상

- Raw Cosine
- Delta-last Cosine
- Pearson
- PAA-16 Cosine
- V0
- V1 (Checkpoint 존재 시)
- V5
- Individual Oracle

모든 방법은 동일한 Query, Channel, Candidate Mask, Support에서 평가한다.

PAA-16은 기존 구현을 확인하고, 각 입력 길이를 16개 동일 길이 구간으로 압축하는 기준을 고정한다.

기존 Canonical 구현과 다른 경우 새로운 Baseline이라고 표시한다.

## 평가 지표

- RetMSE@10
- Recall@10
- NDCG@10
- Oracle Regret
- Query-level Win Rate

P100과 Full을 별도 결과표로 작성한다.

특히 다음을 확인한다.

1. V0가 Raw/Pearson/Delta-last보다 우수한가?
2. V1이 V0보다 우수한가?
3. V5가 독립 V1보다 우수한가?
4. V5가 우수한 조건과 악화되는 조건의 공통적인 특징은 무엇인가?
5. Full과 P100의 Baseline 순위가 달라지는가?

**V0와 V5만 비교해서 Multi-Head 효과를 결론 내리지 않는다.**

V0→V1의 Projection 효과와 V1→V5의 Multi-Head 효과를 분리한다.

산출물:

- `RETRIEVAL_BASELINE_COMPARISON.csv`
- `V0_V1_V5_COMPARISON.csv`

---

# 4. Phase 2 — Candidate Pool vs Ranking Regret

이번 분석의 첫 번째 핵심이다.

## 4.1 Oracle 정의

Query q와 Candidate i의 Future MSE:

\[
d(q,i)=\frac1H\sum_t(Y_q(t)-V_i(t))^2
\]

V_i는 기존 CARTS와 동일한 Last-aligned Candidate Future를 사용한다.

각 Query/Channel에서 다음 세 가지를 계산한다.

### Full Oracle

전체 유효 Candidate 중 Future MSE가 가장 낮은 10개의 평균.

\[
R_F
\]

### P100 Oracle

해당 Query의 P100 Candidate 중 Future MSE가 가장 낮은 10개의 평균.

\[
R_P
\]

### 실제 Retriever

모델이 선택한 Top-10 후보의 평균 Future MSE.

\[
R_M
\]

## 4.2 Regret Decomposition

P100 평가에 대해서:

\[
\text{PoolRegret}=R_P-R_F
\]

\[
\text{RankingRegret}=R_M-R_P
\]

\[
\text{TotalRegret}=R_M-R_F
\]

반드시 다음을 검증한다.

\[
\text{TotalRegret}
=
\text{PoolRegret}+\text{RankingRegret}
\]

Full 평가의 경우:

\[
\text{RankingRegret}_{Full}=R_M-R_F
\]

Full Retriever의 성능이 P100 Oracle보다 나빠야 한다는 잘못된 제약을 적용하지 않는다.

## 4.3 필수 결과

8개 데이터셋 조건 × P100/Full × 모델별로 다음을 계산한다.

- Model RetMSE
- Full Oracle RetMSE
- P100 Oracle RetMSE
- Pool Regret
- Ranking Regret
- Total Regret
- Pool Regret Share
- Ranking Regret Share
- Oracle Coverage
- Query-level Regret 분포
- Channel별 Regret 분포

Regret 비율은 Query별 비율을 평균하지 않고 전체 Regret을 먼저 집계한 후 계산한다.

추가로 Regret 상위 1%, 5%, 10% Query가 전체 검색 손실에 얼마나 기여하는지 분석한다.

산출물:

- `RETRIEVAL_REGRET_DECOMPOSITION.csv`
- `REGRET_BY_QUERY.parquet`
- `REGRET_BY_CHANNEL.csv`
- `POOL_VS_RANKING_REGRET.png`

---

# 5. Phase 3 — 동일 Checkpoint의 P100 vs Full 평가

**이번 실험에서 반드시 새롭게 확인해야 하는 핵심 진단이다.**

기존 P100과 Full 성능은 서로 다른 학습 Checkpoint를 사용하므로, 성능 차이를 후보군만의 효과라고 해석할 수 없다.

따라서 동일한 Checkpoint를 고정한 상태에서 Candidate Support만 변경하는 평가를 수행한다.

## 5.1 Cross-Support Evaluation Matrix

각 Dataset/Horizon/V0 또는 V5에 대해 다음 네 조건을 계산한다.

| 학습된 Checkpoint | 평가 Candidate Support |
|---|---|
| P100-trained | P100 |
| P100-trained | Full |
| Full-trained | Full |
| Full-trained | P100 |

반드시 동일한 Query, Channel, Mask, 평가 Future Space를 사용한다.

Checkpoint 파라미터는 변경하지 않는다.

Support를 변경하면서 Candidate Embedding을 동일 Checkpoint의 Encoder로 재계산한다.

## 5.2 비교 목적

### Comparison A

P100-trained 모델에서 P100→Full로 바꾸면 RetMSE가 개선되는가?

### Comparison B

Full-trained 모델에서 Full→P100으로 바꾸면 RetMSE가 개선되는가?

### Comparison C

동일 P100 Support에서 P100-trained와 Full-trained 중 어느 Checkpoint가 더 좋은가?

### Comparison D

동일 Full Support에서 P100-trained와 Full-trained 중 어느 Checkpoint가 더 좋은가?

이 비교로 다음 두 요인을 분리한다.

- Training Support Difference
- Inference Candidate Support Difference

단, 이는 학습을 다시 통제한 인과 실험이 아니라, **고정 Checkpoint의 Support Intervention**이다.

## 5.3 V5 Softmax Support Sensitivity

V5의 Head별 Softmax는 Candidate Support가 바뀌면 정규화 분모도 달라진다.

따라서 추가 후보를 제공하는 것만으로 기존 P100 후보의 Mean-Mixture 순위가 변경될 수 있다.

이 효과를 명시적으로 분석한다.

동일 Checkpoint에서 다음 두 점수를 계산한다.

### P100-normalized Mixture

각 Head의 Softmax를 P100 내부에서 계산한다.

\[
p_{100}(i)=\frac15\sum_h
\frac{\exp(s_{h,i}/\tau)}
{\sum_{j\in P100}\exp(s_{h,j}/\tau)}
\]

### Full-normalized Mixture

각 Head의 Softmax를 Full Bank에서 계산한다.

\[
p_{Full}(i)=\frac15\sum_h
\frac{\exp(s_{h,i}/\tau)}
{\sum_{j\in Full}\exp(s_{h,j}/\tau)}
\]

이후 동일한 P100 후보들에 대해 두 방식의 상대 순위를 비교한다.

평가 항목:

- P100 후보 순위 Spearman/Kendall
- P100 내 Top-10 변화율
- 동일한 P100 후보를 사용한 RetMSE 차이
- Head별 P100 Probability Mass
- Head별 Softmax Normalization Constant
- Support 변경에 따른 Head의 암묵적인 상대 가중치 변화

**중요:** 이 분석은 Candidate Pool을 바꾸지 않고 Softmax 정규화 범위만 변경한 대조 실험이어야 한다.

이렇게 해야 V5의 Support Sensitivity가 단순 후보 추가 때문인지, Mean-Mixture 정규화 구조 때문인지 부분적으로 분리할 수 있다.

산출물:

- `CROSS_SUPPORT_EVALUATION.csv`
- `SOFTMAX_SUPPORT_SENSITIVITY.csv`
- `HEAD_PROBABILITY_MASS.csv`
- `SUPPORT_RANK_SHIFT.png`

---

# 6. Phase 4 — Global Ranking Quality

현재 Retriever가 Future Relevance 순위를 얼마나 잘 학습했는지 확인한다.

## 6.1 Score 정의

Student Score:

\[
s(q,i)
\]

Teacher Relevance:

\[
r(q,i)=-d(q,i)
\]

V5는 실제 추론에 사용하는 Mean-Mixture Probability를 Ranking Score로 사용한다.

전체 후보와 P100 후보의 Score를 혼용하지 않는다.

## 6.2 Ranking 지표

다음을 계산한다.

- Spearman Correlation
- Kendall Tau
- NDCG@10
- NDCG@50
- NDCG@100
- Recall@10
- Recall@50
- Recall@100

다음 범위를 분리한다.

1. Entire Support Ranking
2. Student Top-100 내부 Ranking
3. Student Top-50 내부 Ranking
4. Oracle Top-100 내부 Ranking

Student 기준과 Oracle 기준의 후보 집합을 구분한다.

## 6.3 Full Candidate 계산 비용

Full Bank의 전체 후보 순위 상관 분석은 계산량이 크다.

모든 Dataset/Horizon/Support에서 필수 Core Metric은 전체 Query/Channel 대상으로 계산한다.

Full Global Ranking Correlation이 비용상 과도한 경우 사전에 고정한 시간 구간의 Query Sample을 사용한다.

Sampling 조건:

- Seed0
- Dataset/Horizon별 고정 Sampling
- 전체 시간 범위에서 균일하게 선택
- 선택된 Query의 모든 Channel 포함
- Test Regret을 보고 Sample을 선택하지 않음
- Exact / Approximate 구분
- 전체 Query 대비 분석 Coverage 기록

샘플 결과를 전체 Query를 평가한 수치처럼 보고하지 않는다.

## 6.4 핵심 질문

- Global Ranking이 개선되어도 RetMSE는 악화되는가?
- Oracle 후보가 전체적으로 낮은 Student Rank를 받는가?
- V5가 V0보다 실제 Ranking Quality를 개선하는가?
- Weather P100에서 V5가 V0보다 순위를 잘못 학습한 구간은 어디인가?

산출물:

- `GLOBAL_RANKING_QUALITY.csv`
- `TOPK_RANKING_QUALITY.csv`
- `RANKING_VS_RETMSE.csv`

---

# 7. Phase 5 — Missed Oracle Candidate Analysis

모델이 놓친 Oracle 후보를 분석한다.

## 7.1 Missed Oracle

각 Query에서:

\[
M=O_{10}\setminus S_{10}
\]

\[
F=S_{10}\setminus O_{10}
\]

M은 Missed Oracle, F는 False Selected이다.

P100 평가에서는 P100 Oracle을 사용한다.

Full 평가에서는 Full Oracle을 사용한다.

P100 밖의 Full Oracle 후보는 `OUT_OF_SUPPORT`로 표시한다.

## 7.2 Missed Oracle Rank Distribution

각 Missed Oracle 후보의 Student Rank를 계산한다.

P100:

- 11–20
- 21–50
- 51–100

Full:

- 11–20
- 21–50
- 51–100
- 101–500
- 501–1000
- 1001+
- Rank Percentile

결과:

- Mean Missed Rank
- Median Missed Rank
- P90/P95 Missed Rank
- Rank Bin Distribution
- Missed Oracle Count
- Query별 Missed Fraction

## 7.3 Top-K Exchange Regret

실제로 잘못 선택된 후보와 놓친 Oracle 후보의 Future MSE 차이를 계산한다.

\[
\text{ExchangeRegret}
=
\frac{
\sum_{j\in F}d_j-\sum_{i\in M}d_i
}{10}
\]

동일 Support에서 Exchange Regret은 Ranking Regret과 일치해야 한다.

추가로:

- Oracle 10위와 11위의 Future MSE 차이
- Student 10위와 11위의 Score 차이
- Student Top-10 내 False Selected 비율
- False Selected의 평균 Future MSE
- Missed Oracle의 평균 Future MSE
- 큰 Regret을 만드는 후보의 순위 및 Score

를 계산한다.

## 7.4 핵심 질문

- 실제 손실이 Top-K Boundary에 집중되어 있는가?
- Oracle 후보가 주로 11–20위에 있는가?
- 좋은 후보들이 50–100위 또는 Full Bank의 더 낮은 순위에 있는가?
- 순위 오류는 많은데 실제 MSE 손실은 작은 경우가 존재하는가?
- RetMSE가 나쁜 Query는 소수의 극단적인 False Selected 때문에 발생하는가?

산출물:

- `MISSED_ORACLE_RANKS.csv`
- `TOPK_EXCHANGE_REGRET.csv`
- `BOUNDARY_ERROR_ANALYSIS.csv`
- `MISSED_ORACLE_RANK_HISTOGRAM.png`

---

# 8. Phase 6 — Weather Recall/RetMSE Anomaly Investigation

Weather Full H96에서 기존 보고서는 다음을 보여준다.

| Model | RetMSE | Recall@10 |
|---|---:|---:|
| V0 | 0.417865 | 0.057687 |
| V5 | 0.299062 | 0.004936 |

V5는 RetMSE를 크게 개선했지만 Oracle Recall은 감소했다.

이를 단순한 오류로 가정하지 말고 독립적으로 검증한다.

## 8.1 구현 검증

- Oracle Top-10 Candidate IDs
- Model Top-10 Candidate IDs
- Candidate Global/Local ID Mapping
- Stable Tie-breaking
- Valid Mask
- Last-aligned Future MSE
- Mean-Mixture Score
- Recall 분모
- Query/Channel Aggregation
- Checkpoint Selected Epoch

위 항목을 재검증한다.

## 8.2 검색 결과 품질 분석

다음을 계산한다.

- Full Oracle Top-10과의 Exact Overlap
- Oracle Top-10과의 Future MSE Gap
- Oracle Top-10 주변의 Near-optimal Candidate 수
- Oracle 10위 Future MSE 대비 Model Top-10 Future MSE
- V0/V5가 선택한 후보들의 Future MSE 분포
- Recall과 RetMSE의 Query-level Correlation

가설:

V5가 정확한 Oracle 후보 ID는 적게 찾더라도, 미래 MSE가 거의 비슷한 다른 좋은 후보들을 많이 선택했을 가능성이 있다.

단, 이 가설은 결과로 검증한다.

## 8.3 Weather 데이터 품질 민감도

기존 Weather의 `-9999` Sentinel 및 Scaler 관련 분석을 확인한다.

공식 전체 채널 지표는 유지한다.

별도로:

- 영향 가능 Channel
- 그 외 Channel
- 전체 Channel

의 결과를 비교한다.

비정상값을 임의로 수정하거나 공식 지표를 변경하지 않는다.

산출물:

- `WEATHER_RECALL_ANOMALY_AUDIT.md`
- `NEAR_ORACLE_CANDIDATE_ANALYSIS.csv`
- `WEATHER_CHANNEL_SENSITIVITY.csv`

---

# 9. Phase 7 — Multi-Head Retrieval Utility Analysis

이번 분석은 기존 V5의 Head가 실제로 유용한지를 평가한다.

단순히 Head 간 검색 결과가 다르다는 사실만으로 전문화가 입증되었다고 결론 내리지 않는다.

## 9.1 개별 Head 검색 품질

V5의 동일 Checkpoint에서 각 Head의 Standalone Top-10을 평가한다.

- H1
- H2
- H3
- H4
- H5
- Mean-Mixture All-5

각각 다음을 계산한다.

- RetMSE@10
- Recall@10
- NDCG@10
- Oracle Regret

Validation에서 RetMSE 기준으로 가장 좋은 Single Head를 선택하고, Test에서 All-5와 비교한다.

Test에서 가장 좋은 Head를 사후 선택한 값은 `POSTHOC_ORACLE_HEAD`로 따로 표시한다.

독립 V1과 V5 내부 Single Head도 분리한다.

## 9.2 Head Diversity

- Pairwise Top-10 Overlap
- Union Size
- Head별 Unique Candidate 수
- Head별 Unique Oracle Candidate 수
- Oracle-positive Coverage
- Head별 Candidate Future MSE

## 9.3 Unique Useful Candidates

각 Head가 나머지 Head가 찾지 못한 유용한 후보를 찾는지 측정한다.

유용성은 Future MSE 기준으로 정의한다.

다음 두 값을 구분한다.

1. 단순히 다른 Candidate ID를 검색했는가?
2. 미래가 실제로 유사한 새로운 Candidate를 검색했는가?

각 Head의 Exclusive Oracle Hits 및 Unique Relevant Candidate 수를 계산한다.

## 9.4 동일 검색 예산 비교

5개 Head가 총 50개 후보를 제공한다고 해서 Single Head Top-10과 그대로 비교하지 않는다.

다음 조건을 비교한다.

- Validation-selected Best Single Head Top-10
- V5 Mean-Mixture Top-10
- Head Union에서 미래를 사용하지 않는 고정 Ranking 규칙으로 선택한 Top-10
- Head Union 내부 Oracle Top-10 (진단 상한)

모두 최종 선택 후보 수 K=10으로 비교한다.

Oracle 기반 Union 선택은 실제 배포 가능한 방법으로 취급하지 않는다.

## 9.5 Head Error Redundancy

기존 오류가 있었던 Error Cosine 분석은 수정된 구현을 사용한다.

다음 지표를 확인한다.

- Head Pair Error Cosine
- Head Top-10 Overlap
- Unique Relevant Coverage
- Head 성능 분산

Error Cosine과 Top-10 Overlap은 다른 개념이다.

오류 Cosine이 낮다고 곧바로 검색 품질이 좋다고 결론 내리지 않는다.

산출물:

- `HEAD_LEVEL_RETRIEVAL_QUALITY.csv`
- `HEAD_UNIQUE_RELEVANCE.csv`
- `BEST_SINGLE_VS_ALL5.csv`
- `HEAD_DIVERSITY_VS_UTILITY.csv`

---

# 10. Phase 8 — Oracle-positive Multimodality Analysis

AMER와 같은 Multi-Query Retrieval을 CARTS에 적용할 타당성을 검증하기 위한 분석이다.

핵심 질문:

**미래가 유사한 Oracle Candidate들이 과거 공간에서는 여러 개의 구분된 집단으로 존재하는가?**

## 10.1 Oracle-positive Set

각 Query/Channel에서 Future MSE가 가장 낮은 후보를 구한다.

- Oracle Top-10
- Oracle Top-30

Full Candidate와 P100 각각에서 분석한다.

Top-30은 Multi-modality 진단을 위한 집합이며, 최종 검색 평가의 K=10은 변경하지 않는다.

## 10.2 과거 표현 공간

Oracle-positive 후보들의 다음 표현을 비교한다.

- Raw History
- Delta-last History
- Pearson-normalized History
- Learned V0 Embedding
- Learned V5 Candidate Embedding

모든 Embedding은 기존 학습된 Checkpoint를 통해 생성한다.

미래값을 Embedding 입력으로 사용하지 않는다.

## 10.3 Multimodality Metrics

다음 지표를 계산한다.

- Pairwise Embedding Distance
- Within-group Dispersion
- Between-group Separation
- Cluster Silhouette
- Cluster Stability
- Cluster별 Candidate Count

Cluster 개수를 미리 고정하지 말고 작은 범위에서 탐색하되, 선택 기준은 Validation에서 고정한다.

Oracle Top-10은 샘플이 매우 적으므로 Cluster 수를 과도하게 늘리지 않는다.

## 10.4 대조군

Multi-modality가 실제로 특별한 현상인지 확인하기 위해 다음과 비교한다.

- 동일 크기 Random Candidate Set
- Past-nearest Candidate Set
- Future-nearest Oracle Set
- Future MSE 분포를 맞춘 Control Set (구성이 가능한 경우)

차원 축소 시각화만 보고 Multi-modality가 존재한다고 주장하지 않는다.

## 10.5 Head별 Cluster Coverage

Oracle-positive 후보가 여러 집단을 형성한다면 다음을 확인한다.

- 각 Head가 어떤 Cluster의 후보를 검색하는가?
- 서로 다른 Head가 다른 Cluster를 담당하는가?
- All-5가 Single Head보다 Cluster Coverage를 증가시키는가?
- Cluster Coverage 증가가 실제 RetMSE 개선과 연결되는가?

단순한 Cluster 다양성 증가를 성능 향상의 증거로 사용하지 않는다.

## 10.6 AMER 도입 근거 판단

다음이 함께 확인되는 경우 AMER-inspired Multi-Target Head Learning을 추천할 수 있다.

1. 유용한 후보가 과거 Embedding에서 여러 분리된 집단으로 나타남.
2. Single Head가 특정 집단에 편중됨.
3. Multi-Head가 서로 다른 유용한 집단을 검색할 가능성이 있음.
4. 그 집단을 활용하면 동일 K=10에서 RetMSE 개선 여지가 존재함.

위 근거가 없으면 AMER 도입을 우선 추천하지 않는다.

산출물:

- `ORACLE_POSITIVE_MULTIMODALITY.csv`
- `HEAD_CLUSTER_COVERAGE.csv`
- `CLUSTER_RETRIEVAL_UTILITY.csv`
- `ORACLE_POSITIVE_EMBEDDING_VISUALIZATION.png`

---

# 11. Phase 9 — Past–Future Disagreement Analysis

검색된 후보의 과거 특징과 미래 유사도 사이의 불일치를 분석한다.

## 11.1 Candidate Groups

- Correctly Retrieved Oracle
- Missed Oracle
- False Selected
- Random Candidate

## 11.2 Past-only Features

- Raw Cosine
- Delta-last Cosine
- Pearson
- PAA-16 Cosine
- Diff-1 Cosine
- Trend Difference
- Volatility Difference
- First-difference Variance
- Autocorrelation Similarity
- Spectral Similarity
- Calendar Similarity (적용 가능한 경우)

## 11.3 분석

각 그룹의 특징 분포를 비교한다.

특히 다음을 확인한다.

- False Selected는 Past Similarity가 높지만 Future MSE가 큰가?
- Missed Oracle은 Past Similarity가 낮지만 Future MSE가 작은가?
- 특정 주기성이나 변화율 조건에서 검색 실패가 집중되는가?
- 이러한 패턴이 P100과 Full에서 동일하게 나타나는가?
- V5가 V0보다 특정 유형의 오류를 줄이는가?

High-Regret/Low-Regret Query를 비교한다.

특징 차이는 Query-paired 기준으로 평가하고 효과 크기 및 Confidence Interval을 기록한다.

기존 `past_neighborhood_ambiguity` 관련 코드가 있으면 재사용한다.

관찰된 특징 차이를 인과적인 실패 원인으로 단정하지 않는다.

산출물:

- `PAST_FUTURE_DISAGREEMENT.csv`
- `FALSE_SELECTED_FEATURES.csv`
- `MISSED_ORACLE_FEATURES.csv`
- `FEATURE_EFFECT_SIZES.csv`

---

# 12. Phase 10 — Past-only Learnability Diagnostic

기존 Encoder가 활용하지 못한 정보가 실제 Past Feature에 존재하는지 확인한다.

우선 재학습 없는 진단을 진행한다.

- Past Similarity와 Future MSE 관계
- 과거가 유사한 후보 집합의 Future Dispersion
- Query별 Future Relevance 불확실성
- High-Regret Query에서의 Future Dispersion 변화

필요하다면 작은 Frozen Feature Probe를 보조 진단으로 실행한다.

허용 모델:

- Ridge Regression
- Small 2-layer MLP

Retriever Encoder 파라미터는 변경하지 않는다.

Probe는 다음 원칙을 따른다.

- Train에서만 Future MSE Label을 사용해 학습
- Validation으로 Probe 설정 선택
- Test는 고정 모델 평가에만 사용
- Input Feature는 Past-only
- Candidate ID나 Query ID를 암기하는 부적절한 입력 금지

Probe가 실패하더라도 미래 관련성이 원천적으로 식별 불가능하다고 결론 내리지 않는다.

산출물:

- `PAST_FUTURE_LEARNABILITY.csv`
- `PAST_ONLY_PROBE_RESULTS.csv`

---

# 13. Phase 11 — 실패 원인 분류

모든 결과를 종합한다.

각 Dataset/Horizon/Support/Model에 대해 다음을 보고한다.

| 항목 | 설명 |
|---|---|
| Primary RetMSE | 실제 검색 후보 품질 |
| Oracle Regret | 개선 가능한 검색 품질 격차 |
| Pool Regret | 후보군 제한 손실 |
| Ranking Regret | 후보군 내부 검색 실패 |
| Global Ranking | 전체 미래 관련성 순위 |
| Missed Oracle Rank | 놓친 후보의 실제 순위 |
| Exchange Regret | 잘못된 후보 선택으로 생긴 손실 |
| Multi-Head Utility | 추가 Head의 실질적 기여 |
| Past–Future Ambiguity | 과거 정보의 식별 가능성 |
| Main Bottleneck | 주요 실패 유형 |

가능한 판정:

- `POOL_BOTTLENECK`
- `GLOBAL_RANKING_BOTTLENECK`
- `TOPK_BOUNDARY_BOTTLENECK`
- `MULTIHEAD_REDUNDANCY`
- `MULTI_TARGET_OPPORTUNITY`
- `SUPPORT_NORMALIZATION_SENSITIVITY`
- `PAST_FUTURE_AMBIGUITY`
- `MIXED`
- `INCONCLUSIVE`

한 조건에 여러 원인이 존재할 수 있다.

임의의 단일 원인으로 강제 분류하지 않는다.

각 판정에는 해당 수치와 비교 실험 근거를 명시한다.

---

# 14. Phase 12 — 후속 연구 아이디어 평가

이번 분석의 최종 결과는 단순한 지표 보고가 아니라 **다음에 어떤 학습 구조를 연구하는 것이 가장 타당한지 결정하는 것**이다.

아래 후보를 비교한다.

## Candidate A — Top-K Regret-aware Ranking Loss

관련 분야:

- Learning-to-Rank
- LambdaRank / LambdaLoss
- Hard Negative Mining

검증할 가설:

실제로 선택한 Top-10과 놓친 유용한 후보의 교환 손실에 집중하면 RetMSE를 개선할 수 있는가?

필요한 근거:

- Ranking Regret이 큼
- 중요한 Top-K 순위 오류가 존재함
- Exchange Regret이 특정 오류에 집중됨

기존 Weighted RankNet과의 차이를 명확히 한다.

## Candidate B — AMER-inspired Predictive Multi-Target Retrieval

관련 분야:

- Multi-Query Retrieval
- Multi-Interest Retrieval
- Multi-Target Assignment

검증할 가설:

서로 다른 과거 패턴이 유사한 미래로 이어지는 경우, 여러 Head가 서로 다른 유용한 후보 집단을 담당하도록 학습하는 것이 효과적인가?

필요한 근거:

- Oracle-positive Multimodality 존재
- Single Head Coverage 한계
- Multi-Head가 유용한 후보 집단을 추가로 포착할 가능성

## Candidate C — PRISM-inspired Multi-Perspective Query/Key

관련 분야:

- Multi-Vector Retrieval
- Multi-Perspective Representation

검증할 가설:

Query-only Projection 대신 Query와 Candidate의 여러 관점을 함께 학습하면 미래 관련성 순위화가 개선되는가?

필요한 근거:

- 특정 Past Feature에서의 정보 손실
- Query/Key 표현 비대칭의 유효성
- 여러 관점의 실질적인 검색 기여

## Candidate D — Learned Preselection

관련 분야:

- Dense Retrieval
- Coarse-to-Fine Retrieval
- Learned ANN Retrieval

검증할 가설:

P100 Candidate Coverage를 높이면서 모델이 실제로 유용한 후보를 선택할 수 있도록 학습할 수 있는가?

필요한 근거:

- Pool Regret이 큼
- Full Oracle 후보가 P100 밖에 다수 존재함
- 고정 Checkpoint에서 Support 변경 효과가 검증됨

## Candidate E — Head Combination / Support Calibration

검증할 가설:

Head별 Softmax Normalization과 후보군 변화가 V5 검색 성능 악화의 주요 원인인가?

필요한 근거:

- 동일 Checkpoint에서 Support 변경만으로 Ranking이 크게 변함
- Head별 Probability Mass 변화가 순위 변동과 연결됨
- 동일 P100 후보에서 정규화 범위 변경만으로 RetMSE 차이가 발생함

이 경우 새로운 Head를 추가하기보다 검색 점수 결합 규칙을 개선하는 방향을 우선 검토한다.

## 최종 평가 기준

각 후보에 대해 다음을 기록한다.

- 실험 근거의 강도
- 기존 CARTS 실험과의 중복 여부
- 예상되는 RetMSE 개선 대상
- 계산 및 GPU 비용
- 구현 복잡도
- 실패 가능성
- 논문 Contribution으로 발전할 가능성
- 필요한 최소 Ablation

분석 후 가장 적합한 연구 아이디어 1개를 1순위로 제안한다.

다만 데이터가 충분하지 않으면 추가로 필요한 최소 진단을 명시하고 `INCONCLUSIVE`로 판정한다.

새로운 Loss나 Architecture를 실제로 학습하지 않는다.

---

# 15. 실행 자원 관리

## GPU

물리 GPU 1번만 사용한다.

장치 UUID/PCI Bus ID로 실제 GPU1 사용 여부를 확인한다.

동일 GPU에서 대형 실험을 무분별하게 병렬 실행하지 않는다.

기존 실행 중인 다른 GPU 작업은 중단하지 않는다.

## 순서

1. ETTh1_96
2. ETTh1_192
3. ETTh1_336
4. ETTh1_720
5. Weather_96
6. Weather_192
7. Weather_336
8. Weather_720

각 조건에서 P100과 Full을 모두 완료한 뒤 다음 조건으로 이동한다.

## Full Candidate 처리

대형 `[Query, Candidate, Horizon, Channel]` Tensor를 전체 크기로 생성하지 않는다.

- Query Chunk
- Candidate Chunk
- Channel-aware Batching
- Streaming Top-K
- Streaming Oracle

를 사용한다.

가능하면 Full Candidate 재인코딩과 Query Score 계산에서 기존 검증 코드를 활용한다.

OOM 발생 시 평가 Batch/Chunk만 변경하고 Support 및 Metric 정의는 유지한다.

Full 분석이 실패하면 오류 원인과 Peak GPU Memory를 즉시 결과 파일에 기록한다.

다른 Support의 수치로 Full 결과를 대체하지 않는다.

## 자원 측정

Dataset/Horizon/Support별로 기록한다.

- Wall-clock Seconds
- Oracle Calculation Seconds
- Ranking Analysis Seconds
- Multi-Head Analysis Seconds
- Feature Analysis Seconds
- Peak GPU Allocated GiB
- Peak GPU Reserved GiB
- GPU Utilization Samples
- CPU RAM Peak
- Query/Channel 수
- Candidate Bank Size

---

# 16. 재현성 및 오류 검증

다음 검증을 반드시 통과해야 한다.

1. 기존 공식 Stage1 RetMSE 재현
2. V5 Mean-Mixture 추론 결과 일치
3. Checkpoint Hash 검증
4. Candidate ID Mapping 일치
5. P100 Pool Cache Hash 일치
6. Full Candidate Mask 일치
7. Last-aligned Future Reconstruction 일치
8. Full Oracle RetMSE ≤ P100 Oracle RetMSE
9. P100 Oracle RetMSE ≤ P100 Model RetMSE
10. Full Oracle RetMSE ≤ Full Model RetMSE
11. Total Regret 분해 항등식 성립
12. Exchange Regret과 Ranking Regret 일치
13. Head Union의 실제 Candidate ID 중복 제거 검증
14. 동일 Support에서 공정한 V0/V1/V5 비교
15. Test Future가 Student Score 계산에 유입되지 않음
16. Validation/Test 결과를 사용한 Checkpoint 재선택 없음
17. Weather 데이터 품질 관련 민감도 결과와 공식 결과 분리
18. Query/Channel 집계가 공식 평균과 일치

통계 검증 시 Sliding Window와 Channel 간 상관관계를 고려한다.

가능하면 시간 Block Bootstrap을 사용한다.

단일 seed0 실험으로 일반적인 통계적 우월성을 주장하지 않는다.

---

# 17. 산출물

새 디렉터리에서만 작업한다.

`experiments/retrieval_failure_mechanism02/`

`results/TRACK-RETRIEVAL-FAILURE-MECHANISM02/`

필수 산출물:

### Audit

- `CHECKPOINT_AUDIT.csv`
- `METRIC_REPRODUCTION.csv`
- `IMPLEMENTATION_AUDIT.md`

### Retrieval Quality

- `RETRIEVAL_BASELINE_COMPARISON.csv`
- `V0_V1_V5_COMPARISON.csv`

### Regret / Ranking

- `RETRIEVAL_REGRET_DECOMPOSITION.csv`
- `GLOBAL_RANKING_QUALITY.csv`
- `MISSED_ORACLE_RANKS.csv`
- `TOPK_EXCHANGE_REGRET.csv`

### Support

- `CROSS_SUPPORT_EVALUATION.csv`
- `SOFTMAX_SUPPORT_SENSITIVITY.csv`
- `HEAD_PROBABILITY_MASS.csv`

### Multi-Head

- `HEAD_LEVEL_RETRIEVAL_QUALITY.csv`
- `BEST_SINGLE_VS_ALL5.csv`
- `HEAD_UNIQUE_RELEVANCE.csv`
- `ORACLE_POSITIVE_MULTIMODALITY.csv`
- `HEAD_CLUSTER_COVERAGE.csv`

### Additional Diagnostics

- `WEATHER_RECALL_ANOMALY_AUDIT.md`
- `PAST_FUTURE_DISAGREEMENT.csv`
- `PAST_FUTURE_LEARNABILITY.csv`

### Reports

- `RESOURCE_SUMMARY.csv`
- `FAILURE_ATTRIBUTION_SUMMARY.csv`
- `LOSS_DESIGN_RECOMMENDATION.md`
- `FINAL_REPORT.md`

Query-level 대형 결과는 Parquet 등 압축된 열 기반 형식으로 저장한다.

최종 보고서는 한국어로 작성하고, 각 수치에 Dataset/Horizon/Support/Model/Checkpoint/Split/Seed를 명시한다.

---

# 18. 최종 보고 요구사항

FINAL_REPORT.md에서는 반드시 다음 질문에 답하라.

1. P100과 Full에서 V0/V5의 검색 성능 차이는 무엇인가?
2. V5의 추가 Head는 독립 V1보다 실제로 유리한가?
3. P100 검색 실패에서 Pool Regret과 Ranking Regret은 각각 얼마나 큰가?
4. 좋은 후보가 검색 가능한데도 놓치는 이유는 무엇인가?
5. 놓친 Oracle 후보는 주로 Student Rank 몇 위에 존재하는가?
6. Global Ranking과 Top-K Ranking은 얼마나 일치하는가?
7. Weather Full H96의 낮은 Recall과 낮은 RetMSE는 왜 동시에 나타나는가?
8. 동일 Checkpoint로 P100/Full을 변경했을 때 검색 성능은 어떻게 변하는가?
9. V5의 Head별 Softmax Normalization이 Support Sensitivity를 유발하는가?
10. 다섯 Head는 서로 다른 유용한 후보를 실제로 추가하는가?
11. Oracle-positive 후보들은 과거 Embedding에서 여러 집단으로 나타나는가?
12. 현재 검색 성능을 개선하기 위해 무엇을 먼저 학습해야 하는가?
13. Top-K Regret-aware Loss와 AMER-style Multi-Target Head 중 어느 방향의 실험 근거가 더 강한가?
14. 새로 학습해야 할 최소 실험은 무엇인가?

### 최종 판단

위 분석에 근거하여 다음 중 하나를 선택한다.

- `GO_TOPK_REGRET_LEARNING`
- `GO_MULTI_TARGET_HEAD_LEARNING`
- `GO_QUERY_KEY_MULTIPERSPECTIVE`
- `GO_LEARNED_PRESELECTION`
- `GO_SUPPORT_CALIBRATION`
- `INCONCLUSIVE`

선택 사유를 실제 수치로 증명한다.

단순히 새로운 방법이 흥미롭다는 이유로 GO 판정을 내리지 않는다.

기존 CARTS 실험에서 이미 실패했거나 중복된 방법은 명확히 표시한다.

### 가장 중요한 지시

**이번 실험은 모델을 더 복잡하게 만드는 것이 아니라, 미래와 유사한 후보 검색이 왜 실패하는지를 밝혀 다음 연구의 학습 목표를 결정하는 실험이다.**

특히 P100과 Full 모두에서 동일한 분석을 수행하고, 별도 학습 Checkpoint 비교와 동일 Checkpoint Support Intervention을 명확히 구분해야 한다.

V5에서 Head 수를 늘리는 것이 필요한지, 단일 Projection의 Ranking Loss를 개선하는 것이 필요한지, 후보군 선택을 개선하는 것이 필요한지 실험 결과로 판단하라.

모든 Phase를 완료 가능한 범위에서 실행하고, 결과 누락·실패·근사 평가 여부를 명확히 기록한다.

**학습 없이 검증 가능한 질문부터 해결하고, 검증되지 않은 가설을 연구 결론으로 제시하지 말 것.**
