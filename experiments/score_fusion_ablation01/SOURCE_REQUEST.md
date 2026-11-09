# CARTS — Multi-Head Score Fusion Ablation 실험 요청

## 1. 실험 목적

현재 CARTS의 V5 Multi-Head Retriever는 각 Head에서 계산한 Cosine Similarity를 Candidate 축으로 Softmax한 후, 다섯 Head의 확률분포를 평균하여 최종 Top-10을 선택한다.

기존 실험 D1~D3에서 다음 현상이 관찰되었다.

- D1: 서로 다른 Head가 일부 고유한 유용 후보를 검색한다.
- D2: Head들이 이미 확보한 후보 안에서도 최종 선택에 따른 상당한 Oracle Regret이 남아 있다.
- D3: 동일 Checkpoint와 후보 ID에서도 Head별 Softmax 정규화 범위에 따라 최종 검색 순위가 달라진다.

따라서 **기존 Cosine Similarity와 학습된 Head는 유지하면서, Head별 Score 결합 방식만 변경하면 검색 품질이 개선되는지** 검증하고자 한다.

Repository: `https://github.com/01-hyeok/CARTS`

**이번 실험에서 새로운 Retriever 학습은 금지한다.**

Encoder, Projection Head, Candidate Embedding, 기존 Checkpoint는 모두 고정한다.

Stage2 Forecasting은 수행하지 않는다.

---

## 2. 실험 범위

### Dataset 및 Horizon

- ETTh1: H96, H192, H336, H720
- Weather: H96, H192, H336, H720

공통 설정:

- `seq_len = pred_len = H`
- Seed = 0
- Top-K = 10
- ETTh1: 7개 채널 전체
- Weather: 21개 채널 전체
- 전체 Validation Query 및 전체 Test Query 평가
- 물리 GPU 1번 사용 (`CUDA_VISIBLE_DEVICES=1`)
- 기존 Train/Validation/Test Split 유지
- 기존 RAFT Candidate Mask 유지
- 기존 Data Normalization 및 Delta-last Future Reconstruction 유지

### Candidate Support

각 조건에서 두 종류의 기존 V5 Checkpoint를 사용한다.

1. Full Candidate에서 학습된 V5
2. P100에서 학습된 V5

그리고 각 Checkpoint를 P100과 Full에서 모두 평가한다.

| Training Support | Evaluation Support | 구분 |
|---|---|---|
| Full | Full | Full Native |
| P100 | P100 | P100 Native |
| Full | P100 | Cross-support |
| P100 | Full | Cross-support |

총 8개 Dataset–Horizon × 2개 Training Support × 2개 Evaluation Support = **32개 평가 조합**이다.

P100은 기존 실험에서 사용한 동일 Candidate Pool을 재사용한다. 새로 생성하거나 Test 결과를 이용해 변경하지 않는다.

Full에서는 전체 유효 Candidate Bank를 사용하고, 임의로 Top-100으로 줄이지 않는다.

---

## 3. 기존 코드 및 Checkpoint 감사

실험 구현 전 다음을 확인한다.

- 기존 V5 Checkpoint 경로 및 SHA256
- Validation으로 선택된 기존 Checkpoint 여부
- Full/P100 Candidate Pool 정의
- Head별 Cosine Score 계산 방식
- 기존 `tau_s`
- Head별 Softmax와 Mean-Mixture 구현
- Candidate Mask 및 Stable Top-K 구현
- Future MSE 및 RetMSE@10 계산 방식

다음 기존 코드를 우선 참고한다.

- `utils/mean_mixture_selection.py`
- `utils/full_candidate_bank.py`
- `scripts/train_retriever_pool01.py`
- `scripts/eval_v5_meanmix_checkpoint_selection01.py`
- `scripts/eval_v5_meanmix_checkpoint_selection_pool01.py`
- `experiments/full_head_analysis01/`

필요한 경우 최신 D1~D6 및 Mechanism02 결과의 실제 생성 코드를 찾아 재사용한다.

파일 경로와 구현이 변경되었을 수 있으므로, 위 경로만 신뢰하지 말고 현재 서버의 실제 코드와 결과를 확인한다.

**기존 V5의 공식 Mean-Mixture Top-10과 RetMSE가 재현되지 않으면 새로운 Score 실험을 진행하지 않는다.**

기존 Checkpoint를 재학습하거나 재선택하지 않는다.

---

# 4. 비교할 Score Fusion 방법

Head별 기존 Cosine Similarity를 다음과 같이 정의한다.

\[
s_h(q,i)=\cos(W_hE(X_q),E(X_i))
\]

모든 방법에서 동일한 다섯 Head의 Score를 사용한다.

## E0 — Original Mean-Mixture

현재 V5의 공식 검색 방식이다.

\[
p_h(i)=
\operatorname{softmax}_{i}
\left(s_h(q,i)/\tau_s\right)
\]

\[
S_{E0}(q,i)=\frac15\sum_{h=1}^{5}p_h(i)
\]

최종 Top-10은 \(S_{E0}\) 기준으로 선택한다.

반드시 기존 공식 코드와 동일한 후보 ID를 재현해야 한다.

## E1 — Raw Cosine Mean

\[
S_{E1}(q,i)=\frac15\sum_{h=1}^{5}s_h(q,i)
\]

Head별 Softmax 없이 Cosine Score를 직접 평균한다.

목적: Candidate Softmax를 제거하는 것만으로 검색 순위가 개선되는지 확인한다.

## E2 — Max Fusion

\[
S_{E2}(q,i)=\max_h s_h(q,i)
\]

다섯 Head 중 가장 높은 Cosine Score를 사용한다.

목적: 특정 Head가 발견한 좋은 후보의 강한 Score를 보존하는 것이 유용한지 확인한다.

## E3 — Top-2 Mean

\[
S_{E3}(q,i)=
\frac{s_{(1)}(q,i)+s_{(2)}(q,i)}2
\]

\(s_{(1)},s_{(2)}\)는 가장 높은 두 Head Score이다.

목적: 하나의 Head만 신뢰하는 Max보다 여러 Head의 증거를 이용하는 것이 유리한지 확인한다.

## E4 — Soft-OR / LogSumExp Fusion

\[
S_{E4}(q,i)=
\gamma\log\left(
\frac15\sum_{h=1}^{5}
\exp(s_h(q,i)/\gamma)
\right)
\]

\(\gamma\)가 작으면 Max에 가까워지고, 크면 Mean에 가까워진다.

검증할 온도:

`gamma ∈ {0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0}`

온도는 **각 Native Checkpoint의 전체 Validation RetMSE@10을 기준으로 선택**한다.

그 후 동일 Checkpoint의 Cross-support 평가에서는 Native Validation에서 선택한 gamma를 변경하지 않는다.

Test 결과로 gamma를 재선택하지 않는다.

LogSumExp는 Overflow를 방지하는 안정적인 방식으로 구현한다.

E0~E4 모두 Encoder와 Head Score는 동일하며, **최종 결합 수식만 달라야 한다.**

---

# 5. 실험 구현

새로운 실험 디렉터리를 생성한다.

`experiments/score_fusion_ablation01/`

최종 결과 디렉터리:

`results/TRACK-SCORE-FUSION-ABLATION01/`

기존 코드·Checkpoint·결과 파일을 덮어쓰지 않는다.

가능하면 다음과 같이 구현한다.

- `score_fusion.py`: E0~E4 Score 결합 함수
- `evaluate.py`: 공통 Retriever 평가
- `run_all.sh`: 전체 8조건 실행
- `summarize.py`: 결과표 및 분석 보고서 생성
- `tests/test_score_fusion.py`: 수식 및 재현 검증

동일 Query, 동일 Candidate, 동일 Head Score를 재사용하여 E0~E4를 비교한다.

Full Candidate에서는 Query/Candidate Chunking으로 GPU 메모리 사용을 제한한다.

Full Candidate의 후보를 임의로 축소하지 않는다.

P100과 Full 평가 시 기존 Candidate Mask 및 Candidate ID Mapping을 유지한다.

---

# 6. 필수 평가 지표

Primary Metric:

**Test RetMSE@10 ↓**

모든 방법에서 다음 지표도 측정한다.

- Oracle Gap
- Random-to-Oracle Gap Recovery
- Recall@10
- NDCG@10
- Top-10 Overlap with E0
- Head Union의 유용 후보 유지율
- Query별 RetMSE Win/Tie/Loss
- Worst Query의 검색 성능 악화
- Full/P100 Support 변경 시 순위 안정성
- 추론시간
- Peak GPU Memory

Random과 Oracle은 반드시 같은 Evaluation Support, Query, Channel, Mask에서 평가한다.

Oracle은 실제 Query Future를 사용하는 진단용 최적 기준으로만 사용한다.

Random은 기존 검증된 정의를 재사용하며, 단일 추출 결과를 여러 번의 평균처럼 표시하지 않는다.

Stage2 MSE나 Forecast Fusion 성능은 계산하지 않는다.

---

# 7. 결과 비교표

## 7.1 Full Native 검색 성능

ETTh1/Weather × H96/192/336/720 모두 포함한다.

| Dataset | H | Random | V0 | 독립 V1 | V5(E0) | E1 | E2 | E3 | E4 | Oracle |
|---|---|---|---|---|---|---|---|---|---|---|

모든 값은 전체 Test Query, K=10, Full Candidate에서의 RetMSE이다.

## 7.2 P100 Native 검색 성능

Full Native와 동일한 형식으로 작성한다.

| Dataset | H | Random | V0 | 독립 V1 | V5(E0) | E1 | E2 | E3 | E4 | Oracle |
|---|---|---|---|---|---|---|---|---|---|---|

모든 값은 전체 Test Query, K=10, P100 Candidate에서의 RetMSE이다.

기존 V0·독립 V1의 공식 전체-query 결과가 저장되어 있다면 동일 평가 Population과 설정을 검증한 후 재사용한다.

검증된 결과가 없다면 누락으로 표시하고 임의 수치를 넣지 않는다.

## 7.3 Cross-support 결과

Full-trained → P100 및 P100-trained → Full을 별도 표로 작성한다.

모델과 Score를 변경하지 않고 평가 Support만 바꾸었을 때의 검색 성능을 비교한다.

Native와 Cross-support 결과를 혼합하지 않는다.

---

# 8. 결과 검증

다음 검증을 반드시 통과해야 한다.

1. 기존 E0의 Score와 Top-10 후보 ID가 공식 V5와 일치
2. 기존 E0 RetMSE가 저장된 공식 결과와 일치
3. 다섯 Head의 Raw Cosine Score가 모든 E0~E4에서 동일
4. E1은 Head 점수의 산술평균과 정확히 일치
5. E2는 Head별 최대값과 일치
6. E3는 상위 두 Head 점수의 평균과 일치
7. E4는 gamma가 충분히 작을 때 Max에, 클 때 Mean에 가까워지는지 검증
8. E1~E4에서 공통 Candidate의 Score가 Candidate Support 변경만으로 달라지지 않음
9. Masked Candidate가 Top-10에 포함되지 않음
10. 중복 Candidate 없이 K=10을 선택
11. Full/P100 Candidate ID Mapping 정확성
12. Validation 선택에 Test 정보가 사용되지 않음
13. 전체 Query 및 전체 채널 평가
14. Query Batch/Chunk 크기 변경에 따른 결과 재현
15. Float32/Float64 및 near-tie에 의한 순위 변경 가능성 확인

기존 V5와 E0가 불일치하면 원인을 수정하기 전까지 본 실험의 Test 결과를 공식 성능으로 사용하지 않는다.

---

# 9. 실행 순서

GPU 1번에서 다음 순서로 실행한다.

1. ETTh1 H96 — Full/P100
2. ETTh1 H192 — Full/P100
3. ETTh1 H336 — Full/P100
4. ETTh1 H720 — Full/P100
5. Weather H96 — Full/P100
6. Weather H192 — Full/P100
7. Weather H336 — Full/P100
8. Weather H720 — Full/P100

기존 GPU 작업을 임의로 종료하거나 변경하지 않는다.

각 조건 완료 시 결과를 즉시 저장하고 Checkpoint SHA256, 평가 Query 수, 소요시간, GPU 메모리를 기록한다.

중간 오류가 발생하면 문제 조건과 실패 원인을 기록한다. 이미 완료된 결과는 보존한다.

---

# 10. 최종 분석 요구사항

최종 보고서는 한국어로 작성한다.

단순히 E4가 몇 조건에서 이겼는지만 보고하지 말고 다음 질문에 답한다.

1. 기존 Mean-Mixture보다 좋은 Score Fusion 방법이 존재하는가?
2. Full과 P100 모두에서 개선되는가?
3. 새로운 Score가 독립 V1보다 우수한 조건은 무엇인가?
4. Max가 좋다면 특정 Head의 강한 검색 증거를 보존하는 효과인가?
5. Mean이 좋다면 기존 Candidate Softmax 결합의 단점이 있는가?
6. Soft-OR의 gamma가 조건별로 어떻게 선택되는가?
7. Head Union의 고유 유용 후보가 실제 최종 Top-10에 더 많이 포함되는가?
8. Support-independent Score가 실제 검색 품질 개선으로 이어지는가?
9. 후보군 확대 시 성능이 악화되는 조건은 무엇인가?
10. 이후 Head Calibration이나 Future-MSE 기반 Score 학습을 진행할 가치가 있는가?

각 결과에 대해 **관찰된 사실, 가능한 설명, 검증되지 않은 가설을 구분**한다.

E4 Soft-OR가 가장 좋다는 결론을 미리 가정하지 않는다.

---

# 11. 최종 산출물

다음 파일을 생성한다.

- `FULL_NATIVE_TEST_RETMSE.csv`
- `P100_NATIVE_TEST_RETMSE.csv`
- `CROSS_SUPPORT_TEST_RETMSE.csv`
- `VALIDATION_GAMMA_SELECTION.csv`
- `HEAD_USEFUL_CANDIDATE_RETENTION.csv`
- `QUERY_LEVEL_WIN_LOSS.csv`
- `RESOURCE_SUMMARY.csv`
- `REPRODUCTION_GATE.json`
- `FINAL_REPORT.md`
- `CHATGPT_ANALYSIS_REPORT.md`

최종 보고서에 전체 8조건의 결과와 소요시간을 모두 포함한다.

**최종 목표는 Soft-OR가 반드시 우수하다는 결과를 만드는 것이 아니라, 기존 V5의 Head별 Cosine Similarity를 어떤 방식으로 결합해야 미래가 유사한 Top-10 후보를 더 잘 선택할 수 있는지 검증하는 것이다.**
