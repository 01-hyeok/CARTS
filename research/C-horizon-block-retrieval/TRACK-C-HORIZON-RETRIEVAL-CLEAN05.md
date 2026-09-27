# TRACK-C-HORIZON-RETRIEVAL-CLEAN05 — ETTh1 Closure + Weather Generalization

**Status: COMPLETE. Verdict: NO-GO for Weather_720 seed=0. ETTh1's own
CLEAN04 result downgraded from "success" to Weak/Inconclusive after proper
ranking/bootstrap audit.**

## 1. CLEAN04 결과 요약

CLEAN04 Primary(공통 τ=0.10, block별 독립 step 선택)는 ETTh1_720에서
val +2.696%, test +0.181%(방향 안 뒤집힘)을 보고했다. 이번 CLEAN05는
그 결과를 바꾸지 않고(설정 고정) 생략됐던 ranking/시간축/bootstrap
진단을 보완하고, 동일 방법을 Weather_720에 그대로 적용했다.

## 2. ETTh1 closure audit 결과

`results/TRACK-C-HORIZON-RETRIEVAL-CLEAN05/ETTh1_720/closure_verdict.json`.
Val/test 재현 정확히 일치(val 2.696%, test 0.181% — CLEAN04과 동일),
A0 test 가중합-vs-보고값 일치(오차 2.85e-08, 기준 1e-5 이내 PASS).

## 3. ETTh1 test block별 개선 분해

| Block | A0 대비 절대 delta | 상대 개선 |
|---|---:|---:|
| B1 | +0.005545 | **+1.33%** |
| B2 | +0.012509 | **+2.31%** |
| **B3** | **-0.006979** | **-0.89%(악화)** |

**Test에서 개선된 block은 2/3.** B3(장기 구간)는 test에서 오히려
나빠졌다 — 전체 H720 +0.18%는 B1/B2의 개선이 B3의 악화를 간신히 상쇄한
결과다.

## 4. Binary NDCG 수정 내용

기존 continuous(utility-graded) NDCG는 Recall@10이 3%대인데도 ~0.93이
나와 Oracle Top-10 검색 성능을 반영하지 못했다. `binary_oracle_ndcg_at_10`
(relevance=Oracle Top-10 소속 여부, 0/1)을 별도로 계산해
`utility_graded_ndcg_at_10`과 다른 키로 저장했다. Test 결과:

| Block | Recall@10 (A0→Primary) | Binary NDCG@10 (A0→Primary) | Rank fraction (A0→Primary) |
|---|---|---|---|
| B1 | 0.0398→0.0412 | 0.0417→0.0431 | 0.2172→0.1677 (개선) |
| B2 | 0.0208→0.0235 | 0.0212→0.0240 | 0.1960→0.1467 (개선) |
| **B3** | **0.0147→0.0142(악화)** | **0.0146→0.0142(악화)** | **0.1683→0.1786(악화)** |

## 5. ETTh1 Primary ranking 개선 여부

**B1/B2는 MSE와 ranking이 함께 개선**됐다(genuine specialization).
**B3는 MSE도 ranking도 둘 다 악화**됐다 — B3에서 "Jaccard가 낮아졌다"는
사실이 있더라도 그것이 성공을 의미하지 않는다는 spec의 경고가 정확히
여기서 적용된다.

## 6. ETTh1 test chronological/bootstrap 결과

Chronological 8-bin: **5/8만 개선**(3개 bin 악화) — CLEAN04이 암시한
"방향 일치"보다 훨씬 약한 신호. Moving-block bootstrap(block length
96/240/720, 2000 reps, seed=0): **세 block length 전부 95% CI가 0을
포함**(`ci_excludes_zero_positive: false`) — mean delta는 양수
(0.00119)지만 query window overlap을 고려하면 **통계적으로 0과
구분되지 않는다.**

**결론: CLEAN04이 "test에서 방향이 안 뒤집혔다"고 보고한 것은 사실이지만,
그 개선이 통계적으로 유의하다거나 전체적으로 고른 것은 아니었다.**
spec의 분류 기준(§12)에 따르면 이는 **Weak GO / Inconclusive**에
해당하며, B3는 명확한 contribution 한계다.

## 7. B2 boundary diagnostic

**미실행.** Weather 학습(Stage B~F)이 이번 세션의 GPU 시간 대부분을
차지해 ETTh1 B2의 675 step 이후 확장(800/900/1125/1350)은 이번 라운드에
포함하지 못했다 — 한계로 명시한다. B2가 675에서 정말 수렴했는지는
미확인 상태로 남는다.

## 8~9. Weather Global Transformer / Expert-wise adapter 구조

기존 CLEAN02와 동일한 스크립트(`train_c_horizon_clean02.py --arm
g_transformer`)를 Weather_720에 그대로 재사용(새 코드 없음, `--out_dir`/
`--checkpoints`만 CLEAN05 경로로 지정). Batch_size=128로 실측 타이밍
확인 후(91분/epoch, peak VRAM 31.7GB) 학습, **best_epoch=1**,
val_global_h720_mse=0.729248, test_global_h720_mse=0.463893.

Frozen 후 block adapter(ETTh1과 동일 구조)를 dataset-size-normalized
epoch-fraction 기준 step schedule(steps/epoch=1108,
{0,111,222,366,554,831,1108,1662,2216,3324,4432,5540})로 학습. **τ는
spec 지시대로 전부 0.10 고정, 재탐색하지 않았다.**

## 10. Weather block별 checkpoint

| Block | Best step | Val block MSE |
|---|---|---|
| B1 | 4432 (~4 epoch) | 0.468555 |
| **B2** | **0 (학습 무효)** | 0.688619 |
| **B3** | **0 (학습 무효)** | 0.817268 |

**B2/B3는 어떤 step에서도 초기 상태(=A0)를 넘지 못했다** — ETTh1에서
가장 크게 개선됐던 B2(+5.70%)가 Weather에서는 정반대로 전혀 작동하지
않았다.

## 11. Weather validation 결과

| | Global(A0) | Composite(Primary) | 개선 |
|---|---|---|---|
| **Val** | 0.728452 | 0.727890 | **+0.077%** |

Chronological 8-bin: 6/8 개선이지만 개선폭이 전부 0.5% 미만(최대
0.45%, bin2/bin4는 오히려 악화). **spec의 validation PASS 기준 1번
("2% 이상 개선")부터 충족하지 못한다.**

## 12. Weather Oracle headroom recovery

계산하지 않았다 — 1번 기준이 이미 명백히 실패해 나머지 기준을 확인하는
의미가 낮다고 판단했다(효율성 원칙). 필요시 후속 진단에서 추가 가능.

## 13. Weather test 결과

| | Global(A0) | Composite(Primary) | 개선 |
|---|---|---|---|
| **Test** | 0.463080 | 0.463058 | **+0.005%** |

사실상 0이다.

## 14. Weather block별 개선 분해

Validation 단계에서 이미 B2/B3가 전혀 작동하지 않음이 확인됐으므로
test에서 block별 delta를 별도 계산하지 않았다(B1만 유효한 신호를 가질
수 있고, 그마저도 나머지 두 block의 0-개선에 희석되어 전체 지표에
거의 안 보인다는 점은 validation 결과로 이미 명확하다).

## 15. Ranking과 forecasting MSE의 일치 여부

Weather에서는 별도로 계산하지 않았다 — MSE 자체가 이미 결정적으로
실패했다(validation 2% 기준 미달, test 사실상 0).

## 16. Temporal 안정성

Weather validation chronological bin은 6/8 개선이지만 개선폭이
미미해(0.02~0.45%) 의미 있는 temporal 안정성 주장을 하기엔 이르다.

## 17. ETTh1과 Weather 비교

| | ETTh1_720 | Weather_720 |
|---|---|---|
| Val 개선 | +2.696% | **+0.077%** |
| Test 개선 | +0.181%(bootstrap CI가 0 포함) | **+0.005%** |
| B1 | 개선 | 개선 |
| B2 | **가장 크게 개선**(+5.70%) | **전혀 작동 안 함**(step=0) |
| B3 | 미미한 개선(val), **test 악화** | 전혀 작동 안 함(step=0) |

**두 데이터셋에서 어떤 block이 유용한지가 정반대다.** ETTh1의 B2 성공은
데이터셋 특이적이었고 Weather로 재현되지 않았다.

## 18. 계산량·파라미터·VRAM

Weather G-Transformer: batch_size=128, peak VRAM 31.7GB, ~91분/epoch,
best_epoch=1(6 epoch에서 조기종료). Frozen adapter 학습(3 block, τ=0.10
고정, 최대 5540 step): 전체 캐시 빌드 각 ~49초(train)+수 초(val), step당
비용은 낮아 3개 block 전부 수십 분 내 완료.

## 19. Strong GO / Weak GO / NO-GO 판정

**Weather_720 seed=0: NO-GO.** Validation PASS 기준의 첫 조건(2% 이상
개선)부터 충족하지 못했다(0.077%). B2/B3는 어떤 학습으로도 A0를
넘지 못했다(best_step=0). Test는 사실상 무의미한 수준(+0.005%).

또한 ETTh1 자체도 이번 audit으로 재평가하면 **Strong GO가 아니라 Weak
GO/Inconclusive**로 하향 조정해야 한다(moving-block bootstrap CI가 0을
포함, B3가 test에서 악화, chronological bin 5/8만 개선).

## 20. 추가 seed 진행 여부

**진행하지 않음.** spec §13: "Weather seed=0이 Weak GO이면 사용자 결정을
받은 뒤 실행"이라고 했는데, 이번은 Weak GO도 아니고 명확한 **NO-GO**이므로
seed=1,2를 자동 실행하지 않는다.

## 21. C의 논문 contribution 가능성

현재 시점에서는 **낮다.** Horizon-block retrieval 아이디어는 (a) Oracle
레벨에서는 강한 근거가 있고(§1 배경, block-specific complementarity
25~40%), (b) 학습된 형태로는 ETTh1에서만, 그것도 B1/B2 두 block에서만,
통계적으로 불확실한 수준으로 재현되며, (c) Weather로는 전혀 확장되지
않는다(B2/B3 완전 무효). 데이터셋 간 "어느 block이 유용한가"가 정반대로
나타난다는 사실은, 지금 구조(zero-init residual adapter + block-wise
KL)가 진짜 horizon-specific 신호를 안정적으로 추출한다기보다 데이터셋
특이적 우연에 더 가깝다는 것을 시사한다.

## 다섯 가지 핵심 질문에 대한 답

- **Horizon별 Oracle 후보는 Global 후보와 다른가?** 예 (Oracle Jaccard
  0.03~0.06, 두 데이터셋 모두 사전 확인됨).
- **학습된 expert가 Oracle ranking에 가까워졌는가?** ETTh1의 B1/B2만
  그렇다(Recall/NDCG/rank fraction 전부 개선). B3와 Weather 전체는 아니다.
- **Ranking 개선이 실제 block MSE 개선으로 이어졌는가?** ETTh1 B1/B2만
  그렇다.
- **Expert별 checkpoint가 ETTh1 외 데이터에서도 유효한가?** **아니다** —
  Weather에서 B2/B3는 완전히 무효(step=0).
- **Long-horizon B3에서도 expert가 실제 가치를 가지는가?** **아니다** —
  ETTh1 test에서 악화, Weather에서 완전 무효. 두 데이터셋 모두에서 B3는
  이 구조의 명확한 한계다.
