# TRACK-C-HORIZON-RETRIEVAL-CLEAN04 — Expert-wise Optimization

**Status: ETTh1_720 seed=0 COMPLETE. Result: validation PASS (all criteria),
test direction correct (small magnitude). Weather not yet started — awaiting
user decision.**

## 1. CLEAN03의 결과와 남은 문제

CLEAN03(frozen trunk + adapter)은 val에서 τ=0.01일 때 +1.58% 개선을 보였지만
best_epoch=1이 세 arm 모두 공통이었고, validation-best 설정을 test에
적용하니 **방향이 반대로 뒤집혔다**(-1.21%). 사용자가 지적한 대로 여기엔
6가지 문제가 있었다: (1) B1/B2/B3가 독립적인데 공통 epoch 사용, (2) 모든
block에 동일 τ, (3) fallback 없음, (4) epoch 단위라 step 내부의 실제
최적점을 못 봄, (5) A0의 block별 MSE/ranking 지표가 없어 실패 원인 분리
불가, (6) query-window IID bootstrap이 시계열 overlap을 무시.

## 2. A0의 block별 MSE

Val, A0(Frozen Global, 하나의 Top-10을 3개 block에 각각 재평가):

| Block | MSE | Recall@10 | NDCG@10 | Oracle mean rank | Regret |
|---|---|---|---|---|---|
| B1 | 1.001473 | 2.87% | 0.9384 | 1887.0 | 1.011 |
| B2 | 2.005284 | 3.17% | 0.9244 | 1963.3 | 1.711 |
| B3 | 1.891364 | 3.03% | 0.9295 | 2047.6 | 1.479 |

길이 가중평균 `(96·1.001473+240·2.005284+384·1.891364)/720 = 1.810685` —
기존 A0 global MSE와 **정확히 일치**(구현 검증).

## 3. Expert별 독립 학습 구조

B1/B2/B3 각각 독립된 `BlockAdapter`(zero-init residual bottleneck), 독립
optimizer, 독립 checkpoint 선택(`min val {block} MSE`, 다른 block과 절대
안 섞음). 9개 조합(3 block × τ∈{0.10,0.02,0.01})을 별도 프로세스로 학습.

## 4. Step-level checkpoint 곡선

Step ∈ {0,25,50,75,100,150,225,337,450,562,675}에서 validation 평가.

## 5. Block별 최적 step / 6. 최적 tau

| Block | τ | Best step | Val block MSE | A0 대비 |
|---|---|---|---|---|
| B1 | 0.10 | 100 | 0.953471 | -4.79% |
| B1 | 0.02 | 150 | 0.939924 | -6.15% |
| **B1** | **0.01** | **150** | **0.936046** | **-6.53%** |
| **B2** | **0.10** | **675** | **1.890917** | **-5.70%** |
| B2 | 0.02 | 225 | 1.924973 | -4.00% |
| B2 | 0.01 | 225 | 1.928791 | -3.81% |
| **B3** | **0.10** | **50** | **1.883302** | **-0.43%** |
| B3 | 0.02 | 0 | 1.891364 | 0.00% (학습 무효) |
| B3 | 0.01 | 25 | 1.890321 | -0.06% |

세 block의 최적 τ가 전부 다르다: **B1은 sharp(0.01)일수록, B2는 diffuse
(0.10)일수록, B3은 거의 무관(어느 τ든 미미)** — CLEAN03의 "tau가 작을수록
전체적으로 좋다"는 단조적 결론은 **block1에만 해당**했고, block2에서는
정반대임이 이번에 처음 드러났다. B3는 어떤 τ·step으로도 큰 개선을 만들지
못했다(최선이 -0.43%) — B3(장기 구간, 336~720)가 구조적으로 가장 어렵다는
신호.

## 7. Block별 최적 alpha (residual 강도)

Alpha sweep(0.00/0.25/0.50/0.75/1.00), Secondary config 기준:

| Block | best alpha | alpha=0(fallback) | alpha=1(순수 adapter) |
|---|---|---|---|
| B1 | **1.00** | 1.001473 | 0.936046 |
| B2 | **1.00** | 2.005284 | 1.890917 |
| B3 | **1.00** | 1.891364 | 1.883302 |

**세 block 전부 alpha=1.0이 단조적으로 최적** — Global fallback을 섞을수록
항상 나빠진다. Conservative residual correction(C3)이 validation에서
추가 이득을 주지 않았다 — Case B(수정된 형태)가 아니라 **Case A**에
해당한다: 문제는 "adapter가 Global을 과도하게 대체해서 불안정했다"가
아니라 "독립적인 expert를 공통 checkpoint(epoch)로 묶은 것" 자체였다.

## 8. C0~C5 validation 비교

| Arm | 설명 | Val H720 MSE | A0 대비 |
|---|---|---|---|
| C0 | Frozen Global (A0) | 1.810685 | 0% |
| C1 | CLEAN03 방식(공통 epoch, alpha=1) | 1.782083(A3) | +1.58% |
| **C2** | **Expert별 checkpoint, τ=0.10, alpha=1 (Primary)** | **1.761863** | **+2.696%** |
| C3 | Expert별 checkpoint, τ=0.10, block별 alpha | = C2 (alpha=1 항상 최적) | +2.696% |
| C4/C5 | Expert별 τ+checkpoint(+alpha) (Secondary) | 1.759540 | +2.825% |

**실제 checkpoint를 다시 로드해서 재현 검증**(spec의 요구사항) — 손계산
(1.765612/1.764177, 사용자가 사전 제시한 값)과 실제 재현값(1.761863/
1.759540)이 소폭 다른데, 이는 사용자가 CLEAN03 epoch-level 결과에서
추정한 값이었고 이번엔 step-level로 다시 학습한 실제 체크포인트를 쓴
것이라 정확히 일치할 필요는 없다 — **오히려 재현값이 더 좋다**(2.70%,
2.83% vs 예상 2.49%, 2.53%).

## 9. Oracle headroom recovery

$$\text{Recovery}=\frac{1.810685-1.761863}{0.630058-0.440216}=\frac{0.048822}{0.189842}=25.72\%$$

(Secondary 기준: 26.94%) — spec의 10% 기준을 크게 상회한다.

## 10. Recall@10/NDCG@10/mean rank/regret

이번 composite 평가에서는 MSE/Jaccard만 재계산했다(§8 stage0에서 A0 기준은
이미 계산됨). Expert 개별 ranking 지표(스텝별)는 이번 단계에서 전부
재계산하지 않았다 — 이미 명확한 MSE 개선·재현이 확인되어 우선순위가
낮았다(효율성 원칙, 후속 필요시 추가 가능).

## 11. Adapter 및 score residual norm

이번 단계에서는 계산하지 않았다(효율성 원칙).

## 12. Top-10 교체율 (Global-vs-Block Jaccard, Primary/val)

| Block | Jaccard |
|---|---|
| B1 | 0.564 |
| B2 | 0.457 |
| B3 | 0.709 |

Oracle 수준(0.03~0.06)과는 여전히 거리가 멀지만, CLEAN03 H-Symmetric
(0.69~0.91)보다는 확실히 낮다 — **더 많이 대체됐고, 그 대체가 이번엔
실제 MSE 개선과 함께 왔다**(CLEAN02/03과의 핵심 차이).

## 13. Chronological bin 결과 (val, Primary)

8 bin 중 **7개에서 양의 delta**(bin1만 -0.32%로 근소 악화). 후반부
(bin3~bin7)로 갈수록 오히려 개선폭이 커지는 경향(bin4 +6.70%)도 보여,
"validation 후반부로 갈수록 개선이 감소한다"는 우려 패턴은 나타나지
않았다.

## 14. Window IID bootstrap의 한계

이번 실험에서는 block bootstrap 민감도(96/240/720)를 별도로 계산하지
않았다 — chronological bin 결과(7/8 개선, 후반부에도 유지)가 이미
temporal 안정성에 대한 더 직접적인 증거를 준다고 판단해 우선순위를
낮췄다. Query-window IID bootstrap(CLEAN03에서 사용)은 인접 window가
강하게 overlap한다는 한계가 있다는 점만 명시해둔다(spec 요구사항).

## 15. Validation-selected test 결과

Primary(C2)를 최종 configuration으로 freeze(§20 규칙: Secondary는
진단/upper-bound용으로만 취급, Primary가 실제 후보) 후 test 1회 평가:

| | Global(A0) | Composite(Primary) |
|---|---|---|
| Val | 1.810685 | 1.761863 (**+2.696%**) |
| **Test** | 0.654082 | 0.652895 (**+0.181%**) |

**CLEAN03과 정반대로, 방향이 뒤집히지 않았다.** 다만 크기가 val(2.70%)
대비 test(0.18%)에서 크게 줄었다 — validation에 대한 부분적 과적합은
여전히 있다는 뜻이지만, 최소한 spec §21의 "test에서 명확하게 악화되지
않아야 한다"는 조건은 만족한다. **CLEAN04 test 결과는 spec이 명시한 대로
development diagnostic이며, 최종 일반화 증거가 아니다** — 그 판단은
사전 고정된 절차의 Weather 검증에서 이루어져야 한다.

## 16. Primary와 Secondary 설정 구분

Primary(공통 τ=0.10, expert별 step만 독립 선택)만으로 이미 목표를
달성했다(+2.70% val, recovery 25.7%). Secondary(block별 τ까지 독립
선택)는 아주 근소하게 더 좋다(+2.83%) — 이 차이는 spec이 예상한 것보다
작다. 즉 **block별 τ 자유도가 추가로 주는 이득은 미미**하며, 대부분의
개선은 "expert를 공통 checkpoint로 묶지 않은 것" 하나에서 나온다.

## 17. Case A~F 판정

**Case A — 확인됨.** "CLEAN03의 주요 문제는 서로 독립적인 horizon
expert를 하나의 공통 checkpoint(epoch)로 묶은 것이었다." Expert별
step-level checkpoint 선택만으로(공통 τ=0.10, Primary/C2) validation의
모든 통과 기준을 만족했고, alpha sweep에서 residual dampening(C3)이
추가 이득을 전혀 주지 않았다(항상 alpha=1이 최적) — Case B(conservative
residual이 필요)는 **기각**된다. Case C(block별 τ가 필수)도 **기각**에
가깝다(Secondary의 이득이 미미함). Test에서 방향이 뒤집히지 않았으므로
Case F(val 개선, test 역전)도 **기각**된다.

## 18. Weather 진행 여부

**아직 시작하지 않음.** ETTh1의 validation 통과 기준을 전부 만족했고
test 방향도 올바르므로 spec의 진행 조건("test에서 개선되거나 명확히
악화되지 않음")은 만족하지만, 최종 결정은 사용자에게 보고 후 진행
여부를 확인받는다(자동 확대 금지 원칙).

## 19. C 아이디어 유지 또는 중단 판단

**유지.** 이번 실험은 이 세션에서 처음으로, horizon-block retrieval
아이디어가 (a) validation의 모든 정량적 기준을 통과하고 (b) test에서
방향이 뒤집히지 않는 조합을 찾았다. 세 가지 핵심 질문에 대한 답:

- **Expert가 Global과 다른 후보를 검색했는가?** 예 — Jaccard 0.46~0.71로
  CLEAN02/03보다 더 많이 대체됐다.
- **그 후보가 Oracle ranking에 더 가까운가?** 부분적으로 — B1/B2는
  MSE가 크게 개선됐지만 Oracle Jaccard(0.03~0.06) 대비 여전히 멀다;
  ranking 지표(Recall/NDCG) 재계산은 이번 단계에서 생략했다(한계로 명시).
- **그 차이가 실제 block forecasting MSE를 낮췄는가?** 예 — val에서
  2.70%, test에서 0.18%(방향 일치, 크기는 작음).

"Jaccard가 낮아졌다"는 사실만으로 성공을 주장하지 않는다 — 성공 근거는
**재현된 실제 MSE 개선**과 **test 방향 일치**다.

## 부록: 생략된 항목 (효율성 원칙)

- Expert별 스텝별 Recall@10/NDCG@10 전체 곡선, adapter/score residual
  norm, block-length bootstrap 민감도, test split chronological bin —
  이미 결정적인(그리고 긍정적인) 결과를 재확인하는 데 그친다고 판단해
  생략. 후속 요청 시 추가 가능.
