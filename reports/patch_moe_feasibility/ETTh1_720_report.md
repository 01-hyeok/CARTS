# ETTh1_720 Patch-level MoE Feasibility — Interim Report

**Status: NOT FINAL.** Stages 0-3 (feasibility funnel through root-cause
diagnosis of the router failure) are complete. Stage 4 (soft-utility
router) and Stage 5 (direct mixture-loss router) have not been run yet —
paused pending a decision, since Stages 1-3 already established that the
router's failure is very unlikely to be a feature-quality problem. Weather_720
has not been started (per explicit instruction: ETTh1 first).

## 1. 한 문장 결론

Query별 patch complementarity와 큰 이론적 오라클 헤드룸(15~19%)은 실재하지만,
그 헤드룸을 과거 정보만으로 회수하려는 첫 시도(hand-crafted feature + hard-winner
router)는 실패했고, 원인은 feature 품질이나 candidate-support 불일치가 아니라
**train과 val/test 구간 사이의 진짜 시간적 분포 변화(temporal regime shift)**로
확인됐다.

## 2. 이전 실험과 이번 업데이트의 차이

이전 실험(`TRACK-A-PATCH-RETRIEVAL-EXPERT01` Phase A)은 "평균적으로 어떤 patch
크기가 가장 좋은가"를 물었다. 이번 업데이트는 그 4개 checkpoint를 재학습 없이
재사용해서 "query마다 좋은 patch가 달라지고, 그게 과거 정보로 예측 가능한가"를
묻는다 — 비교 단위가 arm 평균에서 query 단위로, 목적이 "best arm 선정"에서
"Patch-level MoE go/no-go 판정"으로 바뀌었다.

## 3. 기존 실험 재사용 여부

4개 patch arm(native_p16/p24/p48/p120) 체크포인트는 전부 기존 Phase A 산출물을
**그대로 재사용**했고(재학습 없음), config_fingerprint 대조 결과 patch_len 외
모든 설정(seed=0, batch_size=32, top_k=10, tau_t=0.02, tau_s=0.1, loss 정의,
checkpoint 기준)이 동일함을 확인했다(공정 비교 가능). 이번에 새로 만든 건 전부
**평가/분석 코드**뿐이다(재학습 코드 없음):
`diag_patch_moe_feasibility01.py`, `diag_patch_moe_fusion01.py`,
`train_patch_moe_router01.py`, `diag_patch_moe_deployment_matched01.py`,
`diag_patch_moe_d1_block_random01.py`.

## 4. Patch arm별 평균 성능 (retMSE@10)

| Split | native_p16 | p24 | p48 | p120 (Best Fixed) |
|---|---|---|---|---|
| val | 2.2601 | 2.1444 | 2.0479 | **2.0443** |
| test | 0.9717 | 0.9928 | 0.9885 | **0.9692** |
| train (raw) | **0.7024** | 0.9597 | 0.9460 | 0.9559 |

## 5. Query별 Winner 분포

| Split | native_p16 | p24 | p48 | p120 |
|---|---|---|---|---|
| val | 34.3% | 22.5% | 22.9% | 20.4% |
| test | 31.3% | 17.6% | 23.8% | 27.4% |
| **train (raw)** | **73.2%** | 9.9% | 8.6% | 8.3% |

Tie 비율은 모든 split에서 사실상 0(margin=0%인 query 0건). native_p16의 train
승리는 근소한 차이의 누적이 아니라 **중앙값 12.8%의 큰 relative margin**을 가진
안정적인 승리다(val 중앙값 8.05%보다 오히려 큼) — §8에서 상세.

## 6. Scale-selection Oracle Headroom

| Split | Best Fixed | Oracle | Gain |
|---|---|---|---|
| val | 2.0443 | 1.6636 | **18.62%** (95% CI 0.367~0.395, paired diff) |
| test | 0.9692 | 0.8223 | **15.15%** |
| train (raw) | 0.7024 | 0.6755 | 3.82% |

val/test에서 통계적으로 유의미한 헤드룸이 존재하고 서로 재현됨(방향·크기 일관).
train에서는 native_p16이 이미 거의 최적이라 헤드룸 자체가 작다 — train과
val/test가 근본적으로 다른 문제라는 첫 신호.

## 7. Patch 간 Complementarity (Top-10 Jaccard overlap, val)

| Pair | Jaccard | Top-1 일치율 |
|---|---|---|
| native_p16 vs p120 (최소) | 0.063 | 2.5% |
| p24 vs p48 (최대) | 0.176 | 9.4% |

최대 overlap도 17.6%뿐 — patch들이 실질적으로 다른 후보를 찾는다(단순 중복 expert
아님).

## 8. Fixed-budget Fusion (학습 없음)

| 방법 | val 개선 | test 개선 |
|---|---|---|
| Uniform 확률 평균 | **-7.85%** (손해) | +1.25% |
| Reciprocal Rank Fusion | +0.38% | +1.27% |
| Mean-rank Fusion | **+4.21%** | **+2.29%** |

Mean-rank fusion이 유일하게 일관된 개선을 보였지만, 오라클 헤드룸의 15~23%만
회수 — 학습 없는 단순 결합으로는 헤드룸 대부분을 못 건진다.

## 9. Past-only Router (hard-winner CE)

**실패.** Hand-crafted feature(mean/std/trend/recent-change/FFT band energy/ACF)
+ 소형 MLP router를 train query의 winner label로 학습:

| Split | Router hard acc | Majority baseline | retMSE@10 개선 | 회수된 헤드룸 |
|---|---|---|---|---|
| val | 34.26% | **34.26% (동일)** | **-10.55%** | **-56.7%** |
| test | 31.26% | **31.26% (동일)** | -0.26% | -1.7% |

Confusion matrix 확인 결과 router가 **모든 query에서 예외 없이 arm 0(native_p16)만
예측**했다 — 입력을 전혀 안 보고 train-majority로 collapse. Best Fixed보다도
나쁜 결과.

## 10. 원인 분리 — Candidate-support Mismatch (기각) vs Temporal Shift (지지)

### 10.1 후보 확인: candidate-support 비대칭 발견
`utils/relation_memory.py::RelationMemorySampler`(mask_mode='raft') 추적 결과,
**train query는 자기 주변 최대 2,879개 후보가 self/overlap exclusion으로 제외**
(실질 후보 ~4,322개)되는 반면, **val/test query는 제외가 전혀 없이 전체 7,201개를
다 사용**한다는 구조적 비대칭을 확인했다.

### 10.2 Deployment-matched 재실험 → 이 가설 기각
Train 타임라인 뒷부분(20%)을 val/test와 동일한 방식(strict causal, 제외 없음,
평균 5,746개 후보)으로 재평가:

| | 원래 train | Deployment-matched |
|---|---|---|
| native_p16 승률 | 73.2% | **72.1% (거의 불변)** |

**candidate-support를 완전히 맞췄는데도 결과가 안 바뀜 → 이 가설 기각.**

### 10.3 D1 (block-random holdout, gap≥1440, 3-seed) → temporal shift 지지
11개 블록(720 windows)을 무작위로 train/holdout 배정(고정 개수, seed당 2개
홀드아웃 블록), 모든 query-candidate 쌍에 gap≥1440(=seq_len+pred_len) 강제:

| Seed | native_p16 승률 (block-random 내부 held-out) |
|---|---|
| 0 | 67.9% |
| 1 | 75.9% |
| 2 | 64.6% |
| **평균** | **69.5%** |

같은 train era 안에서는 **진짜 held-out 샘플(겹침 완전 차단)로도 65~76%가
안정적으로 재현** — val/test(31~34%)와 전혀 겹치지 않는 범위.

### 10.4 Block-vs-time winner-rate curve
전체 타임라인(train→val→test, 720-window block)을 이어붙여 native_p16 승률을
그리면(`winner_rate_timeline.png`), train 11개 블록은 66.9~85.7%로 계속 높다가
**train/val 경계(block 10→11)에서 85.7%→45.3%로 한 번에 급락**, 이후 val/test
내내 14~45%를 오간다. 점진적 drift가 아니라 **경계에서의 급격한 전환**.

### 10.5 결론
> Candidate-support mismatch는 deployment-matched 실험에서 기각되었다. 또한
> block-random holdout(3-seed 반복, gap≥L+H 강제)에서도 train-era 내 native_p16
> 우세가 안정적으로 유지된 반면(65~76%), chronological val/test에서는 그 관계가
> 크게 약화되었다(31~34%). block별 winner-rate 곡선은 이 전환이 train/val
> 경계에서 급격하게 일어남을 보여준다. 이는 Router 실패의 주된 원인이 feature
> 자체의 in-distribution 일반화 부족보다는 시간에 따른 feature–utility 관계
> 변화(temporal regime shift)일 가능성을 강하게 지지한다.

## 11. 계산량

모든 분석은 4개 frozen checkpoint의 inference-only 재평가이며 재학습 없음.
GPU1에서 각 (arm × split) 조합당 수 분~수십 분, 총 수 GB 수준의 메모리만 사용
(멀티 프로세스 동시 로드 시에도 4GB 이하로 확인됨).

## 12. 발견된 코드/설정 문제
- 없음(신규 버그 미발견). candidate-support 비대칭은 버그가 아니라 기존
  mask_mode='raft'의 의도된 anti-leakage 설계임을 코드 추적으로 확인.

## 13. Patch-level MoE Go/No-Go 판정 (잠정)

**Stage 4/5(soft router, direct mixture loss) 미실행 상태의 잠정 판정.**
Stage 1-3의 증거(oracle headroom 유의미, complementarity 존재, fusion 미미,
hard-winner router 완전 실패 + 원인이 temporal shift로 특정됨)를 종합하면
**B(단순 fusion 유지) 또는 C(Single Patch 유지)에 가깝다.** Router 자체가
"학습이 안 된 것"이 아니라 "학습해야 할 대상(feature-utility 관계)이
train/val/test 사이에서 실제로 바뀌는 것"이므로, Stage 4(soft utility loss)나
더 정교한 feature(learned embedding, post-retrieval confidence)를 추가해도
근본적인 temporal shift 자체는 해결되지 않을 가능성이 높다 — 다만 이건
아직 직접 검증되지 않은 추정이며, Stage 4/5를 실제로 돌려봐야 확정된다.

## 14. Weather에서 확인해야 할 사항
- ETTh1과 마찬가지로 train/val/test 간 winner 분포·oracle headroom 격차가
  나타나는지 (temporal shift가 ETTh1만의 특이 현상인지, 일반적 패턴인지 확인)
- Best Fixed Patch가 ETTh1과 다른지
- (ETTh1이 B/C 판정이므로) 대규모 신규 학습 없이 기존 완료분만 분석
