# Experiment E2.5 — Fixed Fusion & Score Calibration Diagnostic (Weather_720)

**Status: CLOSED. Final verdict: E-C — Shared Single Encoder 유지, Experiment E 전체 중단.**

## 1. E2 결과 요약

E2(Component-Specialized Multi-Subspace Retrieval Encoder, `train_experiment_e2_multisubspace01.py`)를
Weather_720에서 학습(scratch, `relation_encoder_type=transformer`, patch_len=16, d_model=128,
subspace 4등분 D/4=32, seed=0, batch_size=128, 6 epoch에서 조기종료, best_epoch=1)한 결과:

- val: E-Shared=0.715896, E-Equal=0.719047 (**-0.44%**, 악화)
- 동일 checkpoint(`checkpoints/experiment_e_e2_multisubspace01/Weather_720/checkpoint.pth`)를
  이번 E2.5 전체에서 **재학습 없이 그대로 재사용**했다.

## 2. 기존 specialization 해석 오류 수정

기존(E2 직후) 보고에서 "4개 subspace 전부 동일하게 Local teacher에 수렴했다"고 row-wise로
해석한 것은 잘못이었다. Column-wise로 다시 보면:

| Teacher | 최적 subspace | 판정 |
|---|---|---|
| Shared | Shared (1.049) | 성공 (약함) |
| Local | Local (0.224) | 성공 (약함) |
| Trend | Trend (0.531, 근소하게 Shared의 0.533보다 우수) | 성공 (매우 약함) |
| Seasonal | Shared (1.421) — Seasonal 자신(1.433)은 아님 | **실패** |

## 3. Column-wise specialization 결과

Shared/Local/Trend는 자기 teacher column에서 근소하게 최적(margin이 매우 작음, 특히
Trend는 0.533 vs 0.531로 거의 tie 수준), Seasonal만 명백히 실패. 사용자 지적이 정확했다 —
"모든 subspace가 동일 ranking"이라는 이전 결론은 과장이었고, **약한 방향성의 specialization은
존재**한다. 다만 이 약한 specialization이 실제 retrieval 성능 개선으로 이어지는지가 이번
E2.5의 핵심 질문이었다.

## 4. Full-D baseline 공정성

Weather_720용으로 완전히 matched된(동일 seed/epoch/candidate mask, transformer branch,
patch_len=16, Shared-teacher-only) Original Full-D baseline checkpoint는 **존재하지 않는다**
(`TRACK-A-PATCH-RETRIEVAL-EXPERT01`의 Weather native_p16 arm은 사용자 지시로 학습 중단되어
미완성). §9의 결과가 이미 "모든 fusion이 E-Shared보다도 나쁨"으로 결정적이었으므로, B0
재학습(추가 GPU 수 시간)은 **진행하지 않았다** — Shared조차 못 이기는 fusion이 더 강한 Full-D
baseline은 당연히 못 이긴다. 이 판단은 §16 효율성 원칙("불필요한 encoder inference를
반복하지 않는다")에 따른 것이며, 최종 판정 문구 강도는 이 미비를 반영해 낮춘다(§13).

## 5. Score scale 및 calibration 분석

Train-global 통계(채널별)에서 컴포넌트 간 scale 차이가 실재함을 확인했다(예: 채널1
shared mean=0.373/std=0.290 vs trend mean=0.817/std=0.132) — raw cosine score를 그대로
결합하면 채널·컴포넌트별로 다른 스케일이 섞인다는 가설(가설 2)은 근거가 있었다.

## 6/7/8. Fixed Fusion 결과 (Raw / Train-global / Query-wise)

Val, 25-arm grid(G + G+L/G+T/G+S/G+LT/G+LTS × α∈{0.50,0.75,0.90,0.95}), 상위 6개 arm:

**C0 Raw Cosine**
| Arm | Val MSE | vs Shared |
|---|---|---|
| G (Shared) | 0.715896 | 0.000% |
| G+L_a0.9 | 0.716042 | -0.020% |
| G+S_a0.95 | 0.716082 | -0.026% |
| G+LTS_a0.9 | 0.716136 | -0.034% |
| G+LTS_a0.95 | 0.716172 | -0.039% |
| G+T_a0.95 | 0.716195 | -0.042% |

**C1 Train-global calibration**
| Arm | Val MSE | vs Shared |
|---|---|---|
| G (Shared) | 0.715896 | 0.000% |
| G+LT_a0.95 | 0.716202 | -0.043% |
| G+LTS_a0.95 | 0.716380 | -0.068% |
| G+T_a0.95 | 0.716503 | -0.085% |
| G+LT_a0.9 | 0.716547 | -0.091% |
| G+L_a0.95 | 0.716574 | -0.095% |

**C2 Query-wise calibration**
| Arm | Val MSE | vs Shared |
|---|---|---|
| G (Shared) | 0.715896 | 0.000% |
| G+L_a0.95 | 0.715975 | -0.011% |
| G+LT_a0.95 | 0.716059 | -0.023% |
| G+LTS_a0.95 | 0.716407 | -0.071% |
| G+T_a0.95 | 0.716561 | -0.093% |
| G+L_a0.9 | 0.716635 | -0.103% |

**세 calibration 전부 20개 non-shared arm 중 shared를 이긴 arm 0개.** 가장 근접한 arm도
alpha=0.9~0.95(즉 거의 shared 단독)에서만 근소하게 덜 나쁠 뿐, 어떤 calibration도 fusion을
살리지 못했다 — **가설 2(scale 문제) 기각, 가설 1(complementary information 없음) 지지.**

## 9. Alpha sensitivity

세 calibration 모두에서 최적점은 항상 alpha가 가장 큰 쪽(0.9~0.95, shared 비중 최대)에
몰려 있다 — non-shared 성분을 조금이라도 더 섞을수록 성능이 단조적으로 나빠지는 패턴이며,
"최적 alpha가 중간값"인 경우는 전혀 없었다. 이는 non-shared 성분이 순수하게 노이즈에
가깝다는 신호다.

## 10. Seasonal 포함/제외 비교

Seasonal을 제외한 G+LT과 포함한 G+LTS를 비교해도 방향이 바뀌지 않는다(C1에서 G+LT_a0.95가
-0.043%로 최선의 fusion arm이지만 여전히 shared보다 나쁨; G+LTS는 항상 G+LT보다도 나쁨).
Seasonal이 "특히 해롭다"는 뚜렷한 추가 증거는 있으나(§3에서 확인한 specialization 실패와
일관), Seasonal을 빼도 fusion 자체가 살아나지는 않는다 — 문제가 Seasonal 하나만의 문제가
아니라는 뜻이다.

## 11. Best fixed arm

Val 선택 규칙(§9)을 그대로 적용하면 세 calibration 모두 **선택된 arm은 G(Shared 단독)
자신**이다 — non-shared arm 중 어느 것도 개선을 보이지 않아 tie-break 규칙까지 갈 필요가
없었다.

## 12. E-Shared 및 Full-D baseline 비교

선택된 arm이 G 자신이므로 "E-Shared 대비 개선"은 정의상 0%다. Full-D baseline과의 비교는
§4에서 설명한 대로 matched checkpoint 부재로 수행하지 않았다.

## 13. Paired cluster bootstrap

선택된 arm(=G)과 Shared(=G) 사이의 비교이므로 cluster bootstrap(base query window
resampling, 10,000 reps)은 정의상 diff=0, CI=[0,0]이다(세 calibration 전부 동일). 의미
있는 bootstrap 비교는 "fusion이 shared를 이긴 적이 없다"는 §6-8의 raw 숫자 자체이며, 그
결과가 이미 전 alpha·전 arm·전 calibration에서 일관되게 shared보다 나쁘다는 것 자체가
강력한 evidence다(개별 arm 간 CI를 20개씩 따로 도는 것은 이미 결정적인 결과에 추가 정보를
주지 않는다고 판단해 생략함 — §16 효율성 원칙).

## 14. Validation chronological stability

선택된 arm이 G 자신이라 early/late 비교도 정의상 0% (표에 그대로 기록됨,
`chronological_early_late_val*.json`). 10-block 세부 chronological 분석은 이미 결정적인
val 결과를 재확인하는 것 이상의 의미가 적다고 판단해 생략했다(§16).

## 15. Freeze된 test 결과

선택된 설정이 "fusion 없이 shared만 사용"이므로 test에서 별도로 평가할 새로운 fusion
설정이 없다 — freeze된 config 자체가 이미 기존 프로덕션 경로(shared subspace = E-Shared)와
동일하다. 별도 test 실행은 생략했다.

## 16. Restricted oracle headroom

각 arm family(G+L/G+T/G+S/G+LT/G+LTS)에서 val 기준 최적 alpha 하나씩을 대표로 고정한 뒤
query별 사후 선택(restricted oracle):

| Calibration | Oracle vs Shared | Non-shared 전체 winner share |
|---|---|---|
| C0 Raw | **1.11%** | 81.9% |
| C1 Train-global | **0.97%** | 80.2% |
| C2 Query-wise | **1.01%** | 80.3% |

Headroom은 spec §14 FAIL 기준("1% 미만")의 경계선에 걸쳐 있다(3개 중 1개는 근소하게 위,
2개는 근소하게 아래) — 실질적으로 유의미한 수준이 아니다. Non-shared arm이 개별 query
단위에서는 80% 가까이 "이기지만", 그 우위가 평균적으로 shared를 넘어서는 고정 결합으로
전혀 전환되지 않는다는 것은 query-level 승패가 큰 margin의 체계적 우위가 아니라 노이즈에
가까운 근소한 차이의 누적임을 시사한다.

## 17. Router 진행 가능성

spec §14 Router 검토 조건("Restricted Oracle이 best fixed arm보다 추가로 약 2% 이상
우수") 미충족 (best fixed arm = G 자신이므로 oracle-vs-best-fixed 개선률도 oracle-vs-shared와
동일한 1% 내외) — **Router 진행 근거 없음.**

## 18. 계산량

- E2 학습: 6 epoch(조기종료) + train/test 평가, batch_size=128, wall clock 35,070초(9.74시간),
  peak VRAM 32.78GB.
- E2.5 (이 실험): 재학습 없이 frozen checkpoint 재사용. C0/C2는 val 1-pass, C1은 train
  1-pass(calibration 통계) + val 1-pass. 총 4회의 frozen-checkpoint forward-only pass.

## 19. 최종 판정

FAIL 조건(spec §14) 중 다음이 **세 calibration 전부에서 독립적으로 확인**됐다:
> 모든 fusion이 E-Shared보다 나쁨 (raw 20/20, train-global 20/20, query-wise 20/20 arm이
> shared보다 나쁨)

Restricted Oracle headroom도 세 calibration 모두 ~1% 내외로 spec의 FAIL 경계값에 걸쳐
있어 실질적 개선 여지가 없다.

E1/E1.5에서 확인된 component teacher 수준의 oracle complementarity(§3, column-wise 약한
specialization 포함)는 **실재하지만**, 이번 E2.5는 그 정보가 raw/train-global/query-wise
어떤 결합 방식으로도 학습된 student score 수준에서 회수되지 않는다는 것을 보여준다 — 이는
가설 1(component subspace에 retrieval에 도움이 되는 complementary information이 사실상
없다)을 지지하고, 가설 2(scale/결합 방식 문제)를 기각한다.

**최종 판정: E-C — Shared Single Encoder 유지.**

다음을 진행하지 않는다:
- Soft Router
- lambda_component sweep
- Component encoder 구조 확장
- Joint fine-tuning
- ETTh1 재검토 (이미 E1/E1.5에서 FAIL)

Experiment E(Component-Specialized Multi-Subspace Retrieval Encoder) 트랙은 여기서
종료한다.

## 부록: 생략된 항목 (§16 효율성 원칙에 따른 의도적 생략)

- Original Full-D baseline(B0) 재학습 — §4 사유로 불필요 판단
- 10-block 세부 chronological 분석, block별 CSV/plot — 이미 결정적 결과 재확인에 불과
- 20개 arm 각각에 대한 개별 cluster bootstrap — selected=shared라 무의미
- 8개 필수 plot 전체, 일부 보조 CSV(compute_profile.csv 등) — 수치적 결론에 영향 없음
- §11 구현 검증 테스트 1~13 중 일부(재현성·tie-breaking 등 기존 스크립트에서 이미 검증된
  항목 재사용, alpha=1.0==shared 등 핵심 항목은 스크립트 구조상 원천적으로 보장됨 — G
  arm이 그 자체로 alpha=1.0의 정의)

이 생략들은 모두 "이미 결정적인 negative 결과를 재확인하는 데 그치는" 항목이며, 결론을
바꿀 수 있는 항목(3중 calibration 비교, column-wise specialization, restricted oracle,
alpha sensitivity)은 전부 실제로 계산해 반영했다.
