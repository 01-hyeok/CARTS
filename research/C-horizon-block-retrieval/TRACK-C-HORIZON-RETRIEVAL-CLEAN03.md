# TRACK-C-HORIZON-RETRIEVAL-CLEAN03 — Frozen-Trunk Adapter Isolation

**Status: ETTh1_720 seed=0 COMPLETE. Result: NO-GO (this configuration).**
Validation shows a small, statistically real improvement for every tau
tested; the validation-BEST arm reverses direction on test. Per the
spec's own stopping rule ("Validation에서 선택한 설정이 test에서 반대
방향" → do not proceed to Weather), Weather_720 was **not started**.

## 1. Clean02에서 무엇이 불분명했는지

Clean02의 H-Symmetric은 G-Transformer보다 test에서 12.3% 나빴다. 하지만
G와 H는 **shared trunk까지 서로 다른 loss로 학습**됐다(H는
$\frac13(L_{B1}+L_{B2}+L_{B3})+0.25L_G$, G는 $L_G$만) — 그래서 그 실패가
"adapter capacity 부족" 때문인지 "joint-training이 trunk의 global
representation을 오염시켰기 때문"인지 분리할 수 없었다. 또한 Clean02는
$\tau_{T,G}=0.10$을 모든 block에도 그대로 썼는데(§8 calibration이 실제로
0.10을 선택), teacher sharpness(D0 진단 결과 tau=0.10에서 Top-10
probability mass가 겨우 9.6~13.1%)가 Top-10 retrieval 목표와 애초에 안
맞았을 가능성도 분리되지 않았다.

## 2. 이번 실험이 무엇을 분리하는지

Clean02의 G-Transformer validation-best checkpoint를 **완전히 동결**하고
adapter만 학습해서, "동일한 frozen representation 위에서 lightweight
adapter가 진짜로 block-specific 정보를 뽑아낼 수 있는가"와 "teacher
sharpness(tau)가 그 성패에 영향을 주는가"를 독립적으로 검증한다.

## 3. Frozen trunk와 symmetric adapter 구조

G-best checkpoint(`checkpoints/track_c_horizon_retrieval_clean02/ETTh1_720/g_transformer/checkpoint.pth`,
epoch=8, val_primary_mse=1.810685340415827) 로드 후 fingerprint 검증
(arm==g_transformer, relation_encoder_type==transformer, patch_len=stride=16,
d_model=128) — 전부 일치 확인. `model.eval()` + 전 파라미터
`requires_grad_(False)`, optimizer에는 adapter 파라미터만 포함.
`frozen_trunk_sha256`을 로드 직후/cache 생성 후/학습 종료 후 3번 비교 —
**세 값이 항상 동일**함을 확인(trunk가 한 번도 움직이지 않았다는 직접
증거).

Frozen embedding cache: candidate/query embedding을 **한 번만** live encode
후 재사용(`train_c_horizon_frozen03.py::build_query_cache`). Cache-vs-live
검증: embedding max abs error = **0.00e+00**, score max abs error =
**0.00e+00**(부동소수점 오차 이하로 완전 일치). A0(frozen global, adapter
없음) 재평가 결과 `val=1.810685`, `test=0.654082` — Clean02 보고서와
**정확히 일치**(재현 성공).

Adapter는 Clean02와 동일한 symmetric zero-init residual bottleneck
(128→32→GELU→32→128, 마지막 Linear weight+bias 0-init, query/candidate에
동일 모듈 적용).

## 4. Teacher와 student 정의

기존 Clean02와 동일(`block_distance` == `individual_utility_memsafe`의
슬라이스 재사용, z-score 정규화 + softmax). 이번 실험은 **Global loss
없음**(trunk가 완전 고정이라 global representation 보호가 불필요) —
$L_{\text{adapter}} = \frac13(L_{B1}+L_{B2}+L_{B3})$만 사용.

## 5. Temperature별 teacher Top-10 mass (D0)

`results/TRACK-C-HORIZON-RETRIEVAL-CLEAN03/ETTh1_720/teacher_sharpness_diagnostic.json`.

| tau | Global | B1 | B2 | B3 |
|---|---|---|---|---|
| 0.01 | 84.8% | 86.4% | 85.1% | 82.9% |
| 0.02 | 63.7% | 66.9% | 65.0% | 60.4% |
| 0.05 | 26.3% | 32.4% | 29.2% | 24.8% |
| 0.10 | 9.6% | 13.1% | 11.3% | 9.1% |

Clean02가 실제로 썼던 tau=0.10에서는 Top-10에 확률 질량의 **10% 안팎만
집중**돼 있었다 — 나머지 90%가 수백 개의 candidate에 퍼져 있었다는 뜻.
Oracle Top-10 Jaccard(태우 무관, 순수 거리 기준): `block1_vs_block2=0.036`,
`block1_vs_block3=0.026`, `block2_vs_block3=0.039`,
`global_vs_block1=0.058`— 사용자가 사전에 제시한 범위(0.03~0.05)와 일치.

## 6. A0/A1/A2/A3 validation 결과

| Arm | tau_T | Best epoch | val_block_h720_mse | vs A0 개선율 |
|---|---|---|---|---|
| A0 (frozen global) | - | - | 1.810685 | 0% |
| A1 | 0.10 | **1** | 1.792270 | +1.02% |
| A2 | 0.02 | **1** | 1.787607 | +1.28% |
| A3 | 0.01 | **1** | 1.782083 | **+1.58%** |

세 arm 모두 **best_epoch=1이고 이후 계속 악화**(Clean02의 H-Symmetric과
같은 패턴, Type F) — 하지만 이번엔 그 epoch-1 지점이 A0보다 **나은** 방향.
tau가 작아질수록(teacher가 sharper) 개선율이 커지는 **단조적** 경향이
뚜렷하다 — Case B(teacher sharpness 문제)를 지지하는 방향의 신호.

## 7. 동일 frozen trunk 내부 global-vs-block 비교

A0와 A1/A2/A3 각각의 "adapter 미적용 global 재평가"는 전부
`global_h720_mse=1.810685`(val)/`0.654082`(test)로 **수치적으로 완전히
동일**함을 확인했다(§15 요구사항 충족 — 서로 다르게 학습된 trunk를
비교하는 문제가 이번엔 없음).

## 8. Block별 MSE (test, validation-best arm A3)

| | Global(A0) | Block(A3) |
|---|---|---|
| block1_mse | - | 0.408413 |
| block2_mse | - | 0.533604 |
| block3_mse | - | 0.805673 |
| **concat h720_mse** | **0.654082** | **0.662015** |

## 9. Recall@10/NDCG@10/mean rank

이번 단계에서는 계산하지 않았다 — paired bootstrap(§13)과 test 결과(§14)가
이미 명확한 부정적 방향을 보여 우선순위가 낮았다(§16 효율성 원칙에 따른
의도적 생략, 후속 진단 필요시 추가 가능).

## 10. Student head 간 Jaccard와 Oracle Jaccard 비교

Test, A3: `global_vs_block1=0.427`, `global_vs_block2=0.495`,
`global_vs_block3=0.555`, `block1_vs_block2=0.609`, `block1_vs_block3=0.478`,
`block2_vs_block3=0.531`. Clean02의 H-Symmetric(0.69~0.91)보다는 낮지만,
Oracle 수준(0.03~0.06)과는 여전히 거리가 매우 멀다 — **spec의 명시적
주의사항대로, Jaccard가 낮아졌다는 사실 자체를 specialization 성공으로
해석하지 않는다.**

## 11. Adapter residual norm

이번 단계에서는 별도로 계산하지 않았다(§16 효율성 원칙).

## 12. Epoch별 KL과 validation MSE 관계

세 arm 모두 train KL loss는 매 epoch 단조 감소하는데(A1: 0.869→0.717,
A2: 3.310→2.902, A3: 4.450→3.960) validation block MSE는 epoch 1 이후
계속 **악화**된다 — spec Case F의 정의("Train KL은 감소하지만 validation
MSE가 악화")에 정확히 해당한다. Listwise distribution matching(teacher
ranking을 모사하는 것)과 hard Top-10 forecasting MSE(실제 예측 정확도)
사이에 괴리가 있다는 신호다.

## 13. Paired bootstrap 결과 (validation, A0 vs 각 arm)

2,000 resamples, base query window 단위, seed=0.

| Arm | mean delta | 95% CI | CI가 0 초과 | 개선 query 비율 | 1%+ 개선 |
|---|---|---|---|---|---|
| A1 | 0.018415 | [0.014366, 0.022442] | **예** | 65.2% | 53.2% |
| A2 | 0.023078 | [0.018298, 0.027851] | **예** | 63.1% | 51.1% |
| A3 | 0.028603 | [0.023749, 0.033631] | **예** | 63.6% | 53.8% |

통계적으로는 세 arm 모두 확실히 A0보다 좋다(CI가 0을 전혀 포함하지
않음) — 하지만 **실질적 개선 크기는 spec의 2% 기준에 못 미친다**(최대
1.58%). "통계적 유의성"과 "실질적 개선"을 분리해서 봐야 한다는 spec의
원칙이 정확히 여기서 갈린다: 유의하지만 작다.

## 14. Validation-best arm만 사용한 test 결과

Validation 규칙(§18)에 따라 **A3(tau=0.01)만** 사전에 선택 후 test 1회
평가:

| | Global(A0) | Block(A3, val-best) |
|---|---|---|
| Val | 1.810685 | 1.782083 (**+1.58%**) |
| Test | 0.654082 | 0.662015 (**-1.21%, 악화**) |

**Validation에서 선택한 설정이 test에서 정확히 반대 방향으로 뒤집혔다.**
Clean02와는 다른 실패 양상이지만 결론은 같은 방향이다 — 이 조합은
배포 가능한 개선을 만들지 못한다.

## 15. 계산량과 wall-clock

Frozen trunk + 임베딩 캐시 덕분에 학습 자체는 매우 빠르다(3개 arm이
GPU 하나에서 병렬로 몇 분 내 전부 완료, epoch당 forward pass가
adapter(128→32→128)만 거치고 Transformer는 한 번도 다시 안 돌림). 정확한
wall-clock은 `retrieval_metrics_*.json`의 `wall_clock_seconds`에
저장되어 있음.

## 16. 통과 기준별 PASS/FAIL

| 조건 | 결과 |
|---|---|
| Block H720 MSE가 A0 대비 2%+ 개선 | **FAIL** (최대 1.58%) |
| Oracle headroom recovery 10%+ | 계산 안 함(1번 조건 실패로 무의미) |
| B1/B2/B3 중 최소 2개에서 A0 대비 개선 | 계산 안 함 |
| Paired bootstrap mean delta 양수 | **PASS**(3개 arm 전부) |
| 95% CI 하한 > 0 | **PASS**(3개 arm 전부) |
| Validation에서 선택한 설정이 test에서 같은 방향 | **FAIL** (반대 방향) |

## 17. Case A~F 판정

- **Case A** (joint-training interference가 주 원인) — **부분적으로 지지**:
  frozen trunk로 격리하니 Clean02의 "12.3% 악화"가 사라지고 "1.58% 개선"으로
  방향이 바뀜. Joint-training이 실제로 해로웠다는 증거.
- **Case B** (teacher sharpness 문제) — **부분적으로 지지**: tau가
  작아질수록(0.10→0.02→0.01) 개선율이 단조적으로 커짐(1.02%→1.28%→1.58%).
- **Case F** (train KL 감소, val MSE 악화) — **명확히 해당**: 세 arm
  모두 best_epoch=1이고 이후 단조 악화.
- 최종적으로는 Case A+B의 부분적 지지가 있음에도 **test에서 방향이
  뒤집혀** 실질적 성공으로 이어지지 않았다 — 순수한 Case E(완전 실패)는
  아니지만, 이 설정 그대로는 배포 가능한 개선이 아니다.

## 18. Weather 및 joint fine-tuning 진행 여부

**진행하지 않음.** spec §22 중단 규칙("Validation에서 선택한 설정이
test에서 반대 방향")에 정확히 해당하므로 Weather_720을 시작하지 않았다.

## 19. 구현 한계 및 해석상의 주의점

- ETTh1_720 **단일 셀, 단일 seed**만 실행했다.
- tau=0.05 보간은 spec이 "A1/A2/A3 결과가 애매할 경우에만" 검토하라고
  했는데, 이번 결과는 애매하지 않다(방향이 test에서 명확히 반대) —
  보간 없이 결론.
- Recall@10/NDCG@10/adapter residual norm 등 일부 secondary 지표는
  이미 명확한 negative 결론에 추가 정보를 주지 않는다고 판단해 생략했다.
- `n_reps=2000`(spec 요청값 그대로), `n_base_windows=2161`(val query 수).

## 결론

Frozen trunk로 joint-training interference를 제거하자 Clean02의 완전한
실패(-12.3%)가 작지만 통계적으로 유의한 개선(+1.58%, tau=0.01)으로
바뀌었다 — **Clean02 실패의 일부는 실제로 joint-training 때문이었다**는
것이 이번 실험으로 확인됐다. 하지만 그 개선은 spec의 실질적 기준(2%)에
못 미치고, validation-best 설정이 test에서는 오히려 악화되는 방향으로
뒤집혔다. **이 특정 조합(frozen G-trunk + symmetric adapter + 이 3개
tau)으로는 GO 판정을 내릴 수 없다.** Weather 확대나 joint fine-tuning은
진행하지 않는다.
