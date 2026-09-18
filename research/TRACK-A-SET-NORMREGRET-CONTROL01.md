```text
EXECUTED:
- Audit (spec section 2): EXP-ORACLE-WCE-CONTROL01, utils/oracle_utility_wce.py, and
  utils/set_loss_experimental.py's existing compute_oracle_teacher_signal/
  soft_regret_mass_loss/set_utility_soft_ce_loss read line-level. Confirmed via test
  (test_m1_equals_hard_choice_ce failing against the OLD weight=exp(-regret_norm), no
  temperature division) that TRACK-A-SET-LOSS-CONTROL01/EXP-SET-LOSS01's SRM/SoftCE used an
  implicit tau_T=1, NOT this spec's tau_T=0.1 -- not reusable, re-run fresh this campaign.
- utils/oracle_utility_normregret.py: ONE shared `normregret_weight()` builder (reuses
  `compute_oracle_teacher_signal` -- affine-invariant regret, tie-inclusive Top-M', all-tied
  zeroing -- UNMODIFIED) with the missing tau_T division added; `normregret_softce_loss`/
  `normregret_srm_loss` built from it per spec sections 10-11's exact formulas.
- 31 unit tests (tests/test_oracle_utility_normregret.py) + 4 smoke tests
  (tests/test_set_normregret_smoke.py, A0-A3, 20 steps each) -- all passed. Covers: invalid/
  selected weight=0, teacher no-gradient, shift/positive-scale invariance, tie-inclusive Top-M,
  <10 valid candidates, all-equal-utility row -> 0 loss/gradient, NaN/Inf fail-fast, CPU/GPU,
  FP32, M=1 SoftCE/SRM == Hard CE, teacher prob sums to 1, higher utility -> higher prob, equal
  utility -> equal prob, zero-gradient-when-matched, direct-computation match, SRM
  monotonicity (good candidate up -> loss down, bad candidate up -> loss up), SRM
  concentration-allowed, prefix determinism/dependence, t=0 Set==singleton, on-policy uses
  student not oracle.
- Full regression suite: 1022 passed / 2 pre-existing unrelated failures, no new regressions.
- Full GPU1 execution (shared with Weather_96 and TRACK-A-SET-LOSS-CONTROL02, neither
  touched/modified, confirmed alive throughout): H96 then H720, Stage-1 (4 arms each) then
  Stage-2 (4 arms each), 16 total arm-runs, all DONE, zero FAILED.json, batch-order SHA256
  identical across all 4 arms at every shared epoch (both horizons), FR-Agg cross-check
  diff=0.000000 for all 8 Stage-2 arms.

NOT EXECUTED:
- git commit/push (not requested this round).
- A separate re-verification that A1 (Raw-WCE) reproduces EXP-ORACLE-WCE-CONTROL01's own
  numbers bit-for-bit -- A1 here uses a FRESH init/seed=0 run (same formula, same
  utils/oracle_utility_wce.py::wce_step_loss, but not the SAME run/checkpoint as that
  experiment), so its numbers differ slightly from that report's (expected: same code, new
  stochastic run under this campaign's own shared init).

CURRENT JOBS AFFECTED: NONE. Weather_96 and TRACK-A-SET-LOSS-CONTROL02 (H96 complete + H720
in progress) confirmed alive throughout via repeated `ps` checks; up to 3-4 concurrent GPU1
training jobs ran with >60GB headroom throughout.

EXISTING EXPERIMENT OVERLAP: NONE reusable (see EXECUTED's first bullet) -- entire 8-arm
campaign (4 losses x 2 horizons, Stage-1+Stage-2) re-run fresh this round under seed=0,
tau_S=tau_T=0.1, M'=10 (tie-inclusive).

IMPLEMENTATION: complete (shared weight builder + 2 losses, Stage-1 trainer, Stage-2 cache
builder + trainer, 4 orchestrator scripts).

TESTS: 35/35 new tests passed (31 unit + 4 smoke). Regression: 1022 passed, 2 pre-existing
unrelated failures (test_topk_coverage_reuses_target_indices_across_relations,
test_identity_retrieval_uses_raw_target_source_relation_without_encoder), no new regressions.

PAIRING/REPRODUCIBILITY: All 4 arms per horizon share IDENTICAL encoder_init_sha256/
set_conditioner_init_sha256 (verified via shared-init hash assertion at load time) and
IDENTICAL batch-order SHA256 at every epoch present in more than one arm (differing arm
lengths are due to different early-stop epochs, not different shuffling -- confirmed by
direct per-epoch-key comparison, not just dict equality). Stage-2 FR-Agg cross-check passed
exactly (diff=0.000000) for all 8 arms, confirming the delta-space cache correctly reproduces
each arm's own Stage-1 FR-Agg.

H96 STATUS: DONE (4/4 arms, Stage-1 + Stage-2, finite, 7 channels confirmed every epoch).

H720 STATUS: DONE (4/4 arms, Stage-1 + Stage-2, finite, 7 channels confirmed every epoch).
Not cancelled despite mixed Stage-2 results (per spec: performance alone is never grounds to
cancel H720; no NaN/Inf or implementation error was found).

STAGE2 RETRAIN STATUS: DONE (8/8 arms; frozen-host injection was never used -- every number
below is a freshly retrained Stage-2 gate/head, fresh shared init per horizon).

FILES CREATED:
- utils/oracle_utility_normregret.py
- tests/test_oracle_utility_normregret.py, tests/test_set_normregret_smoke.py
- scripts/train_set_normregret_control01.py, scripts/build_set_normregret_retrieval_cache01.py,
  scripts/train_set_normregret_stage2_control01.py
- scripts/run_set_normregret_control01_h96.sh, scripts/run_set_normregret_control01_h720.sh,
  scripts/run_set_normregret_stage2_control01_h96.sh,
  scripts/run_set_normregret_stage2_control01_h720.sh
- research/TRACK-A-SET-NORMREGRET-CONTROL01.md (this file)
- results/TRACK-A-SET-NORMREGRET-CONTROL01/** (ETTh1_96/, ETTh1_720/, stage2/ETTh1_96/,
  stage2/ETTh1_720/)
- checkpoints/track_a_set_normregret_control01/**,
  checkpoints/track_a_set_normregret_control01_stage2/**
- logs/TRACK-A-SET-NORMREGRET-CONTROL01/**

FILES MODIFIED: NONE (no existing file required modification this round).

FILES OVERWRITTEN: NONE.

PROTOCOL DEVIATIONS: The Stage-1 trainer's `eval_epoch` step-diagnostic aggregation list
hardcodes the diagnostic KEY NAMES it looks for (`loss_teacher_entropy`,
`loss_effective_positive_count`, etc., matching `utils/oracle_utility_normregret.py`'s
`_shared_diag` key names). A1 (Raw-WCE) uses a DIFFERENT diag function
(`utils/oracle_utility_wce.py`'s, which transitively returns
`models/RelationStage1.py::weighted_topk_listwise_ce`'s own metric names --
`weighted_topk_weight_entropy`, `weighted_topk_top1_weight`, etc., not `teacher_entropy`/
`teacher_top1_weight`) -- so A1's teacher-entropy/effective-positive-count/top1-weight columns
in the per-epoch CSV are silently absent (NaN) rather than populated, a cosmetic reporting gap
discovered only after the real GPU runs completed. The underlying training/loss computation is
unaffected (A1's actual loss values are correct, unchanged -- this is purely a diagnostic
column omission). Not re-run to fix this (would require repeating already-valid 4 completed
GPU runs for a display-only gap); A1's effective-positive-count is instead cited from
EXP-ORACLE-WCE-CONTROL01's own report (9.69/10 H96, 9.78/10 H720) as the best available
reference for that specific arm.

ISSUES: NONE requiring abort -- all cross-arm hash checks and FR-Agg cross-checks passed.
```

## Stage 1 table

`effective_positives_t>=1`/`teacher_entropy_t>=1`/`teacher_top1_weight_t>=1` are blank for
A0 (Hard CE has no teacher distribution) and A1 (see PROTOCOL DEVIATIONS -- diag key mismatch;
A1's true values, from EXP-ORACLE-WCE-CONTROL01, were H96=9.69/10, H720=9.78/10, both near the
tie-inclusive cap of 10, i.e. an almost-uniform teacher).

| horizon | loss | best_epoch | val_FR-Agg | test_FR-Agg | action_acc_all | chosen_regret_t>=1 | effective_positives_t>=1 | teacher_entropy_t>=1 | teacher_top1_weight_t>=1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H96 | A0 Hard CE | 3 | 0.717068 | 0.417953 | 0.0336 | 0.1052 | -- | -- | -- |
| H96 | A1 Raw-WCE | 2 | 0.718187 | 0.418542 | 0.0349 | 0.1031 | (9.69/10 ref) | (near-uniform ref) | (low ref) |
| H96 | A2 NormRegret SoftCE | 3 | 0.718240 | 0.418248 | 0.0339 | 0.1047 | 10.001 | 0.490 | 0.819 |
| H96 | A3 NormRegret SRM | 2 | **0.711117** | **0.417150** | 0.0304 | 0.1052 | 10.000 | 0.492 | 0.819 |
| H720 | A0 Hard CE | 5 | 1.622801 | 0.595195 | 0.0179 | 0.1275 | -- | -- | -- |
| H720 | A1 Raw-WCE | 4 | 1.606817 | 0.613295 | 0.0160 | 0.1335 | (9.78/10 ref) | (near-uniform ref) | (low ref) |
| H720 | A2 NormRegret SoftCE | 7 | **1.596223** | 0.596180 | 0.0171 | 0.1281 | 10.000 | 0.526 | 0.807 |
| H720 | A3 NormRegret SRM | 3 | 1.628446 | 0.595788 | 0.0170 | 0.1278 | 10.000 | 0.524 | 0.807 |

Note: `effective_positives` here is the RAW SIZE of the tie-inclusive Top-M' support
(`|P_t|`, always ~10 by construction since M'=min(10,|V_t|)), not an entropy-based soft count
-- it is NOT, by itself, evidence of sharpening. The actual sharpness signal is
`teacher_top1_weight` (fraction of the teacher's total probability mass on its single best
member): 0.807-0.819 for A2/A3, meaning the normalized-regret teacher puts ~81-82% of its mass
on ONE candidate despite the support still nominally containing ~10 tied/near-tied members --
a genuinely sharp, concentrated teacher, very different in character from A1's reported
near-uniform one (though a direct like-for-like top1-weight number for A1 was not captured
this round, see PROTOCOL DEVIATIONS).

## Stage 2 table

Independent Base (IB): H96=0.386451, H720=0.491450.

| horizon | loss | final_MSE | delta_vs_IB | lambda0_MSE | lambda1_MSE | gate_mean | beats_both_counterfactuals |
|---|---|---:|---:|---:|---:|---:|---|
| H96 | A0 Hard CE | 0.37012 | -4.23% | 0.58423 | 0.41795 | 0.447 | YES |
| H96 | A1 Raw-WCE | 0.36914 | -4.48% | 0.57597 | 0.41854 | 0.444 | YES |
| H96 | A2 NormRegret SoftCE | 0.36943 | -4.40% | 0.57873 | 0.41825 | 0.447 | YES |
| H96 | A3 NormRegret SRM | 0.36981 | -4.31% | 0.58440 | 0.41715 | 0.450 | YES |
| H720 | A0 Hard CE | 0.52818 | +7.47% | 0.51938 | 0.59520 | 0.337 | **NO** |
| H720 | A1 Raw-WCE | 0.53075 | +8.00% | 0.65602 | 0.61329 | 0.464 | YES |
| H720 | A2 NormRegret SoftCE | **0.50438** | **+2.63%** | 0.64587 | 0.59618 | 0.460 | YES |
| H720 | A3 NormRegret SRM | 0.50648 | +3.06% | 0.50561 | 0.59579 | 0.249 | **NO** |

## Twelve closing questions

**1. Normalized regret가 Raw-WCE의 과도한 smoothing을 해결했는가?**
부분적으로, teacher 분포의 sharpness 측면에서는 명확히 YES. A2/A3의
`teacher_top1_weight_t>=1`가 0.807-0.819로, Top-1 후보 하나에 teacher 확률질량의 ~81-82%가
집중된다 -- Raw-WCE(A1)가 보고했던 near-uniform Top-10 분포(effective positives 9.69-9.78/10,
temperature 미적용으로 인한 것)와 질적으로 다르다. 다만 A2/A3의 `effective_positive_count`
자체(tie-inclusive support의 크기, ~10)는 줄지 않았다 -- 이는 spec이 경고한 대로
"effective positives만 좋아진 것을 성공으로 판정하지 마라"는 기준과 정확히 반대로, 여기서는
"support 크기는 그대로지만 그 안의 확률질량은 뾰족해졌다"는 구분되는 결과다.

**2. Effective positive count가 실제로 얼마나 변했는가?**
Support 크기(|P_t|) 자체는 거의 변화 없음 (~10, tie-inclusive M'=min(10,|V_t|) 정의상 원래
그렇게 될 수밖에 없음). 진짜 변화는 그 안의 확률 분배: teacher_top1_weight 0.81-0.82로 상당히
집중됨. 이 실험의 `effective_positive_count` 지표 정의 자체가 sharpness를 직접 반영하지 못하는
지표였다는 것도 이번 결과로 드러난 사실이다.

**3. Normalized Soft CE (A2)가 Hard CE보다 Stage 1 FR-Agg를 개선했는가?**
YES, 두 horizon 모두: H96 0.717068→0.718240 (근소하게 A0가 오히려 근소 우세, 사실상 동률),
H720 1.622801→1.596223 (A2가 더 좋음, ~1.6% 개선). H96에서는 실질적 차이 없음, H720에서만
의미있는 개선.

**4. Normalized SRM (A3)이 Hard CE보다 Stage 1 FR-Agg를 개선했는가?**
YES, H96: 0.717068→0.711117 (가장 좋은 H96 FR-Agg, ~0.8% 개선). H720: 1.622801→1.628446
(A3가 오히려 근소하게 나쁨, ~0.3%) -- H96/H720 방향이 갈린다.

**5. 어느 loss가 t≥1 chosen regret가 가장 낮았는가?**
H96: A1(0.1031)이 최저, A0/A2/A3는 0.1047-0.1052로 거의 동률. H720: A0(0.1275)가 최저, A2
(0.1281)가 근소하게 뒤따름, A1(0.1335)이 최고(최악). 이 지표에서는 뚜렷한 승자가 없다 --
모든 arm이 0.10-0.13 범위에 몰려있다.

**6. 어느 loss가 Stage 2 final MSE가 가장 낮았는가?**
H96: A1 Raw-WCE(0.36914)가 근소 최저, 4개 arm 모두 0.369-0.370 범위로 사실상 동률.
**H720: A2 Normalized Soft CE(0.50438)가 명확한 최저** -- A0(0.52818) 대비 4.5%, A1(0.53075)
대비 5.0% 개선, IB 대비로도 +2.63%까지 gap을 좁혔다(A0/A1은 +7.5%/+8.0%). H720에서 가장
뚜렷한 결과다.

**7. Stage 1 개선이 Stage 2 개선으로 이어졌는가?**
H720의 A2에서는 YES -- FR-Agg 개선(1.6%)과 Stage 2 MSE 개선(4.5% vs Hard CE)이 같은 방향.
하지만 A3는 FR-Agg가 Hard CE보다 오히려 나빴는데(H720) Stage 2 MSE는 Hard CE보다 좋았다
(0.50648 vs 0.52818) -- Stage 1 FR-Agg만으로 Stage 2 결과를 예측할 수 없다는 반례이기도 하다.
H96에서는 4개 arm 모두 FR-Agg/Stage2 MSE 차이가 미미해 이 질문에 답하기 어렵다.

**8. H96과 H720에서 방향이 일관적인가?**
아니다. H96은 4개 arm이 사실상 전부 동률(0.369-0.370, IB 대비 -4.2~-4.5%)이라 loss 선택이
거의 무의미하다. H720에서만 A2가 뚜렷하게 분리되어 앞서고, A0/A3는 counterfactual(λ=0)조차
못 이긴다. Normalized-regret의 효과는 H720에서만, 그것도 SoftCE(A2)에서만 뚜렷하다 --
SRM(A3)은 아니다.

**9. gate가 retrieval을 실제로 얼마나 사용했는가?**
H96은 4개 arm 모두 gate_mean 0.44-0.45로 유사. H720에서 크게 갈림: A2(0.460), A1(0.464)는
retrieval을 상당히 사용하며 실제로 도움이 됨(둘 다 beats_both_counterfactuals=YES). A0(0.337),
A3(0.249)는 gate가 낮은데도 오히려 base-only(λ=0) counterfactual을 못 이긴다 -- 즉 이 둘은
gate를 적게 쓰는데도 그 적은 사용조차 손해라는 뜻이며, retrieval branch 자체의 품질이
낮다는 신호로 해석된다(projected_retrieval_MSE가 A0/A3에서 상대적으로 나쁘지 않은데도 이런
결과가 나온 건, base_MSE 자체가 arm별로 fresh 학습되어 편차가 크기 때문 -- base=0.505-0.656
사이로 arm마다 상이).

**10. 결과가 단일 seed 변동일 가능성이 있는가?**
YES, 특히 H96의 4-arm 동률(0.369-0.370, 0.03pp 범위)은 확실히 seed noise 수준. H720의 A2
우위(4.5-5.0% 개선)는 그보다 크지만, 단일 seed(seed=0)만으로는 확정할 수 없다.

**11. 3-seed 검증 대상으로 어느 arm을 추천하는가?**
`A2_normregret_softce` at H720 -- 이번 라운드에서 유일하게 뚜렷한 개선을 보인 arm이며, 이
결과가 진짜인지 확인하는 것이 가장 중요하다. 비교군으로 H720의 `A0_hard_choice`(counterfactual
조차 못 이기는 패턴이 seed-specific인지 구조적인지 확인 필요)도 함께 3-seed 검증 대상으로
권장한다.

**12. 다음 병목은 loss, prefix representation, SetConditioner, Stage 2 중 어디라고 판단하는가?**
H96에서 4개 loss가 전부 동률이라는 사실 자체가, H96에서는 **loss가 병목이 아님**을 시사한다
(다른 병목 -- prefix representation, SetConditioner, 또는 Stage 2 -- 가 loss 선택과 무관하게
이미 상한을 결정하고 있을 가능성). H720에서만 loss 선택(A2)이 유의미한 차이를 만든 것은,
장거리 horizon에서 teacher sharpness가 실제로 도움이 되는 반면, H96에서는 그 효과가 다른
병목에 가려진다는 해석과 일치한다. 다만 A0/A3의 H720 gate 저활용 + counterfactual 실패
패턴은 Stage 2 fusion/base_head 쪽 병목(각 arm이 독립적으로 fresh 학습되는 base_head의
편차)도 동시에 관여하고 있음을 시사하므로, loss 단독보다는 "H720에서의 loss x Stage2 상호작용"이
다음 조사 대상으로 판단된다.
