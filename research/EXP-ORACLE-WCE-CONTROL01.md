```text
EXECUTED:
- Code audit (spec section 3, all 12 required items) via direct call-site
  tracing, not name-based assumption -- see section 1 below.
- Reused, unmodified: scripts/train_factorial_e2e01.py::run_sequence (via
  its `loss_fn` hook, already added and regression-tested this session for
  TRACK-A-SET-LOSS-CONTROL02), individual_utility/greedy_set_utility,
  HostScorer/candidate_weights/encode_raw/arm_score, all diagnostics
  functions, the full 7-channel loop, the checkpoint-selection rule.
- Built utils/oracle_utility_wce.py: ONE shared `wce_step_loss`, built
  entirely from models/RelationStage1.py's `prepare_topk_coverage_targets`
  + `weighted_topk_listwise_ce` (the two functions the spec names), used
  identically by both arms.
- Audit finding (not assumed, found via unit test): the existing
  `scripts/train_onpolicy_rankloss01.py::_wce_step_loss` convenience
  wrapper does NOT apply any student temperature to `u_hat` before
  `log_softmax` -- only the teacher weighting is temperature-scaled. This
  does not match the spec's `p_theta=softmax(s_theta/tau_S)` formula, and
  means TRACK-A-ONPOLICY-RANKLOSS01's R1_wce arm was NOT a temperature-
  matched comparison against R0's Hard Choice CE. `wce_step_loss` fixes
  exactly this gap (applies `tau_choice` to BOTH student and teacher, the
  SAME `tau_choice` Hard Choice CE already uses -- no new/unapproved
  temperature). research/REVIEW_FOR_CHATGPT.md carries a correction block;
  R1_wce's own files were not modified.
- t=0 equivalence verified directly (not assumed): Set Oracle's
  `Aggregate(S_{-1} U {i})` with empty prefix reduces to the singleton
  candidate's own value, independent of host weighting -- confirmed with a
  REALISTIC (non-delta) host weight, max_abs_diff=4.77e-07. Distance,
  Top-M indices, teacher weights, and WCE loss/gradient on identical
  student logits all verified identical at t=0.
  (tests/test_oracle_wce_t0_equivalence.py, 4/4 passed)
- 20 unit tests (tests/test_oracle_utility_wce.py) covering: invalid/
  selected-candidate teacher weight exclusion, no gradient into the
  teacher, <10-candidate handling, teacher probabilities sum to 1, lower
  distance -> higher teacher probability, gradient shrinks as student
  matches teacher, M=1 == Hard Choice CE exactly, direct-computation match,
  CPU/GPU consistency, FP32 finiteness, Individual/Set identical-distance
  -> identical loss/gradient, both arms share the same function object,
  Set-WCE sign convention (better candidate -> lower distance -> higher
  teacher prob), Individual-distance prefix-invariance, Set-distance
  prefix-dependence at t>=1, selected-candidate exclusion, on-policy uses
  model prediction not oracle, 7-channel list check. All 20 + 4 = 24
  passed.
- Smoke test (spec section 11): Individual Hard CE / Individual WCE / Set
  Hard CE / Set WCE, 20 real optimizer steps each on the same ETTh1_96
  small batches, same LR -- all finite, zero NaN/Inf
  (tests/test_oracle_wce_smoke.py, 4/4 passed).
- Full regression suite: 968 passed / 2 pre-existing failures unrelated to
  this work (same names as the documented baseline), no new regressions.
- Full re-execution, GPU1 only, one arm at a time within this experiment
  (shared GPU1 with the already-running Weather_96 factorial and
  TRACK-A-SET-LOSS-CONTROL02 -- neither touched, both confirmed alive
  throughout via repeated `ps` checks):
  - Stage-1: individual_onpolicy_cosine_wce and set_onpolicy_cosine_wce,
    H96 then H720, all 4 runs DONE, batch-order hashes identical across
    both arms at every shared epoch (H96: 7/7 epochs; H720: 6/6 overlapping
    epochs before individual's earlier early-stop -- see section "batch
    order" below for the exact comparison).
  - Stage-2: all 4 arms retrained fresh (frozen Stage-1 encoder/
    SetConditioner, freshly initialized gate/base_head/fusion from a
    shared per-cell init), delta-space schema-versioned cache, FR-Agg
    cross-check gate passed with diff=0.000000 for all 4 arms.
- append-only correction to research/REVIEW_FOR_CHATGPT.md (R1_wce
  temperature gap + frozen-host Stage-2 limitation).

NOT EXECUTED:
- Re-running the Hard Choice CE baseline arms (explicitly forbidden --
  reused as-is from results/TRACK-A-CHOICECE-STAGE2-RETRAIN01-CORRECTED/).
- 3-seed replication (out of scope this round, recommended in the closing
  questions).
- git commit/push (not requested this round).

CURRENT JOBS AFFECTED: NONE. Weather_96 factorial (arm-changing PIDs,
confirmed alive throughout, currently on its final arm
set_onpolicy_asymmetric) and TRACK-A-SET-LOSS-CONTROL02 (currently
A0_hard_choice, epoch 5+/10) were both confirmed alive via `ps` before,
during, and after this experiment's GPU1 work; up to 3 concurrent GPU1
jobs ran with >60GB headroom throughout.

IMPLEMENTATION: complete (Stage-1 trainer, shared WCE loss, Stage-2
trainer, cache builder, 4 orchestrator scripts, 24 unit/gate/smoke tests).

LOSS-ONLY DIFFERENCE VERIFIED: YES at the code level (identical
run_sequence, identical channel loop, identical on-policy prefix advance,
identical checkpoint criterion, identical hyperparameters -- only `target`
and therefore the Oracle-utility source differ, and both feed the SAME
`wce_step_loss`). NOT independently re-verified as an exact-paired
stochastic reproduction against the ORIGINAL Hard Choice CE runs (see
HARD BASELINE COMPATIBILITY below) -- this experiment's OWN two WCE arms
ARE exact-paired against each other (see batch-order table).

7-CHANNEL VERIFIED: YES, all 4 Stage-1 runs report
`processed_channels=[0, 1, 2, 3, 4, 5, 6]`, asserted in-process
(`assert va['processed_channels'] == [0,1,2,3,4,5,6]`) every epoch, never
just printed.

T0 EQUIVALENCE: PASS. Individual and Set Oracle distance, Top-M indices,
teacher weights, and WCE loss/gradient on identical student logits are all
identical at t=0 (max_abs_diff=4.77e-07), confirmed against the REAL
`individual_utility`/`greedy_set_utility` implementations under a
realistic (non-degenerate) host weight, not a contrived special case.

TESTS: 24/24 new tests passed (20 unit + 4 t=0 gate), 4/4 smoke tests
passed, full regression suite 968 passed / 2 pre-existing unrelated
failures (test_topk_coverage_reuses_target_indices_across_relations,
test_identity_retrieval_uses_raw_target_source_relation_without_encoder),
no new regressions.

HARD BASELINE COMPATIBILITY: [ISSUE] Existing Hard Choice run does not
preserve enough artifacts to prove exact paired initialization/batch-order
equality. The comparison is configuration-matched but not exact paired-
stochastic reproduction. Specifically: `individual_onpolicy_cosine` /
`set_onpolicy_cosine` (results/track_a_factorial_e2e/ETTh1_{96,720}/) were
run under `scripts/train_factorial_e2e01.py`, which seeds the model via a
single `torch.manual_seed(cli.seed)` call before both model construction
AND (implicitly, via the global generator) DataLoader shuffling -- no
explicit per-epoch batch-order hash was recorded for those runs, and no
separate loader-seed artifact exists to re-derive it. Both share
`seed=0` and the SAME encoder_init_sha256 (`56588522de17ff76` at H96) as
this experiment's WCE arms (confirmed -- the architecture/seed/construction
order happen to coincide), which is suggestive but not sufficient proof of
identical batch order. Per instruction, the Hard Choice arms were NOT
re-run to close this gap; the WCE arms proceeded under matched
configuration and seed=0 as authorized.

H96 STATUS: DONE (both arms, Stage-1 + Stage-2, all finite, 7 channels
confirmed, batch-order hashes identical for all 7 epochs).

H720 STATUS: DONE (both arms, Stage-1 + Stage-2, all finite, 7 channels
confirmed, batch-order hashes identical for all 6 epochs individual ran
before its earlier early-stop; set continued to identical hashes through
its own epoch 9).

STAGE2 RETRAIN STATUS: DONE (4/4 arms; frozen-host injection results are
NOT used -- every number below is from a freshly retrained Stage-2).

FILES CREATED:
- utils/oracle_utility_wce.py
- tests/test_oracle_utility_wce.py
- tests/test_oracle_wce_t0_equivalence.py
- tests/test_oracle_wce_smoke.py
- scripts/train_oracle_wce_control01.py
- scripts/run_oracle_wce_control01_h96.sh
- scripts/run_oracle_wce_control01_h720.sh
- scripts/build_oracle_wce_retrieval_cache01.py
- scripts/train_oracle_wce_stage2_control01.py
- scripts/run_oracle_wce_stage2_control01_h96.sh
- scripts/run_oracle_wce_stage2_control01_h720.sh
- research/EXP-ORACLE-WCE-CONTROL01.md (this file)
- results/EXP-ORACLE-WCE-CONTROL01/** (Stage-1: ETTh1_96/, ETTh1_720/;
  Stage-2: stage2/ETTh1_96/, stage2/ETTh1_720/)
- checkpoints/exp_oracle_wce_control01/**,
  checkpoints/exp_oracle_wce_control01_stage2/**
- logs/EXP-ORACLE-WCE-CONTROL01/**

FILES MODIFIED:
- research/REVIEW_FOR_CHATGPT.md (append-only correction block: R1_wce's
  missing student temperature + frozen-host Stage-2 limitation)

FILES OVERWRITTEN: NONE. R1_wce's own results/logs/reports and the Hard
Choice CE baseline's results/checkpoints were never touched.

PROTOCOL DEVIATIONS: A real orchestration bug was found and fixed before
any Stage-2 numbers were produced: the first Stage-2 orchestrator draft
used the SAME output directory for both Stage-1 (`results/EXP-ORACLE-WCE-CONTROL01/<cell>/`)
and Stage-2 outputs, and both stages write a same-named `DONE_<arm>.marker`
per cell -- so the Stage-2 orchestrator's first run misread Stage-1's own
marker as "Stage-2 already done" and skipped Stage-2 entirely without
training anything (caught immediately: cache/checkpoint/metrics files were
absent despite the "already done, skipping" message). Fixed by moving ALL
Stage-2 artifacts under a separate `results/EXP-ORACLE-WCE-CONTROL01/stage2/`
subtree before any real Stage-2 training ran; re-verified the fix produced
real cache files, real FR-Agg cross-checks, and real training logs before
trusting any Stage-2 number in this report. No results were lost or
silently accepted from the buggy first attempt (it produced zero output
files, only a misleading log line).

ISSUES:
[ISSUE] Hard Choice CE baseline compatibility is configuration-matched,
not exact-paired (see HARD BASELINE COMPATIBILITY above) -- do not read
any Individual/Set delta-vs-Hard-CE number below as a controlled,
seed-and-batch-order-identical ablation; it is a same-config, same-seed,
independently-executed comparison.
```

## 1. Code audit (spec section 3)

```text
HARD INDIVIDUAL LOSS CALL:
  run_sequence() -> oracle_choice_step_loss(u_hat, u_target, valid_now, tau_choice)
  (scripts/train_oracle_choice01.py), u_target = individual_utility(...).

HARD SET LOSS CALL:
  Same call site/function, u_target = greedy_set_utility(...).

INDIVIDUAL ORACLE UTILITY:
  individual_utility(futures, query_future) = -MSE(y_i, y_q), step-invariant.

SET ORACLE UTILITY:
  greedy_set_utility(prefix_idx, w_host, futures, query_future, chunk_size)
  = -AggregateMSE(S_{t-1} U {i}), host-weighted, recomputed every step from
  the on-policy prefix.

STUDENT LOGIT CONSTRUCTION:
  u_hat = arm_score(h_t, E, metric); cosine (metric=None) for both WCE
  arms. h_t = z_q at t=0; h_t = set_conditioner(z_q, E[prefix].mean(1)) at
  t>=1.

VALID/SELECTED MASK:
  valid_now = cand_mask & ~selected, updated every step via
  selected.scatter(1, nxt, True).

PREFIX UPDATE:
  On-policy: nxt = model_next = u_hat.masked_fill(~valid_now,-inf).argmax(-1),
  detached -- never the oracle's own pick.

CHANNEL LOOP:
  channels = list(range(int(args.enc_in))) = [0..6] for ETTh1 (enc_in=7),
  `for c in channels` already iterates all 7 -- confirmed via saved
  fingerprints AND the in-process assertion added this round.

LOSS REDUCTION:
  Per channel: mean over K=10 steps. Across channels: mean over 7 channels.

CHECKPOINT CRITERION:
  'min val free_running_aggregate_future_mse', identical string to the
  Hard Choice arms' own saved fingerprint.

EXISTING WCE IMPLEMENTATION:
  models/RelationStage1.py::prepare_topk_coverage_targets +
  weighted_topk_listwise_ce (spec-named primitives, reused directly).
  scripts/train_onpolicy_rankloss01.py::_wce_step_loss was audited as a
  reuse candidate and found to NOT apply a student temperature -- NOT
  reused; utils/oracle_utility_wce.py::wce_step_loss built instead,
  directly from the two named primitives, with `tau_choice` applied to
  BOTH student and teacher.

STAGE2 CORRECTED TRAINING PATH:
  scripts/build_choicece_retrieval_cache01.py +
  scripts/train_choicece_stage2_retrain01.py (delta-space cache, FR-Agg
  cross-check gate, frozen Stage-1 + fresh gate/head) -- structurally
  reused for scripts/build_oracle_wce_retrieval_cache01.py +
  scripts/train_oracle_wce_stage2_control01.py.
```

## 2. Batch-order hash verification (spec section 8)

H96 (7 overlapping epochs, both arms):

| epoch | individual hash (16 hex) | set hash (16 hex) | match |
|---|---|---|---|
| 1-7 | (identical to set, every epoch) | (identical to individual, every epoch) | YES, all 7 |

H720 (individual early-stopped at epoch 6, set continued to epoch 9):

| epoch | individual hash | set hash | match |
|---|---|---|---|
| 1 | ed5ae865... | ed5ae865... | YES |
| 2 | 8afdef44... | 8afdef44... | YES |
| 3 | 7ac8d453... | 7ac8d453... | YES |
| 4 | 1dc6c669... | 1dc6c669... | YES |
| 5 | 3be4e607... | 3be4e607... | YES |
| 6 | 6fee58ce... | 6fee58ce... | YES |
| 7-9 | (individual stopped) | e888..., be01..., 3be5... | N/A (not a divergence -- different early-stop epoch, not different shuffle order) |

All epochs present in BOTH arms match exactly at both horizons -- confirms
identical `loader_seed=0` DataLoader shuffling as required.

## 3. Stage 1 table

Regret/rank/NDCG/Spearman columns are Oracle-SPECIFIC (Individual's own
target vs Set's own target) and must NOT be read as directly comparable
absolute-value rankings between Oracles (spec section 13); FR-Agg is the
one Oracle-agnostic comparison metric.

| horizon | oracle | loss | best_epoch | val_FR-Agg | test_FR-Agg | action_acc_all | action_acc_t>=1 | chosen_regret_all | chosen_regret_t>=1 | median_rank_t>=1 | top10_hit_t>=1 | ndcg@10_t>=1 | spearman_t>=1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H96 | Individual | WCE | 2 | 0.731377 | 0.417154 | 0.0084 | 0.0084 | 0.4916 | 0.4931 | 533.26 | 0.0752 | 0.9726 | 0.4362 |
| H96 | Set | WCE | 2 | 0.718187 | 0.418542 | 0.0349 | 0.0378 | 0.1410 | 0.1031 | 43.40 | 0.2719 | 0.6465 | 0.2166 |
| H720 | Individual | WCE | 1 | 1.585016 | 0.588404 | 0.0097 | 0.0106 | 0.4666 | 0.4561 | 357.17 | 0.0859 | 0.9694 | 0.3687 |
| H720 | Set | WCE | 4 | 1.606817 | 0.613295 | 0.0160 | 0.0176 | 0.1723 | 0.1335 | 68.51 | 0.1486 | 0.5729 | 0.3601 |

## 4. Stage 2 table

Independent Base (reused, unaffected by this experiment): H96=0.386451,
H720=0.491450. Hard-CE reference is
`results/TRACK-A-CHOICECE-STAGE2-RETRAIN01-CORRECTED/<cell>/<arm>/metrics_best.json`
(configuration-matched, not exact-paired -- see ISSUES).

| horizon | oracle | loss | base_MSE | base_MAE | retrieval_MSE(proj) | retrieval_MAE | delta_vs_base | relative_delta_vs_base | delta_vs_corresponding_hard | relative_delta_vs_corresponding_hard | gate_mean | gate_median | gate_std | gate_p10 | gate_p90 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| H96 | Individual | WCE | 0.56193 | -- | 0.41715 | -- | -0.19578 | -34.83% | -0.00158 | -0.43% | 0.454 | 0.449 | 0.041 | 0.411 | 0.507 |
| H96 | Set | WCE | 0.57597 | -- | 0.41854 | -- | -0.20683 | -35.91% | +0.00145 | +0.39% | 0.444 | 0.440 | 0.042 | 0.398 | 0.498 |
| H720 | Individual | WCE | 0.47464 | -- | 0.58840 | -- | +0.00442 | +0.93% | -0.00381 | -0.79% | 0.147 | 0.107 | 0.144 | 0.005 | 0.339 |
| H720 | Set | WCE | 0.65602 | -- | 0.61329 | -- | -0.12527 | -19.10% | +0.04091 | +8.35% | 0.464 | 0.460 | 0.103 | 0.340 | 0.593 |

(base_MAE / retrieval_MAE not separately recorded by the current trainer's
`test` dict -- final_mse/final_mae are, base/retrieval branch MAE were not
part of the metric set this trainer emits; noted as a gap, not fabricated.)

Counterfactual check (trained gate vs lambda=0 base-only / lambda=1
retrieval-only, spec section 6/9):

| horizon | oracle | trained_final_mse | lambda0(base) | lambda1(retrieval) | beats_both |
|---|---|---:|---:|---:|---|
| H96 | Individual | 0.36615 | 0.56193 | 0.41715 | YES |
| H96 | Set | 0.36914 | 0.57597 | 0.41854 | YES |
| H720 | Individual | 0.47906 | 0.47464 | 0.58840 | **NO** |
| H720 | Set | 0.53075 | 0.65602 | 0.61329 | YES |

## 5. Ten closing questions

**1. Individual Oracle에서 WCE가 Hard Choice CE보다 개선됐는가?**
YES at both horizons, by a small margin: H96 -0.43% (0.36615 vs 0.36773),
H720 -0.79% (0.47906 vs 0.48288). Both improvements are small relative to
the "configuration-matched, not exact-paired" caveat -- plausibly within
single-seed noise (see Q9).

**2. Set Oracle에서 WCE가 Hard Choice CE보다 개선됐는가?**
NO at both horizons: H96 +0.39% worse (0.36914 vs 0.36769), H720
substantially worse at +8.35% (0.53075 vs 0.48984). The H720 gap is large
enough that it is unlikely to be pure noise, though a single seed cannot
confirm this (Q9/Q10).

**3. WCE 효과는 Individual과 Set 중 어디에서 더 컸는가?**
The DIRECTION differs by Oracle, not just magnitude: WCE helped Individual
at both horizons and hurt Set at both horizons. In absolute terms Set's
H720 change (+8.35%) is the largest single effect observed, but it is a
REGRESSION, not an improvement -- "WCE effect was larger for Set" is true
only if read as "larger magnitude of harm," not "larger benefit."

**4. 동일한 WCE에서 Set Oracle이 Individual Oracle보다 개선됐는가?**
NO at either horizon: Individual-WCE beats Set-WCE at both H96 (0.36615 vs
0.36914) and H720 (0.47906 vs 0.53075, by a wide margin). Under Hard
Choice CE the two Oracles were nearly tied (H96: 0.36773 vs 0.36769; H720:
0.48288 vs 0.48984) -- WCE is what pulls them apart, in Individual's
favor, not Set's.

**5. t=0에서는 두 Oracle이 실제로 동일했는가?**
YES, confirmed directly (not assumed) against the real
`individual_utility`/`greedy_set_utility` implementations under a
realistic host weight: distance, Top-M indices, teacher weights, and WCE
loss/gradient on identical student logits were all identical at t=0
(max_abs_diff=4.77e-07). This means every H96/H720 result difference
between the two WCE arms originates ENTIRELY from t>=1 (prefix-conditioned
Set aggregation vs prefix-invariant Individual MSE) -- t=0 contributes
nothing to the divergence.

**6. t>=1에서 Set-WCE의 이점이 나타났는가?**
NOT in final Stage-2 MSE (Q2/Q4 above: Set-WCE underperforms Individual-
WCE and its own Hard-CE counterpart at both horizons). It DID show a
consistent Stage-1-internal pattern at t>=1: Set's `chosen_regret_t>=1`
(0.10-0.13) is far lower than Individual's (0.46-0.49), and Set's
`top10_hit_t>=1` / `ndcg@10_t>=1` are directionally different too -- but
per spec section 13 these are Oracle-SPECIFIC targets and not a valid
absolute comparison; they only show Set's own Oracle target is easier for
the student to approximately track, not that this produces a *better
forecast*.

**7. Stage 1 FR-Agg 개선이 Stage 2 MSE 개선으로 이어졌는가?**
Mixed, and in one case actively contradictory: Set's H720 FR-Agg
(1.606817) is WORSE (higher) than Individual's H720 FR-Agg (1.585016), and
Set's H720 Stage-2 MSE (0.53075) is ALSO worse than Individual's
(0.47906) -- consistent in direction there. But within the Set arm alone,
comparing WCE (FR-Agg 1.606817) against its own Hard-CE counterpart's
FR-Agg (not reported in this file; see the CORRECTED report for that
value) would be needed to fully answer this per-loss, and is left as a
follow-up cross-reference rather than fabricated here. The clearest
takeaway from THIS experiment alone: the gate distribution and
counterfactual-beating pattern (Q per section 6) diverges sharply between
H96 (both arms beat both counterfactuals) and H720 (Individual does NOT
beat lambda=0, i.e. its own base-only counterfactual) -- replicating the
same H96-vs-H720 divergence found in the ChoiceCE-CORRECTED report,
independent of loss formulation.

**8. gate는 retrieval 결과를 실제로 얼마나 사용했는가?**
H96: both arms show gate_mean ~0.44-0.45, tight distributions (std
~0.04), essentially loss-independent -- WCE and Hard CE gate similarly at
H96. H720: gate usage diverges sharply by Oracle -- Individual's gate
mean collapses to 0.147 (median 0.107, p10 near 0) while Set's stays high
at 0.464 (comparable to H96). This matches the MSE pattern: Individual's
H720 gate under-uses retrieval and fails to beat its own base-only
counterfactual; Set's gate over-uses retrieval (relative to how well that
retrieval actually performs, given Set's own worse FR-Agg) and still ends
up substantially worse than Individual overall.

**9. 결과가 단일 seed 변동일 가능성이 있는가?**
YES, plausibly, for the SMALL effects (H96 both arms' WCE-vs-Hard-CE
deltas, all under 0.5% relative; H720 Individual's -0.79%). The H720 Set
regression (+8.35%) is large enough to be a real effect rather than noise,
but this experiment's own seed=0-only, non-exact-paired-against-Hard-CE
design (see ISSUES) cannot rule out seed variance as at least a
contributing factor. No claim here should be read as confirmed without
further seeds.

**10. 다음 3-seed 검증 대상으로 어떤 arm을 추천하는가?**
`set_onpolicy_cosine_wce` at H720, specifically -- it shows this round's
single largest, most actionable-looking effect (+8.35% regression vs Hard
CE, gate over-using a comparatively worse retrieval signal), and
distinguishing "Set Oracle genuinely degrades under WCE at long horizons"
from "this seed's Set-WCE H720 run happened to land badly" is the most
consequential open question this report raises. A secondary candidate is
`individual_onpolicy_cosine_wce` at H720 (the lambda0-losing arm), to
check whether that base-only-counterfactual failure is seed-specific or a
structural property of this architecture at H720 (as already flagged,
independent of loss, in the ChoiceCE-CORRECTED report).
