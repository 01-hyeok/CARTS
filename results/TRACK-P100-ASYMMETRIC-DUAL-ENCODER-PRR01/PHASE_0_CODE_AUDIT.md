# PHASE 0 -- Repository Audit & Correctness Validation

No new code was run for this Phase beyond existing tests and reading
already-saved artifacts. All verification below is either (a) an
existing, passing unit test, (b) a direct read of an already-saved
JSON artifact, or (c) a direct read of existing source code. Nothing
here was estimated.

## Code paths reviewed (section 3)

| Path | Reviewed | Notes |
|---|---|---|
| `scripts/train_v_sharedtop100_01.py` | YES (full read, this session) | Canonical V5 training loop, dual checkpoint policy (`checkpoint_best_retmse`/`checkpoint_best_kl`) |
| `scripts/train_retriever_pool01.py` | YES (full read, this session) | `CandidatePoolCache`, `compute_scores` (full/coarse_topk modes) |
| `utils/mean_mixture_selection.py` | YES (full read, this session) | `mean_mixture_topk_selection`: `p_h=softmax(masked/tau_s)`, `p_mix=mean_h p_h`, defensively re-masked before topk |
| `utils/candidate_pool.py` | YES (full read, this session) | `compute_coarse_delta_last_cosine_scores`, `build_coarse_pool`, never reads `batch_y`/future by construction (no such argument exists) |
| `research/J-shared-encoder-drift/TRACK-J2-KEY-UPDATE-DECOMPOSITION01.md` | YES (full read, this track) | **Directly relevant prior art for RQ1/RQ2** -- see section below |
| `scripts/train_j2_key_update_decomposition01.py` | Referenced (not re-read line-by-line; J2's own report documents its exact mechanics and fingerprints) | True-Frozen-Key implementation pattern (deepcopy + requires_grad_(False) + assert-never-drifts) is the correct template for any frozen-key control in this track, should one be needed |
| `scripts/train_k_multislot_predictive_retrieval01.py` | YES (`SlotHeads`, `kl_loss_from_prob` read/reused extensively this session) | |
| `scripts/train_r_stage2_lambda01.py` | YES (full read, this session, in the prior audit track) | `load_tensors`, `GlobalLambdaGate` (`lambda=sigmoid(a)`), `eval_gate` |
| `scripts/eval_r_stage2_lambda_signfix01.py` | YES (full read, this session) | Confirms the dual-checkpoint reuse pattern this track's E0 should follow if E0 is re-run |
| `scripts/train_margutil01.py` | YES (`build_experiment`, `memory_value` read/reused extensively this session) | |
| `FINAL_RESEARCH_REPORT.md` | Exists, not re-read line-by-line this Phase (out of this track's immediate scope; the specific baseline numbers needed are independently reproduced below from primary JSON artifacts, not from this file's prose) | |
| `research/RESEARCH_DECISIONS.md` | Referenced throughout this session; TRACK-V-SHARED-TOP100-SIGNFIX01's provenance (teacher-sign bug) already tracked here from earlier in this session | |
| `results/TRACK-HARD-EXPERT-V5-P100-ALLH01/REPORT.md` | Exists; not required for this Phase's specific checks (Hard-expert routing is orthogonal to the Shared-vs-Separate-Encoder question) -- marked NOT VERIFIED / NOT NEEDED | |
| `scripts/train_horizon_retrieval_expert01.py` | YES (`normalized_teacher_prob` read and tested extensively this session, including the original teacher-sign-bug discovery earlier in this session) | |

## A. Teacher Sign -- PASS

`normalized_teacher_prob(d, valid_mask, tau_t)` (verified by direct
read + existing passing tests `tests/test_horizon_retrieval_expert01.py::test_normalized_teacher_prob_masking_and_zscore`
and `::test_teacher_diagnostics_uniform_vs_onehot`, both pass): z
-score-normalizes `d` (lower-is-better future distance) over each
row's valid candidates, then `softmax(-normalized/tau_t)` -- masked
entries get probability exactly 0. This is the SIGN-FIXED call
convention (passing `d` directly, never double-negated) --
`TRACK-V-SHARED-TOP100-SIGNFIX01` is the canonical source for this
fix, established earlier in this session; this track uses ONLY
sign-fixed checkpoints (`checkpoints/track_v_sharedtop100_signfix01/`),
confirmed by checkpoint path audit below. Gradient-to-teacher: `p_t`
is `.detach()`-ed inside `kl_loss`/`kl_loss_from_prob` (verified by
direct read of `utils/mean_mixture_selection.py` and
`train_k_multislot_predictive_retrieval01.py::kl_loss_from_prob` in
this session) -- no gradient reaches the teacher.

## B. Temporal Leakage -- PASS

Existing tests `tests/test_j_shared_encoder_drift01.py::test_item13_no_val_test_leakage_in_candidate_bank`
and `tests/test_p_fusion_semantics_audit01.py::test_item20_no_future_leakage`
both PASS (re-run for this audit, not assumed). `compute_coarse_delta_last_cosine_scores`
(the coarse score underlying Shared-P100 itself) takes `batch_x` and
`memory_x` only -- there is no `batch_y`/future parameter in its
signature by construction, so it is structurally impossible for this
function to leak future information, independent of any test. RAFT
candidate mask convention (`exp._candidate_mask`) is reused UNMODIFIED
from every other track this session -- not re-derived here.

## C. Future Value Alignment -- PASS

`memory_value(args, batch_x, memory_y, memory_x_last, target_channel)`
(`scripts/train_margutil01.py`, read in full this session): when
`relation_value_space == 'delta_last'`, `memory_value_c -=
memory_x_last[:, target_channel]`; `query_offset =
batch_x[:, -1, target_channel]`. `candidate_future =
memory_value_c + query_offset` recovers the ABSOLUTE future space,
matching `batch_y` directly -- this exact reconstruction was
independently cross-validated dozens of times this session (every
Phase of `TRACK-P100-COARSE-TO-FINE-RETRIEVAL01` and
`TRACK-P100-RETRIEVAL-UTILIZATION-AUDIT01` computed AggMSE this way
and the numbers always matched canonical artifacts to 6 decimal
places). No absolute/delta-last space mixing was found.

## D. Gradient -- PASS (query branch) / PASS (key branch, shared-encoder case)

Existing tests `tests/test_v_multiquery_generalization01.py::test_item4_full_gradient_all_branches`
and `::test_item4b_v0_full_gradient_both_branches` (both re-run for
this audit, PASS) directly verify that in the canonical SHARED
-encoder V5 training setup, both the query-branch and key/candidate
-branch gradients are nonzero after backward. This establishes E0's
own gradient correctness as the starting point for Phase 1's
comparison -- not yet verified: E1/E2/E3's own gradient flow (separate
query/key parameters), which does not exist in the repository yet and
is therefore NOT VERIFIED until Phase 1 implements and smoke-tests it.

**Directly relevant prior art (not previously connected to this
question until this audit)**: `TRACK-J2-KEY-UPDATE-DECOMPOSITION01`
(full report read above) already ran a CAUSAL intervention closely
related to this track's E1/E2 hypothesis, on ETTh1 H720, full-memory
(not Shared-P100): J0 (joint shared encoder, gradient both sides) vs
J1 (shared encoder, key-side gradient stopped) vs J2 (TRUE separate
encoders, key encoder frozen at init, never updated). Result: **J0 >
J1 > J2 on retMSE@10/Oracle regret (shared-joint wins), but the
ranking REVERSES on Uniform Aggregate MSE@10 (J1 > J2 > J0)** -- the
same individual-vs-aggregate metric disagreement this session has
repeatedly found elsewhere. This is NOT the same experiment as this
track's E1 (J2's key encoder is FROZEN at init, never jointly
trained, whereas E1 trains both encoders jointly) -- so TRACK-J2 does
not pre-empt this track's result -- but it is strong, directly
relevant prior evidence that simply separating the key representation
from the query (even as an extreme frozen-key case) does NOT uniformly
help on the primary per-candidate retrieval metric, while it CAN help
on the aggregate metric. **This track's Phase 1 report must explicitly
discuss TRACK-J2 in its Comparison A/Case interpretation** -- failing
to do so would risk re-discovering (or contradicting) already
-established evidence without acknowledgment, which this audit flags
now specifically to prevent.

## E. Baseline Reproduction -- PASS (exact, to 5 decimals)

Read directly from `results/TRACK-P100-COARSE-TO-FINE-RETRIEVAL01/STAGE2/metrics/<cell>/V5/metrics.json`
(the ORIGINAL artifact, not re-run, not rounded prose):

| Cell | Spec-quoted canonical V5 | This repo's own saved artifact (`test_mse`, full precision) | Match |
|---|---:|---:|---|
| ETTh1 H96 | 0.36960 | 0.36959707736968994 | PASS (rounds to 0.36960) |
| ETTh1 H720 | 0.51492 | 0.5149174928665161 | PASS (rounds to 0.51492) |
| Weather H96 | 0.17665 | 0.1766497790813446 | PASS (rounds to 0.17665) |
| Weather H720 | 0.38553 | 0.38552603125572205 | PASS (rounds to 0.38553) |

All 4 cells reproduce exactly. Checkpoint provenance for the
underlying V5 Stage1 scores (confirmed via `selected_checkpoint.json`
in an earlier track this session, re-confirmed by path existence
here): `checkpoints/track_v_sharedtop100_signfix01/{ETTh1,Weather}/H{96,720}/seed0/pool_top100/.../V5/checkpoint_epoch{10,9,4,6}.pth`
-- the sign-fixed checkpoints, never the historical
`INVALIDATED_BY_TEACHER_SIGN_BUG`-marked ones.

## Summary table

| Item | Status |
|---|---|
| A. Teacher sign | PASS |
| B. Temporal leakage | PASS |
| C. Future value alignment | PASS |
| D. Gradient (E0 shared-encoder baseline) | PASS |
| D. Gradient (E1/E2/E3, not yet implemented) | NOT VERIFIED (deferred to Phase 1 smoke test, per spec's own sequencing) |
| E. Baseline reproduction (4 cells) | PASS (exact to 5 decimals, from original artifact) |

**No fatal error was found.** Per spec section 2's explicit rule
("Correctness Audit에서 치명적인 오류가 발견되면... 실험을 진행하지
않는다"), Phase 1 is cleared to proceed. The one substantive finding
from this Phase is not a bug but a piece of directly relevant prior
evidence (TRACK-J2) that tempers the prior expectation for E1: a
causally-intervened, non-shared key representation has ALREADY been
shown, on this exact codebase and a closely related cell, to help
aggregate-level metrics while hurting individual-level retrieval
metrics -- Phase 1's own E0-E3 comparison should be read against this
established pattern, not treated as the first evidence on the
question.
