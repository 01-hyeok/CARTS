# TRACK-W-CHECKPOINT-CRITERION-CORRECTION01 -- AUDIT.md

## PART -1: governance policy adopted mid-track (2026-10-01)

The user issued a standing CHECKPOINT/MODEL-SELECTION GOVERNANCE RULE
during this track's execution. It formalizes exactly the principle this
track was already applying (default = validation-objective-aligned
selection; any TYPE C diagnostic/downstream metric used for Stage1
checkpoint selection requires explicit, documented justification on
four axes -- literature, task-alignment, fairness, circularity -- or
is disallowed). TRACK-W's own corrected criteria were already aligned
with the policy's PART 8 "current project default policy" table before
the policy was written down:

| Experiment | Default per new policy | TRACK-W's corrected criterion | Match |
|---|---|---|---|
| Original KL | min val KL | min val KL | YES |
| Multi-Query KL | min val KL | min val KL | YES |
| Hard Choice CE / Set Oracle (B Host-Free) | min val Choice CE | min val_tf_choice_ce | YES |
| Base Forecaster | min val forecast MSE | unchanged (already min val forecast MSE) | YES |
| Raw Cosine Stage1 | no training/no checkpoint | unchanged (not applicable) | YES |

Going forward in this track: (1) every expensive GPU run prints the
mandatory `[MODEL-SELECTION AUDIT]` block (training objective /
validation selection metric / early-stopping metric / identical? /
decision) before starting -- added to
`scripts/train_w_hostfree_corrected01.py`; (2) early-stopping and
checkpoint-selection are confirmed to use the SAME metric everywhere in
this track (no train/select/early-stop metric mixing anywhere -- see
PART 1 below, all three families use one unified validation-objective
metric for both roles).

**Scope note on the policy's PART 7 (historical audit table covering
ALL Stage1 experiments since 2026-08-25)**: that table's scope is
broader than this track's own five-experiment-family scope (Original
KL, Multi-Query, B Host-Free, Base Forecaster, Raw Cosine). A fully
exhaustive historical audit would additionally need to cover every
other Stage1 trainer this session has produced (`train_factorial_
e2e01.py`'s own onpolicy/tf-oracle arms, `train_tf_oracle_learnability01
.py`, `train_setlossctrl01.py`/`train_setlossctrl_stage2_retrain01-02.py`,
`train_k_multislot_predictive_retrieval01.py`, `train_m_relevance_
constrained_multislot01.py`, `train_q_gate_capacity01.py`, and others).
Flagged to the user as a separate scope question rather than silently
expanding this track's own GPU workload further while two jobs already
share GPU1.

## PART -0.5: FULL HISTORICAL AUDIT (policy PART 7, all Stage1 trainers since 2026-08-25)

Per the user's explicit decision (2026-10-01) to start this immediately
via static code reading (no GPU needed). Scanned all 60 `scripts/
train_*.py` files for (a) an explicit `checkpoint_criterion` string (22
files) or an inferred `best['...']` checkpoint-selection pattern (21
more), then cross-checked each one's ACTUAL training loss against its
checkpoint-selection metric by reading the training loop directly. This
is a much larger scope than TRACK-W's own 5-family correction target
(PART 1 below) -- it covers every Stage1 trainer in the repository, not
just the ones this track is actively correcting.

### Confirmed findings

| Script | Experiment lineage | Train Objective | Checkpoint Criterion | Match? | Status |
|---|---|---|---|---|---|
| `train_j_shared_encoder_drift01.py` | Original KL / J0 | KL(p_T\|\|p_S) | val retMSE@10 | NO | **AFFECTED** (TRACK-W target) |
| `train_t_pure_multislot01.py` | Multi-Query / TRACK-T | KL(p_T\|\|mean p_m) | val retMSE@10 | NO | **AFFECTED** (TRACK-W target) |
| `train_u_asymmetry_capacity01.py` | TRACK-U | KL | val retMSE@10 | NO | **AFFECTED** (not in TRACK-W's current GPU scope -- flagged) |
| `train_j2_key_update_decomposition01.py` | J1/J2 StopGrad-Key -- **upstream of the ENTIRE M2/J1 lineage reused by TRACK-K/M/R/S/T/T2/U/V** | KL | val model_top10_individual_mse (=retMSE@10) | NO | **AFFECTED** |
| `train_setoracle_hostfree01.py` | B Host-Free | hard_choice_ce (teacher-forced) | val free_running_aggregate_future_mse | NO | **AFFECTED** (TRACK-W target) |
| `train_factorial_e2e01.py` | root Sequential Set Retriever (ancestor of B Host-Free / tf-oracle / onpolicy families) | oracle_choice_cross_entropy | val free_running_aggregate_future_mse | NO | **AFFECTED** |
| `train_tf_oracle_learnability01.py` | sibling of B Host-Free (HostScorer instead of UniformHost) | hard_choice_ce | val free_running_aggregate_future_mse | NO | **AFFECTED** |
| `train_onpolicy_rankloss01.py` | onpolicy rank-loss family | rank-loss (arm-dependent, e.g. R0_choice_ce) | val free_running_aggregate_future_mse | NO | **AFFECTED** |
| `train_set_loss01.py` / `train_set_loss_control01.py` / `train_set_loss_control02.py` | Set Loss (Control) family | arm-dependent (`loss_name=cli.arm`) | val free_running_aggregate_future_mse | NO (by construction, unless one arm's own loss literally IS this metric) | **AFFECTED** |
| `train_set_normregret_control01.py` | norm-regret control | norm-regret loss | val free_running_aggregate_future_mse | NO | **AFFECTED** |
| `train_oracle_wce_control01.py` | oracle WCE control | weighted-CE loss | val free_running_aggregate_future_mse | NO | **AFFECTED** |
| `train_multipos_choice01.py` | multi-position choice | multi-position choice loss | val free_running_aggregate_future_mse | NO | **AFFECTED** |
| `train_i_pca_future_teacher01.py` | PCA future-teacher | KL (PCA-projected teacher) | val model_top10_individual_mse (=retMSE@10) | NO | **AFFECTED** |
| `train_horizon_retrieval_expert01.py` | horizon retrieval expert | KL (global + per-block blend) | val full-H720 uniform-aggregate hard Top-10 MSE (=AggMSE) | NO | **AFFECTED** |
| `train_patch_retrieval_expert01.py` | patch retrieval expert | KL | val model_top10_individual_mse (=retMSE@10) | NO | **AFFECTED** |
| `train_k_multislot_predictive_retrieval01.py` | TRACK-K (K1/K2) -- **upstream of TRACK-M's M1/M2** | K1: anchor KL + overlap; K2: + aggregate term | val HARD uniform_agg_mse10 (=AggMSE) | NO (K1: AggMSE not even in training loss; K2: AggMSE is PART of the composite loss but selection ignores the anchor-KL/overlap terms) | **AFFECTED** (policy explicitly bans AggMSE for Stage1 selection regardless of partial training-loss overlap) |
| `train_m_relevance_constrained_multislot01.py` | TRACK-M (M1/M2) -- **M2 directly reused by TRACK-R/S/T/T2/U/V as "M2"** | anchor KL + overlap + aggregate + relevance-budget (composite) | val agg_mse10 s.t. feasibility constraint (=AggMSE) | NO | **AFFECTED** |
| `train_f_late_interaction_control01.py` / `train_f_late_interaction_probe01.py` | TRACK-F late interaction | KL | val retMSE@10 (`best_val_retmse10`) | NO | **AFFECTED** |
| `train_g_decoupled_metric_adaptation01.py` | TRACK-G decoupled metric | KL | val retMSE@10 | NO | **AFFECTED** |
| `train_h_direct_set_utility01.py` | TRACK-H direct set utility | KL (individual term) | val retMSE@10 | NO | **AFFECTED** |
| `train_c_horizon_clean02.py` | TRACK-C horizon-clean | KL (global + blockwise, `kl_loss` from `train_horizon_retrieval_expert01`) | val global_h720_mse / block_h720_mse (per-arm, =AggMSE-equivalent) | NO | **AFFECTED** |
| `train_margutil01.py` | margin-utility (set_conditioner + utility_head) | smooth_l1 margin-utility regression loss (`step_losses`) | val overlap@k (TYPE C retrieval diagnostic) | NO | **AFFECTED** |
| `train_experiment_e2_multisubspace01.py` | E2 multi-subspace | (subspace-dependent KL) | self-flagged `[ISSUE] spec ambiguous on which subspace Top-10 to checkpoint on` | N/A | **UNKNOWN** (author self-flagged; not independently resolved here) |
| `train_r_base_forecaster01.py` | Base Forecaster (all later tracks) | forecast MSE | val forecast MSE | YES | UNAFFECTED |
| `train_factorial_e2e01_base_only.py` | base-only control | (needs confirm, likely MSE per name) | val MSE | LIKELY YES | UNAFFECTED (pending final confirm) |
| `build_r_retrieval_cache01.py` (Raw Cosine) | Raw Cosine Stage1 | none (no training) | N/A | N/A | UNAFFECTED/not applicable |
| `train_r_stage2_lambda01.py`, `train_q_gate_capacity01.py`, `train_choicece_stage2_retrain01.py`, `train_setlossctrl_stage2_retrain01.py`, `train_setlossctrl_stage2_retrain02.py`, `train_oracle_wce_stage2_control01.py`, `train_set_normregret_stage2_control01.py` | Stage2 fusion trainers (`exp._loss(y_final, y_base, y_ret, ...)`) | Stage2 fusion forecast loss | val final_mse (Stage2 forecast MSE) | YES | UNAFFECTED (Stage2 family) |
| `train_n_gate_only01.py`, `train_o_frozen_base_consumer01.py`, `train_p_fusion_semantics01.py` | reuse `train_epoch` from `train_setlossctrl_stage2_retrain02.py` unmodified | Stage2 fusion forecast loss (same as above) | val final_mse | YES | UNAFFECTED (Stage2 family) |
| `train_cross_channel_resdirect.py` | residual-direct cross-channel | forecast MSE (`F.mse_loss`) | val_mse | YES | UNAFFECTED |
| `train_cross_channel_ressel.py`, `train_feature_ladder.py`, `train_utility_reranker.py` | shortlist reranker probes (closed, see memory "Shortlist reranking: headroom exists, not learnable") | surrogate choice/KL loss over candidate utility (top1_ce or kl_div -- needed for backprop through a discrete pick) | val forecast MSE (the model's actual deliverable metric, not a retrieval diagnostic) | Train≠checkpoint by the literal TYPE A=TYPE B rule, BUT the checkpoint metric is the end-task forecast MSE itself (not a banned TYPE C retrieval diagnostic like retMSE/AggMSE/D/C) and the surrogate loss is required only because forecast MSE is not directly differentiable through a discrete top-1 selection | **JUSTIFIED-DIAGNOSTIC** (Stage2-deliverable pattern: checkpointing on the true end metric when the train loss is a necessary differentiable surrogate for a discrete selection step is the standard reranker-training pattern, distinct from the Stage1 KL-train/retMSE-checkpoint mismatch this audit is otherwise flagging) |
| `train_c_horizon_clean04_expertwise.py`, `train_c_horizon_frozen03.py` | TRACK-C horizon-clean variants (expertwise / frozen) | KL (`kl_loss` from `train_horizon_retrieval_expert01`) | val block_h720_mse / block_mse (=AggMSE-equivalent) | NO | **AFFECTED** |

### Not yet individually confirmed (flagged, not fabricated)

All files originally in this bucket have now been resolved into the
main table above EXCEPT the following, which do not use the `best['val']`
checkpoint-selection pattern searched for and would each need an
individual read of their own (distinct) selection logic to classify
correctly rather than a guess from a grep miss:
`train_query_residual_selector.py`, `train_toptail_rank01.py`,
`train_utility_classifier.py`, `train_utility_ranker.py`,
`train_oracle_choice01.py`, `train_oracle_rank_gain01.py`,
`train_oracle_scratch01.py` (+ its `_base_only`/`_stage2`/
`_frozenbase01_stage2`/`_tf01` siblings), `train_onpolicy_choice01.py`,
`train_onpolicy_prefix01.py`, `train_patch_moe_router01.py`,
`train_rank_gain01_stage2_fusion.py`, `train_seqfull01.py`,
`train_correction_encoder01.py` (+ `_stage2`), `train_correction_
selector01.py`, `train_correction_stage2_semantics01.py`,
`train_experiment_e2_multisubspace01.py` (self-flagged `[ISSUE]`
ambiguous by its own original author -- already UNKNOWN by its own
admission, now in the main table). These remain **UNKNOWN**; NOT
assumed AFFECTED or UNAFFECTED without verification. Given the
overwhelming pattern above (every confirmed Stage1 KL/CE/composite-loss
trainer this session has produced uses a TYPE C diagnostic metric --
retMSE@10 or AggMSE or free-running-aggregate-MSE -- for checkpoint
selection instead of its own training objective), the PRIOR for most of
these being AFFECTED too is high, but this is stated as a prior, not a
confirmed finding. Most of these (`oracle_choice`, `oracle_rank_gain`,
`onpolicy_choice/prefix`, `seqfull`, `correction_*`) are pre-TRACK-named
EXP-series probes (see `research/EXPERIMENT_LOG.md`) already closed with
a STOP verdict per memory notes -- their checkpoint criterion is a
historical-record question, not a live-experiment risk, so they are
deprioritized relative to TRACK-W's actual GPU-side correction work.

### Headline implication

**The checkpoint-selection/training-objective mismatch is not a
TRACK-W-specific issue -- it is the DEFAULT pattern across nearly every
Stage1 retrieval trainer produced in this entire research session.**
Most critically: `train_j2_key_update_decomposition01.py` (J1/J2,
StopGrad-Key) and `train_k_multislot_predictive_retrieval01.py`/
`train_m_relevance_constrained_multislot01.py` (K1/K2, M1/M2) are
AFFECTED, and these are the checkpoints EVERY LATER TRACK this session
(R, S, T, T2, U, V) has reused, read-only, as "J1" / "M2" ground truth
without ever re-auditing their own selection criterion. TRACK-W's
current GPU-side correction (PART 1-6 below) covers only Original KL /
Multi-Query / B Host-Free; it does NOT yet correct J1/J2/K/M's own
Stage1 checkpoints. This is flagged to the user as a separate,
larger-scope decision (see the main report's open-questions section)
rather than silently expanded into TRACK-W's already-running GPU
workload.

## PART 0: parallel-execution safety (before anything else)

Verified via `nvidia-smi`/`ps` before starting: GPU1 has exactly ONE
existing job -- TRACK-V's `train_t_pure_multislot01.py --num_slots 2
--cell Weather_720` (PID 3638088, Weather H720 V2 Stage1, actively
training). This job is NEVER killed, restarted, or read from mid-write
anywhere in this track. TRACK-W runs as the SECOND top-level job on
GPU1 (`CUDA_VISIBLE_DEVICES=1`), with its own internal steps executed
strictly sequentially (no parallel children within TRACK-W).

## PART 1: code audit (items A-E, confirmed against the actual current code)

### A. Original KL (`scripts/train_j_shared_encoder_drift01.py`)
Confirmed via source read (line ~453-458, ~524-529): training loss is
`kl_loss(p_t, s, cand_mask, tau_s)` (the genuine KL objective);
checkpoint/early-stop criterion is `val_metric = val_sums[
'model_top10_individual_mse'] / ...` (i.e. val retMSE@10), compared via
`if val_metric < best['val']`. **Objective/selection mismatch
confirmed.** The validation loop computes ONLY this hard-selection
metric -- no validation-split KL loss is computed or logged anywhere
in the existing pipeline.

### B. Multi-Query (`scripts/train_t_pure_multislot01.py`)
Confirmed via source read (line ~340-350 training loop, ~376-392
validation loop): training loss is `kl_loss_from_prob(p_t, p_bar,
cand_mask)` where `p_bar` is the mean-of-slot-softmaxes (`KL(p_T ||
mean_m p_m)`); checkpoint criterion is `val_retmse10 < best[
'val_retmse10']` (hard round-robin Top-10 individual MSE). **Same
mismatch pattern as A.** No validation-split KL loss computed or
logged in the existing pipeline either.

### C. B Host-Free (`scripts/train_setoracle_hostfree01.py`, experiment
TRACK-A-SETORACLE-HOSTFREE01, arms `individual_tf_hostfree_cosine` /
`set_tf_hostfree_cosine`)
Confirmed via full source read: training loss is `hard_choice_ce` (via
`run_sequence(..., prefix='tf', ...)`, teacher-forced); checkpoint
criterion is `row['val_free_running_aggregate_future_mse'] < best[
'val']` -- a FREE-RUNNING (not teacher-forced) aggregate-MSE metric,
which is neither the training loss nor even the same prefix-policy
axis (`tf` train vs `free_running` eval-selection metric). **Confirmed
UniformHost-based (`host.scores()` returns all-zero for every valid
candidate -- no Stage-2 host checkpoint loaded, referenced, or hashed
anywhere in this script or its cache builder).** Crucially: this
script ALREADY computes and logs `val_tf_choice_ce` every epoch
(`val_split_tf, _ = teacher_forced_diagnostics(...)` on the full val
split, stored as `val_tf_tf_choice_ce` in `epoch_metrics_{arm}.csv`) --
it was simply never used for selection. This means CASE-A cells can be
corrected via pure re-selection from already-logged data, cross-checked
by reloading.

### D. Base Forecaster (`scripts/train_r_base_forecaster01.py`)
Confirmed via source read: training loss is forecast MSE; checkpoint
criterion is `val_mse < best['val_mse']` -- the SAME objective family
(forecast MSE) on both sides. **No mismatch. NOT a correction target.**
Per PART 7.1's explicit instruction, this is NOT reported as "no
training" -- it trains normally, and its existing checkpoint-selection
was already correct.

### E. Raw Cosine (`scripts/build_r_retrieval_cache01.py::cosine_scores`)
Confirmed via source read: `cosine_scores` computes cosine similarity
directly on the raw delta-last input window (`batch_x[:,:,c] -
batch_x[:,-1:,c]` vs the same for `memory_x`) -- no learned encoder, no
`nn.Module` with trainable parameters anywhere in this code path, no
checkpoint file ever written or loaded for it. **No training occurs;
not a correction target for Stage1.** Stage2 for Raw Cosine (built on
top of this Stage1 cache + a frozen base forecaster + the standard
trainable-lambda fusion) is a SEPARATE question addressed in PART 7.3
below (Stage2 reuse depends only on cache/base-checkpoint/config
hashes matching, not on any Stage1 checkpoint-selection issue, since
Raw Cosine has no Stage1 checkpoint at all).

## PART 2: epoch-checkpoint completeness audit (CASE A vs CASE B)

### B Host-Free (`checkpoints/track_a_setoracle_hostfree01/<cell>/<arm>/`)
`train_epochs=10` default was used for all cells (confirmed via each
cell's `config_fingerprint_{arm}.json`, `epochs: 10`). Checkpoint counts
found on disk:

| Cell | Arm | epoch checkpoints found | Complete (10)? | Case |
|---|---|---:|---|---|
| ETTh1_96 | individual_tf_hostfree_cosine | 6 | NO | **B (retrain)** |
| ETTh1_96 | set_tf_hostfree_cosine | 7 | NO | **B (retrain)** |
| ETTh1_720 | individual_tf_hostfree_cosine | 6 | NO | **B (retrain)** |
| ETTh1_720 | set_tf_hostfree_cosine | 10 | YES | A (reselect) |
| Weather_96 | individual_tf_hostfree_cosine | 6 | NO | **B (retrain)** |
| Weather_96 | set_tf_hostfree_cosine | 10 | YES | A (reselect) |
| Weather_720 | individual_tf_hostfree_cosine | 10 | YES | A (reselect) |
| Weather_720 | set_tf_hostfree_cosine | 6 | NO | **B (retrain)** |

5 of 8 (cell, arm) combinations require fresh retraining (early-stopped
under the OLD free-running-aggregate-MSE criterion before the full
10-epoch trajectory existed); 3 can be corrected by pure re-selection
from already-logged `val_tf_tf_choice_ce` values (cross-validated by
reloading each saved epoch checkpoint).

### Original KL / Multi-Query (Correction B)
Completeness audit for ETTh1/Weather x H96/H720 x S={0,1,2,4,10} is
performed just before each cell's correction step (PART 4.3/4.4 of the
spec) -- see the main report body for the resulting table, since
several of these settings (Weather S=0/1/2) are still being produced
by the concurrently-running TRACK-V job at audit time and must not be
touched until TRACK-V's own completion marker confirms they are done.

## PART 3: reuse verification (Base Forecaster, Raw Cosine)

Deferred to the main report body (requires computing SHA-256 over the
actual checkpoint/cache files per dataset/horizon, done as part of
execution, not static code reading) -- see `BASE_REUSE` /
`RAW_COSINE_STAGE1_REUSE` / `RAW_COSINE_STAGE2_REUSE` tables there.

## Corrected checkpoint-selection principle (fixed for this track)

| Experiment | Training objective | OLD selection | CORRECTED selection |
|---|---|---|---|
| Original KL (S=0) | KL(p_T \|\| p_S) | val retMSE@10 | val KL |
| Multi-Query (S>=1) | KL(p_T \|\| mean_m p_m) | val retMSE@10 | val KL |
| B Host-Free (both arms) | hard_choice_ce (teacher-forced) | val free-running agg MSE | val teacher-forced choice CE |
| Base Forecaster | forecast MSE | val forecast MSE | unchanged (already correct) |
| Raw Cosine Stage1 | none (no training) | N/A | unchanged (not applicable) |

No training math, teacher, candidate mask, optimizer, LR, batch size,
seed, or architecture is changed anywhere in this track -- only WHICH
epoch's checkpoint gets selected (and, where necessary, the
early-stopping patience trigger that decides how many epochs to run).

## PART 5: Correction B (Original KL / Multi-Query) -- in progress, 2026-10-05

### Epoch-checkpoint completeness audit (resolves PART 2's deferred table)

| Cell | S | Source track | Epochs found | Complete (10)? | Case |
|---|---|---|---:|---|---|
| ETTh1_720 | 0 | TRACK-V (V0) | 10 | YES | A |
| ETTh1_720 | 1,2,4,10 | TRACK-T (S1/S2/S4/S10) | 10 each | YES | A |
| ETTh1_96 | 0,1,2 | TRACK-V(V0)/TRACK-T(S1,S2) | 10 each | YES | A |
| ETTh1_96 | 4,10 | TRACK-T (S4/S10) | 6 each | NO (patience-5 stop, not the 10-epoch cap) | **B (retrain)** |
| Weather_96/720 | 0,1,2 | TRACK-V (V0/V1/V2) | not yet audited here (deferred, see below) | - | - |
| Weather_96/720 | 4,10 | **none -- never trained by any prior track** | 0 | N/A | requires NEW Stage1 training if included (user decision pending) |

TRACK-T is ETTh1-only (confirmed: no Weather checkpoints exist under
`checkpoints/track_t_pure_multislot_validation01/`); TRACK-V covers
S={0,1,2,5} for both datasets but never S={4,10}. For S=1/S=2, TRACK-T's
own checkpoints (not TRACK-V's V1/V2) are used as the canonical S=1/S=2
source for Correction B, since S=4/S=10 only exist there and keeping
the whole S-grid sourced from one track avoids an arbitrary TRACK-T/
TRACK-V mix. **Whether to additionally train Weather S=4/S=10 from
scratch (4 new multi-hour-to-30-hour GPU1 runs) is an open question put
to the user; not started pending their answer.**

### CASE A re-selections complete (7/7: ETTh1_720 all 5 S-values + ETTh1_96 S=0,1,2)

All showed a striking pattern: **every corrected checkpoint for
ETTh1_720 selects epoch 1** (S=0,1,2,4,10 alike) -- val KL is best at
the very first epoch and gets monotonically worse every epoch after,
even as the OLD metric (val retmse10) kept improving through epoch 6
-10. ETTh1_96 is different: S=0,1,2 all select epoch 5 (not epoch 1).
This means the ENTIRE TRACK-R/S/T/T2/U/V "H720 multi-query helps"
narrative was built on epoch 6-10 checkpoints that the corrected
(training-objective-aligned) criterion would never have selected at
all -- whether that narrative survives Stage2 re-evaluation on the
corrected checkpoints is not yet known (cache+Stage2 rebuild pending).

### ETTh1 Correction B -- Stage2 RESULTS (2026-10-05, all 10 cache+Stage2 runs complete)

| Horizon | S | OLD (val_retmse10) test_mse | CORRECTED (val_kl) test_mse |
|---|---|---:|---:|
| 96 | 0 | 0.373682 | 0.372457 |
| 96 | 1 | 0.375676 | 0.374271 |
| 96 | 2 | 0.377003 | 0.378916 |
| 96 | 4 | 0.376284 | 0.380433 |
| 96 | 10 | 0.375539 | 0.379549 |
| 720 | 0 | 0.529134 | 0.493465 |
| 720 | 1 | 0.506830 | 0.491350 |
| 720 | 2 | 0.504628 | 0.488027 |
| 720 | 4 | 0.500167 | 0.489150 |
| 720 | 10 | 0.497591 | 0.487977 |

(OLD H96 S4/S10 from TRACK-T's own `summary/stage2_metrics.csv`; OLD
H720 all S and S0 at both horizons from TRACK-V's V0/historical values,
already established earlier this session.)

**Headline finding -- unlike B Host-Free, the qualitative conclusion
MOSTLY SURVIVES correction here.** The H96-degrades/H720-improves
horizon-reversal pattern (more query views hurt at H96, help at H720)
holds in BOTH the OLD and CORRECTED tables: H96's S0 is still the best
arm and S4/S10 still the worst under either criterion; H720's S0 is
still the worst arm and S10 still the best under either criterion. The
ABSOLUTE magnitude changed substantially for H720 (S0: 0.529 -> 0.493,
a large improvement from the epoch10->epoch1 reselection), so any
report quoting the OLD absolute H720 numbers needs updating, but the
qualitative S-scaling direction this session has built an entire
narrative on (TRACK-R/S/T/T2/U/V) is NOT one of the conclusions this
correction overturns -- at least for ETTh1. One pairwise ordering did
flip (OLD: S4 < S2 at H96; CORRECTED: S2 < S4), a minor exception to
the otherwise-preserved monotonic trend.

Weather's own Correction B (S=0,1,2 at minimum; S=4,10 pending the
user's scope decision) has not yet been run -- GPU1's second slot
currently still held by TRACK-W-TIMESTAMP-FUSION01's Phase 1 (Weather_720/C1).

### CASE B retrain: a real orchestration bug found and fixed

`train_w_klmultiquery_corrected01.py`'s first run (ETTh1_96 S=4 then
S=10, same invoking batch script) wrote both arms' checkpoints to the
SAME flat `--checkpoints`/`--out_dir` path (the script, copied from
`train_t_pure_multislot01.py`, relies on the CALLER passing distinct
paths per arm rather than scoping internally) -- S=10's checkpoint
silently overwrote S=4's at every epoch number, and S=4's results were
lost. Fixed by scoping `out_dir`/`ckpt_dir` to `<path>/<cell>/<arm>`
INSIDE the script itself (so this class of mistake cannot recur even if
a future caller forgets to pass distinct paths), and both arms were
rerun cleanly from scratch. No corrupted checkpoint was ever used
downstream (caught before any cache/Stage2 step touched it).

## PART 4: B Host-Free correction -- RESULTS (2026-10-04, all 8 cache+Stage2 runs complete)

All 3 CASE-A re-selections (no retraining) and all 5 CASE-B fresh
retrains (corrected criterion from the start) completed; all 8
corrected checkpoints passed `build_setoracle_hostfree01_retrieval_cache.py`'s
own independent fingerprint/epoch-consistency validation before any
cache was built. Fresh Stage2 retrained for all 8 (never checkpoint
-injected into the OLD Stage2 weights -- spec requirement). Comparison
uses `test_best.final_mse` (the best-val-epoch's test evaluation, the
metric the OLD track itself reported):

| Cell | Arm | OLD best_epoch / test_mse | CORRECTED best_epoch / test_mse | delta |
|---|---|---|---|---:|
| ETTh1_96 | individual | 7 / 0.377988 | 7 / 0.373541 | -0.004446 |
| ETTh1_96 | set | 7 / 0.372898 | 7 / 0.379168 | +0.006269 |
| ETTh1_720 | individual | 2 / 0.493384 | 1 / 0.557159 | +0.063775 |
| ETTh1_720 | set | 6 / 0.574686 | 6 / 0.520145 | -0.054540 |
| Weather_96 | individual | 3 / 0.181150 | 2 / 0.193902 | +0.012753 |
| Weather_96 | set | 7 / 0.184763 | 6 / 0.193685 | +0.008923 |
| Weather_720 | individual | 6 / 0.322206 | 6 / 0.321532 | -0.000674 |
| Weather_720 | set | 6 / 0.324521 | 4 / 0.330419 | +0.005898 |

**Headline finding -- the Individual-vs-Set-Oracle conclusion does NOT
survive correction in 3 of 4 cells.**

| Cell | OLD winner | CORRECTED winner | Flipped? |
|---|---|---|---|
| ETTh1_96 | set (0.3729 < 0.3780) | **individual** (0.3735 < 0.3792) | YES |
| ETTh1_720 | individual (0.4934 < 0.5747) | **set** (0.5572 > 0.5201 -- set now wins) | YES |
| Weather_96 | individual (0.1812 < 0.1848) | **set** (0.1937 vs 0.1939, set marginally ahead) | YES (narrow) |
| Weather_720 | individual (0.3222 < 0.3245) | individual (0.3215 < 0.3304) | no |

This is a direct, concrete answer to the question that motivated this
entire track: the checkpoint-criterion mismatch was not a cosmetic
issue -- it changed which arm the project would have reported as
"better" in 3 of the 4 cells tested. Any existing report/claim built on
the OLD B Host-Free individual-vs-set comparison should be treated as
unreliable until re-read against this corrected table.
