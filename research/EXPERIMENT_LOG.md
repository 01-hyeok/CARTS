# EXPERIMENT_LOG.md

Append-only, factual record of completed experiments. Newest entries go at the
bottom. Keep this file primarily factual; interpretation belongs to the reviewer.

Pre-existing results produced before this log was created are recorded in
`RESULTS_SUMMARY.md` (aggregate tables, 2026-08-04) and under `metrics/` and
`logs/`. They are **not** retro-fitted into this log, to avoid fabricating
history.

---

## Template

Copy this block for each completed experiment.

```markdown
## <EXP-ID, e.g. EXP-0001-short-slug>

### Date
YYYY-MM-DD

### Research Question

### Configuration
Full command line(s), or the driver script path plus any overrides.

### Changed Variable

### Controlled Variables

### Dataset

### Prediction Horizon

### Seed

### Important Hyperparameters

### Result Files
Paths under `results/<EXP-ID>/`, plus the `metrics/` and `logs/` paths.

### Results
Exact numbers, with the baseline number alongside each.

### Sanity Checks
What was run before the expensive job, and the outcome.

### Implementation Notes
What changed in the code; files touched.

### Status
completed | failed | aborted
```

---

<!-- Append experiment entries below this line. -->

## EXP-SEQFULL01 — Full-Memory Set-Conditioned Sequential Retrieval: verdict STOP

### Date
2026-09-05

### Research Question
Does directly imitating the Full-Memory Weighted Set Oracle's discrete greedy
construction via a set-conditioned sequential selector (score against the
WHOLE valid memory bank at every one of K=10 steps, mask only selected/invalid
candidates, never a Top-M shortlist anywhere) let a student learn hard Top-K
retrieval that captures the Individual->Set-Oracle headroom EXP-1/EXP-2
measured, and does that improve Stage-2 Final MSE?

### Configuration
See `results/EXP-SEQFULL01/command.txt`. ETTh1 H96 only, self-only
(`relation_top_n=1`), `top_k=10`, seed 0. Encoder/optimizer/lr/epochs=10/
patience=5/candidate_mask=raft/split/seed taken verbatim from B0's own saved
checkpoint args (`Exp_Stage1_Relation` built from `ckpt['args']`, not
hand-listed flags) -- only the training objective differs from every other
Stage-1 arm in this campaign. No residual teacher, no query/candidate
conditioning, no asymmetric metric (FRR01's mechanisms deliberately excluded
from this pilot). B0 is reused as-is (EXP-3's S0_wce checkpoints), not rerun.

### Changed Variable
Stage-1 training objective and inference procedure only: teacher-forced,
step-wise full-memory cross-entropy against an offline Full-Memory Weighted
Set Oracle sequence (reusing `utils.oracle_intervention.select_greedy_weighted_set`
unmodified), instead of any single-shot WCE/KL/soft-set loss. A new small
SetConditioner module (`concat(q,m) -> Linear -> GELU -> Linear -> residual
-> norm`) conditions each step's query representation on the mean embedding
of the set selected so far (a learned empty-set token at t=1).

### Controlled Variables
Stage-2 architecture, base forecaster, gate (`residual`/`scalar`), fusion,
`top_k=10`, `tau_topk=0.1`, split, seed -- identical to B0. The sequential
selector's output only ever changes WHICH 10 candidates enter Stage-2
(via `RelationStage2.set_forced_selection`, the mechanism EXP-1/EXP-2 already
use and this project has already reviewed); Stage-2's aggregation weighting
for those 10 candidates is B0's own frozen retrieval score, never anything
the sequential model produces.

### Dataset
ETTh1

### Prediction Horizon
96

### Seed
0

### Important Hyperparameters
`d_model=128`, `batch_size=32`, `learning_rate=1e-3` (B0's own), K=10,
`tau_topk=0.1` for the teacher's softmax weighting (fixed, from B0's own
frozen checkpoint, never the student's own changing score).

### Result Files
`results/EXP-SEQFULL01/` (`metrics.csv`, `command.txt`, `notes.md`, `env.txt`,
`working_tree.diff`, `logs/train_full.log`, `train_summary.json`,
`stage2_eval.json`, teacher/checkpoint sha256 fingerprints).

### Results

**Small-N sanity gate: PASS.** 16 queries, 256 candidates, single channel,
400 steps: val overlap@10 with the tiny-universe teacher = 0.875 (chance
~0.4% for 10-of-~240); candidate-side gradient norm 0.081 (nonzero
throughout); 0 duplicate/invalid selections (masking is structural, not
merely observed).

**Full ETTh1 H96 (8449 candidates, 7 self-only channels): the mechanism
trained cleanly but did not generalize.** Across all 10 epochs,
candidate-side gradient norm stayed nonzero (0.044-0.091) and train loss
decreased monotonically (8.571 -> 8.463), but validation Set overlap@10
never rose meaningfully above chance:

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val overlap@10 | 0.0090 | 0.0097 | 0.0103 | 0.0114 | 0.0123 | 0.0111 | 0.0120 | **0.0130** | 0.0112 | 0.0120 |

(chance for random unordered 10-of-8449 overlap ≈ 0.0118). Checkpoint
selected at epoch 8 (best validation overlap). Test-split diagnostics with
that checkpoint:

| Quantity | Value |
|---|---:|
| Individual Oracle MSE (diagnostic) | 0.1910 |
| B0's own A_weighted (natural Top-10) | 0.4073 |
| Full-Memory Weighted Set Oracle A_weighted | 0.1030 |
| Sequential selector's own A_weighted | 0.5304 |
| gap_recovery = (A_B0 - A_seq)/(A_B0 - A_set_oracle) | **-0.404** |
| Sequential Set Recall@10 (test, vs. oracle) | 0.0135 (chance ≈ 0.0118) |
| Duplicate rate / Invalid rate | 0.0 / 0.0 |
| Effective rank of trained encoder | 5.43 (of 128; B0's own ≈ 20.8) |

**Stage-2 (fingerprint-verified protocol match: this run's own B0-unforced
pass reproduces 0.373122 against the recorded 0.37312):**

| Arm | Stage-2 Final MSE | Delta vs B0 |
|---|---:|---:|
| B0 (unforced, this run's own reproduction) | 0.37312 | — |
| Sequential selector (forced Top-10) | 0.40277 | **+0.02965** |

### Sanity Checks
`pytest tests/`: 458 passed (14 new in `tests/test_exp_seqfull01.py`, covering
masking invariants, duplicate impossibility, teacher-forcing prefix
correctness, CE-target correctness, candidate-gradient non-zero and
non-detachment, single shared differentiable embedding tensor across K steps,
and forced-selection reaching Stage-2 without touching any other attribute),
exactly the 2 pre-existing failures, no regression. A dataloader-unpacking
bug (`loader, _ = exp._get_data(...)` instead of `_, loader = ...`) crashed
the first full-run attempt on batch 1; caught immediately, fixed, rerun from
scratch (see `results/EXP-SEQFULL01/notes.md`).

### Implementation Notes
New files only, no existing production code modified: `models/SequentialSetRetriever.py`
(SetConditioner, EmptySetToken, step_logits), `scripts/precompute_seqfull01_teacher.py`
(offline teacher, reuses `select_greedy_weighted_set`/`build_common_support`
unmodified with a full-memory pool), `scripts/train_seqfull01.py`
(teacher-forced sequential training + small-N gate, reuses `Exp_Stage1_Relation`
for all data/memory plumbing and `Model.encoder`/`_relation_tensor` for
embeddings), `scripts/eval_seqfull01_stage2.py` (Stage-2 evaluation via
`set_forced_selection`, reused verbatim). `tests/test_exp_seqfull01.py` (14
tests).

### Status
completed; verdict STOP. Stage-2 Final MSE is worse than B0 by +0.02965 --
roughly 3x the project's own ~0.01 seed-noise reference and in the wrong
direction, so no 3-seed confirmation is warranted per the pre-registered
rule. H1/H2 evidence: small-N memorization succeeded cleanly, but validation/
test Set Recall@10 never exceeded chance on the full 8449-candidate memory,
and `gap_recovery` is negative (the sequential selector's own aggregate is
worse than B0's simple retriever's, not merely short of the Set Oracle's) --
this is H2 evidence (past-only X_q/X_i information does not let this
mechanism generalize the oracle's discrete choice to held-out queries), not
H1 (a soft-relaxation-specific failure). Not run: any other horizon, dataset,
cross-channel relation, or hyperparameter/architecture variant, per the
pre-registered scope.

### ERRATUM (2026-09-06)
The "chance ≈ 0.0118" and "chance ≈ 0.4%" figures quoted above and in the
now-superseded `REVIEW_FOR_CHATGPT.md` write-up were **computed with the
wrong formula**, caught during EXP-SEQDIAG01's mandatory metric/chance-baseline
audit. `overlap_at_k` (`scripts/train_seqfull01.py`) computes the NORMALIZED
set overlap `|S_pred ∩ S_teacher| / K`; its correct chance expectation for two
independent random K-subsets of an N-candidate pool is `K/N`, not `K²/N`
(what was actually used for the full-memory "0.0118" figure) nor `1/N` (what
was actually used for the small-N "0.4%" figure). Monte Carlo-verified
(50,000 trials) correct values:

| Setting | N | K | Wrong figure quoted | Correct chance (K/N) | Measured |
|---|---:|---:|---:|---:|---:|
| ETTh1 full-memory | 8449 | 10 | 0.0118 (used K²/N) | **0.00118** | 0.009–0.013 |
| Small-N gate | ~240 | 10 | 0.004 (used 1/N) | **0.0417** | 0.875 |

The small-N gate's PASS verdict is unchanged (0.875 is ~21x the *correct*
chance, was reported as ~219x the *wrong* one -- still an unambiguous pass
either way). The full-memory reading changes materially: the measured
overlap (0.009–0.013) is **~8–11x the correct chance baseline**, not "at
chance" as originally written -- there is a small but real generalization
signal, not zero. This does **not** change the Stage-2 verdict (Final MSE
+0.02965 worse than B0, `gap_recovery` -0.404, both computed independently of
this labeling error and unaffected by it): a selector that is ~10x better
than random at matching individual set members still produces a *worse*
aggregate than B0's naive retriever, which is if anything a more specific
finding than "no signal at all" -- weak-to-moderate correct-direction signal
in membership does not survive into aggregate quality. The H1/H2 read is
revised accordingly: this is still H2-consistent (generalization is real but
far too weak for the discrete-imitation objective to produce a competitive
aggregate) rather than "zero generalization." See EXP-SEQDIAG01 below, which
was designed in part to separate how much of this weak-and-insufficient
signal is attributable to the representation collapse also recorded in this
entry (effective rank 20.8 → 5.43).

---

## EXP-FRR01 — Full-Memory Forecast-Conditioned Residual Retrieval: model-discovery pilot verdict: STOP

### Date
2026-09-05

### Research Question
Redefine retrieval from "similar future" to "similar historical
forecast-error/residual pattern" via 5 arms building on a residual teacher:
R0 (residual teacher only), R1 (R0 + query base-forecast conditioning), R2
(R0 + candidate historical-residual representation), R12 (R1+R2), R3 (R12 +
asymmetric dual encoder). Full-memory retrieval only, no Top-M shortlist
anywhere. Does any arm improve Stage-2 Final MSE over B0 by a real (>= 0.01,
the pre-registered 3-seed-confirmation threshold) margin?

### Configuration
See `results/EXP-FRR01/command.txt` for exact flags per arm and
`results/EXP-FRR01/notes.md` for the full account, including a real
architecture gap found and fixed mid-run: `RelationStage2.load_stage1_checkpoint`
originally had no path to load the new `query_cond_proj`/`candidate_cond_proj`
conditioning modules, which would have made R1/R2/R12/R3's actual mechanism
silently absent at Stage-2 retrieval time. Fixed (see Implementation Notes)
after confirming the fix with the user before proceeding.

### Changed Variable
Per-arm: which distance the WCE coverage target is graded by (R0), whether
the query/candidate embedding is additively conditioned on a base-forecast /
historical-residual projection (R1/R2/R12), and the retrieval comparison
(cosine vs asymmetric, R3). B0's architecture, loss, and protocol are
otherwise unchanged.

### Controlled Variables
Stage-2 architecture, base forecaster, fusion (`residual`/`scalar` gate),
`relation_top_n=1`, `top_k=10`, `tau_topk=0.1`, `candidate_mask=raft`,
`freeze_stage1_encoder=1`, `stage2_e2e=0`, split, seed. Identical to B0
(EXP-3's `S0_wce` cell) in every respect except the flags each arm adds.

### Dataset
ETTh1

### Prediction Horizon
96 (H96 only, per the pre-registered pilot scope)

### Seed
0 (1-seed pilot; no arm crossed the 3-seed-confirmation threshold, so no
further seeds were run)

### Important Hyperparameters
Same as B0's Stage-1/Stage-2 recipe (d_model=128, batch_size=32, lr=1e-3,
train_epochs=10/patience=5, `stage1_full_memory_gradient_mode full_online`,
`stage1_checkpoint_metric hard_aggregate_mse10`). Residual-teacher cache
rebuilt from the current campaign's own B0 Stage-2 checkpoint (not the older,
differently-sourced `cache/residual_teacher/ETTh1_pred96.pt`).

### Result Files
`results/EXP-FRR01/` (`metrics.csv`, `command.txt`, `notes.md`, `env.txt`,
`working_tree.diff`, `residual_cache_sha256.txt`, `logs/` — copies of
`logs/exp_frr01/*_stage1.log` and `*_stage2.log`).

### Results
Stage-2 Final MSE (ETTh1 H96, lower is better), B0 = 0.37312 (reused EXP-3
checkpoint, not rerun):

| Arm | Stage-1 HardAgg@10 | Recall@10 (orig) | Recall@10 (residual-target) | Stage-2 Final MSE | Delta vs B0 |
|---|---:|---:|---:|---:|---:|
| B0 | 0.4073 | 0.0570 | — | 0.37312 | — |
| R0 | 0.4154 | 0.0507 | 0.0282 | 0.37397 | +0.00085 |
| R1 | 0.4703 | 0.0505 | 0.0477 | 0.37099 | -0.00213 |
| R2 | 0.4667 | 0.0861 | 0.0606 | 0.38023 | +0.00711 |
| R12 | 0.4655 | 0.0866 | 0.0669 | 0.38112 | +0.00800 |
| R3 | 0.4537 | 0.0805 | 0.0596 | 0.37953 | +0.00641 |

No arm improved or worsened Stage-2 Final MSE by >= 0.01 (the pre-registered
threshold for a 3-seed confirmation run), so none was confirmed and none
qualifies as GO. R1's small improvement (-0.00213) is below the project's own
~0.01 seed-noise reference and is not distinguishable from noise at 1 seed.
R2/R12/R3 raised Recall@10 substantially (+51%/+52%/+41% relative) while
Stage-1 `hard_aggregate_mse10` and Stage-2 Final MSE both got worse — the same
"Recall does not predict Stage-2" pattern EXP-C01/EXP-3 established, now
reproduced through a structurally different mechanism (embedding conditioning
rather than a soft aggregate loss).

### Sanity Checks
`pytest tests/` run after every code change (R0/R1/R2 mechanism, Stage-2
wiring): 444 passed both times, exactly the 2 pre-existing failures
(`test_topk_coverage_reuses_target_indices_across_relations`,
`test_identity_retrieval_uses_raw_target_source_relation_without_encoder`), no
regression. 1-epoch smoke test of R3 (exercises every new code path: query
conditioning, candidate conditioning at all embedding sites, asymmetric
metric) run to completion, both at Stage-1 and Stage-2, before the real
10-epoch runs. `[legality]` assertion (memory_residual row count == train
split query count) passed for every Stage-1 arm. `[Retrieval Selection]
configured_metric=asymmetric actual_selection_score_fn=asymmetric` wiring
guard passed for R3 at Stage-2.

### Implementation Notes
`models/RelationStage1.py`: WCE coverage-target distance now switches to
`_residual_mse(query_residual, memory_residual, c)` when
`stage1_residual_teacher` is active (previously only the KL-teacher branch
read the residual teacher; the WCE branch, which is what B0 and every arm
here actually use, ignored it entirely). Added `_condition_query_embedding`/
`_condition_candidate_embedding` and the `stage1_query_base_conditioning`/
`stage1_candidate_residual_conditioning` flags, wired at every embedding site
(query: single site; candidate: key_bank, differentiable_keys, and
`full_online` re-encoding — three sites, mirroring the historical
forced-selection bug pattern of wiring only one of several call sites).

`models/RelationStage2.py`: mirrored the same two conditioning modules and
flags (`stage2_query_base_conditioning`/`stage2_candidate_residual_conditioning`),
extended `load_stage1_checkpoint`'s existing per-module loop to also load
them, and wired them into `build_retrieval_cache` (the static per-split cache
path, which is the one actually used here since `freeze_stage1_encoder=1`
plus ETTh1's self-only relation graph makes `_use_retrieval_cache()` true) and
`build_memory_key_bank` (the candidate bank builder). Added `query_y=`/
`query_residual=` threading and a `target_y=batch_y` fix to the eval branch of
`exp_stage2_relation.py::_run_loader`, which previously omitted `target_y`
entirely.

`exp/exp_stage1_relation.py`: added the `[legality]` assertion and log line
in `_residual_cache`. `exp/exp_stage2_relation.py`: refactored the residual
cache loader out of `_e2e_extras` into `_residual_teacher_cache`/
`_residual_batch`, reachable independent of `stage2_e2e` (it previously
returned `{}` immediately when `stage2_e2e=0`, which every arm here uses).

New files: `tests/test_exp_frr01_arms.py` (8 tests, additive-conditioning and
legality-assert coverage), `scripts/run_exp_frr01_stage1.sh`,
`scripts/run_exp_frr01_stage2.sh`, `cache/residual_teacher_frr01/ETTh1_pred96.pt`
(hash in `results/EXP-FRR01/residual_cache_sha256.txt`).

### Status
completed (model-discovery pilot); verdict STOP — no 3-seed confirmation
triggered, no arm meets GO. Not run: other horizons/datasets, R1/R2/R3
follow-up designs, any hyperparameter sweep, per the pre-registered scope.

---

## EXP-3-CLOSURE — soft_set_mse representative Stage-2 verdict: STOP

### Date
2026-09-04

### Research Question
Does the aggregate-aligned soft_set_mse Stage-1 loss, which showed mixed
Stage-1-internal signal (ETTh1: hard_aggregate_mse10 worse; Weather: better),
actually improve Stage-2 downstream forecasting?

### Configuration
Representative closure, not the full 5-arm sweep: per cell, only S0 (WCE
baseline) vs the single non-baseline arm with the best **validation**
hard_aggregate_mse10 from the already-completed Stage-1 sweep (never TEST,
never each arm's own training objective). Stage-2 architecture, base
forecaster, gate (`residual`/`scalar`), `relation_top_n=1`, `top_k=10`,
`tau_topk=0.1`, split, and protocol identical to every other Stage-2 run in
this project; only the Stage-1 checkpoint (and therefore the selection arm)
differs.

Full 20-run Stage-1×Stage-2 sweep was intentionally paused after Stage-1
completed (20/20) and Stage-2 reached 3/20, once representative closure
became the priority. 2 of 8 representative cells (ETTh1 H96 S0/S2) were
already complete and reused as-is; the other 6 were run fresh.

### Changed Variable
Stage-1 selection arm only (S0 vs best-by-val-HardAgg non-baseline).

### Dataset / Prediction Horizon / Seed
ETTh1 H96/H720, Weather H96/H720; `--seed 0`.

### Result Files
`results/EXP-3-soft-set-mse-closure/logs/` (copy of `logs/soft_set_mse/`),
git commit recorded alongside.

### Results

| Dataset | H | Arm | S2 Final MSE | Delta vs S0 | Rel% | Stage-1 HardAgg@10 | Recall@10 | N_eff | Eff. rank |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 | 96 | S0 (WCE) | 0.37312 | — | — | 0.4073 | 0.0570 | 94.6 | 20.8 |
| ETTh1 | 96 | S2 (lam10) | 0.37911 | **+0.00599** | **+1.61%** | 0.4147 | 0.0516 | 201.1 | 12.1 |
| ETTh1 | 720 | S0 (WCE) | 0.46827 | — | — | 0.5732 | 0.0198 | 68.5 | 17.6 |
| ETTh1 | 720 | S3 (lam30) | 0.50947 | **+0.04120** | **+8.80%** | 0.5864 | 0.0210 | 207.4 | 11.4 |
| Weather | 96 | S0 (WCE) | 0.17553 | — | — | 0.2348 | 0.0579 | 1498.7 | 12.2 |
| Weather | 96 | S1 (set only) | 0.17681 | **+0.00128** | **+0.73%** | 0.2058 | 0.0548 | 8870.6 | 5.8 |
| Weather | 720 | S0 (WCE) | 0.31869 | — | — | 0.5672 | 0.0116 | 81.5 | 20.0 |
| Weather | 720 | S3 (lam30) | 0.38574 | **+0.06705** | **+21.04%** | 0.4650 | 0.0105 | 700.0 | 4.4 |

MSE lower is better; every delta above is positive, i.e. **every representative
arm is worse than WCE on Stage-2 Final MSE, at all 4 cells, with no exception.**

Weather's Stage-1 `hard_aggregate_mse10` appeared to *improve* at both
horizons (−12.3% H96, −18.0% H720) while `N_eff` exploded (+492%, +759%) and
`effective_rank` collapsed (−53%, −78%) — the improvement does not survive
contact with Stage-2 and is consistent with the diffusion/collapse confound
flagged as a risk before Stage-2 was run.

Known seed noise in this repository is ≈0.01 MSE. ETTh1 H720 (+0.041) and
Weather H720 (+0.067) are 4-7x that; Weather H96 (+0.0013) is within it but
still the wrong sign for a claimed improvement; ETTh1 H96 (+0.006) is
borderline but likewise the wrong sign.

### Sanity Checks
Driver safety guard (`expected/executed/completed/skipped/failed` counters,
non-zero exit on missing-checkpoint skip) verified functioning both on a
simulated failure and on this real run
(`[summary] expected=2 executed=2 completed=2 ... failed=0` per cell called;
`overall_status=OK`). `pytest tests/`: 436 passed (+3 new, EXP-SC01 Gate B),
2 pre-existing failures unchanged, no regression.

### Implementation Notes
Fixed a checkpoint-path bug in `run_soft_set_mse_stage2.sh` (`--checkpoints`
implicitly nests under `stage1/`, the lookup glob did not) that had silently
skipped all 20 Stage-2 runs in the prior attempt while the script still
reported success. Added the counters/guard described above. No Stage-2
model/gate/fusion code touched.

### Status
completed — **VERDICT: STOP** (see decision below)

### Verdict: STOP

Per the pre-registered stop rule, all of the following hold:
- ETTh1: representative arm fails to improve Stage-2 baseline at both horizons.
- Weather: the Stage-1-level improvement is strongly coupled to N_eff
  explosion and representation collapse.
- All 4 cells move in the same (worse) direction — not merely inconsistent,
  uniformly negative.
- Two of four deltas exceed seed noise; the other two are the wrong sign to
  claim improvement regardless of magnitude.

**Conclusion (stated within what the data supports):** Set Oracle analysis
showed real headroom in candidate *combination*, but training a full-memory
soft aggregation loss to capture it did not transfer to Top-K retrieval or
downstream forecasting, because of a soft/hard selection mismatch (ETTh1) and
probability diffusion into representation collapse (Weather). No further
lambda/tau/support-penalty sweep is planned on this direction.



## EXP-1 — Test-matched Oracle Intervention (ETTh1 H96 / H720)

### Date
2026-09-02

### Research Question
In Stage-2 forecasting, does what matters come from ten individually good
candidates, or from a set that is good *together*? Equivalently: is the
Individual Oracle — the ground truth behind Recall@10 — actually Stage-2's
optimal retrieval target?

### Configuration
Causal intervention, **no training**. Stage-1 encoder/scorer, Stage-2, base
forecaster, gate and fusion all frozen; only Top-K membership is replaced.

ETTh1, TEST split, `features=M`, `seq_len = pred_len`, `top_k=10`,
`tau_topk=0.1`, `fusion_mode=residual`, `gate_mode=scalar`,
`relation_value_space=delta_last`, `d_model=128/n_heads=4/e_layers=2/d_ff=256`.
Support: cosine-induced fixed common candidate support, P100.

**`relation_top_n=1`, `source_mode=auto`, sources `[[0],[1],[2],[3],[4],[5],[6]]`
— every target's only source is itself. This is a SELF-RETRIEVAL experiment and
carries no evidence about cross-channel contribution** (see CONFLICT-001).

Stage-2 checkpoints (pre-registered as the KL+Cosine baseline, not chosen on
test score):
`checkpoints/stage2/ETTh1/seq{96,720}_pred{96,720}/stage2_carts_s2ls_fixsel_cosine_kl_stage2_*/checkpoint.pth`
(H96 sha256 `d085398b0dd4290d…`, epoch 3; H720 sha256 `041b8e419c4dbc0b…`).

### Changed Variable
Candidate selection rule only. Five arms: R0 current retriever, R1 Individual
Oracle, R2-U uniform set oracle, R2-W weighted set oracle (softmax re-normalised
over the whole selected set at each greedy step), R3 Good+Diverse control.
R2-relation was not run: under the self-only graph it is identical to R2-U, and
is kept as a unit-test invariant.

### Controlled Variables
Stage-1 checkpoint/encoder/scorer, Stage-2 checkpoint, base forecaster, gate,
fusion mode, `tau_topk`, test queries and their order, P100, valid mask, seed,
normalization, K. Verified at runtime: `base_mse` identical across all arms to
9 decimals, and the Stage-2 `state_dict` SHA256 re-checked after every arm.

### Dataset / Prediction Horizon / Seed
ETTh1 TEST; H96 (2785 queries, 19495 query×channel units) and H720 (15127 units);
`--seed 0`.

### Important Hyperparameters
`top_k=10`, `tau_topk=0.1` (the value the checkpoint was trained with — the
Phase-0 calibrated taus were deliberately NOT used, since this experiment
changes selection only), P100, `good_n=30` for R3.

### Result Files
`results/EXP-1-oracle-intervention/` — per-horizon CSVs, pairwise CSVs, logs,
`fingerprints.txt`, `command.txt`, `env.txt`, `working_tree.diff`.
Live copies under `logs/oracle_intervention/`.

### Results

Metric space: TEST split, normalized absolute, target channel only — the same
space as the Stage-2 final MSE (asserted at runtime).

**H96** (base MSE 0.647403 for every arm)

| Selection | I | A_uniform | A_weighted | Final MSE | Retrieval gain | Gain % |
|---|---|---|---|---|---|---|
| R0 Current | 0.67667 | 0.40690 | 0.40679 | 0.37469 | +0.27271 | +42.12 |
| R1 Individual Oracle | 0.31556 | 0.18861 | 0.19022 | 0.25448 | +0.39292 | +60.69 |
| R2-U Uniform Set Oracle | 0.38669 | **0.15718** | 0.15967 | 0.23912 | +0.40828 | +63.06 |
| R2-W Weighted Set Oracle | 0.40221 | 0.16087 | **0.15713** | **0.23754** | +0.40986 | **+63.31** |
| R3 Good+Diverse | 0.43534 | 0.20538 | 0.20751 | 0.27044 | +0.37696 | +58.23 |

**H720** (base MSE 0.671841 for every arm)

| Selection | I | A_uniform | A_weighted | Final MSE | Retrieval gain | Gain % |
|---|---|---|---|---|---|---|
| R0 Current | 0.98132 | 0.63258 | 0.63408 | 0.52857 | +0.14327 | +21.32 |
| R1 Individual Oracle | 0.60071 | 0.36035 | 0.36353 | 0.39313 | +0.27871 | +41.48 |
| R2-U Uniform Set Oracle | 0.66149 | **0.32514** | 0.33036 | 0.37742 | +0.29443 | +43.82 |
| R2-W Weighted Set Oracle | 0.66921 | 0.32806 | **0.32699** | **0.37582** | +0.29602 | **+44.06** |
| R3 Good+Diverse | 0.70525 | 0.36827 | 0.37579 | 0.39906 | +0.27279 | +40.60 |

Supplementary (H96 / H720): V_uniform R0 0.26977 / 0.34873, R1 0.12695 / 0.24036,
R2-U 0.22951 / 0.33635, R2-W 0.24134 / 0.34114, R3 0.22996 / 0.33698.
lambda_mean stays in 0.5137–0.5159 (H96) and 0.5065–0.5362 (H720); lambda_std
0.027–0.036 (H96), 0.103–0.116 (H720).

**Pairwise, query-level paired** (variant − baseline; negative = variant better)

| Horizon | Comparison | ΔA_uniform | ΔA_weighted | ΔFinal MSE | frac better (A_u / A_w / MSE) | overlap |
|---|---|---|---|---|---|---|
| H96 | R1 → R2-U | −0.03143 | −0.03055 | −0.01536 | 0.9963 / 0.9903 / 0.9508 | 0.5178 |
| H96 | R1 → R2-W | −0.02774 | −0.03309 | −0.01694 | 0.9717 / 0.9982 / 0.9630 | 0.4951 |
| H96 | R0 → R1 | −0.21829 | −0.21657 | −0.12021 | 0.9872 / 0.9870 / 0.9989 | 0.1365 |
| H96 | R0 → R2-W | −0.24602 | −0.24966 | −0.13715 | 0.9995 / 1.0000 / 0.9993 | 0.0782 |
| H96 | R2-U → R2-W | +0.00369 | −0.00254 | −0.00158 | 0.1801 / 0.5331 / 0.6434 | 0.8464 |
| H720 | R1 → R2-U | −0.03522 | −0.03318 | −0.01572 | 0.9974 / 0.9856 / 0.9593 | 0.5663 |
| H720 | R1 → R2-W | −0.03229 | −0.03655 | −0.01731 | 0.9884 / 0.9978 / 0.9755 | 0.5515 |
| H720 | R0 → R1 | −0.27223 | −0.27054 | −0.13544 | 0.9940 / 0.9935 / 0.9917 | 0.0818 |
| H720 | R0 → R2-W | −0.30452 | −0.30709 | −0.15275 | 1.0000 / 1.0000 / 0.9981 | 0.0384 |
| H720 | R2-U → R2-W | +0.00293 | −0.00337 | −0.00159 | 0.1547 / 0.6096 / 0.6187 | 0.8380 |

### Sanity Checks
All twelve pre-registered checks pass.

1. R2-W softmax re-normalised over the whole selected set each greedy step
   (unit test: w[0] 1.000 → 0.982 when a second candidate is added).
2. Production path unchanged apart from `forced_idx` — scores, weighting, gate,
   fusion identical (unit-tested against the real retrieval call).
3. Stage-2 `state_dict` SHA256 re-verified after every arm; unchanged.
4. P100 identical across arms (deterministic construction, unit-tested).
5. Valid mask identical across arms; `queries_dropped_below_k = 0` at both horizons.
6. R2-target == R2-relation under the self-only graph (unit test).
7. `I = A_u + V_u`: max residual 6.41e-07 (H96), 6.56e-07 (H720).
8. `I_w = A_w + V_w`: max residual 6.11e-07 (H96), 5.96e-07 (H720).
9. `base_mse` identical across all arms: 0.647403 (H96), 0.671841 (H720).
10. `tau_topk = 0.1` for every arm.
11. `fusion_mode = residual`, `gate_mode = scalar`.
12. `relation_top_n = 1`, sources self-only — logged and flagged.

Repository suite: `425 passed, 2 failed`; the two failures are the pre-existing
ones at HEAD `c306def`. 29 of the passing tests are new
(`tests/test_oracle_intervention.py`).

### Implementation Notes
New `utils/oracle_intervention.py` (support construction, per-arm selection,
uniform/weighted decomposition, overlap, degeneracy detection). The selectors
themselves are imported from `models/RelationStage1.py`, not reimplemented.
`retrieve_relation_future` gained a `forced_idx` argument that replaces the
Top-K while leaving scores, weighting and every downstream read identical.
`RelationStage2.set_forced_selection`, driver `Exp_Stage2_Relation.oracle_intervention`,
four CLI flags with early validation, `scripts/run_oracle_intervention.sh`.

**Two implementation bugs were caught before any result was accepted:**

1. *Checkpoint silently not loaded.* The Stage-2 checkpoint stores weights under
   `model_state_dict`, not `state_dict`; the first loader matched neither and
   `strict=False` let all 36 tensors go missing. The first run therefore executed
   on a randomly initialised Stage-2. The loader now raises on any missing key.
2. *Forced selection not reaching the forward pass.* `retrieve_relation_future`
   is called from two places; only `build_retrieval_cache` had been wired, so the
   first successful-looking run returned a byte-identical Final MSE (0.374692)
   for all five arms while the retrieval-quality metrics moved. Both call sites
   are now wired, and the driver raises if every arm returns an identical
   `ret_mse` — the signature this failure produces.

### Status
completed


## CONFLICT-001 — `relation_top_n` is not constant across campaigns

### Date
2026-09-02 (raised before EXP-1 was run; recorded rather than silently merged)

### What conflicts

`research/REVIEW_FOR_CHATGPT.md` and `research/EXPERIMENT_LOG.md` state
`relation_top_n=3` as a single generic configuration covering everything. That is
wrong: the two campaigns used different relation structures, and the REVIEW's own
*Key Configuration* paragraph contradicts itself — it says `relation_top_n=3`
while quoting the memory bank as `(7, 1, 8449, 128)`, whose second axis **is** the
source-slot count.

### Evidence — [repo] key-bank shapes read from the logs

| Campaign | Key bank | `relation_top_n` | Sources |
|---|---|---|---|
| Pre-campaign (`RESULTS_SUMMARY.md`, through 2026-08-04), e.g. `logs/ETTh1/run_chronos_t5_base_concat_seqeqpred_all.log` | `(7, 3, 8449, 1536)` | **3** | self + 2 cross-channel |
| 2026-09 corrected learned-score Stage-2 (`logs/stage2_learned_score_corrected_selection/`) | `(7, 1, 7201, 128)` | **1** | **self only** |
| 2026-09 pre-fix learned-score Stage-2 (`logs/stage2_learned_score/`) | `(7, 1, 7969, 128)` | **1** | self only |
| 2026-09 `e2_loss` | `(7, 1, 7201, 128)` | **1** | self only |
| Set-oracle diagnostic (`scripts/run_set_oracle.sh`) | — | **1** (explicit flag) | self only |

The relation graph confirms it: `metrics/relation_graphs/ETTh1/pearson_self_top1.json`
has `sources = [[0],[1],[2],[3],[4],[5],[6]]`, every entry `is_self=1`.
`pearson_self_top3.csv` does contain real cross sources (HUFL←MUFL ρ=0.984), but
no 2026-09 run used it.

### Consequence

**The entire 2026-09 diagnostic campaign is a self-retrieval experiment.** It does
not exercise cross-channel retrieval at all — the mechanism the manuscript names
as the project's first contribution. Specifically:

- Findings B1–B17 were produced under `relation_top_n=1`, self-only.
- The pre-campaign findings (A1–A6, `RESULTS_SUMMARY.md`) were produced under
  `relation_top_n=3` with cross-channel sources active.
- **These two groups are therefore not directly comparable**, and no 2026-09
  result may be cited as evidence about cross-channel contribution.

A further consequence, specific to EXP-1: under a self-only graph
`[target ‖ source] = [target ‖ target]`, so the relation space is the target
space duplicated and a relation-space set oracle is identical to a target-space
one. Verified in `tests/test_oracle_intervention.py`.

### Resolution

Not merged. The three configurations are to be documented separately —
*generic/older diagnostics*, *corrected learned-score Stage-2*, and *oracle
intervention* — when the documents are updated after EXP-1, with provenance kept
rather than numbers overwritten. Cross-channel re-verification at
`relation_top_n=3` is a separate future experiment and must not be mixed into
EXP-1.

### Status
open — documents not yet corrected (deferred to the post-EXP-1 update)



## EXP-C01 — 2026-09 retrieval-transfer diagnostic campaign

### Date
2026-08-30 → 2026-09-02 (artifact timestamps span 08-29 22:20 → 09-02 04:05)

### Research Question
Why does improving Stage-1 retrieval (Recall@10) not improve Stage-2 forecasting?

### Configuration
Multiple probes, each in its own log directory. Common setting where stated:
ETTh1, `features=M`, `seq_len = pred_len`, `top_k=10`, `relation_top_n=3`,
`d_model=128 / n_heads=4 / e_layers=2 / d_ff=256 / label_len=0`.
Frozen-encoder probes use a healthy WCE-trained Stage-1 encoder and a fixed
per-query candidate pool P100.

### Changed Variable
Varies per sub-probe — see the per-phase table below. No single controlled
variable; this is a diagnostic campaign, not a single comparison.

### Controlled Variables
Dataset, split protocol, `seq_len = pred_len`, `top_k`, `relation_top_n`, model
size. Oracle-selection probes (Phase A, set oracle) hold the candidate pool fixed
so that only the *selection rule* varies.

### Dataset
ETTh1 (primary); Weather (Stage-2 wiring-fixed verification only).

### Prediction Horizon
96 / 192 / 336 / 720 (Phase C imitation: 96 and 720 only).

### Seed
Not recorded per-probe in the logs. **UNKNOWN** — `--seed` default is 0; this was
not confirmed for these runs.

### Important Hyperparameters
- Phase 0 calibrated temperatures: τ = 0.015 (H96), 0.015 (H192), 0.01 (H336),
  0.02 (H720).
- Imitation probes: fixed pool P100, 3 epochs, LR decayed to 2.5e-4 by epoch 3.
- Candidate memory bank at H96: shape `(7, 1, 8449, 128)`, valid candidate pool
  8449.

### Result Files

| Probe | Directory | Key files |
|---|---|---|
| Score geometry | `logs/score_geometry/` | `ETTh1_pred{96,720}.log` |
| Boundary rank arms | `logs/rank_arms/` | `ETTh1_pred{96,720}_wce_rank.log` |
| Persistent pair mining | `logs/persistent_probe/` | `Dynamic.log`, `Persistent.log` |
| Collapse onset | `logs/collapse_diag/` | `WCE.log`, `Dynamic.log`, `Persistent.log` |
| Frozen encoder + rank isolation | `logs/frozen_scorer/` | `A_cosine.log`, `B_asym_rank.log` |
| Global anchor KL | `logs/global_anchor/` | `beta0p1.log`, `beta1p0.log` |
| Swap / conflict | `logs/swap_conflict/` | `swap_rows.csv`, `fingerprints.txt` |
| Oracle headroom + redundancy | `logs/headroom_redundancy/` | 40 × `ETTh1_pred<H>_<metric>_<loss>.log` |
| Retrieval-off counterfactual | `logs/utilization_diag/` | 40 × same scheme |
| Set-utility oracle (raw) | `logs/set_oracle/` | `set_only_candidates_pred{96,192,336,720}.csv` |
| **Phase 0** — τ calibration | `logs/tau_calibration/` | `pred{96,192,336,720}.log` |
| **Phase A** — pool × K sweep | `logs/set_oracle/` | `pool_k_sweep_pred{96,192,336,720}.csv` |
| **Phase C** — imitation gate | `logs/imitation/` | `pred{96,720}_{individual,set}.log` |
| Stage-2 final MSE | `logs/stage2_learned_score_corrected_selection/` | `ETTh1/pred<H>/<arm>_{stage2,e2e}.log` |
| Weather Stage-2 | `logs/weather_stage2/`, `logs/weather_top3/` | `weather/pred<H>/` |

Phases B, D, E, F of the stated 7-phase plan: **UNKNOWN** — not named, no
directory identified.

### Results

Full numeric tables are in `research/RESEARCH_CONTEXT.md` →
*Established Experimental Findings*, section B (B1–B17), with per-block
provenance marking. Headline factual results:

1. Stage-1 R@10 is improvable ~2–4× by asymmetric / pair-MLP scorers, most at
   H336/H720 (0.0489 → 0.1311 at H336).
2. Weather H96 post-fix: R@10 0.0583 → 0.0044 (−92.5%) while Stage-2 MSE
   0.1925 → 0.1794 (−6.8%). Directions disagree.
3. Retrieval gain is nonzero and horizon-dependent (ETTh1 H96 ≈40.05%,
   H720 ≈6.18%), so "Stage-2 ignores retrieval" is rejected.
4. Spearman vs Stage-2 MSE: Recall@10 +0.032 (ETTh1) / −0.350 (Weather);
   HardAggregateMSE@10 +0.810 / +0.935.
5. Rank-only encoder training collapses the representation
   (effective rank 16.38 → 1.07 within one epoch).
6. With the encoder frozen, rank training improves local pair ordering
   (PairAcc100 0.53185 → 0.54605) while global R@10 falls 0.05775 → 0.02046.
7. Global KL anchor raises Top-10 retention 0.142 → 0.716 with β, but
   `cos(g_rank, g_global) < 0` and only ≈3.7% of anchor gradient reaches Top-10.
8. Swap analysis: added candidates were *better* individually
   (0.59782 vs removed 0.73979), yet candidate variance fell up to 74%.
9. `I = A + V` verified exactly (residual ≤ 3e-9 across all `pool_k_sweep` rows).
10. Set Oracle vs Individual Oracle aggregate gain, recomputed here from raw
    CSVs: **25.9% / 18.9% / 12.3% / 4.0%** at H96/192/336/720, with
    `A_set < A_ind AND I_set ≥ I_ind` holding for **100.0%** of query units.
11. Good+Diverse Oracle loses to Individual Oracle at every horizon
    (−6.8% / −1.8% / −6.8% / −3.8%).
12. Phase 0 τ calibration complete; per-horizon τ as listed above.
13. Phase C: every imitation arm sits at random/uniform on validation
    (TeacherSetRecall@10 0.0847–0.1363 vs random 0.10; imitation loss
    4.589–4.621 vs ln(100) ≈ 4.605) — for the **Individual** target as well as
    the Set target.
14. Greedy Set Oracle restart overlap 0.878 (H96) / 0.902 (H720) → target-noise
    explanation rejected.

### Sanity Checks
- `pytest tests/` at the time of logging: `396 passed, 2 failed`; both failures
  pre-existing at HEAD `c306def` (see *Known Repository Issues*).
- `I = A + V` identity verified numerically on every `pool_k_sweep` row.
- Phase C reference baselines pinned in advance: random Top-10 overlap in P100 =
  0.10, uniform CE = ln(100) ≈ 4.605.
- Greedy Set Oracle restart stability measured before treating it as a target.

### Implementation Notes
Implemented by the uncommitted working-tree changes inventoried in
`RESEARCH_CONTEXT.md` → *Uncommitted Work In Progress* (boundary rank loss,
rank-failure diagnostics, collapse/score geometry, frozen-encoder scorer, global
anchor, set-level loss, set-utility oracle selectors, retrieval-off
counterfactual, Stage-2 selection/redundancy reporting).

**A wiring bug was found and fixed during this campaign:** the configured scorer
was not propagated into Stage-2's actual Top-K selection path, so some runs
selected via the cosine path regardless of the trained scorer. Weather was
re-verified post-fix; the pre-fix **ETTh1 Stage-2 sweep is invalidated** and
needs a rerun. See *Invalidated Results — Do Not Cite*.

### Verification Discrepancies
Recomputation from the artifacts did not reproduce three user-supplied figures.
Same conclusions; different numbers. Flagged for the reviewer:

| Quantity | User-supplied | Recomputed / logged | Source |
|---|---|---|---|
| H336 set-oracle aggregate gain | 11.8% | **12.3%** | `set_only_candidates_pred336.csv` |
| H720 set-oracle aggregate gain | 3.6% | **4.0%** | `set_only_candidates_pred720.csv` |
| Joint-condition fraction | 99.9 / 100 / 97.8 / 95.5% | **100.0%** at all four | recomputed, strict `<` and `≥` |
| H336 N_eff at τ=0.015 | ≈32 | **93.6** (N_eff ≈ 35.9 occurs at τ=0.01) | `tau_calibration/pred336.log` |
| H336 N_eff / Mass@10 at τ=0.1 | ≈1632 / ≈0.065 | **2444.9 / 0.0202** (τ=0.07 gives 1625.1 / 0.0385) | same |
| H96 cosine I / A / V | 0.6612 / 0.3573 / 0.3038 | **0.594556 / 0.318727 / 0.275829** | `pool_k_sweep_pred96.csv`, pool=100 K=10 |

The chosen τ values (0.015 / 0.015 / 0.01 / 0.02) match the logs exactly.
The Phase C imitation numbers match the logs exactly.

### Status
completed (diagnostic campaign); **ETTh1 Stage-2 sweep requires rerun**

---

## EXP-SEQDIAG01 — Cross-Dataset Sequential Collapse Diagnostic (Frozen-B0 control)

### Date
2026-09-06

### Research Question
EXP-SEQFULL01 (full-memory, teacher-forced, sequential imitation of the
Weighted Set Oracle's greedy construction) trained cleanly on ETTh1 H96 but
did not generalize to held-out queries, and its trained encoder's effective
rank collapsed from B0's ~20.8 to 5.43. Two explanations are entangled:

- **H1**: the sequential full-memory cross-entropy objective itself collapses
  the encoder representation, and that collapse is what prevents
  generalization.
- **H2**: even with the representation held stable, past-only `X_q`/`X_i`
  information does not let this mechanism learn a discrete greedy
  set-construction rule that transfers from training queries to held-out
  ones.

A Frozen-B0 control arm (B0's own encoder weights, never updated; only the
new `SetConditioner` trains) was added on both ETTh1 and Weather H96 and
compared against the existing/new Trainable arm on each dataset, to
distinguish these hypotheses and check whether any conclusion is
dataset-dependent.

Pre-registered decision framing: **Case A** — Frozen improves over Trainable
on both datasets (H1 supported: collapse is a genuine contributing cause).
**Case B** — Frozen also fails to improve over Trainable on both datasets
(H2 supported: the failure is not primarily about representation collapse).
**Case C** — the two datasets disagree (dataset-dependent learnability, no
clean call). No exact numeric threshold for "improves" was recorded in the
repository prior to this entry; this entry reports the sign and magnitude of
every arm's `gap_recovery` explicitly so the reviewer can apply their own
threshold.

### Configuration
Stage-1 (`scripts/train_seqfull01.py`) then Stage-2 (`scripts/eval_seqfull01_stage2.py`),
matching EXP-SEQFULL01's protocol exactly except for the `--frozen_encoder`
flag and the dataset. Exact commands (also in `logs/exp_seqdiag01/RESUME_STATE.md`
and `logs/exp_seqdiag01/run_bcd_chain.sh`):

```bash
source /data/pjh_workspace/ts-env/bin/activate
export CUDA_VISIBLE_DEVICES=1

# Arm B: ETTh1 Frozen-B0
python -u scripts/train_seqfull01.py \
  --base_ckpt "$ETTH1_S1_CKPT" --teacher_cache cache/seqfull01_teacher/ETTh1_pred96.pt \
  --checkpoints checkpoints/exp_seqdiag01 \
  --model_id carts_seqdiag01_etth1_frozen --des seqdiag01_etth1_frozen \
  --top_k 10 --seed 0 --frozen_encoder
python -u scripts/eval_seqfull01_stage2.py \
  --stage2_checkpoint "$ETTH1_S2_CKPT" \
  --sequential_checkpoint checkpoints/exp_seqdiag01/seqfull01/ETTh1/seq96_pred96/carts_seqdiag01_etth1_frozen/checkpoint.pth \
  --teacher_cache cache/seqfull01_teacher/ETTh1_pred96.pt \
  --top_k 10 --out results/EXP-SEQDIAG01/armB_etth1_frozen_stage2.json

# Arm C: Weather Trainable (no --frozen_encoder)
python -u scripts/train_seqfull01.py \
  --base_ckpt "$WEATHER_S1_CKPT" --teacher_cache cache/seqfull01_teacher/custom_pred96.pt \
  --checkpoints checkpoints/exp_seqdiag01 \
  --model_id carts_seqdiag01_weather_trainable --des seqdiag01_weather_trainable \
  --top_k 10 --seed 0
python -u scripts/eval_seqfull01_stage2.py \
  --stage2_checkpoint "$WEATHER_S2_CKPT" \
  --sequential_checkpoint checkpoints/exp_seqdiag01/seqfull01/custom/seq96_pred96/carts_seqdiag01_weather_trainable/checkpoint.pth \
  --teacher_cache cache/seqfull01_teacher/custom_pred96.pt \
  --top_k 10 --out results/EXP-SEQDIAG01/armC_weather_trainable_stage2.json

# Arm D: Weather Frozen-B0
python -u scripts/train_seqfull01.py \
  --base_ckpt "$WEATHER_S1_CKPT" --teacher_cache cache/seqfull01_teacher/custom_pred96.pt \
  --checkpoints checkpoints/exp_seqdiag01 \
  --model_id carts_seqdiag01_weather_frozen --des seqdiag01_weather_frozen \
  --top_k 10 --seed 0 --frozen_encoder
python -u scripts/eval_seqfull01_stage2.py \
  --stage2_checkpoint "$WEATHER_S2_CKPT" \
  --sequential_checkpoint checkpoints/exp_seqdiag01/seqfull01/custom/seq96_pred96/carts_seqdiag01_weather_frozen/checkpoint.pth \
  --teacher_cache cache/seqfull01_teacher/custom_pred96.pt \
  --top_k 10 --out results/EXP-SEQDIAG01/armD_weather_frozen_stage2.json
```

ETTh1's Trainable-arm row is **reused, not rerun**, from EXP-SEQFULL01
(`results/EXP-SEQFULL01/metrics.csv`); its `b0_unforced_final_mse` was
cross-checked against Arm B's own `b0_unforced_final_mse` (0.373122 both),
confirming the two experiments' evaluation code paths agree on the same B0
baseline.

`--frozen_encoder`: freezes the Stage-1 `RelationEncoder` weights (loaded
from B0's own Stage-1 checkpoint rather than randomly initialised, unlike
EXP-SEQFULL01's Trainable arms) and B0's own retrieval modules; only the new
`SetConditioner` (and `EmptySetToken`) receive gradient updates. Verified by
unit test and by `encoder_grad_norm=None(frozen)` in every frozen-arm epoch
line of the training logs.

### Changed Variable
`--frozen_encoder` (on/off) x dataset (ETTh1/Weather), 2x2, with ETTh1
Trainable reused from EXP-SEQFULL01.

### Controlled Variables
Everything else matches EXP-SEQFULL01's protocol and B0's own hyperparameters
(loaded from B0's Stage-1 checkpoint `args`): `seq_len=pred_len=96`, `top_k=10`,
d_model=128, teacher-forced step-wise full-memory cross-entropy, no shortlist
at any point (every valid candidate scored at every step; only
already-selected/invalid candidates masked), Stage-2's forced-selection
mechanism (`RelationStage2.set_forced_selection`) reused verbatim, B0's own
frozen retrieval aggregation weights unchanged.

### Dataset
ETTh1 (7 channels, 8449 valid candidates/channel) and Weather (`custom`, 21
channels, 36696 valid candidates/channel).

### Prediction Horizon
96 only.

### Seed
0 (`--seed 0`, explicit on every arm).

### Important Hyperparameters
`--top_k 10`; Stage-1 training 10 epochs each arm; Weather per-epoch wall
clock ~510-525s (Frozen) / ~642s (Trainable), observed directly in
`logs/exp_seqdiag01/armD_weather_frozen_train.log` and prior Arm C log.

### Result Files
- `results/EXP-SEQDIAG01/armB_etth1_frozen_stage2.json`
- `results/EXP-SEQDIAG01/armC_weather_trainable_stage2.json`
- `results/EXP-SEQDIAG01/armD_weather_frozen_stage2.json`
- `results/EXP-SEQFULL01/metrics.csv` (ETTh1 Trainable, reused)
- `results/EXP-SEQDIAG01/notes.md` (this campaign's full working notes)
- `logs/exp_seqdiag01/` (training logs, resume-state note, chain script)
- Checkpoints: `checkpoints/exp_seqdiag01/seqfull01/{ETTh1,custom}/seq96_pred96/carts_seqdiag01_{etth1_frozen,weather_trainable,weather_frozen}/checkpoint.pth`

### Results

`gap_recovery`: fraction of the B0-to-set-oracle-A_weighted gap the arm's
forced selection recovers; 0 = matches B0, negative = worse than B0 (moves
away from the set oracle), 1 = matches the set oracle exactly. Not comparable
in raw MSE across datasets; `gap_recovery` is the normalized quantity meant
to be cross-dataset comparable.

| Arm | Dataset | b0_unforced_mse | sequential_forced_mse | gap_recovery | seq_set_recall@10 | duplicate/invalid rate |
|---|---|---:|---:|---:|---:|---:|
| Trainable (reused, EXP-SEQFULL01) | ETTh1 | 0.37312 | 0.40277 | -0.404 | 0.01354 | 0.0 / 0.0 |
| **Frozen-B0 (Arm B)** | ETTh1 | 0.373122 | 0.387924 | **-0.248** | 0.01242 | 0.0 / 0.0 |
| Trainable (Arm C) | Weather | 0.175529 | 0.268140 | -0.816 | 0.00917 | 0.0 / 0.0 |
| **Frozen-B0 (Arm D)** | Weather | 0.175529 | 0.217331 | **-0.608** | 0.02293 | 0.0 / 0.0 |

Freezing the encoder moved `gap_recovery` toward zero (less negative, i.e.
less bad) on **both** datasets: ETTh1 -0.404 → -0.248 (Δ +0.156), Weather
-0.816 → -0.608 (Δ +0.208). This is the same-direction pattern on both
datasets called out in the pre-registered Case A framing above — but on
neither dataset does the frozen arm come close to `gap_recovery = 0`
(matching B0), let alone a positive value: **every one of the four arms
remains net negative**, meaning the sequential selector's forced Top-10 is
still worse than B0's own unforced selection in all four cases.

Effective-rank diagnostic (`utils.rank_losses.embedding_geometry`, full
candidate bank, channel 0, computed post-hoc from the saved checkpoints —
not saved by the training script itself):

| Encoder | Dataset | n_candidates | effective_rank (of 128) | mean pairwise cosine |
|---|---|---:|---:|---:|
| B0 (frozen arms reuse this exactly) | ETTh1 | 8449 | ~20.8 (recorded in EXP-3's closure, not recomputed here) | — |
| Trainable (EXP-SEQFULL01) | ETTh1 | 8449 | 5.433 | 0.0122 |
| B0 | Weather | 36696 | 19.576 | 0.8123 |
| Trainable (Arm C) | Weather | 36696 | 4.297 | 0.7063 |

The Weather Trainable encoder collapses in effective rank by almost exactly
the same relative amount as ETTh1's did (19.576 → 4.297 is a 78% reduction;
20.8 → 5.433 is a 74% reduction) — a second, independent piece of evidence
for consistent, cross-dataset representation collapse under this objective.
Note the mean-pairwise-cosine reading does **not** move in the same direction
on Weather as it does on ETTh1 (B0's Weather cosine, 0.812, is already high
before training, and the trained arm's cosine, 0.706, is *lower* than B0's,
not higher) — cosine alone is not a reliable collapse indicator on Weather;
effective rank is the metric that agrees across both datasets.

**HardAggregateMSE@10** (`= MSE(mean_i y_i, y_q)`, the equal-weight aggregate
of the forced Top-10, computed post-hoc — `scripts/compute_hard_aggregate_mse_seqdiag01.py`,
not part of the production eval script, which only reports the B0-weighted
`A_weighted` aggregate):

| Arm | Dataset | individual_oracle | set_oracle | sequential | B0 (unforced) | seq / b0 ratio |
|---|---|---:|---:|---:|---:|---:|
| Trainable (reused) | ETTh1 | 0.19101 | 0.30503 | 0.59017 | 0.40741 | 1.449x |
| **Frozen-B0 (Arm B)** | ETTh1 | 0.19101 | 0.30503 | **0.51901** | 0.40741 | **1.274x** |
| Trainable (Arm C) | Weather | 0.03631 | 0.12417 | **4.30396** | 0.23421 | **18.38x** |
| **Frozen-B0 (Arm D)** | Weather | 0.03631 | 0.12417 | **0.92057** | 0.23421 | **3.93x** |

This unweighted metric shows a much larger freeze effect than `gap_recovery`/
`A_weighted` did, especially on Weather: freezing the encoder drops the
sequential arm's unweighted aggregate MSE by **4.7x** (4.304 → 0.921), versus
a much smaller relative improvement on ETTh1 (0.590 → 0.519, 1.14x). Both
datasets move in the *same direction* (freezing helps), corroborating the
`gap_recovery`/effective-rank pattern above with a third, independent
metric — but the *magnitude* of the freeze effect is markedly larger on
Weather under this unweighted metric than under the B0-weighted one, an
asymmetry not visible in `gap_recovery` alone. Every arm's `seq` value still
exceeds its own `b0` value (no arm beats B0 under this metric either).

### Structural invariant verification

**Encoder-freeze invariant** (independently verified beyond the training-time
`requires_grad=False` assertion already built into `scripts/train_seqfull01.py`):
SHA256 of every `encoder.*` parameter tensor in the saved frozen-arm
checkpoints, byte-for-byte, against the B0 Stage-1 checkpoint each was loaded
from —

| Checkpoint | Encoder-only SHA256 |
|---|---|
| B0 ETTh1 Stage-1 | `b7cc8b56...d9f87c3b1` |
| Arm B (ETTh1 Frozen) | `b7cc8b56...d9f87c3b1` (**identical**) |
| B0 Weather Stage-1 | `ecfcc2e9...af73a29db` |
| Arm D (Weather Frozen) | `ecfcc2e9...af73a29db` (**identical**) |
| Arm C (Weather Trainable) | `0402369a...acdee7902` (differs from B0, as expected — this arm is not frozen) |

Confirms, at the weight level and independent of any training-time logging,
that the frozen arms' encoders are exactly B0's own weights with zero drift.

**Full-memory invariant**: `models/SequentialSetRetriever.py::step_logits`
computes `torch.matmul(h_t, candidate_embeddings.transpose(0,1))` against
the full `candidate_embeddings` tensor (N=8449 ETTh1 / N=36696 Weather) at
every one of the K=10 steps; only `selected_mask | ~valid_mask` positions
are set to `-inf`. No Top-M/shortlist/coarse-retrieval stage exists anywhere
in `scripts/train_seqfull01.py` or `scripts/eval_seqfull01_stage2.py` — every
valid candidate is scored at every step, confirmed by direct code
inspection (not merely by the module's own docstring claim).

**Important distinction (do not conflate):** the Frozen-B0 arms are **not**
a different Stage-1/Stage-2 architecture. Every arm in this experiment —
Trainable and Frozen alike — already separates Stage-1 (produces a hard
Top-10 via `set_forced_selection`) from Stage-2 (B0's own unchanged
forecaster/gate/fusion/aggregation, forecasting loss never backpropagated
into Stage-1). "Frozen" refers only to whether the Stage-1 `RelationEncoder`
weights update during Stage-1 training — it is an encoder-representation
control within the existing separated two-stage design, not a new
end-to-end or joint-training configuration. True Stage-1+Stage-2 joint
end-to-end training (Stage-2 forecasting loss backpropagated into Stage-1)
was not implemented and is out of scope for this experiment.

### Sanity Checks
`pytest tests/`: 466 passed, 2 failed (pre-existing at HEAD `c306def`, listed
in *Known Repository Issues*; `tests/test_exp_seqdiag01.py` adds 8 new
passing tests for the frozen-encoder control and metric definitions), no
regression versus the 396-passed/2-failed baseline recorded 2026-09-02 (the
delta from 396 to 458 to 466 passed reflects tests added by EXP-SEQFULL01
and EXP-SEQDIAG01 themselves, not a changed baseline).

`b0_unforced_final_mse` fingerprint check: every arm's evaluation script
independently re-derives B0's own (unforced) selection through the identical
evaluation loop before reporting the forced-selection number. ETTh1:
0.373122 (Arm B) reproduces 0.37312 (EXP-SEQFULL01's recorded B0 baseline)
to 5 significant figures. Weather: 0.175529 reproduced identically by both
Arm C and Arm D (same B0 checkpoint, same evaluation code path, two
independent runs).

Server reboot mid-campaign (2026-09-06, ~03:2x): all background processes
were cleanly SIGTERM'd before the planned reboot (no mid-write), GPU 1
confirmed free before shutdown. Arm D's Stage-1 was 2/10 epochs in at that
point; per `logs/exp_seqdiag01/RESUME_STATE.md`, the 2-epoch checkpoint was
discarded (not resumed — `train_seqfull01.py` has no epoch-resume logic) and
Arm D was rerun from scratch after reboot. Arms B and C were unaffected
(both completed before the reboot).

### Implementation Notes
`--frozen_encoder` support added to `scripts/train_seqfull01.py` (loads B0's
own Stage-1 encoder weights instead of random init, excludes encoder
parameters from the optimizer, keeps the encoder in eval mode, asserts zero
encoder gradient each epoch). `tests/test_exp_seqdiag01.py` (8 tests) covers
the frozen-encoder gradient-isolation contract and the corrected
`overlap_at_k` metric definition (see ERRATUM above this entry).

### Status
completed (diagnostic campaign, 4/4 arms, plus the HardAggregateMSE@10
companion metric and independent structural-invariant re-verification added
after the initial 4-arm result). **Case A pattern observed on both datasets
(Frozen improves `gap_recovery` AND `HardAggregateMSE@10` over Trainable,
same direction on both metrics and both datasets, similar relative
effective-rank collapse magnitude) — partial support for H1: sequential
training's representation collapse is a genuine, consistent contributing
cause. This is not full support: every arm on both datasets remains net
`gap_recovery`-negative and net `HardAggregateMSE@10`-worse-than-B0, so H2
(the discrete greedy rule itself does not transfer to held-out queries from
past-only information, independent of representation quality) remains
necessary to explain the residual failure even once the encoder is held
fixed at B0's values.** The freeze effect's *magnitude* is markedly larger
on Weather than ETTh1 under the unweighted metric (4.7x vs. 1.14x
reduction) — a dataset asymmetry `gap_recovery` alone did not surface.
**Encoder freeze is a within-Stage-1 representation control, not a
different Stage-1/Stage-2 architecture: every arm here (Trainable and
Frozen alike) already separates Stage-1 hard-selection from Stage-2's
unchanged forecaster via `set_forced_selection`, with no forecasting-loss
backpropagation into Stage-1 in either case.** Interpretation, novelty
assessment, and next-experiment recommendation are left to the reviewer per
this project's role division — see `research/REVIEW_FOR_CHATGPT.md`.

---

## EXP-MARGUTIL01 — Full-Memory Set-Conditioned Dense Marginal Utility

### Date
2026-09-06 to 2026-09-07

### Research Question
D-0011 closed exact one-hot next-candidate-ID cross-entropy imitation. This
experiment tests the named successor: does replacing the one-hot target with
a DENSE per-candidate set-utility regression target,
`u_i^(t) = -A_weighted(S*_{t-1} + {i})` for every valid remaining candidate
at every teacher-forced step (oracle prefix `S*_{t-1}`, the cached
`select_greedy_weighted_set` sequence), let a held-out query build a
competitive Top-K set and reach Stage-2 Final MSE? Encoder frozen (B0's own
weights) throughout — this experiment does not re-open the
trainable-encoder question EXP-SEQDIAG01 already answered.

- **H1 (sparse-target):** the prior objective's sparsity, not an
  information ceiling, was the failure mode; dense supervision should raise
  utility correlation, lower regret, improve `gap_recovery`/Stage-2.
- **H2 (information ceiling):** even dense supervision on a frozen,
  uncollapsed representation does not predict held-out marginal utility
  well enough to matter downstream.

### Configuration
`scripts/train_margutil01.py` (Stage-1, dense utility regression, always
`--frozen_encoder`-equivalent — encoder loaded from `--base_ckpt` and
`requires_grad=False` unconditionally) then `scripts/eval_margutil01_stage2.py`
(teacher-forced utility diagnostics + free-running Stage-2 injection via
`RelationStage2.set_forced_selection`). New shared math:
`utils/dense_utility.py` (chunked incremental-weighted-mean utility, same
closed form `select_greedy_weighted_set` uses internally, exposed densely).
New minimal head: `models/DenseUtilityRetriever.py::UtilityHead`
(`u_hat = a*cosine(h_t,e_i)+b`, `a`/`b` the only new learnable scalars — no
new architecture). Full commands: `results/EXP-MARGUTIL01/command.txt`.

H96 teacher caches reused verbatim from EXP-SEQFULL01
(`cache/seqfull01_teacher/{ETTh1,custom}_pred96.pt`). H720 caches newly
built via `scripts/precompute_seqfull01_teacher.py` (existing script,
unmodified) against the S0_wce B0 Stage-2 checkpoints for H720 (both
datasets), which already existed in the repo.

### Changed Variable
Training target: one-hot next-candidate-ID CE (EXP-SEQFULL01/EXP-SEQDIAG01)
→ dense per-candidate set-utility SmoothL1 regression (this experiment).
Encoder: always frozen (no trainable-encoder arm in this experiment).

### Controlled Variables
`top_k=10`, `relation_top_n=1` (self-only), seed 0, oracle-prefix
teacher-forcing during training (no free-running error accumulation mixed
into the training signal), no softmax/KL loss (SmoothL1 only, per spec, to
avoid EXP-3's full-memory probability-diffusion confound), no per-candidate
utility weighting in the loss (plain mean over valid candidates), B0's own
Stage-2 weighting/gate/fusion/base-forecaster/`tau_topk` completely
unchanged — only Top-K membership differs at Stage-2.

### Dataset
ETTh1 and Weather (`custom`), self-only.

### Prediction Horizon
96 and 720. **192 and 336 explicitly out of scope** (not run, per spec).

### Seed
0.

### Important Hyperparameters
Candidate chunk size: 4096 (ETTh1/Weather H96), 1024 (ETTh1 H720), 512
(Weather H720 — never used, see below). `tau_topk` inherited from each
cell's own B0 checkpoint args (0.1 for the S0_wce line, confirmed identical
between B0's saved args and the H96 teacher cache's own recorded `tau`).

### Result Files
`results/EXP-MARGUTIL01/{ETTh1_H96,Weather_H96,ETTh1_H720}_stage2.json`,
`comparison.csv`, `notes.md`, `command.txt`, `env.txt`, `git_commit.txt`,
`checkpoint_fingerprints.txt`. Checkpoints:
`checkpoints/exp_margutil01/margutil01/{ETTh1,custom}/seq<L>_pred<L>/carts_margutil01_<cell>/checkpoint.pth`.

### Results

| Cell | B0 unforced MSE | Dense forced MSE | Delta vs B0 | gap_recovery | seq_set_recall@10 | Utility Spearman | Utility Pearson | HardAgg (seq/b0) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 H96 | 0.37312 | 0.52991 | **+0.157** | -1.832 | 0.00427 | 0.543 | 0.424 | 1.657 / 0.407 |
| Weather H96 | 0.17553 | 0.33253 | **+0.157** | -1.744 | 0.00593 | 0.658 | 0.503 | 4.465 / 0.234 |
| ETTh1 H720 | 0.46827 | 0.49269 | **+0.024** | -1.866 | 0.00147 | 0.383 | 0.413 | 1.961 / 0.572 |

`duplicate_rate`/`invalid_rate` = 0.0 on all 3 cells (structural, verified
by unit test and direct measurement). **Weather H720: no result — training
did not complete a single epoch before being stopped and removed by
explicit user decision (D-0013); this is a scope reduction, not a null or
failed result, and no number is reported or estimated for this cell.**

Every cell's utility Spearman is clearly above 0 (0.38-0.66): the dense
target IS partially learnable from a frozen, past-only representation —
H1's premise about learnability is not rejected. But `gap_recovery` is
strongly negative and `HardAggregateMSE@10` is 3-19x worse than B0's own
unforced selection on every cell — the learned utility does not translate
into a competitive free-running Top-K selection, and Stage-2 Final MSE is
worse than B0 on every cell by more than this project's pre-registered
0.01 noise threshold (ETTh1/Weather H96: +0.157; ETTh1 H720: +0.024). This
is a **larger Stage-2 regression than EXP-SEQDIAG01's Frozen-B0 one-hot
arms** (ETTh1 Frozen: +0.0148; the dense-target successor is not an
improvement over the one-hot predecessor it replaced, on any cell tested).

### Sanity Checks
`pytest tests/`: 466 (pre-EXP-MARGUTIL01 baseline) → 477 (after
`tests/test_exp_margutil01.py`, 11 new) → 485 (after the follow-up
`tests/test_exp_firstanchor_diag.py`, 8 new), same 2 pre-existing failures
throughout, no regression at any point. A 1-epoch ETTh1 H96 GPU smoke run
caught and fixed two implementation bugs before the full 3-cell run
launched: (1) `HardAggregateMSE@10` was off by exactly the horizon `H` in
`eval_margutil01_stage2.py`'s final division (caught by comparing against
EXP-SEQDIAG01's already-verified `individual_oracle_mse`/`set_oracle`
figures — the buggy output was off by a factor of 96 = H, unmistakable);
(2) step-wise `regret` was computed against the oracle's own pick, which is
tautologically ~0 by construction, instead of the model's own predicted
pick under the true oracle prefix — fixed to compare the model's argmax
pick's teacher-utility against the true best. Both fixes verified by
re-running the sanity check and reproducing EXP-SEQDIAG01's exact
`individual_oracle_mse`/`set_oracle_a_weighted` numbers before proceeding.

### Implementation Notes
New files: `models/DenseUtilityRetriever.py`, `utils/dense_utility.py`,
`scripts/train_margutil01.py`, `scripts/eval_margutil01_stage2.py`,
`tests/test_exp_margutil01.py`. Reuses `models.SequentialSetRetriever`'s
`SetConditioner`/`EmptySetToken` verbatim, `utils.oracle_intervention`'s
`select_greedy_weighted_set` (unmodified) for the cached teacher sequence,
and `RelationStage2.set_forced_selection` (unmodified) for Stage-2
injection.

### Status
**3 of 4 approved cells completed; Weather H720 cancelled by user decision
before completing a single training epoch (D-0013).** 0 of 3 completed
cells show meaningful Stage-2 improvement (pre-registered rule) — tracking
toward STOP, but the pre-registered 4-cell decision rule (≥2/4 cells improve
≥0.01, none worsen ≥0.01) cannot be mechanically applied to a 3-cell result
without the reviewer's/user's explicit acknowledgement that the rule now
applies to 3 cells, not 4. Reported as-is; verdict language and
next-direction recommendation left to the reviewer — see
`research/REVIEW_FOR_CHATGPT.md`. Immediately followed by
`EXP-FIRSTANCHOR-DIAG` (below), a causal-decomposition diagnostic over the
3 completed cells' checkpoints, run in parallel on a separate GPU per
explicit user instruction.

---

## EXP-FIRSTANCHOR-DIAG — causal decomposition of EXP-MARGUTIL01's t=1 choice

### Date
2026-09-07

### Research Question
EXP-MARGUTIL01's step-wise regret was overwhelmingly concentrated at t=1
(e.g. ETTh1 H96: 0.854 at t=1, dropping to 0.061 by t=2). Is the first
(t=1) candidate choice's failure the primary cause of the whole sequence's
downstream failure and Stage-2 regression (H1), or does the set-conditioned
Dense selector fail on its own merits at t=2..10 even given a good t=1
anchor (H2)? No new training: reuses each EXP-MARGUTIL01 checkpoint's
frozen encoder, `SetConditioner`, and `UtilityHead` exactly as-is; the only
causal variable is which rule picks candidate 1.

### Configuration
`scripts/eval_firstanchor_diag.py` (new, eval-only). Four arms per query:
**Dense-first** (t=1 also from the Dense model's own argmax — reproduces
EXP-MARGUTIL01's free-running result), **B0-first** (t=1 = B0's own
production retrieval score's argmax — `RelationStage2._retrieval_score_fn()`
/cosine, the exact scorer Stage-2 already uses, not a new score),
**Oracle-first** (t=1 = argmin singleton future MSE, diagnostic-only, uses
`Y_q`), **B0 baseline** (production unforced). t=2..10 for every arm use
the identical, unmodified Dense free-running argmax — no teacher forcing
and no oracle/B0 injection past t=1 in any arm. Verified by unit test
(`test_intervention_affects_only_t1_dense_vs_forced_agree_from_t2`: forcing
t=1 to the value an arm would have picked anyway reproduces the exact
free-running trajectory).

Ran **in parallel with EXP-MARGUTIL01's Weather H720 training**, on GPU 0
while GPU 1 ran Weather H720 — an explicit user instruction issued
mid-session, overriding this experiment's own original spec (which had said
GPU 1 only, wait if busy). Full commands: `results/EXP-FIRSTANCHOR-DIAG/command.txt`.

### Changed Variable
t=1 selection rule only (Dense / B0 / Oracle), applied to an otherwise
completely unmodified EXP-MARGUTIL01 checkpoint's free-running selector.

### Controlled Variables
Same B0 Stage-2 checkpoints, same teacher caches, same `tau_topk`, same
candidate mask/full-memory support as EXP-MARGUTIL01. No training of any
kind. Stage-2 weighting/gate/fusion/base-forecaster/aggregation completely
unchanged — Dense utility scores are never used as a Stage-2 weight.

### Dataset / Horizon
ETTh1 H96, Weather H96, ETTh1 H720 (the 3 EXP-MARGUTIL01 cells that had a
saved checkpoint). **Weather H720: not run — no checkpoint exists** (its
EXP-MARGUTIL01 training was cancelled before saving one).

### Seed
0 (inherited from each reused checkpoint; no new randomness introduced by
this diagnostic beyond none — every arm's t≥2 argmax is deterministic given
the frozen model and the prefix so far).

### Important Hyperparameters
`rank_eval_queries=200` (t=1 rank/top-tail diagnostics subsampled to 200
queries per cell for tractability — candidates are NEVER subsampled, every
diagnostic ranks the FULL valid candidate population for those 200
queries; the main Stage-2/aggregate/candidate-quality metrics run over the
FULL test set, not the 200-query subsample).

### Result Files
`results/EXP-FIRSTANCHOR-DIAG/{cell}_summary.json`,
`{cell}_stepwise_raw.csv`, `comparison.csv`,
`stepwise_aggregate_trajectory.csv`, `t1_rank_diagnostics.csv`,
`t1_candidate_quality.csv`, `notes.md`, `command.txt`, `env.txt`,
`git_commit.txt`, `checkpoint_fingerprints.txt`.

### Results

Stage-2 Final MSE and Recovery-to-B0 (`= (MSE_dense - MSE_arm)/(MSE_dense - MSE_B0)`;
0 = no recovery, 1 = fully recovers to B0, >1 = beats B0):

| Cell | B0 | Dense-first | B0-first (Recovery) | Oracle-first (Recovery) |
|---|---:|---:|---:|---:|
| ETTh1 H96 | 0.37312 | 0.52991 | 0.43661 (**59.5%**) | 0.21717 (**199.5%**) |
| Weather H96 | 0.17553 | 0.33253 | 0.21261 (**76.4%**) | 0.41411 (**-52.0%**) |
| ETTh1 H720 | 0.46827 | 0.49269 | 0.51995 (**-111.6%**) | 0.43635 (**230.7%**) |

**No single clean pattern across cells.** B0-first recovers strongly on
both H96 cells (60-76%) but makes ETTh1 H720 *worse* than Dense-first.
Oracle-first beats B0 outright on both ETTh1 cells but is markedly worse
than Dense-first on Weather H96.

**What IS consistent across all 3 cells:** the stepwise A_weighted(S_t)
trajectory barely moves after t=1 for every arm (e.g. ETTh1 H96
oracle-first: 0.19101 at t=1 → 0.19665 at t=10 — a 3% relative change over
9 further steps). t=1 overwhelmingly determines each arm's final aggregate
on every cell; the disagreement across cells is about WHICH t=1 rule is
good, not about whether t=1 dominates the outcome.

t=1 candidate quality (singleton future MSE, full test set): B0's own pick
beats the Dense model's own t=1 pick 64-70% of the time on every cell
measured (ETTh1 H96: 64.4%, Weather H96: 69.8%, ETTh1 H720: 66.7%), yet
exact matches to the true oracle singleton are rare for both (B0: 1-5%,
Dense: <0.2%). t=1 rank diagnostics (200-query subsample, full candidate
population): the oracle-best candidate's median predicted rank (out of the
full memory bank) is 489-1083 depending on cell — the Dense model's t=1
ranking places the true best candidate nowhere near its own top choices
(Top-1 hit rate 0% on every cell measured); `spearman_within_teacher_top1pct`
is markedly lower (0.04-0.14) than the global teacher-forced Spearman
EXP-MARGUTIL01 reported (0.38-0.66) — global ranking quality does not imply
top-tail ranking quality, and t=1 selection depends entirely on the top
tail.

### Sanity Checks
`pytest tests/`: 485 passed (8 new in `tests/test_exp_firstanchor_diag.py`),
same 2 pre-existing failures, no regression. Structural invariants verified
by test: Oracle-first's t=1 exactly equals the cached greedy oracle's own
first pick on every query checked
(`identity_check_singleton_oracle_eq_teacher_first: true` in every cell's
summary json); forcing t=1 to the value an arm would have picked anyway
reproduces the byte-identical full trajectory; `a_weighted_prefix` never
references the Dense model's own score; `duplicate_rate`/`invalid_rate` =
0.0 on every arm/cell.

### Implementation Notes
New file: `scripts/eval_firstanchor_diag.py`. A GPU sanity run on a real
checkpoint (ETTh1 H96, 10-query subsample) caught one device-placement bug
(`_ndcg_at_k`'s discount tensor was created on CPU while `rel` was on GPU)
before the full run — fixed, re-verified.

### Status
completed (diagnostic, 3 of the intended 4 cells — Weather H720 not run,
no checkpoint exists for it). **No uniform verdict**: t=1 dominates every
cell's outcome (consistent finding), but whether a good t=1 anchor by
itself is *sufficient* to make the Dense selector competitive is
cell-dependent (B0-first: 2/3 cells help, 1/3 hurts; Oracle-first: 2/3
cells help decisively, 1/3 hurts markedly). H1 (first-choice failure as
primary bottleneck) is supported by the trajectory-flatness finding on
every cell, but not by a uniformly effective fix — this sits closer to a
qualified Case A/D (t=1 matters a great deal, deployable B0-first recovers
meaningfully on 2 of 3 cells) than a clean Case C (t=1 is not the
bottleneck) or an unqualified Case A (any reasonable anchor fixes
everything, which the ETTh1 H720 B0-first regression rules out).
Interpretation, novelty assessment, and next-experiment recommendation are
left to the reviewer — see `research/REVIEW_FOR_CHATGPT.md`.

---

## EXP-CONTINUATION-DIAG — exhaustive t=2 continuation diagnostic

### Date
2026-09-07

### Research Question
EXP-MARGUTIL01's regret was concentrated at t=1; EXP-FIRSTANCHOR-DIAG
showed fixing t=1 does not uniformly fix the outcome (cell-dependent).
This experiment tests directly, at t=2, exhaustively over every valid
remaining candidate: does the Dense Marginal Utility selector's own t=2
pick land near the true best continuation, or is there a real,
findable-by-exhaustive-search continuation the selector misses? And is any
failure a global ranking problem or specific to the extreme top tail (the
region greedy Top-K selection actually depends on)?

### Configuration
`scripts/eval_continuation_diag.py` (new, eval-only, no training). Imports
`run_arm`/`a_weighted_prefix`/`encode`/`load_trained_selector` directly
from `scripts/eval_firstanchor_diag.py` (not reimplemented) for t=1/t=2
Dense picks and the B0-weighted aggregate `A(S)`; imports
`dense_utility`/`candidate_weights` from `utils/dense_utility.py`
(EXP-MARGUTIL01's own shared math) for the exhaustive `A(S1+{i})` over
every valid remaining candidate, chunked. For a fixed first candidate `i1`
(one of `dense_first`/`b0_first`/`oracle_first`, identical definitions to
EXP-FIRSTANCHOR-DIAG), computes `A1=A({i1})`, the Dense model's own t=2
pick `i2_dense` (via `run_arm(...,k=2,...)` — the *exact same* call that
produces EXP-FIRSTANCHOR-DIAG's own trajectory, not a reimplementation),
and the exhaustive oracle `i2_oracle=argmin_i A({i1,i})` over every valid
remaining candidate. Full commands: `results/EXP-CONTINUATION-DIAG/command.txt`.

### Changed Variable
None (diagnostic only — no training, no architecture, no objective
change). The only "variable" is which of 3 pre-existing first-anchor
policies fixes `i1`.

### Controlled Variables
Identical checkpoints, B0 score, `tau_topk`, candidate mask, and full-memory
support as EXP-MARGUTIL01/EXP-FIRSTANCHOR-DIAG. `A(S)` computed by the
identical `a_weighted_prefix` function EXP-FIRSTANCHOR-DIAG already uses
(imported, not redefined).

### Dataset / Horizon
ETTh1 H96, Weather H96, ETTh1 H720 (the 3 cells with a saved EXP-MARGUTIL01
checkpoint). Weather H720 not run (no checkpoint — D-0013).

### Seed
0 (inherited from each reused checkpoint).

### Important Hyperparameters
`query_budget=500` per (cell, anchor) combo — first 500 valid queries in
test-split order, full candidate population per query always (never
subsampled; only the number of QUERIES evaluated is bounded, matching the
precedent EXP-FIRSTANCHOR-DIAG set at `rank_eval_queries=200` for its own
per-query diagnostics). Candidate chunk size 4096 (H96 cells) / 1024 (H720).

### Result Files
`results/EXP-CONTINUATION-DIAG/{comparison.csv, continuation_diag_<cell>_<anchor>.csv,
<cell>_<anchor>_summary.json, <cell>_<anchor>_top20_catastrophic.json,
REPORT.md, notes.md, command.txt, env.txt, git_commit.txt,
checkpoint_fingerprints.txt}`.

### Results

| Dataset | H | Anchor | A1 | Dense A2 | Oracle A2 | Cont. Regret | Dense hurt % | Oracle improvable % | Dense hurts & Oracle improves % | Dense true rank median | Oracle pred rank median | Global ρ | Top10% ρ | Top1% ρ |
|---|--:|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| ETTh1 | 96 | dense_first | 1.056 | 1.067 | 0.228 | 0.839 | 37.6 | 100.0 | 37.6 | 5244 | 8372 | -0.195 | -0.558 | -0.257 |
| ETTh1 | 96 | b0_first | 0.569 | 0.569 | 0.279 | 0.289 | 23.2 | 100.0 | 23.2 | 6866 | 8395 | -0.414 | -0.629 | -0.323 |
| ETTh1 | 96 | oracle_first | 0.223 | 0.223 | 0.164 | 0.059 | 48.2 | 100.0 | 48.2 | 3631 | 8279 | 0.156 | -0.525 | -0.347 |
| Weather | 96 | dense_first | 0.849 | 0.781 | 0.038 | 0.743 | 41.6 | 100.0 | 41.6 | 20536 | 30506 | -0.163 | -0.226 | -0.061 |
| Weather | 96 | b0_first | 0.723 | 0.716 | 0.178 | 0.538 | 39.2 | 93.2 | 32.4 | 24987 | 35710 | -0.236 | -0.399 | -0.171 |
| Weather | 96 | oracle_first | 0.030 | 0.035 | 0.021 | 0.014 | **79.6** | 89.6 | **69.4** | 3155 | 20899 | 0.457 | -0.044 | -0.183 |
| ETTh1 | 720 | dense_first | 1.228 | 1.211 | 0.398 | 0.813 | 20.0 | 100.0 | 20.0 | 5286 | 6951 | -0.268 | -0.392 | -0.145 |
| ETTh1 | 720 | b0_first | 0.812 | 0.812 | 0.454 | 0.358 | 12.0 | 100.0 | 12.0 | 6388 | 7092 | -0.574 | -0.546 | -0.152 |
| ETTh1 | 720 | oracle_first | 0.409 | 0.410 | 0.294 | 0.116 | 20.0 | 100.0 | 20.0 | 4334 | 6809 | -0.091 | -0.417 | -0.151 |

**Top1% ρ is negative in all 9 of 9 combinations tested** — the single
most consistent finding of this experiment. Global ρ is mixed (positive in
2/9). Weather H96's `oracle_first` is a distinct catastrophic case:
`ratio_dense` (A2_dense/A1) reaches p99=721M, max=2.1B, with
`corr(alpha2_dense, delta_dense)=+0.659` (the mis-selected candidate
receives disproportionate aggregation weight specifically in this combo —
not a pattern replicated in the other 8 combos, where this correlation is
weak-to-moderately negative).

### Sanity Checks
`pytest tests/`: 490 passed (5 new in `tests/test_exp_continuation_diag.py`),
same 2 pre-existing failures, no regression. Cross-formula check (two
independent code paths computing the same `A(S1+{i2})`): max abs diff
1.9e-6 (`a2_dense` vs. `a2_dense_check`), 2.4e-7 (`a2_oracle` vs.
`a2_oracle_check`) across all 9 combos — float32 noise only. Full sanity
list (8 items) verified and logged: `results/EXP-CONTINUATION-DIAG/REPORT.md`
→ *Sanity checks*.

### Implementation Notes
New file: `scripts/eval_continuation_diag.py`. Imports (not reimplements)
`run_arm`, `a_weighted_prefix`, `encode`, `load_trained_selector` from
`scripts/eval_firstanchor_diag.py`, and `dense_utility`/`candidate_weights`
from `utils/dense_utility.py`. Ran on GPU 1 (returned to the project's
default single-GPU convention after EXP-FIRSTANCHOR-DIAG's one-time GPU-0
exception, D-0014, per explicit user instruction).

### Status
completed (9/9 approved combos; Weather H720 not run, no checkpoint
exists). **Central hypothesis supported**: Dense selector's global utility
ranking is mixed but its extreme-top-tail ranking is consistently
anti-correlated with true continuation quality across every
dataset/horizon/anchor combination tested — the single most consistent
result across this entire diagnostic. A good first anchor does not fix,
and on 2/3 cells (ETTh1 H96, Weather H96) actively worsens, t=2
continuation failure. Full evidence-based breakdown of all 7 required
questions (including the explicit A/B/C/D/E failure-mode judgement):
`results/EXP-CONTINUATION-DIAG/REPORT.md`. Interpretation, novelty
assessment, and next-experiment recommendation left to the reviewer — see
`research/REVIEW_FOR_CHATGPT.md`.

---

## EXP-TOPTAIL-RANK01 — loss-formulation comparison (R0/R1/R2) for the Dense selector

### Date
2026-09-07

### Research Question
EXP-CONTINUATION-DIAG found extreme-top-tail utility ranking negatively
correlated with true continuation quality in all 9 dataset/horizon/anchor
combinations tested, while global ranking correlation was mixed. Is
EXP-MARGUTIL01's pointwise SmoothL1 regression target itself responsible —
does replacing/augmenting it with a top-tail-focused pairwise ranking loss
improve the decision-relevant top-tail ranking, and does that reach
free-running Top-K and Stage-2?

### Configuration
New file `scripts/train_toptail_rank01.py`, importing (not reimplementing)
`build_experiment`/`encode`/`memory_value`/`run_sequence_dense` from
`scripts/train_margutil01.py`. Three loss modes, everything else identical
to EXP-MARGUTIL01 (frozen B0 encoder, `SetConditioner`, `EmptySetToken`,
`UtilityHead=a*cosine+b`, oracle-prefix teacher forcing, full-memory
candidate support): `--loss_mode smoothl1` (R0, EXP-MARGUTIL01's own arm,
reused not retrained), `--loss_mode pairwise` (R1: logistic pairwise
ranking between true-top-1% positives and predicted-high hard negatives,
fully vectorised/batched — an early per-row-Python-loop implementation was
replaced after real-data timing showed it impractically slow), `--loss_mode
hybrid --lambda_rank 1.0` (R2 = pairwise + 1.0*SmoothL1). Evaluation reuses
`scripts/eval_margutil01_stage2.py` (Stage-2), `scripts/eval_firstanchor_diag.py`
(t=1, new file this campaign), `scripts/eval_continuation_diag.py`
(exhaustive t=2, new file this campaign) unmodified.

### Changed Variable
Training loss only (smoothl1 / pairwise / hybrid). Same architecture, same
target definition (`u_i^(t)=-A_weighted(S*_{t-1}+{i})`), same checkpoint-
selection criterion (`val_overlap@10`) across all three arms.

### Controlled Variables
Frozen B0 encoder, `SetConditioner`/`EmptySetToken`/`UtilityHead`
architecture (`trainable_params=49794`, verified identical across R0/R1/R2),
oracle-prefix teacher forcing, teacher cache, full-memory candidate support,
K=10, seed 0, Stage-2 checkpoint/architecture/aggregation.

### Dataset / Horizon
ETTh1 H96 (all 3 arms, complete). Weather H96: R0 (existing EXP-MARGUTIL01
result) only — **R1 was stopped after 2 epochs by explicit user decision**
(loss had plateaued near `ln(2)≈0.693` on both epochs, the value indicating
the pairwise loss cannot separate positives from negatives; no checkpoint
past epoch 1 quality); **R2 was stopped after training an epoch-1
checkpoint, then that checkpoint was explicitly discarded/unevaluated by
user decision** ("stop하고 새로운 실험 진행할거야") in favour of proceeding
to EXP-ASYM-SCORER01. Weather H96 therefore has NO R1 or R2 result in this
campaign. H720 not run (out of scope).

### Seed
0.

### Result Files
`results/EXP-TOPTAIL-RANK01/{ETTh1_H96,Weather_H96}_stage2.json` (R1),
`results/EXP-TOPTAIL-RANK01/R2/ETTh1_H96_R2_*` (Stage-2, t=1, t=2 for R2),
plus the reused R0 files in `results/EXP-MARGUTIL01/`.

### Results — ETTh1 H96 (complete, 3/3 arms)

| Arm | Stage-2 MSE | Delta vs B0 | gap_recovery | HardAgg (seq) | Global Spearman (t=1, teacher-forced) |
|---|---:|---:|---:|---:|---:|
| B0 | 0.37312 | — | — | 0.407 | — |
| R0 (SmoothL1) | 0.52991 | +0.157 | -1.832 | 1.657 | +0.543 |
| R1 (pairwise) | 0.39978 | +0.027 | -0.303 | 0.600 | -0.162 |
| R2 (hybrid) | 0.39526 | +0.022 | -0.263 | 0.551 | -0.103 |

t=1 (FIRSTANCHOR-DIAG): Spearman-within-true-top-1% went -0.558 (R0) →
+0.163 (R1) → +0.213 (R2) — monotonic. t=2 (CONTINUATION-DIAG,
`dense_first` anchor): `hurt_frac` 37.6% (R0) → 24.6% (R1) → 23.2% (R2),
`continuation_regret` 0.839 → 0.479 → 0.350 — also monotonic on most
metrics, though R1 has a very slightly better t=2 oracle-predicted-rank
and top-tail Spearman than R2 specifically (small differences; see
`research/REVIEW_FOR_CHATGPT.md` for full numbers). R0→R1→R2 is a clean,
consistent improvement on Stage-2/`gap_recovery`/HardAggregate/t=1; every
global teacher-forced Spearman is negative or near-zero for R1/R2 while
positive for R0, reproducing this project's global-vs-top-tail dissociation
finding at three points along a loss-weighting spectrum.

### Sanity Checks
`pytest tests/`: 496 passed (6 new in `tests/test_exp_toptail_rank01.py`),
same 2 pre-existing failures, no regression. `trainable_params=49794`
identical across R0/R1/R2, confirmed by startup log every run.

### Training stability
Both R1 and R2 select `best_epoch=1` by `val_overlap@10` on ETTh1 H96 and
decline afterward; R1's `pairwise_loss` converges to almost exactly
`ln(2)=0.693` by epoch 6 (the degenerate value), while R2's stays in
`0.697-0.703` throughout and `UtilityHead`'s scale `a` declines gradually
(0.921→0.799) rather than collapsing toward 0 — R2 measurably softens but
does not eliminate the degradation pattern.

### Implementation Notes
New files: `scripts/train_toptail_rank01.py`, `tests/test_exp_toptail_rank01.py`.
A per-row Python-loop pairwise-loss implementation was replaced with a
fully vectorised (batched `topk`/`scatter`/`gather`) version after a
real-data GPU sanity run showed the loop version was impractically slow
(a full epoch did not complete within the harness's execution window); the
vectorised rewrite was re-verified against the same test suite before any
further training.

### Status
completed for ETTh1 H96 (3/3 arms); Weather H96 incomplete by explicit
user decision (R1/R2 both stopped, no result for either). R0→R1→R2 shows a
consistent Stage-2/t=1/t=2 improvement on ETTh1 H96, but R2's Stage-2 MSE
(0.39526) is still worse than B0 (+0.022, above the project's 0.01
noise-scale reference). Followed immediately by EXP-ASYM-SCORER01 (below),
testing whether R2's remaining gap is a scorer-capacity bottleneck.
Interpretation and next-direction recommendation left to the reviewer —
see `research/REVIEW_FOR_CHATGPT.md`.

---

## EXP-ASYM-SCORER01 — R2 + asymmetric scorer vs. R2 + cosine

### Date
2026-09-07

### Research Question
Does R2's remaining top-tail/continuation ranking failure and Stage-2 gap
stem from the fixed symmetric cosine geometry's limited expressiveness?
`C0 = R2 + cosine` (existing checkpoint, reused) vs. `C1 = R2 +
asymmetric` (`cos(W_q h_t, W_k e_i)`, `W_q`/`W_k` identity-initialised),
everything else identical.

### Configuration
`models/DenseUtilityRetriever.py::AsymmetricUtilityHead` wraps
`layers.retrieval_metric.RetrievalMetric(kind='asymmetric', layer_norm=False,
output='cosine')` (the project's existing, previously-reviewed asymmetric
scorer implementation — not reimplemented) with the same affine
`scale`/`bias` `UtilityHead` already uses. `scripts/train_toptail_rank01.py`
gained `--scorer_mode {cosine,asymmetric}` (default `cosine`, i.e. every
prior experiment's behaviour is unchanged); `asymmetric` runs an
identity-init equivalence check (`layers.retrieval_metric.cosine_init_deviation`)
before training and aborts if `>= 1e-6`. `eval_margutil01_stage2.py`/
`eval_firstanchor_diag.py` (and `eval_continuation_diag.py`, which imports
from it) read the checkpoint's own saved `scorer_mode` and instantiate the
matching head — old checkpoints (no such key) default to `UtilityHead`,
unaffected. Full commands: `results/EXP-ASYM-SCORER01/command.txt`.

### Changed Variable
Scorer geometry only (cosine vs. asymmetric). R2's hybrid loss
(`lambda_smooth=1.0`), all hyperparameters, and checkpoint-selection
criterion (`val_overlap@10`) held identical.

### Controlled Variables
Frozen B0 encoder (same checkpoint), `SetConditioner`/`EmptySetToken`,
teacher cache/definition/normalization, positive/hard-negative definition
and sampling counts, optimizer/lr/batch/epochs/patience/seed, candidate
validity mask, K=10, full-memory support, free-running inference, Stage-2
architecture/aggregation/gate/fusion, evaluation scripts and query
subsets — all identical between C0 and C1. `trainable_params`: C0=49794,
C1=82562 (= C0 + 32770 for `W_q`/`W_k`, confirmed to be the only source of
the difference by startup log and unit test).

### Dataset / Horizon
ETTh1 H96 only, seed 0. Weather and H720 explicitly not run.

### Result Files
`results/EXP-ASYM-SCORER01/{command.txt, config.json, scorer_init_check.json,
ETTh1_H96_C1_stage2.json, ETTh1_H96_C1_summary.json,
ETTh1_H96_C1_dense_first_summary.json, continuation_diag_ETTh1_H96_C1_dense_first.csv,
ETTh1_H96_C1_top20_catastrophic.json, metrics.csv, REPORT.md, logs/,
working_tree.diff, checkpoint_fingerprints.txt, env.txt, git_commit.txt}`.

### Results

Identity-init check: `max_abs_score_deviation = 0.0` (< 1e-6), verified
before training started. C1 selected `best_epoch=3` (`val_overlap@10=0.0085`),
slightly better than C0's `best_epoch=1` (`0.0085` vs `0.0082`) — the
training-time proxy metric improved. Every downstream metric did not:

| Metric | C0 | C1 | Delta |
|---|---:|---:|---:|
| Stage-2 MSE | 0.39526 | 0.41257 | **+0.01731 (worse)** |
| Delta vs B0 | +0.02214 | +0.03945 | worse |
| gap_recovery | -0.263 | -0.488 | worse |
| HardAggregate@10 (seq) | 0.551 | 0.690 | worse |
| t1 Spearman within true top-1% | 0.213 | 0.159 | worse |
| t1 oracle-best predicted rank median | 99 | 180 | worse |
| t1 Top-50 containment | 30.5% | 26.5% | worse |
| t2 hurt_frac | 23.2% | 24.0% | worse |
| t2 selected-second true rank median | 1934 | 2799 | worse |
| t2 oracle-second predicted rank median | 560 | 673 | worse |
| t2 Spearman within true top-10% | 0.288 | 0.256 | worse |
| t2 continuation regret | 0.350 | 0.375 | worse |
| t2 Spearman GLOBAL | 0.177 | 0.248 | **better** |

`duplicate_rate`/`invalid_rate` = 0.0 for both arms. Every metric is flat
or worse for C1 except global (non-top-tail) Spearman, which improves —
the same global-vs-top-tail dissociation this project has repeatedly found
elsewhere replays at the scorer-geometry level: more expressive geometry
fits the bulk of the ranking marginally better while the decision-critical
extreme tail gets worse. `cond(W_k)` grew monotonically through training
(24.6 at epoch 1 → 58.8 at the selected epoch 3 → 139.6 by epoch 8),
suggesting the candidate-side projection moves toward an increasingly
anisotropic transformation.

### Sanity Checks
`pytest tests/`: 502 passed (6 new in `tests/test_exp_asym_scorer01.py`),
same 2 pre-existing failures, no regression. Identity-init equivalence,
gradient flow to `W_q`/`W_k`, encoder-frozen invariant, and param-count
audit all verified (see REPORT.md section 2-3 for the full checklist).

### Implementation Notes
New class: `models/DenseUtilityRetriever.py::AsymmetricUtilityHead`.
Modified: `scripts/train_toptail_rank01.py` (`--scorer_mode` flag),
`scripts/eval_margutil01_stage2.py`, `scripts/eval_firstanchor_diag.py`
(`load_trained_selector` now scorer-mode-aware). New tests:
`tests/test_exp_asym_scorer01.py`.

### Status
completed (1 cell, C0 vs C1). **Evidence AGAINST the scorer-capacity-
bottleneck hypothesis**: every decision-relevant metric (Stage-2,
HardAggregate, t=2 continuation, most of t=1) is worse with the asymmetric
scorer than with plain cosine, despite the training-time checkpoint-
selection proxy (`val_overlap@10`) preferring the asymmetric arm — this
itself is informative about that proxy's reliability. Not evidence that no
asymmetric scorer could ever help under different hyperparameters/training
budget (not swept, per the pre-registered scope). Per the user's explicit
instruction, no further experiment (Weather, H720, hyperparameter sweep,
Mahalanobis, encoder unfreezing, SetConditioner changes) was started.
Interpretation, novelty assessment, and next-experiment recommendation
left to the reviewer — see `research/REVIEW_FOR_CHATGPT.md`.


## EXP-STRONG-SCORER-DIAG01 — R2 + cosine vs. R2 + strong nonlinear residual pair scorer (COMPLETE: ETTh1 H96 only)

**Date:** 2026-09-07
**Status:** completed (1 cell, C0 vs C2). No next experiment auto-started, per
explicit user instruction.

### Research Question

Is R2's remaining top-tail/continuation ranking failure and Stage-2 gap
caused by the cosine scorer's limited expressiveness, as opposed to the
encoder or loss? `C0 = R2 + cosine` (`UtilityHead`, existing
EXP-TOPTAIL-RANK01/R2 checkpoint, reused verbatim) vs. `C2 = R2 +
StrongResidualPairScorer` (`u_hat = a*cos(h_t,e_i)+b + Delta_phi(h_t,e_i)`,
`Delta_phi` a nonlinear MLP over `[h,e,h⊙e,|h-e|]`, zero-init final layer so
C2 ≡ C0 at construction), everything else held identical.

### Configuration

`models/DenseUtilityRetriever.py::StrongResidualPairScorer` added (zero-init
final MLP layer; `forward` for the full-memory bank, `forward_batched` for
the small per-row gathered pairwise-loss pool). `scripts/train_toptail_rank01.py`
gained `--scorer_mode {cosine,asymmetric,strong_pair}` / `--scorer_chunk_size`
(default `cosine`, prior behaviour unchanged); `strong_pair` runs a zero-init
equivalence check before training (abort if `max_abs_score_deviation >=
1e-6`). Eval scripts (`eval_margutil01_stage2.py`, `eval_firstanchor_diag.py`,
`eval_continuation_diag.py`) read the checkpoint's own `scorer_mode` to
instantiate the matching head; old checkpoints default to `UtilityHead`,
unaffected. `--split {train,val,test}` (default `test`) added to
`eval_firstanchor_diag.py`/`eval_continuation_diag.py` for the mandatory
train-vs-test generalization diagnostic. Full commands:
`results/EXP-STRONG-SCORER-DIAG01/command.txt`.

### OOM fix (mid-experiment, reported to and approved by the user before the full run)

The original `strong_pair` training path held the full autograd graph across
all K teacher-forced steps x channels x candidate chunks before a single
`.backward()`, which OOM'd on GPU 1 (compounded by, but not solely caused by,
concurrent external GPU 1 usage from another user's job). Rewritten to a
memory-safe streaming design: full-memory hard negatives mined in a chunked
`torch.no_grad()` pass (global Top-K, never a shortlist — `FULL MEMORY ->
DIRECT TOP-K` unaffected); the pairwise loss computed with grad on only the
small gathered positive/hard-negative pool and backpropped immediately; the
dense SmoothL1 term computed and backpropped per candidate chunk, freeing
each chunk's graph before the next; `optimizer.step()` called exactly once
per batch by the caller (`run_epoch`), never inside `strong_pair_step()`.
`SetConditioner`'s `h_t` is recomputed fresh per chunk via a zero-arg `m_fn`
callable (not a precomputed tensor) — necessary because the precomputed-
tensor version double-backwards through the trainable `EmptySetToken`'s
shared graph segment at t=0 (`RuntimeError: Trying to backward through the
graph a second time`), caught only on real data; a dedicated regression test
using a real `EmptySetToken` was added
(`test_streaming_with_trainable_m_source_does_not_double_backward`).
Mathematical equivalence to the original unchunked implementation verified
BEFORE the real run by
`tests/test_exp_strong_scorer_streaming.py::test_streaming_gradients_match_unchunked_reference`
(gradient equality on every `SetConditioner`/scorer parameter, `atol=1e-4`).
Loss normalization (chunk-sum / global valid count) and lambda weighting
unchanged from the original R2 hybrid math. After the fix: sanity run
completed in 33min with `peak_gpu_mem=504MiB`, no OOM (previously could not
even complete a sanity epoch).

### Changed Variable

Scorer capacity only (cosine vs. cosine + zero-init nonlinear MLP residual).
R2's hybrid loss (`lambda_smooth=1.0`), all other hyperparameters, and
checkpoint-selection criterion (`val_overlap@10`) held identical.

### Controlled Variables

Frozen B0 encoder (same checkpoint), `SetConditioner`/`EmptySetToken`,
teacher cache/definition/normalization, positive/hard-negative definition
and sampling counts, optimizer/lr/batch/epochs/patience/seed, candidate
validity mask, K=10, full-memory support, free-running inference, Stage-2
architecture/aggregation/gate/fusion, evaluation scripts and query
subsets — all identical between C0 and C2. `trainable_params`: C0=49794,
C2=378243 (`SetConditioner`=49664, `EmptySetToken`=128,
`StrongResidualPairScorer`=328451).

### Dataset / Horizon

ETTh1 H96 only, seed 0. Weather, H720, architecture sweep, encoder
unfreezing explicitly not run.

### Positive Control

PASSED: `tests/test_exp_strong_scorer_diag01.py::test_smalln_positive_control_fits_known_utility_landscape`
— `StrongResidualPairScorer` trained with the real `pairwise_step_loss` fits
a synthetic known utility landscape (16 queries, 256 candidates, 400 steps)
to mean oracle-best predicted rank < 20/256.

### Result Files

`results/EXP-STRONG-SCORER-DIAG01/{command.txt, config.json, smalln_summary.json,
stage2_eval.json, ETTh1_H96_C2_stage2.json, t1_diagnostic.json,
t2_continuation.json, train_vs_val_test.json, metrics.csv,
ETTh1_H96_C2_{test,train}_summary.json,
continuation_diag_ETTh1_H96_C2_{test,train}_dense_first.csv,
ETTh1_H96_C2_{test,train}_dense_first_summary.json, train_summary.json,
REPORT.md, logs/, working_tree.diff, checkpoint_fingerprints.txt, env.txt,
git_commit.txt}`.

### Results

Zero-init check: `max_abs_score_deviation = 0.000e+00`. Training:
`best_epoch=1` (`val_overlap@10=0.0098`), early-stopped at epoch 6; epoch-by-
epoch `val_overlap@10`: 0.0098→0.0047→0.0073→0.0059→0.0057→0.0041 (monotonic
decline after epoch 1, matching R1/R2/C1's own pattern). `wall_clock=8936.2s`,
`peak_gpu_mem=504MiB`.

| Metric | C0 (cosine) | C2 (strong pair) | Delta |
|---|---:|---:|---:|
| Stage-2 MSE | 0.39526 | 0.39708 | worse (+0.00182) |
| Delta vs B0 | +0.02214 | +0.02396 | worse |
| gap_recovery | -0.2626 | -0.3392 | worse |
| HardAggregate@10 (seq) | 0.5514 | 0.6224 | worse |
| t1 Spearman within true top-1% (test) | 0.2128 | 0.2251 | **better** |
| t1 oracle-best predicted rank median (test) | 99 | 97 | **better** |
| t1 Top-50 containment (test) | 30.5% | 32.0% | **better** |
| t1 Spearman within true top-1% (train, C2 only) | — | 0.1250 | (worse than C2-test) |
| t1 oracle-best predicted rank median (train, C2 only) | — | 654 | (worse than C2-test) |
| t2 hurt_frac (test) | 0.232 | 0.214 | **better** |
| t2 selected true rank median (test) | 1934.0 | 3859.5 | worse |
| t2 oracle predicted rank median (test) | 560.0 | 442.0 | **better** |
| t2 Spearman within true top-1% (test) | 0.1404 | 0.1818 | **better** |
| t2 continuation_regret (test) | 0.3498 | 0.3657 | worse |
| t2 hurt_frac / spearman / regret (train, C2 only) | — | 0.324 / 0.058 / 0.294 | (worse than C2-test on hurt_frac, spearman) |

C2 shows small, consistent improvements over C0 on t1 (test) and several t2
(test) top-tail ranking metrics, but Stage-2 MSE, `gap_recovery`,
HardAggregateMSE, `selected true rank`, and `continuation_regret` (the
metrics closest to realized free-running selector behavior) are
flat-to-worse. C2's own train-split t1/t2 diagnostics are notably WORSE than
its test-split diagnostics — the opposite of classic overfitting — consistent
with `best_epoch=1` leaving the checkpoint barely displaced from the C0
initialization before `val_overlap@10` begins declining.

### Sanity Checks

`pytest tests/`: 513 passed (final full sweep after all docs assembled; 6 new in `tests/test_exp_strong_scorer_diag01.py`
+ 6 in `tests/test_exp_strong_scorer_streaming.py`, +1 shared fixture change
elsewhere), same 2 pre-existing failures, no regression. Small-N positive
control, zero-init equivalence, chunked==unchunked scoring equivalence,
streaming-vs-unchunked gradient equivalence, and the `EmptySetToken`
double-backward regression test all verified — full checklist in
`results/EXP-STRONG-SCORER-DIAG01/REPORT.md`.

### Implementation Notes

New class: `models/DenseUtilityRetriever.py::StrongResidualPairScorer`.
Modified: `scripts/train_toptail_rank01.py` (`--scorer_mode strong_pair`,
`mine_pairs()` factored out, `strong_pair_step()` streaming path,
`run_epoch()` streaming branch), `scripts/eval_margutil01_stage2.py`,
`scripts/eval_firstanchor_diag.py`, `scripts/eval_continuation_diag.py`
(scorer-mode-aware loading; `--split` support). New tests:
`tests/test_exp_strong_scorer_diag01.py`,
`tests/test_exp_strong_scorer_streaming.py`.

### Conclusion

No clean fit to any of the 3 pre-registered outcomes (A: capacity bottleneck
confirmed; B: train fits but generalization fails; C: even a strong scorer
can't fit/use supervision). The positive control (§ above) rules out C. The
mixed test-split result (small ranking-metric gains, flat-to-worse Stage-2/
HardAggregate/realized-selection metrics) combined with `best_epoch=1` and
train-worse-than-test on C2's own diagnostics does not match A or a classic
B either — the more parsimonious reading is that the bottleneck is not
per-candidate scorer expressiveness, but something in the sequential
teacher-forced training signal itself (dense marginal utility target,
weighted-set greedy oracle supervision, or train/inference distribution
mismatch) that stalls every scorer/loss arm tried this session at
`best_epoch=1`. Full 12-point breakdown:
`results/EXP-STRONG-SCORER-DIAG01/REPORT.md`. Per the user's explicit
instruction, no further experiment was started.

## EXP-ENCODER-UNFREEZE01 — R2 + cosine + frozen encoder vs. R2 + cosine + trainable encoder (COMPLETE: ETTh1 H96 only)

**Date:** 2026-09-08
**Status:** completed (1 cell, C0 vs E1). No next experiment auto-started, per
explicit user instruction.

### Research Question

Is the frozen B0 encoder representation itself a bottleneck for the R2
set-conditioned utility objective (rather than the scorer or the loss,
both already tested and rejected as the primary bottleneck in
EXP-ASYM-SCORER01/EXP-STRONG-SCORER-DIAG01)? `C0 = R2 + cosine + FROZEN B0
encoder` (existing EXP-TOPTAIL-RANK01/R2 checkpoint, reused verbatim) vs.
`E1 = R2 + cosine + TRAINABLE encoder` (same architecture, initialised from
the same B0 checkpoint), everything else held identical.

### Configuration

`scripts/train_encoder_unfreeze01.py` (new). Only structural change vs.
C0/R2: `model.encoder.parameters()` set `requires_grad=True` instead of
`False`. Query AND candidate embeddings are both re-derived from the
CURRENT encoder parameters at every training step (`encode_raw()` always
re-runs `model.encoder` on raw input, never a cached bank) — verified by a
dedicated sanity test that perturbs encoder parameters and confirms
candidate embeddings change immediately. Memory-safe streaming training
(`encoder_unfreeze_step`): full-memory hard-negative mining in a
`torch.no_grad()` chunked pass (exact global Top-K, `FULL MEMORY -> DIRECT
TOP-K` unaffected); pairwise loss re-encodes only the small gathered
positive/hard-negative candidates WITH grad, backward immediately; dense
SmoothL1 term re-encodes the full candidate bank chunk by chunk, each
chunk getting its own fresh encoder forward and immediate backward.
`q`/`m` are produced by zero-arg callables (`q_fn`/`m_fn`) that rerun the
encoder fresh on every call, generalising EXP-STRONG-SCORER-DIAG01's
`m_fn` fix (needed only for `EmptySetToken` there) to every tensor that now
touches the trainable encoder. `optimizer.step()` called exactly once per
batch. Full commands: `results/EXP-ENCODER-UNFREEZE01/command.txt`.

### Mandatory sanity checks (spec section 12, all before the GPU run)

`tests/test_exp_encoder_unfreeze01.py`, 4/4 PASSED: (A) encoder receives a
nonzero gradient from both query- and candidate-side branches; (B) with
`requires_grad=False` (the C0/R2 configuration), the encoder receives
exactly no gradient; (C) perturbing encoder parameters changes candidate
embeddings on the next forward (no stale/cached bank); (D) unchunked vs.
chunked streaming training from identical initial weights produce matching
losses (atol 1e-4) and matching gradients on every encoder/SetConditioner/
UtilityHead parameter (atol 1e-3) — this also exercises the `q_fn`/`m_fn`
fresh-recompute-per-call contract across many chunk iterations without a
double-backward `RuntimeError`. A real-data smoke run (1 epoch, full
ETTh1 train/val, GPU) confirmed correct execution (97% GPU util, ~6GB mem,
no OOM/RuntimeError) before a targeted timing probe picked
`cand_chunk_size=2048` for the full run.

### Changed Variable

Encoder trainability only (frozen vs. trainable). R2's hybrid loss
(`lambda_smooth`/`lambda_rank=1.0`), cosine `UtilityHead`, `SetConditioner`/
`EmptySetToken`, optimizer/LR (`0.001`, single Adam over every trainable
parameter including the encoder — no separate encoder LR or LR sweep; an
explicit search of `exp/exp_stage1_relation.py`, `models/RelationStage1.py`,
`run.py` found no existing "encoder LR multiplier" policy in this codebase
before this experiment, so none was invented), `train_epochs=10`,
`patience=5`, `batch_size=32`, checkpoint-selection criterion
(`val_overlap@10`) all held identical between C0 and E1 (all inherited
unmodified from the shared base checkpoint's own `args`).

### Dataset / Horizon

ETTh1 H96 only, seed 0. Weather, H720, 3 seeds, encoder depth/width/
architecture sweep explicitly not run.

### Result Files

`results/EXP-ENCODER-UNFREEZE01/{command.txt, config.json, sanity_summary.json,
train_summary.json, encoder_drift.csv, representation_diagnostics.json,
t1_diagnostic.json, t2_continuation.json, train_vs_val_test.json,
stage2_eval.json, ETTh1_H96_E1_stage2.json,
ETTh1_H96_E1_{test,train}_summary.json,
continuation_diag_ETTh1_H96_E1_{test,train}_dense_first.csv,
ETTh1_H96_E1_{test,train}_dense_first_summary.json, metrics.csv, REPORT.md,
logs/, working_tree.diff, checkpoint_fingerprints.txt, env.txt,
git_commit.txt}`.

### Results

Training: `best_epoch=1` (`val_overlap@10=0.0083`, vs. C0's own
`best_epoch=1`/`val_overlap@10=0.008228` — statistically indistinguishable
at this metric), early-stopped at epoch 6 (patience 5). `encoder_grad_norm`
nonzero every epoch (0.046-0.087). `wall_clock=4631.1s` (~77min),
`peak_gpu_mem=515MiB`, no OOM.

**Representation collapse** (channel 0, first 256 memory rows,
`utils.rank_losses.embedding_geometry`):

| | B0 | E1 epoch1 (selected) | E1 epoch3 | E1 epoch6 (final) |
|---|---:|---:|---:|---:|
| `embedding_effective_rank` | 17.48 | 2.96 | 1.89 | 1.49 |
| `embedding_pairwise_cosine_mean` | 0.474 | 0.993 | 0.988 | 0.915 |

Collapse is already severe after 1 epoch (the checkpoint actually used for
every downstream evaluation below) and deepens monotonically for as long
as training continues; a per-epoch drift probe corroborates this
(`encoder_drift.csv`: L2 drift 0.826→0.931, cosine-to-B0 0.650→0.559,
epoch 1→6).

| Metric | C0 (frozen) | E1 (trainable) | Delta |
|---|---:|---:|---:|
| Stage-2 MSE | 0.39526 | 0.40668 | worse |
| gap_recovery | -0.2626 | -0.3703 | worse |
| HardAggregate@10 (seq) | 0.5514 | 0.5466 | better (small) |
| t1 Spearman within true top-1% (test) | 0.2128 | 0.1455 | worse |
| t1 oracle-best predicted rank median (test) | 99 | 210 | worse |
| t1 Top-50 containment (test) | 30.5% | 20.5% | worse |
| t2 hurt_frac (test) | 0.232 | 0.194 | better |
| t2 selected true rank median (test) | 1934.0 | 539.5 | better |
| t2 oracle predicted rank median (test) | 560.0 | 318.0 | better |
| t2 Spearman within true top-1% (test) | 0.1404 | 0.1358 | ~worse (small) |
| t2 continuation_regret (test) | 0.3498 | 0.3987 | worse |

t2 shows a mixed picture: E1 improves `hurt_frac` and both rank-median
metrics but is worse on `continuation_regret` (the metric closest to
realized utility loss). E1's own train-split t1/t2 diagnostics are mostly
worse than its test-split diagnostics (same train-worse-than-test direction
EXP-STRONG-SCORER-DIAG01's C2 showed), consistent with `best_epoch=1`
reflecting only 1 epoch's adaptation rather than train-set overfitting.

### Sanity Checks

`pytest tests/`: 517 passed (4 new in `tests/test_exp_encoder_unfreeze01.py`),
same 2 pre-existing failures, no regression.

### Implementation Notes

New: `scripts/train_encoder_unfreeze01.py`, `tests/test_exp_encoder_unfreeze01.py`.
Modified: `models/DenseUtilityRetriever.py` (`UtilityHead.forward_batched`
added, analogous to `StrongResidualPairScorer`'s own, for the streaming
pairwise-loss step). No changes needed to `eval_margutil01_stage2.py`/
`eval_firstanchor_diag.py`/`eval_continuation_diag.py` — these already load
each checkpoint's own full `model_state_dict` (encoder included) and use
`model.encoder` directly for both query and candidate encoding, so E1's
adapted encoder is picked up automatically with zero eval-side code
changes; verified via checkpoint-fingerprint mismatch between C0/B0's and
E1's saved checkpoints.

### Conclusion

**Outcome D (representation collapse), not A/B/C.** Not Outcome A: Stage-2
MSE and `gap_recovery` (the two metrics most tied to this project's
"did this actually help" question) are both worse than C0. Not Outcome B:
t1 top-tail ranking is uniformly worse, not better, so there is no
local-ranking gain to contrast against a free-running failure. Not
Outcome C: the encoder moved a great deal, and several metrics moved with
it (not "flat"). **Outcome D fits the cleanest, most consistent signal**:
`embedding_effective_rank` collapses ~83% within 1 epoch and deepens every
epoch trained, `pairwise_cosine_mean` rises to 0.99+, replicating
EXP-SEQDIAG01's earlier collapse finding under a different objective, now
also under R2's more careful hybrid loss and a cosine scorer already ruled
out as the bottleneck. The t2 rank-median improvements are real but read as
consistent with a collapsed representation placing picks favorably in a
degenerate space, not as evidence of a genuinely improved representation —
`continuation_regret` and Stage-2 MSE (arguably the more decision-relevant
metrics) both point the other way. Per the pre-registered decision rule,
Outcome D does NOT justify an encoder depth/capacity follow-up — a bigger
encoder trained the same (uncontrolled) way would be expected to collapse
similarly, plausibly faster. No collapse-prevention regularisation,
partial-layer unfreezing, or encoder-LR tuning was started in response.
Full breakdown: `results/EXP-ENCODER-UNFREEZE01/REPORT.md`. Per the user's
explicit instruction, no further experiment was started.

## EXP-ENCODER-ANCHOR01 — R2 + cosine + trainable encoder + B0-anchor regularizer (COMPLETE: ETTh1 H96 only)

**Date:** 2026-09-08
**Status:** completed (1 cell, C0 vs E1 vs E2). Per user's explicit
pre-approval, EXP-ORACLE-CHOICE01 (D1), EXP-TEACHER-FORCING-DIAG01, and
EXP-ONPOLICY-PREFIX01 (T1) proceed next in sequence regardless of this
result.

### Research Question

`EXP-ENCODER-UNFREEZE01` (E1) found that letting the encoder train under
R2's objective causes severe, monotonically-deepening representation
collapse and worse Stage-2 than the frozen baseline C0. Does adding a
B0-anchor regularizer to the SAME trainable-encoder setup PREVENT the
collapse, and if so, does that let encoder adaptation actually beat C0 on
Stage-2 -- or was collapse never the primary bottleneck?

### Configuration

New script `scripts/train_encoder_anchor01.py` (does not modify
`scripts/train_encoder_unfreeze01.py`, preserving E1's own reproducibility).
`encoder_ref = copy.deepcopy(model.encoder)` taken immediately after loading
B0 weights, before `requires_grad=True`; permanently frozen/eval/no_grad,
never touched by the optimizer. Anchor loss `L_anchor = 1 - cos(f_theta(x),
f_theta0(x))` evaluated once per (batch, channel) over the query batch and
the FULL candidate bank (chunked purely for memory, sum-then-divide-by-
global-N normalisation so chunking is exactly loss/gradient-equivalent to
an unchunked pass), added to R2's loss with `lambda_anchor=0.16` (fixed, no
sweep -- selected via a single fixed-batch gradient-norm-ratio diagnostic
on the EXISTING E1 epoch-1 checkpoint, comparing R2-only vs. anchor-only
encoder gradient magnitude; measured ratio 0.1628, rounded to 0.16; see
`results/EXP-ENCODER-ANCHOR01/notes.md` and `lambda_selection_probe.py`).
Every other hyperparameter/architecture/protocol identical to E1 (same
optimizer/LR/epochs/patience, checkpoint-selection criterion
`val_overlap@10`, full-memory memory-safe streaming design, Stage-2
architecture/gate/fusion unchanged). Full commands:
`results/EXP-ENCODER-ANCHOR01/command.txt`.

### Mandatory sanity checks (S1-S8, all before the GPU run)

`tests/test_exp_encoder_anchor01.py`, 8/8 PASSED: S1 zero-step equivalence
(trainable encoder == frozen reference at construction), S2 anchor loss ~0
at init, S3 anchor gradient nonzero after a controlled perturbation, S4
candidate-side gradient flows, S5 query-side gradient flows, S6 no stale
candidate bank, S7 chunk equivalence for the anchor loss (exact, atol
1e-5), S8 `anchor_step` never calls `optimizer.step()` itself.

Two bugs caught and fixed during development, before the real run: (1) S1
initially failed (max|z-z0|=5.478e-02) because `model.encoder` was compared
in `.train()` mode (dropout=0.1 active) against `encoder_ref`'s permanent
`.eval()` mode -- fixed by evaluating S1 in eval mode on both sides before
restoring `model.encoder.train()`. (2) S7 initially failed at a 0.0013
discrepancy because the candidate-chunk anchor loss averaged each chunk's
own mean rather than sum-then-divide-by-global-N, silently over-weighting
a smaller final chunk -- fixed to match `encoder_unfreeze_step`'s own
established chunking convention, re-verified exact (atol 1e-5).

### Changed Variable

The B0-anchor regularizer only (added on top of E1, which is otherwise
unchanged). `lambda_anchor=0.16` fixed, no sweep.

### Dataset / Horizon

ETTh1 H96 only, seed 0. Weather, H720, 3 seeds, lambda_anchor sweep,
VICReg/covariance/variance regularizers, EMA, stop-gradient, separate
encoder LR, encoder depth/width sweep all explicitly not run.

### Result Files

`results/EXP-ENCODER-ANCHOR01/{command.txt, config.json, sanity_summary.json,
notes.md, lambda_selection_probe.py, train_summary.json, stage2_eval.json,
ETTh1_H96_E2_stage2.json, representation_probe.csv,
representation_diagnostics.json, t1_diagnostic.json, t2_continuation.json,
train_vs_val_test.json, ETTh1_H96_E2_{test,train}_summary.json,
continuation_diag_ETTh1_H96_E2_{test,train}_dense_first.csv,
ETTh1_H96_E2_{test,train}_dense_first_summary.json, metrics.csv, REPORT.md,
logs/, working_tree.diff, checkpoint_fingerprints.txt, env.txt,
git_commit.txt}`.

### Results

Training: `best_epoch=1` (`val_overlap@10=0.0094`, best of C0/E1/E2 on
this proxy), early-stopped at epoch 6, `wall_clock=8609.7s`,
`peak_gpu_mem=516MiB`, no OOM.

**Collapse prevented**: `embedding_effective_rank` stayed 15.70-16.56
across all 6 epochs (vs B0's 17.48, vs E1's own collapse to 2.96 at
epoch1 -> 1.49 at epoch6). `cos_to_b0` stayed 0.9965-0.9977 (small,
roughly stable displacement, no runaway drift unlike E1's monotonic
0.650->0.559 decline).

| Metric | C0 (frozen) | E1 (trainable, no anchor) | E2 (trainable + anchor) |
|---|---:|---:|---:|
| Stage-2 MSE | 0.39526 | 0.40668 | 0.40418 |
| gap_recovery | -0.2626 | -0.3703 | -0.3594 |
| HardAggregate@10 (seq) | 0.5514 | 0.5466 | 0.6768 |
| t1 Spearman top-1% (test) | 0.2128 | 0.1455 | 0.2442 |
| t1 oracle rank median (test) | 99 | 210 | 86 |
| t1 Top-50 containment (test) | 30.5% | 20.5% | 31.5% |
| t2 hurt_frac (test) | 0.232 | 0.194 | 0.216 |
| t2 selected true rank median (test) | 1934.0 | 539.5 | 1929.5 |
| t2 continuation_regret (test) | 0.3498 | 0.3987 | 0.3177 |

E2 is the BEST of all three arms on t1 (every metric) and most of t2
(hurt_frac, oracle rank, Spearman, continuation_regret), but WORSE than C0
on Stage-2 (though slightly better than E1) and WORST OF ALL THREE arms on
HardAggregateMSE.

### Sanity Checks

`pytest tests/`: 536 passed (8 new in `tests/test_exp_encoder_anchor01.py`
plus 4+7 for D1/T1 prepared in parallel -- see their own entries), same 2
pre-existing failures, no regression.

### Implementation Notes

New: `scripts/train_encoder_anchor01.py`, `tests/test_exp_encoder_anchor01.py`.
No changes needed to `eval_margutil01_stage2.py`/`eval_firstanchor_diag.py`/
`eval_continuation_diag.py` (E2's checkpoint carries its own adapted
encoder in `model_state_dict`, loaded automatically, same as E1).

### Conclusion

**Outcome B (collapse prevented, downstream Stage-2 does not improve),
with an unresolved internal metric disagreement.** Not Outcome A (Stage-2
worse than C0). Not Outcome C (collapse plainly did not persist). Not a
clean Outcome D either (the encoder moved a real amount and several
metrics changed substantially, not "flat"). The Stage-2/HardAggregate vs.
t1/t2 disagreement (E2 best of all three on top-tail ranking, worst of all
three on HardAggregate, still worse than C0 on the primary Stage-2 metric)
is reported as-is rather than resolved. **The STRONG version of "collapse
fully explains E1's failure" is refuted** (preventing it only closed a
small fraction of the Stage-2 gap and made HardAggregate worse); a WEAK
version ("collapse was a real, partial contributing factor, alongside
something else visible as a per-step-ranking vs. free-running-aggregate
disconnect") is left open. Full breakdown:
`results/EXP-ENCODER-ANCHOR01/REPORT.md`. Per the pre-registered stopping
rule, no larger anchor-strength sweep or different anchor form was
started; per the user's explicit pre-approval, EXP-ORACLE-CHOICE01 (D1)
proceeds next.

## EXP-ORACLE-CHOICE01 — R2 (SmoothL1+pairwise) vs. Oracle-Choice Cross-Entropy (COMPLETE: ETTh1 H96 only)

**Date:** 2026-09-08
**Status:** completed (1 cell, C0 vs D1). Per user's explicit pre-approval,
EXP-TEACHER-FORCING-DIAG01 (diagnostic, no training) and
EXP-ONPOLICY-PREFIX01 (T1) proceed next in sequence.

### Research Question

Does R2's SmoothL1+pairwise surrogate insufficiently target the Set
Oracle's actual greedy next choice, such that training the model to
directly predict `i_t* = argmax_i u_i^(t)` improves free-running selection
and Stage-2?

### Configuration

New script `scripts/train_oracle_choice01.py`. Loss:
`L_choice^(t) = -log p_{i_t*}^(t)`, masked full-memory softmax over valid
candidates only, `tau=0.1` (base checkpoint's own `tau_topk`, no sweep).
SmoothL1 and pairwise terms REMOVED entirely -- Oracle-Choice CE only, per
the spec's explicit prohibition on mixing auxiliary losses. Frozen B0
encoder unchanged (same checkpoint as C0), same `SetConditioner`/
`EmptySetToken`/`UtilityHead`/optimizer/LR/epochs/patience/checkpoint-
selection criterion (`val_overlap@10`). No memory-safe streaming needed
(encoder frozen, same simple single-backward design as R0/R1/R2). Full
commands: `results/EXP-ORACLE-CHOICE01/command.txt`.

### Mandatory pre-training diagnostic

`scripts/diag_oracle_margin01.py` (`oracle_margin_diagnostic.json`), run
BEFORE training, interpretive only: the Oracle's true top-1-vs-top-2
utility margin collapses sharply as the selected set grows --
`near_tie_frac` rises from 8.5% at t=1 to 99.0% at t=10, `rel_margin_mean`
falls from 0.136 to 0.0008. Per spec, this is recorded as a LIMITATION on
one-hot Oracle-Choice CE (later steps' "true" label is often an arbitrary
tie-break), not acted on -- no automatic switch to multi-positive/soft/KL
targets.

### Mandatory sanity checks

`tests/test_exp_oracle_choice01.py`, 4/4 PASSED: full-memory masked CE
includes every valid candidate with invalid probability exactly 0; oracle
label equals `argmax` of the true dense utility exactly; a synthetic
small-N positive control fits >90% top-1 accuracy; `dense_utility`
chunking does not change the resulting logits/loss.

### Changed Variable

Loss only (SmoothL1+pairwise -> Oracle-Choice CE). Encoder, scorer,
SetConditioner, optimizer/LR, checkpoint-selection criterion all identical
to C0.

### Dataset / Horizon

ETTh1 H96 only, seed 0. Weather, H720, 3 seeds, temperature sweep,
multi-positive/soft/KL target change, combination with on-policy prefix,
encoder unfreezing all explicitly not run.

### Result Files

`results/EXP-ORACLE-CHOICE01/{command.txt, config.json, sanity_summary.json,
notes.md, oracle_margin_diagnostic.json, train_summary.json, stage2_eval.json,
ETTh1_H96_D1_stage2.json, t1_diagnostic.json, t2_continuation.json,
ETTh1_H96_D1_{test,train}_summary.json,
continuation_diag_ETTh1_H96_D1_{test,train}_dense_first.csv,
ETTh1_H96_D1_{test,train}_dense_first_summary.json,
oracle_choice_diag_{test,train}.json, metrics.csv, REPORT.md, logs/,
working_tree.diff, checkpoint_fingerprints.txt, env.txt, git_commit.txt}`.

### Results

Training: `best_epoch=10` (the max configured epoch, NEVER early-stopped
-- `val_overlap@10` improved every single epoch, 0.0119->0.0146, unlike
every other arm this session which peaked at epoch 1). `wall_clock=2099.2s`
(~35 min), `peak_gpu_mem=593MiB`, no OOM.

Teacher-forced exact next-choice accuracy stayed low throughout (mean
top1_acc=0.36% across K=10 steps, t1=1.34%, consistent with the near-tie
diagnostic above) -- but every downstream/decision-relevant metric
improved:

| Metric | C0 (R2) | D1 (Oracle-Choice CE) | Delta |
|---|---:|---:|---|
| Stage-2 MSE | 0.39526 | **0.38916** | **better** (FIRST arm this session to beat C0) |
| gap_recovery | -0.2626 | -0.2317 | better |
| HardAggregate@10 (seq) | 0.5514 | 0.5480 | better |
| t1 Spearman top-1% (test) | 0.2128 | 0.2832 | better |
| t1 oracle rank median (test) | 99 | 47 | better |
| t1 Top-50 containment (test) | 30.5% | 52.0% | better |
| t2 hurt_frac (test) | 0.232 | 0.122 | better |
| t2 selected true rank median (test) | 1934.0 | 88.0 | much better |
| t2 Spearman top-1% (test) | 0.1404 | 0.3294 | much better |
| t2 continuation_regret (test) | 0.3498 | 0.1964 | much better |

Every single decision-relevant metric measured improves for D1 over C0 --
the first arm this entire session where every intermediate diagnostic and
the primary Stage-2 metric point the same direction.

### Sanity Checks

`pytest tests/`: 536 passed (4 new in `tests/test_exp_oracle_choice01.py`),
same 2 pre-existing failures, no regression.

### Implementation Notes

New: `scripts/train_oracle_choice01.py`, `scripts/diag_oracle_margin01.py`,
`scripts/eval_oracle_choice_diag.py` (per-step teacher-forced top1/5/10
accuracy, predicted rank, and step regret on the best checkpoint, reusing
`oracle_choice_step_loss` directly), `tests/test_exp_oracle_choice01.py`.
No changes needed to `eval_margutil01_stage2.py`/`eval_firstanchor_diag.py`/
`eval_continuation_diag.py` (D1's checkpoint uses the default `cosine`
`UtilityHead`, already handled).

### Conclusion

**Outcome L-A.** R2's SmoothL1+pairwise surrogate was insufficiently
aligned with the Set Oracle's actual greedy decision; a more direct
choice-prediction objective (Oracle-Choice CE) recovers real, consistent
Stage-2 and downstream-decision value, DESPITE a demonstrable and
explained limitation (near-tie labels at later steps making exact
teacher-forced accuracy low). This is evidence that loss/surrogate
mismatch was a real, and the largest-yet-found, contributing factor to
this project's set-aware retrieval underperforming B0. D1 still does not
close the full gap to B0 (0.38916 vs 0.37312). Full breakdown:
`results/EXP-ORACLE-CHOICE01/REPORT.md`. Per the user's explicit
pre-approval, EXP-TEACHER-FORCING-DIAG01 (diagnostic, no training)
proceeds next, followed by EXP-ONPOLICY-PREFIX01 (T1).

## EXP-TEACHER-FORCING-DIAG01 — Oracle-prefix vs. free-running evaluation of the SAME C0/R2 checkpoint (COMPLETE: ETTh1 H96, diagnostic only, no training)

**Date:** 2026-09-08
**Status:** completed (diagnostic only). Per user's explicit pre-approval,
EXP-ONPOLICY-PREFIX01 (T1) proceeds next.

### Research Question

Before committing to on-policy retraining, does the EXISTING C0/R2
checkpoint reproduce the Set Oracle's actual next choice well when GIVEN
the correct oracle prefix, and how does that compare to its own
free-running behavior?

### Method

`scripts/diag_teacher_forcing01.py` (new, no training). Two evaluation
modes on the SAME frozen C0/R2 checkpoint: (A) Oracle-prefix — state and
target both from the oracle's own prefix at every step; (B) Free-running —
the model's own argmax prefix. Channel 0 only. Stage-2 MSE (within this
single-channel scope) accumulated over the FULL test split; detailed
per-step rank/NDCG/regret/divergence diagnostics subsampled to 200 valid
queries. A bug in the first version (early loader break, producing a
Stage-2 MSE on a tiny non-representative subset) was caught and fixed
before trusting the result — re-run on the full split.

**Scope limitation (documented)**: this diagnostic forces only channel 0
into Stage-2 for both arms, unlike the project's official all-7-channel
Stage-2 evaluations — its Stage-2 numbers are NOT comparable in absolute
terms to C0's canonical 0.39526; only the relative oracle-prefix-vs-
free-running comparison (same scope both arms) is valid.

### Results

| Metric | Oracle-prefix | Free-running |
|---|---:|---:|
| Oracle next-choice rank mean | 1148.8 | — |
| Top-1 accuracy | 0.25% | — |
| Top-10 containment | 3.25% | — |
| NDCG@10 | 0.768 | — |
| Spearman (true top-1%) | 0.279 | — |
| mean step regret | 0.254 | 0.376 |
| mean divergence rate | — | 99.5% |
| first divergence step (mean) | — | ~0 |
| mean prefix overlap | — | 2.7% |
| Stage-2 MSE (single-channel scope) | 0.322 | 0.377 |

Per-step: t1 rank_mean=120.3/NDCG@10=0.965; t2 rank_mean=488.5/NDCG@10=0.756;
t10 rank_mean=2299.2/NDCG@10=0.712 -- rank degrades sharply with t while
NDCG stays comparatively high throughout.

### Case classification

**Mixed -- elements of Case A, B, and C all present, not forced into one.**
Case C (NDCG-high, exact-rank-poor) strongly and clearly present at every
step, reproducing EXP-FIRSTANCHOR-DIAG's original finding on the SAME
checkpoint family. Case A partial support: oracle-prefix performance is
not "good" in absolute terms (Top-1 acc ~0.25%, rank balloons past 1000 by
mid-sequence even with the correct prefix). Case B also has real support:
within this diagnostic's own paired comparison, oracle-prefix beats
free-running on every comparable metric, and free-running diverges from
the oracle's own trajectory almost immediately (99.5% divergence,
overlap collapsing to 2.7%). Net reading: teacher-forcing/exposure-bias
mismatch is a real, independently-evidenced contributor, operating
ALONGSIDE a surrogate/decision-resolution weakness (Case C) that is at
least as severe -- consistent with, and complementary to,
EXP-ORACLE-CHOICE01's finding that a more direct choice-aligned loss
recovers real value.

### Result Files

`results/EXP-TEACHER-FORCING-DIAG01/{REPORT.md, metrics.csv,
oracle_prefix_summary_test.json, free_running_summary_test.json,
full_result_test.json, step_metrics_test.csv, command.txt, env.txt,
checkpoint_fingerprints.txt, working_tree.diff, logs/, git_commit.txt}`.

### Implementation Notes

New: `scripts/diag_teacher_forcing01.py` (reuses `load_trained_selector`/
`encode`/`a_weighted_prefix`/`hard_agg`/`_ndcg_at_k` from
`eval_firstanchor_diag.py`, `load_stage2` from `utils.retrieval_diagnostics`,
`dense_utility`/`candidate_weights` from `utils.dense_utility` -- no
reimplementation of existing diagnostic primitives). No dedicated pytest
file (matches this project's convention for other no-training, read-only
diagnostic scripts like `eval_firstanchor_diag.py` itself); correctness
verified via the full pytest suite (no regression) plus the internal
consistency fix (early-break bug caught and corrected before trusting
results).

### Conclusion

Diagnostic complete; per the user's explicit pre-approval,
`EXP-ONPOLICY-PREFIX01` (T1) proceeds next as the final step of the
pre-approved 3-step sequence (D1 -> this diagnostic -> T1), after which no
further experiment is auto-started.

## EXP-ONPOLICY-PREFIX01 — R2 loss unchanged, Oracle prefix vs. on-policy (model-generated) prefix (COMPLETE: ETTh1 H96 only)

**Date:** 2026-09-08
**Status:** completed (1 cell, C0 vs T1). This was the LAST step of the
user's pre-approved 3-experiment sequence (D1 -> teacher-forcing
diagnostic -> T1). Sequence now COMPLETE; no further experiment auto-started.

### Research Question

Is R2's free-running failure caused, at least in part, by the train-time
Oracle-prefix / inference-time model-prefix state-distribution mismatch
(teacher-forcing/exposure bias), independent of the loss formula itself?

### Configuration

New script `scripts/train_onpolicy_prefix01.py`. Loss formula UNCHANGED
from C0/R2 (SmoothL1 + `lambda_rank=1.0` * pairwise) -- deliberately NOT
combined with EXP-ORACLE-CHOICE01's Oracle-Choice CE, per the
pre-registered single-variable design. Only the prefix source changes:
at every teacher-forced step, both the `SetConditioner` state AND the
dense-utility TARGET are built from the model's own on-policy pick
history (`argmax` under `torch.no_grad()`/detached, never differentiated
through), not the Oracle's sequence. Frozen B0 encoder unchanged (same
checkpoint as C0). Full commands: `results/EXP-ONPOLICY-PREFIX01/command.txt`.

### Mandatory sanity checks

`tests/test_exp_onpolicy_prefix01.py`, 7/7 PASSED (t=0 state identity,
selected index enters prefix, candidate removed from next valid set,
target recomputed from the model's own prefix, no gradient through
argmax, R2 gradient flows normally, full-memory semantics preserved).

### Changed Variable

Prefix source only (Oracle -> on-policy). Loss, encoder, scorer,
SetConditioner, optimizer/LR, checkpoint-selection criterion all identical
to C0.

### Dataset / Horizon

ETTh1 H96 only, seed 0. Weather, H720, 3 seeds, scheduled sampling,
DAgger, curriculum, combination with Oracle-Choice CE all explicitly not run.

### Result Files

`results/EXP-ONPOLICY-PREFIX01/{command.txt, config.json, sanity_summary.json,
notes.md, train_summary.json, stage2_eval.json, ETTh1_H96_T1_stage2.json,
t1_diagnostic.json, t2_continuation.json,
ETTh1_H96_T1_{test,train}_summary.json,
continuation_diag_ETTh1_H96_T1_{test,train}_dense_first.csv,
ETTh1_H96_T1_{test,train}_dense_first_summary.json, metrics.csv, REPORT.md,
logs/, working_tree.diff, checkpoint_fingerprints.txt, env.txt,
git_commit.txt}`.

### Results

Training: `best_epoch=4` (`val_overlap@10=0.0091`), early-stopped at
epoch 9. `wall_clock=1932.8s` (~32min), `peak_gpu_mem=729MiB`, no OOM.
Training-time state diagnostics showed `prefix_overlap` with the Oracle's
own sequence staying near-zero throughout (0.004-0.009) with no improving
trend, and per-step regret against the model's own best continuation
rising slightly (0.755->0.852) -- looked concerning in isolation but was
NOT predictive of the downstream result (see Conclusion).

| Metric | C0 (Oracle prefix) | T1 (on-policy prefix) | Delta |
|---|---:|---:|---|
| Stage-2 MSE | 0.39526 | **0.37455** | **much better -- best of the whole session** |
| gap_recovery | -0.2626 | **-0.0100** | **much better -- best of the whole session** |
| HardAggregate@10 (seq) | 0.5514 | **0.4096** | **much better -- best of the whole session** |
| t1 Spearman top-1% (test) | 0.2128 | 0.1650 | worse |
| t1 oracle rank median (test) | 99 | 144 | worse |
| t2 hurt_frac (test) | 0.232 | 0.264 | worse |
| t2 selected true rank median (test) | 1934.0 | 135.5 | much better |
| t2 Spearman top-1% (test) | 0.1404 | 0.4086 | much better |
| t2 continuation_regret (test) | 0.3498 | 0.2303 | better |

Mixed on t1/t2 Oracle-agreement metrics (several worse), but dramatically
and unambiguously better on Stage-2, HardAggregate, gap_recovery, and t2
rank/Spearman/regret -- the metrics closest to actual realized sequential
decision quality. **T1's Stage-2 MSE (0.37455) is within 0.00143 of B0's
own unforced floor (0.37312) and beats EXP-ORACLE-CHOICE01's own D1
result (0.38916), the best prior arm this session.**

### Sanity Checks

`pytest tests/`: 545 passed (7 new in `tests/test_exp_onpolicy_prefix01.py`,
plus Track B's 9 new sanity tests run in parallel -- see that experiment's
own entry), same 2 pre-existing failures, no regression.

### Implementation Notes

New: `scripts/train_onpolicy_prefix01.py` (`run_sequence_onpolicy`
generalises `run_sequence_dense`'s free-running branch to ALSO compute a
fresh dense-utility target from the model's own growing prefix at every
step, reusing `dense_utility`/`candidate_weights`/`step_losses` from
`utils.dense_utility`/`scripts.train_toptail_rank01` unmodified),
`tests/test_exp_onpolicy_prefix01.py`. No memory-safe streaming needed
(encoder stays frozen, same simple single-backward design as R0/R1/R2/D1).

### Conclusion

**Outcome T-A.** Per the pre-registered priority order (Stage-2 >
HardAggregate/gap_recovery > realized regret > Oracle choice/rank/top-tail
> training proxy), the three HIGHEST-priority criteria are unambiguous,
large wins; only the LOWEST-priority criterion (agreement with the
Oracle's specific choices/ranking) is worse. **Train-time Oracle-prefix /
inference-time model-prefix state-distribution mismatch was a real, and
by Stage-2/HardAggregate magnitude the LARGEST-YET-FOUND, contributing
factor to this project's set-aware retrieval underperforming B0** -- this
is the strongest positive result the entire session has produced, on the
metric this project has always treated as primary. The t1/t2
Oracle-agreement regression is best read as a natural consequence of T1
never being trained to match the Oracle's specific choices at all --
consistent with, not contradicting, the Stage-2 result. Full breakdown:
`results/EXP-ONPOLICY-PREFIX01/REPORT.md`. **This completes the user's
pre-approved 3-experiment sequence (D1 -> teacher-forcing diagnostic ->
T1). No further experiment in this sequence is started; results go to
independent review next**, per the standing project workflow.

## EXP-CORRECTION-ORACLE-DIAG01 (Track B) — Future Set Oracle vs. Correction Set Oracle (COMPLETE: diagnostic only, no training)

**Date:** 2026-09-08
**Status:** completed. Ran in parallel with Track A
(EXP-ENCODER-ANCHOR01 -> EXP-ORACLE-CHOICE01 -> EXP-TEACHER-FORCING-DIAG01
-> EXP-ONPOLICY-PREFIX01); touches none of Track A's code, checkpoints, or
results.

### Research Question

Stage-2's actual fusion is `Y_final = B_q + gamma * Y_ret` (a residual
correction, not a from-scratch reconstruction). Does retrieving candidates
whose OWN frozen-B0 forecast ERROR (`r_i = Y_i - B_i`) resembles the
query's own error better match this downstream semantics than the
existing Future Set Oracle (which retrieves by realized-future similarity)?
Is the Correction target more learnable from a past-only score?

### Method

New script `scripts/diag_correction_oracle01.py`. `B_i = base_head(X_i)`
computed for the full candidate bank ONCE (candidate's own past only,
frozen B0, no_grad); `r_i = Y_i - B_i` in the candidate's OWN frame (no
query-offset transplant, unlike the existing Future Oracle's `_memory_value`
convention -- a residual is already scale-appropriate as an error).
`select_greedy_weighted_set` (existing, unmodified) called twice: once
with raw futures (Future Oracle, faithful reproduction) and once with
residuals (Correction Oracle) -- the function is agnostic to what the
value tensor means. `dense_utility` (existing, unmodified) used for
per-step rank/NDCG/margin/epsilon-optimal diagnostics, cross-checked
online against `select_greedy_weighted_set`'s own picks (0% mismatch at
t=1-4, <=4.5% at t=10). Same B0 checkpoint, same past-only reference score
for weighting, self-only, channel 0.

**Scope limitation**: channel-0-only diagnostic; its MSE numbers are NOT
comparable in absolute terms to Track A's multi-channel Stage-2 numbers.

### Mandatory sanity checks

`tests/test_exp_correction_oracle01.py`, 9/9 PASSED (residual exactness,
`MSE(B_q+C,Y_q)==MSE(C,r_q)` equivalence -- also asserted at runtime on
every real batch, empty correction reduces to B0 exactly, K=1 brute-force
match, small-N exhaustive stepwise match, no invalid/duplicate selection,
chunking-invariant candidate count, no B0 gradient, residual independence
from the query).

Two implementation bugs caught and fixed before trusting results: a
dict-indexing bug (per-step rows looked up on the wrong dict), and a
shared-selected-mask bug (Future and Correction Oracles pick different
candidates per step and need independent "already selected" trajectories,
not one shared mask).

### Result Files

`results/EXP-CORRECTION-ORACLE-DIAG01/{REPORT.md, summary.json,
future_oracle_metrics.csv, correction_oracle_metrics.csv,
utility_margin_diagnostics.csv, epsilon_optimal_metrics.csv,
oracle_overlap.csv, per_step_metrics.csv, sanity_checks.txt, command.txt,
env.txt, checkpoint_fingerprints.txt, working_tree.diff, logs/, git_commit.txt}`.

### Results

| Metric | B0 | Future Set Oracle | Correction Set Oracle |
|---|---:|---:|---:|
| Final MSE (channel-0 scope) | 1.4496 | 0.2109 | 0.1818 |
| Gain vs B0 | -- | 1.2387 | 1.2678 |
| Mean Oracle rank (past-only score) | -- | 757.8 | 632.1 |
| NDCG@10 | -- | 0.524 | 0.513 |
| Top1-Top2 margin (abs) | -- | 0.00838 | 0.00573 |
| Set overlap between the two oracles | -- | 22-28% across all 10 steps |

Correction Oracle: modestly lower downstream MSE, genuinely different
candidate set from Future Oracle, but consistently HIGHER near-tie
fraction and smaller margin at every step (worse separability), and mixed
(mostly worse) NDCG/containment despite a somewhat better mean rank.
Epsilon-optimality is very low for BOTH oracles at every threshold tested
-- the past-only reference score rarely lands near either oracle's true
best pick.

### Sanity Checks

`pytest tests/`: 545 passed (9 new in `tests/test_exp_correction_oracle01.py`),
same 2 pre-existing failures, no regression.

### Conclusion

**Outcome B-B.** Real but modest downstream headroom gain from the
Correction target, alongside consistently mixed-to-worse learnability
signals (higher near-tie fraction, smaller margin) than the existing
Future Oracle. Not strong enough evidence on its own for the spec's clean
Outcome B-A ("proceed to `EXP-CORRECTION-SELECTOR01`" recommendation), nor
does it match B-C (no gain) or B-D (near-identical oracles). Read as weak,
mixed evidence that a future VALUE-AWARE selector (adding `v_i=g(r_i)` as
new observable candidate information, not just changing the retrieval
target) MIGHT be worth considering -- explicitly not started automatically.
Full breakdown: `results/EXP-CORRECTION-ORACLE-DIAG01/REPORT.md`. Per the
spec's STOP rule: no selector, encoder, residual encoder, or Stage-2 gate
was trained. `EXP-CORRECTION-SELECTOR01`'s execution is not decided here.

## EXP-ONPOLICY-CHOICE01 — On-policy prefix + Oracle-Choice CE, combined (COMPLETE: ETTh1 H96 only)

**Date:** 2026-09-08
**Status:** completed. Fills the last cell of Track A's 2x2 (prefix x
loss) table. Ran in parallel with Track B1/B2. Per the user's explicit
STOP rule, no further experiment auto-started.

### Research Question

D1 (Oracle-prefix + Oracle-Choice CE) and T1 (on-policy prefix + R2 loss)
were both independent positive single-variable interventions. Does
combining them (on-policy prefix + Oracle-Choice CE, learning the greedy
Oracle best action AT THE STATES THE MODEL ACTUALLY VISITS) beat T1 alone
and cross B0?

### Configuration

New script `scripts/train_onpolicy_choice01.py`. `run_sequence_onpolicy_choice`
mirrors `train_onpolicy_prefix01.run_sequence_onpolicy`'s on-policy prefix
construction exactly, substituting `oracle_choice_step_loss` (from
`scripts/train_oracle_choice01.py`, unmodified) for R2's SmoothL1+pairwise
at the single loss-computation site -- the target is recomputed from the
CURRENT on-policy state (`u_i^(t) = -A_weighted(S_hat_{t-1}+{i})`), never
the fixed Oracle trajectory's own precomputed target. Frozen B0 encoder,
tau_choice = base checkpoint's own tau_topk (no sweep). 11/11 mandatory
sanity checks passed before the GPU run. Full commands:
`results/EXP-ONPOLICY-CHOICE01/command.txt`.

### Result Files

`results/EXP-ONPOLICY-CHOICE01/{command.txt, config.json, sanity_summary.json,
train_summary.json, stage2_eval.json, ETTh1_H96_OPC1_stage2.json,
t1_diagnostic.json, t2_continuation.json, per_step_metrics.csv,
utility_margin_diagnostics.csv, metrics.csv, REPORT.md, logs/,
working_tree.diff, checkpoint_fingerprints.txt, env.txt, git_commit.txt}`.

### Results

Training: `best_epoch=4`, early-stopped at epoch 9, `wall_clock=2085.1s`,
`peak_gpu_mem=579MiB`, no OOM. `val_top1_acc` reached 5.2% (higher than
D1's own oracle-prefix ~1.3% at t=1 / ~0.36% mean) -- the on-policy state's
utility landscape gives a sharper choice signal than the fixed Oracle
trajectory's states.

| Arm | Stage-2 MSE | gap_recovery | HardAggregate | t2 selected rank median | t2 regret |
|---|---:|---:|---:|---:|---:|
| B0 | 0.37312 | -- | -- | -- | -- |
| C0 | 0.39526 | -0.2626 | 0.5514 | 1934.0 | 0.3498 |
| D1 | 0.38916 | -0.2317 | 0.5480 | 88.0 | 0.1964 |
| T1 | 0.37455 | -0.0100 | 0.4096 | 135.5 | 0.2303 |
| **OPC1** | **0.37340** | **-0.0091** | 0.4110 | **47.0** | **0.1422** |

OPC1 beats T1 on Stage-2, gap_recovery, t2 rank, and t2 regret; is
essentially tied with T1 on HardAggregate; t1/t2 metrics are the best or
near-best of every arm this session.

### Sanity Checks

`pytest tests/`: 563 passed (11 new in `tests/test_exp_onpolicy_choice01.py`,
plus Track B1/B2's own new tests -- see their entries), same 2 pre-existing
failures, no regression.

### Conclusion

**Case B**: `0.37312 <= 0.37340 < 0.37455`. Beats T1, does not beat B0
(0.00028 short). D1's and T1's gains are combinable but SUB-additive
(0.02186 combined Stage-2 gain vs. 0.02071+0.00610=0.02681 naive sum of
individual gains) -- real, diminishing-but-positive returns from stacking
both fixes, not a failure to combine. H2 (hard Choice CE brittle on-policy)
is not supported -- training was stable throughout, no near-tie-driven
degradation observed. This is the closest ANY arm has come to B0 this
entire session. Full breakdown: `results/EXP-ONPOLICY-CHOICE01/REPORT.md`.

---

## EXP-CORRECTION-ORACLE-DIAG02 (Track B1) — multivariate Correction Set Oracle diagnostic (COMPLETE: diagnostic only, no training)

**Date:** 2026-09-08
**Status:** completed. Extends EXP-CORRECTION-ORACLE-DIAG01's channel-0
diagnostic to all 7 ETTh1 channels. Ran in parallel with Track A and B2.

### Configuration

`diag_correction_oracle01.evaluate()` gained an optional `channel`
parameter (defaults to `channels[0]`, preserving the original single-
channel behaviour exactly); new driver `scripts/diag_correction_oracle02.py`
calls it once per channel and aggregates. Reuses
`tests/test_exp_correction_oracle01.py`'s 9/9 sanity checks unmodified.

### Results

| Metric | Future Oracle (mean) | Correction Oracle (mean) |
|---|---:|---:|
| Gain vs B0 | 0.5437 | **0.5514** |
| Mean rank | **1506.9** | 2312.9 |
| NDCG@10 | 0.656 | **0.699** |
| Margin abs | **0.00404** | 0.00279 |
| Channels where Correction wins on MSE | -- | **4 / 7** |

Correction Oracle wins on the cross-channel mean gain and on a majority
(4/7) of individual channels, but loses on 3 channels (3, 5, 6) -- all
three among the easiest-to-forecast (lowest B0 MSE) channels. NDCG now
favors Correction on the cross-channel mean (a REVERSAL from the
channel-0-only result), while mean rank/margin still favor Future --
learnability signals disagree with each other, not just with the exact-
choice metrics as in the channel-0 diagnostic.

### Sanity Checks

`pytest tests/`: 563 passed (0 new -- reuses `test_exp_correction_oracle01.py`
unmodified), same 2 pre-existing failures, no regression.

### Conclusion

Nuances, without overturning, `EXP-CORRECTION-ORACLE-DIAG01`'s Outcome
B-B: the modest downstream advantage generalizes on average and on a
majority of channels but is not universal, concentrated among
harder-to-forecast channels; the negative learnability signal is now
mixed (NDCG reverses direction) rather than uniformly negative. Full
breakdown: `results/EXP-CORRECTION-ORACLE-DIAG02/REPORT.md`.

---

## EXP-CORRECTION-STAGE2-SEMANTICS01 (Track B2) — correction-aligned Stage-2 fusion, fixed reference retrieval (COMPLETE: ETTh1 H96 only)

**Date:** 2026-09-08
**Status:** completed. Ran in parallel with Track A and B1.

### Research Question

Given the SAME reference retrieval (Top-K + alpha, from B0's own existing
production score -- no new selector), does fusing the historical
CORRECTION aggregate beat the existing FUTURE aggregate fusion Stage-2
already uses?

### Configuration

New script `scripts/train_correction_stage2_semantics01.py`. No new
retrieval selector trained; B0/base_forecast/retrieval score stay frozen.
`layers/retrieval_gate.py::RetrievalGate` (existing, unmodified) reused
for both C0 (`fixed_lambda=1.0`, no training) and C1 (`fixed_lambda=-1.0`,
newly trained per channel on TRAIN, selected on VAL). `B_q`/`C_ret`/`Y_q`
precomputed once per split (frozen, gate-independent) for fast gate
training. 7/7 mandatory sanity checks passed, including a positive-control
test confirming the gate training loop can recover a known gamma=0.5 on
synthetic data.

A bug (F0 MSE accumulated from the unsliced multivariate output, giving
the same 0.37312-scale number for every channel) was caught and fixed
before trusting results.

### Results

| Arm | Aggregate MSE |
|---|---:|
| F0 (existing) | **0.37312** |
| C1 (learned gate) | 0.38337 |
| C0 (fixed gamma=1) | 0.39056 |

F0 beats both C0 and C1 on 6 of 7 individual channels and on the
cross-channel aggregate. C1 consistently beats C0 on every channel
(adaptive correction strength helps over a naive full correction), and
the gate's gamma values (0.263-0.983) show real per-channel variation, not
a degenerate collapse -- confirming the gate training itself worked. Only
channel 6 (lowest B0 MSE, easiest to forecast) shows C1 beating F0.

### Sanity Checks

`pytest tests/`: 563 passed (7 new in
`tests/test_exp_correction_stage2_semantics01.py`), same 2 pre-existing
failures, no regression.

### Conclusion

**Outcome B2-D** on the aggregate and 6/7 channels. Per the pre-registered
interpretation rule, this does NOT reject the Correction retrieval
hypothesis overall -- read together with Track B1 (which found real
oracle-level headroom for the Correction VALUE using the TRUE query
future), the likely explanation is that reusing the EXISTING
future-oriented Top-K (rather than a retrieval process aligned with the
correction objective) hands the correction fusion the wrong candidates,
independent of whether the correction value itself has merit. Full
breakdown: `results/EXP-CORRECTION-STAGE2-SEMANTICS01/REPORT.md`. Per the
user's explicit STOP rule, no Correction Selector training or further
experiment was auto-started.
