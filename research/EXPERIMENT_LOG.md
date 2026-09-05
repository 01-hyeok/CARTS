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

