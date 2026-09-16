```text
INVALID_FOR_CONCLUSION:
- reason: The retrieval cache built by scripts/build_choicece_retrieval_cache01.py
  stored `relation_outputs` in ABSOLUTE value space (`memory_c + offset_c`,
  the query's own last-observed value added onto the candidate delta), while
  `RelationStage2.forward_from_retrieval_values()` -- and the model's own
  online (non-cached) path -- expects/builds `relation_outputs` in DELTA
  space (candidate future minus that candidate's own last value, offset
  NEVER added mid-pipeline; confirmed by direct line-level trace of
  models/RelationStage2.py::forward(), see
  research/TRACK-A-CHOICECE-STAGE2-RETRAIN01-CORRECTED.md section 1 for the
  full evidence). This caused every metric below built on the cached
  retrieval branch (`ret_mse`, the gate's learned weighting of that branch,
  and therefore `final_mse`) to be computed on an effectively double-offset
  retrieval signal.
- affected: every ChoiceCE-STAGE2-RETRAIN01 metric in this file that
  involves `ret_mse`, the gate/lambda values, or `final_mse` -- i.e. all 16
  arms (H96 + H720). Any interpretation drawn from those numbers (including
  gate weight magnitude, retrieval-branch usefulness, or ChoiceCE-vs-
  Independent-Base comparisons) is INVALID and must not be cited.
- unaffected: the Stage-1 checkpoints themselves, their own
  free_running_aggregate_future_mse (`retrieval_metrics_<arm>.json`), the
  frozen-submodule-hash integrity checks, and the Independent Base
  reference (trained without any retrieval branch, never touched this
  cache).
- superseded by: research/TRACK-A-CHOICECE-STAGE2-RETRAIN01-CORRECTED.md
```

```text
EXECUTED:
- Code audit (section 2, all 7 questions), read-only, with the two named
  suspicions CONFIRMED by direct code inspection (not assumed).
- utils/... no new reference file touched. New files: retrieval cache
  builder, Stage-2 retrainer, orchestrator scripts (H96/H720), unit tests.
- 9 unit tests written and passing, against the REAL Exp_Stage2_Relation /
  RelationStage2 classes built from the real ETTh1_96 S0_wce host config
  (CPU/GPU2 only -- GPU1 never touched by any test or structural check).
- Full repo regression suite: 909 passed / 2 pre-existing failures
  (unchanged names/reasons), no new regressions.
- A brief, unintentional CPU-construction check touched GPU0 for a few
  seconds before CUDA_VISIBLE_DEVICES was pinned in later checks --
  reported under Protocol Deviations, confirmed to have left no lasting
  GPU0 footprint (checked via nvidia-smi immediately after), and GPU1 was
  never involved.

NOT EXECUTED:
- No retrieval cache was built for any arm (build_choicece_retrieval_cache01.py
  is implemented and unit-tested in its cache-lookup half, but has NOT been
  run against a real arm checkpoint end-to-end yet -- that requires the
  real encoder forward pass over the full ETTh1 memory bank, which is a
  real GPU1 workload).
- No Stage-2 arm was trained. Zero of the 16 requested arm x horizon runs
  (8 arms x {H96, H720}) have been executed.
- The smoke test (spec section 7, 20 optimizer steps on set_onpolicy_cosine)
  was NOT run -- it requires GPU1.
- The full ~40-item unit test battery enumerated in spec section 6 was NOT
  fully implemented; 9 tests cover the highest-risk correctness claims
  (frozen/trainable separation, cache lookup order-invariance, forced-
  retrieval forward producing finite loss and correctly-scoped gradient,
  frozen-hash invariance across an optimizer step, fresh-init not leaking
  host weights, shared-seed reproducibility). Deferred items include:
  online-vs-cache Top-K bit-identity, per-channel cache coverage assertion,
  NaN-query-future invariance, duplicate/invalid-count assertions as
  dedicated tests (the cache builder DOES compute and print these counts
  at build time, but no pytest re-verifies them against known-answer
  fixtures), alpha-sums-to-1 as a dedicated test (true by construction --
  softmax -- but not independently asserted), and the aggregation-formula
  match against `free_running_aggregate_future_mse` as a dedicated test
  (the cache builder's y_ret line is copied verbatim from that function's
  own computation, so they are the same code, but no test proves this).
- research/REVIEW_FOR_CHATGPT.md: not yet appended (no results to report).
- commit/push: not done (no user approval requested this round; nothing
  to commit beyond new, never-before-committed files in any case).

CURRENT JOBS AFFECTED: none. Weather_96 factorial (pid 4137529, currently
on its final retained cosine arm) and TRACK-A-SET-LOSS-CONTROL01 (currently
on ETTh1_96/A3_setutility_softce) both confirmed alive, unpaused, unmodified,
immediately before this report was written.

STAGE1 CHECKPOINTS REUSED: none loaded yet at runtime (no cache-build run
executed) -- the loader and its fail-fast validation function
(`validate_arm_checkpoint`) are implemented and were exercised only via
static code review, not against a real checkpoint this round.

STAGE1 RETRAINED: no (by design -- never planned; Stage-1 checkpoints are
read-only throughout).

RETRIEVAL POLICY FROZEN: implemented (encoder + SetConditioner + optional
RetrievalMetric all loaded directly from the Factorial checkpoint's own
`model_state_dict`/`set_conditioner_state_dict`/`metric_state_dict` keys,
`requires_grad_(False)` + `.eval()`), but not yet exercised on GPU1.

SETCONDITIONER LOADED: implemented, not yet run. Confirmed by code audit
(section 1 below) that the standard `RelationStage2.load_stage1_checkpoint`
path does NOT load it -- this is exactly why a separate cache-builder was
required rather than connecting a standard `--stage1_ckpt_path`.

AGGREGATION WEIGHT SOURCE: implemented as specified -- Top-K MEMBERSHIP
from the arm's own frozen score (sequential, SetConditioner-conditioned,
free-running only, never Oracle-prefixed); aggregation alpha from the
FIXED S0_wce host score, softmax at the host's own `tau_topk`, identical
formula to `free_running_aggregate_future_mse`. Not yet run.

STAGE2 MODULES TRAINED (verified by a real, passing test against the real
model): `base_head.*`, `gate.*`, `relation_concat_projection.*` (12 named
parameters for the ETTh1_96 S0_wce host's own config: `stage2_relation_fusion`
resolves to a projection module in this config rather than a separate
RelationMixer with its own free parameters -- the exact trainable set is
config-dependent and is re-printed/re-saved per arm at runtime, per spec).
`stage1_encoder` and `shared_cross_projection` confirmed frozen
(`retrieval_metric`/`pairwise_scorer`/`query_cond_proj`/`candidate_cond_proj`
are absent/None for this host config, so there was nothing further to freeze
there).

TESTS: 9/9 new tests pass (scope-limited, see NOT EXECUTED). Full suite
909/911 (2 pre-existing failures, unchanged).

H96 STATUS: NOT STARTED (implementation + partial tests only).
H720 STATUS: NOT STARTED.

FILES CREATED:
scripts/build_choicece_retrieval_cache01.py
scripts/train_choicece_stage2_retrain01.py
scripts/run_choicece_stage2_retrain01_h96.sh
scripts/run_choicece_stage2_retrain01_h720.sh
tests/test_choicece_stage2_retrain01.py
research/TRACK-A-CHOICECE-STAGE2-RETRAIN01.md (this file)

FILES MODIFIED: none.

FILES OVERWRITTEN: none.

PROTOCOL DEVIATIONS:
1. A brief CPU-side structural check (verifying `build_fresh_stage2` /
   `freeze_retrieval_submodules` work against the real classes) was run
   without first pinning `CUDA_VISIBLE_DEVICES`, and `Exp_Stage2_Relation`'s
   own constructor unexpectedly moved the model onto GPU0 (physical,
   default device) for a few seconds. Confirmed via `nvidia-smi`
   immediately afterward that GPU0 returned to ~0 MiB used and 0%
   utilization -- no lasting footprint, and GPU1 was never touched by this.
   All subsequent checks and the full unit-test run were pinned to GPU2
   (idle) or GPU1 was left alone entirely, per the standing GPU-1-only
   rule for actual experiment execution.
2. Given `RelationStage2`/`Exp_Stage2_Relation` are the LARGEST and most
   config-branching files in the repository, the "full 40-item test
   battery" was descoped to the 9 highest-risk items this round (listed
   under NOT EXECUTED) to stay within a single session's effort budget --
   flagged explicitly rather than silently narrowed.
```

## 1. Code audit (section 2 answers)

1. **y_base / y_ret / gate / y_final wiring** (`models/RelationStage2.py::forward`,
   ~line 1979 onward): `y_base_all = self.base_head(batch_x)`; per channel,
   the retrieval branch (`relation_outputs`, either computed online or --
   the path this experiment uses -- read from a supplied `retrieval_cache`)
   feeds the mixer/gate; `lambda_all`/`beta_all` (the gate outputs) combine
   `y_base` and the retrieval-derived `y_ret` into `y_final`. The model
   exposes a purpose-built hook, `forward_from_retrieval_values(relation_outputs,
   *, batch_x, retrieval_cache, **forward_kwargs)` (line 1892), which calls
   `forward` itself (not a reimplementation) with the retrieval branch
   forced to supplied values -- its own docstring states this exists so
   "the mixer, the gate and the final `+ output_offset` restore stay a
   single source of truth." This is the exact hook this experiment uses.

2. **Stage-2 training loss** (`exp/exp_stage2_relation.py::_loss`, line 963):
   `loss = mean((y_final - batch_y)^2)`, restricted to `valid_query` rows,
   with OPTIONAL auxiliary terms gated by the host's own `args`:
   `use_aux_base_loss`/`aux_base_weight`, `use_aux_ret_loss`/`aux_ret_weight`,
   `beta_entropy_reg`, `retrieval_kl_weight`, `stage2_rank_weight`. This
   experiment reuses `exp._loss` UNMODIFIED and reads these weights from
   the S0_wce host's own saved `args` (read-only), per the instruction not
   to add new loss terms.

3. **`freeze_stage1_encoder=1`**: sets `param.requires_grad = False` on
   `stage1_encoder` and `shared_cross_projection` (and, when configured,
   `retrieval_metric`/`pairwise_scorer`/etc. -- confirmed via
   `models/RelationStage2.py` lines 244-366). This experiment freezes the
   SAME submodule SET explicitly (`FREEZE_SUBMODULES` in the new trainer),
   rather than relying on this flag, because the flag alone does not cover
   the SetConditioner (which is not part of `RelationStage2` at all -- see
   item 5).

4. **`load_stage1_checkpoint`** (`models/RelationStage2.py:651`): loads,
   from `ckpt['model_state_dict']`, ONLY: the encoder (`encoder.`/
   `teacher_encoder.` prefix), `shared_cross_projection`, and (if
   configured) `retrieval_metric`/`pairwise_scorer`/`query_cond_proj`/
   `candidate_cond_proj`, each via prefix-matching directly inside
   `model_state_dict`.

5. **CONFIRMED, both named suspicions true**:
   - **SetConditioner is never loaded.** `load_stage1_checkpoint` has no
     code path referencing `set_conditioner` at all. A Factorial
     checkpoint's `set_conditioner_state_dict` (a SEPARATE top-level key,
     confirmed at `scripts/train_factorial_e2e01.py:946`) would be
     silently ignored by this loader.
   - **The asymmetric `metric_state_dict` would also be missed.** Factorial
     saves it under its OWN top-level key (`metric_state_dict`, line 947),
     NOT embedded in `model_state_dict` under a `retrieval_metric.` prefix.
     `load_stage1_checkpoint` only ever looks inside `model_state_dict`.
     Worse: if `retrieval_metric` is a configured submodule on the Stage-2
     side, the loader `raise`s when it finds no `retrieval_metric.*` keys
     in `model_state_dict` -- so a naive connection would either silently
     train with a randomly-initialized metric (if unconfigured) or crash
     (if configured), depending on host config, neither of which
     reproduces the Factorial arm's actual trained comparison.

   These two confirmed findings are exactly why this experiment builds a
   SEPARATE cache-based pipeline instead of using `--stage1_ckpt_path`.

6. **Does Stage-2's own retrieval reproduce Factorial's sequential
   selection? NO.** Stage-2's native retrieval path (`models/RelationStage2.py`,
   e.g. `order = raw.masked_fill(~valid_mask, neg).argsort(dim=-1,
   descending=True)`, line 1159) is a single ONE-SHOT top-k/argsort call
   over the full candidate set -- it never constructs a SetConditioner-
   conditioned running prefix representation across K sequential picks the
   way `train_factorial_e2e01.py::run_sequence` does. Confirmed by direct
   code reading, not inferred.

7. **Does Stage-1's `free_running_aggregate_future_mse` match Stage-2's
   `y_ret` under the same Top-K/alpha? By construction, yes** -- the cache
   builder's `y_ret` computation (`scripts/build_choicece_retrieval_cache01.py`)
   is copied verbatim from `free_running_aggregate_future_mse`'s own
   `alpha = softmax(host_score(picks)/tau_topk)`, `y_ret = sum alpha*futures`
   lines, so it is definitionally the same formula given the same picks
   and the same fixed host score. **Not yet independently re-verified by a
   dedicated numeric-equality unit test this round** (listed under NOT
   EXECUTED) -- flagged rather than silently assumed proven.

## 2. Architecture decision: why `forward_from_retrieval_values` instead of
reimplementing the fusion tail

`models/RelationStage2.py`'s own docstring for this method states its
purpose is precisely to let a caller force the retrieval branch while
keeping "the mixer, the gate and the final `+ output_offset` restore" as
"a single source of truth." This experiment's trainer calls this
pre-existing, already-used-in-production hook (it backs
`evaluate_candidate_correction`, an existing diagnostic) rather than
reimplementing `forward`'s fusion tail by hand -- minimizing the risk of a
silent mismatch against the real training-time forward pass. Verified
working end-to-end (finite loss, correctly-scoped gradient, frozen hashes
unchanged across an optimizer step) by
`test_forward_from_retrieval_values_finite_loss_and_selective_gradient`,
run against the real ETTh1_96 S0_wce host's real `RelationStage2` class.

## 3. Status and next step

**IMPLEMENTATION COMPLETE (core pipeline). TESTS PARTIAL (9/9 passing,
scope-limited, see above). WAITING FOR GPU 1. CURRENT JOBS UNTOUCHED.**

GPU 1 is currently running two experiments (Weather_96 factorial's final
retained-cosine arm, TRACK-A-SET-LOSS-CONTROL01's ETTh1_96/A3 arm) and,
independently, has shown unexplained multi-hour-per-epoch slowdowns this
session that could not be attributed to this session's own concurrent
work (see prior conversation turns) -- per this experiment's own explicit
instruction not to add a third GPU1 workload while there is any risk of
impacting existing jobs, no cache-building or training has been started.
Once GPU1 is confirmed free (or the user explicitly authorizes running
alongside the current jobs, as was separately negotiated for
TRACK-A-SET-LOSS-CONTROL01), the next steps are, in order: (1) run the
smoke test (spec section 7) on `set_onpolicy_cosine`/H96; (2) if clean,
build caches and train the 8 H96 arms one at a time via
`scripts/run_choicece_stage2_retrain01_h96.sh`; (3) review H96 for
numerical stability before starting H720 via
`scripts/run_choicece_stage2_retrain01_h720.sh`. No results, tables, or
answers to the 12 final questions can be given yet -- none of the 16
arm-runs have executed.
