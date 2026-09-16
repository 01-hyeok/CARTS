```text
EXECUTED:
- Independent, evidence-based re-trace of models/RelationStage2.py::forward()
  (section 1) confirming the user's bug report and retracting Claude's own
  earlier ("RULED OUT") conclusion.
- Fixed scripts/build_choicece_retrieval_cache01.py to store `relation_outputs`
  in DELTA space (weighted sum of candidate deltas, query offset never
  added), plus a second real bug found during the smoke test (memory_c was
  not batch-broadcast before `.gather()` -- fixed by expanding it to
  [B, N_memory, pred_len] first).
- Rewrote scripts/train_choicece_stage2_retrain01.py: cache schema
  rejection (`cache_schema_version != 'corrected_delta_v1'` aborts with
  [ISSUE]), mandatory pre-training Stage-1 FR-Agg cross-check gate,
  raw_retrieval_aggregate_mse / projected_retrieval_branch_mse split,
  gate-distribution stats, counterfactual (lambda=0/lambda=1) reuse, full
  per-arm deliverable file set.
- 10 new unit tests (schema rejection x2, offset-never-in-cache,
  restore_absolute exactness + double-offset-must-differ, weighted-delta-
  sum exactness, restored-delta-matches-direct-aggregate, encoder/
  SetConditioner receive no gradient, forecast head/gate DO receive
  gradient, FP32 finite loss/gradient, FR-Agg cross-check function matches
  manual computation) -- all passing, GPU2 only (CPU-equivalent
  convention), alongside all 9 pre-existing tests in this file.
- Full repo regression suite: 939 passed / 2 pre-existing failures
  (test_topk_coverage_reuses_target_indices_across_relations,
  test_identity_retrieval_uses_raw_target_source_relation_without_encoder
  -- unrelated to this work, same names as the documented baseline), no
  new regressions.
- Smoke test (1 epoch, ETTh1_96/individual_tf_cosine, GPU1): FR-Agg
  cross-check diff=0.000000 (stage1=0.749611, cache_restored=0.749611).
- Full re-execution: ALL 16 arms (8 H96 + 8 H720), one at a time on GPU1,
  fresh Stage-2 init per cell (NOT continued from the old buggy run's
  checkpoints), every arm's FR-Agg cross-check passed with diff=0.000000
  (grep of the full run log for ISSUE/ABORT/Traceback: zero matches).
- Old report (research/TRACK-A-CHOICECE-STAGE2-RETRAIN01.md) marked
  INVALID_FOR_CONCLUSION, not deleted.
- research/REVIEW_FOR_CHATGPT.md: append-only correction section added,
  retracting the earlier false "RULED OUT" claim.

NOT EXECUTED:
- git commit/push (explicitly withheld this round per instruction).
- Multi-seed replication (this is still a single-seed=1 result per arm,
  same as the original round).
- Online-vs-cache Top-K bit-identity / duplicate-invalid-counting /
  per-channel cache coverage pytest items from the original scope note
  remain not independently re-verified by a dedicated test (exercised
  implicitly by the cache builder's own runtime assertions/prints only).

CURRENT JOBS AFFECTED: NONE. Weather_96 factorial (pid 4137529) and
TRACK-A-SET-LOSS-CONTROL01 H720 Stage-1 (pid 452129) were confirmed alive
via `ps` before, during (via a persistent monitor), and after this round's
GPU1 work; GPU1 ran up to 3 concurrent jobs (the two above + this
experiment's single active arm at a time) with >60GB headroom throughout.

ROOT CAUSE VERIFIED: YES, independently re-traced (see section 1). The
online path's `relation_outputs` (the tensor that reaches
`relation_mixer`/`gate`/`y_final`) is the candidate's own DELTA
(`memory_y[...] - memory_x_last[...]`); `query_offset` is added exactly
once, uniformly, at the very end of `forward()` to `y_final_out`/
`y_base_out`/`y_ret_out`. `_restore_retrieved_value` (which DOES add the
offset) feeds ONLY a diagnostics field, never the training/inference path.
The old cache stored `memory_c + offset_c` (absolute space) and fed that
directly into `forward_from_retrieval_values`, effectively double-counting
the query's own last observation inside the retrieval branch.

FILES CREATED:
- scripts/run_choicece_stage2_retrain01_h96_corrected.sh
- scripts/run_choicece_stage2_retrain01_h720_corrected.sh
- results/TRACK-A-CHOICECE-STAGE2-RETRAIN01-CORRECTED/** (caches, per-arm
  deliverables, checkpoints, metrics -- 16 arms)
- research/TRACK-A-CHOICECE-STAGE2-RETRAIN01-CORRECTED.md (this file)

FILES MODIFIED:
- scripts/build_choicece_retrieval_cache01.py (delta-space fix + gather
  broadcast fix)
- scripts/train_choicece_stage2_retrain01.py (schema check, FR-Agg gate,
  metric split, full deliverable set)
- tests/test_choicece_stage2_retrain01.py (synthetic cache schema fields,
  10 new tests)
- research/TRACK-A-CHOICECE-STAGE2-RETRAIN01.md (INVALID_FOR_CONCLUSION
  block prepended, body otherwise untouched)
- research/REVIEW_FOR_CHATGPT.md (append-only correction section)

FILES OVERWRITTEN: NONE. The old run's results/checkpoints under
results/TRACK-A-CHOICECE-STAGE2-RETRAIN01/ and
checkpoints/track_a_choicece_stage2_retrain01/ are untouched; the
corrected run writes to entirely separate
results/TRACK-A-CHOICECE-STAGE2-RETRAIN01-CORRECTED/ and
checkpoints/track_a_choicece_stage2_retrain01_corrected/ paths.

STAGE1 CHECKPOINTS REUSED: YES, all 16 arms' Stage-1 checkpoints (frozen
encoder + SetConditioner + RetrievalMetric where applicable) reused
read-only from checkpoints/track_a_factorial_e2e/<cell>/<arm>/checkpoint.pth,
unchanged from the original round -- Stage-1 was never affected by this
bug (its own FR-Agg is computed online, not through this cache).

STAGE2 CHECKPOINTS REJECTED: YES. The old run's Stage-2 gate/base_head/
fusion weights (results/TRACK-A-CHOICECE-STAGE2-RETRAIN01/**,
checkpoints/track_a_choicece_stage2_retrain01/**) were NEVER loaded into
this round -- every arm's Stage-2 model starts from a fresh shared init
(per-cell shared_init_out/shared_init_in, seed=1), consistent with the
original protocol; only the schema-checked, newly-built delta-space cache
differs.

CACHE VALIDATION: PASS for all 16 arms. Every arm's cache carries
cache_schema_version='corrected_delta_v1', value_space='delta',
offset_included=False, stage1_checkpoint_path/hash, config_hash; every
arm's pre-training FR-Agg cross-check (cache-restored vs Stage-1's own
reported best_val_free_running_aggregate_future_mse) matched with
diff=0.000000 (well under the 0.02 tolerance) -- see table 1.

CHOICECE STAGE2 STATUS: DONE (16/16 arms, H96 + H720, all DONE markers
present, zero FAILED.json).

SET-LOSS STAGE2 STATUS: see research/EXP-SET-LOSS-STAGE2-RETRAIN01-CORRECTED.md
(this file covers TRACK-A-CHOICECE-STAGE2-RETRAIN01 only).

H720 LOSS STATUS: N/A to this file (see the Set-Loss report).

PROTOCOL DEVIATIONS: The cache builder had a second, independent bug
(discovered during the smoke test, not part of the original bug report):
`memory_c` returned by `memory_value()` is `[N_memory, pred_len]` -- shared
across the whole batch, not per-query -- and needed explicit batch
broadcasting (`.unsqueeze(0).expand(bsz, -1, -1)`) before `.gather()` by
per-query `picks_t`; without this fix `.gather()` raised a dimension-count
RuntimeError immediately (never silently produced wrong numbers). Fixed in
both cache builders before any real cache was built for this round; the
smoke test that caught it is preserved as a paragraph above, not silently
omitted.
```

## 1. Root-cause re-verification (evidence, not assumption)

The user's report was independently re-traced against `models/RelationStage2.py`
rather than accepted on faith, per explicit instruction. Findings:

1. **`memory_value()`'s return-value meaning**: `memory_value_c = memory_y[:, :, c]`
   (optionally minus `memory_x_last[:, c]` when `relation_value_space ==
   'delta_last'`) -- this is the candidate's own future minus the
   candidate's own last-observed value, i.e. a DELTA, computed once per
   memory-bank example, not per query.
2. **Delta vs absolute**: candidate values are DELTA by construction in
   this host's config (`relation_value_space='delta_last'`, confirmed from
   the checkpoint's own saved `args`).
3. **Where `query_offset` is added**: `retrieve_relation_future(...)`
   returns `query_offset` (`batch_x[:, -1, c]`, the QUERY's own last
   value) SEPARATELY from `memory_value_c`; inside `forward()`, offset
   addition happens exactly once, at the very end, uniformly to
   `y_final_out`, `y_base_out`, `y_ret_out` (`... + output_offset`).
4. **What `forward_from_retrieval_values()` expects**: its own docstring
   states it forces the retrieval branch to the supplied values and then
   runs the SAME mixer/gate/offset-restore code the online path uses --
   i.e. it expects the same DELTA-space `relation_outputs` the online
   path's local variable holds, not an already-restored value.
5. **Does Stage-2 internally re-add the offset?**: NO, not on the path
   that reaches `y_ret_c`/`y_final_c`. `_restore_retrieved_value` (which
   DOES add `query_offset`) populates ONLY `debug['relation_outputs']`, a
   diagnostics field, computed in parallel and never fed back into the
   mixer/gate/final output.
6. **What the (old, buggy) cache's `relation_outputs` actually stored**:
   `futures = memory_c + offset_c` -- i.e. exactly the diagnostics-only
   absolute-space quantity from (5), fed directly into
   `forward_from_retrieval_values` as if it were the delta the online path
   builds. This is confirmed a real bug, not a difference of convention.

**Conclusion: [CONFIRMED BUG]**, matching the user's report. Claude's
earlier conclusion in `research/REVIEW_FOR_CHATGPT.md` (which claimed to
have "ruled out" this exact bug by tracing `_restore_retrieved_value`) is
retracted -- that trace conflated the diagnostics-only restoration path
with the training/inference path.

## 2. Table 1 -- all 16 arms, corrected pipeline

Independent Base-only Forecaster (unaffected by this bug, reused
read-only): H96 = 0.386451, H720 = 0.491450.

| cell | arm | fragg_cache | fragg_stage1 | fragg_diff | final_MSE | delta_vs_IB | base_MSE(lambda0) | proj_ret_MSE(lambda1) | raw_ret_agg_MSE | gate_mean | trained_beats_both_counterfactuals |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ETTh1_96 | individual_tf_cosine | 0.749611 | 0.749611 | 0.000000 | 0.37947 | -1.81% | 0.58802 | 0.43863 | 0.43863 | 0.409 | YES |
| ETTh1_96 | individual_onpolicy_cosine | 0.732638 | 0.732638 | 0.000000 | 0.36773 | -4.84% | 0.55212 | 0.41772 | 0.41772 | 0.418 | YES |
| ETTh1_96 | individual_tf_asymmetric | 0.778088 | 0.778088 | 0.000000 | 0.37298 | -3.49% | 0.49153 | 0.45781 | 0.45781 | 0.348 | YES |
| ETTh1_96 | individual_onpolicy_asymmetric | 0.737152 | 0.737152 | 0.000000 | 0.37402 | -3.22% | 0.62239 | 0.42863 | 0.42863 | 0.461 | YES |
| ETTh1_96 | set_tf_cosine | 0.793040 | 0.793040 | 0.000000 | 0.37802 | -2.18% | 0.46086 | 0.49400 | 0.49400 | 0.263 | YES |
| ETTh1_96 | set_onpolicy_cosine | 0.711654 | 0.711654 | 0.000000 | 0.36769 | -4.85% | 0.55772 | 0.41606 | 0.41606 | 0.417 | YES |
| ETTh1_96 | set_tf_asymmetric | 0.845128 | 0.845128 | 0.000000 | 0.37906 | -1.91% | 0.49257 | 0.51869 | 0.51869 | 0.280 | YES |
| ETTh1_96 | set_onpolicy_asymmetric | 0.714051 | 0.714051 | 0.000000 | 0.36815 | -4.74% | 0.55386 | 0.42109 | 0.42109 | 0.421 | YES |
| ETTh1_720 | individual_tf_cosine | 1.805824 | 1.805824 | 0.000000 | 0.50477 | +2.71% | 0.48630 | 0.71112 | 0.71112 | 0.224 | no |
| ETTh1_720 | individual_onpolicy_cosine | 1.631595 | 1.631595 | 0.000000 | 0.48288 | -1.74% | 0.47580 | 0.58216 | 0.58216 | 0.137 | no |
| ETTh1_720 | individual_tf_asymmetric | 1.711876 | 1.711876 | 0.000000 | 0.52256 | +6.33% | 0.50202 | 0.62704 | 0.62704 | 0.362 | no |
| ETTh1_720 | individual_onpolicy_asymmetric | 1.666294 | 1.666294 | 0.000000 | 0.55834 | +13.61% | 0.65392 | 0.59862 | 0.59862 | 0.573 | YES |
| ETTh1_720 | set_tf_cosine | 1.723849 | 1.723849 | 0.000000 | 0.53680 | +9.23% | 0.49968 | 0.72818 | 0.72818 | 0.246 | no |
| ETTh1_720 | set_onpolicy_cosine | 1.615965 | 1.615965 | 0.000000 | 0.48984 | -0.33% | 0.48439 | 0.57301 | 0.57301 | 0.121 | no |
| ETTh1_720 | set_tf_asymmetric | 1.777309 | 1.777309 | 0.000000 | 0.49005 | -0.28% | 0.47744 | 0.67093 | 0.67093 | 0.113 | no |
| ETTh1_720 | set_onpolicy_asymmetric | 1.601006 | 1.601006 | 0.000000 | 0.53977 | +9.83% | 0.50937 | 0.61359 | 0.61359 | 0.357 | no |

Note on `raw_retrieval_aggregate_mse == projected_retrieval_branch_mse`:
this is exact for every arm, not a bug. `RelationMixer.forward` computes
`y_ret = (beta.unsqueeze(-1) * relation_outputs).sum(dim=1)`; every arm
here has exactly ONE retrieval "slot" per channel (the top-K candidates
are already pre-aggregated into a single alpha-weighted delta by the
cache builder before reaching the mixer), so `beta` is `softmax` over a
single element = 1.0 identically, making the mixer a pass-through. The
learned `score_net` inside `RelationMixer` therefore has no effect on
`y_ret` in this experiment's slot configuration.

Gate distribution (best checkpoint, full stats in each arm's
`lambda_distribution.json`): H96 gates cluster tightly around 0.26-0.46
mean with low std (0.035-0.064) and almost no saturation (`gate_gt_0.98`
is 0 in all 8 H96 arms). H720 gates are far more dispersed (std up to
0.159) and several `onpolicy` arms show near-zero medians
(individual_onpolicy_cosine median=0.087, set_onpolicy_cosine
median=0.064) alongside occasionally large means driven by outliers.

## 3. Ten closing questions

**1. Does the cache-restored FR-Agg match Stage-1's own reported FR-Agg
within tolerance?** YES, for all 16 arms, diff=0.000000 exactly (well
under the 0.02 tolerance) -- this is strong, direct evidence the fix is
correct: the corrected cache reproduces Stage-1's own free-running
aggregate metric bit-for-bit-close when restored to absolute space.

**2. Does ChoiceCE retrieval beat Independent Base under the corrected
pipeline?** At H96: YES, all 8/8 arms beat Independent Base (-1.81% to
-4.85%). At H720: MIXED, only 3/8 beat it (individual_onpolicy_cosine
-1.74%, set_onpolicy_cosine -0.33%, set_tf_asymmetric -0.28%); the other
5/8 are WORSE than Independent Base, one substantially so
(individual_onpolicy_asymmetric +13.61%).

**3. Individual vs Set comparison**: at H96, Individual and Set arms are
close and interleaved (best is Set: set_onpolicy_cosine -4.85%, second is
Individual: individual_onpolicy_cosine -4.84% -- a 0.01pp gap, not a
meaningful difference at n=1 seed). At H720 the picture is noisier and
Individual's best (individual_onpolicy_cosine -1.74%) modestly beats Set's
best (set_onpolicy_cosine -0.33%, set_tf_asymmetric -0.28%), but Individual
also contains the single worst arm overall
(individual_onpolicy_asymmetric +13.61%). No consistent Individual-vs-Set
winner across horizons under this corrected pipeline.

**4. Does Stage-1 FR-Agg improvement predict Stage-2 MSE improvement?**
Weakly, within H96: `set_onpolicy_cosine` has a noticeably BETTER (lower)
Stage-1 FR-Agg than `individual_onpolicy_cosine` (0.7117 vs 0.7326, a
~2.9% relative gap) but their Stage-2 `final_mse` are nearly identical
(0.36769 vs 0.36773, a 0.01% gap) -- i.e. a real Stage-1 retrieval-quality
gap did NOT translate proportionally into a Stage-2 gap. This is
consistent with (not proof of) this project's prior "retrieval quality
ceiling doesn't move the downstream needle much" finding, now measured on
a correctly-scaled pipeline rather than a buggy one.

**5. gate/lambda vs old (buggy) run**: not directly comparable number-for-
number since the old run's gate was fit against a corrupted retrieval
signal, but qualitatively: the old H96 gate means were reported near 0.05
(collapsed toward ignoring retrieval); the corrected H96 gate means are
0.26-0.46 (meaningfully engaged with the retrieval branch). This is the
single clearest signature that the bug was actively suppressing the
learned gate's use of retrieval, not merely shifting a diagnostic number.

**6. Does the trained gate beat both counterfactuals (lambda=0, lambda=1)?**
At H96: YES for all 8/8 arms -- the gated fusion is a genuine improvement
over either branch alone. At H720: NO for 7/8 arms -- only
individual_onpolicy_asymmetric beats both; for the other 7, the trained
gated model is WORSE than simply using the base branch alone
(lambda=0/`base_mse`). This is a real, horizon-dependent finding: at H720,
this fresh Stage-2's own base_head (co-trained with the retrieval branch)
underperforms Independent Base's dedicated base-only model, and gating in
retrieval on top of it does not recover that gap for most arms.

**7. H96/H720 directional consistency**: NOT consistent. H96 uniformly
favors retrieval (gate beats base-only, final beats Independent Base);
H720 mostly does not (gate mostly loses to base-only, final mostly loses
to Independent Base). This divergence is itself the most report-worthy
result of this corrected run and should NOT be smoothed over as noise
without further investigation.

**8. Is this plausibly single-seed noise?** The H96 pattern (8/8 arms
uniformly beating both counterfactuals and Independent Base, by margins of
1.8-4.9%) is too uniform across 8 structurally different arms to be pure
noise. The H720 pattern (7/8 arms uniformly losing to the base-only
counterfactual, several also losing to Independent Base) is similarly
uniform in direction. The MAGNITUDE within each horizon (e.g. H720's
-0.28% to +13.61% spread) plausibly includes real single-seed variance,
but the H96-vs-H720 DIRECTIONAL split does not look like noise.

**9. Which of the 4 losses wins under corrected Stage-2?** N/A to this
file -- see `research/EXP-SET-LOSS-STAGE2-RETRAIN01-CORRECTED.md`.

**10. 3-seed follow-up recommendation**: YES, recommended, specifically
targeted at explaining the H96-vs-H720 directional split found here
(question 7) -- a single seed cannot distinguish "H720's fresh base_head
is systematically harder to co-train with a gate" from "this seed's H720
base_head happened to land in a bad basin." 3 seeds x 2 representative
arms (e.g. individual_onpolicy_cosine, set_onpolicy_cosine) at H720 would
be the minimum needed before treating the H720 regression as a real
architectural finding rather than seed noise.
