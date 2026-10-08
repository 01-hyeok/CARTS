Stage-2 Mode: Professor-paper-style Validation-Only Scalar Trust Fusion
GPU: Physical GPU 1 only

# TRACK-V-PROFESSOR-FUSION01

## 1. Research Question

Does Multi-Query (V1/V2/V5) retrieve forecasting-useful history better than
the single-view Original-KL retriever (V0), **independent of** how strong
or adaptive the downstream Stage-2 fusion mechanism is? The existing CARTS
Stage-2 (`train_r_stage2_lambda01.py`, a trainable global scalar lambda) is
expressive enough that an observed forecasting improvement could come from
either (a) the retriever finding better history, or (b) the fusion gate
compensating for a weaker retriever. This track removes confound (b) by
replacing the trainable gate with the simplest possible frozen-base +
validation-only-grid-search-beta fusion used in the professor's paper
("Which Histories Matter for Time Series Forecasting? Learning Predictive
Relevance with Future Supervision"), applied identically to every method.

## 2. Existing CARTS Stage-2 vs This Track's Stage-2

| | Previous CARTS Stage-2 | This track's Stage-2 |
|---|---|---|
| File | `train_r_stage2_lambda01.py` (untouched, reused elsewhere) | `eval_professor_style_fusion01.py` (new) |
| lambda/beta | trainable `nn.Parameter`, `lambda=sigmoid(a)`, Adam, 10 epochs | fixed grid `{0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20}` |
| Selection | min validation MSE over trained epochs | min validation MSE over the grid (test NEVER touched) |
| Optimizer | Adam | **none** -- no train loop, no `.backward()` anywhere |

## 3. Fusion Formula

```
Y_final(beta) = B(q) + beta * (R(q) - B(q))
beta* = argmin_{beta in GRID} validation_MSE(beta)   (ties -> smaller beta)
```
`B(q)`: frozen per-channel-linear Base Forecaster + last-value offset.
`R(q)`: existing cache's `relation_outputs` -- uniform mean of the
selected Top-10 candidate futures, delta-last aligned to the query's
last value. No score-weighted aggregation added.

## 4. Beta Selection Protocol

Grid: `{0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20}`, beta=0 always included
(exact Base fallback). Selected on validation only; `select_beta_on_validation`
takes only validation tensors (test 4, `tests/test_professor_style_fusion01.py`).
Test evaluated exactly once, at the frozen beta*.

## 5. Dataset / Horizon / Method

Full-memory: ETTh1, Weather x H{96,192,336,720} x {V0,V1,V2,V5} = 32 cells.
Shared-Top-100 (P100): audited via filesystem first -- **H192/H336 P100
checkpoints/caches do not exist for either dataset** (never previously
trained); only H96/H720 exist. P100 scope is therefore ETTh1, Weather x
H{96,720} x {V0,V1,V2,V5} = 16 cells, exactly as anticipated. No new P100
Stage1 training was added.

V0/V1 reuse their existing canonical caches unmodified (Mean-Mixture ==
Round-Robin for S=1, already proven in `tests/test_v_meanmix_cache01.py`/
`tests/test_v_meanmix_selection01.py`). V2/V5 use the
TRACK-V-MEANMIX-CHECKPOINT-CORRECTION01 corrected (validation
Mean-Mixture RetMSE@10 selected) checkpoints and caches -- never the
aborted Round-Robin-checkpoint TRACK-V-MEANMIX-INFERENCE01 results.

## 6. Full-Memory Results (test MSE, selected-beta fusion)

**ETTh1**

| H | Base | Original KL (V0) | V1 | V2 | V5 |
|---:|---:|---:|---:|---:|---:|
| 96 | 0.39242 | 0.37769 | 0.37810 | 0.37741 | **0.37718** |
| 192 | 0.44722 | 0.42910 | 0.43126 | 0.42970 | **0.42886** |
| 336 | 0.45778 | 0.44505 | 0.44326 | **0.44223** | 0.44343 |
| 720 | 0.56036 | 0.52698 | 0.52068 | 0.52014 | **0.51880** |
| Avg | 0.46445 | 0.44471 | 0.44332 | 0.44237 | **0.44206** |

**Weather**

| H | Base | Original KL (V0) | V1 | V2 | V5 |
|---:|---:|---:|---:|---:|---:|
| 96 | 0.16936 | 0.16594 | 0.16530 | 0.16514 | **0.16510** |
| 192 | 0.19908 | 0.19734 | 0.19692 | 0.19671 | **0.19641** |
| 336 | 0.24567 | 0.24442 | 0.24232 | 0.24307 | **0.24164** |
| 720 | 0.31943 | 0.32077 | **0.31502** | 0.31600 | 0.31678 |
| Avg | 0.23339 | 0.23212 | 0.22989 | 0.23023 | **0.22998** (V1 avg 0.22989 is marginally lower) |

All selected betas in the main table hit the grid ceiling (0.2) except
where noted in the beta table below -- see section 9's ceiling-effect
caveat.

## 7. Shared-Top-100 (P100) Results (test MSE, selected-beta fusion)

| Dataset | H | Original KL (V0) | V1 | V2 | V5 |
|---|---:|---:|---:|---:|---:|
| ETTh1 | 96 | 0.38532 | 0.39043 | 0.39044 | 0.39044 |
| ETTh1 | 720 | 0.54043 | 0.59770 | 0.60046 | 0.59669 |
| Weather | 96 | 0.16899 | 0.17057 | 0.16936 | 0.17052 |
| Weather | 720 | 0.32412 | 0.32202 | 0.32206 | 0.32215 |

**Full vs P100 degradation** (`MSE_P100 - MSE_Full`, relative %):

| Dataset | H | V0 | V1 | V2 | V5 |
|---|---:|---|---|---|---|
| ETTh1 | 96 | +0.00763 (+2.0%) | +0.01233 (+3.3%) | +0.01303 (+3.5%) | +0.01326 (+3.5%) |
| ETTh1 | 720 | +0.01345 (+2.6%) | **+0.07701 (+14.8%)** | **+0.08032 (+15.4%)** | **+0.07790 (+15.0%)** |
| Weather | 96 | +0.00305 (+1.8%) | +0.00527 (+3.2%) | +0.00422 (+2.6%) | +0.00542 (+3.3%) |
| Weather | 720 | +0.00335 (+1.0%) | +0.00699 (+2.2%) | +0.00605 (+1.9%) | +0.00538 (+1.7%) |

Multi-query arms (V1/V2/V5) degrade MUCH more than V0 when restricted to
a 100-candidate pool, but only at ETTh1 H720 specifically (14-15% vs
V0's 2.6%) -- everywhere else the degradation is small (1-4%) and
roughly similar across arms.

## 8. Selected Beta Analysis

Across all 48 cells: beta=0.2 (grid ceiling) selected in 37/48 (77.1%),
beta=0.025 in 8/48, beta=0.1 in 2/48, **beta=0.0 (Base fallback) in only
1/48** (Weather_96/P100/V2). Weather-only: 1/24 fallback to beta=0,
17/24 at the ceiling -- Weather does NOT show the frequent-fallback
pattern one might expect from the earlier trainable-lambda diagnostic;
validation apparently still prefers nonzero retrieval trust even on
Weather, it simply doesn't transfer perfectly to test (see section 10).

Selected-beta does not straightforwardly track retrieval quality: the
grid ceiling absorbs 77% of cells, meaning for most cells we cannot
distinguish "validation wants a lot of retrieval trust" from
"validation wants MORE than 0.2 but the grid stops there" -- this is a
real limitation of the fixed grid, reported as-is (no grid change made,
per spec section 27).

## 9. Retrieval-Only Analysis

Retrieval-only (beta=1) test MSE is NOT monotonically improving
V0->V1->V2->V5 in any of the 8 Full-memory cells (0/8) -- more query
heads does not mean monotonically better *standalone* retrieval
quality. Final fusion MSE IS monotonically improving in 3/8 cells
(ETTh1 H720, Weather H96, Weather H192). Pairwise ranking agreement
between retrieval-only-MSE ordering and final-fusion-MSE ordering
across the 4 arms: **32/48 pairs (66.7%)** -- moderate, not strong,
agreement. Retrieval-only quality is informative but not a reliable
proxy for final forecasting benefit.

## 10. Comparison with Existing (trainable-lambda) Stage-2

| Method | H | Existing learned-lambda MSE | Professor-style MSE | Which is better |
|---|---:|---:|---:|---|
| V5 | ETTh1 96 | 0.37472 | 0.37718 | existing (close) |
| V5 | ETTh1 192 | 0.42368 | 0.42886 | existing |
| V5 | ETTh1 336 | 0.44241 | 0.44343 | existing (close) |
| V5 | ETTh1 720 | 0.50368 | 0.51880 | existing |
| V5 | Weather 96 | 0.16818 | **0.16510** | **Professor-style** |
| V5 | Weather 192 | 0.20414 | **0.19641** | **Professor-style** |
| V5 | Weather 336 | 0.25133 | **0.24164** | **Professor-style** |
| V5 | Weather 720 | 0.34229 | **0.31678** | **Professor-style, by 7.5%** |

**Clean split by dataset**: on ETTh1, the existing trainable-lambda
Stage-2 is better at every horizon (more adaptive fusion helps when
train/val/test behave similarly). On Weather, the simple validation
-grid Professor-style fusion is better at EVERY horizon, dramatically
so at H720 (7.5% lower MSE) -- directly consistent with the earlier,
separately-verified diagnostic (this same session, interactive
debugging with the user) that the trainable lambda gate on Weather
settles near its initialization (lambda~0.5) because retrieval looks
artificially strong on the TRAIN split specifically (sliding-window
near-duplicate candidates inflate train-side retrieval quality), a
signal that does not reflect validation or test. The simpler,
non-adaptive Professor-style fusion is not fooled by this because it
never fits anything to train data at all.

## 11. Timing / GPU

Aggregate across all 48 cells (see `resource_summary.csv` for
per-cell detail): total wall-clock **365.6 seconds** (~6.1 minutes) for
all 48 evaluations combined (mean 7.6s/cell, max 10.9s/cell -- the
evaluator does no training, only frozen inference + a 7-point grid
scan). Peak allocated VRAM: mean 1195.9 MiB, max 4282.4 MiB across all
cells. Physical GPU 1 confirmed via `nvidia-smi` before each
orchestrator script started and via `CUDA_VISIBLE_DEVICES=1` on every
launched process (verified via `/proc/<pid>/environ` at launch time).
All 48 cells ran strictly sequentially, never in parallel (spec
section 12).

## 12. Channel-First Optimization

**Not applicable to this evaluator.** `BaseForecastHead.forward` loops
over channels only to apply one tiny per-channel `Linear(seq_len,
pred_len)` -- no `transform_relation_features` call, no candidate
re-encoding. Retrieval cache reads are plain tensor index lookups.
There is nothing of the shape A1 (channel-first preprocessing)
optimizes in this file; applying it would be a cosmetic no-op. Stated
explicitly in `scripts/eval_professor_style_fusion01.py`'s own
docstring and `config.json`'s `channel_first_optimization_applicable:
false` field for every cell.

## 13. Failed / Missing Cells

None failed. P100 H192/H336 (both datasets) were never in scope (never
previously trained, audited via filesystem before any code was
written) -- not a failure, an explicit, pre-declared exclusion.

## 14. Final Conclusion -- Answers to Section 28

1. **Do V1/V2/V5 beat Original KL (V0) even under this simple fusion?**
   V5: **8/8** Full-memory cells. V2: 7/8 (fails only ETTh1 H192). V1:
   6/8 (fails ETTh1 H96, H192). Reliability of beating V0 increases
   monotonically with head count (6 -> 7 -> 8).
2. **Is V5 best overall? Where not?** V5 wins outright in 6/8
   Full-memory cells. Loses to V2 at ETTh1 H336 (0.44223 vs 0.44343,
   margin 0.3%) and to V1 at Weather H720 (0.31502 vs 0.31678, margin
   0.6%) -- both narrow margins, not a clear failure mode.
3. **How well does retrieval-only quality rank-predict final MSE?**
   Moderately: 66.7% pairwise agreement (32/48), 0/8 cells fully
   monotonic in retrieval-only MSE, 3/8 fully monotonic in final MSE.
   Retrieval-only quality is suggestive, not predictive.
4. **How does selected beta vary by method?** Dominated by the grid
   ceiling (77% of all cells pick beta=0.2) -- the grid is too narrow
   to resolve fine-grained differences between methods' trust levels;
   this is a real limitation, not a finding about the methods
   themselves.
5. **How often does Weather fall back to beta=0?** Only 1/24 Weather
   cells (Weather_96/P100/V2). Despite Weather's well-documented
   val/test retrieval-quality mismatch (see section 10), validation
   almost always still finds SOME nonzero beta preferable -- the
   fallback-to-Base safety mechanism exists but rarely triggers in
   practice at this grid's resolution.
6. **Full vs P100 gap?** Small (1-4%) for ETTh1 H96 and both Weather
   horizons; large specifically for ETTh1 H720 multi-query arms
   (14-15% vs V0's 2.6%) -- candidate-pool restriction hurts
   multi-query's long-horizon ETTh1 advantage disproportionately.
7. **Does the existing trainable-lambda V5 improvement survive under
   simple fusion?** Dataset-split: NO on ETTh1 (existing wins every
   horizon), YES on Weather (Professor-style wins every horizon, and
   by a wide margin at H720). The ETTh1 gain partly depends on the
   adaptive gate; the Weather gain is, if anything, UNDERSTATED by the
   adaptive gate (which was overfitting to a train-only retrieval
   artifact there).
8. **Can the forecasting gain be attributed to the retriever itself?**
   Partially, and it depends which dataset. On Weather, yes more
   cleanly -- the simplest possible fusion with no adaptation still
   shows V5 beating V0 at every horizon and beating the existing
   trainable-lambda Stage2's own V5 numbers, so the gain survives
   confound removal. On ETTh1, the picture is muddier: V5 still beats
   V0 at every horizon under simple fusion (supporting a real
   retriever-side effect), but the MAGNITUDE of the gain shrinks
   relative to the adaptive-gate numbers, meaning part of the
   previously-reported ETTh1 gain plausibly came from Stage-2
   adaptation, not the retriever alone.
9. **Is there enough evidence to keep Multi-Query as the paper's main
   method?** The evidence is positive but not uniformly strong: V5
   beats V0 in every single Full-memory cell under a deliberately
   dumb, confound-free fusion (Case E territory for 8/8 cells, not
   just a subset) -- this is the single strongest piece of evidence in
   this track for a real retriever-side contribution. Against that:
   retrieval-only ranking only weakly predicts final MSE (66.7%), P100
   restriction erases much of multi-query's ETTh1-long-horizon
   advantage, and on ETTh1 specifically a chunk of the previously
   -reported gain traces to Stage-2 adaptation rather than the
   retriever. Recommendation (not a unilateral decision): the
   V0-vs-V5 comparison under simple fusion is worth featuring
   prominently as confound-controlled evidence, but claims about WHY
   multi-query helps (candidate diversity vs. individual retrieval
   quality) are not yet settled by this track alone.

No result in this report has been reframed positively beyond what the
numbers show; where V5 loses or the evidence is mixed, that is stated
plainly above.
