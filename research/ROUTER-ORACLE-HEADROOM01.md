# ROUTER-ORACLE-HEADROOM01 (Query-Adaptive Retrieval Routing, Phase 1)

**Status: ETTh1_96, ETTh1_720, Weather_96, Weather_720 complete (full
train/val/test). Solar_96/Solar_720 pending -- see section 8.**

This is Phase 1 of the three-phase Query-Adaptive Retrieval Routing
campaign. Phase 2 (`ROUTER-UTILITY-PRED01`, learnable past-only router) and
Phase 3 (`ROUTER-E2E01`, forecasting MSE) are gated on this phase's result
and are **not** implemented in this round.

## 1. Research Question

> Rather than fixing one retrieval rule for every query, can a router that
> only sees the query's own past select, per query, whichever of several
> retrieval strategies is most useful -- and does that headroom actually
> exist to be recovered?

Phase 1 answers only the second half: does the Oracle (future-aware)
upper-bound routing headroom exist at all, before any router is trained.

## 2. Three Retrieval Experts

| | Definition |
|---|---|
| R0 raw_absolute_cosine | `transform_relation_history(x, 'absolute')`, cosine, Top-10 |
| R1 delta_last_cosine | `transform_relation_history(x, 'delta_last')` (`x - x[:,-1:,:]`), cosine, Top-10 |
| R2 future_aligned_individual | The TRAINED Individual-Oracle/TF/Hard-CE/Cosine scratch encoder's own **free-running** K-step policy -- `scripts.train_factorial_e2e01.run_sequence(..., free_running=True)`, reused unmodified |

R2's checkpoint is `TRACK-A-TF-ORACLE-LEARNABILITY01`'s `individual_tf_cosine`
arm for the same cell, chosen ONLY after fingerprint validation (axis_oracle
='individual', axis_prefix='tf', axis_score='cosine', top_k=10, full channel
range) -- never by filename alone, per spec section 3.

## 3. Implementation Audit

- `diagnose_stage1_retrieval.py::run()` confirmed (line-level) that
  `raw_cos`/`input_space` already compute exactly R0 and R1 via
  `transform_relation_history()` + `F.normalize(...).matmul(...)`; this
  experiment's `scripts/diag_router_oracle_headroom01.py` reuses the same
  transform and cosine formula, verified against a direct reference in
  T1/T2.
- `run_sequence(..., free_running=True)` (audited in
  `scripts/train_factorial_e2e01.py`) structurally never reads `futures`/
  `query_future` inside its `if free_running:` branch -- confirmed by code
  read AND by T7 (corrupting `futures`/`query_future` with random garbage
  does not change R2's selected indices).
- Value reconstruction reuses `scripts.train_margutil01.memory_value()`
  unmodified for all three experts -- retrieval INPUT space (what R0/R1/R2
  score candidates by) and retrieval VALUE space (how a candidate's future
  is reconstructed once selected) are kept structurally separate, so no
  expert gets a different future-reconstruction convention.
- R0/R1/R2 are computed inside the SAME batch/channel loop iteration over
  ONE `shuffle=False` loader pass, sharing ONE `exp._candidate_mask(
  batch_start_idx)` call -- query order and candidate-mask identity across
  experts holds by construction, not by post-hoc reconciliation.

## 4. Primary Metric

Host-independent, per spec: uniform `(1/K)`-aggregate MSE,
`E_m(q) = mean_c MSE(mean_{i in TopK_m(q,c)} Y_i, Y_{q,c})`. Secondary:
candidate-mean future MSE (both saved in `query_utility_*.csv`, not
reported in the summary tables below).

## 5. Unit Tests

`tests/test_router_oracle_headroom01.py`, 9/9 passing (T1-T12 collapsed
into 9 functions covering the full checklist): raw/delta cosine match
direct reference; candidate mask is a pure, re-callable function of
`batch_start_idx`; Top-K never selects an invalid candidate; K unique
candidates per row; R2's free-running picks are index-identical across
repeated calls and to the production `run_sequence` path; no future
leakage (T7); query-id uniqueness and manual uniform-aggregate spot-check;
Oracle Router picks the exact row-min expert (synthetic); Best-Static's
function signature takes only validation rows (structurally cannot read
test labels); bootstrap mean matches a manual computation.

Full regression suite: 1038 passed / 2 pre-existing failures (unrelated).

## 6. Results

| Cell | Raw | Delta | Learned | Best Static | Oracle Router | Headroom (abs / rel) |
|---|---:|---:|---:|---|---:|---:|
| ETTh1_96 | 0.6513 | 0.5108 | 0.4440 | learned (0.4440) | 0.4094 | 0.0346 / 7.8% |
| ETTh1_720 | 1.0554 | 0.8726 | 0.7241 | learned (0.7241) | 0.6555 | 0.0687 / 9.5% |
| Weather_96 | 0.5411 | 0.5091 | 0.2930 | raw (0.5411) | 0.2036 | 0.3376 / 62.4% |
| Weather_720 | 0.7121 | 2.0864 | 0.7282 | raw (0.7121) | 0.5439 | 0.1682 / 23.6% |

Headroom bootstrap CIs (2000 resamples) are strictly positive and tight in
every cell (e.g. ETTh1_96: [0.0320, 0.0374]; Weather_96: [0.3121, 0.3652]).

**Anomaly, reported not smoothed over**: Weather_720's Delta (R1) mean test
error is 2.0864 -- roughly 3x Raw/Learned, and its Best-Static selection is
"raw" specifically because validation already flagged Delta as much worse
(val means: raw 1.030, delta 1.219, learned 1.079). Delta cosine retrieval
appears to degrade sharply at Weather's H720 specifically; this is recorded
as a real per-cell finding, not investigated further in this diagnostic
round.

## 7. Winner Fraction / Pairwise Win Matrix

| Cell | raw | delta | learned | fraction static != query-best |
|---|---:|---:|---:|---:|
| ETTh1_96 | 9.4% | 30.3% | 60.3% | 39.7% |
| ETTh1_720 | 16.0% | 26.4% | 57.6% | 42.4% |
| Weather_96 | 32.7% | 21.3% | 46.0% | 67.3% |
| Weather_720 | 47.1% | 17.6% | 35.2% | 52.9% |

Every cell has all three experts winning a non-trivial share of queries (no
expert exceeds ~60% winner share, none falls to 0) -- routing headroom is
not an artifact of one expert dominating almost everywhere. In every cell,
a substantial fraction of queries (40-67%) are NOT won by the single
validation-selected Best-Static expert, i.e. the static choice is
systematically suboptimal for a large minority-to-majority of individual
queries.

Pairwise win rate (fraction of queries where the row expert beats the
column expert), e.g. ETTh1_96: `learned` beats `raw` 85.4% of queries and
beats `delta` 66.9% (`delta_vs_learned`=33.1%, i.e. delta wins 33.1% of the
time against learned) -- learned dominates pairwise but not universally.

## 8. Limitations

- **Solar_96/720 pending.** Solar_96's R2 checkpoint
  (`individual_tf_cosine`, trained this round using the newly-built Solar_96
  S0_wce host) was still training at the time of this report (137 channels,
  ~3+ hours/epoch observed); Solar_720 is additionally blocked by the same
  Solar_720 Stage-1 host-build OOM documented in
  `TRACK-A-TF-ORACLE-LEARNABILITY01.md`. This report covers ETTh1/Weather
  only; a Solar addendum will follow once training completes.
- Oracle Router is a future-aware upper bound, never an inference-time
  policy -- reported as such throughout, never as achievable performance.
- Phase 1 does not train anything; whether a past-only router can recover
  this headroom is exactly Phase 2's open question.

## Interim Conclusion (pending Solar)

Headroom is real, non-trivial (7.8-62.4% relative), and no single expert
dominates in any of the four completed cells -- **Phase 1 supports
proceeding to Phase 2** on the ETTh1/Weather evidence. Final go/no-go
should incorporate the Solar addendum once available, per this project's
"don't decide before all planned cells are in" convention -- this report is
an interim snapshot, not the final Phase-1 verdict.
