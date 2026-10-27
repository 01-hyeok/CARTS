# PHASE A -- Oracle Lambda Diagnostic

No new Stage1/Stage2 training. All B (Base)/R (Retrieval)/Y (future)
tensors reused verbatim from `TRACK-P100-COARSE-TO-FINE-RETRIEVAL01`'s
own cache (`scripts/build_p100_coarse_to_fine_cache01.py` + the same
Base Forecaster checkpoint), loaded via
`scripts/train_r_stage2_lambda01.py::load_tensors` (imported,
unmodified). Code: `scripts/diag_p100_oracle_lambda01.py`. Arms R
tested: canonical V5 and Phase D Set-aware (the two retrieval caches
explicitly named in the audit's own artifact list, section 3).

## Phase 0 -- reproduction check

Read directly from the ORIGINAL (not re-run) `metrics.json` files, so
reproduction is exact (not approximate): Base/V5/Set-aware test MSE
for all 4 cells match `FINAL_REPORT.md`'s published table to full
float precision (e.g. Weather_96: base=0.16936443746089935,
V5=0.1766497790813446, matching the rounded 0.1694/0.1767 published).
No discrepancy found; Phase A proceeds.

## Table 1 -- Global Lambda arms (test MSE)

| Cell | Arm | Base (A0) | Current Global λ (A1) | Val-Optimal Global λ (A2, closed form) | Current λ | Val-Opt λ |
|---|---|---:|---:|---:|---:|---:|
| Weather_96 | V5 | 0.1694 | 0.1767 | **0.1687** | 0.3953 | 0.2258 |
| Weather_96 | PhaseD | 0.1694 | 0.1714 | **0.1671** | 0.4595 | 0.3180 |
| Weather_720 | V5 | 0.3194 | 0.3855 | 0.3224 | 0.5094 | 0.1796 |
| Weather_720 | PhaseD | 0.3194 | 0.3715 | 0.3222 | 0.5093 | 0.1984 |
| ETTh1_720 | V5 | 0.5604 | **0.5149** | 0.5153 | 0.4793 | 0.4867 |
| ETTh1_720 | PhaseD | 0.5604 | **0.5096** | 0.5106 | 0.4712 | 0.4978 |
| ETTh1_96 | V5 | 0.3924 | **0.3696** | 0.3694 | 0.4733 | 0.4241 |
| ETTh1_96 | PhaseD | 0.3924 | 0.3761 | **0.3758** | 0.3781 | 0.3436 |

Bold = best of {Base, Current, Val-Optimal} for that row. **On both
Weather cells, simply re-solving for the global lambda in closed form
(no new model, same frozen B/R) beats the ACTUALLY DEPLOYED
gradient-trained lambda by a real margin** (Weather_96/V5: 0.1687 vs
0.1767, a 4.5% improvement; Weather_720/V5: 0.3224 vs 0.3855, a 16.4%
improvement) -- and on Weather_96 specifically, this closed-form
lambda alone (still a SINGLE global scalar, still deployable, not an
oracle) already beats Base. **On both ETTh1 cells, the deployed
lambda is already within 0.1% of the closed-form optimum** -- there
is essentially no global-lambda utilization headroom left on ETTh1 at
all.

## Table 2 -- Oracle headroom (test, % -- see spec section 10 formulas)

| Cell | Arm | Global λ headroom % | Oracle utilization headroom % (per query-channel) | Base-relative oracle gain % |
|---|---|---:|---:|---:|
| Weather_96 | V5 | 4.51 | 21.39 | 18.00 |
| Weather_96 | PhaseD | 2.53 | 19.41 | 18.44 |
| Weather_720 | V5 | 16.38 | 28.73 | 14.00 |
| Weather_720 | PhaseD | 13.26 | 25.58 | 13.45 |
| ETTh1_720 | V5 | **-0.07** | 9.17 | 16.54 |
| ETTh1_720 | PhaseD | **-0.20** | 9.75 | 17.93 |
| ETTh1_96 | V5 | 0.06 | 11.11 | 16.28 |
| ETTh1_96 | PhaseD | 0.06 | 11.01 | 14.72 |

**Global λ headroom is the single cleanest split in this whole
diagnostic**: essentially zero (even slightly negative, i.e. the
trained lambda is marginally BETTER than the closed-form val-optimal
one applied to test, within val/test sampling noise) on both ETTh1
cells, vs. a real 2.5-16.4% on both Weather cells. This directly
supports **H2 (Utilization Bottleneck) as a genuine, cell-specific
factor on Weather** -- the EXISTING Stage2 training procedure
(gradient descent on a sigmoid-parameterized scalar) converges to a
measurably worse-than-optimal global lambda specifically on Weather,
not on ETTh1.

Per-query-channel oracle utilization headroom (the ceiling if lambda
could vary per query AND channel, using future information) is large
everywhere (9-29%), largest on Weather_720 (25.6-28.7%) -- but this
is a diagnostic upper bound, not evidence that any deployable method
can reach it (that is Phase B's question).

## Table 3 -- Oracle lambda distribution (per query-channel, test)

| Cell | Arm | mean | median | frac(λ=0) | frac(λ=1) | frac(interior) | frac(R better than B) |
|---|---|---:|---:|---:|---:|---:|---:|
| Weather_96 | V5 | 0.463 | 0.381 | 0.348 | 0.293 | 0.359 | 0.458 |
| Weather_96 | PhaseD | 0.455 | 0.352 | 0.352 | 0.290 | 0.357 | 0.447 |
| Weather_720 | V5 | 0.322 | 0.160 | 0.366 | 0.077 | 0.557 | 0.304 |
| Weather_720 | PhaseD | 0.312 | 0.139 | 0.369 | 0.079 | 0.552 | 0.292 |
| ETTh1_720 | V5 | 0.410 | 0.398 | 0.189 | 0.050 | 0.761 | 0.413 |
| ETTh1_720 | PhaseD | 0.418 | 0.415 | 0.200 | 0.050 | 0.750 | 0.427 |
| ETTh1_96 | V5 | 0.384 | 0.283 | 0.326 | 0.156 | 0.518 | 0.361 |
| ETTh1_96 | PhaseD | 0.385 | 0.240 | 0.341 | 0.127 | 0.532 | 0.326 |

The oracle lambda distribution is NOT dominated by 0 or 1 anywhere
(interior fraction 35.7-76.1%) -- the "right answer" per query/
channel is almost never a simple binary in/out decision; a genuinely
continuous, per-row trust level would be needed to realize the full
oracle headroom. Note `frac(R better than B)` is actually HIGHER on
Weather_96 (0.45-0.46) than on ETTh1_96 (0.33-0.36), despite ETTh1
being the dataset where fusion ultimately WORKS and Weather being the
one where it doesn't -- individual retrieval-vs-base accuracy is not
what determines whether fusion helps; this is consistent with this
whole research program's repeated finding (`TRACK-P100-COARSE-TO-FINE-RETRIEVAL01`)
that D/C aggregation-bias effects, not pointwise accuracy, drive the
outcome.

## Table 4 -- Retrieval gain g_q = MSE(B_q,Y_q) - MSE(R_q,Y_q) distribution (per query-channel, test)

| Cell | Arm | mean | median | positive fraction |
|---|---|---:|---:|---:|
| Weather_96 | V5 | -0.095 | -0.001 | 0.458 |
| Weather_96 | PhaseD | -0.049 | -0.001 | 0.447 |
| Weather_720 | V5 | -0.298 | -0.048 | 0.304 |
| Weather_720 | PhaseD | -0.241 | -0.055 | 0.292 |
| ETTh1_720 | V5 | -0.060 | -0.032 | 0.413 |
| ETTh1_720 | PhaseD | -0.048 | -0.028 | 0.427 |
| ETTh1_96 | V5 | -0.017 | -0.018 | 0.361 |
| ETTh1_96 | PhaseD | -0.047 | -0.028 | 0.326 |

Mean retrieval gain is NEGATIVE in every single cell/arm (retrieval
alone is, on average, worse than base alone everywhere, including
ETTh1) -- this confirms retrieval's value, where it exists, comes
entirely through FUSION complementarity, never through retrieval
being independently more accurate. Weather_720's gain is the most
negative of all 8 rows (-0.24 to -0.30), consistent with it being the
cell with the worst current Stage2 outcome.

## Decision gate inputs (feeds Phase C)

- **ETTh1 (both H96, H720): headroom is almost entirely at the
  per-query level (9-18%), NOT at the global-lambda level (~0%).**
  If a deployable per-query signal existed, ETTh1 could improve
  further beyond its already-working global fusion -- but the
  current global lambda itself is not the problem here.
- **Weather (both H96, H720): headroom exists at BOTH levels** --
  global lambda alone recovers 2.5-16.4%, and per-query oracle adds a
  further 19-29% ceiling on top. Weather's problem is a mix of H1
  (even the best deployable global lambda on Weather_720 does not
  clearly beat Base: 0.3224 vs 0.3194) and H2 (the actually-deployed
  lambda is far from even that best global choice).
- Per spec section 19-21's Case framework, this sets up Phase B as
  the key test of whether H2's potential (the per-query oracle
  headroom, 9-29% everywhere) is actually realizable from past-only
  information, or whether it requires reading the future (in which
  case it collapses back toward H1 -- "no deployable additional
  information").
