# TRACK-Q-GATE-CAPACITY-CALIBRATION01

Direct follow-up to TRACK-P. Everything fixed: retriever=M2,
aggregation=Uniform, fusion=mixture, common frozen base=TRACK-M's S0.
Single question: how much gate complexity does exploiting M2's
retrieval actually need? Full audit: `research/Q-gate-capacity-calibration/AUDIT.md`.

## Setup

`relation_mixer`/`RelationStage2.Model` are never constructed for Q2/Q3/
Q5 (AUDIT.md): the cached common base `B` and Uniform M2 retrieval
aggregate `R` are consumed directly by a small gate module in plain
PyTorch. All Stage2 auxiliary loss terms are off for this host config,
so this is exactly the real training objective, not an approximation.
All 4 reproduction gates pass to <4e-7 (`reproduction_gate.json`): Q0
(base), Uniform M2 retrieval, Q1 (TRACK-P's fixed lambda=0.42), Q4
(TRACK-P's P4, reused directly).

## PART 8: Q2 optimizer audit (prerequisite before trusting Q3/Q5)

`trajectory.csv`'s first 10 steps show smooth, nonzero gradients
(0.018-0.038 magnitude) and lambda moving steadily from its 0.5 init
toward the optimum -- no stuck/saturation. Q2 best epoch (2) landed at
`lambda_mean=0.3998`, within `|0.3998-0.42|=0.0202 <= 0.05` of TRACK-N's
validation-selected fixed lambda, and `MSE_Q2=0.463855 ≈ MSE_Q1=0.463969`.
**Q2 passes its own audit cleanly -- Q3/Q5 results can be trusted.**

## Primary comparison table (test split, min-val-selected checkpoints)

| Arm | Gate type | #params | Test MSE | Test MAE | Mean λ | Median λ |
|---|---|---:|---:|---:|---:|---:|
| Q0 | Base | 0 | 0.488790 | 0.486478 | -- | -- |
| Q1 | Fixed global | 0 | 0.463969 | 0.477169 | 0.42 | 0.42 |
| **Q2** | **Trainable global** | **1** | **0.463855** | 0.476804 | 0.400 | 0.400 |
| Q3 | Per-channel | 7 | 0.463766 | 0.476759 | 0.408 | 0.401 |
| Q4 | Query MLP | 184,577 | 0.475686 | 0.480704 | 0.1429 | 0.00033 |
| Q5 | Global-prior Query MLP | 184,578 | 0.474148 | 0.482518 | 0.2598 | 0.1499 |

## Paired bootstrap (test split, query_start_idx unit, 10,000 reps)

| Comparison | mean diff | 95% CI | relative % | practically meaningful? |
|---|---|---|---|---|
| Q2 - Q1 | -0.000114 | [-0.00019,-0.00004] sig. | 0.025% | **No** |
| Q3 - Q2 | -0.000089 | [-0.00013,-0.00004] sig. | 0.019% | **No** |
| Q3 - Q1 | -0.000203 | [-0.00027,-0.00014] sig. | 0.044% | **No** |
| Q4 - Q3 | +0.011920 | [0.01042,0.01344] sig. | 2.57% | **Yes (bad)** |
| Q5 - Q1 | +0.010179 | [0.00902,0.01138] sig. | 2.19% | **Yes (bad)** |
| Q5 - Q4 | -0.001537 | [-0.00290,-0.00026] sig. | 0.32% | Small but real |
| Q2/Q3/Q4/Q5 - Q0 | all significant, -0.013 to -0.025 | all sig. | 2.7-5.1% | Yes (all beat base) |

Every comparison among {Q1,Q2,Q3} is statistically significant (large
n makes even tiny differences detectable) but ALL are well under the
0.2% practical-significance threshold (PART 22) -- global and per
-channel lambda are indistinguishable in any way that matters. Q4/Q5 are
both clearly, meaningfully worse than Q1/Q2/Q3 (>2% relative).

## Calibration vs. oracle lambda* (`oracle_calibration_comparison.csv`)

| Arm | MAE vs oracle | Pearson | Spearman |
|---|---|---|---|
| Q1 | 0.2788 | -- (constant) | -- |
| **Q2** | **0.2752** | -- (constant) | -- |
| Q3 | 0.2766 | 0.017 | 0.163 |
| Q4 | 0.3222 | 0.125 | -0.052 |
| Q5 | 0.3503 | -0.040 | -0.104 |

Strikingly, the CONSTANT-lambda arms (Q1/Q2) have the LOWEST MAE against
the highly variable oracle lambda* (mean 0.355, std 0.315) -- better
than either query-conditioned gate, and Q3's tiny channel-wise
correlation with oracle (Spearman 0.16) doesn't translate into a
meaningful MSE gain over Q2. Q4/Q5's per-query adaptivity does NOT
track the oracle signal well (weak/near-zero/negative correlations) --
their complexity is not being spent usefully.

## Realized gain (`realized_gain.csv`)

| Arm | mean gain | frac gain>0 |
|---|---|---|
| Q1 | 0.02482 | 59.6% |
| Q2 | 0.02493 | 60.4% |
| Q3 | 0.02502 | 60.1% |
| Q4 | 0.01310 | 37.5% |
| Q5 | 0.01464 | 67.1% |

Q1/Q2/Q3 realize essentially the same gain. Q4 realizes barely half the
gain of Q1-Q3 AND helps on fewer queries (37.5% vs ~60%) -- consistent
with its well-known over-suppression problem (PART 1). Q5 helps on MORE
queries than Q4 (67.1%, even more than Q1-Q3) but with smaller average
magnitude per query -- the global prior successfully re-engages
retrieval broadly, but the query-specific correction it learns is not
well-calibrated to when retrieval is actually worth using.

## Hypothesis verdicts (PART 20)

- **H1 (Q2≈Q1): CONFIRMED.** A single trainable global lambda, initialized
  at a neutral 0.5 (NOT at the known-good 0.42), recovers the
  validation-grid-search optimum via plain gradient descent alone.
- **H2 (Q3<Q2 meaningfully): NOT CONFIRMED.** Statistically significant
  but practically negligible (0.019% relative) -- channel-wise variation
  in retrieval utility is real (TRACK-N found per-channel calibration
  differences) but not large enough to matter for M2 Uniform mixture.
- **H3 (Q4 over-complex/miscalibrated): CONFIRMED.** Q4 is ~2.5% relative
  worse than Q3, a large and highly significant gap, entirely explained
  by its severe under-use of retrieval (median λ=0.00033, 59.2% of
  queries below λ=0.02).
- **H4 (Q5<Q4, prior helps): CONFIRMED, but insufficient.** Q5
  significantly beats Q4 (-0.32% relative) and roughly doubles the
  fraction of queries where retrieval helps (67.1% vs 37.5%) -- the
  under-use problem is genuinely alleviated by good calibration/init.
  But Q5 still falls ~2.2% short of the trivial Q1/Q2/Q3 baselines: query
  conditioning, even well-initialized, does not yet add value beyond a
  constant lambda for this arm.

## Decision-table verdict: **Case A -- Global scalar sufficient**

All three of Case A's criteria hold: `Q2≈Q1` (0.025% relative), `Q3` no
meaningful improvement over `Q2` (0.019% relative), `Q5` no meaningful
improvement over `Q2`/`Q3` (in fact 2.2% WORSE). Per PART 23's priority
rule (differences <0.2% relative favor the simpler gate), the final
Stage2 fusion/gate structure is fixed as:

\[
\boxed{\text{M2 + Uniform + Mixture + Trainable Global Lambda (Q2)}}
\]

Q2 is preferred over Q1 despite their near-identical performance because
it self-calibrates during ordinary end-to-end training (no separate
offline validation grid search needed) while remaining exactly as simple
(a single scalar parameter).

## Final answers (spec section 26)

**Q1. Does a single trainable global lambda recover the fixed validation
lambda=0.42 level?** Yes -- `MSE_Q2=0.463855` vs `MSE_Q1=0.463969`
(Q2 is even marginally better), lambda converges to 0.40 (within the
0.05 tolerance), confirmed via a clean optimizer audit (PART 8).

**Q2. Global vs per-channel -- which is better?** Statistically Q3 edges
out Q2 (`-0.000089`, significant) but the difference is 0.019%
relative -- practically identical. Global is preferred for simplicity.

**Q3. Does the current query-conditioned MLP gate (Q4) provide real
additional benefit?** No -- it is significantly and meaningfully WORSE
than every simpler alternative (2.5% relative worse than per-channel,
2.4% worse than base-adjusted global).

**Q4. Does Q5 fix Q4's over-suppression?** Partially. Q5 halves the
under-use problem in a real, measurable sense (queries where retrieval
helps: 37.5% -> 67.1%; `lambda<0.02` fraction drops substantially) and
significantly beats Q4 in test MSE. But it does not fully fix it: Q5
still trails Q1/Q2/Q3 by ~2.2%.

**Q5. Is global-prior + query correction actually better than fixed/
global/per-channel?** No. Despite its more sophisticated architecture
and good calibration/init, Q5's test MSE (0.474148) is clearly worse
than Q1/Q2/Q3's (~0.464).

**Q6. Is the best gate's parameter-count-to-performance tradeoff
justified?** No for Q4/Q5 (184K+ params for meaningfully WORSE
performance than a 1-parameter model). Q3's extra 6 parameters over Q2
are not justified either (0.019% relative gain). **Q2's 1 parameter is
the clear, fully justified choice.**

**Q7. Final Stage2 gate: global / per-channel / query-conditioned /
prior-conditioned-query?** **Global** (Q2, trainable).

**Q8. Ready to generalize to other horizons/Weather/seeds? YES/NO.**
**YES.** A single Stage2 fusion/gate structure (M2 + Uniform + Mixture +
Trainable Global Lambda) is now fixed by this track's evidence, per the
user's own stated protocol for when to move to generalization.

## STOP rule compliance

No Stage1 retriever redesign, no new teacher, no Multi-Slot
objective/consumer, no HostScorer tuning, no Weather, no other horizon,
no additional seed anywhere in this track. Retriever=M2,
aggregation=Uniform, fusion=mixture held fixed throughout; only gate
architecture/capacity was varied (Q0-Q5), exactly as scoped.
