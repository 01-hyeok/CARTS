# TRACK-N-FORECAST-CONDITIONAL-UTILITY01

Direct follow-up to TRACK-M. TRACK-M Stage1 M2 achieved the best
retrieval-quality metrics of any arm (retMSE=0.988612, C=0.424731,
Agg=0.523593 on FULL2161), yet TRACK-M Stage2 found M2 significantly
WORSE than J1 (paired bootstrap `MSE_S3-MSE_S1=+0.009813`, 95% CI
`[0.0067, 0.0130]`), with the trained gate suppressing M2's retrieval to
near-zero (95.5% of test queries at gate<0.02). This track's SOLE
purpose is diagnosing WHY, without training any new retriever. NO
retriever training, NO Stage1 retraining, NO Set Oracle, NO coefficient
sweep, NO Weather, NO additional seed, NO Stage2 base-head joint
training anywhere in this track -- every retriever/base checkpoint is
the existing frozen one.

Full code/checkpoint audit: `research/N-forecast-conditional-utility/AUDIT.md`.

## Common fixed base (PART 4) -- removing the base-training confound

TRACK-M's own Stage2 had a real confound: S0/S1/S2/S3 each jointly
trained their OWN base head alongside their own retrieval branch, so
final-MSE differences conflated "retrieval fusion effect" with
"base-head co-training effect." This track fixes that: `B_q =
base_head(batch_x) + batch_x[:,-1:,:]` computed ONCE from TRACK-M's
frozen S0 checkpoint (`BaseForecastHead.forward` predicts in delta
space; `Model.forward()`'s own `+ output_offset` restore is applied
identically here -- confirmed by reproducing S0's reported test
`base_mse=0.488790` to `3.3e-7`). J1/K2/M2 all condition on this exact
same `B_q` for every downstream analysis in this track.

## Reproduction gates (PART 7) -- PASSED

| Arm | Uniform (expected) | Uniform (got) | abs diff | Host (expected) | Host (got) | abs diff |
|---|---|---|---|---|---|---|
| J1 | 0.564026 | 0.564026 | 1.5e-7 | 0.578322 | 0.578322 | 2.5e-7 |
| K2 | 0.546943 | 0.546943 | 1.6e-7 | 0.573898 | 0.573898 | 2.5e-7 |
| M2 | 0.523593 | 0.523593 | 3.0e-7 | 0.562447 | 0.562447 | 4.4e-8 |

All well under the `<=1e-5` gate. `Agg_w=D_w+C_w` identity confirmed
to `<5e-7` on every row (18/18 required unit tests pass, including this
identity and the reproduction gates as formal asserts).

## PART 8: aggregation damage (Uniform -> Host, test split)

| Arm | MSE_U | MSE_H | Damage | Damage % | C_w_U | C_w_H |
|---|---|---|---|---|---|---|
| J1 | 0.564026 | 0.578322 | 0.014296 | +2.53% | 0.463476 | 0.456612 |
| K2 | 0.546943 | 0.573898 | 0.026955 | +4.93% | 0.434852 | 0.383661 |
| M2 | 0.523593 | 0.562447 | 0.038854 | **+7.42%** | 0.424731 | 0.393278 |

Confirms the hypothesis: damage is worst for M2 (the strongest-complementarity
arm) and smallest for J1 (the weakest-complementarity, best-individual arm).

## PART 9: weighted D/C decomposition -- the mechanism is D_w inflation, NOT C_w reinflation

| Arm | Weighting | D_w | C_w | Agg_w |
|---|---|---|---|---|
| J1 | U | 0.100550 | 0.463476 | 0.564026 |
| J1 | H | 0.121709 | 0.456612 | 0.578322 |
| K2 | U | 0.112091 | 0.434852 | 0.546943 |
| K2 | H | 0.190237 | 0.383661 | 0.573898 |
| M2 | U | 0.098861 | 0.424731 | 0.523593 |
| M2 | H | **0.169169** | 0.393278 | 0.562447 |

`C_w` actually *decreases* under Host weighting for every arm (M2:
0.4247->0.3933) -- HostScorer does not undo complementarity in the
direct cross-term sense. The damage is entirely a `D_w` story: M2's
`D_w` jumps +71% (0.0989->0.1692), K2's +70%, J1's only +21%.
HostScorer's own similarity criterion doesn't know which slot a
multi-slot retriever chose for individual quality vs. complementarity,
so it can up-weight individually-weaker (but complementarity-contributing)
slots -- hitting the multi-slot, complementarity-optimized retrievers
(K2, M2) hardest.

## PARTS 10-12: forecast-conditional correction and analytic oracle lambda

| Arm | Weighting | Residual Cosine | Mean Oracle Gain | Mean Oracle Fused MSE | Frac λ*>0.10 |
|---|---|---|---|---|---|
| J1 | U | 0.187 | 0.0588 | 0.4300 | 0.673 |
| J1 | H | 0.182 | 0.0583 | 0.4305 | 0.660 |
| K2 | U | 0.166 | 0.0422 | 0.4466 | 0.681 |
| K2 | H | 0.165 | 0.0497 | 0.4391 | 0.645 |
| M2 | U | 0.166 | 0.0514 | 0.4374 | 0.688 |
| M2 | H | 0.158 | 0.0525 | 0.4363 | 0.645 |

J1 has the best oracle potential; M2 is a solid second (clearly above
K2 under Uniform); the Uniform-to-Host transition barely moves any
arm's oracle-fused MSE (0.4374 -> 0.4363 for M2 -- essentially flat)
even though it visibly hurts standalone retrieval MSE and cosine. Full
per-arm stats: `oracle_lambda_summary.csv`, `conditional_utility.csv`.

## PART 13: validation-calibrated deployable lambda -- the central finding

A single scalar lambda, chosen ONLY on validation (grid search 0.00-1.00,
never touching test), applied as a fixed linear fusion `B + lambda*(R-B)`:

| Arm | Weighting | Val Global λ | Test MSE (global λ) | Test MSE (per-channel λ) |
|---|---|---|---|---|
| J1 | U | -- | 0.463349 | 0.458901 |
| J1 | H | -- | 0.464746 | 0.459902 |
| K2 | U | -- | 0.471756 | 0.475693 |
| K2 | H | -- | 0.468264 | 0.468594 |
| M2 | U | -- | **0.463969** | 0.463420 |
| M2 | H | -- | **0.468076** | 0.464675 |

Under Uniform, **M2 essentially ties J1** (0.463969 vs 0.463349, a
0.00062 gap). Under Host (matching TRACK-M's actual pipeline
aggregation), the gap widens slightly (0.468076 vs 0.464746, 0.0033)
but remains **~3x smaller than TRACK-M's real trained-gate gap**
(0.0098, CI entirely positive/significant). A trivial, deployable,
non-learned linear fusion already recovers most of M2's competitive
potential; TRACK-M's actual learned gate did not.

## PART 17: correlations (pooled across arms, test split, row-level)

| Pair | Pearson r | Spearman r |
|---|---|---|
| retrieval_mse (standalone AggMSE) vs oracle_gain | 0.14 (U) / 0.14 (H) | 0.14 / 0.11 |
| residual cosine vs oracle_gain | 0.60 (U) / 0.59 (H) | **0.93** / **0.93** |
| lambda_star vs oracle_gain | 0.68 / 0.70 | 0.96 / 0.97 |

Standalone aggregate retrieval quality is a **weak** predictor of
downstream forecast-conditional utility (r~0.10-0.14). What actually
predicts utility is the correction-target residual cosine (r~0.59-0.60
Pearson, ~0.93 Spearman) -- i.e. whether a retriever's error correlates
with the BASE forecaster's own error direction, not how accurate the
retrieval is on its own. Full table: `correlations.csv`.

## PART 18-19: conditional Frozen-Base Gate-Only experiment

**Trigger condition MET**: M2's Uniform oracle/fixed-lambda potential
is clearly competitive with J1's (oracle_gain 0.0514 vs 0.0588;
val-calibrated test MSE 0.463969 vs 0.463349, essentially tied), while
TRACK-M's actual learned Stage2 gate suppressed M2 to near-zero and
produced a significantly worse forecast. Ran G0=J1/G1=K2/G2=M2, all
Uniform aggregation, all conditioned on the SAME frozen S0 base
(`base_head`/`relation_mixer`/`relation_concat_projection` loaded from
S0 and frozen; only `gate`, re-initialized to a shared fresh seed=0
init, is trainable). 10/18 unit tests directly cover this ablation's
structural guarantees (items 15-18).

| Arm | Test final_mse | vs G0 (bootstrap, 10k reps) |
|---|---|---|
| G0 (J1) | 0.489349 | -- |
| G1 (K2) | 0.488784 | -0.000565, CI [-0.00079,-0.00035], significant |
| G2 (M2) | 0.488924 | -0.000425, CI [-0.00063,-0.00023], significant |

PART 19's literal success criterion (`MSE_G2 < MSE_G0`) is technically
met and statistically significant. **But the effect sizes are tiny**
(~0.1% relative to base_mse) and **every gate-only arm dramatically
underperforms its own TRACK-M full-joint-trained counterpart** (G0 J1
gate-only=0.489349 vs the real S1 J1 full-joint result of 0.475650 --
a ~14x smaller improvement over the no-retrieval baseline). This
strongly implicates the FROZEN `relation_mixer` (trained only on S0's
all-zero retrieval, never adapted to a real nonzero signal) as a major
confound in this specific ablation, suppressing ALL three retrievers
roughly equally rather than cleanly isolating "was it specifically the
gate that failed on M2." The gate-only ranking (K2 best, then M2, then
J1) also does not match either the oracle-potential ranking (J1 best)
or TRACK-M's real Stage2 ranking (J1 best, K2 second, M2 worst) --
three diagnostic regimes, three different rankings. Full detail:
`gate_only/verdict.json`.

## Final answers (spec section 23)

**Q1. Why does M2's Stage1 Agg=0.5236 degrade to Stage2's Host-aggregated
0.5624?** HostScorer reweighting inflates `D_w` substantially (+71% for
M2) while `C_w` actually decreases slightly; net damage is a `D_w`
story, largest for the multi-slot, complementarity-optimized retrievers
(M2 +7.42%, K2 +4.93%) and smallest for J1 (+2.53%).

**Q2. Does HostScorer weighting actually destroy M2's complementarity?**
No -- not in the direct cross-term sense. `C_w` decreases (not
increases) under Host weighting for every arm. The damage mechanism is
individual-quality (`D_w`) inflation, not complementarity reversal.

**Q3. Under the same frozen base, is M2's retrieval correction well
aligned with the base residual?** Moderately: cosine 0.166(U)/0.158(H),
comparable to K2's, somewhat below J1's 0.187(U)/0.182(H) -- not low in
absolute terms, just modestly weaker than J1's.

**Q4. Is M2's oracle conditional gain larger than J1's?** No. J1's
(0.0588 U / 0.0583 H) exceeds M2's (0.0514 U / 0.0525 H), though M2
clearly beats K2 under Uniform.

**Q5. Does M2 still beat J1 under a validation-selected deployable
lambda?** Essentially tied under Uniform (0.463969 vs 0.463349); a
small but real gap under Host (0.468076 vs 0.464746) -- in both cases
far closer than TRACK-M's actual trained-gate gap.

**Q6. Primary cause of the original Stage2 failure: A/B/C/D?**
**D -- Mixed, A+B dominant.** Real, M2-disproportionate aggregation
damage (A) plus strong circumstantial evidence of gate/joint-training
underexploitation (B: a trivial calibrated lambda nearly closes the gap
that the learned gate could not), with the general (not M2-specific)
finding that standalone AggMSE weakly predicts downstream utility
(a mild C-type signal present across ALL arms, not unique to M2). The
Gate-Only ablation intended to confirm B cleanly was itself confounded
by the frozen mixer and is only weak, inconclusive supporting evidence.

**Q7. Is new Stage1 training needed? YES/NO.** **NO.** M2's retrieval is
not fundamentally deficient for forecasting -- its oracle and
validation-calibrated conditional utility are competitive with J1's.
The bottleneck is Stage2 CONSUMPTION (aggregation scheme choice and/or
joint mixer+gate training dynamics), not the M2 retrieval objective.
Per the user's own instruction, retriever architecture/loss must not be
changed again until this consumption question is resolved -- the next
step (not run in this track) is improving how Stage2 consumes retrieval
(e.g. proper joint mixer+gate training with real nonzero retrieval
present from the start, and/or reconsidering the Host-aggregation
scheme for multi-slot retrievers), not a new Stage1 objective.

**Bottom line: M2 is not actually bad for forecasting. We were consuming
a genuinely good retrieval set incorrectly.**
