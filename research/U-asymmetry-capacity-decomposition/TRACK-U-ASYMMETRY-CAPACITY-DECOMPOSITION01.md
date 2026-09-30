# TRACK-U-ASYMMETRY-CAPACITY-DECOMPOSITION01

**STATUS: INTERIM REPORT -- Phase A complete (ETTh1 H96 + H720, seed0
only). Weather/multi-seed pending per this track's own decision rule.**

TRACK-T2 found Query-only Projection beats True Original KL by a large
margin at H720. That single change simultaneously adds `D^2` learnable
parameters AND breaks query/candidate symmetry -- this track separates
the two with two parameter-matched controls: **U1** (one shared `D x D`
matrix applied to BOTH sides, symmetry preserved) and **U3** (the
mirror of query-only: candidate/key-only transformation). All four arms
-- U0 (0 params), U1/U2/U3 (`D^2` params each) -- were retrained fresh
with a verified bit-identical initial projection weight across U1/U2/U3.
Full audit: `research/U-asymmetry-capacity-decomposition/AUDIT.md`.

## Setup

GPU1 only (standing rule), confirmed free before launch. Encoder init
hash identical across all 4 arms per setting; projection init SHA-256
(`07e4bdd983fb4332...`) bit-identical across U1/U2/U3 at BOTH horizons
(the seed is dataset-independent, so this is expected and confirms the
generator is genuinely deterministic across processes). Batch order
SHA-256 identical across all 4 arms per setting. 16/16 equivalence unit
tests pass. Both U0 runs reproduced TRACK-T2's T0 exactly at every
logged epoch and at the final test metric.

## Main result table

**ETTh1 H96 (seed0)**

| Arm | Extra params | Recall@10 | retMSE | D | C | Agg | Stage2 MSE | lambda |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| U0 Original | 0 | 0.0535 | 0.6807 | 0.06807 | 0.34585 | 0.41392 | **0.373682** | 0.511 |
| U1 Shared Sym | 16384 | 0.0517 | 0.6860 | 0.06860 | 0.34896 | 0.41756 | 0.375317 | 0.509 |
| U2 Query-only | 16384 | 0.0548 | 0.6324 | 0.06324 | 0.35900 | 0.42224 | 0.375630 | 0.501 |
| U3 Key-only | 16384 | 0.0533 | 0.6534 | 0.06534 | 0.36914 | 0.43449 | 0.379204 | 0.501 |

**ETTh1 H720 (seed0)**

| Arm | Extra params | Recall@10 | retMSE | D | C | Agg | Stage2 MSE | lambda |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| U0 Original | 0 | 0.0224 | 0.9554 | 0.09554 | 0.54884 | 0.64437 | 0.529134 | 0.552 |
| U1 Shared Sym | 16384 | 0.0234 | 0.9543 | 0.09543 | 0.53534 | 0.63077 | 0.524125 | 0.551 |
| U2 Query-only | 16384 | 0.0573 | 0.8254 | 0.08254 | 0.51020 | 0.59274 | 0.506902 | 0.551 |
| U3 Key-only | 16384 | 0.0427 | 0.8755 | 0.08755 | 0.45522 | 0.54277 | **0.492096** | 0.458 |

## Contrasts (PART 13)

| Horizon | Contrast A (Capacity: U1-U0) | Contrast B (Asymmetry: U2-U1) | Contrast B2 (Asymmetry: U3-U1) | Contrast C (Direction: U2-U3) |
|---|---:|---:|---:|---:|
| H96 stage2 | +0.001636 (sig, harmful) | +0.000313 (n.s.) | +0.003887 (sig, harmful) | -0.003574 (sig, U2 better) |
| H720 stage2 | -0.005009 (sig, helps) | -0.017222 (sig, helps a lot) | -0.032029 (sig, helps even more) | **+0.014807 (sig, U3 better)** |

All comparisons are bootstrap-significant (10k reps, query_start_idx
unit) except H96's U2-U1 at Stage2 (CI includes zero).

## Headline finding: capacity is a small effect, asymmetry is a large one, and the key/candidate side matters MORE than the query side

At H720, **U1 recovers only 22.5% of U0-to-U2's total gain and only
13.5% of U0-to-U3's total gain** -- adding `D^2` parameters while
keeping the comparison symmetric buys almost nothing on its own. Both
asymmetric arms (U2, U3) dramatically outperform U1 at the SAME
parameter count. This is decisive evidence for `H_asym` over
`H_capacity`.

**The genuinely unexpected result**: `U3 (key-only) significantly beats
U2 (query-only)` at H720 (Stage2 MSE 0.492096 vs 0.506902, a 0.0148
gap, larger than U1's entire improvement over U0). TRACK-T2's framing
(and this track's own hypothesis section) was built around query-only
projection being the interesting arm -- empirically, transforming the
**candidate/key side** matters more. At H96 the same asymmetry
dominance appears in reverse: U3 is harmed MORE than U2 (both worse
than U0, but U3 significantly worse than U2) -- i.e. whichever
direction a horizon favors, the key-side transformation has the LARGER
effect on it than the query-side transformation does.

## D/C mechanism (relative to U0)

| Horizon | Arm | delta D | delta C | Pattern |
|---|---|---:|---:|---|
| H96 | U1 | +0.78% | +0.90% | both get slightly worse |
| H96 | U2 | -7.09% | +3.80% | D improves, C worsens (TRACK-T2's original finding) |
| H96 | U3 | -4.01% | +6.74% | smaller D gain AND worse C than U2 -- explains why U3 is worst |
| H720 | U1 | -0.12% | -2.46% | D flat, small C gain -- capacity alone is a modest, C-only effect |
| H720 | U2 | -13.61% | -7.04% | both D and C improve substantially |
| H720 | U3 | -8.36% | **-17.06%** | smaller D gain than U2, but MUCH larger C gain -- explains why U3 wins |

The pattern is consistent across both mechanism and outcome: **U3's
effect on `C` is always the largest in magnitude of the three arms, at
both horizons**, and that is what drives it to be both the best arm at
H720 and the worst at H96.

## Post-hoc projection diagnostics (never used for selection)

| Horizon | Arm | \|\|W-I\|\|_F | condition number | effective rank |
|---|---|---:|---:|---:|
| H720 | U1 | 3.35 | 9.99 | 122.7 |
| H720 | U2 | 4.84 | 198.06 | 121.4 |
| H720 | U3 | **1.96** | **4.24** | 126.8 |

At H720 -- the setting where U3 wins -- U3's own learned projection
moved LEAST from identity and stayed the best-conditioned of the three,
while U2's projection became notably ill-conditioned (condition number
198). The best-performing arm here is the most conservative one, not
the most dramatically transformed one -- a genuinely counter-intuitive
diagnostic that argues against a naive "more transformation = more
capacity = better" story.

## Case assessment (PART 15)

**H720**: closest to **Case B** (asymmetry supported -- both U2 and U3
clearly beat U1, whose own gain over U0 is much smaller) but with an
**unregistered direction effect on top**: U3 significantly beats U2.
This does not match the pre-registered Case C (which anticipated
query-role-specificity) -- empirically it is candidate/KEY-role
specificity that dominates.

**H96**: fits no pre-registered case (all four assume some arm beats
U0; here every arm is worse). The same direction-asymmetry recurs in
reverse: U3 is harmed more than U2.

## Q1-Q7 (PART 22)

**Q1 (is H720's gain explained by capacity alone)**: NO. U1 recovers
only 22.5% (vs U2) / 13.5% (vs U3) of the total gain at the same
parameter count.

**Q2 (does asymmetric beat symmetric at equal parameters)**: YES at
H720 (both U2 and U3 significantly beat U1). NO at H96 (U2 not
significantly different from U1 at Stage2; U3 significantly worse).

**Q3 (does query-only vs key-only direction matter)**: YES, strongly,
and in the OPPOSITE direction from what TRACK-T2's own framing
implicitly assumed -- key-only wins at H720, query-only wins at H96;
key-only's effect magnitude is larger at both horizons regardless of
sign.

**Q4 (H720's D/C split between capacity and asymmetry)**: capacity
(U1 vs U0) is a small, C-only effect (D flat, C -2.46%). Asymmetry on
top (U2/U3 vs U1) unlocks both a real D benefit and a much larger C
benefit, with U3's C benefit (-17.06% vs U0) being the single largest
effect measured in this track.

**Q5 (does H96's "D improves, C worsens" pattern appear in shared
projection too)**: only partially -- U1 does not even get U2's small D
improvement; both D and C get slightly worse for U1. The "D improves
but C worsens" story is specific to the asymmetric arms.

**Q6 (which hypothesis to adopt)**: asymmetric query/key matching,
with an open, unexplained sub-finding that the key/candidate-side
transformation matters more than the query-side one. This track's own
diagnostics do not explain WHY; flagged as an open question, not
over-interpreted.

**Q7 (architecture search or generalization next)**: per this track's
own pre-registered decision rule, a Case-B-like outcome means: adopt
asymmetric matching as the mechanism, do not add further new scorer
structures, and move to Weather + multi-seed replication -- carrying
the key-vs-query direction-reversal finding into that replication as
something to check for transfer, not as grounds for a new
architecture-search track.

## STOP/VALIDITY compliance (PART 20)

No trigger encountered: parameter counts verified 0/16384/16384/16384;
projection init bit-identical across U1/U2/U3; encoder init hash
identical across all 4 arms; batch order identical across all 4 arms;
checkpoint criterion identical; Stage2 base checkpoint shared and
SHA-256-verified; both U0 reproductions matched TRACK-T2 exactly.

## Next step

Per PART 21: Weather H96 seed0 and Weather H720 seed0, identical
protocol, no architecture changes, to test whether the asymmetry effect
-- and its H96/H720 key-vs-query direction reversal -- transfers to a
second dataset.
