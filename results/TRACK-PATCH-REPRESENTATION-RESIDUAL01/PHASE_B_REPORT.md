# PHASE B -- Residual-V5, evaluation-only alpha diagnostic

No training. Reads the EXISTING T-V5 selected checkpoint (read-only)
per cell. `r_q = normalize(mean_p h_p)` (PatchMean, per PHASE A's CASE
A1 verdict -- attention discarded). `q_m(alpha) = normalize(g_m +
alpha*r_q)`, fixed alpha sweep on VALIDATION only, test never touched.
Code: `scripts/diag_patch_residual_alpha01.py`.

**Parity check (alpha=0 must exactly reproduce T-V5's own scores)**:
PASS in all 4 cells, max abs score diff = 0.00e+00 (exact).

## Full alpha sweep (validation)

| Cell | alpha | AggMSE@10 | RetMSE@10 | Top-10 overlap | Union Recall(5 heads) |
|---|---:|---:|---:|---:|---:|
| ETTh1_96 | 0.00 | 0.719034 | 1.006430 | 0.3312 | 0.4657 |
| ETTh1_96 | 0.05 | 0.719239 | 1.006569 | 0.3319 | 0.4653 |
| ETTh1_96 | 0.10 | 0.719222 | 1.006767 | 0.3328 | 0.4649 |
| ETTh1_96 | **0.25** | **0.716208** | 1.005343 | 0.3359 | 0.4628 |
| ETTh1_96 | 0.50 | 0.717659 | 1.007899 | 0.3425 | 0.4567 |
| ETTh1_96 | 1.00 | 0.720294 | 1.013833 | 0.3614 | 0.4397 |
| ETTh1_720 | 0.00 | 1.647371 | 2.001347 | 0.5217 | 0.4372 |
| ETTh1_720 | 0.05 | 1.645965 | 1.999860 | 0.5220 | 0.4374 |
| ETTh1_720 | 0.10 | 1.644759 | 1.998582 | 0.5223 | 0.4374 |
| ETTh1_720 | 0.25 | 1.642354 | 1.995860 | 0.5245 | 0.4364 |
| ETTh1_720 | **0.50** | **1.641392** | 1.995618 | 0.5295 | 0.4317 |
| ETTh1_720 | 1.00 | 1.640753 | 1.997421 | 0.5452 | **0.4141 (-5.3%, FAILS diversity gate)** |
| Weather_96 | 0.00 | 0.494205 | 0.691641 | 0.2700 | 0.5258 |
| Weather_96 | **0.05** | **0.493762** | 0.691333 | 0.2698 | 0.5255 |
| Weather_96 | 0.10 | 0.494462 | 0.691915 | 0.2697 | 0.5251 |
| Weather_96 | 0.25 | 0.495405 | 0.683664 | 0.2704 | 0.5224 |
| Weather_96 | 0.50 | 0.505370 | 0.688115 | 0.2740 | 0.5158 |
| Weather_96 | 1.00 | 0.515521 | 0.701808 | 0.2893 | 0.4973 |
| Weather_720 | 0.00 | 0.852451 | 1.013724 | 0.3834 | 0.4714 |
| Weather_720 | 0.05 | 0.852085 | 1.011663 | 0.3852 | 0.4704 |
| Weather_720 | 0.10 | 0.851283 | 1.010379 | 0.3877 | 0.4694 |
| Weather_720 | **0.25** | **0.848979** | 1.009219 | 0.3954 | 0.4655 |
| Weather_720 | 0.50 | 0.850164 | 1.009560 | 0.4147 | **0.4551 (-3.5%, FAILS diversity gate)** |
| Weather_720 | 1.00 | 0.856641 (WORSE than alpha=0) | 1.016983 | 0.4633 | 0.4260 |

Bold = the best alpha for that cell that still satisfies the Phase B
diversity-preservation sub-conditions (Top-10 overlap increase <=0.10
absolute, Union Oracle Recall decrease <=2% relative). Larger alphas
that improve AggMSE further (ETTh1_720 alpha=1.0, Weather_720
alpha=0.5) are EXCLUDED because they violate the diversity gate.

## Primary GO condition, checked directly

**Sub-condition (quality-diversity frontier exists)**: satisfied in
**4/4 cells** -- every cell has at least one alpha>0 that improves
AggMSE, does not degrade RetMSE, and keeps both diversity metrics
inside the allowed band.

**Sub-condition (>=2/4 cells with >=1% AggMSE improvement)**: **0/4
cells reach 1%.** Best safe improvement per cell: ETTh1_96 -0.39%,
ETTh1_720 -0.36%, Weather_96 -0.09%, Weather_720 -0.41%. The maximum
across all cells and all safe alphas is 0.41% (Weather_720) --
well under half of the required 1% bar.

## Verdict: STOP

The quality-diversity frontier genuinely exists and is directionally
consistent across all 4 cells (a real, reproducible signal, not
noise -- small positive alpha reliably improves AggMSE/RetMSE while
barely moving head diversity, and this pattern holds at every
horizon/dataset tested). But per the pre-registered rule (spec Section
12), the magnitude requirement (>=1% AggMSE improvement in >=2/4
cells) is not met by a wide margin. **Residual direction STOPS here --
PHASE C (trainable gamma, full training) is NOT started.** No
hyperparameter sweep, temperature, or additional complication is
added to try to clear the bar; this is reported as a genuine but
too-small effect, not escalated into a training run.

## Reading

This both confirms and bounds PHASE A's finding: contextual patch
-mean representation carries SOME real, consistently-signed retrieval
information beyond what V5's static linear heads already capture, but
the quantity of that information is small (<0.5% AggMSE at the best
alpha, in every cell) -- not large enough, by the pre-registered
threshold, to justify training a residual architecture around it.
