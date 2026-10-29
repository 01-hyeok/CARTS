# PHASE G -- Mechanism Analysis (v1 vs v2)

Script: `scripts/analyze_topk_v1_v2_head_mechanism01.py` (new, reuses Phase C's `analyze_cell` unmodified).
Raw data: `results/TRACK-V5-TOPK-CORRECTION-MECHANISM01/phaseG/v1_v2_head_mechanism_results.json`, 24/24 rows
(S0, D0, S2-v1, D2-v1, S2-v2, D2-v2 x 4 cells).

## Full table

| Cell | Arm | Head-pair cosine | Head-pair overlap | mix_D | mix_C | Within-set-10 cosine |
|---|---|---:|---:|---:|---:|---:|
| ETTh1_96 | S0 | 0.8435 | 0.3754 | 0.066386 | 0.343422 | 0.5219 |
| ETTh1_96 | S2-v1 | 0.8483 | 0.3938 | 0.067260 | 0.344828 | 0.5123 |
| ETTh1_96 | **S2-v2** | 0.8480 | 0.3969 | **0.067119** | **0.342664** | 0.5104 |
| ETTh1_96 | D0 | 0.8986 | 0.5244 | 0.068595 | 0.368108 | 0.5368 |
| ETTh1_96 | D2-v1 | 0.9095 | 0.6116 | 0.069362 | 0.362452 | 0.5196 |
| ETTh1_96 | **D2-v2** | 0.9148 | 0.6151 | **0.068715** | **0.358934** | 0.5224 |
| ETTh1_720 | S0 | 0.9517 | 0.7315 | 0.094263 | 0.526307 | 0.5910 |
| ETTh1_720 | S2-v1 | 0.9192 | 0.6308 | 0.094814 | 0.520145 | 0.5824 |
| ETTh1_720 | **S2-v2** | 0.9497 | 0.7146 | **0.094394** | **0.519083** | 0.5857 |
| ETTh1_720 | D0 | 0.9598 | 0.7579 | 0.091846 | 0.510909 | 0.5842 |
| ETTh1_720 | D2-v1 | 0.9464 | 0.7342 | 0.094424 | 0.511813 | 0.5741 |
| ETTh1_720 | **D2-v2** | 0.9505 | 0.7409 | **0.092390** | **0.506787** | 0.5818 |
| Weather_96 | S0 | 0.6497 | 0.2526 | 0.045562 | 0.218768 | 0.5686 |
| Weather_96 | S2-v1 | 0.6257 | 0.2967 | 0.033048 | 0.171598 | 0.5557 |
| Weather_96 | **S2-v2** | **0.6744** | **0.3651** | **0.032266** | **0.169926** | 0.5651 |
| Weather_96 | D0 | 0.6498 | 0.2538 | 0.055221 | 0.250243 | 0.5651 |
| Weather_96 | D2-v1 | 0.7180 | 0.5403 | 0.033422 | 0.173469 | 0.5798 |
| Weather_96 | **D2-v2** | 0.7175 | **0.4923** | 0.036236 | 0.178481 | 0.5511 |
| Weather_720 | S0 | 0.8085 | 0.4375 | 0.078792 | 0.539010 | 0.7727 |
| Weather_720 | S2-v1 | 0.7723 | 0.3774 | 0.067347 | 0.440491 | 0.7541 |
| Weather_720 | **S2-v2** | **0.7422** | **0.3140** | **0.065801** | **0.433071** | 0.7495 |
| Weather_720 | D0 | 0.8275 | 0.4951 | 0.083301 | 0.565223 | 0.7693 |
| Weather_720 | D2-v1 | 0.7434 | 0.3208 | 0.073443 | 0.462372 | 0.7243 |
| Weather_720 | **D2-v2** | **0.8063** | **0.4385** | **0.070322** | **0.447953** | 0.7289 |

## G1: Does v2 reduce head-pair error cosine/overlap vs v1?

**Mixed, and mostly NO -- the original Phase C finding (L3b does not win via reduced head redundancy) holds
for v2 too, with one partial exception.**

- **S2 (Shared): v2 reduces cosine/overlap on exactly 1 of 4 cells (Weather_720: 0.7723->0.7422,
  0.3774->0.3140) and INCREASES it on the other 3** (ETTh1_96: 0.8483->0.8480 flat/~0, ETTh1_720:
  0.9192->0.9497 up, Weather_96: 0.6257->0.6744 up substantially).
- **D2 (Separate): v2 increases cosine/overlap on 3 of 4 cells** (ETTh1_96: 0.9095->0.9148,
  ETTh1_720: 0.9464->0.9505, Weather_720: 0.7434->0.8063 -- the largest single movement in the table) **and
  decreases it on exactly 1** (Weather_96: 0.7180->0.7175, essentially flat, with overlap actually dropping
  0.5403->0.4923).

So on 6 of 8 (arm, cell) pairs, v2 makes the heads MORE redundant with each other than v1, not less --
despite v2 improving `mix_D`/`mix_C` (and Stage1/Stage2 MSE, per Phase F) in 7 of 8 cases. **The one cell
where v2 genuinely reduces head redundancy for S2 (Weather_720) is also one of the cells where v2 gives the
largest Stage2 improvement (Phase F: -0.46%)** -- but D2 on the SAME cell shows the opposite redundancy
direction while ALSO improving Stage2 (-1.07%, the single best improvement in the whole study) -- proving
reduced redundancy is not a necessary condition for v2's gains.

## G2/G3: Individual accuracy vs error cancellation, v1 vs v2

The pattern Phase C established for v1 holds for v2 as well, and more cleanly: **`mix_D` improves (v2 < v1)
in 7 of 8 cases, exactly mirroring Phase F's Stage1/Stage2 pattern** (the sole exception is Weather_96 D2,
where `mix_D` goes 0.033422 -> 0.036236, worse -- the same arm/cell Phase F already flagged as the one
exception, now confirmed to be a `D`-level effect, not a `C`-level one). The within-set-10 candidate error
cosine (the real `C`-generating quantity) moves by much smaller relative amounts than `mix_D` does, and in
several cases moves in the OPPOSITE direction from `mix_C` itself (e.g. Weather_96 D2-v2: within-set cosine
drops 0.5798->0.5511 -- MORE cancellation by this measure -- while `mix_C` still gets WORSE, 0.173469->
0.178481, because `mix_D` also got worse and both terms interact). **`mix_D` (individual relevance) remains
the dominant, consistent driver of v2's improvement over v1, exactly as it was for v1's improvement over
L0** -- v2 does not introduce or rely on a complementarity mechanism any more than v1 did.

## G4: Convergence improvement vs performance improvement -- two separate claims

These must not be conflated, per spec:

1. **v2 converges closer to the true hard Top-K limit than v1 does.** This is a BY-CONSTRUCTION,
   mathematically proven claim (Phase D1/D2/D3), independent of any downstream performance number -- 16/16
   synthetic tests confirm it, including at exact ties where v1 cannot converge correctly at all.
2. **That better convergence improves Hard AggMSE/Stage2 MSE in 7 of 8 cases (Phase F), by a small amount
   (0.2-1.1%).** This is an EMPIRICAL claim about THIS data/checkpoints at THIS seed, not a mathematical
   consequence of (1) -- a differentiable relaxation converging more correctly to its OWN intended limit
   does not automatically imply that limit itself was the right training target; the small, non-uniform
   size of the gains (and the one clear regression) is consistent with (1) being true while (2) being a
   real but modest, data-dependent effect, not a guaranteed one.

## G5: Encoder separation capacity confound

Unchanged from Phase C's own G5 note: Separate (`D*`) has exactly 2x the encoder parameter count of Shared
(`S*`), and no capacity-matched control was added in this track either (out of scope per spec section 8,
same as the original DUAL-ENCODER01 track). Every D-vs-S comparison above inherits this confound.

## G6: Validation/Test generalization (seed0 only -- no seed-variance claim)

Spot-checked two cases using each arm's own best-validation-epoch `val_metrics.csv` (not the test-set
numbers used everywhere else in this report):

- **ETTh1_96 S2 -> S2-v2**: val Hard AggMSE improves (0.7092 -> 0.7072), same direction as the test-set
  improvement (Phase F). Validation and test agree.
- **Weather_96 D2 -> D2-v2**: val Hard AggMSE is WORSE (0.5079 -> 0.5600), same direction as the test-set
  regression already flagged as this study's one exception. **Validation and test agree that this specific
  run is a genuine regression, not a train/val-vs-test generalization gap** -- consistent with Phase F's
  framing of it as ordinary single-seed training variance rather than evidence v2 is systematically worse
  for Separate+Weather.

Only 2 of 8 (arm,cell) pairs were spot-checked this way (not all 8); this is a sample confirmation, not an
exhaustive validation/test table. **Seed = 0 only throughout this entire track -- this is not a
seed-variance estimate**, consistent with the project's long-standing single-seed-per-track convention and
the spec's own explicit prohibition on claiming otherwise.
