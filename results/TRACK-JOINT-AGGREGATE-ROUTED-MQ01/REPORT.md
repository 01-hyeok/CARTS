# TRACK-JOINT-AGGREGATE-ROUTED-MQ01 -- Phase 3 Report

**Stage-2 Mode: Original CARTS Trainable Global Lambda** (all Stage-2 numbers below; Weather additionally gets the validation-only beta-grid CONTROL, never the primary result).

Initialized from Phase 2's BEST_BACKBONE (Hard; V5 vs Hard differ by 1.77% avg Agg MSE in Phase 2, more than the 0.5% exception threshold, so only Hard is run per spec). Encoder + SlotHeads + Router ALL jointly trainable (no freezing). Soft routing throughout (train AND inference), per Phase 2's BEST_ROUTING_MODE decision. Main loss: `L_route = KL(p_T || p_R)` where `p_R = sum_h pi_h * p_h` -- the individual Hard-argmin loss is REMOVED entirely per Phase 1's finding that Individual Utility is misaligned with Aggregate Utility. Optional auxiliary (alpha in {0, 0.1, 0.5}, no sweep beyond these 3 values): `L_router = KL(r_agg || pi)` where `r_agg` is a DETACHED soft-teacher built from the Aggregate Utility, tau_R taken from Phase 2's own validation-selected value per cell (not re-swept here). Fixed 10 epochs, no early stopping, every epoch checkpoint saved, post-hoc selection by validation Soft-routed Aggregate MSE (via a REAL forward pass each epoch, since the backbone updates).


## Table: Forecast comparison (test Aggregate MSE, Stage-2 MSE)

| Dataset | H | Best alpha | A. V5 MM | B. Hard MM | C. Frozen Router (Phase2) | D/E/F. Joint (best alpha) | G. Aggregate Oracle |
|---|---|---|---|---|---|---|---|
| ETTh1 | 96 | 0.5 | 0.40981 | 0.41450 | 0.41321 | 0.41271 | 0.34374 |
| ETTh1 | 192 | 0.5 | 0.47010 | 0.47007 | 0.46848 | 0.46760 | 0.39426 |
| ETTh1 | 336 | 0.1 | 0.50507 | 0.51644 | 0.51323 | 0.51522 | 0.43052 |
| ETTh1 | 720 | 0.0 | 0.62057 | 0.61505 | 0.61539 | 0.63197 | 0.52806 |
| Weather | 96 | 0.5 | 0.26433 | 0.27907 | 0.25872 | 0.27843 | 0.14726 |
| Weather | 192 | 0.5 | 0.32784 | 0.37150 | 0.36068 | 0.36272 | 0.19409 |
| Weather | 336 | 0.5 | 0.43673 | 0.46706 | 0.44942 | 0.42571 | 0.23050 |
| Weather | 720 | 0.0 | 0.61780 | 0.67867 | 0.66674 | 0.61365 | 0.36158 |

## Table: Stage-2 MSE comparison

| Dataset | H | Frozen Router Stage-2 MSE | Joint Stage-2 MSE | Delta | lambda (Joint) |
|---|---|---|---|---|---|
| ETTh1 | 96 | 0.36942 | 0.37035 | +0.00093 | 0.4752 |
| ETTh1 | 192 | 0.42255 | 0.42224 | -0.00031 | 0.4890 |
| ETTh1 | 336 | 0.44090 | 0.44229 | +0.00139 | 0.4362 |
| ETTh1 | 720 | 0.51204 | 0.51817 | +0.00613 | 0.4968 |
| Weather | 96 | 0.17678 | 0.18112 | +0.00434 | 0.4220 |
| Weather | 192 | 0.22037 | 0.22542 | +0.00505 | 0.4511 |
| Weather | 336 | 0.28514 | 0.28083 | -0.00431 | 0.4987 |
| Weather | 720 | 0.39612 | 0.38780 | -0.00832 | 0.5172 |

## Table: Collapse Diagnostics (best-alpha checkpoint, first vs last epoch)

| Dataset | H | Alpha | Overlap (ep1 -> last) | Union (ep1 -> last) | Active Heads (last) | Router Entropy (ep1 -> last) | Max Routing Weight (last) |
|---|---|---|---|---|---|---|---|
| ETTh1 | 96 | 0.5 | 0.429 -> 0.386 | 25.3 -> 26.9 | 5 | 0.999 -> 0.998 | 0.216 |
| ETTh1 | 192 | 0.5 | 0.475 -> 0.508 | 23.7 -> 22.6 | 5 | 0.998 -> 0.996 | 0.229 |
| ETTh1 | 336 | 0.1 | 0.495 -> 0.549 | 23.1 -> 21.3 | 5 | 0.996 -> 0.953 | 0.310 |
| ETTh1 | 720 | 0.0 | 0.682 -> 0.612 | 17.3 -> 19.4 | 5 | 0.905 -> 0.360 | 0.768 |
| Weather | 96 | 0.5 | 0.260 -> 0.219 | 33.6 -> 35.8 | 5 | 0.998 -> 0.998 | 0.217 |
| Weather | 192 | 0.5 | 0.272 -> 0.245 | 32.6 -> 33.8 | 5 | 0.999 -> 0.997 | 0.226 |
| Weather | 336 | 0.5 | 0.308 -> 0.319 | 30.4 -> 30.0 | 5 | 0.999 -> 0.995 | 0.232 |
| Weather | 720 | 0.0 | 0.349 -> 0.336 | 28.9 -> 28.9 | 5 | 0.904 -> 0.363 | 0.771 |

## Anomaly flagged before interpretation

**At both H720 cells (ETTh1_720 and Weather_720), alpha=0 (no auxiliary loss) was selected as best, and BOTH show the router's entropy collapsing from ~0.90 (near-uniform) to ~0.36 over training, with max routing weight rising to ~0.77-0.77** -- a genuine, verifiable collapse in the router's own output distribution, confined to H720 and to the no-auxiliary variant. Checked directly in `epoch_diagnostics.csv` for both cells (not inferred): entropy falls monotonically every epoch at H720 in a way it does NOT at any other horizon (H96/H192/H336 all stay above 0.95 even at their best epoch). This strongly suggests the main route-KL loss, by itself, pushes the router toward a near-committed (low-entropy) policy specifically at the longest horizon -- plausibly because future-MSE separation between heads becomes starker at longer horizons, giving the KL loss a stronger single-head-favoring gradient absent the aux loss's smoothing.

**The SAME collapse pattern produces OPPOSITE forecasting outcomes on the two datasets**: at ETTh1_720, the collapsed joint router is the worst method in the entire Table 1 (0.63197, worse than Mean-Mixture's own 0.61505 AND the Frozen Router's 0.61539) -- collapse hurts here. At Weather_720, the SAME alpha=0 collapse pattern produces the single BEST result anywhere in this track (0.61365, beating Mean-Mixture 0.67867 and Frozen Router 0.66674 by a wide margin) -- collapse helps here. Traced through the required order:
- **data**: ETTh1_720 and Weather_720 differ in channel count (7 vs 21) and in every retrieval-quality finding established earlier this research program (Weather's retrieval signal is weaker overall but, per `TRACK-HEAD-UTILITY-ALIGNMENT-CORRECTION01`, has a much larger Aggregate Oracle gap -- i.e. more unclaimed headroom for a router to find). A router that commits harder to one head has more to gain where the oracle gap is large (Weather) and more to lose where it is small (ETTh1, where the frozen router had already nearly closed much of the oracle gap: Table 1 shows ETTh1_720's Frozen-to-Oracle gap is 0.61539-0.52806=0.08733, materially smaller in relative terms than Weather_720's 0.66674-0.36158=0.30516).
- **metric**: the collapse diagnostic (entropy, max routing weight) is computed identically for both cells by the same `epoch_val_diagnostics` function -- not a measurement asymmetry.
- **implementation**: both cells use the identical trainer, loss, and hyperparameters (only `tau_r`, `tau_s`, and the data itself differ per cell) -- no code-path divergence between the two H720 runs.
- **evaluation protocol**: both followed the identical validation-only checkpoint-selection and test-only final-evaluation protocol.
- **model structure**: this is the most likely locus -- at H720, 5 independent heads routing over a 100-candidate pool may simply have less to gain from diversity on ETTh1 (fewer channels, tighter retrieval-quality regime per earlier tracks) than on Weather (more channels, much larger untapped oracle gap), so the SAME training dynamic (entropy collapse under alpha=0) exploits real headroom on Weather while over-committing to noise on ETTh1.

This is reported as a genuine, dataset-conditional finding, not smoothed into a single "collapse is good" or "collapse is bad" narrative -- the data says both, depending on dataset, and Phase 3's own alpha sweep (which DOES include non-zero alpha for every other cell) is itself evidence that the auxiliary loss's stabilizing effect is cell-dependent: alpha=0 is NOT selected as best anywhere else in the whole 8-cell grid.

## Phase 3 closing determinations

**GO criteria (spec section 12), checked programmatically**:
1. **Joint improves avg Agg MSE over Frozen Router**: YES. 8-cell average: Joint = 0.46350, Frozen Router = 0.46823.
2. **Joint wins vs Mean-Mixture in >=5/8 cells**: YES, **7/8** (only loses at ETTh1_720).
3. **Joint Stage-2 MSE improves on average in at least one dataset**: YES for **Weather** (0.26879 vs Frozen's 0.26960); NO for ETTh1 (0.43826 vs Frozen's 0.43623, i.e. Joint is WORSE on ETTh1's Stage-2 average) -- the spec requires only ONE dataset to improve, which Weather satisfies.

**PHASE3_GO = TRUE.**

**Per-cell breakdown** (not smoothed): Joint beats BOTH Frozen Router and Mean-Mixture in 3/8 cells (ETTh1_96, ETTh1_192, Weather_720 -- plus Weather_336 which beats both on Agg MSE and Stage-2 MSE together). Joint beats Mean-Mixture but loses to Frozen Router in 3/8 cells (ETTh1_336, Weather_96, Weather_192) -- i.e. joint training still beats the simple uniform baseline there but underperforms the cheaper, much-faster-to-train frozen-router alternative. Joint loses to BOTH at exactly 1/8 cells (ETTh1_720, the collapse anomaly above).

**Honest framing of the win**: Phase 3's GO is real but not uniform -- it is driven disproportionately by Weather_336 and Weather_720 (where Joint produces the best result of any method in this entire multi-phase program), while on ETTh1 the extra cost and complexity of joint training (far longer wall-clock time: Weather cells took 30-45 minutes PER alpha PER cell for training alone, vs Phase 2's frozen-router training which ran in seconds using cached features) buys a worse result at 2/4 horizons and a near-tie at the other 2. A researcher choosing between "ship the cheap Frozen Router" and "ship the expensive Joint Router" should weigh this asymmetry, not just the aggregate GO verdict.

**Proceeding to Phase 4 (TRACK-JOINT-ROUTER-GENERALIZATION01)** per the mechanical GO/STOP gate.
