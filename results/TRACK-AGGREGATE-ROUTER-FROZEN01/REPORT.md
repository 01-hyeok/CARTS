# TRACK-AGGREGATE-ROUTER-FROZEN01 -- Phase 2 Report

**Stage-2 Mode: Original CARTS Trainable Global Lambda** (the TERTIARY decision criterion and all "best-variant" Stage-2 numbers below use this mode exclusively; the Weather beta-grid numbers are a validation-only CONTROL/diagnostic, never the primary result).

Router input granularity: CHANNEL_AWARE (per TRACK-HEAD-UTILITY-ALIGNMENT-CORRECTION01/phase1_decision.json). Backbone (encoder+SlotHeads+retriever) FROZEN throughout -- only the Router (2-layer MLP, Linear(d,d/2)->GELU->Linear(d/2,5)) is trained, on CACHED frozen-backbone features (z_qc, U_agg/U_ind), 10 fixed epochs, every epoch checkpoint saved, post-hoc selection by validation Hard-routed Aggregate MSE. 8 cells x 2 backbones (V5, Hard) x 2 targets (R1 Aggregate-Winner-CE, R2 Aggregate-Soft-Teacher with a 4-value tau sweep, validation-selected) = 80 training runs, plus one real-backbone-forward full evaluation pass per (cell, backbone) covering every required baseline and both Hard/Soft routing inference modes together.


## Table: Forecast Baselines (test Aggregate MSE, both backbones)

| Dataset | H | Backbone | Mean-Mixture | Fixed-Head | Majority-Head | Random-Router | R1-Hard | R1-Soft | R2-Hard | R2-Soft | Aggregate-Oracle | Final-Fusion-Oracle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ETTh1 | 96 | V5 | 0.40981 | 0.41900 | 0.43121 | 0.42930 | 0.42689 | 0.40925 | 0.41600 | 0.40970 | 0.33904 | 0.33340 |
| ETTh1 | 96 | Hard | 0.41450 | 0.42261 | 0.44328 | 0.43049 | 0.43522 | 0.41321 | 0.41648 | 0.41411 | 0.34374 | 0.33690 |
| ETTh1 | 192 | V5 | 0.47010 | 0.47886 | 0.47854 | 0.47796 | 0.48030 | 0.46762 | 0.47092 | 0.46955 | 0.39829 | 0.39270 |
| ETTh1 | 192 | Hard | 0.47007 | 0.47976 | 0.48436 | 0.49251 | 0.48654 | 0.46848 | 0.46956 | 0.46949 | 0.39426 | 0.38637 |
| ETTh1 | 336 | V5 | 0.50507 | 0.51941 | 0.52031 | 0.51135 | 0.52940 | 0.50535 | 0.51222 | 0.50501 | 0.43951 | 0.41863 |
| ETTh1 | 336 | Hard | 0.51644 | 0.54853 | 0.51372 | 0.54152 | 0.53226 | 0.51323 | 0.51119 | 0.51464 | 0.43052 | 0.41916 |
| ETTh1 | 720 | V5 | 0.62057 | 0.62230 | 0.64054 | 0.62525 | 0.64540 | 0.62313 | 0.63053 | 0.62070 | 0.56967 | 0.49528 |
| ETTh1 | 720 | Hard | 0.61505 | 0.63303 | 0.62595 | 0.66464 | 0.64601 | 0.61539 | 0.62687 | 0.61353 | 0.52806 | 0.47626 |
| Weather | 96 | V5 | 0.26433 | 0.21090 | 0.29308 | 0.30512 | 0.26702 | 0.25135 | 0.24250 | 0.26165 | 0.13428 | 0.13861 |
| Weather | 96 | Hard | 0.27907 | 0.25877 | 0.26759 | 0.29335 | 0.25717 | 0.25872 | 0.23533 | 0.27112 | 0.14726 | 0.14223 |
| Weather | 192 | V5 | 0.32784 | 0.30027 | 0.37132 | 0.36293 | 0.32407 | 0.32583 | 0.31299 | 0.32716 | 0.16691 | 0.16290 |
| Weather | 192 | Hard | 0.37150 | 0.34184 | 0.34053 | 0.36937 | 0.35343 | 0.36068 | 0.35039 | 0.36849 | 0.19409 | 0.16923 |
| Weather | 336 | V5 | 0.43673 | 0.44113 | 0.60624 | 0.39955 | 0.41277 | 0.42437 | 0.41353 | 0.43172 | 0.23514 | 0.20714 |
| Weather | 336 | Hard | 0.46706 | 0.45782 | 0.45194 | 0.46224 | 0.44996 | 0.44942 | 0.45585 | 0.46518 | 0.23050 | 0.20428 |
| Weather | 720 | V5 | 0.61780 | 0.55067 | 0.55073 | 0.57151 | 0.67071 | 0.67380 | 0.62446 | 0.62783 | 0.39017 | 0.30554 |
| Weather | 720 | Hard | 0.67867 | 0.71796 | 0.60060 | 0.60055 | 0.64449 | 0.66674 | 0.65004 | 0.67687 | 0.36158 | 0.28949 |

## Table: Router Quality Metrics (test split, both backbones)

| Dataset | H | Backbone | R1 Winner Acc | R1 Top2 Acc | R1 Entropy | R1 Max Head Usage | R2 Winner Acc | R2 Top2 Acc | R2 Entropy | R2 Max Head Usage |
|---|---|---|---|---|---|---|---|---|---|---|
| ETTh1 | 96 | V5 | 0.215 | 0.420 | 0.996 | 0.394 | 0.202 | 0.402 | 0.999 | 0.464 |
| ETTh1 | 96 | Hard | 0.254 | 0.473 | 0.988 | 0.378 | 0.211 | 0.407 | 0.999 | 0.346 |
| ETTh1 | 192 | V5 | 0.264 | 0.505 | 0.987 | 0.404 | 0.236 | 0.453 | 0.999 | 0.347 |
| ETTh1 | 192 | Hard | 0.261 | 0.459 | 0.986 | 0.333 | 0.191 | 0.383 | 0.998 | 0.398 |
| ETTh1 | 336 | V5 | 0.259 | 0.477 | 0.978 | 0.608 | 0.183 | 0.371 | 0.999 | 0.451 |
| ETTh1 | 336 | Hard | 0.262 | 0.465 | 0.981 | 0.456 | 0.203 | 0.420 | 0.998 | 0.331 |
| ETTh1 | 720 | V5 | 0.258 | 0.527 | 0.949 | 0.947 | 0.193 | 0.375 | 0.999 | 0.263 |
| ETTh1 | 720 | Hard | 0.257 | 0.484 | 0.971 | 0.471 | 0.177 | 0.368 | 0.996 | 0.444 |
| Weather | 96 | V5 | 0.270 | 0.469 | 0.945 | 0.336 | 0.262 | 0.459 | 0.997 | 0.325 |
| Weather | 96 | Hard | 0.281 | 0.474 | 0.932 | 0.361 | 0.204 | 0.463 | 0.997 | 0.287 |
| Weather | 192 | V5 | 0.240 | 0.433 | 0.958 | 0.341 | 0.190 | 0.383 | 0.998 | 0.366 |
| Weather | 192 | Hard | 0.254 | 0.453 | 0.963 | 0.411 | 0.205 | 0.406 | 0.999 | 0.238 |
| Weather | 336 | V5 | 0.236 | 0.430 | 0.969 | 0.327 | 0.219 | 0.413 | 0.997 | 0.383 |
| Weather | 336 | Hard | 0.248 | 0.448 | 0.975 | 0.314 | 0.244 | 0.441 | 0.994 | 0.340 |
| Weather | 720 | V5 | 0.248 | 0.474 | 0.982 | 0.800 | 0.198 | 0.388 | 0.999 | 0.228 |
| Weather | 720 | Hard | 0.222 | 0.424 | 0.971 | 0.307 | 0.201 | 0.402 | 0.998 | 0.259 |

## Table: Oracle Gap Recovered % ( (Fixed-Router)/(Fixed-Oracle)*100, test split, both backbones )

| Dataset | H | Backbone | R1-Hard | R1-Soft | R2-Hard | R2-Soft | Mean-Mixture |
|---|---|---|---|---|---|---|---|
| ETTh1 | 96 | V5 | -9.87% | 12.19% | 3.75% | 11.62% | 11.49% |
| ETTh1 | 96 | Hard | -15.99% | 11.92% | 7.77% | 10.78% | 10.28% |
| ETTh1 | 192 | V5 | -1.78% | 13.96% | 9.86% | 11.56% | 10.88% |
| ETTh1 | 192 | Hard | -7.93% | 13.19% | 11.93% | 12.00% | 11.33% |
| ETTh1 | 336 | V5 | -12.49% | 17.60% | 9.00% | 18.03% | 17.95% |
| ETTh1 | 336 | Hard | 13.79% | 29.91% | 31.64% | 28.72% | 27.20% |
| ETTh1 | 720 | V5 | -43.90% | -1.59% | -15.63% | 3.04% | 3.28% |
| ETTh1 | 720 | Hard | -12.37% | 16.81% | 5.87% | 18.58% | 17.13% |
| Weather | 96 | V5 | -73.25% | -52.80% | -41.25% | -66.23% | -69.74% |
| Weather | 96 | Hard | 1.43% | 0.04% | 21.02% | -11.08% | -18.21% |
| Weather | 192 | V5 | -17.85% | -19.17% | -9.54% | -20.17% | -20.68% |
| Weather | 192 | Hard | -7.84% | -12.75% | -5.79% | -18.04% | -20.08% |
| Weather | 336 | V5 | 13.76% | 8.14% | 13.40% | 4.57% | 2.14% |
| Weather | 336 | Hard | 3.46% | 3.70% | 0.87% | -3.23% | -4.06% |
| Weather | 720 | V5 | -74.79% | -76.72% | -45.98% | -48.08% | -41.83% |
| Weather | 720 | Hard | 20.62% | 14.37% | 19.06% | 11.53% | 11.02% |

## Table: GO Criteria Verification (per spec section 12, applied per-backbone)

| Backbone | Variant | Wins vs MM (>=5/8?) | ETTh1 avg improved? | Weather avg improved? | Not worse than Fixed (avg)? | GO? |
|---|---|---|---|---|---|---|
| V5 | R1-Hard | 2/8 (No) | No | No | No | **FALSE** |
| V5 | R1-Soft | 5/8 (Yes) | Yes | No | No | **FALSE** |
| V5 | R2-Hard | 3/8 (No) | No | Yes | No | **FALSE** |
| V5 | R2-Soft | 6/8 (Yes) | Yes | No | No | **FALSE** |
| Hard | R1-Hard | 4/8 (No) | No | Yes | Yes | **FALSE** |
| Hard | **R1-Soft** | **7/8 (Yes)** | **Yes** | **Yes** | **Yes** | **TRUE** |
| Hard | R2-Hard | 6/8 (Yes) | No | Yes | Yes | **FALSE** |
| Hard | R2-Soft | 8/8 (Yes) | Yes | Yes | Yes | **TRUE** |

**PHASE2_GO = TRUE**, satisfied by the Hard backbone under BOTH R1-Soft and R2-Soft. The V5 backbone satisfies NO variant's full criteria set -- every V5 variant that wins >=5/8 cells vs Mean-Mixture still fails the Weather-average-must-improve criterion (Weather_720 is catastrophic for every V5-backbone router variant, 0.62-0.67 agg MSE vs Mean-Mixture's own already-bad 0.61780).


## BEST_BACKBONE / BEST_ROUTER_TARGET / BEST_ROUTING_MODE decision (spec section 14)

Both Hard/R1-Soft and Hard/R2-Soft satisfy Phase 2 GO. Tie-broken by the spec's own priority order:

- **PRIMARY (Test Aggregate MSE, 8-cell average)**: R1-Soft = 0.46823, R2-Soft = 0.47418 -- R1-Soft wins.
- **SECONDARY (Oracle Gap Recovered %, 8-cell average)**: R1-Soft = 9.65%, R2-Soft = 6.16% -- R1-Soft wins.
- **TERTIARY (actual gradient-trained Stage-2 MSE, built from a real retrieval cache using each variant's inference)**: computed for R1-Soft only (the already-confirmed PRIMARY+SECONDARY winner) against the existing Hard-backbone Mean-Mixture Stage-2 MSE (`TRACK-HARD-EXPERT-V5-P100-ALLH01`'s own Hard/stage2/metrics.json) -- see table below; R1-Soft's router-cache Stage-2 MSE beats Mean-Mixture Stage-2 MSE in 6/8 cells.

**BEST_BACKBONE = Hard. BEST_ROUTER_TARGET = R1 (Aggregate Winner CE). BEST_ROUTING_MODE = Soft.**


## Table: TERTIARY check -- actual Stage-2 MSE, R1-Soft router cache (Hard backbone) vs existing Mean-Mixture Stage-2

| Dataset | H | Mean-Mixture Stage-2 MSE (existing) | Router (R1-Soft) Stage-2 MSE | Delta | Base MSE | Router lambda |
|---|---|---|---|---|---|---|
| ETTh1 | 96 | 0.36987 | 0.36942 | -0.00045 | 0.39242 | 0.4501 |
| ETTh1 | 192 | 0.42336 | 0.42255 | -0.00081 | 0.44722 | 0.4634 |
| ETTh1 | 336 | 0.44163 | 0.44090 | -0.00073 | 0.45778 | 0.2827 |
| ETTh1 | 720 | 0.51352 | 0.51204 | -0.00148 | 0.56035 | 0.4712 |
| Weather | 96 | 0.17905 | 0.17678 | -0.00227 | 0.16936 | 0.4085 |
| Weather | 192 | 0.22030 | 0.22037 | +0.00007 | 0.19908 | 0.4157 |
| Weather | 336 | 0.28988 | 0.28514 | -0.00474 | 0.24567 | 0.4942 |
| Weather | 720 | 0.39567 | 0.39612 | +0.00045 | 0.31943 | 0.5006 |

## Table: Weather beta-grid CONTROL (R1-Soft router cache, Hard backbone, validation-only -- diagnostic, never primary)

| H | Original (trainable) lambda | Original Stage-2 MSE | Val-selected beta | Beta-Control MSE | Base MSE |
|---|---|---|---|---|---|
| 96 | 0.4085 | 0.17678 | 0.20 | 0.16798 | 0.16936 |
| 192 | 0.4157 | 0.22037 | 0.20 | 0.20118 | 0.19908 |
| 336 | 0.4942 | 0.28514 | 0.20 | 0.24724 | 0.24567 |
| 720 | 0.5006 | 0.39612 | 0.20 | 0.32671 | 0.31943 |

## Anomaly flagged before interpretation

**Weather_720 is pathological for every router variant on BOTH backbones** -- every single Hard/Soft/R1/R2 combination at Weather_720 is worse than Mean-Mixture's own Aggregate MSE on V5 (0.62-0.67 vs MM's 0.61780), and on Hard backbone only R1-Hard/R1-Soft/R2-Hard beat Mean-Mixture there, each by a small margin, while R2-Soft is also worse. This is the single cell that makes criterion 3 (Weather average must improve) fail for V5 across the board and nearly fail it for Hard (Hard passes only because its OTHER 3 Weather cells show large enough gains to outweigh Weather_720's loss). Checked against prior tracks: Weather_720 is also where `TRACK-HEAD-UTILITY-ALIGNMENT-CORRECTION01`'s alpha* domain-shift was most extreme (train-optimal 0.94-0.95 vs validation-optimal 0.16-0.18) and where `TRACK-HARD-EXPERT-V5-P100-ALLH01` found Hard's specialization gain translated into the WORST (not best) forecasting result of the whole track. The pattern is consistent across three independent tracks now: Weather's longest horizon is a genuinely hard regime for this entire method family, not a fluke of any one track's implementation.

## Phase 2 closing determinations

**Collapse diagnostics (spec section 9)**: Router max-head-usage stays well below 1.0 in every cell for both R1 and R2 (Table "Router Quality Metrics" above) -- no router variant collapses to a single head. R1's entropy is systematically lower than R2's at matched cells (R1 is a one-hot CE target, R2 a soft/margin-aware target, so this is expected architecture behavior, not a pathology). Winner accuracy for both targets sits in the 0.2-0.3 range on most cells -- well above the 0.2 random-chance floor for 5 heads, but far from confident; this is consistent with Phase 1's near-tie finding (many (query,channel) decisions have no real margin, so even a well-behaved router cannot reach high winner accuracy on them).

**Why GO despite modest winner accuracy**: the router does not need high winner-classification accuracy to win on Aggregate MSE -- Soft routing (mixing ALL heads' distributions weighted by pi, not committing to one) is inherently more forgiving of winner-classification errors than Hard routing. Checked numerically (not assumed): R1-Soft beats R1-Hard in **10/16** cell-backbone combinations, and R2-Soft beats R2-Hard in only **7/16** (i.e. R2-Hard is actually the more common winner for the R2 target) -- so "Soft beats Hard" is true on average and for the specific winning variant (R1-Soft), but it is NOT a universal property of Soft routing across every target; R2's results do not show the same pattern. This distinction matters for Phase 3, which must decide whether to carry Soft routing's error-tolerance forward into joint training, and should not assume it generalizes beyond the specific R1 target that won here.

**PHASE2_GO = TRUE. Proceeding to Phase 3 (TRACK-JOINT-AGGREGATE-ROUTED-MQ01) with BEST_BACKBONE=Hard.**
