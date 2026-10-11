# TRACK-HEAD-UTILITY-ALIGNMENT-CORRECTION01 -- Phase 1 Report

**Stage-2 Mode: Original CARTS Trainable Global Lambda** (unchanged from all prior tracks; this report's beta-grid numbers are a CONTROL/diagnostic only, never the primary result).

Corrects two issues in TRACK-HEAD-UTILITY-ALIGNMENT01: (A) winners are now computed at the REAL Hard-trainer granularity, per (query, channel) -- confirmed bit-identical (mod floating-point tie-break noise on exact ties) to the actual `train_hard_expert_v5_p100_allh01.py` winner via a direct integration test against a real checkpoint and real data. (B) the Stage-2 calibration CONTROL beta grid is extended to {0,.025,.05,.075,.10,.15,.20,.30,.40,.50} (this control only -- the shared grid used as the PRIMARY result elsewhere is untouched). Read-only, no retraining, 24/24 settings (8 cells x V5/Soft/Hard).


## Table: Global vs Per-Channel Winner Agreement/Correlation (test split)

| Dataset | H | Arm | Global Agree | Per-Channel Agree | Global rho | Per-Channel rho |
|---|---|---|---|---|---|---|
| ETTh1 | 96 | V5 | 0.624 | 0.578 | 0.693 | 0.654 |
| ETTh1 | 96 | Soft | 0.659 | 0.604 | 0.736 | 0.634 |
| ETTh1 | 96 | Hard | 0.658 | 0.593 | 0.748 | 0.665 |
| ETTh1 | 192 | V5 | 0.611 | 0.586 | 0.667 | 0.656 |
| ETTh1 | 192 | Soft | 0.657 | 0.617 | 0.760 | 0.657 |
| ETTh1 | 192 | Hard | 0.670 | 0.602 | 0.771 | 0.696 |
| ETTh1 | 336 | V5 | 0.605 | 0.600 | 0.707 | 0.667 |
| ETTh1 | 336 | Soft | 0.696 | 0.647 | 0.763 | 0.681 |
| ETTh1 | 336 | Hard | 0.679 | 0.623 | 0.770 | 0.715 |
| ETTh1 | 720 | V5 | 0.704 | 0.657 | 0.765 | 0.696 |
| ETTh1 | 720 | Soft | 0.754 | 0.708 | 0.820 | 0.732 |
| ETTh1 | 720 | Hard | 0.742 | 0.680 | 0.872 | 0.768 |
| Weather | 96 | V5 | 0.649 | 0.627 | 0.782 | 0.726 |
| Weather | 96 | Soft | 0.601 | 0.641 | 0.701 | 0.685 |
| Weather | 96 | Hard | 0.643 | 0.632 | 0.749 | 0.715 |
| Weather | 192 | V5 | 0.640 | 0.639 | 0.745 | 0.734 |
| Weather | 192 | Soft | 0.656 | 0.648 | 0.743 | 0.706 |
| Weather | 192 | Hard | 0.673 | 0.646 | 0.773 | 0.742 |
| Weather | 336 | V5 | 0.669 | 0.654 | 0.770 | 0.746 |
| Weather | 336 | Soft | 0.668 | 0.667 | 0.750 | 0.713 |
| Weather | 336 | Hard | 0.703 | 0.658 | 0.797 | 0.758 |
| Weather | 720 | V5 | 0.690 | 0.655 | 0.780 | 0.725 |
| Weather | 720 | Soft | 0.701 | 0.675 | 0.790 | 0.709 |
| Weather | 720 | Hard | 0.751 | 0.659 | 0.829 | 0.748 |

## Table: Global vs Per-Channel Oracle Headroom (Aggregate criterion, validation-selected fixed head(s), test-evaluated, no leakage)

| Dataset | H | Arm | Global Fixed MSE | Global Oracle MSE | Global Gain % | Per-Channel Fixed MSE | Per-Channel Oracle MSE | Per-Channel Gain % |
|---|---|---|---|---|---|---|---|---|
| ETTh1 | 96 | V5 | 0.42085 | 0.37610 | 10.63% | 0.41900 | 0.33904 | 19.08% |
| ETTh1 | 96 | Soft | 0.41237 | 0.38506 | 6.62% | 0.41333 | 0.36434 | 11.85% |
| ETTh1 | 96 | Hard | 0.42007 | 0.37757 | 10.12% | 0.42261 | 0.34374 | 18.66% |
| ETTh1 | 192 | V5 | 0.46692 | 0.43615 | 6.59% | 0.47886 | 0.39829 | 16.83% |
| ETTh1 | 192 | Soft | 0.47745 | 0.43981 | 7.88% | 0.47216 | 0.41449 | 12.21% |
| ETTh1 | 192 | Hard | 0.47844 | 0.43024 | 10.07% | 0.47976 | 0.39426 | 17.82% |
| ETTh1 | 336 | V5 | 0.51837 | 0.47224 | 8.90% | 0.51941 | 0.43951 | 15.38% |
| ETTh1 | 336 | Soft | 0.51900 | 0.47768 | 7.96% | 0.51971 | 0.45465 | 12.52% |
| ETTh1 | 336 | Hard | 0.54388 | 0.46940 | 13.69% | 0.54853 | 0.43052 | 21.51% |
| ETTh1 | 720 | V5 | 0.61826 | 0.59135 | 4.35% | 0.62230 | 0.56967 | 8.46% |
| ETTh1 | 720 | Soft | 0.60365 | 0.58846 | 2.52% | 0.61074 | 0.57090 | 6.52% |
| ETTh1 | 720 | Hard | 0.60213 | 0.56262 | 6.56% | 0.63303 | 0.52806 | 16.58% |
| Weather | 96 | V5 | 0.22004 | 0.17900 | 18.65% | 0.21090 | 0.13428 | 36.33% |
| Weather | 96 | Soft | 0.27420 | 0.21264 | 22.45% | 0.25311 | 0.18295 | 27.72% |
| Weather | 96 | Hard | 0.29933 | 0.19240 | 35.72% | 0.25877 | 0.14726 | 43.09% |
| Weather | 192 | V5 | 0.35488 | 0.23505 | 33.77% | 0.30027 | 0.16691 | 44.41% |
| Weather | 192 | Soft | 0.36113 | 0.27046 | 25.11% | 0.33395 | 0.22497 | 32.63% |
| Weather | 192 | Hard | 0.35301 | 0.26300 | 25.50% | 0.34184 | 0.19409 | 43.22% |
| Weather | 336 | V5 | 0.44257 | 0.31896 | 27.93% | 0.44113 | 0.23514 | 46.70% |
| Weather | 336 | Soft | 0.42744 | 0.33064 | 22.65% | 0.42651 | 0.28089 | 34.14% |
| Weather | 336 | Hard | 0.49807 | 0.31813 | 36.13% | 0.45782 | 0.23050 | 49.65% |
| Weather | 720 | V5 | 0.63972 | 0.46274 | 27.66% | 0.55067 | 0.39017 | 29.15% |
| Weather | 720 | Soft | 0.58123 | 0.47963 | 17.48% | 0.58150 | 0.43490 | 25.21% |
| Weather | 720 | Hard | 0.64726 | 0.45612 | 29.53% | 0.71796 | 0.36158 | 49.64% |

## Table: Degeneracy Diagnostics (per-channel, test split)

| Dataset | H | Arm | Max Head Usage | Min Head Usage | Normalized Entropy | Near-Tie Fraction (Agg margin) |
|---|---|---|---|---|---|---|
| ETTh1 | 96 | V5 | 0.238 | 0.157 | 0.993 | 0.020 |
| ETTh1 | 96 | Soft | 0.262 | 0.130 | 0.984 | 0.131 |
| ETTh1 | 96 | Hard | 0.217 | 0.172 | 0.998 | 0.030 |
| ETTh1 | 192 | V5 | 0.252 | 0.155 | 0.989 | 0.040 |
| ETTh1 | 192 | Soft | 0.230 | 0.167 | 0.996 | 0.086 |
| ETTh1 | 192 | Hard | 0.219 | 0.183 | 0.998 | 0.020 |
| ETTh1 | 336 | V5 | 0.227 | 0.153 | 0.994 | 0.053 |
| ETTh1 | 336 | Soft | 0.310 | 0.096 | 0.947 | 0.137 |
| ETTh1 | 336 | Hard | 0.269 | 0.166 | 0.990 | 0.024 |
| ETTh1 | 720 | V5 | 0.304 | 0.108 | 0.958 | 0.213 |
| ETTh1 | 720 | Soft | 0.259 | 0.111 | 0.974 | 0.290 |
| ETTh1 | 720 | Hard | 0.285 | 0.129 | 0.963 | 0.085 |
| Weather | 96 | V5 | 0.275 | 0.152 | 0.987 | 0.160 |
| Weather | 96 | Soft | 0.306 | 0.151 | 0.979 | 0.233 |
| Weather | 96 | Hard | 0.240 | 0.157 | 0.993 | 0.158 |
| Weather | 192 | V5 | 0.227 | 0.178 | 0.998 | 0.115 |
| Weather | 192 | Soft | 0.244 | 0.177 | 0.996 | 0.182 |
| Weather | 192 | Hard | 0.263 | 0.159 | 0.990 | 0.121 |
| Weather | 336 | V5 | 0.213 | 0.187 | 0.999 | 0.121 |
| Weather | 336 | Soft | 0.256 | 0.164 | 0.992 | 0.200 |
| Weather | 336 | Hard | 0.232 | 0.173 | 0.997 | 0.089 |
| Weather | 720 | V5 | 0.250 | 0.159 | 0.991 | 0.168 |
| Weather | 720 | Soft | 0.316 | 0.141 | 0.972 | 0.281 |
| Weather | 720 | Hard | 0.233 | 0.179 | 0.997 | 0.125 |

## Table: Stage-2 Calibration CONTROL with extended beta grid (0-0.50) -- diagnostic only, never the main result

| Dataset | H | Arm | Original lambda | Original MSE | Val beta (ext. grid) | Hit Grid Ceiling (0.50)? | Beta-Control MSE | Base MSE |
|---|---|---|---|---|---|---|---|---|
| ETTh1 | 96 | V5 | 0.473 | 0.36960 | 0.400 | False | 0.36948 | 0.39242 |
| ETTh1 | 96 | Soft | 0.459 | 0.37002 | 0.400 | False | 0.36994 | 0.39242 |
| ETTh1 | 96 | Hard | 0.443 | 0.36987 | 0.400 | False | 0.36981 | 0.39242 |
| ETTh1 | 192 | V5 | 0.378 | 0.42284 | 0.400 | False | 0.42265 | 0.44722 |
| ETTh1 | 192 | Soft | 0.394 | 0.42399 | 0.300 | False | 0.42554 | 0.44722 |
| ETTh1 | 192 | Hard | 0.450 | 0.42336 | 0.300 | False | 0.42510 | 0.44722 |
| ETTh1 | 336 | V5 | 0.313 | 0.43913 | 0.300 | False | 0.43929 | 0.45778 |
| ETTh1 | 336 | Soft | 0.300 | 0.44020 | 0.200 | False | 0.44291 | 0.45778 |
| ETTh1 | 336 | Hard | 0.266 | 0.44163 | 0.200 | False | 0.44349 | 0.45778 |
| ETTh1 | 720 | V5 | 0.479 | 0.51492 | 0.500 | True | 0.51603 | 0.56035 |
| ETTh1 | 720 | Soft | 0.473 | 0.51234 | 0.500 | True | 0.51373 | 0.56035 |
| ETTh1 | 720 | Hard | 0.468 | 0.51352 | 0.500 | True | 0.51496 | 0.56035 |
| Weather | 96 | V5 | 0.395 | 0.17665 | 0.200 | False | 0.16811 | 0.16936 |
| Weather | 96 | Soft | 0.407 | 0.17607 | 0.200 | False | 0.16765 | 0.16936 |
| Weather | 96 | Hard | 0.392 | 0.17905 | 0.200 | False | 0.16895 | 0.16936 |
| Weather | 192 | V5 | 0.448 | 0.21758 | 0.200 | False | 0.19951 | 0.19908 |
| Weather | 192 | Soft | 0.420 | 0.21880 | 0.150 | False | 0.19889 | 0.19908 |
| Weather | 192 | Hard | 0.401 | 0.22030 | 0.200 | False | 0.20168 | 0.19908 |
| Weather | 336 | V5 | 0.495 | 0.28290 | 0.200 | False | 0.24720 | 0.24567 |
| Weather | 336 | Soft | 0.493 | 0.28120 | 0.200 | False | 0.24651 | 0.24567 |
| Weather | 336 | Hard | 0.493 | 0.28988 | 0.200 | False | 0.24833 | 0.24567 |
| Weather | 720 | V5 | 0.509 | 0.38553 | 0.200 | False | 0.32411 | 0.31943 |
| Weather | 720 | Soft | 0.507 | 0.38043 | 0.200 | False | 0.32327 | 0.31943 |
| Weather | 720 | Hard | 0.493 | 0.39567 | 0.150 | False | 0.32189 | 0.31943 |

## Table: Alpha Domain Shift (closed-form optimal scalar; test alpha* is diagnostic only)

| Dataset | H | Arm | alpha*_Train | alpha*_Val | alpha*_Test (diag.) |
|---|---|---|---|---|---|
| ETTh1 | 96 | V5 | 0.474 | 0.424 | 0.430 |
| ETTh1 | 96 | Soft | 0.459 | 0.418 | 0.424 |
| ETTh1 | 96 | Hard | 0.443 | 0.406 | 0.416 |
| ETTh1 | 192 | V5 | 0.379 | 0.367 | 0.419 |
| ETTh1 | 192 | Soft | 0.396 | 0.335 | 0.405 |
| ETTh1 | 192 | Hard | 0.452 | 0.340 | 0.417 |
| ETTh1 | 336 | V5 | 0.311 | 0.256 | 0.348 |
| ETTh1 | 336 | Soft | 0.298 | 0.230 | 0.336 |
| ETTh1 | 336 | Hard | 0.259 | 0.243 | 0.320 |
| ETTh1 | 720 | V5 | 0.468 | 0.487 | 0.399 |
| ETTh1 | 720 | Soft | 0.474 | 0.461 | 0.403 |
| ETTh1 | 720 | Hard | 0.444 | 0.477 | 0.406 |
| Weather | 96 | V5 | 0.400 | 0.226 | 0.125 |
| Weather | 96 | Soft | 0.413 | 0.226 | 0.135 |
| Weather | 96 | Hard | 0.397 | 0.217 | 0.107 |
| Weather | 192 | V5 | 0.627 | 0.190 | 0.093 |
| Weather | 192 | Soft | 0.609 | 0.161 | 0.079 |
| Weather | 192 | Hard | 0.595 | 0.191 | 0.067 |
| Weather | 336 | V5 | 0.629 | 0.219 | 0.083 |
| Weather | 336 | Soft | 0.617 | 0.203 | 0.091 |
| Weather | 336 | Hard | 0.612 | 0.211 | 0.075 |
| Weather | 720 | V5 | 0.946 | 0.180 | 0.066 |
| Weather | 720 | Soft | 0.945 | 0.184 | 0.071 |
| Weather | 720 | Hard | 0.939 | 0.163 | 0.055 |

## Anomaly flagged before interpretation

**3/24 cells (all ETTh1_720) hit the extended beta grid's ceiling (beta=0.50)** -- the true validation-optimal beta for these cells may lie beyond 0.50, meaning the reported beta-control MSE at these 3 cells is a LOWER BOUND on how much the calibration control can help, not the true optimum. This does not change this Phase's GO/STOP decision (ETTh1_720's global/per-channel oracle gains are the smallest in the whole table, 4-9%, well below the dataset average), but any future work drawing a strong quantitative conclusion specifically about ETTh1_720's calibration gap should extend the grid further before doing so.

## Phase 1 closing determinations (spec section 7)

**A. 실제 per-channel Hard supervision에서도 Ind vs Agg mismatch가 유의미한가?**
Yes, if anything slightly WORSE than the channel-averaged (global) diagnostic suggested. Per-channel Ind-Agg agreement averages **0.625 (ETTh1)** and **0.650 (Weather)** -- both LOWER than the corresponding global-granularity agreement (0.672 ETTh1, 0.670 Weather) from the original (uncorrected) diagnostic. At the exact granularity the real Hard trainer actually supervises on, roughly **37-38% of (query, channel) winner decisions** point at a head that is not the Aggregate-best head for that same (query, channel). The mismatch is real, large, and if anything larger than the original report suggested.

**B. Global Oracle와 Per-channel Oracle 중 어느 쪽의 headroom이 더 큰가?**
Per-Channel, decisively and consistently -- in all 24/24 cells, per-channel oracle gain exceeds global oracle gain (overall average 26.64% vs 17.44%, a 9.2 percentage-point gap). This makes sense structurally: per-channel oracle has strictly more freedom (an independent head choice per channel) than global oracle (one head for the whole query), so this direction was never in doubt, but the SIZE of the gap (channels clearly benefit from different heads, not just the same head applied more cleverly) is the finding that matters for routing design.

**C. Near-tie가 많은가?**
Moderately, and notably more so on Weather. Average near-tie fraction (Aggregate-utility margin, threshold 1e-4) is **9.4% on ETTh1** and **16.3% on Weather** -- i.e. roughly 1 in 6 (query,channel) decisions on Weather have no real margin between the best and second-best head. This is large enough that any router evaluation metric sensitive to exact-winner accuracy (rather than ranking quality) will understate true performance on these near-tie cases, and should be interpreted with that caveat in Phase 2.

**D. Hard CE target을 써도 되는가, 아니면 Soft Aggregate Teacher가 더 타당한가?**
Soft Aggregate Teacher is more defensible, based on this Phase's own near-tie finding (C) combined with the Ind-vs-Agg mismatch (A). A one-hot Hard CE target forces a confident decision exactly in the cases (near-ties) where the data itself does not support one, and does so using a criterion (Individual) that disagrees with the criterion that matters (Aggregate) over a third of the time. A soft target that reflects the actual Aggregate-utility margin naturally down-weights near-tie queries instead of forcing an arbitrary hard choice on them. This is consistent with (but does not by itself prove) the Phase 2 spec's inclusion of both R1 (Aggregate Winner CE) and R2 (Aggregate Soft Teacher) as targets to compare empirically rather than assuming one in advance.

## NEXT_ROUTER_MODE decision (spec section 8)

Per-channel oracle gain (26.64% overall) exceeds global oracle gain (17.44% overall) by **9.2 percentage points** -- far above the 1pp tie-breaking threshold. **NEXT_ROUTER_MODE = CHANNEL_AWARE.** This holds consistently within both datasets individually (ETTh1: 14.79% vs 7.99%, +6.8pp; Weather: 38.49% vs 26.88%, +11.6pp) and in all three arms, so there is no dataset- or arm-specific exception that would argue for GLOBAL instead.

## Phase 1 GO/STOP decision (spec section 9)

Aggregate Oracle gain average is **7.99% (ETTh1, global) / 14.79% (ETTh1, per-channel)** and **26.88% (Weather, global) / 38.49% (Weather, per-channel)** -- both datasets clear the 3% GO threshold by a wide margin under EITHER granularity.

**PHASE1_GO = TRUE. Proceeding to Phase 2.**
