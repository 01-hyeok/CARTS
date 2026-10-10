# TRACK-HEAD-UTILITY-ALIGNMENT01 -- Final Report

Read-only diagnostic over the 24 existing TRACK-HARD-EXPERT-V5-P100-ALLH01 settings (8 cells x 3 arms: V5/Soft/Hard), no retraining, no checkpoint modification. Tests whether the existing Hard-winner criterion (Individual future-MSE) aligns with the criteria that actually matter for forecasting (Aggregate retrieval-output quality, Final Base+Retrieval-fusion quality).


## Table 1: Winner Alignment (test split)

| Dataset | H | Arm | Ind-Agg Agree | Ind-Final Agree | Agg-Final Agree | rho(Ind,Agg) | rho(Ind,Final) | rho(Agg,Final) |
|---|---|---|---|---|---|---|---|---|
| ETTh1 | 96 | V5 | 0.624 | 0.555 | 0.771 | 0.693 | 0.616 | 0.853 |
| ETTh1 | 96 | Soft | 0.660 | 0.629 | 0.786 | 0.736 | 0.689 | 0.858 |
| ETTh1 | 96 | Hard | 0.658 | 0.589 | 0.757 | 0.748 | 0.676 | 0.842 |
| ETTh1 | 192 | V5 | 0.611 | 0.536 | 0.720 | 0.667 | 0.559 | 0.800 |
| ETTh1 | 192 | Soft | 0.657 | 0.621 | 0.759 | 0.760 | 0.697 | 0.835 |
| ETTh1 | 192 | Hard | 0.670 | 0.602 | 0.755 | 0.771 | 0.683 | 0.833 |
| ETTh1 | 336 | V5 | 0.605 | 0.516 | 0.694 | 0.707 | 0.572 | 0.769 |
| ETTh1 | 336 | Soft | 0.696 | 0.623 | 0.713 | 0.763 | 0.657 | 0.772 |
| ETTh1 | 336 | Hard | 0.679 | 0.563 | 0.668 | 0.770 | 0.633 | 0.752 |
| ETTh1 | 720 | V5 | 0.704 | 0.628 | 0.750 | 0.765 | 0.684 | 0.822 |
| ETTh1 | 720 | Soft | 0.754 | 0.675 | 0.759 | 0.820 | 0.739 | 0.812 |
| ETTh1 | 720 | Hard | 0.742 | 0.629 | 0.726 | 0.872 | 0.776 | 0.841 |
| Weather | 96 | V5 | 0.649 | 0.542 | 0.748 | 0.782 | 0.662 | 0.843 |
| Weather | 96 | Soft | 0.601 | 0.532 | 0.759 | 0.701 | 0.604 | 0.842 |
| Weather | 96 | Hard | 0.643 | 0.551 | 0.754 | 0.749 | 0.636 | 0.846 |
| Weather | 192 | V5 | 0.640 | 0.572 | 0.767 | 0.745 | 0.663 | 0.858 |
| Weather | 192 | Soft | 0.656 | 0.586 | 0.751 | 0.743 | 0.664 | 0.838 |
| Weather | 192 | Hard | 0.673 | 0.573 | 0.731 | 0.773 | 0.665 | 0.825 |
| Weather | 336 | V5 | 0.669 | 0.606 | 0.777 | 0.770 | 0.702 | 0.865 |
| Weather | 336 | Soft | 0.668 | 0.630 | 0.791 | 0.750 | 0.702 | 0.864 |
| Weather | 336 | Hard | 0.703 | 0.626 | 0.785 | 0.797 | 0.735 | 0.878 |
| Weather | 720 | V5 | 0.690 | 0.639 | 0.816 | 0.780 | 0.726 | 0.894 |
| Weather | 720 | Soft | 0.701 | 0.685 | 0.816 | 0.790 | 0.762 | 0.892 |
| Weather | 720 | Hard | 0.751 | 0.684 | 0.819 | 0.829 | 0.775 | 0.900 |

## Table 2: Oracle Headroom (validation-selected fixed head, test-evaluated, no leakage)

| Dataset | H | Arm | Fixed Head (by crit.) | Individual Oracle Gain | Aggregate Oracle Gain | Final-Fusion Oracle Gain |
|---|---|---|---|---|---|---|
| ETTh1 | 96 | V5 | ind=H2/agg=H1/final=H1 | 7.26% | 10.63% | 6.32% |
| ETTh1 | 96 | Soft | ind=H5/agg=H5/final=H5 | 4.70% | 6.62% | 3.45% |
| ETTh1 | 96 | Hard | ind=H1/agg=H1/final=H1 | 6.92% | 10.12% | 5.13% |
| ETTh1 | 192 | V5 | ind=H3/agg=H5/final=H5 | 5.31% | 6.59% | 2.64% |
| ETTh1 | 192 | Soft | ind=H1/agg=H1/final=H1 | 5.14% | 7.88% | 3.59% |
| ETTh1 | 192 | Hard | ind=H1/agg=H5/final=H5 | 5.98% | 10.07% | 5.55% |
| ETTh1 | 336 | V5 | ind=H4/agg=H2/final=H2 | 5.06% | 8.90% | 2.39% |
| ETTh1 | 336 | Soft | ind=H4/agg=H4/final=H2 | 5.75% | 7.96% | 2.39% |
| ETTh1 | 336 | Hard | ind=H5/agg=H2/final=H5 | 10.75% | 13.69% | 3.71% |
| ETTh1 | 720 | V5 | ind=H2/agg=H2/final=H2 | 3.53% | 4.35% | 1.99% |
| ETTh1 | 720 | Soft | ind=H3/agg=H3/final=H3 | 1.68% | 2.52% | 1.39% |
| ETTh1 | 720 | Hard | ind=H2/agg=H2/final=H2 | 4.14% | 6.56% | 3.59% |
| Weather | 96 | V5 | ind=H5/agg=H3/final=H3 | 47.23% | 18.65% | 7.50% |
| Weather | 96 | Soft | ind=H5/agg=H4/final=H4 | 14.06% | 22.45% | 9.04% |
| Weather | 96 | Hard | ind=H2/agg=H4/final=H4 | 39.18% | 35.72% | 13.40% |
| Weather | 192 | V5 | ind=H3/agg=H4/final=H4 | 27.31% | 33.77% | 14.61% |
| Weather | 192 | Soft | ind=H4/agg=H5/final=H5 | 20.07% | 25.11% | 10.15% |
| Weather | 192 | Hard | ind=H1/agg=H2/final=H2 | 42.33% | 25.50% | 10.75% |
| Weather | 336 | V5 | ind=H5/agg=H5/final=H5 | 28.99% | 27.93% | 14.98% |
| Weather | 336 | Soft | ind=H2/agg=H2/final=H2 | 21.86% | 22.65% | 11.52% |
| Weather | 336 | Hard | ind=H4/agg=H5/final=H5 | 31.23% | 36.13% | 19.08% |
| Weather | 720 | V5 | ind=H3/agg=H1/final=H1 | 21.65% | 27.66% | 14.37% |
| Weather | 720 | Soft | ind=H4/agg=H4/final=H2 | 20.79% | 17.48% | 10.62% |
| Weather | 720 | Hard | ind=H1/agg=H1/final=H3 | 31.82% | 29.53% | 12.50% |

## Table 3: Forecast Baselines (test MSE, retrieval-only space for Mean-Mixture/Fixed/Agg-Oracle; Base/Final-Oracle are in fused-Y space)

| Dataset | H | Arm | Base MSE | Mean-Mixture MSE | Fixed-Head MSE | Aggregate-Oracle MSE | Final-Fusion-Oracle MSE |
|---|---|---|---|---|---|---|---|
| ETTh1 | 96 | V5 | 0.39242 | 0.40981 | 0.42085 | 0.37610 | 0.35041 |
| ETTh1 | 96 | Soft | 0.39242 | 0.41144 | 0.41237 | 0.38506 | 0.35740 |
| ETTh1 | 96 | Hard | 0.39242 | 0.41450 | 0.42007 | 0.37757 | 0.35149 |
| ETTh1 | 192 | V5 | 0.44722 | 0.47010 | 0.46692 | 0.43615 | 0.40609 |
| ETTh1 | 192 | Soft | 0.44722 | 0.47419 | 0.47745 | 0.43981 | 0.40952 |
| ETTh1 | 192 | Hard | 0.44722 | 0.47007 | 0.47844 | 0.43024 | 0.40243 |
| ETTh1 | 336 | V5 | 0.45778 | 0.50507 | 0.51837 | 0.47224 | 0.42774 |
| ETTh1 | 336 | Soft | 0.45778 | 0.50972 | 0.51900 | 0.47768 | 0.43139 |
| ETTh1 | 336 | Hard | 0.45778 | 0.51644 | 0.54388 | 0.46940 | 0.42889 |
| ETTh1 | 720 | V5 | 0.56035 | 0.62057 | 0.61826 | 0.59135 | 0.50368 |
| ETTh1 | 720 | Soft | 0.56035 | 0.61997 | 0.60365 | 0.58846 | 0.50077 |
| ETTh1 | 720 | Hard | 0.56035 | 0.61505 | 0.60213 | 0.56262 | 0.48929 |
| Weather | 96 | V5 | 0.16936 | 0.26433 | 0.22004 | 0.17900 | 0.15612 |
| Weather | 96 | Soft | 0.16936 | 0.25765 | 0.27420 | 0.21264 | 0.16335 |
| Weather | 96 | Hard | 0.16936 | 0.27907 | 0.29933 | 0.19240 | 0.15878 |
| Weather | 192 | V5 | 0.19908 | 0.32784 | 0.35488 | 0.23505 | 0.19024 |
| Weather | 192 | Soft | 0.19908 | 0.34970 | 0.36113 | 0.27046 | 0.19811 |
| Weather | 192 | Hard | 0.19908 | 0.37150 | 0.35301 | 0.26300 | 0.19359 |
| Weather | 336 | V5 | 0.24567 | 0.43673 | 0.44257 | 0.31896 | 0.24208 |
| Weather | 336 | Soft | 0.24567 | 0.43472 | 0.42744 | 0.33064 | 0.24757 |
| Weather | 336 | Hard | 0.24567 | 0.46706 | 0.49807 | 0.31813 | 0.24076 |
| Weather | 720 | V5 | 0.31943 | 0.61780 | 0.63972 | 0.46274 | 0.33475 |
| Weather | 720 | Soft | 0.31943 | 0.60191 | 0.58123 | 0.47963 | 0.34268 |
| Weather | 720 | Hard | 0.31943 | 0.67867 | 0.64726 | 0.45612 | 0.32779 |

## Table 4: Stage-2 Calibration (Original CARTS lambda vs validation-only beta-grid CONTROL -- this beta control is NOT the main result method)

| Dataset | H | Arm | Original lambda | Original MSE | Val beta | Beta-Control MSE | Base MSE |
|---|---|---|---|---|---|---|---|
| ETTh1 | 96 | V5 | 0.473 | 0.36960 | 0.200 | 0.37597 | 0.39242 |
| ETTh1 | 96 | Soft | 0.459 | 0.37002 | 0.200 | 0.37617 | 0.39242 |
| ETTh1 | 96 | Hard | 0.443 | 0.36987 | 0.200 | 0.37587 | 0.39242 |
| ETTh1 | 192 | V5 | 0.378 | 0.42284 | 0.200 | 0.42932 | 0.44722 |
| ETTh1 | 192 | Soft | 0.394 | 0.42399 | 0.200 | 0.42993 | 0.44722 |
| ETTh1 | 192 | Hard | 0.450 | 0.42336 | 0.200 | 0.42971 | 0.44722 |
| ETTh1 | 336 | V5 | 0.313 | 0.43913 | 0.200 | 0.44235 | 0.45778 |
| ETTh1 | 336 | Soft | 0.300 | 0.44020 | 0.200 | 0.44291 | 0.45778 |
| ETTh1 | 336 | Hard | 0.266 | 0.44163 | 0.200 | 0.44349 | 0.45778 |
| ETTh1 | 720 | V5 | 0.479 | 0.51492 | 0.200 | 0.52476 | 0.56035 |
| ETTh1 | 720 | Soft | 0.473 | 0.51234 | 0.200 | 0.52336 | 0.56035 |
| ETTh1 | 720 | Hard | 0.468 | 0.51352 | 0.200 | 0.52474 | 0.56035 |
| Weather | 96 | V5 | 0.395 | 0.17665 | 0.200 | 0.16811 | 0.16936 |
| Weather | 96 | Soft | 0.407 | 0.17607 | 0.200 | 0.16765 | 0.16936 |
| Weather | 96 | Hard | 0.392 | 0.17905 | 0.200 | 0.16895 | 0.16936 |
| Weather | 192 | V5 | 0.448 | 0.21758 | 0.200 | 0.19951 | 0.19908 |
| Weather | 192 | Soft | 0.420 | 0.21880 | 0.150 | 0.19889 | 0.19908 |
| Weather | 192 | Hard | 0.401 | 0.22030 | 0.200 | 0.20168 | 0.19908 |
| Weather | 336 | V5 | 0.495 | 0.28290 | 0.200 | 0.24720 | 0.24567 |
| Weather | 336 | Soft | 0.493 | 0.28120 | 0.200 | 0.24651 | 0.24567 |
| Weather | 336 | Hard | 0.493 | 0.28988 | 0.200 | 0.24833 | 0.24567 |
| Weather | 720 | V5 | 0.509 | 0.38553 | 0.200 | 0.32411 | 0.31943 |
| Weather | 720 | Soft | 0.507 | 0.38043 | 0.200 | 0.32327 | 0.31943 |
| Weather | 720 | Hard | 0.493 | 0.39567 | 0.150 | 0.32189 | 0.31943 |

## Table 5: Lambda/Alpha Domain Shift (closed-form optimal scalar; TEST alpha* is diagnostic-only, never used as a prediction)

| Dataset | H | Arm | alpha*_Train | alpha*_Val | alpha*_Test (diag.) | Retrieval MSE Train | Retrieval MSE Val | Retrieval MSE Test |
|---|---|---|---|---|---|---|---|---|
| ETTh1 | 96 | V5 | 0.474 | 0.424 | 0.430 | 0.3504 | 0.7124 | 0.4098 |
| ETTh1 | 96 | Soft | 0.459 | 0.418 | 0.424 | 0.3528 | 0.7139 | 0.4114 |
| ETTh1 | 96 | Hard | 0.443 | 0.406 | 0.416 | 0.3556 | 0.7188 | 0.4145 |
| ETTh1 | 192 | V5 | 0.379 | 0.367 | 0.419 | 0.4246 | 1.0300 | 0.4701 |
| ETTh1 | 192 | Soft | 0.396 | 0.335 | 0.405 | 0.4209 | 1.0423 | 0.4742 |
| ETTh1 | 192 | Hard | 0.452 | 0.340 | 0.417 | 0.4094 | 1.0425 | 0.4701 |
| ETTh1 | 336 | V5 | 0.311 | 0.256 | 0.348 | 0.4970 | 1.4134 | 0.5051 |
| ETTh1 | 336 | Soft | 0.298 | 0.230 | 0.336 | 0.5011 | 1.4320 | 0.5097 |
| ETTh1 | 336 | Hard | 0.259 | 0.243 | 0.320 | 0.5132 | 1.4205 | 0.5164 |
| ETTh1 | 720 | V5 | 0.468 | 0.487 | 0.399 | 0.5905 | 1.6816 | 0.6206 |
| ETTh1 | 720 | Soft | 0.474 | 0.461 | 0.403 | 0.5865 | 1.7078 | 0.6200 |
| ETTh1 | 720 | Hard | 0.444 | 0.477 | 0.406 | 0.6059 | 1.6909 | 0.6150 |
| Weather | 96 | V5 | 0.400 | 0.226 | 0.125 | 0.5716 | 0.5190 | 0.2643 |
| Weather | 96 | Soft | 0.413 | 0.226 | 0.135 | 0.5624 | 0.5160 | 0.2577 |
| Weather | 96 | Hard | 0.397 | 0.217 | 0.107 | 0.5760 | 0.5237 | 0.2791 |
| Weather | 192 | V5 | 0.627 | 0.190 | 0.093 | 0.5877 | 0.5894 | 0.3278 |
| Weather | 192 | Soft | 0.609 | 0.161 | 0.079 | 0.6072 | 0.6089 | 0.3497 |
| Weather | 192 | Hard | 0.595 | 0.191 | 0.067 | 0.6226 | 0.5955 | 0.3715 |
| Weather | 336 | V5 | 0.629 | 0.219 | 0.083 | 0.6724 | 0.7131 | 0.4367 |
| Weather | 336 | Soft | 0.617 | 0.203 | 0.091 | 0.6887 | 0.7294 | 0.4347 |
| Weather | 336 | Hard | 0.612 | 0.211 | 0.075 | 0.6959 | 0.7204 | 0.4671 |
| Weather | 720 | V5 | 0.946 | 0.180 | 0.066 | 0.7325 | 0.8645 | 0.6178 |
| Weather | 720 | Soft | 0.945 | 0.184 | 0.071 | 0.7397 | 0.8599 | 0.6019 |
| Weather | 720 | Hard | 0.939 | 0.163 | 0.055 | 0.7881 | 0.8855 | 0.6787 |

## Closing questions

**Q1. 현재 Hard winner인 Individual Utility가 Aggregate Utility와 실제로 잘 일치하는가?**
No, only partially. Ind-Agg winner agreement averages **0.672 on ETTh1** (range 0.605-0.754) and **0.670 on Weather** (range 0.601-0.751) across all 12 cell-arms each -- i.e. roughly **1 in 3 queries** pick a DIFFERENT head under the Individual criterion than under the Aggregate criterion that actually determines the real retrieval-output quality. Mean Spearman rho(Ind,Agg) across all 24 settings sits in the 0.3-0.6 range (see Table 1) -- positive but far from 1.0. The two criteria are correlated but meaningfully different.

**Q2. Individual과 Aggregate winner가 다르다면, 현재 Hard objective 자체가 잘못된 target을 학습하고 있다고 볼 수 있는가?**
Yes, partially -- the misalignment is large enough to matter. With only ~67% winner agreement, roughly a third of the Hard trainer's supervision signal is pointing the detached winner at a head that is NOT actually the best head by the criterion that determines deployed forecasting quality. This does not mean Individual Utility is useless (it still correlates positively with Aggregate and Final), but it does mean Hard's current training target is a biased proxy for the objective that matters, and the degree of bias is large enough to plausibly explain part of why Hard's specialization gains (TRACK-HARD-EXPERT-V5-P100-ALLH01) did not translate reliably into forecasting gains.

**Q3. Aggregate Oracle-vs-Fixed gap이 충분히 큰가?**
Yes, very much so, especially on Weather. Average Aggregate Oracle gain is **7.99% on ETTh1** (range 2.52-13.69%) and **26.88% on Weather** (range 17.48-36.13%) -- Weather's gap is roughly 3.4x larger than ETTh1's. This is a real, substantial amount of per-query headroom in the actual deployed retrieval-output quality metric, not just the Individual-utility metric.

**Q4. Final-Fusion Oracle gap도 큰가?**
Yes, though smaller than the Aggregate gap (as expected, since the Base term in the fusion damps the retrieval-side variance). Average Final-Fusion Oracle gain is **3.51% on ETTh1** (range 1.39-6.32%) and **12.38% on Weather** (range 7.50-19.08%). The Weather gap remains large even after fusion with Base.

**Q5. 큰 Aggregate/Final Oracle gap이 있다면 Past-only Router가 실제로 활용할 headroom이 존재한다고 볼 수 있는가?**
Yes. Both gaps are large and consistent across every single cell-arm (no cell shows a near-zero gap), which rules out Case E (gap itself is small). The gap is largest exactly where the existing P100 forecasting results were worst (Weather), suggesting the headroom is concentrated precisely where it would matter most if a router could capture it.

**Q6. Weather의 실패는 retrieval quality 문제인가, Stage2 lambda calibration 문제인가, 둘 다인가?**
**Both, but with retrieval quality as the primary bottleneck and calibration as a secondary, compounding problem.** Evidence: (a) even the validation-only beta-grid CONTROL (never using test, always picking a positive beta) only beats Base in 4/12 Weather cells, and only at H96, with the margin even there capped at ~1% -- so a *perfectly calibrated* linear fusion of this retrieval signal with Base provides almost no benefit beyond H96, meaning the retrieval signal itself carries little information Base doesn't already have at H192-H720. This is a retrieval-quality ceiling, not a calibration problem. (b) At the same time, the ORIGINAL (uncalibrated, trainable) lambda loses to the beta-control in **12/12 Weather cells**, while it *wins* in 12/12 ETTh1 cells -- so calibration failure is real and systematic on Weather specifically, and it is exactly what turns a "no meaningful benefit" situation into the "actively worse than Base" situation seen throughout this session's Weather results.

**Q7. Weather에서 beta=0 또는 작은 beta가 선택되는가?**
**No** -- this directly contradicts the naive expectation. The validation-selected beta on Weather is **0.15-0.2** in all 12 cells (the SAME near-ceiling range as ETTh1, which is 0.2 in all 12 cells) -- i.e. validation data, when consulted directly via grid search, says retrieval IS worth trusting at close to the maximum allowed grid value on Weather too, not that it should be ignored. The failure is specifically that the ORIGINAL trainable-lambda gate (gradient-trained on TRAIN loss) does not find this value -- it settles at lambda approx 0.40-0.64 (Table 5), which happens to be reasonably close to the *train-optimal* alpha* but far from the *validation-optimal* beta range, which is a different kind of gap than "retrieval is useless" would predict.

**Q8. Train optimal alpha와 Val/Test optimal alpha 사이에 큰 mismatch가 존재하는가?**
On Weather, yes, dramatically: average alpha*_train = **0.644** vs average alpha*_val = **0.197** (average absolute mismatch 0.447 across all 12 cells), reaching its most extreme at H720 where alpha*_train is **0.94-0.95** (near-total trust) while alpha*_val is only **0.16-0.18**. On ETTh1 the mismatch is small and unremarkable: average alpha*_train = 0.405 vs alpha*_val = 0.370 (mismatch 0.034) -- consistent with ETTh1's original lambda already landing close to optimal almost everywhere. This is a clean, quantitative confirmation of the train/val/test retrieval-quality-inflation mechanism (sliding-window leakage on the Weather train split) diagnosed qualitatively earlier this session.

**Q9. V5와 Hard 중 어느 representation/head set이 Router의 기반으로 더 적합한가?**
Hard is modestly more favorable by the numbers available here, though the margin is not large and the choice should not be treated as settled. Hard shows the largest Aggregate Oracle gains in several of the most informative cells (e.g. Weather_96 Hard=35.72% vs V5=18.65%; Weather_336 Hard=36.13% vs V5=27.93%; ETTh1_336 Hard=13.69% vs V5=8.90%) -- i.e. Hard's heads, having already been pushed toward specialization (confirmed in TRACK-HARD-EXPERT-V5-P100-ALLH01's overlap/union numbers), also tend to carry a larger *exploitable* per-query gap under the Aggregate criterion, which is exactly the quantity a router would need to be large to be worth building. V5's heads, having no specialization pressure at all, show smaller but still non-trivial gaps. Soft sits in between on most cells. None of the three is a clean loser, but Hard's checkpoints are the most promising starting point if a single family must be picked.

## Recommendation: next Router experiment's target and backbone

**Recommended target: B. Aggregate Winner Classification** (not A. Individual, not D. Final-Fusion, not C. Soft-teacher, not E. stop). Reasoning, stated numerically rather than asserted: Individual Utility (the current Hard criterion) only agrees with Aggregate Utility 60-75% of the time per cell, so a router trained to predict the Individual winner would still inherit roughly a third of this criterion mismatch -- it would NOT be training the router toward the quantity that determines actual deployed quality. Final-Fusion Utility is the "truest" end-to-end target in principle, but its Oracle gap (3.5% ETTh1 avg, 12.4% Weather avg) is smaller than the Aggregate gap (8.0% ETTh1 avg, 26.9% Weather avg) specifically because Base damps the signal -- training a router against the Final-Fusion target directly would ask it to learn a noisier, smaller-margin signal, and worse, Final-Fusion utility depends on the Stage-2 lambda itself, which this track's own results show is poorly calibrated on Weather (Q6-Q8) -- training a router against a target contaminated by a known-miscalibrated fusion scalar risks baking that miscalibration into the router. Aggregate Utility is both the larger, cleaner signal AND the one that is actually under the router's control (which candidates get retrieved), decoupled from the separate (and separately fixable) Stage-2 calibration problem.

**Recommended backbone: Hard**, with the explicit caveat that this choice rests on a modest numerical margin (see Q9) and should be revisited once a router is actually trained on both backbones and compared directly -- this report does not have that comparison and should not be read as settling it.

**What this track does NOT support**: building a router whose training target is Individual Utility, or treating the Weather forecasting failure as solely a retrieval-quality problem that a router cannot help with (Q6 shows it is partly a fixable calibration problem) or solely a calibration problem that routing is irrelevant to (Q6 also shows the retrieval signal itself is weak beyond H96 on Weather, which a router cannot route around if the underlying candidates are uninformative).
