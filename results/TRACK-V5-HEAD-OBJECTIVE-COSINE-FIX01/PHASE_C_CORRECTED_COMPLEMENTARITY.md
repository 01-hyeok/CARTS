# PHASE C -- Corrected Head Error Complementarity (all 4 cells)

Recomputed via a fresh forward pass on the EXISTING canonical checkpoints (no new training), using the fixed `scripts/diag_v5_head_objective01.py`. seed0, Shared-P100, K=10, Mean-Mixture inference, same candidate IDs/query order as before.

## Required comparison table -- Old (buggy) vs Corrected Error Cosine

| Cell | Old Error Cosine (mean, pooled over 10 pairs) | Corrected Error Cosine | Ratio (corrected/old) | Error Dot (unchanged) | Negative-Cosine Fraction (unchanged) |
|---|---:|---:|---:|---:|---:|
| ETTh1_96 | 0.0088 | **0.844** | 96.0 (= H exactly) | 0.389 (pooled mean, unchanged) | 0.0-3.3% per pair (unchanged) |
| ETTh1_720 | 0.00132 | **0.952** | 720.0 (= H exactly) | 0.605 (pooled mean, unchanged) | 0.0-0.2% per pair (unchanged) |
| Weather_96 | 0.0068 | **0.650** | 96.0 (= H exactly) | 0.200 (pooled mean, unchanged) | 5.7-14.9% per pair (unchanged) |
| Weather_720 | 0.00112 | **0.809** | 720.0 (= H exactly) | 0.472 (pooled mean, unchanged) | 1.7-4.0% per pair (unchanged) |

The ratio is EXACTLY the horizon length `H` in every cell (96 or 720, to 6 significant figures), confirming the root cause precisely: the old code's `cos_` was the true cosine divided by `H`. Error Dot and Negative-Cosine Fraction are confirmed byte-identical to the pre-fix parquet (`old['dot_H1-H2'].equals(new['dot_H1-H2'])` = True; negative-fraction values match to the row, since sign is invariant to a positive rescale) -- see PHASE D for the full unaffected-metric verification.

## Full pairwise table (test split, corrected cosine)

### ETTh1_96 (pooled mean cosine 0.844; quantiles [p5,p25,p50,p75,p95] = [0.393, 0.821, 0.923, 0.968, 0.993])

| Pair | Overlap | Corrected Cosine | Error Dot | Neg. Frac |
|---|---:|---:|---:|---:|
| H1-H2 | 0.497 | 0.906 | 0.392 | 0.0018 |
| H1-H3 | 0.341 | 0.842 | 0.388 | 0.0138 |
| H1-H4 | 0.349 | 0.849 | 0.388 | 0.0140 |
| H1-H5 | 0.465 | 0.855 | 0.392 | 0.0181 |
| H2-H3 | 0.310 | 0.835 | 0.386 | 0.0149 |
| H2-H4 | 0.509 | 0.892 | 0.406 | 0.0103 |
| H2-H5 | 0.339 | 0.820 | 0.386 | 0.0205 |
| H3-H4 | 0.325 | 0.834 | 0.389 | 0.0168 |
| H3-H5 | 0.384 | 0.831 | 0.391 | 0.0220 |
| H4-H5 | 0.237 | 0.770 | 0.378 | 0.0334 |

### ETTh1_720 (pooled mean cosine 0.952; quantiles = [0.810, 0.951, 0.983, 0.994, 1.000])

| Pair | Overlap | Corrected Cosine | Error Dot | Neg. Frac |
|---|---:|---:|---:|---:|
| H1-H2 | 0.686 | 0.933 | 0.608 | 0.0015 |
| H1-H3 | 0.576 | 0.909 | 0.595 | 0.0022 |
| H1-H4 | 0.685 | 0.940 | 0.614 | 0.0009 |
| H1-H5 | 0.655 | 0.923 | 0.603 | 0.0014 |
| H2-H3 | 0.690 | 0.953 | 0.594 | 0.0003 |
| H2-H4 | 0.901 | 0.987 | 0.619 | 0.0000 |
| H2-H5 | 0.876 | 0.985 | 0.610 | 0.0000 |
| H3-H4 | 0.656 | 0.945 | 0.596 | 0.0007 |
| H3-H5 | 0.762 | 0.967 | 0.600 | 0.0001 |
| H4-H5 | 0.828 | 0.975 | 0.612 | 0.0001 |

### Weather_96 (pooled mean cosine 0.650; quantiles = [-0.395, 0.481, 0.879, 0.972, 0.998])

| Pair | Overlap | Corrected Cosine | Error Dot | Neg. Frac |
|---|---:|---:|---:|---:|
| H1-H2 | 0.148 | 0.549 | 0.205 | 0.1488 |
| H1-H3 | 0.287 | 0.680 | 0.177 | 0.0813 |
| H1-H4 | 0.251 | 0.672 | 0.208 | 0.0853 |
| H1-H5 | 0.220 | 0.639 | 0.225 | 0.1014 |
| H2-H3 | 0.202 | 0.587 | 0.181 | 0.1279 |
| H2-H4 | 0.269 | 0.626 | 0.213 | 0.1136 |
| H2-H5 | 0.255 | 0.638 | 0.221 | 0.1030 |
| H3-H4 | 0.324 | 0.728 | 0.181 | 0.0569 |
| H3-H5 | 0.217 | 0.662 | 0.174 | 0.0850 |
| H4-H5 | 0.353 | 0.718 | 0.215 | 0.0662 |

### Weather_720 (pooled mean cosine 0.809; quantiles = [0.117, 0.758, 0.939, 0.992, 1.000])

| Pair | Overlap | Corrected Cosine | Error Dot | Neg. Frac |
|---|---:|---:|---:|---:|
| H1-H2 | 0.351 | 0.759 | 0.434 | 0.0403 |
| H1-H3 | 0.455 | 0.806 | 0.468 | 0.0323 |
| H1-H4 | 0.446 | 0.806 | 0.478 | 0.0341 |
| H1-H5 | 0.411 | 0.780 | 0.458 | 0.0384 |
| H2-H3 | 0.510 | 0.845 | 0.491 | 0.0183 |
| H2-H4 | 0.416 | 0.812 | 0.470 | 0.0229 |
| H2-H5 | 0.325 | 0.762 | 0.441 | 0.0329 |
| H3-H4 | 0.512 | 0.856 | 0.507 | 0.0172 |
| H3-H5 | 0.489 | 0.834 | 0.489 | 0.0240 |
| H4-H5 | 0.460 | 0.826 | 0.487 | 0.0250 |

## Distribution, not just mean (spec requirement)

| Cell | Pooled cosine mean | p5 | p25 | p50 | p75 | p95 | Per-channel mean range |
|---|---:|---:|---:|---:|---:|---:|---|
| ETTh1_96 | 0.844 | 0.393 | 0.821 | 0.923 | 0.968 | 0.993 | 0.634 - 0.944 |
| ETTh1_720 | 0.952 | 0.810 | 0.951 | 0.983 | 0.994 | 1.000 | 0.903 - 0.981 |
| Weather_96 | 0.650 | **-0.395** | 0.481 | 0.879 | 0.972 | 0.998 | 0.241 - 0.859 |
| Weather_720 | 0.809 | 0.117 | 0.758 | 0.939 | 0.992 | 1.000 | 0.519 - 0.917 |

The median is high everywhere (0.879-0.983) -- most query/channel pairs show strongly redundant head errors, not just the mean. The LOWER tail matters too: Weather_96's 5th percentile is actually negative (-0.395), meaning a non-trivial minority of query-channel-pairs (roughly 5%+, given p5<0 ETTh1_96/Weather_720 land right at/near the boundary too) DO show genuine error cancellation even though the median case is strongly redundant -- the single mean number from the original (bugged OR fixed) report hides this real heterogeneity. Per-channel means also vary substantially (e.g. Weather_96: 0.241 to 0.859) -- complementarity is not uniform across channels either.

## Head error norm and aggregate MSE (reused from existing `head_row` columns, no rerun needed: `||e_m|| = sqrt(aggmse_H{h} * H)`)

| Cell | H1 \|\|e\|\| | H2 | H3 | H4 | H5 |
|---|---:|---:|---:|---:|---:|
| ETTh1_96 | 6.356 | 6.422 | 6.433 | 6.443 | 6.438 |
| ETTh1_720 | 21.556 | 21.098 | 21.101 | 21.279 | 21.051 |
| Weather_96 | 5.606 | 6.077 | 4.596 | 4.915 | 5.254 |
| Weather_720 | 21.462 | 20.153 | 20.317 | 20.912 | 20.590 |

Head error norms are close to each other within each cell (e.g. ETTh1_96: 6.36-6.44, <1.3% spread) except Weather_96 where H3 (4.59) is substantially smaller than H2 (6.07, ~32% smaller) -- consistent with Weather_96 also being the cell with H3 as the best individual/aggregate head (PHASE C of the original audit track). Smaller error norm for the already-best head is expected, not a new finding, but it is useful context for interpreting that head's high pairwise cosine with the others (a smaller-norm vector can still have a high cosine with a larger one if they point the same direction).
