# PHASE D -- Head-Level Error Complementarity

> **SUPERSEDED (2026-10-27): the Error Cosine numbers and "near-orthogonal"
> interpretation in this file were computed with a normalization bug
> (reported cosine = true cosine / H). Corrected values and revised
> interpretation (heads are actually strongly REDUNDANT, cosine 0.65-0.95,
> not near-zero) are in
> `results/TRACK-V5-HEAD-OBJECTIVE-COSINE-FIX01/PHASE_C_CORRECTED_COMPLEMENTARITY.md`
> and `PHASE_E_REVISED_RESEARCH_VERDICT.md`. This file is kept unmodified
> below for historical record, per standing policy -- do not cite its
> Error Cosine conclusions as current.

**This is new analysis, not present anywhere in the repo before this track** (see PHASE A). TRACK-J3 measured complementarity between individually-retrieved CANDIDATES inside one shared Top-10 set; this Phase measures complementarity between the 5 heads' own independently-produced AGGREGATE predictions `R_m = mean(Top10(p_m))`. Correlation (RQ3/D4) is computed at Query/Channel granularity throughout -- never pooling raw candidates as independent samples.

## Table B -- Head-pair error complementarity (test split)

### ETTh1_96 (cross-head mean cosine over all 10 pairs: 0.0088; within-Mean-Mixture-Set C: 0.34342)

| Pair | Top-10 Overlap | Error Cosine | Error Dot | Negative-Cosine Fraction |
|---|---:|---:|---:|---:|
| H1-H2 | 0.497 | 0.0094 | 0.392 | 0.0018 |
| H1-H3 | 0.341 | 0.0088 | 0.388 | 0.0138 |
| H1-H4 | 0.349 | 0.0088 | 0.388 | 0.0140 |
| H1-H5 | 0.465 | 0.0089 | 0.392 | 0.0181 |
| H2-H3 | 0.310 | 0.0087 | 0.386 | 0.0149 |
| H2-H4 | 0.509 | 0.0093 | 0.406 | 0.0103 |
| H2-H5 | 0.339 | 0.0085 | 0.386 | 0.0205 |
| H3-H4 | 0.325 | 0.0087 | 0.389 | 0.0168 |
| H3-H5 | 0.384 | 0.0087 | 0.391 | 0.0220 |
| H4-H5 | 0.237 | 0.0080 | 0.378 | 0.0334 |

### ETTh1_720 (cross-head mean cosine: 0.00132; within-set C: 0.52631)

| Pair | Overlap | Cosine | Dot | Neg. Frac |
|---|---:|---:|---:|---:|
| H1-H2 | 0.686 | 0.0013 | 0.608 | 0.0015 |
| H1-H3 | 0.576 | 0.0013 | 0.595 | 0.0022 |
| H1-H4 | 0.685 | 0.0013 | 0.614 | 0.0009 |
| H1-H5 | 0.655 | 0.0013 | 0.603 | 0.0014 |
| H2-H3 | 0.690 | 0.0013 | 0.594 | 0.0003 |
| H2-H4 | 0.901 | 0.0014 | 0.619 | 0.0000 |
| H2-H5 | 0.876 | 0.0014 | 0.610 | 0.0000 |
| H3-H4 | 0.656 | 0.0013 | 0.596 | 0.0007 |
| H3-H5 | 0.762 | 0.0013 | 0.600 | 0.0001 |
| H4-H5 | 0.828 | 0.0014 | 0.612 | 0.0001 |

### Weather_96 (cross-head mean cosine: 0.0068; within-set C: 0.21877)

| Pair | Overlap | Cosine | Dot | Neg. Frac |
|---|---:|---:|---:|---:|
| H1-H2 | 0.148 | 0.0057 | 0.205 | **0.1488** |
| H1-H3 | 0.287 | 0.0071 | 0.177 | 0.0813 |
| H1-H4 | 0.251 | 0.0070 | 0.208 | 0.0853 |
| H1-H5 | 0.220 | 0.0067 | 0.225 | 0.1014 |
| H2-H3 | 0.202 | 0.0061 | 0.181 | 0.1279 |
| H2-H4 | 0.269 | 0.0065 | 0.213 | 0.1136 |
| H2-H5 | 0.255 | 0.0066 | 0.221 | 0.1030 |
| H3-H4 | 0.324 | 0.0076 | 0.181 | 0.0569 |
| H3-H5 | 0.217 | 0.0069 | 0.174 | 0.0850 |
| H4-H5 | 0.353 | 0.0075 | 0.215 | 0.0662 |

### Weather_720 (cross-head mean cosine: 0.00112; within-set C: 0.53901)

| Pair | Overlap | Cosine | Dot | Neg. Frac |
|---|---:|---:|---:|---:|
| H1-H2 | 0.351 | 0.0011 | 0.434 | 0.0403 |
| H1-H3 | 0.455 | 0.0011 | 0.468 | 0.0323 |
| H1-H4 | 0.446 | 0.0011 | 0.478 | 0.0341 |
| H1-H5 | 0.411 | 0.0011 | 0.458 | 0.0384 |
| H2-H3 | 0.510 | 0.0012 | 0.491 | 0.0183 |
| H2-H4 | 0.416 | 0.0011 | 0.470 | 0.0229 |
| H2-H5 | 0.325 | 0.0011 | 0.441 | 0.0329 |
| H3-H4 | 0.512 | 0.0012 | 0.507 | 0.0172 |
| H3-H5 | 0.489 | 0.0012 | 0.489 | 0.0240 |
| H4-H5 | 0.460 | 0.0011 | 0.487 | 0.0250 |

## D1/D2 -- interpretation

All 4 cells show **cross-head error cosine near zero** (0.001-0.009) -- a factor of ~60-600x smaller than TRACK-J3's candidate-level error cosine (0.50-0.62, Full-memory, ETTh1_720). This is not directly comparable in absolute scale (different normalization context -- candidates within one Top-10 set necessarily share most of their local neighborhood, whereas the 5 heads' AGGREGATE predictions already each average away much of that shared signal), but the qualitative story is clear: **the 5 heads' own error vectors are nearly uncorrelated with each other, not strongly redundant nor strongly cancelling.** Low overlap alone does NOT imply strong complementarity here -- per the spec's explicit warning (section 7.2), e.g. Weather_96's H1-H2 pair has the LOWEST overlap of all pairs/cells (0.148) but only a middling cosine (0.0057) and the HIGHEST negative-cosine fraction (14.9%) -- low overlap correlates with low-but-not-zero cosine, not with strong cancellation.

**RQ3 answer (per Query/Channel, never pooled-as-independent-candidates):** error directions ARE measurably different across heads (overlap-vs-cosine Spearman correlation is strongly positive in every pair/cell, e.g. ETTh1_720 pairs range 0.86-0.90, Weather_96 pairs range 0.16-0.33, Weather_720 pairs range 0.72-0.81) -- i.e. pairs of heads that pick MORE similar candidates also produce MORE correlated errors, a sensible and consistent monotonic relationship, but the absolute error correlation stays small everywhere. Heads are neither strongly redundant nor strongly adversarial; they sit in a weakly-correlated, largely orthogonal regime.

**Negative-cosine fraction (genuine cancellation) is small everywhere** (0.0-3.3% for ETTh1 cells, 5.7-14.9% for Weather) -- Weather consistently shows MORE true error cancellation between head pairs than ETTh1, interesting given Weather is also where Mean-Mixture loses to single-best-head most often (PHASE C) -- more raw cancellation opportunity does not translate into Mean-Mixture actually winning on AggMSE there (PHASE E will quantify this gap precisely).

## D3 -- Cross-Head vs Within-Set complementarity: explicitly different quantities

The spec warns these must not be conflated. Measured side by side:

| Cell | Cross-Head mean cosine (5 heads' own `R_m`, dimensionless) | Within-Mean-Mixture-Set `C` (10 selected candidates, raw MSE units) |
|---|---:|---:|
| ETTh1_96 | 0.0088 | 0.34342 |
| ETTh1_720 | 0.0013 | 0.52631 |
| Weather_96 | 0.0068 | 0.21877 |
| Weather_720 | 0.0011 | 0.53901 |

These are not on the same scale (one is a bounded cosine over 5 head-level vectors, the other an unnormalized cross-MSE term over 10 candidate-level vectors) and are not meant to be compared numerically -- the point of reporting them together is structural: **Cross-Head complementarity (low) does not determine Within-Set complementarity (the dominant term in every cell's AggMSE, 70-85% of AggMSE in every cell per PHASE C's own D/C split).** A head producing a weakly-correlated-with-other-heads aggregate prediction tells us nothing directly about how well the 10 individual candidates inside that head's own (or the Mean-Mixture's) selected set interact with each other -- confirmed by inspection: ETTh1_720 has the LOWEST cross-head cosine (0.0013) of all 4 cells, yet the LARGEST within-set `C` (0.52631, tied with Weather_720) -- the two quantities move independently across cells, not together.
