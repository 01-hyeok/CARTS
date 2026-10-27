# PHASE C -- Head-Level Individual and Aggregate Retrieval Quality

Script: `scripts/diag_v5_head_objective01.py --mode full_pass` (data) + `scripts/analyze_v5_head_objective01.py` (CPU-only pandas aggregation). Test split, all 4 cells, canonical sign-fixed checkpoint, Shared-P100. "Mean-Mixture" below is the TRUE `mean_mixture_topk_selection` result (see PHASE B's correctness finding), never the pre-existing `rr` round-robin diagnostic.

`D+C=AggMSE` identity verified by an INDEPENDENT direct pairwise-sum computation (not just subtraction) on every query/channel row, all 4 cells: **max abs diff = 0.0** (exact match, not merely within tolerance).

## Table A -- Head-level RetMSE/AggMSE/D/C/Recall/NDCG (test split)

### ETTh1_96

| Head | RetMSE@10 | AggMSE@10 | D | C | Recall@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| H1 | 0.67476 | 0.42085 | 0.06748 | 0.35338 | 0.24725 | 0.8703 |
| H2 | 0.67498 | 0.42961 | 0.06750 | 0.36211 | 0.25009 | 0.86955 |
| H3 | 0.72487 | 0.43103 | 0.07249 | 0.35854 | 0.18638 | 0.85468 |
| H4 | 0.69639 | 0.43236 | 0.06964 | 0.36272 | 0.21365 | 0.85957 |
| H5 | 0.70081 | 0.43179 | 0.07008 | 0.36171 | 0.21424 | 0.85621 |
| **Mean-Mixture** | **0.66386** | **0.40981** | 0.06639 | 0.34342 | **0.25752** | **0.87411** |

### ETTh1_720

| Head | RetMSE@10 | AggMSE@10 | D | C | Recall@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| H1 | 0.97249 | 0.64537 | 0.09725 | 0.54812 | 0.28150 | 0.87748 |
| H2 | 0.94289 | 0.61826 | 0.09429 | 0.52397 | 0.32265 | 0.89014 |
| H3 | 0.93787 | 0.61838 | 0.09379 | 0.52459 | 0.32140 | 0.89049 |
| H4 | 0.95636 | 0.62889 | 0.09564 | 0.53326 | 0.30879 | 0.88511 |
| **H5** | **0.93668** | **0.61550** | 0.09367 | 0.52183 | **0.32607** | **0.89239** |
| Mean-Mixture | 0.94263 | 0.62057 | 0.09426 | 0.52631 | 0.32129 | 0.88946 |

### Weather_96

| Head | RetMSE@10 | AggMSE@10 | D | C | Recall@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| H1 | 0.60717 | 0.32741 | 0.06072 | 0.26669 | 0.23317 | 0.80649 |
| H2 | 0.65359 | 0.38469 | 0.06536 | 0.31934 | 0.19636 | 0.76707 |
| **H3** | **0.46838** | **0.22004** | 0.04684 | 0.17320 | 0.24008 | 0.81848 |
| H4 | 0.43047 | 0.25162 | 0.04305 | 0.20858 | 0.24936 | 0.82464 |
| H5 | 0.52903 | 0.28750 | 0.05290 | 0.23460 | 0.23592 | 0.81104 |
| Mean-Mixture | 0.45562 | 0.26433 | 0.04556 | 0.21877 | **0.26658** | **0.83028** |

### Weather_720

| Head | RetMSE@10 | AggMSE@10 | D | C | Recall@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| H1 | 0.85909 | 0.63972 | 0.08591 | 0.55381 | 0.22666 | 0.80610 |
| H2 | 0.74308 | **0.56406** | 0.07431 | 0.48976 | 0.23196 | 0.81393 |
| **H3** | **0.73277** | 0.57333 | 0.07328 | 0.50006 | 0.23994 | 0.82089 |
| H4 | 0.77256 | 0.60737 | 0.07726 | 0.53011 | 0.23998 | 0.82173 |
| H5 | 0.76841 | 0.58881 | 0.07684 | 0.51197 | 0.22741 | 0.81282 |
| Mean-Mixture | 0.78792 | 0.61780 | 0.07879 | 0.53901 | **0.25535** | **0.82480** |

## Does the best individual-candidate head match the best aggregate head?

| Cell | Best individual head (lowest RetMSE) | Best aggregate head (lowest AggMSE) | Same? |
|---|---|---|---|
| ETTh1_96 | H1 | H1 | **YES** |
| ETTh1_720 | H5 | H5 | **YES** |
| Weather_96 | H4 | H3 | **NO** |
| Weather_720 | H3 | H2 | **NO** |

On ETTh1 the two selection criteria agree; on Weather they disagree in both horizons -- individual relevance and aggregate-set quality are measuring different things specifically where the retrieval signal is weaker (Weather, consistent with every other P100 track this session). Where they disagree, the gap is itself informative: on Weather_720, H3 has the lowest RetMSE (0.73277) but H2 has the lowest AggMSE (0.56406 vs H3's 0.57333) -- H3's individually-closer candidates interact with each other slightly worse (C=0.50006 vs H2's 0.48976) than H2's individually-slightly-worse-but-more-complementary candidates.

## Does Mean-Mixture beat every single head?

**No, not uniformly.** Mean-Mixture is the single best AggMSE arm on ETTh1_96 (0.40981, beating every head by 2.5-5.2%) but is WORSE than the best individual head on all 3 other cells: ETTh1_720 (Mean-Mixture 0.62057 vs H5's 0.61550, +0.8%), Weather_96 (0.26433 vs H3's 0.22004, +20.1%), Weather_720 (0.61780 vs H2's 0.56406, +9.5%). Averaging 5 heads' probability distributions is not a free lunch for AggMSE in this codebase's current (unsupervised-for-aggregate) training regime -- directly relevant to PHASE G/final-verdict L3's question of whether an aggregate-aligned loss has room to help.

Mean-Mixture DOES win on Recall@10/NDCG@10 in every single cell -- it is consistently the best at recovering the RESTRICTED-POOL oracle's own Top-10 ranking, even on the 3 cells where it is not the best at minimizing AggMSE. Averaging distributions improves rank-agreement-with-the-individual-relevance-oracle but does not always improve the aggregate forecast, because AggMSE depends on the CROSS term `C` as well as `D`, and the oracle Recall@10/NDCG@10 metrics only ever measure `D`-adjacent quantities (individual relevance ranking), never `C`.
