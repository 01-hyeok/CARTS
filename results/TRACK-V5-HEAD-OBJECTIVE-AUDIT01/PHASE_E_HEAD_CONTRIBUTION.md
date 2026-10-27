# PHASE E -- Head Contribution and Combination Oracle

All 31 non-empty head subsets enumerated (`p_A = mean_{m in A} p_h`, `R_A = mean(Top10(p_A))`), computed fresh by `diag_v5_head_objective01.py` for every query/channel in val and test (new analysis -- no prior track enumerated more than the single-head argmin oracle, see PHASE A). Fixed-subset and single-head selection happens on VAL ONLY, then is evaluated once on TEST (never test-based selection). Per-query Oracle is reported as an explicit, non-deployable upper bound.

## Table C -- Head contribution (test split; subset/single-head chosen on val)

| Cell | All-5 Mean-Mixture AggMSE | Best Single Head (val-selected) AggMSE | Best Fixed Subset (val-selected) AggMSE | Per-Query Oracle AggMSE (upper bound) | Head-Oracle Gap (All-5 vs Oracle) |
|---|---:|---:|---:|---:|---:|
| ETTh1_96 | 0.40981 | 0.42085 (H1) | 0.41079 ({1,2,4,5}) | 0.32441 | 0.08540 (20.84%) |
| ETTh1_720 | 0.62057 | 0.61826 (H2) | 0.61826 ({2}, degenerates to single head) | 0.56499 | 0.05558 (8.96%) |
| Weather_96 | 0.26433 | 0.22004 (H3) | 0.27689 ({1,2,3,5}, **worse than All-5**) | 0.12156 | 0.14277 (54.01%) |
| Weather_720 | 0.61780 | 0.63972 (H1) | 0.64389 ({1,2,5}, **worse than All-5 AND worse than best single head**) | 0.37866 | 0.23914 (38.71%) |

**Important honest finding**: the val-selected "Best Fixed Subset" is WORSE than plain All-5 Mean-Mixture on 2 of 4 cells (Weather_96, Weather_720), and even worse than the val-selected single head on those same 2 cells. With 31 candidate subsets and only ~2161-5175 validation queries (x channels), subset selection by validation AggMSE is exposed to a real multiple-comparisons overfitting risk -- many of the 31 subsets are highly correlated (nested head memberships), so the effective number of independent comparisons is smaller than 31, but the risk is not zero, and this result shows it materializing concretely on Weather. **This is reported as a genuine negative finding for the "Fixed Subset" strategy, not smoothed over** -- a production system choosing a fixed head subset via validation AggMSE over 31 options is not guaranteed to beat using all 5 heads, and on this data it did not, twice out of four cells.

The **Per-Query Oracle** (which cannot be deployed -- it uses the query's own future to choose the best of the 31 subsets, a strict upper bound only) shows a large, real gap everywhere: 9-54% better AggMSE than All-5 Mean-Mixture. This is consistent with the already-established `TRACK-HARD-EXPERT-V5-P100-ALLH01` single-head-oracle gaps (6.3-15.1% on ETTh1, higher on Weather in that track's own single-head-only oracle) -- the FULL 31-subset oracle gap is, as expected, larger than the single-head-only gap (e.g. ETTh1_96: 20.8% here vs 15.1% single-head-only in the prior track), since the 31-subset oracle strictly dominates the 5-single-head oracle (it includes those 5 subsets plus 26 more).

## Leave-One-Head-Out contribution (`Delta_h`, conditional on the other 4 heads; positive = head helps)

| Cell | H1 | H2 | H3 | H4 | H5 |
|---|---:|---:|---:|---:|---:|
| ETTh1_96 | +0.00186 | -0.00023 | +0.00099 | +0.00047 | **+0.00254** |
| ETTh1_720 | -0.00264 | +0.00095 | **+0.00246** | -0.00157 | +0.00167 |
| Weather_96 | -0.00356 | +0.00765 | **+0.01329** | +0.01256 | -0.00927 |
| Weather_720 | **-0.06291** | +0.03000 | +0.03261 | +0.00502 | +0.03689 |

No head is universally helpful or harmful across cells -- e.g. H1 helps on ETTh1_96 (+0.0019) but is the single MOST harmful head on Weather_720 (-0.0629, by a wide margin over every other head's magnitude on that cell) and mildly harmful on ETTh1_720/Weather_96. These are all small in absolute magnitude except Weather_720's H1 -- removing H1 from the Weather_720 Mean-Mixture would improve AggMSE by 0.063, roughly 10% of the All-5 AggMSE (0.618), the single largest LOO effect found in this track. Per the spec's own caveat, this is H1's CONDITIONAL contribution given the other 4 heads are present -- not a claim about H1's independent, context-free utility.

## Oracle winner-subset frequency (top-5 by query-count share, test split)

| Cell | #1 | #2 | #3 | #4 | #5 |
|---|---|---|---|---|---|
| ETTh1_96 | {3}: 8.4% | {5}: 8.3% | {4}: 8.1% | {1}: 7.0% | {2}: 6.3% |
| ETTh1_720 | {3}: 21.6% | {1}: 18.5% | {2}: 12.2% | {5}: 7.0% | {4}: 6.9% |
| Weather_96 | {1}: 13.4% | {5}: 6.5% | {3}: 5.8% | {2}: 5.5% | {4}: 4.5% |
| Weather_720 | {1}: 13.3% | {2}: 12.9% | {5}: 10.3% | {4}: 9.9% | {3}: 6.9% |

Single-head subsets dominate the oracle-winner frequency ranking in every cell (no multi-head subset appears in the top-5 at ETTh1_96/Weather_96/Weather_720; only small multi-head subsets like `{1,3}`/`{1,2}` appear further down the ETTh1_720 list, per the raw JSON). This means the PER-QUERY optimal choice is most often "pick one single good head for this query," not "average several heads" -- yet WHICH single head is optimal varies substantially query-by-query (no single head's frequency exceeds ~22% anywhere), so no FIXED single head captures this -- exactly the tension PHASE E's fixed-subset-overfitting finding above already demonstrates empirically. This is consistent with (not proof of) a genuine per-query head-routing opportunity that no method tested in this repo (fixed assignment, Hard/Soft Expert re-training, or this Phase's own val-selected fixed subset) currently captures reliably.

## E4 -- Past-only predictability (descriptive only, no router trained)

Oracle subsets change query-by-query (no subset exceeds ~22% winner share in any cell, confirmed above) -- this track does not train a router or selector of any kind (per spec section 8.4's explicit prohibition), and this observation alone is **not** evidence that the winning subset is predictable from past-only information; it only establishes that the headroom exists and that no fixed choice (single head or subset) captures it reliably on 2 of 4 cells. A dedicated past-only-feature router experiment (in the spirit of `TRACK-P100-RETRIEVAL-UTILIZATION-AUDIT01`'s Phase B observability probes, which found AUROC clearly above chance but a deployable-gate sign flip on 3/4 cells for a RELATED but different per-query usefulness question) would be needed to test predictability directly -- flagged as a candidate follow-up, not run here.
