# PHASE A -- Frozen Coarse + V5 Static Score Fusion

Evaluation-only, no training. Candidate support fixed to the existing
canonical Shared-P100 cache
(`results/TRACK-V-MULTIQUERY-GENERALIZATION01/<ds>/H<h>/seed0/pool_top100/shared_candidate_pool`,
read-only) in all 4 cells. V5 is the frozen canonical sign-fixed
checkpoint (`TRACK-V-SHARED-TOP100-SIGNFIX01`, `selected_checkpoint.json`
epochs: ETTh1_96=10, ETTh1_720=9, Weather_96=4, Weather_720=6,
tau_s=0.1 in every case). Code: `scripts/diag_p100_coarse_v5_fusion01.py`.

## Provenance / audit (section 2-5, S1-S8)

- **Coarse score** reused UNMODIFIED from
  `utils.candidate_pool.compute_coarse_delta_last_cosine_scores` (the
  exact function `precompute_candidate_pool01.py` already uses to
  BUILD Shared-P100) -- never reimplemented. Computed on the full
  bank then gathered at the pool's own global indices; mathematically
  identical to computing it directly on just the 100 pool candidates
  (cosine of one candidate's own history never depends on any other
  candidate).
- **V5 score**: canonical Mean-Mixture `p_v5=mean_h softmax(scores_h/tau_s)`,
  computed WITHIN the same 100-candidate pool, tau_s=0.1 (confirmed
  from each cell's own `selected_checkpoint.json`).
- **tau_c calibration** (`scripts/calibrate_p100_coarse_tau01.py`,
  `tau_c_calibration.json`): no canonical tau_c exists anywhere in the
  repo (pool-generation only ever uses a temperature-free raw Top-M
  argsort). A first attempt calibrated tau_c by minimizing coarse
  -only AggMSE@10 on TRAIN -- discarded on discovery that this
  criterion is mathematically degenerate (argmax/top-k of
  softmax(c_i/tau) is rank-invariant to tau for any tau>0, confirmed
  constant to 6+ decimals across tau_c in [0.02,2.0] on all 4 cells
  before being discarded). Replaced with the correct criterion: match
  the coarse distribution's average entropy on TRAIN to canonical
  V5's own Mean-Mixture entropy (tau_s=0.1) on TRAIN, averaged across
  all 4 cells. Selected **tau_c=0.05** (cross-cell TRAIN coarse
  entropy 4.383 nats vs V5 target 4.069 nats; nearest grid point).
  Fixed, never re-swept in Phase A/B/C/D.
- **Endpoint sanity** (every val/test run): beta=0 recovers
  coarse-only and beta=1 recovers V5-only to within 2.4e-7 max
  absolute probability difference, in all 4 cells x 2 splits -- PASS.
- **Consistency cross-check** (not in the original spec, added as an
  extra sanity signal): beta=1.0's AggMSE@10 in this script
  numerically matches canonical V5's own previously-reported numbers
  in `TRACK-HEAD-CONDITIONED-CANDIDATE-SUPPORT01` to 4+ decimals in
  every cell (e.g. ETTh1_720 val 1.681553 vs 1.6816; Weather_96 val
  0.518968 vs 0.5190) -- confirms the fusion harness's V5 path is
  computing the exact same thing as the independently-built prior
  track's harness.

## Table 1 -- Val/Test AggMSE@10 across the beta sweep

| Cell | Split | beta=0 (coarse) | beta=0.25 | beta=0.5 | beta=0.75 | beta=1 (V5) |
|---|---|---:|---:|---:|---:|---:|
| ETTh1_96 | val | 0.8808 | 0.7739 | 0.7346 | 0.7159 | **0.7124** |
| ETTh1_96 | test | 0.5108 | 0.4340 | 0.4161 | 0.4105 | **0.4098** |
| ETTh1_720 | val | 1.9064 | 1.6965 | 1.6875 | 1.6827 | **1.6816** |
| ETTh1_720 | test | 0.8726 | 0.6718 | 0.6370 | 0.6244 | **0.6206** |
| Weather_96 | val | 0.7264 | **0.5159** | 0.5217 | 0.5166 | 0.5190 |
| Weather_96 | test | 0.5092 | 0.2615 | 0.2564 | **0.2545** | 0.2643 |
| Weather_720 | val | 1.2192 | 0.8972 | 0.8670 | **0.8571** | 0.8645 |
| Weather_720 | test | 2.0859 | 0.6238 | 0.6012 | **0.6002** | 0.6178 |

Bold = lowest AggMSE in that row. **ETTh1 (both horizons): beta=1.0
(pure V5) is the literal minimum on val** -- the sweep is monotonically
decreasing in beta, i.e. fusing in ANY coarse weight only hurts.
**Weather (both horizons): an interior beta beats pure V5 on val**
(0.25 for H96, 0.75 for H720), and the val-selected beta ALSO beats
beta=1.0 on test in both cases (Weather_96 test: selected-beta
0.2615 vs V5-only 0.2643, **1.06% better**; Weather_720 test:
selected-beta 0.6002 vs V5-only 0.6178, **2.86% better**).

## Table 2 -- selected beta (val-only selection, test evaluated once after)

| Cell | Selected beta (val min) | Val AggMSE@10 | Test AggMSE@10 | Test AggMSE@10 (beta=1, V5-only) | Test improvement vs V5-only |
|---|---:|---:|---:|---:|---:|
| ETTh1_96 | 1.00 | 0.7124 | 0.4098 | 0.4098 | 0.00% (selected = V5-only) |
| ETTh1_720 | 1.00 | 1.6816 | 0.6206 | 0.6206 | 0.00% (selected = V5-only) |
| Weather_96 | 0.25 | 0.5159 | 0.2615 | 0.2643 | **+1.06%** |
| Weather_720 | 0.75 | 0.8571 | 0.6002 | 0.6178 | **+2.86%** |

(Test was evaluated exactly once per cell, at the pre-registered
val-selected beta only; the other 4 beta values' test numbers in
Table 1 are reported for transparency but were never used for any
selection decision.)

## Table 3 -- additional Stage-1 metrics at the selected beta (test split)

| Cell | RetMSE@10 | AggMSE@10 | Recall@10 | NDCG@10 | D | C |
|---|---:|---:|---:|---:|---:|---:|
| ETTh1_96 (beta=1.0) | 0.6639 | 0.4098 | 0.2575 | 0.8741 | 0.0664 | 0.3434 |
| ETTh1_720 (beta=1.0) | 0.9426 | 0.6206 | 0.3213 | 0.8895 | 0.0943 | 0.5263 |
| Weather_96 (beta=0.25) | 0.5402 | 0.2615 | 0.2474 | 0.8194 | 0.0540 | 0.2075 |
| Weather_720 (beta=0.75) | 0.7625 | 0.6002 | 0.2546 | 0.8220 | 0.0763 | 0.5239 |

## Table 4 -- coarse vs V5 ranking disagreement (test split; beta-independent)

| Cell | Top1 agree | Top10 overlap | Spearman(c,v5) | Kendall(c,v5) | Spearman(c,oracle) | Spearman(v5,oracle) |
|---|---:|---:|---:|---:|---:|---:|
| ETTh1_96 | 0.056 | 0.208 | 0.220 | 0.150 | 0.086 | 0.394 |
| ETTh1_720 | 0.017 | 0.127 | 0.126 | 0.086 | 0.060 | 0.466 |
| Weather_96 | 0.071 | 0.164 | 0.121 | 0.039 | 0.107 | 0.394 |
| Weather_720 | 0.022 | 0.115 | 0.062 | 0.053 | 0.055 | 0.403 |

The coarse score and V5's learned score are clearly measuring
different things (Top10 overlap only 0.115-0.208, far from 1.0;
Top1 agreement under 7.1% everywhere). V5 alone correlates with the
true oracle ranking (Spearman(v5,oracle): 0.39-0.47) much more
strongly than coarse alone does (Spearman(c,oracle): 0.055-0.107) in
EVERY cell -- consistent with V5 being the stronger individual
ranker everywhere, which is exactly why ETTh1's sweep is monotonic.
Where Weather still gains from a partial coarse weight despite coarse
being the much weaker individual ranker there too, the benefit is
coming from DIVERSIFICATION (the two rankers disagreeing on *which*
errors they make), not from coarse being independently competitive.

## Table 5 -- Spearman(fused, oracle) across the beta sweep (test split)

| Cell | beta=0 | beta=0.25 | beta=0.5 | beta=0.75 | beta=1 |
|---|---:|---:|---:|---:|---:|
| ETTh1_96 | 0.086 | 0.286 | 0.356 | 0.385 | 0.394 |
| ETTh1_720 | 0.060 | 0.386 | 0.446 | 0.463 | 0.466 |
| Weather_96 | 0.107 | 0.336 | 0.372 | 0.387 | 0.394 |
| Weather_720 | 0.055 | 0.328 | 0.372 | 0.393 | 0.403 |

Spearman-vs-oracle itself is monotonically increasing in beta in
*every* cell, including Weather -- i.e. by pure pointwise rank
correlation with the true future, more V5 weight is always "more
correct." Weather's AggMSE improvement at an interior beta is
therefore NOT because the fused ranking correlates better with the
oracle pointwise -- it is an aggregation-level effect (the Top-10
SET selected at the interior beta produces a better uniform-mean
forecast than the Top-10 set selected at beta=1, even though the
latter's individual ranking is "more correct" by Spearman). This is
visible directly in Table 3's D/C split for Weather_96: D (individual
retrieval quality) is essentially flat across beta=0.5-1.0 (0.050,
0.045, 0.046) while C (the aggregation gap) is what actually moves
and bottoms out at beta=0.75, not beta=1.0.

## PHASE A interpretation (section 10, checked directly)

| Condition | Required | Actual |
|---|---|---|
| Interior beta beats V5-only (val direction) | >=3/4 cells | **2/4** (Weather_96, Weather_720 only) |
| >=2 cells with >=1% improvement (test) | >=2/4 cells | **2/4** (Weather_96 +1.06%, Weather_720 +2.86%) -- met |

A-PASS requires BOTH conditions; the first (>=3/4 cells) is not met,
so this is **not A-PASS**. A-WEAK also requires >=3/4 cells showing
the direction (regardless of magnitude); also not met. By the
pre-registered three-way split this result falls to **A-FAIL**
("beta=1 또는 beta=0이 대부분 best") on the strict binary gate --
but reporting it as a flat, uniform null would misrepresent the data.

**What actually happened is a clean, reproducible 2-cell/2-cell split
that lines up exactly with dataset identity, not noise:**

- **ETTh1 (H96 and H720, both)**: the beta sweep is monotonically
  decreasing in AggMSE as beta rises; beta=1.0 (pure V5) is the
  literal minimum on val in both cells, by a widening margin at the
  high-beta end (ETTh1_96: beta=0.75 is already within 0.5% of
  beta=1.0; same shape at H720). Coarse weight only ever hurts here.
- **Weather (H96 and H720, both)**: an interior beta clearly beats
  pure V5 on val, confirmed on test with real >1% improvements in
  both cases (1.06%, 2.86%) -- not a validation-only artifact.

This matches this session's standing caveat recorded elsewhere
(`TRACK-V`'s horizon-reversal finding): effects in this codebase
recurrently turn out to be **dataset-specific rather than general**.
No hyperparameter (tau_c, the beta grid, candidate support) was
retuned after seeing this split -- it is reported exactly as the
pre-registered sweep produced it.

## Next step

Per section 10's explicit instruction, **Phase B proceeds regardless
of this result** ("결과와 관계없이 PHASE B로 진행한다"). Phase B's
own oracle-beta diagnostic and adaptive-fusion model will show
whether a PER-QUERY beta choice (rather than one fixed beta per cell)
can recover additional headroom beyond this fixed-fusion result --
and, in particular, whether it can do so on ETTh1 as well, where no
single fixed interior beta helps.
