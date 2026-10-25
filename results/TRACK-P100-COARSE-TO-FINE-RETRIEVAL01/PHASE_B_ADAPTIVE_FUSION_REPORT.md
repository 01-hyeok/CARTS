# PHASE B -- Query-Adaptive Confidence Fusion

Candidate support fixed to Shared-P100, same frozen canonical V5
checkpoint as Phase A. Two steps: (1) an evaluation-only cheating
-oracle beta diagnostic (`scripts/diag_p100_oracle_beta01.py`), (2) a
deployable adaptive-beta model trained on past-only features
(`scripts/train_p100_adaptive_fusion01.py`), with mandatory controls
B0-B3 per spec section 15.

## Step 1 -- Oracle Beta diagnostic (cheating oracle, not deployable)

For each query/channel, picks whichever beta in {0,.25,.5,.75,1}
(same fused distributions as Phase A) has the lowest ACTUAL future
AggMSE -- reads the future, diagnostic only.

| Cell | Split | Best global fixed beta | Global fixed AggMSE@10 | Oracle (per-query) AggMSE@10 | Oracle gain % | Winner-share entropy (max=1.609) |
|---|---|---:|---:|---:|---:|---:|
| ETTh1_96 | val | 1.00 | 0.7124 | 0.6284 | 11.79% | 1.569 |
| ETTh1_96 | test | 1.00 | 0.4098 | 0.3505 | 14.46% | 1.571 |
| ETTh1_720 | val | 1.00 | 1.6816 | 1.4576 | 13.32% | 1.572 |
| ETTh1_720 | test | 1.00 | 0.6206 | 0.5430 | 12.50% | 1.592 |
| Weather_96 | val | 0.25 | 0.5159 | 0.4132 | 19.92% | 1.575 |
| Weather_96 | test | 0.75 | 0.2545 | 0.1583 | 37.78% | 1.579 |
| Weather_720 | val | 0.75 | 0.8571 | 0.7358 | 14.15% | 1.592 |
| Weather_720 | test | 0.75 | 0.6002 | 0.4406 | 26.59% | 1.591 |

Real per-query oracle headroom exists in **every** cell, including
both ETTh1 cells (11.8-14.5%) where Phase A's fixed-beta sweep found
no benefit at all -- and the winner-share entropy is close to its
maximum (ln 5 = 1.609) everywhere, meaning the oracle's optimal beta
is genuinely spread across all 5 candidates per query, not dominated
by one value. This headroom is the motivation for step 2: can a
deployable, past-only model capture any of it?

## Step 2 -- deployable adaptive-beta model vs mandatory controls

Tiny 2-layer MLP (`AdaptiveBetaMLP`, 18 past-only features -> hidden 32
-> softmax over the 5 beta candidates), trained via KL against a
TRAIN-only soft oracle-beta target (`softmax(-AggMSE(beta)/tau_beta)`,
tau_beta = TRAIN std of AggMSE(beta), one-time per-cell calibration:
0.54 / 0.78 / 43.08 / 44.00 for ETTh1_96/720, Weather_96/720
respectively). Model selection via cached-feature validation KL loss
(the same quantity optimized); the reported AggMSE@10 below is from
one streaming pass over the TEST split with the selected-epoch model,
exactly mirroring Phase A's "test evaluated once" discipline.

| Cell | B0 fixed (AggMSE@10) | B1 adaptive | B2 static scalar | B3 shuffled control | B1 vs B0 | B1 vs B2 |
|---|---:|---:|---:|---:|---:|---:|
| ETTh1_96 | 0.4098 | 0.4155 | 0.4154 | 0.4154 | -1.39% | -0.03% |
| ETTh1_720 | 0.6206 | 0.6343 | 0.6343 | 0.6329 | -2.22% | -0.01% |
| Weather_96 | 0.2615 | **0.2558** | 0.2563 | 0.2563 | **+2.20%** | +0.22% |
| Weather_720 | 0.6002 | 0.6011 | 0.6011 | 0.6011 | -0.16% | ~0.00% |

(% = improvement of B1 over the comparator; negative = B1 worse.)

**In every cell, B1/B2/B3's learned beta_mean converges to ~0.50-0.55,
with beta_std always tiny (0.0000-0.0230)** -- the "adaptive" model
essentially collapses to behaving like a single near-constant scalar,
indistinguishable in practice from B2 (the explicit no-features
control) and B3 (the shuffled-feature control). This happens
regardless of how large the real oracle headroom is for that cell
(Weather_720 has the single largest oracle gain of any cell/split,
26.6-14.2%, yet B1/B2/B3 are the most tightly clustered of all four
cells there, essentially identical to 4 decimal places).

## Mandatory-control check (section 15: "B1 must beat B0/B2/B3 or there is no adaptive contribution")

| Cell | B1 beats B0? | B1 beats B2? | B1 beats B3? | Verdict |
|---|---|---|---|---|
| ETTh1_96 | NO | NO (marginally worse) | NO | No adaptive contribution |
| ETTh1_720 | NO | NO (marginally worse) | NO | No adaptive contribution |
| Weather_96 | YES (+2.20%) | YES (+0.22%) | YES (+0.22%) | Marginal contribution -- real but thin |
| Weather_720 | NO | ~tie | ~tie | No adaptive contribution |

**3 of 4 cells: no query-adaptive contribution at all** -- B1 is not
just failing to beat the fixed-beta baseline B0, it fails to even
distinguish itself from B2 (zero input features) and B3 (features
scrambled across queries), meaning whatever small amount the model's
output varies per query is not meaningfully informative. **1 of 4
cells (Weather_96): B1 does beat every control**, but the margin over
B2/B3 (+0.22%) is an order of magnitude smaller than its margin over
B0 (+2.20%) -- most of Weather_96's gain comes from the SOFT-target
-trained scalar landing near beta=0.50 (closer to Phase A's own
interior optimum there) rather than from genuine per-query adaptation;
only a sliver (+0.22%) is attributable to the features actually
varying the prediction per query.

## Why the oracle headroom (12-38%) does not translate to deployable gain

Not investigated further here (per the pre-registered rule against
retuning after a negative result), but worth recording as a candidate
explanation for Phase D's eventual interpretation: the 18 past-only
confidence/disagreement features (entropy, margins, overlap,
Spearman, JS-divergence, rank displacement, z_q norm, channel id) may
simply not carry the signal needed to predict WHICH of the 5 betas is
best for a given query -- the oracle's winner-share entropy being
close to maximum (1.57-1.59 out of 1.609) suggests the true optimal
beta is close to uniformly unpredictable from any summary statistic
at this scale, not merely from these particular 18 features. An
alternative explanation -- that the tiny 2-layer MLP lacks capacity,
or that the KL/soft-target training procedure itself washes out
per-query signal into a near-constant prediction -- is also
consistent with the data (B1 and B2 converge to nearly the SAME
scalar in every cell, suggesting the optimization itself is finding a
near-degenerate solution rather than the features being literally
uninformative); this track's scope rules forbid a new architecture or
hyperparameter sweep to disambiguate between these two explanations,
so both are reported as open rather than resolved.

## Verdict

Per the pre-registered rule, no hyperparameter (hidden width, tau_beta
grid, feature set) was changed after seeing this result. **Query
-adaptive fusion, as implemented here, provides no reliable
contribution over simple fixed-beta fusion or even a context-free
static scalar in 3 of 4 cells**, despite substantial oracle headroom
existing in every cell (12-38%). Weather_96 is the one partial
exception, but even there the ADAPTIVE (per-query) component of the
gain is thin (+0.22% over a static scalar) compared to the gain from
simply landing near the right global scalar (+2.20%).

Per section 37's explicit instruction, Phase C proceeds regardless of
this result.
