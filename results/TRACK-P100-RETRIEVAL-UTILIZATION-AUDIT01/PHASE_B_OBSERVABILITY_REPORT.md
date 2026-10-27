# PHASE B -- Past-Only Observability Diagnostic

No new forecasting model, no new Stage1 encoder/router/Transformer --
only the 3 tiny probes spec section 16 allows (B0 constant, B1
logistic regression, B2 2-layer MLP, hidden=32). Scope: canonical V5
arm only (Phase A found V5 and Phase D Set-aware behave almost
identically on every Oracle Lambda metric, so Phase B does not
duplicate that analysis for Set-aware). Code:
`scripts/diag_p100_observability01.py`. 30 features total: 18 reused
unmodified from `train_p100_adaptive_fusion01.extract_features`
(V5-confidence/coarse-confidence/disagreement) + 12 new (query past
mean/std/slope/first-diff-var/ACF-lag1, Base forecast mean/std/slope,
Base-vs-Retrieval disagreement norm, retrieved-Top10 past/embedding
dispersion, P100 score variance) -- all strictly past-only, never
touching `batch_y`. Target `t_q=1[MSE(R_q,Y_q)<MSE(B_q,Y_q)]`,
TRAIN-only label; existing chronological train/val/test split reused,
never re-split.

## Table 1 -- probe metrics (test split)

| Cell | Probe | AUROC | AUPR | Balanced Acc | Brier | ECE | Spearman(prob, gain) |
|---|---|---:|---:|---:|---:|---:|---:|
| ETTh1_96 | B0 | 0.500 | 0.392 | 0.500 | 0.239 | 0.089 | -0.037 |
| ETTh1_96 | B1 | 0.597 | 0.457 | 0.568 | 0.233 | 0.093 | 0.065 |
| ETTh1_96 | B2 | 0.593 | 0.426 | 0.518 | 0.226 | 0.026 | 0.063 |
| ETTh1_720 | B0 | 0.500 | 0.389 | 0.500 | 0.243 | 0.031 | 0.186 |
| ETTh1_720 | B1 | 0.639 | 0.515 | 0.599 | 0.241 | 0.107 | 0.254 |
| ETTh1_720 | B2 | 0.513 | 0.457 | 0.505 | 0.273 | 0.145 | -0.009 |
| Weather_96 | B0 | 0.500 | 0.513 | 0.500 | 0.251 | 0.054 | -0.022 |
| Weather_96 | B1 | 0.681 | 0.683 | 0.631 | 0.219 | 0.041 | 0.279 |
| Weather_96 | B2 | **0.709** | **0.709** | 0.644 | 0.211 | 0.025 | **0.326** |
| Weather_720 | B0 | 0.500 | 0.340 | 0.500 | 0.244 | 0.180 | -0.151 |
| Weather_720 | B1 | 0.682 | 0.519 | 0.630 | 0.219 | 0.166 | 0.306 |
| Weather_720 | B2 | 0.686 | 0.586 | 0.627 | 0.201 | 0.136 | 0.209 |

**Observability is NOT absent anywhere** -- both B1 and B2 beat B0's
chance-level AUROC (0.500) by a real margin in every cell, and
observability is actually STRONGEST on Weather (AUROC 0.68-0.71),
not ETTh1 (0.51-0.64), despite Weather being the dataset where
retrieval currently fails at Stage2. B2 (the tiny MLP) only clearly
beats B1 (plain logistic regression) on Weather; on ETTh1_720 B2 is
actually WORSE than B1 (0.513 vs 0.639) and barely above chance --
the extra capacity does not help uniformly.

## Table 2 -- deployable binary gate (threshold selected on VAL only, evaluated once on TEST)

| Cell | Threshold | Fraction routed to R (test) | Implied gain vs always-Base (val, used for selection) | Implied gain vs always-Base (test) |
|---|---:|---:|---:|---:|
| ETTh1_96 | 0.55 | 2.1% | +0.00175 | **-0.00197** |
| ETTh1_720 | 0.85 | 3.6% | +0.0632 | **+0.0119** |
| Weather_96 | 0.95 | 6.4% | +0.00733 | **+0.00062** |
| Weather_720 | 0.75 | 9.0% | +0.00820 | **-0.00041** |

**This is the decisive table for Phase B.** Despite real, above-chance
AUROC everywhere (Table 1), the deployable binary gate's val-to-test
generalization is inconsistent: it flips sign on ETTh1_96 and
Weather_720 (positive on val, negative on test -- the gate would have
made things WORSE had it been deployed), and shrinks by roughly 10x
on Weather_96 (val suggested +0.0073, test delivered only +0.0006).
Only ETTh1_720 shows a gain that is positive on both splits and of a
broadly similar order of magnitude (+0.063 val, +0.012 test) --
though even there, the gate only routes 3.6% of queries, so the
absolute impact is small.

## Table 3 -- most informative features (by |Spearman(feature, gain)| on TRAIN)

| Cell | #1 feature | #2 feature |
|---|---|---|
| ETTh1_96 | q_past_std (0.082) | base_std (0.054) |
| ETTh1_720 | v5_head_entropy_mean (0.268) | v5_entropy (0.267) |
| Weather_96 | retrieved_top10_past_dispersion (-0.122) | channel_id_norm (0.083) |
| Weather_720 | retrieved_top10_past_dispersion (-0.169) | retrieved_top10_embed_dispersion (-0.106) |

No single feature family dominates across cells. On ETTh1_720, V5's
OWN confidence (entropy, head agreement) is the strongest signal --
when V5 is confident, retrieval tends to help, a self-consistent and
intuitive result. On Weather (both H96/H720), the NEW
candidate-dispersion features this script added (not present in the
original 18-feature set) are the strongest signal -- when the
actually-retrieved Top-10's own past windows are tightly clustered
(low dispersion), retrieval tends to help; when they are spread out,
it tends to hurt. This is a genuinely new, interpretable finding this
diagnostic surfaced that the reused feature set alone would have
missed.

## Decision gate inputs (feeds Phase C)

Per spec sections 19-21's Case framework, this is closest to a
**fragile/partial Case C**: observability clearly exists (AUROC well
above chance everywhere, strongest on Weather) and is not reducible
to "no better than constant" (Case B) -- but the SPECIFIC deployable
binary-gate operationalization of that signal does not reliably
generalize from validation to test (3 of 4 cells flip sign or shrink
drastically). The one cell with a robust, same-signed, reasonably
-sized val/test gain is ETTh1_720 -- which is also the one cell where
Table 2 of Phase A already showed near-zero global-lambda headroom,
meaning ETTh1_720's deployable opportunity is specifically at the
PER-QUERY level this probe targets, consistent with the whole
picture. Weather's larger AUROC does not translate to a reliable
deployable gain, most likely because Weather's problem (per Phase A)
is a MIX of H1 (real sub-Base retrieval ceiling) and H2 (bad global
lambda) -- a good classifier cannot manufacture net-positive value
from a retrieval source whose best achievable per-query oracle still
sits close to or below Base on 2 of Weather's cells' val-optimal
global lambda (Phase A Table 1: Weather_720's val-opt-global MSE,
0.322, is still slightly worse than Base's 0.319).
