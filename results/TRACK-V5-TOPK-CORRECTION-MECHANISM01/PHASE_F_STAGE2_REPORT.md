# PHASE F -- Stage2 Evaluation of S2-v2/D2-v2

All 8 Stage2 caches built via the unmodified `scripts/build_v5_head_loss_stage2_cache01.py` (Mean-Mixture
Top-10 Hard inference, identical to every other arm in this project) and evaluated via the unmodified
`scripts/train_r_stage2_lambda01.py` (Original CARTS Trainable Global Lambda). Base values are this track's
own independently-verified numbers from Phase A (`results/TRACK-V5-TOPK-CORRECTION-MECHANISM01/phaseA/*.json`),
not read off `EXPERIMENT_LOG.md` directly. S0/S1/D0/D1/S2(-v1)/D2(-v1) values are
`TRACK-V5-HEAD-LOSS-DUAL-ENCODER01`'s already-committed numbers (unchanged, reused as-is).

## Full comparison table (all 4 cells, both Stage1 Hard AggMSE and Stage2 Test MSE)

### Stage1 Hard AggMSE

| Cell | S0 | S1 | S2-v1 | **S2-v2** | D0 | D1 | D2-v1 | **D2-v2** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ETTh1_96 | 0.409808 | 0.411558 | 0.412088 | **0.409782** | 0.436703 | 0.431208 | 0.431813 | **0.427649** |
| ETTh1_720 | 0.620570 | 0.626839 | 0.614959 | **0.613478** | 0.602755 | 0.619305 | 0.606237 | **0.599177** |
| Weather_96 | 0.264330 | 0.293258 | 0.204646 | **0.202192** | 0.305465 | 0.306692 | 0.206892 | **0.214717** |
| Weather_720 | 0.617802 | 0.526663 | 0.507839 | **0.498872** | 0.648524 | 0.556950 | 0.535815 | **0.518275** |

### Stage2 Test MSE (the primary metric)

| Cell | Base | S0 | S1 | S2-v1 | **S2-v2** | D0 | D1 | D2-v1 | **D2-v2** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ETTh1_96 | 0.392423 | 0.369597 | 0.370652 | 0.371837 | **0.371040** | 0.376534 | 0.373972 | 0.375010 | **0.373257** |
| ETTh1_720 | 0.560354 | 0.514917 | 0.517572 | 0.514618 | **0.512595** | 0.505956 | 0.512241 | 0.506632 | **0.503495** |
| Weather_96 | 0.169364 | 0.176649 | 0.179473 | 0.168375 | **0.167749** | 0.181700 | 0.178282 | 0.168030 | **0.168352** |
| Weather_720 | 0.319430 | 0.385526 | 0.363349 | 0.359719 | **0.358064** | 0.392910 | 0.374244 | 0.354681 | **0.350882** |

## Does v2's Stage1 improvement transfer to Stage2?

**Yes, in 7 of 8 cases -- the exact same 7/8 pattern as Stage1.** v2 improves over v1 at BOTH stages for:
ETTh1_96 S2 (-0.21%), ETTh1_96 D2 (-0.47%), ETTh1_720 S2 (-0.39%), ETTh1_720 D2 (-0.62%), Weather_96 S2
(-0.37%), Weather_720 S2 (-0.46%), Weather_720 D2 (-1.07%, the single largest Stage2 improvement). The ONE
exception, Weather_96 D2, is worse at BOTH stages (Stage1 +3.78%, Stage2 +0.19%) -- a consistent exception,
not a Stage1/Stage2 disagreement.

**Weather_720 D2-v2 (0.350882) is now the single best Stage2 MSE of all 9 retrieval arms across this
entire two-track research program (DUAL-ENCODER01 + TOPK-CORRECTION-MECHANISM01) on this cell** -- better
than D2-v1 (0.354681), D1, D0, S0/S1/S2-v1/v2, by a clear margin, though still well short of beating Base
(0.319430) on this cell, consistent with Weather_720 remaining the hardest cell for any retrieval arm in
this research program.

## Is the improvement worth v2's ~3x compute cost?

**Marginal, but real and directionally consistent.** The improvements are all small in absolute terms
(0.2-1.1% Stage2 MSE) -- none are large enough on their own to change which arm is "best" on a given cell
(e.g. D0 is still ETTh1_720's best arm overall, not D2-v2; D2-v2 is Weather_720's best arm whether v1 or v2
is used, v2 just makes that win slightly bigger). The value of this fix is less about moving the headline
numbers and more about **confirming the mechanism is working as intended**: v1's documented bug (the K-th
candidate stuck at weight 0.5 regardless of tau) was real and measurably present on real data (Phase D4:
minimum observed K/(K+1) gaps were 2-4 orders of magnitude below the training tau in every cell), and fixing
it produces a small, consistent, same-direction improvement in 7 of 8 Stage1+Stage2 pairs -- exactly the
signature expected if the bug was adding a small amount of noise to an otherwise-correct training signal,
rather than it being irrelevant to real training dynamics. Given the fix is now implemented, tested, and
has no downside risk to the one existing use (`--topk_version` defaults to `v1`, preserving every prior
result unchanged), **using v2 going forward for any new L3b work is recommended on correctness grounds**,
independent of whether the ~0.2-1.1% Stage2 gain alone would justify the extra compute for its own sake.

## The Weather_96 D2 exception

This is the one case where v2 is worse than v1 at both stages. Checking the Phase E training log: D2-v2's
best-retmse checkpoint was selected at epoch 3 (val_retmse10 climbed back up after that, i.e. this run
showed early-epoch overfitting/noisier convergence on this specific cell+arm combination, unlike every
other arm which settled into its best epoch much later: 7-10). This is consistent with ordinary
run-to-run training variance (single seed, no repeat) rather than evidence that v2 is systematically worse
for Separate-encoder arms on Weather_96 -- but it has not been re-run with a different seed to confirm,
and this report does not claim it is noise without that check; it is reported as the one real exception.

## Validation-Optimal Global Lambda secondary control

Not computed this round (deferred) -- would require a dedicated ternary/grid search over the Stage2 cache
tensors per spec section 9, kept clearly separate from the Trainable Lambda numbers above. Given the
Trainable Lambda values here are all close to 0.4-0.5 (not near a degenerate 0 or 1), a validation-optimal
sweep is unlikely to change the ranking conclusions above, but this is a documented gap, not a verified
null result.
