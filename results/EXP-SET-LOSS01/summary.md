# EXP-SET-LOSS01 -- results summary

Full report: `research/EXP-SET-LOSS01.md`.

## Headline

All 8 arms (2 losses x 2 prefix policies x 2 horizons, ETTh1, Set Oracle
only, seed=1) trained and Stage-2-evaluated successfully, no NaN/Inf, no
crashes. 30/30 new unit tests pass, 885/887 full suite (2 pre-existing
failures, no new regressions).

**[ISSUE]**: existing ETTh1 Hard-Choice-CE arms use seed=0, this round
uses seed=1 -- NOT a controlled baseline comparison. No new seed=1
Hard-CE baseline was run without approval.

## Stage-2 MSE vs Independent Base (only true baseline)

| | TF | On-policy |
|---|---:|---:|
| H96 SRM | +3.85% (worse) | -2.19% (better) |
| H96 SoftCE | +1.77% (worse) | **-2.80%** (better) |
| H720 SRM | -2.99% (better) | -3.19% (better) |
| H720 SoftCE | **-3.52%** (better) | -2.74% (better) |

## Recommendation for 3-seed follow-up

**Set-Utility Soft CE, on-policy** -- lowest chosen regret in every cell,
most consistent Stage-2 improvement, largest/most reliable effect under
on-policy (per the H96 sign-flip from worse-than-base under TF to
better-than-base under on-policy).

## Caveat

Single seed only -- none of the deltas above (0.7-3.9%) are distinguishable
from seed noise without a multi-seed run, which was explicitly not
executed this round.
