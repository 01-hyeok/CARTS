# PHASE A -- Semantic Oracle Complementarity Diagnostic

No model, no training. Reuses `scripts/diag_experiment_e_teacher01.py`'s
own `shared`/`local`/`trend`/`seasonal` component-distance definitions
UNMODIFIED, restricted here to the Shared-P100 candidate pool
(Experiment E's own script used the full candidate bank; P100 keeps
this comparable with every other track this session). Period `P` for
each dataset is read from the EXISTING train-only
`period_detection.json` (ETTh1 -> 24, Weather -> 142) -- confirmed
reusable across horizons for the same dataset: both
`Dataset_ETT_hour`/`Dataset_Custom` (`data_provider/data_loader.py`)
fix the train split's boundary (`border1s[0]=0`, `border2s[0]`=a
constant) independent of `seq_len`/`pred_len`, so the train-only ACF
period is identical for H96 and H720 of the same dataset; no new
period computation was done. VALIDATION split only (never test).
Code: `scripts/diag_semantic_oracle_heads01.py`.

## Known issue found during this run: Trend/Seasonal are UNDEFINED for Weather_96

`transform(y, 'trend'/'seasonal', period=142)` requires `period <
H`. Weather's horizon-96 cell has `H=96 < period=142`, so the
moving-average/seasonal-difference window described in
`diag_experiment_e_teacher01.py` produces an EMPTY tensor (`shape
[B,0]`), verified directly:
```
transform(torch.randn(2,96), 'trend', 142).shape    -> torch.Size([2, 0])
transform(torch.randn(2,96), 'seasonal', 142).shape  -> torch.Size([2, 0])
```
This is not a bug in `transform` -- period-142 trend/seasonal
decomposition is mathematically undefined on a 96-step horizon -- but
it means Weather_96's `T`/`S`/`GT`/`GS`/`GTL`/`GTLS` numbers below are
**invalid and excluded from the verdict for that cell**; only `G`,
`L`, `GL` are valid for Weather_96. This was not anticipated by the
spec (no period-capping fallback was specified) and no new period
-capping rule was invented to work around it -- reported as-is.

## Fixed-combination raw-future AggMSE (validation)

| Cell | G | L | T | S | GT | GL | GS | GTL | GTLS | best fixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| ETTh1_96 | 0.3676 | 0.7077 | 0.4280 | 0.6803 | 0.3829 | 0.4870 | 0.4696 | 0.4439 | 0.4772 | **G** |
| ETTh1_720 | 1.0997 | 1.7649 | 1.1453 | 1.8192 | 1.1084 | 1.3138 | 1.3255 | 1.2213 | 1.3112 | **G** |
| Weather_96 | 0.2841 | 0.5019 | *invalid* | *invalid* | *invalid* | 0.3576 | *invalid* | *invalid* | *invalid* | **G** (of valid arms) |
| Weather_720 | 0.5502 | 0.8671 | 0.5757 | 0.8063 | **0.5425** | 0.6297 | 0.6090 | 0.5794 | 0.6011 | **GT** |

In 3/4 cells, plain Global-only is already the best of all 9 arms --
no semantic fusion beats it. Only Weather_720 shows any fusion
(Global+Trend) beating Global alone, and by only
`(0.5502-0.5425)/0.5502 = 1.40%`.

## Pairwise Top-10 overlap (component distinctiveness)

| Cell | G-L | G-T | G-S | L-T | L-S | T-S |
|---|---:|---:|---:|---:|---:|---:|
| ETTh1_96 | 0.289 | 0.549 | 0.337 | 0.157 | 0.291 | 0.261 |
| ETTh1_720 | 0.226 | 0.695 | 0.225 | 0.155 | 0.255 | 0.209 |
| Weather_96 | 0.320 | *invalid* | *invalid* | *invalid* | *invalid* | **1.000 (degenerate, see above)** |
| Weather_720 | 0.283 | 0.625 | 0.380 | 0.204 | 0.364 | 0.289 |

Components ARE distinctive by this metric (several pairs well under
0.7, e.g. G-L everywhere, L-T everywhere) -- the lack of fusion benefit
is not because the components pick identical candidates.

## Restricted per-query oracle headroom (diagnostic only, not deployable)

| Cell | best_fixed AggMSE | restricted oracle AggMSE | oracle gain vs best_fixed | oracle gain vs G alone |
|---|---:|---:|---:|---:|
| ETTh1_96 | 0.3676 | 0.3608 | 1.86% | 1.86% |
| ETTh1_720 | 1.0997 | 1.0888 | 0.99% | 0.99% |
| Weather_96 | 0.2841 | 0.2825 | 0.57% | 0.57% |
| Weather_720 | 0.5425 | 0.5117 | 5.69% | 6.99% |

Even the per-query CHEATING oracle (picks the single best of all 9
fixed arms, per query, with future information) only buys 0.6-5.7% --
real headroom exists at Weather_720 but is modest everywhere else.

## Winner share (fraction of query/channel rows where each arm is individually best)

| Cell | G | L | T | S | GT | GL | GS | GTL | GTLS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ETTh1_96 | 55.0% | 0.7% | 8.3% | 2.1% | 17.9% | 3.9% | 7.4% | 2.6% | 1.9% |
| ETTh1_720 | 50.1% | 0.8% | 8.8% | 0.4% | 16.6% | 5.2% | 5.0% | 7.7% | 5.4% |
| Weather_96 | 78.5% | 3.2% | 0.9%(invalid) | 0.0%(invalid) | 4.4%(invalid) | 8.1% | 0.0%(invalid) | 3.6%(invalid) | 1.3%(invalid) |
| Weather_720 | 33.4% | 2.5% | 11.4% | 2.4% | 17.1% | 6.0% | 6.1% | 12.1% | 9.0% |

Global wins the plurality of individual queries everywhere (33-79%),
consistent with it being the best FIXED arm everywhere except
Weather_720.

## PHASE A GO condition, checked directly

Required: best semantic fixed fusion beats Global-only in >=3/4
cells, AND >=2/4 cells with >=2% improvement, AND at least one useful
component has `overlap(Global, component) <= 0.7`.

| Condition | Required | Actual | Pass? |
|---|---|---|---|
| Best fusion < Global in >=3/4 cells | >=3/4 | **1/4** (Weather_720 only) | **FAIL** |
| >=2/4 cells with >=2% improvement | >=2/4 | **0/4** (max 1.40%, Weather_720) | **FAIL** |
| Distinctiveness (overlap<=0.7 for a useful component) | n/a (moot, no useful component found) | -- | -- |

## Verdict: Scenario D -- STOP

> Global 외 어떤 component도 2% headroom 없음.

Neither gate condition is met. Per the pre-registered rule, **PHASE B
(Semantic-Specialized-V5 training) is NOT started.** No lambda/role
sweep, no architecture change, no attempt to force a passing result.

This is consistent with, and adds a direct causal mechanism to, the
session's standing caveat about Experiment E (seasonal specialization
weak/failed, component fusion rarely beating Shared) -- here, with a
CLEAN oracle-only diagnostic (no learned model confound at all),
semantic component fusion essentially never beats the existing
Global/Shared teacher on raw-future forecasting, at the P100 scale
tested.
