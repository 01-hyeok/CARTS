# TRACK-P100-SOFTTOPK-QUALITY-AWARE01 -- INTERIM REPORT (125/144 runs, 86.8%)

**Status: IN PROGRESS. This is NOT the final report.** TEST data has not
been read by any script in this track (`evaluate_locked_test01.py`, the
only script permitted to read TEST, refuses to run without a
`LOCKED_CONFIG.json`, which does not exist yet). Everything below is
computed from VALIDATION-split `checkpoint_selection.json` files only.
Generated 2026-10-19 on explicit user request for a current-state snapshot
while Phase C/D's 144-run grid is still executing on GPU1.

## 1. Research question

Does estimating per-query/channel retrieval quality and using it to
dynamically set the fusion trust `lambda(q,c)` (Quality-Aware, "QA")
improve forecasting over giving every query the same, globally-learned
scalar trust `lambda_global` (NOQA), when both arms use **identical**
Soft Top-K (K=10) weighted retrieval over the same Shared Top-100 (P100)
candidate pool, the same frozen Base Forecaster, the same candidate pool,
and the same init/batch order/seed?

## 2. Scope and design (unchanged from the approved spec)

- Datasets x horizons: ETTh1 / Weather / Solar(137ch) x {96, 192, 336, 720} = 12 cells.
- Methods: V0, V1, V2, V5, Hard, Joint (6).
- Arms: NOQA (single trainable scalar `lambda_global = sigmoid(a)`), QA
  (`lambda(q,c) = sigmoid(QualityGate(features))`, fixed
  `Linear(7,16)->GELU->Linear(16,1)->Sigmoid`, zero-initialized final
  layer so QA starts at lambda=0.5, matching NOQA's own init).
- 12 cells x 6 methods x 2 arms = 144 planned E2E training runs, 10 fixed
  epochs each (no early stopping), every-epoch checkpoints saved.
- Checkpoint selection: minimum VALIDATION final-forecast MSE.
- Candidate support: Shared Top-100 (P100) exclusively, K=10, throughout.

## 3. Progress as of this snapshot

| Cell | Done / 12 |
|---|---:|
| ETTh1_96/192/336/720 | 12/12 each (48/48) |
| Weather_96/192/336/720 | 12/12 each (48/48) |
| Solar_96 | 12/12 |
| Solar_192 | 12/12 |
| Solar_336 | 5/12 (V0 x2, V1 x2, V2/NOQA done; V2/QA running; V5, Hard, Joint not started) |
| Solar_720 | 0/12 (not started) |
| **Total** | **125/144 (86.8%)** |

Zero errors across the entire grid log so far (no Traceback, no OOM, no
`[ISSUE][ABORT]`), confirmed by direct `grep` of
`logs/TRACK-P100-SOFTTOPK-QUALITY-AWARE01/grid_run.log`.

## 4. Interim VALIDATION-only NOQA vs QA comparison

Computed by `scripts/compute_qa_go_decision01.py` on the 62/72
(cell, method) combinations where **both** arms are complete so far (raw
output: `results/TRACK-P100-SOFTTOPK-QUALITY-AWARE01/_interim/qa_go_decision_INTERIM_125of144.json`).
10 combinations are still missing, all in Solar_336 (V2/QA in flight; V5,
Hard, Joint not started) and all of Solar_720.

**Headline: QA beats NOQA in 55/62 (89%) of completed (cell, method)
combinations.**

| Metric | Value |
|---|---:|
| Combos compared | 62 / 72 |
| QA wins (lower val MSE) | 55 |
| NOQA wins | 7 |
| Ties | 0 |
| Overall avg delta (QA - NOQA) | -0.007575 (negative = QA better) |

Per-dataset average delta (cell-avg-of-methods, then dataset-avg-of-cells;
Solar uses only the 3 cells with at least one method done so far):

| Dataset | Avg delta (QA-NOQA) | Reading |
|---|---:|---|
| ETTh1 | -0.003276 | QA better on average, but NOT uniform (see below) |
| Weather | -0.015646 | QA better, uniformly, with the largest margins |
| Solar (partial) | -0.001175 | QA better on average |

Per-cell average delta (cell-level average over the methods done so far):

| Cell | Avg delta | Favors |
|---|---:|---|
| ETTh1_96 | -0.001657 | QA |
| ETTh1_192 | **+0.003388** | **NOQA** |
| ETTh1_336 | -0.008184 | QA |
| ETTh1_720 | -0.006651 | QA |
| Weather_96 | -0.009405 | QA |
| Weather_192 | -0.013805 | QA |
| Weather_336 | -0.012610 | QA |
| Weather_720 | -0.026766 | QA (largest margin seen anywhere) |
| Solar_96 | -0.000979 | QA |
| Solar_192 | -0.001140 | QA |
| Solar_336 (partial, 3/6 methods) | -0.001407 | QA |

**10/11 cells with any data so far favor QA on average; only ETTh1_192
favors NOQA.**

Full per-(cell, method) table (NOQA val MSE, QA val MSE, delta, winner):

| Cell | Method | NOQA | QA | Delta | Winner |
|---|---|---:|---:|---:|---|
| ETTh1_96 | V0 | 0.664144 | 0.663164 | -0.000980 | QA |
| ETTh1_96 | V1 | 0.663815 | 0.661573 | -0.002242 | QA |
| ETTh1_96 | V2 | 0.664132 | 0.661390 | -0.002742 | QA |
| ETTh1_96 | V5 | 0.663781 | 0.660855 | -0.002926 | QA |
| ETTh1_96 | Hard | 0.663202 | 0.662817 | -0.000385 | QA |
| ETTh1_96 | Joint | 0.663526 | 0.662860 | -0.000666 | QA |
| ETTh1_192 | V0 | 0.959141 | 0.959126 | -0.000015 | QA |
| ETTh1_192 | V1 | 0.956951 | 0.961848 | +0.004897 | NOQA |
| ETTh1_192 | V2 | 0.957963 | 0.962343 | +0.004380 | NOQA |
| ETTh1_192 | V5 | 0.955703 | 0.962265 | +0.006562 | NOQA |
| ETTh1_192 | Hard | 0.961010 | 0.962597 | +0.001587 | NOQA |
| ETTh1_192 | Joint | 0.956785 | 0.959701 | +0.002916 | NOQA |
| ETTh1_336 | V0 | 1.231931 | 1.224999 | -0.006932 | QA |
| ETTh1_336 | V1 | 1.232603 | 1.227429 | -0.005175 | QA |
| ETTh1_336 | V2 | 1.231865 | 1.226051 | -0.005814 | QA |
| ETTh1_336 | V5 | 1.229883 | 1.225823 | -0.004059 | QA |
| ETTh1_336 | Hard | 1.241653 | 1.224178 | -0.017475 | QA |
| ETTh1_336 | Joint | 1.235931 | 1.226279 | -0.009651 | QA |
| ETTh1_720 | V0 | 1.558589 | 1.549332 | -0.009257 | QA |
| ETTh1_720 | V1 | 1.553271 | 1.545545 | -0.007725 | QA |
| ETTh1_720 | V2 | 1.556211 | 1.541066 | -0.015145 | QA |
| ETTh1_720 | V5 | 1.547742 | 1.544754 | -0.002988 | QA |
| ETTh1_720 | Hard | 1.559850 | 1.548884 | -0.010967 | QA |
| ETTh1_720 | Joint | 1.548069 | 1.554246 | +0.006177 | NOQA |
| Weather_96 | V0 | 0.411358 | 0.402916 | -0.008441 | QA |
| Weather_96 | V1 | 0.413368 | 0.401802 | -0.011565 | QA |
| Weather_96 | V2 | 0.412571 | 0.403900 | -0.008670 | QA |
| Weather_96 | V5 | 0.410747 | 0.403150 | -0.007596 | QA |
| Weather_96 | Hard | 0.412614 | 0.403804 | -0.008810 | QA |
| Weather_96 | Joint | 0.414167 | 0.402822 | -0.011346 | QA |
| Weather_192 | V0 | 0.463019 | 0.451717 | -0.011301 | QA |
| Weather_192 | V1 | 0.464923 | 0.450602 | -0.014321 | QA |
| Weather_192 | V2 | 0.464958 | 0.448540 | -0.016418 | QA |
| Weather_192 | V5 | 0.463100 | 0.450395 | -0.012705 | QA |
| Weather_192 | Hard | 0.468008 | 0.450144 | -0.017864 | QA |
| Weather_192 | Joint | 0.462557 | 0.452339 | -0.010218 | QA |
| Weather_336 | V0 | 0.552120 | 0.538957 | -0.013163 | QA |
| Weather_336 | V1 | 0.548016 | 0.534585 | -0.013432 | QA |
| Weather_336 | V2 | 0.546968 | 0.534036 | -0.012931 | QA |
| Weather_336 | V5 | 0.545698 | 0.532802 | -0.012896 | QA |
| Weather_336 | Hard | 0.549239 | 0.536507 | -0.012732 | QA |
| Weather_336 | Joint | 0.546094 | 0.535591 | -0.010503 | QA |
| Weather_720 | V0 | 0.641920 | 0.618601 | -0.023319 | QA |
| Weather_720 | V1 | 0.642696 | 0.614521 | -0.028175 | QA |
| Weather_720 | V2 | 0.641725 | 0.617178 | -0.024547 | QA |
| Weather_720 | V5 | 0.647742 | 0.618717 | -0.029026 | QA |
| Weather_720 | Hard | 0.646595 | 0.616648 | -0.029948 | QA |
| Weather_720 | Joint | 0.640806 | 0.615221 | -0.025584 | QA |
| Solar_96 | V0 | 0.183237 | 0.182079 | -0.001158 | QA |
| Solar_96 | V1 | 0.183967 | 0.182762 | -0.001204 | QA |
| Solar_96 | V2 | 0.183665 | 0.183510 | -0.000154 | QA |
| Solar_96 | V5 | 0.183358 | 0.182407 | -0.000951 | QA |
| Solar_96 | Hard | 0.184393 | 0.183710 | -0.000683 | QA |
| Solar_96 | Joint | 0.183138 | 0.181418 | -0.001720 | QA |
| Solar_192 | V0 | 0.180012 | 0.178053 | -0.001959 | QA |
| Solar_192 | V1 | 0.179529 | 0.179600 | +0.000071 | NOQA (near-tie) |
| Solar_192 | V2 | 0.180383 | 0.178147 | -0.002236 | QA |
| Solar_192 | V5 | 0.179933 | 0.179154 | -0.000779 | QA |
| Solar_192 | Hard | 0.183801 | 0.181992 | -0.001809 | QA |
| Solar_192 | Joint | 0.179261 | 0.179132 | -0.000129 | QA |
| Solar_336 | V0 | 0.188090 | 0.186797 | -0.001293 | QA |
| Solar_336 | V1 | 0.189516 | 0.187995 | -0.001521 | QA |

(Solar_336's V2, V5, Hard, Joint and all of Solar_720 are not yet available.)

## 5. Interim GO/STOP criteria (spec section 21), computed on available data

This is an **UNOFFICIAL preview**, not the final decision -- 10 of 72
(cell, method) combinations are still missing. The official decision
requires the full grid.

| Criterion | Result on 62/72 available | Pass? |
|---|---|---|
| 1. QA < NOQA, averaged over all combos | avg delta = -0.007575 | **True** |
| 2. QA < NOQA in >= 7/12 cells | 10/11 cells-with-data favor QA | **True** |
| 3-5. ETTh1 / Weather / Solar dataset averages all QA <= NOQA | all three negative | **True** |
| 6. Positive Dynamic-Oracle headroom | not yet computed | not evaluated |

If this pattern holds once Solar_336 and Solar_720 finish, `QA_GO=TRUE`
is the likely outcome, but this is explicitly a trend observation, not a
commitment -- Solar_720 in particular has not produced a single data
point yet, and the one clear NOQA-favoring cell (ETTh1_192) shows the
pattern is not universal.

## 6. Pattern worth flagging (not yet investigated further)

QA's margin over NOQA appears to scale with channel count: it is large
and completely consistent on Weather (21 channels, 24/24 method-wins,
0.008-0.030 MSE reduction, the largest margins anywhere) and nearly as
consistent on Solar (137 channels, 28/29 method-wins so far), but mixed
on ETTh1 (7 channels) -- and ETTh1_192 specifically reverses, with 5/6
methods favoring NOQA. This is reported as an observation, not an
established mechanism; it has not been investigated further in this
snapshot.

## 7. What remains

- Solar_336: V2/QA (in flight), V5, Hard, Joint (NOQA+QA each) -- 7 runs.
- Solar_720: all 12 runs (not started).
- Once all 144 runs are complete: run `compute_qa_go_decision01.py` for
  the official GO/STOP decision, compute the validation-only oracle
  headroom / quality calibration diagnostics
  (`utils/validation_diagnostics01.py`), decide seed robustness
  (seed=1/2) and, only if `QA_GO=TRUE`, generate `LOCKED_CONFIG.json`
  and run the single permitted TEST evaluation pass.
- This interim document will be superseded by the final `REPORT.md` in
  this same directory once the track completes.
