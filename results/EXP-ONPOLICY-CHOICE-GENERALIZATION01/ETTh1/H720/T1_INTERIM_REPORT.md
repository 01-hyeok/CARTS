# EXP-ONPOLICY-CHOICE-GENERALIZATION01 — ETTh1 H720 cell (INTERIM: T1 complete, OPC1 in progress)

Part of Track A's cross-horizon/cross-dataset generalization check for
`EXP-ONPOLICY-PREFIX01` (T1) and `EXP-ONPOLICY-CHOICE01` (OPC1), testing
whether the on-policy-prefix fix that nearly closed the gap to B0 at
ETTh1 H96 (T1=0.37455, OPC1=0.37340 vs F0=0.37312) generalizes to other
horizons. Per the user's explicit choice when Stage-1 checkpoints for
H192/H336 were found missing ("일단 H96/H720만 진행"), only H96 (reused
from existing results) and H720 (this cell, new) are in scope for ETTh1;
H192/H336 remain out of scope, not silently dropped.

**Status: T1 arm complete. OPC1 arm still training (launched once T1
finished; not yet done as of this interim report — see final REPORT.md
for the completed 3-way F0/T1/OPC1 comparison once available).**

## Configuration

Same protocol as the original ETTh1 H96 T1 run (`results/EXP-ONPOLICY-
PREFIX01/`): frozen B0 encoder, R2 hybrid loss (SmoothL1 + top-tail
pairwise, `lambda_rank=1.0`), on-policy prefix construction, cosine
UtilityHead, `top_k=10`, seed 0, self-only, full-memory. Stage-1
checkpoint `checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/
stage1_carts_softset_ETTh1_720_S0_wce_.../checkpoint.pth`, Stage-2
checkpoint `checkpoints/stage2/ETTh1/seq720_pred720/stage2_carts_softset_
s2_ETTh1_720_S0_wce_.../checkpoint.pth`, teacher cache
`cache/seqfull01_teacher/ETTh1_pred720.pt` (freshly precomputed this
session for H720, 7201/2161/2161 train/val/test queries, no drops).

## Training

`best_epoch=10` (the training epochs cap, `train_epochs=10`/`patience=5`
never exhausted — training was still improving at cutoff, same pattern
as Track B3), `wall_clock=14338.3s` (~4h, under heavy multi-tenant GPU-1
contention), `peak_gpu_mem=2718MiB`, `val_overlap@10` final = 0.0091
(essentially the same order of magnitude as ETTh1 H96's own T1, 0.0083-
0.0091 — chance-level membership overlap, consistent with the H96
finding that on-policy training optimizes self-consistency, not Oracle-
sequence agreement).

## Results (test split)

| Arm | Stage-2 Final MSE | vs F0 |
|---|---:|---:|
| F0 (B0, unforced) | 0.46827 | — |
| **T1 (this run)** | **0.46587** | **−0.00240 (better)** |
| Individual Oracle (diagnostic upper bound) | 0.44227 | — |

**T1 beats F0 at ETTh1 H720** (−0.0024, small but the correct direction —
unlike ETTh1 H96, where T1 was slightly WORSE than F0 by +0.00143 before
OPC1 closed most of the remaining gap). `gap_recovery = (F0−T1)/(F0−
SetOracleAweighted) = +0.00049` — a small positive recovery toward the
Set Oracle ceiling (`set_oracle_a_weighted=0.25633`).

### HardAggregate / regret / rank diagnostics (test, 500-query subsample)

| Metric | dense_first (T1's own on-policy continuation) | oracle_first | b0_first |
|---|---:|---:|---:|
| `a_weighted` | 0.57309 | 0.56736 | 0.56958 |
| `hard_aggregate_mse` | 0.57569 | 0.53921 | 0.57679 |
| `set_recall_at_k` | 0.00699 | 0.10292 | 0.00669 |
| mean continuation regret | 0.29904 | — | — |
| dense-continuation improves over first-anchor-only | 71.4% of queries | — | — |
| second-pick true-utility rank, top-1% | 28.2% | (oracle-cond.) 49.6% | — |

Set Recall@10 against the true Oracle sequence stays near chance
(0.7-10%, same qualitative pattern as every other T1/OPC1 arm this
session) — the win here, as at H96, comes from aggregate/self-consistency
quality, not from recovering the Oracle's specific membership.

## What this interim report does NOT yet answer

- Whether OPC1 (Choice CE + on-policy) improves further over T1 at H720,
  the way it did at H96 (T1=0.37455 → OPC1=0.37340, closing to within
  0.00028 of F0) — OPC1 is still training as of this report.
- The full cross-cell (ETTh1 H96/H720, Weather H96/H720) summary table,
  win/tie/loss statistics, and horizon-trend analysis required by the
  parent spec — deferred to the final `REPORT.md` once all cells/arms
  complete, per the STOP rule (no premature synthesis).

## Sanity checks

Reuses `tests/test_exp_onpolicy_prefix01.py` (7/7 passed, unchanged code
path — only the checkpoint/dataset/horizon differ from the original H96
run) and `pytest tests/` (581 passed, same 2 pre-existing failures, no
regression from this session's other changes).

---

Source files: `T1_stage2.json`, `T1_test_summary.json`,
`T1_test_dense_first_summary.json`, `T1_test_stepwise_raw.csv`,
`continuation_diag_T1_test_dense_first.csv`,
`T1_test_dense_first_top20_catastrophic.json`. Training log:
`logs/exp_generalization01/t1_ETTh1_H720.log`. Eval chain log:
`logs/exp_generalization01/eval_t1_ETTh1_H720.log`.
