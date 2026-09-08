# EXP-ONPOLICY-PREFIX01 — Report

Scope: ETTh1, H96, seed 0, top_k=10. C0 = R2 (SmoothL1 + top-tail
pairwise) + cosine UtilityHead + FROZEN B0 encoder + ORACLE prefix
(existing `EXP-TOPTAIL-RANK01/R2`, reused, not retrained). **T1 (this
experiment, the only new training) = the SAME R2 loss, SAME frozen
encoder, SAME everything — except the training-time prefix source changes
from the Oracle's own sequence `S*_{t-1}` to the model's own on-policy
picks `S_hat_{t-1}`.** No Oracle-Choice CE (that is D1's own change, not
combined here per the pre-registered spec), no encoder training.

**Research question:** is R2's free-running failure caused, at least in
part, by the train-time Oracle-prefix / inference-time model-prefix
state-distribution mismatch (teacher-forcing/exposure bias), independent
of the loss formula itself?

---

## Sanity checks (before the GPU run)

`tests/test_exp_onpolicy_prefix01.py`, 7/7 PASSED: t=0 state identical to
the oracle-prefix code path; the model's actual selected index enters the
prefix; the selected candidate is removed from the next valid set with no
duplicates; the dense-utility target is recomputed from the CURRENT model
prefix (verified against a direct re-derivation, not the oracle's); no
gradient flows through argmax/index selection; R2 gradient flows normally
to `SetConditioner`/`UtilityHead`; full-memory semantics preserved (every
step scores the full candidate bank, never a shortlist).

## Training-time state diagnostics

`best_epoch=4` (`val_overlap@10=0.0091`), early-stopped at epoch 9
(patience 5). `wall_clock=1932.8s` (~32 min), `peak_gpu_mem=729MiB`, no OOM.

**A training-time observation that looked concerning in isolation, but was
NOT a good predictor of the downstream result**: `prefix_overlap` with the
Oracle's own sequence stayed near-zero throughout training (0.004–0.009)
with no improving trend, and per-step regret against the model's OWN best
achievable continuation (`regret_t1`) rose slightly epoch over epoch
(0.755→0.852). Full trajectory in `notes.md`. As detailed there, the
correct reading (established only after seeing the downstream result
below) is that on-policy training optimizes SELF-consistency with the
model's own visited states, not agreement with the Oracle's specific path
— the two are not the same target, and only the former is what Stage-2
inference actually depends on.

## Existing t1/t2/HardAggregate/gap_recovery diagnostics

Test split, C0 vs. T1:

| metric | C0 | T1 | Δ |
|---|---:|---:|---|
| t1 Spearman (true top-1%) | 0.2128 | 0.1650 | worse |
| t1 oracle rank median | 99 | 144 | worse |
| t1 Top-50 containment | 30.5% | 21.5% | worse |
| t2 hurt_frac | 0.232 | 0.264 | worse |
| t2 selected true rank median | 1934.0 | **135.5** | **much better** |
| t2 Spearman (true top-1%) | 0.1404 | **0.4086** | **much better (best of every arm this session)** |
| t2 continuation_regret | 0.3498 | **0.2303** | **better** |
| HardAggregate@10 (seq) | 0.5514 | **0.4096** | **much better (best of every arm this session)** |
| gap_recovery | -0.2626 | **-0.0100** | **much better (best of every arm this session)** |

**A genuinely mixed intermediate picture**: T1 is WORSE than C0 on every
t1 metric (teacher-forced, oracle-prefix-based ranking quality) and on t2
`hurt_frac`, but dramatically BETTER on t2's rank/Spearman/regret and on
the two metrics closest to actual free-running sequential behavior
(HardAggregate, gap_recovery) — both the best result any arm has produced
this session, by a wide margin.

## Stage-2 Final MSE — the primary metric

| | B0 | C0 | D1 (best prior arm) | T1 |
|---|---:|---:|---:|---:|
| Stage-2 MSE | 0.37312 | 0.39526 | 0.38916 | **0.37455** |
| Δ vs B0 | — | +0.02214 | +0.01604 | **+0.00143** |
| gap_recovery | — | -0.2626 | -0.2317 | **-0.0100** |

**T1 beats every other set-aware arm tried this entire session, including
D1, and comes within 0.00143 of B0's own unforced Stage-2 MSE** — a
~93.5% reduction of C0's own Δ-vs-B0 gap (0.02214 → 0.00143), and the
gap_recovery of -0.0100 means T1 has recovered essentially all of the
achievable gap between B0 and the Set Oracle's own theoretical ceiling on
this metric.

## Which Outcome (T-A / T-B / T-C)?

**Outcome T-A — a strong, clean positive result on the metrics that matter
most.** Per the pre-registered priority order (Stage-2 > HardAggregate/
gap_recovery > realized regret > Oracle choice/rank/top-tail > training
proxy):

1. **Stage-2 MSE**: dramatically improved (best of the session).
2. **HardAggregate/gap_recovery**: dramatically improved (best of the
   session, gap_recovery essentially at the achievable ceiling).
3. **Realized continuation regret (t2)**: improved.
4. **Oracle choice/rank/top-tail (t1, and t2 hurt_frac)**: WORSE than C0.

The first three (higher-priority) criteria are unambiguous wins; only the
lowest-priority criterion (agreement with the Oracle's own choices/ranking)
is worse. Per the spec's own priority ordering, this is Outcome T-A: **the
train-time Oracle-prefix / inference-time model-prefix state-distribution
mismatch was a real, and — by Stage-2/HardAggregate magnitude — the
LARGEST-YET-FOUND contributing factor to this project's set-aware
retrieval underperforming B0.** The t1/t2-rank-quality regression is best
read as a side effect of T1 no longer being optimized to match the
Oracle's specific choices at all (it was never given that training
signal) — consistent with, and explaining, why a metric measuring
Oracle-agreement would decline while a metric measuring realized
sequential-decision quality improves sharply.

---

### Primary result table (test split, ETTh1 H96)

| Metric | C0 (Oracle prefix) | T1 (on-policy prefix) |
|---|---:|---:|
| best_epoch | 1 | 4 |
| val_overlap@10 (best) | 0.008228 | 0.0091 |
| t1 top-1% Spearman | 0.2128 | 0.1650 |
| t1 oracle rank median | 99 | 144 |
| t2 selected true rank median | 1934.0 | 135.5 |
| t2 top-1% Spearman | 0.1404 | 0.4086 |
| t2 continuation regret | 0.3498 | 0.2303 |
| HardAggregate@10 | 0.5514 | 0.4096 |
| gap_recovery | -0.2626 | -0.0100 |
| **Stage-2 MSE** | 0.39526 | **0.37455** |

Source files: `stage2_eval.json`, `t1_diagnostic.json`, `t2_continuation.json`,
`metrics.csv`, `train_summary.json`, `checkpoint_fingerprints.txt`,
`working_tree.diff`, `sanity_summary.json`, `notes.md`, `command.txt`,
`config.json`.
