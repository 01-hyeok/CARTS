# EXP-ORACLE-CHOICE01 — Report

Scope: ETTh1, H96, seed 0, top_k=10, tau=0.1 (= base checkpoint's own
`tau_topk`, no sweep). C0 = R2 (SmoothL1 + top-tail pairwise) + cosine
UtilityHead + FROZEN B0 encoder (existing `EXP-TOPTAIL-RANK01/R2`, reused,
not retrained). **D1 (this experiment, the only new training) = R2's
SmoothL1+pairwise surrogate REPLACED entirely by a single masked
full-memory Oracle-Choice Cross-Entropy loss** — same frozen encoder, same
`SetConditioner`/`EmptySetToken`/`UtilityHead`, same optimizer/LR/epochs/
patience, same Stage-2 protocol.

**Research question:** does the SmoothL1+pairwise surrogate insufficiently
target the Set Oracle's actual greedy decision, such that training the
model to directly predict the Oracle's next pick improves free-running
selection and Stage-2?

---

## Pre-training diagnostic: is one-hot Oracle-Choice CE well-posed?

Run BEFORE any training (`scripts/diag_oracle_margin01.py`,
`oracle_margin_diagnostic.json`), per spec — interpretive only, the loss
was not changed in response. **The Oracle's true top-1-vs-top-2 utility
margin collapses sharply as the selected set grows**: `near_tie_frac` rises
from 8.5% at t=1 to 99.0% at t=10; `rel_margin_mean` falls from 0.136 to
0.0008. One-hot CE targets become increasingly ill-posed at later steps —
this is a genuine LIMITATION of D1's approach, not a bug, and is the most
plausible explanation for why D1's teacher-forced top-1 accuracy stays
very low throughout (see below), independent of whether the downstream
result is positive.

## Sanity checks (before the GPU run)

`tests/test_exp_oracle_choice01.py`, 4/4 PASSED: (1)+(2) the softmax
denominator includes every valid candidate, invalid positions get exactly
zero probability; (3) the Oracle label equals `argmax` of the true dense
utility exactly; (4) a synthetic small-N positive control fits to >90%
top-1 accuracy, confirming the loss/training loop CAN learn a clean
choice signal when one exists; (5) `dense_utility` chunking does not
change the resulting logits/loss (atol 1e-6).

## Teacher-forced decision metrics

Test split (`oracle_choice_diag_test.json`), mean across K=10 steps:
`top1_acc=0.0036`, `top5_acc=0.0161`, `top10_acc=0.0415`,
`pred_rank_mean=380.8`, `step_regret=0.2956`. At t=1 specifically (least
affected by the near-tie problem): `top1_acc=0.0134`, `pred_rank_mean=67.7`.
At t=2: `top1_acc=0.0045`, `pred_rank_mean=153.0`.

**Oracle EXACT next-choice reproduction stays very low throughout** —
consistent with the near-tie diagnostic above. Despite this, D1's
DOWNSTREAM realized-selection quality (below) is the best of any arm this
session.

## Existing t1/t2/HardAggregate/gap_recovery diagnostics

Test split, C0 vs. D1:

| metric | C0 | D1 | Δ |
|---|---:|---:|---|
| t1 Spearman (true top-1%) | 0.2128 | 0.2832 | **better** |
| t1 oracle rank median | 99 | 47 | **better** |
| t1 Top-50 containment | 30.5% | 52.0% | **better** |
| t2 hurt_frac | 0.232 | 0.122 | **better** |
| t2 selected true rank median | 1934.0 | 88.0 | **much better** |
| t2 oracle predicted rank median | 560.0 | 42.5 | **much better** |
| t2 Spearman (true top-1%) | 0.1404 | 0.3294 | **much better** |
| t2 continuation_regret | 0.3498 | 0.1964 | **much better** |
| HardAggregate@10 (seq) | 0.5514 | 0.5480 | **better** |
| gap_recovery | -0.2626 | -0.2317 | **better** |

**Every single one of these metrics improves for D1 over C0**, several by
a large margin (t2 selected true rank median drops from 1934 to 88; t2
continuation_regret nearly halves). This is the first arm this session
where every intermediate/decision-relevant diagnostic points the same
direction.

## Stage-2 Final MSE — the primary metric

| | B0 | C0 | D1 |
|---|---:|---:|---:|
| Stage-2 MSE | 0.37312 | 0.39526 | **0.38916** |
| Δ vs B0 | — | +0.02214 | +0.01604 |
| gap_recovery | — | -0.2626 | -0.2317 |

**D1 beats C0 on the primary metric** (0.38916 < 0.39526, Δ=-0.0061,
recovering about 27% of C0's own gap to B0). This is the first arm this
whole session (across EXP-MARGUTIL01, EXP-TOPTAIL-RANK01, EXP-ASYM-SCORER01,
EXP-STRONG-SCORER-DIAG01, EXP-ENCODER-UNFREEZE01, EXP-ENCODER-ANCHOR01) to
improve Stage-2 MSE over C0. D1 is still worse than B0's own unforced
Stage-2 (0.38916 vs 0.37312) — retrieval-augmented selection still has not
closed the full gap to the base forecaster, but the direction and
magnitude of improvement over the best prior set-aware arm is genuine and
consistent across every metric measured.

## Which Outcome (L-A / L-B / L-C)?

**Outcome L-A**: D1 improves Oracle next-choice PROXY metrics (t1/t2 rank/
Spearman/regret — note: NOT raw teacher-forced top-1 exact-match accuracy,
which stays low due to the near-tie problem, but the ranking-quality
metrics this project treats as decision-relevant), improves free-running
HardAggregate, AND improves Stage-2 MSE over C0.

Per the spec: "기존 R2 surrogate가 Set Oracle의 실제 greedy decision과
충분히 정렬되지 않은 것이 실제 병목의 한 부분이라는 증거" — **this
result is evidence that R2's SmoothL1+pairwise surrogate WAS
insufficiently aligned with the Set Oracle's actual greedy decision, and
that a more direct choice-prediction objective (even one with a
demonstrable near-tie/label-noise limitation, per the margin diagnostic)
recovers real Stage-2 value.**

## Notable training-dynamics observation (not over-interpreted)

D1's `val_overlap@10` improved every single epoch through all 10
configured epochs (never early-stopped), unlike every other arm this
session (R1, R2, C1, C2, E1, E2), which all peaked at epoch 1. This is
reported as an observation, not a claim about WHY — a longer training
budget was never tried for R2 either, so this is not a controlled
comparison of training dynamics, and it should not be read as "D1 trains
better" in a general sense, only as a fact about this specific run.

---

### Primary result table (test split, ETTh1 H96)

| Metric | C0 (R2) | D1 (Oracle-Choice CE) |
|---|---:|---:|
| best_epoch | 1 | 10 (max configured, never early-stopped) |
| val_overlap@10 (best) | 0.008228 | 0.0146 |
| t1 top-1% Spearman | 0.2128 | 0.2832 |
| t1 oracle rank median | 99 | 47 |
| t1 Top-50 containment | 30.5% | 52.0% |
| t2 selected true rank median | 1934.0 | 88.0 |
| t2 continuation regret | 0.3498 | 0.1964 |
| HardAggregate@10 | 0.5514 | 0.5480 |
| gap_recovery | -0.2626 | -0.2317 |
| **Stage-2 MSE** | 0.39526 | **0.38916** |

Source files: `stage2_eval.json`, `t1_diagnostic.json`, `t2_continuation.json`,
`oracle_choice_diag_test.json`, `oracle_choice_diag_train.json`,
`oracle_margin_diagnostic.json`, `metrics.csv`, `train_summary.json`,
`checkpoint_fingerprints.txt`, `working_tree.diff`, `sanity_summary.json`,
`notes.md`, `command.txt`, `config.json`.
