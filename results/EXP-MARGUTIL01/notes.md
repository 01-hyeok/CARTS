# EXP-MARGUTIL01 -- factual notes (final for the approved scope, 2026-09-07)

Status: **3 of the original 4 cells COMPLETE (ETTh1 H96, Weather H96, ETTh1
H720). Weather H720 CANCELLED by explicit user decision (2026-09-07),
scope reduced to 3 cells.** No result exists or is reported for Weather
H720 -- it is not fabricated, approximated, or interpolated from the other
cells.

## What happened to Weather H720

Training launched normally (frozen B0 encoder loaded, `encoder_grad_none`
confirmed each step), ran on GPU 1 for approximately 2 hours without
completing a single epoch (Weather's 35448-candidate, 21-channel, 720-step
horizon full-memory dense-utility computation is substantially heavier than
the other three cells). The user then asked to run a follow-up diagnostic
(EXP-FIRSTANCHOR-DIAG) in parallel on GPU 0 using the 3 already-complete
cells' checkpoints; part-way through, the user decided to stop and remove
the Weather H720 run entirely rather than let it continue. The process was
killed (`kill <pid>`, not `-9`, clean SIGTERM); no checkpoint had ever been
saved for this cell (best_val_overlap was never reached because epoch 1
never finished), so nothing was left to clean up beyond an empty checkpoint
directory. This is a **scope decision**, not an experimental failure --
see `research/RESEARCH_DECISIONS.md` D-0013.

## Design (unchanged from the approved spec)

Frozen B0 encoder throughout (no trainable-encoder re-comparison -- that
question was already answered by EXP-SEQDIAG01). Dense per-candidate
set-utility regression target, `u_i^(t) = -A_weighted(S*_{t-1} + {i})`, at
every teacher-forced step (oracle prefix from the cached
`select_greedy_weighted_set` sequence), SmoothL1 loss, minimal
`SetConditioner` + affine `UtilityHead` (`a*cosine+b`), full memory scored
at every step (chunked for GPU memory, never shortlisted).

## Results (3 cells)

| Cell | B0 unforced | Dense forced | Delta vs B0 | gap_recovery | Utility Spearman | HardAgg (seq/b0) |
|---|---:|---:|---:|---:|---:|---:|
| ETTh1 H96 | 0.37312 | 0.52991 | **+0.157** | -1.832 | 0.543 | 1.657 / 0.407 |
| Weather H96 | 0.17553 | 0.33253 | **+0.157** | -1.744 | 0.658 | 4.465 / 0.234 |
| ETTh1 H720 | 0.46827 | 0.49269 | **+0.024** | -1.866 | 0.383 | 1.961 / 0.572 |

Full numbers: `comparison.csv`. `duplicate_rate` / `invalid_rate` = 0.0 on
every cell.

Per the pre-registered success/stop rule (meaningful improvement ≤ B0−0.01,
meaningful worse ≥ B0+0.01, noise-scale <0.01): **all 3 completed cells are
meaningfully WORSE than B0**, well past the 0.01 threshold. 0 of 3 cells
show any Stage-2 improvement.

## Sanity

`pytest tests/`: 485 passed (11 new in `tests/test_exp_margutil01.py`, plus
8 more from the follow-up EXP-FIRSTANCHOR-DIAG), same 2 pre-existing
failures. A 1-epoch ETTh1 H96 GPU smoke run caught and fixed two bugs
before the full run: (1) `HardAggregateMSE@10` was off by a factor of the
horizon `H` in the eval script's final division; (2) step-wise `regret` was
computed against the oracle's own pick (trivially always ~0) instead of the
model's own predicted pick under the true oracle prefix. Both fixed and
re-verified against known EXP-SEQDIAG01 figures (individual/set-oracle
MSE reproduced exactly) before the full run launched.

## Why this closes the "sparse vs. dense target" question, on the 3 cells run

`utility_spearman_mean` (0.38-0.66) confirms the utility function is
partially learnable from a frozen representation -- the sparsity of the
prior one-hot target (EXP-SEQFULL01/EXP-SEQDIAG01) was not the entire
story, since a dense target IS learnable to a moderate degree. But
`gap_recovery` and `HardAggregateMSE@10` show this learned utility does not
translate into a competitive free-running Top-K on any of the 3 cells --
Stage-2 is worse than B0 everywhere, by a similar or larger margin than the
one-hot CE arms EXP-SEQDIAG01 measured (see EXPERIMENT_LOG.md's full
EXP-MARGUTIL01 entry for the cross-experiment comparison). The follow-up
`EXP-FIRSTANCHOR-DIAG` (in `results/EXP-FIRSTANCHOR-DIAG/`) decomposes WHY:
the first (t=1) candidate choice dominates every downstream aggregate, and
the model's own t=1 choice is markedly worse than B0's.
