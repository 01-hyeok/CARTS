# EXP-MARGUTIL01 — interim results (2026-09-07, Weather H720 still running)

Status: **IN PROGRESS.** 3 of 4 cells complete; Cell 4 (Weather H720) is
training in the background on GPU 1. This file is an informal interim
snapshot for quick sharing -- once the run finishes, the canonical,
complete write-up goes into `research/EXPERIMENT_LOG.md` and
`research/REVIEW_FOR_CHATGPT.md` per this project's normal workflow. Treat
this file as provisional.

## What this experiment is

Successor to EXP-SEQFULL01 (one-hot next-candidate-ID imitation, closed by
D-0011) and EXP-SEQDIAG01 (established that a trainable encoder collapses
under this class of full-memory sequential objective, and that freezing the
encoder helps but is not sufficient). EXP-MARGUTIL01 replaces the one-hot CE
target with a **dense** per-candidate set-utility regression target,
`u_i^(t) = -A_weighted(S*_{t-1} + {i})`, for every valid candidate at every
teacher-forced step (oracle prefix `S*_{t-1}`), using B0's own **frozen**
encoder throughout (no trainable-encoder comparison in this experiment).

Question: does dense utility supervision let a held-out query build a
better Top-K set than exact-ID imitation did, and does that reach Stage-2
Final MSE? Three separate questions, never conflated:

- **A** — is the utility function itself learnable (teacher-forced Spearman/
  Pearson correlation, regret)?
- **B** — does that reach free-running Top-K selection (`gap_recovery`,
  `HardAggregateMSE@10`, duplicate/invalid rate)?
- **C** — does that Top-K improve Stage-2 Final MSE (the primary success
  criterion, not Stage-1 alone)?

4 cells: ETTh1 H96, Weather H96, ETTh1 H720, Weather H720 (seed=0, self-only,
top_k=10, full memory scored at every step, no shortlist anywhere).

## Results so far (3 of 4 cells)

| Cell | B0 unforced MSE | Dense Utility forced MSE | Delta vs B0 | gap_recovery | Utility Spearman | Utility Pearson | HardAgg (seq/b0) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ETTh1 H96 | 0.37312 | 0.52991 | **+0.157** | -1.83 | 0.543 | 0.424 | 1.657 / 0.407 |
| Weather H96 | 0.17553 | 0.33253 | **+0.157** | -1.74 | 0.658 | 0.503 | 4.465 / 0.234 |
| ETTh1 H720 | 0.46827 | 0.49269 | **+0.024** | -1.87 | 0.383 | 0.413 | 1.961 / 0.572 |
| Weather H720 | pending | pending | pending | pending | pending | pending | pending |

`duplicate_rate` / `invalid_rate` = 0.0 on all cells so far (no structural
cheating). Success/stop rule (pre-registered by the user): meaningful
Stage-2 improvement = ≤ B0 − 0.01; meaningful worse = ≥ B0 + 0.01;
noise-scale = |Δ| < 0.01. Every cell so far is **meaningfully worse**, well
past the noise threshold.

## Reading so far (3/3 cells, do not treat as final until Cell 4 lands)

- **A (is utility learnable?):** yes, partially — Spearman 0.38-0.66 across
  cells, moderate positive correlation, not zero. H96 correlates better
  than H720 on ETTh1 (0.543 vs 0.383) — longer horizon makes the dense
  target harder to predict from a frozen, past-only representation.
- **B (does it reach free-running selection?):** no — `gap_recovery` is
  strongly negative (-1.7 to -1.9) on every cell so far, and
  `HardAggregateMSE@10` for the sequential arm is 3-19x worse than B0's own
  unforced selection. The utility signal that does exist (per A) is not
  translating into good Top-K sets when the model has to choose from a
  full ~8k-36k candidate pool on its own.
- **C (Stage-2 improvement?):** no — every cell so far is worse than B0 by
  more than the 0.01 noise threshold (H96: +0.157 on both datasets; H720:
  +0.024 on ETTh1, smaller but still meaningfully worse).

This is currently tracking toward **STOP** per the pre-registered rule
(no cell has shown meaningful Stage-2 improvement), but the verdict is not
final until Weather H720 (Cell 4) completes — do not treat this as the
closing verdict.

## Not yet done

- Weather H720 (Cell 4) training + Stage-2 eval — running.
- Final `pytest tests/` regression check (post-run).
- Full comparison table across all 4 cells, final STRONG GO / MIXED / STOP
  verdict application.
- `research/EXPERIMENT_LOG.md` entry, `research/REVIEW_FOR_CHATGPT.md`
  refresh, `research/RESEARCH_DECISIONS.md` closing decision (if STOP).
- `results/EXP-MARGUTIL01/` full artifact set (metrics.csv, comparison.csv,
  notes.md, command.txt, env.txt, checkpoint fingerprints).

See `logs/exp_margutil01/run_margutil01_chain.sh` and
`logs/exp_margutil01/chain.log` for the exact commands and live log.
