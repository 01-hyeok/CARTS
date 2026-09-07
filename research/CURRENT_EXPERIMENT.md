# CURRENT_EXPERIMENT.md

Status: **EXP-MARGUTIL01 — Full-Memory Set-Conditioned Dense Marginal
Utility — IN PROGRESS (approved and commissioned directly by the user,
2026-09-06).** Implementation complete, sanity-verified end-to-end (1-epoch
ETTh1 H96 smoke run + Stage-2 eval reproduced known B0/individual/set-oracle
figures exactly), 11 new unit tests passing, full `pytest tests/` clean
(477 passed, same 2 pre-existing failures). The 4-cell run (ETTh1 H96,
Weather H96, ETTh1 H720, Weather H720; frozen B0 encoder, dense utility
regression, teacher-forced training + free-running Stage-2 eval) is running
in the background on GPU 1, sequentially, per the user's explicit
instruction to run all 4 cells regardless of intermediate scientific
outcome, stopping only on wiring/leakage/numerical failure. See
`logs/exp_margutil01/` for the chain script and live logs;
`results/EXP-MARGUTIL01/` for per-cell results as they land.

**Predecessor: EXP-SEQDIAG01 — COMPLETE.** Full record in
`research/EXPERIMENT_LOG.md` → EXP-SEQDIAG01, closing decision
`research/RESEARCH_DECISIONS.md` → D-0011 (exact one-hot sequential
imitation closed on both encoder variants; Dense Marginal Utility named as
the next candidate direction — now being executed as EXP-MARGUTIL01, per
explicit user approval, not a Claude Code research-direction decision).

# Research Question

EXP-SEQFULL01's one-hot next-candidate-ID cross-entropy target discards the
dense per-candidate set-utility information the Weighted Set Oracle
computes at every step (`A_weighted(S*_{t-1} + {i})` for every valid `i`,
not just the argmin). EXP-MARGUTIL01 asks: does training on this dense
target instead, `u_i^(t) = -A_weighted(S*_{t-1} + {i})`, let a held-out
query build a better Top-K set than exact-ID imitation did, and does that
reach Stage-2 Final MSE? Encoder is frozen (B0's own weights) throughout,
by explicit instruction, to isolate the objective change from the
encoder-collapse confound EXP-SEQDIAG01 already characterised — this
experiment does **not** re-open the trainable-encoder comparison.

- **H1 (sparse-target hypothesis):** set-level utility is predictable from
  frozen B0 representations; the prior objective's sparsity, not an
  information ceiling, caused EXP-SEQFULL01's failure. Dense supervision
  should raise utility rank correlation, lower regret, improve
  `gap_recovery`, and improve Stage-2 Final MSE.
- **H2 (information-ceiling hypothesis):** even with a dense target and a
  frozen, uncollapsed representation, held-out marginal set utility is not
  well predicted from past-only information. Weak correlation, high regret,
  `gap_recovery <= 0`, no Stage-2 improvement.

# Constraints (unchanged from EXP-SEQFULL01/EXP-SEQDIAG01)

`FULL MEMORY -> DIRECT TOP-K` (D-0010). Full memory scored at every one of
K=10 sequential steps; only already-selected and invalid candidates are
masked. No shortlist, Top-M/Top-100 pool, coarse retrieval, or reranker
anywhere. Candidate-dimension chunking (for GPU memory) is used and is
explicitly not a shortlist -- every valid candidate's utility is still
computed, just not in one tensor.

# Cells (seed=0, self-only, top_k=10, frozen B0 encoder throughout)

1. ETTh1 H96
2. Weather H96
3. ETTh1 H720
4. Weather H720

Executed in this order; all 4 run regardless of intermediate result, per
explicit user instruction (only wiring/leakage/numerical failure stops it
early).

# Success / Stop rule (pre-registered by the user)

Per-cell: meaningful improvement = Stage-2 MSE ≤ B0 − 0.01; noise-scale =
|Δ| < 0.01; meaningful worse = Stage-2 MSE ≥ B0 + 0.01.

- **STRONG GO:** ≥2/4 cells improve ≥0.01, no cell worsens ≥0.01, Stage-1
  utility correlation/regret/`gap_recovery` agree in direction.
- **MIXED:** some cells improve, dataset/horizon dependency is clear —
  report as-is, no automatic follow-up sweep.
- **STOP:** no cell shows meaningful Stage-2 improvement, or most/all cells
  have `gap_recovery ≤ 0` and Stage-2 does not beat B0.

Three questions kept separate in the final report, never conflated: (A) is
the utility function itself learnable (teacher-forced correlation/regret)?
(B) does that reach free-running Top-K selection (`gap_recovery`,
`HardAggregateMSE@10`, duplicate/invalid)? (C) does that Top-K improve
Stage-2 Final MSE?
