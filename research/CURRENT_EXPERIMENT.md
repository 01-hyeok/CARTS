# CURRENT_EXPERIMENT.md

Status: **Both approved experiments COMPLETE for their (reduced) scope.
No experiment currently running. Nothing running on any GPU on behalf of
this project.**

**1) EXP-MARGUTIL01 — Full-Memory Set-Conditioned Dense Marginal Utility —
COMPLETE for 3 of 4 cells (2026-09-07).** ETTh1 H96, Weather H96, ETTh1 H720
finished (Stage-1 + Stage-2, results in `results/EXP-MARGUTIL01/`).
**Weather H720 was cancelled by explicit user decision** (D-0013 in
`research/RESEARCH_DECISIONS.md`) after ~2h without completing a single
epoch; no checkpoint exists for it and no number is reported for it.
Result on the 3 completed cells: dense per-candidate utility is partially
learnable (Spearman 0.38-0.66) but does not translate into a competitive
free-running Top-K (`gap_recovery` strongly negative on all 3 cells) or
Stage-2 improvement (all 3 cells worse than B0 by more than the 0.01 noise
threshold). Full record: `research/EXPERIMENT_LOG.md` → EXP-MARGUTIL01.

**2) EXP-FIRSTANCHOR-DIAG — causal decomposition of EXP-MARGUTIL01's t=1
choice — COMPLETE for the same 3 cells (2026-09-07).** No new training;
reuses EXP-MARGUTIL01 checkpoints, only varying which rule picks the first
(t=1) candidate. Ran in parallel with EXP-MARGUTIL01's (then still active)
Weather H720 on a separate GPU, per explicit user instruction (D-0014) —
that authorisation was a one-time exception, not a standing policy; default
back to GPU 1 only unless the user says otherwise again. Result: t=1
dominates every cell's final aggregate (consistent finding across all 3
cells — the stepwise trajectory barely moves after t=1), but which t=1
rule actually helps is cell-dependent, not uniform: the deployable B0-first
hybrid recovers 60-76% of the gap on both H96 cells but *worsens* ETTh1
H720; the diagnostic-only Oracle-first beats B0 outright on both ETTh1
cells but *worsens* Weather H96. No clean single verdict. Full record:
`research/EXPERIMENT_LOG.md` → EXP-FIRSTANCHOR-DIAG.

**No next experiment is approved.** Per this project's workflow, the next
step is an independent review (ChatGPT/Codex reads
`research/REVIEW_FOR_CHATGPT.md` and writes `research/NEXT_EXPERIMENT.md`);
the user then promotes an approved plan into this file. Do not start a new
experiment until that happens. In particular: do not automatically decide
that "B0-first" or "Oracle-first" is a validated next method — the mixed,
cell-dependent recovery pattern above is exactly the kind of result this
project's workflow reserves for the reviewer's interpretation, not Claude
Code's.

**Standing constraint, unchanged:** `FULL MEMORY -> DIRECT TOP-K` (D-0010).
Top-100/Top-M/shortlist/coarse-retrieval/reranker are not to be proposed as
the current or next research direction. Historical P100 usage remains
factual record in `EXPERIMENT_LOG.md`, not reopened.
