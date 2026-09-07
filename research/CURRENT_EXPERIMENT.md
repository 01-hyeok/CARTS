# CURRENT_EXPERIMENT.md

Status: **All three approved experiments COMPLETE for their (reduced)
scope. No experiment currently running. Nothing running on any GPU on
behalf of this project.**

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

**3) EXP-CONTINUATION-DIAG — exhaustive t=2 continuation diagnostic —
COMPLETE for the same 3 cells (2026-09-07).** No new training; reuses
EXP-MARGUTIL01 checkpoints and EXP-FIRSTANCHOR-DIAG's own `run_arm`/
`a_weighted_prefix` verbatim (via import). Ran on GPU 1 (back to the
project's default single-GPU convention, per explicit user instruction,
after EXP-FIRSTANCHOR-DIAG's one-time GPU-0 exception). For each first
anchor (dense_first/b0_first/oracle_first) × cell, exhaustively evaluates
`A(S1+{i})` over EVERY valid remaining candidate (not sampled) to find the
true best t=2 continuation and compare it against the Dense selector's own
t=2 pick. **Central finding, the most consistent result across this whole
diagnostic campaign: extreme-top-tail utility ranking correlation
(Spearman within the true top 1%) is NEGATIVE in all 9 of 9
dataset×horizon×anchor combinations tested, while global ranking
correlation is mixed (positive in 2/9).** A good (oracle) first anchor does
NOT fix t=2 continuation failure — on 2 of 3 cells (ETTh1 H96, Weather H96)
it makes the failure rate WORSE than Dense's own (weaker) anchor, which was
not predicted going in. Weather H96's `oracle_first` shows a distinct
catastrophic-tail failure mode (aggregate can blow up by up to ~2 billion×)
tied to a cell-specific mechanism (mis-selected candidates receiving
disproportionate aggregation weight there specifically — not general
across the other 8 combos). Full 7-question evidence-based breakdown (most
defensible single-cause-vs-multi-cause judgement: multiple causes, with
set-conditioned top-tail ranking failure as the best-evidenced single
component): `results/EXP-CONTINUATION-DIAG/REPORT.md`. Full record:
`research/EXPERIMENT_LOG.md` → EXP-CONTINUATION-DIAG.

**No next experiment is approved.** Per this project's workflow, the next
step is an independent review (ChatGPT/Codex reads
`research/REVIEW_FOR_CHATGPT.md` and writes `research/NEXT_EXPERIMENT.md`);
the user then promotes an approved plan into this file. Do not start a new
experiment until that happens. In particular: do not automatically decide
that "B0-first" or "Oracle-first" is a validated next method — the mixed,
cell-dependent recovery pattern above is exactly the kind of result this
project's workflow reserves for the reviewer's interpretation, not Claude
Code's. Likewise, do not automatically implement a listwise/top-focused
training objective (or an aggregation-weight cap) as a "fix" for
EXP-CONTINUATION-DIAG's top-tail-ranking finding without the reviewer's/
user's explicit approval — the finding is well-evidenced, but which fix (if
any) to try next is exactly the kind of call this workflow reserves for the
reviewer.

**Standing constraint, unchanged:** `FULL MEMORY -> DIRECT TOP-K` (D-0010).
Top-100/Top-M/shortlist/coarse-retrieval/reranker are not to be proposed as
the current or next research direction. Historical P100 usage remains
factual record in `EXPERIMENT_LOG.md`, not reopened.
