# CURRENT_EXPERIMENT.md

Status: **All six approved experiments COMPLETE for their (reduced)
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

**4) EXP-TOPTAIL-RANK01 — loss-formulation comparison (R0/R1/R2) — COMPLETE
on ETTh1 H96, Weather H96 abandoned (2026-09-07).** No new architecture;
same frozen encoder/`SetConditioner`/`UtilityHead`, only the training loss
varies: R0=SmoothL1 (EXP-MARGUTIL01's own arm), R1=top-tail pairwise
ranking, R2=R1+R0 hybrid. On ETTh1 H96, R0→R1→R2 is a clean, consistent
improvement on Stage-2 (+0.157→+0.027→**+0.022** vs B0), `gap_recovery`,
HardAggregate, and t=1 top-tail ranking (Spearman-within-true-top-1%:
-0.558→+0.163→**+0.213**) — global teacher-forced Spearman goes the
opposite direction (+0.543→-0.162→-0.103), reproducing this project's
global-vs-top-tail dissociation finding a third time. **Weather H96: R1
was stopped after 2 epochs (loss plateaued near `ln(2)`, matching ETTh1's
own signature); R2's epoch-1 checkpoint was explicitly discarded/
unevaluated by user decision — no Weather H96 R1 or R2 result exists.**
Full record: `research/EXPERIMENT_LOG.md` → EXP-TOPTAIL-RANK01.

**5) EXP-ASYM-SCORER01 — R2+cosine vs. R2+asymmetric scorer — COMPLETE,
ETTh1 H96 only (2026-09-07).** No new training objective or architecture
change beyond the scorer: `AsymmetricUtilityHead` wraps the project's
existing `layers.retrieval_metric.RetrievalMetric(kind='asymmetric')`,
identity-initialised (`max_abs_score_deviation=0.0`, verified before
training). **Result: evidence AGAINST the scorer-capacity-bottleneck
hypothesis.** Every decision-relevant metric (Stage-2 +0.017 worse,
`gap_recovery` worse, HardAggregate worse, t=1/t=2 top-tail ranking worse)
is worse with the asymmetric scorer, DESPITE the training-time checkpoint-
selection proxy (`val_overlap@10`) preferring the asymmetric arm — itself
informative about that proxy's reliability for this model family. Not
tested/not concluded: whether different hyperparameters or regularisation
on the growing `cond(W_k)` would change this outcome. Full record:
`research/EXPERIMENT_LOG.md` → EXP-ASYM-SCORER01, full 7-question
evidence-based breakdown: `results/EXP-ASYM-SCORER01/REPORT.md`.

**6) EXP-STRONG-SCORER-DIAG01 — R2+cosine vs. R2+much-stronger nonlinear
residual pair scorer — COMPLETE, ETTh1 H96 only (2026-09-07).** Ceiling
probe: `C2 = R2 + StrongResidualPairScorer` (cosine + zero-init nonlinear
MLP residual over `[h,e,h⊙e,|h-e|]`, `trainable_params=378243`, ~87% of
which is the new scorer head) vs. `C0 = R2 + cosine` (existing
EXP-TOPTAIL-RANK01/R2 checkpoint, reused verbatim). Zero-init check
`max_abs_score_deviation=0.0`. Small-N positive control PASSED (scorer/loss/
training loop can fit a known synthetic utility landscape). Mid-experiment,
the original training path OOM'd on GPU 1 due to holding the full autograd
graph across all K steps x channels x candidate chunks before a single
backward; rewritten to a memory-safe streaming design (chunked, incremental
backward, exactly one `optimizer.step()` per batch, `FULL MEMORY -> DIRECT
TOP-K` semantics unchanged), verified mathematically equivalent to the
original via a dedicated gradient-equality unit test, before the real run
(peak_gpu_mem dropped from OOM to 504MiB). **Result: no clean fit to any of
the 3 pre-registered outcomes.** C2 shows small, real, consistent
improvements over C0 on t1 (test) and several t2 (test) top-tail ranking
metrics (t1 Spearman 0.213→0.225, t1 oracle rank median 99→97, t2 Spearman
0.140→0.182), but Stage-2 MSE, `gap_recovery`, HardAggregateMSE, and the
metrics closest to realized selector behavior (`selected true rank`,
`continuation_regret`) are flat-to-worse (Stage-2 0.39526→0.39708,
HardAgg 0.551→0.622). C2's train-split t1/t2 diagnostics are notably WORSE
than its own test-split diagnostics (opposite of classic overfitting),
consistent with `best_epoch=1` (checkpoint barely displaced from the C0
initialization before `val_overlap@10` starts declining) rather than with a
train-fits/test-fails story. `best_epoch=1`-then-decline on `val_overlap@10`
now replicates across every scorer/loss arm tried this session (R1, R2, C1,
C2). Full record: `research/EXPERIMENT_LOG.md` → EXP-STRONG-SCORER-DIAG01,
full report: `results/EXP-STRONG-SCORER-DIAG01/REPORT.md`.

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
reviewer. Per EXP-ASYM-SCORER01's own explicit instructions: do not
automatically run Weather/H720 for the asymmetric scorer, do not sweep
hyperparameters or add regularisation to `W_q`/`W_k`, do not try
Mahalanobis, and do not modify `SetConditioner` or unfreeze the encoder,
without the reviewer's/user's explicit approval first.

**Standing constraint, unchanged:** `FULL MEMORY -> DIRECT TOP-K` (D-0010).
Top-100/Top-M/shortlist/coarse-retrieval/reranker are not to be proposed as
the current or next research direction. Historical P100 usage remains
factual record in `EXPERIMENT_LOG.md`, not reopened.
