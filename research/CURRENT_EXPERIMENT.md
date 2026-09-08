# CURRENT_EXPERIMENT.md

Status: **Track A (EXP-ENCODER-ANCHOR01 -> EXP-ORACLE-CHOICE01 ->
EXP-TEACHER-FORCING-DIAG01 -> EXP-ONPOLICY-PREFIX01) is now FULLY
COMPLETE (2026-09-08). Track B (EXP-CORRECTION-ORACLE-DIAG01, diagnostic
only, run in parallel, touching none of Track A's code/checkpoints/
results) is also COMPLETE.** Both sequences were pre-approved by the user
to run in full regardless of each step's individual result (only a
genuine technical/validity problem -- sanity failure, NaN, wrong candidate
universe -- was grounds to stop early; none occurred). **STOP. No further
experiment without new user approval.** Eleven experiments COMPLETE this
session (below); results ready for independent review.

## EXP-ENCODER-ANCHOR01 — COMPLETE (2026-09-08)

**Result:** Outcome B (collapse prevented, Stage-2 does not improve), with
an internal metric disagreement reported as-is. `embedding_effective_rank`
stayed 15.70-16.56 across all 6 epochs trained (vs B0's 17.48, vs E1's own
collapse to 2.96-1.49) -- the anchor clearly prevented collapse. t1/t2
top-tail ranking metrics became the BEST of all three arms (C0/E1/E2) on
test split (t1 Spearman 0.244 vs C0's 0.213/E1's 0.146; t1 oracle rank
median 86 vs C0's 99/E1's 210; t2 continuation_regret 0.318, better than
both C0's 0.350 and E1's 0.399). However, by the pre-registered PRIMARY
metric, Stage-2 MSE = 0.40418, still worse than C0's 0.39526 (though
slightly better than E1's 0.40668) -- and HardAggregateMSE = 0.6768 is the
WORST of all three arms (C0=0.5514, E1=0.5466). This three-way
Stage-2/HardAggregate-vs-t1/t2 disagreement is reported as an open,
unresolved finding, not forced into a single narrative. Conclusion: the
STRONG version of "collapse fully explains E1's Stage-2 failure" is
refuted (preventing collapse only closed a small fraction of the gap and
made HardAggregate worse); a WEAK version ("collapse was a real but
partial contributing factor, alongside something else that shows up as a
per-step-ranking vs. free-running-aggregate disconnect") is left open and
plausibly links to the still-pending teacher-forcing question. Per the
pre-registered stopping rule, this result does NOT justify a larger
anchor-strength sweep or a different anchor form -- none was started. Full
report: `results/EXP-ENCODER-ANCHOR01/REPORT.md`, full record:
`research/EXPERIMENT_LOG.md` → EXP-ENCODER-ANCHOR01.

## EXP-ENCODER-ANCHOR01 — original approved plan (2026-09-08, for reference)

**Research question:** in `EXP-ENCODER-UNFREEZE01`, letting the encoder
train under R2's objective caused rapid representation collapse
(`embedding_effective_rank` 17.48→2.96→1.49, `pairwise_cosine_mean`
→0.99+, Stage-2 MSE 0.39526→0.40668, worse than frozen C0). Does adding a
B0-anchor regularizer to the SAME trainable-encoder setup (E1) prevent the
collapse, and if so, does that let encoder adaptation actually improve
Stage-2 over C0 — or was collapse never the real bottleneck?

**Arms:** C0 (existing, reused, EXP-TOPTAIL-RANK01/R2) and E1 (existing,
reused, EXP-ENCODER-UNFREEZE01) are NOT retrained. Only **E2** is new: R2
hybrid loss + cosine UtilityHead + trainable encoder (init from the same B0
checkpoint) + `L_anchor = 1 - cos(normalize(f_theta(x)), normalize(f_{theta_0}(x)))`
added to the loss with a single fixed `lambda_anchor` (chosen via a small
fixed-batch gradient-norm-ratio diagnostic, not a sweep, not by looking at
test Stage-2). B0 reference encoder stays frozen/eval/no_grad throughout.
Every other hyperparameter/architecture/protocol identical to E1 (same
optimizer/LR, epochs/patience, checkpoint-selection criterion
`val_overlap@10`, full-memory memory-safe streaming design, Stage-2
architecture/gate/fusion unchanged). New script:
`scripts/train_encoder_anchor01.py` (does not modify
`scripts/train_encoder_unfreeze01.py`, preserving E1's reproducibility).
Explicitly prohibited: VICReg/covariance/variance regularizers, EMA,
stop-grad, encoder LR sweep/multiplier, scorer changes, Stage-2 fusion
changes, Top-M/shortlist/reranker, other datasets/horizons/seeds. ETTh1
H96 seed 0 only. Primary success metric: Stage-2 Final MSE vs. C0
(0.39526) and E1 (0.40668), in that priority order ahead of HardAggregate/
gap_recovery, t1/t2, and representation diagnostics. Outcome categories
A (collapse prevented + Stage-2 improves) / B (collapse prevented, Stage-2
doesn't improve) / C (collapse persists) / D (over-constrained, ~frozen) —
reported as-is, no forced binary success/failure. No next experiment to be
auto-started regardless of outcome.

Result directory: `results/EXP-ENCODER-ANCHOR01/`.

## Pre-approved next sequence (2026-09-08, explicit user approval)

User has explicitly pre-approved, in this order, regardless of each
individual step's result: (2) EXP-ORACLE-CHOICE01 (D1: R2's SmoothL1+
pairwise surrogate replaced by a direct masked full-memory Oracle-Choice
Cross-Entropy loss, frozen encoder, no other change) → (3)
EXP-TEACHER-FORCING-DIAG01 (diagnostic only, NO new training, reuses the
existing C0/R2 checkpoint verbatim: compares Oracle-prefix vs. free-running
evaluation of the SAME checkpoint to check whether teacher forcing itself
is a plausible bottleneck before committing to on-policy retraining) → (4)
EXP-ONPOLICY-PREFIX01 (T1: R2 loss unchanged, only the training-time prefix
source changes from oracle to the model's own on-policy picks, frozen
encoder). After (4), STOP -- no further experiment (combinations, sweeps,
scheduled sampling, DAgger, etc.) without new explicit user approval. A
merely negative/mixed RESULT at any step is NOT grounds to skip the next
pre-approved step; only a genuine technical/validity failure (sanity check
failure, NaN, wrong candidate universe) is.

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

**7) EXP-ENCODER-UNFREEZE01 — R2+cosine+frozen encoder vs. R2+cosine+
trainable encoder — COMPLETE, ETTh1 H96 only (2026-09-08).** Tests whether
the frozen B0 encoder representation, not the scorer/loss, is the
bottleneck: `E1 = R2 + cosine + TRAINABLE encoder` (initialised from the
same B0 checkpoint, encoder architecture unchanged) vs. `C0 = R2 + cosine +
FROZEN encoder` (existing EXP-TOPTAIL-RANK01/R2 checkpoint, reused
verbatim). Encoder gradient flow (query- and candidate-side) and a
"no stale candidate bank" invariant were verified by 4 mandatory sanity
tests before the GPU run; a memory-safe streaming re-encoding design
(generalising EXP-STRONG-SCORER-DIAG01's OOM fix so every encoder-touching
tensor, not just the scorer head, uses a fresh per-call forward) trained
to completion with `peak_gpu_mem=515MiB`, no OOM. **Result: severe,
monotonically-deepening representation collapse** — `embedding_effective_rank`
falls from B0's 17.48 to 2.96 after 1 epoch (the checkpoint actually
selected, `best_epoch=1` matching every other arm this session) to 1.49 by
epoch 6, `pairwise_cosine_mean` rising to 0.99+ (candidates nearly
indistinguishable), replicating EXP-SEQDIAG01's earlier collapse finding
under a different (one-hot) objective, now also under R2's more careful
hybrid loss. Stage-2 MSE and `gap_recovery` are both worse than C0
(0.39526→0.40668, gap_recovery -0.263→-0.370); t1 top-tail ranking is
uniformly worse; t2 continuation shows a mixed picture (some rank-median
metrics better, `continuation_regret` worse). Verdict: **Outcome D**
(representation collapse) — not Outcome A/B/C, and per the pre-registered
stopping rule this does NOT justify an encoder depth/capacity follow-up
(a bigger encoder trained the same uncontrolled way would be expected to
collapse the same way, plausibly faster); no collapse-prevention
regularisation was added. Full record: `research/EXPERIMENT_LOG.md` →
EXP-ENCODER-UNFREEZE01, full report: `results/EXP-ENCODER-UNFREEZE01/REPORT.md`.

**8) EXP-ENCODER-ANCHOR01 — R2+cosine+trainable encoder+B0-anchor —
COMPLETE, ETTh1 H96 only (2026-09-08).** See the full summary at the top
of this file. Outcome B (collapse prevented, Stage-2 still worse than C0);
HardAggregate worst of all three arms while t1/t2 top-tail ranking best of
all three arms — an unresolved metric disagreement, reported as-is. Full
report: `results/EXP-ENCODER-ANCHOR01/REPORT.md`.

**9) EXP-ORACLE-CHOICE01 — R2 (SmoothL1+pairwise) vs. Oracle-Choice CE —
COMPLETE, ETTh1 H96 only (2026-09-08).** D1 replaces R2's SmoothL1+
pairwise surrogate entirely with a single masked full-memory Oracle-Choice
Cross-Entropy loss (predict the Set Oracle's actual argmax next choice
directly), frozen B0 encoder unchanged, tau=0.1 (base checkpoint's own
`tau_topk`, no sweep). A mandatory pre-training diagnostic found the
Oracle's true top-1-vs-top-2 margin collapses sharply as the selected set
grows (`near_tie_frac` 8.5%@t=1 -> 99.0%@t=10) -- flagged as a limitation
on one-hot CE, not acted on. Consistent with this, D1's exact teacher-forced
top-1 accuracy stayed low throughout (mean 0.36% across K=10 steps).
**Despite that, EVERY downstream/decision-relevant metric improved over
C0** -- t1 Spearman 0.213->0.283, t1 oracle rank median 99->47, t2 selected
true rank median 1934->88, t2 continuation_regret 0.350->0.196,
HardAggregate 0.5514->0.5480, gap_recovery -0.263->-0.232, and **Stage-2
MSE 0.39526->0.38916 -- the FIRST arm in this entire session to beat C0 on
Stage-2**. `val_overlap@10` improved every epoch through all 10 configured
epochs (never early-stopped), unlike every other arm this session (noted
as an observation, not over-interpreted -- no controlled comparison of
training budget was run). Verdict: **Outcome L-A** -- evidence that R2's
SmoothL1+pairwise surrogate was insufficiently aligned with the Set
Oracle's actual greedy decision, and a more direct choice-prediction
objective recovers real Stage-2 value despite its own near-tie-driven
exact-accuracy limitation. Full report: `results/EXP-ORACLE-CHOICE01/REPORT.md`.

**10) EXP-TEACHER-FORCING-DIAG01 — Oracle-prefix vs. free-running
evaluation of the SAME C0/R2 checkpoint — COMPLETE, diagnostic only, no
training (2026-09-08).** Mixed evidence: Case C (NDCG-high but
exact-choice-rank-poor, reproducing EXP-FIRSTANCHOR-DIAG's finding at
every step) strongly present; Case A partial (oracle-prefix itself is not
"good" in absolute terms -- Top-1 acc ~0.25%, rank balloons past 1000
mid-sequence even given the correct prefix); Case B also has real support
(oracle-prefix beats free-running on every comparable metric within this
diagnostic, free-running diverges from the oracle's trajectory almost
immediately -- 99.5% divergence rate, 2.7% prefix overlap). Net reading:
teacher-forcing mismatch is a real contributor alongside a
surrogate/decision-resolution weakness that independently corroborates
EXP-ORACLE-CHOICE01. Full report: `results/EXP-TEACHER-FORCING-DIAG01/REPORT.md`.

**11) EXP-ONPOLICY-PREFIX01 — R2 loss unchanged, Oracle prefix vs.
on-policy prefix — COMPLETE, ETTh1 H96 only (2026-09-08).** T1: same R2
hybrid loss, frozen encoder, cosine UtilityHead as C0 -- only the
training-time prefix source changes from the Oracle's sequence to the
model's own on-policy picks. Training-time diagnostics (prefix overlap
with the Oracle staying near-zero throughout, step-1 regret rising
slightly) looked concerning in isolation but were NOT predictive of the
downstream result. **Result: by far the best Stage-2/HardAggregate result
of the entire session.** Stage-2 MSE 0.39526 (C0) -> **0.37455** (T1),
within 0.00143 of B0's own 0.37312; gap_recovery -0.263 -> **-0.010**
(essentially fully recovered); HardAggregate 0.551 -> **0.410**. t1/t2
Oracle-agreement metrics were mixed-to-worse (t1 Spearman/rank worse, t2
hurt_frac worse) but t2 selected-true-rank/Spearman/regret were much
better. Verdict: **Outcome T-A** -- per the pre-registered priority order
(Stage-2 > HardAggregate/gap_recovery > realized regret > Oracle
choice/rank/top-tail), the three highest-priority criteria are
unambiguous large wins; only the lowest-priority criterion is worse.
Train-time Oracle-prefix / inference-time model-prefix state-distribution
mismatch was a real, and by magnitude the LARGEST-YET-FOUND, contributing
factor to this project's set-aware retrieval underperforming B0. Full
report: `results/EXP-ONPOLICY-PREFIX01/REPORT.md`. **This completes
Track A.**

**12) EXP-CORRECTION-ORACLE-DIAG01 (Track B) — Future Set Oracle vs.
Correction Set Oracle — COMPLETE, diagnostic only, no training
(2026-09-08).** Ran in parallel with Track A, touching none of its
code/checkpoints/results. Question: should retrieval target candidates
whose realized future resembles the query's future (existing Future Set
Oracle), or candidates whose OWN frozen-B0 forecast error resembles the
query's own error (new Correction Set Oracle), given Stage-2's actual
residual-fusion semantics? **Result (channel-0 diagnostic scope, not
comparable in absolute terms to Track A's multi-channel Stage-2 numbers):**
Correction Oracle achieves modestly lower downstream MSE than Future
Oracle (gain-vs-B0 1.2678 vs. 1.2387), selects a genuinely different
candidate set (22-28% overlap), but shows WORSE learnability signals
(higher near-tie fraction, smaller margin, mostly worse NDCG/containment)
despite a somewhat better mean rank. Verdict: **Outcome B-B** -- real but
modest downstream headroom gain, mixed-to-worse learnability, not strong
enough evidence on its own to justify `EXP-CORRECTION-SELECTOR01`
automatically. Full report: `results/EXP-CORRECTION-ORACLE-DIAG01/REPORT.md`.

Per this project's workflow, the next
step is an independent review (ChatGPT/Codex reads
`research/REVIEW_FOR_CHATGPT.md` for Track A and
`research/REVIEW_FOR_CHATGPTB.md` for Track B, split into separate files
per the user's explicit request so the two independent lines of
investigation can be reviewed separately, and writes
`research/NEXT_EXPERIMENT.md`); the user then promotes an approved plan
into this file. Do not start a new
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
