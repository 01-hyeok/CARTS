# RESEARCH_DECISIONS.md

Durable record of research decisions: what was decided, by whom, on what
evidence, and what it closes off. One entry per decision, newest at the bottom.
This file exists so that closed directions are not silently reopened.

Claude Code reads this file before implementing any experiment and does not
add entries on its own judgement — entries are added when the user makes or
confirms a decision.

---

## Template

```markdown
## D-XXXX — <short title>

- **Date:**
- **Decided by:** user | user + ChatGPT review
- **Decision:**
- **Evidence:** (log entry / result files / review)
- **Consequence:** what this opens or closes
- **Status:** active | superseded by D-YYYY
```

---

## D-0001 — Research workflow roles

- **Date:** 2026-09-02
- **Decided by:** user
- **Decision:** Claude Code is the implementation engineer; ChatGPT (web, manual
  copy-paste) is the research PI and independent reviewer; the user is the final
  decision maker. ChatGPT is not called via API. `AGENTS.md` assigns the same
  reviewer role to Codex; either reviewer uses the same artifacts.
- **Evidence:** this setup session; `CLAUDE.md`, `AGENTS.md`.
- **Consequence:** Claude does not choose research direction or interpret results
  as conclusions; it stops after each experiment and produces
  `research/REVIEW_FOR_CHATGPT.md`.
- **Status:** active

## D-0002 — Experiment drivers and result summaries become version-controlled

- **Date:** 2026-09-02
- **Decided by:** Claude Code, under blanket user authorization ("알아서 해줘")
- **Decision:** `.gitignore` changed so that (a) `scripts/**/*.sh` and
  `scripts/**/*.py` are tracked, except hidden `scripts/.*` one-off drivers, and
  (b) under `results/`, `.md` / `.csv` / `.json` / `.txt` / `.diff` files are
  tracked while everything heavier stays ignored.
- **Evidence:** `git check-ignore -v` confirmed `results/` and `scripts/*` were
  fully ignored, which would make every `EXPERIMENT_LOG.md` entry cite files
  absent from history. 231 driver scripts become trackable.
- **Consequence:** experiments logged under this workflow are reproducible from
  the repository alone. `logs/`, `metrics/`, `checkpoints/`, `runs/`,
  `predictions/`, `cache/` and all model/array binaries remain ignored.
  The scripts have **not** been committed — that is the user's call.
- **Status:** active

## D-0003 — RESULTS_SUMMARY.md is kept as an archive, not migrated

- **Date:** 2026-09-02
- **Decided by:** Claude Code, under blanket user authorization
- **Decision:** `RESULTS_SUMMARY.md` (Korean, results through 2026-08-04) stays
  in place as the archive of pre-workflow results and is cited by
  `RESEARCH_CONTEXT.md`. It is not retro-fitted into `EXPERIMENT_LOG.md`.
- **Evidence:** retro-fitting would require inventing configurations, seeds, and
  sanity-check records that the repository does not contain — i.e. fabricating
  research history, which `CLAUDE.md` forbids.
- **Consequence:** `EXPERIMENT_LOG.md` starts empty; findings from before the
  workflow live under *Established Experimental Findings* in
  `RESEARCH_CONTEXT.md` with their source named.
- **Status:** active

## D-0004 — Two pre-existing test failures are documented, not fixed

- **Date:** 2026-09-02
- **Decided by:** Claude Code, under blanket user authorization
- **Decision:** `pytest tests/` reports `396 passed, 2 failed`. Both failures
  reproduce in a clean worktree at HEAD `c306def`, so they predate the
  uncommitted work. They are recorded as the sanity-check baseline rather than
  repaired, because this session's mandate excluded modifying source code.
- **Evidence:** *Known Repository Issues* in `RESEARCH_CONTEXT.md`.
- **Consequence:** an experiment reporting exactly these two failures has not
  regressed; any additional failure has. Repairing them is a separate decision.
- **Status:** active

## D-0005 — Pre-fix ETTh1 Stage-2 sweep is invalidated

- **Date:** 2026-09-02
- **Decided by:** user
- **Decision:** A wiring bug prevented the configured scorer from reaching
  Stage-2's actual Top-K selection path, so some runs selected via cosine
  regardless of the trained scorer. The pre-fix ETTh1 Stage-2 score sweep must
  not be cited as the downstream effect of asymmetric/MLP selection. Weather was
  re-verified post-fix and remains valid.
- **Evidence:** user report, 2026-09-02; `RESEARCH_CONTEXT.md` →
  *Invalidated Results — Do Not Cite*.
- **Consequence:** ETTh1 Stage-2 requires a full rerun over
  {KL, WCE} x {Cosine, Asymmetric, MLP} x {96, 192, 336, 720} under identical
  Stage-2 conditions. Until then ETTh1 Stage-2 is presented as a pre-fix
  historical diagnostic and Weather is the only valid downstream evidence.
- **Status:** active

## D-0006 — Research question moved from recall to set-level utility

- **Date:** 2026-09-02
- **Decided by:** user
- **Decision:** The guiding question changes from "how do we raise Recall@10?" to
  "how do we define a retrieval utility useful for forecasting, and can it be
  learned from past-only information?"
- **Evidence:** Recall@10 vs Stage-2 MSE move in opposite directions on Weather
  H96 (B2); HardAggregateMSE@10 correlates far more strongly with Stage-2 than
  Recall@10 (B4); Set Oracle beats Individual Oracle on aggregate at every
  horizon while being worse individually (B10); direct Oracle imitation is at
  chance on validation for both targets (B15).
- **Consequence:** Recall@10 is retained as a diagnostic, not an objective.
  Pre-campaign hypotheses H-A / H-B / H-C are superseded by Q1 / Q2 / Q3.
  Next experiments test *learnability of the target*, not another loss term.
- **Status:** active

## D-0007 — Generic diversity is not adopted as a mechanism

- **Date:** 2026-09-02
- **Decided by:** user
- **Decision:** Despite the `I = A + V` decomposition showing that variance
  collapse can raise aggregate error, "increase diversity" is **not** adopted as
  the fix. The requirement is target-directed complementarity.
- **Evidence:** Good+Diverse Oracle lost to Individual Oracle at every horizon
  (-6.8 / -1.8 / -6.8 / -3.8%), and at H96 matched Set Oracle's variance
  (0.245785 vs 0.245243) while its aggregate MSE was far worse
  (0.139201 vs 0.096719).
- **Consequence:** any proposal justified only by "raises candidate variance" is
  rejected on existing evidence.
- **Status:** active

<!-- Append decisions below this line. -->

## D-0010 — Standing research constraint: full-memory direct Top-K, no shortlist/reranker direction

- **Date:** 2026-09-06
- **Decided by:** user, stated explicitly and repeatedly across the
  EXP-FRR01, EXP-SEQFULL01, and EXP-SEQDIAG01 experiment specifications.
- **Decision:** This project's retrieval research direction is, until further
  explicit user decision, `FULL MEMORY -> DIRECT TOP-K`. Top-100 / Top-M /
  shortlist / candidate pruning / coarse retrieval / reranker /
  retrieve-then-rerank architectures are not to be proposed as the current or
  next research direction, in experiment designs, in `CURRENT_EXPERIMENT.md`,
  or in `REVIEW_FOR_CHATGPT.md`'s forward-looking questions.
- **Evidence:** Direct, repeated user instruction (not inferred from
  experiment results). The coarse-retrieve-then-rerank design previously in
  `CURRENT_EXPERIMENT.md` (see D-0008's consequence) was superseded and
  removed on this date per this decision.
- **Consequence:** Historical factual records of P100/shortlist use in past
  diagnostics (EXP-1/EXP-2's P100 arms, EXP-C01's coarse-vs-full comparisons)
  remain in `EXPERIMENT_LOG.md` unchanged — this decision governs *future*
  proposals only, not the append-only historical record.
- **Status:** active

## D-0009 — EXP-FRR01 (residual-conditioned full-memory retrieval) closed: STOP, no 3-seed confirmation

- **Date:** 2026-09-05
- **Decided by:** pre-registered rule, specified by the user before the run
  ("최고 arm이 S0 대비 ≥0.01 개선일 때만 3-seed 확인"), mechanically triggered
  by the actual 1-seed results.
- **Decision:** None of R0/R1/R2/R12/R3 (residual-teacher WCE target, query
  base-forecast conditioning, candidate historical-residual conditioning, and
  their combination with an asymmetric dual encoder) improves Stage-2 Final
  MSE over B0 by the pre-registered margin. No 3-seed confirmation run
  triggered. No further arm/hyperparameter variation planned on this specific
  pilot's mechanisms without new evidence.
- **Evidence:** `EXPERIMENT_LOG.md` EXP-FRR01 — deltas vs B0 (0.37312):
  R0 +0.00085, R1 -0.00213, R2 +0.00711, R12 +0.00800, R3 +0.00641; none
  exceed the ±0.01 threshold. R2/R12/R3 raised Recall@10 by ~50% while making
  both `hard_aggregate_mse10` and Stage-2 Final MSE worse.
- **Consequence:** Reproduces EXP-C01/EXP-3's "Recall@10 does not predict
  Stage-2" finding via a second, structurally different mechanism (embedding
  conditioning vs. a soft aggregate loss), strengthening it as a property of
  this retrieval setup rather than an artifact of one loss family. The
  Individual→Set Oracle gap (EXP-1/EXP-2) remains real and unreached by every
  mechanism tried so far.
- **Status:** active

## D-0008 — soft_set_mse (EXP-3) closed: STOP, no further sweep

- **Date:** 2026-09-04
- **Decided by:** user, on representative-closure evidence
- **Decision:** Full-memory `soft_set_mse` as a Stage-1 training objective is
  closed. No further lambda/tau/support-penalty sweep on this direction.
- **Evidence:** `EXPERIMENT_LOG.md` EXP-3-CLOSURE — 0 of 4 representative
  cells (ETTh1 H96/H720, Weather H96/H720) improved Stage-2 Final MSE vs the
  WCE baseline; Weather's apparent Stage-1-level gain was confounded by N_eff
  explosion (+492%/+759%) and representation collapse (effective_rank
  -53%/-78%).
- **Consequence:** The Individual→Set Oracle gap (EXP-1/EXP-2) remains a real,
  measured upper bound and is NOT closed by this result — only this specific
  attempt to reach it (full-memory softmax training) failed. Candidate next
  direction (coarse-retrieve-then-rerank, confined to a fixed shortlist so
  diffusion cannot recur) is designed in `CURRENT_EXPERIMENT.md`, not started.
- **Status:** active

## D-0011 — EXP-SEQDIAG01 closed: exact one-hot sequential imitation (Trainable AND Frozen-B0) STOP; Dense Marginal Utility named as next candidate, not started

- **Date:** 2026-09-06
- **Decided by:** pre-registered Case A/B/C rule, specified by the user
  before this entry, mechanically applied to the actual 4-arm results.
- **Decision:** Discrete, one-hot, teacher-forced full-memory sequential
  cross-entropy imitation of the Weighted Set Oracle's greedy construction
  is closed as a direction — in **both** its Trainable and Frozen-B0
  encoder variants, on **both** ETTh1 and Weather H96. Freezing the encoder
  is not, by itself, sufficient to make this mechanism competitive with B0.
  No further hyperparameter/architecture variant of this exact-imitation
  objective is planned without new evidence.
- **Evidence:** `EXPERIMENT_LOG.md` EXP-SEQDIAG01 — `gap_recovery` is
  negative for all 4 arms (ETTh1 Trainable −0.404, ETTh1 Frozen −0.248,
  Weather Trainable −0.816, Weather Frozen −0.608); `HardAggregateMSE@10` is
  worse than B0's own unforced selection for all 4 arms (1.449x, 1.274x,
  18.38x, 3.93x respectively). Freezing the encoder improves both metrics on
  both datasets (partial H1 support, confirmed by independent SHA256
  encoder-weight verification that the frozen arms truly never update), but
  none of the four arms comes close to matching, let alone beating, B0.
  This is neither a clean Case A (H1 fully vindicates — frozen arm would
  need to beat B0 or reach `gap_recovery≈0`) nor a clean Case B (H1 plays no
  role at all — it clearly does, on both datasets and two metrics); the
  decision to close applies regardless of which sub-case this is, since the
  closing criterion (does any arm reach competitive Stage-2 performance) is
  met identically either way: no.
- **Consequence:** The Individual→Set-Oracle gap (EXP-1/EXP-2) remains real
  and unreached by five structurally different mechanisms now (EXP-3 soft
  relaxation, EXP-FRR01 embedding conditioning, EXP-SEQFULL01/EXP-SEQDIAG01
  discrete sequential imitation in trainable and frozen-encoder variants).
  Per the user's own pre-registered Case B consequence, the next candidate
  direction named (but **not implemented, not started, not approved for
  execution**) is a **Full-Memory Set-Conditioned Dense Marginal Utility**
  objective: `Delta_i(S) = A(S) − A(S ∪ {i})`, a dense per-candidate teacher
  signal rather than a single one-hot next-index target. This still
  requires the user's/reviewer's explicit approval and a `CURRENT_EXPERIMENT.md`
  design before any implementation begins. `FULL MEMORY -> DIRECT TOP-K`
  (D-0010) continues to govern: no shortlist/reranker reopening.
- **Status:** active

## D-0012 — EXP-MARGUTIL01 commissioned by explicit user approval

- **Date:** 2026-09-06
- **Decided by:** user (explicit, detailed 30-section execution instruction
  approving implementation and a full 4-cell run in one message)
- **Decision:** Dense Marginal Utility (named as D-0011's candidate next
  direction, not previously approved) is commissioned as EXP-MARGUTIL01: 4
  cells (ETTh1/Weather × H96/H720), frozen B0 encoder throughout (no
  trainable-encoder re-comparison), run to completion regardless of
  intermediate scientific result. This is not a Claude Code research
  judgement -- the direction was named by D-0011 and approved for execution
  by the user in this message.
- **Evidence:** user message, 2026-09-06, section 0 ("이 프롬프트는 사용자가
  직접 승인한 실행 지시다"), section 5 (4-cell spec), section 22 ("과학적
  negative result 때문에 4-cell을 조기 종료하지 마").
- **Consequence:** `research/CURRENT_EXPERIMENT.md` replaced with
  EXP-MARGUTIL01's spec (EXP-SEQDIAG01's completed status preserved in
  `EXPERIMENT_LOG.md`/`RESEARCH_DECISIONS.md`, not overwritten). New code:
  `models/DenseUtilityRetriever.py`, `utils/dense_utility.py`,
  `scripts/train_margutil01.py`, `scripts/eval_margutil01_stage2.py`,
  `tests/test_exp_margutil01.py`. `FULL MEMORY -> DIRECT TOP-K` (D-0010)
  continues to govern; no shortlist/reranker reopening.
- **Status:** active

## D-0013 — EXP-MARGUTIL01 Weather H720 cancelled; scope reduced to 3 cells

- **Date:** 2026-09-07
- **Decided by:** user (explicit, direct instruction mid-session: "0번
  스탑하자 1번에 하고, weather 720 실험은 없애버리자")
- **Decision:** EXP-MARGUTIL01's Weather H720 cell is cancelled. It had run
  on GPU 1 for ~2 hours without completing a single training epoch (heaviest
  cell: 21 channels, 35448 candidates, 720-step horizon). The process was
  killed and its (empty — never-saved) checkpoint directory removed.
  EXP-MARGUTIL01's approved scope is reduced from 4 to 3 cells (ETTh1 H96,
  Weather H96, ETTh1 H720). No number is reported, estimated, or
  extrapolated for Weather H720.
- **Evidence:** user message, 2026-09-07; process kill and empty-directory
  removal confirmed directly (`ps`, `find` before/after).
- **Consequence:** `research/EXPERIMENT_LOG.md`'s EXP-MARGUTIL01 entry
  reports 3 cells as complete and Weather H720 as cancelled, not as a 4th
  data point. The pre-registered 4-cell STRONG GO/MIXED/STOP decision rule
  (≥2/4 cells improve ≥0.01 Stage-2 MSE, none worsen ≥0.01) cannot be
  mechanically applied to a 3-cell result without the reviewer's/user's
  explicit re-scoping acknowledgement; Claude Code does not unilaterally
  reinterpret the rule for 3 cells. `EXP-FIRSTANCHOR-DIAG` (the causal
  follow-up diagnostic, also user-approved) likewise only covers 3 cells
  for the same reason — it reuses EXP-MARGUTIL01 checkpoints and none
  exists for Weather H720.
- **Status:** active

## D-0014 — EXP-FIRSTANCHOR-DIAG authorised to run in parallel on GPU 0

- **Date:** 2026-09-07
- **Decided by:** user (explicit instruction, overriding this diagnostic's
  own original spec which had said GPU 1 only / wait if busy / no other
  GPU): "지금 실험은 weather 720 실험이 진행중일텐데, 내가 제안한 실험을
  병렬로 실행해줘." Later corrected/clarified by the user (see D-0013 — the
  user subsequently decided to cancel Weather H720 rather than keep both
  running); GPU 0 usage itself is recorded here as a one-time, explicitly
  authorised exception to the project's default single-GPU convention, not
  a standing policy change.
- **Decision:** `scripts/eval_firstanchor_diag.py` ran on GPU 0 while
  EXP-MARGUTIL01's Weather H720 ran on GPU 1, per direct user instruction,
  rather than waiting for GPU 1 to free up as the diagnostic's own written
  spec had originally required.
- **Evidence:** user message, 2026-09-07.
- **Consequence:** None ongoing — the GPU-0 run completed (3/3 available
  cells) before the user's subsequent D-0013 decision to stop GPU 1's
  Weather H720. No standing precedent for using GPUs other than 1 without
  equally explicit future authorisation.
- **Status:** active (historical record of a one-time exception, not a
  standing rule)


