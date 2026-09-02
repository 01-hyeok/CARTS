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
