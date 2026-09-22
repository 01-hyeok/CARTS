# research/ INDEX

This repository's `research/` directory is split by experiment line so
reports don't read as one undifferentiated pile. Workflow files
(`CURRENT_EXPERIMENT.md`, `EXPERIMENT_LOG.md`, `REVIEW_FOR_CHATGPT*.md`,
`RESEARCH_CONTEXT.md`, `RESEARCH_DECISIONS.md`, `NEXT_EXPERIMENT.md`) stay at
the top level per `CLAUDE.md`'s Research Workspace convention -- everything
else is grouped below. Reorganized 2026-09-22 by explicit user request; no
report content was edited, only moved (`git mv`, history preserved).

## A — individual-encoder line
Query/candidate scored by each candidate's own individual future MSE
(no set-aware aggregation).
- `A-individual-encoder/ROUTER-ORACLE-HEADROOM01.md` -- Phase-1 comparison of
  retrieval-similarity experts (raw/delta/future-aligned), all individual-
  utility-based; Oracle Router vs validation-selected Best Static.

The earlier individual-choice pipeline (B0 -> R0/R1/R2 -> C1/C2 -> E1/E2 ->
D1 -> TF-DIAG -> T1 -> OPC1) is recorded in the top-level `EXPERIMENT_LOG.md`
/ `REVIEW_FOR_CHATGPT.md`, not as a standalone file here; its correctness
audit lives in `infra/TRACK_A_STRICT_AUDIT.md`.

## B — set-oracle line
Query scored against the Greedy Set Oracle's set-aware utility (the
Top-K SET that minimizes aggregate MSE), not per-candidate individual MSE.
- `B-set-oracle/EXP-SET-LOSS01.md`
- `B-set-oracle/EXP-SET-LOSS-STAGE2-RETRAIN01.md` (superseded by -CORRECTED)
- `B-set-oracle/EXP-SET-LOSS-STAGE2-RETRAIN01-CORRECTED.md`
- `B-set-oracle/EXP-ORACLE-WCE-CONTROL01.md`
- `B-set-oracle/TRACK-A-SET-LOSS-CONTROL02.md`
- `B-set-oracle/TRACK-A-SET-NORMREGRET-CONTROL01.md`
- `B-set-oracle/TRACK-A-CHOICECE-STAGE2-RETRAIN01.md` (superseded by -CORRECTED)
- `B-set-oracle/TRACK-A-CHOICECE-STAGE2-RETRAIN01-CORRECTED.md`
- `B-set-oracle/TRACK-A-TF-ORACLE-LEARNABILITY01.md` -- Individual vs
  Greedy-Set Oracle, Teacher-Forcing, Hard-Choice-CE comparison (reports
  both `individual_tf_cosine` and `set_tf_cosine` arms together; filed here
  because the report's central question is Set-Oracle learnability).

## C — horizon-block retrieval line
Does retrieval need to be specific to a future sub-horizon (block1
[0:96] / block2 [96:336] / block3 [336:720]) rather than one Top-K set for
the whole H=720 horizon?
- `C-horizon-block-retrieval/TRACK-A-HORIZON-RETRIEVAL-HEADROOM01.md` --
  pure Oracle diagnostic (no training): does block-specific Top-K beat one
  global Top-K? (Case-A finding: yes, ETTh1_720/Weather_720.)
- `C-horizon-block-retrieval/TRACK-A-HORIZON-RETRIEVAL-EXPERT01_INTERIM.md`
  -- first LEARNED horizon-aware retriever following that diagnostic.
  **Interim/in-progress**, not a closed result.

## D — patch-granularity retrieval line
Does varying the PAST input's patch size (not the future horizon) change
which candidates a retriever finds, and does combining multiple patch sizes
beat the best single size? New 2026-09-22, separate code/checkpoints/results
from C (`scripts/train_patch_retrieval_expert01.py`,
`scripts/calibrate_patch_retrieval_expert01_temperature.py`,
`results/TRACK-A-PATCH-RETRIEVAL-EXPERT01/`,
`checkpoints/track_a_patch_retrieval_expert01/`). Report will land at
`D-patch-retrieval/TRACK-A-PATCH-RETRIEVAL-EXPERT01.md` once Phase A
completes -- currently no report file yet (experiment in progress).

## infra — engineering / optimization / audit, not a standalone experiment
Speed, VRAM, or correctness work that supports the lines above but isn't
itself a research result.
- `infra/TRACK-A-WEATHER-OPT01.md` / `OPT02.md` / `OPT03.md` / `OPT04.md` --
  Greedy Set Oracle wall-clock/VRAM optimization for Weather (feeds B).
- `infra/TRACK-A-SOLAR-VRAM-OPT01.md` -- Solar (137-channel) VRAM levers
  (channelwise backward, etc.), used across B/C/D.
- `infra/AUDIT_ORACLE_RANK_GAIN01.md` -- correctness audit,
  EXP-ORACLE-RANK-GAIN01 vs EXP-ORACLE-SCRATCH01.
- `infra/RETRIEVAL_QUALITY_DIAG01.md` -- read-only diagnostic, why retrieval
  fails to help at ETTh1 H720.
- `infra/TRACK_A_STRICT_AUDIT.md` -- correctness audit of the original
  individual-choice pipeline (A).
- `infra/gpt.md` -- stale, superseded EXP-MARGUTIL01 interim note; kept for
  reference only, see the note at its top for the current write-up location.
