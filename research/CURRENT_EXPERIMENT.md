# CURRENT_EXPERIMENT.md

Status: **EXP-SEQDIAG01 — Cross-Dataset Sequential Collapse Diagnostic —
COMPLETE (2026-09-06).** All 4 arms finished (plus a HardAggregateMSE@10
companion metric and independent SHA256 encoder-freeze verification added
after the initial run); full factual record in `research/EXPERIMENT_LOG.md`
→ EXP-SEQDIAG01, reviewer handoff in `research/REVIEW_FOR_CHATGPT.md`,
closing decision in `research/RESEARCH_DECISIONS.md` → D-0011. Result:
partial support for H1 (frozen encoder improves `gap_recovery` AND
`HardAggregateMSE@10` consistently on both ETTh1 and Weather) but not full
support (every arm on both datasets remains net worse than B0 under both
metrics) — H2 remains necessary alongside H1. Per D-0011, exact one-hot
sequential imitation (both encoder variants) is now closed; Dense Marginal
Utility is named as a candidate next direction but **not implemented or
approved**.

**Note: "Frozen" in this experiment means the Stage-1 encoder's own weights
did not update during Stage-1 training — Stage-1 and Stage-2 were already
separated (via `set_forced_selection`, no forecasting-loss backprop into
Stage-1) in every arm, Trainable included. This is not a different
Stage-1/Stage-2 architecture and not end-to-end training.**

**No next experiment is approved.** Per this project's workflow, the next
step is an independent review (ChatGPT/Codex reads
`research/REVIEW_FOR_CHATGPT.md` and writes `research/NEXT_EXPERIMENT.md`);
the user then promotes an approved plan into this file. Do not start a new
experiment until that happens.

**Note on research direction (2026-09-06):** the coarse-retrieve-then-rerank
design that previously occupied this file (a fixed Top-100 shortlist between
full-memory retrieval and Stage-2) has been explicitly superseded by the
user's standing research constraint:

    FULL MEMORY -> DIRECT TOP-K

Top-100 / Top-M / shortlist / coarse retrieval / reranker / retrieve-then-rerank
architectures are not to be proposed as the current or next research
direction. This does not erase the historical fact that P100 supports were
used in earlier diagnostics (EXP-1/EXP-2's P100 arms, EXP-C01's coarse-vs-full
comparisons) — those remain in `EXPERIMENT_LOG.md` as factual history. It
means no *new* experiment in this project proposes reintroducing a shortlist.

---

# Research Question

EXP-SEQFULL01 (full-memory, teacher-forced, sequential imitation of the
Weighted Set Oracle's greedy construction) trained cleanly on ETTh1 H96
(nonzero gradient throughout, clean small-N memorization) but did not
generalize well to held-out queries, and the trained encoder's effective
rank collapsed from B0's ~20.8 to 5.43. Two explanations are entangled:

- **H1**: the sequential full-memory cross-entropy objective itself collapses
  the encoder representation, and that collapse is what prevents
  generalization.
- **H2**: even with the representation held stable (no collapse possible),
  past-only `X_q`/`X_i` information does not let this class of mechanism
  learn a discrete greedy set-construction rule that transfers from training
  queries to held-out ones.

EXP-SEQDIAG01 separates these by adding a **Frozen-B0 control arm** (B0's own
encoder weights, never updated; only the new SetConditioner trains) on both
ETTh1 and Weather H96, and comparing it against the existing Trainable arm.

# Hypothesis / Decision Rule

See `research/EXPERIMENT_LOG.md` → EXP-SEQDIAG01 for the full pre-registered
Case A / Case B / Case C decision rule (representation-collapse hypothesis
strengthened vs. exact-imitation-does-not-generalize hypothesis strengthened
vs. dataset-dependent learnability) and its outcome.

# Constraints (unchanged from EXP-SEQFULL01)

Full memory scored at every one of K=10 sequential steps; only
already-selected and invalid candidates are masked. No shortlist, no
Top-M/Top-100 pool, no reranker, at any point in this design.
