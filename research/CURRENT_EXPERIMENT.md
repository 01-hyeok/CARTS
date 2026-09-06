# CURRENT_EXPERIMENT.md

Status: **EXP-SEQDIAG01 — Cross-Dataset Sequential Collapse Diagnostic.**
Approved and commissioned directly by the user (2026-09-06), executed in the
same session. See `research/EXPERIMENT_LOG.md` → EXP-SEQDIAG01 for the
factual result once complete.

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
