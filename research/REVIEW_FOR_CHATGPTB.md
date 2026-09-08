# REVIEW_FOR_CHATGPTB.md — Track B handoff

Companion to `research/REVIEW_FOR_CHATGPT.md` (Track A). Track B
(`EXP-CORRECTION-ORACLE-DIAG01`) ran independently of, and in parallel
with, Track A (`EXP-ENCODER-ANCHOR01` -> `EXP-ORACLE-CHOICE01` ->
`EXP-TEACHER-FORCING-DIAG01` -> `EXP-ONPOLICY-PREFIX01`) — separate
question, separate code, separate result directory
(`results/EXP-CORRECTION-ORACLE-DIAG01/`), touching none of Track A's
checkpoints or files. Split into its own file per the user's explicit
request (2026-09-08), so the two lines of investigation can be reviewed
independently. `research/EXPERIMENT_LOG.md` (the append-only factual
record) still contains BOTH tracks' entries in chronological order,
unsplit — this file only splits the reviewer-facing handoff document.

For the full Track A synthesis (which of representation / loss /
teacher-forcing was the largest bottleneck) and Track A's own experiment
sections, see `research/REVIEW_FOR_CHATGPT.md`. This file is self-contained
for Track B specifically.

---

# EXP-CORRECTION-ORACLE-DIAG01 (Track B) — Future Set Oracle vs. Correction Set Oracle (COMPLETE: diagnostic only, no training)

## Research Question

Stage-2's actual fusion is `Y_final = B_q + gamma*Y_ret` (residual
correction). Does retrieving by candidate forecast-ERROR similarity
(`r_i = Y_i - B_i`, a new "Correction Set Oracle") better match this
downstream semantics than the existing Future Set Oracle (retrieves by
realized-future similarity)? Ran in parallel with, independent of, Track A.

## Method

`select_greedy_weighted_set`/`dense_utility` (both existing, unmodified)
reused with residual value tensors instead of raw futures -- both
functions are agnostic to what the value tensor means. `r_i` computed in
each candidate's OWN frame (no query-offset transplant, unlike the
existing Future Oracle's convention) since a forecast error is already
scale-appropriate. Channel 0, self-only, same B0 checkpoint and reference
score throughout. 9/9 mandatory sanity checks passed, including a runtime
(not just unit-test) assertion of the `MSE(B_q+C,Y_q)==MSE(C,r_q)`
objective equivalence on every real batch.

**Scope limitation**: channel-0-only; absolute MSE values not comparable
to Track A's multi-channel numbers -- only the relative {B0, Future,
Correction} comparison is used.

## Results

| Metric | B0 | Future Oracle | Correction Oracle |
|---|---:|---:|---:|
| Final MSE (ch-0 scope) | 1.4496 | 0.2109 | 0.1818 |
| Gain vs B0 | -- | 1.2387 | 1.2678 |
| Mean rank | -- | 757.8 | 632.1 |
| NDCG@10 | -- | 0.524 | 0.513 |
| Margin (abs) | -- | 0.00838 | 0.00573 |

Correction Oracle wins modestly on downstream MSE, selects a genuinely
different candidate set (22-28% overlap with Future), but is consistently
LESS separated (higher near-tie fraction, smaller margin at every step)
and mixed-to-worse on NDCG/containment despite a better mean rank.
Epsilon-optimality is very low for both oracles at every threshold tested.

## Conclusion

**Outcome B-B.** Real but modest downstream headroom gain, mixed-to-worse
learnability -- not the clean "lower MSE + same-or-better geometry" pattern
Outcome B-A requires. Read as weak evidence that a VALUE-AWARE selector
(new observable candidate information, distinct from `EXP-STRONG-SCORER-
DIAG01`'s earlier scorer-CAPACITY-only negative result) might be worth
considering later -- not started automatically.

## Sanity checks passed

`pytest tests/`: 545 passed (9 new), same 2 pre-existing failures, no
regression.

## What this does NOT establish

- Whether the channel-0-only scope's headroom gain (2678 vs 2387) would
  hold proportionally under a full 7-channel evaluation.
- Whether a learned Correction Selector (`EXP-CORRECTION-SELECTOR01`)
  would recover the modest MSE gain in practice, given the worse
  learnability signals.
- Whether combining correction-aware retrieval with T1's on-policy prefix
  fix (Track A's own strongest result) would compound or conflict.

## Questions for ChatGPT

42. Correction Oracle's downstream MSE gain (1.2678 vs 1.2387) is modest
    relative to its consistently worse separability (near-tie fraction,
    margin). Is a ~2.3% relative headroom improvement, paired with worse
    learnability, sufficient grounds to run `EXP-CORRECTION-SELECTOR01`,
    or does the mixed evidence argue for testing a full 7-channel version
    of this SAME diagnostic first (removing the scope limitation) before
    committing to a new training run?
43. Given Track A's `EXP-ONPOLICY-PREFIX01` (T1) result (Stage-2 essentially
    at B0's own floor) and Track B's more modest, mixed Correction Oracle
    result, does this suggest the training-signal/prefix-distribution
    axis (Track A) is a substantially larger lever than the
    retrieval-TARGET-semantics axis (Track B) for this project's remaining
    gap to B0 -- and should future research effort be prioritized
    accordingly?

Please answer using the structure in `research/NEXT_EXPERIMENT.md`.
