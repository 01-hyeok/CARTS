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

---

# EXP-CORRECTION-ORACLE-DIAG02 (Track B1) — multivariate Correction Set Oracle diagnostic (COMPLETE: diagnostic only, no training)

## Research Question

Does the channel-0-only Correction Set Oracle advantage found in
`EXP-CORRECTION-ORACLE-DIAG01` reproduce across the full ETTh1
multivariate setting (all 7 channels)?

## Method

`diag_correction_oracle01.evaluate()` gained an optional `channel`
parameter (default `channels[0]`, preserving the original single-channel
behaviour exactly); new driver `scripts/diag_correction_oracle02.py` calls
it once per channel and aggregates. Reuses the original's 9/9 sanity
checks unmodified -- the underlying math does not change per channel.

## Results

| channel | B0 | Future Oracle | Correction Oracle | Correction wins? |
|---|---:|---:|---:|:-:|
| 0 | 1.4496 | 0.2109 | **0.1818** | ✓ |
| 1 | 0.3245 | 0.0778 | **0.0770** | ✓ |
| 2 | 1.5371 | 0.1969 | **0.1748** | ✓ |
| 3 | 0.2701 | **0.0657** | 0.0665 | ✗ |
| 4 | 0.7304 | 0.1295 | **0.1256** | ✓ |
| 5 | 0.1585 | **0.0372** | 0.0386 | ✗ |
| 6 | 0.0570 | **0.00324** | 0.00324 | ✗ (tied) |

Aggregate: gain vs B0 (mean) Future=0.5437, Correction=**0.5514**; mean
rank Future=**1506.9** vs Correction=2312.9; NDCG@10 Future=0.656 vs
Correction=**0.699** (reversed from the channel-0-only result); margin abs
Future=**0.00404** vs Correction=0.00279.

## Conclusion

Correction Oracle wins on the cross-channel mean and on 4/7 individual
channels, losing on the 3 easiest-to-forecast channels (lowest B0 MSE).
Learnability signals are genuinely mixed: NDCG now favors Correction on
average (opposite direction from the channel-0 diagnostic), while mean
rank/margin still favor Future. Nuances, without overturning,
`EXP-CORRECTION-ORACLE-DIAG01`'s Outcome B-B.

## Sanity checks passed

`pytest tests/`: 563 passed (0 new, reuses existing checks), same 2
pre-existing failures, no regression.

---

# EXP-CORRECTION-STAGE2-SEMANTICS01 (Track B2) — correction-aligned Stage-2 fusion, fixed reference retrieval (COMPLETE: ETTh1 H96 only)

## Research Question

Given the SAME reference retrieval (Top-K + alpha from B0's own existing
production score -- no new selector trained), does fusing the historical
CORRECTION aggregate (`C_ret = sum(alpha_i * r_i)`, `r_i = Y_i - B_i`)
beat the existing FUTURE aggregate fusion Stage-2 already uses?

## Method

New script `scripts/train_correction_stage2_semantics01.py`. No new
retrieval selector; B0/base_forecast/retrieval score frozen throughout.
`layers/retrieval_gate.py::RetrievalGate` (existing, unmodified) reused
for C0 (`fixed_lambda=1.0`, no training) and C1 (learned per channel on
TRAIN, selected on VAL, evaluated once on TEST). `B_q`/`C_ret`/`Y_q`
precomputed once per split for fast gate training. 7/7 sanity checks
passed, including a positive-control test on synthetic data confirming the
training loop recovers a known gamma. A bug (F0 MSE computed from the
unsliced multivariate output) was caught and fixed before trusting results.

## Results

| Arm | Aggregate MSE (7-channel mean) |
|---|---:|
| F0 (existing) | **0.37312** |
| C1 (learned gate) | 0.38337 |
| C0 (fixed γ=1) | 0.39056 |

F0 beats both C0 and C1 on 6/7 channels and on the aggregate. C1
consistently beats C0 on every channel (adaptive γ helps over a naive full
correction), and gamma values (0.263-0.983) show genuine per-channel
variation, not collapse -- the gate training itself worked correctly. Only
channel 6 (easiest to forecast) shows C1 beating F0.

## Conclusion

**Outcome B2-D** on the aggregate and 6/7 channels. Per the pre-registered
interpretation rule, this does NOT reject the Correction hypothesis
overall: Track B1 found real oracle-level headroom for the Correction
VALUE using the TRUE query future. The most likely reconciliation is that
reusing the EXISTING future-oriented Top-K (rather than a retrieval
process aligned with the correction objective) hands the correction
fusion the wrong candidates -- Track B1 already showed Future and
Correction Oracles select substantially different sets (22-28% overlap).

## Sanity checks passed

`pytest tests/`: 563 passed (7 new), same 2 pre-existing failures, no
regression.

## What Track B1 + B2 together do NOT establish

- Whether a retrieval process specifically optimized for the correction
  objective (not yet tested, explicitly gated by the STOP rule) would let
  Stage-2 realize B1's oracle-level headroom.
- Whether channel 6's exception (C1 beats F0) generalizes to other
  low-B0-MSE regimes or datasets.

## Questions for ChatGPT

46. Track B1 shows the Correction Oracle has real headroom (using the true
    query future) on 4/7 channels; Track B2 shows that headroom is NOT
    realized when reusing the existing future-oriented Top-K. Does this
    make `EXP-CORRECTION-SELECTOR01` (a NEW selector trained toward the
    Correction objective, not yet approved) a more promising next step
    than further Stage-2-fusion-only experiments, since B2's own negative
    result may be attributable to retrieval-target mismatch rather than a
    flaw in the correction VALUE itself?
47. B2's gate (C1) reliably beats the fixed γ=1 arm (C0) on every channel,
    showing adaptive correction strength is real and learnable even
    without a correction-aligned retrieval. Is this evidence that gate
    architecture/training itself is not the bottleneck, strengthening the
    case that the retrieval SELECTION step (not the fusion step) is where
    the Correction hypothesis's remaining value is locked up?

Please answer using the structure in `research/NEXT_EXPERIMENT.md`.
