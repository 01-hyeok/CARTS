# EXP-FIRSTANCHOR-DIAG -- factual notes (2026-09-07)

Status: **COMPLETE for 3 cells (ETTh1 H96, Weather H96, ETTh1 H720).**
Weather H720 cell **not run** -- EXP-MARGUTIL01's Weather H720 training was
stopped and its checkpoint removed by explicit user decision before this
diagnostic could reuse it (see `results/EXP-MARGUTIL01/notes.md`, D-0013 in
`research/RESEARCH_DECISIONS.md`). No new training was done anywhere in
this experiment -- every arm reuses an EXP-MARGUTIL01 checkpoint's frozen
encoder, `SetConditioner`, and `UtilityHead` exactly as-is; the causal
variable is only which rule picks the t=1 candidate.

## Design

Four arms per cell: **Dense-first** (existing EXP-MARGUTIL01 free-running
result, reused), **B0-first** (t=1 = B0's own production Top-1 score,
t=2..10 = unmodified Dense selector), **Oracle-first** (t=1 = argmin
singleton future MSE, a diagnostic-only intervention using `Y_q`; t=2..10 =
unmodified Dense selector), **B0 baseline** (production, unforced).
`identity_check_singleton_oracle_eq_teacher_first: true` on every cell --
Oracle-first's t=1 is verified, per query, to equal
`select_greedy_weighted_set`'s own cached first pick.

## Primary table (Stage-2 Final MSE, recovery relative to Dense-first's own
failure: 0 = no recovery, 1 = fully recovers to B0, >1 = beats B0)

| Cell | B0 | Dense-first | B0-first | Recovery | Oracle-first | Recovery |
|---|---:|---:|---:|---:|---:|---:|
| ETTh1 H96 | 0.37312 | 0.52991 | 0.43661 | **59.5%** | 0.21717 | **199.5%** |
| Weather H96 | 0.17553 | 0.33253 | 0.21261 | **76.4%** | 0.41411 | **-52.0%** |
| ETTh1 H720 | 0.46827 | 0.49269 | 0.51995 | **-111.6%** | 0.43635 | **230.7%** |

Full per-arm numbers: `comparison.csv`. Stepwise A_weighted(S_t) trajectory
(B0's own fixed score used for the softmax weights at every step, never the
Dense model's own score): `stepwise_aggregate_trajectory.csv`.

## No single clean story across cells

- **B0-first**: strong, consistent recovery on both H96 cells (60-76%), but
  makes ETTh1 H720 WORSE than even Dense-first (-112% "recovery", i.e. a
  regression). Not a uniformly safe intervention.
- **Oracle-first**: beats B0 outright on both ETTh1 cells (H96: 200%, H720:
  231% recovery -- Stage-2 MSE below B0), but is markedly WORSE than
  Dense-first on Weather H96 (-52%). The "perfect" singleton anchor is not
  uniformly good either.
- The stepwise trajectory shows very little movement after t=1 on every
  cell/arm (e.g. ETTh1 H96 oracle-first: A_weighted moves from 0.19101 at
  t=1 to only 0.19665 by t=10) -- **t=1 overwhelmingly determines each arm's
  final aggregate**, consistent across all three cells. This is the one
  finding that IS consistent: whichever candidate lands at t=1, the
  Dense-conditioned steps t=2..10 barely move the aggregate away from it.
  The disagreement across cells is about WHICH t=1 rule is good, not about
  whether t=1 dominates the outcome.

## t=1 candidate quality (singleton future MSE, full test set; `t1_candidate_quality.csv`)

| Cell | Dense mean | B0 mean | Oracle mean | B0 beats Dense | Dense beats B0 |
|---|---:|---:|---:|---:|---:|
| ETTh1 H96 | 0.988 | 0.666 | 0.191 | 64.4% | 35.0% |
| Weather H96 | 0.567 | 0.452 | 0.036 | 69.8% | 27.1% |
| ETTh1 H720 | 1.211 | 0.951 | 0.442 | 66.7% | 33.1% |

B0's own singleton pick is substantially better than Dense's own t=1 pick
on every cell measured (B0 beats Dense roughly 2:1), yet exact matches to
the true oracle singleton are rare for both (B0: 1-5%, Dense: <0.2%).

## t=1 rank diagnostics (`t1_rank_diagnostics.csv`, 200-query subsample per
cell, full candidate population per query -- never a candidate subsample)

ETTh1 H96: oracle-best candidate's median predicted rank (by the Dense
model's own t=1 u_hat, out of ~8449 valid) is 489 (mean percentile rank
14.3%); Top-1 hit rate 0%, Top-50 containment 11.5%;
`spearman_within_teacher_top1pct` 0.080 (weak -- the Dense model's ranking
degrades sharply within the extreme top tail, even though its *global*
teacher-forced Spearman, reported in EXP-MARGUTIL01, was 0.543). This
quantifies exactly what section 10 asked: global ranking quality is not
the same as top-tail ranking quality, and the top-tail is what t=1
selection actually depends on.

## Structural / leakage guarantees (also covered by `tests/test_exp_firstanchor_diag.py`)

- `duplicate_rate` / `invalid_rate` = 0.0 on every arm, every cell.
- `run_arm`'s only intervention point is `first_pick` at t=0; t=1..K-1 use
  the identical unmodified argmax branch regardless of arm (verified by
  forcing t=1 to the value an arm would have picked anyway and confirming
  the full K-step trajectory is byte-identical).
- `a_weighted_prefix` never references `utility_head` -- the Stage-2/
  trajectory metric is always B0's own fixed score, never the Dense
  model's.
- Oracle-first's `Y_q` usage is confined to computing `i1_oracle` (the
  intervention index) and to the diagnostic-only rank/quality metrics;
  `run_arm`'s own source contains no reference to `query_future`/`q_tgt`.

## Sanity

`pytest tests/`: 485 passed (8 new, `tests/test_exp_firstanchor_diag.py`),
same 2 pre-existing failures, no regression.

## Interpretation kept to what this diagnostic design supports

This experiment answers "does fixing t=1 revive the rest of the Dense
selector" per cell, and the answer is cell-dependent, not uniform:

- **ETTh1 (both horizons):** yes, decisively for Oracle-first, and clearly
  helpful (though incomplete) for the deployable B0-first hybrid at H96.
- **Weather H96:** yes for the deployable B0-first hybrid (76% recovery,
  the single best recovery number in this table), but the "should-be-ideal"
  Oracle-first intervention backfires.
- **ETTh1 H720:** B0-first backfires; only the (non-deployable) Oracle-first
  intervention helps.

No cell supports "t=1 is the ONLY bottleneck and any reasonable anchor
fixes it" -- B0-first and Oracle-first disagree with each other on 2 of 3
cells about whether they even help. What IS consistent is that t=1
dominates the final aggregate on every cell (the stepwise trajectories),
so t=1 quality is clearly *necessary* to the outcome -- but which concrete
t=1 rule is sufficient is dataset/horizon-dependent, not settled by this
diagnostic alone. Interpretation, novelty assessment and next-experiment
recommendation belong to the reviewer, not this file.
