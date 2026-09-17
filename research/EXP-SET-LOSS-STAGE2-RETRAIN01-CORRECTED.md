```text
NOT DIRECTLY COMPARABLE TO FACTORIAL SEED-0 BASELINE
UNAUTHORIZED SEED CHANGE
PRESERVED AS A SEED-1 REPLICATE ONLY
```

This file's Stage-1 input (`TRACK-A-SET-LOSS-CONTROL01`) used `--seed 1`
without user approval, not `--seed 0` as the Factorial baseline
(`set_onpolicy_cosine`) uses. The value-space (delta vs absolute) fix this
file documents remains valid on its own terms -- it is an orthogonal bug
from the seed issue -- but none of the numbers below should be compared
directly against the Factorial seed=0 baseline. Superseded, same-seed
(seed=0) re-run: `research/EXP-SET-LOSS-STAGE2-RETRAIN02.md`.

```text
EXECUTED:
- Identical fix and re-verification protocol as
  research/TRACK-A-CHOICECE-STAGE2-RETRAIN01-CORRECTED.md (same root cause,
  same code trace -- see that file section 1 for the full evidence, not
  duplicated here).
- Fixed scripts/build_setlossctrl_retrieval_cache01.py (delta-space fix +
  the batch-broadcast `.gather()` fix found during the sibling's smoke
  test, applied here proactively before any real cache was built).
- Rewrote scripts/train_setlossctrl_stage2_retrain01.py identically to the
  ChoiceCE trainer: cache schema rejection, FR-Agg cross-check gate,
  metric split, gate-distribution stats, counterfactual reuse, full
  per-arm deliverable set.
- 10 new unit tests added to tests/test_setlossctrl_stage2_retrain01.py
  (same battery as the sibling) -- all passing, GPU2 only, alongside the 9
  pre-existing tests in this file (39 total across both sibling test
  files).
- Full repo regression suite: 939 passed / 2 pre-existing failures
  unrelated to this work (same as sibling report), no new regressions.
- Full re-execution: all 4 H96 arms (A0_hard_choice, A1_adaptive_multipos,
  A2_srm, A3_setutility_softce), one at a time on GPU1, fresh Stage-2 init
  (NOT continued from the old buggy run), every arm's FR-Agg cross-check
  passed with diff=0.000000.
- Old report (research/EXP-SET-LOSS-STAGE2-RETRAIN01.md) marked
  INVALID_FOR_CONCLUSION, not deleted.
- research/REVIEW_FOR_CHATGPT.md: append-only correction section added
  (shared with the sibling experiment, same section).

NOT EXECUTED:
- H720: WAITING_FOR_STAGE1. TRACK-A-SET-LOSS-CONTROL01's own H720 Stage-1
  chain (pid 452129, arm A0_hard_choice) was still running at the time
  this round's work finished -- confirmed alive via `ps` immediately
  before this report was written. This experiment's H720 Stage-2 was
  correctly NOT started early; scripts/run_setlossctrl_stage2_retrain01_h96_corrected.sh
  only ever reads that experiment's already-finished DONE markers and
  never starts, modifies, or waits on it.
- git commit/push (explicitly withheld this round).
- Multi-seed replication.

CURRENT JOBS AFFECTED: NONE (see sibling report -- identical GPU1
concurrency situation, same two protected jobs confirmed alive throughout).

ROOT CAUSE VERIFIED: YES -- identical to sibling report; TRACK-A-SET-LOSS-CONTROL01's
retrieval cache used the same buggy `build_setlossctrl_retrieval_cache01.py`
sharing the exact same absolute-space bug as the ChoiceCE builder (both
scripts were written as siblings from the same template this session).

FILES CREATED:
- scripts/run_setlossctrl_stage2_retrain01_h96_corrected.sh
- results/EXP-SET-LOSS-STAGE2-RETRAIN01-CORRECTED/** (4 arms, H96 only)
- research/EXP-SET-LOSS-STAGE2-RETRAIN01-CORRECTED.md (this file)

FILES MODIFIED:
- scripts/build_setlossctrl_retrieval_cache01.py
- scripts/train_setlossctrl_stage2_retrain01.py
- tests/test_setlossctrl_stage2_retrain01.py
- research/EXP-SET-LOSS-STAGE2-RETRAIN01.md (INVALID_FOR_CONCLUSION block)
- research/REVIEW_FOR_CHATGPT.md (append-only, shared with sibling)

FILES OVERWRITTEN: NONE. Old run's results/checkpoints under
results/EXP-SET-LOSS-STAGE2-RETRAIN01/ and
checkpoints/exp_set_loss_stage2_retrain01/ untouched; this round writes to
results/EXP-SET-LOSS-STAGE2-RETRAIN01-CORRECTED/ and
checkpoints/exp_set_loss_stage2_retrain01_corrected/.

STAGE1 CHECKPOINTS REUSED: YES, all 4 arms' TRACK-A-SET-LOSS-CONTROL01
Stage-1 checkpoints (frozen encoder + SetConditioner; cosine-only, no
RetrievalMetric) reused read-only, unchanged.

STAGE2 CHECKPOINTS REJECTED: YES, identical policy to sibling -- old
Stage-2 gate/head/fusion weights never loaded; fresh shared init per cell,
seed=1.

CACHE VALIDATION: PASS for all 4 arms -- schema fields present, FR-Agg
cross-check diff=0.000000 for every arm (see table below).

CHOICECE STAGE2 STATUS: see research/TRACK-A-CHOICECE-STAGE2-RETRAIN01-CORRECTED.md
(this file covers EXP-SET-LOSS-STAGE2-RETRAIN01 only).

SET-LOSS STAGE2 STATUS: DONE (4/4 H96 arms, DONE markers present, zero
FAILED.json).

H720 LOSS STATUS: WAITING_FOR_STAGE1 (not started, by design -- see NOT
EXECUTED above).

PROTOCOL DEVIATIONS: same batch-broadcast `.gather()` fix as the sibling
report (proactively applied here from the start, since both scripts share
the same code pattern) -- no separate deviation beyond that.
```

## Table -- 4 loss arms, H96, corrected pipeline

Independent Base-only Forecaster (unaffected, reused read-only): H96 =
0.386451.

| loss | fragg_cache | fragg_stage1 | fragg_diff | final_MSE | delta_vs_IB | base_MSE(lambda0) | proj_ret_MSE(lambda1) | gate_mean | trained_beats_both_counterfactuals |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A0 Hard Choice CE | 0.712167 | 0.712167 | 0.000000 | 0.37982 | -1.72% | 0.63293 | 0.42187 | 0.439 | YES |
| A1 Adaptive MultiPos | 0.715429 | 0.715429 | 0.000000 | 0.37800 | -2.19% | 0.62407 | 0.41945 | 0.435 | YES |
| A2 Soft Regret Mass | 0.710819 | 0.710819 | 0.000000 | 0.38311 | -0.86% | 0.63735 | 0.42447 | 0.438 | YES |
| A3 Set-Utility Soft CE | 0.711540 | 0.711540 | 0.000000 | 0.37810 | -2.16% | 0.62005 | 0.41895 | 0.434 | YES |

Spread across the 4 losses: final_MSE ranges 0.37800-0.38311 (a 1.34%
relative range), gate_mean ranges 0.434-0.439 (near-identical across
losses). All 4 arms show the SAME qualitative pattern as ChoiceCE's H96
arms: the trained gate beats both the base-only and retrieval-only
counterfactuals, and all 4 clearly beat Independent Base.

## Ten closing questions

**1. FR-Agg match?** YES, all 4 arms, diff=0.000000.

**2. Does the corrected retrieval beat Independent Base?** YES, all 4/4
arms (-0.86% to -2.19%), same qualitative pattern as ChoiceCE's H96 arms
(which also all beat Independent Base) -- consistent with H96 being the
horizon where this architecture's gated fusion genuinely helps.

**3. Individual vs Set**: N/A -- TRACK-A-SET-LOSS-CONTROL01 is Greedy Set
Oracle only (no Individual-oracle arm in this experiment's design).

**4. Does Stage-1 FR-Agg improvement predict Stage-2 MSE improvement?**
Weakly and non-monotonically within this set: A2 (SRM) has the BEST
(lowest) Stage-1 FR-Agg (0.710819) but the WORST (highest) Stage-2
final_MSE (0.38311); A1 (MultiPos) has the worst FR-Agg (0.715429, tied
for a middling position) but the best final_MSE (0.37800). The FR-Agg
spread across all 4 losses is tiny (0.7108-0.7154, a 0.6% relative range)
and does not order the final_MSE ranking -- no evidence here that any one
loss produces a Stage-1 retriever that is meaningfully better for Stage-2
than the others.

**5. Which of the 4 losses wins under corrected Stage-2?** A1 Adaptive
MultiPos, by final_MSE (0.37800), narrowly ahead of A3 Set-Utility Soft CE
(0.37810) and A0 Hard Choice CE (0.37982); A2 Soft Regret Mass is
noticeably last (0.38311). The full spread (0.37800-0.38311, 1.34%
relative) is larger than the old (buggy) run's reported spread
(0.38568-0.38612, 0.044%) -- the corrected pipeline differentiates the 4
losses more than the buggy one did, though still not by a large margin,
and this remains a SINGLE seed (seed=1) per loss.

**6. Does the trained gate beat both counterfactuals?** YES, all 4/4 arms
-- same as ChoiceCE's H96 arms, different from ChoiceCE's H720 arms
(where 7/8 lost to lambda=0). This experiment only has H96 data so far, so
it cannot independently confirm or contradict the H720 regression seen in
the sibling report; that remains an open question pending this
experiment's own H720 run once Stage-1 finishes.

**7. gate/lambda vs old (buggy) run**: old run reported gate_mean
0.048-0.055 (near-collapsed); corrected run reports gate_mean 0.434-0.439
-- an order-of-magnitude increase in how much the trained gate actually
weights the retrieval branch, matching the same qualitative shift seen in
the sibling ChoiceCE report.

**8. H96/H720 directional consistency**: cannot be assessed yet -- H720
is WAITING_FOR_STAGE1 for this experiment. Given the sibling
(ChoiceCE) experiment found H96 and H720 diverge sharply (H96 favors
retrieval, H720 mostly does not), this experiment's own H720 run --
once its Stage-1 dependency finishes -- is the natural next check for
whether that H96/H720 split is a property of the fusion architecture in
general (would replicate here) or specific to the ChoiceCE retriever
(would NOT replicate here, since this experiment uses a fixed
Greedy-Set-Oracle retriever across all 4 arms).

**9. Is this plausibly single-seed noise?** The 4 losses' final_MSE values
are close enough (1.34% relative spread) that the specific ranking (A1 >
A3 > A0 > A2) should NOT be treated as a confident ordering from a single
seed; the shared qualitative finding (all 4 beat both Independent Base and
both counterfactuals, by broadly similar margins) is more robust than the
specific ranking among them.

**10. 3-seed follow-up recommendation**: YES -- two follow-ups, in order
of priority: (a) run this experiment's own H720 chain once
TRACK-A-SET-LOSS-CONTROL01's H720 Stage-1 finishes, to check whether the
sibling's H96-vs-H720 regression replicates here; (b) if (a) confirms the
regression, a 3-seed x representative-arm (e.g. A0_hard_choice and
A1_adaptive_multipos) re-run at both horizons would be needed before
treating either the H96 uniform improvement or the H720 regression as
more than single-seed noise.
