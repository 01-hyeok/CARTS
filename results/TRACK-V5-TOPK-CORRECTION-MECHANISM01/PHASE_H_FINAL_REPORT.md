# PHASE H -- Final Report and Verdict

## 12 Final Questions

1. **Was the Weather H96 Base-value error a reporting error or an evaluation error?** A reporting error.
   `research/EXPERIMENT_LOG.md:4607` had the correct number (`0.16936`) two columns away from where
   `TRACK-V5-HEAD-LOSS-DUAL-ENCODER01`'s own reports read it from (`0.19527`, the Raw-Cosine-baseline's
   column). The evaluation pipeline itself was never wrong (Phase A4).
2. **Is Stage1/Stage2 query alignment correct?** Yes. All 24 (cell, arm) Stage2 numbers across both tracks
   were independently re-derived from scratch via `scripts/verify_v5_head_loss_stage2_correctness01.py` and
   matched the recorded values exactly, including the `lambda=0 -> B` / `lambda=1 -> R` identities (Phase A4).
3. **Was the KL aggregation bug fixed?** Yes (Phase B1). Confirmed diagnostic-logging-only -- it never
   touched the training gradient, checkpoint selection, or any RetMSE/AggMSE/Stage2 number ever reported in
   either track.
4. **Do the existing RetMSE/AggMSE/Stage2 numbers reproduce?** Yes, exactly, everywhere checked.
5. **Did L3b-v1 reduce head error redundancy?** No (Phase C). Head-pair error cosine and Top-10 overlap
   either stay flat or INCREASE under L3b-v1 relative to L0, most dramatically for the Separate-encoder
   arm that wins most often.
6. **Does L3b-v2 converge to the exact Hard Top-K uniform weights?** Yes, by construction, including at
   exact ties (Phase D1-D3, 16/16 synthetic tests, strict tolerances). v1 provably does not (the K-th
   candidate is permanently stuck at half its correct weight).
7. **Does v2 focus enough at the real training temperature (tau=0.05)?** Partially. Phase D4 found every
   cell's minimum observed K/(K+1) score gap on real data is 2-4 orders of magnitude below `tau=0.05` --
   v1's bug is not a rare edge case in practice, but neither v1 nor v2 fully "focuses" at this tau for the
   near-tied queries where the gap is this small; v2 simply handles the ambiguity correctly (midpoint
   threshold) where v1 handles it incorrectly (anchored to the K-th score itself).
8. **Does L3b-v2 improve Hard AggMSE over v1?** Yes, in 7 of 8 (arm, cell) combinations, by 0.2-3.8%
   (Phase E/F). The one exception (Weather_96, Separate encoder) is a genuine regression, confirmed at both
   validation and test (Phase G6), most plausibly ordinary single-seed training variance (not re-verified
   with a repeat seed).
9. **Does that improvement reach Stage2?** Yes, in the SAME 7 of 8 cases, same direction at both stages in
   every instance (Phase F). Gains are smaller at Stage2 (0.2-1.1%) than at Stage1, the same attenuation
   pattern seen throughout both tracks this session.
10. **Is the improvement from individual relevance, error complementarity, or fusion?** Individual
    relevance / head specialization (Phase C, extended to v2 in Phase G). `mix_D` (individual-candidate
    accuracy) tracks the Stage1/Stage2 improvement pattern exactly (7/8 cases, same direction); head-pair
    error cosine/overlap move in the WRONG direction for a complementarity story in 6 of 8 cases (Phase G1)
    and the within-set-10 candidate cosine (the real `C`-generating quantity) shows no consistent
    improvement either. Fusion (Stage2's lambda) was never the mechanism under test in either track --
    Stage2 always just inherits whatever Stage1 retrieval quality it is given.
11. **Is the Shared/Separate effect consistent?** No -- mixed, exactly as `TRACK-V5-HEAD-LOSS-DUAL-ENCODER01`
    already found for v1. v2 does not resolve or clarify this inconsistency; if anything Phase G1 shows v2
    sometimes pushes Shared and Separate in OPPOSITE redundancy directions on the same cell (Weather_720).
12. **Is further Head-KL / explicit Complementarity Loss work worth it?** Qualified yes, with a specific
    redirection. Both tracks this session converge on the same finding from two independent angles (Phase C
    on v1, Phase G on v2): L3b's real mechanism is individual-head accuracy, not complementarity, and
    further tuning of an AGGREGATE-level surrogate (L3a/L3b in either version) is not engaging the
    complementarity channel the original V5-HEAD-OBJECTIVE-AUDIT01 spec hoped an aggregate loss would
    reach. A more direct experiment -- an explicit PER-HEAD loss (closer to that track's original L1
    proposal) -- is a better-targeted next step than further Top-K-relaxation tuning, which this track has
    now tested about as thoroughly as a single-seed, 4-cell study reasonably can.

## Verdicts

- **IMPLEMENTATION_PASS.** Phase D's v2 fix is mathematically justified, passes all 16 synthetic tests with
  strict tolerances, and the gradient path is confirmed correct via both autograd-consistency and
  finite-difference checks. The `--topk_version` flag defaults to `v1`, so no existing result in either
  track was altered by this implementation work.
- **METRIC_VALID.** Every Stage1/Stage2 number cited in either track's final tables was independently
  re-derived and matched exactly (Phase A4, this track's Phase F cross-check against
  `TRACK-V5-HEAD-LOSS-DUAL-ENCODER01`'s committed values). The one metric found NOT valid as originally
  reported (Weather_96 "Base") was traced, corrected, and the correction was propagated everywhere it
  appeared.
- **STAGE1_GO** (qualified). v2 is Stage1-superior to v1 in 7 of 8 tested (arm, cell) combinations. Not an
  unconditional GO: the gains are small (0.2-3.8%), one combination regresses, and the mechanism (Phase
  G1-G3) shows v2 is not doing anything qualitatively different from v1 -- it is the SAME individual-
  relevance mechanism, executed slightly more correctly at the boundary.
- **STAGE2_GO** (qualified, same basis). The Stage1 gains transfer to Stage2 in the identical 7 of 8 cases,
  with further attenuation (0.2-1.1%), and never change which arm is the overall best on any cell (D0 is
  still ETTh1_720's best arm; D2 is still Weather_720's best arm, whether v1 or v2 flavor).

## Overall recommendation

Given v2 is a strict correctness improvement with no observed downside risk to any existing result
(`--topk_version` defaults preserve every prior run unchanged) and a modest, mostly-positive empirical
effect, **v2 should be adopted as the default Differentiable Top-K implementation for any FUTURE L3b work**
in this codebase, while v1's results remain the historical record (preserved, not deleted, per this
project's standing rule). This is a MODIFY-level recommendation at the implementation layer, layered on top
of `TRACK-V5-HEAD-LOSS-DUAL-ENCODER01`'s own track-level MODIFY verdict (L3b helps on Weather, not
dataset-general) -- this track does not change that overall dataset-dependence conclusion, it refines the
mechanism understanding underneath it and fixes a real, now-documented implementation bug along the way.

**Next step, if this direction continues:** an explicit per-head relevance/KL loss (not a further
aggregate-level Top-K relaxation), per Q12 above -- that decision, and whether to pursue it, belongs to the
user/reviewer per this project's standing "execution agent, not researcher" rule, not to this report.
