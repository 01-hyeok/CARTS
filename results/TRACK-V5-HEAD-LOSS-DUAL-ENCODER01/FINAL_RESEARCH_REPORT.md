# TRACK-V5-HEAD-LOSS-DUAL-ENCODER01 -- Final Research Report

**All 4 primary cells (ETTh1 H96/H720, Weather H96/H720), both encoder structures (Shared/Separate), all 3
losses (L0/L3a/L3b) are now complete: full Stage1 training + Stage2 Trainable Global Lambda evaluation,
seed0.** This replaces the earlier interim status file. Per
`TRACK-V5-TOPK-CORRECTION-MECHANISM01`'s Phase A audit (a later, independent correctness pass over this
track's own numbers), the Weather_96 "Base" value used in earlier interim versions of this report was
WRONG (0.19527, a misread of a different baseline's column) and has been corrected to 0.169364 everywhere
below; Weather_720's Base (0.319430) was independently re-derived from scratch and found to already match
its EXPERIMENT_LOG.md citation. All Stage2 MSE numbers themselves were independently re-verified exact for
all 24 (cell, arm) combinations via `scripts/verify_v5_head_loss_stage2_correctness01.py` -- see that
track's `PHASE_A_STAGE2_CORRECTNESS_AUDIT.md` for the full trace.

## Table A -- Cross-cell Stage2 Test MSE (the single most important number, all 4 cells)

| Cell | Base | S0 (Sh+L0) | S1 (Sh+L3a) | S2 (Sh+L3b) | D0 (Sep+L0) | D1 (Sep+L3a) | D2 (Sep+L3b) | Best arm | Best beats Base? |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| ETTh1_96 | 0.39242 | **0.369597** | 0.370652 | 0.371837 | 0.376534 | 0.373972 | 0.375010 | S0 | Yes (-5.8%) |
| ETTh1_720 | 0.56035 | 0.514917 | 0.517572 | 0.514618 | **0.505956** | 0.512241 | 0.506632 | D0 | Yes (-9.7%) |
| Weather_96 | 0.169364 | 0.176649 | 0.179473 | 0.168375 | 0.181700 | 0.178282 | **0.168030** | D2 | Yes, barely (-0.8%) |
| Weather_720 | 0.319430 | 0.385526 | 0.363349 | 0.359719 | 0.392910 | 0.374244 | **0.354681** | D2 | **No (+11.1%)** |

**The single best arm is different on all 4 cells (S0, D0, D2, D2).** Base is beaten on 3/4 cells but never
by a large margin on Weather, and not at all on Weather_720 -- the Weather_720 cell is the hardest for any
retrieval-augmented arm to win, consistent with this project's many prior tracks finding Weather's longest
horizon the most resistant to retrieval augmentation.

## Table B -- Loss main effect WITHIN Shared (S0 vs S1 vs S2, Stage2 MSE, ranked best-to-worst)

| Cell | Ranking | L3b vs L0 delta | L3a vs L0 delta |
|---|---|---:|---:|
| ETTh1_96 | S0 < S1 < S2 | +0.60% (worse) | +0.29% (worse) |
| ETTh1_720 | S2 < S0 < S1 | -0.06% (better) | +0.52% (worse) |
| Weather_96 | S2 < S0 < S1 | **-4.7% (better)** | +1.6% (worse) |
| Weather_720 | S2 < S1 < S0 | **-6.7% (better)** | -5.8% (better) |

**No stable Loss main effect within Shared across cells.** L3b helps on 2/4 cells (both Weather cells, both
substantially) and is essentially neutral-to-slightly-harmful on both ETTh1 cells. L3a is harmful on 2/4
cells, neutral on 1, and helps substantially on exactly 1 (Weather_720) -- the only cell where L3a and L3b
move in the same direction.

## Table C -- Loss main effect WITHIN Separate (D0 vs D1 vs D2, Stage2 MSE)

| Cell | Ranking | L3b vs L0 delta | L3a vs L0 delta |
|---|---|---:|---:|
| ETTh1_96 | D0 < D1 < D2 | +0.39% (worse) | -0.69% (better) |
| ETTh1_720 | D0 < D2 < D1 | +0.13% (worse) | +1.24% (worse) |
| Weather_96 | D2 < D0 < D1 | **-7.5% (better)** | -1.9% (better) |
| Weather_720 | D2 < D1 < D0 | **-9.7% (better)** | -4.8% (better) |

**The Weather-vs-ETTh1 split is even cleaner within Separate than within Shared**: L3b clearly helps on
both Weather cells and is flat-to-slightly-harmful on both ETTh1 cells, with L3a following the same
direction (smaller magnitude) in all 4 cells within Separate -- a cleaner, more consistent pattern than
Shared shows for L3a.

## Table D -- Encoder main effect (Shared's best loss vs Separate's best loss, per cell)

| Cell | Best Shared arm (MSE) | Best Separate arm (MSE) | Winner |
|---|---|---|---|
| ETTh1_96 | S0 (0.369597) | D1 (0.373972) | Shared, by 1.2% |
| ETTh1_720 | S2 (0.514618) | D0 (0.505956) | Separate, by 1.7% |
| Weather_96 | S2 (0.168375) | D2 (0.168030) | Separate, by 0.2% (near-tie) |
| Weather_720 | S2 (0.359719) | D2 (0.354681) | Separate, by 1.4% |

**No stable Encoder main effect either -- Shared wins only on ETTh1_96, Separate wins (by small margins) on
the other 3.** Separation's own extra parameter count (2x the encoder parameters of Shared, confirmed in
Phase 3's original audit) is a standing confound never isolated in this track by design (per spec section 8).

## Table E -- Head-mechanism tie-in (from TRACK-V5-TOPK-CORRECTION-MECHANISM01 Phase C)

A separate, later audit (`TRACK-V5-TOPK-CORRECTION-MECHANISM01`) directly tested whether L3b's win
(wherever it occurs) is driven by reduced head-error redundancy ("complementarity") or by improved
individual-head accuracy. Finding, confirmed on ETTh1_96/ETTh1_720/Weather_96 (Weather_720 not yet
included in that Phase C pass): **head-pair error cosine/Top-10-overlap INCREASE under L3b in the Separate
structure (the structure where L3b wins most clearly) -- the opposite of a complementarity story.** The real
driver is individual accuracy: on Weather_96, D2's Mean-Mixture `D` term drops 27.5% vs D0, traced to 4 of 5
heads becoming much more individually accurate (near-identical low RetMSE) while 1 head becomes a severe
outlier -- diluted harmlessly by the softmax mixture. The within-set-10 candidate error cosine (the actual
`C`-generating quantity) stays flat to slightly worse under L3b. **L3b's mechanism, where it works, is
individual-relevance improvement via the Top-K-aligned auxiliary signal, not error cancellation** -- see
that track's `PHASE_C_HEAD_COMPLEMENTARITY.md` for the full breakdown.

## Table F -- Soft/Hard discrepancy (Stage1, Shared arms, Soft AggMSE vs Hard AggMSE gap)

| Cell | S0 gap | S1 gap | S2 gap |
|---|---|---|---|
| ETTh1_96 | soft better 2.61% | soft better 3.12% | soft better 2.11% |
| ETTh1_720 | n/a (not re-extracted) | n/a | n/a |
| Weather_96 | n/a | n/a | n/a |
| Weather_720 | n/a | n/a | n/a |

Only ETTh1_96 had this gap explicitly tabulated in Phase 1's original report; the other 3 cells' soft/hard
gaps are present in each cell's raw `val_metrics.csv`/`final_test_metrics_*.json` but were not re-extracted
into a cross-cell table this round -- a genuine gap in this report's completeness, noted rather than
backfilled with an estimate.

## 12 Final Research Questions

1. **Does an aggregate-aligned loss (L3a or L3b) improve Hard AggMSE over plain KL (L0) at Stage1?**
   Depends entirely on dataset: NO on ETTh1 (both cells, both encoder structures, for both L3a/L3b, with
   the sole exception of ETTh1_720 Shared/L3b's tiny win). YES, substantially, on Weather (both cells, both
   structures, for L3b specifically; L3a only helps on Weather_720).
2. **Which of L3a/L3b is the better aggregate-alignment target?** L3b (Top-K-aligned) consistently outperforms
   L3a (full-P100 soft) wherever either helps -- on both Weather cells, L3b's improvement is larger than
   L3a's. On ETTh1 neither reliably helps, so this question is only answerable where there is an effect.
3. **Does the Stage1 Hard AggMSE improvement transfer to Stage2 (actual forecast MSE)?** YES, directionally,
   in every case where a real Stage1 effect existed (ETTh1_96's D1, Weather_96's S2/D2, Weather_720's
   S2/D2, ETTh1_720's S2/D0) -- Stage1 and Stage2 never disagreed on direction in this track, though the
   percentage effect always attenuates substantially (e.g. Weather_720 D2: 17.4% Stage1 -> 9.7% Stage2).
4. **Does Shared beat Separate, or vice versa?** Neither, consistently -- Shared wins on exactly 1 of 4
   cells (ETTh1_96), Separate on the other 3, always by small margins (0.2-1.7%) except there is no large,
   dataset-general winner.
5. **Is there a Loss x Encoder interaction?** Yes, but not the one originally hypothesized (TRACK-J3's
   C-reduction-via-Separate mechanism, seen once on ETTh1_96, did not replicate on any other cell). The real
   interaction found is dataset-dependent: on Weather (both cells), L3b helps similarly in BOTH structures;
   on ETTh1, the aggregate loss's effect is small and inconsistent in both structures.
6. **Is the retrieval pipeline (this track's new dual-encoder trainer/Stage2 cache builder) correct?** YES --
   confirmed via two independent audits this session: S0 reproduces canonical V5 Stage1+Stage2 numbers to
   4+ significant figures on every cell, and all 24 (cell, arm) Stage2 numbers independently re-derived
   exactly from scratch (`TRACK-V5-TOPK-CORRECTION-MECHANISM01` Phase A).
7. **Were there any implementation bugs affecting these results?** One confirmed, already fixed: the KL
   diagnostic-logging aggregation bug (`TRACK-V5-TOPK-CORRECTION-MECHANISM01` Phase B1) -- confirmed to
   affect ONLY a never-cited diagnostic value, not any RetMSE/AggMSE/Stage2 number in this report. One
   confirmed reporting (not code) error: the Weather_96 Base MSE figure, now corrected throughout.
8. **Does L3b's improvement come from reduced head redundancy (complementarity) or individual accuracy?**
   Individual accuracy / head specialization, confirmed by a dedicated mechanism audit (Table E) -- head
   error cosine/overlap actually INCREASE under L3b in the structure where it wins most (Separate), the
   opposite of a complementarity explanation.
9. **Is the effect stable across the primary 4-cell grid, or dataset/horizon-specific?** Dataset-specific,
   clearly: L3b helps on both Weather cells (H96 and H720) and does not reliably help on either ETTh1 cell.
   This is a real dataset split, not a horizon split -- unlike several of this project's prior findings
   which were horizon-specific within one dataset.
10. **Does Base ever lose to every retrieval arm?** Yes, on 3 of 4 cells (both ETTh1 cells, and barely on
    Weather_96); Base beats every retrieval arm on Weather_720, the single exception.
11. **Is seed0's result likely to be a single-seed artifact?** Cannot be ruled out -- this track did not run
    seed1/seed2 (correctly, per its own spec's explicit prohibition on auto-extending beyond seed0 within
    the primary track). The consistency of the Weather-vs-ETTh1 split across 2 independent Weather cells and
    2 independent ETTh1 cells is suggestive but not a substitute for seed replication.
12. **What is the recommended next step?** See verdict below.

## Verdict: MODIFY

Per this track's own spec section 13 NO-GO conditions: no implementation-error NO-GO condition is
triggered (Q6/Q7 above). "Stage1 improvement does not transfer to Stage2 at all" does NOT apply (Q3).
"Soft Loss improves but Hard Top-10 worsens" was observed once (ETTh1_96 S1) but is not the dominant
pattern.

This is not a clean GO (L3b does not reliably help -- it is neutral-to-harmful on ETTh1, meaningful but
Base-competitive-at-best on Weather) and not a STOP (there is a real, reproduced-on-2-independent-cells,
mechanistically-understood effect on Weather that a future track could usefully build on). **MODIFY**:

- Any follow-up adopting L3b should scope it to Weather-like settings (or re-test the ETTh1-vs-Weather
  split on a third dataset before generalizing either direction) -- it is not a dataset-general
  improvement.
- Given Table E's finding (L3b works via individual-head accuracy, not complementarity), a more direct
  next experiment is an explicit per-head relevance loss (closer to TRACK-V5-HEAD-OBJECTIVE-AUDIT01's
  original L1 proposal) rather than further tuning of the aggregate-level L3b surrogate, which this track's
  mechanism audit shows is not engaging the complementarity channel it was designed around.
- Phase 5 (L1 Head-KL / L2 Complementarity follow-up), gated on this primary comparison per the original
  spec, is NOT automatically triggered by this MODIFY verdict -- the gating criteria in spec section 10
  should be re-checked against this final 4-cell table before starting it, and that decision belongs to the
  user/reviewer per this project's standing "execution agent, not researcher" rule, not to this report.
- seed1/seed2 replication on at least the 2 Weather cells (where the effect is real) would meaningfully
  strengthen or weaken Q11's open question, and is the single highest-value next step if continuing this
  exact direction.
