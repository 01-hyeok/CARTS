# TRACK-V5-HEAD-LOSS-DUAL-ENCODER01 -- Status / Interim Report

**This track is IN PROGRESS, not complete.** Per the user's explicit continuation instruction, work resumed Phase 0 through Phase 1-4 for the first of 4 primary cells. This file will be rewritten as a true final report once all 4 cells (ETTh1 H96/H720, Weather H96/H720) complete Phase 1-4; until then it records exactly what is done, what is pending, and the real numbers obtained so far -- not a projection.

## Completed

- **PHASE 0** (`PHASE_0_IMPLEMENTATION_AUDIT.md`): CLOSED. L0/L3a/L3b implemented (`utils/differentiable_topk01.py`, `scripts/train_v5_head_loss_dual_encoder01.py`), 24/24 new unit tests pass, L0-canonical parity confirmed exact (`.eval()` mode, byte-identical to `train_retriever_pool01.compute_scores`), gradient probe confirms `lambda_A=1.0` is not pathologically mismatched in scale. A genuine smoke-test-ordering bug (comparing Separate-encoder weights for divergence BEFORE the optimizer step that would cause them to diverge) was found and fixed in the imported `train_p100_dual_encoder01.py` pattern and this track's own script.
- **PHASE 1-4, ETTh1_96, ETTh1_720 AND Weather_96 (all 6 arms, full Stage1 + Stage2, three of four cells)**: COMPLETE. See `PHASE_1_SHARED_LOSS_REPORT.md`, `PHASE_2_SHARED_STAGE2_REPORT.md`, `PHASE_3_SEPARATE_LOSS_REPORT.md`, `PHASE_4_SEPARATE_STAGE2_REPORT.md`. S0 reproduces canonical V5 exactly on both ETTh1 cells (ETTh1_96: Stage1 0.663855/0.409808, Stage2 0.369597; ETTh1_720: Stage1 0.942633/0.620570, Stage2 0.514917), validating the whole new pipeline end to end.

## Three-cell headline result (ETTh1 H96/H720 + Weather_96, seed0 -- Weather_720 not yet run, NOT a 4-cell claim)

| Cell | Base | S0 | S1 | S2 | D0 | D1 | D2 | Best arm |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ETTh1_96 | 0.39242 | **0.369597** | 0.370652 | 0.371837 | 0.376534 | 0.373972 | 0.375010 | S0 |
| ETTh1_720 | 0.56035 | 0.514917 | 0.517572 | 0.514618 | **0.505956** | 0.512241 | 0.506632 | D0 |
| Weather_96 | 0.19527 | 0.176649 | 0.179473 | 0.168375 | 0.181700 | 0.178282 | **0.168030** | D2 |

**The best arm differs on all three cells tested so far (S0, D0, D2)** -- neither a pure "Loss main effect" nor a pure "Encoder main effect" is stable. But Weather_96 shows something new and large: **L3b (S2/D2) beats L0 by 4.7-7.5% at Stage2 in BOTH encoder structures**, the biggest and most consistent Loss effect in the track, and the first cell where ALL 6 retrieval arms beat Base (0.19527) -- a departure from this project's long-standing Weather-vs-Base pattern. ETTh1's two single-cell findings (ETTh1_96's Separate+aggregate-loss C-reduction, ETTh1_720's Separate+L0 dominance) do not explain Weather_96's result -- L3b wins on Weather_96 regardless of encoder structure, a genuinely different mechanism/cell-dependence than either ETTh1 finding. **Weather_720 is still necessary before any RQ1/RQ4/RQ5 answer is final.**

## Explicitly NOT yet done (pending, in order)

1. Weather_720 -- full Phase 1-4 (6 more training runs + cache builds + Stage2 runs), the final primary cell.
2. The full cross-cell Loss x Encoder factorial table (section 11, Tables A-F) -- cannot be written honestly with only 3 of 4 cells.
3. Phase 5 (L1 Head-KL / L2 Complementarity follow-up) -- explicitly gated on completing the primary 4-cell comparison first, per spec section 10's own ordering rule; not started.
4. The 12 final research questions (section 16) -- answerable in full only after all 4 cells are in; a partial, ETTh1_96-only answer to Q1-Q5/Q7/Q9 is already visible in the Phase reports above, but is explicitly flagged there as single-cell, not a final answer.
5. seed1/seed2/H192/H336 extension -- correctly NOT started, per spec's own explicit prohibition on auto-running this within the primary track.

## GO/STOP basis (preliminary, 3 of 4 cells -- not a track-level verdict)

Per spec section 13's NO-GO conditions: "Stage1 개선이 Stage2로 전혀 전이되지 않음" does NOT apply anywhere -- every real Stage1 signal found so far (ETTh1_96's D1 vs D0, Weather_96's S2/D2 vs S0/D0) DID transfer to Stage2, in the same direction. "Soft Loss만 개선되고 Hard Top-10 악화" is PARTIALLY observed for ETTh1_96's S1 only. No implementation-error NO-GO condition is triggered (candidate leakage, teacher error, cosine-recomputation error, gradient instability, missing key gradient -- all checked and clear in Phase 0). Weather_96's result is the strongest positive signal for the track's underlying hypothesis so far (L3b helps, transfers, and is encoder-structure-independent) -- but it is one cell, and ETTh1_96/ETTh1_720 both found L3b/L3a non-helpful or mixed, so this is still **not enough data for a track-level verdict; Weather_720 is the correct next step, not stopping.**
