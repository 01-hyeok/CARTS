# TRACK-V5-HEAD-LOSS-DUAL-ENCODER01 -- Status / Interim Report

**This track is IN PROGRESS, not complete.** Per the user's explicit continuation instruction, work resumed Phase 0 through Phase 1-4 for the first of 4 primary cells. This file will be rewritten as a true final report once all 4 cells (ETTh1 H96/H720, Weather H96/H720) complete Phase 1-4; until then it records exactly what is done, what is pending, and the real numbers obtained so far -- not a projection.

## Completed

- **PHASE 0** (`PHASE_0_IMPLEMENTATION_AUDIT.md`): CLOSED. L0/L3a/L3b implemented (`utils/differentiable_topk01.py`, `scripts/train_v5_head_loss_dual_encoder01.py`), 24/24 new unit tests pass, L0-canonical parity confirmed exact (`.eval()` mode, byte-identical to `train_retriever_pool01.compute_scores`), gradient probe confirms `lambda_A=1.0` is not pathologically mismatched in scale. A genuine smoke-test-ordering bug (comparing Separate-encoder weights for divergence BEFORE the optimizer step that would cause them to diverge) was found and fixed in the imported `train_p100_dual_encoder01.py` pattern and this track's own script.
- **PHASE 1-4, ETTh1_96 AND ETTh1_720 (all 6 arms, full Stage1 + Stage2, both cells)**: COMPLETE. See `PHASE_1_SHARED_LOSS_REPORT.md`, `PHASE_2_SHARED_STAGE2_REPORT.md`, `PHASE_3_SEPARATE_LOSS_REPORT.md`, `PHASE_4_SEPARATE_STAGE2_REPORT.md`. S0 reproduces canonical V5 exactly on BOTH cells (ETTh1_96: Stage1 0.663855/0.409808, Stage2 0.369597; ETTh1_720: Stage1 0.942633/0.620570, Stage2 0.514917), validating the whole new pipeline end to end on two different horizons.

## Two-cell headline result (ETTh1 H96/H720, seed0 -- Weather not yet run, NOT a 4-cell claim)

| Cell | Base | S0 | S1 | S2 | D0 | D1 | D2 | Best arm |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| ETTh1_96 | 0.39242 | **0.369597** | 0.370652 | 0.371837 | 0.376534 | 0.373972 | 0.375010 | S0 |
| ETTh1_720 | 0.56035 | 0.514917 | 0.517572 | 0.514618 | **0.505956** | 0.512241 | 0.506632 | D0 |

**The best arm reverses completely between the two ETTh1 horizons already tested**: H96 favors Shared+L0 (canonical V5) and ranks every Separate arm worst; H720 favors Separate+L0 and ranks every Shared arm worse than D0/D2. Neither a pure "Loss main effect" (L0 best within H96's own groups, but L0 is NOT best within H720's Shared group -- S2/L3b beats S0 there) nor a pure "Encoder main effect" (Shared>Separate on H96, Separate>Shared on H720) is stable across just these two cells. The one Separate-specific aggregate-loss-helps-via-C-reduction finding from ETTh1_96 (D1<D0) also does NOT replicate on ETTh1_720 (D1>D0 there) -- it was itself a single-cell artifact, now falsified by the second cell, which is exactly why this track's spec requires completing the primary 4-cell comparison before drawing any conclusion. **No stable pattern should be assumed from ETTh1 alone; Weather_96/Weather_720 are necessary, not optional, before any RQ1/RQ4/RQ5 answer can be trusted.**

## Explicitly NOT yet done (pending, in order)

1. Weather_96, Weather_720 -- full Phase 1-4 (12 more training runs + cache builds + Stage2 runs). ETTh1_720 took essentially the same wall-clock as ETTh1_96 (~150-170s/arm for training) despite 7.5x the horizon length -- Weather cells were 5-15x slower than ETTh1 for the pure-diagnostic forward-pass-only script in earlier tracks this session (more channels, more queries), so full training+Stage2 on Weather is expected to take meaningfully longer per arm, though not necessarily as a direct multiple of the diagnostic-script ratio.
2. The full cross-cell Loss x Encoder factorial table (section 11, Tables A-F) -- cannot be written honestly with only 1 of 4 cells.
3. Phase 5 (L1 Head-KL / L2 Complementarity follow-up) -- explicitly gated on completing the primary 4-cell comparison first, per spec section 10's own ordering rule; not started.
4. The 12 final research questions (section 16) -- answerable in full only after all 4 cells are in; a partial, ETTh1_96-only answer to Q1-Q5/Q7/Q9 is already visible in the Phase reports above, but is explicitly flagged there as single-cell, not a final answer.
5. seed1/seed2/H192/H336 extension -- correctly NOT started, per spec's own explicit prohibition on auto-running this within the primary track.

## GO/STOP basis (preliminary, ETTh1_96 only -- not a track-level verdict)

Per spec section 13's NO-GO conditions: "Stage1 개선이 Stage2로 전혀 전이되지 않음" does NOT apply here (the one real Stage1 signal found, D1 vs D0, DID transfer). "Soft Loss만 개선되고 Hard Top-10 악화" is PARTIALLY observed for S1 (soft AggMSE improved marginally, 0.398719 vs S0's 0.399121, while Hard AggMSE got worse) -- a genuine instance of the failure mode the spec warns about, on the Shared arm specifically. No implementation-error NO-GO condition is triggered (candidate leakage, teacher error, cosine-recomputation error, gradient instability, missing key gradient -- all checked and clear in Phase 0). **This is not enough data for a track-level verdict; continuing to the remaining 3 cells is the correct next step, not stopping.**
