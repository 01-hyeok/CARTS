# TRACK-V5-HEAD-LOSS-DUAL-ENCODER01 -- Status / Interim Report

**This track is IN PROGRESS, not complete.** Per the user's explicit continuation instruction, work resumed Phase 0 through Phase 1-4 for the first of 4 primary cells. This file will be rewritten as a true final report once all 4 cells (ETTh1 H96/H720, Weather H96/H720) complete Phase 1-4; until then it records exactly what is done, what is pending, and the real numbers obtained so far -- not a projection.

## Completed

- **PHASE 0** (`PHASE_0_IMPLEMENTATION_AUDIT.md`): CLOSED. L0/L3a/L3b implemented (`utils/differentiable_topk01.py`, `scripts/train_v5_head_loss_dual_encoder01.py`), 24/24 new unit tests pass, L0-canonical parity confirmed exact (`.eval()` mode, byte-identical to `train_retriever_pool01.compute_scores`), gradient probe confirms `lambda_A=1.0` is not pathologically mismatched in scale. A genuine smoke-test-ordering bug (comparing Separate-encoder weights for divergence BEFORE the optimizer step that would cause them to diverge) was found and fixed in the imported `train_p100_dual_encoder01.py` pattern and this track's own script.
- **PHASE 1-4, ETTh1_96 (all 6 arms, full Stage1 + Stage2)**: COMPLETE. See `PHASE_1_SHARED_LOSS_REPORT.md`, `PHASE_2_SHARED_STAGE2_REPORT.md`, `PHASE_3_SEPARATE_LOSS_REPORT.md`, `PHASE_4_SEPARATE_STAGE2_REPORT.md`. S0's Stage1 (0.663855/0.409808) and Stage2 (0.369597) numbers reproduce canonical V5 exactly, validating the whole new pipeline end to end, not just at the implementation-unit level.

## ETTh1_96 headline result (one cell, seed0 -- NOT yet a cross-cell or cross-seed claim)

| Arm | Encoder | Loss | Stage2 Test MSE |
|---|---|---|---:|
| Base | -- | -- | 0.39242 |
| **S0** | Shared | L0 | **0.369597 (best)** |
| S1 | Shared | L3a | 0.370652 |
| S2 | Shared | L3b | 0.371837 |
| D0 | Separate | L0 | 0.376534 |
| D1 | Separate | L3a | 0.373972 |
| D2 | Separate | L3b | 0.375010 |

**On this cell: plain L0 + Shared Encoder (= canonical V5) is still the best arm.** Neither aggregate-aligned loss improves over L0 within either encoder structure's own best arm. The one genuinely interesting finding: within the SEPARATE encoder structure specifically, the aggregate loss DOES measurably reduce the within-set cross term `C` (Phase 3) and this DOES transfer to a real Stage2 improvement over the Separate+L0 baseline (D1 < D0, Phase 4) -- an encoder-structure x loss interaction that is absent in the Shared structure (where the aggregate loss does nothing useful at all). This is exactly the kind of interaction RQ5 asked about, found on the first cell tested -- but it is a single cell, single seed, and does not change the overall ranking (every Separate arm is still worse than every Shared arm on this cell).

## Explicitly NOT yet done (pending, in order)

1. ETTh1_720, Weather_96, Weather_720 -- full Phase 1-4 (18 more training runs + cache builds + Stage2 runs). Based on ETTh1_96's wall-clock (~155-180s/arm for training, Weather cells were 5-15x slower for the pure-diagnostic forward-pass-only script in earlier tracks this session, so full training+Stage2 on Weather could run into the tens of minutes per arm).
2. The full cross-cell Loss x Encoder factorial table (section 11, Tables A-F) -- cannot be written honestly with only 1 of 4 cells.
3. Phase 5 (L1 Head-KL / L2 Complementarity follow-up) -- explicitly gated on completing the primary 4-cell comparison first, per spec section 10's own ordering rule; not started.
4. The 12 final research questions (section 16) -- answerable in full only after all 4 cells are in; a partial, ETTh1_96-only answer to Q1-Q5/Q7/Q9 is already visible in the Phase reports above, but is explicitly flagged there as single-cell, not a final answer.
5. seed1/seed2/H192/H336 extension -- correctly NOT started, per spec's own explicit prohibition on auto-running this within the primary track.

## GO/STOP basis (preliminary, ETTh1_96 only -- not a track-level verdict)

Per spec section 13's NO-GO conditions: "Stage1 개선이 Stage2로 전혀 전이되지 않음" does NOT apply here (the one real Stage1 signal found, D1 vs D0, DID transfer). "Soft Loss만 개선되고 Hard Top-10 악화" is PARTIALLY observed for S1 (soft AggMSE improved marginally, 0.398719 vs S0's 0.399121, while Hard AggMSE got worse) -- a genuine instance of the failure mode the spec warns about, on the Shared arm specifically. No implementation-error NO-GO condition is triggered (candidate leakage, teacher error, cosine-recomputation error, gradient instability, missing key gradient -- all checked and clear in Phase 0). **This is not enough data for a track-level verdict; continuing to the remaining 3 cells is the correct next step, not stopping.**
