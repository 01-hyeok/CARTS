> **CORRECTION (TRACK-V5-TOPK-CORRECTION-MECHANISM01, Phase A1)**: this file's Weather_96 "Base" value
> below, `0.19527`, is WRONG -- it was misread from `research/EXPERIMENT_LOG.md:4607`'s "Raw Cosine (R0)
> MSE" column instead of the "Base Forecaster (B0) MSE" column two columns to its left. The correct
> Weather_96 Base test MSE is **0.169364** (`results/TRACK-R-FINAL-METHOD-GENERALIZATION01/Weather/H96/seed0/base/metrics.json`,
> independently re-verified exactly). All Stage2 MSE numbers themselves (S0/S1/S2 = 0.176649/0.179473/0.168375)
> are UNCHANGED and independently re-verified correct -- only the Base comparison point was wrong. With the
> corrected Base, S0 and S1 are actually WORSE than Base on this cell (not better); only S2 (L3b) edges it
> out, by <1%. See `results/TRACK-V5-TOPK-CORRECTION-MECHANISM01/PHASE_A_STAGE2_CORRECTNESS_AUDIT.md` for
> the full trace. This note is a correction, not a retraction -- the original numbers below are left as
> originally written.

# PHASE 2 -- Shared Encoder Stage2 Forecasting (S0/S1/S2)

**Status: ETTh1_96 complete; other 3 cells pending.** Original CARTS Trainable Global Lambda (`scripts/train_r_stage2_lambda01.py`, reused UNMODIFIED), same Base Forecaster checkpoint (`checkpoints/track_r_final_method_generalization01/ETTh1/H96/seed0/base/checkpoint.pth`) for every arm, independent retrieval cache per arm (`scripts/build_v5_head_loss_stage2_cache01.py`, new, cache-schema-identical to `build_v_meanmix_cache_pool01.py`).

## Baseline reproduction (section 7)

| Cell | Spec reference Base MSE | Spec reference V5 MSE | S0 (this track) MSE |
|---|---:|---:|---:|
| ETTh1_96 | 0.39242 | 0.36960 | **0.369597** |

S0 reproduces canonical V5's Stage2 test MSE to 5 decimals (0.369597 vs 0.36960) -- end-to-end pipeline validated (Phase 0's correctness claims hold through Stage1 AND Stage2, not just at the single-batch level).

## ETTh1_96 Stage2 Test MSE

| Cell | Base | S0 (L0) | S1 (L3a) | S2 (L3b) |
|---|---:|---:|---:|---:|
| ETTh1_96 | 0.39242 | **0.369597** | 0.370652 | 0.371837 |

All 3 Shared arms beat Base. **S0 (plain KL) is the best of the three** -- L3a and L3b both make Stage2 slightly WORSE than L0, monotonically (S0 < S1 < S2), mirroring the Stage1 ordering exactly. Lambda values: S0=0.4733, S1=0.4770, S2=0.4949 (all close to 0.5, none near the lambda=0 degenerate case).

## RQ3 answer (ETTh1_96 only)

**The (small, negative) Stage1 change DID transfer directionally to Stage2** -- S1/S2's slightly worse Stage1 Hard AggMSE corresponds to slightly worse Stage2 MSE, in the same rank order. This is a case where Stage1 and Stage2 AGREE (both say L0 > L3a > L3b on this cell), which is itself notable given this project's repeated prior finding (TRACK-M) that Stage1 and Stage2 can disagree -- here they do not, but the direction is "the aggregate loss does not help," not "the aggregate loss helps and it transfers."

## ETTh1_720 Stage2 Test MSE

| Cell | Base | S0 (L0) | S1 (L3a) | S2 (L3b) |
|---|---:|---:|---:|---:|
| ETTh1_720 | 0.56035 | 0.514917 | 0.517572 | **0.514618** |

S0 reproduces canonical V5 Stage2 (0.51492) essentially exactly (0.514917). **S2's small Stage1 win over S0 (Phase 1) DOES transfer to Stage2** -- S2's test MSE (0.514618) is marginally better than S0's (0.514917), the first case in this track where an aggregate-aligned loss beats plain L0 end-to-end. S1 remains the worst of the three, same as Stage1.

## Weather_96 Stage2 Test MSE

| Cell | Base | S0 (L0) | S1 (L3a) | S2 (L3b) |
|---|---:|---:|---:|---:|
| Weather_96 | 0.19527 | 0.176649 | 0.179473 | **0.168375** |

**L3b's Stage1 win transfers to Stage2, and all 3 arms now beat Base** -- a genuine departure from this project's long-standing repeated Weather pattern (Base usually wins or ties on Weather; see PHASE 4's earlier note). S2 beats S0 by 4.7% at Stage2 (smaller than Stage1's 22.6% Hard-AggMSE gap, but still the clearest Stage2 win for an aggregate-aligned loss in this track). S1 is worst of the three at Stage2 too, same order as Stage1. Lambda values: S0=0.3953, S1=0.3837, S2=0.4309 -- all meaningfully below 0.5 on this cell (unlike ETTh1's ~0.47-0.49), consistent with Base being more competitive here.

## Weather_720 Stage2 Test MSE

**Independently re-verified Base (`scripts/verify_v5_head_loss_stage2_correctness01.py --cell Weather_720`,
not taken from any EXPERIMENT_LOG.md table at face value, per the Weather_96 column-misread precedent):
`base_mse(test) = 0.319430`**, which matches the EXPERIMENT_LOG.md citation (0.31943) exactly this time --
unlike Weather_96, there was no error in this particular citation, but it was still independently confirmed
rather than assumed.

| Cell | Base | S0 (L0) | S1 (L3a) | S2 (L3b) |
|---|---:|---:|---:|---:|
| Weather_720 | 0.319430 | 0.385526 | 0.363349 | **0.359719** |

**All 3 Shared arms are WORSE than Base on this cell** -- a departure from Weather_96 (where S2/D2 narrowly
beat Base) and consistent with this project's long-standing repeated finding that Base is hardest to beat
at the longest horizon. L3b's large Stage1 win (17.8% Hard AggMSE) attenuates sharply at Stage2 but is still
the best of the three retrieval arms (S2 beats S0 by 6.7%), the same ordering as every other cell.

**(All 4 primary cells now complete -- see FINAL_RESEARCH_REPORT.md for the full cross-cell factorial.)**
