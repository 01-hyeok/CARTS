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

**(Remaining 3 cells pending -- see FINAL_RESEARCH_REPORT.md.)**
