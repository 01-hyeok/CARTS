# PHASE 4 -- Separate Encoder Stage2 Forecasting (D0/D1/D2)

**Status: ETTh1_96 complete; other 3 cells pending.** Same Base Forecaster checkpoint as the Shared arms (Phase 2), same Stage2 script, independent cache per arm.

## ETTh1_96 full 6-arm comparison (test split, Stage2 MSE -- the single most important number per spec section 16)

| Cell | Base | S0 | S1 | S2 | D0 | D1 | D2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ETTh1_96 | 0.39242 | **0.369597** | 0.370652 | 0.371837 | 0.376534 | 0.373972 | 0.375010 |

| Arm | Lambda | Best epoch |
|---|---:|---:|
| S0 | 0.4733 | 9 |
| S1 | 0.4770 | 10 |
| S2 | 0.4949 | 3 |
| D0 | 0.4341 | 10 |
| D1 | 0.4460 | 9 |
| D2 | 0.4278 | 9 |

**All 6 arms beat Base on this cell.** Full ranking (best to worst): **S0 < S1 < S2 < D1 < D2 < D0.**

## RQ3/Stage2-transfer answer for the Separate arms, this cell

Phase 3's Stage1 finding (D1 beats D0 on Hard AggMSE via a C-reduction mechanism, even with slightly worse D) **DOES transfer to Stage2 on this cell**: D1's Stage2 MSE (0.373972) is better than D0's (0.376534), the same rank order as Stage1. D2 sits between D0 and D1 at Stage2 (0.375010), also matching its Stage1 position. **This is a case where the aggregate-loss's Stage1 mechanism (C-reduction, visible specifically within the Separate structure) DOES carry through to Stage2** -- a positive transfer result, though a modest one in absolute terms (D1 is still worse than every Shared arm, including the Shared arm with the SAME loss, S1) and confirmed on only one cell/one seed so far.

**No cell shows Base beating any retrieval arm yet** (unlike this project's repeated Weather pattern in prior tracks) -- but Weather cells have not been run yet in this track; this ETTh1-only result should NOT be extrapolated to Weather, consistent with every prior P100 track's finding that ETTh1 and Weather diverge.

**(Remaining 3 cells pending -- see FINAL_RESEARCH_REPORT.md's status section for the concrete remaining-work list.)**
