# PHASE 3 -- Separate Query/Key Encoder Loss Comparison (D0/D1/D2)

**Status: ETTh1_96 complete; other 3 cells pending.** Separate encoders initialized identically (`theta_q^(0)=theta_k^(0)`, deepcopy, confirmed by matching `state_hash`), both trained with full, undetached gradient every step (no frozen/stop-grad key -- confirmed distinct from the TRACK-J2 J2 arm by construction and by this track's own smoke test asserting the two encoders' post-step weights diverge).

## ETTh1_96 (test split, primary checkpoint = min val RetMSE@10)

| Arm | Loss | RetMSE@10 | Hard AggMSE@10 | D | C | Soft AggMSE | Best epoch |
|---|---|---:|---:|---:|---:|---:|---:|
| D0 | L0 | 0.685951 | 0.436703 | 0.068595 | 0.368108 | 0.413498 | 10 |
| D1 | L3a | 0.686945 | 0.431208 | 0.068695 | 0.362513 | 0.413269 | 8 |
| D2 | L3b | 0.693619 | 0.431813 | 0.069362 | 0.362451 | 0.416539 | 7 |

## 1. D0 vs S0 -- Encoder separation's basic effect (ETTh1_96)

**Separation hurts, clearly.** D0's RetMSE@10 (0.685951) is 3.3% worse than S0's (0.663855); Hard AggMSE@10 (0.436703) is 6.6% worse than S0's (0.409808) -- a larger relative degradation on the aggregate metric than on the individual one. This is consistent with (not proof of) a genuine capacity/role-separation-confound effect -- Separate has roughly 2x the encoder parameters of Shared, so this alone cannot isolate "role separation helps/hurts" from "more parameters with no shared-regularization helps/hurts" (spec's own required caveat; a capacity-matched control was explicitly NOT added this round, per spec section 8's instruction not to introduce one in the primary experiment).

## 2/3. D1 vs S1, D2 vs S2 -- Separation effect when an Aggregate Loss is present

Separation still hurts under both aggregate losses (D1's 0.431208 Hard AggMSE is worse than S1's 0.411558; D2's 0.431813 worse than S2's 0.412088) -- the direction of the Shared-vs-Separate effect does NOT flip when an aggregate-aligned loss is added, on this cell.

## 4/5. D1 vs D0, D2 vs D0 -- Aggregate Loss effect WITHIN the Separate structure

**A genuinely different pattern from the Shared case, and the most interesting single finding in this cell's data.** Within Shared, `C` barely moves (0.3434 -> 0.3446 -> 0.3448) while `D` rises monotonically as the aggregate term is added -- the aggregate loss does essentially nothing useful there. Within SEPARATE, `C` drops measurably from D0 (0.368108) to D1 (0.362513) and D2 (0.362451) -- a real ~1.5% relative reduction in the cross-interaction term -- while `D` is nearly flat (0.068595 -> 0.068695 -> 0.069362, +0.1% to +1.1%, much smaller relative moves than Shared's D-increase). **D1 beats D0 on Hard AggMSE (0.431208 < 0.436703) even though D1's own RetMSE is very slightly worse than D0's** -- the exact "D worse, Agg better via C" pattern TRACK-J3 established as a real, not merely hypothetical, mechanism, reproducing here for the FIRST time in combination with an explicit aggregate-aligned loss (rather than TRACK-J3's own stop-grad-key mechanism).

## 6. D2 vs D1 -- Top-K alignment's marginal effect within Separate

D2 is slightly worse than D1 on every metric (RetMSE 0.693619 vs 0.686945; Hard AggMSE 0.431813 vs 0.431208 -- nearly tied; Soft AggMSE 0.416539 vs 0.413269, D2 clearly worse). On this cell, the extra Top-K-alignment relaxation of L3b does not improve over L3a's simpler full-pool soft aggregate -- if anything it is marginally worse across the board within the Separate structure, mirroring the Shared-structure finding (S2 also slightly worse than S1).

## RQ4/RQ5 answer (ETTh1_96 only)

**RQ4**: Yes, Shared beats Separate under EVERY loss on this cell -- a consistent, non-interaction-dependent main effect at this cell (seed0).

**RQ5**: Yes, there IS an interaction, and it runs in the DIRECTION the spec's own framing hoped to detect: the Aggregate-aligned loss's mechanism of action (reducing `C`, per TRACK-J3's own interpretation framework) is VISIBLE within the Separate structure (D1/D2 vs D0) but nearly absent within the Shared structure (S1/S2 vs S0, where `D` rises instead and `C` barely moves). This does not mean Separate+Aggregate-Loss is the best arm overall on this cell (D1's Hard AggMSE, 0.431208, is still clearly worse than every Shared arm including the weakest one, S2 at 0.412088) -- but it is genuine evidence that the aggregate loss's intended mechanism engages DIFFERENTLY depending on encoder structure, which Phase 4's Stage2 check and the cross-cell replication (pending) will determine whether this is a reproducible pattern or a single-cell/single-seed artifact.

## Parameter count / capacity record (section 8 requirement)

Separate (`D0/D1/D2`) has exactly 2x the `model.encoder` submodule parameter count of Shared (`S0/S1/S2`) -- both built from the identical base config (`d_model=128, d_ff=256`, confirmed via checkpoint read), since `model_k` is a full `deepcopy` of `model_q`'s architecture. No capacity-matched control was run this round (explicitly out of scope per spec section 8's own instruction).

**(Remaining 3 cells pending -- see FINAL_RESEARCH_REPORT.md.)**
