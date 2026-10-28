# PHASE C -- Head Error Complementarity Analysis (existing checkpoints only, no new training)

Script: `scripts/analyze_v5_topk_head_complementarity01.py` (new, read-only inference on the already-trained
S0/S1/S2/D0/D1/D2 checkpoints from `TRACK-V5-HEAD-LOSS-DUAL-ENCODER01` -- per this track's naming convention
these ARE the spec's "S2-v1"/"D2-v1"; no v2 Top-K implementation has been trained yet, that is Phase D/E).
Raw data: `results/TRACK-V5-TOPK-CORRECTION-MECHANISM01/phaseC/head_complementarity_results.json`, all 3
completed cells (ETTh1_96, ETTh1_720, Weather_96; Weather_720's Stage1 checkpoints exist as of this writing
but were not yet included in this pass -- see note at the end).

**Correctness cross-check**: `mixmix_top10_D`/`mixmix_top10_C`/`mixmix_top10_AggMSE` computed by this script
match the pre-existing `TRACK-V5-HEAD-LOSS-DUAL-ENCODER01` Phase 1/3 reports' D/C/Hard-AggMSE columns
EXACTLY for every one of the 18 (cell, arm) combinations (e.g. ETTh1_96 S0: D=0.06639/C=0.34342 here vs
0.066386/0.343422 in `PHASE_1_SHARED_LOSS_REPORT.md`; Weather_96 S2: D=0.03305/C=0.1716 here vs the
hard_agg=0.204646 reported there, 0.03305+0.1716=0.20465, matching). This independently confirms the new
script's forward pass and candidate pooling are correct before trusting any new number it reports.

**Zero-norm cosine rule** (per spec): where either head's error norm is 0, cosine is defined as exactly
`0.0`, not `+1`/`-1` -- encoded directly in `pairwise_cosine_zero_rule()`. No zero-norm case was actually
observed in any of the 18 runs (every head's error norm was bounded well away from 0), so this rule did not
end up affecting any reported number, but is in place and tested in principle.

**Cosine bug avoidance** (per spec, learned from `TRACK-V5-HEAD-OBJECTIVE-COSINE-FIX01`): `cosine_mn` is
computed from the SAME un-divided `raw_dot_mn = <e_m, e_n>` used for `head_error_dot_matrix`, divided by
`norm_m * norm_n` -- never from an already-normalized quantity. `raw_dot`, `norm`, and `cosine` are all
reported so the dot-into-norm*cosine decomposition in C4 Q5/Q6 below is literal, not inferred.

## Summary table (off-diagonal means; `mix_D`/`mix_C` = Mean-Mixture Top-10's own decomposition)

| Cell | Arm | Head-pair cosine | Head-pair overlap | Neg-cosine frac | mix_D | mix_C | mix_AggMSE | Within-set-10 cosine |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ETTh1_96 | S0 | 0.8435 | 0.3754 | 0.0166 | 0.06639 | 0.34342 | 0.40981 | 0.5219 |
| ETTh1_96 | S2 | 0.8483 | 0.3938 | 0.0157 | 0.06726 | 0.34483 | 0.41209 | 0.5123 |
| ETTh1_96 | D0 | 0.8986 | 0.5244 | 0.0075 | 0.06860 | 0.36811 | 0.43670 | 0.5368 |
| ETTh1_96 | D2 | 0.9095 | 0.6116 | 0.0078 | 0.06936 | 0.36245 | 0.43181 | 0.5196 |
| ETTh1_720 | S0 | 0.9517 | 0.7315 | 0.0007 | 0.09426 | 0.52631 | 0.62057 | 0.5910 |
| ETTh1_720 | S2 | 0.9192 | 0.6308 | 0.0035 | 0.09481 | 0.52014 | 0.61496 | 0.5824 |
| ETTh1_720 | D0 | 0.9598 | 0.7579 | 0.0003 | 0.09185 | 0.51091 | 0.60276 | 0.5842 |
| ETTh1_720 | D2 | 0.9464 | 0.7342 | 0.0014 | 0.09442 | 0.51181 | 0.60624 | 0.5741 |
| Weather_96 | S0 | 0.6497 | 0.2526 | 0.0969 | 0.04556 | 0.21877 | 0.26433 | 0.5686 |
| **Weather_96** | **S2** | 0.6257 | 0.2967 | 0.1175 | **0.03305** | **0.1716** | **0.20465** | 0.5557 |
| Weather_96 | D0 | 0.6498 | 0.2538 | 0.0972 | 0.05522 | 0.25024 | 0.30546 | 0.5651 |
| **Weather_96** | **D2** | **0.7180** | **0.5403** | 0.0918 | **0.03342** | **0.17347** | **0.20689** | 0.5798 |

(S1/D1, the L3a arms, are in the raw JSON but omitted from this summary table since every C4 question below
is specifically about S2/D2 vs S0/D0; their numbers follow the same pattern already established in Phase
1/3 -- L3a never beats L0 on any of the 3 cells.)

## C4 Q1: Does S2(-v1) reduce head error cosine vs S0?

**No, not meaningfully, on any cell.** ETTh1_96: 0.8435 -> 0.8483 (slightly UP). ETTh1_720: 0.9517 -> 0.9192
(down, but ETTh1_720's overall numbers are noisy given the much larger prediction norms at this horizon --
see Q3). Weather_96: 0.6497 -> 0.6257, a real but small (-3.7% relative) decrease. In no case is this a
large or clearly causal effect.

## C4 Q2: Does D2(-v1) reduce head error cosine vs D0?

**No -- it increases on every cell**, most dramatically on Weather_96 (0.6498 -> 0.7180, +10.5% relative,
the single largest cosine movement in this entire table) and also on ETTh1_96 (0.8986 -> 0.9095) and
ETTh1_720 (0.9598 -> 0.9464, this one down slightly). **This directly contradicts the "L3b wins because it
reduces head redundancy" hypothesis for the Separate-encoder arm that happens to win most often (D2 on
Weather_96/Weather_720) -- D2's heads are MORE redundant with each other than D0's, not less.**

## C4 Q3: Does Top-10 overlap decrease alongside (or instead of) cosine?

**No -- overlap moves in the same direction as cosine in every case, and often more dramatically.**
Weather_96 D2 vs D0: overlap 0.2538 -> 0.5403, more than doubling (heads agree on over half their Top-10
picks, vs about a quarter before). ETTh1_96 D2 vs D0: 0.5244 -> 0.6116. Both head-level redundancy measures
(error-direction cosine and literal Top-10 index overlap) tell the SAME story, and it is the opposite of
what a "complementarity improved" narrative would predict for D2 specifically.

## C4 Q4: Does any head-cosine change connect to the real Mean-Mixture Top-10 `C`?

**Weakly, and in the wrong direction for D2.** Restricting to Weather_96, where the `C` effect is real and
large: S0 `C`=0.21877 -> S2 `C`=0.1716 (-21.6%), D0 `C`=0.25024 -> D2 `C`=0.17347
(-30.7%). Both are large C reductions. But Q1/Q2 showed head-pair cosine barely moved for S2 (-3.7%) and
*increased* for D2 (+10.5%). **The within-set-10 candidate-level error cosine (the quantity that actually,
mechanically generates `C` -- see the exact formula in Phase B3's planned direct-pairwise check) is nearly
flat across every arm on Weather_96** (S0 0.5686 vs S2 0.5557, a -2.3% move; D0 0.5651 vs D2 0.5798, a
*+2.6%* move -- i.e. the within-set-10 cosine for D2 goes up slightly too, consistent with the head-level
story, not with a "more cancellation" story). **The head-level cosine/overlap metrics and the `C` term are
NOT causally connected here** -- `C`'s large drop is explained almost entirely by reduced error MAGNITUDE
(see Q5/Q6), not by any change in error DIRECTION/cancellation.

## C4 Q5/Q6: Is the AggMSE improvement from individual accuracy or error cancellation? Did cancellation increase?

**Individual accuracy (`D`), not cancellation (`C`-via-direction).** The `raw_dot = norm * cosine`
decomposition makes this exact, not inferred:

- Weather_96 S2 vs S0: `mix_D` falls 0.04556 -> 0.03305 (-27.5%), while within-set-10 error cosine barely
  moves (0.5686 -> 0.5557, -2.3%). The per-head error NORM values collapse for most heads (S0:
  [3.36, 3.93, 3.07, 3.14, 3.20] -> S2: [3.08, 3.03, **3.81**, 3.32, **4.24**] -- 3 of 5 heads get smaller
  norms, 2 get larger, net effect still favors S2 on `D` because of how Mean-Mixture weights candidates).
- Weather_96 D2 vs D0 shows the most extreme individual-head pattern in the whole dataset: head error norms
  **[3.03, 3.03, 5.33, 3.03, 3.09]** for D2 vs D0's **[3.07, 3.99, 3.61, 3.21, 3.05]** -- head RetMSE for D2
  is **[0.325, 0.326, 7.741, 0.325, 0.342]**: four of five heads become extremely, almost identically
  accurate (RetMSE ~0.33, vs D0's more varied 0.41-1.10), while ONE head (index 2) becomes a severe outlier
  at RetMSE=7.74 (worse than any D0 head). **This is specialization/near-collapse in 4 of 5 heads toward a
  similarly good solution, not a reduction in error correlation between heads** -- and it is EXACTLY why
  head-pair cosine/overlap goes UP for D2 (4 near-identical heads are, almost by construction, highly
  correlated with each other), while the Mean-Mixture's own Top-10 `D` still improves, because those 4
  heads are simply much better individually, and the 1 bad head's damage is diluted across the mixture's
  softmax averaging before any Top-10 selection happens.
- **Conclusion: within-set error cancellation (the real C-generating mechanism) does NOT increase under
  L3b on Weather_96 -- in fact it is flat to slightly worse. The entire improvement traces to 4-of-5 heads
  becoming more individually accurate (lower error norm / lower RetMSE), consistent with this being an
  individual-relevance effect of the Top-K-aligned auxiliary loss, not an error-complementarity effect.**
  This matches the user's own observation going into this analysis (Weather_96 S2's much lower `mix_D`) and
  extends it with the mechanism: norm reduction concentrated in most heads, not cosine/cancellation
  improvement, and a visible one-head outlier cost that the Mean-Mixture happens to absorb without harm to
  the final Top-10 (since that outlier head's candidates are simply outcompeted in the softmax average by
  the other 4 heads' far better scores).

## Negative-cosine fraction (how often do heads actually disagree in error direction?)

Weather_96 (~10-12% of head pairs have negative cosine) is meaningfully higher than ETTh1_96 (~1.5-1.7%) and
ETTh1_720 (~0.1-0.3%) across every arm -- Weather's heads disagree far more often than ETTh1's regardless of
loss, a dataset property, not a loss-induced one (the fraction barely moves between S0/S2 or D0/D2 within
either dataset).

## Horizon pattern (ETTh1_96 vs ETTh1_720)

Head-pair cosine/overlap are already very high at baseline on ETTh1 and get HIGHER with horizon (ETTh1_96
D0: 0.90/0.52 -> ETTh1_720 D0: 0.96/0.76) -- heads converge toward near-identical behavior as the horizon
grows, for every loss. This is consistent with ETTh1_720's much smaller Loss-effect sizes throughout this
track (there is simply less room for any loss variant to differentiate heads that are already this
redundant).

## Limitations / what this Phase does NOT establish

- This is correlational, not causal: no intervention (e.g. explicit per-head decorrelation loss) was tried
  here to confirm that reducing head cosine would actually improve `C`. Phase E/G (if Phase D's v2 Top-K fix
  changes this picture) should re-examine.
- Weather_720 is not included in this pass (checkpoints exist as of this writing but were deliberately left
  for a follow-up run to avoid CPU contention with the still-finishing `TRACK-V5-HEAD-LOSS-DUAL-ENCODER01`
  Weather_720 training inside the same session).
- Head RetMSE/error-norm outlier behavior (the one catastrophic head under D2) was observed but not
  separately diagnosed (e.g. is it the same head index across different seeds/runs, or arbitrary from
  initialization randomness/training dynamics?) -- a single-seed observation, not characterized further.
