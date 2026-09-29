# TRACK-M-RELEVANCE-CONSTRAINED-MULTISLOT01

Direct follow-up to TRACK-J3 (error complementarity), TRACK-K
(multi-slot predictive retrieval, NO-GO), TRACK-L (evaluation alignment).
K2's diagnosis: genuine complementarity gain (lower cross-term `C`) but at
the cost of individual relevance (`D` worse than J1 by +11.48%, exceeding
the pre-registered 10% budget). This track adds a relevance-protection
loss term to K2 -- and only that -- to test whether the complementarity
gain can be kept while individual relevance is restored toward J1's
level.

Full spec, architecture, and PART-by-PART requirements: see the
conversation record and `research/M-relevance-constrained-multislot/AUDIT.md`
(code audit of the K2/L commits this track extends).

## Stage 1

### Setup

- K2 (parent baseline) reused UNCHANGED (not retrained); its saved
  checkpoint/results are the ones in `results/TRACK-L-EVAL-ALIGNMENT-FULLTEST01/`.
- J1's train-split individual-relevance reference `T_J1(q,c)` precomputed
  once (`scripts/precompute_m_j1_reference01.py`), train-only (7201
  queries x 7 channels = 50407 rows, dense, verified no val/test
  leakage), from J1's frozen checkpoint
  (`checkpoints/track_j2_key_update_decomposition01/ETTh1_720/J1_stopgrad_key/checkpoint.pth`).
  J1's full-validation baseline (needed for PART 8's constrained
  checkpoint selection) computed via TRACK-L's own unified evaluator:
  `R_val_J1 = 2.049492`, `A_val_J1 = 1.603132`.
- M1 (`soft_relevance`, control arm): `L = L_K2 + gamma * R_set`,
  `gamma = 1.0`.
- M2 (`relevance_budget`, primary arm): `L = L_K2 + gamma *
  ReLU(R_set - 1.05*T_J1(q,c))`, `gamma = 1.0`, `delta = 0.05`. No
  coefficient sweep (both fixed, PART 6).
- Both trained with `init_seed=loader_seed=0`, identical to K2's pairing
  convention; per-epoch batch-order hashes saved
  (`{M1,M2}_*/batch_order_hashes.csv`).
- Checkpoint selection (PART 8, new rule): among epochs with
  `val_retmse10 <= 1.05 * R_val_J1`, the minimum-`val_agg` epoch is kept.
  Both M1 and M2 selected **epoch 1** (feasible at every epoch observed;
  epoch 1 had the lowest val AggMSE in both runs -- consistent with
  J0/J1/K1/K2 all also selecting `best_epoch=1` in this setup).
- 18/18 required unit tests pass (`tests/test_m_relevance_constrained_multislot01.py`);
  full repo suite: 1214 passed, the same 2 pre-existing unrelated
  failures, no regression.

### Stage 1 comparison table (FULL2161, TRACK-L unified evaluator; J0/J1/K2 reused verbatim from TRACK-L)

| Arm | retMSE@10 | D | C | AggMSE | Recall@10 | retMSE % vs J1 | Agg % vs J1 | K2-gain retention % |
|---|---|---|---|---|---|---|---|---|
| J0 | 0.955397 | 0.095540 | 0.548835 | 0.644375 | 0.022377 | -4.98% | +14.25% | -- |
| J1 | 1.005504 | 0.100550 | 0.463476 | 0.564026 | 0.021855 | 0.00% | 0.00% | -- |
| K2 | 1.120915 | 0.112091 | 0.434852 | 0.546943 | 0.027461 | +11.48% | -3.03% | 100.0% |
| **M1** | 1.020802 | 0.102080 | 0.431197 | 0.533277 | 0.028181 | +1.52% | -5.45% | **180.0%** |
| **M2** | 0.988612 | 0.098861 | 0.424731 | 0.523593 | 0.028479 | **-1.68%** | -7.17% | **236.7%** |

`K2-gain retention % = (Agg_J1 - Agg_new)/(Agg_J1 - Agg_K2) x 100`. Both
M1 and M2 don't just retain K2's aggregate gain -- they exceed it by a
wide margin, while ALSO beating K2 on individual relevance (`D`/`retMSE`)
and the cross term (`C`) simultaneously. M2's test-set retMSE
(0.988612) is even slightly better than J1's own (1.005504).

### Verdict tiers

Thresholds (from TRACK-L's FULL2161 J1/K2 numbers, PART 10):
`retMSE <= 1.05xJ1 = 1.055779`; `Agg <= J1 - 0.5x(Agg_J1-Agg_K2) =
0.555485` (50%-retention GO line); `EXCELLENT: retMSE <= 1.03xJ1 =
1.035669` and `Agg <= ~0.5512`.

| Arm | retMSE<=1.055779 | Agg<=0.555485 | C<C_J1 | retMSE<=1.035669 | Agg<=0.5512 | **Test-based tier** |
|---|---|---|---|---|---|---|
| M1 | Y (1.020802) | Y (0.533277) | Y (0.431197<0.463476) | Y | Y | **STRONG_GO** |
| M2 | Y (0.988612) | Y (0.523593) | Y (0.424731<0.463476) | Y | Y | **STRONG_GO** |

Per PART 13/29, the Stage2 trigger and M* selection must be
**validation-based**, never test-based -- so the same tier logic was
independently re-run on the analogous validation-split quantities
(`R_val_J1=2.049492`, `A_val_J1=1.603132`, and K2's own validation
baseline computed the same way: `R_val_K2=2.200418`, `A_val_K2=1.559554`):

| Arm | val retMSE | val Agg | val C | feasible (<=2.151966) | Agg<=1.581343 (50% val-retention) | C<C_J1_val (1.398182) | **Validation-based tier** |
|---|---|---|---|---|---|---|---|
| M1 | 2.109218 | 1.554727 | 1.343805 | Y | Y | Y | **STRONG_GO** |
| M2 | 2.080521 | 1.541034 | 1.332982 | Y | Y | Y | **STRONG_GO** |

Both splits agree: **STRONG_GO for both M1 and M2**, independently.

### Q1. Does M1's penalty restore D?

Yes, almost completely. M1's `D=0.102080` is only 1.5% worse than J1's
`D=0.100550` (vs K2's `D=0.112091`, +11.5% worse than J1) -- the
undifferentiated set-average penalty alone recovers nearly all of the
individual-relevance loss K2 incurred.

### Q2. How much C is lost doing so?

**None.** This is the central surprise of this track. The pre-registered
expectation (spec PART 12, scenario a) was that M1's naive penalty would
restore D at the cost of destroying the complementarity gain (`C` rising
back toward J1's level). Instead M1's `C=0.431197` is *better* than
K2's own `C=0.434852` -- M1 Pareto-dominates K2 on every axis
simultaneously (`D`, `C`, and `AggMSE` all improve).

### Q3. Is M2's Pareto tradeoff better than M1's?

Yes, on every axis, but by a small margin: `retMSE` 0.988612 vs
1.020802, `C` 0.424731 vs 0.431197, `AggMSE` 0.523593 vs 0.533277. M2
Pareto-dominates M1. However, because M1 (the control arm) already
succeeds strongly, M2's advantage over M1 is incremental rather than
qualitative -- both mechanisms work, M2's budget-gated hinge is simply
somewhat more effective than the unconditional penalty.

### Q4. Is M2's final retMSE within +5% of J1?

Yes, and better than parity: M2's retMSE is 1.68% *lower* than J1's
(0.988612 vs 1.005504), not merely within the +5% budget.

### Q5. Does M2 retain >=50% of K2's aggregate gain?

Yes, 236.7% -- M2's aggregate improvement over J1 is more than double
K2's own.

### Q6. Is there a Strong GO among M1/M2?

Yes -- **both** M1 and M2 independently achieve STRONG_GO, confirmed on
both the validation-based assessment (which governs the Stage2 trigger)
and the test-based table.

### Q7. Does it pass the Stage2 entry gate?

Yes. Per PART 28's pseudocode, `any(arm.verdict in [STRONG_GO,
EXCELLENT_GO])` is true (both qualify), so Stage2 is triggered. Per PART
13's tie-break rule (lower validation AggMSE, then lower validation
retMSE), **M\* = M2** (`val_agg=1.541034` < M1's `1.554727`).

### An important caveat for interpretation

The spec's mechanistic hypothesis (PART 5's stated rationale) was that
constraining relevance **per-slot** would destroy complementarity by
forcing every slot toward the same "safe" candidates, and that only
constraining the **set average** (`R_set`) would preserve it. Neither M1
nor M2 actually implements a per-slot constraint -- both operate on the
same set-average `R_set` (M1 unconditionally, M2 only above a budget).
So this experiment did not directly test the per-slot-vs-set-average
contrast; it tested unconditional-vs-budget-gated set-average pressure,
and found both sufficient. The finding that M1 (the presumed-to-fail
control) also fully succeeds is worth flagging to the reviewer as an
open question: it suggests K2's individual-relevance sacrifice may not
have been an unavoidable trade-off at all, but simply an artifact of K2
never being given ANY relevance-restoring signal -- consistent with, but
not proof of, the idea that the aggregate/complementarity objective and
individual relevance are not fundamentally opposed at this coefficient
scale (`gamma=1.0`, `lambda_agg=1.0`, `beta=0.05`). A dedicated
per-slot-constraint ablation would be needed to test the original
mechanistic claim directly -- out of scope for this run (PART 6 forbids
new arms/sweeps this round).

**Stage 1 verdict: STRONG_GO (both arms) -> Stage2 triggered, M\* = M2.**

## Stage 2

Triggered per PART 28's pseudocode (Stage1 STRONG_GO on both M1 and M2,
validation-based). M\* = M2 (lower val AggMSE, PART 13 tie-break).

### Implementation reused, not redesigned (PART 14)

Audited the repo for the most recently validated Stage2 implementation:
`scripts/train_setlossctrl_stage2_retrain02.py`
(`e7fd86cce19dd9c05716bbbaf0f0f0c83615f9a3`, 2026-09-18) -- the generic,
CLI-driven, cache-based Stage2 trainer already used and fully validated
for TRACK-A-SET-LOSS-CONTROL02 (results + report exist:
`research/B-set-oracle/TRACK-A-SET-LOSS-CONTROL02.md`). Reused
UNMODIFIED. All three PART 14 past-issue classes are already guarded in
this lineage and re-verified here: (1) wrong/legacy cache -- the loader
hard-aborts on any `cache_schema_version` other than
`corrected_delta_v1`; (2) encoder unintentionally trainable --
`FREEZE_SUBMODULES` frozen before training and SHA-reverified
byte-identical after (never mismatched in any of the 4 runs); (3) old
WCE checkpoint / unauthorized seed -- `--seed 0` used throughout, fresh
per-arm output paths, no checkpoint reuse across arms.

Base forecaster head + gate/mixer are freshly initialized (shared across
arms via `--shared_init_out`/`--shared_init_in`, SHA-verified equal for
all 4 arms) and jointly trained; the retrieval-side submodules
(`stage1_encoder`, `shared_cross_projection`, `retrieval_metric`,
`pairwise_scorer`, `query_cond_proj`, `candidate_cond_proj`) are frozen.
J1/K2/M2's own retrievers never appear in the Stage2 model at all --
retrieval is entirely precomputed into a per-arm cache
(`scripts/build_m_stage2_retrieval_cache01.py`, sibling of
`build_setlossctrl_retrieval_cache02.py`) using each retriever's own
frozen checkpoint and future-blind Top-10 selection (J1: `arm_score` +
`stable_topk_indices`; K2/M2: `compute_scores` + `hard_unique_selection`,
unmodified from Stage1), alpha-weighted into a single aggregate by ONE
shared, fixed `HostScorer` (the Stage2 host's own frozen score) --
PART 17: only the retrieval SOURCE differs between arms, the aggregation
function is identical. S0 (base) uses an all-zero retrieval cache (its
own dedicated training run, not a reuse of another arm's
`counterfactual_lambda0_mse`) -- verified zero by unit test (item 9).
10/10 required Stage2 unit tests pass
(`tests/test_m_stage2_relevance_constrained_multislot01.py`).

### Stage2 comparison table (test split, best-val-selected checkpoint per arm)

| Arm | best_epoch | final_mse | final_mae | base_mse | retrieval-branch-only MSE |
|---|---|---|---|---|---|
| S0 (base) | 6 | 0.488790 | 0.486478 | 0.488790 | 1.335121 |
| S1 (J1) | 8 | **0.475650** | 0.484477 | 0.475768 | 0.578322 |
| S2 (K2) | 3 | 0.481361 | 0.487947 | 0.476692 | 0.573898 |
| S3 (M\*=M2) | 6 | 0.485463 | 0.484168 | 0.485434 | **0.562447** |

S1 (J1) has the best final forecast MSE of all four arms. S3 (M2) is
barely better than the no-retrieval baseline S0, despite M2's retrieval
branch being the MOST accurate standalone one of the three (lowest
retrieval-branch-only MSE, 0.562447).

### Gate / lambda distribution (test split, selected checkpoint)

| Arm | gate_mean | gate_median | gate<0.02 | gate>0.10 | gate>0.50 |
|---|---|---|---|---|---|
| S0 | 0.475 (structural: constant zero-input bias, not real retrieval use) | 0.477 | 0% | 100% | 8.8% |
| S1 (J1) | 0.0165 | 0.0002 | 82.4% | 4.5% | 0.06% |
| S2 (K2) | 0.1102 | 0.0738 | 22.3% | 41.0% | 0.7% |
| S3 (M\*) | **0.0044** | 0.0000 | **95.5%** | 1.2% | 0.01% |

The gate engages K2's retrieval most (lowest `gate<0.02` fraction,
highest `gate>0.10` fraction), J1's retrieval moderately, and **M2's
retrieval least of all three real-retrieval arms** -- more suppressed
even than J1's.

### Paired bootstrap (test split, query_start_idx unit, 10,000 replicates)

| Comparison | mean(MSE_a - MSE_b) | 95% CI | excludes 0 |
|---|---|---|---|
| S1 - S0 | -0.013139 | [-0.016155, -0.010208] | Yes (negative) |
| S2 - S0 | -0.007429 | [-0.009759, -0.005162] | Yes (negative) |
| S3 - S0 | -0.003326 | [-0.003821, -0.002840] | Yes (negative) |
| **S3 - S1 (primary)** | **+0.009813** | **[0.006701, 0.013007]** | **Yes (positive)** |
| S3 - S2 | +0.004102 | [0.001701, 0.006525] | Yes (positive) |

All four retrieval-bearing arms beat the no-retrieval baseline S0
significantly. But S3 (M\*) is significantly **worse** than both S1 (J1)
and S2 (K2) -- not merely non-significantly different.

### Stage2 verdict: **NO-GO**

Per PART 21: Stage2 GO requires `MSE_S3 < MSE_S1` with a favorable CI (or
a smaller but real >=0.5% improvement); NO-GO applies when `S3 >= S1`.
Here `S3 - S1 = +0.009813` with a 95% CI entirely above zero -- S3 is
significantly worse than S1, not merely equivalent. STRONG Stage2 GO
(beating S0, S1, AND S2 simultaneously) is not met either (S3 beats only
S0).

### Gate-interpretation case (PART 22): **Case C**

M\* (M2) had the *best* Stage1 retrieval-quality metrics of every arm
(lowest D, lowest C, lowest AggMSE) and even the lowest standalone
Stage2-cache retrieval-branch MSE of the three real-retrieval arms --
yet Stage2 forecasting with M2 is worse than with either J1 or K2, and
the trained gate learns to suppress M2's retrieval more than any other
arm's (95.5% of test queries at gate<0.02). This is PART 22's Case C:
*the Stage1 aggregate-MSE surrogate, even after a genuine, validated
improvement, remains misaligned with what the downstream gated fusion
model finds useful for actually reducing forecast error.*

An open, unresolved question this raises (not something Stage2 was
designed to isolate, flagged for the reviewer): M2's retrieval branch is
the most accurate *on its own*, but the gate seems to value K2's
retrieval more. One plausible reading is that gate utility depends on
how a retrieval branch's errors *correlate with the base forecaster's
own errors*, not on the retrieval branch's standalone accuracy -- the
same kind of complementarity structure TRACK-J3 found between candidates
within a single retrieval set, potentially recurring one level up,
between the retrieval branch and the base branch. This experiment did
not measure that correlation directly; it would need a dedicated
follow-up.

### Final answer to the two-part scientific question (PART 29)

**Can we preserve candidate-level predictive relevance while explicitly
learning complementary retrieval sets?** Yes -- Stage1's answer is
unambiguous: M2 (and even the naive M1 control) achieves near-J1
individual relevance while exceeding K2's own complementarity gain,
Pareto-dominating K2 on every axis.

**Does that retrieval Pareto improvement actually improve forecasting?**
No. Stage2's answer is a clean, honest NO-GO: the Stage1 Pareto gain does
not transfer to downstream forecast accuracy, and in fact M2's retrieval
is the LEAST used by the trained gate of any real-retrieval arm tested.
The hoped-for target pattern (`D_M2≈D_J1`, `C_M2<C_J1`, `Agg_M2<Agg_J1`
-> better forecasting) was realized only in its first half.
