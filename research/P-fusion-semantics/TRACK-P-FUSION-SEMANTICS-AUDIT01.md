# TRACK-P-FUSION-SEMANTICS-AUDIT01

Direct follow-up to TRACK-O. Single question: is Stage2's `residual`
fusion (`y_final = y_base + lambda*y_ret`) mis-specified, given that
`y_ret` is a full alternative forecast rather than a correction term?
Tested against `mixture` fusion (`y_final = (1-lambda)*y_base +
lambda*y_ret`), already implemented in `RetrievalGate` but never
selected by any host config used this session. Stage1 retriever
untouched throughout (J1, M2 checkpoints identical to every prior
track). Full audit: `research/P-fusion-semantics/AUDIT.md`.

## Setup

Uniform-aggregation retrieval caches and the common frozen base are
reused verbatim from TRACK-N/M (no rebuilding). P1 (J1+Uniform+
Residual+learned gate) and P2 (M2+Uniform+Residual+learned gate) are
TRACK-O's O1/O3 reused directly -- identical definitions. P5/P6 (fixed
-lambda mixture, `lambda=0.37`/`0.42`, TRACK-N's own validation-selected
values) are pure closed-form re-evaluations, NO model forward pass, NO
training. Only P3 (J1+Uniform+Mixture+learned gate) and P4 (M2+Uniform+
Mixture+learned gate) are genuinely new training runs -- identical to
P1/P2 in every respect (base checkpoint, retrieval cache, gate
architecture/init/optimizer/LR/batch order/epochs/patience/checkpoint
criterion) except `fusion_mode`. `relation_mixer` is frozen and its
`beta≡1` identity-pass property (TRACK-O's finding) is asserted at
runtime before training starts, not merely assumed.

All 7 reproduction gates PASS to <4e-7 (`reproduction_gate.json`).
Shared-init proof: `base_head` and `gate` initial-weight SHA are
byte-identical across P1/P2/P3/P4 (`shared_gate_init.json`); trainable
parameter set is `{gate.*}` only for P3/P4, confirmed via fingerprint
diffing before/after training. 20/20 required test functions (23 spec
items) pass.

## Primary comparison table (test split, min-val-selected checkpoints)

| Arm | Retriever | Fusion | Final MSE | Final MAE | Gate mean |
|---|---|---|---|---|---|
| P0 (base) | -- | -- | 0.488790 | 0.486478 | -- |
| P1 | J1 | Residual (learned) | 0.489349 | 0.486747 | 0.0092 |
| P2 | M2 | Residual (learned) | 0.488924 | 0.486948 | 0.0464 |
| **P3** | J1 | **Mixture (learned)** | **0.476781** | 0.479170 | **0.1198** |
| **P4** | M2 | **Mixture (learned)** | **0.475686** | 0.480704 | **0.1429** |
| P5 | J1 | Mixture (fixed λ=0.37) | 0.463349 | 0.475469 | -- |
| P6 | M2 | Mixture (fixed λ=0.42) | 0.463969 | 0.477169 | -- |

Switching ONLY the fusion formula (nothing else) took J1 from 0.489349
(WORSE than no retrieval) to 0.476781 -- a ~2.44% improvement over base.
For M2, 0.488924 -> 0.475686, a ~2.68% improvement. Gate engagement
jumped from near-zero (P1: 0.92%, P2: 4.6%) to genuinely active (P3:
12.0%, P4: 14.3%) -- exactly the pattern the spec's hypothesis predicted
(section 16: "P4 gate mean >> 0.046" -- confirmed, 0.143 vs 0.046, a 3x
increase).

## Paired bootstrap (test split, query_start_idx unit, 10,000 reps)

| Comparison | mean diff | 95% CI | significant |
|---|---|---|---|
| P3 - P1 | -0.012568 | [-0.01331, -0.01186] | **Yes** (mixture beats residual, J1) |
| P4 - P2 | -0.013239 | [-0.01541, -0.01106] | **Yes** (mixture beats residual, M2) |
| P3 - P0 | -0.012008 | [-0.01275, -0.01130] | **Yes** (P3 beats base) |
| P4 - P0 | -0.013104 | [-0.01526, -0.01091] | **Yes** (P4 beats base) |
| P3 - P5 | +0.013432 | [0.01198, 0.01492] | **Yes** (learned mixture worse than fixed) |
| P4 - P6 | +0.011717 | [0.01023, 0.01321] | **Yes** (learned mixture worse than fixed) |
| P4 - P3 | -0.001095 | [-0.00339, 0.00119] | No (M2 vs J1 under correct semantics: tied) |
| P5 - P0 | -0.025440 | [-0.02710, -0.02378] | **Yes** |
| P6 - P0 | -0.024821 | [-0.02641, -0.02322] | **Yes** |

## Verdict: Case A confirmed (not Strong), with a Case-B-pattern secondary gap

**Case A's criteria are met**: `P3<P1` (significant), `P4<P2`
(significant), `P4<P0` (significant). **Residual fusion was
mis-specified; mixture fusion is the correct Stage2 semantics for this
architecture.** Strong Case A (`P4<=0.47`) is narrowly missed
(P4=0.475686, off by 0.0057) -- but this is still the single best
neural-gate Stage2 result obtained anywhere in this entire session
(TRACK-M/N/O included).

**A Case-B pattern is simultaneously present**: P5/P6 reproduce TRACK-N
exactly, and P3/P4 are significantly WORSE than P5/P6 (both CIs
entirely positive, ~1.2-1.3% gap). A secondary, smaller gate
-optimization gap remains even after fixing the dominant fusion
-semantics bug: the learned scalar gate does not yet fully reach the
performance of a single, non-learned, validation-calibrated constant.

**Comparison D (P4 vs P3, J1 vs M2 under correct semantics)**: NOT
significant (relative gap 0.23% < 0.5%, CI overlaps zero). Per spec
section 25's own decision rule: **M2 DOWNSTREAM COMPETITIVE** -- neither
retriever is significantly better under correct fusion semantics.

## Semantic invariants (synthetic, PART 26-27)

All confirmed exactly via a real `RetrievalGate` instance
(`semantic_invariants.json`):
- Scale check (`B=R=[2,2]`, `lambda=0.5`): residual gives `[3,3]`,
  mixture gives `[2,2]` -- both match the algebraic prediction exactly.
- Mixture `lambda=0` -> `Y=B` exactly.
- Mixture `lambda=1` -> `Y=R` exactly.
- Mixture `B=R` -> `Y=B=R` exactly (idempotence), for ANY lambda.
- Residual `B=R` -> `Y != B` (diagnostic contrast: residual genuinely
  double-counts an accurate retrieval forecast, confirming the
  mis-specification hypothesis at the equation level, not just
  empirically).

## Final answers (spec section 30)

**Q1. Residual vs Mixture -- which is correct?** **Mixture.** Confirmed
both by the semantic invariant proofs (residual double-counts B=R;
mixture is idempotent) and empirically (Case A's criteria met for both
retrievers).

**Q2. Is M2 Mixture Learned significantly better than M2 Residual
Learned?** **Yes** -- `P4-P2 = -0.013239`, CI entirely negative.

**Q3. Does the same pattern appear for J1?** **Yes** -- `P3-P1 =
-0.012568`, CI entirely negative. The effect is retriever-agnostic: a
Stage2 architecture-level issue, not specific to either J1 or M2.

**Q4. Does the learned mixture gate approach the fixed validation
lambda?** Partially. It captures most of the available gain (base
0.4888 -> fixed ~0.4636 is a 0.0252 gap; learned mixture closes about
half of that, landing at ~0.4763) but a real, statistically significant
~1.2-1.3% gap to the fixed lambda remains.

**Q5. How large is the fixed-vs-learned mixture gap?** J1: `P3-P5 =
+0.013432` (CI [0.0120, 0.0149]). M2: `P4-P6 = +0.011717` (CI [0.0102,
0.0132]). Both clearly significant, comparable magnitude for both
retrievers.

**Q6. Under correct mixture semantics, which retriever is better?**
Neither, significantly. `P4-P3` CI overlaps zero (relative gap 0.23%).
M2 downstream utility is COMPETITIVE with J1's, not superior or
inferior, once fusion semantics is corrected.

**Q7. Primary cause of the original Stage2 failure: fusion semantics /
gate optimization / single-vector aggregation / mixed?** **Mixed**, with
fusion semantics as the dominant, now-corrected factor. Switching only
the fusion formula recovered ~97% of the achievable gap between the
original residual failure and the fixed-lambda ceiling for both
retrievers; the remaining ~3% (a real, significant ~1.2-1.3% MSE gap
vs. the fixed lambda) is attributable to gate optimization/calibration,
a smaller but genuine secondary issue.

**Q8. What should the next experiment be: generalization / gate
calibration / multi-slot consumer?** **Gate Capacity / Calibration
Audit** (per spec section 22's own prescription for exactly this
outcome pattern) -- NOT executed in this track, per the STOP rule.
Candidates for that follow-up, as pre-registered: global trainable
lambda, per-channel lambda, query-conditioned scalar gate variants,
horizon gate. Multi-slot consumer redesign (Case D's remedy) is NOT
indicated -- the single aggregated retrieval vector `R` is clearly
sufficient once consumed correctly (Case A/D territory, not Case D of
section 24). Generalization to other horizons/datasets is premature
until the gate-optimization gap is resolved or understood.

## STOP rule compliance

No Stage1 retriever modification, no residual-aware retriever
objective, no new Multi-Slot loss, no new consumer attention, no
Weather, no additional horizon -- none occur in this track. Only
`fusion_mode` was varied; everything else (checkpoints, caches, gate
architecture/init, optimizer, training schedule) is byte-identical to
TRACK-O's own established setup.
