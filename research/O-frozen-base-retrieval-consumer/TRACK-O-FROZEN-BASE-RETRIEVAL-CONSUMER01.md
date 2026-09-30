# TRACK-O-FROZEN-BASE-RETRIEVAL-CONSUMER01

Direct follow-up to TRACK-N. Core principle honored throughout: **the
Stage1 retriever was never touched.** J1 and M2's checkpoints are the
exact same frozen artifacts used in every prior track; only how Stage2
consumes their retrieval output was varied.

Full audit: `research/O-frozen-base-retrieval-consumer/AUDIT.md`.

## Critical prerequisite finding: `relation_mixer` is structurally gradient-dead

Before trusting any O1-O4 result, this track discovered (empirically,
then confirmed analytically) that `relation_mixer` -- the module PART 4
intended to be the main new trainable "consumer" -- **never receives any
gradient in this pipeline, regardless of arm or input.** `RelationMixer`
computes `beta = softmax(scores, dim=1)` over the source-slot axis, and
`model.num_source_slots() == 1` for this host config (`source_mode=
'auto'`, a `pearson_self_top1` relation graph -- each channel's only
"source" is itself). Softmax over a size-1 axis is the constant function
`beta≡1`, whose Jacobian is exactly zero. Verified two ways: (a) after a
real forward+backward pass on real O1 data, `relation_mixer.score_net`'s
gradient is all-zero while `gate`'s is large and nonzero; (b) after 10
full training epochs, `relation_mixer`'s weight hash is byte-identical
to its random init.

This is an architecture-level fact of the `[N, channels, 1, pred_len]`
single-aggregated-source cache convention used by EVERY Stage2 track
this session (TRACK-A-SET-LOSS-CONTROL02, TRACK-M, TRACK-N), not a bug
introduced here. It also **corrects** TRACK-N's own stated diagnosis for
why its Gate-Only ablation was confounded ("mixer frozen at a state
trained on S0's zero retrieval") -- S0's mixer was never trained at all
(same degeneracy), so "loaded from S0" and "fresh init" are the *same
value* (confirmed by hash equality). The only genuinely trainable
retrieval-consuming module in this whole pipeline is **`gate`**. Reported
transparently: O1-O4 remain a valid, controlled comparison (same frozen
base, same fresh-init gate, same frozen dead-mixer, only
retriever/aggregation differing), but they measure "gate-only
consumption," not "mixer+gate consumer training" as originally
envisioned.

## Setup

Common frozen base `B_q` and reproduction gates reused directly from
TRACK-N/M (no rebuilding needed -- same checkpoints, same code, byte
-identical caches, PART 8 re-verified to <5e-7 against all four target
values). Consumer init: fresh `build_fresh_stage2(seed=0)`, only
`base_head.*` loaded from S0's checkpoint (filtered), `relation_mixer`/
`gate` left at their shared fresh init -- confirmed IDENTICAL across all
four arms via `shared_init_fingerprint.json` (`base_head_identical=True`,
`consumer_identical=True`, `trainable_parameter_names_identical=True`,
trainable set = `{gate.*, relation_mixer.*}` only, per the audited
forward-graph trace -- `relation_concat_projection` correctly excluded
as provably unused by this host's `stage2_relation_fusion='gate'`
config). 21/21 required unit tests pass; full repo suite unaffected.

## PARTS 9-10: aggregation diagnostic (independent of Stage2 training)

| Arm | Weighting | Effective K | I (weighted ind. MSE) | D_w | C_w | Agg |
|---|---|---|---|---|---|---|
| J1 | Uniform | 10.0 | 1.0055 | 0.1006 | 0.4635 | 0.5640 |
| J1 | Host | 8.25 | 0.9891 | 0.1217 | 0.4566 | 0.5783 |
| M2 | Uniform | 10.0 | 0.9886 | 0.0989 | 0.4247 | 0.5236 |
| M2 | Host | **5.76** | 0.9423 | 0.1692 | 0.3933 | 0.5624 |

Mean Spearman(alpha, individual-candidate-MSE): J1 = -0.051, M2 = -0.147
(both negative -- HostScorer IS quality-aware, favoring lower-MSE
candidates, more so for M2). Yet `D_w` still inflates for both. This is
decisive: the weighted individual MSE `I` (quality-aware mean) is
*better* than the plain Uniform mean for both arms (J1: 1.0055->0.9891;
M2: 0.9886->0.9423) -- Host weighting is NOT mis-selecting worse
candidates. The `D_w` inflation is a pure **concentration** effect
(effective K collapsing from 10 to 8.25 for J1, and much more severely
to 5.76 for M2), not a quality-weighting failure. M2's candidates --
selected for complementarity, hence more heterogeneous under an
exogenous fixed scorer -- get concentrated much more sharply by
HostScorer's softmax than J1's already-individual-quality-optimized
candidates.

## Primary Stage2 causal comparisons (test split, min-val-selected checkpoints, 10k-rep paired bootstrap, query_start_idx unit)

| Arm | retriever | agg | final MSE | vs O0 (base=0.488790) |
|---|---|---|---|---|
| O0 | -- | -- | 0.488790 | -- |
| O1 | J1 | Uniform | 0.489349 | +0.000559, CI [0.00035,0.00078], **sig. worse** |
| O2 | J1 | Host | 0.490523 | +0.001733, CI [0.00132,0.00218], **sig. worse** |
| O3 | M2 | Uniform | 0.488924 | +0.000135, CI [-0.00001,0.00028], not sig. |
| O4 | M2 | Host | 0.489104 | +0.000314, CI [0.00010,0.00053], **sig. worse** (small) |

**Comparison A (O3 vs O1)**: M2 Uniform significantly BEATS J1 Uniform:
`-0.000425`, CI `[-0.00063,-0.00023]`.
**Comparison B (O4 vs O3)**: Host vs Uniform for M2 -- NOT significant
(`-0.000179`, CI `[-0.00036, 0.00001]`).
**Comparison C (O2 vs O1)**: Host vs Uniform for J1 -- significantly
WORSE under Host (`+0.001174`, CI `[0.00080,0.00158]`).
**Comparison D (interaction)**: `(O4-O3)-(O2-O1) = -0.000995`, CI
`[-0.00142,-0.00059]`, significant and NEGATIVE -- the Host penalty is
actually SMALLER for M2 (+0.00018) than for J1 (+0.00117) once a
trainable gate can partially compensate. This is the OPPOSITE direction
from TRACK-N's standalone-retrieval-quality damage finding (there M2
suffered *more* %-damage) -- the gate absorbs more of the standalone
Host degradation for M2 than for J1.

**M2 also significantly beats J1 under Host** (O4 vs O2:
`-0.001420`, CI `[-0.00189,-0.00098]`).

## Decision table verdict: Case 4

None of the four arms achieve a meaningful, positive improvement over
`O0`. Three of four differences are statistically significant (large
n=2161x7) but all are practically tiny (<0.4% relative), and the
direction is mostly unhelpful (O1/O2/O4 are worse than O0; O3 is
statistically tied, not better). Per the pre-registered decision table,
**Case 4 applies**: the Stage2 consumer architecture itself fails to
exploit real retrieval information, for any retriever or aggregation
tested. Retriever must NOT be changed; consumer architecture needs
redesign -- now with a specific, identified mechanism (the dead
`relation_mixer`, leaving `gate` alone to do all the work, which this
result shows it cannot do adequately even for the objectively
better-quality M2 retrieval).

This connects directly to TRACK-N's own central finding: a trivial,
non-learned, validation-calibrated single scalar lambda (no gate, no
mixer) already achieved test MSE ~0.463-0.468 for both J1 and M2 --
dramatically better than EVERY neural-gate result obtained across
TRACK-M's full joint training, TRACK-N's Gate-Only ablation, and this
track's O1-O4. The simplest possible deployable fusion rule outperforms
every neural approach tried so far.

## Final answers (spec section 28)

**Q1. With a common frozen base + fresh trainable consumer, is J1 still
better than M2?** No. Under identical treatment, M2 significantly beats
J1 in both aggregation modes (O3<O1, O4<O2, both CI entirely negative).

**Q2. Is M2 Uniform better than, similar to, or worse than J1 Uniform?**
Significantly BETTER (small but real margin, `-0.000425` MSE).

**Q3. Is M2 Host significantly worse than M2 Uniform?** No -- the
difference is not statistically significant (CI includes zero).

**Q4. Is Host damage due to weight concentration, candidate-quality
weighting, or both?** **Concentration**, decisively. The quality
-weighted mean (`I`) is *better* than the Uniform mean for both
retrievers (Host is quality-aware, negative Spearman correlation) --
`D_w` still inflates purely because effective K collapses (J1:
10->8.25; M2: 10->5.76).

**Q5. Is the Host penalty larger for M2 than for J1?** On the standalone
retrieval-quality metric (TRACK-N): yes (+7.42% vs +2.53%). On the
downstream, gate-mediated forecast metric (this track): **no, the
opposite** -- the interaction term is significantly negative, meaning
the trainable gate absorbs more of M2's Host damage than J1's.

**Q6. Primary cause of TRACK-M's Stage2 failure: retrieval quality /
aggregation / consumer initialization-training / mixed?** **Consumer
initialization/training** (specifically, the structurally-dead
`relation_mixer` leaving only `gate` to adapt, and `gate` alone proving
insufficient even for the objectively better M2 retrieval to produce a
meaningful downstream gain) is the dominant, now mechanistically
identified cause. Aggregation (Host) contributes a real but smaller,
and for M2 partially gate-absorbable, penalty.

**Q7. Is there a basis to keep M2 as the main retriever? YES/NO/
INCONCLUSIVE.** **YES.** M2 is never worse than, and often
significantly better than, J1 across every honest, controlled comparison
run so far (TRACK-M Stage1, TRACK-N oracle/calibrated-lambda, and this
track's O3 vs O1 / O4 vs O2). Its underperformance in TRACK-M's original
Stage2 was a consumption artifact, not a property of the retrieval
itself.

**Q8. Should the next experiment be a Stage1 forecast-conditional
retriever, or Stage2 consumer refinement?** **Stage2 consumer
refinement.** Every piece of evidence across TRACK-N and TRACK-O points
the same direction: M2's retrieval already carries real, competitive
-to-superior downstream value; what's missing is a consumer capable of
realizing it. The dead-mixer finding gives a concrete, actionable target
(the single-source-slot cache convention structurally disables the
mixer's attention mechanism; either the mixer needs a genuinely
multi-slot input to attend over, or the fusion architecture needs to be
simplified/replaced with something demonstrably able to match the
trivial calibrated-lambda baseline).
