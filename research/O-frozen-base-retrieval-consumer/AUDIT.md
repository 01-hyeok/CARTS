# TRACK-O-FROZEN-BASE-RETRIEVAL-CONSUMER01 -- Forward-Graph and Checkpoint Audit

NO RETRIEVER TRAINING. This track isolates "does Stage2 consume M2's
retrieval correctly" by training ONLY the retrieval-consuming submodules
from a FRESH init, on top of a fully frozen common base and fully frozen
retrievers.

## Files read

- `models/RelationStage2.py` (`Model.forward`, lines ~1979-2397 and
  `forward_from_retrieval_values`, lines 1892-1911).
- `exp/exp_stage2_relation.py` (`_select_optimizer`, `_candidate_mask`,
  `_get_data`, `_move_batch`, `_build_key_bank`, `_loss`).
- `scripts/train_setlossctrl_stage2_retrain02.py` (the validated Stage2
  trainer this track reuses for `train_epoch`/`eval_epoch`/
  `load_cache_as_lookup`/`gate_distribution`, all UNMODIFIED).

## Forward-graph trace: what actually consumes `relation_outputs`

`Model.forward()` (per target channel `c`, given a cached
`relation_outputs`/`relation_query_embs`):

```
if self.stage2_relation_fusion == 'concat_linear':
    y_ret_c = self.relation_concat_projection(relation_outputs.reshape(B, -1))
else:
    y_ret_c, beta_c, relation_scores = self.relation_mixer(relation_outputs, relation_query_embs)

y_base_c = y_base_all[:, :, c]                    # from self.base_head(batch_x), NOT retrieval-dependent
if self.fusion_mode == 'raft_concat':
    y_final_c = self.raft_concat_head(cat([y_base_c, y_ret_c]))
else:
    y_final_c, lambda_c = self.gate(y_base_c, y_ret_c)
```

**The S2_720 host checkpoint's own config** (read directly from
`checkpoint['args']`, not assumed): `stage2_relation_fusion = 'gate'`
(NOT `'concat_linear'`), `fusion_mode = 'residual'` (NOT
`'raft_concat'`). This means the ACTUAL forward path for this exact
model uses:

- `self.relation_mixer` (`layers/relation_mixer.py`'s `RelationMixer` --
  a self-contained `score_net` MLP that attention-pools the Top-K
  retrieved futures with a query-conditioned softmax; no separate
  external "retrieval projection" module exists inside it) -- USED.
- `self.gate` (`layers/retrieval_gate.py`'s `RetrievalGate`) -- USED.
- `self.relation_concat_projection` -- **NOT used** by this config
  (only reachable via the `concat_linear` branch, which this config
  never takes).
- `self.raft_concat_head` -- **NOT used** (only reachable via
  `fusion_mode == 'raft_concat'`).

**Correction to TRACK-M's own precedent**: TRACK-M's Stage2
(`train_setlossctrl_stage2_retrain02.py`) left `relation_concat_projection`
trainable alongside `base_head`/`gate`/`relation_mixer` (visible in its
own saved `trainable_parameters.json`) simply because it was never
explicitly frozen and defaults to `requires_grad=True` -- but given this
config it is never invoked in forward, so it received zero gradient and
sat inert at its initial values throughout TRACK-M's training. Harmless
there, but per spec PART 5 ("사용되지 않는 module을 임의로 trainable로
만들지 마라") this track explicitly EXCLUDES it from the trainable set.

**TRACK-O trainable set (audited, not assumed): `relation_mixer` +
`gate`, and nothing else.** Frozen: `base_head` (the common frozen
base), `relation_concat_projection` (provably unused), plus the
pre-existing retrieval-irrelevant `FREEZE_SUBMODULES` from
`train_setlossctrl_stage2_retrain02.py` (`stage1_encoder`,
`shared_cross_projection`, `retrieval_metric`, `pairwise_scorer`,
`query_cond_proj`, `candidate_cond_proj` -- moot anyway since retrieval
here is entirely cache-based, these never receive gradient regardless,
frozen for consistency with the established convention).

## Checkpoints (frozen, never retrained this track)

| Role | Checkpoint | SHA-256 |
|---|---|---|
| J1 retriever | `checkpoints/track_j2_key_update_decomposition01/ETTh1_720/J1_stopgrad_key/checkpoint.pth` | `1b14877fce3c41114e83db1b832fdd6bd749df4888c8ff74468e34a6d730626c` |
| M2 retriever | `checkpoints/track_m_relevance_constrained_multislot01/ETTh1_720/M2_relevance_budget/checkpoint.pth` | `9e3ef235281e1efb74aeb5a6ef21de72b4c1dc9d530a13c0c263bbece53606d2` |
| S0 common frozen base | `checkpoints/track_m_relevance_constrained_multislot01/stage2/ETTh1_720/S0_base/checkpoint.pth` | `a7f91972c02bce72a00b05e9402474f597cee71a6c5ce41247e34e17b35e8609` |

## Retrieval caches reused verbatim (NOT rebuilt -- identical checkpoint, identical code, identical schema)

TRACK-N and TRACK-M already built the EXACT four caches this track
needs, from the same frozen checkpoints via the same, unmodified
selection/aggregation code (`build_m_stage2_retrieval_cache01.py` for
Host, `build_n_gate_only_cache01.py` for Uniform). Rebuilding them would
produce byte-identical output (same future-blind hard Top-10 rule,
same `HostScorer`), so this track reuses the existing files directly
rather than duplicating them:

| Arm | Retriever | Aggregation | Cache path (reused) |
|---|---|---|---|
| O1 | J1 | Uniform | `results/TRACK-N-FORECAST-CONDITIONAL-UTILITY01/gate_only/cache/ETTh1_720/G0_J1/{train,val,test}.pt` |
| O2 | J1 | Host | `results/TRACK-M-RELEVANCE-CONSTRAINED-MULTISLOT01/stage2/cache/ETTh1_720/S1_J1/{train,val,test}.pt` |
| O3 | M2 | Uniform | `results/TRACK-N-FORECAST-CONDITIONAL-UTILITY01/gate_only/cache/ETTh1_720/G2_M2/{train,val,test}.pt` |
| O4 | M2 | Host | `results/TRACK-M-RELEVANCE-CONSTRAINED-MULTISLOT01/stage2/cache/ETTh1_720/S3_Mstar/{train,val,test}.pt` |

Reproduction gate (PART 8) is re-verified against these exact files
before any training starts (see `results/.../reproduction_gate.json`).

## CRITICAL FINDING: `relation_mixer` is structurally gradient-dead in this cache convention

Discovered empirically (O1 training run: `relation_mixer.score_net`
parameter hashes identical before/after 10 epochs of training, despite
`requires_grad=True` and nonzero `y_ret`/`lam` values) and then verified
directly: after a real forward+backward pass on real O1 (J1+Uniform)
data, `relation_mixer.score_net`'s gradients are all EXACTLY 0.0 (not
`None` -- the tensor exists, every element is zero), while `gate`'s
gradients are large and nonzero on the same pass.

Root cause: `RelationMixer.forward` computes `beta =
softmax(scores, dim=1)` over the **source-slot** axis (NOT the K=10
retrieved-candidate axis -- those are already fused into ONE vector by
the Uniform/Host aggregation *before* the cache is even built, matching
the `[N, channels, 1, pred_len]` schema every cache in this session
uses). `model.num_source_slots() == 1` for this S2_720 host
(`source_mode='auto'`, `pearson_self_top1` relation graph -- each target
channel's only "source" is itself). `softmax` over a size-1 dimension is
mathematically the constant function `beta ≡ 1.0`, whose Jacobian is
identically zero -- so **no gradient can ever reach `relation_mixer`'s
parameters, for ANY input, in this pipeline.** This is an architectural
fact of the `num_source_slots=1` cache convention, not a bug introduced
by this track: it applies identically to TRACK-A-SET-LOSS-CONTROL02,
TRACK-M's S0-S3, and TRACK-N's gate-only ablation. TRACK-N's own stated
confound diagnosis ("relation_mixer was frozen at a state trained on
S0's zero retrieval") is therefore corrected here: S0's `relation_mixer`
was never trained at all (same degeneracy applied to its own training),
so "loaded from S0" and "freshly initialized" are the SAME value --
confirmed by hash equality between TRACK-N G0's frozen mixer and this
track's fresh mixer init.

**Consequence for this track's design**: PART 4's "consumer는 학습한다:
relation_mixer... trainable" cannot be literally realized with the
existing cache/model convention -- `relation_mixer` remains at its fixed
random init regardless of `requires_grad`. The ONLY genuinely trainable
retrieval-consuming module in this whole pipeline is `gate`. This is
documented transparently rather than silently produced: O1-O4 are still
a valid, meaningful comparison (identical fresh gate init across all
four, identical frozen base, only retriever/aggregation differing -- the
causal control PART 12 requires still holds), but they measure
"gate-only consumption of a fixed-but-shared-across-arms aggregate
retrieval vector," not "consumer network (mixer+gate) training" as
originally envisioned. Reported honestly in the final report's answer
to Q1-Q8; not silently reframed as something it isn't.

## Consumer initialization (PART 11-12)

For every arm: `build_fresh_stage2(S2_720, seed=0)` builds a FRESH model
(base_head, relation_mixer, gate, relation_concat_projection all at
their fresh seed=0 init). ONLY `model.base_head`'s state is then loaded
from S0's checkpoint (filtered from S0's full `model_state_dict` to just
`base_head.*` keys) -- `relation_mixer`/`gate` are NEVER touched by S0's
own (zero-retrieval-trained) weights, staying at the shared fresh init
for all four arms. This directly fixes TRACK-N's Gate-Only confound
(there, the mixer was loaded from S0's own trained-on-zero-retrieval
state and frozen, which TRACK-N's own report flagged as a likely
confound suppressing all three gate-only arms roughly equally).
