# TRACK-J-SHARED-ENCODER-DRIFT01 -- Code Audit

Direct code reads only; no report/`RESEARCH_CONTEXT.md` prose trusted
without a matching grep/read. Answers to the 20 required audit items:

## 1-2. Which script/config is the "corrected, clean" MLP baseline?

There are TWO distinct code paths in this repository that can train a
`relation_encoder_type='mlp'` Stage-1 retriever:

**(a) `run.py --task_name stage1_relation` / `exp/exp_stage1_relation.py`**
(the original, most general production entry point). Its Stage-1
candidate-encoding behavior is controlled by
`--stage1_full_memory_gradient_mode`, **default `'bank'`**
(`run.py:325`) -- i.e. candidates come from a **cached, non-re-encoded
bank by default**; `'full_online'` re-encodes every step with gradient.
`scripts/run_candidate_reencode_kl_full.sh` (the most directly relevant
prior experiment -- it exists specifically to study candidate-side
gradient effects with an MLP encoder) confirms this: its `full_bank_kl`
arm is explicitly commented `"full memory bank, no candidate gradient
(baseline)"` and never passes `--stage1_full_memory_gradient_mode
full_online` in `COMMON_ARGS`.

**(b) `scripts/train_patch_retrieval_expert01.py`-style scripts**
(reused unmodified throughout this session's TRACK-F/G/H/I work). Their
`train_epoch`/`eval_epoch` **always** re-encode candidates from the
current encoder with gradient flowing (`E = encode_raw(model,
exp.memory_x, c)  # NOT detached/cached`), unconditionally -- they do not
even read `stage1_full_memory_gradient_mode`.

**TRACK-J's own research question is specifically about the shared,
simultaneously-updated query/candidate encoder** -- i.e. exactly path
(b)'s behavior, NOT the default `'bank'` mode of path (a). Building A0 on
path (a) with its default args would silently answer a different
question (no candidate gradient at all). **A0 is therefore built on the
same `individual_utility_memsafe` / `normalized_teacher_prob` / `kl_loss`
/ `memory_value` / `encode_raw` / `arm_score` / `exp._candidate_mask`
library already verified bug-fixed and exact-reproducing across
TRACK-F/G/H/I this session**, with `relation_encoder_type='mlp'`,
`relation_self_fill='linear'` substituted for the `transformer`/`zero`
override `train_patch_retrieval_expert01.py` itself forces. This is the
"clean, corrected, full-online, shared-encoder" baseline the spec asks
for -- confirmed correct by direct code read, not assumed.

## 3. Does `patch_len`/`stride` affect MLP encoder output?

No. `RelationEncoder.__init__` (`models/RelationStage1.py:2897-2915`,
`encoder_type == 'mlp'` branch) never reads `configs.patch_len` or
`configs.stride` -- only the `transformer` branch does
(`RelationPatchEmbedding(patch_len=..., stride=...)`, line ~2879-2885).
Confirmed further by a dedicated regression test (see AUDIT unit-test
list, item 6) that trains two identically-seeded MLP encoders with
different `patch_len` values and asserts byte-identical output.

## 4. Shared query/candidate encoder?

Yes. `encode_raw(model, x, c)` (`train_factorial_e2e01.py`) calls
`model.encoder(model._relation_tensor(x, c, c))` for BOTH the query batch
(`x=batch_x`) and the candidate bank (`x=exp.memory_x`) -- the identical
`nn.Module` instance, same parameters, in both calls.

## 5-6. Candidates re-encoded every step, no stale bank?

Yes, by construction of the reused `train_epoch` pattern (see item 1-2):
`E = encode_raw(model, exp.memory_x, c)` is called inside the batch loop,
every batch, every epoch -- never cached across steps, never `.detach()`-ed.

## 7. Candidate memory: train split only?

Yes. `Exp_Stage1_Relation._ensure_memory()` builds `memory_x`/`memory_y`
from `RelationMemorySampler(train_data, ...)` -- `train_data` is the
`flag='train'` split only. Unchanged this track.

## 8. Do train/val/test queries share the same candidate universe?

Yes -- `exp.memory_x`/`exp.memory_y` are built once and reused for every
split's evaluation; only the query batch (`batch_x`/`batch_y`) differs by
split (`flag='train'|'val'|'test'`).

## 9. Overlap/self mask matches current production fix?

`exp._candidate_mask(batch_start_idx)` is called unmodified (same call
site convention as every other script this session reused it in) -- no
TRACK-J-specific masking logic is introduced.

## 10. Raw Future-MSE teacher: exact definition

`individual_utility_memsafe(memory_c, offset_c, query_future, chunk_size)`
(`train_factorial_e2e01.py`): `u_i = -mean_H((Y_i - Y_q)^2)`, chunked over
the candidate axis. `d = -u` is what feeds the teacher softmax.

## 11. `delta_last`: input space vs teacher/value space

Three DISTINCT `relation_*_space` flags exist and are set independently:
`relation_input_space` (what the encoder itself sees), `relation_teacher_space`
(unused by the `memory_value`-based path -- see item 10's caveat below),
`relation_value_space` (what `memory_value()` reconstructs candidate
FUTURES in, i.e. the space `individual_utility_memsafe` computes MSE in).
This track sets all three to `delta_last` (matching
`run_candidate_reencode_kl_full.sh`'s convention and every Track-A/F/G/H/I
script this session used), so the distinction is moot in practice here,
but the three flags are confirmed independent in the code
(`exp_stage1_relation.py._future_distance_inputs` reads
`relation_teacher_space` separately from `memory_value`'s
`relation_value_space` read -- A0 uses ONLY the `memory_value`/
`individual_utility_memsafe` path, never `_future_distance_inputs`,
consistent with every prior track this session).

## 12-13. Teacher normalization / temperature

`normalized_teacher_prob(d, mask, tau_t)`: z-score-normalizes `d` per row
over valid candidates, then `softmax(-normalized/tau_t)`. This IS the
"`teacher_mse_space=normalized`" behavior `run_candidate_reencode_kl_full.sh`
requests explicitly via CLI on the `run.py` path -- confirmed equivalent
in effect (both z-score-normalize the raw distance before the teacher
softmax), though A0 uses the `train_factorial_e2e01`/
`train_horizon_retrieval_expert01` function, not `run.py`'s own internal
implementation (not re-derived independently this round; relied on this
session's repeated exact-reproduction evidence that this function
library is correct).

`tau_t` (teacher temperature): **0.1**, sourced from
`run_candidate_reencode_kl_full.sh --tau_teacher 0.1` -- the most directly
relevant prior MLP-encoder experiment (NOT `train_patch_retrieval_expert01.py`'s
own `tau_t=0.02`, which was a value chosen specifically for that script's
patch-Transformer sweep, not inherited from an original MLP baseline).

## 14. Student temperature

`tau_s = 0.1`, also from `run_candidate_reencode_kl_full.sh --tau_student 0.10`.

## 15. KL direction

`kl_loss(p_t, s, mask, tau_s)`: `p_t` detached,
`term = p_t * (log p_t - log_softmax(s/tau_s))`, summed over valid
candidates, mean over batch -- `KL(p_teacher || p_student)`. Confirmed by
direct formula (unchanged from every prior track this session).

## 16. Checkpoint-selection criterion

**`min val model_top10_individual_mse`** (raw future MSE of the model's
own future-blind Top-10 picks) -- the established, "future-blind at
selection time" convention used consistently by every TRACK-A/F/G/H/I
script this session (`train_patch_retrieval_expert01.py`'s own explicit
phrasing), NOT `run.py`'s generic `--stage1_checkpoint_metric` default
(`'loss'`, i.e. raw val KL) and not
`run_candidate_reencode_kl_full.sh`'s `recall10` override. Reasoning:
(a) it directly measures the quantity TRACK-J's own research question is
about (does training-loss improvement track actual retrieval quality?),
making a KL-loss-based selection criterion circular for this specific
diagnostic; (b) it is the criterion every other script this session's
audits already vetted as "future-blind" in exactly the phrasing spec
section 19 uses. This is a **judgment call**, documented here rather than
silently assumed, per the spec's own instruction not to hardcode
unverified values.

## 17. Full-memory evaluation ranks ALL candidates?

Yes -- `stable_topk_indices` is always called over the complete `[B, N]`
score matrix; no shortlist/Top-M/sampling anywhere in the reused library
(same "full-memory only" discipline as every prior track).

## 18-19. Candidate re-encode fix / train-eval support-mismatch fix present?

Yes to both, by construction of using this specific function library
(items 5-6, 8 above) -- these are exactly the properties
`train_patch_retrieval_expert01.py`'s own docstring documents as fixed
("Candidate-side gradient... recomputed... every batch", "so encoder
gradients from the candidate side are never silently dropped").

## 20. Existing comparable numbers?

None directly comparable: every prior Track-A/F/G/H/I run used
`relation_encoder_type='transformer'` (patch_len=120) or reused a
pre-trained Transformer p120 checkpoint. No MLP-encoder run with this
exact `individual_utility_memsafe`-based full-online library exists in
this repository's results/ directories (`grep`-checked). TRACK-J's own
`B0` reproduction-check run (§ baseline reproduction, instrumentation
OFF) is therefore the **first and only" number to compare instrumentation
ON against -- an internal consistency check, not a reproduction of a
pre-existing external baseline.

## Regression test: patch_len irrelevance for MLP

`tests/test_j_shared_encoder_drift01.py::test_mlp_encoder_output_unaffected_by_patch_len`
constructs two `RelationEncoder(encoder_type='mlp', ...)` instances with
identical seeds but different `patch_len`/`stride` config values and
asserts byte-identical output on the same input -- confirms item 3
empirically, not just by code inspection.

## Final baseline configuration (all values sourced above, not assumed)

| Param | Value | Source |
|---|---|---|
| `relation_encoder_type` | `mlp` | spec requirement |
| `relation_self_fill` | `linear` | `run_candidate_reencode_kl_full.sh` |
| `relation_input_space` / `relation_teacher_space` / `relation_value_space` | `delta_last` | `run_candidate_reencode_kl_full.sh`, matches every prior track |
| `candidate_mask` | `raft` (`exp._candidate_mask` default) | unchanged production default |
| `tau_t` (teacher) | `0.1` | `run_candidate_reencode_kl_full.sh --tau_teacher` |
| `tau_s` (student) | `0.1` | `run_candidate_reencode_kl_full.sh --tau_student` |
| `top_k` | `10` | spec + all priors |
| `batch_size` | `32` | `run_candidate_reencode_kl_full.sh` + all priors |
| `learning_rate` | `1e-3` | `run_candidate_reencode_kl_full.sh` Stage-1 LR |
| `train_epochs` | `10` | `run_candidate_reencode_kl_full.sh` (full mode) |
| `patience` | `5` | `run_candidate_reencode_kl_full.sh` (full mode) |
| `d_model` / `n_heads` / `e_layers` / `d_ff` | `128` / `4` / `2` / `256` | `run_candidate_reencode_kl_full.sh` (n_heads/e_layers unused by MLP, kept for CLI/arch-config compatibility) |
| checkpoint criterion | `min val model_top10_individual_mse` | judgment call, reasoned above |
| optimizer | `Adam`, no weight decay | matches every prior track |
| init_seed / loader_seed | `0` / `0` | spec section 18 |
