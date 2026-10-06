# TRACK-W-TIMESTAMP-FUSION01 -- AUDIT

## PART 0: parallel-execution safety

Preflight GPU1 state recorded at `results/TRACK-W-TIMESTAMP-FUSION01/preflight_gpu_state.txt`
(2026-10-02 08:55 UTC). Existing GPU1 job at that time: TRACK-W-CHECKPOINT
-CRITERION-CORRECTION01's CASE-B retrain chain (`run_w_caseB_all.sh`, PID
3838937 bash runner / active python child PID varies as it steps through
its 5-item queue). That job's own files/checkpoints/results/logs under
`results/TRACK-W-CHECKPOINT-CRITERION-CORRECTION01/`,
`checkpoints/track_w_checkpoint_criterion_correction01/` are NEVER
touched by this track. This track runs as a SECOND, independent top
-level job on `CUDA_VISIBLE_DEVICES=1`, with its OWN internal steps
(Phase1's 12 runs, Phase2's 12 runs, cache builds, Stage2 runs) executed
strictly sequentially via one runner script -- never more than 1 active
child of THIS track on the GPU at a time. Total GPU1 top-level jobs
target: exactly 2 (TRACK-W-CHECKPOINT-CRITERION-CORRECTION01 + this
track), matching the user's explicit instruction.

## PART 1: STEP A -- code audit findings

### A.1 `scripts/run_v_one_setting01.sh` / `train_j_shared_encoder_drift01.py` / `train_t_pure_multislot01.py`

- `build_model(cli, device)` (`train_j_shared_encoder_drift01.py`) calls
  `build_experiment(base_ckpt, overrides)` (`train_margutil01.py`), which
  loads `torch.load(base_ckpt)['args']` and applies `overrides` on top --
  this is how EVERY V-arm/J-arm/K-arm/etc. script inherits `root_path`,
  `data_path`, `freq`, `data`, `embed`, and every other dataset arg from
  the ORIGINAL `soft_set_mse` Stage1 reference checkpoint, never
  specifying them itself. `exp._ensure_memory()` is then called, which
  builds `self.memory_x`/`self.memory_y` via `RelationMemorySampler`
  from the TRAIN split only.
- `encode_raw(model, x, c) = model.encoder(model._relation_tensor(x, c, c))`
  (`train_factorial_e2e01.py`) is the SOLE encoding entry point used by
  every V/J/K/M-arm. `_relation_tensor(x, target_channel, source_channel)`
  with target==source (self relation, which is what every one of these
  arms uses) builds a `[B, 1, L]` tensor via
  `_transform_relation_feature(x[:, :, c], 'delta_last')` = `x_c - x_c[:,-1:].detach()`
  -- i.e. exactly ONE channel's own raw delta-last scalar sequence of
  length `seq_len`, no cross-channel mixing, no timestamp anywhere in
  this path.
- `RelationEncoder.__init__` for the config these arms actually use
  (`relation_encoder_type=mlp`, `relation_self_fill=linear`,
  `relation_input_space=delta_last` -- confirmed directly from the
  reference checkpoints' saved args, both ETTh1 and Weather) builds:
  `self.n_features=1`, `self.seq_len=seq_len` (delta_last doesn't shrink
  length), `input_dim = 1 * seq_len = seq_len`,
  `self.encoder = Sequential(Linear(seq_len, d_ff), GELU, Dropout, Linear(d_ff, d_model))`,
  `self.norm = LayerNorm(d_model)`,
  `self.proj = Sequential(Linear(d_model,d_model), GELU, Linear(d_model,d_model))`.
  `forward`: for `self_fill=linear`, `_prepare_rows` returns the `[B,1,L]`
  tensor UNCHANGED (no role-embedding padding), reshaped to `[B, L]` and
  fed straight into the first `Linear(L, d_ff)` -- i.e. the ENTIRE raw
  scalar sequence is flattened into one vector and immediately mixed by
  a single dense layer. There is NO per-timestep embedding stage in the
  existing architecture to hook into -- the first thing that happens to
  the sequence is a full linear mix across all `L` positions at once.
  **This is the key architectural fact that drives the Phase 1 design
  decision in PART 2 below**: a literal "baseline-preserving, timestamp
  -off reduces to exactly the old V0" implementation is not naturally
  expressible in this architecture family (see PART 2.3), so the
  mandatory C0 capacity-matched control (spec section 6) is the one this
  track relies on, not an exact-equivalence claim.
- `compute_scores_full_grad(model, slot_heads, batch_x, memory_x, c)`
  (`train_t_pure_multislot01.py`) = `z_q=encode_raw(...)`,
  `q=slot_heads(z_q)` (`[B,S,D]`), `k_full=F.normalize(encode_raw(model, memory_x, c))`,
  `scores=einsum('bsd,nd->bsn', q, k_full)`. This function, along with
  `round_robin_topk_selection`, `hard_eval_decomposition`,
  `spearman_batch`, `slot_mechanism_diagnostics`, `SlotHeads`,
  `kl_loss_from_prob`, `slot_overlap_penalty`, `normalized_teacher_prob`,
  `memory_value`, `recall_at_k`/`ndcg_at_k`, is **generic over how `z_q`/
  `k_full` were produced** -- none of it inspects `model.encoder`
  directly except through `encode_raw`. This means this track can reuse
  every one of these functions completely UNMODIFIED, swapping only the
  encode call itself. Verified by direct read, not assumed.
- Checkpoint criterion in `train_t_pure_multislot01.py` (historical
  V0/V1/V2/V5): `min val retmse10` (round-robin Top-10 individual MSE),
  explicitly a TYPE C retrieval diagnostic under this session's own
  checkpoint-selection governance policy (2026-10-01), NOT the training
  objective (KL). The val loop computes NO validation KL anywhere --
  only `retmse10/agg_mse10/recall10/ndcg10` are logged per epoch. See
  PART 3 (historical-comparison validity) for how this is handled.

### A.2 `build_t2_true_original_kl_cache01.py` / `build_t_multislot_cache01.py` / `train_r_stage2_lambda01.py`

Read and confirmed (not reproduced in full here for space): these are
pure POST-HOC cache builders and Stage2 trainers that consume an already
-trained Stage1 checkpoint's embeddings via the SAME `encode_raw`
entry point, then run Stage2 fusion training unmodified. This track
will reuse `train_r_stage2_lambda01.py` and the SAME Stage2
architecture/protocol UNMODIFIED, pointed at NEW retrieval caches built
from this track's own timestamp-aware Stage1 checkpoints -- exactly per
spec section 14's "fresh Stage2 training per arm, no checkpoint
injection" requirement. The cache-builder scripts cannot be reused
byte-for-byte unmodified (they call `encode_raw` directly, with no mark
plumbing), so new, narrowly-scoped cache builders are written for this
track (`build_w_timestamp_phase1_cache01.py` / `build_w_timestamp_phase2_cache01.py`)
that mirror their structure exactly, swapping only the encode call.

### A.3 `models/RelationStage1.py`

See A.1 above for the `RelationEncoder`/`_relation_tensor` findings.
`stable_topk_indices` (used by `round_robin_topk_selection`'s callers
for the oracle ranking) is reused unmodified.

### A.4 `utils/relation_memory.py`

`RelationMemorySampler.__init__` builds
`memory_x = sliding_window_view(data_x, seq_len, axis=0).transpose(0,2,1)[:num_windows]`
from `train_dataset.data_x` (the TRAIN split's own locally-sliced `[T,C]`
array). `Stage1WindowDataset.__init__` sets
`self.starts = arange(len(base_dataset)) + base_dataset.border1`
(converting local indices back to GLOBAL file indices) and
`Stage1WindowDataset.__getitem__` explicitly DISCARDS
`seq_x_mark`/`seq_y_mark` from the base dataset's own `__getitem__`
(`_, seq_x, seq_y, _, _ = item`) -- this is the exact point confirming
the user's framing is accurate: **no existing Stage1 trainer in this
repository has ever had timestamp information reach its encoder**; the
mark tensors are computed by every base `Dataset_*` class but dropped
immediately by `Stage1WindowDataset`.

### A.5 `data_provider/data_loader.py` / `data_provider/data_factory.py`

Both `Dataset_ETT_hour.__read_data__` and `Dataset_Custom.__read_data__`
slice `data_stamp` to the SAME `[border1:border2]` range as `data_x`/
`data_y` (verified by direct read of both classes) -- so
`train_dataset.data_stamp[i]` and `train_dataset.data_x[i]` refer to the
same underlying timestep for every `i`, in every split, for both
dataset classes. This is the alignment guarantee PART 4's `memory_x_mark`
construction below depends on, and it holds by construction in the
EXISTING, unmodified loader code -- nothing needed fixing here, only
confirming.

`data_provider(args, flag, ...)`: `timeenc = 0 if args.embed != 'timeF' else 1`.
Both the ETTh1 and Weather reference checkpoints already have
`embed='timeF'` saved in their args (confirmed by direct checkpoint
load), so `timeenc=1` and `data_stamp = time_features(dates, freq=args.freq)`
automatically -- no override needed for `embed`, only for `freq` (next).

### A.6 `utils/timefeatures.py` + actual dataset cadence (CRITICAL finding)

Measured directly from the CSVs used by the reference checkpoints'
`root_path`/`data_path` (NOT assumed):

| Dataset | date column | n rows | median interval | time_features(freq) used historically | CORRECT cadence |
|---|---|---:|---|---|---|
| ETTh1 (`ETTh1.csv`) | `date` | 17,420 | **1 hour** (17,419/17,419 intervals exactly 1h) | `freq='h'` | **`h` is correct** -- no change needed |
| Weather (`weather.csv`) | `date` | 52,696 | **10 minutes** (52,693/52,695 intervals exactly 10min; 2 anomalous rows, a pre-existing dataset artifact, not touched) | `freq='h'` (same default as ETTh1, inherited from `run.py`'s global default, never overridden for Weather in any existing checkpoint) | **`h` is WRONG; should be `10min`** |

This exactly confirms the user's warning. `time_features_from_frequency_str`
maps `'h'` -> `[HourOfDay, DayOfWeek, DayOfMonth, DayOfYear]` (4 features)
and `'10min'` -> `[MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear]`
(5 features, confirmed by direct call). Under the historical `freq='h'`
mislabeling, Weather's `data_stamp` would be missing `MinuteOfHour`
entirely, making every one of the 6 consecutive 10-minute rows within
the same clock hour share IDENTICAL time features -- i.e. the time
signal would be literally non-identifying at the very granularity this
track's whole hypothesis depends on. **Crucially, this mislabeling never
affected any past result** (PART A.4: no prior trainer ever consumed
`data_stamp` for anything), so no historical experiment needs
correction -- it only matters starting now, for this track, which is
the first to actually use `data_stamp`. This track uses `freq='h'` for
ETTh1 (unchanged, confirmed correct) and `freq='10min'` for Weather (an
explicit, documented override passed via `build_experiment`'s
`overrides` dict), NEVER the `run.py` global default.

### A.7 `research/CURRENT_EXPERIMENT.md` / `RESEARCH_DECISIONS.md` / `EXPERIMENT_LOG.md` (TRACK-V)

Read; TRACK-V's V0/V1/V2/V5 definitions, teacher, candidate mask,
Stage2 structure, `SEQ_LEN=PRED_LEN` protocol, and K=10 round-robin
selection match exactly what sections 2/11/12/13 of this track's spec
require to hold fixed. No conflicts found between the historical
TRACK-V record and this track's own arm definitions.

## PART 2: STEP B -- architecture design

### 2.1 Per-timestep fusion module (`models/TimestampRelationEncoder.py`)

For a single target channel's delta-last scalar sequence `x in R^{B x L}`
and its timestamp feature sequence `c in R^{B x L x K}` (`K`=4 for
ETTh1/`h`, 5 for Weather/`10min`):

```
e_x(t) = P_x(x_t)                 P_x: Linear(1 -> d_proj)
e_t(t) = P_t(c_t)                 P_t: Linear(K -> d_proj)
h_t    = LayerNorm_fuse(e_x(t) + e_t(t))         in R^{d_proj}
H      = [h_1, ..., h_L]                          in R^{B x L x d_proj}
flat   = reshape(H, [B, L*d_proj])
m      = Linear(d_ff) -> GELU -> Dropout -> Linear(d_model)   (applied to flat)
z      = Proj(LayerNorm(m))        Proj: Linear(d_model,d_model)->GELU->Linear(d_model,d_model)
normalized = L2normalize(z)  (cosine; retrieval_similarity=l2 keeps z raw, same convention as RelationEncoder)
```

`d_proj` (new hyperparameter, not specified by the user spec) is set to
**32** (`d_model=128` in every cell tested, so `d_proj=d_model/4`) --
documented here as a deliberate, otherwise-arbitrary implementation
choice, not derived from the spec.

`Query` and `Candidate` share EVERY parameter (`P_x`, `P_t`,
`LayerNorm_fuse`, the main MLP, `norm`, `proj`) -- one `nn.Module`
instance, called twice (`encode(x_q,c_q)` / `encode(x_i,c_i)`), exactly
mirroring `RelationEncoder`'s own single-encoder-called-twice convention
via `encode_raw`.

### 2.2 Parameter-count accounting (mandatory, spec section 5)

For `seq_len=L`, `d_model=128`, `d_ff=256`, `K` time features:

- Historical `RelationEncoder` (`mlp`, `self_fill=linear`):
  `Linear(L,256)` + `Linear(256,128)` + `LayerNorm(128)` + `Linear(128,128)`x2
  = `256*(L+1) + 128*257 + 128*2 + 128*129*2`.
- This track's `TimestampFusionEncoder`:
  adds `Linear(1,32)` (`=64`), `Linear(K,32)` (`=32*(K+1)`),
  `LayerNorm(32)` (`=64`), and replaces the first main `Linear(L,256)`
  with `Linear(L*32, 256)` (`=256*(L*32+1)`) -- i.e. the FIRST main
  layer's input width grows 32x, which dominates the parameter-count
  delta. Exact counts per cell are computed and logged at run time (see
  `config_fingerprints/` -- every run prints and saves
  `old_relation_encoder_param_count` vs
  `timestamp_fusion_encoder_param_count`), not hand-computed here.

### 2.3 Why exact baseline-preservation (spec section 5's preferred option) is not used

As established in A.1, `RelationEncoder`'s `mlp` path has NO per
-timestep embedding stage to begin with -- the raw scalar sequence is
flattened and mixed by a single `Linear(L, d_ff)` in one step. Inserting
a per-timestep `P_x`/`P_t`/fusion stage BEFORE that necessarily changes
the first `Linear`'s input width from `L` to `L*d_proj`, so "timestamp
-off" cannot reduce to literally the same forward computation as the
unmodified `RelationEncoder` (different parameter shapes, not just
different values). Per spec section 5's explicit fallback instruction,
the mandatory **C0 capacity-matched control** (section 6) is used
instead: C0 uses the IDENTICAL `TimestampFusionEncoder` architecture/
parameter count as C1/C2, with `c` (the timestamp branch's input) fixed
to an all-zero tensor. `P_t(0) = bias_t` (a constant, position
-independent offset, since the zero input kills the weight term
entirely) -- so C0's `P_t` carries NO query/candidate-varying
information through `h_t`, while its trainable parameter count, init
hash (before any data-dependent forward pass), and gradient path remain
byte-identical to C1/C2's. This satisfies spec section 6's "C0/C1
parameter count and architecture must be identical" requirement exactly,
documented here as the resolution of an unavoidable spec/architecture
tension per section 22.

### 2.4 Memory bank timestamp alignment (spec section 17)

`memory_x_mark` is built by this track's own code (NOT by modifying
`RelationMemorySampler`, which stays untouched and shared with every
other track) as:

```
memory_x_mark = sliding_window_view(train_dataset.data_stamp, seq_len, axis=0).transpose(0,2,1)[:num_windows]
```

-- the IDENTICAL operation `RelationMemorySampler.__init__` already
applies to `data_x` for `memory_x`, applied to `data_stamp` instead, off
the SAME `train_dataset` object (`exp.train_data_for_memory`, already
exposed by the existing, unmodified `_ensure_memory()`). Since
`data_stamp` and `data_x` are the same length and same row alignment
(A.5), `memory_x_mark[i]` corresponds to EXACTLY the same window as
`memory_x[i]`, by construction, for every `i` -- this is verified by a
dedicated unit test (section 18.A) comparing `memory_x_mark[i]`'s
implied calendar timestamps against `train_dataset.data_stamp[s:s+L]`
for `s=starts[i]` directly, for first/middle/last candidate, both
datasets, train/val/test query windows.

### 2.5 Additive data-pipeline changes (the only 3 shared-file edits this track makes)

1. `utils/relation_memory.py`: `Stage1WindowDataset.__init__` gains
   `include_time_mark=False` (new, optional, default-False kwarg); when
   True, stores `self.data_stamp` and `__getitem__` returns a 4-tuple
   `(seq_x, future, start_idx, seq_x_mark)` instead of the original
   3-tuple. Default behavior (kwarg omitted) is BYTE-IDENTICAL to today.
2. `data_provider/data_factory.py`: `data_provider(...)` gains the same
   `include_time_mark=False` kwarg, threaded to `Stage1WindowDataset`.
3. `exp/exp_stage1_relation.py`: `_get_data(...)` gains the same kwarg,
   threaded to `data_provider`; `_ensure_memory()`'s internal
   `_get_data(flag='train', shuffle=False)` call is changed to pass
   `include_time_mark=getattr(self, '_include_time_mark_for_memory', False)`
   -- an attribute that does not exist on any other track's `exp`
   object, so `getattr(..., False)` preserves their behavior exactly.

Full existing test suite (`pytest tests/`) is run after these 3 edits,
BEFORE any further implementation, to confirm zero regression (see
STEP C below for the result).

### 2.6 C2 (SHUFFLED-TIME) operationalization

Spec section 6 asks for shuffled correspondence "in train" while val/
test's definition should be "fixed to match the research question," and
names "Real vs Shuffled under the identical protocol" as the most
important comparison. This track's reading: C2's broken correspondence
is a TRAINING-time intervention only (to test whether a model forced to
learn from a scrambled timestamp signal ends up worse at the REAL
-world task); VAL/TEST retrieval evaluation uses REAL correspondence for
C0/C1/C2 alike, so that the C1-vs-C2 TEST comparison isolates "did
training against broken correspondence hurt real-world retrieval,"
rather than conflating it with "is retrieval evaluation itself
meaningless under scrambled timestamps." Implementation: ONE fixed
permutation `perm` (`numpy.random.default_rng(seed=0).permutation(num_train_windows)`)
over the train split's own window index space. During TRAINING ONLY,
every train-split window's timestamp block (used both as a query, when
sampled by the train loader, and as a candidate, in the memory bank) is
replaced by `perm[i]`'s timestamp block, via a thin wrapper dataset
(query side) and a fancy-indexed reorder of `memory_x_mark` (candidate
side) -- both built from the exact same `perm` array, so a query at
train index `i` and the candidate at train index `i` are shuffled to the
SAME alternate window consistently. VAL/TEST evaluation for C2 uses the
REAL (unpermuted) `memory_x_mark` and REAL query marks throughout. This
specific interpretation is a judgment call on an underspecified detail,
documented here per spec section 22.

## PART 3: historical V0/V5 comparison validity (spec section 8)

Historical `train_t_pure_multislot01.py` (V0/V1/V2/V5) checkpoints on
`min val retmse10` (TYPE C), not `min val KL` (TYPE A, this track's own
criterion, matching the project's checkpoint-selection governance
policy). Per spec section 8's own fallback: epoch checkpoints are fully
present on disk for every V-arm/cell (TRACK-V never hit this via early
stop before the full trajectory in most cells -- to be confirmed per
cell), so a CASE-A-style POST-HOC re-selection (reload each saved
`checkpoint_epoch{N}.pth`, recompute validation KL via a single
forward-only pass, no backprop, reusing `kl_loss_from_prob`/
`normalized_teacher_prob` exactly) is attempted as a SEPARATE,
lower-priority side task (`scripts/reselect_v_kl_checkpoint01.py`),
run AFTER this track's own Phase 1/2 GPU work is underway, since it is
cheap (forward-only) and does not compete for the same "2 concurrent
jobs" GPU1 budget in any meaningful way. Until that completes, the
V-vs-T-V comparison in the final report is explicitly labeled
**PROVISIONAL (criterion mismatch not yet corrected)** per spec section
8's explicit instruction never to silently present mismatched criteria
as an equal comparison.
