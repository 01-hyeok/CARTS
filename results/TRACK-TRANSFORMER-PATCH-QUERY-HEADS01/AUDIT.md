# TRACK-TRANSFORMER-PATCH-QUERY-HEADS01 -- repository audit + implementation plan

## 1. RelationEncoder transformer branch (direct code read, `models/RelationStage1.py`)

- `encoder_type='transformer'` requires `relation_input_space` with
  `relation_feature_count==1` (line ~2868-2873) -- `delta_last` satisfies
  this (`RELATION_INPUT_FEATURES['delta_last'] = ('delta_last',)`, count 1).
- Requires `self_fill != 'linear'` (line ~2878, transformer-specific
  check) -- spec's `relation_self_fill='zero'` satisfies this.
- `self.patch_embed = RelationPatchEmbedding(seq_len, patch_len, stride,
  d_model, dropout)` (`layers/relation_patch_embed.py`):
  `num_patches = (seq_len - patch_len) // stride + 1`. Verified
  arithmetic: H96/patch_len=16/stride=16 -> 6; H720/patch_len=120/
  stride=120 -> 6. Confirmed empirically too (GPU smoke test, Section 4).
- `forward(..., return_tokens=True)`, `pooling='cls'` branch (line
  ~3009-3019): `out = self.encoder(cat([cls, tokens]))`; `h = out[:,0]`
  (pooled, goes through `self.proj(self.norm(h))` then
  `F.normalize(...)` for `retrieval_similarity='cosine'`); `patch_tokens
  = out[:, 1:]` (CLS excluded, exactly `num_patches` positions). Returns
  `(normalized [B,D], token_bank [B,P,D])`, `token_bank` already
  L2-normalized (`F.normalize(patch_tokens, dim=-1)`, line ~3035) --
  confirmed by this session's own GPU run (smoke tests below), and
  consistent with the TRACK-F docstring this method already carries
  (`test_forward_pooled_output_unchanged_by_return_tokens_flag`).
- This is the FIRST time this track uses the transformer branch with
  REAL patch tokens for retrieval-query construction (not Late
  Interaction, not candidate-side) -- `TRACK-TEMPORAL-QUERY-HEADS01`'s
  masked-MLP tokens were a workaround BECAUSE V5's actual production
  encoder is `relation_encoder_type='mlp'` (no patch structure at all).

## 2. Relationship to prior Transformer/patch tracks (per user's own summary, not re-derived)

- `TRACK-F-LATE-INTERACTION-FEASIBILITY01`/`-CONTROL01`: tested
  query-candidate patch-to-patch late interaction (MaxSim/LSE); the
  CONTROL found the gain attributable mostly to fresh metric/projection
  adaptation, not late interaction's patch-to-patch structure itself.
- `TRACK-G-DECOUPLED-METRIC-ADAPTATION01`: metric-adaptation-focused,
  same family.
- `TRACK-A-PATCH-RETRIEVAL-EXPERT01`: Patch-Mean representation showed
  some aggregate-MSE signal.
- None of these tested patch tokens as input to retrieval-QUERY-HEAD
  construction (this track's actual question) -- they tested
  query-candidate interaction at inference/scoring time. This track is
  not a replay of Late Interaction and does not reuse or contradict
  those findings; it isolates a different mechanism (see Section 3).

## 3. Three effects this track separates (per spec Section 0)

```
MLP-V5 (canonical)         <- backbone: MLP
T-V5   (Transformer, pooled-only, SlotHeads)   <- backbone: Transformer, query: pooled
T-PH5  (Transformer, patch-aware SlotHeads)    <- backbone: Transformer, query: patch-attention
T-PH1  (Transformer, patch-aware, 1 head)      <- isolates multi-head contribution from patch-aware contribution
T-V1   (Transformer, pooled-only, 1 head)      <- isolates multi-head contribution from backbone-only effect
```

Comparison 1 (MLP-V5 vs T-V5): backbone effect alone.
Comparison 2 (T-V5 vs T-PH5): patch-token-utilization effect, backbone held fixed.
Comparison 3 (T-PH1 vs T-PH5): multi-head effect, patch-aware construction held fixed.
Comparison 4 (T-V1 vs T-PH1): patch-aware effect without any multi-head confound.

## 4. GPU1 smoke test results (2026-10-22, before full grid)

| Arm | init_hash (ETTh1_96) | num_patches | encoder_params | slot_heads_params | total_trainable | max_vram_mb | gradient check |
|---|---|---:|---:|---:|---:|---:|---|
| V1 | 61095ac46eec6255 | 6 | 640192 | 16384 | 656576 | 564.1 | PASS |
| V5 | 61095ac46eec6255 | 6 | 640192 | 81920 | 722112 | 565.2 | PASS |
| PH1 | 61095ac46eec6255 | 6 | 640192 | 16384 | 656576 | 564.2 | PASS |
| PH5 | 61095ac46eec6255 | 6 | 640192 | 81920 | 722112 | 565.5 | PASS |

S1 (num_patches): PASS (6 for both H96/16 and H720/120, also directly
verified at H720 scale: `init_hash=2748e55274870d76`, `num_patches=6`,
`total_trainable=2786720`, PH5 smoke PASS).
S3 (Transformer init hash, 4 arms, same cell): PASS -- byte-identical.
S4 (paired SlotHeads init): PASS -- `SlotHeads(d,1).W[0] ==
SlotHeads(d,5).W[0]` verified by unit test (per-slot seeded
construction, `1000+m`, independent of `n_slots` by design -- no new
code needed, just verified).
Section 18 requirement (T-V5 vs T-PH5 identical param count): PASS,
722112 == 722112 (PH-arms add ZERO trainable parameters -- attention
over patch tokens is parameter-free, reusing
`models.TemporalQueryHeads.temporal_attention_heads` unmodified, no
`1/sqrt(D)` scaling -- already diagnosed wrong for L2-unit vectors in
the predecessor track and not reintroduced here).
S8 (gradient reaches encoder + SlotHeads, including through the
attention path for PH1/PH5): PASS, asserted in the smoke-test code
path itself (non-zero gradient on `model.encoder.parameters()` and
`slot_heads.W`).

Candidate branch pooled-only for all 4 arms (Section 8): confirmed by
code construction -- `compute_scores_arm`'s candidate-side call is
`encode_pooled_candidates(lambda x,c: encode_raw(model,x,c), ...)`,
IDENTICAL across all 4 arms, `encode_raw` never passes
`return_tokens=True`.

## 5. What is NOT yet done

Full 4-cell x 4-arm Stage-1 grid (16 runs, fixed 10 epochs each) +
post-hoc checkpoint selection + Mean-Mixture cache + Stage-2 has been
LAUNCHED (sequential, GPU1) but has not completed as of this audit.
MLP-V1/V5 canonical numbers are read-only reused from
`TRACK-V-SHARED-TOP100-SIGNFIX01` (not retrained). Results, tables, and
the CASE 1-5 verdict will follow once the grid finishes.
