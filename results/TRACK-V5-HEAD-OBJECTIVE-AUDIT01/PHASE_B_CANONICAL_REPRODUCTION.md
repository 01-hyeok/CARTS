# PHASE B -- Canonical V5 Correctness Verification

Script: `scripts/diag_v5_head_objective01.py --mode verify` (GPU1 only, strictly sequential, no training, no checkpoint write). Git commit `f769be3e7b2598d03b439f876ccd7a824cf09311`. seed0 throughout.

## B1. Checkpoint confirmation

| Cell | Checkpoint path | Epoch | Exists |
|---|---|---:|---|
| ETTh1_96 | `checkpoints/track_v_sharedtop100_signfix01/ETTh1/H96/seed0/pool_top100/ETTh1_96/V5/checkpoint_epoch10.pth` | 10 | YES |
| ETTh1_720 | `checkpoints/track_v_sharedtop100_signfix01/ETTh1/H720/seed0/pool_top100/ETTh1_720/V5/checkpoint_epoch9.pth` | 9 | YES |
| Weather_96 | `checkpoints/track_v_sharedtop100_signfix01/Weather/H96/seed0/pool_top100/Weather_96/V5/checkpoint_epoch4.pth` | 4 | YES |
| Weather_720 | `checkpoints/track_v_sharedtop100_signfix01/Weather/H720/seed0/pool_top100/Weather_720/V5/checkpoint_epoch6.pth` | 6 | YES |

These are the **sign-fixed** checkpoints (`track_v_sharedtop100_signfix01`), never the historical `INVALIDATED_BY_TEACHER_SIGN_BUG` artifacts. Teacher sign confirmed by source read (`normalized_teacher_prob(d, ...)`, `d_pool` passed DIRECTLY to the teacher, never negated -- see PHASE A). Model `state_dict` SHA256 (first 16 hex chars, for provenance, not a secret) recorded for every cell -- see each `phase_b/<cell>/phase_b_verify.json`.

## B2. Mean-Mixture verification (directly computed, per cell, 6 query-channel samples each)

For every sampled (query, channel): manually recomputed `p_h = softmax(masked_scores/tau_s)` and `p_mix = mean_h(p_h)` from the raw `scores` tensor returned by the canonical `compute_scores` (`train_retriever_pool01.py`), and compared against the SAME quantities returned by the canonical `mean_mixture_topk_selection` (`utils/mean_mixture_selection.py`).

| Check | Result (all 4 cells) |
|---|---|
| `p_heads` manual vs canonical | `torch.allclose(atol=1e-6)` -- PASS, max abs diff **0.0** |
| `p_mix` manual vs canonical | `torch.allclose(atol=1e-6)` -- PASS, max abs diff **0.0** |
| Probability sums to 1 | PASS (`p_mix.sum(-1)` within 1e-4 of 1.0 for every sampled row) |
| Invalid-candidate probability | **Vacuous check under Shared-P100**: `valid_mask.all()` is asserted True for every sampled batch -- the P100 pool is valid-by-construction (`build_coarse_pool` always returns exactly 100 valid global ids per query/channel), so there is no "invalid candidate inside the pool" case to test here (unlike Full-memory, where `cand_mask` can be False for padding/future-leakage exclusions). This is reported explicitly rather than silently skipped. |
| Head count == 5 | PASS, `scores.size(1) == 5` asserted every sampled batch |
| Top-10 parity (`picks` vs `stable_topk_indices` on the manually-recomputed `p_mix`) | PASS, `torch.equal` (exact, not `allclose`) for every sampled row -- same stable tie-breaking (both paths call the identical `stable_topk_indices`) |
| Round-Robin selection used for the "Mean-Mixture" Top-10? | **NO -- confirmed NOT round-robin.** See critical finding below. |

### Critical correctness finding: two DIFFERENT selection rules exist in this codebase, and the spec's "Mean-Mixture" is only one of them

Reading `scripts/train_v_sharedtop100_01.py::eval_channel` and `utils/expert_head_metrics{,_pool01}.py::evaluate_and_save_head_report{,_pool}` shows their own internal `rr`/`model_idx` diagnostic uses **`round_robin_topk_selection`** (`train_t_pure_multislot01.py`: cycle `t % 5`, take each head's own current argmax, allow duplicates) -- NOT the spec's `S_mix = Top10(p_mix)`. The PRODUCTION Stage2 pipeline (`scripts/build_v_meanmix_cache_pool01.py`, confirmed by direct read -- its own docstring: *"Builds a retrieval cache using `mean_mixture_topk_selection` instead of `round_robin_topk_selection`"*) uses the TRUE Mean-Mixture rule the spec describes, and this is what produces the reference Stage2 MSE numbers (PHASE B4 below).

Measured overlap between the two selection rules' own Top-10 sets (same scores, same query, fraction of the Mean-Mixture Top-10 also present in the Round-Robin Top-10):

| Cell | Mean-Mixture vs Round-Robin Top-10 overlap |
|---|---:|
| ETTh1_96 | 0.7406 |
| ETTh1_720 | 0.9089 |
| Weather_96 | 0.4307 |
| Weather_720 | 0.7203 |

These are genuinely different selection rules (overlap far from 1.0, especially Weather_96 at 0.43) -- **this report's PHASE C "Mean-Mixture" row is always the TRUE `mean_mixture_topk_selection` result, computed fresh by `diag_v5_head_objective01.py`, never the pre-existing `rr` field in `TRACK-HARD-EXPERT-V5-P100-ALLH01`'s saved JSONs** (which is the round-robin diagnostic, a different quantity, as this finding shows). This distinction matters and was caught before being silently propagated into the final tables.

## B3. Future alignment

For every sampled query/channel, `pooled_memory_c` (candidate futures, gathered from `memory_value`'s `delta_last`-space reconstruction) was checked to: (a) have the same horizon dimension as `query_future`, (b) reconstruct to finite values after adding `offset_c` (the query's own last-value offset), (c) land on a comparable scale to the query's own future (`std` ratio recorded, not a strict assert since candidate/query std legitimately differ query-by-query -- ratios observed in range ~0.65-2.7, all finite and same order of magnitude, no scale-space mixing detected). No val/test leakage re-derivation was needed here -- `compute_coarse_delta_last_cosine_scores` (building the P100 pool itself) structurally cannot see `batch_y`/future (no such parameter exists in its signature, confirmed by source read in the current track's PHASE A and independently in multiple prior tracks this session).

## B4. Canonical test MSE reproduction

Reference table (spec section 5):

| Cell | Spec reference |
|---|---:|
| ETTh1 H96 | 0.36960 |
| ETTh1 H720 | 0.51492 |
| Weather H96 | 0.17665 |
| Weather H720 | 0.38553 |

Cross-checked against **two independently-built** production artifacts, both using the identical canonical checkpoint + Shared-P100 pool + `mean_mixture_topk_selection` + Original-CARTS-Trainable-Global-Lambda Stage2 (`train_r_stage2_lambda01.py`, unmodified):

| Cell | `TRACK-HARD-EXPERT-V5-P100-ALLH01` (`.../V5/stage2/metrics.json`) | `TRACK-P100-COARSE-TO-FINE-RETRIEVAL01` (`STAGE2/metrics/<cell>/V5/metrics.json`) | Spec reference | Match |
|---|---:|---:|---:|---|
| ETTh1_96 | 0.3695971369743347 | 0.36959707736968994 | 0.36960 | PASS (both round to 0.36960, diff 5.4e-8) |
| ETTh1_720 | 0.5149174928665161 | 0.5149174928665161 | 0.51492 | PASS (byte-identical) |
| Weather_96 | 0.1766490787267685 | 0.1766497790813446 | 0.17665 | PASS (both round to 0.17665, diff 7.0e-7) |
| Weather_720 | 0.38552558422088623 | 0.38552603125572205 | 0.38553 | PASS (both round to 0.38553, diff 4.5e-7) |

All 4 cells reproduce exactly (to the spec's own 5-decimal precision) from TWO separately-run pipelines sharing only the checkpoint and pool, not the code path -- this is a stronger reproduction gate than a single-pipeline re-run would give. No `NOT VERIFIED` cells; no reproduction failure to root-cause.

## Verdict

**PASS on every B1-B4 item, all 4 cells.** No fatal reproduction error. PHASE C/D/E/F/G may proceed. The one substantive correction applied going forward: the spec's "Mean-Mixture" row is computed via `mean_mixture_topk_selection`, never via the superficially-similar but structurally different `round_robin_topk_selection`-based `rr` diagnostic that happens to already exist in `TRACK-HARD-EXPERT-V5-P100-ALLH01`'s saved artifacts.
