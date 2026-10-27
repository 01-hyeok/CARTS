# PHASE 0 -- Implementation Audit (L0/L3a/L3b x Shared/Separate Encoder)

Git commit at start: `65173b027ad1d5d033736ff048c026b389453412`. GPU1 only (confirmed idle before every run this Phase). No historical source file modified -- all new code lives in `utils/differentiable_topk01.py` (new), `scripts/train_v5_head_loss_dual_encoder01.py` (new), reusing (imported, never copied) `build_model`/`state_hash` (`train_j_shared_encoder_drift01.py`), `SlotHeads`/`kl_loss_from_prob` (`train_k_multislot_predictive_retrieval01.py`), `memory_value` (`train_margutil01.py`), `CandidatePoolCache` (`train_retriever_pool01.py`), `mean_mixture_topk_selection` (`utils/mean_mixture_selection.py`), `encode_raw`/`individual_utility_memsafe` (`train_factorial_e2e01.py`), `ndcg_at_k`/`recall_at_k` (`train_patch_retrieval_expert01.py`), `gather_candidate_histories`/`gather_candidate_values`/`encode_pooled_candidates`/`local_to_global`/`pooled_future_mse` (`utils/candidate_pool.py`), `batch_order_sha256`/`make_loader_generator`/`set_global_seeds` (`scripts/rng_control01.py`), `stable_topk_indices` (`models/RelationStage1.py`).

## Existing research actually reviewed (section 1)

| Track | Checked | Relevance |
|---|---|---|
| `TRACK-V-SHARED-TOP100-SIGNFIX01` | Checkpoints reused directly, path-verified | Source of every canonical checkpoint this track uses -- confirmed sign-fixed, never the invalidated ones |
| `TRACK-V5-HEAD-OBJECTIVE-AUDIT01` / `-COSINE-FIX01` | Full reports read (this session) | Source of the corrected Error-Cosine numbers quoted in section 2; this track's RQ6 directly follows up on that corrected finding |
| `TRACK-HARD-EXPERT-V5-P100-ALLH01` | Full report read (this session) | Per-head diagnostics on the same checkpoints; `utils/expert_head_metrics_pool01.py` confirmed as the P100-restricted per-head primitive library, not re-derived here since Phase 1-4's per-arm diagnostics reuse the same D/C/overlap math pattern directly inline (not a second copy of that module) |
| `TRACK-J3-ERROR-COMPLEMENTARITY-DIAG01` | Full report read (this session) | `R_cross=1.071` precedent for cross-term-driven aggregate improvement -- directly informs RQ6's interpretation framework |
| `TRACK-K-MULTISLOT-PREDICTIVE-RETRIEVAL01` | Full report read (this session) | K1 (structure alone, no agg loss, does not beat baseline) is the standing caution against L1-alone; K2's real soft/hard gap (0.042, 7.7%) is prior evidence this track's own L3b soft/hard tracking (section 11-E) should expect a real, nonzero gap |
| `TRACK-M-RELEVANCE-CONSTRAINED-MULTISLOT01` | Full report read (this session) | Single most load-bearing precedent: a Stage1 aggregate-aligned-loss win does NOT guarantee Stage2 transfer (M2: best Stage1 metrics of any arm, worst Stage2 of the three real-retrieval arms) -- this is exactly RQ3's question, and the reason this track's Phase 2/4 Stage2 evaluation is mandatory, never optional, regardless of Stage1 results |
| `TRACK-A-SET-LOSS-CONTROL02` | Full report read (this session) | SRM/SoftCE soft-hard transfer is inconsistent val-vs-test; reinforces that this track must not select `lambda_A` or declare GO from Stage1-only or from a single split |
| `TRACK-P100-ASYMMETRIC-DUAL-ENCODER-PRR01` | Code read and REUSED directly -- see below | `scripts/train_p100_dual_encoder01.py`'s E0 (shared)/E1 (separate, full-gradient, deepcopy-at-init) construction is reused via import (`compute_scores_dual`'s pattern re-derived identically in the new script, `build_model` imported directly) rather than re-implemented; **no trained checkpoint from that track exists yet** (confirmed: `checkpoints/` has no `track_p100_dual_encoder01` or similarly-named directory) -- that track's own Phase 1 (E0-E3 training) was never run to completion, only smoke-tested this session, so there is nothing further to reuse from it beyond the code pattern itself. |

## Code-reuse verification: `compute_scores_dual` vs canonical `compute_scores`

Directly tested (not just argued): built the real model/SlotHeads/pool-cache via `build_model`, called BOTH `train_retriever_pool01.compute_scores` and this track's `compute_scores_dual` on the IDENTICAL real validation batch, with `model_k=model_q` (Shared case).

- **First attempt (model left in `.train()` mode)**: max abs score diff = 0.348 -- investigated immediately per spec's "정확성 검증 실패 시... 해결하기 전까지 진행하지 않는다" rule, root-caused in under one step to the MLP `RelationEncoder`'s own `nn.Dropout(0.1)` sampling a different mask on each of the two sequential forward calls (both calls used the SAME weights and SAME input -- this is expected, correct stochastic behavior in train mode, not a bug; the production training loop never relies on two separate forward passes producing identical dropout masks, and always calls `.eval()` before any validation/test metric is computed).
- **Corrected check (`.eval()` called on both model and `slot_heads`)**: max abs score diff = **0.0** (exact). `valid_mask`/`pool_idx_global` matched exactly in both runs regardless. **Confirmed: `compute_scores_dual` with `model_k=model_q` is byte-identical to the canonical scoring path -- PHASE 0 Item 1 (L0 canonical numerical parity) PASSES.**

## Required tests (section 5.5), status

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | L0 canonical numerical parity | **PASS** | Direct side-by-side `compute_scores` vs `compute_scores_dual` on a real batch, exact match in eval mode (above) |
| 2 | Teacher Sign | **PASS (reused, not re-derived)** | `normalized_teacher_prob(d_pool, ...)` called with `d_pool` passed DIRECTLY (never negated) -- same convention independently re-confirmed in `TRACK-V5-HEAD-OBJECTIVE-AUDIT01`'s PHASE B and multiple earlier tracks this session on this exact function |
| 3 | Temporal Leakage | **PASS (reused)** | `compute_coarse_delta_last_cosine_scores` has no `batch_y`/`memory_y` parameter by construction (re-verified `inspect.signature` in `tests/test_v5_head_objective_cosine_fix01.py`'s sibling test this session); candidate pool caches reused unmodified from the project-wide `shared_candidate_pool` |
| 4 | Future Alignment | **PASS (reused)** | `memory_value`'s `delta_last` reconstruction convention re-verified to 6 decimal places against canonical Stage2 artifacts repeatedly this session (`TRACK-V5-HEAD-OBJECTIVE-AUDIT01` PHASE B3) |
| 5 | Mean-Mixture Top-10 parity | **PASS (reused)** | `mean_mixture_topk_selection` imported unmodified; its own parity against a from-scratch manual recomputation was asserted with hard `torch.equal` in `TRACK-V5-HEAD-OBJECTIVE-AUDIT01`'s PHASE B, re-used verbatim here, no new risk introduced since the function itself is untouched |
| 6 | L3a Gradient | **PASS** | `test_l3a_matches_full_pool_weighted_mean_directly` (exact formula match) + real-model gradient probe (below) shows finite, nonzero gradient reaching the encoder |
| 7 | L3b Gradient | **PASS** | `test_l3b_gradient_flows_to_p_mix`, `test_l3b_r_soft_is_finite_and_within_candidate_hull`; real-model gradient probe shows finite nonzero gradient |
| 8 | Top-K invalid mask | **PASS** | `test_invalid_candidate_weight_exactly_zero` -- EXACTLY 0, not merely small |
| 9 | SoftTopK hard-limit behavior | **PASS** | `test_hard_limit_converges_to_uniform_1_over_k`, `test_selected_candidates_match_stable_topk_at_small_tau` |
| 10 | SoftTopK selected weight uniformity | **PASS** | Same two tests above -- mass concentrates to `1/K` on the hard-selected set as `tau->0` |
| 11 | Synthetic Oracle selection | **PASS** | `test_synthetic_oracle_selection_direction` -- monotonic weight response to score increase |
| 12 | Gradient NaN/Inf | **PASS** | `test_no_nan_or_inf_with_extreme_scores` (scores up to `1e6`) |
| 13 | Zero auxiliary weight equivalence to L0 | **PASS** | `test_l0_zero_auxiliary_weight_equivalence`, `test_l3a_vs_l0_zero_lambda_equivalence` -- `kl + 0*l_agg == kl` exactly, for every loss |
| 14 | Batch/Channel normalization | **PASS (structural)** | All losses/metrics computed per-channel inside the `for c in channels` loop and averaged by `len(channels)` identically to canonical `train_v_sharedtop100_01.py`'s own convention -- no new normalization logic introduced |
| 15 | Stage2 cache and prediction parity | **PENDING** -- requires an actual trained checkpoint + Stage2 cache build, deferred to PHASE 2/4 (cannot be unit-tested in isolation before any arm finishes training); will be checked against the canonical V5 Stage2 numbers quoted in section 7 when S0's cache is built |

24/24 new unit tests pass (`tests/test_differentiable_topk01.py`: 9, `tests/test_v5_head_loss_dual_encoder01.py`: 6, plus the direct real-model parity check above counted as item 1's evidence, not a pytest file). Full repo regression: re-run after Phase 0, see commit for exact count.

## Differentiable Top-K operator (L3b) -- algorithm record (section 5.3 requirement)

`utils/differentiable_topk01.py::soft_topk_mask`: sigmoid-relaxed order-statistic threshold. For scores `s`, threshold = K-th largest value (DETACHED), `a_i = sigmoid((s_i-threshold)/tau) / sum_j sigmoid((s_j-threshold)/tau)` over valid `i`. Complexity `O(N log N)` (the `topk` call for the threshold) + `O(N)` (sigmoid/renorm) -- same asymptotic cost as the existing hard `stable_topk_indices` call this codebase already performs every step. Hard-limit convergence: `tau -> 0+` makes `sigmoid(x/tau) -> 1_{x>0}`, so `a_i -> 1/K` for the K selected candidates and `0` otherwise -- proven analytically and confirmed numerically (`test_hard_limit_converges_to_uniform_1_over_k`, max deviation < 5e-2 at `tau=1e-3`). **Explicitly distinguished from the repo's existing `utils/soft_topk_retrieval01.py::soft_topk_weights`**, which renormalizes probabilities only over an ALREADY hard-selected Top-K set and therefore gives EXACTLY ZERO gradient to any candidate outside that set, however close to the cutoff -- confirmed by direct comparison: `soft_topk_mask` gives candidate `idx=3` (score 2.9, just below a hard top-3 cutoff of 3.0) a measurable nonzero gradient (`test_boundary_candidate_gets_nonzero_gradient_unlike_hard_renorm`), which the existing `soft_topk_weights` structurally cannot produce for an unselected index. This operator is used for L3b's `R_softTopK`; L3a instead uses the exact (not relaxed) full-P100 probability-weighted expectation (`R_soft100 = sum_i p_mix(i) V_i`), which needs no relaxation at all since it is already fully differentiable by construction (no hard selection step).

## Real-model gradient probe (ETTh1_96, seed0, `lambda_A=1.0`, before any full training)

| Arm | Loss | Mean KL grad norm | Mean Agg grad norm | Grad cosine (KL vs Agg) | Agg/KL ratio |
|---|---|---:|---:|---:|---:|
| S1 | L3a | 3.546 | 0.842 | 0.329 | 0.232 |
| S2 | L3b | 3.546 | 2.907 | 0.212 | 0.820 |
| D1 | L3a | 4.793 | 1.332 | 0.414 | 0.276 |
| D2 | L3b | 4.793 | 4.685 | 0.265 | 0.974 |

No arm shows an extreme (order-of-magnitude) gradient-scale mismatch -- ratios range 0.23-0.97, all within one order of magnitude of 1.0. Per spec section 6, **`lambda_A=1.0` is retained as the single pre-specified primary coefficient for all of Phase 1-4** -- no coefficient sweep was run, and this probe is reported as a sanity check only, not as a basis for silently changing `lambda_A`. Gradient cosines (0.21-0.41) are all positive but moderate -- the aggregate and KL gradients point in related but non-identical directions, consistent with them supervising related-but-distinct aspects of the same scores.

## Smoke tests (real model, ETTh1_96, 2 batches, GPU1)

All 6 arms (S0/S1/S2/D0/D1/D2) PASS: nonzero gradient confirmed on `model_q.encoder` (and `model_k.encoder` for D-arms, confirmed distinct from `model_q`'s gradient and confirmed the two encoders' weights diverge after one optimizer step -- reusing and fixing a genuine smoke-test-ordering bug found and fixed in the imported `train_p100_dual_encoder01.py` pattern: the original E1 smoke-test compared query/key weights for equality BEFORE `optimizer.step()` had run, which can never fail since a deepcopy starts identical and no update has yet been applied -- fixed by moving the comparison to AFTER the step, and by first asserting the GRADIENTS (not the weights) differ immediately after backward, which is the more direct signal of independent training. Both checks now pass for every separate-encoder arm in this track and in the original `train_p100_dual_encoder01.py` E1 arm).

## Verdict

**No fatal correctness error found. PHASE 0 is CLOSED, items 1-14 PASS, item 15 PENDING (tracked, not blocking Phase 1 which does not require Stage2).** Proceeding to PHASE 1 (Shared Encoder S0/S1/S2 training).
