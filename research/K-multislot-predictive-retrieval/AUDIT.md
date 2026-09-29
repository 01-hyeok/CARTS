# TRACK-K-MULTISLOT-PREDICTIVE-RETRIEVAL01 -- Audit

Builds directly on TRACK-J2's J1 (StopGrad-Key) architecture and
TRACK-J/J2/J3's shared function library -- all base hyperparameters and
helper functions re-verified from the actual code paths already audited
in `research/J-shared-encoder-drift/AUDIT.md` and re-confirmed at the
start of TRACK-J2/J3 (`config.json`, init hash
`b37fa4031f538e4b5f5c522ae22e7d03613e4ee66283d2637656c7c4f872372e`,
`tau_t=tau_s=0.1`, `batch_size=32`, `lr=1e-3`, `train_epochs=10`,
`patience=5`, `candidate_mask=raft`, `delta_last` throughout, N=7201
train-only candidates, 7 channels, `checkpoint_criterion` changed this
round per spec section 16 to **hard** `uniform_agg_mse10` -- not
`model_top10_individual_mse` -- documented explicitly below since this
is the one deliberate deviation from J0/J1/J2's own criterion).

## Reused unmodified

- `build_model`, `state_hash` (`train_j_shared_encoder_drift01`)
- `encode_raw`, `arm_score`, `individual_utility_memsafe`
  (`train_factorial_e2e01`)
- `normalized_teacher_prob`, `kl_loss` (`train_horizon_retrieval_expert01`)
- `memory_value` (`train_margutil01`)
- `stable_topk_indices` (`RelationStage1`)
- `recall_at_k`, `ndcg_at_k` (`train_patch_retrieval_expert01`)
- `make_loader_generator`, `batch_order_sha256` (`rng_control01`)
- J1's key-branch pattern (candidates re-encoded every step from the
  current encoder, then `.detach()`-ed before any score/loss use) --
  reused exactly, single encoder, no second key encoder (unlike J2).
- `D`/`C` aggregate-MSE decomposition identity from TRACK-J3
  (`diag_j3_error_complementarity01.py`'s per-query math, reimplemented
  inline here since it needs to run on the hard multi-slot-selected
  Top-10, not J0/J1's single-slot Top-10 -- same formula, same identity
  check).

## New this round

- `SlotHeads`: `S=10` independent `D x D` linear maps `W_m`, no bias,
  initialized `W_m = I + eps_m` (`eps_m ~ N(0, 1e-3^2)`, `torch.manual_seed(1000+m)`
  per slot for a fully deterministic, seed-controlled, non-identical
  perturbation -- verified in a unit test that no two slots are
  bit-identical and that step-0 retrieval is close to, but not
  identical to, the single-head J1 baseline).
- Candidate encoding happens ONCE per training step (`K = detach(E_theta(memory))`),
  reused for all 10 slots' score computation via one batched
  `Q @ K^T` -- never re-encoded per slot.
- Collective distribution `p_bar = mean_m softmax(s_m/tau_s)`, anchored
  to the SAME future-MSE teacher via `KL(p_T || p_bar)` -- teacher
  itself is untouched (same `normalized_teacher_prob`/`individual_utility_memsafe`
  call as every other track this session).
- Slot overlap penalty: mean pairwise cosine similarity between the 10
  slot *distributions* `p_m` (not between `W_m` parameters).
- Differentiable aggregate loss (K2 only): per-slot future-blind
  Top-32 (`L_soft=32`) subset selection from the slot's OWN current
  score row, tempered softmax within that subset, weighted candidate-
  future sum, averaged over slots, MSE against `Y_q`. Implemented via
  `gather` on the Top-32 indices -- never materializes `[B,S,N,H]`
  (confirmed by explicit shape assertions in the training loop and a
  dedicated unit test).
- Hard inference: deterministic greedy-unique selection in fixed slot
  order (slot 0 -> 9), each slot's own score row, excluding all
  already-picked candidates from earlier slots in the same query --
  never uses future information (mask is candidate-validity only, not
  future-based).

## Deliberate deviations from J0/J1/J2, documented

- Checkpoint criterion: hard `uniform_agg_mse10` (not
  `model_top10_individual_mse`) -- required by spec section 16, since
  this track's whole purpose is retrieved-SET quality, not per-candidate
  ranking quality. `retMSE@10` is still recorded every epoch for
  comparison, just never used to pick the checkpoint.
- Ten score rows per query instead of one -- everything else in the
  training loop (optimizer, LR, batch size, epochs, patience, candidate
  mask, value space, teacher) is identical to J1.
