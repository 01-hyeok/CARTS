# TRACK-T2-PROJECTION-MULTISLOT-DECOMPOSITION01 -- AUDIT.md

## The confound (PART 0)

TRACK-T's own S1 arm (`train_t_pure_multislot01.py --num_slots 1`) used
`SlotHeads(n_slots=1, std=1e-3)` -- a trainable `W_1 = I + eps_1`
projection matrix, optimizer-tracked and updated every step -- as its
"S=1" baseline. This is **not** TRUE Original KL: Original KL computes
`s(q,k) = cos(Eθ(X_q), Eθ(X_k))` directly, no projection layer at all.
So TRACK-T's own `S1 -> S2 -> S4 -> S10` comparison measured
"single-projected-head -> multi-slot", not "Original KL -> multi-slot".
This track adds a genuine T0 (zero projection parameters) to separate:

```
G_KL   = MSE(Cosine) - MSE(T0)     [reference only, not this track's focus]
G_Proj = MSE(T0) - MSE(T1)         [[projection]] effect
G_MS(S)= MSE(T1) - MSE(T_S)        [[multi-slot]] effect, S=2,4,10
```

## GPU (PART 2)

Standing project rule: **GPU1 only**, queue/wait, never move to another
GPU. Verified via `ps aux` + `nvidia-smi` immediately before starting:
no CARTS process running anywhere, GPU1 at 0% util / 4 MiB used --
fully free. Ran with `CUDA_VISIBLE_DEVICES=1` throughout.

## T1-T10: NOT retrained (PART 18)

`results/TRACK-T-PURE-MULTISLOT-VALIDATION01/` is kept strictly
read-only. T1 = that track's S1, T2/T4/T10 = its S2/S4/S10, read
directly by `scripts/compute_t2_bootstrap01.py` and by the summary
-building step. Only **T0** required new execution.

## T0 implementation (PART 5)

T0 is **not** `SlotHeads(n_slots=1, std=0)` (that would still have a
trainable, optimizer-tracked `W_1` parameter even if initialized to
identity -- exactly the confound this track exists to remove). Instead,
T0's training is `scripts/train_j_shared_encoder_drift01.py` run
**completely unmodified** (`--instrument off` only, for speed -- the
core training loop, loss, checkpoint-selection criterion, and score
computation are untouched) -- this is the actual historical Original-KL
/ J0 reference implementation, not a reimplementation, so there is zero
risk of T0 silently drifting from "true" Original KL. Its forward graph
is exactly: `encode_raw -> arm_score(., ., None)` (plain cosine),
`.detach()`-free on both branches, optimizer = `Adam(model.parameters(),
...)` only (no slot-head params exist to include). This script was
already used, unmodified, for TRACK-S's own "Original KL" arm, and
reproduced byte-identical (SHA-256 match) to the historical TRACK-J J0
checkpoint at ETTh1 H720 -- the same determinism guarantee applies here.

For Stage1-metric parity with T1-T10 (all built via
`train_t_pure_multislot01.py`'s `round_robin_topk_selection` +
`hard_eval_decomposition` + `spearman_batch`), a NEW cache builder
(`scripts/build_t2_true_original_kl_cache01.py`) reuses those same
functions unmodified, feeding them a `[B, 1, N]`-shaped score tensor
(`compute_scores_true_original_kl`, a thin wrapper around
`encode_raw`+`arm_score`, with an explicit runtime assertion that the
loaded checkpoint carries no `slot_heads_state_dict`) -- this size-1
"slot" axis is purely a data-shape convenience to reuse validated code;
`round_robin_topk_selection` with `s=1` is already proven (TRACK-T's own
unit test 3, and this track's own item 10) to reduce EXACTLY to ordinary
greedy Top-10 selection, so no selection-algorithm difference is
introduced.

## Reuse policy (read-only)

| Artifact | Source | Used by |
|---|---|---|
| `train_j_shared_encoder_drift01.py` | unmodified | T0 training |
| Base forecaster checkpoint | `checkpoints/track_r_final_method_generalization01/<D>/H<h>/seed<s>/base/checkpoint.pth` | T0's Stage2 (`--base_checkpoint`), same one T1-T10 already used |
| `train_r_stage2_lambda01.py` | unmodified | T0's Stage2 |
| T1/T2/T4/T10 caches, gates, checkpoints | `results/TRACK-T-PURE-MULTISLOT-VALIDATION01/.../{S1,S2,S4,S10}/` | bootstrap + summary tables, read-only |

## Equivalence tests (PART 6, 10 items)

All 10 written in `tests/test_t2_projection_multislot_decomposition01.py`.
9/10 pass immediately (synthetic/source-level checks); item 8 (same
batch order, T0 vs T1) is `skipif`-guarded on real artifacts and
verified after T0 actually runs (see report body).

## STOP/VALIDITY checks performed before proceeding (PART 19)

- T0 score/loss/gradient equivalence to the Original-KL reference
  pattern: verified analytically (items 2-4), PASS.
- T0 optimizer has no slot-head parameters: verified (item 5, source
  check on the literal `torch.optim.Adam(model.parameters(), ...)`
  line), PASS.
- Teacher, encoder init, batch order, candidate mask/memory, checkpoint
  criterion, Stage2 base checkpoint: all identical across arms by
  construction (T0 uses the same `build_model`/`build_experiment`
  overrides as T1-T10's own trainer, same reference_ckpt/pred_len/
  seq_len/seed inputs, same shared TRACK-R base checkpoint) -- verified
  empirically post-hoc via checkpoint-fingerprint and batch-order-hash
  comparison in the report body.
- H720 reproduction sanity check (PART 16): T0's freshly-trained Stage2
  MSE compared against TRACK-S's historical Original-KL value
  (≈0.529134) as a REFERENCE ONLY, not the main comparison -- see
  report body; if this reproduction had failed badly, execution would
  have stopped here per PART 19 -- it did not.
