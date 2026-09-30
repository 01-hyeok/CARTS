# TRACK-S-KL-CONTRIBUTION-DECOMPOSITION01 -- Audit

Runs in PARALLEL with TRACK-R (never stopped/modified/shared output).
Decomposes the observed forecasting gain into three sequential
architecture-ablation steps: Raw Cosine -> Original KL (S1) -> StopGrad
-Key KL (S2=J1) -> Multi-Slot M2 (S3=M2). No new loss/teacher/slot
design -- pure attribution of gains already obtained.

## PART 1: GPU allocation

`nvidia-smi` at track start:

| GPU | Util | Mem used/total | Note |
|---|---|---|---|
| 0 | 27% | 3226/81920 MiB | other/unrelated process, untouched |
| 1 | 66% | 3126/81920 MiB | **TRACK-R** (Weather H96 seed0 J1 training) -- never touched |
| 2 | 0% | 4/81920 MiB | **fully free -> TRACK-S uses this GPU** |
| 3 | 0% | 0/4096 MiB | display GPU, not used |
| 4 | 0% | 57854/81920 MiB | high memory held but idle (other process), not used |

TRACK-S runs with `CUDA_VISIBLE_DEVICES=2` throughout. TRACK-R's GPU1
process was never killed, throttled, or otherwise interfered with.

## PART 17: Original KL (S1) implementation audit

`scripts/train_j_shared_encoder_drift01.py` (TRACK-J's own "J0" arm) is
read directly, not assumed. Confirmed exactly matching S1's PART 3
definition, at the actual training loss (main loop, lines 444-455 as of
this commit):

```python
z_q = encode_raw(model, batch_x, c)          # query branch, gradient ON
E = encode_raw(model, exp.memory_x, c)        # candidate branch, gradient ON (no .detach())
s = arm_score(z_q, E, None)                   # cosine student score
u = individual_utility_memsafe(...)           # future-MSE teacher utility
p_t = normalized_teacher_prob(-u, cand_mask, tau_t)
ch_loss = kl_loss(p_t, s, cand_mask, tau_s)   # KL(p_t || p_s)
ch_loss.backward()
```

All 12 PART 17 checklist items confirmed: (1) single shared encoder --
yes, one `model` object for both `z_q`/`E`; (2) query gradient ON --
yes, no detach; (3) candidate gradient ON -- yes, `E` is NOT detached
(this is the entire S1-vs-S2 difference); (4) candidates re-encoded
every step through the current (not frozen) encoder -- yes, `E =
encode_raw(model, exp.memory_x, c)` called fresh every batch; (5)
teacher = future-MSE -- yes, `individual_utility_memsafe`, identical
function reused throughout this session; (6) loss = KL -- yes,
`kl_loss` (`train_horizon_retrieval_expert01`), the exact function
`train_j2_key_update_decomposition01.py` (J1/J2) also imports; (7)
scoring = cosine -- yes, `arm_score(., ., None)`; (8) `tau_t=tau_s=0.1`
CLI defaults, identical to J1/M2; (9) candidate mask/universe -- same
`exp._candidate_mask`/`exp.memory_x`, same `raft` convention; (10) input
space -- same `build_model`/`build_experiment` pipeline, `delta_last`
throughout; (11) no future leakage at inference -- Top-K selection uses
only `s` (student cosine score), future used only in the teacher target
(training-time only); (12) checkpoint selection -- `val_metric =
val_sums['model_top10_individual_mse']` (val retMSE@10), **identical
criterion** to J1/J2's own `val_metric = val_metrics['model_top10_individual_mse']`
(`train_j2_key_update_decomposition01.py` line 335) -- PART 10/18's
"same criterion" requirement holds with ZERO reconciliation needed, not
just audited-and-matched.

**No `EXPECTED_INIT_HASH`-style hardcoded constant exists in this
script at all** (grepped, none found) -- unlike J1/M2's scripts, no
`--skip_init_hash_check` flag is needed; `train_j_shared_encoder_drift01.py`
is already safe to reuse unmodified across any horizon/seed. Its own
internal init-hash assertions (`assert init_hash == e0_hash`) are
self-referential (both computed from the just-constructed model in the
SAME run), not compared against any external/hardcoded value.

## PART 18: S1 vs S2 controlled comparison

Since `train_j_shared_encoder_drift01.py` (S1) and
`train_j2_key_update_decomposition01.py` (S2=J1) both call the SAME
`build_model` (`train_j_shared_encoder_drift01.build_model`, imported
unmodified by the J2 script) with identical `--init_seed`/`--loader_seed`,
both produce the IDENTICAL shared-encoder initial weights and IDENTICAL
batch order for the same (horizon, seed) -- confirmed by the session
-wide `EXPECTED_INIT_HASH` constant
(`b37fa4031f538e4b5f5c522ae22e7d03613e4ee66283d2637656c7c4f872372e`)
that J1/M2 assert against for ETTh1_720/seed0. No auxiliary paired rerun
is needed: this is a genuinely controlled comparison by construction,
not merely audited after the fact. The ONLY intended difference is
`E.detach()` (S2) vs no detach (S1) at the score-computation line.

## PART 20: reuse policy (TRACK-R artifacts, read-only)

| Component | Source (read-only) |
|---|---|
| Base forecaster (`B`) | TRACK-R's `checkpoints/track_r_final_method_generalization01/<dataset>/<H>/seed<seed>/base/checkpoint.pth` |
| Cosine cache/Stage2 | TRACK-R's `results/TRACK-R.../<dataset>/<H>/seed<seed>/cosine/` |
| J1 (S2) checkpoint/cache/Stage2 | TRACK-R's `.../J1/` |
| M2 (S3) checkpoint/cache/Stage2 | TRACK-R's `.../M2/` |

TRACK-S's only NEW heavy computation: Original KL (S1) Stage1 training,
its retrieval cache, and its Stage2 global-lambda training -- all via
`scripts/run_s_original_kl01.sh`, entirely independent output/checkpoint
directories (`results/TRACK-S-KL-CONTRIBUTION-DECOMPOSITION01/`,
`checkpoints/track_s_kl_contribution_decomposition01/`).
