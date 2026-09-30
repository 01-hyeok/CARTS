# TRACK-T-PURE-MULTISLOT-VALIDATION01 -- AUDIT.md

## PART 1: GPU allocation

`nvidia-smi` checked directly (3 samples, 4s apart) immediately before
starting this track, with TRACK-R (GPU1) and TRACK-S (GPU2, idle after
its own 2 completed settings) both still live:

| GPU | util | mem used / total | status |
|---|---:|---:|---|
| 0 | ~29% (stable) | 3226 / 81920 MiB | external/unrelated process, stable background load across all 3 samples -- NOT free, left untouched |
| 1 | 72-93% (variable) | 2002 / 81920 MiB | **TRACK-R** (Weather Stage1 M2 training) -- MUST NOT be touched |
| 2 | 32-90% (variable) | 7026 / 81920 MiB | newly busy with an external process not started by this session (was free during TRACK-S's own runs earlier this session; re-checked now and found busy) -- NOT free, left untouched |
| 3 | 0% | 0 / 4096 MiB | display GPU, only 4GB total -- too small for training, not used |
| 4 | 0% (stable, all 3 samples) | 57854 / 81920 MiB | **TRACK-T allocated here**: compute fully free (0% util) despite ~24GB held by an idle other process; ~24GB free is ample for this track's small MLP-encoder models (session-wide precedent: comparable tracks use a few GB) |

**Revised decision (user correction, 2026-09-30): GPU1-only standing
rule applies -- TRACK-T does NOT get its own GPU.** The initial choice
of GPU4 (documented above) was based on reading this track's own PART
"실행 순서" line ("기존 TRACK-R/TRACK-S와 충돌하지 않는 GPU에서 병렬
실행") as authorizing a separate GPU, the way TRACK-S's spec had done
explicitly and repeatedly. The user corrected this: the project's
standing rule is GPU1-only, always, with no per-track exceptions unless
stated with TRACK-S's level of explicitness. The GPU4 run (S1, ~5 min
elapsed, no checkpoint reached `best`) was killed and its partial
artifacts deleted. Separately, TRACK-R's own run (Weather H96 seed0) had
in the meantime crashed on the M2 arm (infeasible checkpoint, recorded
as a negative result, see `results/TRACK-R-FINAL-METHOD-GENERALIZATION01/
Weather/H96/seed0/M2/NEGATIVE_RESULT.md`) and released GPU1 on its own.
**TRACK-T now runs with `CUDA_VISIBLE_DEVICES=1`**, queued to start only
once GPU1 was confirmed free of any TRACK-R process (verified via `ps
aux` + `nvidia-smi` immediately before launch).

## PART 28: pre-execution code audit (required 10 items)

1. **Original KL full-gradient implementation location**:
   `scripts/train_j_shared_encoder_drift01.py` lines 444-456 (main
   training loop). Per channel: `z_q = encode_raw(model, batch_x, c)`;
   `E = encode_raw(model, exp.memory_x, c)` -- candidate bank re-encoded
   every step, **no `.detach()`** anywhere on `E`. `s = arm_score(z_q, E,
   None)` (plain cosine). `p_t = normalized_teacher_prob(d, cand_mask,
   tau_t)`. `ch_loss = kl_loss(p_t, s, cand_mask, tau_s)`. Checkpoint
   criterion: `min val model_top10_individual_mse` (i.e. min val
   retMSE@10, NOT AggMSE).

2. **Where StopGrad-Key enters existing Multi-Slot code**:
   `scripts/train_k_multislot_predictive_retrieval01.py` line 138,
   inside `compute_scores`:
   `k_full = F.normalize(encode_raw(model, memory_x, c), dim=-1).detach()
   # [N, D], stopgrad (J1-style)`. This one `.detach()` is the entire
   difference between TRACK-K/TRACK-M's Multi-Slot (StopGrad-Key
   ancestor) and what TRACK-T needs (full-gradient ancestor).

3. **Was that StopGrad removed for this track?** YES.
   `scripts/train_t_pure_multislot01.py::compute_scores_full_grad` is a
   NEW function (TRACK-K's `compute_scores` is NOT reused for score
   computation) that is byte-identical to TRACK-K's `compute_scores`
   except the `.detach()` is absent: `k_full = F.normalize(encode_raw(
   model, memory_x, c), dim=-1)` -- gradient flows through the candidate
   encoder exactly as it does in Original KL/J0's own `E =
   encode_raw(...)` (no detach). Confirmed by unit test 7 (`grep`-level
   source check) and unit test 5 (candidate-side gradient nonzero,
   numerically).

4. **Teacher identical across T0/T1/T2/T3?** YES. Every arm calls
   `normalized_teacher_prob(-individual_utility_memsafe(...), cand_mask,
   cli.tau_t)` -- same function, same `tau_t=0.1` default, no
   `num_slots`-dependence anywhere in the teacher computation path
   (teacher only sees `batch_x`/`batch_y`/candidate future values, never
   the score or slot count).

5. **Candidate mask identical?** YES. All arms call
   `exp._candidate_mask(batch_start_idx)` (RAFT mask, same
   `build_experiment` override `candidate_mask='raft'` used by every
   other track this session).

6. **Preprocessing identical?** YES. `build_model` (reused unmodified
   from `train_j_shared_encoder_drift01`) applies the SAME
   `build_experiment` overrides (`relation_encoder_type='mlp'`,
   `relation_self_fill='linear'`, `relation_input_space='delta_last'`,
   `relation_teacher_space='delta_last'`, `relation_value_space=
   'delta_last'`, `candidate_mask='raft'`) for every `--num_slots` value
   -- `num_slots` is consumed ONLY by `SlotHeads`'s constructor and
   `compute_scores_full_grad`'s einsum, never by `build_model`.

7. **Checkpoint-selection criterion identical across T0-T3, and does it
   match Original KL's own criterion?** YES to both, by deliberate
   choice: all four arms select on **min val retMSE@10** (`val_retmse10`
   = mean individual MSE of the round-robin-selected Top-10),
   numerically identical in definition to Original KL/J0's own
   `model_top10_individual_mse` criterion (item 1). This was chosen
   over an AggMSE-based criterion (which TRACK-K/TRACK-M use) because
   PART 4 requires T0 to be "as exact a reproduction of Original KL as
   possible," and because AggMSE is exactly the quantity this track is
   trying to measure the EFFECT of S on -- selecting checkpoints on
   AggMSE would let checkpoint selection itself, not the architecture,
   produce any S>1 advantage.

8. **Is S=1 exactly equivalent to Original KL?** Architecturally as
   close as PART 7's "same initialization policy regardless of S"
   permits: `SlotHeads(n_slots=1, std=1e-3)` applies the SAME formula
   (`W_0 = I + eps_0`, `eps_0 ~ N(0, std^2)`, seed=1000) used for every
   other slot in every other arm -- so the actual T0 production run is
   NOT bit-identical to a literal `cos(z_q, z_k)` Original KL score (a
   ~1e-3-scale perturbation is present). This is a **documented,
   deliberate scope resolution**: PART 7 explicitly forbids varying the
   init policy by S, and the perturbation is negligible in scale.
   Separately, unit tests 1 and 2 verify the UNDERLYING CODE PATH
   reduces EXACTLY to raw-cosine Original KL score/loss when `std=0`
   (an isolated test-only configuration, not the production run) --
   this validates that the Multi-Slot formalism itself introduces no
   bug beyond the intentional small perturbation, consistent with this
   session's established pattern of testing algebraic reductions in
   isolation (e.g. TRACK-P's mixture-fusion lambda=0/1 idempotence
   tests).

9. **Is round-robin S=1 exactly equal to standard Top-10?** YES, by
   construction: `round_robin_topk_selection` with `s=1` always computes
   `m = t % 1 = 0`, so every one of the 10 iterations reads slot 0's OWN
   score row and greedily picks its best still-available candidate --
   this is precisely the greedy-iterative-argmax algorithm for ordinary
   Top-10 selection. Verified exactly (index-set equality) by unit
   test 3.

10. **Is the common Stage2 base the exact same checkpoint hash across
    T0-T3?** YES. `scripts/run_t_one_setting01.sh` resolves ONE
    `R_BASE_CKPT` path (TRACK-R's already-trained, per-setting frozen
    base forecaster, `checkpoints/track_r_final_method_generalization01/
    <Dataset>/H<horizon>/seed<seed>/base/checkpoint.pth`) once, aborts
    if it does not exist, and passes that SAME variable to
    `--base_checkpoint` for all 4 `train_r_stage2_lambda01.py` Stage2
    calls in its `for S in 1 2 4 10` loop -- never retrained, never
    copied, never modified (verified by unit test 15: reads the actual
    checkpoint file, confirms `model_state_dict` hash is deterministic,
    and confirms the driver script's source references the identical
    `$R_BASE_CKPT` variable in all 4 invocations).

## Reuse policy (read-only from TRACK-R)

| Artifact | Source | Used by |
|---|---|---|
| Base forecaster checkpoint | `checkpoints/track_r_final_method_generalization01/<D>/H<h>/seed<s>/base/checkpoint.pth` | All T0-T3 Stage2 (`--base_checkpoint`) |
| `train_r_stage2_lambda01.py` | unmodified | Stage2 for all T0-T3 (identical Uniform aggregation + Mixture fusion + Trainable Global Lambda) |

TRACK-T never retrains a base forecaster and never modifies any
TRACK-R/TRACK-S artifact. `checkpoints/track_r_final_method_generalization01/`
and `results/TRACK-R-FINAL-METHOD-GENERALIZATION01/` are read-only from
this track's perspective throughout.

## Checkpoint-criterion divergence note (documented per item 7 above)

TRACK-K (K1/K2) and TRACK-M (M1/M2) select Multi-Slot checkpoints on
**val AggMSE** (K2: `min val agg_mse10`; M2: constrained `min val agg_mse10
s.t. val_retmse10 <= 1.05*R_val_J1`). TRACK-T deliberately selects on
**val retMSE@10** instead, to keep the checkpoint-selection criterion
identical to Original KL/J0's own and identical across T0-T3, isolating
the effect of `num_slots` on the (never-selected-for) AggMSE outcome.
This is a between-track design difference, not an inconsistency within
TRACK-T.
