# TRACK-A-HORIZON-RETRIEVAL-EXPERT01 — INTERIM STATUS (IN PROGRESS)

**Status: NOT COMPLETE.** This is a mid-experiment status snapshot, not a final
report. Do not treat any number here as a closed result. A final report will
replace this document once Solar_720 Stage-1, all three cells' Oracle-contrast
diagnostics, and the full Stage-2 phase are done.

Generated: 2026-09-22. Code state: reuses the S0_wce host-building pipeline and
`train_factorial_e2e01.py` primitives unmodified; new files this experiment:
`scripts/train_horizon_retrieval_expert01.py`,
`scripts/diag_horizon_expert01_oracle_contrast.py`,
`scripts/run_horizon_retrieval_expert01.sh`,
`tests/test_horizon_retrieval_expert01.py` (7/7 passing).

## 1. What this experiment is

First LEARNED horizon-aware retriever, following the pure-Oracle diagnostic
`TRACK-A-HORIZON-RETRIEVAL-HEADROOM01` (which found that a single Top-K
retrieved set does not suffice for the full H=720 horizon; block-specific
Top-K sets recover 25.7% (ETTh1_720) / 39.4% (Weather_720) relative MSE vs a
single global Top-K).

- **Global arm**: single scratch-trained scoring head, `s_G(q,i) = cosine(z_q, e_i)`.
- **Block arm**: shared encoder + 3 zero-init `BlockCorrectionHeads`
  (`s_b(q,i) = cosine(z_q + W_b(z_q), e_i)`), trained jointly with a Global
  term. `s_b == s_G` exactly at init (verified by test).
- **Loss**: forward KL between a z-score-normalized teacher (per-block future
  MSE distribution) and the softmax student score distribution, teacher
  detached. Global loss = `KL_G`. Block loss = `0.5*KL_G + (KL_B1+KL_B2+KL_B3)/6`.
  Gradient-equivalence to soft cross-entropy verified by
  `tests/test_horizon_retrieval_expert01.py::test_kl_gradient_equals_soft_ce_gradient`.
- **Checkpoint selection**: future-blind hard Top-10 uniform-aggregate H720
  MSE on val (NOT the KL loss value).
- Cells: ETTh1_720, Weather_720, Solar_720 (K=10, seed=0, single seed only —
  no reproducibility claim yet).

## 2. Completed: ETTh1_720 (Stage-1, test set)

| Arm | global_h720_mse | block_h720_mse |
|---|---|---|
| Global (baseline) | 0.5644 | — |
| Block (learned correction) | 0.5609 (own global head) | **0.5691** |

Block arm's own block-aggregate (0.5691) is **worse** than the Global arm's
plain global prediction (0.5644) — the learned block correction does not help
on ETTh1_720, and mildly hurts.

Pre-training Oracle contrast (greedy approximate search, seed=0, 256-query
subset, `results/TRACK-A-HORIZON-RETRIEVAL-EXPERT01/ETTh1_720/oracle_contrast_greedy.json`):
- individual-ranked oracle: 0.2976
- greedy uniform-aggregate oracle: 0.2382
- block oracle: 0.2200

The learned models (0.56-0.57) are far from any of these oracle levels —
almost none of the diagnosed headroom is recovered.

## 3. Completed: Weather_720 (Stage-1, test set)

| Arm | global_h720_mse | block_h720_mse |
|---|---|---|
| Global (baseline) | 0.5172 | — |
| Block (learned correction) | 0.5218 (own global head) | **0.4532** |

Block arm's block-aggregate (0.4532) beats the Global arm's global prediction
(0.5172) by 12.4% relative — direction consistent with the Oracle diagnostic's
39.4% headroom finding, but the learned result is well short of that oracle
level (block breakdown: block1=0.1893, block2=0.3981, block3=0.5536).

Weather_720 Oracle-contrast diagnostic has **not** been re-run yet (first
attempt OOM'd from GPU1 contention with the Solar_720 job; deferred rather than
risking the priority training job).

## 4. Common observation across all 4 completed (cell x arm) runs

All four (ETTh1/Weather x global/block) picked **best_epoch=1** by the
future-blind checkpoint-selection metric; val metric monotonically worsens
from epoch 2 onward. Whether this reflects (a) a mis-scaled loss/LR causing
immediate overfitting, (b) a loss that trains ranking but not aggregate MSE,
or (c) something else, is **not yet diagnosed** — this is exactly the kind of
question the final report's required failure-mode classification must answer,
and is deferred until Solar_720 completes so all three datasets can be
compared together.

## 5. In progress: Solar_720 (global arm)

Currently training on GPU1 (`--channelwise_backward`, after fixing an
orchestrator bug that had dropped this flag and caused an earlier OOM).
Through epoch 5:

| epoch | train_loss | val_primary_mse |
|---|---|---|
| 1 | 0.23517 | **0.182211** (best so far) |
| 2 | 0.21323 | 0.182715 |
| 3 | 0.20524 | 0.185257 |
| 4 | 0.19934 | 0.188163 |
| 5 | 0.19442 | 0.189539 |

Same "epoch 1 best, then worsens" pattern as ETTh1/Weather. patience=5, so
likely to early-stop soon. Block arm for Solar_720 has not started yet (runs
after global arm completes, using shared init).

## 6. Separate track, not part of this experiment: Solar_96 TF-Oracle-Learnability01

For context only — `TRACK-A-TF-ORACLE-LEARNABILITY01`'s Solar_96
`individual_tf_cosine` training (a different experiment, different script) is
also running concurrently on GPU1. Diagnosed cause of its slow pace: per-batch
per-channel full-candidate-bank re-encoding inside `teacher_forced_diagnostics()`
plus a non-vectorized per-row Spearman loop in `step_rank_diagnostics()`,
compounded by genuine GPU1 contention with this experiment's Solar_720 job
(both processes share the same physical GPU1, confirmed via `nvidia-smi`).
User decision (2026-09-22): **leave it running as-is, no intervention.**

## 7. Not yet done

- Solar_720 Stage-1 completion (global arm still running; block arm not started).
- Oracle-contrast diagnostic re-run for Weather_720; first run for Solar_720.
- Entire Stage-2 phase (delta-space cache, common frozen base, fresh fusion
  training per arm, λ=0/λ=1 counterfactuals) — not started for any cell.
- Final `research/TRACK-A-HORIZON-RETRIEVAL-EXPERT01.md` report with the
  mandated structure (loss actually implemented, Global/Block controls,
  3-dataset Stage-1 table, Stage-2 results, headroom recovery, failure-mode
  classification).
- Any seed=1/2 repeat plan (deferred until seed=0 is complete across all
  three datasets, per explicit no-single-seed-reproducibility-claim rule).

No interpretation or next-direction recommendation is offered here by design —
that is the reviewer's role per `CLAUDE.md`; this document only records what
has run and what the raw numbers say.
