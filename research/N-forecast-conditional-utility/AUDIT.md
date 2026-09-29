# TRACK-N-FORECAST-CONDITIONAL-UTILITY01 -- Code/Checkpoint Audit

NO TRAINING. This diagnostic decomposes WHY TRACK-M's M2 (best Stage1
retrieval quality of any arm) was significantly worse than J1 in Stage2
forecasting, using only existing frozen checkpoints and existing,
unmodified aggregation/evaluation code.

## Commits read

- TRACK-L @ `9cd3a824bdc94bf47b42c446722b886dfafd537e` --
  `scripts/eval_l_aligned_population01.py`: the unified Stage1 evaluator.
  Reused here for the Uniform-aggregation reproduction gate (PART 7):
  `score_and_decompose` computes `y_sel = memory_c[model_idx] +
  offset_c.view(-1,1,1)`, `agg_pred = y_sel.mean(dim=1)` (this IS the
  Uniform aggregation, `R^U = (1/K)*sum(Y_i)`, identical math to spec
  PART 6) and `agg_mse = ((agg_pred - query_future)**2).mean(-1)` --
  reused UNMODIFIED to reproduce J1/K2/M2's Stage1 AggMSE
  (0.564026 / 0.546943 / 0.523593) to <=1e-5.
- TRACK-M @ `2fe3eade660bc7f5a313626707d7e3a593a1b37a` (HEAD) --
  `scripts/build_m_stage2_retrieval_cache01.py`: the Host-weighted
  aggregation. `HostScorer(stage2_host, device).scores(batch_x, c,
  cand_mask)` gives the FIXED, frozen exogenous score; `alpha =
  softmax(sc_sel / host.tau_topk)`; `delta_ret = (alpha.unsqueeze(-1) *
  tgt_delta).sum(1)` -- this IS the Host aggregation, `R^H =
  sum(alpha_i*Y_i)`, identical math to spec PART 6. Reused UNMODIFIED
  (same `HostScorer`, same `stage2_host` checkpoint) to reproduce the
  TRACK-M Stage2 cache's `raw_retrieval_aggregate_mse` values
  (J1=0.578322, K2=0.573898, M2=0.562447) to <=1e-5.
  `scripts/train_setlossctrl_stage2_retrain02.py`: `eval_epoch`'s
  `raw_retrieval_aggregate_mse` metric (computed by `restore_absolute`
  applied to the cache's `relation_outputs`, i.e. exactly the Host
  aggregation, then compared to `batch_y`) is the quantity these
  numbers were originally measured from -- confirms this audit is
  reproducing the SAME quantity, not a new definition.

## Checkpoints used (frozen, never retrained this track)

| Role | Checkpoint | SHA-256 |
|---|---|---|
| J1 retriever | `checkpoints/track_j2_key_update_decomposition01/ETTh1_720/J1_stopgrad_key/checkpoint.pth` | `1b14877fce3c41114e83db1b832fdd6bd749df4888c8ff74468e34a6d730626c` |
| K2 retriever | `checkpoints/track_k_multislot_predictive_retrieval01/ETTh1_720/K2_multislot_aggregate/checkpoint.pth` | `9827f76d4698ff96a2df6eb867cff2c6eaeaf8f6a80a0db00044a6095ee629c1` |
| M2 retriever | `checkpoints/track_m_relevance_constrained_multislot01/ETTh1_720/M2_relevance_budget/checkpoint.pth` | `9e3ef235281e1efb74aeb5a6ef21de72b4c1dc9d530a13c0c263bbece53606d2` |
| S0 common frozen base (Stage2) | `checkpoints/track_m_relevance_constrained_multislot01/stage2/ETTh1_720/S0_base/checkpoint.pth` | `a7f91972c02bce72a00b05e9402474f597cee71a6c5ce41247e34e17b35e8609` |

## Candidate selection (unchanged from Stage1/TRACK-M, reused verbatim)

- J1: single score vector, `arm_score` + `stable_topk_indices`, future-blind
  hard Top-10.
- K2/M2: 10-slot scores, `compute_scores` + `hard_unique_selection`
  (fixed slot order 0..9, greedy-unique), future-blind hard Top-10.

Candidate SET itself is not changed in this track -- only which
aggregation function (Uniform vs Host) and which downstream fusion
(analytic oracle lambda, validation-calibrated lambda, or the
conditional Gate-Only arm) consumes that same fixed Top-10 set.

## Common frozen base (PART 4, new for this track)

`checkpoints/.../stage2/ETTh1_720/S0_base/checkpoint.pth` (TRACK-M's
own S0 base-only Stage2 arm, `best_epoch=6`) supplies `B_q` -- computed
ONCE per query via that model's `base_head` branch with the retrieval
branch forced to zero (matching how S0 itself was trained), then cached
for train/val/test. J1/K2/M2's oracle-lambda/validation-lambda/gate-only
comparisons all condition on this SAME `B_q`, eliminating the
base-head-co-training confound TRACK-M's own Stage2 had (each of
S0/S1/S2/S3 there jointly trained its own base head alongside its own
retrieval branch, so final-MSE differences conflated retrieval quality
with base-head drift).
