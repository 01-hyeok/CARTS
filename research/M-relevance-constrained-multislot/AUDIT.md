# TRACK-M-RELEVANCE-CONSTRAINED-MULTISLOT01 -- Code Audit

NO ARCHITECTURE CHANGE. This track extends K2 (`06f76672a39bb44814e8f948b1d1976d381946d0`,
`scripts/train_k_multislot_predictive_retrieval01.py`) with exactly one new
loss term per arm and a new (validation-only) checkpoint-selection rule
(PART 8). Everything else -- dataset, seq_len/pred_len, MLP encoder,
10-slot `SlotHeads`, StopGrad-Key, slot init (seed `1000+m`, std=1e-3),
candidate universe/mask (`raft`, N=7201 train-only), teacher
(`normalized_teacher_prob` at `tau_t=0.1`), `tau_s=0.1`, `L_soft=32`, K2's
aggregate objective (`soft_aggregate_loss`, `lambda_agg=1.0`), overlap
objective (`slot_overlap_penalty`, `beta=0.05`), optimizer (Adam,
`lr=1e-3`), batch size (32), and hard inference rule
(`hard_unique_selection`) -- is imported UNMODIFIED from
`train_k_multislot_predictive_retrieval01.py` and called with identical
arguments to K2's own training loop.

## Read at the cited commits

- `scripts/train_k_multislot_predictive_retrieval01.py` @ `06f76672a39bb44814e8f948b1d1976d381946d0`:
  `SlotHeads`, `kl_loss_from_prob`, `slot_overlap_penalty`,
  `soft_aggregate_loss`, `hard_unique_selection`, `compute_scores`,
  `hard_eval_decomposition`, and `main()`'s training/eval loop
  (checkpoint selection there is `if val_agg < best['val_agg']:` -- simple
  min-AggMSE, no feasibility constraint; this is exactly what PART 8
  requires replacing for M1/M2).
- `scripts/eval_l_aligned_population01.py` @ `9cd3a824bdc94bf47b42c446722b886dfafd537e`:
  `load_arm`, `score_and_decompose`, `run_population`, `macro_summary`,
  and the FULL2161 baseline values (`EXPECTED_FULL`) -- reused verbatim
  for J0/J1/K2 in the Stage1 comparison table (PART 9), never recomputed.

## What TRACK-M adds (nothing else)

1. `soft_relevance_loss` (`train_m_relevance_constrained_multislot01.py`) --
   PART 3's `R_m`/`R_set`, using the IDENTICAL per-slot Top-32 selection
   code path as `soft_aggregate_loss` (`stable_topk_indices(s_m, l_soft,
   largest=True)` + `softmax(gathered / tau_s)`), applied to the scalar
   raw future-MSE `d_raw` instead of the future vector. Top-32 selection
   remains student-score-only; future is used only as the loss weight
   target.
2. `precompute_m_j1_reference01.py` -- PART 4's frozen, train-only
   `T_J1(q,c)` table (J1's own hard Top-10, `checkpoints/
   track_j2_key_update_decomposition01/ETTh1_720/J1_stopgrad_key/
   checkpoint.pth`, never retrained) plus J1's full-validation baseline
   (`R_val_J1`, `A_val_J1`), computed via `eval_l_aligned_population01`'s
   own `run_population`/`macro_summary` (not a parallel re-implementation).
3. M1 (`L = L_K2 + gamma * R_set.mean()`) and M2
   (`L = L_K2 + gamma * ReLU(R_set - (1+delta) * T_J1(q,c)).mean()`),
   `gamma=1.0`, `delta=0.05` fixed (no sweep, PART 6).
4. PART 8's constrained checkpoint selection: among epochs with
   `val_retmse10 <= 1.05 * R_val_J1`, the min-`val_agg` epoch is saved as
   `checkpoint.pth`; if no epoch is feasible, `checkpoint.pth` is never
   written and the run is recorded `NO FEASIBLE CHECKPOINT` (diagnostic
   min-violation epoch logged, never used downstream).

## Forbidden and NOT touched

Set Oracle, greedy future oracle, beam search, future-based hard candidate
selection, Transformer, pair scorer, new teacher, Weather, additional
seed, Stage2 architecture modification -- none appear anywhere in
`train_m_relevance_constrained_multislot01.py` or
`precompute_m_j1_reference01.py`.
