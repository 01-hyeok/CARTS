# INVALIDATED_BY_TEACHER_SIGN_BUG

**Status: INVALIDATED. Do not use these numbers. Not deleted -- kept for audit trail.**

## Bug

`scripts/train_v_sharedtop100_01.py` (the Stage-1 trainer that produced this
Shared-Top-100 (P100) result) called
`normalized_teacher_prob(-d_pool, valid_mask, cli.tau_t)` at two sites
(the diagnostic teacher-entropy call and the actual KL training-loss teacher
`p_t`). `d_pool` (from `utils/candidate_pool.py::pooled_future_mse`) is
already a correctly-signed, non-negative, lower-is-better MSE distance.
`normalized_teacher_prob(d, ...)` already internally negates its input
(z-score then `softmax(-z/tau)`). Passing `-d_pool` therefore double-negates:
the teacher distribution ends up assigning HIGH probability to HIGH-MSE (bad)
candidates and LOW probability to LOW-MSE (good) candidates -- the exact
opposite of the intended teacher.

Confirmed by direct code reading, algebraic derivation
(`zscore(-x) = -zscore(x)`), and a numeric synthetic reproduction
(`d=[0.1, 1.0, 10.0]`: correct call puts argmax probability on index 0 /
the lowest-MSE candidate; the buggy call puts it on index 2 / the
highest-MSE candidate). See `tests/test_v_sharedtop100_signfix01.py`.

This bug affects ONLY Stage-1 training that went through
`scripts/train_v_sharedtop100_01.py` (every P100/Shared-Top-100 cell in
this project). It does NOT affect:
- Full-memory arms (all use the `d_raw = -u` pattern correctly, confirmed
  by repo-wide audit and `test_e_full_memory_files_use_correct_utility_negation_pattern`).
- The P100 candidate-pool identity itself
  (`scripts/precompute_candidate_pool01.py` never calls
  `normalized_teacher_prob` -- it only uses a future-blind cosine
  prefilter, `compute_coarse_delta_last_cosine_scores`). The 100 candidates
  in the pool are correct; only the KL-training loss and its diagnostics
  built on top of them were corrupted.

## Fix

Fixed in `scripts/train_v_sharedtop100_01.py`: both call sites changed from
`normalized_teacher_prob(-d_pool, ...)` to `normalized_teacher_prob(d_pool, ...)`.
See `tests/test_v_sharedtop100_signfix01.py` for the regression tests
(Tests A-E per the bug report).

## Corrected results

See `results/TRACK-V-SHARED-TOP100-SIGNFIX01/` once the retrain (16 cells:
ETTh1/Weather x H96/H720 x V0/V1/V2/V5) completes.
