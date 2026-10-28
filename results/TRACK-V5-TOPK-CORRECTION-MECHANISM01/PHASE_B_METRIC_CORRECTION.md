# PHASE B -- Stage1 Metric Calculation Fixes (IN PROGRESS: B1 closed, B2/B3 pending)

## B1. KL cross-batch aggregation bug -- CONFIRMED, ROOT-CAUSED, FIXED

**Confirmed real bug, diagnostic-metric-only (no effect on any reported performance number).**

`kl_loss_from_prob()` (`scripts/train_k_multislot_predictive_retrieval01.py:80`) returns a 0-dim scalar
that is already the MEAN of the per-query KL terms over the current batch's queries -- unlike
`model_ind_mse`/`agg_mse`/`soft_aggmse`/`recall10`/`ndcg10`/`D`/`C` in the same `eval_channel()` return dict,
which are `[bsz]`-shaped per-query tensors.

The val/test aggregation loop in `train_v5_head_loss_dual_encoder01.py` (originally lines ~342-350 and
~397-405) treated every key identically: `.reshape(-1).cpu()`, concatenate across all val/test batches,
`.sum()`, accumulate into `val_sums`/`test_sums`, then divide the grand total by
`n_eff = total_queries_across_all_batches * num_channels` at the very end.

For the per-query-tensor keys this is exactly correct (summing `bsz` individual per-query values per batch,
then dividing by the true total query count, gives the correct query-weighted average). For `kl`, which is
**already a per-batch mean**, this silently introduced an extra division by (effectively) the batch size:
the cross-batch sum of per-batch means was divided by `total_queries`, not by `num_batches`.

**Exact effect size**: when all batches have equal size `bsz0` (the common case except the final batch),
`reported_kl = true_kl / bsz0`. With this track's default `--batch_size 32`, every `val_kl`/`test_kl`
number ever logged by this script was ~32x too small. Confirmed empirically: a 3-batch smoke run
(`ETTh1_96`, `--limit_batches 3`) reported `kl=2.315` after the fix below, versus the magnitude that would
be expected before it (divided by ~32, i.e. ~0.07 -- implausibly tiny for a KL divergence between two
materially different ~100-candidate distributions).

**Scope of impact -- confirmed NOT to affect any reported performance claim**: `kl` is never used for
checkpoint selection (that uses `val_retmse10`/`val_hard_aggmse`/`val_soft_aggmse`, all per-query tensors,
unaffected), never feeds Stage2 (Stage2 only consumes the saved Hard Top-10 retrieval cache, independent of
this diagnostic), and the TRAINING gradient itself was always computed correctly (`eval_channel()`'s
`total_loss = kl + lambda_A * l_agg` backward pass uses the scalar `kl` directly inside the training loop,
a completely separate code path from this val/test *logging* aggregation). This bug only ever corrupted the
`kl` column of `val_metrics.csv`/`final_test_metrics_*.json` -- a diagnostic number that was never cited in
any of this track's or `TRACK-V5-HEAD-LOSS-DUAL-ENCODER01`'s comparison tables or conclusions (those all use
RetMSE@10/AggMSE/Stage2 MSE).

**Fix applied**: `scripts/train_v5_head_loss_dual_encoder01.py`, both the val loop and the test loop --
multiply `kl` by the batch's query count (`bsz`) before the shared `.reshape(-1)` accumulation step, so it
becomes query-weighted like every other key, exactly matching the pattern already used for `D`/`C` in the
test loop. The training-loss gradient computation inside `eval_channel()` is untouched.

**Unit tests** (`tests/test_v5_topk_correction_mechanism01.py`, 4 new, all pass, pure CPU synthetic):
1. `test_kl_loss_from_prob_is_batch_mean_not_sum` -- confirms the exact 0-dim-scalar-mean semantics the bug
   depends on.
2. `test_uneven_batch_sizes_require_query_weighting_not_batch_weighting` -- two batches of different size
   (5 and 3); confirms the naive unweighted mean of per-batch means is WRONG, and the fixed (bsz-weighted)
   formula is exact.
3. `test_equal_batch_sizes_bug_is_invisible_but_fix_is_still_exact` -- equal-size batches; confirms the fix
   introduces no regression when the bug would have been invisible anyway.
4. `test_final_batch_smaller_than_the_rest_matches_real_dataloader_pattern` -- 3 full + 1 smaller trailing
   batch, the realistic `DataLoader(drop_last=False)` pattern; confirms the fixed formula is exact and the
   naive unweighted formula is measurably wrong on this realistic case.

**Re-verification note**: this fix changes only the logged `kl` diagnostic value going forward for any
*new* training run in this track (S2-v2/D2-v2, Phase E). It does NOT require re-running or invalidating any
existing `TRACK-V5-HEAD-LOSS-DUAL-ENCODER01` checkpoint, RetMSE/AggMSE/Stage2 result, since none of those
used the buggy `kl` logging value.

## B2. L3b soft-metric separation -- PENDING

Not yet implemented this round. Requires adding `topk_surrogate_mse` (the actual quantity L3b's auxiliary
term trains against, using the real `soft_topk_mask`/`soft_topk_forecast` call) as a third, separately
logged metric alongside the existing `soft_aggmse` (full-P100 soft, what L3a trains against) and
`agg_mse`/Hard AggMSE, for every arm (not just L3b) so the three can be compared on identical
checkpoints/queries/channels. Deferred to the next work session on this track.

## B3. D/C independent pairwise verification -- PENDING

Not yet implemented this round. Requires a dedicated script that recomputes `C` via direct pairwise
`<e_i, e_j>` dot products over the Top-10 selected set (not `C = AggMSE - D`), for the existing S0/S1/S2-v1/
D0/D1/D2-v1 checkpoints, and compares against the self-consistency identity already used throughout
`TRACK-V5-HEAD-LOSS-DUAL-ENCODER01`. Deferred to the next work session, to run alongside Phase C's head
complementarity analysis (same per-candidate error vectors are needed for both).

**Phase B status: B1 CLOSED (confirmed bug, fixed, tested, scope-limited to a never-cited diagnostic
value). B2/B3 PENDING -- to be completed before Phase C/D/E proceed, per the spec's own required ordering.**
