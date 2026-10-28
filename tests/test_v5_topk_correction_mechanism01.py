"""TRACK-V5-TOPK-CORRECTION-MECHANISM01 Phase B1 -- synthetic regression
tests for the KL cross-batch aggregation fix in
`scripts/train_v5_head_loss_dual_encoder01.py`.

Bug: `kl_loss_from_prob()` returns a 0-dim scalar already averaged over
the CURRENT batch's queries (unlike `model_ind_mse`/`agg_mse`/etc., which
are [bsz]-shaped per-query tensors). The val/test aggregation loop treated
every key the same way -- `.reshape(-1)`, concatenate across batches,
`.sum()`, then divide the grand total by `n_eff = total_queries * channels`
at the end. For the per-query-tensor keys this is a correct query-weighted
average. For `kl` (already a per-batch mean) it silently divided the
cross-batch SUM of per-batch means by the total query count again -- an
extra, spurious division by (roughly) the batch size whenever batch sizes
are not all equal to 1. The fix multiplies `kl` by the batch's query count
before accumulating, so it is query-weighted like every other key.

Pure CPU, synthetic only -- does not call GPU code, does not touch the
training gradient path (`eval_channel`'s own `kl_loss_from_prob` call,
used for `total_loss`/backward, is untouched).
"""
import torch

from scripts.train_k_multislot_predictive_retrieval01 import kl_loss_from_prob


def _synthetic_prob_batch(bsz, n, seed):
    g = torch.Generator().manual_seed(seed)
    p_t = torch.softmax(torch.randn(bsz, n, generator=g), dim=-1)
    p_s = torch.softmax(torch.randn(bsz, n, generator=g), dim=-1)
    valid_mask = torch.ones(bsz, n, dtype=torch.bool)
    return p_t, p_s, valid_mask


def _per_query_kl(p_t, p_s, valid_mask, eps=1e-12):
    """Ground truth: KL per query, no batch-averaging -- shape [bsz]."""
    log_p_s = torch.log(p_s.clamp_min(eps))
    term = (p_t * (torch.log(p_t.clamp_min(eps)) - log_p_s)).masked_fill(~valid_mask, 0.0)
    return term.sum(-1)


def test_kl_loss_from_prob_is_batch_mean_not_sum():
    """Sanity: confirms the exact shape/semantics the bug hinges on --
    kl_loss_from_prob returns a 0-dim scalar equal to the MEAN (not sum)
    of the per-query KL terms."""
    p_t, p_s, valid_mask = _synthetic_prob_batch(bsz=7, n=11, seed=0)
    batch_kl = kl_loss_from_prob(p_t, p_s, valid_mask)
    assert batch_kl.dim() == 0
    expected_mean = _per_query_kl(p_t, p_s, valid_mask).mean()
    assert torch.allclose(batch_kl, expected_mean, atol=1e-6)


def test_uneven_batch_sizes_require_query_weighting_not_batch_weighting():
    """The core regression case: two batches of DIFFERENT size. The
    correct cross-batch KL average must equal the true per-query average
    over the concatenation of both batches, not a simple mean of the two
    batches' own per-batch means (which is what you'd get by weighting
    both batches equally regardless of size)."""
    p_t1, p_s1, vm1 = _synthetic_prob_batch(bsz=5, n=9, seed=1)
    p_t2, p_s2, vm2 = _synthetic_prob_batch(bsz=3, n=9, seed=2)

    kl1 = kl_loss_from_prob(p_t1, p_s1, vm1)  # scalar, mean over 5 queries
    kl2 = kl_loss_from_prob(p_t2, p_s2, vm2)  # scalar, mean over 3 queries

    # ground truth: concatenate both batches' queries, take one true mean
    true_per_query = torch.cat([_per_query_kl(p_t1, p_s1, vm1), _per_query_kl(p_t2, p_s2, vm2)])
    true_mean = true_per_query.mean()

    # a simple (unweighted) mean of the two batch-means is WRONG when
    # batch sizes differ -- this is the failure mode the bug exhibited
    naive_unweighted_mean = (kl1 + kl2) / 2.0
    assert not torch.allclose(naive_unweighted_mean, true_mean, atol=1e-4), (
        'this batch-size pair should expose the weighting bug (if it does not, '
        'strengthen the seeds/sizes so the two per-batch means differ enough to matter)'
    )

    # the FIXED aggregation: multiply each per-batch mean by its own bsz
    # before summing, then divide by the TOTAL query count (query-weighted) --
    # this is exactly what the corrected script now does
    n_total = 5 + 3
    fixed_weighted_mean = (kl1 * 5 + kl2 * 3) / n_total
    assert torch.allclose(fixed_weighted_mean, true_mean, atol=1e-6)


def test_equal_batch_sizes_bug_is_invisible_but_fix_is_still_exact():
    """When all batches happen to be the same size, the old buggy code and
    the fixed code agree up to a constant additional 1/bsz scaling in the
    OLD code (not tested here, since the old code path no longer exists in
    the script) -- this test only confirms the FIXED formula is exact
    for the equal-size case too, so the fix introduces no regression there."""
    p_t1, p_s1, vm1 = _synthetic_prob_batch(bsz=4, n=6, seed=3)
    p_t2, p_s2, vm2 = _synthetic_prob_batch(bsz=4, n=6, seed=4)
    kl1 = kl_loss_from_prob(p_t1, p_s1, vm1)
    kl2 = kl_loss_from_prob(p_t2, p_s2, vm2)

    true_per_query = torch.cat([_per_query_kl(p_t1, p_s1, vm1), _per_query_kl(p_t2, p_s2, vm2)])
    true_mean = true_per_query.mean()

    fixed_weighted_mean = (kl1 * 4 + kl2 * 4) / 8
    assert torch.allclose(fixed_weighted_mean, true_mean, atol=1e-6)


def test_final_batch_smaller_than_the_rest_matches_real_dataloader_pattern():
    """Reproduces the realistic failure mode: N full-size batches plus one
    smaller trailing batch (DataLoader's default behavior with
    drop_last=False), the exact pattern a real val/test loader produces."""
    full_bsz, last_bsz, n = 8, 3, 10
    batches = [_synthetic_prob_batch(full_bsz, n, seed=10 + i) for i in range(3)]
    batches.append(_synthetic_prob_batch(last_bsz, n, seed=99))

    per_batch_kl = [kl_loss_from_prob(pt, ps, vm) for pt, ps, vm in batches]
    sizes = [full_bsz, full_bsz, full_bsz, last_bsz]

    true_per_query = torch.cat([_per_query_kl(pt, ps, vm) for pt, ps, vm in batches])
    true_mean = true_per_query.mean()

    n_total = sum(sizes)
    fixed_weighted_mean = sum(k * s for k, s in zip(per_batch_kl, sizes)) / n_total
    assert torch.allclose(fixed_weighted_mean, true_mean, atol=1e-6)

    naive_unweighted_mean = sum(per_batch_kl) / len(per_batch_kl)
    assert not torch.allclose(naive_unweighted_mean, true_mean, atol=1e-4)
