"""TRACK-V5-HEAD-LOSS-DUAL-ENCODER01 -- PHASE 0 unit tests for
`scripts/train_v5_head_loss_dual_encoder01.py`'s new logic (the parts
not already covered by `tests/test_differentiable_topk01.py` or by the
many pre-existing passing tests this track reuses for Teacher
sign/Temporal leakage/Future alignment/Mean-Mixture parity -- see
PHASE_0_IMPLEMENTATION_AUDIT.md for the full citation list). Pure CPU,
synthetic only.
"""
import torch

from scripts.train_v5_head_loss_dual_encoder01 import aggregate_loss


class _Cli:
    top_k = 3
    tau_topk = 0.05


def _synthetic_batch(bsz=4, m=10, h=5):
    torch.manual_seed(0)
    p_mix = torch.softmax(torch.randn(bsz, m), dim=-1)
    pooled_memory_c = torch.randn(bsz, m, h)
    offset_c = torch.zeros(bsz)
    query_future = torch.randn(bsz, h)
    valid_mask = torch.ones(bsz, m, dtype=torch.bool)
    return p_mix, pooled_memory_c, offset_c, query_future, valid_mask


def test_l0_zero_auxiliary_weight_equivalence():
    """lambda_A * l_agg must be exactly 0 for L0, i.e. total_loss == kl
    regardless of lambda_A -- checked at the call site convention used
    throughout the training script (`kl + lambda_A * l_agg`)."""
    p_mix, pooled_memory_c, offset_c, query_future, valid_mask = _synthetic_batch()
    l_agg, r = aggregate_loss('L0', p_mix, pooled_memory_c, offset_c, query_future, valid_mask, _Cli())
    assert float(l_agg) == 0.0
    assert r is None
    kl = torch.tensor(1.2345)
    for lambda_A in (0.0, 1.0, 5.0):
        total = kl + lambda_A * l_agg
        assert torch.allclose(total, kl), 'L0 total loss must be invariant to lambda_A (l_agg is identically 0)'


def test_l3a_vs_l0_zero_lambda_equivalence():
    """Setting lambda_A=0 on an L3a/L3b arm must make the TOTAL loss
    value identical to the pure L0 loss (same kl, l_agg simply unused)."""
    p_mix, pooled_memory_c, offset_c, query_future, valid_mask = _synthetic_batch()
    l_agg_a, _ = aggregate_loss('L3a', p_mix, pooled_memory_c, offset_c, query_future, valid_mask, _Cli())
    l_agg_b, _ = aggregate_loss('L3b', p_mix, pooled_memory_c, offset_c, query_future, valid_mask, _Cli())
    kl = torch.tensor(0.777)
    assert torch.allclose(kl + 0.0 * l_agg_a, kl)
    assert torch.allclose(kl + 0.0 * l_agg_b, kl)


def test_l3a_matches_full_pool_weighted_mean_directly():
    p_mix, pooled_memory_c, offset_c, query_future, valid_mask = _synthetic_batch()
    l_agg, r_soft = aggregate_loss('L3a', p_mix, pooled_memory_c, offset_c, query_future, valid_mask, _Cli())
    y_pool = pooled_memory_c + offset_c.view(-1, 1, 1)
    r_manual = torch.einsum('bm,bmh->bh', p_mix, y_pool)
    assert torch.allclose(r_soft, r_manual, atol=1e-6)
    assert torch.allclose(l_agg, ((r_manual - query_future) ** 2).mean(), atol=1e-6)


def test_l3b_gradient_flows_to_p_mix():
    p_mix, pooled_memory_c, offset_c, query_future, valid_mask = _synthetic_batch()
    p_mix = p_mix.clone().requires_grad_(True)
    l_agg, r_soft = aggregate_loss('L3b', p_mix, pooled_memory_c, offset_c, query_future, valid_mask, _Cli())
    l_agg.backward()
    assert p_mix.grad is not None
    assert torch.isfinite(p_mix.grad).all()
    assert p_mix.grad.abs().sum() > 0


def test_l3b_r_soft_is_finite_and_within_candidate_hull():
    """R_softTopK must be a convex combination of the pooled candidate
    futures (weights sum to 1, all nonnegative) -- so it must lie
    within [min, max] of the pool along every horizon dimension."""
    p_mix, pooled_memory_c, offset_c, query_future, valid_mask = _synthetic_batch()
    _, r_soft = aggregate_loss('L3b', p_mix, pooled_memory_c, offset_c, query_future, valid_mask, _Cli())
    y_pool = pooled_memory_c + offset_c.view(-1, 1, 1)
    assert torch.isfinite(r_soft).all()
    assert (r_soft <= y_pool.max(dim=1).values + 1e-4).all()
    assert (r_soft >= y_pool.min(dim=1).values - 1e-4).all()


def test_unknown_loss_raises():
    p_mix, pooled_memory_c, offset_c, query_future, valid_mask = _synthetic_batch()
    try:
        aggregate_loss('L_nonexistent', p_mix, pooled_memory_c, offset_c, query_future, valid_mask, _Cli())
        assert False, 'expected ValueError for an unknown loss name'
    except ValueError:
        pass
