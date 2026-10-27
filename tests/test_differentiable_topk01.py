"""TRACK-V5-HEAD-LOSS-DUAL-ENCODER01 -- PHASE 0 unit tests for
`utils/differentiable_topk01.py` (L3b's Differentiable Top-K operator).
Pure CPU, synthetic only.
"""
import torch

from models.RelationStage1 import stable_topk_indices
from utils.differentiable_topk01 import hard_limit_matches_stable_topk, soft_topk_forecast, soft_topk_mask

EPS = 1e-6


def test_weights_nonnegative_and_sum_to_one():
    torch.manual_seed(0)
    scores = torch.randn(8, 30)
    mask = torch.ones(8, 30, dtype=torch.bool)
    a = soft_topk_mask(scores, mask, k=10, tau=0.1)
    assert (a >= 0).all()
    assert torch.allclose(a.sum(-1), torch.ones(8), atol=1e-4)


def test_invalid_candidate_weight_exactly_zero():
    torch.manual_seed(1)
    scores = torch.randn(4, 20)
    mask = torch.ones(4, 20, dtype=torch.bool)
    mask[:, :5] = False  # first 5 candidates invalid
    a = soft_topk_mask(scores, mask, k=5, tau=0.1)
    assert torch.allclose(a[:, :5], torch.zeros(4, 5), atol=0.0)  # EXACTLY zero, not just small
    assert torch.allclose(a.sum(-1), torch.ones(4), atol=1e-4)


def test_hard_limit_converges_to_uniform_1_over_k():
    torch.manual_seed(2)
    scores = torch.randn(16, 50) * 5  # well-separated scores
    mask = torch.ones(16, 50, dtype=torch.bool)
    matches, max_diff = hard_limit_matches_stable_topk(scores, mask, k=10, tau=1e-3, atol=5e-2)
    assert matches, f'small-tau soft top-k must converge to hard top-K uniform weights, max_diff={max_diff}'


def test_selected_candidates_match_stable_topk_at_small_tau():
    torch.manual_seed(3)
    scores = torch.randn(10, 40) * 3
    mask = torch.ones(10, 40, dtype=torch.bool)
    a = soft_topk_mask(scores, mask, k=8, tau=1e-3)
    hard_idx = stable_topk_indices(scores, 8, largest=True)
    mass_on_hard = a.gather(-1, hard_idx).sum(-1)
    assert torch.allclose(mass_on_hard, torch.ones(10), atol=5e-2), \
        'at small tau, essentially all probability mass must sit on the hard Top-K set'


def test_boundary_candidate_gets_nonzero_gradient_unlike_hard_renorm():
    """The key differentiator from the repo's existing
    `soft_topk_weights`: a candidate just OUTSIDE the hard cutoff must
    still receive a nonzero gradient here."""
    scores = torch.tensor([[5.0, 4.0, 3.0, 2.9, 2.0, 1.0]], requires_grad=True)
    mask = torch.ones(1, 6, dtype=torch.bool)
    a = soft_topk_mask(scores, mask, k=3, tau=0.5)  # top-3 by hard rule: idx 0,1,2 (values 5,4,3)
    loss = a[0, 3]  # weight assigned to idx=3 (score 2.9, just outside hard top-3)
    loss.backward()
    assert scores.grad is not None
    assert scores.grad.abs().sum() > 0, \
        'a just-excluded candidate must receive nonzero gradient through this operator'


def test_gradient_flows_through_soft_topk_forecast():
    torch.manual_seed(4)
    scores = torch.randn(5, 20, requires_grad=True)
    mask = torch.ones(5, 20, dtype=torch.bool)
    y_pool = torch.randn(5, 20, 7)
    a = soft_topk_mask(scores, mask, k=10, tau=0.1)
    r = soft_topk_forecast(a, y_pool)
    loss = r.pow(2).mean()
    loss.backward()
    assert scores.grad is not None and torch.isfinite(scores.grad).all()
    assert scores.grad.abs().sum() > 0


def test_no_nan_or_inf_with_extreme_scores():
    scores = torch.tensor([[1e6, -1e6, 0.0, 1e-6, -1e-6, 50.0]], requires_grad=True)
    mask = torch.ones(1, 6, dtype=torch.bool)
    a = soft_topk_mask(scores, mask, k=3, tau=0.1)
    assert torch.isfinite(a).all()
    loss = a.sum()
    loss.backward()
    assert torch.isfinite(scores.grad).all()


def test_synthetic_oracle_selection_direction():
    """Higher score => higher weight, monotonically (holding others
    fixed), consistent with "largest=True" Top-K semantics."""
    base = torch.tensor([[3.0, 1.0, 0.5, 0.2, -1.0]])
    mask = torch.ones(1, 5, dtype=torch.bool)
    a_base = soft_topk_mask(base, mask, k=2, tau=0.3)
    bumped = base.clone()
    bumped[0, 3] = 10.0  # make the 4th candidate the clear best
    a_bumped = soft_topk_mask(bumped, mask, k=2, tau=0.3)
    assert a_bumped[0, 3] > a_base[0, 3]
    assert a_bumped[0, 3] > a_bumped[0, 0]  # now beats the formerly-top candidate


def test_tau_large_approaches_uniform():
    torch.manual_seed(5)
    scores = torch.randn(6, 15)
    mask = torch.ones(6, 15, dtype=torch.bool)
    a = soft_topk_mask(scores, mask, k=5, tau=1000.0)
    uniform = torch.full((6, 15), 1.0 / 15)
    assert torch.allclose(a, uniform, atol=1e-2)
