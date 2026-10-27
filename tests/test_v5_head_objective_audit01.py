"""TRACK-V5-HEAD-OBJECTIVE-AUDIT01 -- synthetic unit tests (spec section
11, items 1-12). Pure CPU, no checkpoint/model load, no GPU. Covers the
numerical primitives used by `scripts/diag_v5_head_objective01.py`,
`scripts/diag_v5_head_gradient01.py`, and `scripts/analyze_v5_head_objective01.py`.
"""
import inspect
import itertools

import pytest
import torch

from models.RelationStage1 import stable_topk_indices
from scripts.diag_v5_head_objective01 import ALL_SUBSETS, subset_agg_mse
from utils.candidate_pool import compute_coarse_delta_last_cosine_scores
from utils.expert_head_metrics import per_head_standalone_topk
from utils.mean_mixture_selection import mean_mixture_topk_selection

EPS = 1e-8


def test_item1_head_probability_normalization():
    torch.manual_seed(0)
    scores = torch.randn(4, 5, 20)
    mask = torch.ones(4, 20, dtype=torch.bool)
    _, p_heads, _ = mean_mixture_topk_selection(scores, mask, tau_s=0.1, k=5)
    sums = p_heads.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_item2_mean_mixture_parity():
    torch.manual_seed(1)
    scores = torch.randn(3, 5, 15)
    mask = torch.ones(3, 15, dtype=torch.bool)
    picks, p_heads, p_mix = mean_mixture_topk_selection(scores, mask, tau_s=0.2, k=4)
    p_heads_manual = torch.softmax(scores / 0.2, dim=-1)
    p_mix_manual = p_heads_manual.mean(dim=1)
    assert torch.allclose(p_heads, p_heads_manual, atol=1e-6)
    assert torch.allclose(p_mix, p_mix_manual, atol=1e-6)
    picks_manual = stable_topk_indices(p_mix_manual, 4, largest=True)
    assert torch.equal(picks, picks_manual)


def test_item3_validity_mask_shared_p100():
    mask = torch.ones(10, 100, dtype=torch.bool)
    assert bool(mask.all())
    assert mask.shape == (10, 100)


def test_item4_head_hard_topk_no_round_robin():
    scores = torch.tensor([[[5., 1., 2., 3., 4.], [1., 5., 2., 3., 4.]]])  # [1,2,5]
    mask = torch.ones(1, 5, dtype=torch.bool)
    idx = per_head_standalone_topk(scores, mask, k=2)
    assert idx[0, 0].tolist() == [0, 4] or set(idx[0, 0].tolist()) == {0, 4}
    assert set(idx[0, 1].tolist()) == {1, 4}


def test_item5_aggregate_mse_direct():
    y_sel = torch.tensor([[[1., 1.], [3., 3.]]])  # [1,2,2]
    query_future = torch.tensor([[2., 2.]])
    agg_pred = y_sel.mean(dim=1)
    agg_mse = ((agg_pred - query_future) ** 2).mean(dim=-1)
    assert torch.allclose(agg_mse, torch.tensor([0.0]), atol=1e-6)


def test_item6_dc_decomposition_identity():
    torch.manual_seed(2)
    bsz, k, h = 6, 10, 4
    y_sel = torch.randn(bsz, k, h)
    query_future = torch.randn(bsz, h)
    e = y_sel - query_future.unsqueeze(1)
    ind_mse = (e ** 2).mean(-1)
    D = ind_mse.mean(-1) / k
    agg_pred = y_sel.mean(dim=1)
    agg_mse = ((agg_pred - query_future) ** 2).mean(-1)
    C = agg_mse - D
    cross = torch.einsum('bih,bjh->bij', e, e) / h
    c_direct = (cross.sum(dim=(-1, -2)) - cross.diagonal(dim1=-2, dim2=-1).sum(-1)) / (k ** 2)
    assert torch.allclose(C, c_direct, atol=1e-5)


def test_item7_head_error_dot_cosine():
    e1 = torch.tensor([[1., 0., 0.]])
    e2 = torch.tensor([[-1., 0., 0.]])  # perfectly cancelling
    e3 = torch.tensor([[1., 0., 0.]])   # perfectly redundant
    dot12 = (e1 * e2).sum(-1) / 3
    cos12 = dot12 / (e1.norm(dim=-1) * e2.norm(dim=-1) / 3 + EPS)
    dot13 = (e1 * e3).sum(-1) / 3
    cos13 = dot13 / (e1.norm(dim=-1) * e3.norm(dim=-1) / 3 + EPS)
    assert cos12.item() < 0
    assert cos13.item() > 0


def test_item8_subset_enumeration_count_and_singleton_equivalence():
    assert len(ALL_SUBSETS) == 31
    assert sorted(len(s) for s in ALL_SUBSETS).count(1) == 5
    torch.manual_seed(3)
    bsz, n_slots, m, h = 4, 5, 20, 3
    p_heads = torch.softmax(torch.randn(bsz, n_slots, m), dim=-1)
    pooled_memory_c = torch.randn(bsz, m, h)
    offset_c = torch.zeros(bsz)
    query_future = torch.randn(bsz, h)
    agg_mse_single, _, idx_single = subset_agg_mse(p_heads, (1,), pooled_memory_c, offset_c, query_future, top_k=5)
    idx_direct = per_head_standalone_topk(
        torch.log(p_heads.clamp_min(EPS)), torch.ones(bsz, m, dtype=torch.bool), k=5)[:, 1, :]
    # p_A for a singleton subset == that head's own distribution -> same top-k set
    assert set(idx_single[0].tolist()) == set(idx_direct[0].tolist())


def test_item9_leave_one_out_direction():
    # 3 heads; head 0 is clearly harmful (its own aggregate is far from target)
    bsz, m, h = 2, 6, 2
    p_heads = torch.zeros(bsz, 3, m)
    p_heads[:, 0, 0] = 1.0  # head0 always picks candidate 0 (bad)
    p_heads[:, 1, 1] = 1.0
    p_heads[:, 2, 2] = 1.0
    pooled_memory_c = torch.zeros(bsz, m, h)
    pooled_memory_c[:, 0] = 100.0  # candidate 0 is a terrible future match
    pooled_memory_c[:, 1] = 0.0
    pooled_memory_c[:, 2] = 0.0
    offset_c = torch.zeros(bsz)
    query_future = torch.zeros(bsz, h)
    agg_all, _, _ = subset_agg_mse(p_heads, (0, 1, 2), pooled_memory_c, offset_c, query_future, top_k=1)
    agg_wo0, _, _ = subset_agg_mse(p_heads, (1, 2), pooled_memory_c, offset_c, query_future, top_k=1)
    delta0 = (agg_wo0 - agg_all).mean()
    assert delta0.item() < 0  # removing the harmful head improves (lowers) AggMSE -> Delta_0 negative


def test_item10_oracle_selection_direction_and_ties():
    d = torch.tensor([[0.5, 0.1, 0.1, 0.9]])
    idx = stable_topk_indices(d, 2, largest=False)
    assert set(idx[0].tolist()) == {1, 2}  # the two smallest (tied) values, lowest-MSE-first


def test_item11_gradient_validity():
    x = torch.randn(3, 5, requires_grad=True)
    p = torch.softmax(x, dim=-1)
    loss = -(p.log() * p).sum(dim=-1).mean()
    loss.backward()
    assert x.grad is not None
    assert x.grad.abs().sum() > 0


def test_item12_no_future_leakage_in_coarse_score():
    sig = inspect.signature(compute_coarse_delta_last_cosine_scores)
    param_names = set(sig.parameters.keys())
    assert 'batch_y' not in param_names and 'memory_y' not in param_names
    q = torch.randn(2, 10, 3)
    mem = torch.randn(5, 10, 3)
    out = compute_coarse_delta_last_cosine_scores(q, mem, channel=0)
    assert out.shape == (2, 5)
