"""TRACK-EXPERT-V5-FULL01 -- unit tests for utils/expert_head_metrics.py
(spec section 12, tests 6-11 and 13 covered directly here with
synthetic tensors; tests 1-5, 14-18 are covered by
tests/test_expert_v5_integration01.py against the real trainer)."""
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.expert_head_metrics import (
    expert_weighted_loss, head_pairwise_overlap, kl_per_head,
    per_head_decomposition, per_head_entropy_and_spread,
    per_head_future_utility, per_head_standalone_topk,
    responsibility_from_utility, winner_margin_stats,
)

TOP_K = 10


def _toy_scores(bsz=4, s=5, n=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(bsz, s, n, generator=g)


# -------------------- test 6: per-head standalone Top10 correctness --------------------

def test_per_head_standalone_topk_matches_manual_argsort():
    scores = _toy_scores()
    cand_mask = torch.ones(4, 20, dtype=torch.bool)
    idx = per_head_standalone_topk(scores, cand_mask, k=TOP_K)
    assert idx.shape == (4, 5, TOP_K)
    for b in range(4):
        for h in range(5):
            expected = scores[b, h].argsort(descending=True)[:TOP_K]
            assert set(idx[b, h].tolist()) == set(expected.tolist())


def test_per_head_standalone_topk_respects_mask():
    scores = _toy_scores(bsz=1, s=1, n=20)
    cand_mask = torch.zeros(1, 20, dtype=torch.bool)
    cand_mask[0, :12] = True  # only 12 valid -> Top10 must come from those 12
    idx = per_head_standalone_topk(scores, cand_mask, k=TOP_K)
    assert idx[0, 0].max() < 12


# -------------------- test 7: head utility arithmetic mean correctness --------------------

def test_per_head_future_utility_is_exact_mean():
    bsz, s, k = 3, 2, 4
    head_topk_idx = torch.randint(0, 10, (bsz, s, k))
    d_raw = torch.randn(bsz, 10)
    U = per_head_future_utility(head_topk_idx, d_raw)
    for b in range(bsz):
        for h in range(s):
            expected = d_raw[b, head_topk_idx[b, h]].mean()
            assert torch.allclose(U[b, h], expected, atol=1e-6)


# -------------------- tests 8/9/10/11: responsibility --------------------

def test_responsibility_shape_and_sums_to_one():
    U = torch.rand(6, 5)
    r = responsibility_from_utility(U, tau_e=1.0)
    assert r.shape == (6, 5)
    assert torch.allclose(r.sum(dim=1), torch.ones(6), atol=1e-5)


def test_lower_utility_gets_higher_responsibility():
    U = torch.tensor([[0.1, 0.5, 0.9, 0.5, 0.5]])
    r = responsibility_from_utility(U, tau_e=1.0)
    assert r[0, 0] == r.max()


def test_responsibility_requires_grad_false():
    U = torch.rand(4, 5, requires_grad=True)
    r = responsibility_from_utility(U, tau_e=1.0)
    assert r.requires_grad is False


def test_identical_utility_gives_uniform_responsibility():
    U = torch.full((2, 5), 0.42)
    r = responsibility_from_utility(U, tau_e=1.0)
    assert torch.allclose(r, torch.full((2, 5), 0.2), atol=1e-5)


# -------------------- test 12: per-head KL [B,S] correctness --------------------

def test_kl_per_head_matches_manual_kl_divergence():
    bsz, s, n = 2, 3, 10
    scores = _toy_scores(bsz, s, n, seed=1)
    cand_mask = torch.ones(bsz, n, dtype=torch.bool)
    p_t = torch.softmax(torch.randn(bsz, n, generator=torch.Generator().manual_seed(2)), dim=-1)
    kl, p_h = kl_per_head(p_t, scores, cand_mask, tau_s=0.1)
    assert kl.shape == (bsz, s)
    for b in range(bsz):
        for h in range(s):
            # log_softmax directly (numerically stable), not softmax().log() --
            # the latter loses precision for the peaked distributions a /0.1
            # temperature produces from randn-scale logits, which is a
            # test-construction artifact, not a property of kl_per_head itself.
            log_p_h_manual = torch.log_softmax(scores[b, h] / 0.1, dim=-1)
            expected = (p_t[b] * (p_t[b].clamp_min(1e-8).log() - log_p_h_manual)).sum()
            assert torch.allclose(kl[b, h], expected, atol=1e-4)


# -------------------- test 13: final scalar loss manual equivalence --------------------

def test_expert_weighted_loss_matches_manual_computation():
    r = torch.rand(5, 5)
    r = r / r.sum(dim=1, keepdim=True)
    kl = torch.rand(5, 5)
    loss = expert_weighted_loss(r, kl)
    manual = (r * kl).sum(dim=1).mean()
    assert torch.allclose(loss, manual, atol=1e-6)


# -------------------- head_pairwise_overlap / winner_margin_stats --------------------

def test_head_pairwise_overlap_identical_heads_gives_overlap_one():
    idx = torch.arange(10).view(1, 1, 10).expand(1, 3, 10).clone()
    overlaps, union = head_pairwise_overlap(idx)
    for pair, val in overlaps.items():
        assert torch.allclose(val, torch.ones(1))
    assert union[0].item() == 10


def test_head_pairwise_overlap_disjoint_heads_gives_overlap_zero():
    idx = torch.stack([torch.arange(10), torch.arange(10, 20)]).view(1, 2, 10)
    overlaps, union = head_pairwise_overlap(idx)
    assert torch.allclose(overlaps[(0, 1)], torch.zeros(1))
    assert union[0].item() == 20


def test_winner_margin_stats_correctness():
    U = torch.tensor([[0.3, 0.1, 0.9, 0.5, 0.2]])
    winner, best, second, margin = winner_margin_stats(U)
    assert winner[0].item() == 1  # index of 0.1, the minimum
    assert torch.allclose(best[0], torch.tensor(0.1))
    assert torch.allclose(second[0], torch.tensor(0.2))
    assert torch.allclose(margin[0], torch.tensor(0.1), atol=1e-6)


def test_per_head_entropy_and_spread_shapes():
    p_h = torch.softmax(_toy_scores(2, 3, 8), dim=-1)
    cand_mask = torch.ones(2, 8, dtype=torch.bool)
    ent, spread = per_head_entropy_and_spread(p_h, cand_mask)
    assert ent.shape == (2, 3)
    assert spread.shape == (2, 3)
    assert (ent >= 0).all()
    assert (spread >= 0).all()


def test_per_head_decomposition_D_C_identity():
    """Agg = D + C must hold exactly, same identity as the historical
    `hard_eval_decomposition` (TRACK-T/V convention)."""
    bsz, s, k = 3, 2, 4
    head_topk_idx = torch.randint(0, 15, (bsz, s, k))
    cand_mask = torch.ones(bsz, 15, dtype=torch.bool)
    memory_c = torch.randn(15, 6)
    offset_c = torch.randn(bsz)
    query_future = torch.randn(bsz, 6)
    d_raw = torch.randn(bsz, 15)
    oracle_idx = torch.randint(0, 15, (bsz, k))
    out = per_head_decomposition(head_topk_idx, cand_mask, memory_c, offset_c, query_future, d_raw, oracle_idx, k)
    assert torch.allclose(out['agg_mse'], out['D'] + out['C'], atol=1e-5)
