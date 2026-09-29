"""Sanity checks for TRACK-H-DIRECT-SET-UTILITY01.

Fast, GPU-optional checks on the core aggregate-loss mechanics: gradient
isolation, lambda=0 equivalence, and metric-reconstruction correctness.
Full-run result-artifact checks (per_query CSV / summary JSON consistency,
baseline reproduction) are added once the 18-run batch exists, mirroring
tests/test_g_decoupled_metric_adaptation01.py's pattern.
"""
import torch
import torch.nn.functional as F
import pytest

from scripts.train_h_direct_set_utility01 import (
    ARMS, FROZEN_ARMS, AGGREGATE_ARMS, aggregate_prediction, effective_number,
    uniform_and_weighted_aggregate, pairwise_mse, entropy_of,
)
from scripts.train_g_decoupled_metric_adaptation01 import AsymMetric


def _toy_batch(bsz=4, n=20, h=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    memory_c = torch.randn(n, h, generator=g)
    offset_c = torch.randn(bsz, generator=g)
    query_future = torch.randn(bsz, h, generator=g)
    cand_mask = torch.ones(bsz, n, dtype=torch.bool)
    cand_mask[:, -3:] = False  # a few masked-out candidates, like real candidate masking
    scores = torch.randn(bsz, n, generator=g, requires_grad=True)
    return memory_c, offset_c, query_future, cand_mask, scores


def test_arm_partition_consistent():
    assert set(FROZEN_ARMS) | {'H2_joint_individual', 'H3_joint_individual_aggregate'} == set(ARMS)
    assert set(AGGREGATE_ARMS) == {'H1_frozen_individual_aggregate', 'H3_joint_individual_aggregate'}
    assert set(FROZEN_ARMS) & set(AGGREGATE_ARMS) == {'H1_frozen_individual_aggregate'}


def test_masked_candidates_get_zero_weight():
    memory_c, offset_c, query_future, cand_mask, scores = _toy_batch()
    y_soft, w = aggregate_prediction(memory_c, offset_c, scores, cand_mask, tau=0.1)
    assert torch.allclose(w[:, -3:], torch.zeros_like(w[:, -3:]), atol=1e-8)
    assert torch.allclose(w.sum(-1), torch.ones(w.size(0)), atol=1e-5)


def test_aggregate_loss_no_grad_into_candidate_future_or_query_future():
    memory_c, offset_c, query_future, cand_mask, scores = _toy_batch()
    memory_c.requires_grad_(True)
    offset_c.requires_grad_(True)
    query_future.requires_grad_(True)
    y_soft, w = aggregate_prediction(memory_c, offset_c, scores, cand_mask, tau=0.1)
    loss = ((y_soft - query_future.detach()) ** 2).mean()
    loss.backward()
    assert scores.grad is not None and scores.grad.abs().sum() > 0
    assert memory_c.grad is None, 'gradient must not flow into candidate futures'
    assert offset_c.grad is None, 'gradient must not flow into candidate offsets'
    assert query_future.grad is None, 'gradient must not flow into the query future'


def test_lambda_zero_reduces_to_individual_only():
    """L = L_ind + 0 * L_agg must be bit-identical to L_ind alone -- the
    aggregate term must not perturb the individual loss's value or its
    gradient when its own coefficient is zero."""
    memory_c, offset_c, query_future, cand_mask, scores = _toy_batch()
    y_soft, _ = aggregate_prediction(memory_c, offset_c, scores, cand_mask, tau=0.1)
    l_agg = ((y_soft - query_future) ** 2).mean()
    l_ind = (scores ** 2).mean()  # stand-in "individual" term, doesn't matter what it is
    lam = 0.0
    l_combined = l_ind + lam * l_agg
    assert torch.allclose(l_combined, l_ind)
    g_combined, = torch.autograd.grad(l_combined, scores, retain_graph=True)
    g_ind, = torch.autograd.grad(l_ind, scores)
    assert torch.allclose(g_combined, g_ind, atol=1e-7)


def test_effective_number_uniform_weights_equals_n():
    n = 16
    w = torch.full((3, n), 1.0 / n)
    neff = effective_number(w)
    assert torch.allclose(neff, torch.full((3,), float(n)), atol=1e-4)


def test_effective_number_one_hot_equals_one():
    w = torch.zeros(2, 10)
    w[:, 0] = 1.0
    neff = effective_number(w)
    assert torch.allclose(neff, torch.ones(2), atol=1e-6)


def test_weighted_aggregate_uses_tau_s_not_uniform_softmax():
    """Regression guard for the exact bug TRACK-H was told not to repeat:
    TRACK-G's weighted-aggregate used un-tempered softmax(scores_sel).
    With tau_s far from 1, the tempered and un-tempered weightings must
    differ measurably."""
    bsz, k, h = 3, 10, 4
    g = torch.Generator().manual_seed(0)
    memory_c = torch.randn(50, h, generator=g)
    offset_c = torch.randn(bsz, generator=g)
    query_future = torch.randn(bsz, h, generator=g)
    picks = torch.randint(0, 50, (bsz, k), generator=g)
    scores_sel = torch.randn(bsz, k, generator=g) * 5  # large-magnitude scores
    _, _, w_tempered = uniform_and_weighted_aggregate(memory_c, offset_c, picks, scores_sel, query_future, tau_s=0.1)
    w_untempered = torch.softmax(scores_sel, dim=-1)
    assert not torch.allclose(w_tempered, w_untempered, atol=1e-3)


def test_pairwise_mse_two_identical_points_is_zero():
    y = torch.ones(2, 5, 3)
    d = pairwise_mse(y)
    assert torch.allclose(d, torch.zeros(2), atol=1e-8)


def test_pairwise_mse_matches_manual_double_loop():
    torch.manual_seed(0)
    y = torch.randn(2, 6, 4)
    d = pairwise_mse(y)
    for b in range(2):
        vals = []
        for i in range(6):
            for j in range(i + 1, 6):
                vals.append(((y[b, i] - y[b, j]) ** 2).mean())
        expected = torch.stack(vals).mean()
        assert torch.allclose(d[b], expected, atol=1e-6)


def test_entropy_uniform_is_log_n():
    n = 8
    w = torch.full((2, n), 1.0 / n)
    ent = entropy_of(w)
    import math
    assert torch.allclose(ent, torch.full((2,), math.log(n)), atol=1e-4)


def test_identity_init_asym_metric_head_matches_cosine():
    torch.manual_seed(0)
    d = 16
    head = AsymMetric(d)
    zq = F.normalize(torch.randn(3, d), dim=-1)
    zk = F.normalize(torch.randn(5, d), dim=-1)
    s = head.project_q(zq) @ head.project_k(zk).T
    assert torch.allclose(s, zq @ zk.T, atol=1e-6)


def test_frozen_arms_require_lambda_zero_and_aggregate_arms_require_lambda_positive():
    for arm in FROZEN_ARMS:
        assert (arm in AGGREGATE_ARMS) == (arm == 'H1_frozen_individual_aggregate')
    assert 'H0_frozen_individual' not in AGGREGATE_ARMS
    assert 'H2_joint_individual' not in AGGREGATE_ARMS
