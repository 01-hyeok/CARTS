"""Unit tests for TRACK-P100-COARSE-TO-FINE-RETRIEVAL01's new, CPU
-only math -- the log-domain fusion endpoint behavior
(`diag_p100_coarse_v5_fusion01`), the vectorized Spearman helper, and
the adaptive-fusion feature extractor's basic shape/range invariants
(`train_p100_adaptive_fusion01`). Does not touch GPU, checkpoints, or
real data -- those are covered by the scripts' own printed sanity
asserts (S1-S8, endpoint_sanity) when run against real cells."""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diag_p100_coarse_v5_fusion01 import BETAS, EPS, spearman_batched
from scripts.train_p100_adaptive_fusion01 import (
    FEATURE_NAMES, N_FEATURES, AdaptiveBetaMLP, StaticScalarBeta, _entropy, _js_divergence, _margin,
    extract_features,
)


def _random_prob(bsz, n, seed):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(bsz, n, generator=g)
    return F.softmax(logits, dim=-1)


def test_fusion_endpoints_recover_pure_arms():
    p_c = _random_prob(4, 100, seed=0)
    p_v5 = _random_prob(4, 100, seed=1)
    for beta, expected in ((0.0, p_c), (1.0, p_v5)):
        raw = (1 - beta) * torch.log(p_c + EPS) + beta * torch.log(p_v5 + EPS)
        p_fused = F.softmax(raw, dim=-1)
        assert torch.allclose(p_fused, expected, atol=1e-5)


def test_fusion_interpolates_monotonically_in_log_domain():
    # a sanity check on the fusion formula itself (not a data claim): at a
    # candidate where p_c and p_v5 disagree in direction, log p_fused must
    # move monotonically from log p_c to log p_v5 as beta sweeps 0->1.
    p_c = torch.tensor([[0.9, 0.1]])
    p_v5 = torch.tensor([[0.1, 0.9]])
    prev = None
    for beta in BETAS:
        raw = (1 - beta) * torch.log(p_c + EPS) + beta * torch.log(p_v5 + EPS)
        if prev is not None:
            assert raw[0, 0] <= prev[0, 0] + 1e-6  # log p(candidate 0) decreases as beta rises
        prev = raw


def test_spearman_batched_matches_scipy_reference():
    scipy_stats = pytest.importorskip('scipy.stats')
    torch.manual_seed(0)
    a = torch.randn(5, 30)
    b = torch.randn(5, 30)
    got = spearman_batched(a, b)
    for i in range(5):
        ref, _ = scipy_stats.spearmanr(a[i].numpy(), b[i].numpy())
        assert abs(float(got[i]) - ref) < 1e-4


def test_spearman_batched_perfect_agreement_and_disagreement():
    a = torch.arange(10).float().unsqueeze(0)
    assert float(spearman_batched(a, a)) > 0.999
    assert float(spearman_batched(a, -a)) < -0.999


def test_entropy_bounds():
    bsz, n = 3, 100
    uniform = torch.full((bsz, n), 1.0 / n)
    onehot = F.one_hot(torch.zeros(bsz, dtype=torch.long), n).float()
    assert torch.allclose(_entropy(uniform), torch.full((bsz,), float(torch.log(torch.tensor(float(n))))), atol=1e-4)
    assert torch.allclose(_entropy(onehot), torch.zeros(bsz), atol=1e-4)


def test_js_divergence_zero_for_identical_and_bounded_by_ln2():
    p = _random_prob(4, 50, seed=2)
    q = _random_prob(4, 50, seed=3)
    assert torch.allclose(_js_divergence(p, p), torch.zeros(4), atol=1e-5)
    js = _js_divergence(p, q)
    assert (js >= 0).all()
    assert (js <= torch.log(torch.tensor(2.0)) + 1e-4).all()


def test_margin_is_nonnegative_and_zero_for_uniform():
    uniform = torch.full((2, 20), 1.0 / 20)
    assert torch.allclose(_margin(uniform, 1, 2), torch.zeros(2), atol=1e-6)
    p = _random_prob(2, 20, seed=4)
    assert (_margin(p, 1, 2) >= -1e-6).all()
    assert (_margin(p, 1, 10) >= _margin(p, 1, 2) - 1e-6).all()


def test_extract_features_shape_and_no_nan():
    bsz, n, n_slots, d = 4, 100, 5, 16
    p_c = _random_prob(bsz, n, seed=5)
    p_v5 = _random_prob(bsz, n, seed=6)
    p_heads = torch.stack([_random_prob(bsz, n, seed=7 + m) for m in range(n_slots)], dim=1)
    z_q = torch.randn(bsz, d)
    feats = extract_features(p_c, p_v5, p_heads, z_q, channel=2, n_channels=7)
    assert feats.shape == (bsz, N_FEATURES)
    assert len(FEATURE_NAMES) == N_FEATURES
    assert not torch.isnan(feats).any()
    assert not torch.isinf(feats).any()


def test_adaptive_beta_mlp_outputs_valid_distribution():
    model = AdaptiveBetaMLP(N_FEATURES, len(BETAS))
    feats = torch.randn(6, N_FEATURES)
    p = model(feats)
    assert p.shape == (6, len(BETAS))
    assert torch.allclose(p.sum(dim=-1), torch.ones(6), atol=1e-5)
    assert (p >= 0).all()


def test_static_scalar_beta_ignores_features_shares_across_batch():
    model = StaticScalarBeta(len(BETAS))
    feats_a = torch.randn(3, N_FEATURES)
    feats_b = torch.randn(3, N_FEATURES) * 100  # wildly different features
    p_a = model(feats_a)
    p_b = model(feats_b)
    assert torch.allclose(p_a, p_b)  # B2 control must be feature-independent by construction
    assert torch.allclose(p_a[0], p_a[1])  # identical across the batch (one global scalar)
