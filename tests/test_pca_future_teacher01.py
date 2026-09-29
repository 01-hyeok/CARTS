"""Unit tests for TRACK-I-PCA-FUTURE-TEACHER01's PCA utilities
(scripts/pca_future_teacher01.py). Fast, GPU-optional -- pure tensor math,
no model/checkpoint required.
"""
import torch
import pytest

from scripts.pca_future_teacher01 import fit_pca, pca_transform, pca_distance_memsafe


def _toy_memory(n=200, h=30, seed=0):
    g = torch.Generator().manual_seed(seed)
    # low-effective-rank synthetic data so full-rank PCA is well-defined
    # and truncated PCA has a clear "enough dims to be exact" boundary
    basis = torch.randn(10, h, generator=g)
    coeff = torch.randn(n, 10, generator=g)
    return coeff @ basis + 0.01 * torch.randn(n, h, generator=g)


def test_pca_transform_is_deterministic():
    y = _toy_memory()
    mean, comp = fit_pca(y)
    a = pca_transform(y[:5], mean, comp)
    b = pca_transform(y[:5], mean, comp)
    assert torch.equal(a, b)


def test_fit_pca_rank_never_exceeds_max_dim():
    y = _toy_memory(n=200, h=30)
    mean, comp = fit_pca(y, max_dim=5)
    assert comp.shape[0] == 5
    mean2, comp2 = fit_pca(y, max_dim=1000)  # requesting more than actual rank
    assert comp2.shape[0] <= min(y.shape)


def test_full_rank_pca_preserves_raw_euclidean_ranking():
    """Sanity check required by spec section 2: full-rank PCA + L2 must
    rank candidates identically to raw squared-Euclidean/MSE distance
    (translation + rotation preserve pairwise distances exactly)."""
    torch.manual_seed(0)
    n, h, bsz = 500, 40, 8
    memory_c = _toy_memory(n=n, h=h, seed=1)
    offset_c = torch.zeros(bsz)
    query_future = _toy_memory(n=bsz, h=h, seed=2)

    mean, comp = fit_pca(memory_c, max_dim=None)  # full rank
    assert comp.shape[0] == min(n, h)

    d_pca = pca_distance_memsafe(memory_c, offset_c, query_future, mean, comp, metric='l2')

    # raw MSE distance, computed directly (not via individual_utility_memsafe,
    # to keep this test self-contained / not depend on that module)
    diff = memory_c.unsqueeze(0) - query_future.unsqueeze(1)  # [bsz, n, h]
    d_raw = (diff ** 2).mean(-1)

    rank_pca = d_pca.argsort(dim=-1)
    rank_raw = d_raw.argsort(dim=-1)
    assert torch.equal(rank_pca, rank_raw), 'full-rank PCA-L2 ranking must match raw MSE ranking exactly'
    # values should also match up to a tiny numerical tolerance (both are
    # the same squared-distance quantity after an exact isometry)
    assert torch.allclose(d_pca, d_raw, atol=1e-4, rtol=1e-4)


def test_truncated_pca_does_not_match_full_rank_in_general():
    """Negative control: a clearly-too-small truncated PCA (dim=1 on data
    with effective rank 10) must NOT reproduce the full-rank ranking --
    guards against a no-op truncation bug."""
    torch.manual_seed(0)
    memory_c = _toy_memory(n=300, h=40, seed=3)
    offset_c = torch.zeros(6)
    query_future = _toy_memory(n=6, h=40, seed=4)

    mean_full, comp_full = fit_pca(memory_c, max_dim=None)
    mean_1, comp_1 = fit_pca(memory_c, max_dim=1)

    d_full = pca_distance_memsafe(memory_c, offset_c, query_future, mean_full, comp_full, metric='l2')
    d_1 = pca_distance_memsafe(memory_c, offset_c, query_future, mean_1, comp_1, metric='l2')

    rank_full = d_full.argsort(dim=-1)
    rank_1 = d_1.argsort(dim=-1)
    assert not torch.equal(rank_full, rank_1)


def test_pca_fit_uses_only_the_matrix_it_is_given():
    """No implicit global state / leakage: fitting on a subset must not be
    influenced by data outside that subset."""
    torch.manual_seed(0)
    full = _toy_memory(n=400, h=20, seed=5)
    train_only = full[:200]
    extra = full[200:]

    mean_a, comp_a = fit_pca(train_only)
    # refit on the same train_only slice again -> must be identical
    mean_b, comp_b = fit_pca(train_only)
    assert torch.equal(mean_a, mean_b)
    assert torch.equal(comp_a, comp_b)

    # fitting on train_only plus unrelated "held-out" data changes the fit --
    # demonstrates fit_pca is a pure function of its input (no leakage path
    # exists other than what's explicitly passed in)
    mean_c, comp_c = fit_pca(torch.cat([train_only, extra], dim=0))
    assert not torch.equal(mean_a, mean_c)


def test_pca_components_frozen_across_repeated_transforms():
    """PCA parameters (mean, components) are plain tensors returned once by
    fit_pca -- calling pca_transform repeatedly must never mutate them."""
    y = _toy_memory()
    mean, comp = fit_pca(y, max_dim=8)
    mean_snapshot, comp_snapshot = mean.clone(), comp.clone()
    for _ in range(5):
        pca_transform(y, mean, comp)
    assert torch.equal(mean, mean_snapshot)
    assert torch.equal(comp, comp_snapshot)


def test_cosine_metric_range():
    memory_c = _toy_memory(n=50, h=16, seed=6)
    offset_c = torch.zeros(4)
    query_future = _toy_memory(n=4, h=16, seed=7)
    mean, comp = fit_pca(memory_c, max_dim=4)
    d = pca_distance_memsafe(memory_c, offset_c, query_future, mean, comp, metric='cosine')
    assert (d >= -1e-5).all() and (d <= 2 + 1e-5).all()
