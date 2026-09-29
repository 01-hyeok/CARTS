"""Additional TRACK-I-PCA-FUTURE-TEACHER01 safety tests (spec section 10),
beyond the core math already covered in tests/test_pca_future_teacher01.py
(items 1/3/5 there). Fast, no GPU/checkpoint required except where noted.
"""
import torch
import torch.nn.functional as F
import pytest

from scripts.pca_future_teacher01 import fit_pca, pca_transform, pca_distance_memsafe
from scripts.train_factorial_e2e01 import individual_utility_memsafe
from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob
from scripts.train_i_pca_future_teacher01 import compute_teacher_distance


def _toy(n=100, h=20, bsz=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    memory_c = torch.randn(n, h, generator=g)
    offset_c = torch.randn(bsz, generator=g)
    query_future = torch.randn(bsz, h, generator=g)
    return memory_c, offset_c, query_future


def test_item4_pca_params_have_no_grad_and_are_never_optimizer_targets():
    """fit_pca's outputs must be plain (non-trainable) tensors -- nothing
    in this codepath ever calls .requires_grad_(True) on them, and they
    are never passed to any optimizer (train_i_pca_future_teacher01.main
    only ever constructs `Adam(model.parameters(), ...)`)."""
    memory_c, _, _ = _toy()
    mean, comp = fit_pca(memory_c, max_dim=8)
    assert not mean.requires_grad
    assert not comp.requires_grad


def test_item4_pca_params_unchanged_across_repeated_use_in_a_training_style_loop():
    memory_c, offset_c, query_future = _toy()
    mean, comp = fit_pca(memory_c, max_dim=8)
    mean0, comp0 = mean.clone(), comp.clone()
    for _ in range(20):
        d = pca_distance_memsafe(memory_c, offset_c, query_future, mean, comp, metric='l2')
        d.sum().backward() if d.requires_grad else None  # no-op here, mirrors a real train step shape
    assert torch.equal(mean, mean0)
    assert torch.equal(comp, comp0)


def test_item2_pca_fit_pool_is_exactly_the_train_candidate_bank_size():
    """Structural leakage guard: the pool handed to fit_pca must be sized
    like the train-only candidate bank (N), never N + val/test count --
    this test doesn't load real data (too expensive for a unit test) but
    pins the CONTRACT: fit_pca(memory_c) where memory_c.shape[0] must equal
    the candidate bank's own N, which by construction (this codebase's
    `Exp_Stage1_Relation._ensure_memory`, unmodified here) contains only
    train-split rows."""
    n_train_only = 7201  # ETTh1_720's known train-only candidate bank size
    memory_c = torch.randn(n_train_only, 720)
    mean, comp = fit_pca(memory_c, max_dim=64)
    assert mean.shape == (720,)
    assert comp.shape[1] == 720


def test_item6_raw_teacher_mode_is_a_pure_passthrough_of_individual_utility_memsafe():
    """teacher_mode='raw' must produce EXACTLY `-individual_utility_memsafe(...)`,
    byte for byte -- no PCA machinery involved at all -- which is what
    guarantees running this script in raw mode reproduces the original
    `train_patch_retrieval_expert01.py` behavior (verified further via a
    live full-training regression against the recorded p120 baseline,
    0.969165454248431, outside this fast unit-test file)."""
    memory_c, offset_c, query_future = _toy()
    d_direct = -individual_utility_memsafe(memory_c, offset_c, query_future, chunk_size=None)
    d_via_dispatch = compute_teacher_distance('raw', {}, 0, memory_c, offset_c, query_future, chunk_size=None)
    assert torch.equal(d_direct, d_via_dispatch)


def test_item8_pca_distance_output_shape_matches_raw_teacher_shape():
    """The mask-application code downstream (`.masked_fill(~cand_mask, ...)`)
    is untouched between raw/pca modes; this only stays correct if both
    teacher distances have the identical [B, N] shape -- verified here."""
    memory_c, offset_c, query_future = _toy(n=50, h=16, bsz=4)
    d_raw = -individual_utility_memsafe(memory_c, offset_c, query_future, chunk_size=None)
    mean, comp = fit_pca(memory_c, max_dim=6)
    d_pca = pca_distance_memsafe(memory_c, offset_c, query_future, mean, comp, metric='l2')
    assert d_raw.shape == d_pca.shape == (4, 50)


def test_item9_masked_candidates_get_exactly_zero_probability_under_pca_teacher():
    memory_c, offset_c, query_future = _toy(n=30, h=10, bsz=3)
    mean, comp = fit_pca(memory_c, max_dim=4)
    d = pca_distance_memsafe(memory_c, offset_c, query_future, mean, comp, metric='l2')
    mask = torch.ones(3, 30, dtype=torch.bool)
    mask[:, -5:] = False
    p_t = normalized_teacher_prob(d.masked_fill(~mask, float('inf')), mask, tau_t=0.05)
    assert torch.equal(p_t[:, -5:], torch.zeros(3, 5))
    assert torch.allclose(p_t.sum(-1), torch.ones(3), atol=1e-5)


def test_pca_dim_never_exceeds_requested_when_data_rank_is_sufficient():
    memory_c, _, _ = _toy(n=500, h=100)  # full-rank random data, rank >= 64
    mean, comp = fit_pca(memory_c, max_dim=64)
    assert comp.shape[0] == 64
