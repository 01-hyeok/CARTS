"""TRACK-V5-HEAD-OBJECTIVE-COSINE-FIX01 -- synthetic unit tests for the
Error Dot / Error Cosine normalization bugfix in
`scripts/diag_v5_head_objective01.py`'s PHASE D block. Pure CPU.
"""
import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-8


def _raw_dot_cos_dot(e_m, e_n):
    """Mirrors the FIXED implementation exactly: raw dot feeds cosine,
    dot/H feeds Error Dot."""
    h = e_m.size(-1)
    raw_dot = (e_m * e_n).sum(-1)
    dot = raw_dot / h
    norm_prod = (e_m.norm(dim=-1) * e_n.norm(dim=-1)).clamp_min(EPS)
    cos = raw_dot / norm_prod
    return dot, cos


def test_test1_identical_vectors_cosine_one():
    e1 = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    dot, cos = _raw_dot_cos_dot(e1, e1.clone())
    assert torch.allclose(cos, torch.tensor([1.0]), atol=1e-6)


def test_test2_opposite_vectors_cosine_minus_one():
    e1 = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    dot, cos = _raw_dot_cos_dot(e1, -e1)
    assert torch.allclose(cos, torch.tensor([-1.0]), atol=1e-6)


def test_test3_orthogonal_vectors_cosine_zero():
    e1 = torch.tensor([[1.0, 0.0]])
    e2 = torch.tensor([[0.0, 1.0]])
    dot, cos = _raw_dot_cos_dot(e1, e2)
    assert torch.allclose(cos, torch.tensor([0.0]), atol=1e-6)
    assert torch.allclose(dot, torch.tensor([0.0]), atol=1e-6)


def test_test4_cosine_invariant_to_horizon_length():
    torch.manual_seed(0)
    base = torch.randn(3, 1)
    e1_96 = base.repeat(1, 96)
    e2_96 = (base * 0.5 + 1.0).repeat(1, 96)
    e1_720 = base.repeat(1, 720)
    e2_720 = (base * 0.5 + 1.0).repeat(1, 720)
    _, cos_96 = _raw_dot_cos_dot(e1_96, e2_96)
    _, cos_720 = _raw_dot_cos_dot(e1_720, e2_720)
    assert torch.allclose(cos_96, cos_720, atol=1e-5), \
        f'cosine must not shrink with horizon length: {cos_96} vs {cos_720}'


def test_test5_error_dot_preserves_mse_scale_definition():
    e1 = torch.tensor([[2.0, 2.0]])
    e2 = torch.tensor([[2.0, 2.0]])
    dot, cos = _raw_dot_cos_dot(e1, e2)
    expected_dot = ((e1 * e2).sum(-1) / e1.size(-1))
    assert torch.allclose(dot, expected_dot, atol=1e-6)
    assert torch.allclose(dot, torch.tensor([4.0]), atol=1e-6)  # (2*2+2*2)/2 = 4


def test_test6_zero_norm_no_nan_inf_not_hardcoded():
    e1 = torch.zeros(1, 5)
    e2 = torch.zeros(1, 5)
    dot, cos = _raw_dot_cos_dot(e1, e2)
    assert torch.isfinite(dot).all() and torch.isfinite(cos).all()
    # zero-vs-zero is a neutral default (0), never an arbitrary +-1
    assert cos.item() != 1.0 and cos.item() != -1.0
    assert torch.allclose(cos, torch.tensor([0.0]), atol=1e-6)

    e3 = torch.zeros(1, 5)
    e4 = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    dot2, cos2 = _raw_dot_cos_dot(e3, e4)
    assert torch.isfinite(dot2).all() and torch.isfinite(cos2).all()
    assert cos2.item() != 1.0 and cos2.item() != -1.0


def test_test7_symmetry():
    torch.manual_seed(1)
    e1 = torch.randn(4, 10)
    e2 = torch.randn(4, 10)
    _, cos_mn = _raw_dot_cos_dot(e1, e2)
    _, cos_nm = _raw_dot_cos_dot(e2, e1)
    assert torch.allclose(cos_mn, cos_nm, atol=1e-6)


def test_test8_range_bounded():
    torch.manual_seed(2)
    e1 = torch.randn(50, 20)
    e2 = torch.randn(50, 20)
    _, cos = _raw_dot_cos_dot(e1, e2)
    assert float(cos.min()) >= -1.0 - 1e-5
    assert float(cos.max()) <= 1.0 + 1e-5


def test_test9_matches_reference_cosine_similarity():
    torch.manual_seed(3)
    e1 = torch.randn(8, 30)
    e2 = torch.randn(8, 30)
    _, cos = _raw_dot_cos_dot(e1, e2)
    ref = F.cosine_similarity(e1, e2, dim=-1)
    assert torch.allclose(cos, ref, atol=1e-5)
    np_cos = np.array([
        float(np.dot(e1[i].numpy(), e2[i].numpy()) /
             (np.linalg.norm(e1[i].numpy()) * np.linalg.norm(e2[i].numpy()) + 1e-12))
        for i in range(8)
    ])
    assert np.allclose(cos.numpy(), np_cos, atol=1e-5)


def test_test10_regression_old_buggy_formula_scaled_by_horizon():
    """Reproduces the ORIGINAL bug on synthetic input: using the
    already-/H dot as the cosine numerator against the un-scaled norm
    product silently divides the true cosine by H."""
    torch.manual_seed(4)
    for h in (96, 720):
        e1 = torch.randn(5, h)
        e2 = torch.randn(5, h)
        raw_dot = (e1 * e2).sum(-1)
        dot_buggy_as_before = raw_dot / h  # this is what the old code called `dot_mn`
        norm_prod = e1.norm(dim=-1) * e2.norm(dim=-1)
        old_buggy_cos = dot_buggy_as_before / (norm_prod + EPS)  # the ORIGINAL bug line
        true_cos = raw_dot / norm_prod.clamp_min(EPS)
        assert torch.allclose(old_buggy_cos * h, true_cos, atol=1e-4), \
            f'expected old_buggy_cos*H == true_cos at H={h}'
        assert not torch.allclose(old_buggy_cos, true_cos, atol=1e-3), \
            'the bug must actually change the numeric result, or this regression test is vacuous'
