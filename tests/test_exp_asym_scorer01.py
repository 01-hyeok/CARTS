"""EXP-ASYM-SCORER01: asymmetric scorer (R2 + W_q/W_k) invariants."""
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.DenseUtilityRetriever import AsymmetricUtilityHead, UtilityHead
from layers.retrieval_metric import cosine_init_deviation


def test_identity_init_asymmetric_equals_cosine_utility_head():
    """Section 4: at construction, AsymmetricUtilityHead's underlying metric
    must reproduce plain cosine to within float tolerance."""
    torch.manual_seed(0)
    head = AsymmetricUtilityHead(dim=16)
    deviation = cosine_init_deviation(head.metric, samples=128)
    assert deviation < 1e-6, f'identity-init deviation too large: {deviation}'


def test_identity_init_full_head_matches_cosine_utility_head_forward():
    """End-to-end: AsymmetricUtilityHead(h,E) must equal UtilityHead(h,E) at
    init (both start with scale=1, bias=0), not just the internal metric."""
    torch.manual_seed(0)
    d, bsz, n = 16, 4, 20
    h = torch.nn.functional.normalize(torch.randn(bsz, d), dim=-1)
    e = torch.nn.functional.normalize(torch.randn(n, d), dim=-1)
    cos_head = UtilityHead()
    asym_head = AsymmetricUtilityHead(dim=d)
    out_cos = cos_head(h, e)
    out_asym = asym_head(h, e)
    assert torch.allclose(out_cos, out_asym, atol=1e-6)


def test_projection_output_is_l2_normalised_before_score():
    """Section 4 warning: without renormalisation after projection, identity
    init would not exactly reproduce cosine once W drifts from I -- verified
    here by checking the metric's internal projections are unit-norm."""
    torch.manual_seed(0)
    d = 16
    head = AsymmetricUtilityHead(dim=d)
    h = torch.randn(3, d) * 5.0  # deliberately unnormalised input
    e = torch.randn(10, d) * 5.0
    q_proj = head.metric.project_query(h)
    k_proj = head.metric.project_key(e)
    q_norm = torch.nn.functional.normalize(q_proj, dim=-1)
    k_norm = torch.nn.functional.normalize(k_proj, dim=-1)
    # the metric's own score() call renormalises internally -- confirm by
    # comparing its output against a manual normalise-then-dot computation.
    manual = torch.matmul(q_norm, k_norm.transpose(0, 1))
    scored = head.metric.score(h, e)
    assert torch.allclose(manual, scored, atol=1e-6)


def test_gradient_flows_to_wq_and_wk():
    torch.manual_seed(0)
    d, bsz, n = 16, 3, 12
    h = torch.nn.functional.normalize(torch.randn(bsz, d), dim=-1)
    e = torch.nn.functional.normalize(torch.randn(n, d), dim=-1)
    head = AsymmetricUtilityHead(dim=d)
    out = head(h, e)
    out.sum().backward()
    assert head.metric.query_projection.weight.grad is not None
    assert head.metric.key_projection.weight.grad is not None
    assert float(head.metric.query_projection.weight.grad.abs().sum()) > 0
    assert float(head.metric.key_projection.weight.grad.abs().sum()) > 0


def test_asymmetric_adds_params_only_to_scorer_not_elsewhere():
    """Section 11: the only new trainable parameters vs. R2-cosine must be
    W_q/W_k (plus the same scale/bias every UtilityHead already has)."""
    d = 16
    cos_head = UtilityHead()
    asym_head = AsymmetricUtilityHead(dim=d)
    n_cos = sum(p.numel() for p in cos_head.parameters())
    n_asym = sum(p.numel() for p in asym_head.parameters())
    assert n_cos == 2  # scale, bias
    assert n_asym == 2 + 2 * d * d  # scale, bias, W_q (dxd), W_k (dxd)


def test_asymmetric_scorer_output_shape_matches_cosine_head():
    d, bsz, n = 16, 5, 30
    h = torch.randn(bsz, d)
    e = torch.randn(n, d)
    head = AsymmetricUtilityHead(dim=d)
    out = head(h, e)
    assert out.shape == (bsz, n)
