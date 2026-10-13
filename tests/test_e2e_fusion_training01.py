import torch

from models.RelationStage1 import stable_topk_indices
from utils.e2e_fusion_training01 import (
    GlobalLambda, combined_loss, fuse_noqa, fuse_qa, soft_topk_forecast,
)
from utils.quality_aware_gate01 import QA_FEATURE_DIM, QualityGate, TrainFeatureNormalizer


def _make_inputs(B=4, N=20, K=10, T=6, seed=0):
    torch.manual_seed(seed)
    logits = torch.randn(B, N)
    p_full = torch.softmax(logits, dim=-1)
    picks = stable_topk_indices(p_full, K)
    y_sel = torch.randn(B, K, T)
    valid_mask = torch.ones(B, N, dtype=torch.bool)
    base = torch.randn(B, T)
    return p_full, picks, y_sel, valid_mask, base


def test_global_lambda_half_at_init():
    gl = GlobalLambda()
    lam = gl(8)
    assert torch.allclose(lam, torch.full((8,), 0.5), atol=1e-6)


def test_fuse_noqa_matches_manual_formula():
    p_full, picks, y_sel, valid_mask, base = _make_inputs()
    r, w = soft_topk_forecast(p_full, picks, y_sel)
    gl = GlobalLambda()
    y_hat, lam = fuse_noqa(base, r, gl)
    manual = base + 0.5 * (r - base)
    assert torch.allclose(y_hat, manual, atol=1e-6)
    assert torch.allclose(lam, torch.full_like(lam, 0.5))


def test_fuse_qa_runs_and_uses_normalizer():
    p_full, picks, y_sel, valid_mask, base = _make_inputs()
    r, w = soft_topk_forecast(p_full, picks, y_sel)
    gate = QualityGate()
    norm = TrainFeatureNormalizer()
    # fit on a separate "train" batch, not the one used for fusion.
    train_p, train_picks, train_y_sel, train_valid, train_base = _make_inputs(seed=99)
    train_r, train_w = soft_topk_forecast(train_p, train_picks, train_y_sel)
    from utils.quality_aware_gate01 import compute_qa_features
    train_feats = compute_qa_features(train_base, train_r, train_y_sel, train_w, train_p, train_picks, train_valid)
    norm.fit(train_feats)
    y_hat, lam, feats = fuse_qa(base, r, y_sel, w, p_full, picks, valid_mask, gate, norm)
    assert y_hat.shape == base.shape
    assert feats.shape == (base.size(0), QA_FEATURE_DIM)
    # At QA gate's own zero-init, lambda must still be 0.5 regardless of fused features.
    assert torch.allclose(lam, torch.full_like(lam, 0.5), atol=1e-6)


def test_combined_loss_gamma_one_equals_plain_sum():
    y_hat = torch.randn(4, 6)
    y = torch.randn(4, 6)
    retrieval_loss = torch.tensor(0.37)
    total, l_fore = combined_loss(y_hat, y, retrieval_loss, gamma=1.0)
    manual = ((y_hat - y) ** 2).mean() + retrieval_loss
    assert torch.allclose(total, manual, atol=1e-6)
    assert torch.allclose(l_fore, ((y_hat - y) ** 2).mean(), atol=1e-6)


def test_fusion_gradient_flows_to_global_lambda_and_retrieval():
    p_full_logits = torch.randn(3, 15, requires_grad=True)
    p_full = torch.softmax(p_full_logits, dim=-1)
    with torch.no_grad():
        picks = stable_topk_indices(p_full, 8)
    y_sel = torch.randn(3, 8, 4)
    base = torch.randn(3, 4)
    r, w = soft_topk_forecast(p_full, picks, y_sel)
    gl = GlobalLambda()
    y_hat, lam = fuse_noqa(base, r, gl)
    y = torch.randn(3, 4)
    retrieval_loss = torch.tensor(0.0, requires_grad=False)
    total, _ = combined_loss(y_hat, y, retrieval_loss, gamma=1.0)
    total.backward()
    assert gl.a.grad is not None and float(gl.a.grad) != 0.0
    assert p_full_logits.grad is not None and torch.any(p_full_logits.grad != 0)
