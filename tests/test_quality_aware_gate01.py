import inspect

import torch

from models.RelationStage1 import stable_topk_indices
from utils.quality_aware_gate01 import (
    QA_FEATURE_DIM, QualityGate, TrainFeatureNormalizer, compute_qa_features,
    validation_oracle_lambda,
)
from utils.soft_topk_retrieval01 import soft_topk_select_and_forecast


def _make_inputs(B=4, N=20, K=10, T=6, seed=0):
    torch.manual_seed(seed)
    logits = torch.randn(B, N)
    valid_mask = torch.ones(B, N, dtype=torch.bool)
    p_full = torch.softmax(logits, dim=-1)
    picks = stable_topk_indices(p_full, K)
    y_sel = torch.randn(B, K, T)
    r, w = soft_topk_select_and_forecast(p_full, picks, y_sel)
    b_forecast = torch.randn(B, T)
    return dict(base_forecast=b_forecast, soft_forecast=r, y_sel=y_sel, weights=w,
                p_full=p_full, picks=picks, valid_mask=valid_mask)


def test_qa_gate_leakage_no_future_ground_truth_argument():
    """`compute_qa_features` must not accept the query's true future `Y`
    under any parameter name -- static signature check, not just a
    runtime check, so this fails loudly if someone adds such a param
    later."""
    sig = inspect.signature(compute_qa_features)
    forbidden = {'y', 'query_future', 'future', 'target', 'ground_truth', 'true_future'}
    for name in sig.parameters:
        assert name.lower() not in forbidden, (
            f'compute_qa_features has a parameter named {name!r} -- this looks like '
            f'it could be the query true future, which QA features must never see')


def test_qa_features_shape_and_finiteness():
    inputs = _make_inputs()
    feats = compute_qa_features(**inputs)
    assert feats.shape == (4, QA_FEATURE_DIM)
    assert torch.isfinite(feats).all()


def test_qa_gate_architecture_fixed():
    gate = QualityGate()
    linear_layers = [m for m in gate.net if isinstance(m, torch.nn.Linear)]
    assert len(linear_layers) == 2
    assert linear_layers[0].in_features == QA_FEATURE_DIM
    assert linear_layers[0].out_features == 16
    assert linear_layers[1].in_features == 16
    assert linear_layers[1].out_features == 1


def test_qa_gate_initialization_fairness_lambda_half_at_init():
    gate = QualityGate()
    feats = torch.randn(32, QA_FEATURE_DIM) * 5.0  # arbitrary, even large-scale inputs
    lam = gate(feats)
    assert torch.allclose(lam, torch.full_like(lam, 0.5), atol=1e-6), (
        'QA gate must output lambda=0.5 for every input at initialization, matching '
        'NOQA lambda_global = sigmoid(0) = 0.5')


def test_train_feature_normalizer_frozen_after_fit():
    torch.manual_seed(0)
    train_feats = torch.randn(100, QA_FEATURE_DIM) * 3.0 + 1.0
    norm = TrainFeatureNormalizer().fit(train_feats)
    val_feats = torch.randn(50, QA_FEATURE_DIM) * 10.0 - 5.0
    out = norm.transform(val_feats)
    # Normalization must use TRAIN stats, not val's own -- so val's own
    # normalized mean/std should NOT be ~0/~1 (that would mean leakage).
    assert not torch.allclose(out.mean(dim=0), torch.zeros(QA_FEATURE_DIM), atol=0.3)


def test_normalizer_to_device_moves_mean_and_std():
    """Regression test: fitting on CPU-collected features then
    transforming GPU tensors during training raised
    'Expected all tensors to be on the same device' until `.to(device)`
    was added -- caught by a real GPU smoke test of
    train_softtopk_e2e_v01.py's QA arm, not by the original test suite
    (which only ever ran on CPU tensors on both sides)."""
    train_feats = torch.randn(20, QA_FEATURE_DIM)
    norm = TrainFeatureNormalizer().fit(train_feats)
    norm.to('cpu')  # exercises the method on every machine, CUDA or not
    assert norm.mean.device.type == 'cpu'
    assert norm.std.device.type == 'cpu'
    if torch.cuda.is_available():
        norm.to('cuda')
        assert norm.mean.device.type == 'cuda'
        assert norm.std.device.type == 'cuda'
        gpu_feats = torch.randn(5, QA_FEATURE_DIM, device='cuda')
        out = norm.transform(gpu_feats)  # must not raise a device-mismatch error
        assert out.device.type == 'cuda'


def test_normalizer_refuses_transform_before_fit():
    norm = TrainFeatureNormalizer()
    import pytest
    with pytest.raises(RuntimeError):
        norm.transform(torch.randn(5, QA_FEATURE_DIM))


def test_validation_oracle_lambda_is_not_used_by_qa_gate():
    """The oracle is a diagnostic only -- confirm it is a free function
    taking query_future explicitly (so it's obviously not something the
    QA gate itself could call), and that it is NOT imported or
    referenced inside QualityGate's module-level forward path."""
    import utils.quality_aware_gate01 as mod
    src = inspect.getsource(mod.QualityGate)
    assert 'validation_oracle_lambda' not in src
    sig = inspect.signature(validation_oracle_lambda)
    assert 'query_future' in sig.parameters


def test_qa_feature_gradient_flows_to_upstream_probabilities():
    """At the exact zero-init point the QA gate's own output has zero
    gradient w.r.t. its input by construction (init fairness -- see
    test_qa_gate_initialization_fairness_lambda_half_at_init), so this
    checks the gradient PATH exists (as it must after any real training
    step moves the final layer off zero), not gradient flow through an
    untrained gate frozen at its fairness-init point."""
    B, N, K, T = 3, 15, 8, 4
    torch.manual_seed(2)
    logits = torch.randn(B, N, requires_grad=True)
    valid_mask = torch.ones(B, N, dtype=torch.bool)
    p_full = torch.softmax(logits, dim=-1)
    with torch.no_grad():
        picks = stable_topk_indices(p_full, K)
    y_sel = torch.randn(B, K, T)
    r, w = soft_topk_select_and_forecast(p_full, picks, y_sel)
    b_forecast = torch.randn(B, T)
    feats = compute_qa_features(b_forecast, r, y_sel, w, p_full, picks, valid_mask)
    gate = QualityGate()
    with torch.no_grad():
        for p in gate.net[-1].parameters():
            p.add_(torch.randn_like(p) * 0.5)  # simulate one post-init training step
    lam = gate(feats)
    loss = lam.sum()
    loss.backward()
    assert logits.grad is not None
    assert torch.any(logits.grad != 0)
