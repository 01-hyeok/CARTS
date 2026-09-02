"""Retrieval-off: the same trained model, asked what it predicts without retrieval.

The point of the intervention is that only the retrieval signal moves. These
pin the two properties that make the resulting difference readable: the gate is
neutralised the way the fusion defines neutral, and nothing else in the forward
changes. Test 5 records that y_ret=0 and lambda=0 coincide under residual
fusion but not under mixture -- a distinction the diagnostic relies on when it
reads base_mse as the retrieval-off error.
"""

import torch

from layers.retrieval_gate import RetrievalGate


def _gate(fusion_mode):
    torch.manual_seed(0)
    return RetrievalGate(pred_len=8, hidden_dim=16, gate_mode='scalar',
                         fusion_mode=fusion_mode)


def test_residual_zero_retrieval_is_exactly_the_base_forecast():
    gate = _gate('residual')
    y_base, y_ret = torch.randn(4, 8), torch.randn(4, 8)
    off, _ = gate(y_base, torch.zeros_like(y_ret))
    assert torch.equal(off, y_base)


def test_residual_zero_retrieval_and_zero_lambda_agree():
    """Test 5: the two counterfactuals coincide here."""
    gate = _gate('residual')
    y_base, y_ret = torch.randn(4, 8), torch.randn(4, 8)
    by_value, _ = gate(y_base, torch.zeros_like(y_ret))
    by_lambda = y_base + torch.zeros(4, 1) * y_ret
    assert torch.allclose(by_value, by_lambda, atol=0)


def test_mixture_zero_retrieval_is_not_the_base_forecast():
    """Under mixture the gate still scales y_base, so the two differ. The
    diagnostic must not read base_mse as retrieval-off for such a run."""
    gate = _gate('mixture')
    y_base, y_ret = torch.randn(4, 8), torch.randn(4, 8)
    off, lam = gate(y_base, torch.zeros_like(y_ret))
    assert not torch.allclose(off, y_base, atol=1e-6)
    assert torch.allclose(off, (1.0 - lam) * y_base, atol=1e-6)


def test_retrieval_still_moves_the_prediction_when_left_on():
    """Guards against a neutral intervention that was neutral all along."""
    gate = _gate('residual')
    y_base, y_ret = torch.randn(4, 8), torch.randn(4, 8)
    on, _ = gate(y_base, y_ret)
    off, _ = gate(y_base, torch.zeros_like(y_ret))
    assert not torch.allclose(on, off, atol=1e-6)


def test_flag_defaults_off_and_is_read_from_config():
    from models.RelationStage2 import Model
    from tests.test_stage2_learned_score import _stage2_config

    assert Model(_stage2_config()).retrieval_off is False
    cfg = _stage2_config()
    cfg.stage2_retrieval_off = 1
    assert Model(cfg).retrieval_off is True


def test_retrieval_off_does_not_touch_the_checkpoint():
    """Test 2: it is an inference switch, not a different set of weights."""
    from models.RelationStage2 import Model
    from tests.test_stage2_learned_score import _stage2_config

    base = Model(_stage2_config())
    cfg = _stage2_config()
    cfg.stage2_retrieval_off = 1
    off = Model(cfg)
    assert base.state_dict().keys() == off.state_dict().keys()
