"""Pin the Stage-2 evaluation protocol so the offset bugs cannot come back.

RelationStage2 fuses in delta space and restores `x[:, -1:, :]` only at the
boundary. Two diagnostics got this wrong in opposite directions -- one dropped
the offset, one added it twice -- and produced numbers that looked like model
differences. These tests make both mistakes fail loudly.
"""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.RelationStage2 import BaseForecastHead
from utils.retrieval_diagnostics import base_forecast, mse_mae


class _Stub(nn.Module):
    """Minimal stand-in exposing just the base head, as base_forecast expects."""

    def __init__(self, seq_len, pred_len, channels):
        super().__init__()
        self.base_head = BaseForecastHead(
            seq_len=seq_len, pred_len=pred_len, channels=channels,
            mode='shared_target_linear',
        )


def test_base_head_alone_is_delta_space():
    """The head de-offsets its input and never restores it."""
    torch.manual_seed(0)
    head = BaseForecastHead(8, 4, 3, mode='shared_target_linear')
    x = torch.randn(5, 8, 3)
    shifted = x + 7.0
    # A constant shift of the input leaves the de-offset head output unchanged.
    assert torch.allclose(head(x), head(shifted), atol=1e-5)


def test_base_forecast_restores_the_offset():
    """base_forecast must match what RelationStage2.forward compares to targets."""
    torch.manual_seed(0)
    model = _Stub(8, 4, 3)
    x = torch.randn(5, 8, 3)
    expected = model.base_head(x) + x[:, -1:, :]
    assert torch.allclose(base_forecast(model, x), expected, atol=1e-6)


def test_base_forecast_tracks_a_constant_shift():
    """Shifting the whole window must shift the forecast by the same amount."""
    torch.manual_seed(0)
    model = _Stub(8, 4, 3)
    x = torch.randn(5, 8, 3)
    delta = 3.5
    moved = base_forecast(model, x + delta) - base_forecast(model, x)
    assert torch.allclose(moved, torch.full_like(moved, delta), atol=1e-5)


def test_rebuilding_final_from_outputs_needs_the_pure_correction():
    """y_ret carries the offset; scaling it directly double-counts it.

    This is the exact arithmetic of the second bug, written so the wrong form
    is visibly wrong and the right form is exact.
    """
    torch.manual_seed(0)
    bsz, horizon, channels = 4, 6, 3
    offset = torch.randn(bsz, 1, channels)
    y_base_delta = torch.randn(bsz, horizon, channels)
    y_ret_delta = torch.randn(bsz, horizon, channels)
    lam = torch.rand(bsz, channels).unsqueeze(1)

    y_final = y_base_delta + lam * y_ret_delta + offset      # what the model does
    y_base_out = y_base_delta + offset
    y_ret_out = y_ret_delta + offset

    wrong = y_base_out + lam * y_ret_out
    right = y_base_out + lam * (y_ret_out - offset)

    assert torch.allclose(right, y_final, atol=1e-6)
    assert not torch.allclose(wrong, y_final, atol=1e-3)
    # The whole discrepancy is exactly lam * offset.
    assert torch.allclose(wrong - y_final, lam * offset, atol=1e-6)


def test_canonical_metric_rejects_broadcasting():
    """Shape mismatch must raise rather than silently broadcast."""
    pred = torch.randn(4, 6, 3)
    target = torch.randn(4, 6, 1)
    try:
        mse_mae(pred, target)
    except Exception:
        return
    raise AssertionError('mse_mae silently broadcast a shape mismatch')
