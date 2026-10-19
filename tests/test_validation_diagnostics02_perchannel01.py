import pytest
import torch

from utils.validation_diagnostics02_perchannel01 import (
    _check_cnt, global_oracle_lambda, oracle_headroom_report_v2, per_channel_oracle_lambda,
)


def test_rejects_non_3d_input_the_original_bug_this_prevents():
    """The original `per_channel_oracle_static_lambda` silently accepted
    [N, T] (2D) and summed over everything -- exactly the bug. This
    module's functions must refuse 2D input outright rather than
    silently flattening the channel axis away."""
    with pytest.raises(ValueError):
        _check_cnt('x', torch.randn(10, 4))  # 2D, missing the channel axis


def test_per_channel_lambda_differs_across_channels_when_optimal_lambda_differs():
    """Construct 2 channels with DIFFERENT true optimal lambda (channel 0
    wants lambda=1, channel 1 wants lambda=0) and confirm the per-channel
    oracle recovers each channel's own value, not one blended global
    scalar."""
    torch.manual_seed(0)
    N, T = 50, 4
    base = torch.randn(2, N, T)
    soft = torch.randn(2, N, T)
    # channel 0: Y == soft exactly (optimal lambda = 1)
    y0 = soft[0]
    # channel 1: Y == base exactly (optimal lambda = 0)
    y1 = base[1]
    y = torch.stack([y0, y1], dim=0)

    lam = per_channel_oracle_lambda(base, soft, y)
    assert lam.shape == (2,)
    assert float(lam[0]) > 0.9
    assert float(lam[1]) < 0.1


def test_global_lambda_is_a_single_blended_scalar_not_per_channel():
    """Same construction as above: the GLOBAL oracle must NOT recover
    each channel's individual optimum -- it blends them into one value
    strictly between the two channels' individual optima, demonstrating
    the original (mislabeled) function's real behavior."""
    torch.manual_seed(1)
    N, T = 50, 4
    base = torch.randn(2, N, T)
    soft = torch.randn(2, N, T)
    y0 = soft[0]
    y1 = base[1]
    y = torch.stack([y0, y1], dim=0)

    g_lam = global_oracle_lambda(base, soft, y)
    assert 0.0 < g_lam < 1.0  # neither channel's own extreme (1.0 or 0.0)


def test_oracle_headroom_report_v2_dynamic_never_worse_than_per_channel_or_global():
    torch.manual_seed(2)
    C, N, T = 3, 40, 5
    base = torch.randn(C, N, T)
    soft = base + torch.randn(C, N, T) * 0.5
    y = base + torch.randn(C, N, T) * 0.3
    report = oracle_headroom_report_v2(base, soft, y, global_lambda=0.5)
    assert report['dynamic_oracle_val_mse'] <= report['global_oracle_val_mse'] + 1e-6
    assert report['dynamic_oracle_val_mse'] <= report['per_channel_oracle_val_mse'] + 1e-6
    assert report['per_channel_improves_on_global']
    assert report['dynamic_improves_on_per_channel']
    assert len(report['per_channel_oracle_lambda']) == C


def test_oracle_headroom_report_v2_requires_channel_dim():
    base = torch.randn(40, 5)  # 2D -- missing channel axis
    soft = torch.randn(40, 5)
    y = torch.randn(40, 5)
    with pytest.raises(ValueError):
        oracle_headroom_report_v2(base, soft, y, global_lambda=0.5)
