import torch

from utils.validation_diagnostics01 import (
    mse, oracle_headroom_report, per_channel_oracle_static_lambda, quality_calibration_report,
    retrieval_advantage,
)


def test_dynamic_oracle_never_worse_than_noqa_or_static_oracle():
    torch.manual_seed(0)
    N, T = 200, 6
    base = torch.randn(N, T)
    soft = base + torch.randn(N, T) * 0.5
    y = base + torch.randn(N, T) * 0.3
    report = oracle_headroom_report(base, soft, y, global_lambda=0.5)
    # Dynamic oracle picks the BEST per-row lambda in [0,1] -- it cannot
    # be worse than a single global or per-channel-constant choice.
    assert report['dynamic_oracle_val_mse'] <= report['global_lambda_noqa_val_mse'] + 1e-6
    assert report['dynamic_oracle_val_mse'] <= report['per_channel_oracle_static_val_mse'] + 1e-6


def test_oracle_headroom_zero_when_retrieval_equals_base():
    N, T = 50, 4
    base = torch.randn(N, T)
    soft = base.clone()  # R == B -> lambda is irrelevant, all three MSEs equal MSE(B,Y)
    y = torch.randn(N, T)
    report = oracle_headroom_report(base, soft, y, global_lambda=0.3)
    assert abs(report['global_lambda_noqa_val_mse'] - report['dynamic_oracle_val_mse']) < 1e-4


def test_retrieval_advantage_sign_matches_which_is_closer():
    base = torch.tensor([[0.0, 0.0]])
    soft_better = torch.tensor([[1.0, 1.0]])
    y = torch.tensor([[1.0, 1.0]])
    adv = retrieval_advantage(base, soft_better, y)
    assert float(adv[0]) > 0  # retrieval (soft) is exactly right, base is not -> retrieval better

    soft_worse = torch.tensor([[5.0, 5.0]])
    adv2 = retrieval_advantage(base, soft_worse, y)
    assert float(adv2[0]) < 0  # retrieval far from target, base closer -> base better


def test_quality_calibration_detects_perfect_positive_correlation():
    torch.manual_seed(1)
    n = 300
    advantage = torch.randn(n)
    lam = advantage.clone()  # perfectly correlated by construction
    report = quality_calibration_report(lam, advantage)
    assert report['pearson'] > 0.99
    assert report['spearman'] > 0.99


def test_quality_calibration_near_zero_correlation_for_independent_signals():
    torch.manual_seed(2)
    n = 2000
    lam = torch.rand(n)
    advantage = torch.randn(n)  # independent of lam
    report = quality_calibration_report(lam, advantage)
    assert abs(report['pearson']) < 0.15


def test_quality_calibration_bins_cover_full_range_and_sum_to_n():
    torch.manual_seed(3)
    n = 600
    lam = torch.rand(n)
    advantage = torch.randn(n)
    report = quality_calibration_report(lam, advantage)
    total = sum(b['n'] for b in report['bins'])
    assert total == n


def test_per_channel_oracle_static_lambda_is_clipped_to_unit_interval():
    N, T = 30, 3
    base = torch.zeros(N, T)
    soft = torch.ones(N, T)
    y = torch.full((N, T), 10.0)  # true value far beyond soft -- raw optimal lambda > 1
    lam = per_channel_oracle_static_lambda(base, soft, y)
    assert 0.0 <= lam <= 1.0
