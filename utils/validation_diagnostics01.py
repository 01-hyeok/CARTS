"""TRACK-P100-SOFTTOPK-QUALITY-AWARE01 -- spec sections 19-20,
VALIDATION-only diagnostics (never used in training, never computed on
TEST before the final method is locked).

Section 19: dynamic-fusion oracle headroom -- per (query, channel),
    lambda*_oracle = clip((R-B)^T(Y-B) / ||R-B||^2, 0, 1)
(already implemented as `utils.quality_aware_gate01.validation_oracle_lambda`,
reused here, not reimplemented) -- and the three comparison MSEs:
Global-lambda (NOQA) Val MSE, Per-channel oracle-static Val MSE (one
lambda per channel, oracle-selected but channel-constant), Dynamic
oracle Val MSE (lambda*_oracle applied per query/channel), vs the
actual QA Val MSE.

Section 20: quality calibration -- does lambda(q,c) actually track
real retrieval quality? `G(q,c) = MSE(B,Y) - MSE(R,Y)` (positive =
retrieval beats base), correlated (Pearson + Spearman) against the
learned lambda, plus a lambda-bin breakdown of retrieval_better_fraction
and mean retrieval advantage.
"""
import torch

from utils.quality_aware_gate01 import validation_oracle_lambda


def per_channel_oracle_static_lambda(base_forecast, soft_forecast, query_future):
    """One lambda per CHANNEL (not per query), chosen as the value in
    [0,1] minimizing MSE over all queries for that channel -- a
    constrained, cheaper oracle than the fully dynamic one. Closed form:
    same ratio as the dynamic oracle, but numerator/denominator summed
    over the query axis before dividing (not clipped per-query)."""
    r_minus_b = soft_forecast - base_forecast  # [N, T]
    y_minus_b = query_future - base_forecast  # [N, T]
    num = (r_minus_b * y_minus_b).sum()
    den = (r_minus_b ** 2).sum().clamp_min(1e-12)
    lam = (num / den).clamp(0.0, 1.0)
    return float(lam)


def mse(y_hat, y):
    return float(((y_hat - y) ** 2).mean())


def oracle_headroom_report(base_forecast, soft_forecast, query_future, global_lambda):
    """All inputs: [N, T] (already flattened over query x channel rows,
    one row per (query, channel)). `global_lambda`: scalar float, the
    trained NOQA lambda. Returns the 4-way comparison table's numbers."""
    noqa_pred = base_forecast + global_lambda * (soft_forecast - base_forecast)
    noqa_mse = mse(noqa_pred, query_future)

    static_lam = per_channel_oracle_static_lambda(base_forecast, soft_forecast, query_future)
    static_pred = base_forecast + static_lam * (soft_forecast - base_forecast)
    static_mse = mse(static_pred, query_future)

    dyn_lam = validation_oracle_lambda(base_forecast, soft_forecast, query_future)  # [N]
    dyn_pred = base_forecast + dyn_lam.unsqueeze(-1) * (soft_forecast - base_forecast)
    dyn_mse = mse(dyn_pred, query_future)

    return {
        'global_lambda_noqa_val_mse': noqa_mse,
        'per_channel_oracle_static_val_mse': static_mse,
        'per_channel_oracle_static_lambda': static_lam,
        'dynamic_oracle_val_mse': dyn_mse,
        'positive_headroom_exists': dyn_mse < noqa_mse,
        'headroom_mse_reduction': noqa_mse - dyn_mse,
    }


def retrieval_advantage(base_forecast, soft_forecast, query_future):
    """G(q,c) = MSE(B,Y) - MSE(R,Y) per row. Positive = retrieval beats base."""
    mse_b = ((base_forecast - query_future) ** 2).mean(dim=-1)
    mse_r = ((soft_forecast - query_future) ** 2).mean(dim=-1)
    return mse_b - mse_r


def _pearson(x, y):
    xc = x - x.mean()
    yc = y - y.mean()
    num = (xc * yc).sum()
    den = (xc.pow(2).sum().sqrt() * yc.pow(2).sum().sqrt()).clamp_min(1e-12)
    return float(num / den)


def _spearman(x, y):
    rx = x.argsort().argsort().float()
    ry = y.argsort().argsort().float()
    return _pearson(rx, ry)


def quality_calibration_report(lam, advantage, bins=((0, .1), (.1, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.0))):
    """lam, advantage: [N] (one QA lambda and one G(q,c) per row)."""
    pearson = _pearson(lam, advantage)
    spearman = _spearman(lam, advantage)
    bin_report = []
    for lo, hi in bins:
        if hi >= 1.0:
            mask = (lam >= lo) & (lam <= hi)
        else:
            mask = (lam >= lo) & (lam < hi)
        n = int(mask.sum())
        if n == 0:
            bin_report.append({'range': [lo, hi], 'n': 0, 'retrieval_better_fraction': None, 'mean_advantage': None})
            continue
        adv_in_bin = advantage[mask]
        bin_report.append({
            'range': [lo, hi], 'n': n,
            'retrieval_better_fraction': float((adv_in_bin > 0).float().mean()),
            'mean_advantage': float(adv_in_bin.mean()),
        })
    return {'pearson': pearson, 'spearman': spearman, 'bins': bin_report}
