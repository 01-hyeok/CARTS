"""TRACK-P100-QA-CAUSAL-VALIDATION01 -- Phase A4 fix of
`utils/validation_diagnostics01.py`'s `per_channel_oracle_static_lambda`.

That function's docstring/name say "per-channel", but its
implementation takes `[N, T]`-shaped input and calls `.sum()` with no
`dim` argument -- summing over EVERYTHING, including the channel axis
if the caller's `N` rows actually span multiple channels (which is how
`oracle_headroom_report` calls it: once, on the whole flattened
query x channel tensor). So as actually used, it computes a single
GLOBAL scalar oracle lambda, not one lambda per channel -- confirmed by
reading both the function and its only caller.

This module keeps the ORIGINAL file completely unmodified (it still
backs the original track's own validation diagnostics) and provides
genuinely channel-aware replacements, operating on explicitly
`[C, N, T]`-shaped tensors (C channels, N queries per channel, T
horizon) so the channel axis can never be silently summed away by
omission:

  `global_oracle_lambda`        -- one scalar, reduced over (C, N, T).
  `per_channel_oracle_lambda`   -- one lambda PER CHANNEL, shape [C],
                                   reduced over (N, T) only.
  per-query/channel dynamic oracle is NOT reimplemented here --
  `utils.quality_aware_gate01.validation_oracle_lambda` already reduces
  only over T (the last dim) for whatever leading shape is given, so it
  is already correct for `[C, N, T]` input (returns `[C, N]`) and is
  reused as-is.
"""
import torch

from utils.quality_aware_gate01 import validation_oracle_lambda


def _check_cnt(name, t):
    if t.dim() != 3:
        raise ValueError(f'{name} must be [C, N, T] (3 dims), got shape {tuple(t.shape)} -- '
                         f'this module exists specifically to prevent the channel axis from being '
                         f'silently flattened away; pass channel-grouped tensors.')


def global_oracle_lambda(base_forecast, soft_forecast, query_future, eps=1e-12):
    """One scalar lambda, oracle-optimal over ALL (channel, query, time)
    rows pooled together. Inputs [C, N, T]. This is the function the
    ORIGINAL `per_channel_oracle_static_lambda` actually computes, under
    its true (not its claimed) name."""
    for name, t in (('base_forecast', base_forecast), ('soft_forecast', soft_forecast),
                    ('query_future', query_future)):
        _check_cnt(name, t)
    r_minus_b = soft_forecast - base_forecast
    y_minus_b = query_future - base_forecast
    num = (r_minus_b * y_minus_b).sum()
    den = (r_minus_b ** 2).sum().clamp_min(eps)
    return float((num / den).clamp(0.0, 1.0))


def per_channel_oracle_lambda(base_forecast, soft_forecast, query_future, eps=1e-12):
    """One lambda PER CHANNEL, oracle-optimal over that channel's (query,
    time) rows only -- the channel axis is preserved, not summed over.
    Inputs [C, N, T]. Returns a [C] tensor, each entry clipped to [0,1]
    independently (clip happens per-channel, matching the spec's
    `lambda_c^* = clip(sum_{q,t}(...) / sum_{q,t}(...), 0, 1)` formula)."""
    for name, t in (('base_forecast', base_forecast), ('soft_forecast', soft_forecast),
                    ('query_future', query_future)):
        _check_cnt(name, t)
    r_minus_b = soft_forecast - base_forecast  # [C, N, T]
    y_minus_b = query_future - base_forecast  # [C, N, T]
    num = (r_minus_b * y_minus_b).sum(dim=(1, 2))  # [C]
    den = (r_minus_b ** 2).sum(dim=(1, 2)).clamp_min(eps)  # [C]
    return (num / den).clamp(0.0, 1.0)


def mse(y_hat, y):
    return float(((y_hat - y) ** 2).mean())


def oracle_headroom_report_v2(base_forecast, soft_forecast, query_future, global_lambda):
    """Channel-aware replacement for `oracle_headroom_report`. Inputs
    [C, N, T]; `global_lambda`: the TRAINED NOQA scalar (not an oracle).

    Returns the 4-way comparison: NOQA (trained global lambda),
    Global Oracle (data-optimal single scalar, pooled over everything),
    Per-channel Oracle (one lambda per channel, broadcast over that
    channel's queries), Dynamic Oracle (one lambda per query AND
    channel)."""
    for name, t in (('base_forecast', base_forecast), ('soft_forecast', soft_forecast),
                    ('query_future', query_future)):
        _check_cnt(name, t)

    noqa_pred = base_forecast + global_lambda * (soft_forecast - base_forecast)
    noqa_mse = mse(noqa_pred, query_future)

    g_lam = global_oracle_lambda(base_forecast, soft_forecast, query_future)
    global_oracle_pred = base_forecast + g_lam * (soft_forecast - base_forecast)
    global_oracle_mse = mse(global_oracle_pred, query_future)

    c_lam = per_channel_oracle_lambda(base_forecast, soft_forecast, query_future)  # [C]
    per_channel_pred = base_forecast + c_lam.view(-1, 1, 1) * (soft_forecast - base_forecast)
    per_channel_mse = mse(per_channel_pred, query_future)

    dyn_lam = validation_oracle_lambda(base_forecast, soft_forecast, query_future)  # [C, N]
    dyn_pred = base_forecast + dyn_lam.unsqueeze(-1) * (soft_forecast - base_forecast)
    dyn_mse = mse(dyn_pred, query_future)

    return {
        'noqa_val_mse': noqa_mse,
        'global_oracle_lambda': g_lam,
        'global_oracle_val_mse': global_oracle_mse,
        'per_channel_oracle_lambda': c_lam.tolist(),
        'per_channel_oracle_val_mse': per_channel_mse,
        'dynamic_oracle_val_mse': dyn_mse,
        'positive_headroom_exists': dyn_mse < noqa_mse,
        'headroom_mse_reduction': noqa_mse - dyn_mse,
        'per_channel_improves_on_global': per_channel_mse <= global_oracle_mse,
        'dynamic_improves_on_per_channel': dyn_mse <= per_channel_mse,
    }
