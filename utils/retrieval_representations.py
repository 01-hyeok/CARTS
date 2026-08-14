"""Input representations for the retrieval-only Recall study.

Every representation maps a window [B, L] (one channel) to a vector that the
similarity is computed on. Three of the four are plain per-window transforms;
`decomposition` is not, because its fusion happens at the score level, so it
exposes two parts and the scorer combines them (see `decompose_trend_seasonal`).

Shapes, for seq_len L:
    raw              -> L
    delta_last       -> L
    arima_residual   -> L - 2
    decomposition    -> (L, L)   two parts, scored separately

Nothing here touches the future. The oracle is always raw-future MSE and is
built independently of the representation, which is what keeps the 192
configurations comparable.
"""

import numpy as np
import torch

REPRESENTATIONS = (
    'raw', 'delta_last', 'arima_residual', 'decomposition', 'sarima_residual',
    'diff1',
)

# Seasonal period per sampling rate, both one day. ETTh1 is hourly and ETTm1 is
# 15-minute, and their autocorrelation at these lags matches almost exactly
# (0.771 vs 0.767), so the two are the corresponding choice rather than two
# unrelated ones.
DEFAULT_PERIOD = {'h': 24, 't': 96, '15min': 96}


def representation_length(seq_len, representation, period=1):
    seq_len = int(seq_len)
    if representation in ('raw', 'delta_last', 'decomposition'):
        return seq_len
    if representation == 'diff1':
        if seq_len < 2:
            raise ValueError('diff1 requires seq_len >= 2')
        return seq_len - 1
    if representation == 'arima_residual':
        if seq_len < 3:
            raise ValueError('arima_residual requires seq_len >= 3')
        # One step is lost to the difference and one more to the AR(1) lag.
        return seq_len - 2
    if representation == 'sarima_residual':
        period = int(period)
        length = seq_len - period - 1
        if length < 2:
            raise ValueError(
                f'sarima_residual with period={period} needs seq_len > {period + 2}, '
                f'got seq_len={seq_len}'
            )
        return length
    raise ValueError(f'Unsupported representation: {representation}')


def moving_average(x, kernel_size):
    """Autoformer-style trend: centred moving average with edge padding.

    x: [..., L] -> [..., L]
    """
    if kernel_size < 1:
        raise ValueError('moving_average kernel_size must be >= 1')
    shape = x.shape
    flat = x.reshape(-1, 1, shape[-1])
    front = kernel_size // 2
    back = kernel_size - 1 - front
    padded = torch.cat(
        [
            flat[..., :1].expand(-1, -1, front),
            flat,
            flat[..., -1:].expand(-1, -1, back),
        ],
        dim=-1,
    )
    trend = torch.nn.functional.avg_pool1d(padded, kernel_size=kernel_size, stride=1)
    return trend.reshape(shape)


def decompose_trend_seasonal(x, kernel_size=25):
    """T = MA(X), S = X - T. Returns the two parts, not a single vector."""
    trend = moving_average(x, kernel_size)
    return trend, x - trend


def fit_ar1_on_differences(series, lag=1, eps=1e-8):
    """Least-squares (c, phi) for  d_t = c + phi * d_{t-1} + r_t  on a difference.

    lag=1 is the plain first difference; lag=period is the seasonal difference
    x_t - x_{t-period}.

    series: 1-D array of one channel over the whole training split. The fit is
    global per channel, so every window is filtered by the same (c, phi) and the
    residual stays comparable across queries -- a per-window fit would re-centre
    each window and wash that comparability out.
    """
    series = np.asarray(series, dtype=np.float64).reshape(-1)
    lag = int(lag)
    if lag < 1:
        raise ValueError('lag must be >= 1')
    d = series[lag:] - series[:-lag]
    if d.size < 3:
        raise ValueError('AR(1) fit needs at least 3 differenced points')
    y = d[1:]
    x = d[:-1]
    x_mean = x.mean()
    y_mean = y.mean()
    var = ((x - x_mean) ** 2).sum()
    if var < eps:
        return float(y_mean), 0.0
    phi = float(((x - x_mean) * (y - y_mean)).sum() / var)
    c = float(y_mean - phi * x_mean)
    return c, phi


def arima_residual(x, c, phi, lag=1):
    """r_t = dx_t - (c + phi * dx_{t-1}) on a lag-`lag` difference.

    lag=1      : x: [..., L] -> [..., L-2]
    lag=period : x: [..., L] -> [..., L-period-1]

    A seasonal lag removes the daily cycle instead of the step-to-step change.
    On 15-minute data the first difference strips ~95% of the series variance
    because neighbouring samples correlate at 0.977, leaving mostly measurement
    noise; the seasonal difference removes ~54% and keeps real structure.
    """
    lag = int(lag)
    dx = x[..., lag:] - x[..., :-lag]
    return dx[..., 1:] - (c + phi * dx[..., :-1])


def transform(x, representation, ar_params=None, moving_avg=25, period=1):
    """Apply a representation to one channel's windows.

    x: [B, L] float tensor.
    Returns a tuple of parts. Every representation returns exactly one part
    except `decomposition`, which returns (trend, seasonal).
    """
    if representation == 'raw':
        return (x,)
    if representation == 'delta_last':
        return (x - x[..., -1:],)
    if representation == 'diff1':
        # The plain first difference, with no AR step. Paired with
        # arima_residual this separates what the differencing does from what
        # removing the AR(1) component on top of it does.
        return (x[..., 1:] - x[..., :-1],)
    if representation == 'arima_residual':
        if ar_params is None:
            raise ValueError('arima_residual requires (c, phi)')
        c, phi = ar_params
        return (arima_residual(x, c, phi),)
    if representation == 'sarima_residual':
        if ar_params is None:
            raise ValueError('sarima_residual requires (c, phi)')
        c, phi = ar_params
        return (arima_residual(x, c, phi, lag=period),)
    if representation == 'decomposition':
        return decompose_trend_seasonal(x, moving_avg)
    raise ValueError(f'Unsupported representation: {representation}')
