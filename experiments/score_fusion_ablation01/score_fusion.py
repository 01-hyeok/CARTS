"""Frozen FP32 head cosines; only fusion changes. No candidate-axis reduction in E1-E4."""
import math
import numpy as np

GAMMAS = (.01, .03, .05, .1, .2, .5, 1.)

def numpy_fusion(g, method, gamma=None):
    z = np.asarray(g, dtype=np.float64)
    assert z.shape[-2] == 5 and np.isfinite(z).all()
    if method == 'E1': return sum(z[..., h, :] for h in range(5)) / 5
    if method == 'E2': return z.max(-2)
    if method == 'E3': return np.sort(z, axis=-2)[..., -2:, :].sum(-2) / 2
    if method == 'E4':
        assert gamma > 0
        m = z.max(-2)
        return m + gamma * np.log(sum(np.exp((z[..., h, :] - m) / gamma) for h in range(5)) / 5)
    raise ValueError(method)

def fuse(g, method, gamma=None, dtype=None):
    import torch
    z = g.to(torch.float64 if dtype is None else dtype)
    assert z.shape[-2] == 5
    if method == 'E1': return (z[..., 0, :] + z[..., 1, :] + z[..., 2, :] + z[..., 3, :] + z[..., 4, :]) / 5
    if method == 'E2': return z.max(-2).values
    if method == 'E3':
        s = z.sort(-2).values
        return (s[..., 4, :] + s[..., 3, :]) / 2
    if method == 'E4':
        assert gamma > 0
        m = z.max(-2).values
        v = sum(torch.exp((z[..., h, :] - m) / gamma) for h in range(5)) / 5
        return m + gamma * v.log()
    raise ValueError(method)

def gamma_select(rows):
    """Input is explicitly native validation ONLY; deterministic smaller-gamma tie."""
    assert len(rows) == len(GAMMAS)
    assert all(r['split'] == 'val' and r['training_support'] == r['evaluation_support'] for r in rows)
    assert set(r['gamma'] for r in rows) == set(GAMMAS)
    assert all(math.isfinite(r['retmse']) for r in rows)
    return min(rows, key=lambda r: (r['retmse'], r['gamma']))['gamma']
