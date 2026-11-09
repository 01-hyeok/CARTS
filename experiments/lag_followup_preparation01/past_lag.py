"""Past-only fixed-window raw patch alignment. No future arguments or I/O.

Positive lag compares query[t] with candidate[t + lag]. All lags use the
identical query patches, within [r, L-r); no circular shift or padding.
This explicit fallback definition awaits a server-side PAMR audit.
"""
import numpy as np


def indices(length, radius=None, patch=16, stride=8):
    r = min(16, length // 8) if radius is None else int(radius)
    assert 0 <= r <= min(16, length // 8)
    starts = np.arange(r, length-r-patch+1, stride, dtype=np.int64)
    assert len(starts) and patch > 1 and stride > 0
    idx = starts[:, None] + np.arange(patch)
    lag = np.arange(-r, r+1)
    assert idx.min()-r >= 0 and idx.max()+r < length
    return idx, lag


def _unit_patch(x, floor=1e-8):
    x = np.asarray(x, np.float64)
    centered = x-x.mean(-1, keepdims=True)
    rms = np.sqrt(np.mean(centered*centered, axis=-1, keepdims=True))
    # Flat patches have zero similarity, not a spurious perfect match.
    unit = np.divide(centered, rms*np.sqrt(x.shape[-1]),
                     out=np.zeros_like(centered), where=rms>floor)
    return unit


def alignment(query_past, candidate_past, radius=None, candidate_chunk=64):
    q, x = np.asarray(query_past), np.asarray(candidate_past)
    assert q.ndim == 3 and x.ndim == 4, 'expected Q,C,L and Q,C,N,L'
    assert x.shape[:2] == q.shape[:2] and x.shape[-1] == q.shape[-1]
    assert np.isfinite(q).all() and np.isfinite(x).all() and candidate_chunk > 0
    ix, lag = indices(q.shape[-1], radius)
    qu = _unit_patch(q[..., ix])
    out = np.empty(x.shape[:-1]+(len(lag),), dtype=np.float64)
    # Candidate and lag loops bound workspace; no loop over channels.
    for a in range(0, x.shape[2], candidate_chunk):
        xx = x[:, :, a:a+candidate_chunk]
        for j, d in enumerate(lag):
            cu = _unit_patch(xx[..., ix+d])
            out[:, :, a:a+candidate_chunk, j] = (qu[:, :, None]*cu).sum(-1).mean(-1)
    return out, lag


def activity(query_past, candidate_past, train_scale):
    """Fixed activity attenuation; train-only channel scales, no future inputs."""
    q, x = np.asarray(query_past), np.asarray(candidate_past)
    scale = np.asarray(train_scale, np.float64)
    assert scale.shape == (q.shape[1],) and np.all(scale >= 1e-8)
    qstd, cstd = q.std(-1)[..., None], x.std(-1)
    return qstd/(qstd+scale[None, :, None])*cstd/(cstd+scale[None, :, None])


def freeze_activity_scale(train_past):
    return np.maximum(1e-8, np.median(np.asarray(train_past).std(-1), axis=0))


def shift_features(scores, lags, tau=.1):
    a = np.asarray(scores, np.float64)
    lags = np.asarray(lags, np.int64)
    assert a.shape[-1] == len(lags) and np.isfinite(a).all() and tau > 0
    zero = int(np.flatnonzero(lags == 0)[0])
    # Exact ties favor zero, then smallest absolute lag, then negative lag.
    order = np.lexsort((lags, np.abs(lags)))
    best_idx = order[np.argmax(a[..., order], axis=-1)]
    best = np.take_along_axis(a, best_idx[..., None], -1)[..., 0]
    p = np.exp((a-a.max(-1, keepdims=True))/tau); p /= p.sum(-1, keepdims=True)
    top = np.sort(a, axis=-1)[..., -2:]
    return dict(zero=a[..., zero], best=best, gain=best-a[..., zero],
                best_lag=lags[best_idx], margin=top[..., 1]-top[..., 0],
                entropy=-(p*np.log(np.maximum(p, 1e-300))).sum(-1))


def torch_alignment(query_past, candidate_past, radius=None, candidate_chunk=64):
    """All channels retained in each GPU operation; tested on CPU when Torch exists."""
    import torch
    q, x = query_past, candidate_past
    assert q.ndim == 3 and x.ndim == 4 and x.shape[:2] == q.shape[:2]
    assert x.shape[-1] == q.shape[-1] and candidate_chunk > 0
    ix, lags = indices(q.shape[-1], radius)
    ix = torch.as_tensor(ix, device=q.device)
    def unit(z):
        z=z.double();z=z-z.mean(-1,keepdim=True)
        rms=z.square().mean(-1,keepdim=True).sqrt()
        return torch.where(rms>1e-8,z/(rms.clamp_min(1e-8)*np.sqrt(z.shape[-1])),torch.zeros_like(z))
    qu=unit(q[...,ix]);out=[]
    for a in range(0,x.shape[2],candidate_chunk):
        xx=x[:,:,a:a+candidate_chunk];parts=[]
        for lag in lags:
            cu=unit(xx[...,ix+int(lag)])
            parts.append((qu[:,:,None]*cu).sum(-1).mean(-1))
        out.append(torch.stack(parts,-1))
    return torch.cat(out,2),lags
