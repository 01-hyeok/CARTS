"""Pure NumPy diagnostic contracts. No file writes, training, or GPU side effects."""
import numpy as np


def ordered(score, mask, largest=True):
    score, mask = np.asarray(score), np.asarray(mask, bool)
    assert score.shape == mask.shape and np.isfinite(score[mask]).all()
    a = np.where(mask, -score if largest else score, np.inf)
    return np.argsort(a, axis=-1, kind='stable')


def topk(score, mask, k=10):
    assert np.all(np.asarray(mask).sum(-1) >= k), 'INVALID: fewer than K valid candidates'
    return ordered(score, mask)[..., :k]


def gather(a, ids):
    return np.take_along_axis(a, ids, axis=-1)


def metrics(score, distance, mask, k=10):
    d = np.asarray(distance, np.float64)
    sel = topk(score, mask, k); oracle = topk(-d, mask, k)
    sd, od = gather(d, sel), gather(d, oracle)
    hit = (sel[..., :, None] == oracle[..., None, :]).any(-1)
    found = (oracle[..., :, None] == sel[..., None, :]).any(-1)
    ret, oret = sd.mean(-1), od.mean(-1)
    exchange = ((sd * ~hit).sum(-1) - (od * ~found).sum(-1)) / k
    assert np.array_equal((~hit).sum(-1), (~found).sum(-1))
    np.testing.assert_allclose(exchange, ret-oret, atol=2e-6, rtol=3e-6)
    assert (ret-oret >= -2e-6).all(), 'negative regret; do not clamp'
    disc = 1 / np.log2(np.arange(2, k+2))
    ndcg = ((1/(1+sd))*disc).sum(-1) / ((1/(1+od))*disc).sum(-1)
    return dict(selected=sel, oracle=oracle, retmse=ret, oracle_retmse=oret,
                regret=ret-oret, exchange=exchange, recall=hit.mean(-1), ndcg=ndcg)


def check_subset(full_ids, full_mask, pool_ids):
    # Production uses canonical global IDs 0..N-1. P100 retains its stored order.
    f, m, p = map(np.asarray, (full_ids, full_mask, pool_ids))
    assert p.shape[-1] == 100 and np.all(np.diff(np.sort(p, axis=-1), axis=-1) > 0)
    assert np.array_equal(f, np.broadcast_to(np.arange(f.shape[-1]), f.shape))
    assert np.all((p >= 0) & (p < f.shape[-1]))
    assert gather(m, p).all(), 'P100 includes masked Full candidate'


def log_softmax(x, mask):
    x = np.asarray(x, np.float64); m = np.asarray(mask, bool)
    z = np.where(m, x, -np.inf)
    a = z.max(-1, keepdims=True)
    logz = a + np.log(np.where(m, np.exp(z-a), 0).sum(-1, keepdims=True))
    return z-logz, logz[..., 0]


def normalization_control(logits, full_mask, pool_ids, tau=.1):
    # (...,heads,N), identical logits and P100 IDs for both normalizers.
    logp_f, zf = log_softmax(np.asarray(logits)/tau, full_mask)
    ph = np.broadcast_to(np.expand_dims(pool_ids, -2), (*logits.shape[:-1], pool_ids.shape[-1]))
    lp = gather(logits, ph)
    logp_p, zp = log_softmax(lp/tau, np.ones_like(lp, bool))
    mass = np.exp(zp-zf)
    restricted = np.exp(gather(logp_f, ph)).mean(-2)
    reconstructed = (mass[..., None]*np.exp(logp_p)).mean(-2)
    np.testing.assert_allclose(restricted, reconstructed, atol=2e-15, rtol=2e-12)
    weights = mass/mass.sum(-1, keepdims=True)
    if logits.shape[-2] == 1:
        np.testing.assert_array_equal(ordered(logp_p[..., 0, :], np.ones_like(pool_ids, bool)),
                                      ordered(gather(logp_f, ph)[..., 0, :], np.ones_like(pool_ids, bool)))
    return dict(p100=np.exp(logp_p).mean(-2), full_restricted=restricted,
                head_mass=mass, head_weight=weights, logZp=zp, logZf=zf)


def decomposition(rfull, rpool, rmodel, support):
    a,b,c = [np.asarray(x, np.float64) for x in (rfull,rpool,rmodel)]
    assert a.shape == b.shape == c.shape and all(np.isfinite(x).all() for x in (a,b,c))
    assert np.all(a <= b+2e-6) and np.all(a <= c+2e-6)
    if support == 'FULL': return dict(ranking=c-a, status='FULL_SELECTION_NO_POOL_DECOMPOSITION')
    assert support == 'P100' and np.all(b <= c+2e-6)
    pool, rank, total = b-a, c-b, c-a
    np.testing.assert_allclose(pool+rank, total, atol=1e-12, rtol=1e-12)
    den = total.sum()
    return dict(pool=pool, ranking=rank, total=total,
                pool_share=None if abs(den)<=1e-12 else pool.sum()/den,
                ranking_share=None if abs(den)<=1e-12 else rank.sum()/den)


def union_topk(head_ids, full_score, k=10):
    ids = np.unique(np.asarray(head_ids).reshape(-1)); assert len(ids)>=k
    # Canonical ascending Full ID tie order; score already normalized on original support.
    return ids[np.argsort(-np.asarray(full_score)[ids], kind='stable')[:k]]


def worst_timestamp_share(regret, fraction):
    r = np.asarray(regret, np.float64); assert r.ndim == 2 and 0<fraction<=1
    by_time = r.mean(1); n = max(1, int(np.ceil(len(r)*fraction)))
    den = by_time.sum()
    return None if abs(den)<=1e-12 else np.sort(by_time)[-n:].sum()/den


def train_label_gate(query_starts, length, horizon, train_start, train_end):
    q = np.asarray(query_starts)
    assert np.all(q>=train_start) and np.all(q+length+horizon<=train_end), 'probe future leaves train split'


def freeze_near_optimal_epsilon(validation_kth_distances):
    a=np.asarray(validation_kth_distances); assert a.ndim==2 and np.isfinite(a).all()
    return np.maximum(1e-8, .01*np.median(a,axis=0))
