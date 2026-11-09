"""CPU/GPU tensor primitives for frozen retrieval diagnostics, not a trainer.

All model scores enter from the audited inference adapter. Ground-truth distances
are evaluation-only. Stable ties retain the inherited support ID order.
"""
import numpy as np
import torch
import torch.nn.functional as F


def stable_order(scores, mask=None, descending=True):
    if mask is None:
        mask = torch.ones_like(scores, dtype=torch.bool)
    assert scores.shape == mask.shape and bool(mask.any(-1).all())
    assert bool(torch.isfinite(scores[mask]).all())
    fill = -float('inf') if descending else float('inf')
    return torch.sort(scores.masked_fill(~mask, fill), dim=-1,
                      descending=descending, stable=True).indices


def selected_ids(scores, candidate_ids, mask, k=10):
    assert bool((mask.sum(-1) >= k).all())
    picks = stable_order(scores, mask)[:, :k]
    ids = candidate_ids.gather(1, picks)
    assert bool(mask.gather(1, picks).all())
    assert bool((ids.sort(-1).values[:, 1:] != ids.sort(-1).values[:, :-1]).all())
    return picks, ids


def ndcg_inverse_mse(distance, ranked, mask, k):
    """NDCG_inverse_MSE_v1; bounded nonnegative direct gain, no exponentiation."""
    assert bool((distance[mask] >= 0).all())
    assert bool((mask.sum(-1) >= k).all())
    gain = (1.0 / (1.0 + distance)).masked_fill(~mask, 0)
    ideal = stable_order(distance, mask, descending=False)[:, :k]
    disc = 1.0 / torch.log2(torch.arange(2, k+2, device=distance.device,
                                       dtype=distance.dtype))
    dcg = (gain.gather(1, ranked[:, :k]) * disc).sum(-1)
    idcg = (gain.gather(1, ideal) * disc).sum(-1)
    return dcg / idcg


def support_metrics(score, distance, mask=None, k=10):
    """Return per-query metrics; no dataset/batch averaging here."""
    if mask is None:
        mask = torch.ones_like(score, dtype=torch.bool)
    assert score.ndim == 2 and distance.shape == score.shape == mask.shape
    assert bool(torch.isfinite(distance[mask]).all())
    assert bool((distance[mask] >= 0).all()) and bool((mask.sum(-1) > k).all())
    model_order = stable_order(score, mask)
    oracle_order = stable_order(distance, mask, False)
    selected, oracle = model_order[:, :k], oracle_order[:, :k]
    # Keep canonical float32 distances/ordering, but accumulate in float64.
    # Weather channel14 can contain large nearly identical distances; separate
    # float32 subset sums otherwise fabricate nonzero exchange loss at ties.
    selected_distance = distance.gather(1, selected).double()
    oracle_distance = distance.gather(1, oracle).double()
    ret = selected_distance.mean(-1)
    oret = oracle_distance.mean(-1)
    hit = (selected[:, :, None] == oracle[:, None, :]).any(-1)
    found = (oracle[:, :, None] == selected[:, None, :]).any(-1)
    assert torch.equal((~hit).sum(-1), (~found).sum(-1))
    false_sum = (selected_distance * (~hit)).sum(-1)
    missed_sum = (oracle_distance * (~found)).sum(-1)
    exchange = (false_sum - missed_sum) / k
    regret = ret - oret
    assert torch.allclose(exchange, regret, atol=2e-6, rtol=3e-6)
    assert bool((regret >= -2e-6).all())
    ranks = torch.empty_like(model_order)
    ranks.scatter_(1, model_order, torch.arange(1, score.shape[-1]+1,
                                              device=score.device).expand_as(model_order))
    missed_ranks = ranks.gather(1, oracle).masked_fill(found, 0)
    sorted_d = distance.gather(1, oracle_order)
    sorted_s = score.gather(1, model_order)
    recall = hit.to(distance.dtype).mean(-1)
    return dict(retmse=ret, retmse_fp32_compat=distance.gather(1,selected).mean(-1), oracle_retmse=oret, ranking_regret=regret,
                exchange_regret=exchange, recall=recall, boundary_error=1-recall,
                oracle_boundary_gap=sorted_d[:, k]-sorted_d[:, k-1],
                student_boundary_gap=sorted_s[:, k-1]-sorted_s[:, k],
                student_top1_topk_gap=sorted_s[:, 0]-sorted_s[:, k-1],
                ndcg_inverse_mse=ndcg_inverse_mse(distance, model_order, mask, k),
                selected=selected, oracle=oracle, missed_ranks=missed_ranks,
                false_selected_mask=~hit, missed_mask=~found)


def regret_decomposition(r_full, r_p100, r_model, model_support):
    """The P100 inequality is deliberately not imposed on Full selections."""
    assert model_support in ('P100', 'FULL')
    a, b, c = [np.asarray(x, dtype=np.float64) for x in (r_full, r_p100, r_model)]
    assert a.shape == b.shape == c.shape
    assert np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(c).all()
    assert np.all(a <= b + 2e-6) and np.all(a <= c + 2e-6)
    if model_support == 'FULL':
        return dict(full_model_regret=c-a, decomposition_status='NOT_APPLICABLE_FULL_SELECTION')
    assert np.all(b <= c + 2e-6)
    pool, ranking, total = b-a, c-b, c-a
    assert np.allclose(total, pool+ranking, atol=1e-12, rtol=1e-12)
    denom = total.sum()
    return dict(pool_regret=pool, ranking_regret=ranking, total_regret=total,
                pool_share=None if abs(denom) <= 1e-12 else float(pool.sum()/denom),
                ranking_share=None if abs(denom) <= 1e-12 else float(ranking.sum()/denom))


def mean_mixture(score, mask, temperature=.1):
    assert score.ndim == 3 and mask.shape == (score.shape[0], score.shape[2])
    assert temperature > 0 and bool(mask.any(-1).all())
    probabilities = torch.softmax((score/temperature).masked_fill(~mask[:, None, :],
                                                                 -float('inf')), -1)
    return probabilities.mean(1).masked_fill(~mask, 0)


class GlobalHeadNormalizer:
    """Two-pass streamed global softmax; every eligible candidate is visited.

    First pass update(score, mask) over chunks. Second pass probability(score,
    mask) using the *global* normalizer. Float64 reduction avoids overflow and
    reduces summation drift. Exact algorithm; bitwise canonical equivalence is
    a separate real-checkpoint gate, not guaranteed by this mathematical unit.
    """
    def __init__(self, temperature=.1):
        assert temperature > 0
        self.temperature = temperature
        self.log_z = None
        self.valid_count = None

    def update(self, score, mask):
        logits = (score.double()/self.temperature).masked_fill(~mask[:, None, :],
                                                              -float('inf'))
        part = torch.logsumexp(logits, -1)
        self.log_z = part if self.log_z is None else torch.logaddexp(self.log_z, part)
        n = mask.sum(-1)
        self.valid_count = n if self.valid_count is None else self.valid_count+n

    def probability(self, score, mask):
        assert self.log_z is not None and bool((self.valid_count > 0).all())
        logits = score.double()/self.temperature
        p = torch.exp(logits-self.log_z[..., None]).masked_fill(~mask[:, None, :], 0)
        return p.mean(1)


class StreamingTopK:
    """Retain exact K smallest distances; ties use ascending canonical bank ID."""
    def __init__(self, k=10):
        self.k = k
        self.distance = self.ids = None

    def update(self, distance, ids, mask):
        assert distance.shape == ids.shape == mask.shape
        d = distance.masked_fill(~mask, float('inf'))
        if self.distance is not None:
            d = torch.cat([self.distance, d], -1)
            ids = torch.cat([self.ids, ids], -1)
        # Secondary-key ordering must precede stable primary-key sorting.
        by_id = torch.sort(ids, stable=True).indices
        ids, d = ids.gather(1, by_id), d.gather(1, by_id)
        take = torch.sort(d, stable=True).indices[:, :self.k]
        self.ids, self.distance = ids.gather(1, take), d.gather(1, take)

    def finalize(self):
        assert self.distance.shape[-1] == self.k
        assert bool(torch.isfinite(self.distance).all())
        assert bool((self.ids.sort(-1).values[:, 1:] != self.ids.sort(-1).values[:, :-1]).all())
        return self.distance, self.ids


def paa16(x):
    assert x.shape[-1] % 16 == 0
    return x.reshape(*x.shape[:-1], 16, x.shape[-1]//16).mean(-1)


def past_scores(query_history, candidate_history):
    """No future arguments accepted; tensors [B,L], [B,P,L]."""
    q, c = query_history, candidate_history
    assert q.ndim == 2 and c.ndim == 3 and c.shape[0] == q.shape[0]
    assert q.shape[-1] == c.shape[-1]
    def cosine(a, b):
        return (F.normalize(a, dim=-1)[:, None, :]*F.normalize(b, dim=-1)).sum(-1)
    qd, cd = q-q[:, -1:], c-c[..., -1:]
    return {'Raw Cosine':cosine(q, c), 'Delta-last Cosine':cosine(qd, cd),
            'Pearson':cosine(q-q.mean(-1, keepdim=True), c-c.mean(-1, keepdim=True)),
            'PAA-16 Cosine':cosine(paa16(qd), paa16(cd)),
            'Diff-1 Cosine':cosine(q[:, 1:]-q[:, :-1], c[..., 1:]-c[..., :-1])}


def fixed_random_scores(query_ids, channel, support_size, seed=0):
    """Per-row stateless RNG; invariant to query batching and unrelated calls."""
    return np.stack([np.random.default_rng(np.random.SeedSequence([seed, int(q), int(channel)]))
                     .random(support_size) for q in query_ids])


def pair_mse_from_dispersion(values):
    """Mean unordered pair future MSE without an N x N future tensor."""
    n = values.shape[-2]
    if n < 2:
        return torch.full(values.shape[:-2], float('nan'), device=values.device)
    centered = values-values.mean(-2, keepdim=True)
    return 2*n/(n-1)*centered.square().mean(dim=(-2, -1))


def bootstrap_query_channel_mean(values, block_length, repetitions=2000, seed=0):
    """Paired difference CI: circular moving blocks; channels stay together."""
    x = np.asarray(values, dtype=np.float64)
    assert x.ndim == 2 and np.isfinite(x).all() and block_length > 0
    q, channels = x.shape
    block_length = min(block_length, q)
    per_time = x.mean(1)
    rng = np.random.default_rng(seed)
    nblocks = (q+block_length-1)//block_length
    means = np.empty(repetitions)
    for i in range(repetitions):
        starts = rng.integers(0, q, nblocks)
        ids = ((starts[:, None]+np.arange(block_length)) % q).ravel()[:q]
        means[i] = per_time[ids].mean()
    return dict(mean=float(x.mean()), ci95=np.quantile(means, [.025, .975]).tolist(),
                query_count=q, channels=channels, block_length=block_length,
                repetitions=repetitions, seed=seed,
                resampling='circular moving query-time blocks; all channels jointly',
                limitation='seed0 exploratory interval, not training-seed uncertainty')


class TrainOnlyRidgeProbe:
    """Small diagnostic probe; cannot fit or choose parameters on test data."""
    def __init__(self, alphas=(.1, 1., 10.)):
        self.alphas = tuple(alphas)
        self.states = []
        self.selected = None

    def fit(self, x, target, split, candidates_future_free):
        assert split == 'train' and candidates_future_free is True
        x, y = np.asarray(x, np.float64), np.asarray(target, np.float64)
        assert x.ndim == 2 and y.shape == (len(x),)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        self.mean = x.mean(0); self.scale = np.maximum(x.std(0), 1e-12)
        z = np.c_[(x-self.mean)/self.scale, np.ones(len(x))]
        penalty = np.eye(z.shape[1]); penalty[-1, -1] = 0
        self.states = [np.linalg.solve(z.T@z+a*penalty, z.T@y) for a in self.alphas]
        return self

    def choose(self, x, target, split):
        assert split == 'val' and self.states
        z = np.c_[(np.asarray(x)-self.mean)/self.scale, np.ones(len(x))]
        losses = [float(np.square(z@w-target).mean()) for w in self.states]
        self.selected = int(np.argmin(losses))
        return dict(alpha=self.alphas[self.selected], validation_mse=losses[self.selected],
                    selection='validation future-MSE regression loss; no test tuning')

    def predict(self, x):
        assert self.selected is not None
        z = np.c_[(np.asarray(x)-self.mean)/self.scale, np.ones(len(x))]
        return z@self.states[self.selected]
