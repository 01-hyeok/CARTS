"""Preparation-only CPU-tested Dynamics/Cross scorer primitives.

No dataset IO, implicit fit, future input, GPU launch, or experiment runner.
Candidate axes are last for scores and penultimate for pair features.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def fit_ar(past, ridge=1e-3, p=4, tail=16, eps=1e-8):
    """Fit per-window differenced AR. Input contains observed history only.

    Coefficients order: intercept, a1,...,ap. The intercept is unpenalized.
    Return coefficient, six-feature vector, recent mean difference, and
    tail sufficient statistics. Float64 solve limits ridge ill-conditioning.
    """
    if past.shape[-1] < p + tail + 1 or ridge <= 0:
        raise ValueError('Need p+tail+1 observed points and positive ridge')
    if not torch.isfinite(past).all():
        raise ValueError('Non-finite history')
    d = past.double().diff(dim=-1)
    windows = d.unfold(-1, p + 1, 1)
    X = torch.cat((torch.ones_like(windows[..., :1]), windows[..., :p].flip(-1)), -1)
    y = windows[..., p]
    reg = torch.eye(p + 1, device=X.device, dtype=X.dtype) * ridge
    reg[0, 0] = 0
    coef = torch.linalg.solve(X.transpose(-1, -2) @ X + reg,
                              X.transpose(-1, -2) @ y[..., None]).squeeze(-1)
    residual = (y - (X * coef[..., None, :]).sum(-1)).square().mean(-1)
    phi = torch.cat((coef[..., 1:], coef[..., :1], (residual + eps).log()[..., None]), -1)
    Xt, yt = X[..., -tail:, :], y[..., -tail:]
    G = Xt.transpose(-1, -2) @ Xt / tail
    b = (Xt.transpose(-1, -2) @ yt[..., None]).squeeze(-1) / tail
    u = yt.square().mean(-1)
    return dict(coef=coef, phi=phi, recent=d[..., -tail:].mean(-1),
                residual=residual, G=G, b=b, u=u)


def cross_error(query, candidate):
    """Pairwise quadratic one-step errors, with broadcastable leading dims."""
    a = candidate['coef']
    value = torch.einsum('...i,...ij,...j->...', a, query['G'], a) - 2 * (a * query['b']).sum(-1) + query['u']
    # Negative cancellation at floating precision is not negative MSE.
    if (value < -1e-8).any():
        raise ValueError('Cross quadratic error is materially negative')
    return value.clamp_min(0)


class FixedStandardizer(nn.Module):
    """One explicit train-only fit; fixed buffers survive state_dict save/load."""
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.register_buffer('mean', torch.zeros(dim, dtype=torch.float64))
        self.register_buffer('scale', torch.ones(dim, dtype=torch.float64))
        self.register_buffer('fitted', torch.tensor(False))
        self.eps = eps

    @torch.no_grad()
    def fit(self, features, split):
        if split != 'train' or bool(self.fitted):
            raise ValueError('Calibration can be fitted once, on train only')
        x = features.detach().double().reshape(-1, self.mean.numel())
        if not len(x) or not torch.isfinite(x).all():
            raise ValueError('Invalid calibration features')
        self.mean.copy_(x.mean(0))
        self.scale.copy_(x.var(0, unbiased=False).sqrt().clamp_min(self.eps))
        self.fitted.fill_(True)
        return self

    def forward(self, features):
        if not bool(self.fitted):
            raise RuntimeError('Train calibration is missing')
        return (features - self.mean) / self.scale


def dynamics_pairs(q, c, normalizer):
    qn, cn = normalizer(q['phi']), normalizer(c['phi'])
    diff = qn - cn
    return torch.cat((diff, qn * cn,
                      q['phi'][..., -1:] + torch.zeros_like(c['phi'][..., -1:]),
                      c['phi'][..., -1:] + torch.zeros_like(q['phi'][..., -1:]),
                      (q['recent'] - c['recent'])[..., None]), -1)


def cross_pairs(q, c, normalizer):
    err = cross_error(q, c)
    self_err = cross_error(q, q)
    return torch.cat((err[..., None], (self_err + torch.zeros_like(err))[..., None],
                      (err - self_err)[..., None], dynamics_pairs(q, c, normalizer)), -1)


class DynamicsL2(nn.Module):
    def __init__(self, dim=6, learned=True):
        super().__init__()
        initial = torch.full((dim,), math.log(math.expm1(1.)), dtype=torch.float64)
        if learned:
            self.raw_weight = nn.Parameter(initial)
        else:
            self.register_buffer('raw_weight', initial)

    def forward(self, normalized_query, normalized_candidate):
        return -(F.softplus(self.raw_weight) * (normalized_query-normalized_candidate).square()).sum(-1)


class PairMLP(nn.Module):
    def __init__(self, dim, width=32):
        super().__init__()
        self.standardizer = FixedStandardizer(dim)
        self.layers = nn.Sequential(nn.Linear(dim, width), nn.SiLU(), nn.Linear(width, 1)).double()

    def forward(self, features):
        return self.layers(self.standardizer(features)).squeeze(-1)


def masked_log_probability(logits, mask):
    if logits.shape != mask.shape or not mask.any(-1).all():
        raise ValueError('Invalid/empty candidate support')
    return logits.masked_fill(~mask, -torch.inf).log_softmax(-1)


def log_v5_score(probability, mask, tau_s=.1):
    """Accept exact official mixture, including its numerical normalization.

    Do NOT standardize this base score: doing so changes beta=0 distribution.
    An additive per-query constant is harmless; multiplicative scale is not.
    """
    if tau_s <= 0 or probability.shape != mask.shape or not mask.any(-1).all():
        raise ValueError('Invalid baseline shape/temperature/support')
    p = probability.double()
    if not torch.isfinite(p).all() or (p[mask] <= 0).any() or (p[~mask] != 0).any():
        raise ValueError('Official probabilities invalid; no epsilon clipping fallback')
    if not torch.allclose(p.sum(-1), torch.ones_like(p.sum(-1)), atol=2e-6, rtol=0):
        raise ValueError('Official mixture is not normalized')
    return tau_s * p.masked_fill(~mask, 1).log().masked_fill(~mask, -torch.inf)


class Fusion(nn.Module):
    def __init__(self, tau_s=.1):
        super().__init__()
        self.tau_s = tau_s
        self.beta = nn.Parameter(torch.tensor(0., dtype=torch.float64))
        self.calibration = FixedStandardizer(1)

    def score(self, official_probability, new_score, mask):
        correction = self.calibration(new_score[..., None]).squeeze(-1)
        return log_v5_score(official_probability, mask, self.tau_s) + self.beta * correction

    def log_probability(self, official_probability, new_score, mask):
        return masked_log_probability(self.score(official_probability, new_score, mask)/self.tau_s, mask)

    @torch.no_grad()
    def ranking_score(self, official_probability, new_score, mask):
        # Preserve canonical FP32 ties and ordered IDs exactly for beta=0.
        # Training never takes this branch: beta must receive a gradient at 0.
        if self.beta.item() == 0:
            return official_probability.masked_fill(~mask, -torch.inf)
        return self.score(official_probability, new_score, mask)


def kl_from_log_probability(teacher, log_student, mask):
    if not torch.isfinite(teacher).all() or (teacher < 0).any() or (teacher[~mask] != 0).any():
        raise ValueError('Invalid teacher')
    if not torch.allclose(teacher.sum(-1), torch.ones_like(teacher.sum(-1)), atol=2e-6, rtol=0):
        raise ValueError('Unnormalized teacher')
    t = teacher.detach()
    logt = t.clamp_min(torch.finfo(t.dtype).tiny).log()
    return (t * (logt - log_student.masked_fill(~mask, 0))).sum(-1).mean()


def assembled_scores(candidate_count, chunk, make_score):
    """Chunk feature/scorer work, then normalize ONCE over global support.

    This retains a full score/graph tensor; not a constant-memory Full trainer.
    GPU peak memory must be measured before large Full runs are authorized.
    """
    if candidate_count < 1 or chunk < 1:
        raise ValueError('Invalid chunks')
    return torch.cat([make_score(i, min(i+chunk, candidate_count))
                      for i in range(0, candidate_count, chunk)], -1)


def eligibility(query_starts, candidate_starts, seq_len, pred_len, policy, split):
    q, c = query_starts[..., None], candidate_starts
    end = c + seq_len + pred_len
    if policy == 'known_at_forecast_origin':
        return end <= q + seq_len
    if policy == 'strict_before_query_start':
        return end <= q
    if policy == 'legacy_raft':
        if split == 'train':
            return (end <= q) | (c >= q + seq_len + pred_len)
        if split in ('val', 'test'):
            return torch.ones_like(q + c, dtype=torch.bool)
    raise ValueError('Unknown eligibility policy/split')
