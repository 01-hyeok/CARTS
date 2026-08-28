"""Learnable retrieval scores that stay a matrix multiplication.

Stage-1 has always compared query and candidate with a fixed cosine. The
diagnostics say that geometry cannot resolve the region that matters: over the
whole bank the student's cosine tracks future-MSE at rho 0.61, but inside its own
Top-100 the correlation is 0.03 and the top candidates sit within 0.004 of each
other.

A pair-conditioned MLP tests the same idea and cannot be indexed -- every query
would need a forward pass per candidate, so training had to fall back to a mined
subset while evaluation ranked the full memory. That mismatch is the likeliest
reason it lost at short horizons. Everything here keeps the score bilinear in the
two embeddings, so a whole bank scores as `q @ K.T` and training can use the same
candidate support as evaluation:

    cosine       <z_q, z_i>                 on unit vectors, nothing learned
    mahalanobis  (L z_q)^T (L z_i)          one shared space, W = L^T L symmetric
    asymmetric   (W_q z_q)^T (W_k z_i)      separate query and key spaces
    bilinear     z_q^T W z_i                unconstrained W, key untouched

The three learned kinds are NOT one ablation ladder. `bilinear` and `asymmetric`
span the same set of functions -- W_q^T W_k ranges over every D x D matrix, so
`asymmetric` is `bilinear` written with redundant parameters, and comparing them
measures optimisation under over-parameterisation, not expressive power. The
ladder that does nest by expressiveness is

    cosine  (W = I)  <  mahalanobis  (W symmetric PSD)  <  asymmetric  (W free)

which is why the experiment arms are cosine / mahalanobis / asymmetric.
`bilinear` is kept because earlier runs used it.

`output='cosine'` renormalises after projecting. That is what makes the identity
initialisation reproduce the incumbent *exactly* -- under `output='dot'` the
score is an unnormalised dot product scaled by 1/sqrt(D), which ranks differently
from cosine at the very first step and puts each kind on a different numeric
scale, so a shared temperature means a different sharpness per arm. Normalising
costs one degree of freedom (the global magnitude) out of D^2 and leaves the
per-direction reweighting that is the whole point, so the experiment uses it and
the arms then differ only in what they are allowed to learn.
"""

import torch
import torch.nn as nn

METRICS = ('cosine', 'mahalanobis', 'asymmetric', 'bilinear')

# Nested by expressiveness: cosine (W = I) < mahalanobis (W = L^T L, symmetric
# PSD) < asymmetric (W free). `bilinear` spans the same functions as
# `asymmetric`, so it is excluded from the ladder.
EXPRESSIVENESS_LADDER = ('cosine', 'mahalanobis', 'asymmetric')


class RetrievalMetric(nn.Module):
    """Score a query batch against a candidate bank. [B, D] x [N, D] -> [B, N].

    `output='cosine'` L2-normalises both sides after projection. Every kind then
    lands in [-1, 1], so one temperature means the same sharpness for all of them
    and no kind can raise a candidate's score by growing its norm instead of
    turning its direction. At identity initialisation the score is then exactly
    the incumbent cosine.

    `output='dot'` leaves the projections unnormalised; `scaled_dot` divides by
    sqrt(D) so the magnitudes stay in a range a cosine-tuned temperature can
    handle. It does not equal cosine at initialisation and it lets the score
    scale drift per kind -- kept for the earlier runs that used it, not for
    comparing kinds against each other.
    """

    def __init__(self, kind='cosine', dim=128, scaled_dot=True, layer_norm=True,
                 output='dot'):
        super().__init__()
        if kind not in METRICS:
            raise ValueError(f'Unsupported retrieval metric: {kind}; expected one of {METRICS}')
        if output not in ('dot', 'cosine'):
            raise ValueError(f'Unsupported metric output: {output}')
        self.kind = kind
        self.dim = int(dim)
        self.output = output
        self.scale = (1.0 / (self.dim ** 0.5)) if scaled_dot else 1.0

        # `shared` means one projection L is applied to both sides, giving a
        # symmetric PSD W = L^T L -- a Mahalanobis metric. Query and candidate
        # keep the same geometry, so the score stays symmetric in its arguments.
        self.shared = (kind == 'mahalanobis')
        if kind in ('bilinear', 'mahalanobis'):
            # One matrix. bilinear folds it into the query side only (the key
            # passes through); mahalanobis applies the same one to both sides.
            self.query_projection = nn.Linear(self.dim, self.dim, bias=False)
            self.key_projection = None
            self.query_projection.weight.data.copy_(torch.eye(self.dim))
        elif kind == 'asymmetric':
            self.query_projection = nn.Linear(self.dim, self.dim, bias=False)
            self.key_projection = nn.Linear(self.dim, self.dim, bias=False)
            self.query_projection.weight.data.copy_(torch.eye(self.dim))
            self.key_projection.weight.data.copy_(torch.eye(self.dim))
        else:
            self.query_projection = None
            self.key_projection = None
        self.norm = (
            nn.LayerNorm(self.dim) if (layer_norm and kind != 'cosine') else None
        )

    def project_query(self, z_q):
        if self.query_projection is None:
            return z_q
        out = self.query_projection(z_q)
        return self.norm(out) if self.norm is not None else out

    def project_key(self, z_k):
        """Applied to candidates.

        Mahalanobis reuses the query projection, so both sides land in the same
        learned space. Bilinear folds its single W into the query side instead
        and the key passes through unchanged. Cosine has no projection at all.
        """
        projection = (
            self.query_projection if self.shared else self.key_projection
        )
        if projection is None:
            return z_k
        out = projection(z_k)
        return self.norm(out) if self.norm is not None else out

    def score(self, z_q, z_k):
        """[B, D] against [N, D] -> [B, N], or [B, M, D] -> [B, M]."""
        q = self.project_query(z_q)
        k = self.project_key(z_k)
        if self.output == 'cosine':
            q = torch.nn.functional.normalize(q, dim=-1)
            k = torch.nn.functional.normalize(k, dim=-1)
            scale = 1.0
        else:
            scale = self.scale if self.kind != 'cosine' else 1.0
        if k.dim() == 2:
            return torch.matmul(q, k.transpose(0, 1)) * scale
        if k.dim() == 3:
            if k.size(0) != q.size(0):
                raise ValueError(
                    f'candidate batch {k.size(0)} does not match query batch {q.size(0)}')
            return (q.unsqueeze(1) * k).sum(-1) * scale
        raise ValueError(f'candidates must be [N, D] or [B, M, D], got {tuple(k.shape)}')

    def forward(self, z_q, z_k):
        return self.score(z_q, z_k)


@torch.no_grad()
def cosine_init_deviation(metric, samples=64, generator=None):
    """Max |metric.score - cosine| at the current parameters, on random inputs.

    The arms are only comparable if the learned kinds *start* where the fixed
    baseline is; otherwise a difference at epoch 1 is the initialisation talking,
    not the learning. This measures that claim instead of asserting it in a
    comment. It is ~0 for a freshly built kind under `output='cosine'` without
    LayerNorm, and visibly non-zero under `output='dot'` or with an affine
    LayerNorm in the path -- both of which change the ranking at step 0.
    """
    device = next(metric.parameters(), torch.zeros(1)).device
    z_q = torch.randn(samples, metric.dim, device=device, generator=generator)
    z_k = torch.randn(samples, metric.dim, device=device, generator=generator)
    reference = torch.matmul(
        torch.nn.functional.normalize(z_q, dim=-1),
        torch.nn.functional.normalize(z_k, dim=-1).transpose(0, 1),
    )
    was_training = metric.training
    metric.eval()
    try:
        deviation = (metric.score(z_q, z_k) - reference).abs().max()
    finally:
        metric.train(was_training)
    return float(deviation)


@torch.no_grad()
def oracle_rank_statistics(scores, oracle_indices, valid_mask, oracle_valid=None):
    """Where the Oracle Top-K sit in the model's own full-memory ranking.

    Recall@10 only counts an Oracle candidate once it reaches the top ten, so it
    stays near zero while the model moves a candidate from rank 4000 to rank 40 --
    a large improvement it cannot see. Mean rank shows that movement directly.
    Rank 1 is the model's highest-scored candidate.
    """
    floor = torch.finfo(scores.dtype).min
    masked = scores.detach().float().masked_fill(~valid_mask, floor)
    order = masked.argsort(dim=-1, descending=True)
    rank = torch.empty_like(order)
    positions = torch.arange(
        1, masked.size(-1) + 1, device=scores.device).expand_as(order)
    rank.scatter_(1, order, positions)

    oracle_rank = rank.gather(1, oracle_indices).float()
    if oracle_valid is not None:
        keep = oracle_valid.bool()
        if not keep.any():
            zero = scores.sum() * 0.0
            return {'oracle_top10_mean_rank': zero, 'oracle_top10_median_rank': zero}
        selected = oracle_rank[keep]
    else:
        selected = oracle_rank.reshape(-1)
    valid_count = valid_mask.sum(-1).float().mean()
    return {
        'oracle_top10_mean_rank': selected.mean(),
        'oracle_top10_median_rank': selected.median(),
        'oracle_top10_rank_fraction': selected.mean() / valid_count.clamp_min(1.0),
        'valid_candidate_pool': valid_count,
    }


@torch.no_grad()
def score_separation_metrics(scores, oracle_indices, valid_mask, top_k=10):
    """Does the model score the Oracle set above everything else?"""
    floor = torch.finfo(scores.dtype).min
    scores = scores.detach().float()
    masked = scores.masked_fill(~valid_mask, floor)
    model_top = masked.topk(min(top_k, masked.size(-1)), dim=-1).indices

    oracle_hit = torch.zeros_like(valid_mask)
    oracle_hit.scatter_(1, oracle_indices, True)
    non_oracle = valid_mask & ~oracle_hit

    def masked_mean(mask):
        total = mask.float().sum().clamp_min(1.0)
        return (scores * mask.float()).sum() / total

    oracle_mean = masked_mean(oracle_hit & valid_mask)
    non_oracle_mean = masked_mean(non_oracle)
    return {
        'oracle_top10_score_mean': oracle_mean,
        'model_top10_score_mean': scores.gather(1, model_top).mean(),
        'non_oracle_score_mean': non_oracle_mean,
        'oracle_vs_nonoracle_score_gap': oracle_mean - non_oracle_mean,
    }
