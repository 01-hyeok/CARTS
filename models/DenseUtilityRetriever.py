"""EXP-MARGUTIL01: Full-Memory Set-Conditioned Dense Marginal Utility.

Reuses `SetConditioner` / `EmptySetToken` from `models/SequentialSetRetriever.py`
verbatim (per the experiment spec's explicit prohibition on new architecture).
The only new piece is `UtilityHead`: a minimal learnable affine rescaling of
the existing cosine score, `u_hat = a * cosine(h_t, e_i) + b`, so the model
outputs a real-valued utility estimate rather than a softmax-normalised
logit. No attention, no pair-MLP, no asymmetric scorer.

EXP-ASYM-SCORER01 adds `AsymmetricUtilityHead`: the same affine head, but
`cosine(h_t, e_i)` is replaced by `cosine(W_q h_t, W_k e_i)`. Rather than
reimplementing query/key projections, this wraps
`layers.retrieval_metric.RetrievalMetric(kind='asymmetric', output='cosine',
layer_norm=False)` -- the project's existing, already-reviewed asymmetric
scorer (identity-initialised, L2-normalised after projection, exactly
`cos` at init) -- so the only new code here is the affine `a*x+b` on top,
matching `UtilityHead`'s own convention exactly.
"""
import torch
import torch.nn as nn

from layers.retrieval_metric import RetrievalMetric


class UtilityHead(nn.Module):
    """u_hat = a * cosine(h_t, e_i) + b, a/b scalar and learnable."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, h_t, candidate_embeddings):
        """h_t: [B, D] (L2-normalised). candidate_embeddings: [N, D] or a
        [C, D] chunk of it (L2-normalised). Returns [B, N] or [B, C]."""
        cosine = torch.matmul(h_t, candidate_embeddings.transpose(0, 1))
        return self.scale * cosine + self.bias


class AsymmetricUtilityHead(nn.Module):
    """u_hat = a * cosine(W_q h_t, W_k e_i) + b.

    W_q/W_k are identity-initialised (via `RetrievalMetric`), so at
    construction this is numerically identical to `UtilityHead` -- verified
    by `layers.retrieval_metric.cosine_init_deviation` before any training
    starts (see `scripts/train_asym_scorer01.py`'s startup check)."""

    def __init__(self, dim):
        super().__init__()
        self.metric = RetrievalMetric(kind='asymmetric', dim=dim, layer_norm=False, output='cosine')
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, h_t, candidate_embeddings):
        cosine = self.metric.score(h_t, candidate_embeddings)
        return self.scale * cosine + self.bias


class StrongResidualPairScorer(nn.Module):
    """EXP-STRONG-SCORER-DIAG01: s_i = a*cos(h_t,e_i) + b + Delta_phi(h_t,e_i).

    `Delta_phi` is a pairwise nonlinear MLP over `[h, e, h*e, |h-e|]`
    (4D input), zero-initialised at its final layer so this is numerically
    identical to `UtilityHead` at construction -- verified by
    `scripts/train_toptail_rank01.py`'s startup check
    (`max_abs_score_deviation < 1e-6`). This is a diagnostic ceiling probe,
    not a proposed production scorer: does substantially more pairwise
    capacity, ADDED ON TOP OF the already-best R2 cosine solution rather
    than trained from scratch, let held-out top-tail utility be learned at
    all? `candidate_embeddings` is always the FULL memory bank (or a chunk
    of it the caller passes in for its own reasons, e.g.
    `utils.dense_utility`'s chunking) -- this module chunks internally over
    the candidate dimension purely for GPU memory (`chunk_size`), never to
    reduce the candidate universe; every valid candidate is still scored.
    """

    def __init__(self, dim, chunk_size=1024):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.bias = nn.Parameter(torch.tensor(0.0))
        d4 = 4 * dim
        self.mlp = nn.Sequential(
            nn.Linear(d4, d4), nn.GELU(),
            nn.Linear(d4, dim), nn.GELU(),
            nn.Linear(dim, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.chunk_size = int(chunk_size)

    def _delta(self, h_t, candidate_embeddings):
        bsz, dim = h_t.shape
        n = candidate_embeddings.size(0)
        cs = self.chunk_size
        out = h_t.new_empty(bsz, n)
        for start in range(0, n, cs):
            end = min(start + cs, n)
            e_chunk = candidate_embeddings[start:end]
            c = e_chunk.size(0)
            h_exp = h_t.unsqueeze(1).expand(bsz, c, dim)
            e_exp = e_chunk.unsqueeze(0).expand(bsz, c, dim)
            feat = torch.cat([h_exp, e_exp, h_exp * e_exp, (h_exp - e_exp).abs()], dim=-1)
            out[:, start:end] = self.mlp(feat).squeeze(-1)
        return out

    def forward(self, h_t, candidate_embeddings):
        base = self.scale * torch.matmul(h_t, candidate_embeddings.transpose(0, 1)) + self.bias
        return base + self._delta(h_t, candidate_embeddings)

    def forward_batched(self, h_t, candidate_embeddings_batched):
        """Per-row candidate set (e.g. a small gathered positive/hard-negative
        pool that differs per query), as opposed to `forward`'s single
        candidate bank shared across the whole batch. `h_t`: [B,D].
        `candidate_embeddings_batched`: [B,M,D]. Returns [B,M]. Used only for
        the small pairwise-loss forward in the memory-safe streaming training
        path (`scripts/train_toptail_rank01.py`'s strong_pair step) -- the
        base+delta formula is identical to `forward`, just per-row candidates
        instead of a shared bank, so no shortcut/approximation is introduced.
        """
        base = self.scale * (h_t.unsqueeze(1) * candidate_embeddings_batched).sum(-1) + self.bias
        h_exp = h_t.unsqueeze(1).expand_as(candidate_embeddings_batched)
        e = candidate_embeddings_batched
        feat = torch.cat([h_exp, e, h_exp * e, (h_exp - e).abs()], dim=-1)
        delta = self.mlp(feat).squeeze(-1)
        return base + delta
