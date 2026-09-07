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
