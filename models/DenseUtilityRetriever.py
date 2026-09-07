"""EXP-MARGUTIL01: Full-Memory Set-Conditioned Dense Marginal Utility.

Reuses `SetConditioner` / `EmptySetToken` from `models/SequentialSetRetriever.py`
verbatim (per the experiment spec's explicit prohibition on new architecture).
The only new piece is `UtilityHead`: a minimal learnable affine rescaling of
the existing cosine score, `u_hat = a * cosine(h_t, e_i) + b`, so the model
outputs a real-valued utility estimate rather than a softmax-normalised
logit. No attention, no pair-MLP, no asymmetric scorer.
"""
import torch
import torch.nn as nn


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
