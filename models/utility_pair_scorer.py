"""Pair scorer for utility-aware candidate selection.

Lifted from scripts/train_utility_ranker.py so the cross-channel arms score
pairs exactly the way the target-only arms did. The only thing the cross-channel
version changes is what `z_q` is -- a contextual embedding instead of a
target-only one -- which is the whole point of the comparison.

    s(q,k) = MLP([z_q, z_k, |z_q - z_k|, z_q * z_k])            arms C / D / E
           = MLP([z_q, z_k, f(R_k), |z_q - z_k|, z_q * z_k])    arm F

R_k is the candidate's historical residual: memory-side, observable at
inference. The query residual never enters the model; it only builds the
utility target.
"""

import torch
import torch.nn as nn


class UtilityPairScorer(nn.Module):
    def __init__(self, dim, horizon=0, residual_dim=64, hidden=256, dropout=0.1):
        super().__init__()
        self.residual_proj = (
            nn.Sequential(nn.Linear(horizon, residual_dim), nn.GELU())
            if horizon else None
        )
        width = 4 * dim + (residual_dim if horizon else 0)
        self.net = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, z_q, z_k, residual=None):
        """z_q [B, 1, d] or [B, M, d], z_k [B, M, d] -> scores [B, M]."""
        z_q = z_q.expand_as(z_k)
        parts = [z_q, z_k, (z_q - z_k).abs(), z_q * z_k]
        if self.residual_proj is not None:
            if residual is None:
                raise ValueError('scorer was built with a residual branch')
            parts.append(self.residual_proj(residual))
        return self.net(torch.cat(parts, dim=-1)).squeeze(-1)
