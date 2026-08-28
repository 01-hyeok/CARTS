"""A learnable score for (query, candidate) pairs.

Stage-1 has always scored retrieval with a cosine similarity, and the diagnostics
say that space cannot express the ordering that matters. Over the whole bank the
student's cosine correlates with future-MSE at rho 0.61, but inside its own
Top-100 the correlation falls to 0.03, and the Top-10 candidates sit within 0.004
cosine of each other. Coarse retrieval works; fine ordering has no signal left.

A pair-conditioned score is the direct test of whether that is the score
function's limit or the data's. Both sides still come from the same shared
encoder -- only the comparison changes from a fixed dot product to a learned
function of the pair.
"""

import torch
import torch.nn as nn

FEATURES = ('pair2', 'pair4')


def pair_feature_multiplier(feature_type):
    """How many copies of the embedding dimension the scorer input holds."""
    if feature_type == 'pair2':
        return 2
    if feature_type == 'pair4':
        return 4
    raise ValueError(f'Unsupported pairwise feature: {feature_type}; expected one of {FEATURES}')


def build_pair_features(z_q, z_k, feature_type):
    """Assemble the scorer input without materialising a repeated query.

    z_q is [B, D] and z_k is [B, M, D]; the query is broadcast rather than
    repeated, so memory stays O(B*M*D) instead of paying twice for it.

    pair2 gives the scorer both representations and lets it learn the comparison.
    pair4 hands it the difference and its magnitude as well: the sign of
    `z_q - z_k` carries direction and `|z_q - z_k|` carries distance per
    dimension, neither of which a network can recover from the concatenation
    without spending capacity on it.
    """
    if z_q.dim() != 2:
        raise ValueError(f'z_q must be [B, D], got {tuple(z_q.shape)}')
    if z_k.dim() != 3 or z_k.size(0) != z_q.size(0) or z_k.size(-1) != z_q.size(-1):
        raise ValueError(
            f'z_k must be [B, M, D] matching z_q {tuple(z_q.shape)}, got {tuple(z_k.shape)}'
        )
    query = z_q.unsqueeze(1).expand_as(z_k)
    if feature_type == 'pair2':
        return torch.cat([query, z_k], dim=-1)
    if feature_type == 'pair4':
        difference = query - z_k
        return torch.cat([query, z_k, difference, difference.abs()], dim=-1)
    raise ValueError(f'Unsupported pairwise feature: {feature_type}; expected one of {FEATURES}')


class PairwiseScorer(nn.Module):
    """[B, M, k*D] -> [B, M]. One scalar per (query, candidate) pair.

    Deliberately small. The question is whether a learned comparison beats a
    fixed one at all, and a scorer large enough to memorise the training pairs
    would answer a different question.
    """

    def __init__(self, embedding_dim, feature_type='pair4', hidden_dim=256,
                 hidden_dim2=128, dropout=0.1):
        super().__init__()
        if feature_type not in FEATURES:
            raise ValueError(
                f'Unsupported pairwise feature: {feature_type}; expected one of {FEATURES}')
        self.feature_type = feature_type
        self.embedding_dim = int(embedding_dim)
        self.input_dim = pair_feature_multiplier(feature_type) * self.embedding_dim
        self.net = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim2),
            nn.GELU(),
            nn.Linear(hidden_dim2, 1),
        )

    def forward(self, z_q, z_k):
        features = build_pair_features(z_q, z_k, self.feature_type)
        if features.size(-1) != self.input_dim:
            raise ValueError(
                f'pair features are {features.size(-1)}-wide, scorer expects {self.input_dim}'
            )
        return self.net(features).squeeze(-1)

    @torch.no_grad()
    def score_bank_in_chunks(self, z_q, bank, chunk_size=1024):
        """Score every candidate in a bank without holding all pairs at once.

        Evaluation runs over the full memory (thousands of candidates per query),
        and a pair feature is k*D wide, so the full matrix would be orders of
        magnitude larger than the cosine one it replaces. The bank is scored a
        slice at a time and only the scalars are kept.
        """
        if bank.dim() != 2:
            raise ValueError(f'bank must be [N, D], got {tuple(bank.shape)}')
        scores = []
        for start in range(0, bank.size(0), int(chunk_size)):
            block = bank[start:start + int(chunk_size)].to(z_q.device, z_q.dtype)
            scores.append(self(z_q, block.unsqueeze(0).expand(z_q.size(0), -1, -1)))
        return torch.cat(scores, dim=-1)
