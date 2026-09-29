"""TRACK-I-PCA-FUTURE-TEACHER01 -- shared PCA-teacher utilities.

Fits a FIXED (frozen) per-channel PCA basis on TRAIN-only future vectors
(the same reconstructed value-space `memory_value()` already produces for
the raw-MSE teacher -- e.g. delta_last -- never reimplemented here), then
exposes a chunked, full-memory distance function with the exact same
[B, N] signature/semantics as `individual_utility_memsafe` from
`train_factorial_e2e01`, so it can be substituted for the raw-MSE `d`
computation with no other code path changed.

No leakage: `fit_pca` must only ever be called with a train-only future
matrix (the candidate memory bank IS train-only by construction throughout
this codebase -- see `exp.memory_y`/`RelationMemorySampler` -- so calling
`fit_pca(memory_c)` on the existing candidate bank is automatically
train-only; this module performs no additional splitting itself and does
not touch val/test data).
"""
import torch
import torch.nn.functional as F


def fit_pca(y_train, max_dim=None, eps_ratio=1e-8):
    """y_train: [N, H] train-only future matrix (one channel).
    Returns (mean [H], components [r, H], explained_rank r).
    r = numerical rank of the centered matrix, clamped to `max_dim` if given
    (PCA dimension can never exceed the actual data rank, per spec)."""
    mean = y_train.mean(dim=0)
    yc = y_train - mean
    # full_matrices=False keeps Vh at [min(N,H), H]
    _, s, vh = torch.linalg.svd(yc, full_matrices=False)
    rank = int((s > s.max().clamp_min(1e-30) * eps_ratio).sum().item())
    rank = max(rank, 1)
    if max_dim is not None:
        rank = min(rank, max_dim)
    components = vh[:rank].contiguous()  # [rank, H]
    return mean, components


def pca_transform(y, mean, components):
    """y: [..., H] -> [..., rank]. Deterministic linear map, no learnable
    state; identical output for identical input every call (see
    test_pca_transform_is_deterministic)."""
    return (y - mean) @ components.T


def pca_distance_memsafe(memory_c, offset_c, query_future, mean, components,
                         metric='l2', chunk_size=None):
    """Chunked full-memory PCA-space distance, mirroring
    `individual_utility_memsafe`'s [B, N] chunking convention exactly
    (same `memory_c`/`offset_c` candidate-future reconstruction, same
    return-shape contract) so it is a drop-in replacement for `-u`.

    metric='l2'  -> mean squared PCA-space distance (scale-consistent with
                    MSE: divided by the PCA dimension, matching how the raw
                    teacher's MSE divides by the full horizon H).
    metric='cosine' -> 1 - cosine_similarity(PCA(q), PCA(k)).

    Returns d: [B, N] (lower = more similar, i.e. same "distance"
    convention as `d = -individual_utility_memsafe(...)` in every other
    script in this repo).
    """
    bsz = offset_c.size(0)
    n, h = memory_c.shape
    chunk_size = chunk_size or n
    q_proj = pca_transform(query_future, mean, components)  # [B, r]
    if metric == 'cosine':
        q_proj = F.normalize(q_proj, dim=-1)
    out = offset_c.new_empty(bsz, n)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        y_c = memory_c[start:end].unsqueeze(0) + offset_c.view(-1, 1, 1)  # [B, chunk, H]
        k_proj = pca_transform(y_c, mean, components)  # [B, chunk, r]
        if metric == 'l2':
            out[:, start:end] = ((k_proj - q_proj.unsqueeze(1)) ** 2).mean(-1)
        elif metric == 'cosine':
            k_proj = F.normalize(k_proj, dim=-1)
            out[:, start:end] = 1.0 - (k_proj * q_proj.unsqueeze(1)).sum(-1)
        else:
            raise ValueError(metric)
    return out
