"""Common candidate-pool abstraction for future retrieval trainers
(Original KL / Multi-Query / Timestamp Router / ...), so each one gets
`--candidate_pool_mode full` (exact legacy behavior) or
`--candidate_pool_mode coarse_topk` (cheap, future-blind, parameter-free
delta-last-cosine pre-filter down to M candidates BEFORE the learned
encoder ever sees them) without reimplementing this math per-script.

Does NOT touch any historical script (`train_j_shared_encoder_drift01.py`,
`train_t_pure_multislot01.py`, `build_t_multislot_cache01.py`,
`build_t2_true_original_kl_cache01.py`) -- those keep their own inline
full-memory logic unchanged, for reproducibility. New trainers
(`scripts/train_retriever_pool01.py` etc.) import this module instead.

Design principle (the whole point of this module): the learned candidate
encoder must see exactly M candidates in `coarse_topk` mode, not N
filtered down to M after the fact -- the sequence is always

    full N candidates -> cheap fixed coarse score -> top-M GLOBAL indices
    -> gather the M candidate histories from memory_x -> THEN encode

never "encode all N, then mask to M".

Local-vs-global index discipline: every function here that returns
candidate positions inside the pool returns them as LOCAL indices in
[0, M); `local_to_global` is the one function that converts back to
memory-bank (global, [0, N)) indices, and every call site operates in
flat memory_x line 10 of their eventual bookkeeping to minimise the
blast radius of sudden mistaken reuse.
"""
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.RelationStage1 import stable_topk_indices

SUPPORTED_POOL_METRICS = ('delta_last_cosine',)
SUPPORTED_POOL_MODES = ('full', 'coarse_topk')


@dataclass(frozen=True)
class CandidatePoolConfig:
    mode: str = 'full'
    size: int = 100
    metric: str = 'delta_last_cosine'

    def __post_init__(self):
        if self.mode not in SUPPORTED_POOL_MODES:
            raise ValueError(f'Unsupported candidate_pool_mode: {self.mode!r}')
        if self.metric not in SUPPORTED_POOL_METRICS:
            raise ValueError(f'Unsupported candidate_pool_metric: {self.metric!r}')
        if self.mode == 'coarse_topk' and self.size < 1:
            raise ValueError(f'candidate_pool_size must be >= 1, got {self.size}')


def _delta_last_1d_time(x):
    """x: [..., L] (a single channel's own scalar history, time axis
    last). Returns x - x[..., -1:]."""
    return x - x[..., -1:].detach()


def compute_coarse_delta_last_cosine_scores(batch_x, memory_x, channel):
    """Future-blind, parameter-free coarse score for one channel.
    `batch_x`: [B, L, C] query histories. `memory_x`: [N, L, C] candidate
    histories (the full memory bank). Returns raw cosine scores [B, N].
    NEVER reads batch_y / memory_y -- there is no such argument here by
    construction, so a caller cannot accidentally leak the future into
    this function."""
    q = batch_x[:, :, channel]            # [B, L]
    k = memory_x[:, :, channel]            # [N, L]
    q_delta = _delta_last_1d_time(q)
    k_delta = _delta_last_1d_time(k)
    q_n = F.normalize(q_delta, dim=-1)
    k_n = F.normalize(k_delta, dim=-1)
    return torch.matmul(q_n, k_n.transpose(0, 1))  # [B, N]


def build_coarse_pool(coarse_scores, cand_mask, pool_size):
    """`coarse_scores`: [B, N] raw scores (higher = more similar).
    `cand_mask`: [B, N] bool, which candidates are even valid for this
    query (RAFT mask etc., channel-independent, computed by the caller
    exactly as every existing script already does via
    `exp._candidate_mask`). Returns `pool_idx_global`: [B, M] int64,
    LOCAL rank 0..M-1 in the pool, but GLOBAL (0..N-1) memory indices --
    i.e. this IS the local->global mapping for the pool itself (pool
    position i already denotes memory_x[pool_idx_global[:, i]]).

    Raises (does not silently truncate) if any query has fewer than
    `pool_size` valid candidates -- spec section 6's explicit
    requirement; the caller must run a preflight check or pick a smaller
    pool_size instead of relying on this to paper over the condition."""
    valid_counts = cand_mask.sum(dim=-1)
    min_valid = int(valid_counts.min().item())
    if min_valid < pool_size:
        raise ValueError(
            f'[ISSUE][ABORT] at least one query has only {min_valid} valid candidates, '
            f'below the requested candidate_pool_size={pool_size}. Candidate pools are '
            'never silently shrunk -- pick a smaller --candidate_pool_size or fix the '
            'candidate mask.')
    masked = coarse_scores.masked_fill(~cand_mask, float('-inf'))
    pool_idx_global = stable_topk_indices(masked, pool_size, largest=True)
    return pool_idx_global


def gather_candidate_histories(memory_x, pool_idx_global):
    """`memory_x`: [N, L, C]. `pool_idx_global`: [B, M]. Returns
    [B, M, L, C] -- every channel's history is carried along (the caller's
    encoder picks its own target channel internally, exactly as the full
    -memory path already does via `encode_raw(model, x, c)`)."""
    return memory_x[pool_idx_global]


def gather_candidate_values(memory_c, pool_idx_global):
    """`memory_c`: [N, H] (one channel's own future-value representation,
    from `train_margutil01.memory_value`). `pool_idx_global`: [B, M].
    Returns [B, M, H]."""
    return memory_c[pool_idx_global]


def local_to_global(pool_idx_global, local_idx):
    """`pool_idx_global`: [B, M]. `local_idx`: [B, K] (positions in
    [0, M), e.g. the output of a Top-K selection run ON the pool).
    Returns [B, K] GLOBAL (0..N-1) memory indices."""
    return pool_idx_global.gather(1, local_idx)


def pooled_future_mse(pooled_memory_c, offset_c, query_future):
    """`pooled_memory_c`: [B, M, H] (from `gather_candidate_values`).
    `offset_c`: [B]. `query_future`: [B, H]. Returns [B, M] -- the same
    `candidate_future = pooled_memory_c + offset_c` convention every
    existing script already uses, restricted to the M pooled candidates."""
    pooled_future = pooled_memory_c + offset_c.view(-1, 1, 1)
    return ((pooled_future - query_future.unsqueeze(1)) ** 2).mean(dim=-1)


def encode_pooled_candidates(encode_fn, pooled_x, channel):
    """`encode_fn`: a 2-arg-plus-channel callable with the SAME signature
    as `train_factorial_e2e01.encode_raw(model, x, c)` (or this track's
    own timestamp-aware analog) -- reused, never reimplemented.
    `pooled_x`: [B, M, L, C]. Flattens to [B*M, L, C], encodes, reshapes
    back to [B, M, D]. Gradient stays ON through this path (no
    `torch.no_grad()` here) -- only the pool SELECTION upstream
    (`build_coarse_pool`) is no-grad/discrete."""
    bsz, pool_m, seq_len, n_channels = pooled_x.shape
    flat = pooled_x.reshape(bsz * pool_m, seq_len, n_channels)
    z_flat = encode_fn(flat, channel)
    d_model = z_flat.size(-1)
    return z_flat.reshape(bsz, pool_m, d_model)
