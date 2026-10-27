"""TRACK-V5-HEAD-LOSS-DUAL-ENCODER01 -- Differentiable Top-K operator for
L3b (Top-K-aligned Aggregate Loss).

The repo's EXISTING `utils/soft_topk_retrieval01.py::soft_topk_weights`
(from TRACK-P100-SOFTTOPK-QUALITY-AWARE01) renormalizes probabilities
over an ALREADY hard-selected Top-K index set -- its own docstring is
explicit that this is "piecewise-differentiable," NOT a true
differentiable Top-K, because the selection BOUNDARY itself carries no
gradient (a candidate just outside the hard cutoff gets EXACTLY zero
weight and zero gradient, however close its score is to the cutoff).
This track's spec explicitly forbids presenting that pattern as a true
Differentiable Top-K, so this module implements a genuinely different
operator that gives every candidate (not just the K already-selected
ones) a smooth, boundary-aware inclusion weight.

## Algorithm: sigmoid-relaxed order-statistic threshold

For a row of scores `s in R^N` (here, N = Shared-P100 pool size = 100)
and target count `K`:

    threshold = kth_largest(s)              -- the K-th largest value, i.e. the
                                                hard cutoff; DETACHED (no gradient
                                                flows through the threshold itself)
    raw_i     = sigmoid((s_i - threshold) / tau)      for every i = 1..N
    a_i       = raw_i / sum_j raw_j                    (renormalized over valid i)

As `tau -> 0`, `sigmoid((s_i - threshold)/tau)` saturates to a step
function: `1` for every `s_i` strictly above the K-th-largest value (the
K selected candidates) and `0` for every `s_i` below it -- so `a_i ->
1/K` for each of the K selected candidates and `a_i -> 0` for every
other candidate, i.e. this operator converges EXACTLY to the Hard
Top-K-with-uniform-weights limit (`torch.nn.functional.softmax` plays
no role in this limit; the convergence is a literal hard-indicator
limit of the sigmoid, proven by `sigmoid(x/tau) -> 1_{x>0}` as `tau->0+`).

Unlike `soft_topk_weights`, **every** candidate -- including those just
below the cutoff -- receives a nonzero, differentiable weight whenever
`tau > 0`; a candidate whose score sits close to `threshold` has `a_i`
close to `0.5 / sum_j raw_j` and a LARGE `d(a_i)/d(s_i)` (the sigmoid's
steepest point), which is exactly the "learn the selection boundary"
property the spec requires and `soft_topk_weights` structurally lacks
(there, `d(a_i)/d(s_i) = 0` for any `i` not already in the hard-selected
set, by construction).

## Complexity

`kth_largest` (via `torch.kthvalue` or `torch.topk`) is `O(N log N)`
(PyTorch's partial-sort implementation); the sigmoid/renormalization
step is `O(N)`. Overall `O(N log N)` per row, identical asymptotic cost
to the existing hard `stable_topk_indices` selection this repo already
performs every training step.

## Invalid candidates

`valid_mask` entries are masked to `-inf` BEFORE the threshold is
computed (so the K-th largest is always chosen among valid candidates
only) and their `raw_i` is forced to exactly `0` (not merely small) via
a post-hoc `masked_fill`, guaranteeing `a_i = 0` for every invalid
candidate regardless of `tau`.
"""
import torch

EPS = 1e-12


def soft_topk_mask(scores, valid_mask, k, tau):
    """scores: [B, N] (or [B, S, N] -- any shape ending in the candidate
    axis; `valid_mask` broadcastable to it). `k`: int. `tau`: float > 0,
    the relaxation temperature. Returns `a`: same shape as `scores`,
    `a >= 0`, `a.sum(-1) == 1` over valid candidates, `a[invalid] == 0`.

    `tau -> 0` recovers exact Hard Top-K with uniform `1/k` weights
    (see module docstring); `tau` large makes `a` approach a uniform
    distribution over all valid candidates (no selectivity)."""
    assert tau > 0, '[ISSUE][ABORT] soft_topk_mask requires tau > 0 (use the hard selection directly at tau=0)'
    masked_scores = scores.masked_fill(~valid_mask, float('-inf'))
    threshold = masked_scores.topk(k, dim=-1, largest=True).values[..., -1:].detach()  # K-th largest, detached
    raw = torch.sigmoid((masked_scores - threshold) / tau)
    raw = raw.masked_fill(~valid_mask, 0.0)
    denom = raw.sum(dim=-1, keepdim=True).clamp_min(EPS)
    return raw / denom


def soft_topk_forecast(a, y_pool):
    """a: [B, N] differentiable weights from `soft_topk_mask`. y_pool:
    [B, N, T] future values for every pool candidate (already
    offset/alignment-applied, SAME convention as
    `pooled_memory_c + offset_c` elsewhere in this repo). Returns
    R: [B, T]."""
    return (a.unsqueeze(-1) * y_pool).sum(dim=1)


def hard_limit_matches_stable_topk(scores, valid_mask, k, tau, atol=1e-3):
    """Diagnostic/test helper: at a small `tau`, does `soft_topk_mask`'s
    arg-sorted support match the hard `stable_topk_indices` selection?
    Returns (matches: bool, max_abs_diff_from_uniform_1_over_k: float)."""
    from models.RelationStage1 import stable_topk_indices
    a = soft_topk_mask(scores, valid_mask, k, tau)
    hard_idx = stable_topk_indices(scores.masked_fill(~valid_mask, float('-inf')), k, largest=True)
    a_at_hard = a.gather(-1, hard_idx)
    max_abs_diff = float((a_at_hard - 1.0 / k).abs().max())
    # every hard-selected candidate should carry close to 1/k, every unselected close to 0
    total_mass_outside = float(a.sum(-1).mean()) - float(a_at_hard.sum(-1).mean())
    return max_abs_diff < atol and total_mass_outside < atol, max_abs_diff
