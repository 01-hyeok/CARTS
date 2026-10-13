"""TRACK-P100-SOFTTOPK-QUALITY-AWARE01 -- Soft Top-K weighted retrieval.

Replaces the Uniform-Top-K-future-mean aggregation
(`y_sel.mean(dim=1)` in `build_v_meanmix_cache_pool01.py` and
equivalents) with a weighted mean over the SAME Top-K index set,
weighted by each candidate's own (renormalized) selection probability:

    I_K = TopK_i(p(i|q,c))                      -- discrete, no gradient needed
    w_i = p_i / sum_{j in I_K} p_j,  i in I_K    -- differentiable w.r.t. p
    R(q,c) = sum_{i in I_K} w_i * Y_i_aligned

The index selection itself is NOT required to be differentiable (it is
ordinary discrete TopK); only the renormalized weights are, via
`torch.gather` on `p_full`, which is a differentiable function of the
upstream retrieval scores. This is accurately described as
"piecewise-differentiable Soft Top-K weighted retrieval" -- not
"fully differentiable TopK" (the index choice itself has no gradient
path; nothing here claims it does).

p_full must already be a valid probability distribution over the
candidate axis (e.g. `mean_mixture_topk_selection`'s own `p_mix`, or any
single-head `softmax(masked_scores / tau_s, dim=-1)`) -- this module
never computes scores or probabilities itself, by design, so every
method (V0/V1/V2/V5/Hard/Joint) supplies its own canonical p_full and
only the K-selection + renormalization + weighted-mean step is shared.
"""
import torch


def soft_topk_weights(p_full, picks, eps=1e-12):
    """p_full: [B, N] probability distribution (rows sum to 1 over valid
    candidates). picks: [B, K] long indices into the N axis (from an
    ordinary TopK on p_full, e.g. `stable_topk_indices`).

    Returns w: [B, K], renormalized so each row sums to 1, differentiable
    w.r.t. p_full via `gather` (picks itself carries no gradient -- it is
    a LongTensor, which is exactly what makes this "index selection is
    discrete, selected probabilities are differentiable")."""
    p_topk = p_full.gather(1, picks)  # [B, K], differentiable
    denom = p_topk.sum(dim=1, keepdim=True).clamp_min(eps)
    return p_topk / denom


def soft_topk_weighted_forecast(weights, y_sel):
    """weights: [B, K]. y_sel: [B, K, T] (future values at the K picks,
    already offset/alignment-applied). Returns R: [B, T]."""
    return (y_sel * weights.unsqueeze(-1)).sum(dim=1)


def soft_topk_select_and_forecast(p_full, picks, y_sel, eps=1e-12):
    """Convenience wrapper: weights = soft_topk_weights(...); R =
    soft_topk_weighted_forecast(weights, y_sel). Returns (R, weights) so
    callers that need the weights for QA features or diagnostics don't
    have to recompute them."""
    w = soft_topk_weights(p_full, picks, eps=eps)
    r = soft_topk_weighted_forecast(w, y_sel)
    return r, w
