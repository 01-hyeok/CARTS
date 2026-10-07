"""TRACK-V-MEANMIX-INFERENCE01 -- the ONLY new piece of math in this
track: replaces `round_robin_topk_selection` (canonical inference) with
a selection rule that matches what V1/V2/V5 are ACTUALLY TRAINED on
(`train_t_pure_multislot01.py`: `p_m = softmax(s_masked/tau_s);
p_bar = p_m.mean(dim=1); L = KL(p_t || p_bar)`):

    p_h(i|q)   = softmax_i(masked_scores_h(i|q) / tau_s)
    p_mix(i|q) = mean_h p_h(i|q)
    T(q)       = TopK_i[p_mix(i|q)]

Candidate-side scoring, encoding, teacher, mask, and the post-selection
Uniform-Top-10-future-mean aggregation are all UNCHANGED and untouched
by this module -- this file only ever receives an already-computed
`scores: [B, S, N]` tensor and a `cand_mask: [B, N]`, exactly the same
inputs `round_robin_topk_selection` takes.
"""
import torch

from models.RelationStage1 import stable_topk_indices


def mean_mixture_topk_selection(scores, cand_mask, tau_s, k=10):
    """scores: [B, S, N] raw logits (S=1 for V0/V1). cand_mask: [B, N].
    Returns (picks: [B, k] long, p_heads: [B, S, N], p_mix: [B, N]) --
    p_heads/p_mix are the REAL probabilities (not -inf'd) for
    diagnostics; `picks` is computed from a defensively re-masked copy
    so an invalid candidate can never be selected even in a
    floating-point edge case."""
    masked = scores.masked_fill(~cand_mask.unsqueeze(1), float('-inf'))
    p_heads = torch.softmax(masked / tau_s, dim=-1)  # [B, S, N], each row sums to 1 over valid i
    p_mix = p_heads.mean(dim=1)  # [B, N]
    p_mix_for_topk = p_mix.masked_fill(~cand_mask, float('-inf'))
    picks = stable_topk_indices(p_mix_for_topk, k, largest=True)
    return picks, p_heads, p_mix
