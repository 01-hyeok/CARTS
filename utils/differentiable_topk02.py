"""TRACK-V5-TOPK-CORRECTION-MECHANISM01 Phase D -- Differentiable Top-K v2.

`utils/differentiable_topk01.py` (v1, PRESERVED UNMODIFIED) sets its
relaxation threshold to the K-th largest score itself:

    threshold = s_(K)                      (K-th largest, detached)
    a_i       = sigmoid((s_i - threshold) / tau) / Z

This has a real convergence bug: for the K-th-ranked candidate itself,
`s_i - threshold == 0` identically, so `sigmoid(0) = 0.5` REGARDLESS of
`tau`. As `tau -> 0`, the top `K-1` candidates (strictly above the
threshold) saturate to raw weight `1`, but the K-th candidate is stuck at
raw weight exactly `0.5` -- so the normalized limit is `a_i -> 1/(K-0.5)`
for ranks `1..K-1` and `a_i -> 0.5/(K-0.5)` for rank `K`, NOT the uniform
`1/K` for all K candidates that v1's own docstring claims. (The ranks
`K+1..N` correctly vanish in the limit; only the boundary candidate's
weight is wrong.)

## v2 fix: midpoint threshold

    threshold = (s_(K) + s_(K+1)) / 2       (midpoint of the K-th and
                                               (K+1)-th largest scores,
                                               detached, same policy as v1)
    a_i       = sigmoid((s_i - threshold) / tau) / Z

Whenever `s_(K) != s_(K+1)` (a strict score gap at the cutoff), every one
of the top-K candidates has `s_i - threshold > 0` and every candidate
ranked `K+1` or below has `s_i - threshold < 0` -- so as `tau -> 0`,
`sigmoid` saturates to exactly `1` for all K selected candidates and
exactly `0` for everyone else, giving the TRUE uniform `1/K` limit v1's
docstring describes but v1's implementation does not deliver.

## Tie-breaking (the case v1's docstring does not address at all)

If `s_(K) == s_(K+1)` (an exact tie straddling the cutoff), the same
degenerate "stuck at 0.5" failure mode recurs for every tied candidate --
the midpoint fix alone does not resolve genuine ties. This module breaks
ties THE SAME WAY `models.RelationStage1.stable_topk_indices` already
does for the repo's hard selection (lower original index wins a tie,
via a stable sort) -- consistency with the hard selection matters because
Phase D/E need v2's hard-tau limit to match the SAME Top-K set the rest
of the pipeline's hard inference would pick, tie or no tie.

The tie-break is implemented as an infinitesimal, strictly-decreasing-in-
-index perturbation applied ONLY inside the (detached, `torch.no_grad()`)
threshold computation, computed in float64 for headroom:

    perturbed_i = s_i - i * TIE_EPS          (TIE_EPS = 1e-9, float64)

This makes the rank ordering of `perturbed` strictly consistent with
"ties broken toward the lower index," exactly matching
`stable_topk_indices`' convention (`stable_argsort` on `-values`, which
keeps the original relative order of equal elements, so the
lower-index element sorts first for `largest=True`). The SAME perturbed
value is what every `sigmoid` numerator uses too (not the original
`s_i`) -- using the unperturbed score for the numerator would leave two
EXACTLY tied candidates with the identical sigmoid argument regardless
of `tau` (the tie-break would only ever decide which one gets LABELED
`s_(K)` vs `s_(K+1)`, never actually separate their output weights).
Since the perturbation is a fixed per-index constant, not a function of
any trainable parameter, `d(perturbed_i)/d(s_i) == 1` -- it changes
nothing about the gradient, only shifts the sigmoid argument by an
amount (`<= 1e-9 * N`) utterly negligible next to any realistic `tau`.
Because the perturbation is `O(1e-9 * N)`, exact convergence to
the correctly-tie-broken hard limit is only guaranteed as `tau -> 0`
(the same honest caveat v1 already carries for near-tied DISTINCT
scores) -- at a fixed moderate `tau` (e.g. the training default 0.05),
an exact tie still produces two nearly-equal weights close to the
untied candidates' weight, not a sharp 1-vs-0 split; this is expected
and tested via a tau-decreasing sweep, not a single fixed tau.

## Threshold gradient: detached (same policy as v1, for the same reason)

`s_(K)` and `s_(K+1)` are themselves selected via a hard order-statistic
operation (effectively `torch.kthvalue`/`torch.sort`) whose derivative
w.r.t. the full score vector has a discontinuity exactly where two
candidates swap rank -- letting gradient flow through the threshold
would make the overall surrogate gradient discontinuous in a
training-destabilizing way at every rank-swap event. Detaching (v1's
existing choice, kept identical here) means `d(threshold)/d(s_i) = 0`
for all `i`, so the ONLY gradient path is `d(a_i)/d(s_i)` through the
`sigmoid` numerator at fixed threshold -- smooth everywhere by
construction.

## Degenerate case: fewer than K+1 valid candidates

If a row has `n_valid <= K` valid candidates, there is no well-defined
`(K+1)`-th largest. The threshold is set to `-inf` in that row, which
makes every valid candidate's `sigmoid((s_i - (-inf))/tau) = sigmoid(+inf)
= 1` -- i.e. ALL valid candidates are selected with equal weight
`1/n_valid`, the only sensible behavior when there are not even K
candidates to choose among (`n_valid < K`) or exactly K (`n_valid == K`,
the correct Top-K IS "everybody").

## Complexity

Identical to v1: `O(N log N)` for the order-statistic threshold (a sort,
done once per row in float64 for tie-break headroom) plus `O(N)` for the
sigmoid/renormalization.
"""
import torch

from utils.differentiable_topk01 import soft_topk_forecast  # re-exported, unchanged

EPS = 1e-12
TIE_EPS = 1e-9


def soft_topk_mask_v2(scores, valid_mask, k, tau):
    """scores: [..., N]. valid_mask: broadcastable to scores. k: int.
    tau: float > 0. Returns `a`: same shape as `scores`, `a >= 0`,
    `a.sum(-1) == 1` over valid candidates, `a[invalid] == 0`.

    `tau -> 0` recovers EXACT Hard Top-K with uniform `1/k` weights,
    including at exact score ties (broken the same way
    `stable_topk_indices` breaks them) -- the bug v1 has at the K-th
    boundary candidate is fixed by the midpoint threshold; see module
    docstring for the full derivation and the tie-break mechanism."""
    assert tau > 0, '[ISSUE][ABORT] soft_topk_mask_v2 requires tau > 0 (use the hard selection directly at tau=0)'
    orig_dtype = scores.dtype
    # The entire forward computation, including the final sigmoid, is done in
    # float64 -- NOT just the threshold. An earlier version cast the threshold
    # back to `orig_dtype` right after averaging s_(K)/s_(K+1); when `orig_dtype`
    # is float32, that cast rounds away the infinitesimal TIE_EPS-scale
    # separation entirely (float32's ULP near O(1) values, ~1.2e-7, is ~100x
    # larger than TIE_EPS*N for any realistic N), silently resurrecting the
    # exact "stuck at sigmoid(0)=0.5" bug this module exists to fix, for any
    # EXACT tie. Staying in float64 end-to-end (scores are losslessly
    # upcast first) avoids this; only the final returned weights are cast
    # back to `orig_dtype`, which is safe since those are probabilities in
    # [0, 1] with no tie-breaking information left to lose.
    masked_scores = scores.double().masked_fill(~valid_mask, float('-inf'))
    n = scores.shape[-1]

    # The TIE_EPS perturbation is applied to `perturbed`, which is what BOTH
    # the threshold AND the final sigmoid numerator use -- not just the
    # threshold. Using the unperturbed `masked_scores` for the numerator
    # would make two EXACTLY tied candidates receive the exact same sigmoid
    # argument regardless of tau (since the numerator depends only on each
    # candidate's own score and the shared threshold, and tied candidates
    # have identical scores) -- the tie-break would then only ever affect
    # which candidate's score gets LABELED s_(K) vs s_(K+1) without ever
    # actually separating their OUTPUT weights. Applying the identical
    # perturbation to the numerator fixes this: `d(perturbed_i)/d(scores_i)
    # == 1` (the offset is a fixed constant per index, not a function of any
    # trainable parameter), so gradients are unaffected, and the Nx(TIE_EPS)
    # shift is utterly negligible for any realistically-separated score.
    idx = torch.arange(n, device=scores.device, dtype=torch.float64)
    perturbed = masked_scores - idx * TIE_EPS

    with torch.no_grad():
        sorted_vals, _ = torch.sort(perturbed.detach(), dim=-1, descending=True)
        n_valid = valid_mask.sum(dim=-1, keepdim=True).clamp(min=1, max=n)

        k_pos = torch.full_like(n_valid, k - 1).clamp(max=n - 1)
        kp1_pos = torch.full_like(n_valid, k).clamp(max=n - 1)
        s_k = torch.gather(sorted_vals, -1, k_pos)
        s_kp1 = torch.gather(sorted_vals, -1, kp1_pos)
        threshold = (s_k + s_kp1) / 2.0  # stays float64, NOT cast back yet

        few_valid = n_valid <= k  # not enough candidates for a well-defined (K+1)-th
        threshold = torch.where(few_valid, torch.full_like(threshold, float('-inf')), threshold)

    raw = torch.sigmoid((perturbed - threshold) / tau)
    raw = raw.masked_fill(~valid_mask, 0.0)
    # sigmoid(nan) can arise from (-inf) - (-inf) on invalid entries when threshold
    # is -inf; already overwritten to 0.0 above, but guard defensively in case any
    # NaN survives into the denom sum from an unexpected dtype/broadcast edge case
    raw = torch.nan_to_num(raw, nan=0.0)
    denom = raw.sum(dim=-1, keepdim=True).clamp_min(EPS)
    return (raw / denom).to(orig_dtype)


def hard_limit_matches_stable_topk_v2(scores, valid_mask, k, tau, atol=1e-3):
    """Diagnostic/test helper, mirrors v1's `hard_limit_matches_stable_topk`
    exactly so v1/v2 can be compared with the identical check."""
    from models.RelationStage1 import stable_topk_indices
    a = soft_topk_mask_v2(scores, valid_mask, k, tau)
    hard_idx = stable_topk_indices(scores.masked_fill(~valid_mask, float('-inf')), k, largest=True)
    a_at_hard = a.gather(-1, hard_idx)
    max_abs_diff = float((a_at_hard - 1.0 / k).abs().max())
    total_mass_outside = float(a.sum(-1).mean()) - float(a_at_hard.sum(-1).mean())
    return max_abs_diff < atol and total_mass_outside < atol, max_abs_diff
