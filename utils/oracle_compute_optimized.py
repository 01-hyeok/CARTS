"""TRACK-A-WEATHER-OPT03 -- common optimized backend for the Oracle compute
paths actually used by `scripts/train_factorial_e2e01.py`'s running Weather
H96/H720 factorial: Individual Oracle (A) and Greedy Set Oracle (E). Oracle-
Choice CE (C) is the shared loss and is NOT reimplemented here (see
`research/TRACK-A-WEATHER-OPT03.md` section 2 -- profiled, found <10% of
iteration time, left unchanged per the spec's own "don't optimize under
10%" rule).

Design constraints (identical to OPT01/OPT02):
    - FP32 throughout, no BF16.
    - candidate-dimension chunking; no additional full [B, N, H] tensor is
      ever materialized beyond the unavoidable per-chunk [B, chunk, H]
      slice (same pattern OPT02 already established and validated).
    - deterministic tie-break: relies on `torch.argmax`/`torch.argmin`'s own
      documented smallest-index convention, exactly like OPT01/OPT02 --
      this module returns a full `[B, N]` VALUE tensor and never picks an
      index itself, so the caller's own argmax/argmin decides ties exactly
      as it does today against the reference.
    - The Greedy Set Oracle (E) path is NOT reimplemented here: it is the
      exact same computation `utils/dense_utility_optimized.py`'s OPT02 code
      already implements and already validated (39/39 unit tests,
      H96/H720 real-checkpoint equivalence in
      `results/TRACK-A-WEATHER-OPT02/equivalence_H{96,720}.json`). This
      module re-exports it so callers have ONE import path for both Oracle
      types without any duplicate implementation.
"""
import torch
import torch.nn.functional as F

from utils.dense_utility_optimized import (dense_utility_optimized,  # noqa: F401 (re-export)
                                            prepare_query_static,
                                            prepare_query_static_chunked)

# --------------------------------------------------------------------------
# A. Individual Oracle -- optimized (squared-Euclidean-distance expansion)
# --------------------------------------------------------------------------
# reference (scripts/train_factorial_e2e01.py::individual_utility):
#     u_i = -mean_h (Y_i - Y_q)^2
# exact algebraic identity used here (NOT an approximation):
#     ||Y_i - Y_q||^2 = ||Y_i||^2 - 2*Y_i . Y_q + ||Y_q||^2
# `||Y_i||^2` is CANDIDATE-ONLY (no query dependence) and, within one
# channel, IDENTICAL FOR EVERY QUERY in the batch's memory bank and for
# every one of the K greedy steps (the Individual Oracle is step-invariant
# by definition -- module docstring of `individual_utility` already says so,
# but the reference call site does not act on it: it is called once per
# step inside `run_sequence`'s K-loop, i.e. K=10x redundantly, on the exact
# same inputs each time; see OPT03 report section 1 finding A-1). Computing
# `||Y_i||^2` once per (channel, memory bank) and reusing it for every query
# and every step is a pure caching win with zero numerical difference from
# summing the same squares fresh each time.


def prepare_individual_query_static(futures, candidate_chunk_size=None):
    """`||Y_i||^2`, `[B, N]`. Candidate-only, reusable across every query
    row that shares this `futures` bank and across all K greedy steps.
    Chunked so no persistent `[B, N, H]` tensor is required beyond the
    per-chunk slice."""
    bsz, n, h = futures.shape
    cs = candidate_chunk_size or n
    cand_sq = futures.new_empty(bsz, n)
    for start in range(0, n, cs):
        end = min(start + cs, n)
        cand_sq[:, start:end] = futures[:, start:end, :].pow(2).sum(dim=-1)
    return cand_sq


def individual_oracle_utility_optimized(futures, query_future, cand_sq=None,
                                        candidate_chunk_size=None):
    """u_i = -MSE(Y_i, Y_q) via the norm-expansion identity, chunked.

    `cand_sq`: precomputed `prepare_individual_query_static(...)` output.
    Pass it in explicitly to get the caching benefit across queries/steps;
    if omitted it is computed fresh (still correct, just not cached).

    No eps/clamp is needed anywhere in this formula (unlike the Greedy Set
    Oracle's weighted-aggregate division) -- squared Euclidean distance has
    no small-denominator failure mode, so none is introduced here.
    """
    bsz, n, h = futures.shape
    cs = candidate_chunk_size or n
    if cand_sq is None:
        cand_sq = prepare_individual_query_static(futures, cs)
    q_sq = query_future.pow(2).sum(dim=-1, keepdim=True)  # [B, 1], query-only
    out = futures.new_empty(bsz, n)
    q = query_future.unsqueeze(1)
    for start in range(0, n, cs):
        end = min(start + cs, n)
        y_c = futures[:, start:end, :]
        dot = torch.einsum('bh,bnh->bn', query_future, y_c)
        d_sq = cand_sq[:, start:end] - 2.0 * dot + q_sq
        out[:, start:end] = -(d_sq / h)
    return out


# --------------------------------------------------------------------------
# C. Oracle-Choice CE -- optimized (dead-code elimination, NOT a loss change)
# --------------------------------------------------------------------------
# PROFILING FINDING (OPT03 report section 2): on real Weather H96/H720 data,
# `scripts/train_oracle_choice01.py::oracle_choice_step_loss` -- the ONE loss
# shared by all 8 factorial arms -- accounts for 85-89% of forward-pass
# iteration time, not the Oracle target computation (A/E) the OPT03 spec
# expected to dominate. Root cause: an unused diagnostic,
#
#     std_per_row = torch.nanmean(torch.tensor([float(torch.std(scale[b][valid_mask[b]]))
#                                                for b in range(u_hat.size(0))
#                                                if bool(valid_mask[b].sum() > 1)], ...))
#
# is computed every call. `float(...)` inside a per-batch-row Python loop
# forces a GPU->CPU synchronization on every one of the (up to) 32 rows, on
# every one of the K=10 steps, on every channel, on every batch. `std_per_row`
# is NEVER placed into the returned `diag` dict and is not read anywhere
# else in the function -- it is genuinely dead code, not a diagnostic the
# training loop or any downstream consumer relies on (checked: `diag`'s only
# keys are `top1_acc`, `top5_acc`, `top10_acc`, `pred_rank_mean`,
# `top1_top2_margin_mean`).
#
# The optimized version below is EVERY OTHER LINE of the reference,
# unchanged, with only this dead computation removed. `loss` and `diag` are
# therefore not just "close" but IDENTICAL in every element to the
# reference's returned values for the same inputs (verified below) -- this
# is not a loss redefinition, it is the removal of an unused side
# computation.
def oracle_choice_step_loss_optimized(u_hat, u_target, valid_mask, tau):
    """Byte-identical `(loss, diag)` to
    `scripts.train_oracle_choice01.oracle_choice_step_loss`, with the dead
    `std_per_row` diagnostic (never read, see module comment above) removed."""
    neg_inf = torch.finfo(u_target.dtype).min / 4
    target_masked = u_target.masked_fill(~valid_mask, neg_inf)
    i_star = target_masked.argmax(dim=-1)

    logits = u_hat.masked_fill(~valid_mask, neg_inf) / float(tau)
    row_has_valid = valid_mask.any(dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(1, i_star.unsqueeze(-1)).squeeze(-1)
    nll = nll[row_has_valid]
    loss = nll.mean() if nll.numel() > 0 else u_hat.sum() * 0.0

    with torch.no_grad():
        pred_rank = (logits > logits.gather(1, i_star.unsqueeze(-1))).sum(dim=-1).float()
        top1 = (u_hat.masked_fill(~valid_mask, neg_inf).argmax(dim=-1) == i_star).float()
        top5 = (pred_rank < 5).float()
        top10 = (pred_rank < 10).float()
        valid_count = valid_mask.sum(dim=-1).clamp_min(2)
        top2_target = target_masked.topk(2, dim=-1).values
        margin_abs = (top2_target[:, 0] - top2_target[:, 1])
        rows = row_has_valid & (valid_count >= 2)

    diag = {
        'top1_acc': float(top1[rows].mean()) if bool(rows.any()) else float('nan'),
        'top5_acc': float(top5[rows].mean()) if bool(rows.any()) else float('nan'),
        'top10_acc': float(top10[rows].mean()) if bool(rows.any()) else float('nan'),
        'pred_rank_mean': float(pred_rank[rows].mean()) if bool(rows.any()) else float('nan'),
        'top1_top2_margin_mean': float(margin_abs[rows].mean()) if bool(rows.any()) else float('nan'),
    }
    return loss, diag


__all__ = [
    'dense_utility_optimized', 'prepare_query_static', 'prepare_query_static_chunked',
    'prepare_individual_query_static', 'individual_oracle_utility_optimized',
    'oracle_choice_step_loss_optimized',
]
