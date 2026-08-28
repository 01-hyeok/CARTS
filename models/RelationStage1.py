import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from layers.relation_patch_embed import RelationPatchEmbedding
from layers.relation_tcn import RelationTCN


from layers.pairwise_scorer import PairwiseScorer, build_pair_features  # noqa: E402
from layers.retrieval_metric import (  # noqa: E402
    METRICS as RETRIEVAL_METRICS,
    RetrievalMetric,
    cosine_init_deviation,
    oracle_rank_statistics,
    score_separation_metrics,
)


def stable_topk_indices(values, k, dim=-1, largest=True):
    """Top-k indices with ties broken by candidate index.

    `torch.topk` leaves tie order unspecified, so ranking a compacted row and
    ranking the same row with its invalid entries masked out could pick
    different — equally scored — candidates. Sorting stably makes the choice a
    property of the data instead of the layout, which is what lets the
    vectorized retrieval metrics be pinned to the per-query reference.
    """
    key = -values if largest else values
    return stable_argsort(key, dim=dim).narrow(dim, 0, int(k))


def stable_argsort(values, dim=-1):
    """Ascending stable argsort that also works on torch < 1.13.

    `torch.argsort` only grew a `stable` keyword in 1.13, while `torch.sort`
    has had one since 1.9. Ties must keep their candidate order, otherwise the
    rank-based retrieval diagnostics move with the sort implementation.
    """
    return torch.sort(values, dim=dim, stable=True).indices


@torch.no_grad()
def select_training_candidates(bank_scores, future_mse, valid_mask, top_m, oracle_k,
                              random_negatives=0, generator=None):
    """Mine Bank Top-M by the model's own score, guaranteeing the Oracle Top-K.

    `bank_scores` is whatever score function the arm uses -- cosine for the
    incumbent, the pair scorer for the learnable arms -- so the training pool is
    self-selected and differs between arms. Arms share the mining *rule*
    (Top-M plus Oracle injection), not the candidate ids. Evaluation is separate
    and always runs over the full memory.

    The memory bank keeps its job of searching the whole candidate pool; this
    only decides *which* candidates the loss will be computed over. Injection is
    training-only supervision: a fresh encoder can rank every good candidate
    outside its Top-M, and then the loss never sees a positive at all.

    `random_negatives` appends uniformly sampled valid candidates. A fixed score
    like cosine extrapolates to pairs it never trained on by construction; a
    learned scorer does not, and a pool made only of Top-M neighbours leaves it
    unconstrained on the rest of the bank -- which is exactly where evaluation
    then asks it to rank. Measured on a one-epoch run without them, the pair
    scorer came out *anti*-correlated with future MSE over the full bank
    (Spearman -0.46) despite training cleanly on its 100 mined candidates.

    Returns
    -------
    selected : [B, M + random_negatives] long
        Candidate indices, best bank rank first, then the sampled negatives.
        Oracle candidates missing from Bank Top-M replace the worst-ranked
        non-Oracle slots, so the mined part stays exactly M.
    stats : dict
        Mining diagnostics measured *before* injection.
    """
    if bank_scores.shape != valid_mask.shape or bank_scores.shape != future_mse.shape:
        raise ValueError(
            'bank_scores, future_mse and valid_mask must share shape; got '
            f'{tuple(bank_scores.shape)}, {tuple(future_mse.shape)}, {tuple(valid_mask.shape)}'
        )
    bsz, num_cand = bank_scores.shape
    valid_mask = valid_mask.bool()
    top_m = min(int(top_m), num_cand)
    oracle_k = min(int(oracle_k), num_cand)
    if top_m <= 0 or oracle_k <= 0:
        raise ValueError('top_m and oracle_k must be positive')
    if oracle_k > top_m:
        raise ValueError(
            f'oracle_k={oracle_k} cannot exceed top_m={top_m}; the Oracle set '
            'would not fit inside the selected candidates'
        )

    device = bank_scores.device
    scores = bank_scores.detach().float().masked_fill(~valid_mask, float('-inf'))
    distances = future_mse.detach().float().masked_fill(~valid_mask, float('inf'))

    bank_top = torch.topk(scores, k=top_m, dim=-1, largest=True).indices
    oracle = torch.topk(distances, k=oracle_k, dim=-1, largest=False).indices

    # A row with fewer than oracle_k valid candidates pads its Oracle set with
    # invalid entries. Those must never be injected.
    oracle_is_valid = valid_mask.gather(1, oracle)

    in_bank = (oracle.unsqueeze(-1) == bank_top.unsqueeze(-2)).any(dim=-1)
    stats = {
        'oracle_count_in_bank_top_m': (in_bank & oracle_is_valid).sum(dim=-1).float(),
        'oracle_valid_count': oracle_is_valid.sum(dim=-1).float(),
    }
    # Treat invalid Oracle slots as already present so they are never injected.
    in_bank = in_bank | ~oracle_is_valid
    missing_per_row = (~in_bank).sum(dim=-1)

    if int(missing_per_row.max()) == 0:
        stats['oracle_missing_count_before_injection'] = missing_per_row.float()
        return _append_random_negatives(
            bank_top, valid_mask, random_negatives, generator, stats), stats

    # Missing Oracle candidates first, each group keeping its Oracle rank order.
    order = stable_argsort(in_bank.int(), dim=1)
    ordered_oracle = oracle.gather(1, order)

    # Walk the bank ranking worst-first and overwrite non-Oracle slots.
    reverse = torch.arange(top_m - 1, -1, -1, device=device)
    bank_reversed = bank_top.index_select(1, reverse)
    holds_oracle = (
        bank_reversed.unsqueeze(-1) == oracle.unsqueeze(-2)
    ).any(dim=-1)
    free_slot = ~holds_oracle
    free_rank = free_slot.long().cumsum(dim=-1) - 1
    take = free_slot & (free_rank < missing_per_row.unsqueeze(1)) & (free_rank >= 0)
    replacement = ordered_oracle.gather(1, free_rank.clamp(0, oracle_k - 1))
    bank_reversed = torch.where(take, replacement, bank_reversed)
    selected = bank_reversed.index_select(1, reverse)

    stats['oracle_missing_count_before_injection'] = missing_per_row.float()
    return _append_random_negatives(
        selected, valid_mask, random_negatives, generator, stats), stats


def _append_random_negatives(selected, valid_mask, count, generator, stats):
    """Add uniformly sampled valid candidates the mining never reaches.

    Sampled without replacement so a duplicate cannot be counted twice in the
    student's softmax denominator. Already-selected columns are given zero
    weight; a row whose valid pool is exhausted falls back to sampling from all
    valid candidates.
    """
    count = int(count)
    stats['random_negative_count'] = selected.new_zeros(
        selected.size(0), dtype=torch.float32) + float(max(count, 0))
    if count <= 0:
        return selected
    weights = valid_mask.float().scatter(1, selected, 0.0)
    exhausted = weights.sum(dim=-1, keepdim=True) < count
    weights = torch.where(exhausted, valid_mask.float(), weights)
    if int(weights.sum(dim=-1).min()) < count:
        raise ValueError(
            f'cannot draw {count} random negatives; some query has too few valid candidates'
        )
    sampled = torch.multinomial(weights, count, replacement=False, generator=generator)
    return torch.cat([selected, sampled], dim=-1)


def future_aware_topk_ranking_loss(
    student_scores,
    future_mse,
    valid_mask,
    top_k,
    rank_margin=0.1,
    min_mse_gap=0.0,
    eps=1e-8,
    return_debug=False,
):
    """Pairwise top-k ranking loss over candidate axis.

    student_scores is raw cosine similarity, not temperature-scaled logits.
    Shapes:
      student_scores: [B, R, M] or [B, M]
      future_mse: [B, R, M] or [B, M]
      valid_mask: [B, R, M] or [B, M]
    """
    squeeze_relation = False
    if student_scores.dim() == 2:
        student_scores = student_scores.unsqueeze(1)
        squeeze_relation = True
    if student_scores.dim() != 3:
        raise ValueError(f'student_scores must be [B, R, M] or [B, M], got {tuple(student_scores.shape)}')

    bsz, rels, num_cand = student_scores.shape
    device = student_scores.device
    dtype = student_scores.dtype
    zero = student_scores.sum() * 0.0

    if num_cand == 0 or int(top_k) <= 0:
        metrics = _empty_rank_metrics(zero, dtype, device)
        return (zero, metrics, {}) if return_debug else (zero, metrics)

    if future_mse.dim() == 2:
        future_mse = future_mse.unsqueeze(1).expand(bsz, rels, num_cand)
    elif future_mse.dim() == 3:
        if future_mse.size(1) == 1 and rels != 1:
            future_mse = future_mse.expand(bsz, rels, num_cand)
    else:
        raise ValueError(f'future_mse must be [B, R, M] or [B, M], got {tuple(future_mse.shape)}')

    if valid_mask.dim() == 2:
        valid_mask = valid_mask.unsqueeze(1).expand(bsz, rels, num_cand)
    elif valid_mask.dim() == 3:
        if valid_mask.size(1) == 1 and rels != 1:
            valid_mask = valid_mask.expand(bsz, rels, num_cand)
    else:
        raise ValueError(f'valid_mask must be [B, R, M] or [B, M], got {tuple(valid_mask.shape)}')

    if future_mse.shape != student_scores.shape:
        raise ValueError(f'future_mse shape {tuple(future_mse.shape)} does not match {tuple(student_scores.shape)}')
    if valid_mask.shape != student_scores.shape:
        raise ValueError(f'valid_mask shape {tuple(valid_mask.shape)} does not match {tuple(student_scores.shape)}')

    valid_mask = valid_mask.bool()
    k = min(int(top_k), int(num_cand))
    if k <= 0:
        metrics = _empty_rank_metrics(zero, dtype, device)
        return (zero, metrics, {}) if return_debug else (zero, metrics)

    masked_score_fill = torch.finfo(dtype).min / 4
    future_detached = future_mse.detach()
    teacher_scores = future_detached.masked_fill(~valid_mask, float('inf'))
    student_scores_masked = student_scores.masked_fill(~valid_mask, masked_score_fill)

    teacher_topk_mse, teacher_topk_idx = torch.topk(teacher_scores, k=k, dim=-1, largest=False)
    student_topk_scores, student_topk_idx = torch.topk(student_scores_masked, k=k, dim=-1, largest=True)

    teacher_topk_valid = torch.isfinite(teacher_topk_mse)
    student_topk_valid = student_topk_scores > masked_score_fill / 2
    effective_k = valid_mask.sum(dim=-1).clamp_max(k)
    active = effective_k > 0

    teacher_in_student = (
        teacher_topk_idx.unsqueeze(-1) == student_topk_idx.unsqueeze(-2)
    ) & teacher_topk_valid.unsqueeze(-1) & student_topk_valid.unsqueeze(-2)
    overlap_count = teacher_in_student.any(dim=-1).float().sum(dim=-1)
    overlap = overlap_count / effective_k.float().clamp_min(1.0)

    missed_mask = teacher_topk_valid & ~teacher_in_student.any(dim=-1)
    hard_mask = student_topk_valid & ~teacher_in_student.any(dim=-2)

    pos_idx = teacher_topk_idx
    neg_idx = student_topk_idx
    pos_scores = student_scores.gather(-1, pos_idx)
    neg_scores = student_scores.gather(-1, neg_idx)
    pos_mse = future_detached.gather(-1, pos_idx)
    neg_mse = future_detached.gather(-1, neg_idx)

    pair_mask = missed_mask.unsqueeze(-1) & hard_mask.unsqueeze(-2)
    pair_mask = pair_mask & ((neg_mse.unsqueeze(-2) - pos_mse.unsqueeze(-1)) > float(min_mse_gap))

    pair_gap = pos_scores.unsqueeze(-1) - neg_scores.unsqueeze(-2)
    pair_loss = F.softplus(float(rank_margin) - pair_gap)

    pair_mask_float = pair_mask.float()
    pair_count = pair_mask_float.sum()
    pair_denom = pair_count.clamp_min(1.0)
    loss = (pair_loss * pair_mask_float).sum() / pair_denom
    pair_accuracy = ((pair_gap > 0).float() * pair_mask_float).sum() / pair_denom
    margin_satisfied = ((pair_gap > float(rank_margin)).float() * pair_mask_float).sum() / pair_denom
    score_gap = (pair_gap * pair_mask_float).sum() / pair_denom

    active_float = active.float()
    active_count = active_float.sum().clamp_min(1.0)
    missed_count = missed_mask.float().sum(dim=-1)
    hard_count = hard_mask.float().sum(dim=-1)
    valid_pair_count_per_rel = pair_mask.float().sum(dim=(-1, -2))

    def active_mean(value):
        return (value * active_float).sum() / active_count

    teacher_topk_future_mse = torch.where(teacher_topk_valid, teacher_topk_mse, torch.zeros_like(teacher_topk_mse))
    student_topk_mse = future_detached.gather(-1, student_topk_idx)
    student_topk_future_mse = torch.where(student_topk_valid, student_topk_mse, torch.zeros_like(student_topk_mse))

    teacher_mse_mean = teacher_topk_future_mse.sum(dim=-1) / teacher_topk_valid.float().sum(dim=-1).clamp_min(1.0)
    student_mse_mean = student_topk_future_mse.sum(dim=-1) / student_topk_valid.float().sum(dim=-1).clamp_min(1.0)
    missed_mse_mean = (pos_mse * missed_mask.float()).sum(dim=-1) / missed_count.clamp_min(1.0)
    hard_mse_mean = (neg_mse * hard_mask.float()).sum(dim=-1) / hard_count.clamp_min(1.0)

    missed_active = missed_count > 0
    hard_active = hard_count > 0
    metrics = {
        'stage1_loss_rank': loss.detach(),
        'rank_teacher_student_topk_overlap': active_mean(overlap).detach(),
        'rank_teacher_student_topk_overlap_count': active_mean(overlap_count).detach(),
        'rank_missed_positive_count': active_mean(missed_count).detach(),
        'rank_hard_negative_count': active_mean(hard_count).detach(),
        'rank_valid_pair_count': active_mean(valid_pair_count_per_rel).detach(),
        'rank_pair_accuracy': pair_accuracy.detach(),
        'rank_score_gap': score_gap.detach(),
        'rank_margin_satisfied_ratio': margin_satisfied.detach(),
        'rank_teacher_topk_future_mse': active_mean(teacher_mse_mean).detach(),
        'rank_student_topk_future_mse': active_mean(student_mse_mean).detach(),
        'rank_missed_positive_future_mse': (
            (missed_mse_mean * missed_active.float()).sum() / missed_active.float().sum().clamp_min(1.0)
        ).detach(),
        'rank_hard_negative_future_mse': (
            (hard_mse_mean * hard_active.float()).sum() / hard_active.float().sum().clamp_min(1.0)
        ).detach(),
    }

    if squeeze_relation:
        student_scores = student_scores.squeeze(1)

    debug = {
        'teacher_topk_idx': teacher_topk_idx,
        'student_topk_idx': student_topk_idx,
        'teacher_topk_valid': teacher_topk_valid,
        'student_topk_valid': student_topk_valid,
        'missed_positive_mask': missed_mask,
        'hard_negative_mask': hard_mask,
        'pair_mask': pair_mask,
    }
    return (loss, metrics, debug) if return_debug else (loss, metrics)


@torch.no_grad()
def prepare_topk_coverage_targets(future_mse, valid_mask, top_k):
    """Prepare target-future Oracle Top-K once for all source relations."""
    if future_mse.dim() != 2:
        raise ValueError(f'future_mse must be [B, N], got {tuple(future_mse.shape)}')
    if valid_mask.shape != future_mse.shape:
        raise ValueError('future_mse and valid_mask must have the same shape')
    if int(top_k) <= 0:
        raise ValueError('topk coverage top_k must be positive')

    valid_mask = valid_mask.bool()
    future = future_mse.detach().float()
    if not torch.isfinite(future[valid_mask]).all():
        raise ValueError('future_mse contains NaN or Inf at a valid candidate')

    bsz, num_cand = future.shape
    k = min(int(top_k), int(num_cand))
    effective_k = valid_mask.sum(dim=-1).clamp_max(k)
    active_query = effective_k > 0
    if k == 0:
        empty_long = torch.empty(bsz, 0, dtype=torch.long, device=future.device)
        empty_float = torch.empty(bsz, 0, dtype=future.dtype, device=future.device)
        empty_bool = torch.empty(bsz, 0, dtype=torch.bool, device=future.device)
        return {
            'oracle_indices': empty_long,
            'oracle_mse': empty_float,
            'oracle_valid': empty_bool,
            'effective_k': effective_k,
            'active_query': active_query,
        }

    masked_future = future.masked_fill(~valid_mask, float('inf'))
    oracle_mse, oracle_indices = torch.topk(
        masked_future, k=k, dim=-1, largest=False
    )
    positions = torch.arange(k, device=future.device).unsqueeze(0)
    oracle_valid = positions < effective_k.unsqueeze(1)
    oracle_mse = oracle_mse.masked_fill(~oracle_valid, 0.0)
    return {
        'oracle_indices': oracle_indices.detach(),
        'oracle_mse': oracle_mse.detach(),
        'oracle_valid': oracle_valid.detach(),
        'effective_k': effective_k.detach(),
        'active_query': active_query.detach(),
    }


def topk_coverage_loss(student_log_prob, targets):
    """Uniform Oracle Top-K cross-entropy over each active query."""
    if student_log_prob.dim() != 2:
        raise ValueError(
            f'student_log_prob must be [B, N], got {tuple(student_log_prob.shape)}'
        )
    required = {
        'oracle_indices', 'oracle_mse', 'oracle_valid',
        'effective_k', 'active_query',
    }
    missing = required.difference(targets)
    if missing:
        raise ValueError(f'topk coverage targets missing keys: {sorted(missing)}')

    oracle_indices = targets['oracle_indices']
    oracle_valid = targets['oracle_valid'].bool()
    effective_k = targets['effective_k']
    active_query = targets['active_query'].bool()
    if oracle_indices.size(0) != student_log_prob.size(0):
        raise ValueError('topk coverage target batch size does not match student_log_prob')
    if oracle_indices.shape != oracle_valid.shape:
        raise ValueError('oracle_indices and oracle_valid must have the same shape')

    zero = student_log_prob.sum() * 0.0
    if oracle_indices.size(1) == 0 or not active_query.any():
        metrics = {
            'topk_coverage_loss': zero.detach(),
            'oracle_topk_probability_mass': zero.detach(),
            'oracle_positive_probability_mean': zero.detach(),
            'oracle_positive_probability_min': zero.detach(),
            'coverage_effective_k': zero.detach(),
            'coverage_oracle_student_overlap': zero.detach(),
        }
        return zero, metrics

    gathered_log_prob = student_log_prob.gather(1, oracle_indices).float()
    if not torch.isfinite(gathered_log_prob[oracle_valid]).all():
        raise ValueError('student_log_prob contains NaN or Inf at an Oracle positive')
    valid_float = oracle_valid.float()
    denominator = effective_k.float().clamp_min(1.0)
    per_query_loss = -(gathered_log_prob * valid_float).sum(dim=-1) / denominator
    loss = per_query_loss[active_query].mean().to(student_log_prob.dtype)
    if not torch.isfinite(loss):
        raise ValueError('topk_coverage_loss is NaN or Inf')

    positive_probability = gathered_log_prob.exp() * valid_float
    probability_mass = positive_probability.sum(dim=-1)
    probability_mean = probability_mass / denominator
    probability_min = positive_probability.masked_fill(
        ~oracle_valid, float('inf')
    ).min(dim=-1).values

    k = oracle_indices.size(1)
    student_indices = torch.topk(student_log_prob, k=k, dim=-1).indices
    positions = torch.arange(k, device=student_log_prob.device).unsqueeze(0)
    student_valid = positions < effective_k.unsqueeze(1)
    oracle_in_student = (
        oracle_indices.unsqueeze(-1) == student_indices.unsqueeze(-2)
    ) & oracle_valid.unsqueeze(-1) & student_valid.unsqueeze(-2)
    overlap = oracle_in_student.any(dim=-1).float().sum(dim=-1) / denominator

    def active_mean(value):
        return value[active_query].mean().detach()

    metrics = {
        'topk_coverage_loss': loss.detach(),
        'oracle_topk_probability_mass': active_mean(probability_mass),
        'oracle_positive_probability_mean': active_mean(probability_mean),
        'oracle_positive_probability_min': active_mean(probability_min),
        'coverage_effective_k': active_mean(effective_k.float()),
        'coverage_oracle_student_overlap': active_mean(overlap),
    }
    return loss, metrics


def weighted_topk_listwise_ce(student_log_prob, targets, tau_teacher, eps=1e-8):
    """Cross-entropy over the Oracle Top-K, weighted by how good each member is.

    `topk_coverage_loss` spreads its target uniformly over the Oracle Top-K, which
    treats the 1st and 10th best candidate as equally worth retrieving. The
    near-tie measurement says that is close to true -- the 10th and 11th differ by
    1.4% -- but not exactly, and the ordering inside the set is what Stage-2's
    weighting would read if it could.

    This keeps the Oracle Top-K as the only positives, and grades them:

        w_i = softmax(-d_i / tau_T) over the Oracle set
        L   = -sum_i w_i * log p_S(i | q)

    The student's softmax still runs over the *whole* candidate set, so a
    negative scoring too highly still costs, even though its target weight is
    zero. That is the difference from a loss defined only on the positives.
    """
    if student_log_prob.dim() != 2:
        raise ValueError(
            f'student_log_prob must be [B, N], got {tuple(student_log_prob.shape)}')
    required = {'oracle_indices', 'oracle_mse', 'oracle_valid', 'active_query'}
    missing = required.difference(targets)
    if missing:
        raise ValueError(f'weighted topk targets missing keys: {sorted(missing)}')
    if float(tau_teacher) <= 0.0:
        raise ValueError('tau_teacher must be positive')

    oracle_indices = targets['oracle_indices']
    oracle_valid = targets['oracle_valid'].bool()
    active_query = targets['active_query'].bool()
    if oracle_indices.size(0) != student_log_prob.size(0):
        raise ValueError('target batch size does not match student_log_prob')

    zero = student_log_prob.sum() * 0.0
    if oracle_indices.size(1) == 0 or not active_query.any():
        return zero, {'weighted_topk_ce_loss': zero.detach()}

    # Weights over the Oracle set only. Centring on the best member keeps exp()
    # finite when the distances are large relative to tau.
    distances = targets['oracle_mse'].detach().float()
    logits = (-distances / float(tau_teacher)).masked_fill(~oracle_valid, float('-inf'))
    logits = logits - logits.max(dim=-1, keepdim=True).values
    weights = logits.exp() * oracle_valid.float()
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(eps)

    gathered = student_log_prob.gather(1, oracle_indices).float()
    if not torch.isfinite(gathered[oracle_valid]).all():
        raise ValueError('student_log_prob contains NaN or Inf at an Oracle positive')
    per_query = -(gathered * weights).sum(dim=-1)
    loss = per_query[active_query].mean().to(student_log_prob.dtype)
    if not torch.isfinite(loss):
        raise ValueError('weighted_topk_listwise_ce is NaN or Inf')

    positive_probability = gathered.exp() * oracle_valid.float()
    # Bound once and reused: writing the effective count inline as
    # `-(...).sum().exp()` negates the exponential rather than the entropy.
    weight_entropy = -(weights * (weights + eps).log()).sum(dim=-1)
    metrics = {
        'weighted_topk_ce_loss': loss.detach(),
        'weighted_topk_oracle_mass': positive_probability.sum(dim=-1)[active_query].mean().detach(),
        'weighted_topk_weight_entropy': weight_entropy[active_query].mean().detach(),
        'weighted_topk_top1_weight': weights.max(dim=-1).values[active_query].mean().detach(),
        'weighted_topk_effective_positives': (
            weight_entropy.exp()[active_query].mean().detach()),
    }
    return loss, metrics


def multi_positive_infonce_loss(student_logits, positive_distance, valid_mask, top_k):
    """Separate the smallest-distance Top-K set from all remaining valid candidates."""
    if student_logits.dim() != 2:
        raise ValueError(
            f'student_logits must be [B, N], got {tuple(student_logits.shape)}'
        )
    if positive_distance.shape != student_logits.shape or valid_mask.shape != student_logits.shape:
        raise ValueError(
            'student_logits, positive_distance, and valid_mask must have the same shape'
        )

    targets = prepare_topk_coverage_targets(positive_distance, valid_mask, top_k)
    positive_indices = targets['oracle_indices']
    positive_valid = targets['oracle_valid'].bool()
    active_query = targets['active_query'].bool()
    zero = student_logits.sum() * 0.0
    if positive_indices.size(1) == 0 or not active_query.any():
        metrics = {
            'stage1_loss_infonce': zero.detach(),
            'infonce_positive_probability_mass': zero.detach(),
            'infonce_effective_positive_count': zero.detach(),
            'infonce_oracle_student_topk_overlap': zero.detach(),
        }
        return zero, metrics

    valid_mask = valid_mask.bool()
    masked_fill = torch.finfo(student_logits.dtype).min / 4
    valid_logits = student_logits.masked_fill(~valid_mask, masked_fill)
    positive_logits = valid_logits.gather(1, positive_indices)
    positive_logits = positive_logits.masked_fill(~positive_valid, masked_fill)

    log_all = torch.logsumexp(valid_logits.float(), dim=-1)
    log_positive = torch.logsumexp(positive_logits.float(), dim=-1)
    per_query_loss = -(log_positive - log_all)
    loss = per_query_loss[active_query].mean().to(student_logits.dtype)
    if not torch.isfinite(loss):
        raise ValueError('multi_positive_infonce_loss is NaN or Inf')

    positive_mass = torch.exp(log_positive - log_all)
    k = positive_indices.size(1)
    student_indices = torch.topk(valid_logits, k=k, dim=-1).indices
    student_valid = (
        torch.arange(k, device=student_logits.device).unsqueeze(0)
        < targets['effective_k'].unsqueeze(1)
    )
    positive_in_student = (
        positive_indices.unsqueeze(-1) == student_indices.unsqueeze(-2)
    ) & positive_valid.unsqueeze(-1) & student_valid.unsqueeze(-2)
    overlap = (
        positive_in_student.any(dim=-1).float().sum(dim=-1)
        / targets['effective_k'].float().clamp_min(1.0)
    )

    metrics = {
        'stage1_loss_infonce': loss.detach(),
        'infonce_positive_probability_mass': positive_mass[active_query].mean().detach(),
        'infonce_effective_positive_count': (
            targets['effective_k'][active_query].float().mean().detach()
        ),
        'infonce_oracle_student_topk_overlap': overlap[active_query].mean().detach(),
    }
    return loss, metrics


@torch.no_grad()
def prepare_query_conditioned_rnc_targets(
    future_mse,
    valid_mask,
    tie_epsilon=0.0,
):
    """Prepare an MSE-based candidate ordering for one relation branch."""
    if float(tie_epsilon) < 0.0:
        raise ValueError('rnc_tie_epsilon must be non-negative')
    if future_mse.shape != valid_mask.shape:
        raise ValueError('future_mse and valid_mask must have the same shape')

    prepared = []
    for query_idx in range(future_mse.size(0)):
        valid_indices = torch.nonzero(valid_mask[query_idx], as_tuple=False).flatten()
        if valid_indices.numel() < 2:
            continue

        row_future = future_mse[query_idx].index_select(0, valid_indices).float()
        if not torch.isfinite(row_future).all():
            raise ValueError('future_mse contains NaN or Inf at a valid candidate')
        order = stable_argsort(row_future, dim=0)
        sorted_future = row_future[order]
        sorted_candidate_indices = valid_indices[order]

        if float(tie_epsilon) == 0.0:
            _, group_counts = torch.unique_consecutive(sorted_future, return_counts=True)
            group_starts = torch.cumsum(group_counts, dim=0) - group_counts
        else:
            starts = []
            counts = []
            start = 0
            while start < sorted_future.numel():
                end = int(torch.searchsorted(
                    sorted_future,
                    sorted_future[start] + float(tie_epsilon),
                    right=True,
                ))
                starts.append(start)
                counts.append(end - start)
                start = end
            group_starts = torch.tensor(starts, device=sorted_future.device, dtype=torch.long)
            group_counts = torch.tensor(counts, device=sorted_future.device, dtype=torch.long)

        denominator_starts = torch.repeat_interleave(group_starts, group_counts)
        denominator_counts = sorted_future.numel() - denominator_starts
        anchor_mask = denominator_counts > 1
        if anchor_mask.any():
            prepared.append({
                'query_index': query_idx,
                'candidate_indices': sorted_candidate_indices,
                'sorted_future_mse': sorted_future,
                'denominator_starts': denominator_starts,
                'anchor_mask': anchor_mask,
            })
    return prepared


def query_conditioned_rnc_loss(
    student_scores,
    future_mse,
    valid_mask,
    temperature=0.2,
    tie_epsilon=0.0,
    prepared_targets=None,
    return_debug=False,
):
    """Query-conditioned RnC loss without materializing a pairwise matrix."""
    if float(temperature) <= 0.0:
        raise ValueError('rnc_temperature must be positive')
    if float(tie_epsilon) < 0.0:
        raise ValueError('rnc_tie_epsilon must be non-negative')
    if student_scores.dim() != 2:
        raise ValueError(f'student_scores must be [B, N], got {tuple(student_scores.shape)}')
    if future_mse.shape != student_scores.shape:
        raise ValueError(
            f'future_mse shape {tuple(future_mse.shape)} does not match '
            f'{tuple(student_scores.shape)}'
        )
    if valid_mask.shape != student_scores.shape:
        raise ValueError(
            f'valid_mask shape {tuple(valid_mask.shape)} does not match '
            f'{tuple(student_scores.shape)}'
        )

    valid_mask = valid_mask.bool()
    zero = student_scores.sum() * 0.0
    validate_row_scores = prepared_targets is None
    if prepared_targets is None:
        prepared_targets = prepare_query_conditioned_rnc_targets(
            future_mse, valid_mask, tie_epsilon=tie_epsilon
        )

    query_losses = []
    anchor_counts = []
    debug_rows = []

    for target in prepared_targets:
        query_idx = target['query_index']
        row_scores = student_scores[query_idx].index_select(
            0, target['candidate_indices']
        )
        if validate_row_scores and not torch.isfinite(row_scores).all():
            raise ValueError('student_scores contains NaN or Inf at a valid candidate')
        sorted_logits = row_scores.float() / float(temperature)
        suffix_lse = torch.flip(
            torch.logcumsumexp(torch.flip(sorted_logits, dims=[0]), dim=0),
            dims=[0],
        )
        denominator_starts = target['denominator_starts']
        anchor_mask = target['anchor_mask']
        per_candidate = suffix_lse[denominator_starts] - sorted_logits
        query_losses.append(per_candidate[anchor_mask].mean())
        anchor_counts.append(anchor_mask.float().sum())
        if return_debug:
            debug_rows.append({
                'query_index': query_idx,
                'order': target['candidate_indices'],
                'sorted_future_mse': target['sorted_future_mse'],
                'sorted_logits': sorted_logits,
                'suffix_logsumexp': suffix_lse,
                'denominator_starts': denominator_starts,
                'anchor_mask': anchor_mask,
                'per_candidate_loss': per_candidate,
            })

    if not query_losses:
        metrics = {
            'rnc_loss': zero.detach(),
            'rnc_valid_query_count': zero.detach(),
            'rnc_anchor_count': zero.detach(),
        }
        return (zero, metrics, debug_rows) if return_debug else (zero, metrics)

    loss = torch.stack(query_losses).mean().to(student_scores.dtype)
    if not torch.isfinite(loss):
        raise ValueError('rnc_loss is NaN or Inf')
    metrics = {
        'rnc_loss': loss.detach(),
        'rnc_valid_query_count': student_scores.new_tensor(float(len(query_losses))),
        'rnc_anchor_count': torch.stack(anchor_counts).mean().detach(),
    }
    return (loss, metrics, debug_rows) if return_debug else (loss, metrics)


def expected_future_mse_loss(
    student_prob,
    future_mse,
    valid_mask,
    normalization='mean',
    eps=1e-8,
):
    """Expected future MSE under the existing full student distribution."""
    if normalization not in ('none', 'mean', 'median'):
        raise ValueError(f'Unsupported expected_mse_normalization: {normalization}')
    if student_prob.dim() != 2:
        raise ValueError(f'student_prob must be [B, N], got {tuple(student_prob.shape)}')
    if future_mse.shape != student_prob.shape or valid_mask.shape != student_prob.shape:
        raise ValueError('student_prob, future_mse, and valid_mask must have the same shape')

    valid_mask = valid_mask.bool()
    zero = student_prob.sum() * 0.0
    valid_query = valid_mask.any(dim=-1)
    if not valid_query.any():
        metrics = {
            'expected_mse_loss': zero.detach(),
            'student_expected_future_mse_raw': zero.detach(),
            'student_expected_future_mse_normalized': zero.detach(),
        }
        return zero, metrics

    prob = student_prob.float().masked_fill(~valid_mask, 0.0)
    future = future_mse.detach().float().masked_fill(~valid_mask, 0.0)
    if not torch.isfinite(prob[valid_mask]).all():
        raise ValueError('student_prob contains NaN or Inf at a valid candidate')
    if not torch.isfinite(future[valid_mask]).all():
        raise ValueError('future_mse contains NaN or Inf at a valid candidate')

    raw_per_query = (prob * future).sum(dim=-1)
    if normalization == 'none':
        normalized_per_query = raw_per_query
    elif normalization == 'mean':
        denominator = (
            future.sum(dim=-1) / valid_mask.sum(dim=-1).clamp_min(1)
        ).detach()
        normalized_per_query = raw_per_query / (denominator + float(eps))
    else:
        normalized_rows = []
        for query_idx in torch.nonzero(valid_query, as_tuple=False).flatten().tolist():
            row_mask = valid_mask[query_idx]
            denominator = future[query_idx, row_mask].median().detach()
            normalized_rows.append(raw_per_query[query_idx] / (denominator + float(eps)))
        normalized_per_query = torch.stack(normalized_rows)

    if normalization == 'median':
        loss = normalized_per_query.mean().to(student_prob.dtype)
    else:
        loss = normalized_per_query[valid_query].mean().to(student_prob.dtype)
    raw_loss = raw_per_query[valid_query].mean().to(student_prob.dtype)
    if not torch.isfinite(loss) or not torch.isfinite(raw_loss):
        raise ValueError('expected future MSE loss is NaN or Inf')
    metrics = {
        'expected_mse_loss': loss.detach(),
        'student_expected_future_mse_raw': raw_loss.detach(),
        'student_expected_future_mse_normalized': loss.detach(),
    }
    return loss, metrics


@torch.no_grad()
def normalize_teacher_scores(scores, valid_mask, mode, eps=1e-8):
    """Put a teacher's scores on a per-query unit scale so tau means one thing.

    Future-MSE differences and utility differences live on very different scales
    -- at tau=0.01 the future teacher keeps ~3 effective candidates while the
    utility teacher keeps ~22 -- so comparing arms at a shared tau without this
    would compare sharpness, not targets.

    Only the scale is divided out, never the mean. Softmax is shift-invariant so
    that costs nothing for a plain KL, and it matters for the NULL arms, where a
    utility of exactly zero is the abstention action's own score and must keep
    its meaning.
    """
    if mode == 'none':
        return scores
    if mode != 'per_query_scale':
        raise ValueError(f'Unsupported teacher normalization: {mode}')
    count = valid_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(scores.dtype)
    masked = scores.masked_fill(~valid_mask, 0.0)
    mean = masked.sum(dim=-1, keepdim=True) / count
    variance = (scores - mean).masked_fill(~valid_mask, 0.0).square().sum(dim=-1, keepdim=True) / count
    return scores / variance.clamp_min(eps).sqrt()


@torch.no_grad()
def external_pool_utility_metrics(student_scores, utility, valid_mask, seed=0, depth=10):
    """How much measured downstream gain the student's own ranking collects.

    Reported for every arm on the fixed pool, whatever teacher trained it, so
    checkpoint selection can run on downstream gain instead of on Future Recall
    -- the alignment study showed those two disagree.

    Gap recovery is framed against a shuffled ranking rather than against zero:
    0 means no better than picking at random from the same pool, 1 means matching
    the best selection available in it.
    """
    masked_fill = torch.finfo(student_scores.dtype).min
    utility = utility.float()
    width = min(depth, student_scores.size(-1))

    def collect(scores):
        return utility.gather(
            1, scores.masked_fill(~valid_mask, masked_fill).topk(width, dim=-1).indices
        )

    retrieved = collect(student_scores)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    shuffled = torch.rand(student_scores.shape, generator=generator).to(student_scores.device)
    random = collect(shuffled)
    oracle = collect(utility)

    graded = utility.clamp_min(0.0).masked_fill(~valid_mask, 0.0)
    discount = 1.0 / torch.log2(
        torch.arange(width, device=utility.device, dtype=torch.float32) + 2.0
    )
    by_student = student_scores.masked_fill(~valid_mask, masked_fill).topk(width, dim=-1).indices
    ideal = graded.topk(width, dim=-1).indices
    dcg = (graded.gather(1, by_student) * discount).sum(-1)
    idcg = (graded.gather(1, ideal) * discount).sum(-1)
    keep = idcg > 0
    ndcg = (dcg[keep] / idcg[keep]).mean() if keep.any() else dcg.new_zeros(())

    retrieved_mean, random_mean = retrieved.mean(), random.mean()
    return {
        'utility_retrieved_at_10': retrieved_mean,
        'utility_random_at_10': random_mean,
        'utility_oracle_at_10': oracle.mean(),
        'utility_positive_rate_at_10': (retrieved > 0).float().mean(),
        'utility_best_available': utility.masked_fill(~valid_mask, masked_fill).max(-1).values.mean(),
        'utility_ndcg_at_10': ndcg,
        'utility_gap_recovery_at_10': (
            (retrieved_mean - random_mean) / (oracle.mean() - random_mean).clamp_min(1e-8)
        ),
    }


def utility_teacher_loss(
    student_scores,
    teacher_scores,
    utility,
    valid_mask,
    tau_student,
    tau_teacher,
    objective='kl',
    normalize='per_query_scale',
    null_logit=None,
    eps=1e-8,
):
    """Retrieval supervision from an externally measured teacher.

    Two objectives over the same fixed candidate pool:

        kl                KL(teacher || student), teacher = softmax(scores / tau)
        expected_utility  -E_{k ~ student}[U(q, k)], maximising the actual gain
                          the student's own distribution would collect

    `null_logit` adds an explicit abstention action whose utility is exactly zero
    -- the "retrieve nothing" branch. Without it a softmax retriever must always
    pick someone, which is wrong whenever every candidate in the pool is harmful,
    and in the long horizons that is most queries.

    Shapes are [B, M] throughout; `null_logit` is [B] or [B, 1].
    """
    if objective not in ('kl', 'expected_utility'):
        raise ValueError(f'Unsupported teacher objective: {objective}')
    for name, tensor in (('teacher_scores', teacher_scores), ('utility', utility),
                         ('valid_mask', valid_mask)):
        if tensor.shape != student_scores.shape:
            raise ValueError(
                f'{name} shape {tuple(tensor.shape)} does not match student '
                f'{tuple(student_scores.shape)}'
            )
    masked_fill = torch.finfo(student_scores.dtype).min

    teacher_source = normalize_teacher_scores(
        teacher_scores.detach().float(), valid_mask, normalize, eps
    )
    utility = utility.detach().float()

    student_logits = (student_scores / tau_student).masked_fill(~valid_mask, masked_fill)
    teacher_logits = (teacher_source / tau_teacher).masked_fill(~valid_mask, masked_fill)
    # The abstention action is always available, so its column is never masked.
    if null_logit is not None:
        null_logit = null_logit.reshape(-1, 1)
        student_logits = torch.cat([student_logits, null_logit / tau_student], dim=-1)
        null_teacher = normalize_teacher_scores(
            torch.zeros_like(null_logit), valid_mask.new_ones(null_logit.shape), 'none'
        )
        teacher_logits = torch.cat([teacher_logits, null_teacher / tau_teacher], dim=-1)
        utility = torch.cat([utility, torch.zeros_like(null_logit, dtype=utility.dtype)], dim=-1)
        valid_mask = torch.cat([valid_mask, valid_mask.new_ones(null_logit.shape)], dim=-1)

    student_log_prob = torch.log_softmax(student_logits, dim=-1)
    student_prob = student_log_prob.exp()
    teacher_prob = torch.softmax(teacher_logits, dim=-1).detach()

    if objective == 'kl':
        per_query = (teacher_prob * ((teacher_prob + eps).log() - student_log_prob)).sum(dim=-1)
    else:
        per_query = -(student_prob * utility).sum(dim=-1)

    active = valid_mask.sum(dim=-1) > (1 if null_logit is not None else 0)
    per_query = per_query[active]
    if per_query.numel() == 0 or not torch.isfinite(per_query).all():
        return None, {}
    loss = per_query.mean()

    zero = loss.detach() * 0.0
    expected_utility = (student_prob * utility).sum(dim=-1)
    depth = min(10, student_scores.size(-1))
    retrieved = utility[:, :student_scores.size(-1)].gather(
        1, student_scores.masked_fill(~valid_mask[:, :student_scores.size(-1)], masked_fill)
        .topk(depth, dim=-1).indices
    )
    metrics = {
        'utility_teacher_loss': loss.detach(),
        'utility_teacher_expected_utility': expected_utility[active].mean().detach(),
        'utility_teacher_entropy': -(
            teacher_prob * (teacher_prob + eps).log()
        ).sum(dim=-1)[active].mean().detach(),
        'utility_student_entropy': -(
            student_prob * student_log_prob
        ).sum(dim=-1)[active].mean().detach(),
        'utility_retrieved_at_10': retrieved.mean().detach(),
        'utility_positive_rate_at_10': (retrieved > 0).float().mean().detach(),
        'utility_best_available': utility.masked_fill(~valid_mask, masked_fill)
        .max(dim=-1).values[active].mean().detach(),
        'utility_null_probability': (
            student_prob[:, -1][active].mean().detach() if null_logit is not None else zero
        ),
        'utility_teacher_null_probability': (
            teacher_prob[:, -1][active].mean().detach() if null_logit is not None else zero
        ),
    }
    return loss, metrics

def _student_retrieval_metrics_reference(student_scores, student_prob, future_mse, valid_mask, eps=1e-8):
    """Per-query reference implementation, kept as the correctness oracle.

    `_student_retrieval_metrics` is the vectorized form of exactly this; the
    differential test in tests/test_stage1_metric_vectorization.py pins them
    together so the fast path can never silently drift.
    """
    valid_mask = valid_mask.bool()
    rows = []
    for query_idx in range(student_scores.size(0)):
        row_mask = valid_mask[query_idx]
        count = int(row_mask.sum())
        if count == 0:
            continue
        scores = student_scores[query_idx, row_mask].detach().float()
        prob = student_prob[query_idx, row_mask].detach().float()
        distances = future_mse[query_idx, row_mask].detach().float()
        if not torch.isfinite(scores).all() or not torch.isfinite(prob).all():
            raise ValueError('student distribution contains NaN or Inf at a valid candidate')
        if not torch.isfinite(distances).all():
            raise ValueError('future_mse contains NaN or Inf at a valid candidate')

        entropy = -(prob * torch.log(prob.clamp_min(float(eps)))).sum()
        entropy_norm = entropy / math.log(count) if count > 1 else entropy * 0.0
        row = {
            'student_entropy_normalized': entropy_norm,
            'student_max_probability': prob.max(),
            'student_top5_probability_mass': prob[
                stable_topk_indices(prob, min(5, count))
            ].sum(),
        }

        mean_distance = distances.mean().clamp_min(float(eps))
        relevance = 1.0 / (1.0 + distances / mean_distance)
        oracle_best_idx = torch.argmin(distances)
        for k in (1, 5, 10):
            effective_k = min(k, count)
            student_idx = stable_topk_indices(scores, effective_k, largest=True)
            oracle_idx = stable_topk_indices(distances, effective_k, largest=False)
            overlap = (
                student_idx[:, None] == oracle_idx[None, :]
            ).any(dim=1).float().mean()
            retrieved_mse = distances[student_idx].mean()
            oracle_mse = distances[oracle_idx].mean()
            row[f'oracle_recall_at_{k}'] = overlap
            row[f'oracle_best_hit_at_{k}'] = (
                student_idx == oracle_best_idx
            ).any().float()
            row[f'topk_probability_mass_at_{k}'] = prob[student_idx].sum()
            row[f'oracle_topk_probability_mass_at_{k}'] = prob[oracle_idx].sum()
            row[f'retrieved_future_mse_at_{k}'] = retrieved_mse
            row[f'best_future_mse_at_{k}'] = distances[student_idx].min()
            row[f'oracle_future_mse_at_{k}'] = oracle_mse
            row[f'retrieval_regret_at_{k}'] = retrieved_mse - oracle_mse

            if k in (5, 10):
                discounts = 1.0 / torch.log2(
                    torch.arange(effective_k, device=scores.device, dtype=torch.float32) + 2.0
                )
                dcg = (relevance[student_idx] * discounts).sum()
                idcg = (relevance[oracle_idx] * discounts).sum()
                row[f'ndcg_at_{k}'] = dcg / idcg.clamp_min(float(eps))

        if count > 1:
            score_order = stable_argsort(scores)
            quality_order = stable_argsort(-distances)
            score_rank = torch.empty_like(scores)
            quality_rank = torch.empty_like(distances)
            score_rank[score_order] = torch.arange(count, device=scores.device, dtype=torch.float32)
            quality_rank[quality_order] = torch.arange(count, device=scores.device, dtype=torch.float32)
            score_rank = score_rank - score_rank.mean()
            quality_rank = quality_rank - quality_rank.mean()
            denominator = torch.sqrt(
                score_rank.square().sum() * quality_rank.square().sum()
            ).clamp_min(float(eps))
            row['spearman_score_vs_negative_mse'] = (
                score_rank * quality_rank
            ).sum() / denominator
        else:
            row['spearman_score_vs_negative_mse'] = scores.new_tensor(0.0)
        rows.append(row)

    if not rows:
        zero = student_scores.detach().sum() * 0.0
        keys = [
            'student_entropy_normalized', 'student_max_probability',
            'student_top5_probability_mass', 'spearman_score_vs_negative_mse',
        ]
        for k in (1, 5, 10):
            keys.extend([
                f'oracle_recall_at_{k}', f'oracle_best_hit_at_{k}',
                f'topk_probability_mass_at_{k}',
                f'oracle_topk_probability_mass_at_{k}',
                f'retrieved_future_mse_at_{k}', f'best_future_mse_at_{k}',
                f'oracle_future_mse_at_{k}', f'retrieval_regret_at_{k}',
            ])
        keys.extend(['ndcg_at_5', 'ndcg_at_10'])
        return {key: zero for key in keys}

    return {
        key: torch.stack([row[key] for row in rows]).mean().detach()
        for key in rows[0]
    }


_STUDENT_RETRIEVAL_KEYS = [
    'student_entropy_normalized', 'student_max_probability',
    'student_top5_probability_mass', 'spearman_score_vs_negative_mse',
    *[
        f'{name}_at_{k}'
        for k in (1, 5, 10)
        for name in (
            'oracle_recall', 'oracle_best_hit', 'topk_probability_mass',
            'oracle_topk_probability_mass', 'retrieved_future_mse',
            'best_future_mse', 'oracle_future_mse', 'retrieval_regret',
        )
    ],
    'ndcg_at_5', 'ndcg_at_10',
]


def _masked_ranks(sort_key, valid_mask, counts):
    """Dense 0-based ranks of the valid entries, ascending by sort_key.

    Invalid entries are pushed past every valid one so the valid entries keep
    exactly the ranks 0..count-1 they would get from ranking a compacted row.
    """
    pushed = sort_key.masked_fill(~valid_mask, float('inf'))
    order = stable_argsort(pushed, dim=-1)
    ranks = torch.empty_like(pushed)
    positions = torch.arange(
        pushed.size(-1), device=pushed.device, dtype=pushed.dtype
    ).expand_as(pushed)
    ranks.scatter_(-1, order, positions)
    return ranks


@torch.no_grad()
def _student_retrieval_metrics(student_scores, student_prob, future_mse, valid_mask, eps=1e-8):
    """Future-aware diagnostics; NDCG uses relevance=1/(1+mean-normalized MSE).

    Vectorized over the batch. `_student_retrieval_metrics_reference` is the
    per-query original and stays the definition of correct; every rule it has is
    reproduced here rather than approximated:

    * queries with no valid candidate are dropped from the mean, not zero-filled
    * each query uses its own ``effective_k = min(k, count)``
    * ranks and means are taken over that query's valid candidates only
    """
    valid_mask = valid_mask.bool()
    counts = valid_mask.sum(dim=-1)
    active = counts > 0
    zero = student_scores.detach().sum() * 0.0
    if not bool(active.any()):
        return {key: zero for key in _STUDENT_RETRIEVAL_KEYS}

    scores = student_scores.detach().float()
    prob = student_prob.detach().float()
    distances = future_mse.detach().float()
    if not torch.isfinite(scores[valid_mask]).all() or not torch.isfinite(prob[valid_mask]).all():
        raise ValueError('student distribution contains NaN or Inf at a valid candidate')
    if not torch.isfinite(distances[valid_mask]).all():
        raise ValueError('future_mse contains NaN or Inf at a valid candidate')

    num_cand = scores.size(-1)
    count_f = counts.clamp_min(1).float()
    # Losing-end fills so masked entries can never win a topk or a min.
    score_pick = scores.masked_fill(~valid_mask, float('-inf'))
    dist_pick = distances.masked_fill(~valid_mask, float('inf'))
    prob_pick = prob.masked_fill(~valid_mask, float('-inf'))
    prob_zero = prob.masked_fill(~valid_mask, 0.0)
    dist_zero = distances.masked_fill(~valid_mask, 0.0)

    entropy = -(prob_zero * torch.log(prob_zero.clamp_min(float(eps)))).sum(dim=-1)
    entropy = torch.where(valid_mask.any(dim=-1), entropy, torch.zeros_like(entropy))
    log_count = torch.log(count_f)
    entropy_norm = torch.where(
        counts > 1, entropy / log_count.clamp_min(float(eps)), torch.zeros_like(entropy)
    )

    out = {
        'student_entropy_normalized': entropy_norm,
        'student_max_probability': prob_pick.max(dim=-1).values,
    }

    def take(values, indices, keep, fill):
        gathered = values.gather(1, indices)
        return gathered.masked_fill(~keep, fill)

    positions = torch.arange(num_cand, device=scores.device).unsqueeze(0)

    top5 = min(5, num_cand)
    idx5 = stable_topk_indices(prob_pick, top5, largest=True)
    keep5 = positions[:, :top5] < counts.clamp_max(top5).unsqueeze(1)
    out['student_top5_probability_mass'] = take(prob, idx5, keep5, 0.0).sum(dim=-1)

    mean_distance = (dist_zero.sum(dim=-1) / count_f).clamp_min(float(eps))
    relevance = 1.0 / (1.0 + distances / mean_distance.unsqueeze(1))
    oracle_best_idx = dist_pick.argmin(dim=-1)

    for k in (1, 5, 10):
        width = min(k, num_cand)
        effective_k = counts.clamp_max(width)
        keep = positions[:, :width] < effective_k.unsqueeze(1)
        denom = effective_k.clamp_min(1).float()

        student_idx = stable_topk_indices(score_pick, width, largest=True)
        oracle_idx = stable_topk_indices(dist_pick, width, largest=False)

        pair = (student_idx.unsqueeze(-1) == oracle_idx.unsqueeze(-2))
        pair = pair & keep.unsqueeze(-1) & keep.unsqueeze(-2)
        out[f'oracle_recall_at_{k}'] = pair.any(dim=-1).float().sum(dim=-1) / denom
        out[f'oracle_best_hit_at_{k}'] = (
            (student_idx == oracle_best_idx.unsqueeze(1)) & keep
        ).any(dim=-1).float()

        retrieved_mse = take(distances, student_idx, keep, 0.0).sum(dim=-1) / denom
        oracle_mse = take(distances, oracle_idx, keep, 0.0).sum(dim=-1) / denom
        out[f'topk_probability_mass_at_{k}'] = take(prob, student_idx, keep, 0.0).sum(dim=-1)
        out[f'oracle_topk_probability_mass_at_{k}'] = take(
            prob, oracle_idx, keep, 0.0
        ).sum(dim=-1)
        out[f'retrieved_future_mse_at_{k}'] = retrieved_mse
        out[f'best_future_mse_at_{k}'] = take(
            distances, student_idx, keep, float('inf')
        ).min(dim=-1).values
        out[f'oracle_future_mse_at_{k}'] = oracle_mse
        out[f'retrieval_regret_at_{k}'] = retrieved_mse - oracle_mse

        if k in (5, 10):
            # Discount at rank i does not depend on effective_k, so one row of
            # discounts serves every query once the tail is masked away.
            discounts = 1.0 / torch.log2(
                torch.arange(width, device=scores.device, dtype=torch.float32) + 2.0
            )
            dcg = (take(relevance, student_idx, keep, 0.0) * discounts).sum(dim=-1)
            idcg = (take(relevance, oracle_idx, keep, 0.0) * discounts).sum(dim=-1)
            out[f'ndcg_at_{k}'] = dcg / idcg.clamp_min(float(eps))

    score_rank = _masked_ranks(scores, valid_mask, counts)
    quality_rank = _masked_ranks(-distances, valid_mask, counts)
    score_rank = (score_rank - (score_rank * valid_mask).sum(dim=-1, keepdim=True) / count_f.unsqueeze(1)) * valid_mask
    quality_rank = (quality_rank - (quality_rank * valid_mask).sum(dim=-1, keepdim=True) / count_f.unsqueeze(1)) * valid_mask
    denominator = torch.sqrt(
        score_rank.square().sum(dim=-1) * quality_rank.square().sum(dim=-1)
    ).clamp_min(float(eps))
    spearman = (score_rank * quality_rank).sum(dim=-1) / denominator
    out['spearman_score_vs_negative_mse'] = torch.where(
        counts > 1, spearman, torch.zeros_like(spearman)
    )

    return {
        key: value[active].mean().detach()
        for key, value in out.items()
    }


def student_retrieval_metric_aliases(metrics):
    """Republish `_student_retrieval_metrics` rows under their `student_` names.

    The teacher branches name the same quantities `student_oracle_recall_at_k`,
    `student_retrieval_regret_at_k`, ... through `_ranking_source_topk_metrics`.
    Teacher-free objectives never reach that helper, so this restates the keys
    they do produce under the shared names instead of recomputing anything.
    """
    aliases = {}
    for k in (1, 5, 10):
        for source, target in (
            (f'oracle_recall_at_{k}', f'student_oracle_recall_at_{k}'),
            (f'oracle_best_hit_at_{k}', f'student_oracle_best_hit_at_{k}'),
            (f'topk_probability_mass_at_{k}', f'student_topk_probability_mass_at_{k}'),
            (
                f'oracle_topk_probability_mass_at_{k}',
                f'student_oracle_topk_probability_mass_at_{k}',
            ),
            (f'retrieved_future_mse_at_{k}', f'student_retrieved_future_mse_at_{k}'),
            (f'best_future_mse_at_{k}', f'student_best_future_mse_at_{k}'),
            (f'retrieval_regret_at_{k}', f'student_retrieval_regret_at_{k}'),
        ):
            if source in metrics:
                aliases[target] = metrics[source]
    for k in (5, 10):
        if f'ndcg_at_{k}' in metrics:
            aliases[f'student_ndcg_at_{k}'] = metrics[f'ndcg_at_{k}']
    if 'spearman_score_vs_negative_mse' in metrics:
        aliases['student_spearman_score_vs_negative_mse'] = metrics[
            'spearman_score_vs_negative_mse'
        ]
    return aliases


@torch.no_grad()
def _score_rank_spearman(first_scores, second_scores, valid_mask, eps=1e-8):
    """Mean per-query Spearman correlation over the same valid candidate pool."""
    if first_scores.shape != second_scores.shape or first_scores.shape != valid_mask.shape:
        raise ValueError(
            'first_scores, second_scores, and valid_mask must have the same shape'
        )

    valid_mask = valid_mask.bool()
    correlations = []
    for query_idx in range(first_scores.size(0)):
        row_mask = valid_mask[query_idx]
        count = int(row_mask.sum())
        if count <= 1:
            continue

        first = first_scores[query_idx, row_mask].detach().float()
        second = second_scores[query_idx, row_mask].detach().float()
        if not torch.isfinite(first).all() or not torch.isfinite(second).all():
            raise ValueError('rank-correlation scores contain NaN or Inf')

        first_order = stable_argsort(first)
        second_order = stable_argsort(second)
        first_rank = torch.empty_like(first)
        second_rank = torch.empty_like(second)
        rank_values = torch.arange(
            count, device=first.device, dtype=torch.float32
        )
        first_rank[first_order] = rank_values
        second_rank[second_order] = rank_values
        first_rank = first_rank - first_rank.mean()
        second_rank = second_rank - second_rank.mean()
        denominator = torch.sqrt(
            first_rank.square().sum() * second_rank.square().sum()
        ).clamp_min(float(eps))
        correlations.append((first_rank * second_rank).sum() / denominator)

    if not correlations:
        return first_scores.detach().sum() * 0.0
    return torch.stack(correlations).mean().detach()


@torch.no_grad()
def _teacher_student_distribution_metrics_reference(
    teacher_prob,
    student_prob,
    valid_mask,
    eps=1e-8,
):
    """Per-query reference for `_teacher_student_distribution_metrics`."""
    if teacher_prob.dim() != 2:
        raise ValueError(
            f'teacher_prob must be [B, N], got {tuple(teacher_prob.shape)}'
        )
    if teacher_prob.shape != student_prob.shape or teacher_prob.shape != valid_mask.shape:
        raise ValueError(
            'teacher_prob, student_prob, and valid_mask must have the same shape'
        )

    valid_mask = valid_mask.bool()
    rows = []
    for query_idx in range(teacher_prob.size(0)):
        row_mask = valid_mask[query_idx]
        count = int(row_mask.sum())
        if count == 0:
            continue

        teacher = teacher_prob[query_idx, row_mask].detach().float()
        student = student_prob[query_idx, row_mask].detach().float()
        if not torch.isfinite(teacher).all() or not torch.isfinite(student).all():
            raise ValueError('teacher/student distribution contains NaN or Inf')
        if (teacher < 0).any() or (student < 0).any():
            raise ValueError('teacher/student distribution contains a negative probability')

        teacher = teacher / teacher.sum().clamp_min(float(eps))
        student = student / student.sum().clamp_min(float(eps))
        teacher_log = torch.log(teacher.clamp_min(float(eps)))
        student_log = torch.log(student.clamp_min(float(eps)))
        midpoint = 0.5 * (teacher + student)
        midpoint_log = torch.log(midpoint.clamp_min(float(eps)))

        teacher_entropy = -(teacher * teacher_log).sum()
        student_entropy = -(student * student_log).sum()
        l1 = torch.abs(teacher - student).sum()
        row = {
            'teacher_student_kl_divergence': (
                teacher * (teacher_log - student_log)
            ).sum(),
            'student_teacher_kl_divergence': (
                student * (student_log - teacher_log)
            ).sum(),
            'teacher_student_js_divergence': 0.5 * (
                (teacher * (teacher_log - midpoint_log)).sum()
                + (student * (student_log - midpoint_log)).sum()
            ),
            'teacher_student_prob_l1': l1,
            'teacher_student_total_variation': 0.5 * l1,
            'teacher_student_hellinger_distance': torch.sqrt(
                0.5 * (
                    torch.sqrt(teacher) - torch.sqrt(student)
                ).square().sum().clamp_min(0.0)
            ),
            'teacher_student_probability_cosine': F.cosine_similarity(
                teacher.unsqueeze(0),
                student.unsqueeze(0),
                dim=-1,
                eps=float(eps),
            ).squeeze(0),
            'teacher_student_entropy_gap': student_entropy - teacher_entropy,
            'teacher_student_entropy_abs_gap': torch.abs(
                student_entropy - teacher_entropy
            ),
        }

        if count > 1:
            teacher_order = stable_argsort(teacher)
            student_order = stable_argsort(student)
            rank_values = torch.arange(
                count, device=teacher.device, dtype=torch.float32
            )
            teacher_rank = torch.empty_like(teacher)
            student_rank = torch.empty_like(student)
            teacher_rank[teacher_order] = rank_values
            student_rank[student_order] = rank_values
            teacher_rank = teacher_rank - teacher_rank.mean()
            student_rank = student_rank - student_rank.mean()
            rank_denominator = torch.sqrt(
                teacher_rank.square().sum() * student_rank.square().sum()
            ).clamp_min(float(eps))
            row['student_teacher_spearman'] = (
                teacher_rank * student_rank
            ).sum() / rank_denominator
        else:
            row['student_teacher_spearman'] = teacher.new_tensor(0.0)

        for k in (1, 5, 10):
            effective_k = min(k, count)
            teacher_topk = stable_topk_indices(teacher, effective_k, largest=True)
            student_topk = stable_topk_indices(student, effective_k, largest=True)
            overlap = (
                teacher_topk[:, None] == student_topk[None, :]
            ).any(dim=1).float().mean()
            row[f'teacher_student_topk_overlap_at_{k}'] = overlap
            # With equal-size Top-K sets, precision and recall are both overlap/K.
            row[f'student_teacher_recall_at_{k}'] = overlap
        rows.append(row)

    keys = [
        'teacher_student_kl_divergence',
        'student_teacher_kl_divergence',
        'teacher_student_js_divergence',
        'teacher_student_prob_l1',
        'teacher_student_total_variation',
        'teacher_student_hellinger_distance',
        'teacher_student_probability_cosine',
        'teacher_student_entropy_gap',
        'teacher_student_entropy_abs_gap',
        'student_teacher_spearman',
    ]
    for k in (1, 5, 10):
        keys.extend([
            f'teacher_student_topk_overlap_at_{k}',
            f'student_teacher_recall_at_{k}',
        ])
    if not rows:
        zero = teacher_prob.detach().sum() * 0.0
        return {key: zero for key in keys}
    return {
        key: torch.stack([row[key] for row in rows]).mean().detach()
        for key in keys
    }


_DISTRIBUTION_KEYS = [
    'teacher_student_kl_divergence',
    'student_teacher_kl_divergence',
    'teacher_student_js_divergence',
    'teacher_student_prob_l1',
    'teacher_student_total_variation',
    'teacher_student_hellinger_distance',
    'teacher_student_probability_cosine',
    'teacher_student_entropy_gap',
    'teacher_student_entropy_abs_gap',
    'student_teacher_spearman',
    *[
        f'{name}_at_{k}'
        for k in (1, 5, 10)
        for name in ('teacher_student_topk_overlap', 'student_teacher_recall')
    ],
]


@torch.no_grad()
def _teacher_student_distribution_metrics(
    teacher_prob,
    student_prob,
    valid_mask,
    eps=1e-8,
):
    """Compare teacher and student distributions on each valid candidate pool.

    Vectorized form of `_teacher_student_distribution_metrics_reference`; the
    two are pinned together by tests/test_stage1_metric_vectorization.py. Each
    query is renormalized over its own valid candidates, so masked entries are
    zeroed before every sum rather than being allowed to leak in.
    """
    if teacher_prob.dim() != 2:
        raise ValueError(
            f'teacher_prob must be [B, N], got {tuple(teacher_prob.shape)}'
        )
    if teacher_prob.shape != student_prob.shape or teacher_prob.shape != valid_mask.shape:
        raise ValueError(
            'teacher_prob, student_prob, and valid_mask must have the same shape'
        )

    valid_mask = valid_mask.bool()
    counts = valid_mask.sum(dim=-1)
    active = counts > 0
    zero = teacher_prob.detach().sum() * 0.0
    if not bool(active.any()):
        return {key: zero for key in _DISTRIBUTION_KEYS}

    teacher = teacher_prob.detach().float()
    student = student_prob.detach().float()
    if not torch.isfinite(teacher[valid_mask]).all() or not torch.isfinite(student[valid_mask]).all():
        raise ValueError('teacher/student distribution contains NaN or Inf')
    if (teacher[valid_mask] < 0).any() or (student[valid_mask] < 0).any():
        raise ValueError('teacher/student distribution contains a negative probability')

    teacher = teacher.masked_fill(~valid_mask, 0.0)
    student = student.masked_fill(~valid_mask, 0.0)
    teacher = teacher / teacher.sum(dim=-1, keepdim=True).clamp_min(float(eps))
    student = student / student.sum(dim=-1, keepdim=True).clamp_min(float(eps))

    def masked_sum(values):
        return (values * valid_mask).sum(dim=-1)

    teacher_log = torch.log(teacher.clamp_min(float(eps)))
    student_log = torch.log(student.clamp_min(float(eps)))
    midpoint_log = torch.log((0.5 * (teacher + student)).clamp_min(float(eps)))

    teacher_entropy = -masked_sum(teacher * teacher_log)
    student_entropy = -masked_sum(student * student_log)
    l1 = masked_sum(torch.abs(teacher - student))

    out = {
        'teacher_student_kl_divergence': masked_sum(teacher * (teacher_log - student_log)),
        'student_teacher_kl_divergence': masked_sum(student * (student_log - teacher_log)),
        'teacher_student_js_divergence': 0.5 * (
            masked_sum(teacher * (teacher_log - midpoint_log))
            + masked_sum(student * (student_log - midpoint_log))
        ),
        'teacher_student_prob_l1': l1,
        'teacher_student_total_variation': 0.5 * l1,
        'teacher_student_hellinger_distance': torch.sqrt(
            (0.5 * masked_sum((torch.sqrt(teacher) - torch.sqrt(student)).square())).clamp_min(0.0)
        ),
        'teacher_student_probability_cosine': F.cosine_similarity(
            teacher, student, dim=-1, eps=float(eps)
        ),
        'teacher_student_entropy_gap': student_entropy - teacher_entropy,
        'teacher_student_entropy_abs_gap': torch.abs(student_entropy - teacher_entropy),
    }

    count_f = counts.clamp_min(1).float()
    teacher_rank = _masked_ranks(teacher, valid_mask, counts)
    student_rank = _masked_ranks(student, valid_mask, counts)
    teacher_rank = (teacher_rank - masked_sum(teacher_rank).unsqueeze(1) / count_f.unsqueeze(1)) * valid_mask
    student_rank = (student_rank - masked_sum(student_rank).unsqueeze(1) / count_f.unsqueeze(1)) * valid_mask
    rank_denominator = torch.sqrt(
        teacher_rank.square().sum(dim=-1) * student_rank.square().sum(dim=-1)
    ).clamp_min(float(eps))
    spearman = (teacher_rank * student_rank).sum(dim=-1) / rank_denominator
    out['student_teacher_spearman'] = torch.where(
        counts > 1, spearman, torch.zeros_like(spearman)
    )

    num_cand = teacher.size(-1)
    positions = torch.arange(num_cand, device=teacher.device).unsqueeze(0)
    teacher_pick = teacher.masked_fill(~valid_mask, float('-inf'))
    student_pick = student.masked_fill(~valid_mask, float('-inf'))
    for k in (1, 5, 10):
        width = min(k, num_cand)
        effective_k = counts.clamp_max(width)
        keep = positions[:, :width] < effective_k.unsqueeze(1)
        teacher_topk = stable_topk_indices(teacher_pick, width, largest=True)
        student_topk = stable_topk_indices(student_pick, width, largest=True)
        pair = teacher_topk.unsqueeze(-1) == student_topk.unsqueeze(-2)
        pair = pair & keep.unsqueeze(-1) & keep.unsqueeze(-2)
        overlap = pair.any(dim=-1).float().sum(dim=-1) / effective_k.clamp_min(1).float()
        out[f'teacher_student_topk_overlap_at_{k}'] = overlap
        # With equal-size Top-K sets, precision and recall are both overlap/K.
        out[f'student_teacher_recall_at_{k}'] = overlap

    return {key: value[active].mean().detach() for key, value in out.items()}


@torch.no_grad()
def _ranking_source_topk_metrics_reference(
    student_scores,
    teacher_scores,
    future_mse,
    future_cosine,
    valid_mask,
):
    """Per-query reference for `_ranking_source_topk_metrics`."""
    tensors = {
        'student': student_scores,
        'teacher': teacher_scores,
        'oracle_mse': -future_mse,
        'oracle_cos': future_cosine,
    }
    expected_shape = valid_mask.shape
    if any(value.shape != expected_shape for value in tensors.values()):
        shapes = {key: tuple(value.shape) for key, value in tensors.items()}
        raise ValueError(
            f'all ranking sources and valid_mask must have shape {tuple(expected_shape)}; '
            f'got {shapes}'
        )

    pairs = (
        ('teacher', 'student'),
        ('oracle_mse', 'student'),
        ('teacher', 'oracle_mse'),
        ('oracle_cos', 'student'),
        ('teacher', 'oracle_cos'),
        ('oracle_mse', 'oracle_cos'),
    )
    rows = []
    valid_mask = valid_mask.bool()
    for query_idx in range(valid_mask.size(0)):
        row_mask = valid_mask[query_idx]
        count = int(row_mask.sum())
        if count == 0:
            continue
        row_scores = {
            key: value[query_idx, row_mask].detach().float()
            for key, value in tensors.items()
        }
        if any(not torch.isfinite(value).all() for value in row_scores.values()):
            raise ValueError('ranking source contains NaN or Inf at a valid candidate')

        row = {}
        for k in (1, 5, 10):
            effective_k = min(k, count)
            topk = {
                key: stable_topk_indices(value, effective_k, largest=True)
                for key, value in row_scores.items()
            }
            for first, second in pairs:
                overlap = (
                    topk[first][:, None] == topk[second][None, :]
                ).any(dim=1).float().mean()
                row[f'{first}_{second}_topk_overlap_at_{k}'] = overlap
        rows.append(row)

    keys = [
        f'{first}_{second}_topk_overlap_at_{k}'
        for first, second in pairs
        for k in (1, 5, 10)
    ]
    if not rows:
        zero = student_scores.detach().sum() * 0.0
        metrics = {key: zero for key in keys}
    else:
        metrics = {
            key: torch.stack([row[key] for row in rows]).mean().detach()
            for key in keys
        }
    for k in (1, 5, 10):
        metrics[f'student_oracle_recall_at_{k}'] = metrics[
            f'oracle_mse_student_topk_overlap_at_{k}'
        ]
        metrics[f'student_oracle_cos_recall_at_{k}'] = metrics[
            f'oracle_cos_student_topk_overlap_at_{k}'
        ]
    return metrics


_RANKING_SOURCE_PAIRS = (
    ('teacher', 'student'),
    ('oracle_mse', 'student'),
    ('teacher', 'oracle_mse'),
    ('oracle_cos', 'student'),
    ('teacher', 'oracle_cos'),
    ('oracle_mse', 'oracle_cos'),
)


@torch.no_grad()
def _ranking_source_topk_metrics(
    student_scores,
    teacher_scores,
    future_mse,
    future_cosine,
    valid_mask,
):
    """Pairwise Top-K overlap among Student, Teacher, and two future Oracles.

    Vectorized form of `_ranking_source_topk_metrics_reference`; the two are
    pinned together by tests/test_stage1_metric_vectorization.py.
    """
    tensors = {
        'student': student_scores,
        'teacher': teacher_scores,
        'oracle_mse': -future_mse,
        'oracle_cos': future_cosine,
    }
    expected_shape = valid_mask.shape
    if any(value.shape != expected_shape for value in tensors.values()):
        shapes = {key: tuple(value.shape) for key, value in tensors.items()}
        raise ValueError(
            f'all ranking sources and valid_mask must have shape {tuple(expected_shape)}; '
            f'got {shapes}'
        )

    valid_mask = valid_mask.bool()
    counts = valid_mask.sum(dim=-1)
    active = counts > 0
    keys = [
        f'{first}_{second}_topk_overlap_at_{k}'
        for first, second in _RANKING_SOURCE_PAIRS
        for k in (1, 5, 10)
    ]

    def finish(metrics):
        for k in (1, 5, 10):
            metrics[f'student_oracle_recall_at_{k}'] = metrics[
                f'oracle_mse_student_topk_overlap_at_{k}'
            ]
            metrics[f'student_oracle_cos_recall_at_{k}'] = metrics[
                f'oracle_cos_student_topk_overlap_at_{k}'
            ]
        return metrics

    zero = student_scores.detach().sum() * 0.0
    if not bool(active.any()):
        return finish({key: zero for key in keys})

    ranked = {}
    for key, value in tensors.items():
        value = value.detach().float()
        if not torch.isfinite(value[valid_mask]).all():
            raise ValueError('ranking source contains NaN or Inf at a valid candidate')
        ranked[key] = value.masked_fill(~valid_mask, float('-inf'))

    num_cand = valid_mask.size(-1)
    positions = torch.arange(num_cand, device=valid_mask.device).unsqueeze(0)
    out = {}
    for k in (1, 5, 10):
        width = min(k, num_cand)
        effective_k = counts.clamp_max(width)
        keep = positions[:, :width] < effective_k.unsqueeze(1)
        denom = effective_k.clamp_min(1).float()
        topk = {
            key: stable_topk_indices(value, width, largest=True)
            for key, value in ranked.items()
        }
        for first, second in _RANKING_SOURCE_PAIRS:
            pair = topk[first].unsqueeze(-1) == topk[second].unsqueeze(-2)
            pair = pair & keep.unsqueeze(-1) & keep.unsqueeze(-2)
            out[f'{first}_{second}_topk_overlap_at_{k}'] = (
                pair.any(dim=-1).float().sum(dim=-1) / denom
            )

    return finish({
        key: value[active].mean().detach() for key, value in out.items()
    })


@torch.no_grad()
def relation_bank_collapse_metrics(
    key_bank,
    sample_size=256,
    dead_std_threshold=1e-3,
    eps=1e-8,
):
    """Measure representation collapse per relation, then aggregate relations."""
    if key_bank.dim() != 4:
        raise ValueError(
            f'key_bank must be [C, S, N, D], got {tuple(key_bank.shape)}'
        )
    if int(sample_size) < 2:
        raise ValueError('collapse metric sample_size must be at least 2')
    if float(dead_std_threshold) < 0.0:
        raise ValueError('collapse dead-dimension threshold must be non-negative')

    _, _, num_candidates, embedding_dim = key_bank.shape
    if num_candidates < 2:
        raise ValueError('collapse metrics require at least two candidate embeddings')

    count = min(int(sample_size), int(num_candidates))
    sample_indices = torch.linspace(
        0,
        num_candidates - 1,
        steps=count,
        device=key_bank.device,
    ).round().long()
    rows = []

    for target_channel in range(key_bank.size(0)):
        for source_slot in range(key_bank.size(1)):
            embeddings = key_bank[target_channel, source_slot].index_select(
                0, sample_indices
            ).float()
            embeddings = F.normalize(embeddings, dim=-1, eps=float(eps))

            similarity = torch.matmul(embeddings, embeddings.transpose(0, 1))
            diagonal = similarity.diagonal()
            pair_count = count * (count - 1)
            pair_sum = similarity.sum() - diagonal.sum()
            pair_square_sum = similarity.square().sum() - diagonal.square().sum()
            pair_mean = pair_sum / pair_count
            pair_variance = (
                pair_square_sum / pair_count - pair_mean.square()
            ).clamp_min(0.0)

            centered = embeddings - embeddings.mean(dim=0, keepdim=True)
            dimension_variance = centered.square().mean(dim=0)
            dimension_std = torch.sqrt(dimension_variance.clamp_min(0.0))
            total_variance = dimension_variance.sum()
            dead_fraction = (
                dimension_std < float(dead_std_threshold)
            ).float().mean()

            covariance = torch.matmul(centered.transpose(0, 1), centered)
            covariance = covariance / max(count - 1, 1)
            eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0.0)
            eigenvalue_sum = eigenvalues.sum()
            if eigenvalue_sum <= float(eps):
                effective_rank = eigenvalue_sum * 0.0
                top_eigenvalue_ratio = eigenvalue_sum * 0.0 + 1.0
            else:
                spectrum = eigenvalues / eigenvalue_sum
                effective_rank = torch.exp(
                    -(spectrum * torch.log(spectrum.clamp_min(float(eps)))).sum()
                )
                top_eigenvalue_ratio = eigenvalues.max() / eigenvalue_sum

            rows.append({
                'pairwise_cosine_mean': pair_mean,
                'pairwise_cosine_std': torch.sqrt(pair_variance),
                'embedding_variance': total_variance,
                'dimension_std_mean': dimension_std.mean(),
                'dead_dimension_fraction': dead_fraction,
                'effective_rank': effective_rank,
                'effective_rank_ratio': effective_rank / float(embedding_dim),
                'top_eigenvalue_ratio': top_eigenvalue_ratio,
            })

    stacked = {
        key: torch.stack([row[key] for row in rows])
        for key in rows[0]
    }
    return {
        'pairwise_cosine_mean': stacked['pairwise_cosine_mean'].mean(),
        'pairwise_cosine_mean_max': stacked['pairwise_cosine_mean'].max(),
        'pairwise_cosine_std': stacked['pairwise_cosine_std'].mean(),
        'embedding_variance_mean': stacked['embedding_variance'].mean(),
        'embedding_variance_min': stacked['embedding_variance'].min(),
        'dimension_std_mean': stacked['dimension_std_mean'].mean(),
        'dead_dimension_fraction_mean': stacked['dead_dimension_fraction'].mean(),
        'dead_dimension_fraction_max': stacked['dead_dimension_fraction'].max(),
        'effective_rank_mean': stacked['effective_rank'].mean(),
        'effective_rank_min': stacked['effective_rank'].min(),
        'effective_rank_ratio_mean': stacked['effective_rank_ratio'].mean(),
        'effective_rank_ratio_min': stacked['effective_rank_ratio'].min(),
        'top_eigenvalue_ratio_mean': stacked['top_eigenvalue_ratio'].mean(),
        'top_eigenvalue_ratio_max': stacked['top_eigenvalue_ratio'].max(),
    }


def relation_variance_covariance_loss(
    embeddings,
    variance_target=1.0,
    eps=1e-4,
):
    """VICReg-style anti-collapse losses for one relation's online batch."""
    if embeddings.dim() != 2:
        raise ValueError(
            'relation embeddings must be [B, D], '
            f'got {tuple(embeddings.shape)}'
        )
    if float(variance_target) <= 0.0:
        raise ValueError('variance_target must be positive')

    dimension_variance = embeddings.var(dim=0, unbiased=False)
    dimension_std = torch.sqrt(dimension_variance + float(eps))
    variance_loss = F.relu(float(variance_target) - dimension_std).mean()

    centered = embeddings - embeddings.mean(dim=0, keepdim=True)
    covariance = torch.matmul(centered.transpose(0, 1), centered)
    covariance = covariance / max(embeddings.size(0) - 1, 1)
    covariance_off_diagonal = covariance - torch.diag_embed(
        covariance.diagonal()
    )
    covariance_loss = covariance_off_diagonal.square().sum() / embeddings.size(1)
    return variance_loss, covariance_loss, dimension_std.mean()


def _empty_rank_metrics(zero, dtype, device):
    value = zero.detach().to(device=device, dtype=dtype)
    return {
        'stage1_loss_rank': value,
        'rank_teacher_student_topk_overlap': value,
        'rank_teacher_student_topk_overlap_count': value,
        'rank_missed_positive_count': value,
        'rank_hard_negative_count': value,
        'rank_valid_pair_count': value,
        'rank_pair_accuracy': value,
        'rank_score_gap': value,
        'rank_margin_satisfied_ratio': value,
        'rank_teacher_topk_future_mse': value,
        'rank_student_topk_future_mse': value,
        'rank_missed_positive_future_mse': value,
        'rank_hard_negative_future_mse': value,
    }


def first_order_difference(x):
    """First-order difference along the relation sequence axis."""
    if x.size(-2) < 2:
        raise ValueError('diff1 requires a relation input with at least two time steps')
    return x[..., 1:, :] - x[..., :-1, :]


# A relation input space is an ordered tuple of single-feature transforms. Every
# legacy space is a 1-tuple and keeps its exact previous behaviour; multi-feature
# spaces stack their transforms as separate encoder input channels.
RELATION_INPUT_FEATURES = {
    'absolute': ('absolute',),
    'delta_last': ('delta_last',),
    'diff1': ('diff1',),
    'delta_last_diff1': ('delta_last', 'diff1'),
}


def relation_input_features(relation_input_space):
    try:
        return RELATION_INPUT_FEATURES[relation_input_space]
    except KeyError:
        raise ValueError(f'Unsupported relation_input_space: {relation_input_space}') from None


def relation_feature_count(relation_input_space):
    """Number of encoder input channels one relation role contributes."""
    return len(relation_input_features(relation_input_space))


def _feature_sequence_length(seq_len, feature):
    if feature == 'diff1':
        if seq_len < 2:
            raise ValueError('relation_input_space=diff1 requires seq_len >= 2')
        return seq_len - 1
    if feature in ('absolute', 'delta_last'):
        return seq_len
    raise ValueError(f'Unsupported relation input feature: {feature}')


def relation_sequence_length(seq_len, relation_input_space):
    """Common relation length across the space's features.

    Features of different natural length (delta_last is L, diff1 is L-1) are
    aligned by keeping their last `min` steps, so nothing synthetic is padded in
    and the length convention of the single-feature spaces is unchanged.
    """
    seq_len = int(seq_len)
    return min(
        _feature_sequence_length(seq_len, feature)
        for feature in relation_input_features(relation_input_space)
    )


def _transform_relation_feature(x, feature):
    if feature == 'absolute':
        return x
    if feature == 'delta_last':
        return x - x[:, -1:, :].detach()
    if feature == 'diff1':
        return first_order_difference(x)
    raise ValueError(f'Unsupported relation input feature: {feature}')


def transform_relation_features(x, relation_input_space):
    """Transform [B, L, C] histories into a list of [B, L', C] feature views.

    All views are cropped to the same trailing length so they can be stacked as
    encoder input channels.
    """
    features = relation_input_features(relation_input_space)
    views = [_transform_relation_feature(x, feature) for feature in features]
    length = min(view.size(-2) for view in views)
    return [view[..., -length:, :] for view in views]


def transform_relation_history(x, relation_input_space):
    """Transform [B, L, C] histories without changing their channel layout.

    Single-feature spaces only; multi-feature spaces have no single [B, L', C]
    view, so callers that need them must use transform_relation_features.
    """
    if relation_feature_count(relation_input_space) != 1:
        raise ValueError(
            f'relation_input_space={relation_input_space} has '
            f'{relation_feature_count(relation_input_space)} features and cannot be '
            'expressed as a single [B, L, C] history; use transform_relation_features'
        )
    return _transform_relation_feature(x, relation_input_features(relation_input_space)[0])


class RelationEncoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.encoder_type = getattr(configs, 'relation_encoder_type', 'transformer')
        self.pooling = getattr(configs, 'relation_pooling', 'cls')
        self.self_fill = getattr(configs, 'relation_self_fill', 'linear')
        # With retrieval_similarity=l2 the encoder must hand back raw embeddings:
        # -||q-k||^2 on normalised vectors is a monotone map of the dot product,
        # so normalising here would make l2 and cosine rank candidates the same.
        # Stage-1 and Stage-2 both read this, which is what keeps the metric the
        # encoder is trained under equal to the one it is retrieved with.
        self.retrieval_similarity = getattr(configs, 'retrieval_similarity', 'cosine')
        if self.retrieval_similarity not in ('cosine', 'l2'):
            raise ValueError(
                f'Unsupported retrieval_similarity: {self.retrieval_similarity}'
            )
        self.relation_input_space = getattr(
            configs, 'relation_input_space', 'absolute'
        )
        # Every relation role contributes n_features encoder input channels, so a
        # relation input is [B, R * F, L] with the rows ordered role-major:
        # target features first, then the optional source features.
        self.n_features = relation_feature_count(self.relation_input_space)
        self.seq_len = relation_sequence_length(
            configs.seq_len, self.relation_input_space
        )
        self.d_model = configs.d_model

        if self.encoder_type == 'transformer':
            if self.n_features != 1:
                raise ValueError(
                    'relation_encoder_type=transformer only supports single-feature '
                    f'relation input spaces, got {self.relation_input_space}; '
                    'use relation_encoder_type=tcn or mlp for multi-feature input'
                )
            if self.pooling not in ('cls', 'mean'):
                raise ValueError(f'Unsupported relation_pooling for transformer: {self.pooling}')
            if self.self_fill == 'linear':
                raise ValueError('relation_self_fill=linear is only supported by the MLP relation encoder')
            self.patch_embed = RelationPatchEmbedding(
                seq_len=self.seq_len,
                patch_len=configs.patch_len,
                stride=configs.stride,
                d_model=configs.d_model,
                dropout=configs.dropout,
            )
            self.cls_token = nn.Parameter(torch.zeros(1, 1, configs.d_model))
            layer = nn.TransformerEncoderLayer(
                d_model=configs.d_model,
                nhead=configs.n_heads,
                dim_feedforward=configs.d_ff,
                dropout=configs.dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=configs.e_layers)
        elif self.encoder_type == 'mlp':
            if self.pooling != 'cls':
                raise ValueError('relation_pooling is only configurable for transformer encoder')
            if self.self_fill not in ('zero', 'repeat', 'linear'):
                raise ValueError(f'Unsupported relation_self_fill for mlp: {self.self_fill}')
            rows = self.n_features if self.self_fill == 'linear' else 2 * self.n_features
            input_dim = rows * self.seq_len
            if self.self_fill == 'linear':
                self.role_embedding = None
            else:
                self.role_embedding = nn.Parameter(
                    torch.zeros(1, 2 * self.n_features, self.seq_len)
                )
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, configs.d_ff),
                nn.GELU(),
                nn.Dropout(configs.dropout),
                nn.Linear(configs.d_ff, configs.d_model),
            )
        elif self.encoder_type == 'tcn':
            if self.pooling not in ('last', 'mean'):
                raise ValueError(
                    'relation_pooling for tcn must be last or mean, got '
                    f'{self.pooling}'
                )
            if self.self_fill not in ('zero', 'repeat', 'linear'):
                raise ValueError(f'Unsupported relation_self_fill for tcn: {self.self_fill}')
            in_channels = (
                self.n_features if self.self_fill == 'linear' else 2 * self.n_features
            )
            # The conv input channel index already separates target from source,
            # so the TCN needs no additive role embedding.
            self.role_embedding = None
            hidden = int(getattr(configs, 'relation_tcn_channels', 0)) or configs.d_model
            tcn_dropout = getattr(configs, 'relation_tcn_dropout', None)
            if tcn_dropout is None or float(tcn_dropout) < 0:
                tcn_dropout = configs.dropout
            self.encoder = RelationTCN(
                in_channels=in_channels,
                hidden_channels=hidden,
                out_channels=configs.d_model,
                num_layers=int(getattr(configs, 'relation_tcn_layers', 4)),
                kernel_size=int(getattr(configs, 'relation_tcn_kernel_size', 3)),
                dropout=float(tcn_dropout),
                pooling=self.pooling,
            )
        else:
            raise ValueError(f'Unsupported relation_encoder_type: {self.encoder_type}')

        self.norm = nn.LayerNorm(configs.d_model)
        self.proj = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.GELU(),
            nn.Linear(configs.d_model, configs.d_model),
        )

    def _prepare_rows(self, relation_x):
        """Validate [B, R*F, L] and apply the self_fill policy.

        Returns [B, F, L] for self_fill=linear (the cross branch is already
        mixed by the shared 2L->L projection) and [B, 2F, L] otherwise, where
        the second role block is zero-filled or repeated for self relations.
        """
        features = self.n_features
        if relation_x.dim() != 3:
            raise ValueError(
                f'relation input must be [B, R*F, L], got {tuple(relation_x.shape)}'
            )
        bsz, rows, seq_len = relation_x.shape
        if rows not in (features, 2 * features):
            raise ValueError(
                f'relation row count must be {features} (self) or {2 * features} '
                f'(cross) for {features}-feature input, got {rows}'
            )
        if seq_len != self.seq_len:
            raise ValueError(f'expected seq_len={self.seq_len}, got {seq_len}')
        if self.self_fill == 'linear':
            if rows != features:
                raise ValueError(
                    f'relation_self_fill=linear expects [B, {features}, L] from either '
                    'a direct self input or the shared cross 2L->L projection, '
                    f'got {tuple(relation_x.shape)}'
                )
            return relation_x
        padded = relation_x.new_zeros(bsz, 2 * features, self.seq_len)
        padded[:, :rows] = relation_x
        if rows == features and self.self_fill == 'repeat':
            padded[:, features:] = relation_x
        if self.role_embedding is None:
            return padded
        return padded + self.role_embedding

    def forward(self, relation_x, return_pre_normalized=False):
        if self.encoder_type == 'transformer':
            tokens = self.patch_embed(relation_x)
            if self.pooling == 'cls':
                cls = self.cls_token.expand(tokens.size(0), -1, -1)
                out = self.encoder(torch.cat([cls, tokens], dim=1))
                h = out[:, 0]
            else:
                out = self.encoder(tokens)
                h = out.mean(dim=1)
        else:
            rows = self._prepare_rows(relation_x)
            if self.encoder_type == 'tcn':
                h = self.encoder(rows)
            else:
                h = self.encoder(rows.reshape(rows.size(0), -1))
        z = self.proj(self.norm(h))
        # l2 scoring needs the norm kept, so the "normalized" slot carries the raw
        # embedding instead. Callers that only want a key/query vector stay
        # unchanged; the ones that asked for the pre-normalised copy still get z.
        normalized = z if self.retrieval_similarity == 'l2' else F.normalize(z, dim=-1)
        if return_pre_normalized:
            return normalized, z
        return normalized


def build_relation_encoder_input(
    x,
    target_channel,
    source_channel,
    relation_input_space='absolute',
    shared_cross_projection=None,
    self_fill='linear',
):
    """Build the Stage-1/Stage-2 relation input sent to RelationEncoder.

    The result is [B, R*F, L], role-major: the F feature rows of the target
    first, then the F feature rows of the source. F is 1 for the single-feature
    spaces, so this is the previous [B, R, L] layout unchanged.

    Self relation uses the target series directly. Cross relation depends on
    self_fill, which is also what sets the encoder input width:

    linear (F rows)
        [target, source] goes through the shared 2L -> L projection, applied
        per feature with the same weights, so the encoder sees only the target-
        width rows and the channels are already mixed for it.
    repeat / zero (2F rows)
        target and source stay as raw rows and the encoder's first layer does
        the mixing. Self relation still returns F rows; RelationEncoder fills
        the source block per the self_fill policy.
    """
    features = transform_relation_features(x, relation_input_space)
    target = torch.stack([view[..., target_channel] for view in features], dim=1)

    if source_channel == target_channel:
        return target

    source = torch.stack([view[..., source_channel] for view in features], dim=1)

    if self_fill != 'linear':
        return torch.cat([target, source], dim=1)

    if shared_cross_projection is None:
        raise RuntimeError(
            'Cross-relation input requires shared_cross_projection. '
            'Use a Stage-1 checkpoint produced with shared_cross_projection.'
        )
    # nn.Linear applies over the last axis, so one 2L -> L projection is shared
    # across the feature rows and its weight shape stays independent of F.
    return shared_cross_projection(torch.cat([target, source], dim=-1))


def build_direct_relation_embedding(
    x,
    target_channel,
    source_channel,
    relation_input_space='absolute',
    eps=1e-8,
):
    """Encoder-free relation vector used by the direct cosine baseline.

    Self relations duplicate the target role so every relation has one fixed
    width. This does not alter self-relation cosine similarity and makes the
    direct baseline comparable to cross relations built as [target || source].
    Multi-feature spaces concatenate their feature views in order, which keeps
    the baseline defined on exactly the input the encoder arms receive.
    """
    features = transform_relation_features(x, relation_input_space)
    parts = [view[..., target_channel] for view in features]
    parts += [view[..., source_channel] for view in features]
    return F.normalize(torch.cat(parts, dim=-1), dim=-1, eps=eps)


class Model(nn.Module):
    """Stage-1 relation-wise retrieval encoder.

    Inputs are normalized sliding windows:
      query_x: [B, L, C], query_y: [B, H, C]
      memory_y: [N, H, C], cand_mask: [B, N]
    EMA teachers either preserve the legacy future-relation target or encode
    the same past relation input as the student (ema_input).
    The student branch uses an epoch-refreshed relation key memory bank.
    """

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.tau_student = float(configs.tau_student)
        self.tau_teacher = float(configs.tau_teacher)
        self.teacher_mse_space = configs.teacher_mse_space
        self.teacher_mode = getattr(configs, 'stage1_teacher_mode', 'mse')
        self.relation_input_space = getattr(configs, 'relation_input_space', 'absolute')
        self.relation_seq_len = relation_sequence_length(
            self.seq_len, self.relation_input_space
        )
        self.relation_teacher_space = getattr(configs, 'relation_teacher_space', 'absolute')
        self.source_mode = configs.source_mode
        self.relation_graph_threshold = int(getattr(configs, 'relation_graph_threshold', 21))
        self.target_mode = configs.target_mode
        self.target_channel = configs.target_channel
        self.key_chunk_size = int(getattr(configs, 'stage1_key_chunk_size', 1024))
        # Learnable pair score. Cosine correlates with future-MSE at rho 0.61 over
        # the whole bank and 0.03 inside the model's own Top-100 -- coarse
        # retrieval works, fine ordering has no signal left in that space. A
        # pair-conditioned score tests whether that is the score function's limit.
        # Learnable but still indexable retrieval metric. Unlike a pair MLP these
        # stay bilinear in the two embeddings, so a whole bank scores as one
        # matmul and training can use the same candidate support as evaluation.
        self.retrieval_metric_kind = getattr(configs, 'stage1_retrieval_metric', 'cosine')
        if self.retrieval_metric_kind not in RETRIEVAL_METRICS:
            raise ValueError(
                f'Unsupported stage1_retrieval_metric: {self.retrieval_metric_kind}')
        self.retrieval_metric = (
            RetrievalMetric(
                kind=self.retrieval_metric_kind,
                dim=int(configs.d_model),
                scaled_dot=bool(int(getattr(configs, 'stage1_metric_scaled_dot', 1))),
                layer_norm=bool(int(getattr(configs, 'stage1_metric_layer_norm', 1))),
                output=getattr(configs, 'stage1_metric_output', 'dot'),
            )
            if self.retrieval_metric_kind != 'cosine' else None
        )
        # A learned kind is only comparable to the cosine baseline if it *starts*
        # there; otherwise an epoch-1 difference is the initialisation, not the
        # learning. The projections initialise to identity, but that alone is not
        # enough -- an unnormalised output or an affine LayerNorm in the path
        # still changes the ranking at step 0 -- so measure it rather than assume
        # it. Reported, not enforced: `output='dot'` runs are deliberate.
        if self.retrieval_metric is not None:
            deviation = cosine_init_deviation(self.retrieval_metric)
            self.metric_cosine_init_deviation = deviation
            note = 'matches cosine' if deviation < 1e-5 else (
                'DOES NOT match cosine at init -- arms differ before training')
            print(f'[stage1][metric] kind={self.retrieval_metric_kind} '
                  f'cosine_init_deviation={deviation:.3e} ({note})')
        else:
            self.metric_cosine_init_deviation = 0.0
        # How the candidate side receives gradient while the denominator stays
        # the full memory. 'bank' scores everything from the detached bank, so
        # only the query branch trains; 'selected_reencode' re-encodes a chosen
        # subset and scatters those scores back into the full logits.
        self.full_memory_gradient_mode = getattr(
            configs, 'stage1_full_memory_gradient_mode', 'bank')
        if self.full_memory_gradient_mode not in ('bank', 'selected_reencode', 'full_online'):
            raise ValueError(
                'stage1_full_memory_gradient_mode must be bank, selected_reencode '
                f'or full_online; got {self.full_memory_gradient_mode}')
        self.full_memory_hard_negatives = int(
            getattr(configs, 'stage1_full_memory_hard_negatives', 100))
        self.full_memory_random_negatives = int(
            getattr(configs, 'stage1_full_memory_random_negatives', 128))
        self.retrieval_score = getattr(configs, 'stage1_retrieval_score', 'cosine')
        if self.retrieval_score not in ('cosine', 'pairwise_mlp'):
            raise ValueError(
                f'Unsupported stage1_retrieval_score: {self.retrieval_score}')
        self.pairwise_feature = getattr(configs, 'stage1_pairwise_feature', 'pair4')
        if self.retrieval_score == 'pairwise_mlp':
            subset_mode = getattr(configs, 'stage1_candidate_subset_mode', 'none')
            # A pair scorer needs candidate embeddings in the graph. Reading them
            # from the detached bank would train the scorer against a candidate
            # side that never receives gradient, which is the exact failure this
            # experiment is meant to remove. Two ways to get that: mine a subset
            # and re-encode it, or re-encode the whole memory.
            #
            # Mining used to be the only option here because materialising every
            # pair looked prohibitive -- a pair feature is 2-4x the embedding
            # width and cannot be folded into a matmul. That was an estimate, not
            # a measurement: at N=8449 it is ~2.4 GiB per target channel, which
            # the card has. Requiring mining is what forced the earlier runs to
            # train on 228 candidates and be evaluated over 8449, so a loss there
            # could not be told apart from the support mismatch causing it.
            if subset_mode != 'selected_reencode' and (
                    self.full_memory_gradient_mode != 'full_online'):
                raise ValueError(
                    'stage1_retrieval_score=pairwise_mlp needs the candidate side '
                    'in the graph: use --stage1_candidate_subset_mode '
                    'selected_reencode, or --stage1_full_memory_gradient_mode '
                    f'full_online to score the whole memory; got subset_mode='
                    f'{subset_mode} and full_memory_gradient_mode='
                    f'{self.full_memory_gradient_mode}'
                )
        self.pairwise_scorer = None
        if self.retrieval_score == 'pairwise_mlp':
            self.pairwise_scorer = PairwiseScorer(
                embedding_dim=int(configs.d_model),
                feature_type=self.pairwise_feature,
                hidden_dim=int(getattr(configs, 'stage1_pairwise_hidden', 256)),
                hidden_dim2=int(getattr(configs, 'stage1_pairwise_hidden2', 128)),
                dropout=float(getattr(configs, 'stage1_pairwise_dropout', 0.1)),
            )
        requested_loss_mode = getattr(configs, 'stage1_loss_mode', 'kl')
        legacy_use_rank_loss = bool(int(getattr(configs, 'stage1_use_rank_loss', 0)))
        if requested_loss_mode not in (
            'kl', 'kl_infonce', 'kl_rank', 'rnc', 'kl_expected_mse',
            'topk_coverage', 'weighted_topk_ce'
        ):
            raise ValueError(f'Unsupported stage1_loss_mode: {requested_loss_mode}')
        # Preserve old rank scripts, which only set stage1_use_rank_loss=1.
        self.loss_mode = 'kl_rank' if requested_loss_mode == 'kl' and legacy_use_rank_loss else requested_loss_mode
        self.use_rank_loss = self.loss_mode == 'kl_rank'
        self.infonce_weight = float(
            getattr(configs, 'stage1_infonce_weight', 0.5)
        )
        requested_infonce_top_k = int(
            getattr(configs, 'stage1_infonce_top_k', -1)
        )
        self.infonce_top_k = (
            requested_infonce_top_k
            if requested_infonce_top_k > 0
            else int(getattr(configs, 'top_k', 10))
        )
        self.infonce_positive_source = getattr(
            configs, 'stage1_infonce_positive_source', 'target_mse'
        )
        self.rank_weight = float(getattr(configs, 'stage1_rank_weight', 0.1))
        self.rank_margin = float(getattr(configs, 'stage1_rank_margin', 0.1))
        self.rank_min_mse_gap = float(getattr(configs, 'stage1_rank_min_mse_gap', 0.0))
        self.rank_top_k = getattr(configs, 'stage1_rank_top_k', None)
        if self.rank_top_k is None or int(self.rank_top_k) <= 0:
            self.rank_top_k = int(getattr(configs, 'top_k', 10))
        else:
            self.rank_top_k = int(self.rank_top_k)
        requested_coverage_top_k = int(
            getattr(configs, 'stage1_coverage_top_k', -1)
        )
        self.coverage_top_k = (
            requested_coverage_top_k
            if requested_coverage_top_k > 0
            else int(getattr(configs, 'top_k', 10))
        )
        if self.coverage_top_k <= 0:
            raise ValueError('stage1_coverage_top_k must be positive after fallback')
        if not 0.0 <= self.infonce_weight <= 1.0:
            raise ValueError('stage1_infonce_weight must be between 0 and 1')
        if self.infonce_top_k <= 0:
            raise ValueError('stage1_infonce_top_k must be positive after fallback')
        if self.infonce_positive_source not in ('target_mse', 'ema_cosine'):
            raise ValueError(
                'stage1_infonce_positive_source must be target_mse or ema_cosine'
            )
        if (
            self.infonce_positive_source == 'ema_cosine'
            and self.teacher_mode != 'ema_target'
        ):
            raise ValueError(
                'stage1_infonce_positive_source=ema_cosine requires '
                'stage1_teacher_mode=ema_target'
            )
        self.rnc_temperature = float(getattr(configs, 'rnc_temperature', 0.2))
        self.rnc_tie_epsilon = float(getattr(configs, 'rnc_tie_epsilon', 0.0))
        self.rnc_quality_source = getattr(configs, 'rnc_quality_source', 'future_mse')
        self.expected_mse_weight = float(getattr(configs, 'expected_mse_weight', 0.1))
        self.expected_mse_normalization = getattr(
            configs, 'expected_mse_normalization', 'mean'
        )
        self.variance_weight = float(
            getattr(configs, 'stage1_variance_weight', 0.0)
        )
        self.covariance_weight = float(
            getattr(configs, 'stage1_covariance_weight', 0.0)
        )
        self.variance_target = float(
            getattr(configs, 'stage1_variance_target', 1.0)
        )
        if self.rnc_temperature <= 0.0:
            raise ValueError('stage1_rnc_temperature must be positive')
        if self.rnc_tie_epsilon < 0.0:
            raise ValueError('stage1_rnc_tie_epsilon must be non-negative')
        if self.rnc_quality_source not in ('future_mse', 'ema_cosine'):
            raise ValueError('rnc_quality_source must be future_mse or ema_cosine')
        if self.rnc_quality_source == 'ema_cosine' and self.teacher_mode != 'ema_target':
            raise ValueError('rnc_quality_source=ema_cosine requires stage1_teacher_mode=ema_target')
        if not 0.0 <= self.expected_mse_weight <= 1.0:
            raise ValueError('expected_mse_weight must be between 0 and 1')
        if self.expected_mse_normalization not in ('none', 'mean', 'median'):
            raise ValueError(
                'stage1_expected_mse_normalization must be one of: none, mean, median'
            )
        if self.variance_weight < 0.0:
            raise ValueError('stage1_variance_weight must be non-negative')
        if self.covariance_weight < 0.0:
            raise ValueError('stage1_covariance_weight must be non-negative')
        if self.variance_target <= 0.0:
            raise ValueError('stage1_variance_target must be positive')
        # Residual teacher over an arbitrary pool, including the full bank.
        # Scores are computed inline from cached base-forecast residuals rather
        # than read from a per-pool score matrix, which is what makes full-bank
        # residual supervision affordable at all.
        self.residual_teacher_active = bool(int(getattr(configs, 'stage1_residual_teacher', 0)))
        self.reference_pool_size = int(getattr(configs, 'stage1_pool_size', 0))

        # Externally measured retrieval teacher. The supervision signal moves
        # from "which candidate's future looks like mine" to "which candidate
        # actually improves the forecast", which cannot be computed inside this
        # loop -- it needs a Stage-2 forward -- so it arrives precomputed over a
        # fixed candidate pool shared by every teacher arm.
        self.external_teacher_target = getattr(configs, 'stage1_teacher_target', 'future')
        if self.external_teacher_target not in ('future', 'residual', 'utility'):
            raise ValueError(
                'stage1_teacher_target must be one of: future, residual, utility; '
                f'got {self.external_teacher_target}'
            )
        self.external_teacher_objective = getattr(configs, 'stage1_teacher_loss', 'kl')
        if self.external_teacher_objective not in ('kl', 'expected_utility'):
            raise ValueError(
                'stage1_teacher_loss must be kl or expected_utility; '
                f'got {self.external_teacher_objective}'
            )
        self.external_teacher_normalize = getattr(
            configs, 'stage1_teacher_normalize', 'per_query_scale'
        )
        self.external_teacher_tau = float(getattr(configs, 'stage1_teacher_tau', 0.05))
        if self.external_teacher_tau <= 0.0:
            raise ValueError('stage1_teacher_tau must be positive')
        # Explicit abstention. A softmax retriever always sums to one, so it must
        # pick someone even when every candidate in the pool is harmful -- which
        # the utility diagnostic showed is most queries at long horizons.
        self.null_mode = getattr(configs, 'stage1_null_mode', 'off')
        if self.null_mode not in ('off', 'fixed', 'query'):
            raise ValueError(
                f'stage1_null_mode must be off, fixed or query; got {self.null_mode}'
            )
        self.null_head = (
            nn.Linear(int(configs.d_model), 1) if self.null_mode == 'query' else None
        )

        # Training-only candidate subset. The memory bank keeps mining the whole
        # pool; these decide which mined candidates the loss runs over and
        # whether the candidate side carries gradient.
        self.candidate_subset_mode = getattr(
            configs, 'stage1_candidate_subset_mode', 'none'
        )
        if self.candidate_subset_mode not in (
            'none', 'selected_detached', 'selected_reencode'
        ):
            raise ValueError(
                'stage1_candidate_subset_mode must be one of: none, '
                f'selected_detached, selected_reencode; got {self.candidate_subset_mode}'
            )
        self.candidate_mine_top_m = int(
            getattr(configs, 'stage1_candidate_mine_top_m', 100)
        )
        # Random negatives drawn from the whole bank. A learned score is only
        # constrained where it has seen pairs, and evaluation ranks the full
        # memory; without them the scorer is unconstrained on 99% of it.
        self.candidate_random_negatives = int(
            getattr(configs, 'stage1_candidate_random_negatives', 0)
        )
        if self.candidate_random_negatives < 0:
            raise ValueError('stage1_candidate_random_negatives must be non-negative')
        requested_inject_k = int(
            getattr(configs, 'stage1_candidate_oracle_inject_k', -1)
        )
        self.candidate_oracle_inject_k = (
            requested_inject_k
            if requested_inject_k > 0
            else int(getattr(configs, 'top_k', 10))
        )
        if self.candidate_subset_mode != 'none':
            if self.candidate_mine_top_m <= 0:
                raise ValueError('stage1_candidate_mine_top_m must be positive')
            if self.candidate_oracle_inject_k > self.candidate_mine_top_m:
                raise ValueError(
                    'stage1_candidate_oracle_inject_k cannot exceed '
                    'stage1_candidate_mine_top_m'
                )
            # The experiment isolates the candidate-gradient effect, so the
            # objective is pinned to one of the two future-supervised losses:
            # the KL distillation the pipeline ships with, or the explicit
            # Oracle Top-K coverage loss the tiny-overfit diagnostic used.
            if self.loss_mode not in ('kl', 'topk_coverage', 'weighted_topk_ce'):
                raise ValueError(
                    'stage1_candidate_subset_mode requires --stage1_loss_mode kl, '
                    f'topk_coverage or weighted_topk_ce; got {self.loss_mode}'
                )
            if self.loss_mode == 'kl' and self.teacher_mode != 'mse':
                raise ValueError(
                    'stage1_candidate_subset_mode with kl requires '
                    f'--stage1_teacher_mode mse; got {self.teacher_mode}'
                )

        self.eps = 1e-8
        self.encoder = RelationEncoder(configs)
        self.shared_cross_projection = nn.Linear(
            2 * self.relation_seq_len, self.relation_seq_len
        )
        if self.teacher_mode not in ('mse', 'pearson', 'ema_target', 'ema_input'):
            raise ValueError(f'Unsupported stage1_teacher_mode: {self.teacher_mode}')
        if self.relation_teacher_space == 'delta_last' and self.teacher_mse_space == 'raw':
            raise ValueError(
                'relation_teacher_space=delta_last is only supported with '
                'teacher_mse_space=normalized because query_x/memory_x offsets are normalized'
            )
        if (
            self.teacher_mode == 'ema_target'
            and self.loss_mode != 'topk_coverage'
            and self.seq_len != self.pred_len
        ):
            raise ValueError(
                'stage1_teacher_mode=ema_target requires seq_len == pred_len '
                f'for shared EMA encoder shapes, got seq_len={self.seq_len}, pred_len={self.pred_len}'
            )
        if self.teacher_mode == 'ema_target' and self.encoder.n_features != 1:
            # ema_target pushes candidate futures through the shared encoder in
            # relation_teacher_space, which has no multi-feature counterpart.
            # ema_input encodes the same past input as the student and does.
            raise ValueError(
                f'relation_input_space={self.relation_input_space} is multi-feature and '
                'is not supported by stage1_teacher_mode=ema_target; use ema_input or mse'
            )
        self.teacher_encoder = copy.deepcopy(self.encoder)
        self.teacher_shared_cross_projection = copy.deepcopy(self.shared_cross_projection)
        for param in self.teacher_encoder.parameters():
            param.requires_grad = False
        for param in self.teacher_shared_cross_projection.parameters():
            param.requires_grad = False
        self._shape_logged = False
        self._subset_logged = False
        self.relation_sources = None

    def set_relation_graph(self, graph):
        if graph is None:
            self.relation_sources = None
            return
        sources = [[int(source) for source in row] for row in graph['sources']]
        if len(sources) != self.channels:
            raise ValueError('relation graph target count does not match model channels')
        self.relation_sources = sources

    def uses_sparse_relation_graph(self):
        return self.relation_sources is not None

    def requires_ema_teacher_bank(self):
        if self.loss_mode in ('topk_coverage', 'weighted_topk_ce'):
            return False
        if self.teacher_mode not in ('ema_target', 'ema_input'):
            return False
        return self.loss_mode != 'rnc' or self.rnc_quality_source == 'ema_cosine'

    def source_channels(self, target_channel):
        if self.relation_sources is not None:
            return self.relation_sources[int(target_channel)]
        if self.source_mode in ('auto', 'topk_corr'):
            raise RuntimeError('sparse source mode requires a loaded relation graph')
        return list(range(self.channels))

    def source_slot(self, target_channel, source_channel):
        sources = self.source_channels(target_channel)
        try:
            return sources.index(int(source_channel))
        except ValueError as exc:
            raise ValueError(
                f'source channel {source_channel} is not selected for target {target_channel}'
            ) from exc

    def target_channels(self):
        if self.target_mode == 'single':
            if self.target_channel is None:
                raise ValueError('target_mode=single requires --target_channel')
            return [int(self.target_channel)]
        if self.target_mode != 'all':
            raise ValueError(f'Unsupported target_mode: {self.target_mode}')
        return list(range(self.channels))

    def _relation_tensor(self, x, target_channel, source_channel):
        return build_relation_encoder_input(
            x,
            target_channel,
            source_channel,
            relation_input_space=self.relation_input_space,
            shared_cross_projection=self.shared_cross_projection,
            self_fill=self.encoder.self_fill,
        )

    def _relation_key_tensor(self, cand_x, target_channel, source_channel):
        bsz, num_cand, seq_len, _ = cand_x.shape
        flat = cand_x.reshape(bsz * num_cand, seq_len, -1)
        return self._relation_tensor(flat, target_channel, source_channel)

    def _future_distance_inputs(self, query_x, query_y, memory_y, memory_x_last, target_channel):
        q = query_y[:, :, target_channel]
        k = memory_y[:, :, target_channel].to(q.device)
        if self.teacher_mse_space not in ('normalized', 'raw'):
            raise ValueError(f'Unsupported teacher_mse_space: {self.teacher_mse_space}')
        if self.relation_teacher_space == 'delta_last':
            if memory_x_last is None:
                raise ValueError('relation_teacher_space=delta_last requires memory_x_last')
            q = q - query_x[:, -1:, target_channel].detach()
            k = k - memory_x_last[:, target_channel].to(q.device).unsqueeze(-1)
        return q, k

    def _relation_future_distance_inputs(
        self,
        query_x,
        query_y,
        memory_y,
        memory_x_last,
        target_channel,
        source_channel,
    ):
        q_target, k_target = self._future_distance_inputs(
            query_x, query_y, memory_y, memory_x_last, target_channel
        )
        q_source, k_source = self._future_distance_inputs(
            query_x, query_y, memory_y, memory_x_last, source_channel
        )
        return (
            torch.cat([q_target, q_source], dim=-1),
            torch.cat([k_target, k_source], dim=-1),
        )

    def _future_mse(
        self,
        query_x,
        query_y,
        memory_y,
        memory_x_last,
        target_channel,
        source_channel,
    ):
        q, k = self._relation_future_distance_inputs(
            query_x,
            query_y,
            memory_y,
            memory_x_last,
            target_channel,
            source_channel,
        )
        # Relation-aware future MSE over [target future || source future].
        # The self branch concatenates the target with itself, which preserves
        # the original target-only ordering while keeping one definition.
        q2 = (q ** 2).mean(dim=-1, keepdim=True)
        k2 = (k ** 2).mean(dim=-1).unsqueeze(0)
        qk = torch.matmul(q, k.transpose(0, 1)) / q.size(-1)
        return (q2 + k2 - 2.0 * qk).clamp_min(0.0)

    def _future_cosine(
        self,
        query_x,
        query_y,
        memory_y,
        memory_x_last,
        target_channel,
        source_channel,
    ):
        q, k = self._relation_future_distance_inputs(
            query_x,
            query_y,
            memory_y,
            memory_x_last,
            target_channel,
            source_channel,
        )
        q = F.normalize(q, dim=-1, eps=self.eps)
        k = F.normalize(k, dim=-1, eps=self.eps)
        return torch.matmul(q, k.transpose(0, 1))

    def _residual_mse(self, query_residual, memory_residual, target_channel):
        """Pairwise MSE between query and candidate base-forecast residuals.

        Same expansion `_future_mse` uses -- ||q||^2 + ||k||^2 - 2<q,k> over the
        horizon -- so the residual teacher costs one matmul against the whole bank
        and scales to any pool size, unlike measured utility.
        """
        q = query_residual[:, :, target_channel]
        k = memory_residual[:, :, target_channel]
        q2 = q.square().mean(dim=-1, keepdim=True)
        k2 = k.square().mean(dim=-1).unsqueeze(0)
        qk = torch.matmul(q, k.transpose(0, 1)) / q.size(-1)
        return (q2 + k2 - 2.0 * qk).clamp_min(0.0)

    def reference_pool(self, reference_scores, valid_mask, pool_size):
        """Top-M candidate ids under a frozen reference encoder.

        The pool has to be identical across teacher arms, so it comes from a
        checkpoint that never moves rather than from the encoder being trained --
        otherwise each arm would be scored on a pool it chose for itself.
        """
        if pool_size <= 0 or pool_size >= reference_scores.size(-1):
            return None
        masked = reference_scores.masked_fill(~valid_mask, torch.finfo(reference_scores.dtype).min)
        return masked.topk(pool_size, dim=-1).indices

    def _teacher_logits(
        self,
        query_x,
        query_y,
        memory_y,
        memory_x_last,
        target_channel,
        source_channel,
    ):
        q, k = self._relation_future_distance_inputs(
            query_x,
            query_y,
            memory_y,
            memory_x_last,
            target_channel,
            source_channel,
        )
        q2 = (q ** 2).mean(dim=-1, keepdim=True)
        k2 = (k ** 2).mean(dim=-1).unsqueeze(0)
        qk = torch.matmul(q, k.transpose(0, 1)) / q.size(-1)
        mse = (q2 + k2 - 2.0 * qk).clamp_min(0.0)
        if self.teacher_mode != 'pearson':
            return -mse / self.tau_teacher, mse

        q_centered = q - q.mean(dim=-1, keepdim=True)
        k_centered = k - k.mean(dim=-1, keepdim=True)
        q_var = (q_centered ** 2).mean(dim=-1, keepdim=True)
        k_var = (k_centered ** 2).mean(dim=-1).unsqueeze(0)
        qk_centered = torch.matmul(q_centered, k_centered.transpose(0, 1)) / q.size(-1)
        corr = qk_centered / torch.sqrt((q_var * k_var).clamp_min(self.eps))
        corr = corr.clamp(min=-1.0, max=1.0)
        return corr / self.tau_teacher, mse

    def _teacher_relation_tensor(
        self,
        future,
        target_channel,
        source_channel,
        offset=None,
    ):
        target = future[..., target_channel]
        if self.relation_teacher_space == 'delta_last':
            if offset is None:
                raise ValueError('relation_teacher_space=delta_last requires a teacher offset')
            target = target - offset[:, target_channel].to(future.device).unsqueeze(-1)

        if source_channel == target_channel:
            return target.unsqueeze(1)

        source = future[..., source_channel]
        if self.relation_teacher_space == 'delta_last':
            source = source - offset[:, source_channel].to(future.device).unsqueeze(-1)
        # The teacher has to compose the pair exactly like the student does,
        # otherwise the KL target is defined in a different relation space.
        if self.encoder.self_fill != 'linear':
            return torch.stack([target, source], dim=1)
        projected = self.teacher_shared_cross_projection(
            torch.cat([target, source], dim=-1)
        )
        return projected.unsqueeze(1)

    @torch.no_grad()
    def _teacher_embedding_scores(
        self,
        query_x,
        query_y,
        teacher_key_bank,
        target_channel,
        source_channel,
        source_slot=None,
    ):
        if self.teacher_mode == 'ema_input':
            q_rel = build_relation_encoder_input(
                query_x,
                target_channel,
                source_channel,
                relation_input_space=self.relation_input_space,
                shared_cross_projection=self.teacher_shared_cross_projection,
                self_fill=self.teacher_encoder.self_fill,
            )
        else:
            query_offset = query_x[:, -1, :]
            q_rel = self._teacher_relation_tensor(
                query_y,
                target_channel,
                source_channel,
                query_offset,
            )
        z_q = self.teacher_encoder(q_rel)
        if source_slot is None:
            source_slot = self.source_slot(target_channel, source_channel)
        z_k = teacher_key_bank[target_channel, source_slot].to(
            device=query_y.device,
            dtype=z_q.dtype,
        )
        if self.encoder.retrieval_similarity == 'l2':
            # The encoder stopped normalising for l2, so a bare dot product here
            # would be dominated by the embedding norm and collapse the teacher
            # onto a single candidate. Score it the same way the student is.
            q_l2 = z_q.float()
            k_l2 = z_k.float()
            return -(
                q_l2.pow(2).sum(dim=-1, keepdim=True)
                + k_l2.pow(2).sum(dim=-1).unsqueeze(0)
                - 2.0 * torch.matmul(q_l2, k_l2.transpose(0, 1))
            ) / float(q_l2.size(-1))
        return torch.matmul(z_q, z_k.transpose(0, 1))

    @torch.no_grad()
    def _teacher_embedding_logits(
        self,
        query_x,
        query_y,
        teacher_key_bank,
        target_channel,
        source_channel,
        source_slot=None,
    ):
        return self._teacher_embedding_scores(
            query_x,
            query_y,
            teacher_key_bank,
            target_channel,
            source_channel,
            source_slot,
        ) / self.tau_teacher

    def _encode_keys(self, k_rel):
        if self.key_chunk_size <= 0 or k_rel.size(0) <= self.key_chunk_size:
            return self.encoder(k_rel)

        chunks = []
        for start in range(0, k_rel.size(0), self.key_chunk_size):
            cur = k_rel[start:start + self.key_chunk_size]
            if self.training and torch.is_grad_enabled():
                chunks.append(checkpoint(self.encoder, cur, use_reentrant=False))
            else:
                chunks.append(self.encoder(cur))
        return torch.cat(chunks, dim=0)

    def candidate_subset_active(self):
        """Training-only. Validation and test always score the full memory bank."""
        return self.candidate_subset_mode != 'none' and self.training

    def external_teacher_active(self):
        """Whether this model is supervised by a precomputed non-Future teacher."""
        return (
            self.external_teacher_target != 'future'
            or self.external_teacher_objective != 'kl'
            or self.null_mode != 'off'
        )

    def null_logit(self, z_q):
        """Score of the explicit no-retrieval action, or None when disabled."""
        if self.null_mode == 'off':
            return None
        if self.null_mode == 'fixed':
            return z_q.new_zeros(z_q.size(0), 1)
        if z_q.size(-1) != self.null_head.in_features:
            raise ValueError(
                f'null head expects {self.null_head.in_features} dims, got {z_q.size(-1)}'
            )
        return self.null_head(z_q)

    @torch.no_grad()
    def _candidate_mining_metrics(self, bank_scores, future_mse, valid_mask, stats):
        """How much of the global Oracle the bank finds *before* injection.

        Injection makes the training candidate set artificially good, so these
        are the only honest read on what the current encoder retrieves.
        """
        valid_mask = valid_mask.bool()
        scores = bank_scores.detach().float().masked_fill(~valid_mask, float('-inf'))
        distances = future_mse.detach().float().masked_fill(~valid_mask, float('inf'))
        num_cand = scores.size(-1)
        oracle_k = min(int(self.candidate_oracle_inject_k), num_cand)
        oracle = torch.topk(distances, k=oracle_k, dim=-1, largest=False).indices
        oracle_valid = valid_mask.gather(1, oracle)
        denominator = oracle_valid.sum(dim=-1).float().clamp_min(1.0)

        metrics = {}
        for depth in (oracle_k, min(int(self.candidate_mine_top_m), num_cand)):
            bank_top = torch.topk(scores, k=depth, dim=-1, largest=True).indices
            hit = (
                (oracle.unsqueeze(-1) == bank_top.unsqueeze(-2)).any(dim=-1)
                & oracle_valid
            )
            metrics[f'bank_oracle_recall_at_{depth}'] = (
                hit.sum(dim=-1).float() / denominator
            ).mean()
        metrics['oracle_count_in_bank_top_m'] = stats['oracle_count_in_bank_top_m'].mean()
        metrics['oracle_missing_count_before_injection'] = stats[
            'oracle_missing_count_before_injection'
        ].mean()
        return metrics

    def _full_memory_gradient_indices(self, student_scores, future_mse, cand_mask,
                                      oracle_k, generator=None):
        """Which candidates get their score recomputed with gradient.

        The denominator stays the whole memory either way; this only decides
        where the candidate branch contributes to the encoder's gradient.

          Oracle Top-K    what the loss wants promoted
          hard negatives  what the model currently ranks highly and should not --
                          the false positives full-memory evaluation surfaces
          random          background, so the encoder is not shaped only by the
                          extremes
        """
        floor = torch.finfo(student_scores.dtype).min
        scores = student_scores.detach().float().masked_fill(~cand_mask, floor)
        distances = future_mse.detach().float().masked_fill(~cand_mask, float('inf'))
        num_cand = scores.size(-1)

        oracle_k = min(int(oracle_k), num_cand)
        oracle = torch.topk(distances, k=oracle_k, dim=-1, largest=False).indices
        picked = torch.zeros_like(cand_mask)
        picked.scatter_(1, oracle, True)

        parts = [oracle]
        hard_width = min(int(self.full_memory_hard_negatives), num_cand)
        if hard_width > 0:
            hard = scores.masked_fill(picked, floor).topk(hard_width, dim=-1).indices
            picked.scatter_(1, hard, True)
            parts.append(hard)
        random_width = min(int(self.full_memory_random_negatives), num_cand)
        if random_width > 0:
            weights = (cand_mask & ~picked).float()
            weights = torch.where(
                weights.sum(-1, keepdim=True) >= random_width, weights, cand_mask.float())
            if int(weights.sum(-1).min()) >= random_width:
                parts.append(torch.multinomial(
                    weights, random_width, replacement=False, generator=generator))
        return torch.cat(parts, dim=-1)

    def _full_online_replaces_scores(self, external_pool):
        """True when `_apply_full_memory_gradient` will discard the bank scores.

        full_online rescores every candidate from the live encoder and returns
        that instead of merging into what came before, so whatever the bank pass
        produced is thrown away. Worth knowing before building a graph over it.
        """
        return (
            self.full_memory_gradient_mode == 'full_online'
            and not self.candidate_subset_active()
            and external_pool is None
            and self.training
        )

    def _apply_full_memory_gradient(self, student_scores, z_q, candidate_x, future_mse,
                                    cand_mask, target_channel, source_channel):
        """Put the candidate branch back in the graph without shrinking the support.

        The bank is detached, so scoring against it trains only the query side.
        Re-encoding a chosen subset and scattering those scores into the full
        [B, N] logits keeps every candidate in the softmax denominator while the
        encoder still receives gradient from the candidate side -- for the
        candidates that matter. Positions outside the subset keep their bank
        score and contribute no candidate-side gradient; that is the
        approximation this mode makes, and it is stated rather than hidden.
        """
        if self.full_memory_gradient_mode == 'bank':
            return student_scores, {}
        if candidate_x is None:
            raise ValueError(
                'stage1_full_memory_gradient_mode needs candidate_x [N, L, C]')

        if self.full_memory_gradient_mode == 'full_online':
            z_all = self.encoder(
                self._relation_tensor(candidate_x, target_channel, source_channel))
            if self.pairwise_scorer is not None:
                # The pair scorer over the whole memory. Mining existed because a
                # pair feature is 2-4x the embedding width and cannot be folded
                # into a matmul, so the earlier runs trained on a mined subset and
                # were evaluated over the full bank -- the support mismatch that
                # made those results unreadable. Materialising every pair costs
                # ~2.4 GiB per target channel at N=8449, which the card has, so
                # the subset is not actually necessary and training scores the
                # same candidate set evaluation does.
                online = self._pairwise_bank_scores(z_q, z_all)
            elif self.retrieval_metric is not None:
                online = self.retrieval_metric.score(z_q, z_all)
            else:
                online = torch.matmul(z_q, z_all.transpose(0, 1))
            return online, {'full_memory_reencoded': z_q.new_tensor(float(z_all.size(0)))}

        selected = self._full_memory_gradient_indices(
            student_scores, future_mse, cand_mask, self.coverage_top_k)
        z_sel, unique_count = self._reencode_selected_candidates(
            candidate_x, selected, target_channel, source_channel)
        online = (
            self.retrieval_metric.score(z_q, z_sel.to(dtype=z_q.dtype))
            if self.retrieval_metric is not None
            else (z_q.unsqueeze(1) * z_sel.to(dtype=z_q.dtype)).sum(-1)
        )
        merged = student_scores.scatter(1, selected, online)
        return merged, {
            'full_memory_reencoded': z_q.new_tensor(float(unique_count)),
            'full_memory_grad_candidates': z_q.new_tensor(float(selected.size(-1))),
        }

    def _pairwise_bank_scores(self, z_q, z_bank, chunk_size=None):
        """Score a query batch against a whole key bank with the pair scorer.

        Returns [B, N]. Kept in the graph when the scorer is training, because
        the query side still carries gradient even where the bank does not.
        """
        if self.pairwise_scorer is None:
            raise RuntimeError('no pairwise scorer configured')
        chunk = int(chunk_size or self.key_chunk_size)
        scores = []
        for start in range(0, z_bank.size(0), chunk):
            block = z_bank[start:start + chunk].to(z_q.device, z_q.dtype)
            scores.append(self.pairwise_scorer(
                z_q, block.unsqueeze(0).expand(z_q.size(0), -1, -1)))
        return torch.cat(scores, dim=-1)

    def _reencode_selected_candidates(
        self, candidate_x, selected_indices, target_channel, source_channel
    ):
        """Embed the selected candidate pasts with the *current* encoder.

        Queries and candidates share one encoder and one parameter set, and the
        result stays inside the graph, so the candidate side carries gradient.
        Rows overlap heavily after Top-M mining, so the unique index set is
        encoded once and scattered back instead of encoding B*M relations.
        """
        bsz, top_m = selected_indices.shape
        flat = selected_indices.reshape(-1)
        unique_indices, inverse = torch.unique(flat, return_inverse=True)
        cand_x = candidate_x.index_select(0, unique_indices)
        k_rel = self._relation_tensor(cand_x, target_channel, source_channel)
        # Deliberately not _encode_keys: its gradient checkpointing
        # (use_reentrant=False) retains memory across optimizer steps on the
        # installed torch, which grew ~0.25 GiB/step until the run died. The
        # unique set is bounded by batch_size * top_m, so a plain forward is
        # both small enough and flat across steps.
        z_k_unique = self.encoder(k_rel)
        z_k = z_k_unique.index_select(0, inverse).view(bsz, top_m, -1)
        return z_k, unique_indices.numel()

    @torch.no_grad()
    def build_embedding_bank(self, memory_x, device, chunk_size=None):
        """Build stale relation-wise key bank [C, S, N, D] for one epoch."""
        was_training = self.training
        self.eval()
        chunk_size = int(chunk_size or self.key_chunk_size)
        memory_x = torch.as_tensor(memory_x, dtype=torch.float32)
        banks = []

        for c in range(self.channels):
            source_banks = []
            for r in self.source_channels(c):
                encoded = []
                for start in range(0, memory_x.size(0), chunk_size):
                    cur = memory_x[start:start + chunk_size].to(device)
                    rel = self._relation_tensor(cur, c, r)
                    encoded_chunk = self.encoder(rel).cpu()
                    if self.uses_sparse_relation_graph():
                        encoded_chunk = encoded_chunk.half()
                    encoded.append(encoded_chunk)
                source_banks.append(torch.cat(encoded, dim=0))
            banks.append(torch.stack(source_banks, dim=0))

        if was_training:
            self.train()
            self.teacher_encoder.eval()
            self.teacher_shared_cross_projection.eval()
        return torch.stack(banks, dim=0)

    @torch.no_grad()
    def build_direct_embedding_bank(self, memory_x, device, chunk_size=None):
        """Build the encoder-free cosine bank [C, S, N, 2*F*relation_seq_len]."""
        chunk_size = int(chunk_size or self.key_chunk_size)
        memory_x = torch.as_tensor(memory_x, dtype=torch.float32)
        banks = []
        for c in range(self.channels):
            source_banks = []
            for r in self.source_channels(c):
                encoded = []
                for start in range(0, memory_x.size(0), chunk_size):
                    cur = memory_x[start:start + chunk_size].to(device)
                    encoded.append(build_direct_relation_embedding(
                        cur,
                        c,
                        r,
                        relation_input_space=self.relation_input_space,
                        eps=self.eps,
                    ).cpu())
                source_banks.append(torch.cat(encoded, dim=0))
            banks.append(torch.stack(source_banks, dim=0))
        return torch.stack(banks, dim=0)

    @torch.no_grad()
    def build_teacher_embedding_bank(self, memory_values, device, chunk_size=None, memory_x_last=None):
        """Build relation-wise EMA key bank [C, S, N, D] for one epoch.

        ema_target embeds candidate futures for the legacy objective. ema_input
        embeds candidate past histories for Experiment 2.
        """
        was_training = self.training
        self.teacher_encoder.eval()
        self.teacher_shared_cross_projection.eval()
        chunk_size = int(chunk_size or self.key_chunk_size)
        memory_values = torch.as_tensor(memory_values, dtype=torch.float32)
        if memory_x_last is not None:
            memory_x_last = torch.as_tensor(memory_x_last, dtype=torch.float32)
        banks = []

        for c in range(self.channels):
            source_banks = []
            for r in self.source_channels(c):
                encoded = []
                for start in range(0, memory_values.size(0), chunk_size):
                    cur = memory_values[start:start + chunk_size].to(device)
                    cur_offset = (
                        None
                        if memory_x_last is None
                        else memory_x_last[start:start + chunk_size].to(device)
                    )
                    if self.teacher_mode == 'ema_input':
                        rel = build_relation_encoder_input(
                            cur,
                            c,
                            r,
                            relation_input_space=self.relation_input_space,
                            shared_cross_projection=self.teacher_shared_cross_projection,
                            self_fill=self.teacher_encoder.self_fill,
                        )
                    else:
                        rel = self._teacher_relation_tensor(cur, c, r, cur_offset)
                    encoded_chunk = self.teacher_encoder(rel).cpu()
                    if self.uses_sparse_relation_graph():
                        encoded_chunk = encoded_chunk.half()
                    encoded.append(encoded_chunk)
                source_banks.append(torch.cat(encoded, dim=0))
            banks.append(torch.stack(source_banks, dim=0))

        if was_training:
            self.train()
            self.teacher_encoder.eval()
            self.teacher_shared_cross_projection.eval()
        return torch.stack(banks, dim=0)

    @torch.no_grad()
    def update_ema_teacher(self, momentum):
        for teacher_param, student_param in zip(self.teacher_encoder.parameters(), self.encoder.parameters()):
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)
        for teacher_buffer, student_buffer in zip(self.teacher_encoder.buffers(), self.encoder.buffers()):
            teacher_buffer.copy_(student_buffer)
        for teacher_param, student_param in zip(
            self.teacher_shared_cross_projection.parameters(),
            self.shared_cross_projection.parameters(),
        ):
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)
        for teacher_buffer, student_buffer in zip(
            self.teacher_shared_cross_projection.buffers(),
            self.shared_cross_projection.buffers(),
        ):
            teacher_buffer.copy_(student_buffer)

    def forward(
        self,
        query_x,
        query_y,
        cand_mask,
        memory_y,
        key_bank,
        teacher_key_bank=None,
        memory_x_last=None,
        active_target_channels=None,
        compute_detailed_metrics=True,
        direct_retrieval=False,
        candidate_x=None,
        differentiable_keys=False,
        external_pool=None,
        external_teacher=None,
        external_utility=None,
        query_residual=None,
        memory_residual=None,
        reference_scores=None,
        mining_scores=None,
    ):
        bsz, num_cand = cand_mask.shape
        if mining_scores is not None:
            # [B, C, N] frozen scores that pick the training candidates. Kept
            # separate from the student's own scores so "which candidates the
            # loss saw" can be held fixed while the score function changes.
            if mining_scores.dim() != 3 or mining_scores.shape[0] != bsz:
                raise ValueError(
                    f'mining_scores must be [B, C, N] with B={bsz}, '
                    f'got {tuple(mining_scores.shape)}'
                )
            if mining_scores.shape[-1] != num_cand:
                raise ValueError(
                    f'mining_scores covers {mining_scores.shape[-1]} candidates, '
                    f'mask covers {num_cand}'
                )
            if self.candidate_subset_mode == 'none':
                raise ValueError(
                    'mining_scores only applies to the candidate-subset training '
                    'path; enable stage1_candidate_subset_mode'
                )
            # Evaluation scores the full memory and has no mining step, so the
            # scores are simply unused there rather than an error.
        if self.residual_teacher_active and (query_residual is None or memory_residual is None):
            raise ValueError(
                'stage1_residual_teacher needs the cached query and memory residuals'
            )
        if self.reference_pool_size > 0:
            if reference_scores is None:
                raise ValueError('a reference pool needs frozen reference scores [B, C, N]')
            if external_pool is not None:
                raise ValueError(
                    'reference pool and precomputed pool are two ways of choosing the '
                    'same columns; enable only one'
                )
            external_pool = torch.stack([
                self.reference_pool(reference_scores[:, c], cand_mask, self.reference_pool_size)
                for c in range(self.channels)
            ], dim=1)
            # The pool restriction block below expects the measured teachers too;
            # with a residual teacher those are computed inline instead.
            external_teacher = torch.zeros_like(external_pool, dtype=cand_mask.new_zeros(1).float().dtype)
            external_utility = torch.zeros_like(external_teacher)
        external_active = external_pool is not None
        if external_active:
            if external_teacher is None or external_utility is None:
                raise ValueError(
                    'an external candidate pool needs both its teacher scores and '
                    'its measured utility'
                )
            for name, tensor in (('external_teacher', external_teacher),
                                 ('external_utility', external_utility)):
                if tensor.shape != external_pool.shape:
                    raise ValueError(
                        f'{name} shape {tuple(tensor.shape)} does not match the pool '
                        f'{tuple(external_pool.shape)}'
                    )
            if external_pool.dim() != 3 or external_pool.size(0) != bsz:
                raise ValueError(
                    f'external_pool must be [B, C, M] with B={bsz}, '
                    f'got {tuple(external_pool.shape)}'
                )
            if self.candidate_subset_active():
                raise ValueError(
                    'an external candidate pool and mined candidate subsets are two '
                    'ways of choosing the same columns; enable only one'
                )
        if differentiable_keys:
            # Diagnostic path: re-encode every candidate history with the current
            # parameters inside the graph, so the key side carries gradient too.
            if direct_retrieval:
                raise ValueError(
                    'differentiable candidate encoding is incompatible with '
                    'encoder-free direct retrieval'
                )
            if candidate_x is None:
                raise ValueError(
                    'differentiable candidate encoding requires candidate_x [N, L, C]'
                )
            if candidate_x.dim() != 3 or candidate_x.size(0) != num_cand:
                raise ValueError(
                    f'candidate_x must be [{num_cand}, L, C], '
                    f'got {tuple(candidate_x.shape)}'
                )
        elif key_bank is None:
            raise ValueError('full-memory Stage-1 requires a relation key memory bank')
        if self.requires_ema_teacher_bank() and teacher_key_bank is None:
            raise ValueError('EMA Stage-1 teacher requires a teacher key memory bank')
        if self.requires_ema_teacher_bank() and teacher_key_bank.dim() != 4:
            raise ValueError(
                'relation-wise EMA teacher key bank must be [C, S, N, D], '
                f'got {tuple(teacher_key_bank.shape)}'
            )

        valid_query = cand_mask.sum(dim=1) > 0
        if valid_query.sum() == 0:
            zero = query_x.sum() * 0.0
            return zero, {'skipped_batches': 1.0}

        if not self._shape_logged:
            print(f'[stage1] batch_x={tuple(query_x.shape)} batch_y={tuple(query_y.shape)}')
            key_shape = None if key_bank is None else tuple(key_bank.shape)
            print(f'[stage1] key_bank={key_shape} memory_y={tuple(memory_y.shape)} mask={tuple(cand_mask.shape)}')
            if differentiable_keys:
                print(
                    '[stage1] differentiable candidate encoding enabled: '
                    f'candidate_x={tuple(candidate_x.shape)} (key bank bypassed)'
                )
            if teacher_key_bank is not None:
                print(f'[stage1] teacher_key_bank={tuple(teacher_key_bank.shape)} teacher_mode={self.teacher_mode}')
            print(
                f'[stage1] relation_input_space={self.relation_input_space} '
                f'relation_seq_len={self.relation_seq_len} '
                f'relation_features={self.encoder.n_features} '
                f'encoder={self.encoder.encoder_type} direct={direct_retrieval}'
            )
            if self.encoder.encoder_type == 'tcn':
                print(
                    f'[stage1] tcn receptive_field={self.encoder.encoder.receptive_field} '
                    f'in_channels={self.encoder.encoder.in_channels} '
                    f'layers={self.encoder.encoder.num_layers}'
                )
            self._shape_logged = True

        masked_fill = torch.finfo(query_x.dtype).min / 4
        losses = []
        kl_losses = []
        rank_losses = []
        rnc_losses = []
        infonce_losses = []
        expected_mse_losses = []
        metric_rows = []
        self_rows = []
        cross_rows = []

        targets = self.target_channels() if active_target_channels is None else active_target_channels
        # The candidate subset narrows cand_mask/valid_query to the selected
        # columns, and the selection is per relation branch. Keep the full-pool
        # originals so each branch starts from them.
        full_cand_mask = cand_mask
        full_valid_query = valid_query
        for c in targets:
            for source_slot, r in enumerate(self.source_channels(c)):
                cand_mask = full_cand_mask
                valid_query = full_valid_query
                future_mse = self._future_mse(
                    query_x,
                    query_y,
                    memory_y,
                    memory_x_last,
                    c,
                    r,
                )
                if self.loss_mode == 'rnc':
                    if self.rnc_quality_source != 'ema_cosine':
                        rnc_quality_distance = future_mse
                        rnc_targets = prepare_query_conditioned_rnc_targets(
                            rnc_quality_distance,
                            cand_mask,
                            tie_epsilon=self.rnc_tie_epsilon,
                        )
                elif self.loss_mode in ('topk_coverage', 'weighted_topk_ce'):
                    coverage_targets = prepare_topk_coverage_targets(
                        future_mse,
                        cand_mask,
                        self.coverage_top_k,
                    )
                else:
                    if self.teacher_mode not in ('ema_target', 'ema_input'):
                        teacher_logits, future_mse = self._teacher_logits(
                            query_x,
                            query_y,
                            memory_y,
                            memory_x_last,
                            c,
                            r,
                        )
                        if self.residual_teacher_active:
                            # future_mse stays as it is: it still feeds Recall and
                            # regret, which remain auxiliary diagnostics here.
                            teacher_logits = -self._residual_mse(
                                query_residual, memory_residual, c
                            ) / self.tau_teacher
                        teacher_logits = teacher_logits.masked_fill(
                            ~cand_mask, masked_fill
                        )
                        teacher_prob = torch.softmax(
                            teacher_logits, dim=-1
                        ).detach()
                    if compute_detailed_metrics:
                        oracle_rank = torch.argmin(
                            future_mse.masked_fill(
                                ~cand_mask, float('inf')
                            ),
                            dim=-1,
                        )
                        random_mse = (
                            future_mse.masked_fill(
                                ~cand_mask, 0.0
                            ).sum(dim=-1)
                            / cand_mask.sum(dim=-1).clamp_min(1)
                        ).detach()
                        if self.teacher_mode not in ('ema_target', 'ema_input'):
                            teacher_entropy = -(
                                teacher_prob
                                * torch.log(teacher_prob + self.eps)
                            ).sum(dim=-1)
                            teacher_rank = torch.argmax(
                                teacher_prob, dim=-1
                            )

                if direct_retrieval:
                    if self.encoder.retrieval_similarity != 'cosine':
                        raise ValueError('Diff1 Direct requires retrieval_similarity=cosine')
                    z_q = build_direct_relation_embedding(
                        query_x,
                        c,
                        r,
                        relation_input_space=self.relation_input_space,
                        eps=self.eps,
                    )
                    z_q_pre_normalized = z_q
                else:
                    q_rel = self._relation_tensor(query_x, c, r)
                    z_q, z_q_pre_normalized = self.encoder(
                        q_rel,
                        return_pre_normalized=True,
                    )
                if (
                    not direct_retrieval
                    and (self.variance_weight > 0.0 or self.covariance_weight > 0.0)
                ):
                    variance_loss, covariance_loss, embedding_std_mean = (
                        relation_variance_covariance_loss(
                            z_q_pre_normalized,
                            variance_target=self.variance_target,
                        )
                    )
                else:
                    variance_loss = z_q.sum() * 0.0
                    covariance_loss = variance_loss
                    embedding_std_mean = variance_loss.detach()
                weighted_variance_loss = self.variance_weight * variance_loss
                weighted_covariance_loss = self.covariance_weight * covariance_loss
                regularization_loss = (
                    weighted_variance_loss + weighted_covariance_loss
                )
                if differentiable_keys:
                    k_rel = self._relation_tensor(candidate_x, c, r)
                    z_k = self._encode_keys(k_rel).to(dtype=z_q.dtype)
                else:
                    z_k = key_bank[c, source_slot].to(
                        device=query_x.device, dtype=z_q.dtype
                    )

                if self.encoder.retrieval_similarity == 'l2':
                    # The encoder already skipped its L2 normalisation, so z_q and
                    # the key bank hold raw embeddings; score them with the
                    # negative mean squared distance. On normalised vectors this
                    # would be a monotone map of the dot product and change
                    # nothing about the ranking.
                    q_l2 = z_q.float()
                    k_l2 = z_k.float()
                    student_scores = -(
                        q_l2.pow(2).sum(dim=-1, keepdim=True)
                        + k_l2.pow(2).sum(dim=-1).unsqueeze(0)
                        - 2.0 * torch.matmul(q_l2, k_l2.transpose(0, 1))
                    ) / float(q_l2.size(-1))
                elif self.retrieval_metric is not None:
                    # One matmul over the whole bank, exactly as cosine was.
                    student_scores = self.retrieval_metric.score(z_q, z_k)
                elif self.pairwise_scorer is not None:
                    # Full-bank scoring. The key bank is rebuilt from the current
                    # encoder immediately before validation, so these embeddings
                    # are not stale -- only the pair comparison changes. Chunked
                    # because a pair feature is 2-4x wider than the embedding and
                    # the bank has thousands of rows.
                    if self.candidate_subset_active() or self._full_online_replaces_scores(
                            external_pool):
                        # Training: this result is discarded -- either it only
                        # mines which candidates the loss runs over, or
                        # full_online is about to rescore the whole bank from the
                        # live encoder. Selection is discrete anyway, so building
                        # a graph over all N candidates would cost a large
                        # transient allocation for a tensor nothing
                        # backpropagates through.
                        with torch.no_grad():
                            student_scores = self._pairwise_bank_scores(z_q.detach(), z_k)
                    else:
                        student_scores = self._pairwise_bank_scores(z_q, z_k)
                else:
                    student_scores = torch.matmul(z_q, z_k.transpose(0, 1))

                # Full-memory gradient path: the denominator stays [B, N] and the
                # candidate side re-enters the graph for the candidates that matter.
                full_memory_metrics = {}
                if (
                    self.full_memory_gradient_mode != 'bank'
                    and not self.candidate_subset_active()
                    and external_pool is None
                    and self.training
                ):
                    student_scores, full_memory_metrics = self._apply_full_memory_gradient(
                        student_scores, z_q, candidate_x, future_mse, cand_mask, c, r
                    )
                # Reported for every loss mode, not just the coverage branch that
                # happened to merge it first. A blank `full_memory_reencoded` on a
                # full_online run reads as "this arm did not re-encode", which is
                # the opposite of true and would make a loss comparison across
                # modes look like a gradient-mode comparison.

                subset_metrics = {}
                selected_indices = None
                external_teacher_c = None
                external_utility_c = None
                if external_active:
                    # A fixed pool, identical across teacher arms, so "subset cost"
                    # and "teacher effect" never end up folded into one number.
                    selected = external_pool[:, c]
                    selected_indices = selected
                    student_scores = student_scores.gather(1, selected)
                    future_mse = future_mse.gather(1, selected)
                    cand_mask = cand_mask.gather(1, selected)
                    valid_query = cand_mask.sum(dim=1) > 0
                    external_teacher_c = external_teacher[:, c]
                    external_utility_c = external_utility[:, c]
                    subset_metrics = external_pool_utility_metrics(
                        student_scores.detach(), external_utility_c, cand_mask
                    )
                    if self.loss_mode in ('topk_coverage', 'weighted_topk_ce'):
                        coverage_targets = prepare_topk_coverage_targets(
                            future_mse, cand_mask, self.coverage_top_k
                        )
                    else:
                        teacher_logits = teacher_logits.gather(1, selected).masked_fill(
                            ~cand_mask, masked_fill
                        )
                        teacher_prob = torch.softmax(teacher_logits, dim=-1).detach()
                    if compute_detailed_metrics:
                        oracle_rank = torch.argmin(
                            future_mse.masked_fill(~cand_mask, float('inf')), dim=-1
                        )
                        random_mse = (
                            future_mse.masked_fill(~cand_mask, 0.0).sum(dim=-1)
                            / cand_mask.sum(dim=-1).clamp_min(1)
                        ).detach()
                        if self.loss_mode not in ('topk_coverage', 'weighted_topk_ce'):
                            teacher_entropy = -(
                                teacher_prob * torch.log(teacher_prob + self.eps)
                            ).sum(dim=-1)
                            teacher_rank = torch.argmax(teacher_prob, dim=-1)
                elif self.candidate_subset_active():
                    if candidate_x is None:
                        raise ValueError(
                            'stage1_candidate_subset_mode requires candidate_x '
                            '[N, L, C] so selected candidates can be re-encoded'
                        )
                    # Common mining keeps the candidate ids identical across
                    # arms; self mining lets each arm pick its own, which is a
                    # final-system question rather than a score-function one.
                    bank_scores_for_mining = (
                        mining_scores[:, c] if mining_scores is not None else student_scores
                    )
                    selected, mining_stats = select_training_candidates(
                        bank_scores_for_mining,
                        future_mse,
                        cand_mask,
                        top_m=self.candidate_mine_top_m,
                        oracle_k=self.candidate_oracle_inject_k,
                        random_negatives=self.candidate_random_negatives,
                    )
                    subset_metrics = self._candidate_mining_metrics(
                        student_scores, future_mse, cand_mask, mining_stats
                    )
                    selected_indices = selected

                    if self.candidate_subset_mode == 'selected_reencode':
                        # Same encoder, same parameters, no detach: the candidate
                        # side is part of the graph the KL backpropagates through.
                        z_k_sel, unique_count = self._reencode_selected_candidates(
                            candidate_x, selected, c, r
                        )
                        if self.encoder.retrieval_similarity == 'l2':
                            student_scores = -(
                                z_q.float().pow(2).sum(dim=-1, keepdim=True)
                                + z_k_sel.float().pow(2).sum(dim=-1)
                                - 2.0 * (z_q.float().unsqueeze(1) * z_k_sel.float()).sum(dim=-1)
                            ) / float(z_q.size(-1))
                        elif self.retrieval_metric is not None:
                            # Same bilinear form as the full-bank path, now with
                            # both sides in the graph.
                            student_scores = self.retrieval_metric.score(
                                z_q, z_k_sel.to(dtype=z_q.dtype))
                        elif self.pairwise_scorer is not None:
                            # Both sides are in the graph here: z_q from the query
                            # forward and z_k_sel from re-encoding the selected
                            # candidates, so the shared encoder gets gradient
                            # through the candidate branch as well as the query.
                            student_scores = self.pairwise_scorer(
                                z_q, z_k_sel.to(dtype=z_q.dtype))
                        else:
                            student_scores = (
                                z_q.unsqueeze(1) * z_k_sel.to(dtype=z_q.dtype)
                            ).sum(dim=-1)
                        # The metric averager stacks tensors, so keep every row
                        # entry a tensor rather than a python scalar.
                        subset_metrics['candidate_unique_encoded'] = z_q.new_tensor(
                            float(unique_count)
                        )
                        if not self._subset_logged:
                            print(
                                '[stage1 candidate-subset] mode=selected_reencode '
                                f'top_m={selected.size(1)} '
                                f'oracle_inject_k={self.candidate_oracle_inject_k} '
                                f'unique_encoded={unique_count}/{selected.numel()} '
                                f'z_q.requires_grad={z_q.requires_grad} '
                                f'z_k.requires_grad={z_k_sel.requires_grad}'
                            )
                            self._subset_logged = True
                    else:
                        # Control arm: identical candidate set, but the scores keep
                        # coming from the detached bank embeddings.
                        if self.pairwise_scorer is not None:
                            # student_scores here were produced by the full-bank
                            # pairwise path, so gathering is correct -- but the
                            # candidate side is detached, which is the control this
                            # mode exists for. Flagged rather than silently mixed.
                            student_scores = student_scores.gather(1, selected)
                        else:
                            student_scores = student_scores.gather(1, selected)
                        if not self._subset_logged:
                            print(
                                '[stage1 candidate-subset] mode=selected_detached '
                                f'top_m={selected.size(1)} '
                                f'oracle_inject_k={self.candidate_oracle_inject_k} '
                                '(bank embeddings, no candidate gradient)'
                            )
                            self._subset_logged = True

                    # Every downstream tensor moves onto the selected columns so
                    # the loss, its target and the metrics all live on [B, M].
                    future_mse = future_mse.gather(1, selected)
                    cand_mask = cand_mask.gather(1, selected)
                    valid_query = cand_mask.sum(dim=1) > 0
                    if self.loss_mode in ('topk_coverage', 'weighted_topk_ce'):
                        # The Oracle positives were prepared over the full pool,
                        # so their indices mean nothing on the selected columns.
                        # Injection guarantees the global Oracle Top-K survived
                        # the mining, so recomputing here reselects the same
                        # candidates under their new positions.
                        coverage_targets = prepare_topk_coverage_targets(
                            future_mse, cand_mask, self.coverage_top_k
                        )
                    else:
                        teacher_logits = teacher_logits.gather(1, selected).masked_fill(
                            ~cand_mask, masked_fill
                        )
                        teacher_prob = torch.softmax(teacher_logits, dim=-1).detach()
                    if compute_detailed_metrics:
                        oracle_rank = torch.argmin(
                            future_mse.masked_fill(~cand_mask, float('inf')), dim=-1
                        )
                        random_mse = (
                            future_mse.masked_fill(~cand_mask, 0.0).sum(dim=-1)
                            / cand_mask.sum(dim=-1).clamp_min(1)
                        ).detach()
                        if self.loss_mode not in ('topk_coverage', 'weighted_topk_ce'):
                            teacher_entropy = -(
                                teacher_prob * torch.log(teacher_prob + self.eps)
                            ).sum(dim=-1)
                            teacher_rank = torch.argmax(teacher_prob, dim=-1)

                if self.loss_mode == 'rnc':
                    if self.rnc_quality_source == 'ema_cosine':
                        teacher_scores = self._teacher_embedding_scores(
                            query_x,
                            query_y,
                            teacher_key_bank,
                            c,
                            r,
                            source_slot,
                        )
                        rnc_quality_distance = (1.0 - teacher_scores).detach()
                        rnc_targets = prepare_query_conditioned_rnc_targets(
                            rnc_quality_distance,
                            cand_mask,
                            tie_epsilon=self.rnc_tie_epsilon,
                        )
                    valid_student_scores = student_scores[cand_mask]
                    if (
                        valid_student_scores.numel() == 0
                        or not torch.isfinite(valid_student_scores).all()
                    ):
                        continue

                    rnc_loss, rnc_metrics = query_conditioned_rnc_loss(
                        student_scores,
                        rnc_quality_distance,
                        cand_mask,
                        temperature=self.rnc_temperature,
                        tie_epsilon=self.rnc_tie_epsilon,
                        prepared_targets=rnc_targets,
                    )
                    if not torch.isfinite(rnc_loss):
                        continue

                    zero_metric = rnc_loss.detach() * 0.0
                    total_loss = rnc_loss + regularization_loss
                    row = {
                        'stage1_loss_total': total_loss.detach(),
                        'stage1_loss_kl': zero_metric,
                        'stage1_loss_rank': zero_metric,
                        'stage1_loss_rank_weighted': zero_metric,
                        'total_loss': total_loss.detach(),
                        'kl_loss': zero_metric,
                        'weighted_kl_loss': zero_metric,
                        'rank_loss': zero_metric,
                        'rnc_loss': rnc_loss.detach(),
                        'expected_mse_loss': zero_metric,
                        'weighted_expected_mse_loss': zero_metric,
                        'stage1_loss_variance': variance_loss.detach(),
                        'stage1_loss_variance_weighted': weighted_variance_loss.detach(),
                        'stage1_loss_covariance': covariance_loss.detach(),
                        'stage1_loss_covariance_weighted': weighted_covariance_loss.detach(),
                        'embedding_std_mean': embedding_std_mean.detach(),
                    }
                    row.update(rnc_metrics)
                    if compute_detailed_metrics:
                        random_mse = (
                            future_mse.masked_fill(~cand_mask, 0.0).sum(dim=-1)
                            / cand_mask.sum(dim=-1).clamp_min(1)
                        ).detach()
                        student_logits = (student_scores / self.tau_student).masked_fill(
                            ~cand_mask, masked_fill
                        )
                        student_log_prob = torch.log_softmax(student_logits, dim=-1)
                        student_prob = student_log_prob.exp()
                        student_entropy = -(
                            student_prob * student_log_prob
                        ).masked_fill(~cand_mask, 0.0).sum(dim=-1)
                        retrieval_metrics = _student_retrieval_metrics(
                            student_scores, student_prob, future_mse, cand_mask, eps=self.eps
                        )
                        topk_weighted = (
                            student_prob * future_mse.masked_fill(~cand_mask, 0.0)
                        ).sum(dim=-1)
                        row.update({
                            'student_entropy': student_entropy[valid_query].detach().mean(),
                            'student_effective_candidates': torch.exp(
                                student_entropy[valid_query]
                            ).detach().mean(),
                            'retrieved_future_mse_topk_weighted': topk_weighted[
                                valid_query
                            ].detach().mean(),
                            'random_future_mse': random_mse[valid_query].detach().mean(),
                        })
                        row.update(retrieval_metrics)
                        row['recall@1'] = row['oracle_recall_at_1']
                        row['recall@5'] = row['oracle_recall_at_5']
                        row['retrieved_future_mse_top1'] = row['retrieved_future_mse_at_1']
                        row['retrieval_gain'] = (
                            row['random_future_mse']
                            - row['retrieved_future_mse_topk_weighted']
                        )
                    losses.append(total_loss)
                    rnc_losses.append(rnc_loss)
                    metric_rows.append(row)
                    (self_rows if c == r else cross_rows).append(row)
                    continue

                if external_active and self.external_teacher_active():
                    teacher_source = {
                        'future': -future_mse,
                        'residual': external_teacher_c,
                        'utility': external_utility_c,
                    }[self.external_teacher_target]
                    utility_loss, utility_metrics = utility_teacher_loss(
                        student_scores,
                        teacher_source,
                        external_utility_c,
                        cand_mask,
                        tau_student=self.tau_student,
                        tau_teacher=self.external_teacher_tau,
                        objective=self.external_teacher_objective,
                        normalize=self.external_teacher_normalize,
                        null_logit=self.null_logit(z_q),
                        eps=self.eps,
                    )
                    if utility_loss is None:
                        continue
                    zero_metric = utility_loss.detach() * 0.0
                    total_loss = utility_loss + regularization_loss
                    student_logits = (student_scores / self.tau_student).masked_fill(
                        ~cand_mask, masked_fill
                    )
                    student_log_prob = torch.log_softmax(student_logits, dim=-1)
                    student_prob = student_log_prob.exp()
                    row = {
                        'stage1_loss_total': total_loss.detach(),
                        'stage1_loss_kl': zero_metric,
                        'stage1_loss_rank': zero_metric,
                        'stage1_loss_rank_weighted': zero_metric,
                        'total_loss': total_loss.detach(),
                        'kl': zero_metric,
                        'kl_loss': zero_metric,
                        'weighted_kl_loss': zero_metric,
                        'rank_loss': zero_metric,
                        'rnc_loss': zero_metric,
                        'expected_mse_loss': zero_metric,
                        'weighted_expected_mse_loss': zero_metric,
                        'stage1_loss_variance': variance_loss.detach(),
                        'stage1_loss_variance_weighted': weighted_variance_loss.detach(),
                        'stage1_loss_covariance': covariance_loss.detach(),
                        'stage1_loss_covariance_weighted': weighted_covariance_loss.detach(),
                        'embedding_std_mean': embedding_std_mean.detach(),
                    }
                    row.update(subset_metrics)
                    row.update(utility_metrics)
                    if compute_detailed_metrics:
                        row.update(_student_retrieval_metrics(
                            student_scores, student_prob, future_mse, cand_mask, eps=self.eps
                        ))
                        row.update(student_retrieval_metric_aliases(row))
                    losses.append(total_loss)
                    metric_rows.append(row)
                    (self_rows if c == r else cross_rows).append(row)
                    continue

                if self.loss_mode in ('topk_coverage', 'weighted_topk_ce'):
                    valid_student_scores = student_scores[cand_mask]
                    if (
                        valid_student_scores.numel() == 0
                        or not torch.isfinite(valid_student_scores).all()
                    ):
                        continue
                    student_logits = (student_scores / self.tau_student).masked_fill(
                        ~cand_mask, masked_fill
                    )
                    student_log_prob = torch.log_softmax(student_logits, dim=-1)
                    student_prob = student_log_prob.exp()
                    if self.loss_mode == 'weighted_topk_ce':
                        # Same Oracle Top-K positives as the coverage loss, but
                        # graded by future quality instead of spread uniformly.
                        coverage_loss, coverage_metrics = weighted_topk_listwise_ce(
                            student_log_prob, coverage_targets, self.tau_teacher,
                            eps=self.eps,
                        )
                    else:
                        coverage_loss, coverage_metrics = topk_coverage_loss(
                            student_log_prob,
                            coverage_targets,
                        )
                    if not torch.isfinite(coverage_loss):
                        continue

                    zero_metric = coverage_loss.detach() * 0.0
                    total_loss = coverage_loss + regularization_loss
                    row = {
                        'stage1_loss_total': total_loss.detach(),
                        'stage1_loss_kl': zero_metric,
                        'stage1_loss_rank': zero_metric,
                        'stage1_loss_rank_weighted': zero_metric,
                        'total_loss': total_loss.detach(),
                        'kl': zero_metric,
                        'kl_loss': zero_metric,
                        'weighted_kl_loss': zero_metric,
                        'rank_loss': zero_metric,
                        'rnc_loss': zero_metric,
                        'expected_mse_loss': zero_metric,
                        'weighted_expected_mse_loss': zero_metric,
                        'stage1_loss_variance': variance_loss.detach(),
                        'stage1_loss_variance_weighted': weighted_variance_loss.detach(),
                        'stage1_loss_covariance': covariance_loss.detach(),
                        'stage1_loss_covariance_weighted': weighted_covariance_loss.detach(),
                        'embedding_std_mean': embedding_std_mean.detach(),
                    }
                    row.update(coverage_metrics)
                    row.update(full_memory_metrics)
                    if compute_detailed_metrics and coverage_targets['oracle_indices'].numel():
                        # Where the Oracle sits in the model's own ranking. Recall@10
                        # stays near zero while a candidate moves from rank 4000 to
                        # rank 40; mean rank shows that movement directly.
                        row.update(oracle_rank_statistics(
                            student_scores, coverage_targets['oracle_indices'],
                            cand_mask, coverage_targets['oracle_valid']))
                        row.update(score_separation_metrics(
                            student_scores, coverage_targets['oracle_indices'], cand_mask))
                    if compute_detailed_metrics:
                        random_mse = (
                            future_mse.masked_fill(~cand_mask, 0.0).sum(dim=-1)
                            / cand_mask.sum(dim=-1).clamp_min(1)
                        ).detach()
                        student_entropy = -(
                            student_prob * student_log_prob
                        ).masked_fill(~cand_mask, 0.0).sum(dim=-1)
                        retrieval_metrics = _student_retrieval_metrics(
                            student_scores,
                            student_prob,
                            future_mse,
                            cand_mask,
                            eps=self.eps,
                        )
                        topk_weighted = (
                            student_prob * future_mse.masked_fill(~cand_mask, 0.0)
                        ).sum(dim=-1)
                        row.update({
                            'student_entropy': student_entropy[valid_query].detach().mean(),
                            'student_effective_candidates': torch.exp(
                                student_entropy[valid_query]
                            ).detach().mean(),
                            'retrieved_future_mse_topk_weighted': topk_weighted[
                                valid_query
                            ].detach().mean(),
                            'random_future_mse': random_mse[valid_query].detach().mean(),
                        })
                        row.update(retrieval_metrics)
                        row.update(
                            student_retrieval_metric_aliases(retrieval_metrics)
                        )
                        row['recall@1'] = row['oracle_recall_at_1']
                        row['recall@5'] = row['oracle_recall_at_5']
                        row['retrieved_future_mse_top1'] = row[
                            'retrieved_future_mse_at_1'
                        ]
                        row['retrieval_gain'] = (
                            row['random_future_mse']
                            - row['retrieved_future_mse_topk_weighted']
                        )
                    losses.append(total_loss)
                    metric_rows.append(row)
                    (self_rows if c == r else cross_rows).append(row)
                    continue

                if self.teacher_mode in ('ema_target', 'ema_input'):
                    teacher_logits = self._teacher_embedding_logits(
                        query_x,
                        query_y,
                        teacher_key_bank,
                        c,
                        r,
                        source_slot,
                    )
                    teacher_logits = teacher_logits.masked_fill(~cand_mask, masked_fill)
                    teacher_prob = torch.softmax(teacher_logits, dim=-1).detach()
                    if compute_detailed_metrics:
                        teacher_entropy = -(
                            teacher_prob * torch.log(teacher_prob + self.eps)
                        ).sum(dim=-1)
                        teacher_rank = torch.argmax(teacher_prob, dim=-1)

                student_logits = student_scores / self.tau_student
                student_logits = student_logits.masked_fill(~cand_mask, masked_fill)
                student_log_prob = torch.log_softmax(student_logits, dim=-1)
                student_prob = student_log_prob.exp()

                kl = (teacher_prob * (torch.log(teacher_prob + self.eps) - student_log_prob)).sum(dim=-1)
                kl = kl[valid_query]
                if kl.numel() == 0 or not torch.isfinite(kl).all():
                    continue
                kl_loss = kl.mean()

                rank_loss = student_scores.sum() * 0.0
                rank_metrics = _empty_rank_metrics(rank_loss, query_x.dtype, query_x.device)
                if self.use_rank_loss and self.rank_weight != 0.0:
                    rank_loss, rank_metrics = future_aware_topk_ranking_loss(
                        student_scores,
                        future_mse,
                        cand_mask,
                        top_k=self.rank_top_k,
                        rank_margin=self.rank_margin,
                        min_mse_gap=self.rank_min_mse_gap,
                    )

                expected_mse_loss = student_scores.sum() * 0.0
                expected_mse_metrics = {}
                if self.loss_mode == 'kl_expected_mse':
                    expected_mse_loss, expected_mse_metrics = expected_future_mse_loss(
                        student_prob,
                        future_mse,
                        cand_mask,
                        normalization=self.expected_mse_normalization,
                        eps=self.eps,
                    )

                infonce_loss = student_scores.sum() * 0.0
                infonce_metrics = {}
                if self.loss_mode == 'kl_infonce':
                    if self.infonce_positive_source == 'target_mse':
                        infonce_positive_distance = future_mse
                    else:
                        # The EMA future distribution is branch-specific. Negating its
                        # logits lets the shared smallest-distance Top-K helper select
                        # the largest cosine-similarity candidates for this branch.
                        infonce_positive_distance = -teacher_logits.detach()
                    infonce_loss, infonce_metrics = multi_positive_infonce_loss(
                        student_logits,
                        infonce_positive_distance,
                        cand_mask,
                        top_k=self.infonce_top_k,
                    )

                if self.loss_mode == 'kl':
                    total_loss = kl_loss
                elif self.loss_mode == 'kl_infonce':
                    total_loss = (
                        (1.0 - self.infonce_weight) * kl_loss
                        + self.infonce_weight * infonce_loss
                    )
                    infonce_losses.append(infonce_loss)
                elif self.loss_mode == 'kl_rank':
                    total_loss = kl_loss + self.rank_weight * rank_loss
                    rank_losses.append(rank_loss)
                else:
                    total_loss = (
                        (1.0 - self.expected_mse_weight) * kl_loss
                        + self.expected_mse_weight * expected_mse_loss
                    )
                    expected_mse_losses.append(expected_mse_loss)
                total_loss = total_loss + regularization_loss
                losses.append(total_loss)
                kl_losses.append(kl_loss)

                row = {
                    'kl': kl.detach().mean(),
                    'stage1_loss_total': total_loss.detach(),
                    'stage1_loss_kl': kl_loss.detach(),
                    'stage1_loss_infonce': infonce_loss.detach(),
                    'stage1_loss_infonce_weighted': (
                        self.infonce_weight * infonce_loss
                    ).detach(),
                    'stage1_loss_rank': rank_metrics['stage1_loss_rank'],
                    'stage1_loss_rank_weighted': (self.rank_weight * rank_loss).detach(),
                    'total_loss': total_loss.detach(),
                    'kl_loss': kl_loss.detach(),
                    'weighted_kl_loss': (
                        (1.0 - self.expected_mse_weight) * kl_loss
                        if self.loss_mode == 'kl_expected_mse'
                        else (
                            (1.0 - self.infonce_weight) * kl_loss
                            if self.loss_mode == 'kl_infonce'
                            else kl_loss
                        )
                    ).detach(),
                    'rank_loss': rank_loss.detach(),
                    'rnc_loss': (student_scores.sum() * 0.0).detach(),
                    'expected_mse_loss': expected_mse_loss.detach(),
                    'weighted_expected_mse_loss': (
                        self.expected_mse_weight * expected_mse_loss
                    ).detach(),
                    'stage1_loss_variance': variance_loss.detach(),
                    'stage1_loss_variance_weighted': weighted_variance_loss.detach(),
                    'stage1_loss_covariance': covariance_loss.detach(),
                    'stage1_loss_covariance_weighted': weighted_covariance_loss.detach(),
                    'embedding_std_mean': embedding_std_mean.detach(),
                }
                row.update(rank_metrics)
                row.update(expected_mse_metrics)
                row.update(infonce_metrics)
                row.update(subset_metrics)
                row.update(full_memory_metrics)
                if compute_detailed_metrics:
                    retrieval_metrics = _student_retrieval_metrics(
                        student_scores, student_prob, future_mse, cand_mask, eps=self.eps
                    )
                    teacher_retrieval_metrics = _student_retrieval_metrics(
                        teacher_logits,
                        teacher_prob,
                        future_mse,
                        cand_mask,
                        eps=self.eps,
                    )
                    distribution_metrics = _teacher_student_distribution_metrics(
                        teacher_prob,
                        student_prob,
                        cand_mask,
                        eps=self.eps,
                    )
                    future_cosine = self._future_cosine(
                        query_x,
                        query_y,
                        memory_y,
                        memory_x_last,
                        c,
                        r,
                    )
                    if selected_indices is not None:
                        # Every ranking source has to live on the same columns.
                        future_cosine = future_cosine.gather(1, selected_indices)
                    ranking_source_metrics = _ranking_source_topk_metrics(
                        student_scores,
                        teacher_logits,
                        future_mse,
                        future_cosine,
                        cand_mask,
                    )
                    top1_student = torch.argmax(student_prob, dim=-1)
                    top1_match = (top1_student == oracle_rank).float()
                    teacher_top1_match = (top1_student == teacher_rank).float()
                    top1_mse = future_mse.gather(1, top1_student[:, None]).squeeze(1)
                    topk_weighted = (
                        student_prob * future_mse.masked_fill(~cand_mask, 0.0)
                    ).sum(dim=-1)
                    student_entropy = -(
                        student_prob * student_log_prob
                    ).masked_fill(~cand_mask, 0.0).sum(dim=-1)
                    teacher_top1_prob = teacher_prob.max(dim=-1).values
                    student_top1_prob = student_prob.max(dim=-1).values
                    student_prob_on_teacher_top1 = student_prob.gather(
                        1, teacher_rank[:, None]
                    ).squeeze(1)
                    row.update({
                        'teacher_entropy': teacher_entropy[valid_query].detach().mean(),
                        'student_entropy': student_entropy[valid_query].detach().mean(),
                        'teacher_effective_candidates': torch.exp(
                            teacher_entropy[valid_query]
                        ).detach().mean(),
                        'student_effective_candidates': torch.exp(
                            student_entropy[valid_query]
                        ).detach().mean(),
                        'teacher_top1_prob': teacher_top1_prob[valid_query].detach().mean(),
                        'student_top1_prob': student_top1_prob[valid_query].detach().mean(),
                        'student_prob_on_teacher_top1': student_prob_on_teacher_top1[
                            valid_query
                        ].detach().mean(),
                        'teacher_student_top5_overlap': distribution_metrics[
                            'teacher_student_topk_overlap_at_5'
                        ],
                        'student_teacher_top1_match': teacher_top1_match[
                            valid_query
                        ].detach().mean(),
                        'top1_teacher_rank_match': top1_match[valid_query].detach().mean(),
                        'recall@1': top1_match[valid_query].detach().mean(),
                        'recall@5': retrieval_metrics['oracle_best_hit_at_5'],
                        'retrieved_future_mse_top1': top1_mse[valid_query].detach().mean(),
                        'retrieved_future_mse_topk_weighted': topk_weighted[
                            valid_query
                        ].detach().mean(),
                        'random_future_mse': random_mse[valid_query].detach().mean(),
                    })
                    row.update(distribution_metrics)
                    row.update(ranking_source_metrics)
                    row.update(retrieval_metrics)
                    for metric_k in (1, 5, 10):
                        row[f'student_oracle_recall_at_{metric_k}'] = (
                            retrieval_metrics[f'oracle_recall_at_{metric_k}']
                        )
                        row[f'student_oracle_best_hit_at_{metric_k}'] = (
                            retrieval_metrics[f'oracle_best_hit_at_{metric_k}']
                        )
                        row[f'student_topk_probability_mass_at_{metric_k}'] = (
                            retrieval_metrics[f'topk_probability_mass_at_{metric_k}']
                        )
                        row[f'student_oracle_topk_probability_mass_at_{metric_k}'] = (
                            retrieval_metrics[
                                f'oracle_topk_probability_mass_at_{metric_k}'
                            ]
                        )
                        row[f'student_retrieved_future_mse_at_{metric_k}'] = (
                            retrieval_metrics[f'retrieved_future_mse_at_{metric_k}']
                        )
                        row[f'student_best_future_mse_at_{metric_k}'] = (
                            retrieval_metrics[f'best_future_mse_at_{metric_k}']
                        )
                        row[f'student_retrieval_regret_at_{metric_k}'] = (
                            retrieval_metrics[f'retrieval_regret_at_{metric_k}']
                        )
                        row[f'teacher_oracle_recall_at_{metric_k}'] = (
                            teacher_retrieval_metrics[f'oracle_recall_at_{metric_k}']
                        )
                        row[f'teacher_oracle_best_hit_at_{metric_k}'] = (
                            teacher_retrieval_metrics[f'oracle_best_hit_at_{metric_k}']
                        )
                        row[f'teacher_topk_probability_mass_at_{metric_k}'] = (
                            teacher_retrieval_metrics[
                                f'topk_probability_mass_at_{metric_k}'
                            ]
                        )
                        row[f'teacher_oracle_topk_probability_mass_at_{metric_k}'] = (
                            teacher_retrieval_metrics[
                                f'oracle_topk_probability_mass_at_{metric_k}'
                            ]
                        )
                        row[f'teacher_retrieved_future_mse_at_{metric_k}'] = (
                            teacher_retrieval_metrics[f'retrieved_future_mse_at_{metric_k}']
                        )
                        row[f'teacher_best_future_mse_at_{metric_k}'] = (
                            teacher_retrieval_metrics[f'best_future_mse_at_{metric_k}']
                        )
                        row[f'teacher_retrieval_regret_at_{metric_k}'] = (
                            teacher_retrieval_metrics[f'retrieval_regret_at_{metric_k}']
                        )
                    for metric_k in (5, 10):
                        row[f'student_ndcg_at_{metric_k}'] = retrieval_metrics[
                            f'ndcg_at_{metric_k}'
                        ]
                        row[f'teacher_ndcg_at_{metric_k}'] = teacher_retrieval_metrics[
                            f'ndcg_at_{metric_k}'
                        ]
                    row['student_spearman_score_vs_negative_mse'] = retrieval_metrics[
                        'spearman_score_vs_negative_mse'
                    ]
                    row['student_oracle_spearman'] = retrieval_metrics[
                        'spearman_score_vs_negative_mse'
                    ]
                    row['teacher_spearman_score_vs_negative_mse'] = (
                        teacher_retrieval_metrics['spearman_score_vs_negative_mse']
                    )
                    row['teacher_oracle_spearman'] = teacher_retrieval_metrics[
                        'spearman_score_vs_negative_mse'
                    ]
                    row['teacher_entropy_normalized'] = teacher_retrieval_metrics[
                        'student_entropy_normalized'
                    ]
                    row['teacher_top5_probability_mass'] = teacher_retrieval_metrics[
                        'student_top5_probability_mass'
                    ]
                    row['retrieval_gain'] = (
                        row['random_future_mse']
                        - row['retrieved_future_mse_topk_weighted']
                    )
                metric_rows.append(row)
                (self_rows if c == r else cross_rows).append(row)

        if not losses:
            zero = query_x.sum() * 0.0
            return zero, {'skipped_batches': 1.0}

        loss = torch.stack(losses).mean()
        metrics = self._average_metrics(metric_rows)
        metrics['loss'] = loss.detach()
        metrics['total_loss'] = loss.detach()
        if kl_losses:
            metrics['stage1_loss_kl'] = torch.stack(kl_losses).mean().detach()
        if rank_losses:
            rank_loss_mean = torch.stack(rank_losses).mean()
            metrics['stage1_loss_rank'] = rank_loss_mean.detach()
            metrics['stage1_loss_rank_weighted'] = (self.rank_weight * rank_loss_mean).detach()
        else:
            zero_metric = loss.detach() * 0.0
            metrics['stage1_loss_rank'] = zero_metric
            metrics['stage1_loss_rank_weighted'] = zero_metric
        zero_metric = loss.detach() * 0.0
        metrics['kl_loss'] = metrics['stage1_loss_kl']
        metrics['weighted_kl_loss'] = (
            (1.0 - self.expected_mse_weight) * metrics['kl_loss']
            if self.loss_mode == 'kl_expected_mse'
            else (
                (1.0 - self.infonce_weight) * metrics['kl_loss']
                if self.loss_mode == 'kl_infonce'
                else metrics['kl_loss']
            )
        )
        infonce_loss_mean = (
            torch.stack(infonce_losses).mean().detach()
            if infonce_losses else zero_metric
        )
        metrics['stage1_loss_infonce'] = infonce_loss_mean
        metrics['stage1_loss_infonce_weighted'] = (
            self.infonce_weight * infonce_loss_mean
        )
        metrics['rank_loss'] = metrics['stage1_loss_rank']
        metrics['rnc_loss'] = (
            torch.stack(rnc_losses).mean().detach() if rnc_losses else zero_metric
        )
        expected_mse_mean = (
            torch.stack(expected_mse_losses).mean().detach()
            if expected_mse_losses else metrics.get('expected_mse_loss', zero_metric)
        )
        metrics['expected_mse_loss'] = expected_mse_mean
        metrics['weighted_expected_mse_loss'] = (
            self.expected_mse_weight * expected_mse_mean
        )
        metrics['stage1_loss_total'] = loss.detach()
        metrics['skipped_batches'] = torch.tensor(0.0, device=query_x.device)
        metrics.update(self._prefixed_average('self_', self_rows, query_x.device))
        metrics.update(self._prefixed_average('cross_', cross_rows, query_x.device))
        return loss, metrics

    @torch.no_grad()
    def distribution_probe(
        self,
        query_x,
        query_y,
        cand_mask,
        memory_y,
        key_bank,
        teacher_key_bank=None,
        memory_x_last=None,
        target_channel=0,
        source_channel=0,
        query_index=0,
        top_n=50,
    ):
        if key_bank is None:
            raise ValueError('distribution_probe requires a relation key memory bank')
        if self.teacher_mode in ('ema_target', 'ema_input') and teacher_key_bank is None:
            raise ValueError('EMA Stage-1 teacher requires a teacher key memory bank')
        if self.teacher_mode in ('ema_target', 'ema_input') and teacher_key_bank.dim() != 4:
            raise ValueError(
                'relation-wise EMA teacher key bank must be [C, S, N, D], '
                f'got {tuple(teacher_key_bank.shape)}'
            )

        c = int(target_channel)
        r = int(source_channel)
        if c < 0 or c >= self.channels or r < 0 or r >= self.channels:
            raise ValueError(f'invalid probe channels target={c}, source={r}, channels={self.channels}')

        valid = torch.nonzero(cand_mask.sum(dim=1) > 0, as_tuple=False).flatten()
        if valid.numel() == 0:
            return None
        q_pos = int(query_index)
        q_pos = max(0, min(q_pos, valid.numel() - 1))
        q_idx = valid[q_pos]
        source_slot = self.source_slot(c, r)

        masked_fill = torch.finfo(query_x.dtype).min / 4
        mse_teacher_logits, future_mse = self._teacher_logits(
            query_x,
            query_y,
            memory_y,
            memory_x_last,
            c,
            r,
        )
        if self.teacher_mode in ('ema_target', 'ema_input'):
            teacher_logits = self._teacher_embedding_logits(
                query_x,
                query_y,
                teacher_key_bank,
                c,
                r,
                source_slot,
            )
        else:
            teacher_logits = mse_teacher_logits
        teacher_logits = teacher_logits.masked_fill(~cand_mask, masked_fill)
        teacher_prob = torch.softmax(teacher_logits, dim=-1)

        q_rel = self._relation_tensor(query_x, c, r)
        z_q = self.encoder(q_rel)
        z_k = key_bank[c, source_slot].to(device=query_x.device, dtype=z_q.dtype)
        student_logits = torch.matmul(z_q, z_k.transpose(0, 1)) / self.tau_student
        student_logits = student_logits.masked_fill(~cand_mask, masked_fill)
        student_prob = torch.softmax(student_logits, dim=-1)

        valid_mask = cand_mask[q_idx]
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return None
        q_row = int(q_idx.item())
        distribution_metrics = _teacher_student_distribution_metrics(
            teacher_prob[q_row:q_row + 1],
            student_prob[q_row:q_row + 1],
            cand_mask[q_row:q_row + 1],
            eps=self.eps,
        )
        k = min(max(int(top_n), 1), valid_count)
        ranked = torch.topk(teacher_prob[q_idx].masked_fill(~valid_mask, -1.0), k=k, dim=-1).indices
        top5_student = torch.topk(student_prob[q_idx].masked_fill(~valid_mask, -1.0), k=min(5, valid_count), dim=-1).indices
        top5_teacher = torch.topk(
            teacher_prob[q_idx].masked_fill(~valid_mask, -1.0),
            k=min(5, valid_count),
            dim=-1,
        ).indices
        top5_overlap = (
            top5_student[:, None] == top5_teacher[None, :]
        ).any(dim=1).float().mean()
        teacher_top1 = ranked[0]
        student_top1 = torch.argmax(student_prob[q_idx].masked_fill(~valid_mask, -1.0), dim=-1)

        return {
            'query_index': q_idx.detach().cpu(),
            'target_channel': torch.tensor(c),
            'source_channel': torch.tensor(r),
            'candidate_indices': ranked.detach().cpu(),
            'teacher_prob': teacher_prob[q_idx, ranked].detach().cpu(),
            'student_prob': student_prob[q_idx, ranked].detach().cpu(),
            'future_mse': future_mse[q_idx, ranked].detach().cpu(),
            'teacher_top1': teacher_top1.detach().cpu(),
            'student_top1': student_top1.detach().cpu(),
            'student_prob_on_teacher_top1': student_prob[q_idx, teacher_top1].detach().cpu(),
            'top5_overlap': top5_overlap.detach().cpu(),
            'teacher_student_kl_divergence': distribution_metrics[
                'teacher_student_kl_divergence'
            ].detach().cpu(),
            'student_teacher_kl_divergence': distribution_metrics[
                'student_teacher_kl_divergence'
            ].detach().cpu(),
            'teacher_student_js_divergence': distribution_metrics[
                'teacher_student_js_divergence'
            ].detach().cpu(),
            'teacher_student_prob_l1': distribution_metrics[
                'teacher_student_prob_l1'
            ].detach().cpu(),
            'teacher_student_total_variation': distribution_metrics[
                'teacher_student_total_variation'
            ].detach().cpu(),
            'teacher_student_hellinger_distance': distribution_metrics[
                'teacher_student_hellinger_distance'
            ].detach().cpu(),
            'teacher_student_probability_cosine': distribution_metrics[
                'teacher_student_probability_cosine'
            ].detach().cpu(),
            'teacher_student_entropy_gap': distribution_metrics[
                'teacher_student_entropy_gap'
            ].detach().cpu(),
            'teacher_student_entropy_abs_gap': distribution_metrics[
                'teacher_student_entropy_abs_gap'
            ].detach().cpu(),
            'student_teacher_spearman': distribution_metrics[
                'student_teacher_spearman'
            ].detach().cpu(),
            'teacher_student_topk_overlap_at_1': distribution_metrics[
                'teacher_student_topk_overlap_at_1'
            ].detach().cpu(),
            'teacher_student_topk_overlap_at_5': distribution_metrics[
                'teacher_student_topk_overlap_at_5'
            ].detach().cpu(),
            'teacher_student_topk_overlap_at_10': distribution_metrics[
                'teacher_student_topk_overlap_at_10'
            ].detach().cpu(),
        }

    def _average_metrics(self, rows):
        if not rows:
            return {}
        return {k: torch.stack([row[k] for row in rows]).mean() for k in rows[0]}

    def _prefixed_average(self, prefix, rows, device):
        if not rows:
            return {prefix + 'kl': torch.tensor(0.0, device=device)}
        return {prefix + k: v for k, v in self._average_metrics(rows).items()}
