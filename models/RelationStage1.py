import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from layers.relation_patch_embed import RelationPatchEmbedding
from layers.relation_tcn import RelationTCN


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
        order = torch.argsort(row_future, dim=0, stable=True)
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
def _student_retrieval_metrics(student_scores, student_prob, future_mse, valid_mask, eps=1e-8):
    """Future-aware diagnostics; NDCG uses relevance=1/(1+mean-normalized MSE)."""
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
            'student_top5_probability_mass': torch.topk(prob, k=min(5, count)).values.sum(),
        }

        mean_distance = distances.mean().clamp_min(float(eps))
        relevance = 1.0 / (1.0 + distances / mean_distance)
        oracle_best_idx = torch.argmin(distances)
        for k in (1, 5, 10):
            effective_k = min(k, count)
            student_idx = torch.topk(scores, k=effective_k, largest=True).indices
            oracle_idx = torch.topk(distances, k=effective_k, largest=False).indices
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
            score_order = torch.argsort(scores, stable=True)
            quality_order = torch.argsort(-distances, stable=True)
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

        first_order = torch.argsort(first, stable=True)
        second_order = torch.argsort(second, stable=True)
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
def _teacher_student_distribution_metrics(
    teacher_prob,
    student_prob,
    valid_mask,
    eps=1e-8,
):
    """Compare teacher and student distributions on each valid candidate pool."""
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
            teacher_order = torch.argsort(teacher, stable=True)
            student_order = torch.argsort(student, stable=True)
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
            teacher_topk = torch.topk(
                teacher, k=effective_k, largest=True
            ).indices
            student_topk = torch.topk(
                student, k=effective_k, largest=True
            ).indices
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


@torch.no_grad()
def _ranking_source_topk_metrics(
    student_scores,
    teacher_scores,
    future_mse,
    future_cosine,
    valid_mask,
):
    """Pairwise Top-K overlap among Student, Teacher, and two future Oracles."""
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
                key: torch.topk(value, k=effective_k, largest=True).indices
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
        requested_loss_mode = getattr(configs, 'stage1_loss_mode', 'kl')
        legacy_use_rank_loss = bool(int(getattr(configs, 'stage1_use_rank_loss', 0)))
        if requested_loss_mode not in (
            'kl', 'kl_infonce', 'kl_rank', 'rnc', 'kl_expected_mse',
            'topk_coverage'
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
        if self.loss_mode == 'topk_coverage':
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
    ):
        bsz, num_cand = cand_mask.shape
        if key_bank is None:
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
            print(f'[stage1] key_bank={tuple(key_bank.shape)} memory_y={tuple(memory_y.shape)} mask={tuple(cand_mask.shape)}')
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
        for c in targets:
            for source_slot, r in enumerate(self.source_channels(c)):
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
                elif self.loss_mode == 'topk_coverage':
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
                else:
                    student_scores = torch.matmul(z_q, z_k.transpose(0, 1))
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

                if self.loss_mode == 'topk_coverage':
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
