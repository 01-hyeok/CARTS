import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from layers.relation_patch_embed import RelationPatchEmbedding


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
def prepare_query_conditioned_rnc_targets(
    future_mse,
    valid_mask,
    tie_epsilon=0.0,
):
    """Prepare target-only RnC ordering once for reuse across source relations."""
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
            row[f'retrieved_future_mse_at_{k}'] = retrieved_mse
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
                f'oracle_recall_at_{k}', f'retrieved_future_mse_at_{k}',
                f'oracle_future_mse_at_{k}', f'retrieval_regret_at_{k}',
            ])
        keys.extend(['ndcg_at_5', 'ndcg_at_10'])
        return {key: zero for key in keys}

    return {
        key: torch.stack([row[key] for row in rows]).mean().detach()
        for key in rows[0]
    }


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


class RelationEncoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.encoder_type = getattr(configs, 'relation_encoder_type', 'transformer')
        self.pooling = getattr(configs, 'relation_pooling', 'cls')
        self.self_fill = getattr(configs, 'relation_self_fill', 'zero')
        self.seq_len = configs.seq_len
        self.d_model = configs.d_model

        if self.encoder_type == 'transformer':
            if self.pooling not in ('cls', 'mean'):
                raise ValueError(f'Unsupported relation_pooling for transformer: {self.pooling}')
            self.patch_embed = RelationPatchEmbedding(
                seq_len=configs.seq_len,
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
            if self.self_fill not in ('zero', 'repeat'):
                raise ValueError(f'Unsupported relation_self_fill for mlp: {self.self_fill}')
            self.role_embedding = nn.Parameter(torch.zeros(1, 2, configs.seq_len))
            self.encoder = nn.Sequential(
                nn.Linear(2 * configs.seq_len, configs.d_ff),
                nn.GELU(),
                nn.Dropout(configs.dropout),
                nn.Linear(configs.d_ff, configs.d_model),
            )
        else:
            raise ValueError(f'Unsupported relation_encoder_type: {self.encoder_type}')

        self.norm = nn.LayerNorm(configs.d_model)
        self.proj = nn.Sequential(
            nn.Linear(configs.d_model, configs.d_model),
            nn.GELU(),
            nn.Linear(configs.d_model, configs.d_model),
        )

    def forward(self, relation_x):
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
            if relation_x.dim() != 3:
                raise ValueError(f'relation input must be [B, R, L], got {tuple(relation_x.shape)}')
            bsz, roles, seq_len = relation_x.shape
            if roles not in (1, 2):
                raise ValueError(f'relation role count must be 1 or 2, got {roles}')
            if seq_len != self.seq_len:
                raise ValueError(f'expected seq_len={self.seq_len}, got {seq_len}')
            padded = relation_x.new_zeros(bsz, 2, self.seq_len)
            padded[:, :roles] = relation_x
            if roles == 1 and self.self_fill == 'repeat':
                padded[:, 1] = relation_x[:, 0]
            h = self.encoder((padded + self.role_embedding).reshape(bsz, -1))

        z = self.proj(self.norm(h))
        return F.normalize(z, dim=-1)


class Model(nn.Module):
    """Stage-1 relation-wise retrieval encoder.

    Inputs are normalized sliding windows:
      query_x: [B, L, C], query_y: [B, H, C]
      memory_y: [N, H, C], cand_mask: [B, N]
    The teacher branch uses target-channel future similarity over all valid memory.
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
        self.relation_teacher_space = getattr(configs, 'relation_teacher_space', 'absolute')
        self.source_mode = configs.source_mode
        self.relation_graph_threshold = int(getattr(configs, 'relation_graph_threshold', 21))
        self.target_mode = configs.target_mode
        self.target_channel = configs.target_channel
        self.key_chunk_size = int(getattr(configs, 'stage1_key_chunk_size', 1024))
        requested_loss_mode = getattr(configs, 'stage1_loss_mode', 'kl')
        legacy_use_rank_loss = bool(int(getattr(configs, 'stage1_use_rank_loss', 0)))
        if requested_loss_mode not in ('kl', 'kl_rank', 'rnc', 'kl_expected_mse'):
            raise ValueError(f'Unsupported stage1_loss_mode: {requested_loss_mode}')
        # Preserve old rank scripts, which only set stage1_use_rank_loss=1.
        self.loss_mode = 'kl_rank' if requested_loss_mode == 'kl' and legacy_use_rank_loss else requested_loss_mode
        self.use_rank_loss = self.loss_mode == 'kl_rank'
        self.rank_weight = float(getattr(configs, 'stage1_rank_weight', 0.1))
        self.rank_margin = float(getattr(configs, 'stage1_rank_margin', 0.1))
        self.rank_min_mse_gap = float(getattr(configs, 'stage1_rank_min_mse_gap', 0.0))
        self.rank_top_k = getattr(configs, 'stage1_rank_top_k', None)
        if self.rank_top_k is None or int(self.rank_top_k) <= 0:
            self.rank_top_k = int(getattr(configs, 'top_k', 10))
        else:
            self.rank_top_k = int(self.rank_top_k)
        self.rnc_temperature = float(getattr(configs, 'rnc_temperature', 0.2))
        self.rnc_tie_epsilon = float(getattr(configs, 'rnc_tie_epsilon', 0.0))
        self.rnc_quality_source = getattr(configs, 'rnc_quality_source', 'future_mse')
        self.expected_mse_weight = float(getattr(configs, 'expected_mse_weight', 0.1))
        self.expected_mse_normalization = getattr(
            configs, 'expected_mse_normalization', 'mean'
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
        self.eps = 1e-8
        self.encoder = RelationEncoder(configs)
        if self.teacher_mode not in ('mse', 'pearson', 'ema_target'):
            raise ValueError(f'Unsupported stage1_teacher_mode: {self.teacher_mode}')
        if self.relation_teacher_space == 'delta_last' and self.teacher_mse_space == 'raw':
            raise ValueError(
                'relation_teacher_space=delta_last is only supported with '
                'teacher_mse_space=normalized because query_x/memory_x offsets are normalized'
            )
        if self.teacher_mode == 'ema_target' and self.seq_len != self.pred_len:
            raise ValueError(
                'stage1_teacher_mode=ema_target requires seq_len == pred_len '
                f'for shared EMA encoder shapes, got seq_len={self.seq_len}, pred_len={self.pred_len}'
            )
        self.teacher_encoder = copy.deepcopy(self.encoder)
        for param in self.teacher_encoder.parameters():
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
        if self.teacher_mode != 'ema_target':
            return False
        return self.loss_mode != 'rnc' or self.rnc_quality_source == 'ema_cosine'

    def source_channels(self, target_channel):
        if self.relation_sources is not None:
            return self.relation_sources[int(target_channel)]
        if self.source_mode == 'topk_corr' or (
            self.source_mode == 'auto' and self.channels >= self.relation_graph_threshold
        ):
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
        target = x[..., target_channel]
        if self.relation_input_space == 'delta_last':
            target = target - target[:, -1:].detach()
        if source_channel == target_channel:
            return target.unsqueeze(1)
        source = x[..., source_channel]
        if self.relation_input_space == 'delta_last':
            source = source - source[:, -1:].detach()
        return torch.stack([target, source], dim=1)

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

    def _future_mse(self, query_x, query_y, memory_y, memory_x_last, target_channel):
        q, k = self._future_distance_inputs(
            query_x, query_y, memory_y, memory_x_last, target_channel
        )
        # MSE(q, k) over H without materializing [B, N, H]. This is also
        # retained as a teacher-independent quality metric for Pearson mode.
        q2 = (q ** 2).mean(dim=-1, keepdim=True)
        k2 = (k ** 2).mean(dim=-1).unsqueeze(0)
        qk = torch.matmul(q, k.transpose(0, 1)) / q.size(-1)
        return (q2 + k2 - 2.0 * qk).clamp_min(0.0)

    def _teacher_logits(self, query_x, query_y, memory_y, memory_x_last, target_channel):
        q, k = self._future_distance_inputs(
            query_x, query_y, memory_y, memory_x_last, target_channel
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

    def _teacher_target_relation(self, future, target_channel, offset=None):
        target = future[..., target_channel]
        if self.relation_teacher_space == 'delta_last':
            if offset is None:
                raise ValueError('relation_teacher_space=delta_last requires a teacher offset')
            target = target - offset[:, target_channel].to(future.device).unsqueeze(-1)
        return target.unsqueeze(1)

    @torch.no_grad()
    def _teacher_embedding_scores(self, query_x, query_y, teacher_key_bank, target_channel):
        query_offset = query_x[:, -1, :]
        q_rel = self._teacher_target_relation(query_y, target_channel, query_offset)
        z_q = self.teacher_encoder(q_rel)
        z_k = teacher_key_bank[target_channel].to(query_y.device)
        return torch.matmul(z_q, z_k.transpose(0, 1))

    @torch.no_grad()
    def _teacher_embedding_logits(self, query_x, query_y, teacher_key_bank, target_channel):
        return self._teacher_embedding_scores(
            query_x, query_y, teacher_key_bank, target_channel
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
                    cur = memory_x[start:start + chunk_size]
                    rel = self._relation_tensor(cur, c, r).to(device)
                    encoded_chunk = self.encoder(rel).cpu()
                    if self.uses_sparse_relation_graph():
                        encoded_chunk = encoded_chunk.half()
                    encoded.append(encoded_chunk)
                source_banks.append(torch.cat(encoded, dim=0))
            banks.append(torch.stack(source_banks, dim=0))

        if was_training:
            self.train()
        return torch.stack(banks, dim=0)

    @torch.no_grad()
    def build_teacher_embedding_bank(self, memory_y, device, chunk_size=None, memory_x_last=None):
        """Build EMA target-future teacher key bank [C, N, D] for one epoch."""
        was_training = self.training
        self.teacher_encoder.eval()
        chunk_size = int(chunk_size or self.key_chunk_size)
        memory_y = torch.as_tensor(memory_y, dtype=torch.float32)
        if memory_x_last is not None:
            memory_x_last = torch.as_tensor(memory_x_last, dtype=torch.float32)
        banks = []

        for c in range(self.channels):
            encoded = []
            for start in range(0, memory_y.size(0), chunk_size):
                cur = memory_y[start:start + chunk_size]
                cur_offset = None if memory_x_last is None else memory_x_last[start:start + chunk_size]
                rel = self._teacher_target_relation(cur, c, cur_offset).to(device)
                encoded.append(self.teacher_encoder(rel).cpu())
            banks.append(torch.cat(encoded, dim=0))

        if was_training:
            self.train()
        return torch.stack(banks, dim=0)

    @torch.no_grad()
    def update_ema_teacher(self, momentum):
        for teacher_param, student_param in zip(self.teacher_encoder.parameters(), self.encoder.parameters()):
            teacher_param.data.mul_(momentum).add_(student_param.data, alpha=1.0 - momentum)
        for teacher_buffer, student_buffer in zip(self.teacher_encoder.buffers(), self.encoder.buffers()):
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
    ):
        bsz, num_cand = cand_mask.shape
        if key_bank is None:
            raise ValueError('full-memory Stage-1 requires a relation key memory bank')
        if self.requires_ema_teacher_bank() and teacher_key_bank is None:
            raise ValueError('stage1_teacher_mode=ema_target requires a teacher key memory bank')

        valid_query = cand_mask.sum(dim=1) > 0
        if valid_query.sum() == 0:
            zero = query_x.sum() * 0.0
            return zero, {'skipped_batches': 1.0}

        if not self._shape_logged:
            print(f'[stage1] batch_x={tuple(query_x.shape)} batch_y={tuple(query_y.shape)}')
            print(f'[stage1] key_bank={tuple(key_bank.shape)} memory_y={tuple(memory_y.shape)} mask={tuple(cand_mask.shape)}')
            if teacher_key_bank is not None:
                print(f'[stage1] teacher_key_bank={tuple(teacher_key_bank.shape)} teacher_mode={self.teacher_mode}')
            print(f'[stage1] self_relation={(bsz, 1, self.seq_len)} cross_relation={(bsz, 2, self.seq_len)}')
            self._shape_logged = True

        masked_fill = torch.finfo(query_x.dtype).min / 4
        losses = []
        kl_losses = []
        rank_losses = []
        rnc_losses = []
        expected_mse_losses = []
        metric_rows = []
        self_rows = []
        cross_rows = []

        targets = self.target_channels() if active_target_channels is None else active_target_channels
        for c in targets:
            if self.loss_mode == 'rnc':
                future_mse = self._future_mse(
                    query_x, query_y, memory_y, memory_x_last, c
                )
                if self.rnc_quality_source == 'ema_cosine':
                    teacher_scores = self._teacher_embedding_scores(
                        query_x, query_y, teacher_key_bank, c
                    )
                    rnc_quality_distance = (1.0 - teacher_scores).detach()
                else:
                    rnc_quality_distance = future_mse
                rnc_targets = prepare_query_conditioned_rnc_targets(
                    rnc_quality_distance,
                    cand_mask,
                    tie_epsilon=self.rnc_tie_epsilon,
                )
            else:
                mse_teacher_logits, future_mse = self._teacher_logits(
                    query_x, query_y, memory_y, memory_x_last, c
                )
                if self.teacher_mode == 'ema_target':
                    teacher_logits = self._teacher_embedding_logits(
                        query_x, query_y, teacher_key_bank, c
                    )
                else:
                    teacher_logits = mse_teacher_logits
                teacher_logits = teacher_logits.masked_fill(~cand_mask, masked_fill)
                teacher_prob = torch.softmax(teacher_logits, dim=-1).detach()
                if compute_detailed_metrics:
                    teacher_entropy = -(
                        teacher_prob * torch.log(teacher_prob + self.eps)
                    ).sum(dim=-1)
                    oracle_rank = torch.argmin(
                        future_mse.masked_fill(~cand_mask, float('inf')), dim=-1
                    )
                    teacher_rank = torch.argmax(teacher_prob, dim=-1)
                    random_mse = (
                        future_mse.masked_fill(~cand_mask, 0.0).sum(dim=-1)
                        / cand_mask.sum(dim=-1).clamp_min(1)
                    ).detach()

            for source_slot, r in enumerate(self.source_channels(c)):
                q_rel = self._relation_tensor(query_x, c, r)
                z_q = self.encoder(q_rel)
                z_k = key_bank[c, source_slot].to(
                    device=query_x.device, dtype=z_q.dtype
                )

                student_scores = torch.matmul(z_q, z_k.transpose(0, 1))
                if self.loss_mode == 'rnc':
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
                    row = {
                        'stage1_loss_total': rnc_loss.detach(),
                        'stage1_loss_kl': zero_metric,
                        'stage1_loss_rank': zero_metric,
                        'stage1_loss_rank_weighted': zero_metric,
                        'total_loss': rnc_loss.detach(),
                        'kl_loss': zero_metric,
                        'weighted_kl_loss': zero_metric,
                        'rank_loss': zero_metric,
                        'rnc_loss': rnc_loss.detach(),
                        'expected_mse_loss': zero_metric,
                        'weighted_expected_mse_loss': zero_metric,
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
                    losses.append(rnc_loss)
                    rnc_losses.append(rnc_loss)
                    metric_rows.append(row)
                    (self_rows if c == r else cross_rows).append(row)
                    continue

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

                if self.loss_mode == 'kl':
                    total_loss = kl_loss
                elif self.loss_mode == 'kl_rank':
                    total_loss = kl_loss + self.rank_weight * rank_loss
                    rank_losses.append(rank_loss)
                else:
                    total_loss = (
                        (1.0 - self.expected_mse_weight) * kl_loss
                        + self.expected_mse_weight * expected_mse_loss
                    )
                    expected_mse_losses.append(expected_mse_loss)
                losses.append(total_loss)
                kl_losses.append(kl_loss)

                row = {
                    'kl': kl.detach().mean(),
                    'stage1_loss_total': total_loss.detach(),
                    'stage1_loss_kl': kl_loss.detach(),
                    'stage1_loss_rank': rank_metrics['stage1_loss_rank'],
                    'stage1_loss_rank_weighted': (self.rank_weight * rank_loss).detach(),
                    'total_loss': total_loss.detach(),
                    'kl_loss': kl_loss.detach(),
                    'weighted_kl_loss': (
                        (1.0 - self.expected_mse_weight) * kl_loss
                        if self.loss_mode == 'kl_expected_mse' else kl_loss
                    ).detach(),
                    'rank_loss': rank_loss.detach(),
                    'rnc_loss': (student_scores.sum() * 0.0).detach(),
                    'expected_mse_loss': expected_mse_loss.detach(),
                    'weighted_expected_mse_loss': (
                        self.expected_mse_weight * expected_mse_loss
                    ).detach(),
                }
                row.update(rank_metrics)
                row.update(expected_mse_metrics)
                if compute_detailed_metrics:
                    retrieval_metrics = _student_retrieval_metrics(
                        student_scores, student_prob, future_mse, cand_mask, eps=self.eps
                    )
                    top1_student = torch.argmax(student_prob, dim=-1)
                    top5_student = torch.topk(
                        student_prob, k=min(5, num_cand), dim=-1
                    ).indices
                    top5_teacher = torch.topk(
                        teacher_prob, k=min(5, num_cand), dim=-1
                    ).indices
                    top1_match = (top1_student == oracle_rank).float()
                    teacher_top1_match = (top1_student == teacher_rank).float()
                    recall5 = (top5_student == oracle_rank[:, None]).any(dim=-1).float()
                    top5_overlap = (
                        top5_student[:, :, None] == top5_teacher[:, None, :]
                    ).any(dim=-1).float().mean(dim=-1)
                    top1_mse = future_mse.gather(1, top1_student[:, None]).squeeze(1)
                    topk_weighted = (
                        student_prob * future_mse.masked_fill(~cand_mask, 0.0)
                    ).sum(dim=-1)
                    student_entropy = -(
                        student_prob * student_log_prob
                    ).masked_fill(~cand_mask, 0.0).sum(dim=-1)
                    prob_l1 = torch.abs(student_prob - teacher_prob).masked_fill(
                        ~cand_mask, 0.0
                    ).sum(dim=-1)
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
                        'teacher_student_prob_l1': prob_l1[valid_query].detach().mean(),
                        'teacher_student_top5_overlap': top5_overlap[
                            valid_query
                        ].detach().mean(),
                        'student_teacher_top1_match': teacher_top1_match[
                            valid_query
                        ].detach().mean(),
                        'top1_teacher_rank_match': top1_match[valid_query].detach().mean(),
                        'recall@1': top1_match[valid_query].detach().mean(),
                        'recall@5': recall5[valid_query].detach().mean(),
                        'retrieved_future_mse_top1': top1_mse[valid_query].detach().mean(),
                        'retrieved_future_mse_topk_weighted': topk_weighted[
                            valid_query
                        ].detach().mean(),
                        'random_future_mse': random_mse[valid_query].detach().mean(),
                    })
                    row.update(retrieval_metrics)
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
            if self.loss_mode == 'kl_expected_mse' else metrics['kl_loss']
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
        if self.teacher_mode == 'ema_target' and teacher_key_bank is None:
            raise ValueError('stage1_teacher_mode=ema_target requires a teacher key memory bank')

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

        masked_fill = torch.finfo(query_x.dtype).min / 4
        mse_teacher_logits, future_mse = self._teacher_logits(query_x, query_y, memory_y, memory_x_last, c)
        if self.teacher_mode == 'ema_target':
            teacher_logits = self._teacher_embedding_logits(query_x, query_y, teacher_key_bank, c)
        else:
            teacher_logits = mse_teacher_logits
        teacher_logits = teacher_logits.masked_fill(~cand_mask, masked_fill)
        teacher_prob = torch.softmax(teacher_logits, dim=-1)

        q_rel = self._relation_tensor(query_x, c, r)
        z_q = self.encoder(q_rel)
        source_slot = self.source_slot(c, r)
        z_k = key_bank[c, source_slot].to(device=query_x.device, dtype=z_q.dtype)
        student_logits = torch.matmul(z_q, z_k.transpose(0, 1)) / self.tau_student
        student_logits = student_logits.masked_fill(~cand_mask, masked_fill)
        student_prob = torch.softmax(student_logits, dim=-1)

        valid_mask = cand_mask[q_idx]
        valid_count = int(valid_mask.sum().item())
        if valid_count == 0:
            return None
        k = min(max(int(top_n), 1), valid_count)
        ranked = torch.topk(teacher_prob[q_idx].masked_fill(~valid_mask, -1.0), k=k, dim=-1).indices
        top5_student = torch.topk(student_prob[q_idx].masked_fill(~valid_mask, -1.0), k=min(5, valid_count), dim=-1).indices
        top5_teacher = ranked[:min(5, k)]
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
        }

    def _average_metrics(self, rows):
        if not rows:
            return {}
        return {k: torch.stack([row[k] for row in rows]).mean() for k in rows[0]}

    def _prefixed_average(self, prefix, rows, device):
        if not rows:
            return {prefix + 'kl': torch.tensor(0.0, device=device)}
        return {prefix + k: v for k, v in self._average_metrics(rows).items()}
