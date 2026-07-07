import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from layers.relation_patch_embed import RelationPatchEmbedding


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
        self.target_mode = configs.target_mode
        self.target_channel = configs.target_channel
        self.key_chunk_size = int(getattr(configs, 'stage1_key_chunk_size', 1024))
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

    def source_channels(self, target_channel):
        if self.source_mode != 'all':
            raise NotImplementedError('source_mode=topk_corr is reserved for Stage-2/source selection work')
        return list(range(self.channels))

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

    def _teacher_logits(self, query_x, query_y, memory_y, memory_x_last, target_channel):
        q = query_y[:, :, target_channel]
        k = memory_y[:, :, target_channel]
        if self.teacher_mse_space not in ('normalized', 'raw'):
            raise ValueError(f'Unsupported teacher_mse_space: {self.teacher_mse_space}')
        if self.relation_teacher_space == 'delta_last':
            if memory_x_last is None:
                raise ValueError('relation_teacher_space=delta_last requires memory_x_last')
            q = q - query_x[:, -1:, target_channel].detach()
            k = k - memory_x_last[:, target_channel].to(memory_y.device).unsqueeze(-1)
        # MSE(q, k) over H without materializing [B, N, H]. This is also
        # retained as a teacher-independent quality metric for Pearson mode.
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
    def _teacher_embedding_logits(self, query_x, query_y, teacher_key_bank, target_channel):
        query_offset = query_x[:, -1, :]
        q_rel = self._teacher_target_relation(query_y, target_channel, query_offset)
        z_q = self.teacher_encoder(q_rel)
        z_k = teacher_key_bank[target_channel].to(query_y.device)
        return torch.matmul(z_q, z_k.transpose(0, 1)) / self.tau_teacher

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
        """Build stale relation-wise key bank [C, C, N, D] for one epoch."""
        was_training = self.training
        self.eval()
        chunk_size = int(chunk_size or self.key_chunk_size)
        memory_x = torch.as_tensor(memory_x, dtype=torch.float32)
        banks = []

        for c in range(self.channels):
            source_banks = []
            for r in range(self.channels):
                encoded = []
                for start in range(0, memory_x.size(0), chunk_size):
                    cur = memory_x[start:start + chunk_size].to(device)
                    rel = self._relation_tensor(cur, c, r)
                    encoded.append(self.encoder(rel).cpu())
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
                cur = memory_y[start:start + chunk_size].to(device)
                cur_offset = None if memory_x_last is None else memory_x_last[start:start + chunk_size].to(device)
                rel = self._teacher_target_relation(cur, c, cur_offset)
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

    def forward(self, query_x, query_y, cand_mask, memory_y, key_bank, teacher_key_bank=None, memory_x_last=None):
        bsz, num_cand = cand_mask.shape
        if key_bank is None:
            raise ValueError('full-memory Stage-1 requires a relation key memory bank')
        if self.teacher_mode == 'ema_target' and teacher_key_bank is None:
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
        metric_rows = []
        self_rows = []
        cross_rows = []

        for c in self.target_channels():
            mse_teacher_logits, future_mse = self._teacher_logits(query_x, query_y, memory_y, memory_x_last, c)
            if self.teacher_mode == 'ema_target':
                teacher_logits = self._teacher_embedding_logits(query_x, query_y, teacher_key_bank, c)
            else:
                teacher_logits = mse_teacher_logits
            teacher_logits = teacher_logits.masked_fill(~cand_mask, masked_fill)
            teacher_prob = torch.softmax(teacher_logits, dim=-1).detach()
            teacher_entropy = -(teacher_prob * torch.log(teacher_prob + self.eps)).sum(dim=-1)
            oracle_rank = torch.argmin(future_mse.masked_fill(~cand_mask, float('inf')), dim=-1)
            teacher_rank = torch.argmax(teacher_prob, dim=-1)
            random_mse = (future_mse.masked_fill(~cand_mask, 0.0).sum(dim=-1) / cand_mask.sum(dim=-1).clamp_min(1)).detach()

            for r in self.source_channels(c):
                q_rel = self._relation_tensor(query_x, c, r)
                z_q = self.encoder(q_rel)
                z_k = key_bank[c, r].to(query_x.device)

                student_logits = torch.matmul(z_q, z_k.transpose(0, 1)) / self.tau_student
                student_logits = student_logits.masked_fill(~cand_mask, masked_fill)
                student_log_prob = torch.log_softmax(student_logits, dim=-1)
                student_prob = student_log_prob.exp()

                kl = (teacher_prob * (torch.log(teacher_prob + self.eps) - student_log_prob)).sum(dim=-1)
                kl = kl[valid_query]
                if kl.numel() == 0 or not torch.isfinite(kl).all():
                    continue
                losses.append(kl.mean())

                top1_student = torch.argmax(student_prob, dim=-1)
                top5_student = torch.topk(student_prob, k=min(5, num_cand), dim=-1).indices
                top5_teacher = torch.topk(teacher_prob, k=min(5, num_cand), dim=-1).indices
                top1_match = (top1_student == oracle_rank).float()
                teacher_top1_match = (top1_student == teacher_rank).float()
                recall5 = (top5_student == oracle_rank[:, None]).any(dim=-1).float()
                top5_overlap = (
                    top5_student[:, :, None] == top5_teacher[:, None, :]
                ).any(dim=-1).float().mean(dim=-1)
                top1_mse = future_mse.gather(1, top1_student[:, None]).squeeze(1)
                topk_weighted = (student_prob * future_mse.masked_fill(~cand_mask, 0.0)).sum(dim=-1)
                student_entropy = -(student_prob * student_log_prob).masked_fill(~cand_mask, 0.0).sum(dim=-1)
                prob_l1 = torch.abs(student_prob - teacher_prob).masked_fill(~cand_mask, 0.0).sum(dim=-1)
                teacher_top1_prob = teacher_prob.max(dim=-1).values
                student_top1_prob = student_prob.max(dim=-1).values
                student_prob_on_teacher_top1 = student_prob.gather(1, teacher_rank[:, None]).squeeze(1)

                row = {
                    'kl': kl.detach().mean(),
                    'teacher_entropy': teacher_entropy[valid_query].detach().mean(),
                    'student_entropy': student_entropy[valid_query].detach().mean(),
                    'teacher_effective_candidates': torch.exp(teacher_entropy[valid_query]).detach().mean(),
                    'student_effective_candidates': torch.exp(student_entropy[valid_query]).detach().mean(),
                    'teacher_top1_prob': teacher_top1_prob[valid_query].detach().mean(),
                    'student_top1_prob': student_top1_prob[valid_query].detach().mean(),
                    'student_prob_on_teacher_top1': student_prob_on_teacher_top1[valid_query].detach().mean(),
                    'teacher_student_prob_l1': prob_l1[valid_query].detach().mean(),
                    'teacher_student_top5_overlap': top5_overlap[valid_query].detach().mean(),
                    'student_teacher_top1_match': teacher_top1_match[valid_query].detach().mean(),
                    'top1_teacher_rank_match': top1_match[valid_query].detach().mean(),
                    'recall@1': top1_match[valid_query].detach().mean(),
                    'recall@5': recall5[valid_query].detach().mean(),
                    'retrieved_future_mse_top1': top1_mse[valid_query].detach().mean(),
                    'retrieved_future_mse_topk_weighted': topk_weighted[valid_query].detach().mean(),
                    'random_future_mse': random_mse[valid_query].detach().mean(),
                }
                row['retrieval_gain'] = row['random_future_mse'] - row['retrieved_future_mse_topk_weighted']
                metric_rows.append(row)
                (self_rows if c == r else cross_rows).append(row)

        if not losses:
            zero = query_x.sum() * 0.0
            return zero, {'skipped_batches': 1.0}

        loss = torch.stack(losses).mean()
        metrics = self._average_metrics(metric_rows)
        metrics['loss'] = loss.detach()
        metrics['skipped_batches'] = torch.tensor(0.0, device=query_x.device)
        metrics.update(self._prefixed_average('self_', self_rows, query_x.device))
        metrics.update(self._prefixed_average('cross_', cross_rows, query_x.device))
        return loss, metrics

    def _average_metrics(self, rows):
        if not rows:
            return {}
        return {k: torch.stack([row[k] for row in rows]).mean() for k in rows[0]}

    def _prefixed_average(self, prefix, rows, device):
        if not rows:
            return {prefix + 'kl': torch.tensor(0.0, device=device)}
        return {prefix + k: v for k, v in self._average_metrics(rows).items()}
