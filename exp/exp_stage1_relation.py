import json
import os
from pathlib import Path
import time
import math

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models.RelationStage1 import relation_bank_collapse_metrics
from utils.relation_memory import RelationMemorySampler, build_memory_index
from utils.relation_graph import load_or_build_relation_graph
from utils.stage1_metrics import MetricAverager, format_metrics
from utils.tensorboard_logger import build_summary_writer, write_metric_scalars
from utils.tools import adjust_learning_rate


TINY_STEP_METRIC_KEYS = [
    'topk_coverage_loss',
    'student_oracle_recall_at_1',
    'student_oracle_recall_at_5',
    'student_oracle_recall_at_10',
    'coverage_oracle_student_overlap',
    'student_retrieval_regret_at_10',
    'encoder_grad_norm',
    'student_retrieved_future_mse_at_10',
    'oracle_future_mse_at_10',
    'student_oracle_topk_probability_mass_at_10',
]


class _FixedBatchSampler:
    def __init__(self, batch_size, steps):
        self.indices = list(range(int(batch_size)))
        self.steps = int(steps)

    def __iter__(self):
        for _ in range(self.steps):
            yield self.indices

    def __len__(self):
        return self.steps


class Exp_Stage1_Relation(Exp_Basic):
    def __init__(self, args):
        super(Exp_Stage1_Relation, self).__init__(args)
        self.train_data_for_memory = None
        self.memory_sampler = None
        self.key_bank = None
        self.teacher_key_bank = None
        self.memory_y = None
        self.memory_x = None
        self.memory_x_last = None
        self.memory_x_np = None
        self.memory_y_np = None
        self.memory_x_last_np = None
        self.tiny_candidate_indices = None
        self.global_update_step = 0
        self.total_update_steps = 1
        self.val_probe_batch = None
        self.relation_graph = None
        self.bank_collapse_metrics = {}
        self.writer = None
        self.tiny_step = 0

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, shuffle=None):
        return data_provider(self.args, flag, shuffle=shuffle)

    def _select_optimizer(self):
        return optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=self.args.learning_rate)

    def _ensure_memory(self):
        if self.memory_sampler is not None:
            return
        train_data, _ = self._get_data(flag='train', shuffle=False)
        self.train_data_for_memory = train_data
        self.memory_sampler = RelationMemorySampler(
            train_data,
            seq_len=self.args.seq_len,
            pred_len=self.args.pred_len,
            mask_mode=self.args.candidate_mask,
        )
        self.relation_graph = load_or_build_relation_graph(
            train_data, self.args, require_existing=False
        )
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.set_relation_graph(self.relation_graph)
        if self.relation_graph is not None:
            self.args.relation_channel_names = self.relation_graph['channel_names']
        self.memory_x_np = self.memory_sampler.memory_x
        self.memory_y_np = self.memory_sampler.memory_y
        self.memory_x_last_np = self.memory_sampler.memory_x[:, -1, :]
        self._sync_memory_tensors()

    def _sync_memory_tensors(self):
        self.memory_y = torch.from_numpy(self.memory_y_np).float().to(self.device)
        self.memory_x_last = torch.from_numpy(self.memory_x_last_np).float().to(self.device)
        # Candidate histories only reach the device for the modes that re-encode
        # them during training; the full-bank baseline never needs them.
        needs_history = self._candidate_subset_enabled() or (
            self._differentiable_keys_enabled()
            and self.tiny_candidate_indices is not None
        )
        self.memory_x = (
            torch.from_numpy(self.memory_x_np).float().to(self.device)
            if needs_history
            else None
        )
        if needs_history:
            print(
                f'[stage1] candidate history on device: {tuple(self.memory_x.shape)} '
                f'({self.memory_x.element_size() * self.memory_x.nelement() / 2**20:.0f} MiB)'
            )

    def _tiny_overfit_enabled(self):
        return int(getattr(self.args, 'stage1_overfit_queries', 0)) > 0

    def _candidate_subset_enabled(self):
        return getattr(self.args, 'stage1_candidate_subset_mode', 'none') != 'none'

    def _differentiable_keys_enabled(self):
        return (
            self._tiny_overfit_enabled()
            and bool(int(getattr(self.args, 'stage1_overfit_differentiable_keys', 0)))
        )

    def _direct_eval_enabled(self):
        return bool(int(getattr(self.args, 'stage1_direct_eval', 0)))

    def _configure_tiny_overfit(self, train_data):
        query_count = int(self.args.stage1_overfit_queries)
        candidate_count = int(self.args.stage1_overfit_candidates)
        steps = int(self.args.stage1_overfit_steps)
        oracle_per_query = int(self.args.stage1_overfit_oracle_per_query)
        if candidate_count <= 0 or steps <= 0:
            raise ValueError(
                'tiny-set overfit requires positive --stage1_overfit_candidates '
                'and --stage1_overfit_steps'
            )
        if self.args.target_mode != 'single' or self.args.target_channel is None:
            raise ValueError(
                'tiny-set overfit requires --target_mode single and --target_channel'
            )
        if query_count > len(train_data):
            raise ValueError(
                f'stage1_overfit_queries={query_count} exceeds train size={len(train_data)}'
            )

        query_indices = torch.linspace(0, len(train_data) - 1, steps=query_count).long().unique()
        if query_indices.numel() != query_count:
            raise ValueError('could not construct the requested number of unique tiny-set queries')
        query_starts = torch.as_tensor(
            [int(train_data[int(index)][2]) for index in query_indices], dtype=torch.long
        )
        full_mask, _ = self.memory_sampler.valid_mask_batch(query_starts.numpy())
        eligible_candidates = torch.nonzero(
            full_mask.any(dim=0), as_tuple=False
        ).flatten()
        if eligible_candidates.numel() < candidate_count:
            raise ValueError(
                f'only {eligible_candidates.numel()} candidates are valid for the tiny queries; '
                f'requested {candidate_count}'
            )

        query_x = torch.stack([
            torch.as_tensor(train_data[int(index)][0], dtype=torch.float32)
            for index in query_indices
        ])
        query_y = torch.stack([
            torch.as_tensor(train_data[int(index)][1], dtype=torch.float32)
            for index in query_indices
        ])
        target = int(self.args.target_channel)
        candidate_y = torch.from_numpy(self.memory_sampler.memory_y).float()
        q_future = query_y[:, :, target]
        k_future = candidate_y[:, :, target]
        if self.args.relation_teacher_space == 'delta_last':
            q_future = q_future - query_x[:, -1:, target]
            candidate_last = torch.from_numpy(
                self.memory_sampler.memory_x[:, -1, target]
            ).float()
            k_future = k_future - candidate_last.unsqueeze(-1)
        future_mse = (
            q_future.square().mean(dim=-1, keepdim=True)
            + k_future.square().mean(dim=-1).unsqueeze(0)
            - 2.0 * torch.matmul(q_future, k_future.transpose(0, 1)) / q_future.size(-1)
        ).clamp_min(0.0)
        future_mse = future_mse.masked_fill(~full_mask, float('inf'))

        ranked_candidates = torch.argsort(future_mse, dim=1)
        selected_candidates = []
        selected_set = set()
        oracle_depth = min(max(oracle_per_query, 1), ranked_candidates.size(1))
        for rank in range(oracle_depth):
            for row in range(ranked_candidates.size(0)):
                candidate_index = int(ranked_candidates[row, rank])
                if candidate_index not in selected_set:
                    selected_set.add(candidate_index)
                    selected_candidates.append(candidate_index)
                    if len(selected_candidates) == candidate_count:
                        break
            if len(selected_candidates) == candidate_count:
                break

        if len(selected_candidates) < candidate_count:
            generator = torch.Generator().manual_seed(0)
            random_order = torch.randperm(
                eligible_candidates.numel(), generator=generator
            )
            for position in random_order.tolist():
                candidate_index = int(eligible_candidates[position])
                if candidate_index not in selected_set:
                    selected_set.add(candidate_index)
                    selected_candidates.append(candidate_index)
                    if len(selected_candidates) == candidate_count:
                        break

        self.tiny_candidate_indices = torch.as_tensor(
            selected_candidates, dtype=torch.long
        )
        candidate_np = self.tiny_candidate_indices.numpy()
        self.memory_x_np = self.memory_sampler.memory_x[candidate_np]
        self.memory_y_np = self.memory_sampler.memory_y[candidate_np]
        self.memory_x_last_np = self.memory_sampler.memory_x[candidate_np, -1, :]
        self._sync_memory_tensors()

        model = self.model.module if hasattr(self.model, 'module') else self.model
        if bool(int(self.args.stage1_overfit_self_only)):
            model.relation_sources = [[channel] for channel in range(model.channels)]

        tiny_subset = Subset(train_data, query_indices.tolist())
        train_loader = DataLoader(
            tiny_subset,
            batch_sampler=_FixedBatchSampler(query_count, steps),
            num_workers=self.args.num_workers,
        )
        eval_loader = DataLoader(
            tiny_subset,
            batch_size=query_count,
            shuffle=False,
            num_workers=self.args.num_workers,
            drop_last=False,
        )
        oracle_covered = sum(
            1 for row in range(ranked_candidates.size(0))
            if int(ranked_candidates[row, 0]) in selected_set
        )
        selected_valid_counts = full_mask.index_select(
            1, self.tiny_candidate_indices
        ).sum(dim=1)
        print(
            '[stage1 tiny-overfit] '
            f'queries={query_count} candidates={candidate_count} steps={steps} '
            f'target={target} self_only={bool(int(self.args.stage1_overfit_self_only))} '
            f'key_refresh={"differentiable" if self._differentiable_keys_enabled() else self.args.stage1_overfit_key_refresh} '
            f'input_space={self.args.relation_input_space} '
            f'teacher_space={self.args.relation_teacher_space} '
            f'oracle_top1_covered={oracle_covered}/{query_count} '
            f'valid_candidates_per_query='
            f'{int(selected_valid_counts.min())}-{int(selected_valid_counts.max())}'
        )
        return train_loader, eval_loader

    def _move_batch(self, batch_x, batch_y, batch_start_idx):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_start_idx = batch_start_idx.long()
        return batch_x, batch_y, batch_start_idx

    def _teacher_cache(self, split):
        """Precomputed teacher pool for a split, loaded once and kept.

        The pool ids are frozen at precomputation time, so every teacher arm in
        the ablation ranks exactly the same candidates and a difference between
        arms cannot be a difference in what they were shown.
        """
        root = getattr(self.args, 'stage1_teacher_cache', '')
        if not root:
            return None
        if not hasattr(self, '_teacher_caches'):
            self._teacher_caches = {}
        if split not in self._teacher_caches:
            from utils.utility_teacher import load_cache

            path = Path(root) / f'{split}.pt'
            if not path.exists():
                raise FileNotFoundError(f'teacher cache missing: {path}')
            cache = load_cache(path)
            meta = cache['meta']
            if meta['dataset'] != self.args.data or meta['pred_len'] != int(self.args.pred_len):
                raise ValueError(
                    f'teacher cache {path} was built for {meta["dataset"]}/'
                    f'{meta["pred_len"]}, not {self.args.data}/{self.args.pred_len}'
                )
            self._teacher_caches[split] = cache
            print(f'[stage1] teacher cache {path}: queries={meta["queries"]} '
                  f'pool={meta["pool_m"]} reference={meta.get("reference_stage2", "?")}')
        return self._teacher_caches[split]

    def _teacher_batch(self, split, batch_start_idx):
        """Pool ids, teacher scores and utilities for this batch, or empty dict."""
        cache = self._teacher_cache(split)
        if cache is None:
            return {}
        from utils.utility_teacher import rows_for_starts

        rows = rows_for_starts(cache, batch_start_idx)
        if rows is None:
            raise KeyError(
                f'teacher cache for {split} does not cover every window in this batch; '
                'rebuild it over the full split'
            )
        return {
            'external_pool': cache['pool'].index_select(0, rows).to(self.device),
            'external_teacher': cache['residual'].index_select(0, rows).to(self.device),
            'external_utility': cache['utility'].index_select(0, rows).to(self.device),
        }

    def _residual_cache(self):
        """Cached base-forecast residuals for the Residual-KL teacher."""
        root = getattr(self.args, 'stage1_residual_teacher_cache', '')
        if not root:
            return None
        if not hasattr(self, '_residual_cache_value'):
            from scripts.precompute_residual_teacher import load

            path = Path(root)
            if path.is_dir():
                path = path / f'{self.args.data}_pred{self.args.pred_len}.pt'
            cache = load(path)
            meta = cache['meta']
            if meta['dataset'] != self.args.data or meta['pred_len'] != int(self.args.pred_len):
                raise ValueError(
                    f'residual cache {path} was built for {meta["dataset"]}/{meta["pred_len"]}'
                )
            cache['memory_residual'] = cache['memory_residual'].to(self.device)
            self._residual_cache_value = cache
            print(f'[stage1] residual cache {path}: '
                  f'memory_residual={tuple(cache["memory_residual"].shape)} '
                  f'reference={meta.get("reference_stage2", "?")}')
        return self._residual_cache_value

    def _reference_key_bank(self):
        """Frozen key bank that defines the candidate pool, shared by every arm.

        Built once from a checkpoint that does not move, so the pool cannot drift
        with the encoder being trained -- which would make each arm's pool its own
        and destroy the comparison.
        """
        path = getattr(self.args, 'stage1_pool_reference_ckpt', '')
        if not path or int(getattr(self.args, 'stage1_pool_size', 0)) <= 0:
            return None
        if not hasattr(self, '_reference_key_bank_value'):
            import copy

            model = self.model.module if hasattr(self.model, 'module') else self.model
            reference = copy.deepcopy(model)
            state = torch.load(path, map_location='cpu')
            reference.load_state_dict(state.get('model_state_dict', state), strict=False)
            reference.eval().to(self.device)
            with torch.no_grad():
                bank = reference.build_embedding_bank(self.memory_x_np, self.device)
            self._reference_key_bank_value = bank
            self._reference_encoder = reference
            print(f'[stage1] reference pool bank {tuple(bank.shape)} from {path}')
        return self._reference_key_bank_value

    @torch.no_grad()
    def _reference_scores(self, batch_x):
        """Frozen-encoder similarity of each query against the whole bank."""
        bank = self._reference_key_bank()
        if bank is None:
            return None
        model = self._reference_encoder
        rows = []
        for c in model.target_channels():
            slot = model.source_slot(c, c)
            z_q = model.encoder(model._relation_tensor(batch_x, c, c))
            keys = bank[c, slot].to(device=z_q.device, dtype=z_q.dtype)
            rows.append(torch.matmul(z_q, keys.transpose(0, 1)))
        return torch.stack(rows, dim=1)

    def _residual_batch(self, split, batch_start_idx):
        cache = self._residual_cache()
        if cache is None:
            return {}
        part = cache['splits'][split]
        try:
            rows = [part['start_to_row'][int(v)] for v in batch_start_idx.cpu().tolist()]
        except KeyError:
            raise KeyError(f'residual cache for {split} does not cover this batch')
        index = torch.tensor(rows, dtype=torch.long)
        return {
            'query_residual': part['query_residual'].index_select(0, index).to(self.device),
            'memory_residual': cache['memory_residual'],
        }

    def _candidate_mask(self, batch_start_idx):
        cand_mask, counts = self.memory_sampler.valid_mask_batch(batch_start_idx.cpu().numpy())
        if self.tiny_candidate_indices is not None:
            cand_mask = cand_mask.index_select(1, self.tiny_candidate_indices)
            counts = cand_mask.sum(dim=1)
        return (
            cand_mask.bool().to(self.device),
            counts,
        )

    def _build_key_bank(self, log=True):
        model = self.model.module if hasattr(self.model, 'module') else self.model
        self.key_bank = None
        if self._direct_eval_enabled():
            self.key_bank = model.build_direct_embedding_bank(
                self.memory_x_np,
                self.device,
                chunk_size=self.args.stage1_key_chunk_size,
            ).to(device=self.device, dtype=torch.float32)
        else:
            self.key_bank = model.build_embedding_bank(
                self.memory_x_np,
                self.device,
                chunk_size=self.args.stage1_key_chunk_size,
            ).to(device=self.device, dtype=torch.float32)
        if log:
            print(
                f'[stage1] built relation key memory bank: {tuple(self.key_bank.shape)} '
                f'device={self.key_bank.device} dtype={self.key_bank.dtype}'
            )
        if model.requires_ema_teacher_bank():
            teacher_memory = (
                self.memory_x_np
                if model.teacher_mode == 'ema_input'
                else self.memory_y_np
            )
            self.teacher_key_bank = model.build_teacher_embedding_bank(
                teacher_memory,
                self.device,
                chunk_size=self.args.stage1_key_chunk_size,
                memory_x_last=self.memory_x_last_np,
            ).to(self.device)
            if log:
                print(
                    '[stage1] built relation-wise EMA teacher key memory bank: '
                    f'{tuple(self.teacher_key_bank.shape)} '
                    f'device={self.teacher_key_bank.device} dtype={self.teacher_key_bank.dtype}'
                )
        else:
            self.teacher_key_bank = None

        self.bank_collapse_metrics = {}
        if log and bool(int(getattr(self.args, 'stage1_collapse_metrics', 1))):
            collapse_kwargs = {
                'sample_size': int(self.args.stage1_collapse_sample_size),
                'dead_std_threshold': float(
                    self.args.stage1_collapse_dead_std_threshold
                ),
            }
            online_metrics = relation_bank_collapse_metrics(
                self.key_bank,
                **collapse_kwargs,
            )
            self.bank_collapse_metrics.update({
                f'online_collapse_{key}': value.detach().float().cpu().item()
                for key, value in online_metrics.items()
            })
            if self.teacher_key_bank is not None:
                ema_metrics = relation_bank_collapse_metrics(
                    self.teacher_key_bank,
                    **collapse_kwargs,
                )
                self.bank_collapse_metrics.update({
                    f'ema_collapse_{key}': value.detach().float().cpu().item()
                    for key, value in ema_metrics.items()
                })
            print(format_metrics('[stage1 collapse]', self.bank_collapse_metrics))

    def _ema_momentum(self):
        if self.total_update_steps <= 1:
            return float(self.args.stage1_ema_momentum_final)
        progress = min(float(self.global_update_step) / float(self.total_update_steps - 1), 1.0)
        base = float(self.args.stage1_ema_momentum_base)
        final = float(self.args.stage1_ema_momentum_final)
        return final - (final - base) * (math.cos(math.pi * progress) + 1.0) / 2.0

    def _update_ema_teacher(self):
        model = self.model.module if hasattr(self.model, 'module') else self.model
        if not model.requires_ema_teacher_bank():
            return None
        momentum = self._ema_momentum()
        model.update_ema_teacher(momentum)
        self.global_update_step += 1
        return momentum

    def _checkpoint_score(self, metrics):
        """Return (score, label); higher is always better.

        Selecting on validation loss picks the checkpoint whose distribution
        matches the teacher, which is not the same as the checkpoint that
        retrieves best. When the requested retrieval metric is missing the
        criterion falls back to loss rather than silently keeping epoch 1.
        """
        criterion = getattr(self.args, 'stage1_checkpoint_metric', 'loss')
        candidates = {
            'recall10': (
                ('student_oracle_recall_at_10', 'oracle_recall_at_10'), 1.0
            ),
            'retrieval_regret10': (
                ('student_retrieval_regret_at_10', 'retrieval_regret_at_10'), -1.0
            ),
            # Selecting on Future Recall picks the checkpoint that best copies the
            # Future-MSE Oracle, which the alignment study showed is not the one
            # that helps the forecast. These select on measured downstream gain.
            'utility_gap_recovery': (('utility_gap_recovery_at_10',), 1.0),
            'utility_ndcg': (('utility_ndcg_at_10',), 1.0),
            'retrieved_utility': (('utility_retrieved_at_10',), 1.0),
        }
        if criterion in candidates:
            keys, sign = candidates[criterion]
            for key in keys:
                if key in metrics:
                    return sign * float(metrics[key]), f'{criterion}({key})'
            print(
                f'[stage1] checkpoint metric {criterion} unavailable this epoch; '
                'falling back to validation loss'
            )
        return -float(metrics.get('loss', float('inf'))), 'loss'

    def _log_tiny_step(self, metrics, step):
        printable = {
            key: (value.item() if hasattr(value, 'item') else float(value))
            for key, value in metrics.items()
            if key in TINY_STEP_METRIC_KEYS
        }
        print(format_metrics(f'[tiny-step {step}]', printable))
        write_metric_scalars(
            self.writer, 'tiny_step', metrics, step, TINY_STEP_METRIC_KEYS
        )

    def _run_loader(self, loader, optimizer=None, compute_detailed_metrics=False,
                    split_name='train'):
        train = optimizer is not None
        self.model.train(train)
        model = self.model.module if hasattr(self.model, 'module') else self.model
        if model.requires_ema_teacher_bank():
            model.teacher_encoder.eval()
            model.teacher_shared_cross_projection.eval()
        avg = MetricAverager()

        all_targets = model.target_channels()
        target_chunk_size = int(getattr(self.args, 'relation_target_chunk_size', 0))
        if target_chunk_size <= 0 or target_chunk_size >= len(all_targets):
            target_chunks = [all_targets]
        else:
            target_chunks = [
                all_targets[start:start + target_chunk_size]
                for start in range(0, len(all_targets), target_chunk_size)
            ]

        differentiable_keys = self._differentiable_keys_enabled()
        log_every = int(getattr(self.args, 'stage1_overfit_log_every', 0))
        step_logging = train and self._tiny_overfit_enabled() and log_every > 0

        for batch_idx, (batch_x, batch_y, batch_start_idx) in enumerate(loader):
            if (
                train
                and self._tiny_overfit_enabled()
                and not differentiable_keys
                and self.args.stage1_overfit_key_refresh == 'step'
            ):
                self._build_key_bank(log=False)
            batch_x, batch_y, batch_start_idx = self._move_batch(batch_x, batch_y, batch_start_idx)
            cand_mask, counts = self._candidate_mask(batch_start_idx)
            teacher_batch = self._teacher_batch(split_name, batch_start_idx)
            teacher_batch.update(self._residual_batch(split_name, batch_start_idx))
            reference_scores = self._reference_scores(batch_x)
            if reference_scores is not None:
                teacher_batch['reference_scores'] = reference_scores
            metrics_extra = {
                'valid_candidate_count_mean': counts.float().mean().item(),
                'valid_candidate_count_min': counts.min().item() if counts.numel() else 0.0,
            }

            if train:
                self.tiny_step += 1
                log_this_step = step_logging and (
                    self.tiny_step == 1 or self.tiny_step % log_every == 0
                )
                optimizer.zero_grad()
                loss, metrics = self.model(
                    batch_x, batch_y, cand_mask,
                    memory_y=self.memory_y,
                    key_bank=self.key_bank,
                    teacher_key_bank=self.teacher_key_bank,
                    memory_x_last=self.memory_x_last,
                    active_target_channels=target_chunks[batch_idx % len(target_chunks)],
                    compute_detailed_metrics=log_this_step,
                    direct_retrieval=self._direct_eval_enabled(),
                    candidate_x=self.memory_x,
                    differentiable_keys=differentiable_keys,
                    **teacher_batch,
                )
                if batch_idx == 0 and self._candidate_subset_enabled():
                    if not loss.requires_grad:
                        raise RuntimeError(
                            'candidate subset training produced a loss detached from '
                            'the encoder; check the candidate re-encoding path'
                        )
                if torch.isfinite(loss) and loss.requires_grad:
                    loss.backward()
                    grad_sq = torch.zeros((), device=self.device)
                    grad_modules = (model.encoder, model.shared_cross_projection)
                    for module in grad_modules:
                        for param in module.parameters():
                            if param.grad is not None:
                                grad_sq = grad_sq + param.grad.detach().float().pow(2).sum()
                    metrics = dict(metrics)
                    metrics['encoder_grad_norm'] = torch.sqrt(grad_sq)
                    optimizer.step()
                    ema_momentum = self._update_ema_teacher()
                    if ema_momentum is not None:
                        metrics = dict(metrics)
                        metrics['ema_momentum'] = ema_momentum
                if log_this_step:
                    self._log_tiny_step(metrics, self.tiny_step)
            else:
                with torch.no_grad():
                    loss, metrics = self.model(
                        batch_x, batch_y, cand_mask,
                        memory_y=self.memory_y,
                        key_bank=self.key_bank,
                        teacher_key_bank=self.teacher_key_bank,
                        memory_x_last=self.memory_x_last,
                        active_target_channels=target_chunks[batch_idx % len(target_chunks)],
                        compute_detailed_metrics=compute_detailed_metrics,
                        direct_retrieval=self._direct_eval_enabled(),
                        candidate_x=self.memory_x,
                        differentiable_keys=differentiable_keys,
                        **teacher_batch,
                    )
                    if model.requires_ema_teacher_bank():
                        metrics = dict(metrics)
                        metrics['ema_momentum'] = self._ema_momentum()

            metrics = dict(metrics)
            metrics.update(metrics_extra)
            avg.update(metrics, weight=batch_x.size(0))

        metrics = avg.average()
        metrics.update(self.bank_collapse_metrics)
        return metrics

    def _set_validation_probe(self, vali_loader):
        model = self.model.module if hasattr(self.model, 'module') else self.model
        if model.loss_mode in ('rnc', 'topk_coverage'):
            return
        if not bool(int(getattr(self.args, 'stage1_probe_vis', 1))):
            return
        if self.val_probe_batch is not None:
            return
        try:
            batch = next(iter(vali_loader))
        except StopIteration:
            return
        self.val_probe_batch = batch

    def _plot_validation_probe(self, writer, setting, epoch):
        model = self.model.module if hasattr(self.model, 'module') else self.model
        if model.loss_mode in ('rnc', 'topk_coverage'):
            return
        if not bool(int(getattr(self.args, 'stage1_probe_vis', 1))):
            return
        if self.val_probe_batch is None:
            return
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            print('[stage1_probe] matplotlib is unavailable; skip validation probe plot')
            return

        was_training = self.model.training
        self.model.eval()
        batch_x, batch_y, batch_start_idx = self.val_probe_batch
        batch_x, batch_y, batch_start_idx = self._move_batch(batch_x, batch_y, batch_start_idx)
        cand_mask, _ = self._candidate_mask(batch_start_idx)
        with torch.no_grad():
            probe = model.distribution_probe(
                batch_x,
                batch_y,
                cand_mask,
                memory_y=self.memory_y,
                key_bank=self.key_bank,
                teacher_key_bank=self.teacher_key_bank,
                memory_x_last=self.memory_x_last,
                target_channel=getattr(self.args, 'stage1_probe_target_channel', 0),
                source_channel=getattr(self.args, 'stage1_probe_source_channel', 0),
                query_index=getattr(self.args, 'stage1_probe_query', 0),
                top_n=getattr(self.args, 'stage1_probe_top_n', 50),
            )
        if was_training:
            self.model.train()
        if probe is None:
            return

        teacher = probe['teacher_prob'].numpy()
        student = probe['student_prob'].numpy()
        x = range(len(teacher))
        width = 0.42
        fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
        ax.bar([i - width / 2 for i in x], teacher, width=width, label='teacher', color='tab:blue', alpha=0.75)
        ax.bar([i + width / 2 for i in x], student, width=width, label='student', color='tab:orange', alpha=0.75)
        ax.set_title(
            'Stage1 val probe '
            f'q={int(probe["query_index"])} c={int(probe["target_channel"])} r={int(probe["source_channel"])} '
            f'top5_overlap={float(probe["top5_overlap"]):.3f} '
            f'p_student@teacher_top1={float(probe["student_prob_on_teacher_top1"]):.3f}'
        )
        ax.set_xlabel('teacher-ranked candidate')
        ax.set_ylabel('probability')
        ax.legend()

        out_dir = os.path.join(
            getattr(self.args, 'stage1_probe_dir', './stage1_vis'),
            self.args.data,
            f'seq{self.args.seq_len}_pred{self.args.pred_len}',
            setting,
        )
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f'epoch_{epoch:03d}_val_probe.png')
        fig.savefig(out_path, dpi=160)
        print(f'[stage1_probe] saved validation teacher/student distribution to {out_path}')
        if writer is not None:
            writer.add_figure('val_probe/teacher_student_topn', fig, epoch)
            writer.add_scalar('val_probe/top5_overlap', float(probe['top5_overlap']), epoch)
            writer.add_scalar(
                'val_probe/student_prob_on_teacher_top1',
                float(probe['student_prob_on_teacher_top1']),
                epoch,
            )
            for metric_name in (
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
                'teacher_student_topk_overlap_at_1',
                'teacher_student_topk_overlap_at_5',
                'teacher_student_topk_overlap_at_10',
            ):
                writer.add_scalar(
                    f'val_probe/{metric_name}',
                    float(probe[metric_name]),
                    epoch,
                )
        plt.close(fig)

    def vali(self, vali_data, vali_loader):
        return self._run_loader(
            vali_loader,
            optimizer=None,
            compute_detailed_metrics=True,
            split_name='val',
        )

    def train(self, setting):
        self._ensure_memory()
        train_data, train_loader = self._get_data(flag='train', shuffle=True)
        if self._tiny_overfit_enabled():
            train_loader, vali_loader = self._configure_tiny_overfit(train_data)
            vali_data = train_data
        else:
            vali_data, vali_loader = self._get_data(flag='val', shuffle=False)
        self._set_validation_probe(vali_loader)
        self.total_update_steps = max(1, len(train_loader) * int(self.args.train_epochs))

        path = os.path.join(self.args.checkpoints, 'stage1', self.args.data, f'seq{self.args.seq_len}_pred{self.args.pred_len}', setting)
        os.makedirs(path, exist_ok=True)
        optimizer = self._select_optimizer()

        best_val_loss = float('inf')
        best_val_score = -float('inf')
        best_path = os.path.join(path, 'checkpoint.pth')
        bad_epochs = 0
        writer = build_summary_writer(self.args, 'stage1', setting)
        self.writer = writer
        tiny_eval_history = []
        tb_keys = [
            'loss', 'kl', 'self_kl', 'cross_kl',
            'stage1_loss_total', 'stage1_loss_kl', 'stage1_loss_rank', 'stage1_loss_rank_weighted',
            'stage1_loss_infonce', 'stage1_loss_infonce_weighted',
            'stage1_loss_variance', 'stage1_loss_variance_weighted',
            'stage1_loss_covariance', 'stage1_loss_covariance_weighted',
            'embedding_std_mean',
            'total_loss', 'kl_loss', 'weighted_kl_loss', 'rank_loss', 'rnc_loss',
            'expected_mse_loss', 'weighted_expected_mse_loss',
            'topk_coverage_loss',
            'oracle_topk_probability_mass',
            'oracle_positive_probability_mean',
            'oracle_positive_probability_min',
            'coverage_effective_k',
            'coverage_oracle_student_overlap',
            'infonce_positive_probability_mass',
            'infonce_effective_positive_count',
            'infonce_oracle_student_topk_overlap',
            'retrieval_gain', 'self_retrieval_gain', 'cross_retrieval_gain',
            'recall@1', 'recall@5',
            'self_recall@1', 'self_recall@5',
            'cross_recall@1', 'cross_recall@5',
            'teacher_entropy', 'student_entropy',
            'teacher_entropy_normalized',
            'teacher_top5_probability_mass',
            'student_entropy_normalized', 'student_max_probability',
            'student_top5_probability_mass',
            'student_expected_future_mse_raw',
            'student_expected_future_mse_normalized',
            'teacher_effective_candidates', 'student_effective_candidates',
            'teacher_top1_prob', 'student_top1_prob',
            'student_prob_on_teacher_top1',
            'teacher_student_prob_l1',
            'teacher_student_kl_divergence',
            'student_teacher_kl_divergence',
            'teacher_student_js_divergence',
            'teacher_student_total_variation',
            'teacher_student_hellinger_distance',
            'teacher_student_probability_cosine',
            'teacher_student_entropy_gap',
            'teacher_student_entropy_abs_gap',
            'teacher_student_top5_overlap',
            'student_teacher_top1_match',
            'teacher_student_topk_overlap_at_1',
            'teacher_student_topk_overlap_at_5',
            'teacher_student_topk_overlap_at_10',
            'student_teacher_recall_at_1',
            'student_teacher_recall_at_5',
            'student_teacher_recall_at_10',
            'rank_teacher_student_topk_overlap',
            'rank_teacher_student_topk_overlap_count',
            'rank_missed_positive_count',
            'rank_hard_negative_count',
            'rank_valid_pair_count',
            'rank_pair_accuracy',
            'rank_score_gap',
            'rank_margin_satisfied_ratio',
            'rank_teacher_topk_future_mse',
            'rank_student_topk_future_mse',
            'rank_missed_positive_future_mse',
            'rank_hard_negative_future_mse',
            'rnc_valid_query_count', 'rnc_anchor_count',
            'oracle_recall_at_1', 'oracle_recall_at_5', 'oracle_recall_at_10',
            'oracle_best_hit_at_1', 'oracle_best_hit_at_5', 'oracle_best_hit_at_10',
            'topk_probability_mass_at_1', 'topk_probability_mass_at_5',
            'topk_probability_mass_at_10',
            'oracle_topk_probability_mass_at_1',
            'oracle_topk_probability_mass_at_5',
            'oracle_topk_probability_mass_at_10',
            'retrieved_future_mse_at_1', 'retrieved_future_mse_at_5',
            'retrieved_future_mse_at_10',
            'best_future_mse_at_1', 'best_future_mse_at_5',
            'best_future_mse_at_10',
            'oracle_future_mse_at_1', 'oracle_future_mse_at_5',
            'oracle_future_mse_at_10',
            'retrieval_regret_at_1', 'retrieval_regret_at_5',
            'retrieval_regret_at_10',
            'ndcg_at_5', 'ndcg_at_10', 'spearman_score_vs_negative_mse',
            'student_oracle_recall_at_1', 'student_oracle_recall_at_5',
            'student_oracle_recall_at_10',
            'student_oracle_best_hit_at_1', 'student_oracle_best_hit_at_5',
            'student_oracle_best_hit_at_10',
            'student_topk_probability_mass_at_1',
            'student_topk_probability_mass_at_5',
            'student_topk_probability_mass_at_10',
            'student_oracle_topk_probability_mass_at_1',
            'student_oracle_topk_probability_mass_at_5',
            'student_oracle_topk_probability_mass_at_10',
            'student_retrieved_future_mse_at_1',
            'student_retrieved_future_mse_at_5',
            'student_retrieved_future_mse_at_10',
            'student_best_future_mse_at_1', 'student_best_future_mse_at_5',
            'student_best_future_mse_at_10',
            'student_retrieval_regret_at_1', 'student_retrieval_regret_at_5',
            'student_retrieval_regret_at_10',
            'student_ndcg_at_5', 'student_ndcg_at_10',
            'student_spearman_score_vs_negative_mse',
            'student_oracle_spearman',
            'teacher_oracle_recall_at_1', 'teacher_oracle_recall_at_5',
            'teacher_oracle_recall_at_10',
            'teacher_oracle_best_hit_at_1', 'teacher_oracle_best_hit_at_5',
            'teacher_oracle_best_hit_at_10',
            'teacher_topk_probability_mass_at_1',
            'teacher_topk_probability_mass_at_5',
            'teacher_topk_probability_mass_at_10',
            'teacher_oracle_topk_probability_mass_at_1',
            'teacher_oracle_topk_probability_mass_at_5',
            'teacher_oracle_topk_probability_mass_at_10',
            'teacher_retrieved_future_mse_at_1',
            'teacher_retrieved_future_mse_at_5',
            'teacher_retrieved_future_mse_at_10',
            'teacher_best_future_mse_at_1', 'teacher_best_future_mse_at_5',
            'teacher_best_future_mse_at_10',
            'teacher_retrieval_regret_at_1', 'teacher_retrieval_regret_at_5',
            'teacher_retrieval_regret_at_10',
            'teacher_ndcg_at_5', 'teacher_ndcg_at_10',
            'teacher_spearman_score_vs_negative_mse',
            'teacher_oracle_spearman',
            'student_teacher_spearman',
            'oracle_mse_student_topk_overlap_at_1',
            'oracle_mse_student_topk_overlap_at_5',
            'oracle_mse_student_topk_overlap_at_10',
            'teacher_oracle_mse_topk_overlap_at_1',
            'teacher_oracle_mse_topk_overlap_at_5',
            'teacher_oracle_mse_topk_overlap_at_10',
            'oracle_cos_student_topk_overlap_at_1',
            'oracle_cos_student_topk_overlap_at_5',
            'oracle_cos_student_topk_overlap_at_10',
            'student_oracle_cos_recall_at_1',
            'student_oracle_cos_recall_at_5',
            'student_oracle_cos_recall_at_10',
            'teacher_oracle_cos_topk_overlap_at_1',
            'teacher_oracle_cos_topk_overlap_at_5',
            'teacher_oracle_cos_topk_overlap_at_10',
            'oracle_mse_oracle_cos_topk_overlap_at_1',
            'oracle_mse_oracle_cos_topk_overlap_at_5',
            'oracle_mse_oracle_cos_topk_overlap_at_10',
            'ema_momentum',
            'encoder_grad_norm',
            'bank_oracle_recall_at_10', 'bank_oracle_recall_at_100',
            'oracle_count_in_bank_top_m',
            'oracle_missing_count_before_injection',
            'candidate_unique_encoded',
            'online_collapse_pairwise_cosine_mean',
            'online_collapse_pairwise_cosine_mean_max',
            'online_collapse_pairwise_cosine_std',
            'online_collapse_embedding_variance_mean',
            'online_collapse_embedding_variance_min',
            'online_collapse_dimension_std_mean',
            'online_collapse_dead_dimension_fraction_mean',
            'online_collapse_dead_dimension_fraction_max',
            'online_collapse_effective_rank_mean',
            'online_collapse_effective_rank_min',
            'online_collapse_effective_rank_ratio_mean',
            'online_collapse_effective_rank_ratio_min',
            'online_collapse_top_eigenvalue_ratio_mean',
            'online_collapse_top_eigenvalue_ratio_max',
            'ema_collapse_pairwise_cosine_mean',
            'ema_collapse_pairwise_cosine_mean_max',
            'ema_collapse_pairwise_cosine_std',
            'ema_collapse_embedding_variance_mean',
            'ema_collapse_embedding_variance_min',
            'ema_collapse_dimension_std_mean',
            'ema_collapse_dead_dimension_fraction_mean',
            'ema_collapse_dead_dimension_fraction_max',
            'ema_collapse_effective_rank_mean',
            'ema_collapse_effective_rank_min',
            'ema_collapse_effective_rank_ratio_mean',
            'ema_collapse_effective_rank_ratio_min',
            'ema_collapse_top_eigenvalue_ratio_mean',
            'ema_collapse_top_eigenvalue_ratio_max',
        ]

        topk_teacher_student_keys = [
            *[
                f'teacher_student_topk_overlap_at_{k}'
                for k in (1, 5, 10)
            ],
            *[
                f'student_teacher_recall_at_{k}'
                for k in (1, 5, 10)
            ],
            *[
                f'{owner}_oracle_recall_at_{k}'
                for owner in ('student', 'teacher')
                for k in (1, 5, 10)
            ],
            *[
                f'{owner}_oracle_best_hit_at_{k}'
                for owner in ('student', 'teacher')
                for k in (1, 5, 10)
            ],
            *[
                f'{owner}_topk_probability_mass_at_{k}'
                for owner in ('student', 'teacher')
                for k in (1, 5, 10)
            ],
            *[
                f'{owner}_oracle_topk_probability_mass_at_{k}'
                for owner in ('student', 'teacher')
                for k in (1, 5, 10)
            ],
            *[
                f'{owner}_retrieved_future_mse_at_{k}'
                for owner in ('student', 'teacher')
                for k in (1, 5, 10)
            ],
            *[
                f'{owner}_best_future_mse_at_{k}'
                for owner in ('student', 'teacher')
                for k in (1, 5, 10)
            ],
            *[
                f'{owner}_retrieval_regret_at_{k}'
                for owner in ('student', 'teacher')
                for k in (1, 5, 10)
            ],
            *[
                f'{owner}_ndcg_at_{k}'
                for owner in ('student', 'teacher')
                for k in (5, 10)
            ],
            'student_spearman_score_vs_negative_mse',
            'teacher_spearman_score_vs_negative_mse',
            'student_oracle_spearman',
            'teacher_oracle_spearman',
            'student_teacher_spearman',
        ]
        tb_keys.extend(
            f'{relation_prefix}{key}'
            for relation_prefix in ('self_', 'cross_')
            for key in topk_teacher_student_keys
        )

        if self._tiny_overfit_enabled():
            self._build_key_bank()
            initial_metrics = self.vali(vali_data, vali_loader)
            print(format_metrics('Tiny Overfit Initial', initial_metrics))
            write_metric_scalars(writer, 'tiny_eval', initial_metrics, 0, tb_keys)
            tiny_eval_history.append((0, initial_metrics))

        try:
            for epoch in range(self.args.train_epochs):
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)
                epoch_time = time.time()
                phase_time = epoch_time
                self._build_key_bank()
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)
                key_bank_train_time = time.time() - phase_time
                phase_time = time.time()
                train_metrics = self._run_loader(train_loader, optimizer=optimizer)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)
                train_time = time.time() - phase_time
                phase_time = time.time()
                self._build_key_bank()
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)
                key_bank_val_time = time.time() - phase_time
                phase_time = time.time()
                val_metrics = self.vali(vali_data, vali_loader)
                if self.device.type == 'cuda':
                    torch.cuda.synchronize(self.device)
                val_time = time.time() - phase_time
                val_loss = val_metrics.get('loss', float('inf'))

                print(format_metrics(f'Epoch {epoch + 1} Train', train_metrics))
                eval_label = 'Tiny Eval' if self._tiny_overfit_enabled() else 'Vali'
                print(format_metrics(f'Epoch {epoch + 1} {eval_label}', val_metrics))
                print(
                    '[stage1 timing] key_bank_train={:.2f}s train={:.2f}s '
                    'key_bank_val={:.2f}s validation={:.2f}s'.format(
                        key_bank_train_time, train_time, key_bank_val_time, val_time
                    )
                )
                print('Epoch: {} cost time: {:.2f}s'.format(epoch + 1, time.time() - epoch_time))
                write_metric_scalars(writer, 'train', train_metrics, epoch + 1, tb_keys)
                eval_tag = 'tiny_eval' if self._tiny_overfit_enabled() else 'vali'
                write_metric_scalars(writer, eval_tag, val_metrics, epoch + 1, tb_keys)
                if self._tiny_overfit_enabled():
                    tiny_eval_history.append((epoch + 1, val_metrics))
                self._plot_validation_probe(writer, setting, epoch + 1)
                adjust_learning_rate(optimizer, epoch + 1, self.args)

                val_score, score_label = self._checkpoint_score(val_metrics)
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_val_loss = val_loss
                    bad_epochs = 0
                    print(
                        f'[stage1] new best on {score_label}: {val_score:+.6f} '
                        f'(epoch {epoch + 1})'
                    )
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'args': vars(self.args),
                        'relation_graph': self.relation_graph,
                        'epoch': epoch + 1,
                        'best_val_loss': best_val_loss,
                    }, best_path)
                    print(f'Saved best Stage-1 checkpoint to {best_path}')
                else:
                    bad_epochs += 1
                    if bad_epochs >= self.args.patience:
                        print('Early stopping')
                        break
        finally:
            if self._tiny_overfit_enabled():
                self._report_tiny_overfit_summary(setting, tiny_eval_history)
            if writer is not None:
                writer.close()
            self.writer = None

        if getattr(self.args, 'build_memory_index', False):
            build_memory_index(self.model, train_data, self.args)
        return self.model

    def _report_tiny_overfit_summary(self, setting, history):
        """Print, and optionally persist, the tiny-set train Recall@1/5/10."""
        if not history:
            return

        def recall(metrics, k):
            for key in (f'student_oracle_recall_at_{k}', f'oracle_recall_at_{k}'):
                if key in metrics:
                    return float(metrics[key])
            return float('nan')

        final_epoch, final_metrics = history[-1]
        best_epoch, best_metrics = max(history, key=lambda item: recall(item[1], 10))
        condition = {
            'setting': setting,
            'relation_input_space': self.args.relation_input_space,
            'relation_teacher_space': self.args.relation_teacher_space,
            'retrieval_similarity': getattr(self.args, 'retrieval_similarity', 'cosine'),
            'stage1_loss_mode': self.args.stage1_loss_mode,
            'candidate_mode': (
                'differentiable'
                if self._differentiable_keys_enabled()
                else f'key_bank_{self.args.stage1_overfit_key_refresh}_refresh'
            ),
            'queries': int(self.args.stage1_overfit_queries),
            'candidates': int(self.args.stage1_overfit_candidates),
            'coverage_top_k': int(self.args.stage1_coverage_top_k)
            if int(self.args.stage1_coverage_top_k) > 0
            else int(self.args.top_k),
            'target_channel': self.args.target_channel,
            'self_only': bool(int(self.args.stage1_overfit_self_only)),
            'seed': int(self.args.seed),
            'final_epoch': int(final_epoch),
            'best_epoch': int(best_epoch),
            'topk_coverage_loss': float(
                final_metrics.get('topk_coverage_loss', float('nan'))
            ),
        }
        for k in (1, 5, 10):
            condition[f'final_train_recall_at_{k}'] = recall(final_metrics, k)
            condition[f'best_train_recall_at_{k}'] = recall(best_metrics, k)
        for key in (
            'coverage_oracle_student_overlap',
            'student_retrieval_regret_at_10',
            'student_retrieved_future_mse_at_10',
            'oracle_future_mse_at_10',
            'student_oracle_topk_probability_mass_at_10',
        ):
            if key in final_metrics:
                condition[f'final_{key}'] = float(final_metrics[key])

        print('=' * 100)
        print('[stage1 tiny-overfit summary]')
        print(
            '  condition: input_space={relation_input_space} candidates={candidate_mode} '
            'teacher_space={relation_teacher_space} similarity={retrieval_similarity} '
            'K={coverage_top_k} Q={queries} N={candidates} seed={seed}'.format(**condition)
        )
        print(
            '  final  (epoch {final_epoch}): Recall@1={final_train_recall_at_1:.4f} '
            'Recall@5={final_train_recall_at_5:.4f} '
            'Recall@10={final_train_recall_at_10:.4f}'.format(**condition)
        )
        print(
            '  best   (epoch {best_epoch}): Recall@1={best_train_recall_at_1:.4f} '
            'Recall@5={best_train_recall_at_5:.4f} '
            'Recall@10={best_train_recall_at_10:.4f}'.format(**condition)
        )
        target = condition['best_train_recall_at_10']
        verdict = 'PASS' if target >= 0.95 else 'FAIL'
        print(f'  success criterion train Recall@10 >= 0.95: {verdict} ({target:.4f})')
        print('=' * 100)

        summary_path = getattr(self.args, 'stage1_overfit_summary_path', '')
        if summary_path:
            os.makedirs(os.path.dirname(os.path.abspath(summary_path)), exist_ok=True)
            with open(summary_path, 'w') as handle:
                json.dump(condition, handle, indent=2)
            print(f'[stage1 tiny-overfit] wrote summary to {summary_path}')

    def test(self, setting, test=0):
        self._ensure_memory()
        checkpoint_path = os.path.join(
            self.args.checkpoints,
            'stage1',
            self.args.data,
            f'seq{self.args.seq_len}_pred{self.args.pred_len}',
            setting,
            'checkpoint.pth',
        )
        if self._direct_eval_enabled():
            if getattr(self.args, 'relation_input_space', None) != 'diff1':
                raise ValueError('stage1_direct_eval is reserved for relation_input_space=diff1')
            print('[stage1] evaluating encoder-free Diff1 Direct retrieval')
        else:
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    f'Stage-1 test checkpoint not found: {checkpoint_path}'
                )
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state = checkpoint.get('model_state_dict', checkpoint)
            self.model.load_state_dict(state)
        self._build_key_bank()
        test_data, test_loader = self._get_data(flag='test', shuffle=False)
        metrics = self._run_loader(
            test_loader,
            optimizer=None,
            compute_detailed_metrics=True,
            split_name='test',
        )
        print(format_metrics('Stage1 Test', metrics))
        return metrics
