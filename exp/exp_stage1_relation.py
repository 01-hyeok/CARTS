import json
import os
from pathlib import Path
import time
import math

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, Subset

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models.RelationStage1 import (
    collapse_geometry, relation_bank_collapse_metrics,
)
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
        needs_history = (
            self._candidate_subset_enabled()
            or (
                self._differentiable_keys_enabled()
                and self.tiny_candidate_indices is not None
            )
            # Full-memory gradient modes re-encode candidates from raw input too,
            # so they need the histories on device even without a mined subset.
            or getattr(self.args, 'stage1_full_memory_gradient_mode', 'bank') != 'bank'
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
        # 0 means no Oracle injection at all: the pool is drawn at random.
        #
        # Injection fills the pool round-robin from each query's best candidates,
        # which is what makes a 16-query tiny set solvable. With many queries it
        # does the opposite -- the first few hundred queries consume every slot, so
        # later queries and every held-out query face a pool that excludes their
        # own answers. A pool that is random is the same pool for every query, and
        # that is what a generalization reading needs.
        oracle_depth = (
            0 if oracle_per_query == 0
            else min(oracle_per_query, ranked_candidates.size(1))
        )
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
        wanted = (
            int(getattr(self.args, 'stage1_pool_size', 0)) > 0
            or getattr(self.args, 'stage1_mining_score', 'self') == 'reference'
        )
        if not path or not wanted:
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

    # Retrieval quality has several defensible readings and they disagree.
    # Recall@10 asks for exact identity inside an Oracle set whose 10th and 11th
    # members differ by 1.4%, while retrieved future MSE asks whether the
    # candidates actually picked were any good. Selecting on one and reporting
    # the others would hide an arm whose advantage is on a different axis, so
    # each criterion keeps its own checkpoint. Early stopping still follows the
    # single configured criterion, so training length is unchanged.
    SIDE_CHECKPOINT_METRICS = {
        'recall10': (('student_oracle_recall_at_10', 'oracle_recall_at_10'), 1.0),
        'ndcg10': (('student_ndcg_at_10', 'ndcg_at_10'), 1.0),
        'retrieved_mse10': (
            ('student_retrieved_future_mse_at_10', 'retrieved_future_mse_at_10'), -1.0),
        'hard_aggregate_mse10': (('hard_aggregate_mse10',), -1.0),
    }

    def _side_checkpoint_scores(self, metrics):
        """Score each side criterion, skipping any the epoch did not report."""
        out = {}
        for name, (keys, sign) in self.SIDE_CHECKPOINT_METRICS.items():
            for key in keys:
                if key in metrics:
                    out[name] = sign * float(metrics[key])
                    break
        return out

    def _save_side_checkpoints(self, best_path, val_metrics, optimizer, epoch):
        """Keep one checkpoint per retrieval criterion alongside the main one."""
        from pathlib import Path as _Path

        best_path = _Path(best_path)
        if not hasattr(self, '_side_best'):
            self._side_best = {}
        payload = None
        for name, score in self._side_checkpoint_scores(val_metrics).items():
            if name in self._side_best and score <= self._side_best[name]:
                continue
            self._side_best[name] = score
            if payload is None:
                payload = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'args': vars(self.args),
                    'relation_graph': self.relation_graph,
                }
            target = best_path.with_name(f'checkpoint_best_{name}.pth')
            torch.save({**payload, 'epoch': epoch, 'selection_metric': name,
                        'selection_score': score}, target)
            print(f'[stage1] new best on {name}: {score:+.6f} (epoch {epoch}) -> {target.name}')

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
            # The quantity Stage-2 actually consumes: how good the futures in the
            # model's own Top-10 are. Recall and NDCG both score the ranking
            # against the Future-MSE Oracle, which is one step further from the
            # forecast than this is.
            'retrieved_mse10': (
                ('student_retrieved_future_mse_at_10', 'retrieved_future_mse_at_10'), -1.0
            ),
            # The error of the one aggregate Stage-2 builds from the Top-10, as
            # opposed to the mean of those candidates' individual errors. The two
            # differ by the spread among them, and the arms here trade one for the
            # other, so every arm is selected on this and none on its own loss.
            'hard_aggregate_mse10': (('hard_aggregate_mse10',), -1.0),
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


    def _probe_rank_gradient(self, model, batch_idx):
        """Choose the ranking weight from what reaches the encoder.

        A margin hinge and a cross-entropy over thousands of candidates are not
        on a comparable scale, so matching their values says nothing about which
        one steers training. Both terms are back-propagated separately here and
        lambda is solved for directly:

            lambda = target_share * ||g_wce|| / ||g_rank||

        The cosine between the two gradients comes with it. The terms disagree
        by construction on a candidate that beats a selected one without
        entering the global Oracle Top-K -- the cross-entropy pushes it down,
        the ranking loss pushes it up -- so a strongly negative cosine is the
        expected reading, not a bug.
        """
        limit = int(os.environ.get('CARTS_GRAD_PROBE_BATCHES', '4'))
        if batch_idx >= limit:
            return
        terms = getattr(model, '_probe_terms', None)
        if not terms or not terms.get('rank'):
            print(f'[probe] batch {batch_idx}: no ranking pairs were mined')
            return
        params = [p for p in model.encoder.parameters() if p.requires_grad]
        if model.shared_cross_projection is not None:
            params += [p for p in model.shared_cross_projection.parameters()
                       if p.requires_grad]

        def flat_grad(term):
            grads = torch.autograd.grad(term, params, retain_graph=True,
                                        allow_unused=True)
            parts = [g.reshape(-1) for g in grads if g is not None]
            return torch.cat(parts) if parts else None

        g_w = flat_grad(torch.stack(terms['wce']).mean())
        g_r = flat_grad(torch.stack(terms['rank']).mean())
        model._probe_terms = {'wce': [], 'rank': []}
        if g_w is None or g_r is None or float(g_r.norm()) == 0.0:
            print(f'[probe] batch {batch_idx}: the ranking term reached no '
                  f'encoder parameter')
            return
        nw, nr = float(g_w.norm()), float(g_r.norm())
        cos = float(torch.dot(g_w, g_r) / (g_w.norm() * g_r.norm() + 1e-12))
        share = float(os.environ.get('CARTS_GRAD_PROBE_SHARE', '0.1'))
        print(f'[probe] batch {batch_idx}  |g_wce|={nw:.6f}  |g_rank|={nr:.6f}  '
              f'cos={cos:+.4f}  ratio={nw / nr:.4f}  '
              f'lambda@{share:.0%}={share * nw / nr:.5f}')

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
            # The loader hands one start per batch, not one per query. The
            # evaluation loaders are sequential and unshuffled, so a query's start
            # -- what a frozen pair is keyed by -- is that batch start plus its row.
            starts = torch.as_tensor(batch_start_idx).reshape(-1)
            if starts.numel() == 1 and batch_x.size(0) > 1:
                starts = int(starts) + torch.arange(batch_x.size(0))
            model._current_starts = starts
            teacher_batch = self._teacher_batch(split_name, batch_start_idx)
            teacher_batch.update(self._residual_batch(split_name, batch_start_idx))
            reference_scores = self._reference_scores(batch_x)
            if reference_scores is not None:
                teacher_batch['reference_scores'] = reference_scores
                if getattr(self.args, 'stage1_mining_score', 'self') == 'reference':
                    # Common mining: every arm's loss runs over the *same*
                    # candidate ids, so a difference between arms is a difference
                    # in score function or objective, not in what they were shown.
                    teacher_batch['mining_scores'] = reference_scores
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
                if os.environ.get('CARTS_GRAD_PROBE') == '1' and train:
                    self._probe_rank_gradient(model, batch_idx)
                if os.environ.get('CARTS_COLLAPSE_PROBE') == '1' and train:
                    marks = {0, 1, 5, 10, 20, 50, 100}
                    grad_marks = {0, 10, 50}
                    if batch_idx in marks:
                        self._collapse_probe(f'step{batch_idx}',
                                             with_grad=batch_idx in grad_marks)
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

    @torch.no_grad()
    def _swap_rows(self, writer, arm, beta, starts, c, base, new, d, cand_mask, acc):
        """One row per candidate that entered or left the Top-10.

        Written per candidate rather than summarised in place so a single query
        can be traced afterwards: which candidate was dropped, where it had been
        ranked, and what its future error was.
        """
        neg = torch.finfo(base.dtype).min / 4
        bm = base.masked_fill(~cand_mask, neg)
        nm = new.masked_fill(~cand_mask, neg)
        b_ord, n_ord = bm.argsort(-1, descending=True), nm.argsort(-1, descending=True)
        b_rank, n_rank = torch.empty_like(b_ord), torch.empty_like(n_ord)
        ar = torch.arange(bm.size(-1), device=bm.device).expand_as(b_ord)
        b_rank.scatter_(1, b_ord, ar)
        n_rank.scatter_(1, n_ord, ar)
        for row in range(bm.size(0)):
            B = set(b_ord[row, :10].tolist())
            N = set(n_ord[row, :10].tolist())
            acc['ret10'].append(len(B & N) / 10.0)
            acc['ret100'].append(len(set(b_ord[row, :100].tolist())
                                     & set(n_ord[row, :100].tolist())) / 100.0)
            rem, add = sorted(B - N), sorted(N - B)
            acc['swaps'].append(len(add))
            b_mse = float(d[row, sorted(B)].mean())
            n_mse = float(d[row, sorted(N)].mean())
            acc['t10delta'].append(n_mse - b_mse)
            # Individual quality and aggregate quality differ by the spread among
            # the ten, which is why a swap can lower every candidate's own error
            # while the aggregate the forecaster receives gets worse.
            acc['b_var'].append(float(d[row, sorted(B)].var(unbiased=False)))
            acc['n_var'].append(float(d[row, sorted(N)].var(unbiased=False)))
            if rem:
                acc['removed'].append(float(d[row, rem].mean()))
            if add:
                acc['added'].append(float(d[row, add].mean()))
            if rem and add:
                acc['delta'].append(float(d[row, add].mean()) - float(d[row, rem].mean()))
            for tag, ids in (('removed', rem), ('added', add)):
                for cid in ids:
                    if tag == 'added':
                        acc['added_rank'].append(int(b_rank[row, cid]))
                    writer.writerow([
                        arm, beta, 'val', int(starts[row]), int(c), int(cid), tag,
                        f'{float(d[row, cid]):.6f}',
                        int(b_rank[row, cid]), int(n_rank[row, cid]),
                        f'{float(base[row, cid]):.6f}', f'{float(new[row, cid]):.6f}',
                        f'{b_mse:.6f}', f'{n_mse:.6f}', f'{n_mse - b_mse:.6f}',
                        len(rem), len(add)])
                    acc['rows'] += 1

    def swap_conflict_diag(self):
        """Retention, Top-10 swaps, anchor gradient by rank band, and whether the
        two objectives pull the scorer apart -- all from the loaded checkpoint.

        Retention is recomputed here for every arm off the same retrieval path,
        including the one trained without an anchor, whose value was previously
        reported from a formatting default rather than measured.
        """
        import csv
        from models.RelationStage1 import boundary_hard_rank_loss, global_anchor_kl

        arm = os.environ.get('CARTS_SWAP_ARM', 'unnamed')
        beta = float(os.environ.get('CARTS_SWAP_BETA', '0'))
        out_dir = os.environ.get('CARTS_SWAP_OUT', 'logs/swap_conflict')
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, 'swap_rows.csv')
        fresh = not os.path.exists(path)
        handle = open(path, 'a', newline='')
        writer = csv.writer(handle)
        if fresh:
            writer.writerow(['arm', 'beta', 'split', 'query_id', 'channel_id',
                             'candidate_id', 'swap_type', 'future_mse',
                             'baseline_rank', 'new_rank', 'baseline_score',
                             'new_score', 'baseline_top10_future_mse',
                             'new_top10_future_mse', 'top10_mse_delta',
                             'num_removed', 'num_added'])

        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.eval()
        # Fingerprint the weights that decide the ranking. Two arms that load
        # different checkpoints must not produce the same fingerprint; when the
        # diagnostic ran before the checkpoint was applied every arm reported
        # the identity initialisation and the numbers looked plausible.
        if model.retrieval_metric is not None:
            fingerprint = float(sum(q.detach().double().abs().sum()
                                    for q in model.retrieval_metric.parameters()))
        else:
            fingerprint = 0.0
        # Recorded to a file rather than the environment: each arm runs in its
        # own process, so an in-process record would never see the others.
        seen_path = os.path.join(out_dir, 'fingerprints.txt')
        prior = {}
        if os.path.exists(seen_path):
            for line in open(seen_path):
                name, value = line.strip().split(' ', 1)
                prior[name] = value
        key = f'{fingerprint:.10f}'
        for name, value in prior.items():
            if value == key and name != arm:
                raise RuntimeError(
                    f'{arm} loaded the same scorer weights as {name}; the two '
                    f'arms are not distinct and their numbers would agree for '
                    f'that reason alone')
        with open(seen_path, 'a') as fh:
            fh.write(f'{arm} {key}\n')
        print(f'[swap] scorer fingerprint {fingerprint:.6f}')
        self._ensure_memory()
        self._build_key_bank()
        _, loader = self._get_data(flag='val', shuffle=False)
        limit = int(os.environ.get('CARTS_SWAP_BATCHES', '2'))
        tau = float(getattr(model, 'global_anchor_tau', 0.1)) or 0.1

        acc = dict(ret10=[], ret100=[], swaps=[], removed=[], added=[],
                   delta=[], t10delta=[], added_rank=[], rows=0, examples=[],
                   b_var=[], n_var=[])
        ga = dict(mass=[0.0, 0.0, 0.0], grad=[0.0, 0.0, 0.0], n=0)
        conflict = []

        for batch_idx, (bx, by, start) in enumerate(loader):
            if batch_idx >= limit:
                break
            bx, by, start = self._move_batch(bx, by, start)
            cand_mask, _ = self._candidate_mask(start)
            starts = torch.as_tensor(start).reshape(-1)
            if starts.numel() == 1 and bx.size(0) > 1:
                starts = int(starts) + torch.arange(bx.size(0))
            for c in model.target_channels():
                for r in model.source_channels(c):
                    with torch.no_grad():
                        z_q = model.encoder(model._relation_tensor(bx, c, r))
                        z_k = self.key_bank[c, 0].to(z_q.dtype)
                        base = (F.normalize(z_q.float(), dim=-1)
                                @ F.normalize(z_k.float(), dim=-1).t())
                        d = model._future_mse(bx, by, self.memory_y,
                                              self.memory_x_last, c, r).float()
                    with torch.enable_grad():
                        new = (model.retrieval_metric.score(z_q, z_k).float()
                               if model.retrieval_metric is not None
                               else base.clone().requires_grad_(True))
                    self._swap_rows(writer, arm, beta, starts, c,
                                    base, new.detach(), d, cand_mask, acc)
                    if c == 0 and len(acc['examples']) < 3:
                        acc['examples'].append(int(starts[0]))

                    # Autograd on the scores, not a re-derivation, so masking and
                    # normalisation match the objective that was trained.
                    kl, _ = global_anchor_kl(base, new, cand_mask, tau=tau)
                    g_s, = torch.autograd.grad(kl, new, retain_graph=True)
                    with torch.no_grad():
                        neg = torch.finfo(base.dtype).min / 4
                        b_ord = base.masked_fill(~cand_mask, neg).argsort(-1, descending=True)
                        p = torch.softmax(base.masked_fill(~cand_mask, neg) / tau, dim=-1)
                        for bi, (lo, hi) in enumerate([(0, 10), (10, 100),
                                                       (100, base.size(-1))]):
                            sel = b_ord[:, lo:hi]
                            ga['mass'][bi] += float(p.gather(1, sel).sum())
                            ga['grad'][bi] += float(g_s.abs().gather(1, sel).sum())
                        ga['n'] += 1

                    params = [q for q in model.retrieval_metric.parameters()
                              if q.requires_grad] if model.retrieval_metric else []
                    if params:
                        rk, _ = boundary_hard_rank_loss(
                            new, d, cand_mask, top_k=10, pool_end=100, margin=0.01,
                            pairs_per_query=32, mining_mode='candidate')
                        gr = torch.autograd.grad(rk, params, retain_graph=True,
                                                 allow_unused=True)
                        gg = torch.autograd.grad(kl, params, retain_graph=True,
                                                 allow_unused=True)
                        fr = [x.reshape(-1) for x in gr if x is not None]
                        fg = [x.reshape(-1) for x in gg if x is not None]
                        if fr and fg:
                            fr, fg = torch.cat(fr), torch.cat(fg)
                            if float(fr.norm()) > 0 and float(fg.norm()) > 0:
                                conflict.append((
                                    float(torch.dot(fr, fg) / (fr.norm() * fg.norm())),
                                    float(fr.norm()), float(fg.norm())))
        handle.close()

        def m(xs):
            return sum(xs) / len(xs) if xs else float('nan')
        print(f'[swap] arm={arm} beta={beta} rows={acc["rows"]} '
              f'pairs_analysed={len(acc["ret10"])} csv={path}')
        print(f'[swap] retention10={m(acc["ret10"]):.5f} '
              f'retention100={m(acc["ret100"]):.5f} '
              f'swaps_mean={m(acc["swaps"]):.4f} '
              f'removed_mse={m(acc["removed"]):.5f} added_mse={m(acc["added"]):.5f} '
              f'swapdelta={m(acc["delta"]):.5f} t10delta={m(acc["t10delta"]):.5f}')
        print(f'[swap] top10_future_var base={m(acc["b_var"]):.5f} '
              f'new={m(acc["n_var"]):.5f} '
              f'delta={m(acc["n_var"]) - m(acc["b_var"]):+.5f}')
        bands = [(10, 20), (20, 50), (50, 100), (100, 500)]
        ranks = acc['added_rank']
        if ranks:
            frac = ' '.join(
                f'{lo + 1}-{hi}={sum(1 for x in ranks if lo <= x < hi) / len(ranks):.3f}'
                for lo, hi in bands)
            print(f'[swap] added_from_rank {frac} '
                  f'501+={sum(1 for x in ranks if x >= 500) / len(ranks):.3f}')
        if ga['n']:
            tm, tg = sum(ga['mass']), sum(ga['grad'])
            print(f'[swap] ga_mass top10={ga["mass"][0]/tm:.5f} '
                  f'11_100={ga["mass"][1]/tm:.5f} rest={ga["mass"][2]/tm:.5f}')
            print(f'[swap] ga_grad top10={ga["grad"][0]/tg:.5f} '
                  f'11_100={ga["grad"][1]/tg:.5f} rest={ga["grad"][2]/tg:.5f}')
        if conflict:
            cos = [x[0] for x in conflict]
            print(f'[swap] conflict cos_mean={m(cos):+.5f} '
                  f'neg_frac={sum(1 for x in cos if x < 0)/len(cos):.4f} '
                  f'g_rank={m([x[1] for x in conflict]):.6f} '
                  f'g_ga={m([x[2] for x in conflict]):.6f}')

    @torch.no_grad()
    @torch.no_grad()
    def tau_calibration_diag(self):
        """Step-0: pick tau_set from the score distribution alone, no future labels.

        Reconstructed from `logs/tau_calibration/pred{96,192,336,720}.log`, the
        only surviving record of this diagnostic -- the method itself was lost
        before being committed. Every sweep point (entropy, N_eff, Mass@10,
        Mass@100, max_p) at all four horizons was checked against that log and
        matches to 4-5 significant figures; this is not a re-derivation from the
        formula in prose, it is the formula the log's numbers pin down.

        N_eff is mean_q(exp(H_q)) -- the mean of each query's own exponentiated
        entropy, not exp(mean_q H_q) and not the inverse-participation ratio
        1/sum_i p_i^2 in either aggregation order. Those alternatives were ruled
        out numerically: at pred=336, tau=0.01, they give 18.46 and (17.82 or
        5.14) respectively, against a logged N_eff of 35.9 that only
        mean_q(exp(H_q)) reproduces.
        """
        import os as _os
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.eval()
        self._ensure_memory()
        self._build_key_bank()
        _, loader = self._get_data(flag='train', shuffle=False)

        batches = int(_os.environ.get('CARTS_TAUCAL_BATCHES', '4'))
        taus = [float(t) for t in _os.environ.get(
            'CARTS_TAUCAL_TAUS',
            '0.005,0.0075,0.01,0.0125,0.015,0.02,0.03,0.05,0.07,0.1').split(',')]
        target_n_eff = tuple(float(x) for x in _os.environ.get(
            'CARTS_TAUCAL_TARGET_NEFF', '30.0,60.0').split(','))
        target_mass10 = tuple(float(x) for x in _os.environ.get(
            'CARTS_TAUCAL_TARGET_MASS10', '0.5,0.8').split(','))

        all_scores, all_valid = [], []
        for batch_idx, (bx, by, start) in enumerate(loader):
            if batch_idx >= batches:
                break
            bx, by, start = self._move_batch(bx, by, start)
            cand_mask, _ = self._candidate_mask(start)
            for c in model.target_channels():
                for r in model.source_channels(c):
                    z_q = model.encoder(model._relation_tensor(bx, c, r))
                    z_k = self.key_bank[c, 0].to(z_q.dtype)
                    if model.retrieval_metric is not None:
                        scores = model.retrieval_metric.score(z_q, z_k)
                    else:
                        scores = (F.normalize(z_q.float(), dim=-1)
                                  @ F.normalize(z_k.float(), dim=-1).t())
                    all_scores.append(scores.float())
                    all_valid.append(cand_mask)

        scores = torch.cat(all_scores, dim=0)
        valid = torch.cat(all_valid, dim=0)
        neg = torch.finfo(scores.dtype).min / 4
        order = scores.masked_fill(~valid, neg).argsort(dim=-1, descending=True)

        pred = self.args.pred_len
        print(f'[taucal] pred={pred} split=train batches={batches} '
              f'target N_eff={target_n_eff} Mass@10={target_mass10}')
        print('[taucal]      tau   entropy     N_eff   Mass@10  Mass@100     max_p')

        rows = []
        for tau in taus:
            logits = (scores / tau).masked_fill(~valid, neg)
            p = F.softmax(logits, dim=-1)
            entropy = -(p * (p + 1e-12).log()).sum(-1)
            n_eff = float(entropy.exp().mean())
            h_mean = float(entropy.mean())
            mass10 = float(p.gather(1, order[:, :10]).sum(-1).mean())
            mass100 = float(p.gather(1, order[:, :100]).sum(-1).mean())
            maxp = float(p.max(-1).values.mean())
            rows.append((tau, h_mean, n_eff, mass10, mass100, maxp))
            print(f'[taucal]  {tau:>7.4g}  {h_mean:>7.4f}  {n_eff:>8.1f}  '
                  f'{mass10:>7.4f}  {mass100:>7.4f}  {maxp:>8.5f}')

        in_band = [r for r in rows if target_n_eff[0] <= r[2] <= target_n_eff[1]]
        pool = in_band if in_band else rows
        rule = 'inside target range' if in_band else 'NO tau inside target N_eff range'
        # The tie-break target is the band's lower bound, not its midpoint --
        # verified against all four ground-truth horizons: at pred=96 and
        # pred=192 two taus land inside the N_eff band, and both times the one
        # logged as chosen is the one closer to 0.50, not to 0.65 (the (0.5,0.8)
        # midpoint). A denser memory bank pushes Mass@10 down at fixed tau, so
        # anchoring to the permissive edge is what keeps the choice from
        # drifting to an unnecessarily sharp tau as the candidate pool grows.
        target_center = target_mass10[0]
        chosen = min(pool, key=lambda r: abs(r[3] - target_center))
        print(f'[taucal] rule: {rule}; Mass@10 closest to {target_center:.2f}')
        print(f'[taucal] TAU_{pred} = {chosen[0]:g}  '
              f'(N_eff={chosen[2]:.1f} Mass@10={chosen[3]:.4f})')
        return {'rows': rows, 'chosen_tau': chosen[0], 'chosen_n_eff': chosen[2],
                'chosen_mass10': chosen[3], 'in_band': bool(in_band)}

    def set_oracle_diag(self):
        """Is a good Top-K a set of individually good candidates, or a set that
        is good together?

        Four selections over one shared pool, so pool coverage cannot explain a
        difference between them, swept over pool sizes because the retriever's
        own Top-100 excludes most of what the Oracle would pick -- widening the
        pool separates "no complementarity here" from "the coarse retriever
        never offered it". K is swept by prefix of one K=max run.

        Everything reads the true future, so it bounds what a perfect selector
        could do inside each pool; it does not say a model can reach it.
        """
        import csv
        from models.RelationStage1 import (
            select_good_diverse, select_greedy_set, select_individual_oracle,
            set_utility_metrics,
        )
        out_dir = os.environ.get('CARTS_SETORACLE_OUT', 'logs/set_oracle')
        os.makedirs(out_dir, exist_ok=True)
        tag = os.environ.get('CARTS_SETORACLE_TAG', f'pred{self.args.pred_len}')
        pools = [x.strip() for x in
                 os.environ.get('CARTS_SETORACLE_POOL', '100').split(',')]
        ks = sorted(int(x) for x in
                    os.environ.get('CARTS_SETORACLE_K', '10').split(','))
        k_max = max(ks)
        good_n = int(os.environ.get('CARTS_SETORACLE_GOOD', '30'))
        limit = int(os.environ.get('CARTS_SETORACLE_BATCHES', '4'))
        arms = ['cosine', 'individual', 'good_diverse', 'set']

        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.eval()
        self._ensure_memory()
        self._build_key_bank()
        _, loader = self._get_data(flag='val', shuffle=False)

        # cell[(pool, K, arm)] -> running metric lists
        cell = {}
        extra = {}

        def note(key, field, values):
            cell.setdefault(key, {}).setdefault(field, []).extend(values)

        for batch_idx, (bx, by, start) in enumerate(loader):
            if batch_idx >= limit:
                break
            bx, by, start = self._move_batch(bx, by, start)
            cand_mask, _ = self._candidate_mask(start)
            for c in model.target_channels():
                for r in model.source_channels(c):
                    z_q = model.encoder(model._relation_tensor(bx, c, r))
                    z_k = self.key_bank[c, 0].to(z_q.dtype)
                    cos = (F.normalize(z_q.float(), dim=-1)
                           @ F.normalize(z_k.float(), dim=-1).t())
                    q_fut, k_fut = model._relation_future_distance_inputs(
                        bx, by, self.memory_y, self.memory_x_last, c, r)
                    q_fut, k_fut = q_fut.float(), k_fut.float()
                    neg = torch.finfo(cos.dtype).min / 4
                    masked = cos.masked_fill(~cand_mask, neg)

                    for spec in pools:
                        width = (int(cand_mask.sum(-1).min()) if spec == 'full'
                                 else min(int(spec), masked.size(-1)))
                        width = max(width, k_max)
                        pool_idx = masked.topk(width, dim=-1).indices
                        pool_y = k_fut[pool_idx]
                        d_pool = ((pool_y - q_fut.unsqueeze(1)) ** 2).mean(-1)
                        order = torch.arange(width, device=cos.device).expand_as(pool_idx)

                        picks = {
                            'cosine': order[:, :k_max],
                            'individual': d_pool.argsort(dim=-1)[:, :k_max],
                            'good_diverse': select_good_diverse(
                                pool_y, d_pool, k_max, good_n),
                            'set': select_greedy_set(pool_y, q_fut, k_max),
                        }
                        for kk in ks:
                            per = {}
                            for arm, idx in picks.items():
                                take = idx[:, :kk]
                                chosen = torch.gather(
                                    pool_y, 1,
                                    take.unsqueeze(-1).expand(-1, -1, pool_y.size(-1)))
                                I, A, V, res = set_utility_metrics(chosen, q_fut)
                                if float(res.abs().max()) > 1e-4:
                                    raise RuntimeError(
                                        f'{arm}: I = A + V failed by '
                                        f'{float(res.abs().max()):.3e}')
                                key = (spec, kk, arm)
                                note(key, 'I', I.tolist()); note(key, 'A', A.tolist())
                                note(key, 'V', V.tolist()); note(key, 'res', res.tolist())
                                per[arm] = (I, A, take)
                            for row in range(pool_y.size(0)):
                                si = set(per['individual'][2][row].tolist())
                                ss = set(per['set'][2][row].tolist())
                                sd = set(per['good_diverse'][2][row].tolist())
                                e = extra.setdefault((spec, kk), {
                                    'ov_is': [], 'ov_ds': [], 'gain': [], 'cost': []})
                                e['ov_is'].append(len(si & ss) / kk)
                                e['ov_ds'].append(len(sd & ss) / kk)
                                a_i = float(per['individual'][1][row])
                                a_s = float(per['set'][1][row])
                                e['gain'].append((a_i - a_s) / max(a_i, 1e-12))
                                e['cost'].append(float(per['set'][0][row])
                                                 - float(per['individual'][0][row]))

        def m(xs):
            return sum(xs) / len(xs) if xs else float('nan')

        csv_path = os.path.join(out_dir, f'pool_k_sweep_{tag}.csv')
        with open(csv_path, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['tag', 'pool', 'K', 'arm', 'I', 'A', 'V', 'residual', 'n'])
            for (spec, kk, arm), v in sorted(cell.items(), key=lambda x: str(x[0])):
                w.writerow([tag, spec, kk, arm, f'{m(v["I"]):.6f}', f'{m(v["A"]):.6f}',
                            f'{m(v["V"]):.6f}', f'{m(v["res"]):.3e}', len(v['A'])])

        print(f'\n[setoracle] {tag} VAL  pools={pools} K={ks} good_n={good_n} '
              f'-> {csv_path}')
        head = (f"{'pool':>6}{'K':>4}" + ''.join(f'{a[:9]:>11}' for a in arms)
                + f"{'SetGain':>10}{'IndCost':>10}{'ov(i,s)':>9}{'joint':>8}")
        print('[setoracle] ' + head)
        for spec in pools:
            for kk in ks:
                if (spec, kk, 'set') not in cell:
                    continue
                a_i = m(cell[(spec, kk, 'individual')]['A'])
                a_s = m(cell[(spec, kk, 'set')]['A'])
                e = extra[(spec, kk)]
                joint = sum(1 for g, cst in zip(e['gain'], e['cost'])
                            if g > 0 and cst >= 0) / len(e['gain'])
                print(f'[setoracle] {spec:>6}{kk:>4}'
                      + ''.join(f"{m(cell[(spec, kk, a)]['A']):>11.5f}" for a in arms)
                      + f'{(a_i - a_s) / a_i:>10.4f}{m(e["cost"]):>10.5f}'
                      f'{m(e["ov_is"]):>9.4f}{joint:>8.4f}')
        if ('100', 1, 'set') in cell:
            a1 = m(cell[('100', 1, 'set')]['A'])
            b1 = m(cell[('100', 1, 'individual')]['A'])
            if abs(a1 - b1) > 1e-6:
                raise RuntimeError(f'K=1 must agree: set={a1} individual={b1}')
            print('[setoracle] sanity: K=1 set == individual  OK')

    class _ProbeDoneSignal(Exception):
        pass

    def _collapse_subset(self):
        """One fixed set of queries and candidates, drawn once.

        Geometry is only comparable across steps and arms if the same rows are
        measured every time; re-drawing would let a change in the sample explain
        a change in the numbers.
        """
        cached = getattr(self, '_collapse_cached', None)
        if cached is not None:
            return cached
        n_q = int(os.environ.get('CARTS_COLLAPSE_QUERIES', '64'))
        n_c = int(os.environ.get('CARTS_COLLAPSE_CANDIDATES', '1024'))
        gen = torch.Generator().manual_seed(1234)
        total = int(self.memory_x.size(0))
        cand = torch.randperm(total, generator=gen)[:min(n_c, total)]
        query = torch.randperm(total, generator=gen)[:min(n_q, total)]
        cached = (query.to(self.device), cand.to(self.device))
        self._collapse_cached = cached
        return cached

    def _collapse_probe(self, tag, with_grad=False):
        """Geometry, and optionally gradients, without touching the weights.

        Uses autograd.grad rather than backward so nothing lands in .grad and no
        optimiser state moves; a checksum over the parameters is compared before
        and after, because a diagnostic that quietly trains the model would look
        exactly like the collapse it is meant to explain.
        """
        model = self.model.module if hasattr(self.model, 'module') else self.model
        before = float(sum(p.detach().float().sum() for p in model.parameters()))
        query_idx, cand_idx = self._collapse_subset()
        channel = 0
        was_training = model.training
        model.eval()
        try:
            q_rel = model._relation_tensor(
                self.memory_x[query_idx].to(self.device), channel, channel)
            c_rel = model._relation_tensor(
                self.memory_x[cand_idx].to(self.device), channel, channel)
            with torch.enable_grad() if with_grad else torch.no_grad():
                z_q = model.encoder(q_rel)
                z_c = model.encoder(c_rel)
                if with_grad and not z_q.requires_grad:
                    # A frozen encoder produces embeddings outside the graph;
                    # re-attaching them measures how much gradient the ranking
                    # loss would deliver to the representation, which is the
                    # comparison against the trainable arms.
                    z_q = z_q.detach().requires_grad_(True)
                    z_c = z_c.detach().requires_grad_(True)
                stats = collapse_geometry(z_q, z_c)
                if with_grad:
                    # The ranking hinge on this subset: raise the candidate the
                    # query should prefer, lower the one it currently prefers.
                    scores = F.normalize(z_q, dim=-1) @ F.normalize(z_c, dim=-1).t()
                    top = scores.topk(2, dim=-1).indices
                    s_i = scores.gather(1, top[:, :1]).squeeze(1)
                    s_j = scores.gather(1, top[:, 1:2]).squeeze(1)
                    hinge = (self.args.rank_margin - s_j + s_i).clamp_min(0).mean()
                    # With the encoder frozen there is nothing to differentiate
                    # on its side, and asking would raise rather than report a
                    # zero. The embedding gradients still answer the question
                    # this probe exists for.
                    params = [p for p in model.encoder.parameters()
                              if p.requires_grad]
                    targets = params + [z_q, z_c]
                    if not any(t.requires_grad for t in targets):
                        stats['encoder_grad_norm'] = 0.0
                        stats['grad_z_query_norm'] = 0.0
                        stats['grad_z_cand_norm'] = 0.0
                        stats['hinge_on_probe'] = float(hinge)
                        raise self._ProbeDoneSignal
                    grads = torch.autograd.grad(hinge, targets,
                                                allow_unused=True)
                    enc = [g.reshape(-1) for g in grads[:len(params)] if g is not None]
                    stats['encoder_grad_norm'] = float(
                        torch.cat(enc).norm()) if enc else 0.0
                    gq, gc = grads[len(params)], grads[len(params) + 1]
                    stats['grad_z_query_norm'] = float(gq.norm()) if gq is not None else 0.0
                    stats['grad_z_cand_norm'] = float(gc.norm()) if gc is not None else 0.0
                    stats['hinge_on_probe'] = float(hinge)
        except self._ProbeDoneSignal:
            pass
        finally:
            if was_training:
                model.train()
        after = float(sum(p.detach().float().sum() for p in model.parameters()))
        if abs(after - before) > 1e-6:
            raise RuntimeError('the collapse probe changed model parameters')
        line = ' '.join(f'{k}={v:.6g}' for k, v in stats.items())
        print(f'[collapse] {tag} {line}')
        return stats

    def _train_diag_loader(self, train_data):
        """A fixed slice of train queries, evaluated exactly as validation is.

        Diagnostics computed inside the optimisation loop are taken on shuffled,
        augmented batches under whatever the encoder looked like mid-epoch, which
        is why they came out unusable. This is a held-constant subset run through
        the same evaluator with the same full-memory retrieval, so a train and a
        validation number differ only in which queries they cover -- which is
        what separates memorising the ordering from learning it.
        """
        cached = getattr(self, '_train_diag_cached', None)
        if cached is not None:
            return cached
        size = int(os.environ.get('CARTS_TRAIN_DIAG_QUERIES', '256'))
        _, loader = self._get_data(flag='train', shuffle=False)
        subset = torch.utils.data.Subset(
            loader.dataset, list(range(min(size, len(loader.dataset)))))
        out = torch.utils.data.DataLoader(
            subset, batch_size=loader.batch_size, shuffle=False,
            num_workers=0, drop_last=False)
        self._train_diag_cached = out
        return out

    def _set_frozen_split(self, split):
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model._frozen_split = split

    def train_diag(self, train_data):
        loader = self._train_diag_loader(train_data)
        self._set_frozen_split('train')
        metrics = self._run_loader(
            loader, optimizer=None, compute_detailed_metrics=True,
            split_name='train_diag',
        )
        missing = [k for k in ('missed_better_100_mean', 'pair_acc_top100_all')
                   if k in metrics and metrics[k] != metrics[k]]
        if missing:
            raise RuntimeError(
                f'train diagnostic produced NaN for {missing}; a silent NaN here '
                f'is what made Case B and Case D indistinguishable')
        return metrics

    def vali(self, vali_data, vali_loader):
        self._set_frozen_split('val')
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
            train_loader, tiny_vali_loader = self._configure_tiny_overfit(train_data)
            if int(getattr(self.args, 'stage1_overfit_holdout_val', 0)):
                # Held-out queries against the same tiny candidate set. The mask is
                # built per batch from the full memory and then indexed down to the
                # tiny candidates, so val queries get a correct mask for free.
                #
                # Without this, tiny-overfit reports val = train and a Recall@10 of
                # 1.0 says only that the encoder memorised its sixteen queries -- it
                # cannot say whether anything transfers to a query it never saw.
                vali_data, vali_loader = self._get_data(flag='val', shuffle=False)
            else:
                vali_loader = tiny_vali_loader
                vali_data = train_data
        else:
            vali_data, vali_loader = self._get_data(flag='val', shuffle=False)
        self._set_validation_probe(vali_loader)
        self.total_update_steps = max(1, len(train_loader) * int(self.args.train_epochs))
        frozen_ref = None
        if int(getattr(self.args, 'stage1_freeze_encoder', 0)):
            net = self.model.module if hasattr(self.model, 'module') else self.model
            frozen_ref = float(sum(p.detach().double().sum()
                                   for p in net.encoder.parameters()))
            trainable = [n for n, p in net.named_parameters() if p.requires_grad]
            print(f'[freeze] encoder held fixed; trainable parameters: {trainable}')
            if any(n.startswith('encoder.') for n in trainable):
                raise RuntimeError('encoder parameters are still trainable')

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
                if (epoch == 0 and getattr(self.args, 'rank_mining_mode',
                                           'pair') == 'persistent'):
                    # One forward-only pass before any update, so every
                    # persistent pair comes from the checkpoint training starts
                    # at rather than a model already moved partway through the
                    # first epoch. It needs the key bank, hence its place here.
                    print('[persistent] mining the fixed training pair set '
                          'from the initial checkpoint')
                    self._run_loader(train_loader, optimizer=None,
                                     compute_detailed_metrics=False,
                                     split_name='persistent_build')
                    net = (self.model.module if hasattr(self.model, 'module')
                           else self.model)
                    built = sum(len(v) for v in
                                getattr(net, '_persistent_store', {}).values())
                    print(f'[persistent] {built} queries fixed')
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
                if frozen_ref is not None:
                    net = (self.model.module if hasattr(self.model, 'module')
                           else self.model)
                    now = float(sum(p.detach().double().sum()
                                    for p in net.encoder.parameters()))
                    if abs(now - frozen_ref) > 1e-6:
                        raise RuntimeError(
                            f'the frozen encoder moved: checksum {frozen_ref} '
                            f'-> {now}')
                if os.environ.get('CARTS_COLLAPSE_PROBE') == '1':
                    self._collapse_probe(f'epoch{epoch + 1}', with_grad=True)
                val_metrics = self.vali(vali_data, vali_loader)
                if os.environ.get('CARTS_TRAIN_DIAG') == '1':
                    tr = self.train_diag(train_data)
                    # Must name the frozen metrics as they are actually keyed,
                    # or the primary criterion silently drops out of the train
                    # line while validation still shows it.
                    keys = ('missed_better_100_mean', 'oracle_model_rank_median',
                            'pair_acc_top100_all', 'pair_acc_top100_gap_p50',
                            'pair_acc_top100_gap_p75',
                            'student_retrieved_future_mse_at_10',
                            'hard_aggregate_mse10', 'student_oracle_recall_at_10',
                            'frozen_pair_correct_order_frac',
                            'frozen_signed_gap_mean', 'frozen_signed_gap_p25',
                            'frozen_signed_gap_p50', 'frozen_signed_gap_p75',
                            'frozen_margin_satisfied_frac', 'frozen_pair_count',
                            'rank_positive_unique_covered_frac')
                    # Signed gaps live near 1e-5, so six decimals prints them
                    # all as zero and hides whether a margin is forming.
                    line = ' | '.join(f'{k}: {tr[k]:.8f}' for k in keys if k in tr)
                    print(f'Epoch {epoch + 1} TrainDiag | {line}')
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

                self._save_side_checkpoints(best_path, val_metrics, optimizer, epoch + 1)
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
        # These come from the validation loader, which in the original tiny-overfit
        # is the training set itself -- hence the name. With
        # --stage1_overfit_holdout_val the loader holds unseen queries, so the same
        # numbers would be validation wearing a training label. Record which split
        # produced them and publish them under a matching name.
        holdout = bool(int(getattr(self.args, 'stage1_overfit_holdout_val', 0)))
        split_name = 'val' if holdout else 'train'
        condition['eval_split'] = split_name
        condition['holdout_val'] = holdout
        for k in (1, 5, 10):
            condition[f'final_{split_name}_recall_at_{k}'] = recall(final_metrics, k)
            condition[f'best_{split_name}_recall_at_{k}'] = recall(best_metrics, k)
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
            '  final  (epoch {final_epoch}) [{eval_split}]: '
            'Recall@1={final_recall_1:.4f} Recall@5={final_recall_5:.4f} '
            'Recall@10={final_recall_10:.4f}'.format(
                final_recall_1=condition[f'final_{split_name}_recall_at_1'],
                final_recall_5=condition[f'final_{split_name}_recall_at_5'],
                final_recall_10=condition[f'final_{split_name}_recall_at_10'],
                **condition)
        )
        print(
            '  best   (epoch {best_epoch}) [{eval_split}]: '
            'Recall@1={best_recall_1:.4f} Recall@5={best_recall_5:.4f} '
            'Recall@10={best_recall_10:.4f}'.format(
                best_recall_1=condition[f'best_{split_name}_recall_at_1'],
                best_recall_5=condition[f'best_{split_name}_recall_at_5'],
                best_recall_10=condition[f'best_{split_name}_recall_at_10'],
                **condition)
        )
        target = condition[f'best_{split_name}_recall_at_10']
        # The 0.95 bar was set for memorisation. On held-out queries it is not the
        # right bar, so say which question the verdict answers.
        question = ('transfers to unseen queries' if holdout
                    else 'memorises its own training queries')
        verdict = 'PASS' if target >= 0.95 else 'FAIL'
        print(f'  criterion {split_name} Recall@10 >= 0.95 ({question}): '
              f'{verdict} ({target:.4f})')
        print('=' * 100)

        summary_path = getattr(self.args, 'stage1_overfit_summary_path', '')
        if summary_path:
            os.makedirs(os.path.dirname(os.path.abspath(summary_path)), exist_ok=True)
            with open(summary_path, 'w') as handle:
                json.dump(condition, handle, indent=2)
            print(f'[stage1 tiny-overfit] wrote summary to {summary_path}')

    def test(self, setting, test=0):
        if os.environ.get('CARTS_TAUCAL_DIAG') == '1':
            # Before the strict load: the encoder is already in place from
            # --stage1_ckpt_path and the scorer must stay at its identity
            # initialisation, which is the geometry every arm starts from.
            self.tau_calibration_diag()
            return {}
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
        if os.environ.get('CARTS_SETORACLE_DIAG') == '1':
            self.set_oracle_diag()
            return {}
        if os.environ.get('CARTS_SWAP_DIAG') == '1':
            # After the checkpoint is applied, or every arm would be diagnosed
            # at its identity initialisation and look identical to the baseline.
            self.swap_conflict_diag()
            return {}
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
