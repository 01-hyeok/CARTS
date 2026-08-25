import os
import time
import csv
import math

from pathlib import Path

import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.relation_memory import RelationMemoryBank
from utils.relation_graph import load_or_build_relation_graph, relation_graph_enabled
from utils.stage1_metrics import MetricAverager, format_metrics
from utils.tensorboard_logger import build_summary_writer, write_metric_scalars
from utils.tools import adjust_learning_rate


class Exp_Stage2_Relation(Exp_Basic):
    def __init__(self, args):
        super(Exp_Stage2_Relation, self).__init__(args)
        self.memory_bank = None
        self.memory_y = None
        self.memory_x_last = None
        self.key_bank = None
        self.retrieval_caches = {}
        self.best_checkpoint_path = None
        self.current_setting = None
        self.relation_graph = None

    def _channel_names(self, num_channels):
        graph_names = getattr(self.args, 'relation_channel_names', None)
        if graph_names is not None and len(graph_names) == num_channels:
            return list(graph_names)
        if self.args.data in ('ETTh1', 'ETTh2', 'ETTm1', 'ETTm2') and num_channels == 7:
            return ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
        return [f'ch{i}' for i in range(num_channels)]

    def _focus_channel_index(self, num_channels):
        names = self._channel_names(num_channels)
        focus = getattr(self.args, 'focus_channel', 'OT')
        if focus in names:
            return names.index(focus), focus
        return None, focus

    def _nan(self, device):
        return torch.tensor(float('nan'), device=device)

    def _safe_corr(self, x, y):
        x = x.reshape(-1).float()
        y = y.reshape(-1).float()
        valid = torch.isfinite(x) & torch.isfinite(y)
        x = x[valid]
        y = y[valid]
        if x.numel() < 2:
            return self._nan(x.device if x.numel() else self.device)
        x = x - x.mean()
        y = y - y.mean()
        denom = torch.sqrt((x.pow(2).mean() * y.pow(2).mean()).clamp_min(1e-12))
        if denom <= 0:
            return self._nan(x.device)
        return (x * y).mean() / denom

    def _rank_average(self, x):
        order = torch.argsort(x, dim=-1, descending=False)
        ranks = torch.empty_like(order, dtype=torch.float32)
        values = torch.arange(1, x.size(-1) + 1, device=x.device, dtype=torch.float32)
        values = values.view(*([1] * (x.dim() - 1)), -1).expand_as(ranks)
        ranks.scatter_(-1, order, values)
        return ranks

    def _csv_base_dir(self, setting):
        return os.path.join(
            getattr(self.args, 'metrics_csv_dir', './metrics'),
            'stage2',
            self.args.data,
            f'seq{self.args.seq_len}_pred{self.args.pred_len}',
            setting,
        )

    def _csv_context(self, epoch, split, setting):
        return {
            'epoch': epoch,
            'split': split,
            'data': self.args.data,
            'pred_len': self.args.pred_len,
            'seq_len': self.args.seq_len,
            'model_id': self.args.model_id,
            'des': self.args.des,
            'setting': setting,
            'encoder_type': self.args.relation_encoder_type,
            'pooling': self.args.relation_pooling,
            'self_fill': self.args.relation_self_fill,
            'relation_input_space': getattr(self.args, 'relation_input_space', 'absolute'),
            'relation_teacher_space': getattr(self.args, 'relation_teacher_space', 'absolute'),
            'relation_value_space': getattr(self.args, 'relation_value_space', 'absolute'),
            'fusion_mode': self.args.fusion_mode,
            'stage2_relation_fusion': getattr(self.args, 'stage2_relation_fusion', 'gate'),
            'stage2_retrieval_encoder': getattr(self.args, 'stage2_retrieval_encoder', 'online'),
            'stage2_retrieval_backbone': getattr(self.args, 'stage2_retrieval_backbone', 'stage1'),
            'relation_mixer_input': self.args.relation_mixer_input,
            'stage1_encoder_init': self.args.stage1_encoder_init,
            'disable_retrieval': int(getattr(self.args, 'disable_retrieval', 0)),
            'source_mode': self.args.source_mode,
            'relation_top_n': int(getattr(self.args, 'relation_top_n', 3)),
            'relation_graph_path': getattr(self.args, 'relation_graph_path', ''),
        }

    def _append_csv(self, path, rows, fieldnames):
        if not rows:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        exists = os.path.exists(path)
        with open(path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not exists:
                writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, '') for key in fieldnames})

    def _to_float(self, value):
        if isinstance(value, torch.Tensor):
            value = value.detach().float().mean().item()
        try:
            value = float(value)
        except (TypeError, ValueError):
            return ''
        if not math.isfinite(value):
            return ''
        return value

    def _relation_metric_key(self, prefix, target_name, source_name):
        safe_target = ''.join(ch if ch.isalnum() else '_' for ch in str(target_name))
        safe_source = ''.join(ch if ch.isalnum() else '_' for ch in str(source_name))
        return f'{prefix}_{safe_target}_{safe_source}'

    def _retrieval_disabled(self):
        return bool(int(getattr(self.args, 'disable_retrieval', 0)))

    def _oracle_train_mode(self):
        return getattr(self.args, 'stage2_oracle_train_mode', 'none')

    def _encoder_free_full_oracle(self):
        return self._oracle_train_mode() == 'full'

    def _active_relation_order(self):
        if self._retrieval_disabled():
            return None
        model = self.model.module if hasattr(self.model, 'module') else self.model
        return [
            model.source_channels(channel)
            for channel in model.target_channels()
        ]

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        encoder_init = getattr(self.args, 'stage1_encoder_init', 'checkpoint')
        retrieval_backbone = getattr(self.args, 'stage2_retrieval_backbone', 'stage1')
        if self._retrieval_disabled():
            print('[stage2] retrieval disabled; skipping Stage-1 encoder checkpoint initialization')
        elif self._encoder_free_full_oracle():
            print('[stage2] encoder-free Full Oracle; skipping Stage-1 encoder checkpoint initialization')
        elif retrieval_backbone == 'identity':
            print('[stage2] using encoder-free identity relation retrieval')
        elif retrieval_backbone == 'pearson':
            print('[stage2] using encoder-free raw Pearson relation retrieval')
        elif retrieval_backbone == 'chronos':
            print(f'[stage2] using frozen Chronos backbone: {self.args.chronos_model_id}')
        elif encoder_init == 'checkpoint':
            if not self.args.stage1_ckpt_path:
                raise ValueError('--stage1_ckpt_path is required when --stage1_encoder_init checkpoint')
            model.load_stage1_checkpoint(self.args.stage1_ckpt_path, strict=True)
        elif encoder_init == 'random':
            print('[stage2] using random Stage-1 encoder initialization')
        else:
            raise ValueError(f'Unsupported stage1_encoder_init: {encoder_init}')
        if self._oracle_train_mode() != 'none':
            if self._retrieval_disabled():
                raise ValueError('Stage-2 oracle training requires retrieval enabled')
            if (
                not self._encoder_free_full_oracle()
                and not bool(int(getattr(self.args, 'freeze_stage1_encoder', 0)))
            ):
                raise ValueError('Stage-2 oracle training requires --freeze_stage1_encoder 1')
            if getattr(self.args, 'relation_mixer_input', 'retrieved') != 'retrieved':
                raise ValueError(
                    'Stage-2 oracle training requires --relation_mixer_input retrieved '
                    'because encoder-free Oracle branches do not provide query embeddings'
                )
            print(f'[stage2] oracle training mode: {self._oracle_train_mode()}')
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag, shuffle=None):
        return data_provider(self.args, flag, shuffle=shuffle)

    def _select_optimizer(self):
        trainable = [
            (name, param)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        ]
        if self._retrieval_disabled():
            unexpected = [
                name
                for name, _ in trainable
                if not (
                    name.startswith('base_head.')
                    or name.startswith('module.base_head.')
                )
            ]
            if unexpected:
                raise RuntimeError(
                    'No Retrieval must train only the base predictor; '
                    f'unexpected trainable parameters: {unexpected}'
                )
        if not trainable:
            raise RuntimeError('Stage-2 has no trainable parameters')
        if self._chronos_finetune():
            # The pretrained encoder needs a far smaller step than the freshly
            # initialised heads; a single Stage-2 learning rate would destroy it.
            chronos_params = [
                param for name, param in trainable if '_chronos_encoder.' in name
            ]
            other_params = [
                param for name, param in trainable if '_chronos_encoder.' not in name
            ]
            if not chronos_params:
                raise RuntimeError(
                    'chronos_finetune=1 but no Chronos parameters are trainable; '
                    'the encoder was not registered as a submodule'
                )
            chronos_lr = float(getattr(self.args, 'chronos_lr', -1.0))
            if chronos_lr <= 0:
                # Default to the single Stage-2 learning rate; pass --chronos_lr
                # to give the pretrained encoder a smaller step if it diverges.
                chronos_lr = float(self.args.learning_rate)
            print(
                f'[stage2] chronos fine-tuning: {len(chronos_params)} tensors at lr={chronos_lr}, '
                f'{len(other_params)} tensors at lr={self.args.learning_rate}'
            )
            # initial_lr is what keeps adjust_learning_rate from collapsing the
            # two groups onto a single value at the first epoch boundary;
            # lr_decay=0 additionally holds the encoder at chronos_lr instead of
            # halving it out of existence within a few epochs.
            chronos_lr_decay = bool(int(getattr(self.args, 'chronos_lr_decay', 0)))
            print(
                f'[stage2] chronos lr schedule: '
                f'{"decays with the Stage-2 schedule" if chronos_lr_decay else f"held at {chronos_lr}"}'
            )
            return optim.Adam([
                {'params': other_params, 'lr': self.args.learning_rate,
                 'initial_lr': self.args.learning_rate},
                {'params': chronos_params, 'lr': chronos_lr,
                 'initial_lr': chronos_lr, 'lr_decay': chronos_lr_decay},
            ])
        params = [param for _, param in trainable]
        return optim.Adam(params, lr=self.args.learning_rate)

    def _ensure_memory(self):
        if self._retrieval_disabled():
            self.memory_bank = None
            self.memory_y = None
            self.memory_x_last = None
            return
        if self.memory_bank is not None:
            return
        train_data, _ = self._get_data(flag='train', shuffle=False)
        self.memory_bank = RelationMemoryBank(
            train_data,
            seq_len=self.args.seq_len,
            pred_len=self.args.pred_len,
            mask_mode=self.args.candidate_mask,
        )
        self.relation_graph = load_or_build_relation_graph(
            train_data,
            self.args,
            require_existing=(
                relation_graph_enabled(self.args)
                and getattr(self.args, 'stage1_encoder_init', 'checkpoint') == 'checkpoint'
                and getattr(self.args, 'stage2_retrieval_backbone', 'stage1') == 'stage1'
            ),
        )
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.set_relation_graph(self.relation_graph)
        if self.relation_graph is not None:
            self.args.relation_channel_names = self.relation_graph['channel_names']
        self.memory_y = torch.from_numpy(self.memory_bank.memory_y).float()
        self.memory_x_last = torch.from_numpy(self.memory_bank.memory_x[:, -1, :]).float()
        print(f'[stage2] memory_x={tuple(self.memory_bank.memory_x.shape)} memory_y={tuple(self.memory_bank.memory_y.shape)}')

    def _build_key_bank(self, force=False):
        if self._retrieval_disabled() or self._encoder_free_full_oracle():
            self.key_bank = None
            self.teacher_key_bank = None
            return
        if self.key_bank is not None and not force:
            return
        model = self.model.module if hasattr(self.model, 'module') else self.model
        self.key_bank = model.build_memory_key_bank(
            self.memory_bank.memory_x,
            self.device,
            chunk_size=self.args.memory_chunk_size,
        )
        print(f'[stage2] built relation key memory bank: {tuple(self.key_bank.shape)}')
        if self._uses_ema_retrieval_teacher():
            # The EMA weights move with the student, so the teacher bank is stale
            # the moment the student bank is. Rebuild them together, always.
            self.teacher_key_bank = model.build_teacher_key_bank(
                self.memory_bank.memory_y,
                self.device,
                self.memory_x_last,
                chunk_size=self.args.memory_chunk_size,
            )
            print(
                f'[stage2] built EMA teacher future key bank: '
                f'{tuple(self.teacher_key_bank.shape)}'
            )

    def _uses_ema_retrieval_teacher(self):
        return (
            float(getattr(self.args, 'retrieval_kl_weight', 0.0)) != 0.0
            and getattr(self.args, 'retrieval_kl_teacher', 'ema') == 'ema'
            and not self._retrieval_disabled()
        )

    def _ema_momentum(self):
        total = max(int(getattr(self, 'total_update_steps', 1)), 1)
        base = float(getattr(self.args, 'stage1_ema_momentum_base', 0.99))
        final = float(getattr(self.args, 'stage1_ema_momentum_final', 0.9995))
        if total <= 1:
            return final
        progress = min(float(getattr(self, 'global_update_step', 0)) / float(total - 1), 1.0)
        return final - (final - base) * (math.cos(math.pi * progress) + 1.0) / 2.0

    def _chronos_finetune(self):
        return (
            getattr(self.args, 'stage2_retrieval_backbone', 'stage1') == 'chronos'
            and bool(int(getattr(self.args, 'chronos_finetune', 0)))
        )

    def _chronos_projection_trainable(self):
        return (
            getattr(self.args, 'stage2_retrieval_backbone', 'stage1') == 'chronos'
            and bool(int(getattr(self.args, 'chronos_projection_trainable', 0)))
        )

    def _chronos_retrieval_trainable(self):
        """Anything that makes the Chronos retrieval space move during training."""
        return self._chronos_finetune() or self._chronos_projection_trainable()

    def _retrieval_space_trainable(self):
        """True whenever the embedding space behind the key bank still moves.

        The key bank has to be rebuilt every epoch in exactly these cases, and
        freeze_stage1_encoder alone cannot decide it: a Chronos run passes
        freeze_stage1_encoder=1 because it has no Stage-1 encoder at all, which
        previously suppressed the rebuild while Chronos itself was training.
        """
        if getattr(self.args, 'stage2_retrieval_backbone', 'stage1') == 'chronos':
            return self._chronos_retrieval_trainable()
        return not bool(int(getattr(self.args, 'freeze_stage1_encoder', 1)))

    def _use_retrieval_cache(self):
        if self._chronos_retrieval_trainable():
            # Retrieval outputs change every step once the encoder trains, so a
            # cache computed before the first epoch would silently be reused.
            return False
        return (
            not self._retrieval_disabled()
            and (
                self._encoder_free_full_oracle()
                or bool(int(getattr(self.args, 'freeze_stage1_encoder', 0)))
            )
            and (
                self._oracle_train_mode() != 'none'
                or not relation_graph_enabled(self.args)
                or getattr(self.args, 'stage2_retrieval_backbone', 'stage1')
                in ('identity', 'pearson', 'chronos')
            )
        )

    def _use_oracle_evaluation_cache(self, split):
        """Allow a test-only cache for oracle metrics with a frozen retriever.

        Relation-graph Stage-1 backbones intentionally do not use the normal
        train/validation retrieval cache.  Oracle candidate evaluation still
        needs the same per-query retrieval outputs plus ground-truth oracle
        candidates, so build that cache only for the test split.
        """
        # A trainable retrieval encoder is still fixed by the time test() runs:
        # it restores the best checkpoint and forces a key-bank rebuild before
        # this cache is built, so the cache matches the weights being evaluated.
        # Requiring freeze_stage1_encoder here would deny oracle metrics to
        # exactly the end-to-end runs that need them most - recall@k is how a
        # collapsed retriever is caught.
        return (
            split == 'test'
            and bool(int(getattr(self.args, 'oracle_candidate_eval', 0)))
            and not self._retrieval_disabled()
        )

    def _use_retrieval_cache_for_split(self, split):
        return self._use_retrieval_cache() or self._use_oracle_evaluation_cache(split)

    def _build_retrieval_cache(self, split, loader):
        if not self._use_retrieval_cache_for_split(split) or split in self.retrieval_caches:
            return
        model = self.model.module if hasattr(self.model, 'module') else self.model
        was_training = self.model.training
        self.model.eval()
        if self._encoder_free_full_oracle():
            cache_parts = {
                # The mixer is configured with input_mode=retrieved, so this
                # zero-width tensor satisfies the common forward interface
                # without retaining an unused encoder representation.
                'relation_query_embs': [],
            }
        else:
            cache_parts = {
                'relation_outputs': [],
                'relation_query_embs': [],
                'alpha_entropy': [],
                'alpha_top1': [],
                'alpha_margin': [],
                'top_k_effective': [],
                'topk_mean_similarity': [],
                'topk_weight_entropy': [],
            }
        oracle_training = self._oracle_train_mode() != 'none'
        oracle_evaluation = (
            split == 'test'
            and bool(int(getattr(self.args, 'oracle_candidate_eval', 0)))
        )
        if oracle_training:
            oracle_output_key = {
                'candidate': 'candidate_oracle_relation_outputs',
                'relation': 'relation_oracle_relation_outputs',
                'full': 'full_oracle_relation_outputs',
            }[self._oracle_train_mode()]
            cache_parts[oracle_output_key] = []
        if oracle_evaluation:
            cache_parts.update({
                'candidate_oracle_mse_sc': [],
                'candidate_oracle_mae_sc': [],
                'full_oracle_mse_sc': [],
                'full_oracle_mae_sc': [],
                'candidate_oracle_top_k_effective_sc': [],
                'candidate_oracle_indices_sc': [],
                'candidate_oracle_mse_topk_sc': [],
                'candidate_oracle_valid_topk_sc': [],
                'student_relation_oracle_recall_at_1_sc': [],
                'student_relation_oracle_recall_at_5_sc': [],
                'student_relation_oracle_recall_at_10_sc': [],
            })
        starts = []
        active_key_bank = self.key_bank
        if (
            active_key_bank is not None
            and getattr(self.args, 'stage2_retrieval_backbone', 'stage1') == 'chronos'
        ):
            active_key_bank = active_key_bank.to(self.device)
        with torch.no_grad():
            for batch_x, batch_y, batch_start_idx in loader:
                batch_x, batch_y, batch_start_idx = self._move_batch(batch_x, batch_y, batch_start_idx)
                cand_mask, counts = self._candidate_mask(batch_start_idx)
                cache = model.build_retrieval_cache(
                    batch_x=batch_x,
                    memory_y=self.memory_y,
                    valid_mask=cand_mask,
                    key_bank=active_key_bank,
                    memory_x_last=self.memory_x_last,
                    oracle_target_y=(
                        batch_y
                        if oracle_training or oracle_evaluation
                        else None
                    ),
                    full_oracle_only=self._encoder_free_full_oracle(),
                )
                for key in cache_parts:
                    if key in cache:
                        cache_parts[key].append(cache[key].detach().cpu())
                starts.extend(int(value) for value in batch_start_idx.cpu().tolist())
        if was_training:
            self.model.train()
        built = {
            key: torch.cat(parts, dim=0)
            for key, parts in cache_parts.items()
            if parts
        }
        built['starts'] = torch.tensor(starts, dtype=torch.long)
        built['start_to_row'] = {start: row for row, start in enumerate(starts)}
        self.retrieval_caches[split] = built
        del active_key_bank
        print(f'[stage2] built {split} retrieval cache: {len(starts)} windows')

    def _cached_retrieval_for_batch(self, split, batch_start_idx):
        if not self._use_retrieval_cache_for_split(split) or split not in self.retrieval_caches:
            return None
        cache = self.retrieval_caches[split]
        try:
            rows = [
                cache['start_to_row'][int(value)]
                for value in batch_start_idx.cpu().tolist()
            ]
        except KeyError:
            return None
        row_idx = torch.tensor(rows, dtype=torch.long)
        selected = {
            key: value.index_select(0, row_idx)
            for key, value in cache.items()
            if key not in {'starts', 'start_to_row'}
        }
        oracle_key = {
            'candidate': 'candidate_oracle_relation_outputs',
            'relation': 'relation_oracle_relation_outputs',
            'full': 'full_oracle_relation_outputs',
        }.get(self._oracle_train_mode())
        if oracle_key is not None:
            selected['relation_outputs'] = selected[oracle_key]
        return selected

    def _e2e_extras(self, split, batch_start_idx):
        """Raw candidate windows and residual teachers for the end-to-end path."""
        if not bool(int(getattr(self.args, 'stage2_e2e', 0))):
            return {}
        if not hasattr(self, '_candidate_x'):
            self._candidate_x = torch.from_numpy(
                self.memory_bank.memory_x).float().to(self.device)
            print(f'[stage2] e2e candidate histories {tuple(self._candidate_x.shape)}')
        extras = {'candidate_x': self._candidate_x}
        root = getattr(self.args, 'stage2_residual_cache', '')
        if not root:
            return extras
        if not hasattr(self, '_residual_cache_value'):
            from scripts.precompute_residual_teacher import load

            path = Path(root)
            if path.is_dir():
                path = path / f'{self.args.data}_pred{self.args.pred_len}.pt'
            cache = load(path)
            meta = cache['meta']
            if meta['dataset'] != self.args.data or meta['pred_len'] != int(self.args.pred_len):
                raise ValueError(f'residual cache {path} was built for another setting')
            cache['memory_residual'] = cache['memory_residual'].to(self.device)
            self._residual_cache_value = cache
            print(f'[stage2] residual teacher {path} '
                  f'memory_residual={tuple(cache["memory_residual"].shape)}')
        cache = self._residual_cache_value
        # Stage-2 names the validation split 'vali'; the cache was built through
        # the data provider, which calls it 'val'.
        key = split if split in cache['splits'] else {'vali': 'val', 'val': 'vali'}.get(split)
        if key not in cache['splits']:
            raise KeyError(
                f'residual cache has splits {sorted(cache["splits"])}, not {split!r}'
            )
        part = cache['splits'][key]
        try:
            rows = [part['start_to_row'][int(v)] for v in batch_start_idx.cpu().tolist()]
        except KeyError:
            raise KeyError(f'residual cache for {split} does not cover this batch')
        index = torch.tensor(rows, dtype=torch.long)
        extras['query_residual'] = part['query_residual'].index_select(0, index).to(self.device)
        extras['memory_residual'] = cache['memory_residual']
        return extras

    def _gradient_norms(self):
        """Per-module gradient norms, so "end-to-end" is verified, not assumed.

        The point of this experiment is that the forecast loss reaches the Stage-1
        encoder. If `grad_norm_stage1_encoder` is zero the arm is not end-to-end,
        whatever the flags say, so it is measured every step rather than trusted.
        """
        if not bool(int(getattr(self.args, 'stage2_e2e', 0))):
            return {}
        model = self.model.module if hasattr(self.model, 'module') else self.model
        modules = {
            'grad_norm_stage1_encoder': getattr(model, 'stage1_encoder', None),
            'grad_norm_base_head': getattr(model, 'base_head', None),
            'grad_norm_relation_mixer': getattr(model, 'relation_mixer', None),
            'grad_norm_gate': getattr(model, 'gate', None),
        }
        out = {}
        for name, module in modules.items():
            if module is None:
                continue
            total = 0.0
            for parameter in module.parameters():
                if parameter.grad is not None:
                    total += float(parameter.grad.detach().float().pow(2).sum())
            out[name] = total ** 0.5
        return out

    def _ema_enabled(self):
        """EMA teacher is off in the end-to-end arms; the flag keeps it reproducible."""
        return bool(int(getattr(self.args, 'use_ema_teacher', 1)))

    def _candidate_mask(self, batch_start_idx):
        cand_mask, counts = self.memory_bank.valid_mask_batch(batch_start_idx.cpu().numpy())
        return cand_mask.bool().to(self.device), counts

    def _move_batch(self, batch_x, batch_y, batch_start_idx):
        return (
            batch_x.float().to(self.device),
            batch_y.float().to(self.device),
            batch_start_idx.long(),
        )

    def _metrics(self, y_final, y_base, y_ret, batch_y, beta, lam, debug, counts, valid_query):
        y_final = y_final[valid_query]
        y_base = y_base[valid_query]
        y_ret = y_ret[valid_query]
        batch_y = batch_y[valid_query]
        beta = beta[valid_query]
        lam = lam[valid_query]

        final_mse = torch.mean((y_final - batch_y) ** 2)
        final_mae = torch.mean(torch.abs(y_final - batch_y))
        if self._retrieval_disabled():
            return {
                'final_mse': final_mse.detach(),
                'final_mae': final_mae.detach(),
            }

        base_mse = torch.mean((y_base - batch_y) ** 2)
        ret_mse = torch.mean((y_ret - batch_y) ** 2)
        base_mae = torch.mean(torch.abs(y_base - batch_y))
        ret_mae = torch.mean(torch.abs(y_ret - batch_y))
        ret_gain = base_mse - ret_mse
        base_err_sc = ((y_base - batch_y) ** 2).mean(dim=1)
        ret_err_sc = ((y_ret - batch_y) ** 2).mean(dim=1)
        final_err_sc = ((y_final - batch_y) ** 2).mean(dim=1)
        ret_advantage = base_err_sc - ret_err_sc
        ret_better = ret_err_sc < base_err_sc
        final_mse_by_channel = ((y_final - batch_y) ** 2).mean(dim=(0, 1))
        base_mse_by_channel = ((y_base - batch_y) ** 2).mean(dim=(0, 1))
        ret_mse_by_channel = ((y_ret - batch_y) ** 2).mean(dim=(0, 1))
        channel_gain = base_mse_by_channel - final_mse_by_channel

        source_indices = debug.get('source_indices')
        if source_indices is None:
            source_indices = torch.arange(
                beta.size(1), device=beta.device, dtype=torch.long
            ).unsqueeze(0).expand(beta.size(1), -1)
        source_indices = source_indices.to(beta.device)
        target_indices = torch.arange(
            beta.size(1), device=beta.device, dtype=torch.long
        ).unsqueeze(-1)
        self_mask = (source_indices == target_indices).unsqueeze(0)
        self_beta = beta.masked_select(self_mask).mean()
        cross_mask = ~self_mask
        cross_beta = (
            beta.masked_select(cross_mask).mean()
            if cross_mask.sum() > 0
            else torch.tensor(0.0, device=beta.device)
        )
        beta_sorted = torch.sort(beta, dim=-1, descending=True).values
        beta_max = beta_sorted[..., 0].mean()
        beta_margin = (
            beta_sorted[..., 0] - beta_sorted[..., 1]
            if beta_sorted.size(-1) > 1
            else beta_sorted[..., 0]
        ).mean()

        metrics = {
            'stage2_loss': final_mse.detach(),
            'final_mse': final_mse.detach(),
            'final_mae': final_mae.detach(),
            'base_mse': base_mse.detach(),
            'base_mae': base_mae.detach(),
            'ret_mse': ret_mse.detach(),
            'ret_mae': ret_mae.detach(),
            'retrieval_gain': (base_mse - final_mse).detach(),
            'retrieval_gain_vs_base': (base_mse - final_mse).detach(),
            'ret_gain': ret_gain.detach(),
            'ret_better_frac': ret_better.float().mean().detach(),
            'beta_self_mean': self_beta.detach(),
            'beta_cross_mean': cross_beta.detach(),
            'beta_self_minus_cross': (self_beta - cross_beta).detach(),
            'beta_max_mean': beta_max.detach(),
            'beta_margin_mean': beta_margin.detach(),
            'lambda_mean': lam.mean().detach(),
            'lambda_std': lam.std(unbiased=False).detach(),
            'lambda_p10': torch.quantile(lam.float().reshape(-1), 0.10).detach(),
            'lambda_p50': torch.quantile(lam.float().reshape(-1), 0.50).detach(),
            'lambda_p90': torch.quantile(lam.float().reshape(-1), 0.90).detach(),
            'lambda_ret_adv_corr': self._safe_corr(lam, ret_advantage).detach(),
            'lambda_min': lam.min().detach(),
            'lambda_max': lam.max().detach(),
            'channel_gain_mean': channel_gain.mean().detach(),
            'channel_gain_min': channel_gain.min().detach(),
            'channel_gain_max': channel_gain.max().detach(),
            'frac_channels_improved': (channel_gain > 0).float().mean().detach(),
            'num_channels_improved': (channel_gain > 0).float().sum().detach(),
            'relation_source_count': float(beta.size(-1)),
            'valid_candidate_count_mean': counts.float().mean().item(),
            'valid_candidate_count_min': counts.min().item() if counts.numel() else 0.0,
            'valid_candidate_fraction': (
                counts.float().mean().item() / max(float(self.memory_y.size(0)), 1.0)
                if self.memory_y is not None
                else 0.0
            ),
        }
        source_correlations = debug.get('source_correlations')
        if source_correlations is not None:
            selected_cross_corr = source_correlations.to(beta.device).masked_select(
                cross_mask.squeeze(0)
            )
            metrics.update({
                'pearson_selected_mean': selected_cross_corr.mean().detach(),
                'abs_pearson_selected_mean': selected_cross_corr.abs().mean().detach(),
                'negative_pearson_frac': (selected_cross_corr < 0).float().mean().detach(),
            })
        channel_names = self._channel_names(beta.size(1))
        focus_idx, focus_name = self._focus_channel_index(beta.size(1))
        if focus_idx is not None and focus_idx < y_final.size(-1):
            metrics.update({
                f'final_mse_{focus_name}': final_mse_by_channel[focus_idx].detach(),
                f'base_mse_{focus_name}': base_mse_by_channel[focus_idx].detach(),
                f'ret_mse_{focus_name}': ret_mse_by_channel[focus_idx].detach(),
                f'final_gain_{focus_name}': channel_gain[focus_idx].detach(),
                f'ret_gain_{focus_name}': (base_mse_by_channel[focus_idx] - ret_mse_by_channel[focus_idx]).detach(),
                f'ret_better_frac_{focus_name}': ret_better[:, focus_idx].float().mean().detach(),
                f'lambda_{focus_name}': lam[:, focus_idx].mean().detach(),
                f'lambda_ret_adv_corr_{focus_name}': self._safe_corr(
                    lam[:, focus_idx], ret_advantage[:, focus_idx]
                ).detach(),
            })
        if 'relation_outputs' in debug:
            relation_outputs = debug['relation_outputs'][valid_query]
            target_y = batch_y.permute(0, 2, 1).unsqueeze(2)
            relation_mse = ((relation_outputs - target_y) ** 2).mean(dim=-1)
            relation_mse_mean = relation_mse.mean()
            relation_mse_best, best_relation = relation_mse.min(dim=-1)
            relation_mae = torch.abs(relation_outputs - target_y).mean(dim=-1)
            relation_mae_best = relation_mae.gather(
                -1, best_relation.unsqueeze(-1)
            ).squeeze(-1)
            beta_choice = beta.argmax(dim=-1)
            beta_expected_mse = (beta * relation_mse).sum(dim=-1)

            relation_mse_self = relation_mse.masked_select(self_mask).mean()
            relation_mse_cross = relation_mse.masked_fill(self_mask, float('inf'))
            relation_mse_cross_best = relation_mse_cross.min(dim=-1).values
            relation_mse_cross_best = relation_mse_cross_best[torch.isfinite(relation_mse_cross_best)].mean()

            metrics.update({
                'relation_mse_mean': relation_mse_mean.detach(),
                'relation_mse_best': relation_mse_best.mean().detach(),
                'relation_mse_self': relation_mse_self.detach(),
                'relation_mse_cross_best': relation_mse_cross_best.detach(),
                'beta_expected_relation_mse': beta_expected_mse.mean().detach(),
                'beta_best_relation_top1_match': (beta_choice == best_relation).float().mean().detach(),
                'beta_gain_vs_uniform': (relation_mse_mean - beta_expected_mse.mean()).detach(),
                'beta_regret_vs_best_relation': (beta_expected_mse - relation_mse_best).mean().detach(),
            })
            source_indices_cpu = source_indices.detach().long().cpu()
            for tgt_idx, tgt_name in enumerate(channel_names):
                if tgt_idx >= relation_mse.size(1):
                    continue
                for source_slot in range(relation_mse.size(2)):
                    source_idx = int(source_indices_cpu[tgt_idx, source_slot].item())
                    source_name = channel_names[source_idx]
                    metrics[self._relation_metric_key('relation_mse', tgt_name, source_name)] = (
                        relation_mse[:, tgt_idx, source_slot].mean().detach()
                    )
                    metrics[self._relation_metric_key('relation_mae', tgt_name, source_name)] = (
                        relation_mae[:, tgt_idx, source_slot].mean().detach()
                    )
            if 'topk_mean_similarity' in debug:
                topk_similarity = debug['topk_mean_similarity'][valid_query]
                topk_entropy = debug.get('topk_weight_entropy')
                topk_entropy = None if topk_entropy is None else topk_entropy[valid_query]
                for tgt_idx, tgt_name in enumerate(channel_names):
                    if tgt_idx >= topk_similarity.size(1):
                        continue
                    for source_slot in range(topk_similarity.size(2)):
                        source_idx = int(source_indices_cpu[tgt_idx, source_slot].item())
                        source_name = channel_names[source_idx]
                        metrics[self._relation_metric_key('topk_mean_similarity', tgt_name, source_name)] = (
                            topk_similarity[:, tgt_idx, source_slot].mean().detach()
                        )
                        if topk_entropy is not None:
                            metrics[self._relation_metric_key('topk_weight_entropy', tgt_name, source_name)] = (
                                topk_entropy[:, tgt_idx, source_slot].mean().detach()
                            )
            beta_rank = self._rank_average(-beta)
            quality_rank = self._rank_average(relation_mse)
            metrics['beta_relation_rank_corr'] = self._safe_corr(beta_rank, quality_rank).detach()
            if 'OT' in channel_names:
                ot_idx = channel_names.index('OT')
                if ot_idx < beta.size(1):
                    ot_beta = beta[:, ot_idx, :]
                    ot_relation_mse = relation_mse[:, ot_idx, :]
                    ot_best_relation = ot_relation_mse.argmin(dim=-1)
                    ot_beta_choice = ot_beta.argmax(dim=-1)
                    ot_beta_top1 = ot_beta.max(dim=-1).values
                    ot_mse_best = ot_relation_mse.gather(1, ot_best_relation[:, None]).squeeze(1)
                    ot_mse_beta_top1 = ot_relation_mse.gather(1, ot_beta_choice[:, None]).squeeze(1)
                    ot_beta_on_best_relation = ot_beta.gather(1, ot_best_relation[:, None]).squeeze(1)
                    metrics['beta_OT_best_relation_top1_match'] = (
                        ot_beta_choice == ot_best_relation
                    ).float().mean().detach()
                    metrics['beta_OT_top1_mean'] = ot_beta_top1.mean().detach()
                    metrics['beta_OT_on_best_relation_mean'] = ot_beta_on_best_relation.mean().detach()
                    metrics['relation_mse_OT_best'] = ot_mse_best.mean().detach()
                    metrics['relation_mse_OT_beta_top1'] = ot_mse_beta_top1.mean().detach()
                    metrics['relation_mse_OT_beta_regret'] = (
                        ot_mse_beta_top1 - ot_mse_best
                    ).mean().detach()
        if 'candidate_oracle_mse_sc' in debug:
            candidate_oracle_mse_sc = debug['candidate_oracle_mse_sc'][valid_query]
            candidate_oracle_mae_sc = debug['candidate_oracle_mae_sc'][valid_query]
            full_oracle_mse_sc = debug['full_oracle_mse_sc'][valid_query]
            full_oracle_mae_sc = debug['full_oracle_mae_sc'][valid_query]
            candidate_oracle_top_k_effective_sc = debug[
                'candidate_oracle_top_k_effective_sc'
            ][valid_query]

            metrics.update({
                'candidate_oracle_mse': candidate_oracle_mse_sc.mean().detach(),
                'candidate_oracle_mae': candidate_oracle_mae_sc.mean().detach(),
                'candidate_oracle_gain_vs_base': (
                    base_err_sc - candidate_oracle_mse_sc
                ).mean().detach(),
                'candidate_oracle_better_frac': (
                    candidate_oracle_mse_sc < base_err_sc
                ).float().mean().detach(),
                'candidate_oracle_top_k_effective': (
                    candidate_oracle_top_k_effective_sc.float().mean().detach()
                ),
                'relation_oracle_mse': relation_mse_best.mean().detach(),
                'relation_oracle_mae': relation_mae_best.mean().detach(),
                'relation_oracle_gain_vs_base': (
                    base_err_sc - relation_mse_best
                ).mean().detach(),
                'relation_oracle_better_frac': (
                    relation_mse_best < base_err_sc
                ).float().mean().detach(),
                'full_oracle_mse': full_oracle_mse_sc.mean().detach(),
                'full_oracle_mae': full_oracle_mae_sc.mean().detach(),
                'full_oracle_gain_vs_base': (
                    base_err_sc - full_oracle_mse_sc
                ).mean().detach(),
                'full_oracle_better_frac': (
                    full_oracle_mse_sc < base_err_sc
                ).float().mean().detach(),
            })
            for metric_k in (1, 5, 10):
                recall_key = (
                    f'student_relation_oracle_recall_at_{metric_k}_sc'
                )
                if recall_key in debug:
                    recall_sc = debug[recall_key][valid_query]
                    metrics[
                        f'student_relation_oracle_recall_at_{metric_k}'
                    ] = recall_sc.mean().detach()
                    source_indices_cpu = source_indices.detach().long().cpu()
                    for target_idx, target_name in enumerate(channel_names):
                        for source_slot in range(recall_sc.size(2)):
                            source_idx = int(
                                source_indices_cpu[
                                    target_idx, source_slot
                                ].item()
                            )
                            source_name = channel_names[source_idx]
                            metrics[self._relation_metric_key(
                                f'student_relation_oracle_recall_at_{metric_k}',
                                target_name,
                                source_name,
                            )] = recall_sc[
                                :, target_idx, source_slot
                            ].mean().detach()
        for name in ('alpha_entropy', 'beta_entropy', 'top_k_effective'):
            if name in debug:
                metrics[name] = debug[name].detach()
        # Score/weight geometry and ranking diagnostics: this experiment is about
        # whether separation appears at all, so they are logged every epoch.
        for name in (
            'rank_loss', 'rank_loss_term', 'rank_pairs', 'rank_order_accuracy',
            'rank_margin_satisfied', 'rank_mean_student_gap', 'rank_mean_teacher_gap',
            'rank_positives', 'rank_hard_negatives', 'rank_random_negatives',
            'rank_candidates',
            # v2 scope/scale diagnostics: whether the loss is aimed where it
            # was corrected to aim, independent of whether that helped.
            'rank_effective_margin', 'rank_topk_spread', 'rank_pairs_inside_topk',
            'rank_fraction_inside_topk', 'rank_loss_inside_topk',
            'rank_loss_outside_topk', 'rank_loss_share_inside',
            'rank_gap_inside_topk', 'rank_gap_outside_topk',
            'topk_score_mean', 'topk_score_std', 'top1_minus_top10', 'top1_minus_top2',
            'score_range', 'weight_entropy', 'normalized_weight_entropy', 'effective_k',
            'max_weight', 'min_weight', 'max_min_weight_ratio',
            'embedding_pairwise_cosine_mean', 'embedding_pairwise_cosine_std',
            'embedding_variance', 'embedding_dimension_std_mean',
            'embedding_effective_rank', 'embedding_effective_rank_ratio',
            'embedding_dead_dimension_fraction',
        ):
            if name in debug:
                metrics[name] = debug[name].detach()
        if 'alpha_entropy' in metrics:
            denom = math.log(max(int(self.args.top_k), 2))
            metrics['alpha_entropy_norm'] = (metrics['alpha_entropy'] / denom).detach()
        if 'alpha_top1' in debug:
            metrics['alpha_top1_mean'] = debug['alpha_top1'].detach()
        if 'alpha_margin' in debug:
            metrics['alpha_margin_mean'] = debug['alpha_margin'].detach()
        if 'beta_entropy' in metrics:
            metrics['beta_effective_relations'] = torch.exp(metrics['beta_entropy']).detach()
            denom = math.log(max(beta.size(-1), 2))
            metrics['beta_entropy_norm'] = (metrics['beta_entropy'] / denom).detach()
        return metrics

    def _loss(self, y_final, y_base, y_ret, batch_y, debug, valid_query):
        y_final = y_final[valid_query]
        y_base = y_base[valid_query]
        y_ret = y_ret[valid_query]
        batch_y = batch_y[valid_query]

        loss = torch.mean((y_final - batch_y) ** 2)
        if self._retrieval_disabled():
            return loss
        if int(self.args.use_aux_base_loss):
            loss = loss + float(self.args.aux_base_weight) * torch.mean((y_base - batch_y) ** 2)
        if int(self.args.use_aux_ret_loss):
            loss = loss + float(self.args.aux_ret_weight) * torch.mean((y_ret - batch_y) ** 2)
        if float(self.args.beta_entropy_reg) != 0.0 and 'beta_entropy' in debug:
            loss = loss - float(self.args.beta_entropy_reg) * debug['beta_entropy']
        kl_weight = float(getattr(self.args, 'retrieval_kl_weight', 0.0))
        if kl_weight != 0.0 and 'retrieval_kl_per_query' in debug:
            kl = debug['retrieval_kl_per_query'][valid_query]
            if kl.numel() and torch.isfinite(kl).all():
                loss = loss + kl_weight * kl.mean()
        # L = L_forecast + alpha * L_rank. The ranking term supplements the
        # forecast objective; it never replaces it.
        rank_weight = float(getattr(self.args, 'stage2_rank_weight', 0.0))
        if rank_weight != 0.0 and 'rank_loss_term' in debug:
            rank_term = debug['rank_loss_term']
            if torch.isfinite(rank_term):
                loss = loss + rank_weight * rank_term
        return loss

    def _init_csv_accumulators(self):
        return {
            'lambda_bins': [
                {'lo': i / 5.0, 'hi': (i + 1) / 5.0, 'count': 0.0,
                 'base': 0.0, 'ret': 0.0, 'final': 0.0, 'ret_better': 0.0}
                for i in range(5)
            ],
            'ot_relation': None,
            'relation_branches': None,
            'oracle_candidates': [],
        }

    def _update_oracle_candidate_rows(self, acc, retrieval_cache, batch_start_idx, valid_query):
        if retrieval_cache is None or 'candidate_oracle_indices_sc' not in retrieval_cache:
            return
        indices = retrieval_cache['candidate_oracle_indices_sc'].detach().cpu()
        mse = retrieval_cache['candidate_oracle_mse_topk_sc'].detach().cpu()
        valid = retrieval_cache['candidate_oracle_valid_topk_sc'].detach().cpu().bool()
        query_starts = batch_start_idx.detach().cpu().tolist()
        valid_rows = valid_query.detach().cpu().bool().tolist()
        channel_names = self._channel_names(indices.size(1))
        memory_starts = self.memory_bank.memory_starts
        model = self.model.module if hasattr(self.model, 'module') else self.model

        for batch_row, is_valid_query in enumerate(valid_rows):
            if not is_valid_query:
                continue
            query_start = int(query_starts[batch_row])
            for target_idx, target_name in enumerate(channel_names):
                source_indices = model.source_channels(target_idx)
                for source_slot, source_idx in enumerate(source_indices):
                    source_name = channel_names[source_idx]
                    for rank in range(indices.size(-1)):
                        memory_index = int(
                            indices[
                                batch_row,
                                target_idx,
                                source_slot,
                                rank,
                            ].item()
                        )
                        is_valid = bool(
                            valid[
                                batch_row,
                                target_idx,
                                source_slot,
                                rank,
                            ].item()
                        )
                        memory_start = (
                            int(memory_starts[memory_index])
                            if (
                                is_valid
                                and 0 <= memory_index < len(memory_starts)
                            )
                            else ''
                        )
                        acc['oracle_candidates'].append({
                            'query_start': query_start,
                            'target_index': target_idx,
                            'target_channel': target_name,
                            'source_slot': source_slot,
                            'source_index': source_idx,
                            'source_channel': source_name,
                            'oracle_rank': rank + 1,
                            'memory_index': (
                                memory_index if is_valid else ''
                            ),
                            'memory_start': memory_start,
                            'relation_future_mse': (
                                float(
                                    mse[
                                        batch_row,
                                        target_idx,
                                        source_slot,
                                        rank,
                                    ].item()
                                )
                                if is_valid
                                else ''
                            ),
                            'valid': int(is_valid),
                        })

    def _update_csv_accumulators(self, acc, y_final, y_base, y_ret, batch_y, beta, lam, debug, valid_query):
        with torch.no_grad():
            y_final = y_final[valid_query]
            y_base = y_base[valid_query]
            y_ret = y_ret[valid_query]
            batch_y = batch_y[valid_query]
            beta = beta[valid_query]
            lam = lam[valid_query]
            if y_final.numel() == 0:
                return

            base_err = ((y_base - batch_y) ** 2).mean(dim=1)
            ret_err = ((y_ret - batch_y) ** 2).mean(dim=1)
            final_err = ((y_final - batch_y) ** 2).mean(dim=1)
            flat_lam = lam.reshape(-1)
            flat_base = base_err.reshape(-1)
            flat_ret = ret_err.reshape(-1)
            flat_final = final_err.reshape(-1)
            flat_better = (flat_ret < flat_base).float()

            for idx, item in enumerate(acc['lambda_bins']):
                lo = item['lo']
                hi = item['hi']
                if idx == len(acc['lambda_bins']) - 1:
                    mask = (flat_lam >= lo) & (flat_lam <= hi)
                else:
                    mask = (flat_lam >= lo) & (flat_lam < hi)
                count = float(mask.sum().item())
                if count <= 0:
                    continue
                item['count'] += count
                item['base'] += float(flat_base[mask].sum().item())
                item['ret'] += float(flat_ret[mask].sum().item())
                item['final'] += float(flat_final[mask].sum().item())
                item['ret_better'] += float(flat_better[mask].sum().item())

            if 'relation_outputs' not in debug:
                return
            relation_outputs = debug['relation_outputs'][valid_query]
            channel_names = self._channel_names(beta.size(1))
            target_y = batch_y.permute(0, 2, 1).unsqueeze(2)
            relation_mse = ((relation_outputs - target_y) ** 2).mean(dim=-1)
            beta_top1 = beta.argmax(dim=-1)
            best_relation = relation_mse.argmin(dim=-1)
            source_indices = debug.get('source_indices')
            if source_indices is None:
                source_indices = torch.arange(beta.size(-1)).unsqueeze(0).expand(beta.size(1), -1)
            source_indices = source_indices.detach().long().cpu()
            source_correlations = debug.get('source_correlations')
            if source_correlations is not None:
                source_correlations = source_correlations.detach().float().cpu()
            if acc['relation_branches'] is None:
                num_targets = beta.size(1)
                num_sources = beta.size(2)
                acc['relation_branches'] = {
                    'channel_names': channel_names,
                    'source_indices': source_indices,
                    'source_correlations': source_correlations,
                    'count': torch.zeros(num_targets, dtype=torch.float64),
                    'beta_sum': torch.zeros(num_targets, num_sources, dtype=torch.float64),
                    'mse_sum': torch.zeros(num_targets, num_sources, dtype=torch.float64),
                    'top1_count': torch.zeros(num_targets, num_sources, dtype=torch.float64),
                    'best_count': torch.zeros(num_targets, num_sources, dtype=torch.float64),
                }
            branch_state = acc['relation_branches']
            count_by_target = torch.full((beta.size(1),), beta.size(0), dtype=torch.float64)
            branch_state['count'] += count_by_target
            branch_state['beta_sum'] += beta.detach().double().cpu().sum(dim=0)
            branch_state['mse_sum'] += relation_mse.detach().double().cpu().sum(dim=0)
            branch_state['top1_count'] += torch.nn.functional.one_hot(
                beta_top1.detach().cpu(), num_classes=beta.size(-1)
            ).double().sum(dim=0)
            branch_state['best_count'] += torch.nn.functional.one_hot(
                best_relation.detach().cpu(), num_classes=beta.size(-1)
            ).double().sum(dim=0)

            focus_idx, focus_name = self._focus_channel_index(beta.size(1))
            if focus_idx is None or focus_idx >= beta.size(1):
                return
            ot_beta = beta[:, focus_idx, :]
            ot_mse = relation_mse[:, focus_idx, :]
            if acc['ot_relation'] is None:
                focus_source_indices = source_indices[focus_idx].tolist()
                focus_source_correlations = (
                    None
                    if source_correlations is None
                    else source_correlations[focus_idx].tolist()
                )
                acc['ot_relation'] = {
                    'target_channel': focus_name,
                    'target_index': focus_idx,
                    'source_indices': focus_source_indices,
                    'source_names': [channel_names[index] for index in focus_source_indices],
                    'source_correlations': focus_source_correlations,
                    'count': 0.0,
                    'beta_sum': torch.zeros(beta.size(-1), dtype=torch.float64),
                    'mse_sum': torch.zeros(beta.size(-1), dtype=torch.float64),
                }
            count = float(ot_beta.size(0))
            acc['ot_relation']['count'] += count
            acc['ot_relation']['beta_sum'] += ot_beta.detach().double().cpu().sum(dim=0)
            acc['ot_relation']['mse_sum'] += ot_mse.detach().double().cpu().sum(dim=0)

    def _lambda_bin_rows(self, acc, context):
        rows = []
        for item in acc['lambda_bins']:
            count = item['count']
            row = dict(context)
            row['lambda_bin'] = f'{item["lo"]:.1f}-{item["hi"]:.1f}'
            row['count'] = int(count)
            if count > 0:
                row['base_mse'] = item['base'] / count
                row['ret_mse'] = item['ret'] / count
                row['final_mse'] = item['final'] / count
                row['ret_better_frac'] = item['ret_better'] / count
            else:
                row['base_mse'] = ''
                row['ret_mse'] = ''
                row['final_mse'] = ''
                row['ret_better_frac'] = ''
            rows.append(row)
        return rows

    def _ot_relation_rows(self, acc, context):
        state = acc['ot_relation']
        if state is None or state['count'] <= 0:
            return []
        beta_mean = state['beta_sum'] / state['count']
        mse_mean = state['mse_sum'] / state['count']
        beta_order = torch.argsort(beta_mean, descending=True)
        mse_order = torch.argsort(mse_mean, descending=False)
        rank_beta = torch.empty_like(beta_order, dtype=torch.long)
        rank_mse = torch.empty_like(mse_order, dtype=torch.long)
        rank_beta[beta_order] = torch.arange(1, beta_order.numel() + 1, dtype=torch.long)
        rank_mse[mse_order] = torch.arange(1, mse_order.numel() + 1, dtype=torch.long)
        best_slot = int(mse_order[0].item())
        rows = []
        for source_slot, src_name in enumerate(state['source_names']):
            source_index = int(state['source_indices'][source_slot])
            pearson = (
                ''
                if state['source_correlations'] is None
                else float(state['source_correlations'][source_slot])
            )
            row = dict(context)
            row.update({
                'target_channel': state['target_channel'],
                'target_index': int(state['target_index']),
                'source_channel': src_name,
                'source_index': source_index,
                'source_slot': source_slot,
                'pearson': pearson,
                'abs_pearson': '' if pearson == '' else abs(pearson),
                'beta_mean': float(beta_mean[source_slot].item()),
                'relation_mse': float(mse_mean[source_slot].item()),
                'rank_by_beta': int(rank_beta[source_slot].item()),
                'rank_by_relation_mse': int(rank_mse[source_slot].item()),
                'is_self': int(source_index == state['target_index']),
                'is_best_relation': int(source_slot == best_slot),
            })
            rows.append(row)
        return rows

    def _relation_branch_rows(self, acc, context):
        state = acc['relation_branches']
        if state is None:
            return []
        rows = []
        names = state['channel_names']
        source_indices = state['source_indices']
        source_correlations = state['source_correlations']
        for tgt_idx, tgt_name in enumerate(names):
            count = float(state['count'][tgt_idx].item())
            if count <= 0:
                continue
            beta_mean = state['beta_sum'][tgt_idx] / count
            mse_mean = state['mse_sum'][tgt_idx] / count
            top1_frac = state['top1_count'][tgt_idx] / count
            best_frac = state['best_count'][tgt_idx] / count
            beta_order = torch.argsort(beta_mean, descending=True)
            mse_order = torch.argsort(mse_mean, descending=False)
            rank_beta = torch.empty_like(beta_order, dtype=torch.long)
            rank_mse = torch.empty_like(mse_order, dtype=torch.long)
            rank_beta[beta_order] = torch.arange(1, beta_order.numel() + 1, dtype=torch.long)
            rank_mse[mse_order] = torch.arange(1, mse_order.numel() + 1, dtype=torch.long)
            for source_slot in range(source_indices.size(1)):
                source_index = int(source_indices[tgt_idx, source_slot].item())
                src_name = names[source_index]
                pearson = (
                    ''
                    if source_correlations is None
                    else float(source_correlations[tgt_idx, source_slot].item())
                )
                row = dict(context)
                row.update({
                    'target_index': tgt_idx,
                    'target_channel': tgt_name,
                    'source_index': source_index,
                    'source_channel': src_name,
                    'source_slot': source_slot,
                    'pearson': pearson,
                    'abs_pearson': '' if pearson == '' else abs(pearson),
                    'branch': f'{tgt_name}<-{src_name}',
                    'beta_mean': float(beta_mean[source_slot].item()),
                    'beta_top1_frac': float(top1_frac[source_slot].item()),
                    'relation_mse': float(mse_mean[source_slot].item()),
                    'best_relation_frac': float(best_frac[source_slot].item()),
                    'rank_by_beta': int(rank_beta[source_slot].item()),
                    'rank_by_relation_mse': int(rank_mse[source_slot].item()),
                    'is_self': int(source_index == tgt_idx),
                })
                rows.append(row)
        return rows

    def _write_stage2_metric_csvs(self, setting, split, epoch, metrics, acc):
        context = self._csv_context(epoch, split, setting)
        base_dir = self._csv_base_dir(setting)
        if self._retrieval_disabled():
            no_retrieval_row = dict(context)
            no_retrieval_row.update({
                'final_mse': self._to_float(
                    metrics.get('final_mse', float('nan'))
                ),
                'final_mae': self._to_float(
                    metrics.get('final_mae', float('nan'))
                ),
            })
            self._append_csv(
                os.path.join(base_dir, 'metrics_main.csv'),
                [no_retrieval_row],
                list(context.keys()) + ['final_mse', 'final_mae'],
            )
            return
        focus = getattr(self.args, 'focus_channel', 'OT')
        oracle_keys = [
            'base_mse', 'base_mae', 'ret_mse', 'ret_mae',
            'candidate_oracle_mse', 'candidate_oracle_mae',
            'candidate_oracle_gain_vs_base', 'candidate_oracle_better_frac',
            'candidate_oracle_top_k_effective',
            'student_relation_oracle_recall_at_1',
            'student_relation_oracle_recall_at_5',
            'student_relation_oracle_recall_at_10',
            'relation_oracle_mse', 'relation_oracle_mae',
            'relation_oracle_gain_vs_base', 'relation_oracle_better_frac',
            'full_oracle_mse', 'full_oracle_mae',
            'full_oracle_gain_vs_base', 'full_oracle_better_frac',
        ]
        if any(key in metrics for key in ('candidate_oracle_mse', 'full_oracle_mse')):
            oracle_row = dict(context)
            oracle_row['oracle_candidate_definition'] = (
                'concat_target_source_future_mse_topk_encoder_alpha'
            )
            for key in oracle_keys:
                oracle_row[key] = self._to_float(metrics.get(key, float('nan')))
            self._append_csv(
                os.path.join(base_dir, 'metrics_oracle_topk.csv'),
                [oracle_row],
                list(context.keys()) + ['oracle_candidate_definition'] + oracle_keys,
            )
            self._append_csv(
                os.path.join(base_dir, 'metrics_oracle_candidates.csv'),
                [dict(context, **row) for row in acc.get('oracle_candidates', [])],
                list(context.keys()) + [
                    'query_start', 'target_index', 'target_channel',
                    'source_slot', 'source_index', 'source_channel',
                    'oracle_rank', 'memory_index', 'memory_start',
                    'relation_future_mse', 'valid',
                ],
            )
            model = self.model.module if hasattr(self.model, 'module') else self.model
            channel_names = self._channel_names(int(self.args.enc_in))
            branch_recall_rows = []
            for target_idx, target_name in enumerate(channel_names):
                for source_slot, source_idx in enumerate(
                    model.source_channels(target_idx)
                ):
                    source_name = channel_names[source_idx]
                    row = dict(context)
                    row.update({
                        'target_index': target_idx,
                        'target_channel': target_name,
                        'source_slot': source_slot,
                        'source_index': source_idx,
                        'source_channel': source_name,
                    })
                    for metric_k in (1, 5, 10):
                        metric_key = self._relation_metric_key(
                            f'student_relation_oracle_recall_at_{metric_k}',
                            target_name,
                            source_name,
                        )
                        row[f'recall_at_{metric_k}'] = self._to_float(
                            metrics.get(metric_key, float('nan'))
                        )
                    branch_recall_rows.append(row)
            self._append_csv(
                os.path.join(
                    base_dir, 'metrics_oracle_branch_recall.csv'
                ),
                branch_recall_rows,
                list(context.keys()) + [
                    'target_index', 'target_channel',
                    'source_slot', 'source_index', 'source_channel',
                    'recall_at_1', 'recall_at_5', 'recall_at_10',
                ],
            )
            if bool(int(getattr(self.args, 'oracle_candidate_eval', 0))):
                return

        main_keys = [
            'stage2_loss',
            'final_mse', 'final_mae', 'base_mse', 'base_mae', 'ret_mse', 'ret_mae',
            'retrieval_gain', 'retrieval_gain_vs_base', 'ret_gain',
            f'final_mse_{focus}', f'base_mse_{focus}', f'ret_mse_{focus}',
            f'final_gain_{focus}', f'ret_gain_{focus}',
            'ret_better_frac', f'ret_better_frac_{focus}',
            'lambda_mean', 'lambda_std', 'lambda_p10', 'lambda_p50', 'lambda_p90',
            f'lambda_{focus}', 'lambda_ret_adv_corr', f'lambda_ret_adv_corr_{focus}',
            'alpha_entropy_norm', 'alpha_top1_mean', 'alpha_margin_mean',
            'beta_entropy_norm', 'beta_effective_relations', 'beta_max_mean', 'beta_margin_mean',
            'beta_self_mean', 'beta_cross_mean', 'beta_self_minus_cross',
            'relation_source_count', 'pearson_selected_mean',
            'valid_candidate_count_mean', 'valid_candidate_fraction',
            'abs_pearson_selected_mean', 'negative_pearson_frac',
            'relation_mse_self', 'relation_mse_cross_best', 'relation_mse_best',
            'beta_expected_relation_mse', 'beta_best_relation_top1_match', 'beta_relation_rank_corr',
            'beta_gain_vs_uniform', 'beta_regret_vs_best_relation',
            'beta_OT_best_relation_top1_match', 'beta_OT_top1_mean',
            'beta_OT_on_best_relation_mean',
            'relation_mse_OT_best', 'relation_mse_OT_beta_top1', 'relation_mse_OT_beta_regret',
            'channel_gain_mean', 'channel_gain_min', 'channel_gain_max',
            'frac_channels_improved', 'num_channels_improved',
        ]
        main_row = dict(context)
        for key in main_keys:
            main_row[key] = self._to_float(metrics.get(key, float('nan')))
        self._append_csv(
            os.path.join(base_dir, 'metrics_main.csv'),
            [main_row],
            list(context.keys()) + main_keys,
        )

        ot_rows = self._ot_relation_rows(acc, context)
        self._append_csv(
            os.path.join(base_dir, 'metrics_ot_relation.csv'),
            ot_rows,
            list(context.keys()) + [
                'target_index', 'target_channel', 'source_index', 'source_channel',
                'source_slot', 'beta_mean', 'relation_mse',
                'pearson', 'abs_pearson',
                'rank_by_beta', 'rank_by_relation_mse', 'is_self', 'is_best_relation',
            ],
        )

        branch_rows = self._relation_branch_rows(acc, context)
        self._append_csv(
            os.path.join(base_dir, 'metrics_relation_branches.csv'),
            branch_rows,
            list(context.keys()) + [
                'target_index', 'target_channel', 'source_index', 'source_channel',
                'source_slot', 'branch',
                'pearson', 'abs_pearson',
                'beta_mean', 'beta_top1_frac', 'relation_mse', 'best_relation_frac',
                'rank_by_beta', 'rank_by_relation_mse', 'is_self',
            ],
        )

        lambda_rows = self._lambda_bin_rows(acc, context)
        self._append_csv(
            os.path.join(base_dir, 'metrics_lambda_bins.csv'),
            lambda_rows,
            list(context.keys()) + [
                'lambda_bin', 'count', 'base_mse', 'ret_mse', 'final_mse', 'ret_better_frac',
            ],
        )

    def _run_loader(self, loader, optimizer=None, split=None, epoch=None, setting=None):
        train = optimizer is not None
        self.model.train(train)
        model = self.model.module if hasattr(self.model, 'module') else self.model
        if bool(int(self.args.freeze_stage1_encoder)) and model.stage1_encoder is not None:
            model.stage1_encoder.eval()
            model.shared_cross_projection.eval()
        avg = MetricAverager()
        csv_acc = self._init_csv_accumulators()

        for batch_x, batch_y, batch_start_idx in loader:
            batch_x, batch_y, batch_start_idx = self._move_batch(batch_x, batch_y, batch_start_idx)
            if self._retrieval_disabled():
                cand_mask = None
                counts = torch.zeros(batch_x.size(0), dtype=torch.float32, device=batch_x.device)
                valid_query = torch.ones(batch_x.size(0), dtype=torch.bool, device=batch_x.device)
            else:
                cand_mask, counts = self._candidate_mask(batch_start_idx)
                valid_query = counts.to(batch_x.device) > 0
            retrieval_cache = self._cached_retrieval_for_batch(split, batch_start_idx)
            e2e_extras = self._e2e_extras(split, batch_start_idx)
            if valid_query.sum() == 0:
                avg.update({
                    'skipped_batches': 1.0,
                    'valid_candidate_count_mean': counts.float().mean().item(),
                    'valid_candidate_count_min': counts.min().item() if counts.numel() else 0.0,
                }, weight=batch_x.size(0))
                continue

            if train:
                optimizer.zero_grad()
                y_final, y_base, y_ret, beta, lam, debug = self.model(
                    batch_x=batch_x,
                    memory_y=self.memory_y,
                    valid_mask=cand_mask,
                    key_bank=self.key_bank,
                    memory_x_last=self.memory_x_last,
                    retrieval_cache=retrieval_cache,
                    target_y=batch_y,
                    teacher_key_bank=getattr(self, 'teacher_key_bank', None),
                    **e2e_extras,
                )
                loss = self._loss(y_final, y_base, y_ret, batch_y, debug, valid_query)
                loss.backward()
                grad_metrics = self._gradient_norms()
                if grad_metrics:
                    avg.update(grad_metrics)
                optimizer.step()
                if self._uses_ema_retrieval_teacher():
                    if self._ema_enabled():
                        model = self.model.module if hasattr(self.model, 'module') else self.model
                        model.update_ema_teacher(self._ema_momentum())
                    self.global_update_step = getattr(self, 'global_update_step', 0) + 1
            else:
                with torch.no_grad():
                    y_final, y_base, y_ret, beta, lam, debug = self.model(
                        batch_x=batch_x,
                        memory_y=self.memory_y,
                        valid_mask=cand_mask,
                        key_bank=self.key_bank,
                        memory_x_last=self.memory_x_last,
                        retrieval_cache=retrieval_cache,
                        **e2e_extras,
                    )
                    loss = self._loss(y_final, y_base, y_ret, batch_y, debug, valid_query)

            if retrieval_cache is not None and 'candidate_oracle_mse_sc' in retrieval_cache:
                for key in (
                    'candidate_oracle_mse_sc',
                    'candidate_oracle_mae_sc',
                    'full_oracle_mse_sc',
                    'full_oracle_mae_sc',
                    'candidate_oracle_top_k_effective_sc',
                    'student_relation_oracle_recall_at_1_sc',
                    'student_relation_oracle_recall_at_5_sc',
                    'student_relation_oracle_recall_at_10_sc',
                ):
                    debug[key] = retrieval_cache[key].to(batch_x.device)

            with torch.no_grad():
                metrics = self._metrics(y_final, y_base, y_ret, batch_y, beta, lam, debug, counts, valid_query)
                metrics['loss'] = loss.detach()
                metrics['skipped_batches'] = 0.0
                avg.update(metrics, weight=int(valid_query.sum().item()))
                if split is not None and epoch is not None and setting is not None:
                    self._update_csv_accumulators(
                        csv_acc, y_final, y_base, y_ret, batch_y, beta, lam, debug, valid_query
                    )
                    self._update_oracle_candidate_rows(
                        csv_acc, retrieval_cache, batch_start_idx, valid_query
                    )

        averaged = avg.average()
        if split is not None and epoch is not None and setting is not None:
            self._write_stage2_metric_csvs(setting, split, epoch, averaged, csv_acc)
        return averaged

    def vali(self, vali_data, vali_loader, epoch=None, setting=None):
        return self._run_loader(vali_loader, optimizer=None, split='vali', epoch=epoch, setting=setting)

    def train(self, setting):
        if self._chronos_retrieval_trainable() and not bool(int(self.args.refresh_memory_every_epoch)):
            raise ValueError(
                'a trainable Chronos retrieval space requires --refresh_memory_every_epoch 1; '
                'otherwise the memory keys stay in the old embedding space while the query moves'
            )
        self._ensure_memory()
        model = self.model.module if hasattr(self.model, 'module') else self.model
        train_data, train_loader = self._get_data(flag='train', shuffle=True)
        vali_data, vali_loader = self._get_data(flag='val', shuffle=False)
        _, train_cache_loader = self._get_data(flag='train', shuffle=False)

        path = os.path.join(self.args.checkpoints, 'stage2', self.args.data, f'seq{self.args.seq_len}_pred{self.args.pred_len}', setting)
        os.makedirs(path, exist_ok=True)
        optimizer = self._select_optimizer()
        best_val_loss = float('inf')
        best_path = os.path.join(path, 'checkpoint.pth')
        self.best_checkpoint_path = best_path
        bad_epochs = 0

        refresh_each_epoch = bool(int(self.args.refresh_memory_every_epoch))
        self.global_update_step = 0
        self.total_update_steps = max(len(train_loader) * int(self.args.train_epochs), 1)
        self._build_key_bank(force=True)
        self._build_retrieval_cache('train', train_cache_loader)
        self._build_retrieval_cache('vali', vali_loader)
        writer = build_summary_writer(self.args, 'stage2', setting)
        tb_keys = [
            'loss', 'stage2_loss',
            'final_mse', 'final_mae',
            'base_mse', 'base_mae',
            'ret_mse', 'ret_mae',
            'retrieval_gain', 'retrieval_gain_vs_base',
            'lambda_mean',
            'beta_self_mean', 'beta_cross_mean',
            'beta_self_minus_cross', 'beta_max_mean', 'beta_margin_mean',
            'relation_source_count', 'pearson_selected_mean',
            'abs_pearson_selected_mean', 'negative_pearson_frac',
            'alpha_entropy', 'alpha_entropy_norm', 'alpha_top1_mean', 'alpha_margin_mean',
            'beta_entropy', 'beta_entropy_norm',
            'beta_effective_relations',
            'relation_mse_mean', 'relation_mse_best',
            'relation_mse_self', 'relation_mse_cross_best',
            'beta_expected_relation_mse',
            'beta_best_relation_top1_match',
            'beta_relation_rank_corr',
            'beta_gain_vs_uniform', 'beta_regret_vs_best_relation',
            'top_k_effective',
            'valid_candidate_fraction',
            'beta_OT_best_relation_top1_match',
            'beta_OT_top1_mean',
            'beta_OT_on_best_relation_mean',
            'relation_mse_OT_best',
            'relation_mse_OT_beta_top1',
            'relation_mse_OT_beta_regret',
        ]
        focus = getattr(self.args, 'focus_channel', 'OT')
        tb_keys.extend([
            f'lambda_{focus}',
            f'final_mse_{focus}',
            f'base_mse_{focus}',
            f'ret_mse_{focus}',
            f'final_gain_{focus}',
            f'ret_gain_{focus}',
            f'ret_better_frac_{focus}',
            f'lambda_ret_adv_corr_{focus}',
        ])

        try:
            for epoch in range(self.args.train_epochs):
                epoch_time = time.time()
                if refresh_each_epoch and epoch > 0 and self._retrieval_space_trainable():
                    self._build_key_bank(force=True)
                    self.retrieval_caches.clear()

                train_metrics = self._run_loader(
                    train_loader,
                    optimizer=optimizer,
                    split='train',
                    epoch=epoch + 1,
                    setting=setting,
                )
                val_metrics = self.vali(vali_data, vali_loader, epoch=epoch + 1, setting=setting)
                val_loss = val_metrics.get('loss', float('inf'))

                print(format_metrics(f'Epoch {epoch + 1} Train', train_metrics))
                print(format_metrics(f'Epoch {epoch + 1} Vali', val_metrics))
                print('Epoch: {} cost time: {:.2f}s'.format(epoch + 1, time.time() - epoch_time))
                write_metric_scalars(writer, 'train', train_metrics, epoch + 1, tb_keys)
                write_metric_scalars(writer, 'vali', val_metrics, epoch + 1, tb_keys)
                adjust_learning_rate(optimizer, epoch + 1, self.args)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    bad_epochs = 0
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'stage1_ckpt_path': getattr(self.args, 'stage1_ckpt_path', ''),
                        'stage2_retrieval_encoder': getattr(self.args, 'stage2_retrieval_encoder', 'online'),
                        'stage2_retrieval_backbone': getattr(self.args, 'stage2_retrieval_backbone', 'stage1'),
                        'stage2_relation_fusion': getattr(self.args, 'stage2_relation_fusion', 'gate'),
                        'active_relation_order': self._active_relation_order(),
                        'args': vars(self.args),
                        'epoch': epoch + 1,
                        'best_val_loss': best_val_loss,
                    }, best_path)
                    print(f'Saved best Stage-2 checkpoint to {best_path}')
                else:
                    bad_epochs += 1
                    if bad_epochs >= self.args.patience:
                        print('Early stopping')
                        break
        finally:
            if writer is not None:
                writer.close()
        return self.model

    def test(self, setting, test=0):
        if (
            bool(int(getattr(self.args, 'oracle_candidate_eval', 0)))
            and not self._use_oracle_evaluation_cache('test')
        ):
            raise ValueError(
                '--oracle_candidate_eval requires retrieval enabled and a frozen '
                'retrieval backbone (or encoder-free full oracle)'
            )
        path = os.path.join(self.args.checkpoints, 'stage2', self.args.data, f'seq{self.args.seq_len}_pred{self.args.pred_len}', setting)
        ckpt_path = self.best_checkpoint_path or os.path.join(path, 'checkpoint.pth')
        if os.path.exists(ckpt_path):
            print(f'loading Stage-2 checkpoint from {ckpt_path}')
            ckpt = torch.load(ckpt_path, map_location=self.device)
            state = ckpt.get('model_state_dict', ckpt)
            self.model.load_state_dict(state)
        else:
            raise FileNotFoundError(
                f'Stage-2 test checkpoint not found: {ckpt_path}'
            )
        self._ensure_memory()
        saved_relation_order = ckpt.get('active_relation_order')
        if saved_relation_order is not None:
            current_relation_order = self._active_relation_order()
            if saved_relation_order != current_relation_order:
                raise RuntimeError(
                    'Stage-2 checkpoint active relation order does not match current config: '
                    f'checkpoint={saved_relation_order} current={current_relation_order}'
                )
        # A fine-tuned retrieval encoder makes the key bank checkpoint-specific:
        # test() restores the best epoch, so keys from the last epoch are stale.
        self._build_key_bank(force=self._retrieval_space_trainable())
        test_data, test_loader = self._get_data(flag='test', shuffle=False)
        self._build_retrieval_cache('test', test_loader)
        metrics = self._run_loader(test_loader, optimizer=None, split='test', epoch=0, setting=setting)
        if not self._retrieval_disabled():
            print(format_metrics('Stage2 Test', metrics))
        if 'final_mse' in metrics and 'final_mae' in metrics:
            print(
                'Stage2 Test Final\n'
                f'final_mse: {float(metrics["final_mse"]):.6f}\n'
                f'final_mae: {float(metrics["final_mae"]):.6f}'
            )
        if 'candidate_oracle_mse' in metrics:
            print(
                'Stage2 Candidate Oracle Test\n'
                f'candidate_oracle_mse: {float(metrics["candidate_oracle_mse"]):.6f}\n'
                f'candidate_oracle_mae: {float(metrics["candidate_oracle_mae"]):.6f}\n'
                f'candidate_oracle_top_k_effective: '
                f'{float(metrics["candidate_oracle_top_k_effective"]):.2f}\n'
                f'student_relation_oracle_recall_at_1: '
                f'{float(metrics["student_relation_oracle_recall_at_1"]):.6f}\n'
                f'student_relation_oracle_recall_at_5: '
                f'{float(metrics["student_relation_oracle_recall_at_5"]):.6f}\n'
                f'student_relation_oracle_recall_at_10: '
                f'{float(metrics["student_relation_oracle_recall_at_10"]):.6f}\n'
                f'relation_oracle_mse: {float(metrics["relation_oracle_mse"]):.6f}\n'
                f'relation_oracle_mae: {float(metrics["relation_oracle_mae"]):.6f}\n'
                f'full_oracle_mse: {float(metrics["full_oracle_mse"]):.6f}\n'
                f'full_oracle_mae: {float(metrics["full_oracle_mae"]):.6f}'
            )
        return metrics
