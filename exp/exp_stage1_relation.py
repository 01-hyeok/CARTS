import os
import time
import math

import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.relation_memory import RelationMemorySampler, build_memory_index
from utils.stage1_metrics import MetricAverager, format_metrics
from utils.tensorboard_logger import build_summary_writer, write_metric_scalars


class Exp_Stage1_Relation(Exp_Basic):
    def __init__(self, args):
        super(Exp_Stage1_Relation, self).__init__(args)
        self.train_data_for_memory = None
        self.memory_sampler = None
        self.key_bank = None
        self.teacher_key_bank = None
        self.memory_y = None
        self.memory_x_last = None
        self.global_update_step = 0
        self.total_update_steps = 1

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=self.args.learning_rate)

    def _ensure_memory(self):
        if self.memory_sampler is not None:
            return
        train_data, _ = self._get_data(flag='train')
        self.train_data_for_memory = train_data
        self.memory_sampler = RelationMemorySampler(
            train_data,
            seq_len=self.args.seq_len,
            pred_len=self.args.pred_len,
            mask_mode=self.args.candidate_mask,
        )
        self.memory_y = torch.from_numpy(self.memory_sampler.memory_y).float().to(self.device)
        self.memory_x_last = torch.from_numpy(self.memory_sampler.memory_x[:, -1, :]).float().to(self.device)

    def _move_batch(self, batch_x, batch_y, batch_start_idx):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_start_idx = batch_start_idx.long()
        return batch_x, batch_y, batch_start_idx

    def _candidate_mask(self, batch_start_idx):
        cand_mask, counts = self.memory_sampler.valid_mask_batch(batch_start_idx.cpu().numpy())
        return (
            cand_mask.bool().to(self.device),
            counts,
        )

    def _build_key_bank(self):
        model = self.model.module if hasattr(self.model, 'module') else self.model
        self.key_bank = model.build_embedding_bank(
            self.memory_sampler.memory_x,
            self.device,
            chunk_size=self.args.stage1_key_chunk_size,
        ).to(self.device)
        print(f'[stage1] built relation key memory bank: {tuple(self.key_bank.shape)}')
        if self.args.stage1_teacher_mode == 'ema_target':
            self.teacher_key_bank = model.build_teacher_embedding_bank(
                self.memory_sampler.memory_y,
                self.device,
                chunk_size=self.args.stage1_key_chunk_size,
                memory_x_last=self.memory_sampler.memory_x[:, -1, :],
            ).to(self.device)
            print(f'[stage1] built EMA target teacher key memory bank: {tuple(self.teacher_key_bank.shape)}')
        else:
            self.teacher_key_bank = None

    def _ema_momentum(self):
        if self.total_update_steps <= 1:
            return float(self.args.stage1_ema_momentum_final)
        progress = min(float(self.global_update_step) / float(self.total_update_steps - 1), 1.0)
        base = float(self.args.stage1_ema_momentum_base)
        final = float(self.args.stage1_ema_momentum_final)
        return final - (final - base) * (math.cos(math.pi * progress) + 1.0) / 2.0

    def _update_ema_teacher(self):
        if self.args.stage1_teacher_mode != 'ema_target':
            return None
        momentum = self._ema_momentum()
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.update_ema_teacher(momentum)
        self.global_update_step += 1
        return momentum

    def _run_loader(self, loader, optimizer=None):
        train = optimizer is not None
        self.model.train(train)
        if self.args.stage1_teacher_mode == 'ema_target':
            model = self.model.module if hasattr(self.model, 'module') else self.model
            model.teacher_encoder.eval()
        avg = MetricAverager()

        for batch_x, batch_y, batch_start_idx in loader:
            batch_x, batch_y, batch_start_idx = self._move_batch(batch_x, batch_y, batch_start_idx)
            cand_mask, counts = self._candidate_mask(batch_start_idx)
            metrics_extra = {
                'valid_candidate_count_mean': counts.float().mean().item(),
                'valid_candidate_count_min': counts.min().item() if counts.numel() else 0.0,
            }

            if train:
                optimizer.zero_grad()
                loss, metrics = self.model(
                    batch_x, batch_y, cand_mask,
                    memory_y=self.memory_y,
                    key_bank=self.key_bank,
                    teacher_key_bank=self.teacher_key_bank,
                    memory_x_last=self.memory_x_last,
                )
                if torch.isfinite(loss) and loss.requires_grad:
                    loss.backward()
                    optimizer.step()
                    ema_momentum = self._update_ema_teacher()
                    if ema_momentum is not None:
                        metrics = dict(metrics)
                        metrics['ema_momentum'] = ema_momentum
            else:
                with torch.no_grad():
                    loss, metrics = self.model(
                        batch_x, batch_y, cand_mask,
                        memory_y=self.memory_y,
                        key_bank=self.key_bank,
                        teacher_key_bank=self.teacher_key_bank,
                        memory_x_last=self.memory_x_last,
                    )
                    if self.args.stage1_teacher_mode == 'ema_target':
                        metrics = dict(metrics)
                        metrics['ema_momentum'] = self._ema_momentum()

            metrics = dict(metrics)
            metrics.update(metrics_extra)
            avg.update(metrics)

        return avg.average()

    def vali(self, vali_data, vali_loader):
        return self._run_loader(vali_loader, optimizer=None)

    def train(self, setting):
        self._ensure_memory()
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        self.total_update_steps = max(1, len(train_loader) * int(self.args.train_epochs))

        path = os.path.join(self.args.checkpoints, 'stage1', self.args.data, f'seq{self.args.seq_len}_pred{self.args.pred_len}', setting)
        os.makedirs(path, exist_ok=True)
        optimizer = self._select_optimizer()

        best_val_loss = float('inf')
        best_path = os.path.join(path, 'checkpoint.pth')
        bad_epochs = 0
        writer = build_summary_writer(self.args, 'stage1', setting)
        tb_keys = [
            'loss', 'kl', 'self_kl', 'cross_kl',
            'retrieval_gain', 'self_retrieval_gain', 'cross_retrieval_gain',
            'recall@1', 'recall@5',
            'self_recall@1', 'self_recall@5',
            'cross_recall@1', 'cross_recall@5',
            'teacher_entropy', 'student_entropy',
            'teacher_effective_candidates', 'student_effective_candidates',
            'teacher_top1_prob', 'student_top1_prob',
            'student_prob_on_teacher_top1',
            'teacher_student_prob_l1',
            'teacher_student_top5_overlap',
            'student_teacher_top1_match',
            'ema_momentum',
        ]

        try:
            for epoch in range(self.args.train_epochs):
                epoch_time = time.time()
                self._build_key_bank()
                train_metrics = self._run_loader(train_loader, optimizer=optimizer)
                self._build_key_bank()
                val_metrics = self.vali(vali_data, vali_loader)
                val_loss = val_metrics.get('loss', float('inf'))

                print(format_metrics(f'Epoch {epoch + 1} Train', train_metrics))
                print(format_metrics(f'Epoch {epoch + 1} Vali', val_metrics))
                print('Epoch: {} cost time: {:.2f}s'.format(epoch + 1, time.time() - epoch_time))
                write_metric_scalars(writer, 'train', train_metrics, epoch + 1, tb_keys)
                write_metric_scalars(writer, 'vali', val_metrics, epoch + 1, tb_keys)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    bad_epochs = 0
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'args': vars(self.args),
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
            if writer is not None:
                writer.close()

        if getattr(self.args, 'build_memory_index', False):
            build_memory_index(self.model, train_data, self.args)
        return self.model

    def test(self, setting, test=0):
        self._ensure_memory()
        test_data, test_loader = self._get_data(flag='test')
        metrics = self._run_loader(test_loader, optimizer=None)
        print(format_metrics('Stage1 Test', metrics))
        return metrics
