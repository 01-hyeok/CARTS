import os
import time
import math

import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.relation_memory import RelationMemorySampler, build_memory_index
from utils.relation_graph import load_or_build_relation_graph
from utils.stage1_metrics import MetricAverager, format_metrics
from utils.tensorboard_logger import build_summary_writer, write_metric_scalars
from utils.tools import adjust_learning_rate


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
        self.val_probe_batch = None
        self.relation_graph = None

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
        self.memory_y = torch.from_numpy(self.memory_sampler.memory_y).float()
        self.memory_x_last = torch.from_numpy(self.memory_sampler.memory_x[:, -1, :]).float()

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
        self.key_bank = None
        self.key_bank = model.build_embedding_bank(
            self.memory_sampler.memory_x,
            self.device,
            chunk_size=self.args.stage1_key_chunk_size,
        )
        print(f'[stage1] built relation key memory bank: {tuple(self.key_bank.shape)}')
        if model.requires_ema_teacher_bank():
            self.teacher_key_bank = model.build_teacher_embedding_bank(
                self.memory_sampler.memory_y,
                self.device,
                chunk_size=self.args.stage1_key_chunk_size,
                memory_x_last=self.memory_sampler.memory_x[:, -1, :],
            )
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
        model = self.model.module if hasattr(self.model, 'module') else self.model
        if not model.requires_ema_teacher_bank():
            return None
        momentum = self._ema_momentum()
        model.update_ema_teacher(momentum)
        self.global_update_step += 1
        return momentum

    def _run_loader(self, loader, optimizer=None):
        train = optimizer is not None
        self.model.train(train)
        model = self.model.module if hasattr(self.model, 'module') else self.model
        if model.requires_ema_teacher_bank():
            model.teacher_encoder.eval()
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

        for batch_idx, (batch_x, batch_y, batch_start_idx) in enumerate(loader):
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
                    active_target_channels=target_chunks[batch_idx % len(target_chunks)],
                    compute_detailed_metrics=False,
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
                        active_target_channels=target_chunks[batch_idx % len(target_chunks)],
                        compute_detailed_metrics=True,
                    )
                    if model.requires_ema_teacher_bank():
                        metrics = dict(metrics)
                        metrics['ema_momentum'] = self._ema_momentum()

            metrics = dict(metrics)
            metrics.update(metrics_extra)
            avg.update(metrics)

        return avg.average()

    def _set_validation_probe(self, vali_loader):
        model = self.model.module if hasattr(self.model, 'module') else self.model
        if model.loss_mode == 'rnc':
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
        if model.loss_mode == 'rnc':
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
        plt.close(fig)

    def vali(self, vali_data, vali_loader):
        return self._run_loader(vali_loader, optimizer=None)

    def train(self, setting):
        self._ensure_memory()
        train_data, train_loader = self._get_data(flag='train', shuffle=True)
        vali_data, vali_loader = self._get_data(flag='val', shuffle=False)
        self._set_validation_probe(vali_loader)
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
            'stage1_loss_total', 'stage1_loss_kl', 'stage1_loss_rank', 'stage1_loss_rank_weighted',
            'total_loss', 'kl_loss', 'weighted_kl_loss', 'rank_loss', 'rnc_loss',
            'expected_mse_loss', 'weighted_expected_mse_loss',
            'retrieval_gain', 'self_retrieval_gain', 'cross_retrieval_gain',
            'recall@1', 'recall@5',
            'self_recall@1', 'self_recall@5',
            'cross_recall@1', 'cross_recall@5',
            'teacher_entropy', 'student_entropy',
            'student_entropy_normalized', 'student_max_probability',
            'student_top5_probability_mass',
            'student_expected_future_mse_raw',
            'student_expected_future_mse_normalized',
            'teacher_effective_candidates', 'student_effective_candidates',
            'teacher_top1_prob', 'student_top1_prob',
            'student_prob_on_teacher_top1',
            'teacher_student_prob_l1',
            'teacher_student_top5_overlap',
            'student_teacher_top1_match',
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
            'retrieved_future_mse_at_1', 'retrieved_future_mse_at_5',
            'retrieved_future_mse_at_10',
            'oracle_future_mse_at_1', 'oracle_future_mse_at_5',
            'oracle_future_mse_at_10',
            'retrieval_regret_at_1', 'retrieval_regret_at_5',
            'retrieval_regret_at_10',
            'ndcg_at_5', 'ndcg_at_10', 'spearman_score_vs_negative_mse',
            'ema_momentum',
        ]

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
                print(format_metrics(f'Epoch {epoch + 1} Vali', val_metrics))
                print(
                    '[stage1 timing] key_bank_train={:.2f}s train={:.2f}s '
                    'key_bank_val={:.2f}s validation={:.2f}s'.format(
                        key_bank_train_time, train_time, key_bank_val_time, val_time
                    )
                )
                print('Epoch: {} cost time: {:.2f}s'.format(epoch + 1, time.time() - epoch_time))
                write_metric_scalars(writer, 'train', train_metrics, epoch + 1, tb_keys)
                write_metric_scalars(writer, 'vali', val_metrics, epoch + 1, tb_keys)
                self._plot_validation_probe(writer, setting, epoch + 1)
                adjust_learning_rate(optimizer, epoch + 1, self.args)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    bad_epochs = 0
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
            if writer is not None:
                writer.close()

        if getattr(self.args, 'build_memory_index', False):
            build_memory_index(self.model, train_data, self.args)
        return self.model

    def test(self, setting, test=0):
        self._ensure_memory()
        test_data, test_loader = self._get_data(flag='test', shuffle=False)
        metrics = self._run_loader(test_loader, optimizer=None)
        print(format_metrics('Stage1 Test', metrics))
        return metrics
