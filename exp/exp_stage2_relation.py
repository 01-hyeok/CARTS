import os
import time
import csv
import math

import torch
import torch.nn as nn
from torch import optim

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.relation_memory import RelationMemoryBank
from utils.stage1_metrics import MetricAverager, format_metrics
from utils.tensorboard_logger import build_summary_writer, write_metric_scalars


class Exp_Stage2_Relation(Exp_Basic):
    def __init__(self, args):
        super(Exp_Stage2_Relation, self).__init__(args)
        self.memory_bank = None
        self.memory_y = None
        self.memory_x_last = None
        self.key_bank = None
        self.best_checkpoint_path = None
        self.current_setting = None

    def _channel_names(self, num_channels):
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
            'relation_mixer_input': self.args.relation_mixer_input,
            'stage1_encoder_init': self.args.stage1_encoder_init,
            'disable_retrieval': int(getattr(self.args, 'disable_retrieval', 0)),
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

    def _retrieval_disabled(self):
        return bool(int(getattr(self.args, 'disable_retrieval', 0)))

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        encoder_init = getattr(self.args, 'stage1_encoder_init', 'checkpoint')
        if self._retrieval_disabled():
            print('[stage2] retrieval disabled; skipping Stage-1 encoder checkpoint initialization')
        elif encoder_init == 'checkpoint':
            if not self.args.stage1_ckpt_path:
                raise ValueError('--stage1_ckpt_path is required when --stage1_encoder_init checkpoint')
            model.load_stage1_checkpoint(self.args.stage1_ckpt_path, strict=True)
        elif encoder_init == 'random':
            print('[stage2] using random Stage-1 encoder initialization')
        else:
            raise ValueError(f'Unsupported stage1_encoder_init: {encoder_init}')
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        return optim.Adam(params, lr=self.args.learning_rate)

    def _ensure_memory(self):
        if self._retrieval_disabled():
            self.memory_bank = None
            self.memory_y = None
            self.memory_x_last = None
            return
        if self.memory_bank is not None:
            return
        train_data, _ = self._get_data(flag='train')
        self.memory_bank = RelationMemoryBank(
            train_data,
            seq_len=self.args.seq_len,
            pred_len=self.args.pred_len,
            mask_mode=self.args.candidate_mask,
        )
        self.memory_y = torch.from_numpy(self.memory_bank.memory_y).float().to(self.device)
        self.memory_x_last = torch.from_numpy(self.memory_bank.memory_x[:, -1, :]).float().to(self.device)
        print(f'[stage2] memory_x={tuple(self.memory_bank.memory_x.shape)} memory_y={tuple(self.memory_bank.memory_y.shape)}')

    def _build_key_bank(self, force=False):
        if self._retrieval_disabled():
            self.key_bank = None
            return
        if self.key_bank is not None and not force:
            return
        model = self.model.module if hasattr(self.model, 'module') else self.model
        self.key_bank = model.build_memory_key_bank(
            self.memory_bank.memory_x,
            self.device,
            chunk_size=self.args.memory_chunk_size,
        ).to(self.device)
        print(f'[stage2] built relation key memory bank: {tuple(self.key_bank.shape)}')

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
        base_mse = torch.mean((y_base - batch_y) ** 2)
        ret_mse = torch.mean((y_ret - batch_y) ** 2)
        final_mae = torch.mean(torch.abs(y_final - batch_y))
        base_mae = torch.mean(torch.abs(y_base - batch_y))
        ret_mae = torch.mean(torch.abs(y_ret - batch_y))
        ret_gain = base_mse - ret_mse
        base_err_sc = ((y_base - batch_y) ** 2).mean(dim=1)
        ret_err_sc = ((y_ret - batch_y) ** 2).mean(dim=1)
        final_err_sc = ((y_final - batch_y) ** 2).mean(dim=1)
        oracle_gate_mse = torch.minimum(base_err_sc, ret_err_sc).mean()
        ret_advantage = base_err_sc - ret_err_sc
        ret_better = ret_err_sc < base_err_sc
        final_mse_by_channel = ((y_final - batch_y) ** 2).mean(dim=(0, 1))
        base_mse_by_channel = ((y_base - batch_y) ** 2).mean(dim=(0, 1))
        ret_mse_by_channel = ((y_ret - batch_y) ** 2).mean(dim=(0, 1))
        channel_gain = base_mse_by_channel - final_mse_by_channel

        diag = torch.eye(beta.size(-1), device=beta.device, dtype=torch.bool)[None, :, :]
        self_beta = beta.masked_select(diag).mean()
        cross_beta = beta.masked_select(~diag).mean() if beta.size(-1) > 1 else torch.tensor(0.0, device=beta.device)
        beta_sorted = torch.sort(beta, dim=-1, descending=True).values
        beta_max = beta_sorted[..., 0].mean()
        beta_margin = (
            beta_sorted[..., 0] - beta_sorted[..., 1]
            if beta_sorted.size(-1) > 1
            else beta_sorted[..., 0]
        ).mean()

        metrics = {
            'final_mse': final_mse.detach(),
            'final_mae': final_mae.detach(),
            'base_mse': base_mse.detach(),
            'base_mae': base_mae.detach(),
            'ret_mse': ret_mse.detach(),
            'ret_mae': ret_mae.detach(),
            'retrieval_gain': (base_mse - final_mse).detach(),
            'ret_gain': ret_gain.detach(),
            'oracle_gate_mse': oracle_gate_mse.detach(),
            'oracle_gate_gain': (base_mse - oracle_gate_mse).detach(),
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
            'valid_candidate_count_mean': counts.float().mean().item(),
            'valid_candidate_count_min': counts.min().item() if counts.numel() else 0.0,
        }
        channel_names = self._channel_names(beta.size(-1))
        focus_idx, focus_name = self._focus_channel_index(beta.size(-1))
        if focus_idx is not None and focus_idx < y_final.size(-1):
            focus_oracle = torch.minimum(base_err_sc[:, focus_idx], ret_err_sc[:, focus_idx]).mean()
            metrics.update({
                f'final_mse_{focus_name}': final_mse_by_channel[focus_idx].detach(),
                f'base_mse_{focus_name}': base_mse_by_channel[focus_idx].detach(),
                f'ret_mse_{focus_name}': ret_mse_by_channel[focus_idx].detach(),
                f'final_gain_{focus_name}': channel_gain[focus_idx].detach(),
                f'ret_gain_{focus_name}': (base_mse_by_channel[focus_idx] - ret_mse_by_channel[focus_idx]).detach(),
                f'oracle_gate_mse_{focus_name}': focus_oracle.detach(),
                f'oracle_gate_gain_{focus_name}': (base_mse_by_channel[focus_idx] - focus_oracle).detach(),
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
            relation_mse_best, oracle_relation = relation_mse.min(dim=-1)
            beta_choice = beta.argmax(dim=-1)
            beta_expected_mse = (beta * relation_mse).sum(dim=-1)

            diag_cc = torch.eye(beta.size(-1), device=beta.device, dtype=torch.bool)[None, :, :]
            relation_mse_self = relation_mse.masked_select(diag_cc).mean()
            relation_mse_cross = relation_mse.masked_fill(diag_cc, float('inf'))
            relation_mse_cross_best = relation_mse_cross.min(dim=-1).values
            relation_mse_cross_best = relation_mse_cross_best[torch.isfinite(relation_mse_cross_best)].mean()

            metrics.update({
                'relation_mse_mean': relation_mse_mean.detach(),
                'relation_mse_best': relation_mse_best.mean().detach(),
                'relation_mse_self': relation_mse_self.detach(),
                'relation_mse_cross_best': relation_mse_cross_best.detach(),
                'beta_expected_relation_mse': beta_expected_mse.mean().detach(),
                'beta_oracle_top1_match': (beta_choice == oracle_relation).float().mean().detach(),
                'beta_gain_vs_uniform': (relation_mse_mean - beta_expected_mse.mean()).detach(),
                'beta_regret_vs_oracle': (beta_expected_mse - relation_mse_best).mean().detach(),
            })
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
                    ot_beta_on_oracle = ot_beta.gather(1, ot_best_relation[:, None]).squeeze(1)
                    metrics['beta_OT_oracle_top1_match'] = (
                        ot_beta_choice == ot_best_relation
                    ).float().mean().detach()
                    metrics['beta_OT_top1_mean'] = ot_beta_top1.mean().detach()
                    metrics['beta_OT_on_oracle_mean'] = ot_beta_on_oracle.mean().detach()
                    metrics['relation_mse_OT_best'] = ot_mse_best.mean().detach()
                    metrics['relation_mse_OT_beta_top1'] = ot_mse_beta_top1.mean().detach()
                    metrics['relation_mse_OT_beta_regret'] = (
                        ot_mse_beta_top1 - ot_mse_best
                    ).mean().detach()
        for name in ('alpha_entropy', 'beta_entropy', 'top_k_effective'):
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
        if int(self.args.use_aux_base_loss):
            loss = loss + float(self.args.aux_base_weight) * torch.mean((y_base - batch_y) ** 2)
        if int(self.args.use_aux_ret_loss):
            loss = loss + float(self.args.aux_ret_weight) * torch.mean((y_ret - batch_y) ** 2)
        if float(self.args.beta_entropy_reg) != 0.0 and 'beta_entropy' in debug:
            loss = loss - float(self.args.beta_entropy_reg) * debug['beta_entropy']
        return loss

    def _init_csv_accumulators(self):
        return {
            'lambda_bins': [
                {'lo': i / 5.0, 'hi': (i + 1) / 5.0, 'count': 0.0,
                 'base': 0.0, 'ret': 0.0, 'final': 0.0, 'ret_better': 0.0}
                for i in range(5)
            ],
            'ot_relation': None,
        }

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
            channel_names = self._channel_names(beta.size(-1))
            focus_idx, focus_name = self._focus_channel_index(beta.size(-1))
            if focus_idx is None or focus_idx >= beta.size(1):
                return
            target_y = batch_y.permute(0, 2, 1).unsqueeze(2)
            relation_mse = ((relation_outputs - target_y) ** 2).mean(dim=-1)
            ot_beta = beta[:, focus_idx, :]
            ot_mse = relation_mse[:, focus_idx, :]
            if acc['ot_relation'] is None:
                acc['ot_relation'] = {
                    'target_channel': focus_name,
                    'source_names': channel_names,
                    'count': 0.0,
                    'beta_sum': torch.zeros(len(channel_names), dtype=torch.float64),
                    'mse_sum': torch.zeros(len(channel_names), dtype=torch.float64),
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
        oracle_idx = int(mse_order[0].item())
        rows = []
        for src_idx, src_name in enumerate(state['source_names']):
            row = dict(context)
            row.update({
                'target_channel': state['target_channel'],
                'source_channel': src_name,
                'beta_mean': float(beta_mean[src_idx].item()),
                'relation_mse': float(mse_mean[src_idx].item()),
                'rank_by_beta': int(rank_beta[src_idx].item()),
                'rank_by_relation_mse': int(rank_mse[src_idx].item()),
                'is_self': int(src_name == state['target_channel']),
                'is_oracle_best': int(src_idx == oracle_idx),
            })
            rows.append(row)
        return rows

    def _write_stage2_metric_csvs(self, setting, split, epoch, metrics, acc):
        context = self._csv_context(epoch, split, setting)
        base_dir = self._csv_base_dir(setting)
        focus = getattr(self.args, 'focus_channel', 'OT')
        main_keys = [
            'final_mse', 'final_mae', 'base_mse', 'base_mae', 'ret_mse', 'ret_mae',
            'retrieval_gain', 'ret_gain',
            f'final_mse_{focus}', f'base_mse_{focus}', f'ret_mse_{focus}',
            f'final_gain_{focus}', f'ret_gain_{focus}',
            'oracle_gate_mse', 'oracle_gate_gain', 'ret_better_frac',
            f'oracle_gate_mse_{focus}', f'oracle_gate_gain_{focus}', f'ret_better_frac_{focus}',
            'lambda_mean', 'lambda_std', 'lambda_p10', 'lambda_p50', 'lambda_p90',
            f'lambda_{focus}', 'lambda_ret_adv_corr', f'lambda_ret_adv_corr_{focus}',
            'alpha_entropy_norm', 'alpha_top1_mean', 'alpha_margin_mean',
            'beta_entropy_norm', 'beta_effective_relations', 'beta_max_mean', 'beta_margin_mean',
            'beta_self_mean', 'beta_cross_mean', 'beta_self_minus_cross',
            'relation_mse_self', 'relation_mse_cross_best', 'relation_mse_best',
            'beta_expected_relation_mse', 'beta_oracle_top1_match', 'beta_relation_rank_corr',
            'beta_gain_vs_uniform', 'beta_regret_vs_oracle',
            'beta_OT_oracle_top1_match', 'beta_OT_top1_mean', 'beta_OT_on_oracle_mean',
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
                'target_channel', 'source_channel', 'beta_mean', 'relation_mse',
                'rank_by_beta', 'rank_by_relation_mse', 'is_self', 'is_oracle_best',
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
        if bool(int(self.args.freeze_stage1_encoder)):
            model.stage1_encoder.eval()
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
            if valid_query.sum() == 0:
                avg.update({
                    'skipped_batches': 1.0,
                    'valid_candidate_count_mean': counts.float().mean().item(),
                    'valid_candidate_count_min': counts.min().item() if counts.numel() else 0.0,
                })
                continue

            if train:
                optimizer.zero_grad()
                y_final, y_base, y_ret, beta, lam, debug = self.model(
                    batch_x=batch_x,
                    memory_y=self.memory_y,
                    valid_mask=cand_mask,
                    key_bank=self.key_bank,
                    memory_x_last=self.memory_x_last,
                )
                loss = self._loss(y_final, y_base, y_ret, batch_y, debug, valid_query)
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    y_final, y_base, y_ret, beta, lam, debug = self.model(
                        batch_x=batch_x,
                        memory_y=self.memory_y,
                        valid_mask=cand_mask,
                        key_bank=self.key_bank,
                        memory_x_last=self.memory_x_last,
                    )
                    loss = self._loss(y_final, y_base, y_ret, batch_y, debug, valid_query)

            with torch.no_grad():
                metrics = self._metrics(y_final, y_base, y_ret, batch_y, beta, lam, debug, counts, valid_query)
                metrics['loss'] = loss.detach()
                metrics['skipped_batches'] = 0.0
                avg.update(metrics)
                if split is not None and epoch is not None and setting is not None:
                    self._update_csv_accumulators(
                        csv_acc, y_final, y_base, y_ret, batch_y, beta, lam, debug, valid_query
                    )

        averaged = avg.average()
        if split is not None and epoch is not None and setting is not None:
            self._write_stage2_metric_csvs(setting, split, epoch, averaged, csv_acc)
        return averaged

    def vali(self, vali_data, vali_loader, epoch=None, setting=None):
        return self._run_loader(vali_loader, optimizer=None, split='vali', epoch=epoch, setting=setting)

    def train(self, setting):
        self._ensure_memory()
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, 'stage2', self.args.data, f'seq{self.args.seq_len}_pred{self.args.pred_len}', setting)
        os.makedirs(path, exist_ok=True)
        optimizer = self._select_optimizer()
        best_val_loss = float('inf')
        best_path = os.path.join(path, 'checkpoint.pth')
        self.best_checkpoint_path = best_path
        bad_epochs = 0

        refresh_each_epoch = bool(int(self.args.refresh_memory_every_epoch))
        self._build_key_bank(force=True)
        writer = build_summary_writer(self.args, 'stage2', setting)
        tb_keys = [
            'loss',
            'final_mse', 'final_mae',
            'base_mse', 'base_mae',
            'ret_mse', 'ret_mae',
            'retrieval_gain',
            'lambda_mean',
            'beta_self_mean', 'beta_cross_mean',
            'beta_self_minus_cross', 'beta_max_mean', 'beta_margin_mean',
            'alpha_entropy', 'alpha_entropy_norm', 'alpha_top1_mean', 'alpha_margin_mean',
            'beta_entropy', 'beta_entropy_norm',
            'beta_effective_relations',
            'relation_mse_mean', 'relation_mse_best',
            'relation_mse_self', 'relation_mse_cross_best',
            'beta_expected_relation_mse',
            'beta_oracle_top1_match',
            'beta_relation_rank_corr',
            'beta_gain_vs_uniform', 'beta_regret_vs_oracle',
            'top_k_effective',
            'beta_OT_oracle_top1_match',
            'beta_OT_top1_mean',
            'beta_OT_on_oracle_mean',
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
            f'oracle_gate_mse_{focus}',
            f'oracle_gate_gain_{focus}',
            f'ret_better_frac_{focus}',
            f'lambda_ret_adv_corr_{focus}',
        ])

        try:
            for epoch in range(self.args.train_epochs):
                epoch_time = time.time()
                if refresh_each_epoch and (epoch > 0 or not bool(int(self.args.freeze_stage1_encoder))):
                    self._build_key_bank(force=True)

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
        path = os.path.join(self.args.checkpoints, 'stage2', self.args.data, f'seq{self.args.seq_len}_pred{self.args.pred_len}', setting)
        ckpt_path = self.best_checkpoint_path or os.path.join(path, 'checkpoint.pth')
        if os.path.exists(ckpt_path):
            print(f'loading Stage-2 checkpoint from {ckpt_path}')
            ckpt = torch.load(ckpt_path, map_location=self.device)
            state = ckpt.get('model_state_dict', ckpt)
            self.model.load_state_dict(state)
        else:
            print(f'[stage2] checkpoint not found, testing current model: {ckpt_path}')
        self._ensure_memory()
        self._build_key_bank(force=True)
        test_data, test_loader = self._get_data(flag='test')
        metrics = self._run_loader(test_loader, optimizer=None, split='test', epoch=0, setting=setting)
        print(format_metrics('Stage2 Test', metrics))
        return metrics
