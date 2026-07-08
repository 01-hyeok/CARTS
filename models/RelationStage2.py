import torch
import torch.nn as nn

from layers.relation_mixer import RelationMixer
from layers.retrieval_gate import RetrievalGate
from models.RelationStage1 import RelationEncoder
from utils.retrieval_ops import retrieve_relation_future


class BaseForecastHead(nn.Module):
    def __init__(self, seq_len, pred_len, channels, mode='per_channel_linear'):
        super().__init__()
        if mode not in ('per_channel_linear', 'shared_target_linear'):
            raise ValueError(f'Unsupported base_head_mode: {mode}')
        self.mode = mode
        self.channels = channels
        if mode == 'per_channel_linear':
            self.heads = nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(channels)])
        else:
            self.shared = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        offset = x[:, -1:, :].detach()
        x = x - offset
        outs = []
        for c in range(self.channels):
            xc = x[:, :, c]
            if self.mode == 'per_channel_linear':
                outs.append(self.heads[c](xc))
            else:
                outs.append(self.shared(xc))
        return torch.stack(outs, dim=-1)


class Model(nn.Module):
    """Stage-2 retrieval-guided forecasting model."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.top_k = int(configs.top_k)
        self.tau_topk = float(configs.tau_topk)
        self.source_mode = configs.source_mode
        self.target_mode = configs.target_mode
        self.target_channel = configs.target_channel
        self.relation_input_space = getattr(configs, 'relation_input_space', 'absolute')
        self.relation_value_space = getattr(configs, 'relation_value_space', 'absolute')
        self.memory_chunk_size = int(configs.memory_chunk_size)
        self.freeze_stage1_encoder = bool(int(configs.freeze_stage1_encoder))
        self.disable_retrieval = bool(int(getattr(configs, 'disable_retrieval', 0)))

        self.stage1_encoder = RelationEncoder(configs)
        if self.freeze_stage1_encoder:
            for param in self.stage1_encoder.parameters():
                param.requires_grad = False

        self.base_head = BaseForecastHead(
            seq_len=configs.seq_len,
            pred_len=configs.pred_len,
            channels=configs.enc_in,
            mode=configs.base_head_mode,
        )
        self.relation_mixer = RelationMixer(
            pred_len=configs.pred_len,
            emb_dim=configs.d_model,
            hidden_dim=configs.relation_mixer_hidden,
            input_mode=configs.relation_mixer_input,
        )
        self.gate = RetrievalGate(
            pred_len=configs.pred_len,
            hidden_dim=configs.gate_hidden,
            gate_mode=configs.gate_mode,
            fusion_mode=configs.fusion_mode,
            fixed_lambda=configs.fixed_lambda,
        )
        self._shape_logged = False

    def source_channels(self, target_channel):
        if self.source_mode != 'all':
            raise NotImplementedError('source_mode=topk_corr is reserved for later source selection work')
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

    def load_stage1_checkpoint(self, ckpt_path, strict=True):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        ckpt_args = ckpt.get('args', {})
        expected = {
            'relation_encoder_type': self.stage1_encoder.encoder_type,
            'relation_pooling': self.stage1_encoder.pooling,
            'relation_self_fill': self.stage1_encoder.self_fill,
            'relation_input_space': self.relation_input_space,
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'enc_in': self.channels,
        }
        mismatches = []
        for key, expected_value in expected.items():
            actual_value = ckpt_args.get(key, 'absolute') if key == 'relation_input_space' else ckpt_args.get(key)
            if actual_value is not None and actual_value != expected_value:
                mismatches.append((key, actual_value, expected_value))
        if mismatches:
            details = ', '.join(
                f'{key}: checkpoint={actual} current={expected}'
                for key, actual, expected in mismatches
            )
            raise RuntimeError(f'Stage-1 checkpoint config mismatch for {ckpt_path}: {details}')

        state = ckpt.get('model_state_dict', ckpt)
        encoder_state = {}
        for key, value in state.items():
            clean_key = key[7:] if key.startswith('module.') else key
            if clean_key.startswith('encoder.'):
                encoder_state[clean_key[len('encoder.'):]] = value
        if not encoder_state:
            raise RuntimeError(f'No Stage-1 encoder weights found in checkpoint: {ckpt_path}')
        missing, unexpected = self.stage1_encoder.load_state_dict(encoder_state, strict=strict)
        print(f'[stage2] loaded Stage-1 encoder from {ckpt_path}')
        if missing or unexpected:
            msg = (
                f'Stage-1 encoder checkpoint mismatch for {ckpt_path}\n'
                f'missing keys: {missing}\n'
                f'unexpected keys: {unexpected}'
            )
            if strict:
                raise RuntimeError(msg)
            print(f'[stage2] {msg}')

    @torch.no_grad()
    def build_memory_key_bank(self, memory_x, device, chunk_size=None):
        was_training = self.training
        self.stage1_encoder.eval()
        chunk_size = int(chunk_size or self.memory_chunk_size)
        memory_x = torch.as_tensor(memory_x, dtype=torch.float32)
        banks = []

        for c in range(self.channels):
            source_banks = []
            for r in range(self.channels):
                encoded = []
                for start in range(0, memory_x.size(0), chunk_size):
                    cur = memory_x[start:start + chunk_size].to(device)
                    rel = self._relation_tensor(cur, c, r)
                    encoded.append(self.stage1_encoder(rel).cpu())
                source_banks.append(torch.cat(encoded, dim=0))
            banks.append(torch.stack(source_banks, dim=0))

        if was_training:
            self.train()
        return torch.stack(banks, dim=0)

    def _memory_value(self, batch_x, memory_y, memory_x_last, target_channel):
        memory_value_c = memory_y[:, :, target_channel]
        query_offset = batch_x[:, -1, target_channel].detach()
        if self.relation_value_space == 'delta_last':
            if memory_x_last is None:
                raise ValueError('relation_value_space=delta_last requires memory_x_last')
            memory_value_c = memory_value_c - memory_x_last[:, target_channel].to(memory_y.device).unsqueeze(-1)
        return memory_value_c, query_offset

    def _restore_retrieved_value(self, retrieved, query_offset):
        if self.relation_value_space == 'delta_last':
            while query_offset.dim() < retrieved.dim():
                query_offset = query_offset.unsqueeze(-1)
            return retrieved + query_offset
        return retrieved

    @torch.no_grad()
    def build_retrieval_cache(self, batch_x, memory_y, valid_mask, key_bank, memory_x_last=None):
        bsz = batch_x.size(0)
        relation_outputs_all = torch.zeros(
            bsz,
            self.channels,
            self.channels,
            self.pred_len,
            device=batch_x.device,
            dtype=batch_x.dtype,
        )
        relation_query_embs_all = torch.zeros(
            bsz,
            self.channels,
            self.channels,
            key_bank.size(-1),
            device=batch_x.device,
            dtype=batch_x.dtype,
        )
        debug_rows = []

        was_training = self.training
        self.stage1_encoder.eval()
        for c in self.target_channels():
            memory_value_c, query_offset_c = self._memory_value(batch_x, memory_y, memory_x_last, c)
            relation_debug_rows = []
            for r in self.source_channels(c):
                q_rel = self._relation_tensor(batch_x, c, r)
                z_q = self.stage1_encoder(q_rel)
                z_mem = key_bank[c, r].to(batch_x.device)
                r_cr, alpha, top_idx, top_scores, ret_debug = retrieve_relation_future(
                    z_q=z_q,
                    z_mem=z_mem,
                    memory_value_c=memory_value_c,
                    valid_mask=valid_mask,
                    top_k=self.top_k,
                    tau_topk=self.tau_topk,
                )
                relation_outputs_all[:, c, r] = r_cr
                relation_query_embs_all[:, c, r] = z_q
                alpha_entropy = -(alpha * torch.log(alpha + 1e-8)).sum(dim=-1)
                alpha_sorted = torch.sort(alpha, dim=-1, descending=True).values
                alpha_top1 = alpha_sorted[:, 0]
                alpha_margin = (
                    alpha_sorted[:, 0] - alpha_sorted[:, 1]
                    if alpha_sorted.size(-1) > 1
                    else alpha_sorted[:, 0]
                )
                relation_debug_rows.append({
                    'alpha_entropy': alpha_entropy,
                    'alpha_top1': alpha_top1,
                    'alpha_margin': alpha_margin,
                    'top_k_effective': ret_debug['top_k_effective'],
                })
            debug_rows.append({
                'alpha_entropy': torch.stack([row['alpha_entropy'] for row in relation_debug_rows], dim=1).mean(dim=1),
                'alpha_top1': torch.stack([row['alpha_top1'] for row in relation_debug_rows], dim=1).mean(dim=1),
                'alpha_margin': torch.stack([row['alpha_margin'] for row in relation_debug_rows], dim=1).mean(dim=1),
                'top_k_effective': torch.stack([row['top_k_effective'] for row in relation_debug_rows], dim=1).mean(dim=1),
            })
        if was_training:
            self.train()
        cache = {
            'relation_outputs': relation_outputs_all.detach(),
            'relation_query_embs': relation_query_embs_all.detach(),
        }
        if debug_rows:
            cache['alpha_entropy'] = torch.stack([row['alpha_entropy'] for row in debug_rows], dim=1).mean(dim=1).detach()
            cache['alpha_top1'] = torch.stack([row['alpha_top1'] for row in debug_rows], dim=1).mean(dim=1).detach()
            cache['alpha_margin'] = torch.stack([row['alpha_margin'] for row in debug_rows], dim=1).mean(dim=1).detach()
            cache['top_k_effective'] = torch.stack([row['top_k_effective'] for row in debug_rows], dim=1).mean(dim=1).detach()
        return cache

    def forward(self, batch_x, memory_y, valid_mask, key_bank, memory_x_last=None, retrieval_cache=None):
        bsz = batch_x.size(0)
        output_offset = batch_x[:, -1:, :].detach()
        y_base_all = self.base_head(batch_x)
        y_ret_all = torch.zeros_like(y_base_all)
        y_final_all = y_base_all.clone()
        lambda_all = torch.zeros(bsz, self.channels, device=batch_x.device, dtype=batch_x.dtype)
        beta_all = torch.zeros(bsz, self.channels, self.channels, device=batch_x.device, dtype=batch_x.dtype)
        relation_outputs_all = torch.zeros(
            bsz,
            self.channels,
            self.channels,
            self.pred_len,
            device=batch_x.device,
            dtype=batch_x.dtype,
        )

        if self.disable_retrieval:
            y_base_out = y_base_all + output_offset
            y_ret_out = y_ret_all + output_offset
            debug = {
                'beta': beta_all,
                'lambda': lambda_all,
            }
            return y_base_out, y_base_out, y_ret_out, beta_all, lambda_all, debug

        debug_rows = []
        first_debug = None
        cached_relation_outputs = None
        cached_relation_query_embs = None
        if retrieval_cache is not None:
            cached_relation_outputs = retrieval_cache['relation_outputs'].to(batch_x.device)
            cached_relation_query_embs = retrieval_cache['relation_query_embs'].to(batch_x.device)

        for c in self.target_channels():
            relation_outputs = []
            relation_query_embs = []
            relation_debug_rows = []

            if cached_relation_outputs is None:
                memory_value_c, query_offset_c = self._memory_value(batch_x, memory_y, memory_x_last, c)
                for r in self.source_channels(c):
                    q_rel = self._relation_tensor(batch_x, c, r)
                    if self.freeze_stage1_encoder:
                        with torch.no_grad():
                            z_q = self.stage1_encoder(q_rel)
                    else:
                        z_q = self.stage1_encoder(q_rel)
                    z_mem = key_bank[c, r].to(batch_x.device)
                    r_cr, alpha, top_idx, top_scores, ret_debug = retrieve_relation_future(
                        z_q=z_q,
                        z_mem=z_mem,
                        memory_value_c=memory_value_c,
                        valid_mask=valid_mask,
                        top_k=self.top_k,
                        tau_topk=self.tau_topk,
                    )
                    relation_outputs.append(r_cr)
                    relation_query_embs.append(z_q)
                    alpha_entropy = -(alpha * torch.log(alpha + 1e-8)).sum(dim=-1)
                    alpha_sorted = torch.sort(alpha, dim=-1, descending=True).values
                    alpha_top1 = alpha_sorted[:, 0]
                    alpha_margin = (
                        alpha_sorted[:, 0] - alpha_sorted[:, 1]
                        if alpha_sorted.size(-1) > 1
                        else alpha_sorted[:, 0]
                    )
                    relation_debug_rows.append({
                        'alpha_entropy': alpha_entropy,
                        'alpha_top1': alpha_top1,
                        'alpha_margin': alpha_margin,
                        'top_k_effective': ret_debug['top_k_effective'],
                    })
                    if first_debug is None:
                        first_debug = {
                            'z_q': z_q,
                            'z_mem': z_mem,
                            'top_idx': top_idx,
                            'v_top': ret_debug['v_top'],
                            'alpha': alpha,
                            'r_cr': r_cr,
                        }
                relation_outputs = torch.stack(relation_outputs, dim=1)
                relation_query_embs = torch.stack(relation_query_embs, dim=1)
            else:
                source_idx = self.source_channels(c)
                relation_outputs = cached_relation_outputs[:, c, source_idx]
                relation_query_embs = cached_relation_query_embs[:, c, source_idx]

            y_ret_c, beta_c, relation_scores = self.relation_mixer(relation_outputs, relation_query_embs)
            y_base_c = y_base_all[:, :, c]
            y_final_c, lambda_c = self.gate(y_base_c, y_ret_c)

            y_ret_all[:, :, c] = y_ret_c
            y_final_all[:, :, c] = y_final_c
            beta_all[:, c, :] = beta_c
            lambda_all[:, c] = lambda_c.mean(dim=-1)
            relation_outputs_all[:, c] = self._restore_retrieved_value(
                relation_outputs.detach(),
                output_offset[:, 0, c],
            )

            beta_entropy = -(beta_c * torch.log(beta_c + 1e-8)).sum(dim=-1)
            row = {'beta_entropy': beta_entropy}
            if relation_debug_rows:
                row.update({
                    'alpha_entropy': torch.stack([item['alpha_entropy'] for item in relation_debug_rows], dim=1).mean(dim=1),
                    'alpha_top1': torch.stack([item['alpha_top1'] for item in relation_debug_rows], dim=1).mean(dim=1),
                    'alpha_margin': torch.stack([item['alpha_margin'] for item in relation_debug_rows], dim=1).mean(dim=1),
                    'top_k_effective': torch.stack([item['top_k_effective'] for item in relation_debug_rows], dim=1).mean(dim=1),
                })
            debug_rows.append(row)

            if not self._shape_logged and first_debug is not None:
                print(f'[stage2] batch_x={tuple(batch_x.shape)} memory_y={tuple(memory_y.shape)} valid_mask={tuple(valid_mask.shape)}')
                print(f'[stage2] key_bank={tuple(key_bank.shape)} z_q={tuple(first_debug["z_q"].shape)} z_mem={tuple(first_debug["z_mem"].shape)}')
                print(f'[stage2] top_idx={tuple(first_debug["top_idx"].shape)} V_top={tuple(first_debug["v_top"].shape)} alpha={tuple(first_debug["alpha"].shape)} R_cr={tuple(first_debug["r_cr"].shape)}')
                print(f'[stage2] relation_outputs={tuple(relation_outputs.shape)} beta={tuple(beta_c.shape)} y_ret_c={tuple(y_ret_c.shape)} y_base_c={tuple(y_base_c.shape)} lambda_c={tuple(lambda_c.shape)} y_final={tuple(y_final_all.shape)}')
                self._shape_logged = True

        debug = {
            'beta': beta_all,
            'lambda': lambda_all,
            'relation_outputs': relation_outputs_all,
        }
        if debug_rows:
            debug['beta_entropy'] = torch.stack([row['beta_entropy'] for row in debug_rows], dim=1).mean()
            for key in ('alpha_entropy', 'alpha_top1', 'alpha_margin', 'top_k_effective'):
                if key in debug_rows[0]:
                    debug[key] = torch.stack([row[key] for row in debug_rows], dim=1).mean()
        if retrieval_cache is not None:
            for key in ('alpha_entropy', 'alpha_top1', 'alpha_margin', 'top_k_effective'):
                if key in retrieval_cache:
                    debug[key] = retrieval_cache[key].to(batch_x.device).mean()
        y_final_out = y_final_all + output_offset
        y_base_out = y_base_all + output_offset
        y_ret_out = y_ret_all + output_offset
        return y_final_out, y_base_out, y_ret_out, beta_all, lambda_all, debug
