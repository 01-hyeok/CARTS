import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.relation_graph_threshold = int(getattr(configs, 'relation_graph_threshold', 21))
        self.relation_top_n = int(getattr(configs, 'relation_top_n', 3))
        self.target_mode = configs.target_mode
        self.target_channel = configs.target_channel
        self.relation_input_space = getattr(configs, 'relation_input_space', 'absolute')
        self.relation_value_space = getattr(configs, 'relation_value_space', 'absolute')
        self.memory_chunk_size = int(configs.memory_chunk_size)
        self.freeze_stage1_encoder = bool(int(configs.freeze_stage1_encoder))
        self.disable_retrieval = bool(int(getattr(configs, 'disable_retrieval', 0)))
        self.fusion_mode = configs.fusion_mode

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
        if self.fusion_mode == 'raft_concat':
            self.gate = None
            self.raft_concat_head = nn.Linear(2 * configs.pred_len, configs.pred_len)
        else:
            self.gate = RetrievalGate(
                pred_len=configs.pred_len,
                hidden_dim=configs.gate_hidden,
                gate_mode=configs.gate_mode,
                fusion_mode=configs.fusion_mode,
                fixed_lambda=configs.fixed_lambda,
            )
            self.raft_concat_head = None
        self._shape_logged = False
        self.relation_sources = None
        self.relation_correlations = None
        self.checkpoint_relation_graph = None

    def set_relation_graph(self, graph):
        if graph is None:
            if self.checkpoint_relation_graph is not None:
                raise RuntimeError(
                    'Stage-1 checkpoint was trained with a sparse relation graph, '
                    'but Stage-2 did not load one'
                )
            self.relation_sources = None
            self.relation_correlations = None
            return
        sources = [[int(source) for source in row] for row in graph['sources']]
        if len(sources) != self.channels:
            raise ValueError('relation graph target count does not match model channels')
        correlations = [
            [float(corr) for corr in row] for row in graph['correlations']
        ]
        if self.checkpoint_relation_graph is not None:
            checkpoint_sources = [
                [int(source) for source in row]
                for row in self.checkpoint_relation_graph['sources']
            ]
            checkpoint_correlations = [
                [float(corr) for corr in row]
                for row in self.checkpoint_relation_graph['correlations']
            ]
            if sources != checkpoint_sources or correlations != checkpoint_correlations:
                raise RuntimeError(
                    'Stage-2 relation graph differs from the graph saved in the Stage-1 checkpoint'
                )
        self.relation_sources = sources
        self.relation_correlations = correlations

    def uses_sparse_relation_graph(self):
        return self.relation_sources is not None

    def num_source_slots(self):
        if self.disable_retrieval and self.relation_sources is None:
            return self.channels
        return len(self.source_channels(self.target_channels()[0]))

    def source_index_tensor(self, device):
        return torch.tensor(
            [self.source_channels(target) for target in range(self.channels)],
            dtype=torch.long,
            device=device,
        )

    def source_correlation_tensor(self, device):
        if self.relation_correlations is None:
            return None
        return torch.tensor(
            self.relation_correlations, dtype=torch.float32, device=device
        )

    def source_channels(self, target_channel):
        if self.relation_sources is not None:
            return self.relation_sources[int(target_channel)]
        if self.source_mode == 'topk_corr' or (
            self.source_mode == 'auto' and self.channels >= self.relation_graph_threshold
        ):
            raise RuntimeError('sparse source mode requires a loaded relation graph')
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
        self.checkpoint_relation_graph = ckpt.get('relation_graph')
        expected = {
            'relation_encoder_type': self.stage1_encoder.encoder_type,
            'relation_pooling': self.stage1_encoder.pooling,
            'relation_self_fill': self.stage1_encoder.self_fill,
            'relation_input_space': self.relation_input_space,
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'enc_in': self.channels,
            'source_mode': self.source_mode,
            'relation_graph_threshold': self.relation_graph_threshold,
            'relation_top_n': self.relation_top_n,
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
            for r in self.source_channels(c):
                encoded = []
                for start in range(0, memory_x.size(0), chunk_size):
                    cur = memory_x[start:start + chunk_size]
                    rel = self._relation_tensor(cur, c, r).to(device)
                    encoded_chunk = self.stage1_encoder(rel).cpu()
                    if self.uses_sparse_relation_graph():
                        encoded_chunk = encoded_chunk.half()
                    encoded.append(encoded_chunk)
                source_banks.append(torch.cat(encoded, dim=0))
            banks.append(torch.stack(source_banks, dim=0))

        if was_training:
            self.train()
        return torch.stack(banks, dim=0)

    def _memory_value(self, batch_x, memory_y, memory_x_last, target_channel):
        memory_value_c = memory_y[:, :, target_channel].to(batch_x.device)
        query_offset = batch_x[:, -1, target_channel].detach()
        if self.relation_value_space == 'delta_last':
            if memory_x_last is None:
                raise ValueError('relation_value_space=delta_last requires memory_x_last')
            memory_value_c = memory_value_c - memory_x_last[:, target_channel].to(batch_x.device).unsqueeze(-1)
        return memory_value_c, query_offset

    def _restore_retrieved_value(self, retrieved, query_offset):
        if self.relation_value_space == 'delta_last':
            while query_offset.dim() < retrieved.dim():
                query_offset = query_offset.unsqueeze(-1)
            return retrieved + query_offset
        return retrieved

    def _oracle_topk_candidates(
        self,
        memory_value_c,
        valid_mask,
        oracle_target_c,
        query_offset,
    ):
        """Select ground-truth Top-K memory futures for one target channel."""
        target_value_c = oracle_target_c
        if self.relation_value_space == 'delta_last':
            target_value_c = target_value_c - query_offset.unsqueeze(-1)

        horizon = float(memory_value_c.size(-1))
        target_sq = target_value_c.pow(2).mean(dim=-1, keepdim=True)
        memory_sq = memory_value_c.pow(2).mean(dim=-1).unsqueeze(0)
        future_mse = (
            target_sq
            + memory_sq
            - 2.0 * torch.matmul(target_value_c, memory_value_c.transpose(0, 1)) / horizon
        ).clamp_min(0.0)
        future_mse = future_mse.masked_fill(~valid_mask, float('inf'))

        k = min(self.top_k, memory_value_c.size(0))
        oracle_mse, oracle_idx = torch.topk(future_mse, k=k, dim=-1, largest=False)
        oracle_valid = torch.isfinite(oracle_mse)
        oracle_values = memory_value_c[oracle_idx]
        return oracle_idx, oracle_valid, oracle_values, oracle_mse

    def _weight_oracle_candidates(self, scores, oracle_idx, oracle_valid, oracle_values):
        """Apply the normal encoder-score weighting to an Oracle-selected Top-K set."""
        masked_fill = torch.finfo(scores.dtype).min / 4
        oracle_scores = scores.gather(1, oracle_idx)
        scaled_scores = (oracle_scores / self.tau_topk).masked_fill(
            ~oracle_valid, masked_fill
        )
        alpha = F.softmax(scaled_scores, dim=-1) * oracle_valid.float()
        alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return (alpha.unsqueeze(-1) * oracle_values).sum(dim=1)

    @torch.no_grad()
    def build_retrieval_cache(
        self,
        batch_x,
        memory_y,
        valid_mask,
        key_bank,
        memory_x_last=None,
        oracle_target_y=None,
    ):
        bsz = batch_x.size(0)
        source_slots = self.num_source_slots()
        relation_outputs_all = torch.zeros(
            bsz,
            self.channels,
            source_slots,
            self.pred_len,
            device=batch_x.device,
            dtype=batch_x.dtype,
        )
        relation_query_embs_all = torch.zeros(
            bsz,
            self.channels,
            source_slots,
            key_bank.size(-1),
            device=batch_x.device,
            dtype=batch_x.dtype,
        )
        candidate_oracle_relations_all = None
        candidate_oracle_top_k_effective_sc = None
        candidate_oracle_indices_sc = None
        candidate_oracle_mse_topk_sc = None
        candidate_oracle_valid_topk_sc = None
        if oracle_target_y is not None:
            candidate_oracle_relations_all = torch.zeros_like(relation_outputs_all)
            candidate_oracle_top_k_effective_sc = batch_x.new_zeros(bsz, self.channels)
            oracle_k = min(self.top_k, memory_y.size(0))
            candidate_oracle_indices_sc = torch.full(
                (bsz, self.channels, oracle_k),
                -1,
                dtype=torch.long,
                device=batch_x.device,
            )
            candidate_oracle_mse_topk_sc = batch_x.new_full(
                (bsz, self.channels, oracle_k),
                float('inf'),
            )
            candidate_oracle_valid_topk_sc = torch.zeros(
                bsz,
                self.channels,
                oracle_k,
                dtype=torch.bool,
                device=batch_x.device,
            )
        debug_rows = []

        was_training = self.training
        self.stage1_encoder.eval()
        for c in self.target_channels():
            memory_value_c, query_offset_c = self._memory_value(batch_x, memory_y, memory_x_last, c)
            oracle_idx = None
            oracle_valid = None
            oracle_values = None
            if candidate_oracle_relations_all is not None:
                oracle_idx, oracle_valid, oracle_values, oracle_mse = self._oracle_topk_candidates(
                    memory_value_c=memory_value_c,
                    valid_mask=valid_mask,
                    oracle_target_c=oracle_target_y[:, :, c],
                    query_offset=query_offset_c,
                )
                candidate_oracle_top_k_effective_sc[:, c] = oracle_valid.float().sum(dim=-1)
                candidate_oracle_indices_sc[:, c, :oracle_idx.size(-1)] = oracle_idx
                candidate_oracle_mse_topk_sc[:, c, :oracle_mse.size(-1)] = oracle_mse
                candidate_oracle_valid_topk_sc[:, c, :oracle_valid.size(-1)] = oracle_valid
            relation_debug_rows = []
            for source_slot, r in enumerate(self.source_channels(c)):
                q_rel = self._relation_tensor(batch_x, c, r)
                z_q = self.stage1_encoder(q_rel)
                z_mem = key_bank[c, source_slot].to(
                    device=batch_x.device, dtype=z_q.dtype
                )
                r_cr, alpha, top_idx, top_scores, ret_debug = retrieve_relation_future(
                    z_q=z_q,
                    z_mem=z_mem,
                    memory_value_c=memory_value_c,
                    valid_mask=valid_mask,
                    top_k=self.top_k,
                    tau_topk=self.tau_topk,
                )
                relation_outputs_all[:, c, source_slot] = r_cr
                relation_query_embs_all[:, c, source_slot] = z_q
                if candidate_oracle_relations_all is not None:
                    oracle_retrieved = self._weight_oracle_candidates(
                        scores=ret_debug['scores'],
                        oracle_idx=oracle_idx,
                        oracle_valid=oracle_valid,
                        oracle_values=oracle_values,
                    )
                    candidate_oracle_relations_all[:, c, source_slot] = (
                        self._restore_retrieved_value(oracle_retrieved, query_offset_c)
                    )
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
        if candidate_oracle_relations_all is not None:
            candidate_oracle_mse_sc = batch_x.new_zeros(bsz, self.channels)
            candidate_oracle_mae_sc = batch_x.new_zeros(bsz, self.channels)
            full_oracle_mse_sc = batch_x.new_zeros(bsz, self.channels)
            full_oracle_mae_sc = batch_x.new_zeros(bsz, self.channels)
            for c in self.target_channels():
                source_count = len(self.source_channels(c))
                relation_outputs_c = relation_outputs_all[:, c, :source_count]
                relation_query_embs_c = relation_query_embs_all[:, c, :source_count]
                candidate_relations_c = candidate_oracle_relations_all[:, c, :source_count]
                _, beta_c, _ = self.relation_mixer(
                    relation_outputs_c, relation_query_embs_c
                )

                candidate_prediction_c = (
                    beta_c.unsqueeze(-1) * candidate_relations_c
                ).sum(dim=1)
                target_c = oracle_target_y[:, :, c]
                candidate_oracle_mse_sc[:, c] = (
                    (candidate_prediction_c - target_c) ** 2
                ).mean(dim=-1)
                candidate_oracle_mae_sc[:, c] = torch.abs(
                    candidate_prediction_c - target_c
                ).mean(dim=-1)

                candidate_relation_mse = (
                    (candidate_relations_c - target_c.unsqueeze(1)) ** 2
                ).mean(dim=-1)
                candidate_relation_mae = torch.abs(
                    candidate_relations_c - target_c.unsqueeze(1)
                ).mean(dim=-1)
                full_mse_c, full_relation_c = candidate_relation_mse.min(dim=-1)
                full_mae_c = candidate_relation_mae.gather(
                    -1, full_relation_c.unsqueeze(-1)
                ).squeeze(-1)
                full_oracle_mse_sc[:, c] = full_mse_c
                full_oracle_mae_sc[:, c] = full_mae_c

            cache.update({
                'candidate_oracle_mse_sc': candidate_oracle_mse_sc.detach(),
                'candidate_oracle_mae_sc': candidate_oracle_mae_sc.detach(),
                'full_oracle_mse_sc': full_oracle_mse_sc.detach(),
                'full_oracle_mae_sc': full_oracle_mae_sc.detach(),
                'candidate_oracle_top_k_effective_sc': (
                    candidate_oracle_top_k_effective_sc.detach()
                ),
                'candidate_oracle_indices_sc': candidate_oracle_indices_sc.detach(),
                'candidate_oracle_mse_topk_sc': candidate_oracle_mse_topk_sc.detach(),
                'candidate_oracle_valid_topk_sc': candidate_oracle_valid_topk_sc.detach(),
            })
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
        source_slots = self.num_source_slots()
        lambda_all = torch.zeros(bsz, self.channels, device=batch_x.device, dtype=batch_x.dtype)
        beta_all = torch.zeros(bsz, self.channels, source_slots, device=batch_x.device, dtype=batch_x.dtype)
        relation_outputs_all = torch.zeros(
            bsz,
            self.channels,
            source_slots,
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
                'source_indices': self.source_index_tensor(batch_x.device),
            }
            source_correlations = self.source_correlation_tensor(batch_x.device)
            if source_correlations is not None:
                debug['source_correlations'] = source_correlations
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
                for source_slot, r in enumerate(self.source_channels(c)):
                    q_rel = self._relation_tensor(batch_x, c, r)
                    if self.freeze_stage1_encoder:
                        with torch.no_grad():
                            z_q = self.stage1_encoder(q_rel)
                    else:
                        z_q = self.stage1_encoder(q_rel)
                    z_mem = key_bank[c, source_slot].to(
                        device=batch_x.device, dtype=z_q.dtype
                    )
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
                source_count = len(self.source_channels(c))
                relation_outputs = cached_relation_outputs[:, c, :source_count]
                relation_query_embs = cached_relation_query_embs[:, c, :source_count]

            y_ret_c, beta_c, relation_scores = self.relation_mixer(relation_outputs, relation_query_embs)
            y_base_c = y_base_all[:, :, c]
            if self.fusion_mode == 'raft_concat':
                fusion_input = torch.cat([y_base_c, y_ret_c], dim=-1)
                y_final_c = self.raft_concat_head(fusion_input)
                lambda_c = y_base_c.new_ones(y_base_c.size(0), 1)
            else:
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
            'source_indices': self.source_index_tensor(batch_x.device),
        }
        source_correlations = self.source_correlation_tensor(batch_x.device)
        if source_correlations is not None:
            debug['source_correlations'] = source_correlations
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
