import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.relation_mixer import RelationMixer
from layers.pairwise_scorer import PairwiseScorer
from layers.retrieval_gate import RetrievalGate
from layers.retrieval_metric import METRICS as RETRIEVAL_METRICS, RetrievalMetric
from models.ChronosRelationEncoder import ChronosRelationEncoder
from models.RelationStage1 import (
    RelationEncoder,
    build_relation_encoder_input,
    relation_feature_count,
    relation_sequence_length,
    transform_relation_features,
    transform_relation_history,
)
from utils.retrieval_ops import retrieve_relation_future, reweight_selected_candidates
from utils.rank_losses import (
    MODES as RANK_MODES,
    embedding_geometry,
    mine_ranking_candidates,
    ranking_loss,
    score_geometry,
    weight_geometry,
)


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
        self.relation_input_space = getattr(configs, 'relation_input_space', 'absolute')
        self.relation_seq_len = relation_sequence_length(
            self.seq_len, self.relation_input_space
        )
        self.relation_n_features = relation_feature_count(self.relation_input_space)
        self.retrieval_backbone = getattr(configs, 'stage2_retrieval_backbone', 'stage1')
        if self.retrieval_backbone not in ('stage1', 'identity', 'pearson', 'chronos'):
            raise ValueError(f'Unsupported stage2_retrieval_backbone: {self.retrieval_backbone}')
        chronos_embedding_dim = int(getattr(configs, 'chronos_embedding_dim', 768))
        self.chronos_finetune = bool(int(getattr(configs, 'chronos_finetune', 0)))
        if self.chronos_finetune and self.retrieval_backbone != 'chronos':
            raise ValueError('chronos_finetune requires stage2_retrieval_backbone=chronos')
        self.chronos_projection_dim = int(getattr(configs, 'chronos_projection_dim', 0))
        self.chronos_projection_mode = getattr(configs, 'chronos_projection_mode', 'cross_only')
        self.chronos_projection_trainable = bool(
            int(getattr(configs, 'chronos_projection_trainable', 0))
        )
        if self.chronos_projection_mode not in ('cross_only', 'uniform'):
            raise ValueError(
                f'Unsupported chronos_projection_mode: {self.chronos_projection_mode}'
            )
        if self.chronos_projection_dim and self.retrieval_backbone != 'chronos':
            raise ValueError('chronos_projection_dim requires stage2_retrieval_backbone=chronos')
        if self.chronos_projection_trainable and not self.chronos_projection_dim:
            raise ValueError('chronos_projection_trainable requires chronos_projection_dim > 0')
        if (
            self.chronos_projection_dim
            and self.chronos_projection_mode == 'cross_only'
            and self.chronos_projection_dim != chronos_embedding_dim
        ):
            # cross_only mirrors shared_cross_projection: 2D -> D so that the
            # unprojected self branch already has the right width.
            raise ValueError(
                'chronos_projection_mode=cross_only requires chronos_projection_dim == '
                f'chronos_embedding_dim ({chronos_embedding_dim}), got {self.chronos_projection_dim}'
            )
        self.relation_emb_dim = (
            (self.chronos_projection_dim or 2 * chronos_embedding_dim)
            if self.retrieval_backbone == 'chronos'
            else (
                2 * self.relation_n_features * self.relation_seq_len
                if self.retrieval_backbone in ('identity', 'pearson')
                else configs.d_model
            )
        )
        self.chronos_context_length = int(
            getattr(configs, 'chronos_context_length', 512)
        )
        self.top_k = int(configs.top_k)
        self.tau_topk = float(configs.tau_topk)
        # 'l2' has to skip the L2 normalisation everywhere an embedding is
        # produced: on normalised vectors -||q-k||^2 is a monotone map of the dot
        # product, so the Top-K would be identical to cosine.
        # Soft attention over the whole bank instead of Top-K, so the forecasting
        # loss reaches every candidate score rather than only the k already picked.
        # End-to-end retrieval: the forecast loss reaches the Stage-1 encoder
        # through the Top-K weights. Selection stays exactly as it was -- the
        # retrieval universe is still the full bank -- but the selected
        # candidates are re-encoded live so the candidate side carries gradient
        # too, which the precomputed key bank cannot.
        self.e2e_retrieval = bool(int(getattr(configs, 'stage2_e2e', 0)))
        self.rank_loss_mode = getattr(configs, 'stage2_rank_loss', 'none')
        self.rank_loss_weight = float(getattr(configs, 'stage2_rank_weight', 0.0))
        self.rank_margin = float(getattr(configs, 'stage2_rank_margin', 0.05))
        self.rank_top_p = int(getattr(configs, 'stage2_rank_top_p', 10))
        self.rank_hard_negatives = int(getattr(configs, 'stage2_rank_hard_negatives', 30))
        self.rank_random_negatives = int(getattr(configs, 'stage2_rank_random_negatives', 10))
        # v2 corrections. The audit found the v1 loss aimed almost entirely
        # outside the Top-K it was built to decompress: 3.7% of pairs inside,
        # carrying 1.9% of the loss, against pairs already 24x wider.
        self.rank_topk_gamma = float(getattr(configs, 'stage2_rank_topk_gamma', -1.0))
        self.rank_margin_mode = getattr(configs, 'stage2_rank_margin_mode', 'absolute')
        self.rank_margin_cap = float(getattr(configs, 'stage2_rank_margin_cap', 0.2))
        self.rank_sigma_mode = getattr(configs, 'stage2_rank_sigma_mode', 'fixed')
        if self.rank_loss_mode not in RANK_MODES:
            raise ValueError(
                f'stage2_rank_loss must be one of {RANK_MODES}; got {self.rank_loss_mode}'
            )
        if self.rank_loss_mode != 'none' and not self.e2e_retrieval:
            # The ranking loss needs differentiable candidate scores, which only
            # the end-to-end path produces. Stage-wise ranking lives in Stage-1.
            raise ValueError(
                'stage2_rank_loss requires stage2_e2e=1; for a stage-wise ranking '
                'control use the Stage-1 ranking loss instead'
            )
        self.retrieval_soft_all = bool(int(getattr(configs, 'retrieval_soft_all', 0)))
        self.retrieval_similarity = getattr(configs, 'retrieval_similarity', 'cosine')
        if self.retrieval_similarity not in ('cosine', 'l2'):
            raise ValueError(
                f'Unsupported retrieval_similarity: {self.retrieval_similarity}'
            )
        # The comparison Stage-1 was trained with. Without these Stage-2 reads the
        # embeddings back through a plain dot product, which is a different
        # function from the one they were shaped for -- a Stage-1 arm that learned
        # an asymmetric metric would have that metric silently discarded at
        # retrieval time, and its encoder judged by cosine.
        # Re-encode the whole memory every step instead of reading the epoch bank.
        # Only meaningful while the encoder trains, so callers gate it on e2e.
        self.e2e_full_online = bool(int(getattr(configs, 'stage2_e2e_full_online', 0)))

        self.retrieval_metric_kind = getattr(configs, 'stage1_retrieval_metric', 'cosine')
        if self.retrieval_metric_kind not in RETRIEVAL_METRICS:
            raise ValueError(
                f'Unsupported stage1_retrieval_metric: {self.retrieval_metric_kind}')
        self.retrieval_metric = (
            RetrievalMetric(
                kind=self.retrieval_metric_kind,
                dim=int(configs.d_model),
                scaled_dot=bool(int(getattr(configs, 'stage1_metric_scaled_dot', 1))),
                layer_norm=bool(int(getattr(configs, 'stage1_metric_layer_norm', 1))),
                output=getattr(configs, 'stage1_metric_output', 'dot'),
            )
            if self.retrieval_metric_kind != 'cosine' else None
        )
        self.retrieval_score = getattr(configs, 'stage1_retrieval_score', 'cosine')
        if self.retrieval_score not in ('cosine', 'pairwise_mlp'):
            raise ValueError(f'Unsupported stage1_retrieval_score: {self.retrieval_score}')
        self.pairwise_scorer = (
            PairwiseScorer(
                embedding_dim=int(configs.d_model),
                feature_type=getattr(configs, 'stage1_pairwise_feature', 'pair4'),
                hidden_dim=int(getattr(configs, 'stage1_pairwise_hidden', 256)),
                hidden_dim2=int(getattr(configs, 'stage1_pairwise_hidden2', 128)),
                dropout=float(getattr(configs, 'stage1_pairwise_dropout', 0.1)),
            )
            if self.retrieval_score == 'pairwise_mlp' else None
        )
        if self.retrieval_metric is not None and self.pairwise_scorer is not None:
            raise ValueError(
                'stage1_retrieval_metric and stage1_retrieval_score=pairwise_mlp are two '
                'different comparisons; pick one')

        # End-to-end retrieval supervision: lambda * KL(future-MSE teacher || cosine student).
        # Zero keeps the pure forecasting objective, i.e. the current behaviour.
        self.retrieval_kl_weight = float(getattr(configs, 'retrieval_kl_weight', 0.0))
        self.retrieval_kl_teacher = getattr(configs, 'retrieval_kl_teacher', 'ema')
        if self.retrieval_kl_teacher not in ('ema', 'future_mse'):
            raise ValueError(f'Unsupported retrieval_kl_teacher: {self.retrieval_kl_teacher}')
        if (
            self.retrieval_kl_teacher == 'ema'
            and self.retrieval_kl_weight > 0.0
            and self.relation_n_features != 1
        ):
            # The EMA teacher encodes candidate futures in relation_value_space,
            # which has no multi-feature counterpart to match the student's rows.
            raise ValueError(
                f'relation_input_space={self.relation_input_space} is multi-feature and '
                'is not supported by retrieval_kl_teacher=ema; use future_mse'
            )
        self.tau_teacher = float(getattr(configs, 'tau_teacher', 0.1))
        self.tau_student = float(getattr(configs, 'tau_student', 0.1))
        self.source_mode = configs.source_mode
        self.relation_graph_threshold = int(getattr(configs, 'relation_graph_threshold', 21))
        self.relation_top_n = int(getattr(configs, 'relation_top_n', 3))
        self.target_mode = configs.target_mode
        self.target_channel = configs.target_channel
        self.relation_value_space = getattr(configs, 'relation_value_space', 'absolute')
        self.memory_chunk_size = int(configs.memory_chunk_size)
        self.freeze_stage1_encoder = bool(int(configs.freeze_stage1_encoder))
        self.disable_retrieval = bool(int(getattr(configs, 'disable_retrieval', 0)))
        self.encoder_free_full_oracle = (
            getattr(configs, 'stage2_oracle_train_mode', 'none') == 'full'
        )
        self.fusion_mode = configs.fusion_mode
        self.stage2_relation_fusion = getattr(configs, 'stage2_relation_fusion', 'gate')
        if self.stage2_relation_fusion not in ('concat_linear', 'gate'):
            raise ValueError(f'Unsupported stage2_relation_fusion: {self.stage2_relation_fusion}')
        self.stage2_retrieval_encoder = getattr(configs, 'stage2_retrieval_encoder', 'online')
        if self.stage2_retrieval_encoder not in ('online', 'ema'):
            raise ValueError(f'Unsupported stage2_retrieval_encoder: {self.stage2_retrieval_encoder}')

        self.chronos_projection = None
        if self.disable_retrieval or self.encoder_free_full_oracle:
            self.stage1_encoder = None
            self.shared_cross_projection = None
        elif self.retrieval_backbone in ('identity', 'pearson', 'chronos'):
            self.stage1_encoder = None
            self.shared_cross_projection = None
            if self.retrieval_backbone == 'chronos':
                chronos_encoder = ChronosRelationEncoder(
                    model_id=getattr(configs, 'chronos_model_id', 'amazon/chronos-t5-base'),
                    embedding_dim=chronos_embedding_dim,
                    dtype=getattr(configs, 'chronos_dtype', 'bfloat16'),
                    random_init=bool(int(getattr(configs, 'chronos_random_init', 0))),
                    finetune=self.chronos_finetune,
                    grad_checkpointing=bool(int(getattr(configs, 'chronos_grad_checkpointing', 1))),
                    pooling=getattr(configs, 'chronos_pooling', 'mean'),
                )
                if self.chronos_finetune:
                    # Register normally so the weights reach the optimizer and are
                    # restored with the Stage-2 checkpoint. Load eagerly on CPU so
                    # state_dict keys already exist before any load_state_dict and
                    # before the optimizer is built; .to(device) moves them later.
                    self._chronos_encoder = chronos_encoder
                    chronos_encoder._load(torch.device('cpu'))
                else:
                    # Keep the frozen pretrained model out of Stage-2 state_dict/checkpoints.
                    object.__setattr__(self, '_chronos_encoder', chronos_encoder)
                if self.chronos_projection_dim:
                    self.chronos_projection = nn.Linear(
                        2 * chronos_embedding_dim, self.chronos_projection_dim
                    )
                    if not self.chronos_projection_trainable:
                        for param in self.chronos_projection.parameters():
                            param.requires_grad = False
                else:
                    self.chronos_projection = None
        else:
            self.stage1_encoder = RelationEncoder(configs)
            self.shared_cross_projection = nn.Linear(
                2 * self.relation_seq_len, self.relation_seq_len
            )
        # EMA teacher for the retrieval KL. It embeds candidate *futures*, which
        # is the objective Stage-1 optimises, so end-to-end keeps the same
        # retrieval target instead of swapping in a different one.
        self.teacher_encoder = None
        self.teacher_shared_cross_projection = None
        if self.retrieval_kl_teacher == 'ema' and self.stage1_encoder is not None:
            self.teacher_encoder = copy.deepcopy(self.stage1_encoder)
            self.teacher_shared_cross_projection = copy.deepcopy(self.shared_cross_projection)
            for module in (self.teacher_encoder, self.teacher_shared_cross_projection):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
        if self.freeze_stage1_encoder and self.stage1_encoder is not None:
            for param in self.stage1_encoder.parameters():
                param.requires_grad = False
            if self.shared_cross_projection is not None:
                for param in self.shared_cross_projection.parameters():
                    param.requires_grad = False

        self.base_head = BaseForecastHead(
            seq_len=configs.seq_len,
            pred_len=configs.pred_len,
            channels=configs.enc_in,
            mode=configs.base_head_mode,
        )
        self.relation_mixer = RelationMixer(
            pred_len=configs.pred_len,
            emb_dim=self.relation_emb_dim,
            hidden_dim=configs.relation_mixer_hidden,
            input_mode=configs.relation_mixer_input,
        )
        self.relation_concat_projection = nn.Linear(
            self._configured_source_slots() * configs.pred_len,
            configs.pred_len,
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
        if self.disable_retrieval:
            retrieval_only_modules = (
                self.relation_mixer,
                self.relation_concat_projection,
                self.gate,
                self.raft_concat_head,
            )
            for module in retrieval_only_modules:
                if module is None:
                    continue
                for param in module.parameters():
                    param.requires_grad = False
        self._chronos_pooled_cache = None
        self._shape_logged = False
        self.relation_sources = None
        self.relation_correlations = None
        self.checkpoint_relation_graph = None

    def _configured_source_slots(self):
        if self.disable_retrieval:
            return self.channels
        if self.source_mode in ('auto', 'topk_corr'):
            return min(self.relation_top_n, self.channels)
        return self.channels

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
        if self.disable_retrieval and self.relation_sources is None:
            return torch.arange(
                self.channels,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0).expand(self.channels, -1)
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
        if self.source_mode in ('auto', 'topk_corr'):
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
        if self.retrieval_backbone != 'stage1':
            raise RuntimeError('_relation_tensor is only available for the Stage-1 backbone')
        return build_relation_encoder_input(
            x,
            target_channel,
            source_channel,
            relation_input_space=self.relation_input_space,
            shared_cross_projection=self.shared_cross_projection,
            self_fill=self.stage1_encoder.self_fill,
        )

    def _channel_embeddings(self, x):
        if self.retrieval_backbone != 'chronos':
            return None
        x = transform_relation_history(x, self.relation_input_space)
        if x.size(1) > self.chronos_context_length:
            x = x[:, -self.chronos_context_length:]
        return self._chronos_encoder.encode_channel_tokens(x)

    def _branch_embedding(self, x, target_channel, source_channel, channel_embeddings=None):
        if self.retrieval_backbone in ('identity', 'pearson'):
            views = transform_relation_features(x, self.relation_input_space)
            parts = [view[..., target_channel] for view in views]
            parts += [view[..., source_channel] for view in views]
            relation = torch.cat(parts, dim=-1)
            if self.retrieval_backbone == 'pearson':
                # Cosine over jointly mean-centered [target || source] is exactly
                # the Pearson correlation RAFT scores its candidates with, which
                # also makes the score invariant to the relation_input_space
                # offset choice.
                relation = relation - relation.mean(dim=-1, keepdim=True)
            return self._maybe_normalize(relation)
        if self.retrieval_backbone == 'chronos':
            if channel_embeddings is None:
                channel_embeddings = self._channel_embeddings(x)
            hidden, mask = channel_embeddings
            pair_mask = mask[:, target_channel] & mask[:, source_channel]
            pair_hidden = torch.cat(
                [hidden[:, target_channel], hidden[:, source_channel]], dim=-1
            )
            pair_mask_float = pair_mask.unsqueeze(-1).to(pair_hidden.dtype)
            pooled = self._chronos_branch_pooled(
                x, target_channel, source_channel, channel_embeddings
            )
            return self._apply_chronos_projection(
                pooled, is_self=(source_channel == target_channel)
            )
        # RelationEncoder already skips its normalisation when similarity is l2,
        # so the same call covers both metrics.
        return self.stage1_encoder(
            self._relation_tensor(x, target_channel, source_channel)
        )

    def _retrieval_score_fn(self):
        """The comparison to score candidates with, or None for the dot product.

        Returned as a callable so the shared retrieval ops stay unaware of which
        module produced it; they fall back to their own similarity when it is
        None, which is what every pre-existing arm does.
        """
        if self.pairwise_scorer is not None:
            def pair_score(z_q, z_k):
                if z_k.dim() == 2:
                    # [N, D] bank: score in chunks, a pair feature is 2-4x wider
                    # than the embedding and the bank has thousands of rows.
                    if self.training and self.e2e_retrieval:
                        return self.pairwise_scorer(
                            z_q, z_k.unsqueeze(0).expand(z_q.size(0), -1, -1))
                    return self.pairwise_scorer.score_bank_in_chunks(z_q, z_k)
                return self.pairwise_scorer(z_q, z_k)
            return pair_score
        if self.retrieval_metric is not None:
            return lambda z_q, z_k: self.retrieval_metric.score(z_q, z_k)
        return None

    def _maybe_normalize(self, z):
        """L2-normalise for cosine scoring; leave the norm alone for l2."""
        if self.retrieval_similarity == 'l2':
            return z
        return F.normalize(z, dim=-1)

    def _apply_chronos_projection(self, pooled, is_self=False):
        """Project pooled Chronos features and L2-normalise for cosine Top-K.

        cross_only leaves the self branch untouched, exactly like
        shared_cross_projection does for the Stage-1 relation encoder: self is
        already D wide, only the concatenated cross branch needs 2D -> D.
        """
        skip = (
            self.chronos_projection is None
            or (self.chronos_projection_mode == 'cross_only' and is_self)
        )
        if not skip:
            pooled = self.chronos_projection(
                pooled.to(self.chronos_projection.weight.dtype)
            )
        return self._maybe_normalize(pooled)

    def _chronos_branch_pooled(self, x, target_channel, source_channel, channel_embeddings):
        """Pre-projection pooled features; cached so a trainable projection does
        not force re-encoding the memory with the frozen T5 every epoch.

        Width is D for an unprojected self branch and 2D otherwise.
        """
        if channel_embeddings is None:
            channel_embeddings = self._channel_embeddings(x)
        hidden, mask = channel_embeddings
        if (
            self.chronos_projection is not None
            and self.chronos_projection_mode == 'cross_only'
            and source_channel == target_channel
        ):
            target_mask = mask[:, target_channel].unsqueeze(-1).to(hidden.dtype)
            return (
                (hidden[:, target_channel] * target_mask).sum(dim=1)
                / target_mask.sum(dim=1).clamp_min(1.0)
            )
        pair_mask = mask[:, target_channel] & mask[:, source_channel]
        pair_hidden = torch.cat(
            [hidden[:, target_channel], hidden[:, source_channel]], dim=-1
        )
        pair_mask_float = pair_mask.unsqueeze(-1).to(pair_hidden.dtype)
        return (
            (pair_hidden * pair_mask_float).sum(dim=1)
            / pair_mask_float.sum(dim=1).clamp_min(1.0)
        )

    def _branch_memory(self, key_bank, target_channel, source_slot, source_channel, dtype, device):
        if self.retrieval_backbone == 'chronos':
            return key_bank[target_channel, source_slot].to(device=device)
        return key_bank[target_channel, source_slot].to(device=device, dtype=dtype)

    def load_stage1_checkpoint(self, ckpt_path, strict=True):
        if self.retrieval_backbone != 'stage1':
            raise RuntimeError(
                f'{self.retrieval_backbone} retrieval does not use a Stage-1 checkpoint'
            )
        ckpt = torch.load(ckpt_path, map_location='cpu')
        ckpt_args = ckpt.get('args', {})
        self.checkpoint_relation_graph = ckpt.get('relation_graph')
        expected = {
            'relation_encoder_type': self.stage1_encoder.encoder_type,
            'relation_input_space': self.relation_input_space,
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'enc_in': self.channels,
            'source_mode': self.source_mode,
            'relation_graph_threshold': self.relation_graph_threshold,
            'relation_top_n': self.relation_top_n,
        }
        expected.update({
            'relation_pooling': self.stage1_encoder.pooling,
            'relation_self_fill': self.stage1_encoder.self_fill,
        })
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
        if self.stage2_retrieval_encoder == 'online':
            encoder_prefix = 'encoder.'
            projection_prefix = 'shared_cross_projection.'
        else:
            encoder_prefix = 'teacher_encoder.'
            projection_prefix = 'teacher_shared_cross_projection.'

        encoder_state = {}
        projection_state = {}
        for key, value in state.items():
            clean_key = key[7:] if key.startswith('module.') else key
            if clean_key.startswith(encoder_prefix):
                encoder_state[clean_key[len(encoder_prefix):]] = value
            if clean_key.startswith(projection_prefix):
                projection_state[clean_key[len(projection_prefix):]] = value
        if not encoder_state:
            raise RuntimeError(
                f'No Stage-1 {self.stage2_retrieval_encoder} encoder weights found in checkpoint: {ckpt_path}'
            )
        if self.retrieval_backbone == 'stage1' and not projection_state:
            raise RuntimeError(
                f'No Stage-1 {self.stage2_retrieval_encoder} shared_cross_projection weights found in checkpoint: {ckpt_path}. '
                'This Stage-2 implementation requires a Stage-1 checkpoint trained with shared_cross_projection.'
            )
        # The learned comparison travels with the encoder. Loading one without the
        # other leaves a randomly initialised metric scoring trained embeddings,
        # which is worse than either alone and fails silently.
        for attr, prefix in (('retrieval_metric', 'retrieval_metric.'),
                             ('pairwise_scorer', 'pairwise_scorer.')):
            module = getattr(self, attr, None)
            if module is None:
                continue
            sub = {k[len(prefix):]: v for k, v in
                   ((key[7:] if key.startswith('module.') else key, value)
                    for key, value in state.items())
                   if k.startswith(prefix)}
            if not sub:
                raise RuntimeError(
                    f'{attr} is configured but the Stage-1 checkpoint has no {prefix}* '
                    f'weights: {ckpt_path}. Stage-1 must have been trained with the same '
                    'comparison.')
            module.load_state_dict(sub, strict=True)
            module.eval()

        missing, unexpected = self.stage1_encoder.load_state_dict(encoder_state, strict=strict)
        proj_missing, proj_unexpected = [], []
        if self.shared_cross_projection is not None:
            proj_missing, proj_unexpected = self.shared_cross_projection.load_state_dict(
                projection_state,
                strict=strict,
            )
        self.stage1_encoder.eval()
        if self.shared_cross_projection is not None:
            self.shared_cross_projection.eval()
        print(
            f'[stage2] loaded Stage-1 {self.stage2_retrieval_encoder} encoder '
            f'for {self.retrieval_backbone} retrieval from {ckpt_path}'
        )
        if missing or unexpected or proj_missing or proj_unexpected:
            msg = (
                f'Stage-1 retrieval backbone checkpoint mismatch for {ckpt_path}\n'
                f'missing keys: {missing}\n'
                f'unexpected keys: {unexpected}\n'
                f'projection missing keys: {proj_missing}\n'
                f'projection unexpected keys: {proj_unexpected}'
            )
            if strict:
                raise RuntimeError(msg)
            print(f'[stage2] {msg}')

    @torch.no_grad()
    def build_memory_key_bank(self, memory_x, device, chunk_size=None):
        was_training = self.training
        chunk_size = int(chunk_size or self.memory_chunk_size)
        memory_x = torch.as_tensor(memory_x, dtype=torch.float32)

        if self.retrieval_backbone in ('identity', 'pearson', 'chronos'):
            encoded = [
                [[] for _ in self.source_channels(c)]
                for c in range(self.channels)
            ]
            reuse_pooled = (
                self.retrieval_backbone == 'chronos'
                and self.chronos_projection_trainable
                and not self.chronos_finetune
            )
            if reuse_pooled and self._chronos_pooled_cache is not None:
                # The frozen T5 output never changes, so re-indexing after a
                # projection update only has to redo the cheap linear map.
                for c in range(self.channels):
                    for source_slot, _ in enumerate(self.source_channels(c)):
                        pooled = self._chronos_pooled_cache[c][source_slot]
                        branch = self._apply_chronos_projection(
                            pooled.to(device=device, dtype=torch.float32),
                            is_self=(self.source_channels(c)[source_slot] == c),
                        ).cpu().half()
                        encoded[c][source_slot].append(branch)
            else:
                pooled_cache = (
                    [[[] for _ in self.source_channels(c)] for c in range(self.channels)]
                    if reuse_pooled
                    else None
                )
                for start in range(0, memory_x.size(0), chunk_size):
                    cur = memory_x[start:start + chunk_size].to(device)
                    features = self._channel_embeddings(cur)
                    for c in range(self.channels):
                        for source_slot, r in enumerate(self.source_channels(c)):
                            if reuse_pooled:
                                pooled = self._chronos_branch_pooled(
                                    cur, c, r, channel_embeddings=features
                                )
                                pooled_cache[c][source_slot].append(pooled.cpu().half())
                                branch = self._apply_chronos_projection(
                                    pooled, is_self=(r == c)
                                ).cpu()
                            else:
                                branch = self._branch_embedding(
                                    cur, c, r, channel_embeddings=features
                                ).cpu()
                            if self.retrieval_backbone == 'chronos':
                                branch = branch.half()
                            encoded[c][source_slot].append(branch)
                if reuse_pooled:
                    self._chronos_pooled_cache = [
                        [torch.cat(parts, dim=0) for parts in row]
                        for row in pooled_cache
                    ]
                    print(
                        '[stage2] cached frozen Chronos pooled features for '
                        'projection-only re-indexing'
                    )
            bank = torch.stack([
                torch.stack([
                    torch.cat(parts, dim=0) for parts in target_parts
                ], dim=0)
                for target_parts in encoded
            ], dim=0)
            if was_training:
                self.train()
            return bank

        self.stage1_encoder.eval()
        self.shared_cross_projection.eval()
        banks = []

        for c in range(self.channels):
            source_banks = []
            for r in self.source_channels(c):
                encoded = []
                for start in range(0, memory_x.size(0), chunk_size):
                    cur = memory_x[start:start + chunk_size].to(device)
                    rel = self._relation_tensor(cur, c, r)
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

    def _relation_future_mse(
        self,
        batch_x,
        memory_y,
        memory_x_last,
        valid_mask,
        target_y,
        target_channel,
        source_channel,
    ):
        """Future MSE of every candidate for one relation branch: [B, N].

        This is the Stage-1 teacher signal. Unlike the oracle helpers below it
        keeps all candidates instead of taking a Top-K, because the retrieval KL
        has to reach the candidates the student did *not* select - that is the
        whole point of the term.
        """
        memory_target, query_target_offset = self._memory_value(
            batch_x, memory_y, memory_x_last, target_channel
        )
        memory_source, query_source_offset = self._memory_value(
            batch_x, memory_y, memory_x_last, source_channel
        )
        query_target = target_y[:, :, target_channel]
        query_source = target_y[:, :, source_channel]
        if self.relation_value_space == 'delta_last':
            query_target = query_target - query_target_offset.unsqueeze(-1)
            query_source = query_source - query_source_offset.unsqueeze(-1)

        query_relation = torch.cat([query_target, query_source], dim=-1)
        memory_relation = torch.cat([memory_target, memory_source], dim=-1)
        relation_length = float(query_relation.size(-1))
        query_sq = query_relation.pow(2).mean(dim=-1, keepdim=True)
        memory_sq = memory_relation.pow(2).mean(dim=-1).unsqueeze(0)
        relation_mse = (
            query_sq
            + memory_sq
            - 2.0
            * torch.matmul(query_relation, memory_relation.transpose(0, 1))
            / relation_length
        ).clamp_min(0.0)
        return relation_mse.masked_fill(~valid_mask, float('inf'))

    def _teacher_relation_tensor(self, future, target_channel, source_channel, offset):
        """Relation built from futures, for the EMA teacher. Mirrors Stage-1."""
        target = future[..., target_channel]
        if self.relation_value_space == 'delta_last':
            target = target - offset[:, target_channel].to(future.device).unsqueeze(-1)
        if source_channel == target_channel:
            return target.unsqueeze(1)
        source = future[..., source_channel]
        if self.relation_value_space == 'delta_last':
            source = source - offset[:, source_channel].to(future.device).unsqueeze(-1)
        # The teacher has to compose the pair exactly like the student does,
        # otherwise the KL target is defined in a different relation space.
        if self.stage1_encoder.self_fill != 'linear':
            return torch.stack([target, source], dim=1)
        pair = torch.cat([target, source], dim=-1)
        return self.teacher_shared_cross_projection(pair).unsqueeze(1)

    @torch.no_grad()
    def build_teacher_key_bank(self, memory_y, device, memory_x_last, chunk_size=None):
        """EMA embeddings of every candidate future: [C, S, N, D].

        Rebuilt on the same schedule as the student key bank, because the EMA
        weights move with the student.
        """
        if self.teacher_encoder is None:
            return None
        chunk_size = int(chunk_size or self.memory_chunk_size)
        memory_y = torch.as_tensor(memory_y, dtype=torch.float32)
        memory_x_last = torch.as_tensor(memory_x_last, dtype=torch.float32)
        banks = []
        for c in range(self.channels):
            source_banks = []
            for r in self.source_channels(c):
                encoded = []
                for start in range(0, memory_y.size(0), chunk_size):
                    cur = memory_y[start:start + chunk_size].to(device)
                    cur_offset = memory_x_last[start:start + chunk_size].to(device)
                    rel = self._teacher_relation_tensor(cur, c, r, cur_offset)
                    encoded.append(self.teacher_encoder(rel).cpu())
                source_banks.append(torch.cat(encoded, dim=0))
            banks.append(torch.stack(source_banks, dim=0))
        return torch.stack(banks, dim=0)

    @torch.no_grad()
    def update_ema_teacher(self, momentum):
        if self.teacher_encoder is None:
            return
        pairs = (
            (self.teacher_encoder, self.stage1_encoder),
            (self.teacher_shared_cross_projection, self.shared_cross_projection),
        )
        for teacher, student in pairs:
            for t_param, s_param in zip(teacher.parameters(), student.parameters()):
                t_param.data.mul_(momentum).add_(s_param.data, alpha=1.0 - momentum)
            for t_buf, s_buf in zip(teacher.buffers(), student.buffers()):
                t_buf.copy_(s_buf)

    @torch.no_grad()
    def _teacher_scores(self, target_y, memory_x_last_query, teacher_key_bank,
                        target_channel, source_slot):
        """Teacher similarity over all candidates, scored like the student.

        cosine: cos(EMA(query future), EMA(candidate futures)). l2: the negative
        mean squared distance - the encoder stops normalising for l2, so a bare
        dot product would be dominated by the embedding norm and collapse the
        teacher onto a single candidate.
        """
        rel = self._teacher_relation_tensor(
            target_y, target_channel, self.source_channels(target_channel)[source_slot],
            memory_x_last_query,
        )
        z_q = self.teacher_encoder(rel)
        z_k = teacher_key_bank[target_channel, source_slot].to(
            device=z_q.device, dtype=z_q.dtype
        )
        if self.retrieval_similarity == 'l2':
            q_l2 = z_q.float()
            k_l2 = z_k.float()
            return -(
                q_l2.pow(2).sum(dim=-1, keepdim=True)
                + k_l2.pow(2).sum(dim=-1).unsqueeze(0)
                - 2.0 * torch.matmul(q_l2, k_l2.transpose(0, 1))
            ) / float(q_l2.size(-1))
        return torch.matmul(z_q, z_k.transpose(0, 1))

    def _retrieval_kl_from_teacher_scores(self, student_scores, teacher_scores, valid_mask):
        """KL(teacher || student) where the teacher is already a similarity."""
        masked_fill = torch.finfo(student_scores.dtype).min / 4
        teacher_logits = (teacher_scores / self.tau_teacher).masked_fill(~valid_mask, masked_fill)
        teacher_prob = torch.softmax(teacher_logits, dim=-1).detach()
        student_logits = (student_scores / self.tau_student).masked_fill(~valid_mask, masked_fill)
        student_log_prob = torch.log_softmax(student_logits, dim=-1)
        return (
            teacher_prob * (torch.log(teacher_prob + 1e-8) - student_log_prob)
        ).sum(dim=-1)

    def _retrieval_kl(self, student_scores, future_mse, valid_mask):
        """KL(teacher || student) over every candidate, Stage-1's convention.

        teacher = softmax(-future_mse / tau_teacher), student = scores / tau_student.
        """
        masked_fill = torch.finfo(student_scores.dtype).min / 4
        teacher_logits = (-future_mse / self.tau_teacher).masked_fill(~valid_mask, masked_fill)
        teacher_prob = torch.softmax(teacher_logits, dim=-1).detach()

        student_logits = (student_scores / self.tau_student).masked_fill(~valid_mask, masked_fill)
        student_log_prob = torch.log_softmax(student_logits, dim=-1)

        kl = (
            teacher_prob * (torch.log(teacher_prob + 1e-8) - student_log_prob)
        ).sum(dim=-1)
        return kl

    def _relation_oracle_topk_candidates(
        self,
        batch_x,
        memory_y,
        memory_x_last,
        valid_mask,
        oracle_target_y,
        target_channel,
        source_channel,
    ):
        """Select branch-specific Top-K using concatenated target/source futures."""
        memory_target, query_target_offset = self._memory_value(
            batch_x, memory_y, memory_x_last, target_channel
        )
        memory_source, query_source_offset = self._memory_value(
            batch_x, memory_y, memory_x_last, source_channel
        )
        query_target = oracle_target_y[:, :, target_channel]
        query_source = oracle_target_y[:, :, source_channel]
        if self.relation_value_space == 'delta_last':
            query_target = query_target - query_target_offset.unsqueeze(-1)
            query_source = query_source - query_source_offset.unsqueeze(-1)

        query_relation = torch.cat([query_target, query_source], dim=-1)
        memory_relation = torch.cat([memory_target, memory_source], dim=-1)
        relation_length = float(query_relation.size(-1))
        query_sq = query_relation.pow(2).mean(dim=-1, keepdim=True)
        memory_sq = memory_relation.pow(2).mean(dim=-1).unsqueeze(0)
        relation_mse = (
            query_sq
            + memory_sq
            - 2.0
            * torch.matmul(query_relation, memory_relation.transpose(0, 1))
            / relation_length
        ).clamp_min(0.0)
        relation_mse = relation_mse.masked_fill(~valid_mask, float('inf'))

        k = min(self.top_k, memory_target.size(0))
        oracle_mse, oracle_idx = torch.topk(
            relation_mse, k=k, dim=-1, largest=False
        )
        oracle_valid = torch.isfinite(oracle_mse)
        oracle_target_values = memory_target[oracle_idx]
        return (
            oracle_idx,
            oracle_valid,
            oracle_target_values,
            oracle_mse,
        )

    def _weight_oracle_candidates(self, scores, oracle_idx, oracle_valid, oracle_values):
        """Weight Oracle-selected values with branch-specific encoder scores."""
        masked_fill = torch.finfo(scores.dtype).min / 4
        oracle_scores = scores.gather(1, oracle_idx)
        scaled_scores = (oracle_scores / self.tau_topk).masked_fill(
            ~oracle_valid, masked_fill
        )
        alpha = F.softmax(scaled_scores, dim=-1) * oracle_valid.float()
        alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        retrieved = (alpha.unsqueeze(-1) * oracle_values).sum(dim=1)
        return retrieved, alpha

    def _weight_full_oracle_candidates(self, oracle_mse, oracle_valid, oracle_values):
        """Weight Oracle-selected values directly with negative future MSE."""
        masked_fill = torch.finfo(oracle_mse.dtype).min / 4
        scaled_scores = (-oracle_mse / self.tau_topk).masked_fill(
            ~oracle_valid, masked_fill
        )
        alpha = F.softmax(scaled_scores, dim=-1) * oracle_valid.float()
        alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        retrieved = (alpha.unsqueeze(-1) * oracle_values).sum(dim=1)
        return retrieved, alpha

    @torch.no_grad()
    def build_retrieval_cache(
        self,
        batch_x,
        memory_y,
        valid_mask,
        key_bank,
        memory_x_last=None,
        oracle_target_y=None,
        full_oracle_only=False,
    ):
        if full_oracle_only and oracle_target_y is None:
            raise ValueError('full_oracle_only requires oracle_target_y')
        if not full_oracle_only and key_bank is None:
            raise ValueError('encoder-based retrieval requires a relation key bank')
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
        topk_mean_similarity_all = torch.zeros(
            bsz,
            self.channels,
            source_slots,
            device=batch_x.device,
            dtype=batch_x.dtype,
        )
        topk_weight_entropy_all = torch.zeros_like(topk_mean_similarity_all)
        cache_query_embeddings = (
            self.stage2_relation_fusion == 'gate'
            and self.relation_mixer.input_mode == 'retrieved_plus_query'
        )
        cached_query_dim = self.relation_emb_dim if cache_query_embeddings else 0
        relation_query_embs_all = torch.zeros(
            bsz,
            self.channels,
            source_slots,
            cached_query_dim,
            device=batch_x.device,
            dtype=batch_x.dtype,
        )
        candidate_oracle_relations_all = None
        candidate_oracle_model_relations_all = None
        candidate_oracle_top_k_effective_sc = None
        candidate_oracle_indices_sc = None
        candidate_oracle_mse_topk_sc = None
        candidate_oracle_valid_topk_sc = None
        student_oracle_recall_sc = None
        full_oracle_relations_all = None
        full_oracle_model_relations_all = None
        if oracle_target_y is not None:
            if not full_oracle_only:
                candidate_oracle_relations_all = torch.zeros_like(relation_outputs_all)
                candidate_oracle_model_relations_all = torch.zeros_like(relation_outputs_all)
            full_oracle_relations_all = torch.zeros_like(relation_outputs_all)
            full_oracle_model_relations_all = torch.zeros_like(relation_outputs_all)
            candidate_oracle_top_k_effective_sc = batch_x.new_zeros(
                bsz, self.channels, source_slots
            )
            oracle_k = min(self.top_k, memory_y.size(0))
            candidate_oracle_indices_sc = torch.full(
                (bsz, self.channels, source_slots, oracle_k),
                -1,
                dtype=torch.long,
                device=batch_x.device,
            )
            candidate_oracle_mse_topk_sc = batch_x.new_full(
                (bsz, self.channels, source_slots, oracle_k),
                float('inf'),
            )
            candidate_oracle_valid_topk_sc = torch.zeros(
                bsz,
                self.channels,
                source_slots,
                oracle_k,
                dtype=torch.bool,
                device=batch_x.device,
            )
            if not full_oracle_only:
                student_oracle_recall_sc = {
                    metric_k: batch_x.new_zeros(
                        bsz, self.channels, source_slots
                    )
                    for metric_k in (1, 5, 10)
                }
        debug_rows = []

        was_training = self.training
        if self.stage1_encoder is not None:
            self.stage1_encoder.eval()
        query_channel_embeddings = (
            self._channel_embeddings(batch_x) if not full_oracle_only else None
        )
        for c in self.target_channels():
            memory_value_c, query_offset_c = self._memory_value(batch_x, memory_y, memory_x_last, c)
            relation_debug_rows = []
            for source_slot, r in enumerate(self.source_channels(c)):
                oracle_idx = None
                oracle_valid = None
                oracle_values = None
                if oracle_target_y is not None:
                    (
                        oracle_idx,
                        oracle_valid,
                        oracle_values,
                        oracle_mse,
                    ) = self._relation_oracle_topk_candidates(
                        batch_x=batch_x,
                        memory_y=memory_y,
                        memory_x_last=memory_x_last,
                        valid_mask=valid_mask,
                        oracle_target_y=oracle_target_y,
                        target_channel=c,
                        source_channel=r,
                    )
                    candidate_oracle_top_k_effective_sc[
                        :, c, source_slot
                    ] = oracle_valid.float().sum(dim=-1)
                    candidate_oracle_indices_sc[
                        :, c, source_slot, :oracle_idx.size(-1)
                    ] = oracle_idx
                    candidate_oracle_mse_topk_sc[
                        :, c, source_slot, :oracle_mse.size(-1)
                    ] = oracle_mse
                    candidate_oracle_valid_topk_sc[
                        :, c, source_slot, :oracle_valid.size(-1)
                    ] = oracle_valid
                    full_retrieved, _ = self._weight_full_oracle_candidates(
                        oracle_mse=oracle_mse,
                        oracle_valid=oracle_valid,
                        oracle_values=oracle_values,
                    )
                    full_oracle_model_relations_all[
                        :, c, source_slot
                    ] = full_retrieved
                    full_oracle_relations_all[
                        :, c, source_slot
                    ] = self._restore_retrieved_value(
                        full_retrieved, query_offset_c
                    )

                if not full_oracle_only:
                    z_q = self._branch_embedding(
                        batch_x, c, r, channel_embeddings=query_channel_embeddings
                    )
                    z_mem = self._branch_memory(
                        key_bank, c, source_slot, r, z_q.dtype, batch_x.device
                    )
                    if self.retrieval_backbone == 'chronos':
                        if self.chronos_finetune:
                            # Keep the differentiable query in full precision and
                            # promote the half-stored keys instead.
                            z_mem = z_mem.to(z_q.dtype)
                        else:
                            z_q = z_q.to(z_mem.dtype)
                    r_cr, alpha, top_idx, top_scores, ret_debug = retrieve_relation_future(
                        z_q=z_q,
                        z_mem=z_mem,
                        memory_value_c=memory_value_c,
                        valid_mask=valid_mask,
                        top_k=self.top_k,
                        tau_topk=self.tau_topk,
                        similarity=self.retrieval_similarity,
                        soft_all=self.retrieval_soft_all,
                        score_fn=self._retrieval_score_fn(),
                    )
                    relation_outputs_all[:, c, source_slot] = r_cr
                    if student_oracle_recall_sc is not None:
                        student_valid = ret_debug['top_valid']
                        for metric_k in (1, 5, 10):
                            effective_k = min(
                                metric_k,
                                top_idx.size(-1),
                                oracle_idx.size(-1),
                            )
                            student_idx_k = top_idx[:, :effective_k]
                            oracle_idx_k = oracle_idx[:, :effective_k]
                            student_valid_k = student_valid[:, :effective_k]
                            oracle_valid_k = oracle_valid[:, :effective_k]
                            matched_oracle = (
                                oracle_idx_k.unsqueeze(-1)
                                == student_idx_k.unsqueeze(-2)
                            )
                            matched_oracle = (
                                matched_oracle
                                & oracle_valid_k.unsqueeze(-1)
                                & student_valid_k.unsqueeze(-2)
                            )
                            overlap = (
                                matched_oracle.any(dim=-1).float().sum(dim=-1)
                                / oracle_valid_k.float().sum(dim=-1).clamp_min(1.0)
                            )
                            student_oracle_recall_sc[metric_k][
                                :, c, source_slot
                            ] = overlap
                    if cache_query_embeddings:
                        relation_query_embs_all[:, c, source_slot] = z_q
                    if candidate_oracle_relations_all is not None:
                        oracle_retrieved, _ = self._weight_oracle_candidates(
                            scores=ret_debug['scores'],
                            oracle_idx=oracle_idx,
                            oracle_valid=oracle_valid,
                            oracle_values=oracle_values,
                        )
                        candidate_oracle_relations_all[:, c, source_slot] = (
                            self._restore_retrieved_value(oracle_retrieved, query_offset_c)
                        )
                        candidate_oracle_model_relations_all[:, c, source_slot] = oracle_retrieved
                    alpha_entropy = -(alpha * torch.log(alpha + 1e-8)).sum(dim=-1)
                    top_valid = ret_debug['top_valid']
                    topk_mean_similarity = (
                        top_scores.masked_fill(~top_valid, 0.0).sum(dim=-1)
                        / top_valid.float().sum(dim=-1).clamp_min(1.0)
                    )
                    topk_mean_similarity_all[:, c, source_slot] = topk_mean_similarity
                    topk_weight_entropy_all[:, c, source_slot] = alpha_entropy
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
            if not full_oracle_only:
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
            'topk_mean_similarity': topk_mean_similarity_all.detach(),
            'topk_weight_entropy': topk_weight_entropy_all.detach(),
        }
        if oracle_target_y is not None:
            candidate_oracle_mse_sc = batch_x.new_zeros(bsz, self.channels)
            candidate_oracle_mae_sc = batch_x.new_zeros(bsz, self.channels)
            full_oracle_mse_sc = batch_x.new_zeros(bsz, self.channels)
            full_oracle_mae_sc = batch_x.new_zeros(bsz, self.channels)
            relation_oracle_outputs_all = torch.zeros_like(relation_outputs_all)
            for c in self.target_channels():
                source_count = len(self.source_channels(c))
                target_c = oracle_target_y[:, :, c]
                full_model_relations_c = full_oracle_model_relations_all[
                    :, c, :source_count
                ]
                if self.stage2_relation_fusion == 'concat_linear':
                    full_prediction_model_c = self.relation_concat_projection(
                        full_model_relations_c.reshape(
                            full_model_relations_c.size(0), -1
                        )
                    )
                else:
                    _, full_beta_c, _ = self.relation_mixer(
                        full_model_relations_c,
                        None,
                    )
                    full_prediction_model_c = (
                        full_beta_c.unsqueeze(-1) * full_model_relations_c
                    ).sum(dim=1)
                full_prediction_c = self._restore_retrieved_value(
                    full_prediction_model_c,
                    batch_x[:, -1, c],
                )
                full_oracle_mse_sc[:, c] = (
                    (full_prediction_c - target_c) ** 2
                ).mean(dim=-1)
                full_oracle_mae_sc[:, c] = torch.abs(
                    full_prediction_c - target_c
                ).mean(dim=-1)

                if not full_oracle_only:
                    relation_outputs_c = relation_outputs_all[:, c, :source_count]
                    relation_query_embs_c = (
                        relation_query_embs_all[:, c, :source_count]
                        if cache_query_embeddings
                        else None
                    )
                    candidate_model_relations_c = (
                        candidate_oracle_model_relations_all[
                            :, c, :source_count
                        ]
                    )
                    if self.stage2_relation_fusion == 'concat_linear':
                        candidate_prediction_model_c = self.relation_concat_projection(
                            candidate_model_relations_c.reshape(
                                candidate_model_relations_c.size(0), -1
                            )
                        )
                    else:
                        _, beta_c, _ = self.relation_mixer(
                            candidate_model_relations_c,
                            relation_query_embs_c,
                        )
                        candidate_prediction_model_c = (
                            beta_c.unsqueeze(-1) * candidate_model_relations_c
                        ).sum(dim=1)
                    candidate_prediction_c = self._restore_retrieved_value(
                        candidate_prediction_model_c,
                        batch_x[:, -1, c],
                    )
                    candidate_oracle_mse_sc[:, c] = (
                        (candidate_prediction_c - target_c) ** 2
                    ).mean(dim=-1)
                    candidate_oracle_mae_sc[:, c] = torch.abs(
                        candidate_prediction_c - target_c
                    ).mean(dim=-1)

                    normal_relations_c = self._restore_retrieved_value(
                        relation_outputs_c,
                        batch_x[:, -1, c],
                    )
                    normal_relation_mse = (
                        (normal_relations_c - target_c.unsqueeze(1)) ** 2
                    ).mean(dim=-1)
                    normal_relation_idx = normal_relation_mse.argmin(dim=-1)
                    gather_idx = normal_relation_idx[:, None, None].expand(
                        -1, 1, self.pred_len
                    )
                    selected_normal = relation_outputs_c.gather(1, gather_idx)
                    relation_oracle_outputs_all[:, c, :source_count] = (
                        selected_normal.expand(-1, source_count, -1)
                    )

            cache.update({
                'full_oracle_mse_sc': full_oracle_mse_sc.detach(),
                'full_oracle_mae_sc': full_oracle_mae_sc.detach(),
                'candidate_oracle_top_k_effective_sc': (
                    candidate_oracle_top_k_effective_sc.detach()
                ),
                'candidate_oracle_indices_sc': candidate_oracle_indices_sc.detach(),
                'candidate_oracle_mse_topk_sc': candidate_oracle_mse_topk_sc.detach(),
                'candidate_oracle_valid_topk_sc': candidate_oracle_valid_topk_sc.detach(),
                'full_oracle_relation_outputs': full_oracle_model_relations_all.detach(),
            })
            if not full_oracle_only:
                cache.update({
                    'candidate_oracle_mse_sc': candidate_oracle_mse_sc.detach(),
                    'candidate_oracle_mae_sc': candidate_oracle_mae_sc.detach(),
                    'candidate_oracle_relation_outputs': (
                        candidate_oracle_model_relations_all.detach()
                    ),
                    'relation_oracle_relation_outputs': relation_oracle_outputs_all.detach(),
                })
                for metric_k, recall_sc in student_oracle_recall_sc.items():
                    cache[
                        f'student_relation_oracle_recall_at_{metric_k}_sc'
                    ] = recall_sc.detach()
        if debug_rows:
            cache['alpha_entropy'] = torch.stack([row['alpha_entropy'] for row in debug_rows], dim=1).mean(dim=1).detach()
            cache['alpha_top1'] = torch.stack([row['alpha_top1'] for row in debug_rows], dim=1).mean(dim=1).detach()
            cache['alpha_margin'] = torch.stack([row['alpha_margin'] for row in debug_rows], dim=1).mean(dim=1).detach()
            cache['top_k_effective'] = torch.stack([row['top_k_effective'] for row in debug_rows], dim=1).mean(dim=1).detach()
        return cache

    def _residual_pair_mse(self, query_residual, memory_residual, target_channel):
        """Pairwise MSE between query and candidate base-forecast residuals.

        Same expansion the Stage-1 residual teacher uses, so the two agree by
        construction: one matmul against the whole bank, no per-candidate forward.
        """
        q = query_residual[:, :, target_channel]
        k = memory_residual[:, :, target_channel].to(q.device, q.dtype)
        return (
            q.square().mean(-1, keepdim=True)
            + k.square().mean(-1).unsqueeze(0)
            - 2.0 * torch.matmul(q, k.transpose(0, 1)) / q.size(-1)
        ).clamp_min(0.0)

    def encode_candidate_histories(self, candidate_x, target_channel, source_channel):
        """Embed raw candidate windows with the *live* encoder, gradient on.

        The key bank is built once per epoch and detached, so scores taken against
        it carry no candidate-side gradient. Re-encoding the handful of selected
        candidates restores it. Unique ids are encoded once; gradient checkpointing
        is deliberately not used here because it leaks memory on this torch build.
        """
        if self.retrieval_backbone != 'stage1':
            raise RuntimeError('end-to-end re-encoding requires the Stage-1 backbone')
        relation = self._relation_tensor(candidate_x, target_channel, source_channel)
        return self.stage1_encoder(relation)

    def _reencode_all_candidates(self, candidate_x, target_channel, source_channel):
        """Embed the whole memory with the *current* encoder, in the graph.

        The key bank is built once per epoch. Under joint training the encoder
        keeps moving inside that epoch, so selection reads embeddings the encoder
        has already left behind while only the chosen Top-K get re-encoded live --
        the candidate that would now rank first may never be looked at. Stage-1
        measured what that costs: removing the same staleness raised Recall@10 by
        5.6-41.9%.

        Serving still uses an index; this only changes how the same function is
        computed during training, so there is no train/serve mismatch.
        """
        return self.encode_candidate_histories(
            candidate_x, target_channel, source_channel
        )

    def _reencode_indices(self, candidate_x, indices, target_channel, source_channel):
        """Embeddings for [B, K] candidate ids, shaped [B, K, D]."""
        flat = indices.reshape(-1)
        unique, inverse = torch.unique(flat, return_inverse=True)
        embeddings = self.encode_candidate_histories(
            candidate_x.index_select(0, unique), target_channel, source_channel
        )
        return embeddings.index_select(0, inverse).view(*indices.shape, -1)

    def expand_candidate_indices(self, candidate_indices, bsz):
        """Accept [K], [B, K] or [B, C, K] candidate ids and return [B, C, K].

        Pools are mined per query and per channel -- each target channel has its
        own key bank -- but a diagnostic that shares one global pool is just the
        degenerate case, so all three shapes stay valid callers.
        """
        idx = candidate_indices.long()
        if idx.dim() == 1:
            idx = idx.view(1, 1, -1).expand(bsz, self.channels, -1)
        elif idx.dim() == 2:
            if idx.size(0) != bsz:
                raise ValueError(f'candidate_indices batch {idx.size(0)} != {bsz}')
            idx = idx.unsqueeze(1).expand(-1, self.channels, -1)
        elif idx.dim() == 3:
            if idx.shape[:2] != (bsz, self.channels):
                raise ValueError(
                    f'candidate_indices leading shape {tuple(idx.shape[:2])} '
                    f'!= {(bsz, self.channels)}'
                )
        else:
            raise ValueError(f'candidate_indices must be 1-D, 2-D or 3-D, got {idx.dim()}-D')
        return idx.contiguous()

    def relation_values_for_candidates(self, batch_x, memory_y, memory_x_last, candidate_indices):
        """Retrieval-branch value of individual candidates, in the model's own space.

        `build_retrieval_cache` stores `r_cr` -- the top-k weighted memory value --
        *without* restoring the query offset, and `forward` consumes it in exactly
        that space. Retrieving a single candidate is the degenerate case where one
        weight is 1, so its branch value is just that candidate's `_memory_value`
        row. Deriving it here keeps analysis code from restating the delta_last
        convention, which is where the earlier residual diagnostics went wrong.

        Returns [B, channels, K, pred_len].
        """
        index = self.expand_candidate_indices(candidate_indices, batch_x.size(0)).to(batch_x.device)
        values = [
            self._memory_value(batch_x, memory_y, memory_x_last, c)[0]
            .index_select(0, index[:, c].reshape(-1))
            .view(index.size(0), index.size(2), self.pred_len)
            for c in range(self.channels)
        ]
        return torch.stack(values, dim=1)

    def forward_from_retrieval_values(self, relation_outputs, *, batch_x, retrieval_cache,
                                      **forward_kwargs):
        """Production forward with the retrieval branch forced to supplied values.

        This calls `forward` rather than re-implementing the fusion tail on
        purpose: the mixer, the gate and the final `+ output_offset` restore have
        to stay a single source of truth. `relation_outputs` is
        [B, channels, source_slots, pred_len] in the same space the cache holds,
        i.e. not offset-restored.
        """
        if retrieval_cache is None:
            raise ValueError('forward_from_retrieval_values needs a retrieval cache to override')
        expected = (batch_x.size(0), self.channels, self.num_source_slots(), self.pred_len)
        if tuple(relation_outputs.shape) != expected:
            raise ValueError(
                f'relation_outputs shape {tuple(relation_outputs.shape)} != {expected}'
            )
        cache = dict(retrieval_cache)
        cache['relation_outputs'] = relation_outputs
        return self.forward(batch_x=batch_x, retrieval_cache=cache, **forward_kwargs)

    @torch.no_grad()
    def evaluate_candidate_correction(self, *, batch_x, batch_y, candidate_indices,
                                      memory_y, valid_mask, key_bank, memory_x_last=None,
                                      retrieval_cache=None, candidate_chunk=16):
        """Per-(query, candidate, channel) forecast utility, measured through `forward`.

        Each candidate is injected alone into the retrieval branch and the real
        Stage-2 forward is run, so the returned number is that candidate's actual
        downstream effect rather than a residual-algebra stand-in:

            utility[q, k, c] = MSE_c(Y_q, no-retrieval) - MSE_c(Y_q, final given k)

        When a target channel has several source slots they are all set to the
        same candidate, that being the analogue of "only this window is retrieved".

        `candidate_indices` may be [K] (one shared pool), [B, K] (per query) or
        [B, C, K] (per query and target channel).

        Returns (utility [B, K, C], base_mse [B, C]).
        """
        bsz, horizon, channels = batch_y.shape
        slots = self.num_source_slots()
        values = self.relation_values_for_candidates(
            batch_x, memory_y, memory_x_last, candidate_indices
        )
        num_candidates = values.size(2)
        base_mse = None
        chunks = []
        for start in range(0, num_candidates, candidate_chunk):
            block = values[:, :, start:start + candidate_chunk, :]
            group = block.size(2)
            # Query b under the k-th pooled candidate lands at row k * bsz + b.
            x_rep = batch_x.unsqueeze(0).expand(group, -1, -1, -1).reshape(group * bsz, *batch_x.shape[1:])
            branch = (
                block.permute(2, 0, 1, 3)           # [g, B, C, H]
                .unsqueeze(3).expand(-1, -1, -1, slots, -1)
                .reshape(group * bsz, channels, slots, horizon)
            )
            cache = {
                key: (
                    value.unsqueeze(0).expand(group, *([-1] * value.dim()))
                    .reshape(group * value.size(0), *value.shape[1:])
                    if torch.is_tensor(value) and value.size(0) == bsz else value
                )
                for key, value in retrieval_cache.items()
            }
            y_final, y_base = self.forward_from_retrieval_values(
                branch,
                batch_x=x_rep,
                retrieval_cache=cache,
                memory_y=memory_y,
                valid_mask=valid_mask.unsqueeze(0).expand(group, -1, -1).reshape(group * bsz, -1),
                key_bank=key_bank,
                memory_x_last=memory_x_last,
            )[:2]
            y_final = y_final.view(group, bsz, horizon, channels)
            if base_mse is None:
                base = y_base.view(group, bsz, horizon, channels)[0]
                if base.shape != batch_y.shape:
                    raise ValueError(f'base shape {tuple(base.shape)} != {tuple(batch_y.shape)}')
                base_mse = (base - batch_y).square().mean(dim=1)
            target = batch_y.unsqueeze(0).expand_as(y_final)
            chunks.append((y_final - target).square().mean(dim=2))
        final_mse = torch.cat(chunks, dim=0).permute(1, 0, 2)
        return base_mse.unsqueeze(1) - final_mse, base_mse

    def forward(self, batch_x, memory_y, valid_mask, key_bank, memory_x_last=None, retrieval_cache=None,
                target_y=None, teacher_key_bank=None, candidate_x=None,
                query_residual=None, memory_residual=None):
        bsz = batch_x.size(0)
        rank_loss_terms = []
        rank_metric_rows = []
        geometry_rows = []
        output_offset = batch_x[:, -1:, :].detach()
        # target_y is the ground-truth future; it only ever feeds the detached
        # teacher of the retrieval KL, never the forecast path.
        collect_retrieval_kl = self.retrieval_kl_weight != 0.0 and target_y is not None
        if collect_retrieval_kl and self.retrieval_kl_teacher == 'ema':
            if teacher_key_bank is None:
                raise ValueError(
                    'retrieval_kl_teacher=ema requires a teacher key bank; it has to be '
                    'rebuilt whenever the student key bank is, because the EMA weights move'
                )
        retrieval_kl_terms = []
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
        topk_mean_similarity_all = torch.zeros(
            bsz,
            self.channels,
            source_slots,
            device=batch_x.device,
            dtype=batch_x.dtype,
        )
        topk_weight_entropy_all = torch.zeros_like(topk_mean_similarity_all)

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
        if self.encoder_free_full_oracle and retrieval_cache is None:
            raise RuntimeError(
                'Full Oracle forward requires the precomputed future-MSE retrieval cache'
            )

        debug_rows = []
        first_debug = None
        cached_relation_outputs = None
        cached_relation_query_embs = None
        if retrieval_cache is not None:
            cached_relation_outputs = retrieval_cache['relation_outputs'].to(batch_x.device)
            cached_relation_query_embs = retrieval_cache['relation_query_embs'].to(batch_x.device)
            if 'topk_mean_similarity' in retrieval_cache:
                topk_mean_similarity_all = retrieval_cache['topk_mean_similarity'].to(batch_x.device)
            if 'topk_weight_entropy' in retrieval_cache:
                topk_weight_entropy_all = retrieval_cache['topk_weight_entropy'].to(batch_x.device)

        query_channel_embeddings = None
        if cached_relation_outputs is None and self.retrieval_backbone == 'chronos':
            query_channel_embeddings = self._channel_embeddings(batch_x)

        for c in self.target_channels():
            relation_outputs = []
            relation_query_embs = []
            relation_debug_rows = []

            if cached_relation_outputs is None:
                memory_value_c, query_offset_c = self._memory_value(batch_x, memory_y, memory_x_last, c)
                for source_slot, r in enumerate(self.source_channels(c)):
                    # The query embedding is only detached when nothing in the
                    # retrieval space is trainable. Detaching it whenever the
                    # backbone is Chronos would silently make --chronos_finetune
                    # and a trainable projection no-ops: the encoder graph would
                    # still be built and held, but no gradient could reach it.
                    if self.retrieval_backbone == 'chronos':
                        detach_query = not (
                            self.chronos_finetune or self.chronos_projection_trainable
                        )
                    else:
                        detach_query = bool(self.freeze_stage1_encoder)
                    if detach_query:
                        with torch.no_grad():
                            z_q = self._branch_embedding(
                                batch_x, c, r, channel_embeddings=query_channel_embeddings
                            )
                    else:
                        z_q = self._branch_embedding(
                            batch_x, c, r, channel_embeddings=query_channel_embeddings
                        )
                    if self.e2e_full_online and self.training:
                        if candidate_x is None:
                            raise ValueError(
                                'stage2_e2e_full_online=1 needs candidate_x [N, L, C]')
                        z_mem = self._reencode_all_candidates(
                            candidate_x, c, r).to(z_q.dtype)
                    else:
                        z_mem = self._branch_memory(
                            key_bank, c, source_slot, r, z_q.dtype, batch_x.device
                        )
                    if self.retrieval_backbone == 'chronos':
                        if self.chronos_finetune:
                            # Keep the differentiable query in full precision and
                            # promote the half-stored keys instead.
                            z_mem = z_mem.to(z_q.dtype)
                        else:
                            z_q = z_q.to(z_mem.dtype)
                    r_cr, alpha, top_idx, top_scores, ret_debug = retrieve_relation_future(
                        z_q=z_q,
                        z_mem=z_mem,
                        memory_value_c=memory_value_c,
                        valid_mask=valid_mask,
                        top_k=self.top_k,
                        tau_topk=self.tau_topk,
                        similarity=self.retrieval_similarity,
                        soft_all=self.retrieval_soft_all,
                    )
                    if self.e2e_retrieval:
                        if candidate_x is None:
                            raise ValueError(
                                'stage2_e2e=1 needs candidate_x [N, L, C] so the selected '
                                'candidates can be re-encoded differentiably'
                            )
                        # Selection is untouched; only the scores behind the Top-K
                        # weights become differentiable on both sides.
                        z_k_sel = self._reencode_indices(candidate_x, top_idx, c, r)
                        r_cr, alpha, top_scores = reweight_selected_candidates(
                            z_q=z_q, z_k_sel=z_k_sel, values=ret_debug['v_top'],
                            top_valid=ret_debug['top_valid'], tau_topk=self.tau_topk,
                            similarity=self.retrieval_similarity,
                            score_fn=self._retrieval_score_fn(),
                        )
                        ret_debug['top_scores'] = top_scores
                        ret_debug['alpha'] = alpha
                        geometry_rows.append({
                            **score_geometry(top_scores.detach(), ret_debug['top_valid']),
                            **weight_geometry(alpha.detach(), ret_debug['top_valid']),
                            **embedding_geometry(z_q.detach(), z_k_sel.detach()),
                        })
                        if self.rank_loss_mode != 'none':
                            if query_residual is None or memory_residual is None:
                                raise ValueError(
                                    'a ranking loss needs the cached base-forecast '
                                    'residuals for its teacher'
                                )
                            teacher = -self._residual_pair_mse(
                                query_residual, memory_residual, c
                            )
                            mined, mining_counts = mine_ranking_candidates(
                                teacher, ret_debug['scores'], valid_mask,
                                top_p=self.rank_top_p,
                                hard_negatives=self.rank_hard_negatives,
                                random_negatives=self.rank_random_negatives,
                            )
                            z_k_rank = self._reencode_indices(candidate_x, mined, c, r)
                            if self.retrieval_similarity == 'l2':
                                # Score the pairs the same way retrieval does, or
                                # the loss trains an ordering the retriever never uses.
                                rank_scores = -(
                                    z_q.float().pow(2).sum(-1, keepdim=True)
                                    + z_k_rank.float().pow(2).sum(-1)
                                    - 2.0 * (z_q.float().unsqueeze(1) * z_k_rank.float()).sum(-1)
                                ) / float(z_q.size(-1))
                            else:
                                rank_scores = (z_q.unsqueeze(1) * z_k_rank.to(z_q.dtype)).sum(-1)
                            # Which mined slots are the Top-K Stage-2 actually weights.
                            mined_in_topk = (
                                mined.unsqueeze(-1) == top_idx.unsqueeze(-2)
                            ).any(-1)
                            rank_term, rank_metrics = ranking_loss(
                                teacher.gather(1, mined),
                                rank_scores,
                                valid_mask.gather(1, mined),
                                mode=self.rank_loss_mode,
                                margin=self.rank_margin,
                                topk_mask=mined_in_topk,
                                gamma=(self.rank_topk_gamma
                                       if self.rank_topk_gamma >= 0.0 else None),
                                margin_mode=self.rank_margin_mode,
                                margin_cap=self.rank_margin_cap,
                                sigma_mode=self.rank_sigma_mode,
                            )
                            if rank_term is not None:
                                rank_loss_terms.append(rank_term)
                                rank_metric_rows.append({**rank_metrics, **mining_counts})
                    if collect_retrieval_kl:
                        # ret_debug['scores'] is the cosine over *every* candidate,
                        # before the non-differentiable Top-K, so the KL gradient
                        # can promote a candidate the student currently ranks out.
                        if self.retrieval_kl_teacher == 'ema':
                            teacher_scores = self._teacher_scores(
                                target_y, batch_x[:, -1, :].detach(),
                                teacher_key_bank, c, source_slot,
                            )
                            retrieval_kl_terms.append(
                                self._retrieval_kl_from_teacher_scores(
                                    ret_debug['scores'], teacher_scores, valid_mask
                                )
                            )
                        else:
                            branch_future_mse = self._relation_future_mse(
                                batch_x, memory_y, memory_x_last, valid_mask,
                                target_y, c, r,
                            )
                            retrieval_kl_terms.append(
                                self._retrieval_kl(
                                    ret_debug['scores'], branch_future_mse, valid_mask
                                )
                            )
                    relation_outputs.append(r_cr)
                    relation_query_embs.append(z_q)
                    alpha_entropy = -(alpha * torch.log(alpha + 1e-8)).sum(dim=-1)
                    top_valid = ret_debug['top_valid']
                    topk_mean_similarity = (
                        top_scores.masked_fill(~top_valid, 0.0).sum(dim=-1)
                        / top_valid.float().sum(dim=-1).clamp_min(1.0)
                    )
                    topk_mean_similarity_all[:, c, source_slot] = topk_mean_similarity
                    topk_weight_entropy_all[:, c, source_slot] = alpha_entropy
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

            if self.stage2_relation_fusion == 'concat_linear':
                expected = self.relation_concat_projection.in_features
                actual = relation_outputs.size(1) * self.pred_len
                if actual != expected:
                    raise RuntimeError(
                        'Stage-2 relation_concat_projection input dimension mismatch: '
                        f'checkpoint/config expects {expected}, current active relation order gives {actual}. '
                        'Check relation_top_n/source_mode and resume with matching Stage-2 config.'
                    )
                relation_concat = relation_outputs.reshape(relation_outputs.size(0), -1)
                y_ret_c = self.relation_concat_projection(relation_concat)
                beta_c = relation_outputs.new_full(
                    (relation_outputs.size(0), relation_outputs.size(1)),
                    1.0 / max(relation_outputs.size(1), 1),
                )
                relation_scores = relation_outputs.new_zeros(
                    relation_outputs.size(0),
                    relation_outputs.size(1),
                )
            else:
                y_ret_c, beta_c, relation_scores = self.relation_mixer(
                    relation_outputs,
                    relation_query_embs,
                )
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
            'topk_mean_similarity': topk_mean_similarity_all,
            'topk_weight_entropy': topk_weight_entropy_all,
            'source_indices': self.source_index_tensor(batch_x.device),
        }
        if retrieval_kl_terms:
            # [B, R] -> per-query mean, so _loss can drop the leaking queries.
            debug['retrieval_kl_per_query'] = torch.stack(retrieval_kl_terms, dim=1).mean(dim=1)
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
        if rank_loss_terms:
            # Averaged over relation branches, matching how every other Stage-2
            # per-branch quantity is reduced.
            debug['rank_loss_term'] = torch.stack(rank_loss_terms).mean()
        if rank_metric_rows:
            for key in rank_metric_rows[0]:
                values = [row[key] for row in rank_metric_rows if key in row]
                debug[key] = torch.stack([
                    v if torch.is_tensor(v) else batch_x.new_tensor(float(v)) for v in values
                ]).float().mean()
        if geometry_rows:
            for key in geometry_rows[0]:
                debug[key] = torch.stack([
                    row[key] for row in geometry_rows if key in row
                ]).float().mean()
        y_final_out = y_final_all + output_offset
        y_base_out = y_base_all + output_offset
        y_ret_out = y_ret_all + output_offset
        return y_final_out, y_base_out, y_ret_out, beta_all, lambda_all, debug
