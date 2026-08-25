"""Target-conditioned cross-channel context for utility-aware selection.

The question this exists to answer: target-channel past alone did not identify
which historical correction is useful (ResDirect beat ResSel 8/8, and the
shuffle-controlled hybrid found no complementary query subset). Does the
simultaneous past of *related source channels* disambiguate the future
correction regime?

Three pieces, deliberately small:

  SharedTemporalEncoder   X_c [B,L] -> z_c [B,d], one set of weights for every
                          channel and for query and candidate alike
  CrossChannelMixer       target-as-query attention over the source embeddings,
                          z_ctx = z_E + gamma * z_src
  ContextEncoder          the two above behind one call, where
                          use_cross_channel_context=0 makes it exactly z_E

The fairness rule the comparison depends on: every arm (ResDirect and ResSel,
target-only and cross-channel) instantiates the *same* encoder class with the
same width. Target-only arms simply do not build a mixer. So a difference
between arms is a difference in what information reaches the head, never a
difference in encoder capacity.

On the encoder being an MLP rather than the patch-conv/Transformer stack: the
selection arms re-encode pooled candidates on every optimisation step, so
encoder cost multiplies by pool width. An MLP keeps a full arm to minutes, and
since capacity is held identical across arms the D-vs-B verdict does not depend
on which family the encoder comes from. This study is not an architecture
comparison.
"""

import torch
import torch.nn as nn


class SharedTemporalEncoder(nn.Module):
    """X [.., L] -> z [.., d]. Shared by every channel, query and candidate."""

    def __init__(self, seq_len, d_model=128, d_ff=256, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.net = nn.Sequential(
            nn.Linear(seq_len, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class CrossChannelMixer(nn.Module):
    """z_E + gamma * attention(Q=z_E, K=V=z_sources).

    gamma starts near zero on purpose: at initialisation the contextual
    representation is the target-only representation, so the cross-channel arm
    begins from the baseline rather than from a perturbed version of it, and any
    gain has to be learned rather than handed over by initialisation noise.
    """

    def __init__(self, d_model, scale_init=1e-2, channel_wise_scale=False,
                 num_targets=None):
        super().__init__()
        self.d_model = d_model
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        if channel_wise_scale:
            if not num_targets:
                raise ValueError('channel_wise_scale requires num_targets')
            self.gamma = nn.Parameter(torch.full((num_targets,), float(scale_init)))
        else:
            self.gamma = nn.Parameter(torch.tensor(float(scale_init)))
        self.channel_wise_scale = channel_wise_scale

    def forward(self, z_target, z_sources, target_slot=None,
                return_attention=False):
        """z_target [B, d], z_sources [B, K, d] -> z_ctx [B, d]."""
        q = self.query_proj(z_target).unsqueeze(1)               # [B, 1, d]
        k = self.key_proj(z_sources)                             # [B, K, d]
        v = self.value_proj(z_sources)
        logits = (q * k).sum(-1) / (self.d_model ** 0.5)         # [B, K]
        attention = torch.softmax(logits, dim=-1)
        z_src = (attention.unsqueeze(-1) * v).sum(1)             # [B, d]
        gamma = self.gamma
        if self.channel_wise_scale:
            if target_slot is None:
                raise ValueError('channel_wise_scale requires target_slot')
            gamma = gamma[target_slot]
        z_ctx = z_target + gamma * z_src
        if return_attention:
            return z_ctx, attention
        return z_ctx


class ContextEncoder(nn.Module):
    """Shared encoder plus optional cross-channel mixer.

    `source_index` is [C, K] long: row c holds the source channels selected for
    target c, self excluded. It is registered as a buffer so it travels with the
    checkpoint -- an arm reloaded with a different source set would not be the
    arm that was trained.
    """

    def __init__(self, seq_len, source_index, d_model=128, d_ff=256,
                 dropout=0.1, use_cross_channel_context=True,
                 scale_init=1e-2, channel_wise_scale=False):
        super().__init__()
        self.encoder = SharedTemporalEncoder(seq_len, d_model, d_ff, dropout)
        source_index = torch.as_tensor(source_index, dtype=torch.long)
        if source_index.dim() != 2:
            raise ValueError('source_index must be [channels, num_sources]')
        self.register_buffer('source_index', source_index)
        self.use_cross_channel_context = bool(use_cross_channel_context)
        self.mixer = (
            CrossChannelMixer(d_model, scale_init, channel_wise_scale,
                              num_targets=source_index.size(0))
            if self.use_cross_channel_context else None
        )

    @property
    def d_model(self):
        return self.encoder.d_model

    def encode_channels(self, x):
        """x [B, L, C] -> z [B, C, d], every channel through the same weights."""
        return self.encoder(x.permute(0, 2, 1))

    def forward(self, x, target_channel, z_channels=None, return_attention=False):
        """x [B, L, C], one target channel -> contextual embedding [B, d].

        With the context switched off this is exactly encoder(x[:, :, target]),
        which is what makes the target-only arm a true baseline of this model
        rather than a separate model that happens to be smaller.
        """
        if z_channels is None:
            z_channels = self.encode_channels(x)
        z_target = z_channels[:, target_channel]
        if self.mixer is None:
            return (z_target, None) if return_attention else z_target
        sources = self.source_index[target_channel]
        if sources.numel() == 0:
            return (z_target, None) if return_attention else z_target
        z_sources = z_channels[:, sources]                       # [B, K, d]
        return self.mixer(
            z_target, z_sources, target_slot=target_channel,
            return_attention=return_attention,
        )


class ResidualHead(nn.Module):
    """z [B, d] -> predicted base-forecast error [B, T]."""

    def __init__(self, d_model, pred_len, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, pred_len),
        )

    def forward(self, z):
        return self.net(z)


def build_source_index(experiment, saved_args, topk, mode='pearson_topk',
                       metrics_root='./metrics'):
    """Per-target source channels, self excluded, from the train split only.

    Reuses the repo's own relation-graph builder, so the correlation matrix, the
    train-only rule and the on-disk format are the ones Stage-1 already uses.
    That builder keeps self in slot 0 and asks for `relation_top_n` sources
    *including* self, hence topk + 1 here and the slot-0 drop below.
    """
    from types import SimpleNamespace

    from utils.relation_graph import load_or_build_relation_graph

    if mode != 'pearson_topk':
        raise ValueError(f'Unsupported cross_channel_source_mode: {mode}')

    channels = int(saved_args.enc_in)
    topk = min(int(topk), channels - 1)
    train_dataset, _ = experiment._get_data(flag='train', shuffle=False)
    graph_args = SimpleNamespace(
        enc_in=channels,
        data_path=saved_args.data_path,
        source_mode='topk_corr',
        relation_top_n=topk + 1,
        relation_graph_path='',
        metrics_csv_dir=metrics_root,
        relation_graph_threshold=int(getattr(saved_args, 'relation_graph_threshold', 21)),
    )
    graph = load_or_build_relation_graph(train_dataset, graph_args)

    source_index, correlations = [], []
    for target in range(channels):
        row = graph['sources'][target]
        corr_row = graph['correlations'][target]
        if int(row[0]) != target:
            raise ValueError(f'relation graph lost self at slot 0 for target {target}')
        # Slot 0 is self by construction and must not become its own source.
        source_index.append([int(s) for s in row[1:]])
        correlations.append([float(c) for c in corr_row[1:]])
    for target, row in enumerate(source_index):
        if target in row:
            raise ValueError(f'target {target} appears in its own source set')
    return (
        torch.tensor(source_index, dtype=torch.long),
        torch.tensor(correlations, dtype=torch.float),
        graph.get('channel_names', [f'ch{i}' for i in range(channels)]),
    )


def count_parameters(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable
