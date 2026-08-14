import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    """Drop the right-side padding a causal dilated conv adds."""

    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[..., :-self.chomp_size].contiguous()


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel axis of [B, C, L], independently per step.

    Normalising per time step keeps the block causal, unlike a norm that also
    pools over L.
    """

    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x):
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class TemporalBlock(nn.Module):
    """Two causal dilated convolutions plus a residual connection.

    Normalisation is a per-step channel LayerNorm rather than the reference
    TCN's weight_norm: the Stage-1 EMA teacher is a ``copy.deepcopy`` of the
    encoder and averages it parameter by parameter, and weight_norm's
    reparametrisation supports neither.
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.norm1 = ChannelLayerNorm(out_channels)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.norm2 = ChannelLayerNorm(out_channels)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)
        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )
        self.out_act = nn.GELU()

    def forward(self, x):
        out = self.drop1(self.act1(self.norm1(self.chomp1(self.conv1(x)))))
        out = self.drop2(self.act2(self.norm2(self.chomp2(self.conv2(out)))))
        residual = x if self.downsample is None else self.downsample(x)
        return self.out_act(out + residual)


class RelationTCN(nn.Module):
    """Dilated causal TCN over relation inputs shaped [B, C_in, L].

    C_in stacks the relation roles (target, optional source) and the encoder
    input representations (delta_last, diff1, ...) as separate conv channels,
    so a single convolution mixes trajectory and local-dynamics features at
    every time step instead of after a flatten like the MLP encoder does.

    Convolutions are causal, which makes the ``last`` pooling read a state that
    has seen the whole window; ``mean`` pools over every position instead.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        kernel_size,
        dropout,
        pooling='last',
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f'relation_tcn_layers must be >= 1, got {num_layers}')
        if kernel_size < 2:
            raise ValueError(f'relation_tcn_kernel_size must be >= 2, got {kernel_size}')
        if pooling not in ('last', 'mean'):
            raise ValueError(f'Unsupported relation TCN pooling: {pooling}')

        self.in_channels = int(in_channels)
        self.kernel_size = int(kernel_size)
        self.num_layers = int(num_layers)
        self.pooling = pooling

        widths = [int(hidden_channels)] * (num_layers - 1) + [int(out_channels)]
        blocks = []
        prev = self.in_channels
        for level, width in enumerate(widths):
            blocks.append(TemporalBlock(
                in_channels=prev,
                out_channels=width,
                kernel_size=self.kernel_size,
                dilation=2 ** level,
                dropout=dropout,
            ))
            prev = width
        self.blocks = nn.Sequential(*blocks)

    @property
    def receptive_field(self):
        """Steps of history the pooled state can see (2 convs per block)."""
        return 1 + 2 * (self.kernel_size - 1) * (2 ** self.num_layers - 1)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f'relation TCN input must be [B, C, L], got {tuple(x.shape)}')
        if x.size(1) != self.in_channels:
            raise ValueError(
                f'expected {self.in_channels} relation TCN input channels, got {x.size(1)}'
            )
        h = self.blocks(x)
        if self.pooling == 'last':
            return h[..., -1]
        return h.mean(dim=-1)
