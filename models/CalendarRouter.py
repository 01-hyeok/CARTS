"""TRACK-V-CALENDAR-ROUTER01: the Calendar Router itself. Deliberately
tiny -- one `nn.Linear(6, 5)` PER CHANNEL, zero-initialized so routing
starts as exactly uniform (pi = [0.2]*5). Takes ONLY the 6-D cyclic
calendar feature (`utils/calendar_features.py`) -- never a time-series
tensor, never a learned date embedding, never an MLP/Transformer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

N_CALENDAR_FEATURES = 6


class CalendarRouter(nn.Module):
    def __init__(self, n_channels, n_heads=5):
        super().__init__()
        self.n_channels = n_channels
        self.n_heads = n_heads
        self.linear = nn.ModuleList([nn.Linear(N_CALENDAR_FEATURES, n_heads) for _ in range(n_channels)])
        for lin in self.linear:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)

    def logits(self, calendar_feat, channel):
        """calendar_feat: [B, 6]. Returns [B, n_heads] raw logits."""
        return self.linear[channel](calendar_feat)

    def forward(self, calendar_feat, channel):
        """Returns p_R(h|q,c): [B, n_heads] softmax probability."""
        return F.softmax(self.logits(calendar_feat, channel), dim=-1)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())
