import torch
import torch.nn as nn


class RetrievalGate(nn.Module):
    def __init__(self, pred_len, hidden_dim=128, gate_mode='scalar', fusion_mode='residual', fixed_lambda=-1.0):
        super().__init__()
        if gate_mode not in ('scalar', 'horizon'):
            raise ValueError(f'Unsupported gate_mode: {gate_mode}')
        if fusion_mode not in ('residual', 'mixture', 'raft_concat'):
            raise ValueError(f'Unsupported fusion_mode: {fusion_mode}')
        self.gate_mode = gate_mode
        self.fusion_mode = fusion_mode
        self.fixed_lambda = float(fixed_lambda)
        out_dim = 1 if gate_mode == 'scalar' else pred_len
        self.net = nn.Sequential(
            nn.Linear(pred_len * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, y_base, y_ret):
        if self.fixed_lambda >= 0.0:
            out_dim = 1 if self.gate_mode == 'scalar' else y_base.size(-1)
            lam = y_base.new_full((y_base.size(0), out_dim), self.fixed_lambda)
        else:
            gate_feat = torch.cat([y_base, y_ret], dim=-1)
            lam = torch.sigmoid(self.net(gate_feat))
        if self.fusion_mode == 'raft_concat':
            raise RuntimeError('fusion_mode=raft_concat is handled by RelationStage2, not RetrievalGate')
        if self.fusion_mode == 'residual':
            y_final = y_base + lam * y_ret
        else:
            y_final = (1.0 - lam) * y_base + lam * y_ret
        return y_final, lam
