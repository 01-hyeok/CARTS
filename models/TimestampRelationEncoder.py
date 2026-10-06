"""TRACK-W-TIMESTAMP-FUSION01: per-timestep Value/Time fusion encoder.

New, additive module -- does not modify `models/RelationStage1.py`'s
`RelationEncoder` at all. See `research/W-timestamp-fusion/AUDIT.md`
PART 2 for why a literal "timestamp-off reduces to the exact old
RelationEncoder" implementation is not naturally expressible (the
existing `mlp` encoder flattens the raw scalar sequence straight into
its first Linear, with no per-timestep embedding stage to hook a value
-projection into), and for the resulting C0 capacity-matched-control
design this module is built to support.

    e_x(t) = P_x(x_t)                       P_x: Linear(1 -> d_proj)
    e_t(t) = P_t(c_t)                       P_t: Linear(K -> d_proj)
    h_t    = LayerNorm_fuse(e_x(t) + e_t(t))                  in R^{d_proj}
    H      = [h_1, ..., h_L]                                   in R^{B,L,d_proj}
    m      = Linear(L*d_proj, d_ff) -> GELU -> Dropout -> Linear(d_ff, d_model)
    z      = Proj(LayerNorm(m))             Proj: Linear(d_model,d_model)->GELU->Linear(d_model,d_model)

Query and candidate share every parameter (one module instance, called
twice) -- mirrors `RelationEncoder`'s own `encode_raw`-called-twice
convention exactly.

`c` (the timestamp branch's input) is ALWAYS a real tensor, never
`None` -- C0 ("NO-TIME") passes an all-zero `c`, keeping the module's
architecture/parameter-count/init-hash identical across C0/C1/C2 (see
AUDIT.md PART 2.3): `P_t(0) = bias_t`, a constant, query/candidate
-independent offset, so C0's time branch carries no VARYING signal even
though its parameters exist and receive (bias-only) gradient.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimestampFusionEncoder(nn.Module):
    def __init__(self, seq_len, d_model, d_ff, time_feat_dim, time_proj_dim=32,
                dropout=0.1, retrieval_similarity='cosine'):
        super().__init__()
        if retrieval_similarity not in ('cosine', 'l2'):
            raise ValueError(f'Unsupported retrieval_similarity: {retrieval_similarity}')
        self.retrieval_similarity = retrieval_similarity
        self.seq_len = int(seq_len)
        self.time_feat_dim = int(time_feat_dim)
        self.time_proj_dim = int(time_proj_dim)

        self.value_proj = nn.Linear(1, self.time_proj_dim)
        self.time_proj = nn.Linear(self.time_feat_dim, self.time_proj_dim)
        self.fuse_norm = nn.LayerNorm(self.time_proj_dim)

        self.main = nn.Sequential(
            nn.Linear(self.seq_len * self.time_proj_dim, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x, c):
        """x: [B, L] raw delta-last scalar sequence (one channel).
        c: [B, L, K] timestamp feature sequence -- REQUIRED, always a
        real tensor (zero-filled for the no-time control, never None)."""
        if x.dim() != 2 or x.size(1) != self.seq_len:
            raise ValueError(f'expected x=[B,{self.seq_len}], got {tuple(x.shape)}')
        if c.dim() != 3 or c.size(1) != self.seq_len or c.size(2) != self.time_feat_dim:
            raise ValueError(
                f'expected c=[B,{self.seq_len},{self.time_feat_dim}], got {tuple(c.shape)}')
        e_x = self.value_proj(x.unsqueeze(-1))   # [B, L, d_proj]
        e_t = self.time_proj(c)                  # [B, L, d_proj]
        h = self.fuse_norm(e_x + e_t)            # [B, L, d_proj]
        flat = h.reshape(h.size(0), -1)           # [B, L*d_proj]
        m = self.main(flat)                        # [B, d_model]
        z = self.proj(self.norm(m))
        normalized = z if self.retrieval_similarity == 'l2' else F.normalize(z, dim=-1)
        return normalized

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


def zero_time_features(x, time_feat_dim):
    """C0 ("NO-TIME") helper: an all-zero timestamp tensor of the right
    shape for a given value tensor x=[B,L]."""
    return x.new_zeros(x.size(0), x.size(1), time_feat_dim)
