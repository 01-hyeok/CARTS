import torch
import torch.nn as nn


class RelationPatchEmbedding(nn.Module):
    """Patch/embed relation inputs shaped [B, R, L].

    R is 1 for self relation c<-c and 2 for cross relation c<-r. The first
    channel role is always target; the optional second role is source.
    """

    def __init__(self, seq_len, patch_len, stride, d_model, dropout=0.1):
        super().__init__()
        if patch_len <= 0 or stride <= 0:
            raise ValueError('patch_len and stride must be positive')
        if seq_len < patch_len:
            raise ValueError('seq_len must be >= patch_len')

        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (seq_len - patch_len) // stride + 1

        self.value_embedding = nn.Linear(patch_len, d_model)
        self.role_embedding = nn.Embedding(2, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, R, L] -> patches: [B, R, N, patch_len]
        if x.dim() != 3:
            raise ValueError(f'relation input must be [B, R, L], got {tuple(x.shape)}')
        bsz, roles, seq_len = x.shape
        if roles not in (1, 2):
            raise ValueError(f'relation role count must be 1 or 2, got {roles}')
        if seq_len != self.seq_len:
            raise ValueError(f'expected seq_len={self.seq_len}, got {seq_len}')

        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        tokens = self.value_embedding(patches)

        role_ids = torch.arange(roles, device=x.device)
        tokens = tokens + self.role_embedding(role_ids)[None, :, None, :]
        tokens = tokens + self.position_embedding[:, None, :, :]
        tokens = tokens.reshape(bsz, roles * self.num_patches, -1)
        return self.dropout(tokens)
