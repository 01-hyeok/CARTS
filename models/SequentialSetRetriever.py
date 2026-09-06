"""EXP-SEQFULL01: Full-Memory Set-Conditioned Sequential Retrieval.

Deliberately small and separate from RelationStage1.Model rather than added
to its already-large forward(): this pilot reuses that Model's `encoder`
(RelationEncoder) and `_relation_tensor` for embeddings, but the sequential,
teacher-forced training procedure is structurally different from every
existing Stage-1 loss mode (a single-shot full-memory score), so it gets its
own small module and its own training loop (scripts/train_seqfull01.py)
instead of another branch inside an already deeply-branched forward().

No shortlist anywhere: every step scores the full valid memory bank; only
already-selected and invalid candidates are masked out.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SetConditioner(nn.Module):
    """h_t = SetConditioner(q, m_{t-1}).

    Deliberately the smallest module the spec allows: concat -> Linear ->
    GELU -> Linear -> residual -> normalise. No attention, no cross-attention,
    no pairwise MLP -- those are explicitly excluded from this pilot.
    """

    def __init__(self, dim, hidden=None):
        super().__init__()
        hidden = hidden or dim
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, q, m):
        """q, m: [B, D] (both already the encoder's own embedding space).
        Returns h_t: [B, D], L2-normalised so its dot product with a
        similarly-normalised candidate embedding is a cosine similarity,
        matching the scoring convention every other arm in this project uses.
        """
        h = self.norm(q + self.net(torch.cat([q, m], dim=-1)))
        return F.normalize(h, dim=-1)


class EmptySetToken(nn.Module):
    """The set-state representation at t=1 (S_0 = empty set).

    Choice made and recorded here per the experiment spec's requirement to
    record it explicitly: a single LEARNED vector, not a zero vector. A zero
    vector is a fixed, out-of-distribution point the conditioner never sees
    again after step 1 and cannot calibrate against; a learned vector lets
    "nothing selected yet" sit wherever training finds useful in the same
    space as `m_{t-1}` for t>1, at the cost of one extra [D]-sized parameter
    -- the smallest possible increase in model complexity for that benefit.
    """

    def __init__(self, dim):
        super().__init__()
        self.empty = nn.Parameter(torch.zeros(dim))

    def forward(self, batch_size, device, dtype):
        return self.empty.to(device=device, dtype=dtype).unsqueeze(0).expand(batch_size, -1)


@torch.no_grad()
def _mean_selected(candidate_embeddings, selected_idx, step):
    """m_{t-1} = mean of the embeddings selected so far. candidate_embeddings:
    [N, D] (shared across the batch); selected_idx: [B, K] (only the first
    `step` columns are used, -1-padded columns beyond `step` are ignored).
    step=0 returns None (caller substitutes the learned empty token).
    """
    if step == 0:
        return None
    idx = selected_idx[:, :step]
    gathered = candidate_embeddings[idx]  # [B, step, D]
    return gathered.mean(dim=1)


def step_logits(h_t, candidate_embeddings, selected_mask, valid_mask):
    """One step's full-memory logits, masked.

    h_t: [B, D] (L2-normalised). candidate_embeddings: [N, D] (L2-normalised,
    the same live tensor reused across all K steps of this optimisation step).
    selected_mask: [B, N] bool, True where already chosen at an earlier step.
    valid_mask: [B, N] bool, True where the candidate is a legal choice at all.
    Returns [B, N] logits with selected/invalid positions at -inf.
    """
    scores = torch.matmul(h_t, candidate_embeddings.transpose(0, 1))
    neg = torch.finfo(scores.dtype).min / 4
    return scores.masked_fill(selected_mask | ~valid_mask, neg)
