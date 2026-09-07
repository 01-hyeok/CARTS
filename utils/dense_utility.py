"""EXP-MARGUTIL01: Full-Memory Set-Conditioned Dense Marginal Utility.

Shared, tested math used by the training script, the Stage-2/utility eval
script, and the unit tests -- one implementation, not three copies.

Reuses the exact closed-form incremental-weighted-mean trick already used by
`utils.oracle_intervention.select_greedy_weighted_set` (validated in
EXP-1/EXP-2/EXP-SEQFULL01): with `w_i = exp(s_i/tau)`, `Z_S = sum_{j in S} w_j`
and `M_S = sum_{j in S} w_j*Y_j`,

    A_weighted(S + {i}) = MSE( (M_S + w_i*Y_i) / (Z_S + w_i), Y_q )

which is *exactly* the same `err` tensor `select_greedy_weighted_set` computes
and argmins at every step -- this module returns the whole `A_i` row (dense),
not just its argmin, chunked over the candidate dimension so it scales to a
full memory bank (8449-36696 candidates) without a shortlist.
"""
import torch


@torch.no_grad()
def prefix_weighted_sums(prefix_idx, w, futures):
    """Z_S, M_S for an oracle/model prefix `prefix_idx` [B, t-1] (t-1 may be 0).

    w: [B, N] candidate weights (already 0 at invalid positions).
    futures: [B, N, H] candidate futures (already offset-adjusted).
    Returns (Z_S [B, 1], M_S [B, H]).
    """
    bsz, _, h = futures.shape
    if prefix_idx.size(1) == 0:
        return (futures.new_zeros(bsz, 1), futures.new_zeros(bsz, h))
    w_sel = w.gather(1, prefix_idx)
    y_sel = futures.gather(1, prefix_idx.unsqueeze(-1).expand(-1, -1, h))
    z_s = w_sel.sum(dim=-1, keepdim=True)
    m_s = (w_sel.unsqueeze(-1) * y_sel).sum(dim=1)
    return z_s, m_s


def dense_utility(prefix_idx, w, futures, query_future, chunk_size=None, eps=1e-12):
    """A_i = MSE(aggregate(S_prefix + {i}), Y_q) for EVERY candidate i, full
    memory, chunked over the candidate dimension (chunking changes nothing
    about the candidate support -- every candidate is still scored, just not
    all in one tensor). Gradient-safe: does not require grad-tracking through
    `prefix_idx`'s own selection (the prefix is a fixed oracle/model sequence
    at this call site, never differentiated through).

    Returns raw A [B, N] (caller masks invalid/already-selected positions and
    negates to get utility `u = -A`).
    """
    with torch.no_grad():
        z_s, m_s = prefix_weighted_sums(prefix_idx, w, futures)
    bsz, n, h = futures.shape
    chunk_size = chunk_size or n
    out = futures.new_empty(bsz, n)
    q = query_future.unsqueeze(1)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        w_c = w[:, start:end]
        y_c = futures[:, start:end, :]
        trial_num = m_s.unsqueeze(1) + w_c.unsqueeze(-1) * y_c
        trial_den = (z_s + w_c).unsqueeze(-1).clamp_min(eps)
        y_ret = trial_num / trial_den
        out[:, start:end] = (y_ret - q).pow(2).mean(dim=-1)
    return out


def candidate_weights(scores, valid_mask, tau, eps=0.0):
    """w_i = exp(s_i/tau), row-max-shifted for numerical stability (matches
    `select_greedy_weighted_set`'s own convention exactly), zeroed at invalid
    positions."""
    s = scores.float().masked_fill(~valid_mask, float('-inf'))
    e = torch.exp((s - s.max(dim=-1, keepdim=True).values) / float(tau))
    return torch.where(valid_mask, e, torch.zeros_like(e))


def normalize_utility(u, valid_mask, eps=1e-6):
    """Per-query, per-step z-score over the VALID (not yet selected, legal)
    candidates only -- invalid/selected positions never enter mean/std and
    are excluded from the loss by the caller's own mask, not by this
    function (this only returns the normalised tensor, unmasked positions
    are still finite but meaningless)."""
    valid_f = valid_mask.float()
    count = valid_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (u * valid_f).sum(dim=-1, keepdim=True) / count
    var = ((u - mean).pow(2) * valid_f).sum(dim=-1, keepdim=True) / count
    std = var.clamp_min(0.0).sqrt()
    return (u - mean) / (std + eps)
