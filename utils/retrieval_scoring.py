"""Similarities and Recall@K against a fixed raw-future-MSE oracle."""

import torch

SIMILARITIES = ('cosine', 'pearson', 'mse')


def similarity_scores(query, memory, similarity, eps=1e-8):
    """Score every candidate for every query. Higher is better, always.

    query:  [B, D]
    memory: [N, D]
    returns [B, N]

    cosine  : cos(q, k)
    pearson : cos on mean-centred vectors, i.e. the Pearson correlation
    mse     : -MSE(q, k). Ranking-equivalent to -||q-k||^2, kept as the mean so
              the scale does not depend on D.
    """
    if similarity == 'pearson':
        query = query - query.mean(dim=-1, keepdim=True)
        memory = memory - memory.mean(dim=-1, keepdim=True)
        similarity = 'cosine'
    if similarity == 'cosine':
        q = torch.nn.functional.normalize(query, dim=-1, eps=eps)
        k = torch.nn.functional.normalize(memory, dim=-1, eps=eps)
        return q @ k.transpose(0, 1)
    if similarity == 'mse':
        d = query.size(-1)
        q2 = (query ** 2).mean(dim=-1, keepdim=True)
        k2 = (memory ** 2).mean(dim=-1).unsqueeze(0)
        qk = (query @ memory.transpose(0, 1)) / d
        return -(q2 + k2 - 2.0 * qk).clamp_min(0.0)
    raise ValueError(f'Unsupported similarity: {similarity}')


def zscore_rows(scores, valid_mask, eps=1e-8):
    """Per-query z-score over valid candidates only."""
    mask = valid_mask.float()
    count = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (scores * mask).sum(dim=-1, keepdim=True) / count
    var = (((scores - mean) ** 2) * mask).sum(dim=-1, keepdim=True) / count
    return (scores - mean) / var.sqrt().clamp_min(eps)


def fuse_decomposition_scores(part_scores, valid_mask):
    """s = mean_j z(s_j). For trend/seasonal this is (1/2)z(s^T) + (1/2)z(s^S).

    Each part is z-scored per query before averaging, so a part whose raw scores
    happen to spread wider cannot dominate the fusion.
    """
    stacked = torch.stack(
        [zscore_rows(s, valid_mask) for s in part_scores], dim=0
    )
    return stacked.mean(dim=0)


def raw_future_mse(query_future, memory_future):
    """Oracle distance: MSE between raw futures. [B, H] x [N, H] -> [B, N].

    This never sees the representation, the encoder or the similarity, which is
    the fairness condition the study depends on.
    """
    h = query_future.size(-1)
    q2 = (query_future ** 2).mean(dim=-1, keepdim=True)
    k2 = (memory_future ** 2).mean(dim=-1).unsqueeze(0)
    qk = (query_future @ memory_future.transpose(0, 1)) / h
    return (q2 + k2 - 2.0 * qk).clamp_min(0.0)


def recall_at_k(scores, oracle_distance, valid_mask, ks=(1, 5, 10)):
    """|retrieved top-k ∩ oracle top-k| / k, averaged over queries.

    Invalid candidates are pushed to the losing end of both rankings so they can
    never enter either top-k.
    """
    neg_inf = torch.finfo(scores.dtype).min / 4
    pos_inf = torch.finfo(oracle_distance.dtype).max / 4
    masked_scores = scores.masked_fill(~valid_mask, neg_inf)
    masked_oracle = oracle_distance.masked_fill(~valid_mask, pos_inf)

    valid_counts = valid_mask.sum(dim=-1)
    out = {}
    for k in ks:
        effective = int(min(k, scores.size(-1)))
        retrieved = masked_scores.topk(effective, dim=-1, largest=True).indices
        oracle = masked_oracle.topk(effective, dim=-1, largest=False).indices
        hit = (retrieved.unsqueeze(-1) == oracle.unsqueeze(-2)).any(dim=-1)
        # A query with fewer valid candidates than k would inflate the ratio.
        usable = (valid_counts >= effective)
        per_query = hit.float().sum(dim=-1) / float(effective)
        out[k] = per_query[usable]
    return out
