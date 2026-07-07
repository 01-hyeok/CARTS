import torch
import torch.nn.functional as F


def retrieve_relation_future(z_q, z_mem, memory_value_c, valid_mask, top_k, tau_topk):
    """Retrieve target-channel futures with top-k cosine scores.

    Args:
        z_q: [B, D], normalized query embeddings.
        z_mem: [N, D], normalized memory key embeddings.
        memory_value_c: [N, H], target-channel future values.
        valid_mask: [B, N], True for non-leaking candidates.
    """
    bsz, num_memory = valid_mask.shape
    k = min(int(top_k), int(num_memory))
    if k <= 0:
        raise ValueError('top_k must be positive and memory bank must be non-empty')

    masked_fill = torch.finfo(z_q.dtype).min / 4
    scores = torch.matmul(z_q, z_mem.transpose(0, 1))
    scores = scores.masked_fill(~valid_mask, masked_fill)

    top_scores, top_idx = torch.topk(scores, k=k, dim=-1)
    top_valid = top_scores > masked_fill / 2
    v_top = memory_value_c[top_idx]

    scaled_scores = (top_scores / float(tau_topk)).masked_fill(~top_valid, masked_fill)
    alpha = F.softmax(scaled_scores, dim=-1) * top_valid.float()
    alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    retrieved = (alpha.unsqueeze(-1) * v_top).sum(dim=1)

    debug = {
        'scores': scores,
        'top_idx': top_idx,
        'top_scores': top_scores,
        'top_valid': top_valid,
        'v_top': v_top,
        'alpha': alpha,
        'top_k_effective': top_valid.float().sum(dim=-1),
    }
    return retrieved, alpha, top_idx, top_scores, debug
