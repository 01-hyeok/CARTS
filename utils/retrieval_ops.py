import torch
import torch.nn.functional as F


def retrieve_relation_future(z_q, z_mem, memory_value_c, valid_mask, top_k, tau_topk,
                             similarity='cosine', soft_all=False, score_fn=None):
    """Retrieve target-channel futures by similarity over the candidate bank.

    Args:
        z_q: [B, D], query embeddings.
        z_mem: [N, D], memory key embeddings.
        memory_value_c: [N, H], target-channel future values.
        valid_mask: [B, N], True for non-leaking candidates.
        similarity: 'cosine' scores with a dot product and expects both sides to
            be L2-normalised. 'l2' scores with the negative mean squared
            distance and expects them *not* to be normalised - on normalised
            vectors -||q-k||^2 = 2<q,k> - 2, a monotone map of the dot product,
            so the two would return an identical Top-K. Dividing by D keeps the
            l2 scores on a comparable scale to cosine so tau_topk still applies.
        soft_all: weight *every* valid candidate with softmax(scores/tau_topk)
            instead of taking a Top-K first. Top-K picks indices, which is not
            differentiable, so the forecasting loss can currently only reshape
            the weights over candidates that were already selected. Weighting the
            whole bank puts every candidate score on the gradient path, which is
            what makes a single end-to-end loss able to train retrieval.

            tau_topk matters far more here: the softmax runs over N candidates
            rather than k, so a temperature tuned for Top-K leaves the weights
            spread over hundreds of them. Top-K indices are still computed for
            the recall and oracle diagnostics.
    """
    bsz, num_memory = valid_mask.shape
    k = min(int(top_k), int(num_memory))
    if k <= 0:
        raise ValueError('top_k must be positive and memory bank must be non-empty')

    masked_fill = torch.finfo(z_q.dtype).min / 4
    if score_fn is not None:
        # A learned comparison in place of the fixed dot product. Stage-1 can be
        # trained with one, and without this the retriever it produced would be
        # read back through cosine at retrieval time -- scoring the embeddings
        # with a different function than the one they were shaped for.
        scores = score_fn(z_q, z_mem)
        if scores.shape != valid_mask.shape:
            raise ValueError(
                f'score_fn returned {tuple(scores.shape)}, expected {tuple(valid_mask.shape)}'
            )
    elif similarity == 'cosine':
        scores = torch.matmul(z_q, z_mem.transpose(0, 1))
    elif similarity == 'l2':
        # Un-normalised keys are stored in half for the Chronos bank; squaring
        # them there overflows, so the distance is always taken in float32.
        z_q = z_q.float()
        z_mem = z_mem.float()
        dim = float(z_q.size(-1))
        q_sq = z_q.pow(2).sum(dim=-1, keepdim=True)
        k_sq = z_mem.pow(2).sum(dim=-1).unsqueeze(0)
        scores = -(q_sq + k_sq - 2.0 * torch.matmul(z_q, z_mem.transpose(0, 1))) / dim
    else:
        raise ValueError(f'Unsupported retrieval similarity: {similarity}')
    scores = scores.masked_fill(~valid_mask, masked_fill)

    # Kept in both modes: recall@k and the oracle metrics are defined on the
    # Top-K set, and soft_all still needs them to stay comparable.
    top_scores, top_idx = torch.topk(scores, k=k, dim=-1)
    top_valid = top_scores > masked_fill / 2
    v_top = memory_value_c[top_idx]

    if soft_all:
        alpha_all = F.softmax(scores / float(tau_topk), dim=-1) * valid_mask.float()
        alpha_all = alpha_all / alpha_all.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        retrieved = torch.matmul(alpha_all, memory_value_c.to(alpha_all.dtype))
        # exp(H(alpha)) over the whole bank: how many candidates the weighting
        # actually keeps. Top-K reports the same quantity over its k entries.
        eff = torch.exp(-(alpha_all * torch.log(alpha_all + 1e-12)).sum(dim=-1))
        alpha = alpha_all.gather(-1, top_idx)      # diagnostics only, not renormalised
    else:
        scaled_scores = (top_scores / float(tau_topk)).masked_fill(~top_valid, masked_fill)
        alpha = F.softmax(scaled_scores, dim=-1) * top_valid.float()
        alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        retrieved = (alpha.unsqueeze(-1) * v_top).sum(dim=1)
        eff = top_valid.float().sum(dim=-1)

    debug = {
        'scores': scores,
        'top_idx': top_idx,
        'top_scores': top_scores,
        'top_valid': top_valid,
        'v_top': v_top,
        'alpha': alpha,
        'top_k_effective': eff,
    }
    return retrieved, alpha, top_idx, top_scores, debug


def reweight_selected_candidates(z_q, z_k_sel, values, top_valid, tau_topk,
                                 similarity='cosine', score_fn=None):
    """Recompute Top-K scores and weights with both sides differentiable.

    `retrieve_relation_future` scores the query against a precomputed key bank,
    so a forecast loss can only reach the query encoder. Re-encoding the selected
    candidates with the live encoder and rescoring here puts the candidate side on
    the gradient path too, without touching which candidates were selected --
    Top-K stays exactly as chosen, and the retrieval universe stays the full bank.

    Args:
        z_q: [B, D] query embeddings, gradient on.
        z_k_sel: [B, K, D] embeddings of the already-selected candidates.
        values: [B, K, H] their target-channel futures.
        top_valid: [B, K] which selected slots are real.
    Returns (retrieved [B, H], alpha [B, K], top_scores [B, K]).
    """
    if z_k_sel.dim() != 3 or z_k_sel.size(0) != z_q.size(0):
        raise ValueError(
            f'z_k_sel must be [B, K, D] with B={z_q.size(0)}, got {tuple(z_k_sel.shape)}'
        )
    if values.shape[:2] != z_k_sel.shape[:2]:
        raise ValueError(
            f'values {tuple(values.shape)} and z_k_sel {tuple(z_k_sel.shape)} disagree on [B, K]'
        )
    masked_fill = torch.finfo(z_q.dtype).min / 4
    z_k_sel = z_k_sel.to(z_q.dtype)
    if score_fn is not None:
        # Same learned comparison as selection used, so the end-to-end gradient
        # reshapes the weights under the function that chose them.
        top_scores = score_fn(z_q, z_k_sel)
        if top_scores.shape != top_valid.shape:
            raise ValueError(
                f'score_fn returned {tuple(top_scores.shape)}, expected {tuple(top_valid.shape)}'
            )
    elif similarity == 'cosine':
        top_scores = (z_q.unsqueeze(1) * z_k_sel).sum(dim=-1)
    elif similarity == 'l2':
        dim = float(z_q.size(-1))
        top_scores = -(
            z_q.float().pow(2).sum(-1, keepdim=True)
            + z_k_sel.float().pow(2).sum(-1)
            - 2.0 * (z_q.float().unsqueeze(1) * z_k_sel.float()).sum(-1)
        ) / dim
    else:
        raise ValueError(f'Unsupported retrieval similarity: {similarity}')

    scaled = (top_scores / float(tau_topk)).masked_fill(~top_valid, masked_fill)
    alpha = F.softmax(scaled, dim=-1) * top_valid.float()
    alpha = alpha / alpha.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    retrieved = (alpha.unsqueeze(-1) * values.to(alpha.dtype)).sum(dim=1)
    return retrieved, alpha, top_scores
