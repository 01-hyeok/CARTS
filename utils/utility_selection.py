"""Shared machinery for utility-aware candidate selection.

Every arm -- the existing binary classifier, the utility rankers, the residual
selector -- has to be scored on the *same* candidate pool, otherwise a better
number could just mean an easier pool. This builds that pool once per split and
hands every arm the identical tensors.

Leakage rule, enforced by construction:
  * query futures produce `utility` (a training target / oracle) and nothing else
  * candidate futures and candidate residuals are memory-side observables and
    may be used at inference
  * base forecasts come from `base_head(past)` only
"""

import torch

EPS = 1e-8


def utility_from_residuals(q_res, k_res, alpha=1.0):
    """U(q,k) = MSE(Y_q, base) - MSE(Y_q, base + alpha*R_k), in closed form.

    Expanding the two MSEs cancels the query-only term, leaving one matmul:
        U = 2*alpha*<R_q,R_k>/T - alpha^2*||R_k||^2/T
    """
    horizon = float(k_res.size(-1))
    return (
        2.0 * alpha * torch.matmul(q_res, k_res.transpose(0, 1)) / horizon
        - (alpha ** 2) * k_res.square().mean(-1).unsqueeze(0)
    )


@torch.no_grad()
def build_selection_cache(experiment, model, data, pool_m, alpha=1.0, chunk=256):
    """Pool indices, utilities and query embeddings for one split.

    Returns tensors indexed [query, channel, ...]. Candidate embeddings are NOT
    materialized -- they are gathered from the key bank by index at use time,
    which keeps a 500-wide pool over thousands of queries in memory.
    """
    device = experiment.device
    n_query = data['query_y'].size(0)
    channels = data['query_y'].size(-1)
    targets = [c for c in range(channels)
               if c in model.source_channels(c)]

    pool_idx = torch.zeros(n_query, len(targets), pool_m, dtype=torch.long)
    utility = torch.zeros(n_query, len(targets), pool_m)
    valid = torch.zeros(n_query, len(targets), pool_m, dtype=torch.bool)
    z_query = None
    rank_scores = torch.zeros(n_query, len(targets), pool_m)

    for start in range(0, n_query, chunk):
        stop = min(start + chunk, n_query)
        cand_mask, _ = experiment._candidate_mask(data['query_start'][start:stop])
        cand_mask = cand_mask.to(device)
        for slot_i, c in enumerate(targets):
            sources = model.source_channels(c)
            slot = sources.index(c)
            z_q = model._branch_embedding(data['query_x'][start:stop], c, c)
            if z_query is None:
                z_query = torch.zeros(n_query, len(targets), z_q.size(-1))
            z_query[start:stop, slot_i] = z_q.detach().cpu()

            bank = experiment.key_bank[c, slot].to(z_q.device, z_q.dtype)
            scores = torch.matmul(z_q, bank.transpose(0, 1)).masked_fill(
                ~cand_mask, float('-inf'))
            width = min(pool_m, bank.size(0))
            top = scores.topk(width, dim=-1)
            pool_idx[start:stop, slot_i, :width] = top.indices.cpu()
            rank_scores[start:stop, slot_i, :width] = top.values.cpu()
            valid[start:stop, slot_i, :width] = cand_mask.gather(
                1, top.indices).cpu()

            u = utility_from_residuals(
                data['query_residual'][start:stop, :, c],
                data['memory_residual'][:, :, c], alpha,
            ).gather(1, top.indices)
            utility[start:stop, slot_i, :width] = u.cpu()

    return {
        'targets': targets, 'pool_idx': pool_idx, 'utility': utility,
        'valid': valid, 'z_query': z_query, 'retriever_score': rank_scores,
        'alpha': alpha, 'pool_m': pool_m,
    }


def masked_utility(cache):
    """Utility with invalid pool slots pushed to -inf."""
    return cache['utility'].masked_fill(~cache['valid'], float('-inf'))


@torch.no_grad()
def selection_metrics(cache, scores, top_r=1):
    """How good is what a scorer picks, independent of the forecast.

    Reported against both ends of the achievable range: a random pick from the
    same pool, and the pool's own best candidate. Exact-identity accuracy is
    kept as a diagnostic only -- near-ties make it pessimistic even when the
    selected utility is essentially optimal.
    """
    utility = masked_utility(cache)
    valid = cache['valid']
    scores = scores.masked_fill(~valid, float('-inf'))
    width = min(top_r, scores.size(-1))
    picked = scores.topk(width, dim=-1).indices
    picked_u = utility.gather(-1, picked)
    # The "@1" figures always describe the single highest-scored candidate, so
    # they stay comparable across arms that aggregate different numbers of them.
    first = picked[..., :1]
    first_u = utility.gather(-1, first).squeeze(-1)

    oracle_best = utility.max(-1).values
    oracle_idx = utility.argmax(-1)
    random_u = (
        cache['utility'].masked_fill(~valid, 0.0).sum(-1)
        / valid.sum(-1).clamp_min(1)
    )
    selected_u = picked_u.mean(-1)
    finite = torch.isfinite(oracle_best) & torch.isfinite(selected_u)
    take = lambda t: float(t[finite].mean())
    return {
        'positive_at_1': take((first_u > 0).float()),
        'selected_utility_at_1': take(first_u),
        'selected_best_utility_at_1': take(picked_u.max(-1).values),
        'selected_utility_at_r': take(selected_u),
        'oracle_pool_utility': take(oracle_best),
        'random_utility': take(random_u),
        'utility_regret_at_1': take(oracle_best - first_u),
        'selection_recovery_at_1': take(
            (first_u - random_u) / (oracle_best - random_u + EPS)),
        'top1_identity_accuracy': take((first.squeeze(-1) == oracle_idx).float()),
    }


@torch.no_grad()
def forecast_from_selection(experiment, data, cache, scores, top_r=1,
                            weights=None, chunk=256):
    """base + alpha * aggregate(residuals of the top-r scored candidates).

    A query whose whole pool is invalid keeps its base forecast rather than
    being handed an arbitrary correction.
    """
    device = experiment.device
    alpha = cache['alpha']
    prediction = data['query_base'].clone()
    valid = cache['valid']
    scores = scores.masked_fill(~valid, float('-inf'))
    width = min(top_r, scores.size(-1))
    picked = scores.topk(width, dim=-1)

    for slot_i, c in enumerate(cache['targets']):
        k_res = data['memory_residual'][:, :, c]
        idx = cache['pool_idx'][:, slot_i].gather(1, picked.indices[:, slot_i])
        keep = valid[:, slot_i].gather(1, picked.indices[:, slot_i])
        for start in range(0, idx.size(0), chunk):
            stop = min(start + chunk, idx.size(0))
            residual = k_res[idx[start:stop].to(device)]        # [b, r, T]
            mask = keep[start:stop].to(device).float()
            if weights is None:
                w = mask
            else:
                w = weights[start:stop, slot_i].gather(
                    1, picked.indices[start:stop, slot_i].to(weights.device)
                ).to(device) * mask
            w = w / w.sum(-1, keepdim=True).clamp_min(EPS)
            correction = (residual * w.unsqueeze(-1)).sum(1)
            has = (mask.sum(-1, keepdim=True) > 0).float()
            prediction[start:stop, :, c] = (
                data['query_base'][start:stop, :, c] + alpha * correction * has
            )
    return prediction
