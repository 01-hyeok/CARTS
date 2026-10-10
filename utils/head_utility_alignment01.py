"""TRACK-HEAD-UTILITY-ALIGNMENT01 -- read-only diagnostic primitives.
Computes, per query and per head (standalone Top-K within the P100
pool, reusing `per_head_standalone_topk` unmodified), three utilities:

  U_ind(h,q)   = mean_i MSE(aligned_future_i, Y_q)          (existing Hard winner criterion)
  U_agg(h,q)   = MSE(mean_i aligned_future_i, Y_q)          (actual Stage-2 retrieval-output quality)
  U_final(h,q) = MSE(B(q) + lambda_arm*(R_h(q)-B(q)), Y_q)  (future-aware oracle diagnostic, lambda_arm FROZEN)

All three are computed PER (query, channel) using the exact same
`compute_scores_pool_channel_first_grad` scoring path and
`gather_candidate_values`/`pooled_future_mse` alignment semantics
already used throughout this project's P100 code -- no new alignment
logic. U_ind/U_agg are then averaged across channels to produce one
scalar per query per head (matching this project's existing
per-query-scalar convention for Stage-2 MSE); U_final is computed once
per query directly in multi-channel space (since Stage-2's lambda
fusion is a single global scalar applied across all channels
simultaneously, not per-channel), using a FIXED head index across all
channels for that query (an explicit simplifying choice for this
oracle diagnostic, documented in each run's config.json).

Everything here runs under `torch.no_grad()` -- this track performs no
training and no parameter is ever required to carry a gradient.
"""
import torch

from models.RelationStage1 import stable_topk_indices
from scripts.train_margutil01 import memory_value
from utils.candidate_pool import gather_candidate_values, pooled_future_mse
from utils.expert_head_metrics import per_head_future_utility, per_head_standalone_topk
from utils.full_candidate_bank import compute_scores_pool_channel_first_grad

NUM_SLOTS = 5
TOP_K = 10


@torch.no_grad()
def per_query_head_quantities(model, slot_heads, base, lambda_arm, exp, args, channels, device, loader,
                              pool_cache, top_k=TOP_K):
    """One full pass over `loader`. Returns a dict of per-query tensors:
    `U_ind` [N,5], `U_agg` [N,5], `U_final` [N,5], `B_mse` [N] (Base
    alone MSE, for reference/sanity), `query_start_idx` [N] (for
    cross-split bookkeeping, same convention as the existing P100
    caches)."""
    all_u_ind, all_u_agg, all_u_final, all_b_mse, all_starts = [], [], [], [], []
    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        bsz = batch_x.size(0)
        u_ind_pc, u_agg_pc = [], []  # per-channel [B,5] lists, averaged over channels at the end
        r_h_per_channel = []  # for U_final: [C] list of [B,5,pred_len]
        for c in channels:
            scores, pool_valid_mask, pool_idx_global = compute_scores_pool_channel_first_grad(
                model, slot_heads, batch_x, exp, c, pool_cache, batch_start_idx, device)
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            pooled_memory_c = gather_candidate_values(memory_c, pool_idx_global)
            d_pool = pooled_future_mse(pooled_memory_c, offset_c, query_future)

            head_topk_idx = per_head_standalone_topk(scores, pool_valid_mask, k=top_k)  # [B,5,K] local idx
            u_ind_c = per_head_future_utility(head_topk_idx, d_pool)  # [B,5]
            bsz_c = head_topk_idx.size(0)
            flat_idx = head_topk_idx.reshape(bsz_c, -1)  # [B, 5*K]
            y_sel = pooled_memory_c.gather(1, flat_idx.unsqueeze(-1).expand(-1, -1, pooled_memory_c.size(-1)))
            y_sel = y_sel.reshape(bsz_c, NUM_SLOTS, top_k, -1) + offset_c.view(-1, 1, 1, 1)
            r_h_c = y_sel.mean(dim=2)  # [B,5,pred_len] -- head h's aggregate retrieval forecast, channel c
            u_agg_c = ((r_h_c - query_future.unsqueeze(1)) ** 2).mean(dim=-1)  # [B,5]

            u_ind_pc.append(u_ind_c)
            u_agg_pc.append(u_agg_c)
            r_h_per_channel.append(r_h_c)

        u_ind = torch.stack(u_ind_pc, dim=0).mean(dim=0)  # [B,5], channel-averaged
        u_agg = torch.stack(u_agg_pc, dim=0).mean(dim=0)  # [B,5], channel-averaged

        R_h = torch.stack(r_h_per_channel, dim=-1)  # [B,5,pred_len,C]
        b_q = base(batch_x) + batch_x[:, -1:, :].detach()  # [B,pred_len,C], same convention as train_r_stage2_lambda01.load_tensors
        y_q = batch_y  # [B,pred_len,C]
        b_mse = ((b_q - y_q) ** 2).mean(dim=(-1, -2))  # [B]
        y_hat_h = b_q.unsqueeze(1) + lambda_arm * (R_h - b_q.unsqueeze(1))  # [B,5,pred_len,C]
        u_final = ((y_hat_h - y_q.unsqueeze(1)) ** 2).mean(dim=(-1, -2))  # [B,5]

        all_u_ind.append(u_ind.cpu())
        all_u_agg.append(u_agg.cpu())
        all_u_final.append(u_final.cpu())
        all_b_mse.append(b_mse.cpu())
        all_starts.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx) else torch.as_tensor(batch_start_idx))

    return {
        'U_ind': torch.cat(all_u_ind), 'U_agg': torch.cat(all_u_agg), 'U_final': torch.cat(all_u_final),
        'B_mse': torch.cat(all_b_mse), 'query_start_idx': torch.cat(all_starts),
    }


def winners(U):
    """U: [N,5], lower=better. Returns argmin per query, [N]."""
    return U.argmin(dim=1)


def agreement(w1, w2):
    return float((w1 == w2).float().mean())


def confusion_matrix(w1, w2, num_slots=NUM_SLOTS):
    idx = w1 * num_slots + w2
    counts = torch.bincount(idx, minlength=num_slots * num_slots).float().reshape(num_slots, num_slots)
    return counts


def winner_fraction_stats(w, num_slots=NUM_SLOTS):
    counts = torch.bincount(w, minlength=num_slots).float()
    frac = counts / counts.sum().clamp_min(1)
    p = frac.clamp_min(1e-12)
    entropy = float(-(p * p.log()).sum())
    entropy_norm = entropy / float(torch.log(torch.tensor(float(num_slots))))
    return {'winner_fraction': frac.tolist(), 'normalized_winner_entropy': entropy_norm,
            'max_winner_fraction': float(frac.max()), 'min_winner_fraction': float(frac.min())}


def spearman_per_query(U1, U2):
    """U1, U2: [N,5]. Returns per-query Spearman rho (ties broken by
    stable rank order), [N]."""
    r1 = U1.argsort(dim=1).argsort(dim=1).float()
    r2 = U2.argsort(dim=1).argsort(dim=1).float()
    r1 = r1 - r1.mean(dim=1, keepdim=True)
    r2 = r2 - r2.mean(dim=1, keepdim=True)
    num = (r1 * r2).sum(dim=1)
    den = (r1.pow(2).sum(dim=1).sqrt() * r2.pow(2).sum(dim=1).sqrt()).clamp_min(1e-12)
    return num / den


def correlation_summary(rho):
    valid = rho[~torch.isnan(rho)]
    q = torch.quantile(valid, torch.tensor([0.25, 0.5, 0.75]))
    return {'mean': float(valid.mean()), 'median': float(q[1]), 'p25': float(q[0]), 'p75': float(q[2]),
            'negative_fraction': float((valid < 0).float().mean())}


def fixed_head_from_val(U_val):
    """Validation-only fixed-head selection -- mean utility across all
    validation queries, argmin head. NEVER uses test."""
    return int(U_val.mean(dim=0).argmin())


def oracle_vs_fixed_test(U_test, fixed_head):
    fixed_mse = float(U_test[:, fixed_head].mean())
    oracle_mse = float(U_test.min(dim=1).values.mean())
    gain_pct = (fixed_mse - oracle_mse) / max(fixed_mse, 1e-12) * 100.0
    return {'fixed_head': fixed_head, 'fixed_test_mse': fixed_mse, 'oracle_test_mse': oracle_mse,
            'oracle_gain_pct': gain_pct}


def closed_form_alpha(B, R, Y):
    """alpha* = argmin_alpha ||B + alpha(R-B) - Y||^2 = (d.e)/(d.d),
    d=R-B, e=Y-B, summed over all elements."""
    d = (R - B).reshape(-1)
    e = (Y - B).reshape(-1)
    denom = float((d * d).sum())
    if denom < 1e-12:
        return 0.0
    return float((d * e).sum() / denom)
