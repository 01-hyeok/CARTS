"""TRACK-HEAD-UTILITY-ALIGNMENT-CORRECTION01 -- per-(query, channel)
granularity Individual/Aggregate head utilities, matching the EXACT
granularity the real Hard trainer (`train_hard_expert_v5_p100_allh01.py`)
uses for its winner decision (one winner per query PER CHANNEL, not one
winner per query averaged across channels, which is what
TRACK-HEAD-UTILITY-ALIGNMENT01 computed). Read-only; no training, no
backward, no checkpoint modification.
"""
import torch

from scripts.train_margutil01 import memory_value
from utils.candidate_pool import gather_candidate_values, pooled_future_mse
from utils.expert_head_metrics import per_head_future_utility, per_head_standalone_topk
from utils.full_candidate_bank import compute_scores_pool_channel_first_grad

NUM_SLOTS = 5
TOP_K = 10


@torch.no_grad()
def per_query_channel_head_quantities(model, slot_heads, exp, args, channels, device, loader, pool_cache,
                                      top_k=TOP_K):
    """One full pass. Returns `U_ind` [N,C,5] and `U_agg` [N,C,5] --
    PER CHANNEL, never averaged -- plus `query_start_idx` [N]."""
    all_u_ind, all_u_agg, all_starts = [], [], []
    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        u_ind_pc, u_agg_pc = [], []
        for c in channels:
            scores, pool_valid_mask, pool_idx_global = compute_scores_pool_channel_first_grad(
                model, slot_heads, batch_x, exp, c, pool_cache, batch_start_idx, device)
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            pooled_memory_c = gather_candidate_values(memory_c, pool_idx_global)
            d_pool = pooled_future_mse(pooled_memory_c, offset_c, query_future)

            head_topk_idx = per_head_standalone_topk(scores, pool_valid_mask, k=top_k)  # [B,5,K]
            u_ind_c = per_head_future_utility(head_topk_idx, d_pool)  # [B,5]

            bsz_c = head_topk_idx.size(0)
            flat_idx = head_topk_idx.reshape(bsz_c, -1)
            y_sel = pooled_memory_c.gather(1, flat_idx.unsqueeze(-1).expand(-1, -1, pooled_memory_c.size(-1)))
            y_sel = y_sel.reshape(bsz_c, NUM_SLOTS, top_k, -1) + offset_c.view(-1, 1, 1, 1)
            r_h_c = y_sel.mean(dim=2)  # [B,5,pred_len]
            u_agg_c = ((r_h_c - query_future.unsqueeze(1)) ** 2).mean(dim=-1)  # [B,5]

            u_ind_pc.append(u_ind_c)
            u_agg_pc.append(u_agg_c)

        all_u_ind.append(torch.stack(u_ind_pc, dim=1).cpu())  # [B,C,5]
        all_u_agg.append(torch.stack(u_agg_pc, dim=1).cpu())  # [B,C,5]
        all_starts.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx) else torch.as_tensor(batch_start_idx))

    return {
        'U_ind': torch.cat(all_u_ind), 'U_agg': torch.cat(all_u_agg),
        'query_start_idx': torch.cat(all_starts),
    }


def per_channel_winners(U):
    """U: [N,C,5]. Returns [N,C] winner per (query,channel). Uses
    `U.sort(dim=-1)[1][...,0]` -- NOT `argmin` -- to match
    `train_hard_expert_v5_p100_allh01.winner_margin_stats`'s EXACT
    tie-breaking operation bit-for-bit (PyTorch's `sort` and `argmin`
    are not guaranteed to agree on tie-breaking for exact ties,
    confirmed empirically: the only 4/224 mismatches found while
    building `test_critical_diagnostic_winner_matches_real_hard_trainer_exactly`
    all had margin exactly 0.0, i.e. multiple heads tied at the
    identical utility value)."""
    return U.sort(dim=-1)[1][..., 0]


def global_winners_from_channel_avg(U):
    """U: [N,C,5]. Channel-average then argmin -- the ORIGINAL
    (uncorrected) TRACK-HEAD-UTILITY-ALIGNMENT01 definition, kept here
    for the required Global-vs-Per-Channel comparison."""
    return U.mean(dim=1).argmin(dim=-1)


def margin_stats(U):
    """U: [N,C,5]. Returns per-(query,channel) margin (2nd-best minus
    best), flattened to [N*C], plus summary stats."""
    sorted_u, _ = U.sort(dim=-1)
    margin = (sorted_u[..., 1] - sorted_u[..., 0]).reshape(-1)
    scale = sorted_u[..., 0].abs().reshape(-1).clamp_min(1e-8)
    margin_norm = margin / scale
    return {
        'mean': float(margin.mean()), 'median': float(margin.median()),
        'near_tie_fraction': float((margin.abs() < 1e-4).float().mean()),
        'normalized_margin_mean': float(margin_norm.mean()),
        'normalized_margin_median': float(margin_norm.median()),
    }


def per_channel_fixed_head_from_val(U_val):
    """U_val: [N,C,5]. Returns [C] -- one fixed head index per channel,
    selected from VALIDATION mean utility only."""
    return U_val.mean(dim=0).argmin(dim=-1)  # [C]


def global_fixed_head_from_val(U_val):
    """U_val: [N,C,5], channel-averaged first. Returns scalar head index."""
    return int(U_val.mean(dim=(0, 1)).argmin())


def per_channel_oracle_vs_fixed_test(U_test, fixed_heads_per_channel):
    """U_test: [N,C,5]. fixed_heads_per_channel: [C]. Per-channel
    fixed uses a DIFFERENT head per channel (never test-selected);
    per-channel oracle picks the best head independently per
    (query,channel). Both are then averaged over channels for a
    per-query scalar, then averaged over queries for the reported MSE."""
    n, c, h = U_test.shape
    fixed_mse_per_channel = U_test.gather(-1, fixed_heads_per_channel.view(1, c, 1).expand(n, c, 1)).squeeze(-1)
    fixed_mse = float(fixed_mse_per_channel.mean())
    oracle_mse = float(U_test.min(dim=-1).values.mean())
    gain_pct = (fixed_mse - oracle_mse) / max(fixed_mse, 1e-12) * 100.0
    return {'fixed_mse': fixed_mse, 'oracle_mse': oracle_mse, 'oracle_gain_pct': gain_pct}


def global_oracle_vs_fixed_test(U_test, fixed_head):
    """U_test: [N,C,5]. Global oracle/fixed both apply ONE head
    uniformly across all channels of a query (channel-averaged U)."""
    U_avg = U_test.mean(dim=1)  # [N,5]
    fixed_mse = float(U_avg[:, fixed_head].mean())
    oracle_mse = float(U_avg.min(dim=-1).values.mean())
    gain_pct = (fixed_mse - oracle_mse) / max(fixed_mse, 1e-12) * 100.0
    return {'fixed_head': fixed_head, 'fixed_mse': fixed_mse, 'oracle_mse': oracle_mse, 'oracle_gain_pct': gain_pct}
