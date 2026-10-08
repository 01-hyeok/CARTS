"""TRACK-HARD-EXPERT-V5-P100-ALLH01 -- Shared-Top-100 (P100) pool-aware
counterparts of `utils/expert_head_metrics.py`'s Full-memory-only
pieces. Every function here operates on a pool-restricted score/distance
tensor (`[B, S, M]` / `[B, M]`, M=pool size, LOCAL indices 0..M-1) rather
than the full candidate bank (`[B, S, N]` / `[B, N]`, GLOBAL indices).

The following primitives from `utils/expert_head_metrics.py` are
candidate-space-size-agnostic and are REUSED UNMODIFIED (imported, never
copied) because they only ever index into whatever last-dimension-sized
tensor they are handed: `per_head_standalone_topk`, `per_head_future_utility`,
`responsibility_from_utility`, `kl_per_head`, `expert_weighted_loss`,
`hard_expert_loss`, `winner_margin_stats`, `head_pairwise_overlap`,
`per_head_entropy_and_spread`.

Only `per_head_decomposition` and `evaluate_and_save_head_report` are
genuinely Full-memory-specific (they index `memory_c[idx_h]` by GLOBAL
id and recompute a FULL-candidate oracle) -- this module's
`per_head_decomposition_pool` / `evaluate_and_save_head_report_pool` are
the P100-restricted counterparts.

`d_pool` here is ALWAYS produced by `utils.candidate_pool.pooled_future_mse`
(already a correctly-signed, non-negative, lower-is-better MSE distance)
and passed DIRECTLY (never negated) to `normalized_teacher_prob` --
the exact fix from TRACK-V-SHARED-TOP100-SIGNFIX01. There is no
`-d_pool` anywhere in this module.
"""
import json
from pathlib import Path

import torch

from models.RelationStage1 import stable_topk_indices
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k
from utils.expert_head_metrics import (
    head_pairwise_overlap, kl_per_head, per_head_entropy_and_spread, per_head_future_utility,
    per_head_standalone_topk, responsibility_from_utility, winner_margin_stats,
)
from utils.mean_mixture_selection import mean_mixture_topk_selection


def per_head_decomposition_pool(head_topk_idx, pooled_memory_c, offset_c, query_future, d_pool, oracle_idx_pool,
                                top_k=10):
    """Pool-aware counterpart of `per_head_decomposition` -- `head_topk_idx`
    (and `oracle_idx_pool`) are LOCAL indices into the M-sized pool
    dimension, so values are gathered from `pooled_memory_c`/`d_pool`
    (both already pool-restricted, per-query) via `.gather`, NEVER via
    `memory_c[idx_h]` (that pattern assumes GLOBAL ids into a shared
    [N,...] bank and would silently mis-index here)."""
    bsz, s, k = head_topk_idx.shape
    out = {name: torch.zeros(bsz, s, device=d_pool.device)
          for name in ('retmse10', 'agg_mse', 'D', 'C', 'recall10', 'ndcg10')}
    pool_valid_mask = torch.ones(bsz, pooled_memory_c.size(1), dtype=torch.bool, device=d_pool.device)
    for h in range(s):
        idx_h = head_topk_idx[:, h, :]
        ind_mse_i = d_pool.gather(1, idx_h)
        y_sel = pooled_memory_c.gather(1, idx_h.unsqueeze(-1).expand(-1, -1, pooled_memory_c.size(-1))) \
            + offset_c.view(-1, 1, 1)
        agg_pred = y_sel.mean(dim=1)
        agg_mse_h = ((agg_pred - query_future) ** 2).mean(dim=-1)
        D_h = ind_mse_i.mean(dim=-1) / top_k
        C_h = agg_mse_h - D_h
        out['retmse10'][:, h] = ind_mse_i.mean(dim=-1)
        out['agg_mse'][:, h] = agg_mse_h
        out['D'][:, h] = D_h
        out['C'][:, h] = C_h
        out['recall10'][:, h] = recall_at_k(idx_h, oracle_idx_pool, top_k)
        out['ndcg10'][:, h] = ndcg_at_k(idx_h, d_pool, pool_valid_mask, top_k)
    return out


def full_head_diagnostics_pool(scores, pooled_memory_c, offset_c, query_future, d_pool, p_t, tau_s, top_k,
                               tau_e=1.0):
    """Pool-aware counterpart of `full_head_diagnostics` -- same
    structure, `d_raw`-sized-N quantities replaced by pool-sized-M ones
    throughout. `scores`: [B,S,M] from the pool-restricted scorer."""
    from scripts.train_t_pure_multislot01 import spearman_batch
    bsz, s, m = scores.shape
    pool_valid_mask = torch.ones(bsz, m, dtype=torch.bool, device=scores.device)
    with torch.no_grad():
        head_topk_idx = per_head_standalone_topk(scores, pool_valid_mask, k=top_k)
        U = per_head_future_utility(head_topk_idx, d_pool)
        responsibility = responsibility_from_utility(U, tau_e=tau_e)
        oracle_idx_pool = stable_topk_indices(d_pool, top_k, largest=False)
        decomp = per_head_decomposition_pool(head_topk_idx, pooled_memory_c, offset_c, query_future, d_pool,
                                             oracle_idx_pool, top_k)
        winner, u_best, u_second, margin = winner_margin_stats(U)
        overlaps, union_size = head_pairwise_overlap(head_topk_idx)
    kl_vals, p_h = kl_per_head(p_t, scores, pool_valid_mask, tau_s)
    with torch.no_grad():
        entropy, spread = per_head_entropy_and_spread(p_h, pool_valid_mask)
        spearman_h = [spearman_batch(scores[:, h, :], d_pool, pool_valid_mask) for h in range(s)]
    return dict(U=U, responsibility=responsibility, decomp=decomp, winner=winner, margin=margin,
               overlaps=overlaps, union_size=union_size, kl=kl_vals.detach(), entropy=entropy, spread=spread,
               spearman_h=spearman_h, oracle_idx=oracle_idx_pool)


def evaluate_and_save_head_report_pool(model, slot_heads, exp, args, cli, channels, device, train_loader,
                                       val_loader, test_loader, pool_caches, out_dir, best_epoch, num_slots=5,
                                       tau_e=1.0):
    """Pool-aware counterpart of `evaluate_and_save_head_report` --
    IDENTICAL output-file layout/semantics, restricted throughout to
    the precomputed Shared-Top-100 pool (`pool_caches[split]`, a
    `CandidatePoolCache` from `scripts/train_retriever_pool01.py`,
    produced by the bug-independent `precompute_candidate_pool01.py`).
    Oracle/fixed-head comparisons here are RESTRICTED-pool oracle (the
    best achievable within the same 100 candidates every arm sees),
    NOT the Full-memory oracle -- callers wanting a Full-vs-P100 oracle
    degradation number must compute the Full oracle separately (see
    this track's `compute_p100_coverage_pool01.py`)."""
    from scripts.train_margutil01 import memory_value
    from utils.candidate_pool import gather_candidate_values, pooled_future_mse
    from utils.full_candidate_bank import compute_scores_pool_channel_first_grad

    model.eval()

    def full_pass(loader, split):
        rr_sums, n = {}, 0
        head_decomp_sums = {name: torch.zeros(num_slots) for name in
                            ('retmse10', 'agg_mse', 'D', 'C', 'recall10', 'ndcg10')}
        head_kl_sum = torch.zeros(num_slots)
        head_U_sum = torch.zeros(num_slots)
        head_resp_sum = torch.zeros(num_slots)
        head_entropy_sum = torch.zeros(num_slots)
        head_spread_sum = torch.zeros(num_slots)
        head_spearman_sum = torch.zeros(num_slots)
        win_count = torch.zeros(num_slots)
        overlap_sum = {}
        union_sum = 0.0
        margin_vals = []
        mean_ret_sum, mean_agg_sum, mean_recall_sum, mean_ndcg_sum = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch_x, batch_y, batch_start_idx in loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                bsz = batch_x.size(0)
                per_ch_rr = {}
                for c in channels:
                    scores, pool_valid_mask, pool_idx_global = compute_scores_pool_channel_first_grad(
                        model, slot_heads, batch_x, exp, c, pool_caches[split], batch_start_idx, device)
                    memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
                    query_future = batch_y[:, :, c]
                    pooled_memory_c = gather_candidate_values(memory_c, pool_idx_global)
                    d_pool = pooled_future_mse(pooled_memory_c, offset_c, query_future)
                    from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob
                    p_t = normalized_teacher_prob(d_pool, pool_valid_mask, cli.tau_t)

                    mean_idx, _, _ = mean_mixture_topk_selection(scores, pool_valid_mask, cli.tau_s, k=cli.top_k)
                    mean_ret = d_pool.gather(1, mean_idx)
                    y_sel_mean = pooled_memory_c.gather(
                        1, mean_idx.unsqueeze(-1).expand(-1, -1, pooled_memory_c.size(-1))) + offset_c.view(-1, 1, 1)
                    mean_agg = ((y_sel_mean.mean(dim=1) - query_future) ** 2).mean(dim=-1)
                    oracle_idx_pool = stable_topk_indices(d_pool, cli.top_k, largest=False)
                    mean_recall = recall_at_k(mean_idx, oracle_idx_pool, cli.top_k)
                    mean_ndcg = ndcg_at_k(mean_idx, d_pool, pool_valid_mask, cli.top_k)
                    mean_ret_sum += float(mean_ret.mean(dim=-1).sum())
                    mean_agg_sum += float(mean_agg.sum())
                    mean_recall_sum += float(mean_recall.sum())
                    mean_ndcg_sum += float(mean_ndcg.sum())

                    diag = full_head_diagnostics_pool(scores, pooled_memory_c, offset_c, query_future, d_pool, p_t,
                                                      cli.tau_s, cli.top_k, tau_e=tau_e)
                    for name in head_decomp_sums:
                        head_decomp_sums[name] += diag['decomp'][name].sum(dim=0).cpu()
                    head_kl_sum += diag['kl'].sum(dim=0).cpu()
                    head_U_sum += diag['U'].sum(dim=0).cpu()
                    head_resp_sum += diag['responsibility'].sum(dim=0).cpu()
                    head_entropy_sum += diag['entropy'].sum(dim=0).cpu()
                    head_spread_sum += diag['spread'].sum(dim=0).cpu()
                    head_spearman_sum += torch.tensor(diag['spearman_h'])
                    win_count += torch.nn.functional.one_hot(diag['winner'], num_slots).float().sum(dim=0).cpu()
                    for pair, v in diag['overlaps'].items():
                        overlap_sum[pair] = overlap_sum.get(pair, 0.0) + float(v.sum())
                    union_sum += float(diag['union_size'].sum())
                    margin_vals.append(diag['margin'].cpu())
                n += bsz
        denom = max(n * len(channels), 1)
        return {
            'rr': {'retmse10': mean_ret_sum / denom, 'agg_mse10': mean_agg_sum / denom,
                  'recall10': mean_recall_sum / denom, 'ndcg10': mean_ndcg_sum / denom},
            'head_decomp': {name: (t / denom).tolist() for name, t in head_decomp_sums.items()},
            'head_kl': (head_kl_sum / denom).tolist(),
            'head_U': (head_U_sum / denom).tolist(),
            'head_responsibility': (head_resp_sum / denom).tolist(),
            'head_entropy': (head_entropy_sum / denom).tolist(),
            'head_spread': (head_spread_sum / denom).tolist(),
            'head_spearman': (head_spearman_sum / denom).tolist(),
            'win_count': win_count.tolist(),
            'win_fraction': (win_count / denom).tolist(),
            'overlap_matrix_mean': {f'H{p[0]+1}-H{p[1]+1}': v / denom for p, v in overlap_sum.items()},
            'union_size_mean': union_sum / denom,
            'margin': torch.cat(margin_vals),
            'n_queries_seen': n,
        }

    train_full = full_pass(train_loader, 'train')
    val_full = full_pass(val_loader, 'val')
    test_full = full_pass(test_loader, 'test')
    final_test_metrics = {**test_full['rr'], 'n_queries_seen': test_full['n_queries_seen'], 'best_epoch': best_epoch}

    def margin_stats(margin):
        q = torch.quantile(margin, torch.tensor([0.5, 0.9]))
        return {'mean': float(margin.mean()), 'median': float(q[0]), 'p90': float(q[1]),
               'near_tie_fraction': float((margin.abs() < 1e-4).float().mean())}

    val_head_retmse = torch.tensor(val_full['head_decomp']['retmse10'])
    h_fixed = int(val_head_retmse.argmin())
    fixed_retmse_test = test_full['head_decomp']['retmse10'][h_fixed]
    oracle_test_mean_U = test_full['head_U']  # per-head mean U, not per-query min -- see oracle_per_query below

    def oracle_per_query_mean(loader):
        from utils.candidate_pool import gather_candidate_values, pooled_future_mse
        total, n = 0.0, 0
        with torch.no_grad():
            for batch_x, batch_y, batch_start_idx in loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                bsz = batch_x.size(0)
                for c in channels:
                    scores, pool_valid_mask, pool_idx_global = compute_scores_pool_channel_first_grad(
                        model, slot_heads, batch_x, exp, c, pool_caches['test'], batch_start_idx, device)
                    memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
                    query_future = batch_y[:, :, c]
                    pooled_memory_c = gather_candidate_values(memory_c, pool_idx_global)
                    d_pool = pooled_future_mse(pooled_memory_c, offset_c, query_future)
                    head_topk_idx = per_head_standalone_topk(scores, pool_valid_mask, k=cli.top_k)
                    U = per_head_future_utility(head_topk_idx, d_pool)
                    total += U.min(dim=1).values.sum().item()
                n += bsz * len(channels)
        return total / max(n, 1)

    oracle_test = oracle_per_query_mean(test_loader)
    oracle_vs_fixed = {
        'fixed_head': f'H{h_fixed+1}', 'fixed_retmse10_test': fixed_retmse_test,
        'oracle_retmse10_test_per_query_mean': oracle_test,
        'oracle_gain_pct': (fixed_retmse_test - oracle_test) / max(fixed_retmse_test, 1e-9) * 100.0,
        'note': 'RESTRICTED-POOL oracle (best of the same 100 candidates every arm sees), '
               'oracle is argmin_h U_h(q) applied per query, not the column-wise min of the mean table.',
    }

    out_dir = Path(out_dir) if not hasattr(out_dir, 'mkdir') else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'final_test_metrics.json').write_text(json.dumps(final_test_metrics, indent=2, default=str))
    for split_name, full in (('train', train_full), ('val', val_full), ('test', test_full)):
        (out_dir / f'per_head_{split_name}_metrics.json').write_text(json.dumps(
            {k_: v for k_, v in full.items() if k_ != 'margin'}, indent=2, default=str))
    (out_dir / 'per_head_kl.json').write_text(json.dumps({'test': test_full['head_kl']}, indent=2))
    (out_dir / 'per_head_future_utility.json').write_text(json.dumps({'test': test_full['head_U']}, indent=2))
    (out_dir / 'responsibility_metrics.json').write_text(json.dumps(
        {'test_mean_responsibility': test_full['head_responsibility']}, indent=2))
    (out_dir / 'winner_fraction.json').write_text(json.dumps(
        {'train': train_full['win_fraction'], 'val': val_full['win_fraction'], 'test': test_full['win_fraction']},
        indent=2))
    (out_dir / 'utility_margin.json').write_text(json.dumps(margin_stats(test_full['margin']), indent=2))
    (out_dir / 'head_overlap_matrix.json').write_text(json.dumps(test_full['overlap_matrix_mean'], indent=2))
    (out_dir / 'head_union_metrics.json').write_text(json.dumps(
        {'test_union_size_mean': test_full['union_size_mean']}, indent=2))
    (out_dir / 'oracle_vs_fixed.json').write_text(json.dumps(oracle_vs_fixed, indent=2))
    (out_dir / 'specialization_metrics.json').write_text(json.dumps({
        'mean_pairwise_top10_overlap_test': sum(test_full['overlap_matrix_mean'].values()) /
                                            max(len(test_full['overlap_matrix_mean']), 1),
        'union_size_mean_test': test_full['union_size_mean'],
        'winner_fraction_test': test_full['win_fraction'],
        'max_head_usage_test': max(test_full['win_fraction']),
        'min_head_usage_test': min(test_full['win_fraction']),
        'active_head_count_test': sum(1 for x in test_full['win_fraction'] if x > 0.01),
    }, indent=2))
    return final_test_metrics
