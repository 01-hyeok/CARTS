#!/usr/bin/env python3
"""TRACK-V5-TOPK-CORRECTION-MECHANISM01 Phase C -- Head Error Complementarity
analysis on EXISTING checkpoints only (S0/S1/S2-v1/D0/D1/D2-v1). No new
training. Read-only inference.

For each query/channel: per-head Top-10 retrieval S_m, per-head error
e_m = R_m - Y (R_m = mean of V_i over S_m), then head-pairwise:
  - raw_dot_mn  = <e_m, e_n>              (NOT divided by H a second time)
  - norm_m      = ||e_m||
  - cosine_mn   = raw_dot_mn / (norm_m * norm_n), with the explicit
                  zero-norm rule: if norm_m==0 or norm_n==0, cosine_mn
                  is defined as exactly 0.0 (a head with a perfect,
                  zero-error prediction has no "direction" to compare;
                  0 encodes "no measurable (dis)agreement" rather than
                  arbitrarily claiming perfect alignment (+1) or perfect
                  opposition (-1)).
  - overlap_mn  = |S_m ∩ S_n| / 10         (Top-10 index overlap fraction)
This avoids TRACK-V5-HEAD-OBJECTIVE-COSINE-FIX01's exact bug (reusing an
already-/H'd dot product as the cosine numerator) by computing raw_dot
and cosine from the SAME un-divided dot product, just as that track's own
fix did.

Also computes, for the Mean-Mixture's own Top-10 selection (not per-head):
within-set candidate-level error cosine/dot/norm (pairwise over the 10
actually-selected candidates), and the direct D/C decomposition, so the
head-level redundancy finding can be connected to the real Mean-Mixture
C term (section C4, question 4) rather than asserted by analogy.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.train_j_shared_encoder_drift01 import build_model
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_margutil01 import memory_value
from scripts.train_retriever_pool01 import CandidatePoolCache
from scripts.train_v5_head_loss_dual_encoder01 import compute_scores_dual
from utils.candidate_pool import gather_candidate_values, pooled_future_mse
from utils.mean_mixture_selection import mean_mixture_topk_selection

N_HEADS = 5
TOP_K = 10


def pairwise_cosine_zero_rule(E):
    """E: [B, K, H] error vectors for K items (heads or candidates).
    Returns raw_dot [B,K,K], norm [B,K], cosine [B,K,K] with cosine=0.0
    wherever either norm is 0 (documented rule above)."""
    raw_dot = torch.einsum('bkh,bjh->bkj', E, E)
    norm = torch.sqrt(torch.diagonal(raw_dot, dim1=-2, dim2=-1).clamp_min(0.0))
    denom = norm.unsqueeze(-1) * norm.unsqueeze(-2)
    zero_mask = denom <= 1e-12
    cosine = torch.where(zero_mask, torch.zeros_like(raw_dot), raw_dot / denom.clamp_min(1e-12))
    return raw_dot, norm, cosine


@torch.no_grad()
def analyze_cell(cell_cfg, arm, ckpt_path, pool_cache_path, out_rows_accum, channels_limit=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class Cli:
        reference_ckpt = cell_cfg['reference_ckpt']
        pred_len = cell_cfg['pred_len']
        seq_len = cell_cfg['seq_len']
        patch_len = 16
        init_seed = 0
        candidate_mask = 'raft'
        batch_size = 32
        top_k = TOP_K
    exp, args, model_q = build_model(Cli, device)
    bl = torch.load(ckpt_path, map_location=device)
    model_q.load_state_dict(bl['model_q_state_dict'])
    model_q.eval()
    shared = bool(bl['shared_encoder'])
    if shared:
        model_k = model_q
    else:
        import copy
        model_k = copy.deepcopy(model_q)
        model_k.load_state_dict(bl['model_k_state_dict'])
        model_k.eval()
    slot_heads = SlotHeads(int(args.d_model), n_slots=N_HEADS, std=1e-3).to(device)
    slot_heads.load_state_dict(bl['slot_heads_state_dict'])
    slot_heads.eval()

    expected_meta = {'seq_len': cell_cfg['seq_len'], 'pred_len': cell_cfg['pred_len'],
                     'candidate_mask': 'raft', 'candidate_pool_size': 100,
                     'candidate_pool_metric': 'delta_last_cosine'}
    pool_cache = CandidatePoolCache(Path(pool_cache_path) / 'test.pt', expected_meta)
    channels = list(range(int(args.enc_in)))
    if channels_limit:
        channels = channels[:channels_limit]
    _, loader = exp._get_data(flag='test', shuffle=False)
    tau_s = 0.1

    head_dot_sum = torch.zeros(N_HEADS, N_HEADS)
    head_cos_sum = torch.zeros(N_HEADS, N_HEADS)
    head_overlap_sum = torch.zeros(N_HEADS, N_HEADS)
    head_norm_sum = torch.zeros(N_HEADS)
    head_retmse_sum = torch.zeros(N_HEADS)
    neg_cos_count = torch.zeros(N_HEADS, N_HEADS)
    n_pairs_total = 0
    n_queries_total = 0

    mix_D_sum = mix_C_sum = mix_Agg_sum = 0.0
    mix_within_cos_sum = mix_within_dot_sum = 0.0
    mix_within_neg_count = 0.0
    mix_within_pairs = 0

    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        bsz = batch_x.size(0)
        for c in channels:
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            scores, valid_mask, pool_idx_global = compute_scores_dual(
                model_q, model_k, slot_heads, batch_x, exp, c, pool_cache, batch_start_idx, device)
            pooled_memory_c = gather_candidate_values(memory_c, pool_idx_global)
            y_pool = pooled_memory_c + offset_c.view(-1, 1, 1)
            d_pool = pooled_future_mse(pooled_memory_c, offset_c, query_future)

            mix_idx, p_heads, p_mix = mean_mixture_topk_selection(scores, valid_mask, tau_s, k=TOP_K)

            # --- per-head Top-10 and errors ---
            head_errors = []
            for m in range(N_HEADS):
                idx_m = stable_topk_indices(p_heads[:, m, :].masked_fill(~valid_mask, float('-inf')),
                                            TOP_K, largest=True)
                r_m = y_pool.gather(1, idx_m.unsqueeze(-1).expand(-1, -1, y_pool.size(-1))).mean(dim=1)
                e_m = r_m - query_future
                head_errors.append(e_m)
                head_retmse_sum[m] += float(d_pool.gather(1, idx_m).mean(dim=-1).sum())
                if m == 0:
                    head_idx_all = idx_m.unsqueeze(1)
                else:
                    head_idx_all = torch.cat([head_idx_all, idx_m.unsqueeze(1)], dim=1)
            E = torch.stack(head_errors, dim=1)  # [B, 5, H]
            raw_dot, norm, cosine = pairwise_cosine_zero_rule(E)
            head_dot_sum += raw_dot.sum(dim=0).cpu()
            head_cos_sum += cosine.sum(dim=0).cpu()
            neg_cos_count += (cosine < 0).sum(dim=0).float().cpu()
            head_norm_sum += norm.sum(dim=0).cpu()

            # Vectorized Top-10 overlap fraction via one-hot membership over the
            # local candidate-pool slots (size = pool_idx_global.size(1)), no
            # Python-level set() loop -- avoids the CPU bottleneck of the naive
            # per-query nested loop over (bsz x 25 head-pairs).
            n_pool = pool_idx_global.size(1)
            onehot = torch.zeros(bsz, N_HEADS, n_pool, device=device)
            onehot.scatter_(2, head_idx_all, 1.0)
            overlap_counts = torch.einsum('bmk,bnk->bmn', onehot, onehot)  # [B, 5, 5], counts in [0,10]
            head_overlap_sum += (overlap_counts.sum(dim=0) / TOP_K).cpu()
            n_pairs_total += bsz
            n_queries_total += bsz

            # --- Mean-Mixture Top-10 within-set candidate error cosine + D/C ---
            y_sel = y_pool.gather(1, mix_idx.unsqueeze(-1).expand(-1, -1, y_pool.size(-1)))
            e_cand = y_sel - query_future.unsqueeze(1)  # [B, 10, H]
            agg_pred = y_sel.mean(dim=1)
            agg_mse = ((agg_pred - query_future) ** 2).mean(dim=-1)
            ind_mse = d_pool.gather(1, mix_idx)
            D_ = ind_mse.mean(dim=-1) / TOP_K
            C_ = agg_mse - D_
            mix_D_sum += float(D_.sum()); mix_C_sum += float(C_.sum()); mix_Agg_sum += float(agg_mse.sum())
            raw_dot_c, norm_c, cosine_c = pairwise_cosine_zero_rule(e_cand)
            iu = torch.triu_indices(TOP_K, TOP_K, offset=1)
            mix_within_dot_sum += float(raw_dot_c[:, iu[0], iu[1]].sum())
            mix_within_cos_sum += float(cosine_c[:, iu[0], iu[1]].sum())
            mix_within_neg_count += float((cosine_c[:, iu[0], iu[1]] < 0).sum())
            mix_within_pairs += bsz * iu.shape[1]

    n_head_pair_obs = n_pairs_total  # per (m,n) cell, summed over this many queries (all channels pooled)
    head_cos_mean = (head_cos_sum / max(n_head_pair_obs, 1)).tolist()
    head_dot_mean = (head_dot_sum / max(n_head_pair_obs, 1)).tolist()
    head_overlap_mean = (head_overlap_sum / max(n_head_pair_obs, 1)).tolist()
    head_norm_mean = (head_norm_sum / max(n_queries_total, 1)).tolist()
    head_retmse_mean = (head_retmse_sum / max(n_queries_total, 1)).tolist()
    neg_frac = (neg_cos_count / max(n_head_pair_obs, 1))
    # mean over off-diagonal pairs only
    off_mask = ~torch.eye(N_HEADS, dtype=torch.bool)
    mean_offdiag_cosine = float(torch.tensor(head_cos_mean)[off_mask].mean())
    mean_offdiag_overlap = float(torch.tensor(head_overlap_mean)[off_mask].mean())
    mean_neg_frac = float(neg_frac[off_mask].mean())

    result = {
        'cell': cell_cfg['name'], 'arm': arm, 'checkpoint': str(ckpt_path),
        'n_queries_total_channel_pooled': n_queries_total,
        'head_error_cosine_matrix': head_cos_mean,
        'head_error_dot_matrix': head_dot_mean,
        'head_top10_overlap_matrix': head_overlap_mean,
        'head_error_norm': head_norm_mean,
        'head_retmse': head_retmse_mean,
        'mean_offdiag_head_error_cosine': mean_offdiag_cosine,
        'mean_offdiag_head_top10_overlap': mean_offdiag_overlap,
        'mean_offdiag_negative_cosine_fraction': mean_neg_frac,
        'mixmix_top10_D': mix_D_sum / max(n_queries_total, 1),
        'mixmix_top10_C': mix_C_sum / max(n_queries_total, 1),
        'mixmix_top10_AggMSE': mix_Agg_sum / max(n_queries_total, 1),
        'mixmix_within_set_candidate_error_cosine_mean': mix_within_cos_sum / max(mix_within_pairs, 1),
        'mixmix_within_set_candidate_error_dot_mean': mix_within_dot_sum / max(mix_within_pairs, 1),
        'mixmix_within_set_negative_cosine_fraction': mix_within_neg_count / max(mix_within_pairs, 1),
    }
    out_rows_accum.append(result)
    print(f"[phaseC] {cell_cfg['name']}/{arm}: mean_offdiag_cos={mean_offdiag_cosine:.4f} "
         f"overlap={mean_offdiag_overlap:.4f} mix_D={result['mixmix_top10_D']:.6f} "
         f"mix_C={result['mixmix_top10_C']:.6f} within_set_cos={result['mixmix_within_set_candidate_error_cosine_mean']:.4f}")
    return result


CELLS = {
    'ETTh1_96': dict(name='ETTh1_96', reference_ckpt='checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth',
                       pool='results/TRACK-V-MULTIQUERY-GENERALIZATION01/ETTh1/H96/seed0/pool_top100/shared_candidate_pool',
                       pred_len=96, seq_len=96),
    'ETTh1_720': dict(name='ETTh1_720', reference_ckpt='checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/stage1_carts_softset_ETTh1_720_S0_wce_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth',
                        pool='results/TRACK-V-MULTIQUERY-GENERALIZATION01/ETTh1/H720/seed0/pool_top100/shared_candidate_pool',
                        pred_len=720, seq_len=720),
    'Weather_96': dict(name='Weather_96', reference_ckpt='checkpoints/soft_set_mse/stage1/custom/seq96_pred96/stage1_carts_softset_Weather_96_S0_wce_RelationStage1_custom_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl96_pl96_0/checkpoint.pth',
                        pool='results/TRACK-V-MULTIQUERY-GENERALIZATION01/Weather/H96/seed0/pool_top100/shared_candidate_pool',
                        pred_len=96, seq_len=96),
    'Weather_720': dict(name='Weather_720', reference_ckpt='checkpoints/soft_set_mse/stage1/custom/seq720_pred720/stage1_carts_softset_Weather_720_S0_wce_RelationStage1_custom_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl720_pl720_0/checkpoint.pth',
                         pool='results/TRACK-V-MULTIQUERY-GENERALIZATION01/Weather/H720/seed0/pool_top100/shared_candidate_pool',
                         pred_len=720, seq_len=720),
}
ARMS = ('S0', 'S1', 'S2', 'D0', 'D1', 'D2')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', nargs='+', default=list(CELLS.keys()))
    ap.add_argument('--out_dir', default='results/TRACK-V5-TOPK-CORRECTION-MECHANISM01/phaseC')
    cli = ap.parse_args()
    rows = []
    for cell in cli.cells:
        cfg = CELLS[cell]
        for arm in ARMS:
            sub = 'phase1' if arm.startswith('S') else 'phase3'
            ckpt = Path(f'checkpoints/track_v5_head_loss_dual_encoder01/{cell}/{arm}/checkpoint_best_retmse.pth')
            if not ckpt.exists():
                print(f'[phaseC][SKIP] missing checkpoint {ckpt}')
                continue
            analyze_cell(cfg, arm, ckpt, cfg['pool'], rows)
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'head_complementarity_results.json'
    json.dump(rows, open(out_path, 'w'), indent=2)
    print(f'[phaseC] wrote {out_path}')


if __name__ == '__main__':
    main()
