#!/usr/bin/env python3
"""TRACK-V5-TOPK-CORRECTION-MECHANISM01 Phase D4 -- real-data v1/v2
Differentiable Top-K comparison on the 4 primary cells' existing
checkpoints. Read-only inference + one backward pass per cell (for
gradient-norm/cosine diagnostics only, no optimizer step, no checkpoint
written). Uses the PRIMARY training temperature `tau_topk=0.05` (per
spec section D5 -- no tau sweep used to pick this value after the fact).
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_j_shared_encoder_drift01 import build_model
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_margutil01 import memory_value
from scripts.train_retriever_pool01 import CandidatePoolCache
from scripts.train_v5_head_loss_dual_encoder01 import compute_scores_dual
from utils.candidate_pool import gather_candidate_values, pooled_future_mse
from utils.mean_mixture_selection import mean_mixture_topk_selection
from utils.differentiable_topk01 import soft_topk_mask as soft_topk_mask_v1, soft_topk_forecast
from utils.differentiable_topk02 import soft_topk_mask_v2

TOP_K = 10
TAU_TOPK = 0.05  # primary training temperature, per spec D5

CELLS = {
    'ETTh1_96': dict(reference_ckpt='checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth',
                       pool='results/TRACK-V-MULTIQUERY-GENERALIZATION01/ETTh1/H96/seed0/pool_top100/shared_candidate_pool',
                       pred_len=96, seq_len=96, ckpt='checkpoints/track_v5_head_loss_dual_encoder01/ETTh1_96/S0/checkpoint_best_retmse.pth'),
    'ETTh1_720': dict(reference_ckpt='checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/stage1_carts_softset_ETTh1_720_S0_wce_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth',
                        pool='results/TRACK-V-MULTIQUERY-GENERALIZATION01/ETTh1/H720/seed0/pool_top100/shared_candidate_pool',
                        pred_len=720, seq_len=720, ckpt='checkpoints/track_v5_head_loss_dual_encoder01/ETTh1_720/S0/checkpoint_best_retmse.pth'),
    'Weather_96': dict(reference_ckpt='checkpoints/soft_set_mse/stage1/custom/seq96_pred96/stage1_carts_softset_Weather_96_S0_wce_RelationStage1_custom_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl96_pl96_0/checkpoint.pth',
                        pool='results/TRACK-V-MULTIQUERY-GENERALIZATION01/Weather/H96/seed0/pool_top100/shared_candidate_pool',
                        pred_len=96, seq_len=96, ckpt='checkpoints/track_v5_head_loss_dual_encoder01/Weather_96/S0/checkpoint_best_retmse.pth'),
    'Weather_720': dict(reference_ckpt='checkpoints/soft_set_mse/stage1/custom/seq720_pred720/stage1_carts_softset_Weather_720_S0_wce_RelationStage1_custom_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl720_pl720_0/checkpoint.pth',
                         pool='results/TRACK-V-MULTIQUERY-GENERALIZATION01/Weather/H720/seed0/pool_top100/shared_candidate_pool',
                         pred_len=720, seq_len=720, ckpt='checkpoints/track_v5_head_loss_dual_encoder01/Weather_720/S2/Weather_720/S2/checkpoint_best_retmse.pth'),
}


def analyze_cell(name, cfg, n_batches=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    class Cli:
        reference_ckpt = cfg['reference_ckpt']
        pred_len = cfg['pred_len']
        seq_len = cfg['seq_len']
        patch_len = 16
        init_seed = 0
        candidate_mask = 'raft'
        batch_size = 32
        top_k = TOP_K
    exp, args, model_q = build_model(Cli, device)
    bl = torch.load(cfg['ckpt'], map_location=device)
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
    slot_heads = SlotHeads(int(args.d_model), n_slots=5, std=1e-3).to(device)
    slot_heads.load_state_dict(bl['slot_heads_state_dict'])
    slot_heads.eval()

    expected_meta = {'seq_len': cfg['seq_len'], 'pred_len': cfg['pred_len'], 'candidate_mask': 'raft',
                     'candidate_pool_size': 100, 'candidate_pool_metric': 'delta_last_cosine'}
    pool_cache = CandidatePoolCache(Path(cfg['pool']) / 'test.pt', expected_meta)
    _, loader = exp._get_data(flag='test', shuffle=False)

    margins, kgaps, v1_mass, v2_mass, v1_eff_n, v2_eff_n = [], [], [], [], [], []
    pred_diffs, grad_norms_v1, grad_norms_v2, grad_cosines = [], [], [], []

    for bi, (batch_x, batch_y, batch_start_idx) in enumerate(loader):
        if bi >= n_batches:
            break
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        c = 0  # one channel per batch is enough for this diagnostic
        memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
        query_future = batch_y[:, :, c]
        with torch.no_grad():
            scores, valid_mask, pool_idx_global = compute_scores_dual(
                model_q, model_k, slot_heads, batch_x, exp, c, pool_cache, batch_start_idx, device)
            pooled_memory_c = gather_candidate_values(memory_c, pool_idx_global)
            y_pool = pooled_memory_c + offset_c.view(-1, 1, 1)
            _, p_heads, p_mix = mean_mixture_topk_selection(scores, valid_mask, 0.1, k=TOP_K)
            log_p_mix = torch.log(p_mix.clamp_min(1e-12))

            sorted_scores, _ = torch.sort(log_p_mix.masked_fill(~valid_mask, float('-inf')), dim=-1, descending=True)
            margin = sorted_scores[:, 0] - sorted_scores[:, TOP_K - 1]
            kgap = sorted_scores[:, TOP_K - 1] - sorted_scores[:, TOP_K]
            margins.append(margin.cpu()); kgaps.append(kgap.cpu())

        # gradient diagnostics: need grad w.r.t. a leaf tensor derived from the
        # real model forward pass, so re-run the slot_heads/model forward with
        # grad enabled this time, then take the TopK-loss gradient w.r.t.
        # log_p_mix.detach().requires_grad_() as a clean, isolated probe of the
        # v1-vs-v2 operator itself (isolates the operator's own gradient
        # behavior from the rest of the model, which is identical for both).
        log_p_mix_leaf = log_p_mix.detach().clone().requires_grad_(True)
        a_v1 = soft_topk_mask_v1(log_p_mix_leaf, valid_mask, k=TOP_K, tau=TAU_TOPK)
        r_v1 = soft_topk_forecast(a_v1, y_pool)
        loss_v1 = ((r_v1 - query_future) ** 2).mean()
        g_v1, = torch.autograd.grad(loss_v1, log_p_mix_leaf)

        log_p_mix_leaf2 = log_p_mix.detach().clone().requires_grad_(True)
        a_v2 = soft_topk_mask_v2(log_p_mix_leaf2, valid_mask, k=TOP_K, tau=TAU_TOPK)
        r_v2 = soft_topk_forecast(a_v2, y_pool)
        loss_v2 = ((r_v2 - query_future) ** 2).mean()
        g_v2, = torch.autograd.grad(loss_v2, log_p_mix_leaf2)

        v1_mass.append(a_v1.sum(-1).cpu())
        v2_mass.append(a_v2.sum(-1).cpu())
        hard_idx_mask = torch.zeros_like(a_v1, dtype=torch.bool)
        mix_idx, _, _ = mean_mixture_topk_selection(scores, valid_mask, 0.1, k=TOP_K)
        hard_idx_mask.scatter_(-1, mix_idx, True)
        v1_eff_n.append((a_v1.gather(-1, mix_idx).sum(-1)).cpu())
        v2_eff_n.append((a_v2.gather(-1, mix_idx).sum(-1)).cpu())
        pred_diffs.append((r_v1 - r_v2).norm(dim=-1).cpu())
        grad_norms_v1.append(g_v1.norm(dim=-1).cpu())
        grad_norms_v2.append(g_v2.norm(dim=-1).cpu())
        cos = F.cosine_similarity(g_v1, g_v2, dim=-1)
        grad_cosines.append(cos.cpu())

    result = {
        'cell': name,
        'mean_top10_score_margin': float(torch.cat(margins).mean()),
        'mean_kth_kplus1_gap': float(torch.cat(kgaps).mean()),
        'min_kth_kplus1_gap': float(torch.cat(kgaps).min()),
        'v1_mean_total_mass': float(torch.cat(v1_mass).mean()),
        'v2_mean_total_mass': float(torch.cat(v2_mass).mean()),
        'v1_mean_mass_in_hard_top10': float(torch.cat(v1_eff_n).mean()),
        'v2_mean_mass_in_hard_top10': float(torch.cat(v2_eff_n).mean()),
        'v1_mean_mass_outside_top10': float(torch.cat(v1_mass).mean() - torch.cat(v1_eff_n).mean()),
        'v2_mean_mass_outside_top10': float(torch.cat(v2_mass).mean() - torch.cat(v2_eff_n).mean()),
        'mean_v1_v2_prediction_l2_diff': float(torch.cat(pred_diffs).mean()),
        'mean_grad_norm_v1': float(torch.cat(grad_norms_v1).mean()),
        'mean_grad_norm_v2': float(torch.cat(grad_norms_v2).mean()),
        'mean_grad_cosine_v1_v2': float(torch.cat(grad_cosines).mean()),
    }
    print(f"[D4] {name}: margin={result['mean_top10_score_margin']:.4f} "
         f"Kgap={result['mean_kth_kplus1_gap']:.6f} (min={result['min_kth_kplus1_gap']:.8f}) "
         f"v1_mass_in_top10={result['v1_mean_mass_in_hard_top10']:.4f} "
         f"v2_mass_in_top10={result['v2_mean_mass_in_hard_top10']:.4f} "
         f"pred_diff={result['mean_v1_v2_prediction_l2_diff']:.6f} "
         f"grad_cos={result['mean_grad_cosine_v1_v2']:.4f}")
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', nargs='+', default=list(CELLS.keys()))
    ap.add_argument('--out_dir', default='results/TRACK-V5-TOPK-CORRECTION-MECHANISM01/phaseD')
    cli = ap.parse_args()
    rows = []
    for cell in cli.cells:
        rows.append(analyze_cell(cell, CELLS[cell]))
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out_dir / 'v1_v2_real_data_comparison.json', 'w'), indent=2)
    print('[D4] wrote', out_dir / 'v1_v2_real_data_comparison.json')


if __name__ == '__main__':
    main()
