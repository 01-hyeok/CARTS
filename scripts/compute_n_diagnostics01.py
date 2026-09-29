#!/usr/bin/env python3
"""TRACK-N-FORECAST-CONDITIONAL-UTILITY01 PARTS 5-13 -- NO TRAINING.

For each retriever (J1/K2/M2) and split (val/test), regenerates the
EXACT SAME future-blind hard Top-10 candidate set used throughout
Stage1/TRACK-M (J1: arm_score+stable_topk_indices; K2/M2:
compute_scores+hard_unique_selection -- unmodified), then computes, per
(query_start_idx, channel) row:

  - Uniform aggregation R^U = mean_i(Y_i)  (== Stage1's AggMSE definition)
  - Host aggregation R^H = sum_i(alpha_i*Y_i), alpha from the SAME fixed
    HostScorer TRACK-M's Stage2 cache used (== TRACK-M's
    raw_retrieval_aggregate_mse definition)
  - weighted D/C decomposition for both weightings (identity Agg=D+C
    verified per row): since sum_i w_i=1 for both weightings,
    Agg_w = MSE(R_w, Y) directly; D_w = sum_i w_i^2 * individual_mse_i;
    C_w = Agg_w - D_w.
  - forecast-conditional quantities against the COMMON frozen base B_q
    (precomputed by precompute_n_common_base01.py): correction c=R_w-B,
    target residual t=Y-B, correction_norm, target_residual_norm,
    cosine(c,t), dot(c,t), analytic optimal lambda* = clip(<t,c>/(|c|^2+eps),0,1),
    oracle_fused_mse, oracle_gain = MSE(B,Y) - oracle_fused_mse.

No [[B,S,N,H]] tensor is ever built; only [B,10] index tensors and
[B,H] value tensors exist at any point (per-channel loop, matching the
established convention throughout this session).
"""
import json
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.train_factorial_e2e01 import HostScorer, arm_score, encode_raw, individual_utility_memsafe
from scripts.train_j_shared_encoder_drift01 import build_model, state_hash
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads, compute_scores, hard_unique_selection
from scripts.train_margutil01 import memory_value

EPS = 1e-8
TOP_K = 10
S1_720 = ('checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/'
         'stage1_carts_softset_ETTh1_720_S0_wce_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_'
         'df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth')
S2_720 = ('checkpoints/stage2/ETTh1/seq720_pred720/stage2_carts_softset_s2_ETTh1_720_S0_wce_RelationStage2_ETTh1_'
         'ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_'
         'sl720_pl720_0/checkpoint.pth')
EXPECTED_INIT_HASH = 'b37fa4031f538e4b5f5c522ae22e7d03613e4ee66283d2637656c7c4f872372e'

CKPTS = {
    'J1': REPO_ROOT / 'checkpoints/track_j2_key_update_decomposition01/ETTh1_720/J1_stopgrad_key/checkpoint.pth',
    'K2': REPO_ROOT / 'checkpoints/track_k_multislot_predictive_retrieval01/ETTh1_720/K2_multislot_aggregate/checkpoint.pth',
    'M2': REPO_ROOT / 'checkpoints/track_m_relevance_constrained_multislot01/ETTh1_720/M2_relevance_budget/checkpoint.pth',
}
IS_MULTISLOT = {'J1': False, 'K2': True, 'M2': True}
BASE_DIR = REPO_ROOT / 'results/TRACK-N-FORECAST-CONDITIONAL-UTILITY01/ETTh1_720/common_base_predictions'
OUT_DIR = REPO_ROOT / 'results/TRACK-N-FORECAST-CONDITIONAL-UTILITY01/ETTh1_720'


class _Cli:
    reference_ckpt = S1_720
    cell = 'ETTh1_720'
    pred_len = 720
    seq_len = 720
    patch_len = 16
    top_k = 10
    tau_t = 0.1
    tau_s = 0.1
    batch_size = 32
    learning_rate = 1e-3
    chunk_size = 4096
    init_seed = 0
    loader_seed = 0


def load_base_cache(split):
    d = torch.load(BASE_DIR / f'base_predictions_{split}.pt', map_location='cpu')
    lut = {int(s): i for i, s in enumerate(d['batch_start_idx'].tolist())}
    return d, lut


@torch.no_grad()
def run_retriever_split(arm, base_model, d_model, exp, args, host, base_cache, base_lut, split, device):
    slot_heads = None
    bl = torch.load(CKPTS[arm], map_location=device)
    base_model.load_state_dict(bl['model_state_dict'])
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)
    if IS_MULTISLOT[arm]:
        slot_heads = SlotHeads(d_model).to(device)
        slot_heads.load_state_dict(bl['slot_heads_state_dict'])
        slot_heads.eval()

    channels = list(range(int(args.enc_in)))
    _, loader = exp._get_data(flag=split, shuffle=False)
    rows = []

    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, counts = exp._candidate_mask(batch_start_idx)
        bsz = batch_x.size(0)
        b_idx = [base_lut[int(s)] for s in batch_start_idx.tolist()]
        b_q_full = base_cache['base_predictions'][b_idx].to(device)  # [B, H, C_all]

        for c in channels:
            z_q = encode_raw(base_model, batch_x, c)
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            u = individual_utility_memsafe(memory_c, offset_c, query_future, 4096)
            d_raw = -u  # [B, N] individual future-MSE for every candidate
            host_scores = host.scores(batch_x, c, cand_mask)

            if IS_MULTISLOT[arm]:
                scores = compute_scores(base_model, slot_heads, batch_x, exp.memory_x, c)
                picks_t = hard_unique_selection(scores, cand_mask, s=TOP_K)
            else:
                z_k = encode_raw(base_model, exp.memory_x, c)
                s = arm_score(z_q, z_k, None).masked_fill(~cand_mask, float('-inf'))
                picks_t = stable_topk_indices(s, TOP_K, largest=True)

            ind_mse_i = d_raw.gather(1, picks_t)  # [B, K]
            y_sel = memory_c[picks_t] + offset_c.view(-1, 1, 1)  # [B, K, H] absolute space

            sc_sel = host_scores.gather(1, picks_t)
            alpha = torch.softmax(sc_sel / float(host.tau_topk), dim=-1)  # [B, K]

            r_u = y_sel.mean(dim=1)  # [B, H]
            r_h = (alpha.unsqueeze(-1) * y_sel).sum(dim=1)  # [B, H]

            d_w_u = ((1.0 / TOP_K) ** 2 * ind_mse_i).sum(dim=1)  # [B]
            d_w_h = (alpha ** 2 * ind_mse_i).sum(dim=1)  # [B]

            agg_u = ((r_u - query_future) ** 2).mean(dim=-1)  # [B]
            agg_h = ((r_h - query_future) ** 2).mean(dim=-1)  # [B]
            c_w_u = agg_u - d_w_u
            c_w_h = agg_h - d_w_h

            b_q = b_q_full[:, :, c]  # [B, H]
            base_mse = ((b_q - query_future) ** 2).mean(dim=-1)  # [B]

            t = query_future - b_q  # [B, H]
            t_norm = t.norm(dim=-1)  # [B]

            for tag, r_w in (('U', r_u), ('H', r_h)):
                corr = r_w - b_q  # [B, H]
                corr_norm = corr.norm(dim=-1)
                dot_ct = (corr * t).sum(dim=-1)
                cos_ct = dot_ct / (corr_norm * t_norm + EPS)
                lam_star = (dot_ct / (corr_norm ** 2 + EPS)).clamp(0.0, 1.0)
                fused = b_q + lam_star.unsqueeze(-1) * corr
                oracle_mse = ((fused - query_future) ** 2).mean(dim=-1)
                oracle_gain = base_mse - oracle_mse
                if tag == 'U':
                    corr_norm_u, dot_u, cos_u, lam_u, oracle_mse_u, gain_u = \
                        corr_norm, dot_ct, cos_ct, lam_star, oracle_mse, oracle_gain
                else:
                    corr_norm_h, dot_h, cos_h, lam_h, oracle_mse_h, gain_h = \
                        corr_norm, dot_ct, cos_ct, lam_star, oracle_mse, oracle_gain

            for b in range(bsz):
                rows.append(dict(
                    query_start_idx=int(batch_start_idx[b]), channel=c, arm=arm, split=split,
                    base_mse=float(base_mse[b]),
                    retrieval_mse_U=float(agg_u[b]), retrieval_mse_H=float(agg_h[b]),
                    D_w_U=float(d_w_u[b]), C_w_U=float(c_w_u[b]),
                    D_w_H=float(d_w_h[b]), C_w_H=float(c_w_h[b]),
                    correction_norm_U=float(corr_norm_u[b]), correction_norm_H=float(corr_norm_h[b]),
                    target_residual_norm=float(t_norm[b]),
                    cosine_U=float(cos_u[b]), cosine_H=float(cos_h[b]),
                    dot_U=float(dot_u[b]), dot_H=float(dot_h[b]),
                    lambda_star_U=float(lam_u[b]), lambda_star_H=float(lam_h[b]),
                    oracle_fused_mse_U=float(oracle_mse_u[b]), oracle_fused_mse_H=float(oracle_mse_h[b]),
                    oracle_gain_U=float(gain_u[b]), oracle_gain_H=float(gain_h[b]),
                ))
    return pd.DataFrame(rows)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp, args, base_model = build_model(_Cli(), device)
    assert state_hash(base_model) == EXPECTED_INIT_HASH
    d_model = int(args.d_model)
    host = HostScorer(S2_720, device)

    all_frames = []
    for split in ('val', 'test'):
        base_cache, base_lut = load_base_cache(split)
        for arm in ('J1', 'K2', 'M2'):
            df = run_retriever_split(arm, base_model, d_model, exp, args, host, base_cache, base_lut, split, device)
            out_path = REPO_ROOT / f'results/TRACK-N-FORECAST-CONDITIONAL-UTILITY01/ETTh1_720/retrieval/{arm}/{split}.parquet'
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out_path, index=False)
            macro_u = df['retrieval_mse_U'].mean()
            macro_h = df['retrieval_mse_H'].mean()
            print(f'[compute_n] {arm}/{split}: n={len(df)} macro_retrieval_mse_U={macro_u:.6f} '
                 f'macro_retrieval_mse_H={macro_h:.6f}')
            all_frames.append(df)

    full = pd.concat(all_frames, ignore_index=True)
    full.to_parquet(OUT_DIR / 'all_diagnostics.parquet', index=False)
    print(f'[compute_n] wrote {OUT_DIR / "all_diagnostics.parquet"} ({len(full)} rows)')


if __name__ == '__main__':
    main()
