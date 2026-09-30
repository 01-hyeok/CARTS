#!/usr/bin/env python3
"""TRACK-O-FROZEN-BASE-RETRIEVAL-CONSUMER01 PARTS 9-10 -- NO TRAINING,
independent of Stage2 training. For J1 and M2 (test split), per
(query_start_idx, channel), computes:

  - effective_K = 1/sum(alpha_i^2)                     (Uniform: always 10)
  - I_w = sum(w_i * MSE_i)  (weighted individual MSE; PART 9's I_U/I_H)
  - rho = Spearman(alpha_i, MSE_i) over the 10 candidates (Host only --
    Uniform's alpha is constant, correlation undefined)
  - D_w = sum(w_i^2 * MSE_i), C_w = Agg_w - D_w, Agg_w = MSE(R_w, Y_q)

reusing the EXACT SAME future-blind hard Top-10 selection and HostScorer
as every other track this session (`arm_score`/`stable_topk_indices` for
J1, `compute_scores`/`hard_unique_selection` for M2).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

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
    'M2': REPO_ROOT / 'checkpoints/track_m_relevance_constrained_multislot01/ETTh1_720/M2_relevance_budget/checkpoint.pth',
}
IS_MULTISLOT = {'J1': False, 'M2': True}
OUT_DIR = REPO_ROOT / 'results/TRACK-O-FROZEN-BASE-RETRIEVAL-CONSUMER01/ETTh1_720/aggregation_diagnostic'


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


@torch.no_grad()
def run_arm(arm, base_model, d_model, exp, args, host, device):
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
    _, loader = exp._get_data(flag='test', shuffle=False)
    rows = []

    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, counts = exp._candidate_mask(batch_start_idx)
        bsz = batch_x.size(0)

        for c in channels:
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            u = individual_utility_memsafe(memory_c, offset_c, query_future, 4096)
            d_raw = -u
            host_scores = host.scores(batch_x, c, cand_mask)

            if IS_MULTISLOT[arm]:
                scores = compute_scores(base_model, slot_heads, batch_x, exp.memory_x, c)
                picks_t = hard_unique_selection(scores, cand_mask, s=TOP_K)
            else:
                z_q = encode_raw(base_model, batch_x, c)
                z_k = encode_raw(base_model, exp.memory_x, c)
                s = arm_score(z_q, z_k, None).masked_fill(~cand_mask, float('-inf'))
                picks_t = stable_topk_indices(s, TOP_K, largest=True)

            ind_mse_i = d_raw.gather(1, picks_t)  # [B, K]
            y_sel = memory_c[picks_t] + offset_c.view(-1, 1, 1)  # [B, K, H]
            sc_sel = host_scores.gather(1, picks_t)
            alpha = torch.softmax(sc_sel / float(host.tau_topk), dim=-1)  # [B, K]
            w_u = torch.full_like(alpha, 1.0 / TOP_K)

            r_u = y_sel.mean(dim=1)
            r_h = (alpha.unsqueeze(-1) * y_sel).sum(dim=1)
            agg_u = ((r_u - query_future) ** 2).mean(dim=-1)
            agg_h = ((r_h - query_future) ** 2).mean(dim=-1)

            eff_k_u = 1.0 / (w_u ** 2).sum(dim=1)  # always 10
            eff_k_h = 1.0 / (alpha ** 2).sum(dim=1)
            i_u = (w_u * ind_mse_i).sum(dim=1)
            i_h = (alpha * ind_mse_i).sum(dim=1)
            d_w_u = (w_u ** 2 * ind_mse_i).sum(dim=1)
            d_w_h = (alpha ** 2 * ind_mse_i).sum(dim=1)
            c_w_u = agg_u - d_w_u
            c_w_h = agg_h - d_w_h

            alpha_np = alpha.cpu().numpy()
            mse_np = ind_mse_i.cpu().numpy()

            for b in range(bsz):
                rho, _ = spearmanr(alpha_np[b], mse_np[b])
                rows.append(dict(
                    query_start_idx=int(batch_start_idx[b]), channel=c, arm=arm,
                    effective_K_U=float(eff_k_u[b]), effective_K_H=float(eff_k_h[b]),
                    I_U=float(i_u[b]), I_H=float(i_h[b]),
                    D_w_U=float(d_w_u[b]), D_w_H=float(d_w_h[b]),
                    C_w_U=float(c_w_u[b]), C_w_H=float(c_w_h[b]),
                    Agg_U=float(agg_u[b]), Agg_H=float(agg_h[b]),
                    spearman_alpha_mse=float(rho) if rho == rho else float('nan'),  # NaN-safe
                ))
    return pd.DataFrame(rows)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp, args, base_model = build_model(_Cli(), device)
    assert state_hash(base_model) == EXPECTED_INIT_HASH
    d_model = int(args.d_model)
    host = HostScorer(S2_720, device)

    frames = []
    for arm in ('J1', 'M2'):
        df = run_arm(arm, base_model, d_model, exp, args, host, device)
        frames.append(df)
        print(f'[compute_o_agg_diag] {arm}: n={len(df)} mean_effK_H={df["effective_K_H"].mean():.3f} '
             f'mean_rho={df["spearman_alpha_mse"].mean():.4f}')
    full = pd.concat(frames, ignore_index=True)
    full.to_parquet(OUT_DIR / 'raw_per_query_channel.parquet', index=False)

    # effective_k.csv
    ek_rows = []
    for arm in ('J1', 'M2'):
        sub = full[full['arm'] == arm]
        ek_rows.append(dict(arm=arm, mean_effK_U=sub['effective_K_U'].mean(), mean_effK_H=sub['effective_K_H'].mean(),
                            median_effK_H=sub['effective_K_H'].median()))
    pd.DataFrame(ek_rows).to_csv(OUT_DIR / 'effective_k.csv', index=False)

    # weighted_individual_mse.csv
    wi_rows = []
    for arm in ('J1', 'M2'):
        sub = full[full['arm'] == arm]
        wi_rows.append(dict(arm=arm, mean_I_U=sub['I_U'].mean(), mean_I_H=sub['I_H'].mean(),
                            I_damage=sub['I_H'].mean() - sub['I_U'].mean()))
    pd.DataFrame(wi_rows).to_csv(OUT_DIR / 'weighted_individual_mse.csv', index=False)

    # alpha_quality_correlation.csv
    aq_rows = []
    for arm in ('J1', 'M2'):
        sub = full[full['arm'] == arm]
        rho_vals = sub['spearman_alpha_mse'].dropna()
        aq_rows.append(dict(arm=arm, mean_rho=rho_vals.mean(), median_rho=rho_vals.median(),
                            frac_rho_gt_0=(rho_vals > 0).mean(), frac_rho_lt_0=(rho_vals < 0).mean(),
                            n=len(rho_vals)))
    pd.DataFrame(aq_rows).to_csv(OUT_DIR / 'alpha_quality_correlation.csv', index=False)

    # dc_decomposition.csv
    dc_rows = []
    for arm in ('J1', 'M2'):
        sub = full[full['arm'] == arm]
        for tag in ('U', 'H'):
            dc_rows.append(dict(arm=arm, weighting=tag, effective_K=sub[f'effective_K_{tag}'].mean(),
                                I=sub[f'I_{tag}'].mean(), D_w=sub[f'D_w_{tag}'].mean(),
                                C_w=sub[f'C_w_{tag}'].mean(), Agg=sub[f'Agg_{tag}'].mean()))
    pd.DataFrame(dc_rows).to_csv(OUT_DIR / 'dc_decomposition.csv', index=False)

    print('[compute_o_agg_diag] wrote effective_k.csv, weighted_individual_mse.csv, '
         'alpha_quality_correlation.csv, dc_decomposition.csv')


if __name__ == '__main__':
    main()
