#!/usr/bin/env python3
"""TRACK-N-FORECAST-CONDITIONAL-UTILITY01 PARTS 8, 9, 12, 13, 14, 17 --
NO TRAINING. Reads `all_diagnostics.parquet` (produced by
`compute_n_diagnostics01.py`) and produces every required summary table.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / 'results/TRACK-N-FORECAST-CONDITIONAL-UTILITY01/ETTh1_720'
ARMS = ('J1', 'K2', 'M2')
EPS = 1e-8
PRED_LEN = 720  # `.norm()`-based t_norm2/c_norm2/dot_ct are SUM-over-H scale; divide by PRED_LEN to
                # match the MEAN-over-H "MSE" scale used everywhere else (base_mse, retrieval_mse_*, ...)


def main():
    df = pd.read_parquet(OUT_DIR / 'all_diagnostics.parquet')
    val = df[df['split'] == 'val'].copy()
    test = df[df['split'] == 'test'].copy()

    # ---- PART 8: aggregation damage (test split) ----
    damage_rows = []
    for arm in ARMS:
        sub = test[test['arm'] == arm]
        mse_u = sub['retrieval_mse_U'].mean()
        mse_h = sub['retrieval_mse_H'].mean()
        damage = mse_h - mse_u
        damage_pct = damage / mse_u * 100
        damage_rows.append(dict(arm=arm, MSE_U=mse_u, MSE_H=mse_h, Damage=damage, Damage_pct=damage_pct,
                                C_w_U=sub['C_w_U'].mean(), C_w_H=sub['C_w_H'].mean(),
                                C_restored_by_host=sub['C_w_H'].mean() - sub['C_w_U'].mean()))
    pd.DataFrame(damage_rows).to_csv(OUT_DIR / 'uniform_vs_host.csv', index=False)

    # ---- PART 9: weighted D/C decomposition summary (test split) ----
    decomp_rows = []
    for arm in ARMS:
        sub = test[test['arm'] == arm]
        for tag in ('U', 'H'):
            d = sub[f'D_w_{tag}'].mean()
            c = sub[f'C_w_{tag}'].mean()
            agg = sub[f'retrieval_mse_{tag}'].mean()
            decomp_rows.append(dict(arm=arm, weighting=tag, D_w=d, C_w=c, Agg_w=agg,
                                    identity_abs_err=abs(d + c - agg)))
    pd.DataFrame(decomp_rows).to_csv(OUT_DIR / 'weighted_decomposition.csv', index=False)

    # ---- PART 10: conditional-correction macro summary (test split) ----
    cond_rows = []
    for arm in ARMS:
        sub = test[test['arm'] == arm]
        for tag in ('U', 'H'):
            cond_rows.append(dict(
                arm=arm, weighting=tag, base_mse=sub['base_mse'].mean(),
                retrieval_mse=sub[f'retrieval_mse_{tag}'].mean(),
                mean_correction_norm=sub[f'correction_norm_{tag}'].mean(),
                mean_target_residual_norm=sub['target_residual_norm'].mean(),
                mean_cosine=sub[f'cosine_{tag}'].mean(), median_cosine=sub[f'cosine_{tag}'].median(),
                mean_dot=sub[f'dot_{tag}'].mean(),
            ))
    pd.DataFrame(cond_rows).to_csv(OUT_DIR / 'conditional_utility.csv', index=False)

    # ---- PART 12: analytic oracle lambda summary (test split) ----
    oracle_rows = []
    for arm in ARMS:
        sub = test[test['arm'] == arm]
        for tag in ('U', 'H'):
            lam = sub[f'lambda_star_{tag}']
            gain = sub[f'oracle_gain_{tag}']
            oracle_rows.append(dict(
                arm=arm, weighting=tag,
                mean_lambda_star=lam.mean(), median_lambda_star=lam.median(),
                frac_lambda_gt_0=(lam > 0).mean(), frac_lambda_gt_0p02=(lam > 0.02).mean(),
                frac_lambda_gt_0p10=(lam > 0.10).mean(), frac_lambda_gt_0p25=(lam > 0.25).mean(),
                frac_lambda_gt_0p50=(lam > 0.50).mean(),
                mean_oracle_gain=gain.mean(), median_oracle_gain=gain.median(),
                frac_oracle_gain_gt_0=(gain > 0).mean(),
                mean_oracle_fused_mse=sub[f'oracle_fused_mse_{tag}'].mean(),
                mean_residual_cosine=sub[f'cosine_{tag}'].mean(), median_residual_cosine=sub[f'cosine_{tag}'].median(),
            ))
    pd.DataFrame(oracle_rows).to_csv(OUT_DIR / 'oracle_lambda_summary.csv', index=False)

    # ---- PART 13: validation-calibrated lambda (global + per-channel) ----
    grid = np.round(np.arange(0.0, 1.001, 0.01), 2)
    val_lambda = {}
    test_calib_rows = []
    for arm in ARMS:
        val_lambda[arm] = {}
        for tag in ('U', 'H'):
            v = val[(val['arm'] == arm)]
            b_v, r_v, y_ok = None, None, None
            # reconstruct per-row squared errors at each candidate lambda from stored scalars:
            # MSE(B + lam*(R-B), Y) = mean_h( (1-lam)*(B-Y) + lam*(R-Y) )^2 -- but we only have norms/dot,
            # not full vectors; use the closed form via target_residual (t=Y-B) and correction (c=R-B):
            # err(lam) = mean_h( t - lam*c )^2 = |t|^2/H - 2*lam*dot(c,t)/H + lam^2*|c|^2/H  (per row, H implicit
            # in the norms already being defined as vector norms over H after /H mean -- see note below).
            t_norm2 = (v['target_residual_norm'] ** 2).values
            c_norm2 = (v[f'correction_norm_{tag}'] ** 2).values
            dot_ct = (v[f'dot_{tag}']).values
            best_lam, best_mse = 0.0, float('inf')
            for lam in grid:
                mse = ((t_norm2 - 2 * lam * dot_ct + (lam ** 2) * c_norm2) / PRED_LEN).mean()
                if mse < best_mse:
                    best_mse, best_lam = mse, float(lam)
            val_lambda[arm][tag] = {'global_lambda': best_lam, 'val_mse_at_global_lambda': float(best_mse)}

            # per-channel
            per_channel = {}
            for ch in range(7):
                vc = v[v['channel'] == ch]
                t2 = (vc['target_residual_norm'] ** 2).values
                c2 = (vc[f'correction_norm_{tag}'] ** 2).values
                dct = (vc[f'dot_{tag}']).values
                bl, bm = 0.0, float('inf')
                for lam in grid:
                    mse = ((t2 - 2 * lam * dct + (lam ** 2) * c2) / PRED_LEN).mean()
                    if mse < bm:
                        bm, bl = mse, float(lam)
                per_channel[str(ch)] = {'lambda': bl, 'val_mse': float(bm)}
            val_lambda[arm][tag]['per_channel'] = per_channel

            # apply to TEST
            te = test[test['arm'] == arm]
            te_t2 = (te['target_residual_norm'] ** 2).values
            te_c2 = (te[f'correction_norm_{tag}'] ** 2).values
            te_dct = (te[f'dot_{tag}']).values
            g_lam = val_lambda[arm][tag]['global_lambda']
            test_mse_global = float(((te_t2 - 2 * g_lam * te_dct + (g_lam ** 2) * te_c2) / PRED_LEN).mean())

            te_channel_lams = np.array([val_lambda[arm][tag]['per_channel'][str(c)]['lambda']
                                        for c in te['channel'].values])
            test_mse_perchannel = float(((te_t2 - 2 * te_channel_lams * te_dct
                                        + (te_channel_lams ** 2) * te_c2) / PRED_LEN).mean())

            test_calib_rows.append(dict(arm=arm, weighting=tag, base_mse_test=te['base_mse'].mean(),
                                        retrieval_mse_test=te[f'retrieval_mse_{tag}'].mean(),
                                        oracle_fused_mse_test=te[f'oracle_fused_mse_{tag}'].mean(),
                                        val_global_lambda=g_lam, test_mse_val_global_lambda=test_mse_global,
                                        test_mse_val_channel_lambda=test_mse_perchannel))
    (OUT_DIR / 'validation_lambda.json').write_text(json.dumps(val_lambda, indent=2))
    pd.DataFrame(test_calib_rows).to_csv(OUT_DIR / 'test_calibrated_fusion.csv', index=False)

    # ---- PART 14: primary comparison table ----
    primary_rows = []
    for arm in ARMS:
        for tag in ('U', 'H'):
            te = test[test['arm'] == arm]
            tc = next(r for r in test_calib_rows if r['arm'] == arm and r['weighting'] == tag)
            primary_rows.append(dict(
                Retriever=arm, Aggregation=tag, Common_Base_MSE=te['base_mse'].mean(),
                Retrieval_MSE=te[f'retrieval_mse_{tag}'].mean(),
                Residual_Cosine=te[f'cosine_{tag}'].mean(),
                Oracle_Fusion_MSE=te[f'oracle_fused_mse_{tag}'].mean(),
                Val_Global_Lambda_Test_MSE=tc['test_mse_val_global_lambda'],
                Val_Channel_Lambda_Test_MSE=tc['test_mse_val_channel_lambda'],
            ))
    pd.DataFrame(primary_rows).to_csv(OUT_DIR / 'primary_comparison.csv', index=False)

    # ---- PART 17: correlations (pooled across arms, test split, row-level) ----
    corr_rows = []
    for tag in ('U', 'H'):
        cols = {'retrieval_mse': test[f'retrieval_mse_{tag}'], 'D_w': test[f'D_w_{tag}'],
               'C_w': test[f'C_w_{tag}'], 'cosine': test[f'cosine_{tag}'],
               'oracle_gain': test[f'oracle_gain_{tag}'], 'lambda_star': test[f'lambda_star_{tag}']}
        names = list(cols.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = cols[names[i]].values, cols[names[j]].values
                pear = pearsonr(a, b)
                spear = spearmanr(a, b)
                corr_rows.append(dict(weighting=tag, var_a=names[i], var_b=names[j],
                                      pearson_r=pear[0], pearson_p=pear[1],
                                      spearman_r=spear.correlation, spearman_p=spear.pvalue))
    pd.DataFrame(corr_rows).to_csv(OUT_DIR / 'correlations.csv', index=False)

    print('[compute_n_summary] wrote uniform_vs_host.csv, weighted_decomposition.csv, conditional_utility.csv, '
         'oracle_lambda_summary.csv, validation_lambda.json, test_calibrated_fusion.csv, primary_comparison.csv, '
         'correlations.csv')
    print(pd.DataFrame(primary_rows).to_string(index=False))


if __name__ == '__main__':
    main()
