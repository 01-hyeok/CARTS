#!/usr/bin/env python3
"""TRACK-Q-GATE-CAPACITY-CALIBRATION01 -- PARTS 13/16/17/18: per-(query,
channel) lambda + realized gain for every arm on the TEST split, oracle
-lambda calibration comparison, lambda-bin analysis, and the primary
comparison table. NO retriever/base/consumer training here -- reads
already-trained gate checkpoints only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_p_fusion_semantics01 import CACHE_DIRS as P_CACHE_DIRS
from scripts.train_p_fusion_semantics01 import build_fresh_stage2_mixture, S2_720
from scripts.train_q_gate_capacity01 import (
    GATE_CLASSES, load_tensors,
)
from scripts.train_setlossctrl_stage2_retrain02 import load_cache_as_lookup

OUT_DIR = REPO_ROOT / 'results/TRACK-Q-GATE-CAPACITY-CALIBRATION01/ETTh1_720'
N_CHANNELS = 7


@torch.no_grad()
def q4_per_query_channel(device):
    bl = torch.load('checkpoints/track_p_fusion_semantics_audit01/ETTh1_720/P4_M2_mixture/checkpoint.pth',
                    map_location=device)
    exp, args, model, host_ck = build_fresh_stage2_mixture(S2_720, seed=0, fusion_mode='mixture')
    model.to(device)
    exp._ensure_memory()
    exp._build_key_bank(force=True)
    model.load_state_dict(bl['model_state_dict'])
    model.eval()
    cache, lut = load_cache_as_lookup(P_CACHE_DIRS['P4_M2_mixture'] / 'test.pt')
    _, test_loader = exp._get_data(flag='test', shuffle=False)
    rows = []
    for batch_x, batch_y, batch_start_idx in test_loader:
        batch_x, batch_y, batch_start_idx = exp._move_batch(batch_x, batch_y, batch_start_idx)
        cand_mask, counts = exp._candidate_mask(batch_start_idx)
        idx = torch.tensor([lut[int(s)] for s in batch_start_idx.tolist()], dtype=torch.long)
        rcache = {'relation_outputs': cache['relation_outputs'][idx].to(device),
                  'relation_query_embs': cache['relation_query_embs'][idx].to(device),
                  'query_offset': cache['query_offset'][idx].to(device)}
        y_final, y_base, y_ret, beta, lam, debug = model.forward_from_retrieval_values(
            rcache['relation_outputs'], batch_x=batch_x, retrieval_cache=rcache,
            memory_y=exp.memory_y, valid_mask=cand_mask, key_bank=exp.key_bank,
            memory_x_last=exp.memory_x_last, target_y=batch_y)
        se = ((y_final - batch_y) ** 2).mean(dim=1)  # [B, C]
        base_se = ((y_base - batch_y) ** 2).mean(dim=1)  # [B, C]
        gain = base_se - se
        for b in range(batch_x.size(0)):
            for c in range(N_CHANNELS):
                rows.append({'query_start_idx': int(batch_start_idx[b]), 'channel': c, 'arm': 'Q4',
                            'lambda': float(lam[b, c]), 'gain': float(gain[b, c]),
                            'final_se': float(se[b, c]), 'base_se': float(base_se[b, c])})
    return pd.DataFrame(rows)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # oracle lambda* (test, M2, Uniform) -- reused verbatim from TRACK-N
    oracle_df = pd.read_parquet(REPO_ROOT / 'results/TRACK-N-FORECAST-CONDITIONAL-UTILITY01/ETTh1_720/'
                                'all_diagnostics.parquet')
    oracle_df = oracle_df[(oracle_df['arm'] == 'M2') & (oracle_df['split'] == 'test')][
        ['query_start_idx', 'channel', 'lambda_star_U']].rename(columns={'lambda_star_U': 'oracle_lambda'})

    te_starts, B_te, R_te, Y_te = load_tensors('test', device)
    starts_np = te_starts.numpy()

    all_frames = []

    # Q1 fixed
    lam = torch.full((B_te.size(0), N_CHANNELS), 0.42, device=device)
    fused = B_te + lam.unsqueeze(1) * (R_te - B_te)
    se = ((fused - Y_te) ** 2).mean(dim=1)
    base_se = ((B_te - Y_te) ** 2).mean(dim=1)
    gain = base_se - se
    rows = [{'query_start_idx': int(starts_np[i]), 'channel': c, 'arm': 'Q1', 'lambda': 0.42,
            'gain': float(gain[i, c]), 'final_se': float(se[i, c]), 'base_se': float(base_se[i, c])}
           for i in range(B_te.size(0)) for c in range(N_CHANNELS)]
    all_frames.append(pd.DataFrame(rows))
    print('[compute_q] Q1 rows:', len(rows))

    # Q2/Q3/Q5 -- load trained gate checkpoints
    for gate_type, arm in (('global', 'Q2'), ('per_channel', 'Q3'), ('query_prior', 'Q5')):
        arm_dir = {'Q2': 'Q2_trainable_global', 'Q3': 'Q3_per_channel', 'Q5': 'Q5_prior_query_gate'}[arm]
        ck = torch.load(OUT_DIR / arm_dir / 'checkpoint.pth', map_location=device)
        gate = GATE_CLASSES[gate_type]().to(device)
        gate.load_state_dict(ck['gate_state_dict'])
        gate.eval()
        with torch.no_grad():
            lam = gate(B_te, R_te)  # [N, C]
            fused = B_te + lam.unsqueeze(1) * (R_te - B_te)
            se = ((fused - Y_te) ** 2).mean(dim=1)
            base_se = ((B_te - Y_te) ** 2).mean(dim=1)
            gain = base_se - se
        rows = [{'query_start_idx': int(starts_np[i]), 'channel': c, 'arm': arm,
                'lambda': float(lam[i, c]), 'gain': float(gain[i, c]),
                'final_se': float(se[i, c]), 'base_se': float(base_se[i, c])}
               for i in range(B_te.size(0)) for c in range(N_CHANNELS)]
        all_frames.append(pd.DataFrame(rows))
        print(f'[compute_q] {arm} rows:', len(rows))

    # Q4 via real model forward
    all_frames.append(q4_per_query_channel(device))
    print('[compute_q] Q4 rows:', len(all_frames[-1]))

    full = pd.concat(all_frames, ignore_index=True)
    full = full.merge(oracle_df, on=['query_start_idx', 'channel'], how='left')
    full.to_parquet(OUT_DIR / 'per_query_channel_all_arms.parquet', index=False)

    # oracle_calibration_comparison.csv
    calib_rows = []
    for arm in ('Q1', 'Q2', 'Q3', 'Q4', 'Q5'):
        sub = full[full['arm'] == arm]
        mae = (sub['lambda'] - sub['oracle_lambda']).abs().mean()
        if sub['lambda'].std() > 1e-9:
            pearson = sub['lambda'].corr(sub['oracle_lambda'], method='pearson')
            spearman = sub['lambda'].corr(sub['oracle_lambda'], method='spearman')
        else:
            pearson, spearman = float('nan'), float('nan')  # constant predictor -- correlation undefined
        calib_rows.append({'arm': arm, 'mae_vs_oracle': mae, 'pearson_vs_oracle': pearson,
                           'spearman_vs_oracle': spearman})
    pd.DataFrame(calib_rows).to_csv(OUT_DIR / 'oracle_calibration_comparison.csv', index=False)

    # realized_gain.csv
    gain_rows = []
    for arm in ('Q1', 'Q2', 'Q3', 'Q4', 'Q5'):
        sub = full[full['arm'] == arm]
        gain_rows.append({'arm': arm, 'mean_gain': sub['gain'].mean(), 'median_gain': sub['gain'].median(),
                          'frac_gain_gt_0': (sub['gain'] > 0).mean(), 'frac_gain_lt_0': (sub['gain'] < 0).mean()})
    pd.DataFrame(gain_rows).to_csv(OUT_DIR / 'realized_gain.csv', index=False)

    # lambda-bin analysis (PART 17)
    bins = [(i / 10, (i + 1) / 10) for i in range(10)]
    bin_rows = []
    for arm in ('Q1', 'Q2', 'Q3', 'Q4', 'Q5'):
        sub = full[full['arm'] == arm]
        for lo, hi in bins:
            m = (sub['lambda'] >= lo) & (sub['lambda'] < hi if hi < 1.0 else sub['lambda'] <= hi)
            b = sub[m]
            bin_rows.append({'arm': arm, 'bin_lo': lo, 'bin_hi': hi, 'predicted_count': len(b),
                             'oracle_mean_lambda': b['oracle_lambda'].mean() if len(b) else float('nan'),
                             'realized_gain': b['gain'].mean() if len(b) else float('nan')})
    pd.DataFrame(bin_rows).to_csv(OUT_DIR / 'lambda_bin_analysis.csv', index=False)

    print('[compute_q] wrote per_query_channel_all_arms.parquet, oracle_calibration_comparison.csv, '
         'realized_gain.csv, lambda_bin_analysis.csv')
    print(pd.DataFrame(calib_rows).to_string(index=False))
    print(pd.DataFrame(gain_rows).to_string(index=False))


if __name__ == '__main__':
    main()
