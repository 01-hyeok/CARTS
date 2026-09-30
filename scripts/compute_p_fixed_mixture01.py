#!/usr/bin/env python3
"""TRACK-P-FUSION-SEMANTICS-AUDIT01 PART 8 (P5/P6) -- NO TRAINING. Fixed
-lambda mixture fusion `Y_hat = B + lambda*(R-B)`, evaluated directly
from cached common-base predictions and cached Uniform retrieval
aggregates (both reused verbatim from TRACK-N). `lambda` is the
TRACK-N validation-selected global lambda (J1=0.37, M2=0.42) -- NOT
re-selected here, per PART 14's reproduction-gate requirement.
"""
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_setlossctrl_stage2_retrain02 import build_fresh_stage2, load_cache_as_lookup, restore_absolute

S2_720 = ('checkpoints/stage2/ETTh1/seq720_pred720/stage2_carts_softset_s2_ETTh1_720_S0_wce_RelationStage2_ETTh1_'
         'ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_'
         'sl720_pl720_0/checkpoint.pth')
BASE_DIR = REPO_ROOT / 'results/TRACK-N-FORECAST-CONDITIONAL-UTILITY01/ETTh1_720/common_base_predictions'
CACHE_DIRS = {'P5_J1_fixed_mixture': (REPO_ROOT / 'results/TRACK-N-FORECAST-CONDITIONAL-UTILITY01/gate_only/cache/ETTh1_720/G0_J1', 0.37),
             'P6_M2_fixed_mixture': (REPO_ROOT / 'results/TRACK-N-FORECAST-CONDITIONAL-UTILITY01/gate_only/cache/ETTh1_720/G2_M2', 0.42)}
OUT_DIR = REPO_ROOT / 'results/TRACK-P-FUSION-SEMANTICS-AUDIT01/ETTh1_720'


@torch.no_grad()
def per_query_arm(arm, cache_dir, lam, split, device):
    exp, args, model, host_ck = build_fresh_stage2(S2_720, seed=0)
    _, loader = exp._get_data(flag=split, shuffle=False)
    base = torch.load(BASE_DIR / f'base_predictions_{split}.pt', map_location='cpu')
    base_lut = {int(s): i for i, s in enumerate(base['batch_start_idx'].tolist())}
    cache, lut = load_cache_as_lookup(cache_dir / f'{split}.pt')

    starts, se_list, ae_list = [], [], []
    for batch_x, batch_y, batch_start_idx in loader:
        b_idx = [base_lut[int(s)] for s in batch_start_idx.tolist()]
        b_q = base['base_predictions'][b_idx]  # [B, H, C]
        r_idx = [lut[int(s)] for s in batch_start_idx.tolist()]
        delta = cache['relation_outputs'][r_idx][:, :, 0, :]
        offset = cache['query_offset'][r_idx]
        r_abs = restore_absolute(delta, offset).permute(0, 2, 1)  # [B, H, C]
        fused = b_q + lam * (r_abs - b_q)
        se = ((fused - batch_y.float()) ** 2).mean(dim=(1, 2))
        ae = (fused - batch_y.float()).abs().mean(dim=(1, 2))
        for b in range(batch_x.size(0)):
            starts.append(int(batch_start_idx[b]))
            se_list.append(float(se[b]))
            ae_list.append(float(ae[b]))
    return starts, se_list, ae_list


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    expected = {'P5_J1_fixed_mixture': 0.463349, 'P6_M2_fixed_mixture': 0.463969}
    for arm, (cache_dir, lam) in CACHE_DIRS.items():
        starts, se_list, ae_list = per_query_arm(arm, cache_dir, lam, 'test', device)
        mse = sum(se_list) / len(se_list)
        mae = sum(ae_list) / len(ae_list)
        diff = abs(mse - expected[arm])
        out = {'arm': arm, 'lambda': lam, 'final_mse': mse, 'final_mae': mae,
              'expected_mse': expected[arm], 'absdiff': diff, 'reproduction_pass': bool(diff <= 1e-5),
              'n_queries': len(se_list)}
        (OUT_DIR / arm / 'metrics.json').write_text(json.dumps(out, indent=2))
        print(f'[compute_p_fixed] {arm}: lambda={lam} mse={mse:.6f} mae={mae:.6f} '
             f'expected={expected[arm]} diff={diff:.2e} PASS={diff<=1e-5}')
        # save per-query for the bootstrap script to reuse
        torch.save({'batch_start_idx': torch.tensor(starts), 'se_per_query': torch.tensor(se_list)},
                  OUT_DIR / arm / 'per_query_se.pt')


if __name__ == '__main__':
    main()
