#!/usr/bin/env python3
"""TRACK-O-FROZEN-BASE-RETRIEVAL-CONSUMER01 PART 21 -- paired bootstrap,
query_start_idx unit, 10,000 reps, seed=0, all 7 channels resampled
jointly per query (per-query MSE already averages over channel+horizon).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_o_frozen_base_consumer01 import CACHE_DIRS, S2_720
from scripts.train_setlossctrl_stage2_retrain02 import build_fresh_stage2, load_cache_as_lookup

ARMS = ('O1_J1_uniform', 'O2_J1_host', 'O3_M2_uniform', 'O4_M2_host')
CKPT_DIR = REPO_ROOT / 'checkpoints/track_o_frozen_base_retrieval_consumer01/ETTh1_720'
OUT_DIR = REPO_ROOT / 'results/TRACK-O-FROZEN-BASE-RETRIEVAL-CONSUMER01/ETTh1_720'
N_BOOT = 10000
SEED = 0


@torch.no_grad()
def per_query_mse_arm(arm, device):
    bl = torch.load(CKPT_DIR / arm / 'checkpoint.pth', map_location=device)
    exp, args, model, host_ck = build_fresh_stage2(S2_720, seed=0)
    model.to(device)
    exp._ensure_memory()
    exp._build_key_bank(force=True)
    model.load_state_dict(bl['model_state_dict'])
    model.eval()

    cache, lut = load_cache_as_lookup(CACHE_DIRS[arm] / 'test.pt')
    _, test_loader = exp._get_data(flag='test', shuffle=False)

    starts, per_q = [], []
    for batch_x, batch_y, batch_start_idx in test_loader:
        batch_x, batch_y, batch_start_idx = exp._move_batch(batch_x, batch_y, batch_start_idx)
        cand_mask, counts = exp._candidate_mask(batch_start_idx)
        valid_query = counts.to(batch_x.device) > 0
        rows = [lut[int(s)] for s in batch_start_idx.tolist()]
        idx = torch.tensor(rows, dtype=torch.long)
        rcache = {'relation_outputs': cache['relation_outputs'][idx].to(device),
                  'relation_query_embs': cache['relation_query_embs'][idx].to(device),
                  'query_offset': cache['query_offset'][idx].to(device)}
        y_final, y_base, y_ret, beta, lam, debug = model.forward_from_retrieval_values(
            rcache['relation_outputs'], batch_x=batch_x, retrieval_cache=rcache,
            memory_y=exp.memory_y, valid_mask=cand_mask, key_bank=exp.key_bank,
            memory_x_last=exp.memory_x_last, target_y=batch_y)
        se_per_q = ((y_final - batch_y) ** 2).mean(dim=(1, 2))
        for b in range(batch_x.size(0)):
            if bool(valid_query[b]):
                starts.append(int(batch_start_idx[b]))
                per_q.append(float(se_per_q[b]))
    return dict(zip(starts, per_q))


@torch.no_grad()
def per_query_mse_base(device):
    d = torch.load(REPO_ROOT / 'results/TRACK-N-FORECAST-CONDITIONAL-UTILITY01/ETTh1_720/'
                   'common_base_predictions/base_predictions_test.pt', map_location='cpu')
    exp, args, model, host_ck = build_fresh_stage2(S2_720, seed=0)
    _, test_loader = exp._get_data(flag='test', shuffle=False)
    lut = {int(s): i for i, s in enumerate(d['batch_start_idx'].tolist())}
    starts, per_q = [], []
    for batch_x, batch_y, batch_start_idx in test_loader:
        idx = [lut[int(s)] for s in batch_start_idx.tolist()]
        b_q = d['base_predictions'][idx]
        se = ((b_q - batch_y.float()) ** 2).mean(dim=(1, 2))
        for b in range(batch_x.size(0)):
            starts.append(int(batch_start_idx[b]))
            per_q.append(float(se[b]))
    return dict(zip(starts, per_q))


def paired_bootstrap(diffs, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(diffs)
    diffs = np.asarray(diffs)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = diffs[idx].mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return {'mean_diff': float(diffs.mean()), 'ci_lo_2.5': float(lo), 'ci_hi_97.5': float(hi),
           'ci_excludes_zero': bool(lo > 0 or hi < 0), 'n_queries': n}


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    per_arm = {'O0_base': per_query_mse_base(device)}
    for arm in ARMS:
        per_arm[arm] = per_query_mse_arm(arm, device)
        print(f'[o_bootstrap] {arm}: n={len(per_arm[arm])} mean_mse={np.mean(list(per_arm[arm].values())):.6f}')
    print(f'[o_bootstrap] O0_base: n={len(per_arm["O0_base"])} mean_mse={np.mean(list(per_arm["O0_base"].values())):.6f}')

    common_ids = sorted(set.intersection(*(set(d.keys()) for d in per_arm.values())))
    arrs = {a: np.array([per_arm[a][q] for q in common_ids]) for a in per_arm}
    print(f'[o_bootstrap] common queries: {len(common_ids)}')

    comparisons = [('O1_J1_uniform', 'O0_base'), ('O2_J1_host', 'O0_base'), ('O3_M2_uniform', 'O0_base'),
                  ('O4_M2_host', 'O0_base'), ('O3_M2_uniform', 'O1_J1_uniform'), ('O4_M2_host', 'O2_J1_host'),
                  ('O3_M2_uniform', 'O4_M2_host'), ('O1_J1_uniform', 'O2_J1_host')]
    results = {}
    for a, b in comparisons:
        diffs = arrs[a] - arrs[b]
        res = paired_bootstrap(diffs, N_BOOT, SEED)
        results[f'{a}_minus_{b}'] = res
        print(f'[o_bootstrap] {a} - {b}: mean_diff={res["mean_diff"]:.6f} '
             f'CI=[{res["ci_lo_2.5"]:.6f}, {res["ci_hi_97.5"]:.6f}] excludes_zero={res["ci_excludes_zero"]}')

    # interaction: (O4-O3) - (O2-O1)
    diff_host_penalty_M2 = arrs['O4_M2_host'] - arrs['O3_M2_uniform']
    diff_host_penalty_J1 = arrs['O2_J1_host'] - arrs['O1_J1_uniform']
    interaction = diff_host_penalty_M2 - diff_host_penalty_J1
    interaction_res = paired_bootstrap(interaction, N_BOOT, SEED)

    out = {'n_boot': N_BOOT, 'seed': SEED, 'resampling_unit': 'query_start_idx',
          'n_common_queries': len(common_ids), 'per_arm_mean_mse': {a: float(arrs[a].mean()) for a in per_arm},
          'comparisons': results}
    (OUT_DIR / 'bootstrap.json').write_text(json.dumps(out, indent=2))
    (OUT_DIR / 'interaction_analysis.json').write_text(json.dumps(
        {'definition': '(O4_M2_host - O3_M2_uniform) - (O2_J1_host - O1_J1_uniform)',
         'interpretation': 'positive => Host penalty is LARGER for M2 than for J1',
         **interaction_res}, indent=2))
    print(f'[o_bootstrap] interaction=(O4-O3)-(O2-O1): mean_diff={interaction_res["mean_diff"]:.6f} '
         f'CI=[{interaction_res["ci_lo_2.5"]:.6f}, {interaction_res["ci_hi_97.5"]:.6f}] '
         f'excludes_zero={interaction_res["ci_excludes_zero"]}')
    print(f'[o_bootstrap] wrote bootstrap.json, interaction_analysis.json')


if __name__ == '__main__':
    main()
