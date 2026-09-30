#!/usr/bin/env python3
"""TRACK-P-FUSION-SEMANTICS-AUDIT01 PART 20 -- paired bootstrap,
query_start_idx unit, 10,000 reps, seed=0.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_o_frozen_base_consumer01 import CACHE_DIRS as O_CACHE_DIRS
from scripts.train_p_fusion_semantics01 import CACHE_DIRS as P_CACHE_DIRS
from scripts.train_p_fusion_semantics01 import S2_720, build_fresh_stage2_mixture
from scripts.train_setlossctrl_stage2_retrain02 import build_fresh_stage2, load_cache_as_lookup

OUT_DIR = REPO_ROOT / 'results/TRACK-P-FUSION-SEMANTICS-AUDIT01/ETTh1_720'
N_BOOT, SEED = 10000, 0

O_CKPT_DIR = REPO_ROOT / 'checkpoints/track_o_frozen_base_retrieval_consumer01/ETTh1_720'
P_CKPT_DIR = REPO_ROOT / 'checkpoints/track_p_fusion_semantics_audit01/ETTh1_720'


@torch.no_grad()
def per_query_mse_learned(ckpt_path, cache_dir, device, fusion_mode='residual'):
    bl = torch.load(ckpt_path, map_location=device)
    if fusion_mode == 'mixture':
        exp, args, model, host_ck = build_fresh_stage2_mixture(S2_720, seed=0, fusion_mode='mixture')
    else:
        exp, args, model, host_ck = build_fresh_stage2(S2_720, seed=0)
    model.to(device)
    exp._ensure_memory()
    exp._build_key_bank(force=True)
    model.load_state_dict(bl['model_state_dict'])
    model.eval()

    cache, lut = load_cache_as_lookup(cache_dir / 'test.pt')
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


def per_query_mse_fixed(arm):
    d = torch.load(OUT_DIR / arm / 'per_query_se.pt', map_location='cpu')
    return dict(zip(d['batch_start_idx'].tolist(), d['se_per_query'].tolist()))


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
    per_arm = {}
    per_arm['P0'] = per_query_mse_base(device)
    print(f'[p_bootstrap] P0: n={len(per_arm["P0"])} mean={np.mean(list(per_arm["P0"].values())):.6f}')
    per_arm['P1'] = per_query_mse_learned(O_CKPT_DIR / 'O1_J1_uniform' / 'checkpoint.pth', O_CACHE_DIRS['O1_J1_uniform'], device)
    print(f'[p_bootstrap] P1: n={len(per_arm["P1"])} mean={np.mean(list(per_arm["P1"].values())):.6f}')
    per_arm['P2'] = per_query_mse_learned(O_CKPT_DIR / 'O3_M2_uniform' / 'checkpoint.pth', O_CACHE_DIRS['O3_M2_uniform'], device)
    print(f'[p_bootstrap] P2: n={len(per_arm["P2"])} mean={np.mean(list(per_arm["P2"].values())):.6f}')
    per_arm['P3'] = per_query_mse_learned(P_CKPT_DIR / 'P3_J1_mixture' / 'checkpoint.pth', P_CACHE_DIRS['P3_J1_mixture'], device, fusion_mode='mixture')
    print(f'[p_bootstrap] P3: n={len(per_arm["P3"])} mean={np.mean(list(per_arm["P3"].values())):.6f}')
    per_arm['P4'] = per_query_mse_learned(P_CKPT_DIR / 'P4_M2_mixture' / 'checkpoint.pth', P_CACHE_DIRS['P4_M2_mixture'], device, fusion_mode='mixture')
    print(f'[p_bootstrap] P4: n={len(per_arm["P4"])} mean={np.mean(list(per_arm["P4"].values())):.6f}')
    per_arm['P5'] = per_query_mse_fixed('P5_J1_fixed_mixture')
    print(f'[p_bootstrap] P5: n={len(per_arm["P5"])} mean={np.mean(list(per_arm["P5"].values())):.6f}')
    per_arm['P6'] = per_query_mse_fixed('P6_M2_fixed_mixture')
    print(f'[p_bootstrap] P6: n={len(per_arm["P6"])} mean={np.mean(list(per_arm["P6"].values())):.6f}')

    common_ids = sorted(set.intersection(*(set(d.keys()) for d in per_arm.values())))
    print(f'[p_bootstrap] common queries: {len(common_ids)}')
    arrs = {a: np.array([per_arm[a][q] for q in common_ids]) for a in per_arm}

    comparisons = [('P3', 'P1'), ('P4', 'P2'), ('P3', 'P5'), ('P4', 'P6'), ('P4', 'P3'),
                  ('P3', 'P0'), ('P4', 'P0'), ('P5', 'P0'), ('P6', 'P0')]
    results = {}
    for a, b in comparisons:
        diffs = arrs[a] - arrs[b]
        res = paired_bootstrap(diffs, N_BOOT, SEED)
        results[f'{a}_minus_{b}'] = res
        print(f'[p_bootstrap] {a} - {b}: mean_diff={res["mean_diff"]:.6f} '
             f'CI=[{res["ci_lo_2.5"]:.6f}, {res["ci_hi_97.5"]:.6f}] excludes_zero={res["ci_excludes_zero"]}')

    out = {'n_boot': N_BOOT, 'seed': SEED, 'resampling_unit': 'query_start_idx',
          'n_common_queries': len(common_ids), 'per_arm_mean_mse': {a: float(arrs[a].mean()) for a in per_arm},
          'comparisons': results}
    (OUT_DIR / 'bootstrap.json').write_text(json.dumps(out, indent=2))
    print('[p_bootstrap] wrote bootstrap.json')


if __name__ == '__main__':
    main()
