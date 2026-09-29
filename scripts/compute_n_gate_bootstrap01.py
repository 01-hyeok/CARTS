#!/usr/bin/env python3
"""TRACK-N-FORECAST-CONDITIONAL-UTILITY01 PART 18-19 -- paired bootstrap
for the Gate-Only arms (G0=J1, G1=K2, G2=M2), query_start_idx unit,
10,000 reps -- same convention as TRACK-M's own Stage2 bootstrap."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_n_gate_only01 import S2_720, S0_CKPT, freeze_all_but_gate
from scripts.train_setlossctrl_stage2_retrain02 import build_fresh_stage2, load_cache_as_lookup

ARMS = ('G0_J1', 'G1_K2', 'G2_M2')
CKPT_DIR = REPO_ROOT / 'checkpoints/track_n_forecast_conditional_utility01/gate_only/ETTh1_720'
CACHE_DIR = REPO_ROOT / 'results/TRACK-N-FORECAST-CONDITIONAL-UTILITY01/gate_only/cache/ETTh1_720'
OUT_DIR = REPO_ROOT / 'results/TRACK-N-FORECAST-CONDITIONAL-UTILITY01/gate_only'
N_BOOT = 10000
SEED = 0


@torch.no_grad()
def per_query_mse(arm, device):
    bl = torch.load(CKPT_DIR / arm / 'checkpoint.pth', map_location=device)
    exp, args, model, host_ck = build_fresh_stage2(S2_720, seed=0)
    model.to(device)
    exp._ensure_memory()
    exp._build_key_bank(force=True)
    model.load_state_dict(bl['model_state_dict'])
    model.eval()

    cache, lut = load_cache_as_lookup(CACHE_DIR / arm / 'test.pt')
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
    for arm in ARMS:
        per_arm[arm] = per_query_mse(arm, device)
        print(f'[gate_bootstrap] {arm}: n={len(per_arm[arm])} mean_mse={np.mean(list(per_arm[arm].values())):.6f}')

    common_ids = sorted(set.intersection(*(set(d.keys()) for d in per_arm.values())))
    arrs = {arm: np.array([per_arm[arm][q] for q in common_ids]) for arm in ARMS}

    comparisons = [('G1_K2', 'G0_J1'), ('G2_M2', 'G0_J1'), ('G2_M2', 'G1_K2')]
    results = {}
    for a, b in comparisons:
        diffs = arrs[a] - arrs[b]
        res = paired_bootstrap(diffs, N_BOOT, SEED)
        results[f'{a}_minus_{b}'] = res
        print(f'[gate_bootstrap] {a} - {b}: mean_diff={res["mean_diff"]:.6f} '
             f'CI=[{res["ci_lo_2.5"]:.6f}, {res["ci_hi_97.5"]:.6f}] excludes_zero={res["ci_excludes_zero"]}')

    out = {'n_boot': N_BOOT, 'seed': SEED, 'resampling_unit': 'query_start_idx',
          'n_common_queries': len(common_ids), 'per_arm_mean_mse': {a: float(arrs[a].mean()) for a in ARMS},
          'comparisons': results}
    (OUT_DIR / 'bootstrap.json').write_text(json.dumps(out, indent=2))
    print(f'[gate_bootstrap] wrote {OUT_DIR / "bootstrap.json"}')


if __name__ == '__main__':
    main()
