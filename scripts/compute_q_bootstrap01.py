#!/usr/bin/env python3
"""TRACK-Q-GATE-CAPACITY-CALIBRATION01 PART 19 -- paired bootstrap,
query_start_idx unit, 10,000 reps, seed=0, channels grouped (per-query
squared error already averages over the 7 channels + horizon).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_p_fusion_semantics01 import CACHE_DIRS as P_CACHE_DIRS
from scripts.train_p_fusion_semantics01 import build_fresh_stage2_mixture, S2_720
from scripts.train_q_gate_capacity01 import GATE_CLASSES, load_tensors
from scripts.train_setlossctrl_stage2_retrain02 import load_cache_as_lookup

OUT_DIR = REPO_ROOT / 'results/TRACK-Q-GATE-CAPACITY-CALIBRATION01/ETTh1_720'
N_BOOT, SEED = 10000, 0


@torch.no_grad()
def q4_per_query(device):
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
    starts, per_q = [], []
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
        se = ((y_final - batch_y) ** 2).mean(dim=(1, 2))
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
    te_starts, B_te, R_te, Y_te = load_tensors('test', device)
    starts_np = te_starts.numpy().tolist()

    per_arm = {}
    base_se = ((B_te - Y_te) ** 2).mean(dim=(1, 2))
    per_arm['Q0'] = dict(zip(starts_np, base_se.cpu().tolist()))

    lam1 = torch.full((B_te.size(0), 7), 0.42, device=device)
    fused1 = B_te + lam1.unsqueeze(1) * (R_te - B_te)
    se1 = ((fused1 - Y_te) ** 2).mean(dim=(1, 2))
    per_arm['Q1'] = dict(zip(starts_np, se1.cpu().tolist()))

    for gate_type, arm, arm_dir in (('global', 'Q2', 'Q2_trainable_global'),
                                    ('per_channel', 'Q3', 'Q3_per_channel'),
                                    ('query_prior', 'Q5', 'Q5_prior_query_gate')):
        ck = torch.load(OUT_DIR / arm_dir / 'checkpoint.pth', map_location=device)
        gate = GATE_CLASSES[gate_type]().to(device)
        gate.load_state_dict(ck['gate_state_dict'])
        gate.eval()
        with torch.no_grad():
            lam = gate(B_te, R_te)
            fused = B_te + lam.unsqueeze(1) * (R_te - B_te)
            se = ((fused - Y_te) ** 2).mean(dim=(1, 2))
        per_arm[arm] = dict(zip(starts_np, se.cpu().tolist()))

    per_arm['Q4'] = q4_per_query(device)

    for a in per_arm:
        print(f'[q_bootstrap] {a}: n={len(per_arm[a])} mean={np.mean(list(per_arm[a].values())):.6f}')

    common_ids = sorted(set.intersection(*(set(d.keys()) for d in per_arm.values())))
    print(f'[q_bootstrap] common queries: {len(common_ids)}')
    arrs = {a: np.array([per_arm[a][q] for q in common_ids]) for a in per_arm}

    comparisons = [('Q2', 'Q1'), ('Q3', 'Q1'), ('Q4', 'Q1'), ('Q5', 'Q1'),
                  ('Q3', 'Q2'), ('Q4', 'Q3'), ('Q5', 'Q4'),
                  ('Q2', 'Q0'), ('Q3', 'Q0'), ('Q4', 'Q0'), ('Q5', 'Q0')]
    results = {}
    for a, b in comparisons:
        diffs = arrs[a] - arrs[b]
        res = paired_bootstrap(diffs, N_BOOT, SEED)
        results[f'{a}_minus_{b}'] = res
        print(f'[q_bootstrap] {a} - {b}: mean_diff={res["mean_diff"]:.6f} '
             f'CI=[{res["ci_lo_2.5"]:.6f}, {res["ci_hi_97.5"]:.6f}] excludes_zero={res["ci_excludes_zero"]}')

    out = {'n_boot': N_BOOT, 'seed': SEED, 'resampling_unit': 'query_start_idx',
          'n_common_queries': len(common_ids), 'per_arm_mean_mse': {a: float(arrs[a].mean()) for a in per_arm},
          'comparisons': results}
    (OUT_DIR / 'bootstrap.json').write_text(json.dumps(out, indent=2))
    print('[q_bootstrap] wrote bootstrap.json')


if __name__ == '__main__':
    main()
