#!/usr/bin/env python3
"""TRACK-R-FINAL-METHOD-GENERALIZATION01 PART 22 -- per-setting paired
bootstrap (query_start_idx unit, 10,000 reps, seed=0 for the bootstrap
RNG itself). Computes M2-vs-Base, M2-vs-Cosine, M2-vs-J1, J1-vs-Cosine
for one (dataset, horizon, seed) setting using the already-trained
gate checkpoints and cached (B, R, Y) tensors -- NO retraining.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage2 import BaseForecastHead
from scripts.train_margutil01 import build_experiment
from scripts.train_r_stage2_lambda01 import GlobalLambdaGate
from scripts.train_setlossctrl_stage2_retrain02 import restore_absolute


def per_query_se(base, gate, exp, cache_dir, device, n_channels):
    _, loader = exp._get_data(flag='test', shuffle=False)
    cache = torch.load(Path(cache_dir) / 'test.pt', map_location='cpu')
    lut = {int(s): i for i, s in enumerate(cache['query_start_idx'].tolist())}
    starts, se = [], []
    with torch.no_grad():
        for batch_x, batch_y, batch_start_idx in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            offset = batch_x[:, -1:, :].detach()
            b_q = base(batch_x) + offset
            idx = [lut[int(s)] for s in batch_start_idx.tolist()]
            r_q = cache['relation_outputs'][idx].to(device)
            if gate is None:
                fused = b_q
            else:
                lam = gate()
                fused = b_q + lam * (r_q - b_q)
            se_q = ((fused - batch_y) ** 2).mean(dim=(1, 2))
            for b in range(batch_x.size(0)):
                starts.append(int(batch_start_idx[b]))
            se.append(se_q.cpu())
    return dict(zip(starts, torch.cat(se).tolist()))


def paired_bootstrap(a, b, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    diffs = np.asarray(a) - np.asarray(b)
    n = len(diffs)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot[i] = diffs[idx].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {'mean_diff': float(diffs.mean()), 'ci_lo_2.5': float(lo), 'ci_hi_97.5': float(hi),
           'ci_excludes_zero': bool(lo > 0 or hi < 0), 'n_queries': n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--setting_root', required=True)
    ap.add_argument('--checkpoint_root', required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--horizon', type=int, required=True)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.seq_len, 'batch_size': 32, 'seed': cli.seed,
    })
    channels = int(args.enc_in)
    root = Path(cli.setting_root)
    ckroot = Path(cli.checkpoint_root)

    base = BaseForecastHead(seq_len=cli.seq_len, pred_len=cli.pred_len, channels=channels,
                            mode='per_channel_linear').to(device)
    bl = torch.load(ckroot / 'base/checkpoint.pth', map_location=device)
    base.load_state_dict(bl['model_state_dict'])
    base.eval()

    per_arm = {'B0': per_query_se(base, None, exp, root / 'cosine', device, channels)}
    for arm, cache_sub in (('B1', 'cosine'), ('B2', 'J1/cache'), ('B3', 'M2/cache')):
        gate = GlobalLambdaGate().to(device)
        gck = torch.load(root / cache_sub.split('/')[0] / 'stage2' / 'checkpoint.pth'
                         if '/' in cache_sub else root / cache_sub / 'stage2' / 'checkpoint.pth', map_location=device)
        gate.load_state_dict(gck['gate_state_dict'])
        gate.eval()
        per_arm[arm] = per_query_se(base, gate, exp, root / cache_sub, device, channels)

    common = sorted(set.intersection(*(set(d.keys()) for d in per_arm.values())))
    arrs = {a: [per_arm[a][q] for q in common] for a in per_arm}

    comparisons = [('B3', 'B0'), ('B3', 'B1'), ('B3', 'B2'), ('B2', 'B1'), ('B1', 'B0'), ('B2', 'B0')]
    out = {'dataset': cli.dataset, 'horizon': cli.horizon, 'seed': cli.seed,
          'resampling_unit': 'query_start_idx', 'n_boot': 10000,
          'per_arm_mean_mse': {a: float(np.mean(arrs[a])) for a in arrs}, 'comparisons': {}}
    for a, b in comparisons:
        res = paired_bootstrap(arrs[a], arrs[b])
        out['comparisons'][f'{a}_minus_{b}'] = res
        print(f'[compute_r_bootstrap] {cli.dataset} H{cli.horizon} seed{cli.seed}: {a}-{b} '
             f'mean_diff={res["mean_diff"]:.6f} CI=[{res["ci_lo_2.5"]:.6f},{res["ci_hi_97.5"]:.6f}] '
             f'sig={res["ci_excludes_zero"]}')

    out_path = root / 'bootstrap.json'
    out_path.write_text(json.dumps(out, indent=2))
    print(f'[compute_r_bootstrap] wrote {out_path}')


if __name__ == '__main__':
    main()
