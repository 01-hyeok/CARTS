#!/usr/bin/env python3
"""TRACK-S-KL-CONTRIBUTION-DECOMPOSITION01 PART 19 -- paired bootstrap
(query_start_idx unit, 10,000 reps) comparing Original KL against
Cosine/J1/M2 for one setting. Reuses TRACK-R's already-trained
Cosine/J1/M2 gates + base (read-only) plus TRACK-S's own Original KL
gate -- NO retraining.
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


def per_query_se(base, gate, exp, cache_dir, device):
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
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--horizon', type=int, required=True)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.seq_len, 'batch_size': 32, 'seed': cli.seed,
    })
    channels = int(args.enc_in)

    r_root = REPO_ROOT / f'results/TRACK-R-FINAL-METHOD-GENERALIZATION01/{cli.dataset}/H{cli.horizon}/seed{cli.seed}'
    r_ckroot = REPO_ROOT / f'checkpoints/track_r_final_method_generalization01/{cli.dataset}/H{cli.horizon}/seed{cli.seed}'
    s_root = REPO_ROOT / f'results/TRACK-S-KL-CONTRIBUTION-DECOMPOSITION01/{cli.dataset}/H{cli.horizon}/seed{cli.seed}'

    base = BaseForecastHead(seq_len=cli.seq_len, pred_len=cli.pred_len, channels=channels,
                            mode='per_channel_linear').to(device)
    bl = torch.load(r_ckroot / 'base/checkpoint.pth', map_location=device)
    base.load_state_dict(bl['model_state_dict'])
    base.eval()

    arm_paths = {
        'Cosine': (r_root / 'cosine', r_root / 'cosine' / 'stage2'),
        'OriginalKL': (s_root / 'original_kl' / 'cache', s_root / 'original_kl' / 'stage2'),
        'J1': (r_root / 'J1' / 'cache', r_root / 'J1' / 'stage2'),
        'M2': (r_root / 'M2' / 'cache', r_root / 'M2' / 'stage2'),
    }
    per_arm = {}
    for name, (cache_dir, gate_dir) in arm_paths.items():
        gate = GlobalLambdaGate().to(device)
        gck = torch.load(gate_dir / 'checkpoint.pth', map_location=device)
        gate.load_state_dict(gck['gate_state_dict'])
        gate.eval()
        per_arm[name] = per_query_se(base, gate, exp, cache_dir, device)
        print(f'[compute_s_bootstrap] {name}: n={len(per_arm[name])} mean={np.mean(list(per_arm[name].values())):.6f}')

    common = sorted(set.intersection(*(set(d.keys()) for d in per_arm.values())))
    arrs = {a: [per_arm[a][q] for q in common] for a in per_arm}

    comparisons = [('OriginalKL', 'Cosine'), ('J1', 'OriginalKL'), ('M2', 'J1'), ('M2', 'OriginalKL')]
    out = {'dataset': cli.dataset, 'horizon': cli.horizon, 'seed': cli.seed, 'n_boot': 10000,
          'resampling_unit': 'query_start_idx', 'comparisons': {}}
    for a, b in comparisons:
        res = paired_bootstrap(arrs[a], arrs[b])
        out['comparisons'][f'{a}_minus_{b}'] = res
        print(f'[compute_s_bootstrap] {cli.dataset} H{cli.horizon} seed{cli.seed}: {a}-{b} '
             f'mean_diff={res["mean_diff"]:.6f} CI=[{res["ci_lo_2.5"]:.6f},{res["ci_hi_97.5"]:.6f}] '
             f'sig={res["ci_excludes_zero"]}')

    out_path = s_root / 'bootstrap.json'
    out_path.write_text(json.dumps(out, indent=2))
    print(f'[compute_s_bootstrap] wrote {out_path}')


if __name__ == '__main__':
    main()
