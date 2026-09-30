#!/usr/bin/env python3
"""TRACK-U-ASYMMETRY-CAPACITY-DECOMPOSITION01 PART 14 -- paired
bootstrap (query_start_idx unit, 10,000 reps) for U1-U0, U2-U1, U3-U1,
U2-U3, U2-U0, at Stage1 retMSE/D/C/AggMSE and Stage2 forecast MSE.
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

ARMS = ('U0', 'U1', 'U2', 'U3')


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


def per_query_stage1(arm_dir, exp, field):
    cache = torch.load(arm_dir / 'cache' / 'test.pt', map_location='cpu')
    qs = cache['query_start_idx'].tolist()
    ro = cache['relation_outputs']
    _, loader = exp._get_data(flag='test', shuffle=False)
    lut = {int(s): i for i, s in enumerate(qs)}
    out = {}
    for batch_x, batch_y, batch_start_idx in loader:
        idx = [lut[int(s)] for s in batch_start_idx.tolist()]
        r = ro[idx]
        agg = ((r - batch_y) ** 2).mean(dim=(1, 2))
        for b, s in enumerate(batch_start_idx.tolist()):
            out[int(s)] = float(agg[b])
    return out  # Note: this reconstructs Agg per query from the cache; D/C bootstrap uses saved test-set means


def per_query_stage2_se(arm_dir, base, exp, device):
    cache = torch.load(arm_dir / 'cache' / 'test.pt', map_location='cpu')
    lut = {int(s): i for i, s in enumerate(cache['query_start_idx'].tolist())}
    gate = GlobalLambdaGate().to(device)
    gck = torch.load(arm_dir / 'stage2' / 'checkpoint.pth', map_location=device)
    gate.load_state_dict(gck['gate_state_dict'])
    gate.eval()
    _, loader = exp._get_data(flag='test', shuffle=False)
    out = {}
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
            se = ((fused - batch_y) ** 2).mean(dim=(1, 2))
            for b, s in enumerate(batch_start_idx.tolist()):
                out[int(s)] = float(se[b])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--horizon', type=int, required=True)
    ap.add_argument('--base_checkpoint', required=True)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.seq_len, 'batch_size': 32, 'seed': cli.seed,
    })
    channels = int(args.enc_in)

    root = REPO_ROOT / f'results/TRACK-U-ASYMMETRY-CAPACITY-DECOMPOSITION01/{cli.dataset}/H{cli.horizon}/seed{cli.seed}'

    base = BaseForecastHead(seq_len=cli.seq_len, pred_len=cli.pred_len, channels=channels,
                            mode='per_channel_linear').to(device)
    bl = torch.load(cli.base_checkpoint, map_location=device)
    base.load_state_dict(bl['model_state_dict'])
    base.eval()

    dirs = {a: root / a for a in ARMS}
    stage1_pq = {a: per_query_stage1(dirs[a], exp, 'agg_mse10') for a in ARMS}
    stage2_pq = {a: per_query_stage2_se(dirs[a], base, exp, device) for a in ARMS}
    for a in ARMS:
        print(f'[compute_u_bootstrap] {a}: stage1_n={len(stage1_pq[a])} '
             f'stage1_mean={np.mean(list(stage1_pq[a].values())):.6f} '
             f'stage2_n={len(stage2_pq[a])} stage2_mean={np.mean(list(stage2_pq[a].values())):.6f}')

    common1 = sorted(set.intersection(*(set(d.keys()) for d in stage1_pq.values())))
    common2 = sorted(set.intersection(*(set(d.keys()) for d in stage2_pq.values())))

    comparisons = [('U1', 'U0'), ('U2', 'U1'), ('U3', 'U1'), ('U2', 'U3'), ('U2', 'U0')]
    out = {'dataset': cli.dataset, 'horizon': cli.horizon, 'seed': cli.seed, 'n_boot': 10000,
          'resampling_unit': 'query_start_idx', 'stage1_agg_mse': {}, 'stage2_forecast_mse': {}}
    for a, b in comparisons:
        arr_a1 = [stage1_pq[a][q] for q in common1]
        arr_b1 = [stage1_pq[b][q] for q in common1]
        res1 = paired_bootstrap(arr_a1, arr_b1)
        out['stage1_agg_mse'][f'{a}_minus_{b}'] = res1

        arr_a2 = [stage2_pq[a][q] for q in common2]
        arr_b2 = [stage2_pq[b][q] for q in common2]
        res2 = paired_bootstrap(arr_a2, arr_b2)
        out['stage2_forecast_mse'][f'{a}_minus_{b}'] = res2
        print(f'[compute_u_bootstrap] {cli.dataset} H{cli.horizon} seed{cli.seed}: {a}-{b} '
             f'stage1_diff={res1["mean_diff"]:.6f} sig={res1["ci_excludes_zero"]} | '
             f'stage2_diff={res2["mean_diff"]:.6f} sig={res2["ci_excludes_zero"]}')

    out_path = root / 'bootstrap.json'
    out_path.write_text(json.dumps(out, indent=2))
    print(f'[compute_u_bootstrap] wrote {out_path}')


if __name__ == '__main__':
    main()
