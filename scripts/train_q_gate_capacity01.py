#!/usr/bin/env python3
"""TRACK-Q-GATE-CAPACITY-CALIBRATION01 -- Q2 (trainable global lambda,
1 param), Q3 (trainable per-channel lambda, 7 params), Q5 (global-prior
query gate: lambda_q = sigmoid(b + MLP([B,R])), b initialized to
logit(0.42), MLP last layer zero-init so training starts at exactly
lambda_q=0.42 for every query).

Per AUDIT.md, `relation_mixer`/`Model` are never constructed: the cached
Uniform M2 retrieval aggregate `R` and the common frozen base `B` are
consumed directly by a small gate module in plain PyTorch (all Stage2
auxiliary loss terms are off for this host config, so the real training
objective IS exactly `mean((y_final-Y)**2)` -- verified in AUDIT.md).
Fixed retriever=M2, aggregation=Uniform, fusion=mixture throughout.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_setlossctrl_stage2_retrain02 import build_fresh_stage2, load_cache_as_lookup, restore_absolute

S2_720 = ('checkpoints/stage2/ETTh1/seq720_pred720/stage2_carts_softset_s2_ETTh1_720_S0_wce_RelationStage2_ETTh1_'
         'ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_'
         'sl720_pl720_0/checkpoint.pth')
BASE_DIR = REPO_ROOT / 'results/TRACK-N-FORECAST-CONDITIONAL-UTILITY01/ETTh1_720/common_base_predictions'
CACHE_DIR = REPO_ROOT / 'results/TRACK-N-FORECAST-CONDITIONAL-UTILITY01/gate_only/cache/ETTh1_720/G2_M2'
PRED_LEN = 720
N_CHANNELS = 7
PRIOR_LAMBDA = 0.42
BATCH_SIZE = 32
LR = 1e-3
TRAIN_EPOCHS = 10
PATIENCE = 5


class GlobalLambdaGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(1))

    def forward(self, B, R):  # B, R: [batch, H, C]
        lam = torch.sigmoid(self.a)
        return lam.expand(B.size(0), N_CHANNELS)  # [batch, C]


class PerChannelLambdaGate(nn.Module):
    def __init__(self, n_channels=N_CHANNELS):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(n_channels))

    def forward(self, B, R):
        lam = torch.sigmoid(self.a)  # [C]
        return lam.unsqueeze(0).expand(B.size(0), -1)  # [batch, C]


class PriorQueryGate(nn.Module):
    def __init__(self, pred_len=PRED_LEN, hidden=128, prior_lambda=PRIOR_LAMBDA):
        super().__init__()
        b0 = math.log(prior_lambda / (1.0 - prior_lambda))
        self.b = nn.Parameter(torch.tensor([b0]))
        self.mlp = nn.Sequential(nn.Linear(pred_len * 2, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, B, R):  # B, R: [batch, H, C]
        lams = []
        for c in range(B.size(-1)):
            feat = torch.cat([B[:, :, c], R[:, :, c]], dim=-1)  # [batch, 2H]
            delta = self.mlp(feat).squeeze(-1)  # [batch]
            lams.append(torch.sigmoid(self.b.squeeze(0) + delta))
        return torch.stack(lams, dim=-1)  # [batch, C]


GATE_CLASSES = {'global': GlobalLambdaGate, 'per_channel': PerChannelLambdaGate, 'query_prior': PriorQueryGate}


def load_tensors(split, device):
    """Returns (starts[N], B[N,H,C], R[N,H,C], Y[N,H,C]) aligned by query_start_idx order."""
    exp, args, model, host_ck = build_fresh_stage2(S2_720, seed=0)
    _, loader = exp._get_data(flag=split, shuffle=False)
    base = torch.load(BASE_DIR / f'base_predictions_{split}.pt', map_location='cpu')
    base_lut = {int(s): i for i, s in enumerate(base['batch_start_idx'].tolist())}
    cache, lut = load_cache_as_lookup(CACHE_DIR / f'{split}.pt')

    starts, b_list, r_list, y_list = [], [], [], []
    for batch_x, batch_y, batch_start_idx in loader:
        b_idx = [base_lut[int(s)] for s in batch_start_idx.tolist()]
        b_q = base['base_predictions'][b_idx]
        r_idx = [lut[int(s)] for s in batch_start_idx.tolist()]
        delta = cache['relation_outputs'][r_idx][:, :, 0, :]
        offset = cache['query_offset'][r_idx]
        r_abs = restore_absolute(delta, offset).permute(0, 2, 1)  # [B, H, C]
        for b in range(batch_x.size(0)):
            starts.append(int(batch_start_idx[b]))
        b_list.append(b_q)
        r_list.append(r_abs)
        y_list.append(batch_y.float())
    return (torch.tensor(starts), torch.cat(b_list).to(device), torch.cat(r_list).to(device),
           torch.cat(y_list).to(device))


def fuse(gate, B, R):
    lam = gate(B, R)  # [batch, C]
    fused = B + lam.unsqueeze(1) * (R - B)  # [batch, H, C]
    return fused, lam


def eval_split(gate, B, R, Y):
    with torch.no_grad():
        fused, lam = fuse(gate, B, R)
        se = (fused - Y) ** 2
        ae = (fused - Y).abs()
        mse = float(se.mean())
        mae = float(ae.mean())
        lam_flat = lam.flatten()
        q = torch.quantile(lam_flat, torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90], device=lam_flat.device))
        dist = {'gate_mean': float(lam_flat.mean()), 'gate_std': float(lam_flat.std()),
               'gate_median': float(q[2]), 'gate_p10': float(q[0]), 'gate_p25': float(q[1]),
               'gate_p75': float(q[3]), 'gate_p90': float(q[4]),
               'gate_lt_0.02': float((lam_flat < 0.02).float().mean()),
               'gate_gt_0.10': float((lam_flat > 0.10).float().mean()),
               'gate_gt_0.25': float((lam_flat > 0.25).float().mean()),
               'gate_gt_0.50': float((lam_flat > 0.50).float().mean())}
        return mse, mae, dist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gate_type', required=True, choices=tuple(GATE_CLASSES.keys()))
    ap.add_argument('--out_dir', default='results/TRACK-Q-GATE-CAPACITY-CALIBRATION01/ETTh1_720')
    ap.add_argument('--seed', type=int, default=0)
    cli = ap.parse_args()
    arm_name = {'global': 'Q2_trainable_global', 'per_channel': 'Q3_per_channel',
               'query_prior': 'Q5_prior_query_gate'}[cli.gate_type]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cli.seed)
    gen = torch.Generator().manual_seed(cli.seed)

    print(f'[track_q] {arm_name}: loading tensors...')
    tr_starts, B_tr, R_tr, Y_tr = load_tensors('train', device)
    va_starts, B_va, R_va, Y_va = load_tensors('val', device)
    te_starts, B_te, R_te, Y_te = load_tensors('test', device)
    print(f'[track_q] {arm_name}: n_train={B_tr.size(0)} n_val={B_va.size(0)} n_test={B_te.size(0)}')

    gate = GATE_CLASSES[cli.gate_type]().to(device)
    n_params = sum(p.numel() for p in gate.parameters())
    initial_state = {k: v.clone() for k, v in gate.state_dict().items()}
    if cli.gate_type == 'query_prior':
        init_lam = float(torch.sigmoid(gate.b.detach()))
        assert abs(init_lam - PRIOR_LAMBDA) < 1e-6, f'initial lambda {init_lam} != {PRIOR_LAMBDA}'
        assert float(gate.mlp[-1].weight.abs().sum()) == 0.0
        assert float(gate.mlp[-1].bias.abs().sum()) == 0.0

    optimizer = torch.optim.Adam(gate.parameters(), lr=LR)
    out_dir = Path(cli.out_dir) / arm_name
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {'gate_type': cli.gate_type, 'arm': arm_name, 'n_trainable_params': n_params,
             'seed': cli.seed, 'lr': LR, 'batch_size': BATCH_SIZE, 'train_epochs': TRAIN_EPOCHS,
             'patience': PATIENCE}
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2))
    print(f'[track_q] {arm_name}: n_params={n_params}')

    n_train = B_tr.size(0)
    trajectory = []
    best = {'val': float('inf'), 'epoch': -1, 'state': None}
    step = 0
    for epoch in range(1, TRAIN_EPOCHS + 1):
        perm = torch.randperm(n_train, generator=gen)
        tot_loss, n_batches = 0.0, 0
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            b_b, r_b, y_b = B_tr[idx], R_tr[idx], Y_tr[idx]
            optimizer.zero_grad()
            fused, lam = fuse(gate, b_b, r_b)
            loss = ((fused - y_b) ** 2).mean()
            loss.backward()
            grad_norm = sum(float(p.grad.abs().sum()) for p in gate.parameters() if p.grad is not None)
            optimizer.step()
            tot_loss += float(loss.detach())
            n_batches += 1
            step += 1
            if epoch == 1 and step <= 10:
                cur_lam = float(torch.sigmoid(gate.a).mean()) if cli.gate_type == 'global' else None
                trajectory.append({'step': step, 'epoch': epoch, 'train_loss': float(loss.detach()),
                                  'grad_norm': grad_norm, 'lambda_snapshot': cur_lam})
        train_loss = tot_loss / max(n_batches, 1)
        val_mse, val_mae, val_dist = eval_split(gate, B_va, R_va, Y_va)
        cur_lam_mean = val_dist['gate_mean']
        trajectory.append({'step': step, 'epoch': epoch, 'train_loss': train_loss, 'val_mse': val_mse,
                          'lambda_mean': cur_lam_mean, 'grad_norm': None})
        if val_mse < best['val']:
            best = {'val': val_mse, 'epoch': epoch, 'state': {k: v.clone() for k, v in gate.state_dict().items()}}
        print(f'[track_q] {arm_name} epoch {epoch} train_loss={train_loss:.6f} val_mse={val_mse:.6f} '
             f'lambda_mean={cur_lam_mean:.4f} (best={best["epoch"]}:{best["val"]:.6f})')
        if epoch - best['epoch'] >= PATIENCE:
            print(f'[track_q] early stop at epoch {epoch} (best={best["epoch"]})')
            break

    with open(out_dir / 'trajectory.csv', 'w', newline='') as fh:
        keys = sorted({k for r in trajectory for k in r})
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in trajectory:
            w.writerow(r)

    gate.load_state_dict(best['state'])
    torch.save({'gate_state_dict': gate.state_dict(), 'epoch': best['epoch'], 'val_mse': best['val'],
               'gate_type': cli.gate_type}, out_dir / 'checkpoint.pth')
    test_mse, test_mae, test_dist = eval_split(gate, B_te, R_te, Y_te)
    val_mse_best, val_mae_best, val_dist_best = eval_split(gate, B_va, R_va, Y_va)

    result = {'epoch': best['epoch'], 'n_trainable_params': n_params,
             'val': {'mse': val_mse_best, 'mae': val_mae_best, **val_dist_best},
             'test': {'final_mse': test_mse, 'final_mae': test_mae, **test_dist}}
    if cli.gate_type == 'per_channel':
        lam_c = torch.sigmoid(gate.a.detach()).cpu().tolist()
        result['lambda_per_channel'] = {str(c): lam_c[c] for c in range(N_CHANNELS)}
        (out_dir / 'lambda_channels.json').write_text(json.dumps(result['lambda_per_channel'], indent=2))
    if cli.gate_type == 'global':
        result['lambda_global'] = float(torch.sigmoid(gate.a.detach()))

    (out_dir / 'metrics_best.json').write_text(json.dumps(result, indent=2))
    print(f'[track_q] done. {arm_name} best_epoch={best["epoch"]} test_final_mse={test_mse:.6f} '
         f'gate_mean={test_dist["gate_mean"]:.4f}')


if __name__ == '__main__':
    main()
