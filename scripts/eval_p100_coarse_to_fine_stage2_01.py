#!/usr/bin/env python3
"""TRACK-P100-COARSE-TO-FINE-RETRIEVAL01 -- Stage2: Original CARTS
Trainable Global Lambda (`Y_final = B + lambda*(R-B)`, lambda=sigmoid(a),
a single trainable scalar), applied identically to all 5 of this
track's arms. Reuses `scripts/train_r_stage2_lambda01.py`'s
`load_tensors`/`GlobalLambdaGate`/`eval_gate` UNMODIFIED (imported,
exactly like `scripts/eval_r_stage2_lambda_signfix01.py` already does
for the sign-fix track) -- never touches that historical file, since
its own `--arm` choices are specific to that track's own arm names.
Same Base Forecaster checkpoint, same training protocol (10 epochs,
Adam lr=1e-3, batch 32, early-stop patience 5, validation-selected,
test evaluated once) as every other track that has used this gate.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage2 import BaseForecastHead
from scripts.rng_control01 import set_global_seeds
from scripts.train_margutil01 import build_experiment
from scripts.train_r_stage2_lambda01 import load_tensors

LR = 1e-3
BATCH_SIZE = 32
TRAIN_EPOCHS = 10
PATIENCE = 5


class GlobalLambdaGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(1))

    def forward(self):
        return torch.sigmoid(self.a)


def eval_gate(gate, B, R, Y):
    with torch.no_grad():
        lam = gate()
        fused = B + lam * (R - B)
        mse = float(((fused - Y) ** 2).mean())
        mae = float((fused - Y).abs().mean())
    return mse, mae, float(lam)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--arm', required=True,
                    choices=('V5', 'PhaseA_fusion', 'PhaseB_adaptive', 'PhaseC_residual', 'PhaseD_setaware'))
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--base_checkpoint', required=True)
    ap.add_argument('--cache_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_global_seeds(cli.seed)
    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.seq_len, 'batch_size': BATCH_SIZE, 'seed': cli.seed,
    })
    channels = int(args.enc_in)

    base = BaseForecastHead(seq_len=cli.seq_len, pred_len=cli.pred_len, channels=channels,
                            mode='per_channel_linear').to(device)
    bl = torch.load(cli.base_checkpoint, map_location=device)
    base.load_state_dict(bl['model_state_dict'])
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)

    print(f'[stage2] arm={cli.arm} loading tensors from {cli.cache_dir} ...')
    t0 = time.time()
    _, B_tr, R_tr, Y_tr = load_tensors(exp, base, cli.cache_dir, 'train', device, channels)
    _, B_va, R_va, Y_va = load_tensors(exp, base, cli.cache_dir, 'val', device, channels)
    _, B_te, R_te, Y_te = load_tensors(exp, base, cli.cache_dir, 'test', device, channels)
    print(f'[stage2] n_train={B_tr.size(0)} n_val={B_va.size(0)} n_test={B_te.size(0)} '
         f'load_seconds={time.time()-t0:.1f}')

    gate = GlobalLambdaGate().to(device)
    optimizer = torch.optim.Adam(gate.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(cli.seed)
    n_train = B_tr.size(0)

    best = {'val': float('inf'), 'epoch': -1, 'state': None}
    epoch_rows = []
    for epoch in range(1, TRAIN_EPOCHS + 1):
        perm = torch.randperm(n_train, generator=gen)
        tot_loss, n_batches = 0.0, 0
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            b_b, r_b, y_b = B_tr[idx], R_tr[idx], Y_tr[idx]
            optimizer.zero_grad()
            lam = gate()
            fused = b_b + lam * (r_b - b_b)
            loss = ((fused - y_b) ** 2).mean()
            loss.backward()
            optimizer.step()
            tot_loss += float(loss.detach()); n_batches += 1
        train_loss = tot_loss / max(n_batches, 1)
        val_mse, val_mae, val_lam = eval_gate(gate, B_va, R_va, Y_va)
        epoch_rows.append({'epoch': epoch, 'train_loss': train_loss, 'val_mse': val_mse, 'lambda': val_lam})
        if val_mse < best['val']:
            best = {'val': val_mse, 'epoch': epoch, 'state': {k: v.clone() for k, v in gate.state_dict().items()}}
        print(f'[stage2] {cli.arm} epoch={epoch} train_loss={train_loss:.6f} val_mse={val_mse:.6f} '
             f'lambda={val_lam:.4f} (best={best["epoch"]}:{best["val"]:.6f})')
        if epoch - best['epoch'] >= PATIENCE:
            print(f'[stage2] {cli.arm} early stop at epoch {epoch}')
            break

    gate.load_state_dict(best['state'])
    test_mse, test_mae, test_lam = eval_gate(gate, B_te, R_te, Y_te)
    retrieval_only_agg_mse = float(((R_te - Y_te) ** 2).mean())
    base_only_mse = float(((B_te - Y_te) ** 2).mean())

    result = {'stage2_mode': 'Original CARTS Trainable Global Lambda', 'arm': cli.arm,
             'best_epoch': best['epoch'], 'val_mse': best['val'], 'lambda': test_lam,
             'test_mse': test_mse, 'test_mae': test_mae,
             'retrieval_only_agg_mse_test': retrieval_only_agg_mse, 'base_only_mse_test': base_only_mse,
             'seed': cli.seed}
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'metrics.json').write_text(json.dumps(result, indent=2))
    with open(out_dir / 'epoch_metrics.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['epoch', 'train_loss', 'val_mse', 'lambda'])
        w.writeheader()
        for r in epoch_rows:
            w.writerow(r)
    print(f'[stage2] done. arm={cli.arm} test_mse={test_mse:.6f} test_mae={test_mae:.6f} lambda={test_lam:.4f} '
         f'base_only_mse={base_only_mse:.6f} retrieval_only_agg_mse={retrieval_only_agg_mse:.6f}')


if __name__ == '__main__':
    main()
