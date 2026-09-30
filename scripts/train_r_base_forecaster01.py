#!/usr/bin/env python3
"""TRACK-R-FINAL-METHOD-GENERALIZATION01 -- standalone base forecaster
(`BaseForecastHead`, `models/RelationStage2.py`, `mode='per_channel_linear'`
-- the exact architecture every B0/B1/B2/B3 arm this session has shared)
trained FRESH per (dataset, horizon, seed), unlike TRACK-N-Q which reused
one frozen ETTh1_720 checkpoint for everything. This is the B0 arm and
the common base every retriever (Cosine/J1/M2) mixes with downstream
(PART 11: `B^{B1}=B^{B2}=B^{B3}` exactly, same seed/setting).

Trained via plain `mean((B(x)+offset - y)**2)`, Adam lr=1e-3,
batch_size=32, 10 epochs, patience=5 -- the SAME hyperparameters used
for every Stage2 training this entire session (audited in AUDIT.md).
"""
import argparse
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
from scripts.rng_control01 import batch_order_sha256, make_loader_generator, set_global_seeds
from scripts.train_factorial_e2e01 import state_sha
from scripts.train_margutil01 import build_experiment

LR = 1e-3
BATCH_SIZE = 32
TRAIN_EPOCHS = 10
PATIENCE = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--checkpoint_path', required=True)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_global_seeds(cli.seed)
    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.seq_len, 'batch_size': BATCH_SIZE, 'seed': cli.seed,
    })
    channels = int(args.enc_in)
    base = BaseForecastHead(seq_len=cli.seq_len, pred_len=cli.pred_len, channels=channels,
                            mode='per_channel_linear').to(device)
    init_sha = state_sha(base.state_dict())

    optimizer = torch.optim.Adam(base.parameters(), lr=LR)
    train_gen = make_loader_generator(cli.seed)
    _, train_loader = exp._get_data(flag='train', shuffle=True, generator=train_gen)
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path(cli.checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    def run_eval(loader):
        base.eval()
        se, ae, cnt = 0.0, 0.0, 0
        with torch.no_grad():
            for batch_x, batch_y, batch_start_idx in loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                offset = batch_x[:, -1:, :].detach()
                pred = base(batch_x) + offset
                se += float(((pred - batch_y) ** 2).sum())
                ae += float((pred - batch_y).abs().sum())
                cnt += batch_y.numel()
        return se / cnt, ae / cnt

    t0 = time.time()
    best = {'val_mse': float('inf'), 'epoch': -1}
    epoch_rows = []
    batch_order_epoch1 = None
    for epoch in range(1, TRAIN_EPOCHS + 1):
        base.train()
        starts = []
        for batch_x, batch_y, batch_start_idx in train_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            starts.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx)
                         else torch.as_tensor(batch_start_idx))
            optimizer.zero_grad()
            offset = batch_x[:, -1:, :].detach()
            pred = base(batch_x) + offset
            loss = ((pred - batch_y) ** 2).mean()
            loss.backward()
            optimizer.step()
        if epoch == 1:
            batch_order_epoch1 = batch_order_sha256(starts)
        val_mse, val_mae = run_eval(val_loader)
        epoch_rows.append({'epoch': epoch, 'val_mse': val_mse, 'val_mae': val_mae})
        payload = {'model_state_dict': base.state_dict(), 'epoch': epoch, 'val_mse': val_mse,
                  'config': {'seq_len': cli.seq_len, 'pred_len': cli.pred_len, 'channels': channels,
                            'mode': 'per_channel_linear', 'seed': cli.seed}}
        if val_mse < best['val_mse']:
            best = {'val_mse': val_mse, 'epoch': epoch}
            torch.save(payload, ckpt_path)
        print(f'[track_r_base] epoch={epoch} val_mse={val_mse:.6f} (best={best["epoch"]}:{best["val_mse"]:.6f})')
        if epoch - best['epoch'] >= PATIENCE:
            print(f'[track_r_base] early stop at epoch {epoch} (best={best["epoch"]})')
            break

    bl = torch.load(ckpt_path, map_location=device)
    base.load_state_dict(bl['model_state_dict'])
    test_mse, test_mae = run_eval(test_loader)
    final_sha = state_sha(base.state_dict())

    result = {'best_epoch': best['epoch'], 'val_mse': best['val_mse'], 'test_mse': test_mse, 'test_mae': test_mae,
             'init_sha256': init_sha, 'final_sha256': final_sha, 'batch_order_epoch1_sha256': batch_order_epoch1,
             'wall_clock_seconds': time.time() - t0, 'seed': cli.seed, 'pred_len': cli.pred_len,
             'seq_len': cli.seq_len, 'channels': channels}
    (out_dir / 'metrics.json').write_text(json.dumps(result, indent=2))
    print(f'[track_r_base] done. best_epoch={best["epoch"]} test_mse={test_mse:.6f} test_mae={test_mae:.6f}')


if __name__ == '__main__':
    main()
