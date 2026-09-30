#!/usr/bin/env python3
"""TRACK-R-FINAL-METHOD-GENERALIZATION01 -- generalized Stage2: a single
trainable global scalar lambda mixing a per-setting frozen base forecaster
(`train_r_base_forecaster01.py`'s checkpoint) with a per-setting frozen
retriever's Uniform-aggregated cache (`build_r_retrieval_cache01.py`).

`Y_hat = (1-lambda)*B + lambda*R`, `lambda=sigmoid(a)`, `a` initialized
to 0 (`lambda_0=0.5`) -- the ONLY trainable parameter. No relation_mixer,
no query-conditioned gate, no per-channel gate, no HostScorer (PART 4).
Base forecaster and retriever are both frozen (never constructed with
`requires_grad`, loaded read-only). B0 (base-only) needs no training at
all here -- it's just the base forecaster's own test metric, already
saved by `train_r_base_forecaster01.py`.
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
from scripts.rng_control01 import set_global_seeds
from scripts.train_margutil01 import build_experiment

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


def load_tensors(exp, base, cache_dir, split, device, channels):
    _, loader = exp._get_data(flag=split, shuffle=False)
    cache = torch.load(Path(cache_dir) / f'{split}.pt', map_location='cpu')
    lut = {int(s): i for i, s in enumerate(cache['query_start_idx'].tolist())}
    starts, B_list, R_list, Y_list = [], [], [], []
    with torch.no_grad():
        for batch_x, batch_y, batch_start_idx in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            offset = batch_x[:, -1:, :].detach()
            b_q = base(batch_x) + offset
            idx = [lut[int(s)] for s in batch_start_idx.tolist()]
            r_q = cache['relation_outputs'][idx].to(device)
            for b in range(batch_x.size(0)):
                starts.append(int(batch_start_idx[b]))
            B_list.append(b_q); R_list.append(r_q); Y_list.append(batch_y)
    return torch.tensor(starts), torch.cat(B_list), torch.cat(R_list), torch.cat(Y_list)


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
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
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

    print('[track_r_stage2] loading tensors...')
    tr_starts, B_tr, R_tr, Y_tr = load_tensors(exp, base, cli.cache_dir, 'train', device, channels)
    va_starts, B_va, R_va, Y_va = load_tensors(exp, base, cli.cache_dir, 'val', device, channels)
    te_starts, B_te, R_te, Y_te = load_tensors(exp, base, cli.cache_dir, 'test', device, channels)
    print(f'[track_r_stage2] n_train={B_tr.size(0)} n_val={B_va.size(0)} n_test={B_te.size(0)}')

    gate = GlobalLambdaGate().to(device)
    optimizer = torch.optim.Adam(gate.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(cli.seed)

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_train = B_tr.size(0)
    t0 = time.time()
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
        print(f'[track_r_stage2] epoch={epoch} train_loss={train_loss:.6f} val_mse={val_mse:.6f} '
             f'lambda={val_lam:.4f} (best={best["epoch"]}:{best["val"]:.6f})')
        if epoch - best['epoch'] >= PATIENCE:
            print(f'[track_r_stage2] early stop at epoch {epoch} (best={best["epoch"]})')
            break

    gate.load_state_dict(best['state'])
    torch.save({'gate_state_dict': gate.state_dict(), 'epoch': best['epoch'], 'val_mse': best['val']},
              out_dir / 'checkpoint.pth')
    test_mse, test_mae, test_lam = eval_gate(gate, B_te, R_te, Y_te)
    val_mse_best, val_mae_best, val_lam_best = eval_gate(gate, B_va, R_va, Y_va)

    result = {'best_epoch': best['epoch'], 'val_mse': val_mse_best, 'val_mae': val_mae_best,
             'lambda': test_lam, 'test_mse': test_mse, 'test_mae': test_mae, 'seed': cli.seed,
             'wall_clock_seconds': time.time() - t0}
    (out_dir / 'metrics.json').write_text(json.dumps(result, indent=2))
    with open(out_dir / 'epoch_metrics.csv', 'w') as fh:
        fh.write('epoch,train_loss,val_mse,lambda\n')
        for r in epoch_rows:
            fh.write(f"{r['epoch']},{r['train_loss']},{r['val_mse']},{r['lambda']}\n")
    print(f'[track_r_stage2] done. best_epoch={best["epoch"]} test_mse={test_mse:.6f} lambda={test_lam:.4f}')


if __name__ == '__main__':
    main()
