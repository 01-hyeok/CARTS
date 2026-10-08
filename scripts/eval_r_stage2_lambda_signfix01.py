#!/usr/bin/env python3
"""TRACK-V-SHARED-TOP100-SIGNFIX01 -- Stage-2 re-evaluation using the
ORIGINAL CARTS trainable global-lambda gate (NOT the Professor-paper-style
validation-only beta fusion). Reuses `scripts/train_r_stage2_lambda01.py`'s
`load_tensors`, `GlobalLambdaGate`, and `eval_gate` UNMODIFIED (imported) --
same formula `Y_final = B + lambda*(R-B)`, `lambda=sigmoid(a)`, frozen Base
and retrieval cache, global scalar `a` is the ONLY trainable parameter,
trained with gradient descent on the train split, best epoch chosen by
validation MSE, evaluated once on test. The only additions over
`train_r_stage2_lambda01.py`'s own training loop are (a) saving every
epoch's checkpoint (not just the best one) and (b) resource/timing/VRAM
logging -- both purely additive side effects; the training dynamics and
selection criterion are byte-for-byte identical to the original script.
`train_r_stage2_lambda01.py` itself is never modified (many other tracks
depend on its exact existing behavior).

Applies ONLY to the already sign-fixed P100 Stage-1 caches under
results/TRACK-V-SHARED-TOP100-SIGNFIX01/<ds>/H<h>/seed0/pool_top100/<cell>/<arm>/cache/
-- never the old bugged caches. Writes to a SEPARATE output directory
(.../stage2_lambda_original/) so it never overwrites the Professor-style
Stage-2 results already saved under .../stage2_professor/.
"""
import argparse
import csv
import json
import subprocess
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


def gpu_snapshot(gpu_index):
    try:
        out = subprocess.run(
            ['nvidia-smi', '-i', str(gpu_index), '--query-gpu=name,memory.used,utilization.gpu',
             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True).stdout.strip()
        name, mem, util = [x.strip() for x in out.split(',')]
        return {'gpu_name': name, 'memory_used_mib': int(mem), 'utilization_pct': int(util)}
    except Exception as e:
        return {'error': str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--arm', required=True, choices=('V0', 'V1', 'V2', 'V5'))
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--base_checkpoint', required=True)
    ap.add_argument('--cache_dir', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--gpu_index', type=int, default=1)
    cli = ap.parse_args()

    t_total0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_before = gpu_snapshot(cli.gpu_index)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

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

    print('[r_stage2_signfix] loading tensors (sign-fixed P100 cache)...')
    t_load0 = time.time()
    tr_starts, B_tr, R_tr, Y_tr = load_tensors(exp, base, cli.cache_dir, 'train', device, channels)
    va_starts, B_va, R_va, Y_va = load_tensors(exp, base, cli.cache_dir, 'val', device, channels)
    te_starts, B_te, R_te, Y_te = load_tensors(exp, base, cli.cache_dir, 'test', device, channels)
    load_seconds = time.time() - t_load0
    print(f'[r_stage2_signfix] n_train={B_tr.size(0)} n_val={B_va.size(0)} n_test={B_te.size(0)}')

    gate = GlobalLambdaGate().to(device)
    optimizer = torch.optim.Adam(gate.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(cli.seed)

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    epoch_ckpt_dir = out_dir / 'epoch_checkpoints'
    epoch_ckpt_dir.mkdir(parents=True, exist_ok=True)
    n_train = B_tr.size(0)
    t_train0 = time.time()
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
        torch.save({'gate_state_dict': gate.state_dict(), 'epoch': epoch, 'val_mse': val_mse},
                  epoch_ckpt_dir / f'checkpoint_epoch{epoch}.pth')
        if val_mse < best['val']:
            best = {'val': val_mse, 'epoch': epoch, 'state': {k: v.clone() for k, v in gate.state_dict().items()}}
        print(f'[r_stage2_signfix] epoch={epoch} train_loss={train_loss:.6f} val_mse={val_mse:.6f} '
             f'lambda={val_lam:.4f} (best={best["epoch"]}:{best["val"]:.6f})')
        if epoch - best['epoch'] >= PATIENCE:
            print(f'[r_stage2_signfix] early stop at epoch {epoch} (best={best["epoch"]})')
            break
    train_seconds = time.time() - t_train0

    gate.load_state_dict(best['state'])
    torch.save({'gate_state_dict': gate.state_dict(), 'epoch': best['epoch'], 'val_mse': best['val']},
              out_dir / 'checkpoint.pth')
    test_mse, test_mae, test_lam = eval_gate(gate, B_te, R_te, Y_te)
    val_mse_best, val_mae_best, val_lam_best = eval_gate(gate, B_va, R_va, Y_va)

    gpu_after = gpu_snapshot(cli.gpu_index)
    max_vram_mb = float(torch.cuda.max_memory_allocated(device) / 2**20) if device.type == 'cuda' else 0.0
    total_seconds = time.time() - t_total0

    result = {'stage2_mode': 'Original CARTS Trainable Global Lambda', 'arm': cli.arm,
             'best_epoch': best['epoch'], 'val_mse': val_mse_best, 'val_mae': val_mae_best,
             'lambda': test_lam, 'test_mse': test_mse, 'test_mae': test_mae, 'seed': cli.seed,
             'wall_clock_seconds': total_seconds}
    (out_dir / 'metrics.json').write_text(json.dumps(result, indent=2))
    with open(out_dir / 'epoch_metrics.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['epoch', 'train_loss', 'val_mse', 'lambda'])
        w.writeheader()
        for r in epoch_rows:
            w.writerow(r)
    (out_dir / 'resource_metrics.json').write_text(json.dumps({
        'gpu_index': cli.gpu_index, 'gpu_before': gpu_before, 'gpu_after': gpu_after,
        'max_vram_mb': max_vram_mb, 'load_tensors_seconds': load_seconds,
        'train_seconds': train_seconds, 'total_wall_clock_seconds': total_seconds,
        'n_epochs_run': len(epoch_rows), 'every_epoch_checkpoint_dir': str(epoch_ckpt_dir),
    }, indent=2))
    print(f'[r_stage2_signfix] done. arm={cli.arm} best_epoch={best["epoch"]} '
         f'test_mse={test_mse:.6f} test_mae={test_mae:.6f} lambda={test_lam:.4f} '
         f'wall={total_seconds:.1f}s max_vram={max_vram_mb:.1f}MB')


if __name__ == '__main__':
    main()
