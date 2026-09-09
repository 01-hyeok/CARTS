#!/usr/bin/env python3
"""EXP-ORACLE-SCRATCH-FROZENBASE01 -- STEP A: Base Predictor pretraining,
retrieval fully disabled. Produces ONE checkpoint per horizon
(`base_H96_checkpoint`/`base_H720_checkpoint` in the spec's own naming) that
every retrieval arm in STEP E shares, frozen, unmodified -- Base Predictor
fine-tuning/adaptation is explicitly out of scope for every later step (see
the frozen-base sanity assertions in
`scripts/train_oracle_scratch_frozenbase01_stage2.py`).

Reuses `models.RelationStage2.BaseForecastHead` unmodified (the SAME class
production Stage-2 uses as `self.base_head`, `mode='shared_target_linear'`
matching the reference checkpoint's own `base_head_mode`) and
`build_experiment` (args reconstruction only, exactly as in
`scripts/train_oracle_scratch01.py`) for data loading.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage2 import BaseForecastHead
from scripts.train_margutil01 import build_experiment


def run_epoch(base_head, loader, train, optimizer, device):
    base_head.train(train)
    total_se, total_n = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch_x, batch_y, _ in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            if train:
                optimizer.zero_grad()
            offset = batch_x[:, -1:, :]
            y_pred = base_head(batch_x) + offset
            loss = (y_pred - batch_y).pow(2).mean()
            if train:
                loss.backward()
                optimizer.step()
            total_se += float(loss.detach()) * batch_x.size(0)
            total_n += batch_x.size(0)
    return total_se / max(total_n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--checkpoints', default='checkpoints/exp_oracle_scratch_frozenbase01')
    ap.add_argument('--model_id', required=True)
    ap.add_argument('--des', required=True)
    ap.add_argument('--train_epochs', type=int, default=10)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--base_head_mode', default='shared_target_linear')
    cli = ap.parse_args()

    overrides = {'is_training': 1, 'model_id': cli.model_id, 'des': cli.des,
                 'checkpoints': cli.checkpoints, 'seed': cli.seed,
                 'pred_len': cli.pred_len, 'seq_len': cli.pred_len,
                 'stage1_retrieval_metric': 'cosine'}
    exp, args = build_experiment(cli.reference_ckpt, overrides)
    torch.manual_seed(args.seed)
    device = exp.device

    base_head = BaseForecastHead(int(args.seq_len), int(args.pred_len), int(args.enc_in),
                                  mode=cli.base_head_mode).to(device)
    n_params = sum(p.numel() for p in base_head.parameters())
    print(f'[base_only] pred_len={cli.pred_len} trainable_params={n_params}')
    optimizer = torch.optim.Adam(base_head.parameters(), lr=float(args.learning_rate))

    ckpt_dir = Path(cli.checkpoints) / 'base_only' / args.data / f'seq{args.seq_len}_pred{args.pred_len}' / cli.model_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float('inf')
    best_epoch = -1
    patience_left = int(args.patience)
    history = []
    wall_start = time.time()
    for epoch in range(int(args.train_epochs)):
        t0 = time.time()
        _, train_loader = exp._get_data(flag='train', shuffle=True)
        _, val_loader = exp._get_data(flag='val', shuffle=False)
        train_mse = run_epoch(base_head, train_loader, True, optimizer, device)
        val_mse = run_epoch(base_head, val_loader, False, optimizer, device)
        dt = time.time() - t0
        history.append({'epoch': epoch + 1, 'train_mse': train_mse, 'val_mse': val_mse, 'seconds': dt})
        print(f'[base_only] epoch {epoch+1} train_mse={train_mse:.5f} val_mse={val_mse:.5f} time={dt:.1f}s')
        ckpt_payload = {'base_head_state_dict': base_head.state_dict(), 'args': vars(args),
                         'epoch': epoch + 1, 'val_mse': val_mse, 'base_head_mode': cli.base_head_mode,
                         'seq_len': int(args.seq_len), 'pred_len': int(args.pred_len), 'enc_in': int(args.enc_in)}
        torch.save(ckpt_payload, ckpt_dir / f'checkpoint_epoch{epoch+1}.pth')
        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch + 1
            patience_left = int(args.patience)
            torch.save(ckpt_payload, ckpt_dir / 'checkpoint.pth')
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f'[base_only] early stop at epoch {epoch+1} (best={best_epoch})')
                break

    _, test_loader = exp._get_data(flag='test', shuffle=False)
    best_ckpt = torch.load(ckpt_dir / 'checkpoint.pth', map_location=device)
    base_head.load_state_dict(best_ckpt['base_head_state_dict'])
    test_mse = run_epoch(base_head, test_loader, False, optimizer, device)

    wall_time = time.time() - wall_start
    summary = {'best_epoch': best_epoch, 'best_val_mse': best_val, 'test_mse': test_mse,
               'wall_clock_seconds': wall_time, 'history': history, 'trainable_params': n_params,
               'checkpoint': str(ckpt_dir / 'checkpoint.pth'), 'base_head_mode': cli.base_head_mode,
               'pred_len': cli.pred_len}
    with open(ckpt_dir / 'summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f'[base_only] done. best_epoch={best_epoch} best_val_mse={best_val:.5f} test_mse={test_mse:.5f}')


if __name__ == '__main__':
    main()
