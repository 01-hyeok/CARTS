#!/usr/bin/env python3
"""TRACK-AGGREGATE-ROUTER-FROZEN01 -- trains the (tiny) AggregateRouter
on CACHED features (`precompute_router_features01.py`'s output) --
z_qc/U_agg never change across epochs since the backbone is frozen, so
this script never touches the backbone, the candidate pool, or any
GPU-heavy forward pass; it only trains a 2-layer MLP on a fixed
in-memory tensor dataset. Fixed 10 epochs, no early stopping, every
epoch checkpoint saved. Post-hoc checkpoint selection via VALIDATION
Routed (Hard) Aggregate MSE -- computed directly from cached
`U_agg_val` by looking up each query-channel's chosen head's own
cached utility (exact, no approximation, and requires no further
backbone computation): val_agg_mse(epoch) = mean_{(q,c)} U_agg_val[q,c,argmax(pi_epoch(z_qc))].
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.aggregate_router01 import (
    AggregateRouter, aggregate_soft_teacher, r1_ce_loss, r2_kl_loss, router_entropy, top2_accuracy,
    winner_accuracy,
)

TRAIN_EPOCHS = 10
BATCH_SIZE = 256
LR = 1e-3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features_dir', required=True)
    ap.add_argument('--target', required=True, choices=('R1', 'R2'))
    ap.add_argument('--tau_r', type=float, default=None, help='required for R2')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--checkpoints', required=True)
    cli = ap.parse_args()
    if cli.target == 'R2' and cli.tau_r is None:
        raise SystemExit('[ISSUE][ABORT] --tau_r is required for R2')

    torch.manual_seed(cli.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    features_dir = Path(cli.features_dir)
    train_f = torch.load(features_dir / 'train.pt', map_location=device)
    val_f = torch.load(features_dir / 'val.pt', map_location=device)

    n_tr, c_tr, d_model = train_f['z_qc'].shape
    z_tr = train_f['z_qc'].reshape(-1, d_model)
    U_agg_tr = train_f['U_agg'].reshape(-1, 5)
    h_star_tr = U_agg_tr.argmin(dim=-1)

    z_va = val_f['z_qc'].reshape(-1, d_model)
    U_agg_va = val_f['U_agg'].reshape(-1, 5)
    h_star_va = U_agg_va.argmin(dim=-1)

    router = AggregateRouter(d_model).to(device)
    optimizer = torch.optim.Adam(router.parameters(), lr=LR)

    config = {'exp': 'TRACK-AGGREGATE-ROUTER-FROZEN01', 'target': cli.target, 'tau_r': cli.tau_r,
             'd_model': d_model, 'train_epochs': TRAIN_EPOCHS, 'batch_size': BATCH_SIZE, 'lr': LR,
             'n_train_query_channel': int(z_tr.size(0)), 'n_val_query_channel': int(z_va.size(0)),
             'checkpoint_criterion': 'min validation Hard-routed Aggregate MSE (cached U_agg lookup)'}
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2))

    n_train = z_tr.size(0)
    t0 = time.time()
    epoch_rows = []
    best = {'val_agg_mse': float('inf'), 'epoch': -1}
    for epoch in range(1, TRAIN_EPOCHS + 1):
        router.train()
        perm = torch.randperm(n_train, device=device)
        tot_loss, n_batches = 0.0, 0
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            z_b, U_b, h_b = z_tr[idx], U_agg_tr[idx], h_star_tr[idx]
            optimizer.zero_grad()
            logits = router(z_b)
            if cli.target == 'R1':
                loss = r1_ce_loss(logits, h_b)
            else:
                teacher = aggregate_soft_teacher(U_b, cli.tau_r)
                loss = r2_kl_loss(logits, teacher)
            loss.backward()
            optimizer.step()
            tot_loss += float(loss.detach()); n_batches += 1
        train_loss = tot_loss / max(n_batches, 1)

        router.eval()
        with torch.no_grad():
            val_logits = router(z_va)
            h_hat_va = val_logits.argmax(dim=-1)
            val_agg_mse = float(U_agg_va.gather(1, h_hat_va.unsqueeze(1)).squeeze(1).mean())
            val_winner_acc = winner_accuracy(val_logits, h_star_va)
            val_top2_acc = top2_accuracy(val_logits, h_star_va)
            val_entropy = float(router_entropy(val_logits).mean())
            val_max_head_usage = float(torch.bincount(h_hat_va, minlength=5).float().div(h_hat_va.numel()).max())
            if cli.target == 'R1':
                val_loss = float(r1_ce_loss(val_logits, h_star_va))
            else:
                val_teacher = aggregate_soft_teacher(U_agg_va, cli.tau_r)
                val_loss = float(r2_kl_loss(val_logits, val_teacher))

        epoch_rows.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
                          'val_agg_mse': val_agg_mse, 'val_winner_acc': val_winner_acc,
                          'val_top2_acc': val_top2_acc, 'val_entropy': val_entropy,
                          'val_max_head_usage': val_max_head_usage})
        torch.save({'router_state_dict': router.state_dict(), 'epoch': epoch, 'val_agg_mse': val_agg_mse},
                  ckpt_dir / f'checkpoint_epoch{epoch}.pth')
        if val_agg_mse < best['val_agg_mse']:
            best = {'val_agg_mse': val_agg_mse, 'epoch': epoch}
        print(f'[router_train] {cli.target}{"" if cli.tau_r is None else f"(tau={cli.tau_r})"} '
             f'epoch={epoch} train_loss={train_loss:.6f} val_loss={val_loss:.6f} val_agg_mse={val_agg_mse:.6f} '
             f'val_winner_acc={val_winner_acc:.3f} (best={best["epoch"]}:{best["val_agg_mse"]:.6f})')
        # NO early-stop break -- fixed TRAIN_EPOCHS always.

    with open(out_dir / 'epoch_metrics.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(epoch_rows[0].keys()))
        w.writeheader()
        for r in epoch_rows:
            w.writerow(r)
    (out_dir / 'checkpoint_selection.json').write_text(json.dumps({
        'selection_criterion': 'min validation Hard-routed Aggregate MSE (never test)',
        'selected_epoch': best['epoch'], 'selected_val_agg_mse': best['val_agg_mse'],
        'selected_checkpoint_path': str(ckpt_dir / f'checkpoint_epoch{best["epoch"]}.pth'),
    }, indent=2))
    print(f'[router_train] done. {cli.target} best_epoch={best["epoch"]} val_agg_mse={best["val_agg_mse"]:.6f} '
         f'wall_seconds={time.time()-t0:.1f}')


if __name__ == '__main__':
    main()
