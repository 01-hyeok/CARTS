#!/usr/bin/env python3
"""EXP-ORACLE-SCRATCH01 -- Stage-2: given ONE frozen Stage-1 retrieval arm's
precomputed cache (`scripts/eval_oracle_scratch01.py`'s `train/val/test.pt`,
{batch_x, Y_q, Y_ret}), train a FRESH `BaseForecastHead` + `RetrievalGate`
TOGETHER (jointly, from scratch) -- no separate Base Predictor
pretraining/freezing round (per the user's explicit revision: there is no
reason to pretrain/freeze the Base for this experiment). Only Stage-1
(encoder/SetConditioner/asymmetric W_q,W_k, all baked into the cached
Y_ret) stays frozen -- it is not even imported here (`Exp_Stage1_Relation`/
`RelationEncoder` never appear in this file), so Stage-1 receiving a
gradient during Stage-2 training is structurally impossible.

Fairness (spec): the 4 arms (Individual-Cos/Asym, Set-Cos/Asym) in the SAME
horizon share ONE random `BaseForecastHead` initialisation (`--shared_
base_init_out`/`--shared_base_init_in`) and ONE random `RetrievalGate`
initialisation (`--shared_gate_init_out`/`--shared_gate_init_in`) -- same
mechanism `scripts/train_oracle_scratch01.py` already uses to share the
Stage-1 encoder's own initial weights. Architecture, optimizer, lr, epochs,
patience, batch size, seed, `top_k`/`tau_topk` (baked into the cache) and
checkpoint-selection rule (best VAL Stage-2 MSE, never TEST) are identical
across arms -- the ONLY thing that differs is which frozen Stage-1 arm
produced `Y_ret`.
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

from layers.retrieval_gate import RetrievalGate
from models.RelationStage2 import BaseForecastHead


def gated_forecast(base_head, gate, batch_x, y_ret):
    """y_ret: [B, pred_len, C] (precomputed, frozen Stage-1 retrieval).
    Reshapes to [B*C, pred_len] so ONE shared gate handles every channel,
    matching `BaseForecastHead`'s own shared-across-channels convention."""
    offset = batch_x[:, -1:, :]
    y_base = base_head(batch_x) + offset  # [B, pred_len, C] -- TRAINABLE here
    bsz, pred_len, c = y_base.shape
    y_base_flat = y_base.permute(0, 2, 1).reshape(bsz * c, pred_len)
    y_ret_flat = y_ret.permute(0, 2, 1).reshape(bsz * c, pred_len)
    y_final_flat, lam_flat = gate(y_base_flat, y_ret_flat)
    y_final = y_final_flat.reshape(bsz, c, pred_len).permute(0, 2, 1)
    lam = lam_flat.reshape(bsz, c, -1)
    return y_final, lam, y_base


def run_epoch(base_head, gate, cache, train, optimizer, device, batch_size):
    base_head.train(train)
    gate.train(train)
    x, y_q, y_ret = cache['batch_x'], cache['Y_q'], cache['Y_ret']
    n = x.size(0)
    idx_order = torch.randperm(n) if train else torch.arange(n)
    total_se, total_n = 0.0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for start in range(0, n, batch_size):
            idx = idx_order[start:start + batch_size]
            batch_x = x[idx].to(device)
            batch_yq = y_q[idx].to(device)
            batch_yret = y_ret[idx].to(device)
            if train:
                optimizer.zero_grad()
            y_final, lam, y_base = gated_forecast(base_head, gate, batch_x, batch_yret)
            loss = (y_final - batch_yq).pow(2).mean()
            if train:
                loss.backward()
                optimizer.step()
            total_se += float(loss.detach()) * batch_x.size(0)
            total_n += batch_x.size(0)
    return total_se / max(total_n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--retrieval_cache_dir', required=True, help='dir with train/val/test .pt from eval_oracle_scratch01.py (or none, for a no-retrieval control)')
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--enc_in', type=int, required=True)
    ap.add_argument('--checkpoints', default='checkpoints/exp_oracle_scratch01')
    ap.add_argument('--model_id', required=True)
    ap.add_argument('--des', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--train_epochs', type=int, default=10)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--base_head_mode', default='shared_target_linear')
    ap.add_argument('--no_retrieval', action='store_true', help='Base-only control: y_ret forced to zero, gate untrained (fixed_lambda=0)')
    ap.add_argument('--shared_base_init_out', default=None)
    ap.add_argument('--shared_base_init_in', default=None)
    ap.add_argument('--shared_gate_init_out', default=None)
    ap.add_argument('--shared_gate_init_in', default=None)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cli.seed)

    if cli.no_retrieval:
        caches = {}
        for split in ('train', 'val', 'test'):
            ref = torch.load(Path(cli.retrieval_cache_dir) / f'{split}.pt', map_location='cpu')
            caches[split] = {'batch_x': ref['batch_x'], 'Y_q': ref['Y_q'], 'Y_ret': torch.zeros_like(ref['Y_ret'])}
    else:
        cache_dir = Path(cli.retrieval_cache_dir)
        caches = {split: torch.load(cache_dir / f'{split}.pt', map_location='cpu') for split in ('train', 'val', 'test')}

    torch.manual_seed(cli.seed)
    base_head = BaseForecastHead(cli.seq_len, cli.pred_len, cli.enc_in, mode=cli.base_head_mode).to(device)
    if cli.shared_base_init_in:
        base_head.load_state_dict(torch.load(cli.shared_base_init_in, map_location='cpu'))
        print(f'[oracle_scratch01_stage2] loaded SHARED base_head init from {cli.shared_base_init_in}')
    if cli.shared_base_init_out:
        Path(cli.shared_base_init_out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(base_head.state_dict(), cli.shared_base_init_out)
        print(f'[oracle_scratch01_stage2] saved this run\'s fresh base_head init to {cli.shared_base_init_out}')

    torch.manual_seed(cli.seed)
    fixed_lambda = 0.0 if cli.no_retrieval else -1.0
    gate = RetrievalGate(cli.pred_len, gate_mode='scalar', fusion_mode='residual', fixed_lambda=fixed_lambda).to(device)
    if not cli.no_retrieval:
        if cli.shared_gate_init_in:
            gate.load_state_dict(torch.load(cli.shared_gate_init_in, map_location='cpu'))
            print(f'[oracle_scratch01_stage2] loaded SHARED gate init from {cli.shared_gate_init_in}')
        if cli.shared_gate_init_out:
            Path(cli.shared_gate_init_out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(gate.state_dict(), cli.shared_gate_init_out)
            print(f'[oracle_scratch01_stage2] saved this run\'s fresh gate init to {cli.shared_gate_init_out}')

    trainable_params = list(base_head.parameters()) + (list(gate.parameters()) if not cli.no_retrieval else [])
    n_params = sum(p.numel() for p in trainable_params)
    print(f'[oracle_scratch01_stage2] no_retrieval={cli.no_retrieval} trainable_params={n_params}')
    optimizer = torch.optim.Adam(trainable_params, lr=cli.lr)

    ckpt_dir = Path(cli.checkpoints) / 'stage2' / cli.model_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float('inf')
    best_epoch = -1
    patience_left = cli.patience
    history = []
    wall_start = time.time()

    for epoch in range(cli.train_epochs):
        t0 = time.time()
        train_mse = run_epoch(base_head, gate, caches['train'], True, optimizer, device, cli.batch_size)
        val_mse = run_epoch(base_head, gate, caches['val'], False, optimizer, device, cli.batch_size)
        dt = time.time() - t0
        history.append({'epoch': epoch + 1, 'train_mse': train_mse, 'val_mse': val_mse, 'seconds': dt})
        print(f'[oracle_scratch01_stage2] epoch {epoch+1} train_mse={train_mse:.5f} val_mse={val_mse:.5f} time={dt:.1f}s')

        ckpt_payload = {'base_head_state_dict': base_head.state_dict(), 'gate_state_dict': gate.state_dict(),
                         'epoch': epoch + 1, 'val_mse': val_mse, 'retrieval_cache_dir': cli.retrieval_cache_dir,
                         'no_retrieval': cli.no_retrieval}
        torch.save(ckpt_payload, ckpt_dir / f'checkpoint_epoch{epoch+1}.pth')
        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch + 1
            patience_left = cli.patience
            torch.save(ckpt_payload, ckpt_dir / 'checkpoint.pth')
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f'[oracle_scratch01_stage2] early stop at epoch {epoch+1} (best={best_epoch})')
                break

    best_ckpt = torch.load(ckpt_dir / 'checkpoint.pth', map_location=device)
    base_head.load_state_dict(best_ckpt['base_head_state_dict'])
    gate.load_state_dict(best_ckpt['gate_state_dict'])
    test_mse = run_epoch(base_head, gate, caches['test'], False, optimizer, device, cli.batch_size)

    wall_time = time.time() - wall_start
    summary = {'best_epoch': best_epoch, 'best_val_mse': best_val, 'test_mse': test_mse,
               'wall_clock_seconds': wall_time, 'history': history, 'checkpoint': str(ckpt_dir / 'checkpoint.pth'),
               'retrieval_cache_dir': cli.retrieval_cache_dir, 'no_retrieval': cli.no_retrieval,
               'trainable_params': n_params}
    with open(out_dir / 'summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f'[oracle_scratch01_stage2] done. best_epoch={best_epoch} best_val_mse={best_val:.5f} test_mse={test_mse:.5f}')
    print(f'[oracle_scratch01_stage2] summary written to {out_dir / "summary.json"}')


if __name__ == '__main__':
    main()
