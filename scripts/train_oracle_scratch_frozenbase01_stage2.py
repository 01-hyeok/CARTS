#!/usr/bin/env python3
"""EXP-ORACLE-SCRATCH-FROZENBASE01 -- STEP E: Frozen Base Predictor (from
STEP A) + frozen Stage-1 retriever (its precomputed retrieval cache, from
`scripts/eval_oracle_scratch01.py`) + a TRAINABLE fusion module only
(`layers.retrieval_gate.RetrievalGate`, reused unmodified, `fusion_mode=
"residual"`, `gate_mode="scalar"`, one shared instance applied per-channel
via a [B,C,pred_len] -> [B*C,pred_len] reshape). Neither the Base Predictor
nor the Stage-1 retriever (encoder/SetConditioner/asymmetric W_q/W_k) is
even IMPORTED here -- `Exp_Stage1_Relation`/`RelationEncoder` never appear in
this file (structurally guarantees Stage-1 cannot receive gradient; see
`tests/test_exp_oracle_scratch01.py::test_10`), and the Base Predictor's
parameters are excluded from the optimizer's own parameter list by
construction, not merely by a `requires_grad=False` flag.

Mandatory sanity assertions (spec section 22), checked before AND after
every Stage-2 run, written to `sanity_summary.json`:
1. every Base Predictor parameter's `.grad` is None after every backward
2. Base Predictor `state_dict()` is bitwise/numerically unchanged, start to end
3. no Base Predictor parameter's `id()` appears in the optimizer's own
   parameter list
4. the trainable gate's own gradient norm is > 0 (the run is actually
   learning something, not silently dead)
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


def load_frozen_base(base_ckpt_path, device):
    ck = torch.load(base_ckpt_path, map_location='cpu')
    base_head = BaseForecastHead(ck['seq_len'], ck['pred_len'], ck['enc_in'],
                                  mode=ck['base_head_mode']).to(device)
    base_head.load_state_dict(ck['base_head_state_dict'])
    base_head.eval()
    for p in base_head.parameters():
        p.requires_grad_(False)
    frozen_reference = {k: v.detach().clone() for k, v in base_head.state_dict().items()}
    return base_head, frozen_reference, ck


def base_predictor_sanity(base_head, frozen_reference, optimizer):
    grad_none = all(p.grad is None for p in base_head.parameters())
    requires_grad_false = all(not p.requires_grad for p in base_head.parameters())
    unchanged = all(torch.allclose(v, base_head.state_dict()[k], atol=0.0, rtol=0.0)
                     for k, v in frozen_reference.items())
    base_ids = {id(p) for p in base_head.parameters()}
    optimizer_ids = {id(p) for group in optimizer.param_groups for p in group['params']}
    not_in_optimizer = base_ids.isdisjoint(optimizer_ids)
    return {'grad_none': grad_none, 'requires_grad_false': requires_grad_false,
            'weights_unchanged': unchanged, 'not_in_optimizer': not_in_optimizer,
            'all_pass': grad_none and requires_grad_false and unchanged and not_in_optimizer}


def gated_forecast(base_head, gate, batch_x, y_ret):
    """y_ret: [B, pred_len, C] (precomputed, frozen Stage-1 retrieval).
    Reshapes to [B*C, pred_len] so ONE shared gate handles every channel,
    matching `BaseForecastHead`'s own shared-across-channels convention."""
    offset = batch_x[:, -1:, :]
    with torch.no_grad():
        y_base = base_head(batch_x) + offset  # [B, pred_len, C], no grad reaches base_head
    bsz, pred_len, c = y_base.shape
    y_base_flat = y_base.permute(0, 2, 1).reshape(bsz * c, pred_len)
    y_ret_flat = y_ret.permute(0, 2, 1).reshape(bsz * c, pred_len)
    y_final_flat, lam_flat = gate(y_base_flat, y_ret_flat)
    y_final = y_final_flat.reshape(bsz, c, pred_len).permute(0, 2, 1)
    lam = lam_flat.reshape(bsz, c, -1)
    return y_final, lam, y_base


def run_epoch(base_head, gate, cache, train, optimizer, device, batch_size):
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
    ap.add_argument('--base_checkpoint', required=True, help='STEP A frozen Base Predictor checkpoint')
    ap.add_argument('--retrieval_cache_dir', required=True, help='dir with train/val/test .pt from eval_oracle_scratch01.py')
    ap.add_argument('--checkpoints', default='checkpoints/exp_oracle_scratch_frozenbase01')
    ap.add_argument('--model_id', required=True)
    ap.add_argument('--des', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--train_epochs', type=int, default=50)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--shared_gate_init_out', default=None)
    ap.add_argument('--shared_gate_init_in', default=None)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cli.seed)
    base_head, frozen_reference, base_ck = load_frozen_base(cli.base_checkpoint, device)
    pred_len = int(base_ck['pred_len'])

    cache_dir = Path(cli.retrieval_cache_dir)
    caches = {split: torch.load(cache_dir / f'{split}.pt', map_location='cpu') for split in ('train', 'val', 'test')}

    torch.manual_seed(cli.seed)
    gate = RetrievalGate(pred_len, gate_mode='scalar', fusion_mode='residual', fixed_lambda=-1.0).to(device)
    if cli.shared_gate_init_in:
        gate.load_state_dict(torch.load(cli.shared_gate_init_in, map_location='cpu'))
        print(f'[frozenbase01_stage2] loaded SHARED gate init from {cli.shared_gate_init_in}')
    if cli.shared_gate_init_out:
        Path(cli.shared_gate_init_out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(gate.state_dict(), cli.shared_gate_init_out)
        print(f'[frozenbase01_stage2] saved this run\'s fresh gate init to {cli.shared_gate_init_out}')

    optimizer = torch.optim.Adam(gate.parameters(), lr=cli.lr)
    assert {id(p) for p in base_head.parameters()}.isdisjoint(
        {id(p) for group in optimizer.param_groups for p in group['params']}), \
        'Base Predictor parameter leaked into the Stage-2 optimizer'

    ckpt_dir = Path(cli.checkpoints) / 'stage2' / cli.model_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val = float('inf')
    best_epoch = -1
    patience_left = cli.patience
    history = []
    sanity_per_epoch = []
    wall_start = time.time()

    for epoch in range(cli.train_epochs):
        t0 = time.time()
        train_mse = run_epoch(base_head, gate, caches['train'], True, optimizer, device, cli.batch_size)
        gate_grad_norm = sum(p.grad.norm().item() ** 2 for p in gate.parameters() if p.grad is not None) ** 0.5
        val_mse = run_epoch(base_head, gate, caches['val'], False, optimizer, device, cli.batch_size)
        dt = time.time() - t0
        sanity = base_predictor_sanity(base_head, frozen_reference, optimizer)
        sanity['gate_grad_norm'] = gate_grad_norm
        sanity['gate_grad_norm_positive'] = gate_grad_norm > 0.0
        sanity_per_epoch.append({'epoch': epoch + 1, **sanity})
        assert sanity['all_pass'], f'Base Predictor freeze sanity FAILED at epoch {epoch+1}: {sanity}'
        assert sanity['gate_grad_norm_positive'], f'gate gradient norm is exactly 0 at epoch {epoch+1} -- STOP'
        history.append({'epoch': epoch + 1, 'train_mse': train_mse, 'val_mse': val_mse, 'seconds': dt})
        print(f'[frozenbase01_stage2] epoch {epoch+1} train_mse={train_mse:.5f} val_mse={val_mse:.5f} '
              f'gate_grad_norm={gate_grad_norm:.5f} time={dt:.1f}s')

        ckpt_payload = {'gate_state_dict': gate.state_dict(), 'epoch': epoch + 1, 'val_mse': val_mse,
                         'base_checkpoint': cli.base_checkpoint, 'retrieval_cache_dir': cli.retrieval_cache_dir}
        torch.save(ckpt_payload, ckpt_dir / f'checkpoint_epoch{epoch+1}.pth')
        if val_mse < best_val:
            best_val = val_mse
            best_epoch = epoch + 1
            patience_left = cli.patience
            torch.save(ckpt_payload, ckpt_dir / 'checkpoint.pth')
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f'[frozenbase01_stage2] early stop at epoch {epoch+1} (best={best_epoch})')
                break

    best_ckpt = torch.load(ckpt_dir / 'checkpoint.pth', map_location=device)
    gate.load_state_dict(best_ckpt['gate_state_dict'])
    test_mse = run_epoch(base_head, gate, caches['test'], False, optimizer, device, cli.batch_size)
    final_sanity = base_predictor_sanity(base_head, frozen_reference, optimizer)
    assert final_sanity['all_pass'], f'Base Predictor freeze sanity FAILED at final eval: {final_sanity}'

    wall_time = time.time() - wall_start
    summary = {'best_epoch': best_epoch, 'best_val_mse': best_val, 'test_mse': test_mse,
               'wall_clock_seconds': wall_time, 'history': history, 'checkpoint': str(ckpt_dir / 'checkpoint.pth'),
               'base_checkpoint': cli.base_checkpoint, 'retrieval_cache_dir': cli.retrieval_cache_dir}
    with open(out_dir / 'summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2, default=str)
    with open(out_dir / 'sanity_summary.json', 'w') as fh:
        json.dump({'per_epoch': sanity_per_epoch, 'final': final_sanity}, fh, indent=2, default=str)
    print(f'[frozenbase01_stage2] done. best_epoch={best_epoch} best_val_mse={best_val:.5f} test_mse={test_mse:.5f}')
    print(f'[frozenbase01_stage2] summary written to {out_dir / "summary.json"}')


if __name__ == '__main__':
    main()
