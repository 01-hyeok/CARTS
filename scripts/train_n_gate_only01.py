#!/usr/bin/env python3
"""TRACK-N-FORECAST-CONDITIONAL-UTILITY01 PART 18-19 -- conditional
Frozen-Base Gate-Only experiment. Runs ONLY because PART 15's trigger
condition was met: M2's Uniform oracle/fixed-lambda potential is
competitive with J1's (oracle_gain 0.0514 vs 0.0588; val-calibrated
global-lambda test MSE 0.463969 vs 0.463349 -- nearly tied), yet
TRACK-M's actual learned Stage2 gate suppressed M2 to near-zero
(gate_mean=0.0044, 95.5% of queries at gate<0.02) and produced a
significantly WORSE final forecast than J1's arm.

G0=J1, G1=K2, G2=M2, all Uniform aggregation (PART 18's own spec), all
conditioned on the SAME frozen S0 base (base_head/relation_mixer/
relation_concat_projection loaded from TRACK-M's S0_base checkpoint and
FROZEN -- PART 16's `stage1_encoder` etc. freeze-set also applied,
though retrieval here is entirely cache-based so those never receive
gradient regardless). ONLY `model.gate` is trainable, and it is
RE-INITIALIZED to a fresh, shared (seed=0) init BEFORE training --
i.e. NOT the value S0's own joint training left it at -- so G0/G1/G2 all
start from the literal same gate weights, differing only in which
retrieval cache they see.

Reuses `train_epoch`/`eval_epoch`/`load_cache_as_lookup`/`gate_distribution`
from `train_setlossctrl_stage2_retrain02.py` UNMODIFIED -- only the
freeze-set and the retrieval cache differ.
"""
import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_factorial_e2e01 import state_sha
from scripts.train_setlossctrl_stage2_retrain02 import (
    build_fresh_stage2, eval_epoch, load_cache_as_lookup, train_epoch,
)

S2_720 = ('checkpoints/stage2/ETTh1/seq720_pred720/stage2_carts_softset_s2_ETTh1_720_S0_wce_RelationStage2_ETTh1_'
         'ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_'
         'sl720_pl720_0/checkpoint.pth')
S0_CKPT = REPO_ROOT / 'checkpoints/track_m_relevance_constrained_multislot01/stage2/ETTh1_720/S0_base/checkpoint.pth'

EXTRA_FREEZE = ('base_head', 'relation_mixer', 'relation_concat_projection')
BASE_FREEZE = ('stage1_encoder', 'shared_cross_projection', 'retrieval_metric', 'pairwise_scorer',
               'query_cond_proj', 'candidate_cond_proj')


def freeze_all_but_gate(model):
    frozen_shas = {}
    for name in EXTRA_FREEZE + BASE_FREEZE:
        sub = getattr(model, name, None)
        if sub is None:
            continue
        for p in sub.parameters():
            p.requires_grad_(False)
        sub.eval()
        frozen_shas[name] = state_sha(sub.state_dict())
    assert any(p.requires_grad for p in model.gate.parameters()), 'gate must remain trainable'
    return frozen_shas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=('G0_J1', 'G1_K2', 'G2_M2'))
    ap.add_argument('--cell', default='ETTh1_720')
    ap.add_argument('--cache_dir', required=True)
    ap.add_argument('--checkpoints', default='checkpoints/track_n_forecast_conditional_utility01/gate_only')
    ap.add_argument('--out_dir', default='results/TRACK-N-FORECAST-CONDITIONAL-UTILITY01/gate_only')
    ap.add_argument('--shared_gate_init_out', default=None)
    ap.add_argument('--shared_gate_init_in', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train_epochs', type=int, default=10)
    ap.add_argument('--patience', type=int, default=5)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp, args, model, host_ck = build_fresh_stage2(S2_720, seed=cli.seed)
    model.to(device)
    exp._ensure_memory()
    exp._build_key_bank(force=True)

    # capture the fresh (seed=cli.seed) gate init BEFORE loading S0's trained weights over it
    fresh_gate_state = copy.deepcopy(model.gate.state_dict())

    bl = torch.load(S0_CKPT, map_location=device)
    model.load_state_dict(bl['model_state_dict'])  # base_head/relation_mixer/relation_concat_projection/gate <- S0

    if cli.shared_gate_init_in:
        blob = torch.load(cli.shared_gate_init_in, map_location='cpu')
        model.gate.load_state_dict(blob['gate_state_dict'])
        got = state_sha(model.gate.state_dict())
        if got != blob['sha256']:
            raise SystemExit(f'[ABORT] shared gate init SHA mismatch: {got} != {blob["sha256"]}')
        gate_init_sha = got
    else:
        model.gate.load_state_dict(fresh_gate_state)  # overwrite S0's trained gate with the fresh shared init
        gate_init_sha = state_sha(model.gate.state_dict())
        if cli.shared_gate_init_out:
            Path(cli.shared_gate_init_out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({'gate_state_dict': {k: v.detach().cpu() for k, v in model.gate.state_dict().items()},
                       'sha256': gate_init_sha}, cli.shared_gate_init_out)

    frozen_shas_before = freeze_all_but_gate(model)
    base_head_sha = state_sha(model.base_head.state_dict())
    trainable_names = sorted(n for n, p in model.named_parameters() if p.requires_grad)
    optimizer = exp._select_optimizer()

    cache_dir = Path(cli.cache_dir)
    train_cache, train_lut = load_cache_as_lookup(cache_dir / 'train.pt')
    val_cache, val_lut = load_cache_as_lookup(cache_dir / 'val.pt')
    test_cache, test_lut = load_cache_as_lookup(cache_dir / 'test.pt')

    _, train_loader = exp._get_data(flag='train', shuffle=True)
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)

    out_dir = Path(cli.out_dir) / cli.cell / cli.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints) / cli.cell / cli.arm
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = {'arm': cli.arm, 'seed': cli.seed, 'gate_init_sha256': gate_init_sha,
                  'base_head_sha256': base_head_sha, 'frozen_submodule_sha256_before': frozen_shas_before,
                  'trainable_parameter_names': trainable_names,
                  'trainable_parameter_count': sum(p.numel() for n, p in model.named_parameters() if p.requires_grad)}
    (out_dir / 'config.json').write_text(json.dumps(fingerprint, indent=2))
    print(f'[gate_only] {cli.arm} gate_init_sha={gate_init_sha[:16]} base_head_sha={base_head_sha[:16]} '
         f'trainable_params={fingerprint["trainable_parameter_count"]}')

    epoch_rows = []
    best = {'val': float('inf'), 'epoch': -1}
    t0 = time.time()
    for epoch in range(1, cli.train_epochs + 1):
        tr_loss = train_epoch(exp, model, optimizer, train_loader, train_cache, train_lut, device)
        va = eval_epoch(exp, model, val_loader, val_cache, val_lut, device)
        row = {'epoch': epoch, 'train_loss': tr_loss, **{f'val_{k}': v for k, v in va.items()}}
        epoch_rows.append(row)
        payload = {'model_state_dict': model.state_dict(), 'epoch': epoch, 'val_final_mse': va['final_mse']}
        torch.save(payload, ckpt_dir / f'checkpoint_epoch{epoch}.pth')
        if va['final_mse'] < best['val']:
            best = {'val': va['final_mse'], 'epoch': epoch}
            torch.save(payload, ckpt_dir / 'checkpoint.pth')
        print(f"[gate_only] {cli.arm} epoch {epoch} train_loss={tr_loss:.5f} val_final_mse={va['final_mse']:.6f} "
             f"gate_mean={va['gate_mean']:.4f}")
        if epoch - best['epoch'] >= cli.patience:
            print(f'[gate_only] early stop at epoch {epoch} (best={best["epoch"]})')
            break

    frozen_shas_after = freeze_all_but_gate(model)
    mismatched = {k: v for k, v in frozen_shas_after.items() if frozen_shas_before.get(k) != v}
    if mismatched:
        raise SystemExit(f'[ISSUE][ABORT] frozen submodule(s) changed during training: {mismatched}')
    base_head_sha_after = state_sha(model.base_head.state_dict())
    if base_head_sha_after != base_head_sha:
        raise SystemExit('[ISSUE][ABORT] base_head changed during gate-only training')

    bl_best = torch.load(ckpt_dir / 'checkpoint.pth', map_location=device)
    model.load_state_dict(bl_best['model_state_dict'])
    te_best = eval_epoch(exp, model, test_loader, test_cache, test_lut, device)

    with open(out_dir / 'epoch_metrics.csv', 'w', newline='') as fh:
        keys = sorted({k for r in epoch_rows for k in r})
        w = csv.DictWriter(fh, fieldnames=['epoch'] + [k for k in keys if k != 'epoch'])
        w.writeheader()
        for r in epoch_rows:
            w.writerow(r)
    (out_dir / 'metrics_best.json').write_text(json.dumps(
        {'epoch': best['epoch'], 'val': epoch_rows[best['epoch'] - 1], 'test': te_best}, indent=2))
    (out_dir / 'fingerprint.json').write_text(json.dumps({**fingerprint, 'frozen_submodule_sha256_after': frozen_shas_after,
                                                          'base_head_sha256_after': base_head_sha_after,
                                                          'wall_clock_seconds': time.time() - t0}, indent=2))
    print(f"[gate_only] done. {cli.arm} best_epoch={best['epoch']} test_final_mse={te_best['final_mse']:.6f}")


if __name__ == '__main__':
    main()
