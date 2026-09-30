#!/usr/bin/env python3
"""TRACK-O-FROZEN-BASE-RETRIEVAL-CONSUMER01 -- Frozen Base + Trainable
Retrieval Consumer.

Per AUDIT.md's forward-graph trace (S2_720 host config:
`stage2_relation_fusion='gate'`, `fusion_mode='residual'`), the ONLY
modules that actually consume `relation_outputs` in the forward pass are
`relation_mixer` and `gate` -- `relation_concat_projection` exists but
is never invoked by this config (TRACK-M left it trainable but inert;
this track correctly excludes it, PART 5).

For every arm (O1=J1+Uniform, O2=J1+Host, O3=M2+Uniform, O4=M2+Host):
  1. `build_fresh_stage2(S2_720, seed=0)` -- fresh model, ALL submodules
     (base_head, relation_mixer, gate, relation_concat_projection) at
     their shared seed=0 init.
  2. Load ONLY `base_head.*` from TRACK-M's S0 checkpoint (filtered from
     its full state_dict) -- relation_mixer/gate are NEVER touched by
     S0's own zero-retrieval-trained weights (fixing TRACK-N's Gate-Only
     confound, where the mixer was loaded from S0 and frozen).
  3. Freeze base_head + relation_concat_projection + the pre-existing
     retrieval-irrelevant `FREEZE_SUBMODULES`. Leave relation_mixer +
     gate trainable.
  4. Train via `train_epoch`/`eval_epoch` reused UNMODIFIED from
     `train_setlossctrl_stage2_retrain02.py`.

Retrieval caches are NOT rebuilt (AUDIT.md): reused verbatim from
TRACK-N (Uniform) and TRACK-M (Host).
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
    FREEZE_SUBMODULES, build_fresh_stage2, eval_epoch, load_cache_as_lookup, train_epoch,
)

S2_720 = ('checkpoints/stage2/ETTh1/seq720_pred720/stage2_carts_softset_s2_ETTh1_720_S0_wce_RelationStage2_ETTh1_'
         'ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_'
         'sl720_pl720_0/checkpoint.pth')
S0_CKPT = REPO_ROOT / 'checkpoints/track_m_relevance_constrained_multislot01/stage2/ETTh1_720/S0_base/checkpoint.pth'

CONSUMER_TRAINABLE = ('relation_mixer', 'gate')  # audited: the only forward-path-used retrieval consumers
EXTRA_FROZEN = ('relation_concat_projection',)   # provably unused by this config's forward path

CACHE_DIRS = {
    'O1_J1_uniform': REPO_ROOT / 'results/TRACK-N-FORECAST-CONDITIONAL-UTILITY01/gate_only/cache/ETTh1_720/G0_J1',
    'O2_J1_host': REPO_ROOT / 'results/TRACK-M-RELEVANCE-CONSTRAINED-MULTISLOT01/stage2/cache/ETTh1_720/S1_J1',
    'O3_M2_uniform': REPO_ROOT / 'results/TRACK-N-FORECAST-CONDITIONAL-UTILITY01/gate_only/cache/ETTh1_720/G2_M2',
    'O4_M2_host': REPO_ROOT / 'results/TRACK-M-RELEVANCE-CONSTRAINED-MULTISLOT01/stage2/cache/ETTh1_720/S3_Mstar',
}


def build_model_for_arm(seed=0):
    exp, args, model, host_ck = build_fresh_stage2(S2_720, seed=seed)
    return exp, args, model, host_ck


def load_only_base_head(model, device):
    bl = torch.load(S0_CKPT, map_location=device)
    base_head_state = {k[len('base_head.'):]: v for k, v in bl['model_state_dict'].items()
                       if k.startswith('base_head.')}
    assert len(base_head_state) > 0, '[ISSUE][ABORT] no base_head.* keys found in S0 checkpoint'
    model.base_head.load_state_dict(base_head_state)
    return state_sha(model.base_head.state_dict())


def freeze_non_consumer(model):
    frozen_shas = {}
    for name in ('base_head',) + EXTRA_FROZEN + FREEZE_SUBMODULES:
        sub = getattr(model, name, None)
        if sub is None:
            continue
        for p in sub.parameters():
            p.requires_grad_(False)
        sub.eval()
        frozen_shas[name] = state_sha(sub.state_dict())
    for name in CONSUMER_TRAINABLE:
        sub = getattr(model, name, None)
        assert sub is not None, f'[ISSUE][ABORT] expected consumer submodule {name} not found on model'
        assert any(p.requires_grad for p in sub.parameters()), f'{name} must remain trainable'
    return frozen_shas


def consumer_init_sha(model):
    return {name: state_sha(getattr(model, name).state_dict()) for name in CONSUMER_TRAINABLE}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=tuple(CACHE_DIRS.keys()))
    ap.add_argument('--cell', default='ETTh1_720')
    ap.add_argument('--checkpoints', default='checkpoints/track_o_frozen_base_retrieval_consumer01')
    ap.add_argument('--out_dir', default='results/TRACK-O-FROZEN-BASE-RETRIEVAL-CONSUMER01/ETTh1_720')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train_epochs', type=int, default=10)
    ap.add_argument('--patience', type=int, default=5)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp, args, model, host_ck = build_model_for_arm(seed=cli.seed)
    model.to(device)
    exp._ensure_memory()
    exp._build_key_bank(force=True)

    consumer_init_before_load = consumer_init_sha(model)  # captured before base_head load (unaffected by it anyway)
    base_head_sha = load_only_base_head(model, device)
    frozen_shas = freeze_non_consumer(model)
    consumer_init_after = consumer_init_sha(model)
    assert consumer_init_before_load == consumer_init_after, \
        'loading base_head must not perturb relation_mixer/gate init'

    trainable_names = sorted(n for n, p in model.named_parameters() if p.requires_grad)
    optimizer = exp._select_optimizer()

    cache_dir = CACHE_DIRS[cli.arm]
    train_cache, train_lut = load_cache_as_lookup(cache_dir / 'train.pt')
    val_cache, val_lut = load_cache_as_lookup(cache_dir / 'val.pt')
    test_cache, test_lut = load_cache_as_lookup(cache_dir / 'test.pt')

    _, train_loader = exp._get_data(flag='train', shuffle=True)
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)

    out_dir = Path(cli.out_dir) / cli.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints) / cli.cell / cli.arm
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = {'arm': cli.arm, 'seed': cli.seed, 'base_head_sha256': base_head_sha,
                  'consumer_init_sha256': consumer_init_after, 'frozen_submodule_sha256_before': frozen_shas,
                  'trainable_parameter_names': trainable_names,
                  'trainable_parameter_count': sum(p.numel() for n, p in model.named_parameters() if p.requires_grad),
                  'cache_dir': str(cache_dir)}
    (out_dir / 'config.json').write_text(json.dumps(fingerprint, indent=2))
    print(f'[track_o] {cli.arm} base_head_sha={base_head_sha[:16]} consumer_sha={consumer_init_after} '
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
        print(f"[track_o] {cli.arm} epoch {epoch} train_loss={tr_loss:.5f} val_final_mse={va['final_mse']:.6f} "
             f"gate_mean={va['gate_mean']:.4f}")
        if epoch - best['epoch'] >= cli.patience:
            print(f'[track_o] early stop at epoch {epoch} (best={best["epoch"]})')
            break

    frozen_shas_after = freeze_non_consumer(model)
    mismatched = {k: v for k, v in frozen_shas_after.items() if frozen_shas.get(k) != v}
    if mismatched:
        raise SystemExit(f'[ISSUE][ABORT] frozen submodule(s) changed during training: {mismatched}')

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
    (out_dir / 'fingerprint.json').write_text(json.dumps(
        {**fingerprint, 'frozen_submodule_sha256_after': frozen_shas_after, 'wall_clock_seconds': time.time() - t0},
        indent=2))
    print(f"[track_o] done. {cli.arm} best_epoch={best['epoch']} test_final_mse={te_best['final_mse']:.6f}")


if __name__ == '__main__':
    main()
