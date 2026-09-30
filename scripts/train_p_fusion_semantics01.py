#!/usr/bin/env python3
"""TRACK-P-FUSION-SEMANTICS-AUDIT01 -- P3 (J1+Uniform+Mixture) and P4
(M2+Uniform+Mixture), the only two genuinely NEW training runs this
track performs (P1/P2 are TRACK-O's O1/O3 reused verbatim; P5/P6 are
NO-TRAINING closed-form fixed-lambda evaluations).

Identical to `train_o_frozen_base_consumer01.py` in every respect
EXCEPT: (a) `args.fusion_mode` is overridden to `'mixture'` before
`Exp_Stage2_Relation(args)` construction (PART 9: this must be the ONLY
difference from the residual arms), and (b) per PART 11, `relation_mixer`
is explicitly frozen here (not nominally-trainable-but-dead as in
TRACK-O) -- ONLY `gate.*` is trainable. Since `relation_mixer`'s
gradient is structurally zero regardless (TRACK-O's finding, re-asserted
here via the beta==1 identity-pass check, PART 7), this changes nothing
numerically, only makes the frozen set explicit and correct per this
track's own spec.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp.exp_stage2_relation import Exp_Stage2_Relation
from scripts.train_factorial_e2e01 import state_sha
from scripts.train_o_frozen_base_consumer01 import CACHE_DIRS as O_CACHE_DIRS
from scripts.train_o_frozen_base_consumer01 import S0_CKPT, S2_720, load_only_base_head
from scripts.train_setlossctrl_stage2_retrain02 import (
    FREEZE_SUBMODULES, eval_epoch, load_cache_as_lookup, train_epoch,
)

CACHE_DIRS = {
    'P3_J1_mixture': O_CACHE_DIRS['O1_J1_uniform'],  # same Uniform J1 cache, no rebuild
    'P4_M2_mixture': O_CACHE_DIRS['O3_M2_uniform'],  # same Uniform M2 cache, no rebuild
}
FROZEN_SET = ('base_head', 'relation_mixer', 'relation_concat_projection') + FREEZE_SUBMODULES


def build_fresh_stage2_mixture(stage2_host, seed, fusion_mode):
    host_ck = torch.load(stage2_host, map_location='cpu')
    args = SimpleNamespace(**host_ck['args'])
    args.num_workers = 0
    assert args.fusion_mode == 'residual', f'host default fusion_mode changed unexpectedly: {args.fusion_mode}'
    args.fusion_mode = fusion_mode  # PART 9: the ONLY architectural difference from the residual arms
    torch.manual_seed(seed)
    exp = Exp_Stage2_Relation(args)
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    assert model.gate.fusion_mode == fusion_mode
    return exp, args, model, host_ck


def freeze_all_but_gate(model):
    frozen_shas = {}
    for name in FROZEN_SET:
        sub = getattr(model, name, None)
        if sub is None:
            continue
        for p in sub.parameters():
            p.requires_grad_(False)
        sub.eval()
        frozen_shas[name] = state_sha(sub.state_dict())
    assert any(p.requires_grad for p in model.gate.parameters())
    trainable_prefixes = sorted(set(n.split('.')[0] for n, p in model.named_parameters() if p.requires_grad))
    assert trainable_prefixes == ['gate'], f'[ISSUE][ABORT] unexpected trainable prefixes: {trainable_prefixes}'
    return frozen_shas


@torch.no_grad()
def assert_mixer_identity_pass(model, exp, cache, lut, device):
    """PART 7: relation_mixer must be a beta==1 identity pass -- y_ret_c ==
    relation_outputs[:, 0, :] exactly. Verified once, on a real batch,
    before training starts."""
    _, loader = exp._get_data(flag='val', shuffle=False)
    batch_x, _batch_y_unused, batch_start_idx = next(iter(loader))
    batch_x, _batch_y_unused, batch_start_idx = exp._move_batch(batch_x, _batch_y_unused, batch_start_idx)
    cand_mask, counts = exp._candidate_mask(batch_start_idx)
    idx = [lut[int(s)] for s in batch_start_idx.tolist()]
    relation_outputs = cache['relation_outputs'][idx].to(device)
    relation_query_embs = cache['relation_query_embs'][idx].to(device)
    assert relation_outputs.shape[2] == 1, f'expected num_source_slots=1, got {relation_outputs.shape[2]}'
    y_ret_c, beta_c, _ = model.relation_mixer(relation_outputs[:, 0], relation_query_embs[:, 0])
    assert torch.allclose(beta_c, torch.ones_like(beta_c), atol=1e-6), f'beta != 1: {beta_c[:5]}'
    assert torch.allclose(y_ret_c, relation_outputs[:, 0, 0], atol=1e-4), 'relation_mixer is not an identity pass'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=tuple(CACHE_DIRS.keys()))
    ap.add_argument('--cell', default='ETTh1_720')
    ap.add_argument('--checkpoints', default='checkpoints/track_p_fusion_semantics_audit01')
    ap.add_argument('--out_dir', default='results/TRACK-P-FUSION-SEMANTICS-AUDIT01/ETTh1_720')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train_epochs', type=int, default=10)
    ap.add_argument('--patience', type=int, default=5)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp, args, model, host_ck = build_fresh_stage2_mixture(S2_720, seed=cli.seed, fusion_mode='mixture')
    model.to(device)
    exp._ensure_memory()
    exp._build_key_bank(force=True)

    gate_init_sha = state_sha(model.gate.state_dict())
    base_head_sha = load_only_base_head(model, device)
    frozen_shas = freeze_all_but_gate(model)

    trainable_names = sorted(n for n, p in model.named_parameters() if p.requires_grad)
    optimizer = exp._select_optimizer()

    cache_dir = CACHE_DIRS[cli.arm]
    train_cache, train_lut = load_cache_as_lookup(cache_dir / 'train.pt')
    val_cache, val_lut = load_cache_as_lookup(cache_dir / 'val.pt')
    test_cache, test_lut = load_cache_as_lookup(cache_dir / 'test.pt')

    assert_mixer_identity_pass(model, exp, val_cache, val_lut, device)

    _, train_loader = exp._get_data(flag='train', shuffle=True)
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)

    out_dir = Path(cli.out_dir) / cli.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints) / cli.cell / cli.arm
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = {'arm': cli.arm, 'seed': cli.seed, 'fusion_mode': 'mixture', 'gate_init_sha256': gate_init_sha,
                  'base_head_sha256': base_head_sha, 'frozen_submodule_sha256_before': frozen_shas,
                  'trainable_parameter_names': trainable_names,
                  'trainable_parameter_count': sum(p.numel() for n, p in model.named_parameters() if p.requires_grad),
                  'cache_dir': str(cache_dir)}
    (out_dir / 'config.json').write_text(json.dumps(fingerprint, indent=2))
    print(f'[track_p] {cli.arm} gate_init_sha={gate_init_sha[:16]} base_head_sha={base_head_sha[:16]} '
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
        print(f"[track_p] {cli.arm} epoch {epoch} train_loss={tr_loss:.5f} val_final_mse={va['final_mse']:.6f} "
             f"gate_mean={va['gate_mean']:.4f}")
        if epoch - best['epoch'] >= cli.patience:
            print(f'[track_p] early stop at epoch {epoch} (best={best["epoch"]})')
            break

    frozen_shas_after = freeze_all_but_gate(model)
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
    (out_dir / 'gate_distribution.json').write_text(json.dumps(
        {k: v for k, v in te_best.items() if k.startswith('gate_')}, indent=2))
    (out_dir / 'fingerprint.json').write_text(json.dumps(
        {**fingerprint, 'frozen_submodule_sha256_after': frozen_shas_after, 'wall_clock_seconds': time.time() - t0},
        indent=2))
    print(f"[track_p] done. {cli.arm} best_epoch={best['epoch']} test_final_mse={te_best['final_mse']:.6f}")


if __name__ == '__main__':
    main()
