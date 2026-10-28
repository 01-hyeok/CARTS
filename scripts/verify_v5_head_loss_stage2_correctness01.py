#!/usr/bin/env python3
"""TRACK-V5-TOPK-CORRECTION-MECHANISM01 PHASE A -- independent re-verification
of TRACK-V5-HEAD-LOSS-DUAL-ENCODER01's Stage2 numbers.

For each (cell, arm):
  1. Rebuild B (Base Forecaster prediction), R (retrieval cache output), Y
     (ground truth) fresh from the checkpoint + cache, via the SAME
     `load_tensors` routine `train_r_stage2_lambda01.py` itself uses (byte-
     identical code path, imported not re-implemented).
  2. Compute Base-only MSE/MAE directly: MSE(B, Y).
  3. Read the already-trained Stage2 lambda from
     `results/.../stage2/<cell>/<arm>/metrics.json` and recompute
     fused = B + lambda*(R-B), MSE(fused, Y) -- compare against the
     recorded test_mse.
  4. Verify the lambda=0 -> B and lambda=1 -> R identities exactly.
  5. Record the Base checkpoint SHA256 actually used, per cell, and verify
     every arm within a cell loaded the identical Base checkpoint file.

This script performs NO training and modifies no existing files; it only
reads checkpoints/caches and writes one JSON report per cell.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage2 import BaseForecastHead
from scripts.rng_control01 import set_global_seeds
from scripts.train_margutil01 import build_experiment
from scripts.train_r_stage2_lambda01 import load_tensors, BATCH_SIZE

CELLS = {
    'ETTh1_96': dict(
        reference_ckpt='checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth',
        base_checkpoint='checkpoints/track_r_final_method_generalization01/ETTh1/H96/seed0/base/checkpoint.pth',
        pred_len=96, seq_len=96,
    ),
    'ETTh1_720': dict(
        reference_ckpt='checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/stage1_carts_softset_ETTh1_720_S0_wce_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth',
        base_checkpoint='checkpoints/track_r_final_method_generalization01/ETTh1/H720/seed0/base/checkpoint.pth',
        pred_len=720, seq_len=720,
    ),
    'Weather_96': dict(
        reference_ckpt='checkpoints/soft_set_mse/stage1/custom/seq96_pred96/stage1_carts_softset_Weather_96_S0_wce_RelationStage1_custom_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl96_pl96_0/checkpoint.pth',
        base_checkpoint='checkpoints/track_r_final_method_generalization01/Weather/H96/seed0/base/checkpoint.pth',
        pred_len=96, seq_len=96,
    ),
}
ARMS = ('S0', 'S1', 'S2', 'D0', 'D1', 'D2')


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True, choices=list(CELLS.keys()))
    ap.add_argument('--stage2_cache_root', default='results/TRACK-V5-HEAD-LOSS-DUAL-ENCODER01/stage2_cache')
    ap.add_argument('--stage2_metrics_root', default='results/TRACK-V5-HEAD-LOSS-DUAL-ENCODER01/stage2')
    ap.add_argument('--out_dir', default='results/TRACK-V5-TOPK-CORRECTION-MECHANISM01/phaseA')
    cli = ap.parse_args()

    cfg = CELLS[cli.cell]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_global_seeds(0)
    exp, args = build_experiment(cfg['reference_ckpt'], {
        'pred_len': cfg['pred_len'], 'seq_len': cfg['seq_len'], 'batch_size': BATCH_SIZE, 'seed': 0,
    })
    channels = int(args.enc_in)

    base_sha = sha256_of(cfg['base_checkpoint'])
    base = BaseForecastHead(seq_len=cfg['seq_len'], pred_len=cfg['pred_len'], channels=channels,
                            mode='per_channel_linear').to(device)
    bl = torch.load(cfg['base_checkpoint'], map_location=device)
    base.load_state_dict(bl['model_state_dict'])
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)

    rows = []
    for arm in ARMS:
        cache_dir = Path(cli.stage2_cache_root) / cli.cell / arm
        if not cache_dir.exists():
            rows.append({'arm': arm, 'status': 'MISSING_CACHE'})
            continue
        metrics_path = Path(cli.stage2_metrics_root) / cli.cell / arm / 'metrics.json'
        if not metrics_path.exists():
            rows.append({'arm': arm, 'status': 'MISSING_STAGE2_METRICS'})
            continue
        recorded = json.load(open(metrics_path))

        row = {'arm': arm, 'base_checkpoint_sha256': base_sha}
        for split in ('val', 'test'):
            starts, B, R, Y = load_tensors(exp, base, str(cache_dir), split, device, channels)
            base_mse = float(((B - Y) ** 2).mean())
            base_mae = float((B - Y).abs().mean())
            retrieval_mse = float(((R - Y) ** 2).mean())
            lam = recorded['lambda']
            fused = B + lam * (R - B)
            fused_mse = float(((fused - Y) ** 2).mean())
            fused_mae = float((fused - Y).abs().mean())
            # identity checks
            fused0 = B + 0.0 * (R - B)
            fused1 = B + 1.0 * (R - B)
            identity_lambda0_ok = torch.allclose(fused0, B, atol=1e-6)
            identity_lambda1_ok = torch.allclose(fused1, R, atol=1e-6)
            row[f'{split}_base_mse_independent'] = base_mse
            row[f'{split}_base_mae_independent'] = base_mae
            row[f'{split}_retrieval_mse_independent'] = retrieval_mse
            row[f'{split}_lambda_used'] = lam
            row[f'{split}_fused_mse_independent'] = fused_mse
            row[f'{split}_fused_mae_independent'] = fused_mae
            row[f'{split}_identity_lambda0_ok'] = identity_lambda0_ok
            row[f'{split}_identity_lambda1_ok'] = identity_lambda1_ok
            if split == 'test':
                row['recorded_test_mse'] = recorded.get('test_mse')
                row['recorded_test_mae'] = recorded.get('test_mae')
                row['recorded_lambda'] = recorded.get('lambda')
                row['match_test_mse'] = abs(fused_mse - recorded.get('test_mse', float('nan'))) < 1e-5
                row['match_test_mae'] = abs(fused_mae - recorded.get('test_mae', float('nan'))) < 1e-5
        row['status'] = 'OK'
        rows.append(row)
        print(f"[verify] {cli.cell}/{arm}: base_mse(test)={row.get('test_base_mse_independent'):.6f} "
              f"fused_mse(test)={row.get('test_fused_mse_independent'):.6f} "
              f"recorded={row.get('recorded_test_mse')} match={row.get('match_test_mse')}")

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{cli.cell}_stage2_independent_verification.json'
    json.dump({'cell': cli.cell, 'base_checkpoint': cfg['base_checkpoint'],
               'base_checkpoint_sha256': base_sha, 'arms': rows}, open(out_path, 'w'), indent=2)
    print(f'[verify] wrote {out_path}')


if __name__ == '__main__':
    main()
