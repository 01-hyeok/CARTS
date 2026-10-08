#!/usr/bin/env python3
"""TRACK-V-PROFESSOR-FUSION01 -- Professor-paper-style Stage-2 evaluator.

Stage-2 Mode: Professor-paper-style Validation-Only Scalar Trust Fusion
(as in "Which Histories Matter for Time Series Forecasting? Learning
Predictive Relevance with Future Supervision"), NOT the existing CARTS
trainable-lambda Stage-2 (`scripts/train_r_stage2_lambda01.py`, left
completely UNMODIFIED -- this is a separate, new file).

    Y_final(beta) = B(q) + beta * (R(q) - B(q))

`beta` is NEVER optimizer-trained. It is chosen by a plain validation
-only grid search over a FIXED candidate set, then frozen and applied
to the test split exactly once:

    beta* = argmin_{beta in GRID} validation_MSE(beta)   (ties -> smaller beta)

No train loop, no optimizer, no `.backward()`, no `requires_grad=True`
parameter anywhere in this file. Test labels are never read by the
selection function -- `select_beta_on_validation` takes ONLY
validation tensors as arguments, structurally incapable of touching
test.

Reuses `load_tensors`/`BaseForecastHead`-loading logic from the
EXISTING `train_r_stage2_lambda01.py` (`load_tensors`, imported
directly, never copied/modified) so `B(q)` and `R(q)` are computed
identically to the existing pipeline -- `R(q)` is already the
uniform-mean-of-Top-K-candidate-futures the cache stores (no
score-weighted aggregation added here).

No channel-first (A1) optimization is applied in this file: unlike
the candidate-encoder trainers, `BaseForecastHead.forward` loops over
channels only to apply one tiny per-channel `Linear(seq_len,pred_len)`
(no `transform_relation_features` call, no candidate re-encoding) --
there is nothing of the shape A1 optimizes, so applying it here would
be a cosmetic no-op, not a real speedup. Retrieval cache reads are a
plain tensor index lookup, not a re-encode.
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage2 import BaseForecastHead
from scripts.rng_control01 import set_global_seeds
from scripts.train_j_shared_encoder_drift01 import state_hash
from scripts.train_margutil01 import build_experiment
from scripts.train_r_stage2_lambda01 import load_tensors

BETA_GRID = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20]
FIXED_BETA = 0.10


def fused_metrics(B, R, Y, beta):
    """Y_final = B + beta*(R-B). No gradient anywhere -- caller is
    expected to be under torch.no_grad() (enforced by main())."""
    fused = B + beta * (R - B)
    mse = float(((fused - Y) ** 2).mean())
    mae = float((fused - Y).abs().mean())
    return mse, mae


def select_beta_on_validation(B_va, R_va, Y_va, grid=BETA_GRID):
    """Validation-only beta selection. Signature takes ONLY validation
    tensors -- there is no parameter through which a test tensor could
    be passed in, by construction (spec section 17 test 4)."""
    rows = []
    best_beta, best_mse = None, float('inf')
    for beta in grid:  # ascending; strict '<' below means ties keep the smaller beta already seen
        mse, mae = fused_metrics(B_va, R_va, Y_va, beta)
        rows.append({'beta': beta, 'val_mse': mse, 'val_mae': mae})
        if mse < best_mse:
            best_mse, best_beta = mse, beta
    return best_beta, best_mse, rows


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
    ap.add_argument('--candidate_support', required=True, choices=('full', 'pool_top100'))
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--gpu_index', type=int, default=1)
    cli = ap.parse_args()

    t_total0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_before = gpu_snapshot(cli.gpu_index)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    set_global_seeds(cli.seed)
    t_setup0 = time.time()
    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.seq_len, 'batch_size': 32, 'seed': cli.seed,
    })
    channels = int(args.enc_in)
    base = BaseForecastHead(seq_len=cli.seq_len, pred_len=cli.pred_len, channels=channels,
                            mode='per_channel_linear').to(device)
    bl = torch.load(cli.base_checkpoint, map_location=device)
    base.load_state_dict(bl['model_state_dict'])
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    base_sha = state_hash(base)
    t_setup1 = time.time()

    with torch.no_grad():
        t_val0 = time.time()
        va_starts, B_va, R_va, Y_va = load_tensors(exp, base, cli.cache_dir, 'val', device, channels)
        t_val1 = time.time()
        te_starts, B_te, R_te, Y_te = load_tensors(exp, base, cli.cache_dir, 'test', device, channels)
        t_test_load1 = time.time()

        base_val_mse, base_val_mae = fused_metrics(B_va, R_va, Y_va, 0.0)
        ret_val_mse, ret_val_mae = fused_metrics(B_va, R_va, Y_va, 1.0)
        base_test_mse, base_test_mae = fused_metrics(B_te, R_te, Y_te, 0.0)
        ret_test_mse, ret_test_mae = fused_metrics(B_te, R_te, Y_te, 1.0)

        t_sweep0 = time.time()
        best_beta, best_val_mse, sweep_rows = select_beta_on_validation(B_va, R_va, Y_va)
        t_sweep1 = time.time()
        best_val_mse_check, best_val_mae = fused_metrics(B_va, R_va, Y_va, best_beta)
        assert abs(best_val_mse_check - best_val_mse) < 1e-9

        fixed_val_mse, fixed_val_mae = fused_metrics(B_va, R_va, Y_va, FIXED_BETA)

        t_final0 = time.time()
        final_test_mse, final_test_mae = fused_metrics(B_te, R_te, Y_te, best_beta)
        t_final1 = time.time()
        fixed_test_mse, fixed_test_mae = fused_metrics(B_te, R_te, Y_te, FIXED_BETA)

    t_total1 = time.time()
    gpu_after = gpu_snapshot(cli.gpu_index)
    peak_alloc = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
    peak_reserved = torch.cuda.max_memory_reserved(device) / 2**20 if device.type == 'cuda' else 0.0

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = {
        'stage2_mode': 'Professor-paper-style Validation-Only Scalar Trust Fusion',
        'previous_carts_stage2': 'trainable global lambda (train_r_stage2_lambda01.py) -- NOT used here',
        'arm': cli.arm, 'candidate_support': cli.candidate_support,
        'pred_len': cli.pred_len, 'seq_len': cli.seq_len, 'seed': cli.seed,
        'beta_grid': BETA_GRID, 'fixed_beta_control': FIXED_BETA,
        'reference_ckpt': cli.reference_ckpt, 'base_checkpoint': cli.base_checkpoint,
        'base_checkpoint_sha256': base_sha, 'cache_dir': cli.cache_dir,
        'channel_first_optimization_applicable': False,
        'channel_first_optimization_reason': 'no transform_relation_features/candidate re-encode in this '
                                             'evaluator -- BaseForecastHead per-channel loop is a tiny linear '
                                             'layer, cache reads are plain index lookups; A1 is a no-op here',
    }
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2))

    metrics = {
        'base_val_mse': base_val_mse, 'base_val_mae': base_val_mae,
        'base_test_mse': base_test_mse, 'base_test_mae': base_test_mae,
        'retrieval_only_val_mse': ret_val_mse, 'retrieval_only_val_mae': ret_val_mae,
        'retrieval_only_test_mse': ret_test_mse, 'retrieval_only_test_mae': ret_test_mae,
        'fixed_beta': FIXED_BETA, 'fixed_beta_val_mse': fixed_val_mse, 'fixed_beta_val_mae': fixed_val_mae,
        'fixed_beta_test_mse': fixed_test_mse, 'fixed_beta_test_mae': fixed_test_mae,
        'selected_beta': best_beta, 'selected_beta_val_mse': best_val_mse, 'selected_beta_val_mae': best_val_mae,
        'selected_beta_test_mse': final_test_mse, 'selected_beta_test_mae': final_test_mae,
        'improvement_vs_base_pct': (base_test_mse - final_test_mse) / base_test_mse * 100.0,
        'n_val': int(B_va.size(0)), 'n_test': int(B_te.size(0)),
    }
    (out_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2))

    with open(out_dir / 'val_beta_sweep.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['beta', 'val_mse', 'val_mae'])
        w.writeheader()
        for r in sweep_rows:
            w.writerow(r)

    resource = {
        'gpu_index': cli.gpu_index, 'gpu_before': gpu_before, 'gpu_after': gpu_after,
        'peak_allocated_mib': peak_alloc, 'peak_reserved_mib': peak_reserved,
        'timing_sec': {
            'setup_time': t_setup1 - t_setup0,
            'val_tensor_load_time_incl_base_inference': t_val1 - t_val0,
            'test_tensor_load_time_incl_base_inference': t_test_load1 - t_val1,
            'beta_sweep_time': t_sweep1 - t_sweep0,
            'final_test_eval_time': t_final1 - t_final0,
            'total_wall_clock': t_total1 - t_total0,
        },
    }
    (out_dir / 'resource_metrics.json').write_text(json.dumps(resource, indent=2))

    print(f'[eval_professor_fusion] arm={cli.arm} support={cli.candidate_support} '
         f'base_test_mse={base_test_mse:.6f} selected_beta={best_beta} '
         f'final_test_mse={final_test_mse:.6f} improvement={metrics["improvement_vs_base_pct"]:+.2f}% '
         f'wall={resource["timing_sec"]["total_wall_clock"]:.1f}s')


if __name__ == '__main__':
    main()
