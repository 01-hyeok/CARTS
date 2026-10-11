#!/usr/bin/env python3
"""TRACK-HEAD-UTILITY-ALIGNMENT-CORRECTION01 -- corrects two issues in
TRACK-HEAD-UTILITY-ALIGNMENT01: (A) winners are now computed at the
REAL Hard-trainer granularity, per (query, channel), not
channel-averaged first; (B) the Stage-2 calibration control beta grid
is extended to include 0.30/0.40/0.50 (this track's own control only --
`scripts/eval_professor_style_fusion01.py`'s BETA_GRID, used as the
PRIMARY result elsewhere, is left untouched). Read-only: loads an
EXISTING TRACK-HARD-EXPERT-V5-P100-ALLH01 checkpoint, no retraining,
no checkpoint modification, no backward pass.
"""
import argparse
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
from scripts.eval_professor_style_fusion01 import fused_metrics, select_beta_on_validation
from scripts.train_j_shared_encoder_drift01 import build_model
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_r_stage2_lambda01 import load_tensors
from scripts.train_retriever_pool01 import CandidatePoolCache
from utils.head_utility_alignment01 import closed_form_alpha, spearman_per_query, correlation_summary
from utils.head_utility_alignment_correction01 import (
    global_fixed_head_from_val, global_oracle_vs_fixed_test, global_winners_from_channel_avg, margin_stats,
    per_channel_fixed_head_from_val, per_channel_oracle_vs_fixed_test, per_channel_winners,
    per_query_channel_head_quantities,
)

NUM_SLOTS = 5
TOP_K = 10
BETA_GRID_EXTENDED = [0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]


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
    ap.add_argument('--ds', required=True)
    ap.add_argument('--h', type=int, required=True)
    ap.add_argument('--arm', required=True, choices=('V5', 'Soft', 'Hard'))
    ap.add_argument('--retriever_checkpoint', required=True)
    ap.add_argument('--base_checkpoint', required=True)
    ap.add_argument('--arm_cache_dir', required=True)
    ap.add_argument('--arm_stage2_metrics', required=True)
    ap.add_argument('--candidate_pool_cache', required=True)
    ap.add_argument('--candidate_pool_size', type=int, default=100)
    ap.add_argument('--candidate_mask', default='raft')
    ap.add_argument('--patch_len', type=int, default=16)
    ap.add_argument('--tau_s', type=float, required=True)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--gpu_index', type=int, default=1)
    ap.add_argument('--out_dir', required=True)
    cli = ap.parse_args()
    cli.init_seed = cli.seed
    cli.pred_len = cli.h
    cli.seq_len = cli.h
    cli.top_k = TOP_K

    t_total0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_before = gpu_snapshot(cli.gpu_index)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    exp, args, model = build_model(cli, device)
    channels = list(range(int(args.enc_in)))
    d_model = int(args.d_model)

    bl = torch.load(cli.retriever_checkpoint, map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    slot_heads = SlotHeads(d_model, n_slots=NUM_SLOTS).to(device)
    slot_heads.load_state_dict(bl['slot_heads_state_dict'])
    model.eval(); slot_heads.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in slot_heads.parameters():
        p.requires_grad_(False)

    base = BaseForecastHead(seq_len=cli.h, pred_len=cli.h, channels=int(args.enc_in),
                            mode='per_channel_linear').to(device)
    bbl = torch.load(cli.base_checkpoint, map_location=device)
    base.load_state_dict(bbl['model_state_dict'])
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)

    lambda_arm = float(json.load(open(cli.arm_stage2_metrics))['lambda'])

    expected_meta = {'seq_len': cli.seq_len, 'pred_len': cli.pred_len, 'candidate_mask': cli.candidate_mask,
                     'candidate_pool_size': cli.candidate_pool_size, 'candidate_pool_metric': 'delta_last_cosine'}
    cache_dir = Path(cli.candidate_pool_cache)
    pool_caches = {s: CandidatePoolCache(cache_dir / f'{s}.pt', expected_meta) for s in ('train', 'val', 'test')}

    _, train_loader = exp._get_data(flag='train', shuffle=False)
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)
    preprocessing_seconds = time.time() - t0

    t0 = time.time()
    per_split = {}
    for split, loader in (('train', train_loader), ('val', val_loader), ('test', test_loader)):
        per_split[split] = per_query_channel_head_quantities(model, slot_heads, exp, args, channels, device,
                                                              loader, pool_caches[split])
    inference_seconds = time.time() - t0

    t0 = time.time()
    config = {
        'exp': 'TRACK-HEAD-UTILITY-ALIGNMENT-CORRECTION01', 'dataset': cli.ds, 'horizon': cli.h, 'arm': cli.arm,
        'retriever_checkpoint': cli.retriever_checkpoint, 'base_checkpoint': cli.base_checkpoint,
        'lambda_arm_frozen': lambda_arm, 'tau_s': cli.tau_s, 'top_k': TOP_K, 'num_slots': NUM_SLOTS,
        'correction_a': 'winners computed per (query, channel), matching the real Hard trainer granularity exactly',
        'correction_b': f'beta-grid control extended to {BETA_GRID_EXTENDED}',
    }
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2))

    # ---- global vs per-channel winner agreement / correlation (test split) ----
    U_ind_te, U_agg_te = per_split['test']['U_ind'], per_split['test']['U_agg']  # [N,C,5]
    n, c, h = U_ind_te.shape
    w_ind_pc = per_channel_winners(U_ind_te)  # [N,C]
    w_agg_pc = per_channel_winners(U_agg_te)
    w_ind_g = global_winners_from_channel_avg(U_ind_te)  # [N]
    w_agg_g = global_winners_from_channel_avg(U_agg_te)

    per_channel_agree = float((w_ind_pc == w_agg_pc).float().mean())
    global_agree = float((w_ind_g == w_agg_g).float().mean())

    rho_pc = spearman_per_query(U_ind_te.reshape(n * c, h), U_agg_te.reshape(n * c, h))
    rho_pc_summary = correlation_summary(rho_pc)
    rho_g = spearman_per_query(U_ind_te.mean(dim=1), U_agg_te.mean(dim=1))
    rho_g_summary = correlation_summary(rho_g)

    winner_agreement = {
        'global_agree_ind_agg': global_agree, 'per_channel_agree_ind_agg': per_channel_agree,
        'global_rho_ind_agg': rho_g_summary, 'per_channel_rho_ind_agg': rho_pc_summary,
    }
    (out_dir / 'winner_alignment.json').write_text(json.dumps(winner_agreement, indent=2))

    confusion_pc = torch.zeros(NUM_SLOTS, NUM_SLOTS)
    idx = w_ind_pc.reshape(-1) * NUM_SLOTS + w_agg_pc.reshape(-1)
    confusion_pc += torch.bincount(idx, minlength=NUM_SLOTS * NUM_SLOTS).float().reshape(NUM_SLOTS, NUM_SLOTS)
    win_frac_pc = torch.bincount(w_ind_pc.reshape(-1), minlength=NUM_SLOTS).float()
    win_frac_pc = win_frac_pc / win_frac_pc.sum()
    p_ = win_frac_pc.clamp_min(1e-12)
    entropy_pc = float(-(p_ * p_.log()).sum()) / float(torch.log(torch.tensor(float(NUM_SLOTS))))
    margin_info = margin_stats(U_agg_te)
    (out_dir / 'winner_confusion_matrices.json').write_text(json.dumps({
        'per_channel_confusion_ind_vs_agg': confusion_pc.tolist(),
        'per_channel_winner_fraction_ind': win_frac_pc.tolist(),
        'per_channel_normalized_entropy_ind': entropy_pc,
        'max_head_usage': float(win_frac_pc.max()), 'min_head_usage': float(win_frac_pc.min()),
        'aggregate_margin_stats': margin_info,
    }, indent=2))

    # ---- Table: Global vs Per-Channel Oracle headroom (Aggregate criterion, val-selected fixed head) ----
    U_agg_va = per_split['val']['U_agg']
    fixed_pc = per_channel_fixed_head_from_val(U_agg_va)
    fixed_g = global_fixed_head_from_val(U_agg_va)
    oracle_pc = per_channel_oracle_vs_fixed_test(U_agg_te, fixed_pc)
    oracle_g = global_oracle_vs_fixed_test(U_agg_te, fixed_g)
    oracle_headroom = {'global': oracle_g, 'per_channel': {**oracle_pc, 'fixed_heads_per_channel': fixed_pc.tolist()}}
    (out_dir / 'oracle_headroom.json').write_text(json.dumps(oracle_headroom, indent=2))

    # ---- Stage-2 calibration control with EXTENDED beta grid (correction B) ----
    B_tr, R_tr, Y_tr = load_tensors(exp, base, cli.arm_cache_dir, 'train', device, int(args.enc_in))[1:]
    B_va, R_va, Y_va = load_tensors(exp, base, cli.arm_cache_dir, 'val', device, int(args.enc_in))[1:]
    B_te, R_te, Y_te = load_tensors(exp, base, cli.arm_cache_dir, 'test', device, int(args.enc_in))[1:]
    best_beta, best_val_mse, beta_rows = select_beta_on_validation(B_va, R_va, Y_va, grid=BETA_GRID_EXTENDED)
    beta_test_mse, beta_test_mae = fused_metrics(B_te, R_te, Y_te, best_beta)
    original_mse = json.load(open(cli.arm_stage2_metrics))['test_mse']
    stage2_calibration = {
        'beta_grid_used': BETA_GRID_EXTENDED, 'original_lambda': lambda_arm, 'original_test_mse': original_mse,
        'val_selected_beta': best_beta, 'beta_control_test_mse': beta_test_mse, 'beta_control_test_mae': beta_test_mae,
        'base_test_mse': float(((B_te - Y_te) ** 2).mean()), 'beta_sweep_val': beta_rows,
        'beta_hit_grid_ceiling': best_beta == max(BETA_GRID_EXTENDED),
    }
    (out_dir / 'stage2_calibration.json').write_text(json.dumps(stage2_calibration, indent=2))

    alpha_domain_shift = {
        'alpha_star_train': closed_form_alpha(B_tr, R_tr, Y_tr),
        'alpha_star_val': closed_form_alpha(B_va, R_va, Y_va),
        'alpha_star_test_DIAGNOSTIC_ONLY_NEVER_USED_AS_PREDICTION': closed_form_alpha(B_te, R_te, Y_te),
        'retrieval_mse_train': float(((R_tr - Y_tr) ** 2).mean()),
        'retrieval_mse_val': float(((R_va - Y_va) ** 2).mean()),
        'retrieval_mse_test': float(((R_te - Y_te) ** 2).mean()),
    }
    (out_dir / 'alpha_domain_shift.json').write_text(json.dumps(alpha_domain_shift, indent=2))
    calibration_seconds = time.time() - t0

    torch.save({s: per_split[s] for s in per_split}, out_dir / 'per_query_channel_head_utilities.pt')

    gpu_after = gpu_snapshot(cli.gpu_index)
    max_vram_mb = float(torch.cuda.max_memory_allocated(device) / 2**20) if device.type == 'cuda' else 0.0
    (out_dir / 'resource_metrics.json').write_text(json.dumps({
        'preprocessing_seconds': preprocessing_seconds, 'inference_seconds': inference_seconds,
        'calibration_seconds': calibration_seconds, 'total_wall_clock_seconds': time.time() - t_total0,
        'gpu_index': cli.gpu_index, 'gpu_before': gpu_before, 'gpu_after': gpu_after, 'max_vram_mb': max_vram_mb,
    }, indent=2))

    print(f'[hua_correction] {cli.ds}_{cli.h}/{cli.arm} done. '
         f'global_agree={global_agree:.3f} per_channel_agree={per_channel_agree:.3f} '
         f'global_oracle_gain={oracle_g["oracle_gain_pct"]:.2f}% per_channel_oracle_gain={oracle_pc["oracle_gain_pct"]:.2f}% '
         f'val_beta={best_beta} beta_hit_ceiling={stage2_calibration["beta_hit_grid_ceiling"]} '
         f'alpha*_train={alpha_domain_shift["alpha_star_train"]:.3f} alpha*_val={alpha_domain_shift["alpha_star_val"]:.3f}')


if __name__ == '__main__':
    main()
