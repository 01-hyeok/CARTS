#!/usr/bin/env python3
"""TRACK-HEAD-UTILITY-ALIGNMENT01 -- read-only diagnostic. Loads an
EXISTING TRACK-HARD-EXPERT-V5-P100-ALLH01 Mean-Mixture-selected
checkpoint (never modified, never retrained) and the existing P100
candidate pool (`results/TRACK-V-MULTIQUERY-GENERALIZATION01/<ds>/H<h>/seed0/pool_top100/shared_candidate_pool`,
confirmed bug-independent during TRACK-V-SHARED-TOP100-SIGNFIX01), and
computes per-query Individual / Aggregate / Final-Fusion head
utilities, winner agreement/correlation, oracle-vs-fixed-head
headroom under all three criteria, a validation-only beta-grid Stage-2
calibration CONTROL (never the main result), and the closed-form
optimal scalar alpha* on train/val/test to diagnose train-to-test
trust-calibration mismatch. No training, no backward pass, no
optimizer anywhere in this file.
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
from scripts.eval_professor_style_fusion01 import BETA_GRID, fused_metrics, select_beta_on_validation
from scripts.train_j_shared_encoder_drift01 import build_model
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_r_stage2_lambda01 import load_tensors
from scripts.train_retriever_pool01 import CandidatePoolCache
from utils.head_utility_alignment01 import (
    agreement, closed_form_alpha, confusion_matrix, correlation_summary, fixed_head_from_val,
    oracle_vs_fixed_test, per_query_head_quantities, spearman_per_query, winner_fraction_stats, winners,
)

NUM_SLOTS = 5
TOP_K = 10


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
    ap.add_argument('--arm_cache_dir', required=True, help='existing ALLH01 arm cache dir (for Mean-Mixture R)')
    ap.add_argument('--arm_stage2_metrics', required=True, help='existing ALLH01 arm stage2/metrics.json (frozen lambda)')
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
    model.eval()
    slot_heads.eval()
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
    load_seconds = time.time() - t0

    t0 = time.time()
    per_split = {}
    for split, loader in (('train', train_loader), ('val', val_loader), ('test', test_loader)):
        per_split[split] = per_query_head_quantities(model, slot_heads, base, lambda_arm, exp, args, channels,
                                                      device, loader, pool_caches[split])
    head_inference_seconds = time.time() - t0

    t0 = time.time()
    config = {
        'exp': 'TRACK-HEAD-UTILITY-ALIGNMENT01', 'dataset': cli.ds, 'horizon': cli.h, 'arm': cli.arm,
        'retriever_checkpoint': cli.retriever_checkpoint, 'base_checkpoint': cli.base_checkpoint,
        'lambda_arm_frozen': lambda_arm, 'tau_s': cli.tau_s, 'top_k': TOP_K, 'num_slots': NUM_SLOTS,
        'note_u_ind_u_agg_are_channel_averaged_per_query': True,
        'note_u_final_uses_a_single_fixed_head_index_across_all_channels_per_query': True,
    }
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2))

    torch.save({s: {k: v for k, v in per_split[s].items()} for s in per_split}, out_dir / 'per_query_head_utilities.pt')

    w = {s: {crit: winners(per_split[s][f'U_{crit}']) for crit in ('ind', 'agg', 'final')} for s in per_split}

    winner_alignment = {}
    winner_confusion = {}
    for s in ('train', 'val', 'test'):
        winner_alignment[s] = {
            'agreement_ind_agg': agreement(w[s]['ind'], w[s]['agg']),
            'agreement_ind_final': agreement(w[s]['ind'], w[s]['final']),
            'agreement_agg_final': agreement(w[s]['agg'], w[s]['final']),
            'rho_ind_agg': correlation_summary(spearman_per_query(per_split[s]['U_ind'], per_split[s]['U_agg'])),
            'rho_ind_final': correlation_summary(spearman_per_query(per_split[s]['U_ind'], per_split[s]['U_final'])),
            'rho_agg_final': correlation_summary(spearman_per_query(per_split[s]['U_agg'], per_split[s]['U_final'])),
        }
        winner_confusion[s] = {
            'ind_vs_agg': confusion_matrix(w[s]['ind'], w[s]['agg']).tolist(),
            'ind_vs_final': confusion_matrix(w[s]['ind'], w[s]['final']).tolist(),
            'agg_vs_final': confusion_matrix(w[s]['agg'], w[s]['final']).tolist(),
            'winner_fraction_ind': winner_fraction_stats(w[s]['ind']),
            'winner_fraction_agg': winner_fraction_stats(w[s]['agg']),
            'winner_fraction_final': winner_fraction_stats(w[s]['final']),
        }
    (out_dir / 'winner_alignment.json').write_text(json.dumps(winner_alignment, indent=2))
    (out_dir / 'winner_confusion_matrices.json').write_text(json.dumps(winner_confusion, indent=2))

    oracle_headroom = {}
    for crit in ('ind', 'agg', 'final'):
        fh = fixed_head_from_val(per_split['val'][f'U_{crit}'])
        oracle_headroom[crit] = oracle_vs_fixed_test(per_split['test'][f'U_{crit}'], fh)
    (out_dir / 'oracle_headroom.json').write_text(json.dumps(oracle_headroom, indent=2))

    # ---- Table 3 baselines ----
    base_mse_test = float(per_split['test']['B_mse'].mean())
    mean_mixture_mse_test = json.load(open(Path(cli.arm_cache_dir) / 'stage1_metrics.json'))['agg_mse10']
    fh_agg = oracle_headroom['agg']['fixed_head']
    fixed_head_mse_test = float(per_split['test']['U_agg'][:, fh_agg].mean())
    agg_oracle_mse_test = oracle_headroom['agg']['oracle_test_mse']
    final_oracle_mse_test = oracle_headroom['final']['oracle_test_mse']
    uniform_ensemble_mse_test = float(per_split['test']['U_agg'].mean(dim=1).mean())  # proxy: mean over heads of each head's own agg MSE is NOT the same as MSE of the mean forecast, flagged below
    baselines = {
        'base_mse': base_mse_test, 'mean_mixture_mse': mean_mixture_mse_test,
        'fixed_head_mse': fixed_head_mse_test, 'aggregate_oracle_mse': agg_oracle_mse_test,
        'final_fusion_oracle_mse': final_oracle_mse_test,
        'note_uniform_ensemble_mse_is_approximate': 'mean_h(U_agg) upper-bounds true Uniform-Ensemble MSE by Jensen; not the exact R_uniform forecast MSE',
        'uniform_ensemble_mse_upper_bound_proxy': uniform_ensemble_mse_test,
    }
    (out_dir / 'baselines.json').write_text(json.dumps(baselines, indent=2))

    # ---- Table 4: Stage-2 calibration control (validation-only beta grid on the SAME Mean-Mixture R) ----
    base_mdl = base  # reuse already-loaded frozen base
    B_tr, R_tr, Y_tr = load_tensors(exp, base_mdl, cli.arm_cache_dir, 'train', device, int(args.enc_in))[1:]
    B_va, R_va, Y_va = load_tensors(exp, base_mdl, cli.arm_cache_dir, 'val', device, int(args.enc_in))[1:]
    B_te, R_te, Y_te = load_tensors(exp, base_mdl, cli.arm_cache_dir, 'test', device, int(args.enc_in))[1:]
    best_beta, best_val_mse, beta_rows = select_beta_on_validation(B_va, R_va, Y_va, grid=BETA_GRID)
    beta_test_mse, beta_test_mae = fused_metrics(B_te, R_te, Y_te, best_beta)
    original_mse = json.load(open(cli.arm_stage2_metrics))['test_mse']
    stage2_calibration = {
        'original_lambda': lambda_arm, 'original_test_mse': original_mse,
        'val_selected_beta': best_beta, 'beta_control_test_mse': beta_test_mse, 'beta_control_test_mae': beta_test_mae,
        'base_test_mse': float(((B_te - Y_te) ** 2).mean()),
        'beta_sweep_val': beta_rows,
    }
    (out_dir / 'stage2_calibration.json').write_text(json.dumps(stage2_calibration, indent=2))

    # ---- Table 5: alpha* domain shift ----
    alpha_domain_shift = {
        'alpha_star_train': closed_form_alpha(B_tr, R_tr, Y_tr),
        'alpha_star_val': closed_form_alpha(B_va, R_va, Y_va),
        'alpha_star_test_DIAGNOSTIC_ONLY_NEVER_USED_AS_PREDICTION': closed_form_alpha(B_te, R_te, Y_te),
        'retrieval_mse_train': float(((R_tr - Y_tr) ** 2).mean()),
        'retrieval_mse_val': float(((R_va - Y_va) ** 2).mean()),
        'retrieval_mse_test': float(((R_te - Y_te) ** 2).mean()),
    }
    (out_dir / 'alpha_domain_shift.json').write_text(json.dumps(alpha_domain_shift, indent=2))
    utility_compute_seconds = time.time() - t0

    gpu_after = gpu_snapshot(cli.gpu_index)
    max_vram_mb = float(torch.cuda.max_memory_allocated(device) / 2**20) if device.type == 'cuda' else 0.0
    (out_dir / 'resource_metrics.json').write_text(json.dumps({
        'load_seconds': load_seconds, 'head_inference_seconds': head_inference_seconds,
        'utility_compute_seconds': utility_compute_seconds, 'stage2_control_seconds': 0.0,
        'total_wall_clock_seconds': time.time() - t_total0, 'gpu_index': cli.gpu_index,
        'gpu_before': gpu_before, 'gpu_after': gpu_after, 'max_vram_mb': max_vram_mb,
    }, indent=2))

    print(f'[head_utility_alignment] {cli.ds}_{cli.h}/{cli.arm} done. '
         f'agree(ind,agg)={winner_alignment["test"]["agreement_ind_agg"]:.3f} '
         f'agree(ind,final)={winner_alignment["test"]["agreement_ind_final"]:.3f} '
         f'agg_oracle_gain={oracle_headroom["agg"]["oracle_gain_pct"]:.2f}% '
         f'final_oracle_gain={oracle_headroom["final"]["oracle_gain_pct"]:.2f}% '
         f'val_beta={best_beta} beta_mse={beta_test_mse:.5f} orig_mse={original_mse:.5f} '
         f'alpha*_train={alpha_domain_shift["alpha_star_train"]:.3f} alpha*_val={alpha_domain_shift["alpha_star_val"]:.3f}')


if __name__ == '__main__':
    main()
