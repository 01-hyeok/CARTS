#!/usr/bin/env python3
"""TRACK-M-RELEVANCE-CONSTRAINED-MULTISLOT01.

Direct extension of TRACK-K's K2 (multislot_aggregate) arm -- PART 0
constraint: dataset, seq_len, pred_len, MLP encoder, 10-slot architecture,
StopGrad-Key, slot init, candidate universe/mask, teacher, tau_t, tau_s,
L_soft, K2's aggregate objective, overlap objective, optimizer, LR, batch
size, and hard inference rule are ALL byte-identical to K2 -- this script
adds exactly one new loss term per arm (M1: soft relevance penalty; M2:
relevance budget hinge) and a new constrained checkpoint-selection rule
(PART 8). Nothing else differs from `train_k_multislot_predictive_retrieval01.py`.

M1 (soft_relevance, CONTROL arm):  L = L_K2 + gamma * R_set
M2 (relevance_budget, PRIMARY arm): L = L_K2 + gamma * ReLU(R_set - (1+delta)*T_J1(q,c))

R_m = sum_{i in Top32(s_m)} softmax(s_mi/tau_s) * d_i   (d_i = raw future-MSE,
i.e. exactly K2's soft_aggregate_loss Top-32 gather, applied to the SCALAR
future-MSE target instead of the future vector -- same selection, same
temperature, same L_soft=32; future used only as the loss weight target,
selection is student-score-only). R_set = mean_m R_m.

T_J1(q,c) is J1's OWN frozen train-split individual relevance, precomputed
once by `precompute_m_j1_reference01.py` (train-only, no val/test leakage)
and looked up by (query_start_idx, channel) here -- never recomputed
in-loop.

gamma=1.0, delta=0.05 FIXED this round (PART 6: no coefficient sweep).

Reused UNMODIFIED from `train_k_multislot_predictive_retrieval01`:
SlotHeads, kl_loss_from_prob, slot_overlap_penalty, soft_aggregate_loss,
hard_unique_selection, compute_scores, hard_eval_decomposition.
Reused UNMODIFIED from elsewhere: `build_model`/`state_hash`
(`train_j_shared_encoder_drift01`), `encode_raw`/`individual_utility_memsafe`
(`train_factorial_e2e01`), `normalized_teacher_prob`
(`train_horizon_retrieval_expert01`), `memory_value` (`train_margutil01`),
`stable_topk_indices` (`RelationStage1`), `recall_at_k`/`ndcg_at_k`
(`train_patch_retrieval_expert01`), `make_loader_generator`/
`batch_order_sha256` (`rng_control01`).
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.rng_control01 import batch_order_sha256, make_loader_generator
from scripts.train_factorial_e2e01 import individual_utility_memsafe
from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob
from scripts.train_j_shared_encoder_drift01 import build_model, state_hash
from scripts.train_k_multislot_predictive_retrieval01 import (
    L_SOFT, N_SLOTS, SlotHeads, compute_scores, hard_eval_decomposition,
    kl_loss_from_prob, slot_overlap_penalty, soft_aggregate_loss,
)
from scripts.train_margutil01 import memory_value

EPS = 1e-8
BETA_DEFAULT = 0.05
LAMBDA_AGG_DEFAULT = 1.0
GAMMA_DEFAULT = 1.0
DELTA_DEFAULT = 0.05
FEAS_MARGIN = 0.05  # PART 8: feasible epoch iff val_retMSE <= (1+FEAS_MARGIN) * R_val_J1
EXPECTED_INIT_HASH = 'b37fa4031f538e4b5f5c522ae22e7d03613e4ee66283d2637656c7c4f872372e'
J1_REF_DIR = REPO_ROOT / 'results/TRACK-M-RELEVANCE-CONSTRAINED-MULTISLOT01/ETTh1_720/j1_reference'


def soft_relevance_loss(scores_masked, d_raw, tau_s, l_soft=L_SOFT):
    """PART 3. R_m = sum_{i in Top32(s_m)} softmax(s_mi/tau_s) * d_i.
    R_set = mean over slots. Identical Top-32 selection/softmax code path
    to `soft_aggregate_loss` (same `stable_topk_indices` call, same
    `softmax(gathered / tau_s)`), applied to the scalar `d_raw` future-MSE
    target instead of the future vector `memory_c`. Top-32 selection is
    ALWAYS student-score-only (`s_m`); `d_raw` (future-derived) is used
    only as the weighted-sum target, never for selection. No [B,S,N,H]
    tensor: `d_raw` is [B,N], gathered to [B,L] per slot."""
    bsz, s, n = scores_masked.shape
    r_sum = d_raw.new_zeros(bsz)
    for m in range(s):
        s_m = scores_masked[:, m, :]
        idx_m = stable_topk_indices(s_m, l_soft, largest=True)  # [B, L], future-blind
        gathered = s_m.gather(1, idx_m)
        w_m = torch.softmax(gathered / tau_s, dim=-1)  # [B, L]
        d_gathered = d_raw.gather(1, idx_m)  # [B, L]
        r_sum = r_sum + (w_m * d_gathered).sum(dim=1)
    return r_sum / s  # R_set, [B]


def load_j1_reference_table(device):
    df = pd.read_parquet(J1_REF_DIR / 'train_reference.parquet')
    n_q = int(df['query_start_idx'].max()) + 1
    n_c = int(df['channel'].max()) + 1
    assert len(df) == n_q * n_c, f'[ISSUE] J1 reference table not dense: {len(df)} rows != {n_q}x{n_c}'
    table = torch.zeros(n_q, n_c, dtype=torch.float32)
    table[torch.as_tensor(df['query_start_idx'].values, dtype=torch.long),
         torch.as_tensor(df['channel'].values, dtype=torch.long)] = torch.as_tensor(df['T_J1'].values,
                                                                                     dtype=torch.float32)
    return table.to(device)


def load_j1_val_baseline():
    d = json.loads((J1_REF_DIR / 'validation_baseline.json').read_text())
    return d['retmse10'], d['agg_mse10']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--arm', required=True, choices=('M1_soft_relevance', 'M2_relevance_budget'))
    ap.add_argument('--cell', default='ETTh1_720')
    ap.add_argument('--pred_len', type=int, default=720)
    ap.add_argument('--seq_len', type=int, default=720)
    ap.add_argument('--patch_len', type=int, default=16)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--tau_t', type=float, default=0.1)
    ap.add_argument('--tau_s', type=float, default=0.1)
    ap.add_argument('--beta', type=float, default=BETA_DEFAULT)
    ap.add_argument('--lambda_agg', type=float, default=LAMBDA_AGG_DEFAULT)
    ap.add_argument('--gamma', type=float, default=GAMMA_DEFAULT)
    ap.add_argument('--delta', type=float, default=DELTA_DEFAULT)
    ap.add_argument('--l_soft', type=int, default=L_SOFT)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--train_epochs', type=int, default=10)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--init_seed', type=int, default=0)
    ap.add_argument('--loader_seed', type=int, default=0)
    ap.add_argument('--out_dir', default='results/TRACK-M-RELEVANCE-CONSTRAINED-MULTISLOT01')
    ap.add_argument('--checkpoints', default='checkpoints/track_m_relevance_constrained_multislot01')
    ap.add_argument('--limit_batches', type=int, default=0, help='SMOKE ONLY')
    ap.add_argument('--smoke_test', action='store_true')
    ap.add_argument('--j1_ref_dir', default=None,
                    help='TRACK-R generalization: override J1_REF_DIR for non-ETTh1_720 settings; '
                         'defaults to the original TRACK-M path (unchanged behavior) if omitted')
    ap.add_argument('--skip_init_hash_check', action='store_true',
                    help='TRACK-R generalization: EXPECTED_INIT_HASH is ETTh1_720/seed0-specific; a '
                         'different horizon/dataset/seed legitimately produces a different init hash.')
    cli = ap.parse_args()
    if cli.j1_ref_dir is not None:
        global J1_REF_DIR
        J1_REF_DIR = Path(cli.j1_ref_dir)
    is_m2 = cli.arm == 'M2_relevance_budget'
    assert cli.gamma == GAMMA_DEFAULT and cli.delta == DELTA_DEFAULT, \
        '[ISSUE][ABORT] PART 6 forbids gamma/delta sweep -- fixed values only'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(cli.out_dir) / cli.cell / cli.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints) / cli.cell / cli.arm
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    exp, args, model = build_model(cli, device)
    channels = list(range(int(args.enc_in)))
    if not cli.skip_init_hash_check:
        assert state_hash(model) == EXPECTED_INIT_HASH, '[ISSUE][ABORT] init hash mismatch vs J0/J1/K2'
    d_model = int(args.d_model)

    slot_heads = SlotHeads(d_model, N_SLOTS).to(device)
    j1_table = load_j1_reference_table(device) if not cli.smoke_test else None
    r_val_j1, a_val_j1 = load_j1_val_baseline() if not cli.smoke_test else (float('inf'), float('inf'))
    feas_threshold = (1.0 + FEAS_MARGIN) * r_val_j1

    for p in model.parameters():
        p.requires_grad_(True)
    params = list(model.parameters()) + list(slot_heads.parameters())
    optimizer = torch.optim.Adam(params, lr=cli.learning_rate)

    train_gen = make_loader_generator(cli.loader_seed)
    _, train_loader = exp._get_data(flag='train', shuffle=True, generator=train_gen)
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)

    config = {'cell': cli.cell, 'arm': cli.arm, 'n_slots': N_SLOTS, 'l_soft': cli.l_soft, 'beta': cli.beta,
              'lambda_agg': cli.lambda_agg, 'gamma': cli.gamma, 'delta': cli.delta if is_m2 else None,
              'tau_t': cli.tau_t, 'tau_s': cli.tau_s, 'top_k': cli.top_k, 'batch_size': cli.batch_size,
              'learning_rate': cli.learning_rate, 'train_epochs': cli.train_epochs, 'patience': cli.patience,
              'init_seed': cli.init_seed, 'loader_seed': cli.loader_seed,
              'n_candidates': int(exp.memory_x.size(0)), 'r_val_j1': r_val_j1, 'a_val_j1': a_val_j1,
              'feas_threshold': feas_threshold,
              'checkpoint_criterion': 'min val agg_mse10 s.t. val_retmse10 <= 1.05*R_val_J1 (PART 8)'}
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2))
    print(f'[track_m] arm={cli.arm} N={config["n_candidates"]} init_hash={state_hash(model)[:16]} '
         f'R_val_J1={r_val_j1:.6f} feas_threshold={feas_threshold:.6f}')

    def relevance_term(s_masked, d_raw, batch_start_idx, c):
        r_set = soft_relevance_loss(s_masked, d_raw, cli.tau_s, cli.l_soft)  # [B]
        if not is_m2:
            return r_set.mean(), float(r_set.mean().detach())
        t_j1 = j1_table[torch.as_tensor(batch_start_idx, device=device, dtype=torch.long), c]  # [B]
        l_budget = F.relu(r_set - (1.0 + cli.delta) * t_j1).mean()
        return l_budget, float(l_budget.detach())

    def eval_channel(batch_x, batch_y, batch_start_idx, c):
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        scores = compute_scores(model, slot_heads, batch_x, exp.memory_x, c)
        memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
        query_future = batch_y[:, :, c]
        u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
        d_raw = -u
        oracle_idx = stable_topk_indices(d_raw.masked_fill(~cand_mask, float('inf')), cli.top_k, largest=False)
        res = hard_eval_decomposition(scores, cand_mask, memory_c, offset_c, query_future, d_raw, oracle_idx,
                                      cli.top_k)
        return res

    if cli.smoke_test:
        # tiny synthetic J1 table so the smoke test does not require the full precompute
        j1_table = torch.zeros(int(exp.memory_x.size(0)), len(channels), device=device) + 0.5
        model.train()
        for bi, (batch_x, batch_y, batch_start_idx) in enumerate(train_loader):
            if bi >= 2:
                break
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            cand_mask, _ = exp._candidate_mask(batch_start_idx)
            optimizer.zero_grad()
            for c in channels:
                scores = compute_scores(model, slot_heads, batch_x, exp.memory_x, c)
                assert scores.shape[1] == N_SLOTS
                s_masked = scores.masked_fill(~cand_mask.unsqueeze(1), float('-inf'))
                p_m = torch.softmax(s_masked / cli.tau_s, dim=-1)
                p_bar = p_m.mean(dim=1)
                memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
                query_future = batch_y[:, :, c]
                with torch.no_grad():
                    u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
                    p_t = normalized_teacher_prob(-u, cand_mask, cli.tau_t)
                d_raw = -u
                l_anchor = kl_loss_from_prob(p_t, p_bar, cand_mask)
                l_overlap = slot_overlap_penalty(p_m)
                l_agg = soft_aggregate_loss(s_masked, memory_c, offset_c, query_future, cli.tau_s, cli.l_soft)
                l = l_anchor + cli.beta * l_overlap + cli.lambda_agg * l_agg
                l_rel, _ = relevance_term(s_masked, d_raw, batch_start_idx, c)
                l = l + cli.gamma * l_rel
                (l / len(channels)).backward()
            assert all(p.grad is not None for p in slot_heads.parameters())
            optimizer.step()
        vram = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
        print(f'[track_m] {cli.arm} SMOKE PASS max_vram_mb={vram:.1f}')
        return

    epoch_val_rows, step_rows, batch_order_hashes = [], [], []
    best = {'val_agg': float('inf'), 'epoch': -1, 'feasible': False}
    diag_min_violation = {'epoch': -1, 'val_retmse10': float('inf'), 'violation': float('inf')}
    t0 = time.time()
    global_step = 0
    for epoch in range(1, cli.train_epochs + 1):
        model.train()
        starts = []
        for bi, (batch_x, batch_y, batch_start_idx) in enumerate(train_loader):
            if cli.limit_batches and bi >= cli.limit_batches:
                break
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            starts.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx) else torch.as_tensor(batch_start_idx))
            cand_mask, _ = exp._candidate_mask(batch_start_idx)
            optimizer.zero_grad()
            batch_loss, batch_anchor, batch_overlap, batch_agg, batch_rel = 0.0, 0.0, 0.0, 0.0, 0.0
            for c in channels:
                scores = compute_scores(model, slot_heads, batch_x, exp.memory_x, c)
                s_masked = scores.masked_fill(~cand_mask.unsqueeze(1), float('-inf'))
                p_m = torch.softmax(s_masked / cli.tau_s, dim=-1)
                p_bar = p_m.mean(dim=1)
                memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
                query_future = batch_y[:, :, c]
                with torch.no_grad():
                    u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
                    p_t = normalized_teacher_prob(-u, cand_mask, cli.tau_t)
                d_raw = -u
                l_anchor = kl_loss_from_prob(p_t, p_bar, cand_mask)
                l_overlap = slot_overlap_penalty(p_m)
                l_agg = soft_aggregate_loss(s_masked, memory_c, offset_c, query_future, cli.tau_s, cli.l_soft)
                l = l_anchor + cli.beta * l_overlap + cli.lambda_agg * l_agg
                l_rel, l_rel_val = relevance_term(s_masked, d_raw, batch_start_idx, c)
                l = l + cli.gamma * l_rel
                (l / len(channels)).backward()
                batch_loss += float(l.detach()) / len(channels)
                batch_anchor += float(l_anchor.detach()) / len(channels)
                batch_overlap += float(l_overlap.detach()) / len(channels)
                batch_agg += float(l_agg.detach()) / len(channels)
                batch_rel += l_rel_val / len(channels)
            optimizer.step()
            global_step += 1
            step_rows.append({'global_step': global_step, 'epoch': epoch, 'batch_in_epoch': bi,
                              'train_loss': batch_loss, 'train_anchor': batch_anchor,
                              'train_overlap': batch_overlap, 'train_agg': batch_agg, 'train_rel': batch_rel})
        batch_order_hashes.append({'epoch': epoch, 'batch_order_sha256': batch_order_sha256(starts)})

        # full validation (hard inference) -- IDENTICAL to K2's eval path
        model.eval()
        val_sums, n = {}, 0
        with torch.no_grad():
            for batch_x, batch_y, batch_start_idx in val_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                bsz = batch_x.size(0)
                per_ch = {}
                for c in channels:
                    res = eval_channel(batch_x, batch_y, batch_start_idx, c)
                    per_ch.setdefault('retmse10', []).append(res['model_ind_mse'].cpu())
                    per_ch.setdefault('agg_mse10', []).append(res['agg_mse'].cpu())
                    per_ch.setdefault('recall10', []).append(res['recall10'].cpu())
                    per_ch.setdefault('ndcg10', []).append(res['ndcg10'].cpu())
                    per_ch.setdefault('oracle_regret', []).append((res['model_ind_mse'] - res['oracle_ind_mse']).cpu())
                for k_, vals in per_ch.items():
                    val_sums[k_] = val_sums.get(k_, 0.0) + torch.cat(vals).sum().item()
                n += bsz
        val_metrics = {k_: v / max(n * len(channels), 1) for k_, v in val_sums.items()}
        val_agg = val_metrics['agg_mse10']
        val_retmse = val_metrics['retmse10']
        val_metrics['epoch'] = epoch
        epoch_val_rows.append(val_metrics)

        violation = max(0.0, val_retmse - feas_threshold)
        if violation < diag_min_violation['violation']:
            diag_min_violation = {'epoch': epoch, 'val_retmse10': val_retmse, 'violation': violation}
        feasible = val_retmse <= feas_threshold

        payload = {'model_state_dict': model.state_dict(), 'slot_heads_state_dict': slot_heads.state_dict(),
                  'epoch': epoch, 'val_agg': val_agg, 'val_retmse10': val_retmse, 'feasible': feasible,
                  'config': config}
        torch.save(payload, ckpt_dir / f'checkpoint_epoch{epoch}.pth')
        if feasible and val_agg < best['val_agg']:
            best = {'val_agg': val_agg, 'epoch': epoch, 'feasible': True}
            torch.save(payload, ckpt_dir / 'checkpoint.pth')
        print(f'[track_m] {cli.arm} epoch={epoch} val_agg={val_agg:.6f} val_retmse10={val_retmse:.6f} '
             f'feasible={feasible} (best={best["epoch"]}:{best["val_agg"]:.6f} feasible={best["feasible"]})')
        if best['epoch'] > 0 and epoch - best['epoch'] >= cli.patience:
            print(f'[track_m] {cli.arm} early stop at epoch {epoch} (best={best["epoch"]})')
            break

    no_feasible = best['epoch'] == -1
    if no_feasible:
        print(f'[track_m] {cli.arm} NO FEASIBLE CHECKPOINT (feas_threshold={feas_threshold:.6f}); '
             f'diagnostic min-violation epoch={diag_min_violation["epoch"]} '
             f'val_retmse10={diag_min_violation["val_retmse10"]:.6f} -- NOT usable for Stage2')
    else:
        bl = torch.load(ckpt_dir / 'checkpoint.pth', map_location=device)
        model.load_state_dict(bl['model_state_dict'])
        slot_heads.load_state_dict(bl['slot_heads_state_dict'])
        model.eval()

    # final test (hard inference only -- same protocol as K2's retmse10/D/C/recall/ndcg block)
    test_metrics = {'best_epoch': best['epoch'], 'no_feasible_checkpoint': no_feasible,
                    'diagnostic_min_violation_epoch': diag_min_violation}
    if not no_feasible:
        test_sums, n = {}, 0
        with torch.no_grad():
            for batch_x, batch_y, batch_start_idx in test_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                bsz = batch_x.size(0)
                per_ch = {}
                for c in channels:
                    res = eval_channel(batch_x, batch_y, batch_start_idx, c)
                    per_ch.setdefault('retmse10', []).append(res['model_ind_mse'].cpu())
                    per_ch.setdefault('agg_mse10', []).append(res['agg_mse'].cpu())
                    per_ch.setdefault('recall10', []).append(res['recall10'].cpu())
                    per_ch.setdefault('ndcg10', []).append(res['ndcg10'].cpu())
                    per_ch.setdefault('D', []).append(res['D'].cpu())
                    per_ch.setdefault('C', []).append(res['C'].cpu())
                for k_, vals in per_ch.items():
                    test_sums[k_] = test_sums.get(k_, 0.0) + torch.cat(vals).sum().item()
                n += bsz
        test_metrics.update({k_: v / max(n * len(channels), 1) for k_, v in test_sums.items()})
        test_metrics['n_queries_seen'] = n

    out_dir_top = Path(cli.out_dir) / cli.cell / cli.arm
    with open(out_dir_top / 'train_metrics.csv', 'w', newline='') as fh:
        fieldnames = sorted({k for r in step_rows for k in r})
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in step_rows:
            w.writerow(r)
    with open(out_dir_top / 'val_metrics.csv', 'w', newline='') as fh:
        fieldnames = sorted({k for r in epoch_val_rows for k in r})
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in epoch_val_rows:
            w.writerow(r)
    with open(out_dir_top / 'batch_order_hashes.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['epoch', 'batch_order_sha256'])
        w.writeheader()
        for r in batch_order_hashes:
            w.writerow(r)
    (out_dir_top / 'final_test_metrics.json').write_text(json.dumps(test_metrics, indent=2, default=str))
    (out_dir_top / 'checkpoint_fingerprint.json').write_text(json.dumps(
        {'init_hash': EXPECTED_INIT_HASH, 'best_epoch': best['epoch'], 'no_feasible_checkpoint': no_feasible,
         'final_model_hash': state_hash(model) if not no_feasible else None,
         'batch_order_hashes': batch_order_hashes}, indent=2))
    vram = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
    print(f'[track_m] done. {cli.arm} best_epoch={best["epoch"]} no_feasible={no_feasible} '
         f'test_retMSE@10={test_metrics.get("retmse10", float("nan")):.6f} max_vram_mb={vram:.1f} '
         f'wall_seconds={time.time()-t0:.1f}')


if __name__ == '__main__':
    main()
