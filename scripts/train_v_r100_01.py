#!/usr/bin/env python3
"""TRACK-V-R100-EFFICIENCY01 -- V1/V2/V5 (SlotHeads multi-query, Full
Candidate support) under the R100 efficiency scheme.

Canonical implementation: `train_t_pure_multislot01.py --num_slots
{1,2,5}` (via `run_v_one_setting01.sh`), reused UNMODIFIED everywhere
except the two approved efficiency changes:

  A. channel-first preprocessing (`utils.full_candidate_bank.
     encode_raw_channel_first`; numerically equivalent to
     `train_factorial_e2e01.encode_raw`, see
     `tests/test_full_candidate_bank01.py`).
  B. the FULL-N candidate embedding bank (shared, unmodified, across
     every arm in a cell) is built once and refreshed every
     `--refresh_interval` optimizer steps instead of re-encoded fresh
     every step. Query encoding AND SlotHeads are unaffected (always
     fresh, gradient ON); the cached bank has `requires_grad=False`
     (candidate-side gradient OFF between refreshes -- spec section 5).

Candidate support stays the full memory bank for every slot/head at
every step -- V5 still scores `[B, 5, N]`, never reduced to Top-M/P100.
Loss (`L = KL(p_teacher || mean_slot softmax(score/tau_s))`), optimizer,
LR, batch size, epochs, patience, seed, init (per-slot seeded SlotHeads
perturbation, so W_1(S=1)==W_1(S=2)==W_1(S=5) etc. holds here exactly as
in the canonical track), candidate mask, round-robin unique Top-10
selection, and checkpoint criterion (min val retMSE@10) are all
UNCHANGED. Validation and test ALWAYS rebuild a fresh full bank (spec
section 6).

Checkpoint schema (`model_state_dict` + `slot_heads_state_dict`) is
byte-compatible with `build_t_multislot_cache01.py`, reused UNMODIFIED
for the Stage-2 retrieval cache.

Reused UNMODIFIED: `build_model`/`state_hash`
(`train_j_shared_encoder_drift01`), `individual_utility_memsafe`
(`train_factorial_e2e01`), `normalized_teacher_prob`
(`train_horizon_retrieval_expert01`), `SlotHeads`/`kl_loss_from_prob`/
`slot_overlap_penalty` (`train_k_multislot_predictive_retrieval01`),
`memory_value` (`train_margutil01`), `stable_topk_indices`
(`RelationStage1`), `round_robin_topk_selection`/
`hard_eval_decomposition`/`spearman_batch`/`slot_mechanism_diagnostics`/
`naive_round_robin_picks` (`train_t_pure_multislot01`).
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

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
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads, kl_loss_from_prob
from scripts.train_margutil01 import memory_value
from scripts.train_t_pure_multislot01 import (
    VALID_NUM_SLOTS, hard_eval_decomposition, round_robin_topk_selection,
    slot_mechanism_diagnostics, spearman_batch,
)
from utils.full_candidate_bank import FullCandidateBank, encode_raw_channel_first

TOP_K = 10


def compute_scores_cached(model, slot_heads, batch_x, bank, c):
    """Training step: query+SlotHeads fresh (grad ON), candidate from
    the cached bank (grad OFF between refreshes)."""
    z_q = encode_raw_channel_first(model, batch_x, c)
    q = slot_heads(z_q)  # [B, S, D], already L2-normalized inside SlotHeads
    k_full = F.normalize(bank.get(c), dim=-1)  # [N, D], cached
    scores = torch.einsum('bsd,nd->bsn', q, k_full)  # [B, S, N]
    return scores


def compute_scores_fresh(model, slot_heads, batch_x, memory_x, c):
    """Validation/Test: ALWAYS a fresh full bank (spec section 6)."""
    z_q = encode_raw_channel_first(model, batch_x, c)
    q = slot_heads(z_q)
    k_full = F.normalize(encode_raw_channel_first(model, memory_x, c), dim=-1)
    scores = torch.einsum('bsd,nd->bsn', q, k_full)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--num_slots', type=int, required=True, choices=VALID_NUM_SLOTS)
    ap.add_argument('--cell', default='ETTh1_720')
    ap.add_argument('--pred_len', type=int, default=720)
    ap.add_argument('--seq_len', type=int, default=720)
    ap.add_argument('--patch_len', type=int, default=16)
    ap.add_argument('--top_k', type=int, default=TOP_K)
    ap.add_argument('--tau_t', type=float, default=0.1)
    ap.add_argument('--tau_s', type=float, default=0.1)
    ap.add_argument('--slot_std', type=float, default=1e-3)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--train_epochs', type=int, default=10)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--init_seed', type=int, default=0)
    ap.add_argument('--loader_seed', type=int, default=0)
    ap.add_argument('--refresh_interval', type=int, default=100,
                    help='full-candidate-bank refresh period in optimizer steps. '
                         '1 == mathematically equivalent to always-fresh (used for the A1 benchmark).')
    ap.add_argument('--out_dir', default='results/TRACK-V-R100-EFFICIENCY01')
    ap.add_argument('--checkpoints', default='checkpoints/track_v_r100_efficiency01')
    ap.add_argument('--limit_batches', type=int, default=0, help='SMOKE/BENCHMARK ONLY')
    ap.add_argument('--smoke_test', action='store_true')
    ap.add_argument('--skip_init_hash_check', action='store_true')
    ap.add_argument('--expected_init_hash', default=None)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    arm = f'V{cli.num_slots}'
    out_dir = Path(cli.out_dir) / cli.cell / arm
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints) / cli.cell / arm
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    exp, args, model = build_model(cli, device)
    channels = list(range(int(args.enc_in)))
    init_hash = state_hash(model)
    if cli.expected_init_hash and not cli.skip_init_hash_check:
        assert init_hash == cli.expected_init_hash, \
            f'[ISSUE][ABORT] init hash mismatch: {init_hash} != {cli.expected_init_hash}'
    d_model = int(args.d_model)
    n_candidates = int(exp.memory_x.size(0))

    slot_heads = SlotHeads(d_model, n_slots=cli.num_slots, std=cli.slot_std).to(device)
    for p in model.parameters():
        p.requires_grad_(True)
    params = list(model.parameters()) + list(slot_heads.parameters())
    optimizer = torch.optim.Adam(params, lr=cli.learning_rate)

    train_gen = make_loader_generator(cli.loader_seed)
    _, train_loader = exp._get_data(flag='train', shuffle=True, generator=train_gen)
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)

    config = {'cell': cli.cell, 'arm': arm, 'num_slots': cli.num_slots, 'slot_std': cli.slot_std,
             'tau_t': cli.tau_t, 'tau_s': cli.tau_s, 'top_k': cli.top_k, 'batch_size': cli.batch_size,
             'learning_rate': cli.learning_rate, 'train_epochs': cli.train_epochs, 'patience': cli.patience,
             'init_seed': cli.init_seed, 'loader_seed': cli.loader_seed, 'n_candidates': n_candidates,
             'refresh_interval': cli.refresh_interval,
             'checkpoint_criterion': 'min val retmse10 (round-robin Top-10 individual MSE) -- '
                                     'IDENTICAL to canonical T-track/V-track criterion'}
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2))
    print(f'[v_r100] arm={arm} cell={cli.cell} N={n_candidates} init_hash={init_hash[:16]} '
         f'refresh_interval={cli.refresh_interval}')

    bank = FullCandidateBank(channels)
    t_refresh0 = time.time()
    bank.refresh(model, exp.memory_x)
    refresh_rows = [{'global_step': 0, 'wall_seconds': time.time() - t_refresh0}]

    def eval_channel(batch_x, batch_y, batch_start_idx, c, cand_mask, want_diag=False):
        scores = compute_scores_fresh(model, slot_heads, batch_x, exp.memory_x, c)
        memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
        query_future = batch_y[:, :, c]
        u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
        d_raw = -u
        oracle_idx = stable_topk_indices(d_raw.masked_fill(~cand_mask, float('inf')), cli.top_k, largest=False)
        res = hard_eval_decomposition(scores, cand_mask, memory_c, offset_c, query_future, d_raw, oracle_idx,
                                      cli.top_k)
        diag = None
        if want_diag:
            s_masked = scores.masked_fill(~cand_mask.unsqueeze(1), float('-inf'))
            p_m = torch.softmax(s_masked / cli.tau_s, dim=-1)
            spearman = spearman_batch(scores.mean(dim=1), d_raw, cand_mask)
            mech = slot_mechanism_diagnostics(scores, cand_mask, p_m, cli.tau_s)
            diag = {'spearman': spearman, **mech}
        return res, diag

    if cli.smoke_test:
        model.train()
        for bi, (batch_x, batch_y, batch_start_idx) in enumerate(train_loader):
            if bi >= 2:
                break
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            cand_mask, _ = exp._candidate_mask(batch_start_idx)
            optimizer.zero_grad()
            for c in channels:
                scores = compute_scores_cached(model, slot_heads, batch_x, bank, c)
                assert scores.shape[1] == cli.num_slots
                s_masked = scores.masked_fill(~cand_mask.unsqueeze(1), float('-inf'))
                p_m = torch.softmax(s_masked / cli.tau_s, dim=-1)
                p_bar = p_m.mean(dim=1)
                memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
                query_future = batch_y[:, :, c]
                with torch.no_grad():
                    u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
                    p_t = normalized_teacher_prob(-u, cand_mask, cli.tau_t)
                l = kl_loss_from_prob(p_t, p_bar, cand_mask)
                (l / len(channels)).backward()
            assert all(p.grad is not None for p in slot_heads.parameters())
            assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters()), \
                '[ISSUE] query-side encoder grad is zero'
            for c in channels:
                assert bank.get(c).requires_grad is False
            optimizer.step()
        vram = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
        print(f'[v_r100] {arm} SMOKE PASS max_vram_mb={vram:.1f}')
        return

    epoch_val_rows, step_rows, batch_order_hashes = [], [], []
    best = {'val_retmse10': float('inf'), 'epoch': -1}
    query_encoder_calls = 0
    t0 = time.time()
    non_refresh_step_times, refresh_step_times = [], []
    global_step = 0
    for epoch in range(1, cli.train_epochs + 1):
        model.train()
        starts = []
        for bi, (batch_x, batch_y, batch_start_idx) in enumerate(train_loader):
            if cli.limit_batches and bi >= cli.limit_batches:
                break
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t_step0 = time.time()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            starts.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx) else torch.as_tensor(batch_start_idx))
            cand_mask, _ = exp._candidate_mask(batch_start_idx)
            optimizer.zero_grad()
            batch_loss = 0.0
            for c in channels:
                scores = compute_scores_cached(model, slot_heads, batch_x, bank, c)
                query_encoder_calls += 1
                s_masked = scores.masked_fill(~cand_mask.unsqueeze(1), float('-inf'))
                p_m = torch.softmax(s_masked / cli.tau_s, dim=-1)
                p_bar = p_m.mean(dim=1)
                memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
                query_future = batch_y[:, :, c]
                with torch.no_grad():
                    u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
                    p_t = normalized_teacher_prob(-u, cand_mask, cli.tau_t)
                l = kl_loss_from_prob(p_t, p_bar, cand_mask)
                (l / len(channels)).backward()
                batch_loss += float(l.detach()) / len(channels)
            optimizer.step()
            global_step += 1
            did_refresh = False
            if global_step % cli.refresh_interval == 0:
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t_r0 = time.time()
                bank.refresh(model, exp.memory_x)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                refresh_rows.append({'global_step': global_step, 'wall_seconds': time.time() - t_r0})
                did_refresh = True
            else:
                bank.tick()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            step_wall = time.time() - t_step0
            (refresh_step_times if did_refresh else non_refresh_step_times).append(step_wall)
            step_rows.append({'global_step': global_step, 'epoch': epoch, 'batch_in_epoch': bi,
                              'train_loss': batch_loss, 'step_wall_seconds': step_wall, 'did_refresh': did_refresh})
        batch_order_hashes.append({'epoch': epoch, 'batch_order_sha256': batch_order_sha256(starts)})

        model.eval()
        val_sums, n = {}, 0
        with torch.no_grad():
            for batch_x, batch_y, batch_start_idx in val_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                cand_mask, _ = exp._candidate_mask(batch_start_idx)
                bsz = batch_x.size(0)
                per_ch = {}
                for c in channels:
                    res, _ = eval_channel(batch_x, batch_y, batch_start_idx, c, cand_mask, want_diag=False)
                    per_ch.setdefault('retmse10', []).append(res['model_ind_mse'].cpu())
                    per_ch.setdefault('agg_mse10', []).append(res['agg_mse'].cpu())
                    per_ch.setdefault('recall10', []).append(res['recall10'].cpu())
                    per_ch.setdefault('ndcg10', []).append(res['ndcg10'].cpu())
                for k_, vals in per_ch.items():
                    val_sums[k_] = val_sums.get(k_, 0.0) + torch.cat(vals).sum().item()
                n += bsz
        val_metrics = {k_: v / max(n * len(channels), 1) for k_, v in val_sums.items()}
        val_retmse10 = val_metrics['retmse10']
        val_metrics['epoch'] = epoch
        epoch_val_rows.append(val_metrics)

        payload = {'model_state_dict': model.state_dict(), 'slot_heads_state_dict': slot_heads.state_dict(),
                  'epoch': epoch, 'val_retmse10': val_retmse10, 'config': config}
        if val_retmse10 < best['val_retmse10']:
            best = {'val_retmse10': val_retmse10, 'epoch': epoch}
            torch.save(payload, ckpt_dir / 'checkpoint.pth')
        print(f'[v_r100] {arm} epoch={epoch} val_retmse10={val_retmse10:.6f} val_agg={val_metrics["agg_mse10"]:.6f} '
             f'(best={best["epoch"]}:{best["val_retmse10"]:.6f})')
        if epoch - best['epoch'] >= cli.patience:
            print(f'[v_r100] {arm} early stop at epoch {epoch} (best={best["epoch"]})')
            break

    bl = torch.load(ckpt_dir / 'checkpoint.pth', map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    slot_heads.load_state_dict(bl['slot_heads_state_dict'])
    model.eval()

    test_sums, n = {}, 0
    diag_accum = {}
    with torch.no_grad():
        for batch_x, batch_y, batch_start_idx in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            cand_mask, _ = exp._candidate_mask(batch_start_idx)
            bsz = batch_x.size(0)
            per_ch = {}
            for c in channels:
                res, diag = eval_channel(batch_x, batch_y, batch_start_idx, c, cand_mask, want_diag=True)
                per_ch.setdefault('retmse10', []).append(res['model_ind_mse'].cpu())
                per_ch.setdefault('agg_mse10', []).append(res['agg_mse'].cpu())
                per_ch.setdefault('recall10', []).append(res['recall10'].cpu())
                per_ch.setdefault('ndcg10', []).append(res['ndcg10'].cpu())
                per_ch.setdefault('D', []).append(res['D'].cpu())
                per_ch.setdefault('C', []).append(res['C'].cpu())
                for k_, v in diag.items():
                    diag_accum.setdefault(k_, []).append(v)
            for k_, vals in per_ch.items():
                test_sums[k_] = test_sums.get(k_, 0.0) + torch.cat(vals).sum().item()
            n += bsz
    test_metrics = {k_: v / max(n * len(channels), 1) for k_, v in test_sums.items()}
    test_metrics['n_queries_seen'] = n
    test_metrics['best_epoch'] = best['epoch']
    slot_mech_metrics = {k_: (sum(v) / len(v) if v else float('nan')) for k_, v in diag_accum.items()}

    vram_allocated = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
    vram_reserved = torch.cuda.max_memory_reserved(device) / 2**20 if device.type == 'cuda' else 0.0
    bank_mib = sum(bank.get(c).numel() * bank.get(c).element_size() for c in channels) / 2**20
    wall_total = time.time() - t0
    refresh_wall_total = sum(r['wall_seconds'] for r in refresh_rows)
    timing = {
        'total_training_wall_seconds': wall_total,
        'avg_sec_per_step': (sum(non_refresh_step_times) + sum(refresh_step_times)) / max(global_step, 1),
        'avg_non_refresh_step_seconds': (sum(non_refresh_step_times) / len(non_refresh_step_times)
                                         if non_refresh_step_times else float('nan')),
        'avg_refresh_step_seconds': (sum(refresh_step_times) / len(refresh_step_times)
                                     if refresh_step_times else float('nan')),
        'single_refresh_mean_seconds': (sum(r['wall_seconds'] for r in refresh_rows[1:]) / max(len(refresh_rows) - 1, 1)
                                        if len(refresh_rows) > 1 else refresh_rows[0]['wall_seconds']),
        'pct_wall_time_in_refresh': 100.0 * refresh_wall_total / max(wall_total, 1e-9),
        'total_optimizer_steps': global_step,
    }
    vram = {'peak_allocated_mib': vram_allocated, 'peak_reserved_mib': vram_reserved,
           'full_embedding_bank_mib': bank_mib}
    workload = {**bank.counters(), 'query_encoder_calls': query_encoder_calls}

    def write_csv(path, rows):
        if not rows:
            return
        fieldnames = sorted({k for r in rows for k in r})
        with open(path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    write_csv(out_dir / 'train_metrics.csv', step_rows)
    write_csv(out_dir / 'val_metrics.csv', epoch_val_rows)
    write_csv(out_dir / 'batch_order_hashes.csv', batch_order_hashes)
    write_csv(out_dir / 'refresh_metrics.csv', refresh_rows)
    (out_dir / 'final_test_metrics.json').write_text(json.dumps(test_metrics, indent=2, default=str))
    (out_dir / 'slot_mechanism_metrics.json').write_text(json.dumps(slot_mech_metrics, indent=2, default=str))
    (out_dir / 'timing_metrics.json').write_text(json.dumps(timing, indent=2))
    (out_dir / 'vram_metrics.json').write_text(json.dumps(vram, indent=2))
    (out_dir / 'workload_metrics.json').write_text(json.dumps(workload, indent=2))
    (out_dir / 'checkpoint_fingerprints.json').write_text(json.dumps(
        {'init_hash': init_hash, 'best_epoch': best['epoch'], 'final_model_hash': state_hash(model),
         'batch_order_hashes': batch_order_hashes}, indent=2))
    print(f'[v_r100] done. {arm} best_epoch={best["epoch"]} test_retMSE@10={test_metrics["retmse10"]:.6f} '
         f'test_agg_mse10={test_metrics["agg_mse10"]:.6f} n_refreshes={bank.n_refreshes} '
         f'max_vram_mb={vram_allocated:.1f} wall_seconds={wall_total:.1f}')


if __name__ == '__main__':
    main()
