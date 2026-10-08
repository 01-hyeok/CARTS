#!/usr/bin/env python3
"""TRACK-HARD-EXPERT-V5-P100-ALLH01 -- Soft Expert-V5, Shared-Top-100
(P100) candidate support. Exactly one change relative to the canonical
P100 multi-query trainer (`train_v_sharedtop100_01.py`'s V5 arm, i.e.
plain `KL(p_T || mean_h softmax(s_h/tau_s))`): the loss becomes the
responsibility-weighted per-head KL from `TRACK-EXPERT-V5-FULL01`
(`train_expert_v5_full01.py`), applied to pool-restricted scores/distance
instead of full-candidate ones. Reuses UNMODIFIED: `SlotHeads`
(`train_k_multislot_predictive_retrieval01`), `normalized_teacher_prob`
(`train_horizon_retrieval_expert01`), `memory_value` (`train_margutil01`),
`CandidatePoolCache` (`train_retriever_pool01`), `gather_candidate_values`
/`pooled_future_mse` (`utils.candidate_pool`), `compute_scores_pool_channel_first_grad`
(`utils.full_candidate_bank`, new this track but itself a pure A1
drop-in with no architecture change), `per_head_standalone_topk`/
`per_head_future_utility`/`responsibility_from_utility`/`kl_per_head`/
`expert_weighted_loss` (`utils.expert_head_metrics`, candidate-space-size
agnostic). `d_pool` (from `pooled_future_mse`) is ALREADY a correctly
signed lower-is-better distance and is passed DIRECTLY to
`normalized_teacher_prob` -- never negated (the TRACK-V-SHARED-TOP100-SIGNFIX01
fix, carried forward here from the start).

Fixed-epoch policy (standing, D-0015): exactly `--train_epochs` (10)
epochs always run, NO early stopping, every epoch checkpoint saved.
Primary checkpoint selection happens in a SEPARATE post-hoc pass
(`eval_v5_meanmix_checkpoint_selection_pool01.py`, reused unmodified --
it is architecture-only, agnostic to which loss trained the
checkpoint) using validation Mean-Mixture RetMSE@10, exactly matching
the actual inference rule. The in-training per-epoch validation
numbers logged here are diagnostics only.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.rng_control01 import batch_order_sha256, make_loader_generator
from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob
from scripts.train_j_shared_encoder_drift01 import build_model, state_hash
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_margutil01 import memory_value
from scripts.train_retriever_pool01 import CandidatePoolCache
from scripts.train_t_pure_multislot01 import round_robin_topk_selection
from utils.candidate_pool import CandidatePoolConfig, gather_candidate_values, pooled_future_mse
from utils.expert_head_metrics import (
    expert_weighted_loss, kl_per_head, per_head_future_utility, per_head_standalone_topk,
    responsibility_from_utility,
)
from utils.full_candidate_bank import compute_scores_pool_channel_first_grad
from utils.mean_mixture_selection import mean_mixture_topk_selection

TOP_K = 10
NUM_SLOTS = 5
TAU_E = 1.0  # fixed, never swept -- identical to TRACK-EXPERT-V5-FULL01


def soft_step(model, slot_heads, batch_x, batch_y, batch_start_idx, c, exp, args, cli, pool_cache):
    scores, pool_valid_mask, pool_idx_global = compute_scores_pool_channel_first_grad(
        model, slot_heads, batch_x, exp, c, pool_cache, batch_start_idx, batch_x.device)
    memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
    query_future = batch_y[:, :, c]
    pooled_memory_c = gather_candidate_values(memory_c, pool_idx_global)
    with torch.no_grad():
        d_pool = pooled_future_mse(pooled_memory_c, offset_c, query_future)
        p_t = normalized_teacher_prob(d_pool, pool_valid_mask, cli.tau_t)  # NEVER -d_pool
        head_topk_idx = per_head_standalone_topk(scores, pool_valid_mask, k=cli.top_k)
        U = per_head_future_utility(head_topk_idx, d_pool)
        responsibility = responsibility_from_utility(U, tau_e=TAU_E)
    kl_vals, p_h = kl_per_head(p_t, scores, pool_valid_mask, cli.tau_s)
    loss_c = expert_weighted_loss(responsibility, kl_vals)
    return loss_c, scores, pool_valid_mask, pool_idx_global, pooled_memory_c, offset_c, query_future, d_pool


@torch.no_grad()
def epoch_val_diagnostics(model, slot_heads, exp, args, cli, channels, device, val_loader, pool_cache):
    win_count = torch.zeros(NUM_SLOTS)
    soft_loss_sum, rr_sum, mean_sum, n = 0.0, 0.0, 0.0, 0
    for batch_x, batch_y, batch_start_idx in val_loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        bsz = batch_x.size(0)
        for c in channels:
            loss_c, scores, pool_valid_mask, pool_idx_global, pooled_memory_c, offset_c, query_future, d_pool = \
                soft_step(model, slot_heads, batch_x, batch_y, batch_start_idx, c, exp, args, cli, pool_cache)
            soft_loss_sum += float(loss_c) * bsz

            mean_idx, _, _ = mean_mixture_topk_selection(scores, pool_valid_mask, cli.tau_s, k=cli.top_k)
            mean_ret = d_pool.gather(1, mean_idx).mean(dim=-1)
            mean_sum += float(mean_ret.sum())

            rr_idx = round_robin_topk_selection(scores, pool_valid_mask, k=cli.top_k)
            rr_ret = d_pool.gather(1, rr_idx).mean(dim=-1)
            rr_sum += float(rr_ret.sum())

            head_topk_idx = per_head_standalone_topk(scores, pool_valid_mask, k=cli.top_k)
            U = per_head_future_utility(head_topk_idx, d_pool)
            winner = U.argmin(dim=1)
            win_count += torch.nn.functional.one_hot(winner, NUM_SLOTS).float().sum(0).cpu()
        n += bsz
    denom = max(n * len(channels), 1)
    winner_fraction = win_count / denom
    return {
        'val_mean_retmse10': mean_sum / denom, 'val_soft_loss': soft_loss_sum / denom,
        'val_rr_retmse10': rr_sum / denom, 'winner_fraction': winner_fraction.tolist(),
        'max_winner_fraction': float(winner_fraction.max()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--patch_len', type=int, default=16)
    ap.add_argument('--top_k', type=int, default=TOP_K)
    ap.add_argument('--tau_t', type=float, default=0.1)
    ap.add_argument('--tau_s', type=float, default=0.1)
    ap.add_argument('--slot_std', type=float, default=1e-3)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--train_epochs', type=int, default=10)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--init_seed', type=int, default=0)
    ap.add_argument('--loader_seed', type=int, default=0)
    ap.add_argument('--candidate_mask', default='raft')
    ap.add_argument('--candidate_pool_size', type=int, default=100)
    ap.add_argument('--candidate_pool_cache', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--checkpoints', required=True)
    ap.add_argument('--limit_batches', type=int, default=0, help='SMOKE/DEBUG ONLY')
    ap.add_argument('--smoke_test', action='store_true')
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(cli.out_dir) / cli.cell / 'Soft'
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints) / cli.cell / 'Soft'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    exp, args, model = build_model(cli, device)
    channels = list(range(int(args.enc_in)))
    init_hash = state_hash(model)
    d_model = int(args.d_model)

    slot_heads = SlotHeads(d_model, n_slots=NUM_SLOTS, std=cli.slot_std).to(device)
    for p in model.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(slot_heads.parameters()), lr=cli.learning_rate)
    assert not any('router' in n.lower() or 'gate' in n.lower() for n, _ in slot_heads.named_parameters())

    expected_meta = {'seq_len': cli.seq_len, 'pred_len': cli.pred_len, 'candidate_mask': cli.candidate_mask,
                     'candidate_pool_size': cli.candidate_pool_size, 'candidate_pool_metric': 'delta_last_cosine'}
    cache_dir = Path(cli.candidate_pool_cache)
    pool_caches = {s: CandidatePoolCache(cache_dir / f'{s}.pt', expected_meta) for s in ('train', 'val', 'test')}

    train_gen = make_loader_generator(cli.loader_seed)
    _, train_loader = exp._get_data(flag='train', shuffle=True, generator=train_gen)
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)

    config = {'exp': 'TRACK-HARD-EXPERT-V5-P100-ALLH01', 'cell': cli.cell, 'arm': 'Soft', 'num_slots': NUM_SLOTS,
             'tau_t': cli.tau_t, 'tau_s': cli.tau_s, 'tau_e': TAU_E, 'top_k': cli.top_k,
             'candidate_pool_size': cli.candidate_pool_size, 'batch_size': cli.batch_size,
             'learning_rate': cli.learning_rate, 'train_epochs': cli.train_epochs, 'early_stopping': False,
             'init_seed': cli.init_seed, 'loader_seed': cli.loader_seed, 'init_hash': init_hash,
             'loss': 'L_soft = mean_q[ sum_h r_h(q) * KL_h(q) ], r detached, tau_E=1.0, pool-restricted (P100)',
             'inference_rule': 'Mean-Mixture Top-K (NOT Round-Robin)',
             'checkpoint_criterion': 'PRIMARY selected post-hoc via eval_v5_meanmix_checkpoint_selection_pool01.py '
                                     '(validation Mean-Mixture RetMSE@10); in-training val numbers are diagnostic.'}
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2))
    print(f'[soft_p100] cell={cli.cell} init_hash={init_hash[:16]}')

    if cli.smoke_test:
        model.train()
        for bi, (batch_x, batch_y, batch_start_idx) in enumerate(train_loader):
            if bi >= 2:
                break
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            optimizer.zero_grad()
            for c in channels:
                loss_c, scores, *_ = soft_step(model, slot_heads, batch_x, batch_y, batch_start_idx, c, exp, args,
                                               cli, pool_caches['train'])
                assert scores.shape[1] == NUM_SLOTS
                (loss_c / len(channels)).backward()
            assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters()), \
                '[ISSUE] encoder grad is zero'
            optimizer.step()
        vram = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
        print(f'[soft_p100] SMOKE PASS max_vram_mb={vram:.1f}')
        return

    step_rows, epoch_val_rows, batch_order_hashes = [], [], []
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
            optimizer.zero_grad()
            batch_loss = 0.0
            for c in channels:
                loss_c, *_ = soft_step(model, slot_heads, batch_x, batch_y, batch_start_idx, c, exp, args, cli,
                                       pool_caches['train'])
                (loss_c / len(channels)).backward()
                batch_loss += float(loss_c.detach()) / len(channels)
            optimizer.step()
            global_step += 1
            step_rows.append({'global_step': global_step, 'epoch': epoch, 'batch_in_epoch': bi,
                              'train_loss': batch_loss})
        batch_order_hashes.append({'epoch': epoch, 'batch_order_sha256': batch_order_sha256(starts)})

        model.eval()
        diag = epoch_val_diagnostics(model, slot_heads, exp, args, cli, channels, device, val_loader,
                                     pool_caches['val'])
        epoch_val_rows.append({'epoch': epoch, **{k: v for k, v in diag.items() if k != 'winner_fraction'}})
        payload = {'model_state_dict': model.state_dict(), 'slot_heads_state_dict': slot_heads.state_dict(),
                  'epoch': epoch, 'config': config}
        torch.save(payload, ckpt_dir / f'checkpoint_epoch{epoch}.pth')
        print(f'[soft_p100] epoch={epoch} val_mean_retmse10(diag)={diag["val_mean_retmse10"]:.6f} '
             f'val_soft_loss(diag)={diag["val_soft_loss"]:.6f} val_rr_retmse10(diag)={diag["val_rr_retmse10"]:.6f} '
             f'winner_fraction={[round(x,3) for x in diag["winner_fraction"]]}')
        # NO early-stop break -- fixed train_epochs always, per spec section 5.

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
    (out_dir / 'checkpoint_fingerprints.json').write_text(json.dumps(
        {'init_hash': init_hash, 'batch_order_hashes': batch_order_hashes}, indent=2))
    vram = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
    print(f'[soft_p100] done. {cli.cell} all {cli.train_epochs} epochs saved. '
         f'max_vram_mb={vram:.1f} wall_seconds={time.time()-t0:.1f}')


if __name__ == '__main__':
    main()
