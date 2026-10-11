#!/usr/bin/env python3
"""TRACK-JOINT-AGGREGATE-ROUTED-MQ01 (Phase 3) -- end-to-end joint
training of Encoder + SlotHeads + AggregateRouter, initialized from
Phase 2's BEST_BACKBONE (Hard), Soft routing throughout (train AND
inference), per phase2_decision.json. The individual Hard-argmin loss
is REMOVED entirely per spec section 7 (Phase 1 found Individual
Utility misaligned with Aggregate Utility); the main loss is the
Router-mixed KL against the SAME future-MSE teacher every other P100
trainer in this project uses:

    p_R(i|q,c) = sum_h pi_h(q,c) * p_h(i|q,c)          (Soft routing)
    L_route    = KL(p_T(.|q,c) || p_R(.|q,c))          -- MAIN loss

Optional auxiliary (spec section 5, alpha in {0, 0.1, 0.5}, no sweep
beyond these 3 values):

    r_h^agg(q,c) = softmax(-zscore_h(U_agg(h,q,c)) / tau_R)   -- DETACHED
    L_router     = KL(r_agg || pi)
    L_total      = L_route + alpha * L_router

`tau_R` is the cell-specific validation-selected value from Phase 2's
R2 tau sweep (`R2_best_tau_selection.json`) -- not re-swept here; Phase
3 only sweeps alpha, per spec.

Fixed 10 epochs, NO early stopping, every epoch checkpoint saved
(model + slot_heads + router jointly). Collapse diagnostics (router
entropy, effective/active heads, max/min routing weight, pairwise
head overlap, union size, per-head RetMSE, Aggregate winner accuracy,
Oracle Gap Recovered) computed every epoch on validation via a REAL
forward pass (the backbone updates every step, so Phase 2's
cached-features trick does not apply here). Post-hoc checkpoint
selection: min validation Soft-routed Aggregate MSE.
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
from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob
from scripts.train_j_shared_encoder_drift01 import build_model, state_hash
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads, kl_loss_from_prob
from scripts.train_margutil01 import memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k
from scripts.train_retriever_pool01 import CandidatePoolCache
from utils.aggregate_router01 import AggregateRouter, aggregate_soft_teacher, r2_kl_loss, router_entropy
from utils.candidate_pool import gather_candidate_values, pooled_future_mse
from utils.expert_head_metrics import head_pairwise_overlap, per_head_standalone_topk
from utils.full_candidate_bank import compute_scores_pool_channel_first_grad, encode_raw_channel_first

NUM_SLOTS = 5
TOP_K = 10


def per_head_aggregate_utility(scores_detached, pooled_memory_c, offset_c, query_future, d_pool, top_k=TOP_K):
    """Detached, no-grad helper: returns U_agg [B,5] and head_topk_idx
    [B,5,K] from already-detached scores."""
    pool_valid = torch.ones(scores_detached.size(0), scores_detached.size(-1), dtype=torch.bool,
                            device=scores_detached.device)
    head_topk_idx = per_head_standalone_topk(scores_detached, pool_valid, k=top_k)
    U_agg = torch.zeros(scores_detached.size(0), NUM_SLOTS, device=scores_detached.device)
    for h in range(NUM_SLOTS):
        idx_h = head_topk_idx[:, h, :]
        y_sel = pooled_memory_c.gather(1, idx_h.unsqueeze(-1).expand(-1, -1, pooled_memory_c.size(-1))) \
            + offset_c.view(-1, 1, 1)
        r_h = y_sel.mean(dim=1)
        U_agg[:, h] = ((r_h - query_future) ** 2).mean(dim=-1)
    return U_agg, head_topk_idx


def joint_step(model, slot_heads, router, batch_x, batch_y, batch_start_idx, c, exp, args, cli, pool_cache,
              tau_r, alpha):
    scores, pool_valid_mask, pool_idx_global = compute_scores_pool_channel_first_grad(
        model, slot_heads, batch_x, exp, c, pool_cache, batch_start_idx, batch_x.device)  # [B,5,M] grad ON
    memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
    query_future = batch_y[:, :, c]
    pooled_memory_c = gather_candidate_values(memory_c, pool_idx_global)
    d_pool = pooled_future_mse(pooled_memory_c, offset_c, query_future)  # pure data, no grad needed
    with torch.no_grad():
        p_t = normalized_teacher_prob(d_pool, pool_valid_mask, cli.tau_t)

    p_h = torch.softmax(scores / cli.tau_s, dim=-1)  # [B,5,M] grad ON
    z_qc = encode_raw_channel_first(model, batch_x, c)  # grad ON
    router_logits = router(z_qc)  # grad ON
    pi = torch.softmax(router_logits, dim=-1)  # [B,5]
    p_R = (pi.unsqueeze(-1) * p_h).sum(dim=1)  # [B,M] grad ON through both model and router

    L_route = kl_loss_from_prob(p_t, p_R, pool_valid_mask)

    L_router = torch.tensor(0.0, device=batch_x.device)
    if alpha > 0:
        with torch.no_grad():
            U_agg, _ = per_head_aggregate_utility(scores.detach(), pooled_memory_c, offset_c, query_future, d_pool)
            r_agg = aggregate_soft_teacher(U_agg, tau_r)
        L_router = r2_kl_loss(router_logits, r_agg)

    L_total = L_route + alpha * L_router
    return L_total, L_route, L_router, scores, p_h, router_logits, pooled_memory_c, offset_c, query_future, d_pool


@torch.no_grad()
def epoch_val_diagnostics(model, slot_heads, router, exp, args, cli, channels, device, val_loader, pool_cache,
                          fixed_head_per_channel, tau_r, top_k=TOP_K):
    agg_sum, rr_retmse_sum, overlap_sum, union_sum, n = 0.0, 0.0, 0.0, 0.0, 0
    winner_correct, winner_total = 0, 0
    entropy_sum, max_w_sum, min_w_sum = 0.0, 0.0, 0.0
    fixed_agg_sum, oracle_agg_sum = 0.0, 0.0
    head_retmse_sum = torch.zeros(NUM_SLOTS)
    for batch_x, batch_y, batch_start_idx in val_loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        bsz = batch_x.size(0)
        for c in channels:
            scores, pool_valid_mask, pool_idx_global = compute_scores_pool_channel_first_grad(
                model, slot_heads, batch_x, exp, c, pool_cache, batch_start_idx, device)
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            pooled_memory_c = gather_candidate_values(memory_c, pool_idx_global)
            d_pool = pooled_future_mse(pooled_memory_c, offset_c, query_future)
            p_h = torch.softmax(scores / cli.tau_s, dim=-1)
            z_qc = encode_raw_channel_first(model, batch_x, c)
            router_logits = router(z_qc)
            pi = torch.softmax(router_logits, dim=-1)
            p_R = (pi.unsqueeze(-1) * p_h).sum(dim=1)

            picks = p_R.topk(top_k, dim=-1).indices
            ind_mse_i = d_pool.gather(1, picks)
            y_sel = pooled_memory_c.gather(1, picks.unsqueeze(-1).expand(-1, -1, pooled_memory_c.size(-1))) \
                + offset_c.view(-1, 1, 1)
            agg_pred = y_sel.mean(dim=1)
            agg_mse = ((agg_pred - query_future) ** 2).mean(dim=-1)
            agg_sum += float(agg_mse.sum())
            rr_retmse_sum += float(ind_mse_i.mean(dim=-1).sum())

            U_agg, head_topk_idx = per_head_aggregate_utility(scores, pooled_memory_c, offset_c, query_future,
                                                               d_pool)
            overlaps, union_size = head_pairwise_overlap(head_topk_idx)
            overlap_sum += float(torch.stack(list(overlaps.values())).mean(0).sum())
            union_sum += float(union_size.sum())
            for h in range(NUM_SLOTS):
                idx_h = head_topk_idx[:, h, :]
                head_retmse_sum[h] += float(d_pool.gather(1, idx_h).mean(dim=-1).sum())

            winner_pred = router_logits.argmax(dim=-1)
            winner_true = U_agg.argmin(dim=-1)
            winner_correct += int((winner_pred == winner_true).sum())
            winner_total += bsz

            ent = router_entropy(router_logits)
            entropy_sum += float(ent.sum())
            pi_max = pi.max(dim=-1).values
            pi_min = pi.min(dim=-1).values
            max_w_sum += float(pi_max.sum())
            min_w_sum += float(pi_min.sum())

            fh = int(fixed_head_per_channel[c])
            fh_idx = p_h[:, fh, :].topk(top_k, dim=-1).indices
            y_sel_fh = pooled_memory_c.gather(1, fh_idx.unsqueeze(-1).expand(-1, -1, pooled_memory_c.size(-1))) \
                + offset_c.view(-1, 1, 1)
            fixed_agg_sum += float(((y_sel_fh.mean(dim=1) - query_future) ** 2).mean(dim=-1).sum())
            oracle_agg_sum += float(U_agg.min(dim=-1).values.sum())
        n += bsz
    denom = max(n * len(channels), 1)
    active_heads = int((head_retmse_sum > 0).sum())
    return {
        'val_agg_mse': agg_sum / denom, 'val_retmse10': rr_retmse_sum / denom,
        'val_pairwise_overlap': overlap_sum / denom, 'val_union_size': union_sum / denom,
        'val_winner_acc': winner_correct / max(winner_total, 1), 'val_router_entropy': entropy_sum / denom,
        'val_max_routing_weight': max_w_sum / denom, 'val_min_routing_weight': min_w_sum / denom,
        'val_per_head_retmse': (head_retmse_sum / denom).tolist(), 'active_head_count': active_heads,
        'val_fixed_head_agg_mse': fixed_agg_sum / denom, 'val_aggregate_oracle_agg_mse': oracle_agg_sum / denom,
        'val_oracle_gap_recovered_pct': (fixed_agg_sum - agg_sum) / max(fixed_agg_sum - oracle_agg_sum, 1e-12) * 100.0,
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
    ap.add_argument('--tau_s', type=float, required=True)
    ap.add_argument('--tau_r', type=float, required=True)
    ap.add_argument('--alpha', type=float, required=True, choices=(0.0, 0.1, 0.5))
    ap.add_argument('--init_checkpoint', required=True, help='Phase2 BEST_BACKBONE (Hard) checkpoint, used as init')
    ap.add_argument('--fixed_head_json', required=True, help='Phase1 oracle_headroom.json for fixed_heads_per_channel')
    ap.add_argument('--candidate_pool_cache', required=True)
    ap.add_argument('--candidate_pool_size', type=int, default=100)
    ap.add_argument('--candidate_mask', default='raft')
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--train_epochs', type=int, default=10)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--init_seed', type=int, default=0)
    ap.add_argument('--loader_seed', type=int, default=0)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--checkpoints', required=True)
    ap.add_argument('--limit_batches', type=int, default=0, help='SMOKE/DEBUG ONLY')
    ap.add_argument('--smoke_test', action='store_true')
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(cli.out_dir) / cli.cell / f'alpha{cli.alpha}'
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints) / cli.cell / f'alpha{cli.alpha}'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    exp, args, model = build_model(cli, device)
    channels = list(range(int(args.enc_in)))
    d_model = int(args.d_model)
    init_hash = state_hash(model)

    bl = torch.load(cli.init_checkpoint, map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    slot_heads = SlotHeads(d_model, n_slots=NUM_SLOTS).to(device)
    slot_heads.load_state_dict(bl['slot_heads_state_dict'])
    for p in model.parameters():
        p.requires_grad_(True)
    for p in slot_heads.parameters():
        p.requires_grad_(True)

    router = AggregateRouter(d_model).to(device)
    assert not any('router' in n.lower() and 'gate' in n.lower() for n, _ in router.named_parameters())

    params = list(model.parameters()) + list(slot_heads.parameters()) + list(router.parameters())
    optimizer = torch.optim.Adam(params, lr=cli.learning_rate)

    expected_meta = {'seq_len': cli.seq_len, 'pred_len': cli.pred_len, 'candidate_mask': cli.candidate_mask,
                     'candidate_pool_size': cli.candidate_pool_size, 'candidate_pool_metric': 'delta_last_cosine'}
    cache_dir = Path(cli.candidate_pool_cache)
    pool_caches = {s: CandidatePoolCache(cache_dir / f'{s}.pt', expected_meta) for s in ('train', 'val', 'test')}

    oh = json.load(open(cli.fixed_head_json))
    fixed_head_per_channel = torch.tensor(oh['per_channel']['fixed_heads_per_channel'], dtype=torch.long)

    train_gen = make_loader_generator(cli.loader_seed)
    _, train_loader = exp._get_data(flag='train', shuffle=True, generator=train_gen)
    _, val_loader = exp._get_data(flag='val', shuffle=False)

    config = {'exp': 'TRACK-JOINT-AGGREGATE-ROUTED-MQ01', 'cell': cli.cell, 'alpha': cli.alpha, 'tau_r': cli.tau_r,
             'tau_s': cli.tau_s, 'tau_t': cli.tau_t, 'init_checkpoint': cli.init_checkpoint,
             'init_hash': init_hash, 'train_epochs': cli.train_epochs, 'early_stopping': False,
             'loss': 'L_total = KL(p_T || p_R) + alpha * KL(r_agg || pi); r_agg detached; NO individual Hard-argmin loss',
             'routing_mode': 'Soft (train AND inference)', 'checkpoint_criterion': 'min validation Soft-routed Aggregate MSE'}
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2))
    print(f'[joint_router] cell={cli.cell} alpha={cli.alpha} tau_r={cli.tau_r} init_hash={init_hash[:16]}')

    if cli.smoke_test:
        model.train(); slot_heads.train(); router.train()
        for bi, (batch_x, batch_y, batch_start_idx) in enumerate(train_loader):
            if bi >= 2:
                break
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            optimizer.zero_grad()
            for c in channels:
                L_total, *_ = joint_step(model, slot_heads, router, batch_x, batch_y, batch_start_idx, c, exp,
                                         args, cli, pool_caches['train'], cli.tau_r, cli.alpha)
                (L_total / len(channels)).backward()
            assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters()), \
                '[ISSUE] encoder grad is zero'
            assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in router.parameters()), \
                '[ISSUE] router grad is zero'
            optimizer.step()
        vram = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
        print(f'[joint_router] SMOKE PASS max_vram_mb={vram:.1f}')
        return

    step_rows, epoch_rows, batch_order_hashes = [], [], []
    best = {'val_agg_mse': float('inf'), 'epoch': -1}
    t0 = time.time()
    global_step = 0
    for epoch in range(1, cli.train_epochs + 1):
        model.train(); slot_heads.train(); router.train()
        starts = []
        for bi, (batch_x, batch_y, batch_start_idx) in enumerate(train_loader):
            if cli.limit_batches and bi >= cli.limit_batches:
                break
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            starts.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx) else torch.as_tensor(batch_start_idx))
            optimizer.zero_grad()
            batch_loss, batch_route, batch_aux = 0.0, 0.0, 0.0
            for c in channels:
                L_total, L_route, L_router, *_ = joint_step(model, slot_heads, router, batch_x, batch_y,
                                                             batch_start_idx, c, exp, args, cli,
                                                             pool_caches['train'], cli.tau_r, cli.alpha)
                (L_total / len(channels)).backward()
                batch_loss += float(L_total.detach()) / len(channels)
                batch_route += float(L_route.detach()) / len(channels)
                batch_aux += float(L_router.detach()) / len(channels)
            optimizer.step()
            global_step += 1
            step_rows.append({'global_step': global_step, 'epoch': epoch, 'batch_in_epoch': bi,
                              'train_loss': batch_loss, 'train_L_route': batch_route, 'train_L_router': batch_aux})
        batch_order_hashes.append({'epoch': epoch, 'batch_order_sha256': batch_order_sha256(starts)})

        model.eval(); slot_heads.eval(); router.eval()
        diag = epoch_val_diagnostics(model, slot_heads, router, exp, args, cli, channels, device, val_loader,
                                     pool_caches['val'], fixed_head_per_channel, cli.tau_r)
        diag['epoch'] = epoch
        epoch_rows.append(diag)
        torch.save({'model_state_dict': model.state_dict(), 'slot_heads_state_dict': slot_heads.state_dict(),
                   'router_state_dict': router.state_dict(), 'epoch': epoch, 'config': config},
                  ckpt_dir / f'checkpoint_epoch{epoch}.pth')
        if diag['val_agg_mse'] < best['val_agg_mse']:
            best = {'val_agg_mse': diag['val_agg_mse'], 'epoch': epoch}
        print(f'[joint_router] cell={cli.cell} alpha={cli.alpha} epoch={epoch} val_agg_mse={diag["val_agg_mse"]:.6f} '
             f'val_winner_acc={diag["val_winner_acc"]:.3f} val_overlap={diag["val_pairwise_overlap"]:.3f} '
             f'val_union={diag["val_union_size"]:.2f} active_heads={diag["active_head_count"]} '
             f'val_entropy={diag["val_router_entropy"]:.3f} oracle_gap%={diag["val_oracle_gap_recovered_pct"]:.2f} '
             f'(best={best["epoch"]}:{best["val_agg_mse"]:.6f})')
        # NO early-stop break -- fixed train_epochs always.

    with open(out_dir / 'train_metrics.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(step_rows[0].keys()))
        w.writeheader()
        for r in step_rows:
            w.writerow(r)
    with open(out_dir / 'epoch_diagnostics.csv', 'w', newline='') as fh:
        fieldnames = sorted({k for r in epoch_rows for k in r if k != 'val_per_head_retmse'})
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in epoch_rows:
            w.writerow({k: v for k, v in r.items() if k != 'val_per_head_retmse'})
    (out_dir / 'checkpoint_selection.json').write_text(json.dumps({
        'selection_criterion': 'min validation Soft-routed Aggregate MSE (never test)',
        'selected_epoch': best['epoch'], 'selected_val_agg_mse': best['val_agg_mse'],
        'selected_checkpoint_path': str(ckpt_dir / f'checkpoint_epoch{best["epoch"]}.pth'),
    }, indent=2))
    (out_dir / 'checkpoint_fingerprints.json').write_text(json.dumps(
        {'init_hash': init_hash, 'batch_order_hashes': batch_order_hashes}, indent=2))
    vram = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
    print(f'[joint_router] done. cell={cli.cell} alpha={cli.alpha} best_epoch={best["epoch"]} '
         f'val_agg_mse={best["val_agg_mse"]:.6f} max_vram_mb={vram:.1f} wall_seconds={time.time()-t0:.1f}')


if __name__ == '__main__':
    main()
