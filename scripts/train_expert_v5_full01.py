#!/usr/bin/env python3
"""TRACK-EXPERT-V5-FULL01 -- Expert-V5: responsibility-weighted per-head
KL loss on top of the EXISTING Full-Candidate V5 (num_slots=5)
architecture, with NO other change.

Canonical Current-V5 (`train_t_pure_multislot01.py --num_slots 5`,
reused UNMODIFIED by `run_v_one_setting01.sh`) trains:

    p_bar(q) = mean_h softmax(s_h(q,:)/tau_S)
    L_current = KL(p_T || p_bar)

-- individual heads never need to match the teacher on their own, only
their AVERAGE does. This script keeps EVERY other part of that
training loop byte-for-byte (full candidate N, fresh re-encode every
optimizer step via `compute_scores_full_grad`, candidate-side gradient
ON, SlotHeads architecture/init, teacher, tau_T, tau_S, top_k=10,
optimizer/lr/epochs/patience/seed, candidate mask, checkpoint
criterion = min val round-robin retMSE@10) and changes ONLY the loss:

    T_h(q)      = standalone Top-10 of head h alone (NOT round-robin)
    U_h(q)      = mean future-MSE over T_h(q)
    r(q)        = softmax(-zscore_h(U)/tau_E), tau_E=1.0, DETACHED
    KL_h(q)     = KL(p_T(q) || softmax(s_h(q,:)/tau_S))   -- per head, [B,5]
    L_expert    = mean_q [ sum_h r_h(q) * KL_h(q) ]        -- the ONLY loss term

No R100/EMA/stale-bank/Top-M/router/diversity/load-balance/aggregate
term is added anywhere in this file (spec section 25).

Reused UNMODIFIED: `build_model`/`state_hash`
(`train_j_shared_encoder_drift01`), `individual_utility_memsafe`
(`train_factorial_e2e01`), `normalized_teacher_prob`
(`train_horizon_retrieval_expert01`), `SlotHeads`
(`train_k_multislot_predictive_retrieval01`), `memory_value`
(`train_margutil01`), `stable_topk_indices` (`RelationStage1`),
`compute_scores_full_grad`/`round_robin_topk_selection`/
`hard_eval_decomposition`/`spearman_batch`/`slot_mechanism_diagnostics`/
`naive_round_robin_picks` (`train_t_pure_multislot01`), every primitive
in `utils/expert_head_metrics.py`.
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
from scripts.train_factorial_e2e01 import individual_utility_memsafe
from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob
from scripts.train_j_shared_encoder_drift01 import build_model, state_hash
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_margutil01 import memory_value
from scripts.train_t_pure_multislot01 import (
    compute_scores_full_grad, hard_eval_decomposition, round_robin_topk_selection,
    slot_mechanism_diagnostics, spearman_batch,
)
from utils.expert_head_metrics import (
    evaluate_and_save_head_report, expert_weighted_loss, full_head_diagnostics, head_pairwise_overlap,
    kl_per_head, per_head_decomposition, per_head_entropy_and_spread, per_head_future_utility,
    per_head_standalone_topk, responsibility_from_utility, winner_margin_stats,
)

TOP_K = 10
NUM_SLOTS = 5
TAU_E = 1.0  # fixed per spec section 5 -- never swept


def expert_step(model, slot_heads, batch_x, batch_y, batch_start_idx, c, cand_mask, exp, args, cli):
    """One channel's worth of the Expert-V5 forward+loss (no backward).
    Returns (loss_c, scores, p_t, d_raw) -- scores/p_t/d_raw are reused
    by the caller for diagnostics without recomputation."""
    scores = compute_scores_full_grad(model, slot_heads, batch_x, exp.memory_x, c)  # [B,5,N], full grad, fresh
    memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
    query_future = batch_y[:, :, c]
    with torch.no_grad():
        u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
        d_raw = -u
        p_t = normalized_teacher_prob(d_raw, cand_mask, cli.tau_t)
        head_topk_idx = per_head_standalone_topk(scores, cand_mask, k=cli.top_k)
        U = per_head_future_utility(head_topk_idx, d_raw)
        responsibility = responsibility_from_utility(U, tau_e=TAU_E)
    kl_vals, p_h = kl_per_head(p_t, scores, cand_mask, cli.tau_s)
    loss_c = expert_weighted_loss(responsibility, kl_vals)
    return loss_c, scores, p_t, d_raw, memory_c, offset_c, query_future


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
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
    ap.add_argument('--out_dir', default='results/TRACK-EXPERT-V5-FULL01')
    ap.add_argument('--checkpoints', default='checkpoints/track_expert_v5_full01')
    ap.add_argument('--limit_batches', type=int, default=0, help='SMOKE ONLY')
    ap.add_argument('--smoke_test', action='store_true')
    ap.add_argument('--skip_init_hash_check', action='store_true')
    ap.add_argument('--expected_init_hash', default=None)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(cli.out_dir) / cli.cell
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints) / cli.cell
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    exp, args, model = build_model(cli, device)
    channels = list(range(int(args.enc_in)))
    init_hash = state_hash(model)
    if cli.expected_init_hash and not cli.skip_init_hash_check:
        assert init_hash == cli.expected_init_hash, \
            f'[ISSUE][ABORT] init hash mismatch: {init_hash} != {cli.expected_init_hash}'
    d_model = int(args.d_model)
    n_candidates = int(exp.memory_x.size(0))

    slot_heads = SlotHeads(d_model, n_slots=NUM_SLOTS, std=cli.slot_std).to(device)
    for p in model.parameters():
        p.requires_grad_(True)
    params = list(model.parameters()) + list(slot_heads.parameters())
    optimizer = torch.optim.Adam(params, lr=cli.learning_rate)

    train_gen = make_loader_generator(cli.loader_seed)
    _, train_loader = exp._get_data(flag='train', shuffle=True, generator=train_gen)
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)

    config = {'cell': cli.cell, 'arm': 'Expert-V5-Full', 'num_slots': NUM_SLOTS, 'slot_std': cli.slot_std,
             'tau_t': cli.tau_t, 'tau_s': cli.tau_s, 'tau_e': TAU_E, 'top_k': cli.top_k,
             'batch_size': cli.batch_size, 'learning_rate': cli.learning_rate, 'train_epochs': cli.train_epochs,
             'patience': cli.patience, 'init_seed': cli.init_seed, 'loader_seed': cli.loader_seed,
             'n_candidates': n_candidates,
             'loss': 'L_expert = mean_q[ sum_h r_h(q) * KL_h(q) ], r detached, tau_E=1.0 fixed',
             'checkpoint_criterion': 'min val round-robin retmse10 (IDENTICAL to canonical Current-V5)'}
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2))
    print(f'[expert_v5] cell={cli.cell} N={n_candidates} init_hash={init_hash[:16]}')

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
                loss_c, scores, *_ = expert_step(model, slot_heads, batch_x, batch_y, batch_start_idx, c,
                                                  cand_mask, exp, args, cli)
                assert scores.shape[1] == NUM_SLOTS
                (loss_c / len(channels)).backward()
            assert all(p.grad is not None for p in slot_heads.parameters())
            assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters()), \
                '[ISSUE] query/candidate-side encoder grad is zero'
            optimizer.step()
        vram = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
        print(f'[expert_v5] SMOKE PASS max_vram_mb={vram:.1f}')
        return

    step_rows, epoch_val_rows, batch_order_hashes, per_head_epoch_rows = [], [], [], []
    best = {'val_retmse10': float('inf'), 'epoch': -1}
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
            batch_loss = 0.0
            for c in channels:
                loss_c, *_ = expert_step(model, slot_heads, batch_x, batch_y, batch_start_idx, c, cand_mask,
                                         exp, args, cli)
                (loss_c / len(channels)).backward()
                batch_loss += float(loss_c.detach()) / len(channels)
            optimizer.step()
            global_step += 1
            step_rows.append({'global_step': global_step, 'epoch': epoch, 'batch_in_epoch': bi,
                              'train_loss': batch_loss})
        batch_order_hashes.append({'epoch': epoch, 'batch_order_sha256': batch_order_sha256(starts)})

        # ---- end-of-epoch validation: PRIMARY round-robin metric (checkpoint
        # selection, section 18) + per-head trajectory row (section 10) ----
        model.eval()
        rr_sums, n = {}, 0
        head_accum = {name: [] for name in ('retmse10', 'kl', 'win_fraction_count', 'responsibility')}
        overlap_accum, union_accum = [], []
        with torch.no_grad():
            for batch_x, batch_y, batch_start_idx in val_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                cand_mask, _ = exp._candidate_mask(batch_start_idx)
                bsz = batch_x.size(0)
                per_ch_rr = {}
                for c in channels:
                    scores = compute_scores_full_grad(model, slot_heads, batch_x, exp.memory_x, c)
                    memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
                    query_future = batch_y[:, :, c]
                    u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
                    d_raw = -u
                    p_t = normalized_teacher_prob(d_raw, cand_mask, cli.tau_t)
                    oracle_idx = stable_topk_indices(d_raw.masked_fill(~cand_mask, float('inf')), cli.top_k,
                                                      largest=False)
                    res_rr = hard_eval_decomposition(scores, cand_mask, memory_c, offset_c, query_future, d_raw,
                                                     oracle_idx, cli.top_k)
                    per_ch_rr.setdefault('retmse10', []).append(res_rr['model_ind_mse'].cpu())
                    per_ch_rr.setdefault('agg_mse10', []).append(res_rr['agg_mse'].cpu())

                    diag = full_head_diagnostics(scores, cand_mask, memory_c, offset_c, query_future, d_raw, p_t,
                                                 cli.tau_s, cli.top_k)
                    head_accum['retmse10'].append(diag['decomp']['retmse10'].cpu())
                    head_accum['kl'].append(diag['kl'].cpu())
                    head_accum['responsibility'].append(diag['responsibility'].cpu())
                    winner_onehot = torch.nn.functional.one_hot(diag['winner'], num_classes=NUM_SLOTS).float()
                    head_accum['win_fraction_count'].append(winner_onehot.cpu())
                    overlap_accum.append(torch.stack([v for v in diag['overlaps'].values()]).mean(0).cpu())
                    union_accum.append(diag['union_size'].cpu())
                for k_, vals in per_ch_rr.items():
                    rr_sums[k_] = rr_sums.get(k_, 0.0) + torch.cat(vals).sum().item()
                n += bsz
        val_retmse10 = rr_sums['retmse10'] / max(n * len(channels), 1)
        val_agg10 = rr_sums['agg_mse10'] / max(n * len(channels), 1)
        epoch_val_rows.append({'epoch': epoch, 'retmse10': val_retmse10, 'agg_mse10': val_agg10})

        head_retmse = torch.cat(head_accum['retmse10']).mean(dim=0)  # [5]
        head_kl = torch.cat(head_accum['kl']).mean(dim=0)
        head_resp = torch.cat(head_accum['responsibility']).mean(dim=0)
        head_win_frac = torch.cat(head_accum['win_fraction_count']).mean(dim=0)
        pairwise_overlap_mean = torch.cat(overlap_accum).mean().item()
        union_mean = torch.cat(union_accum).mean().item()
        row = {'epoch': epoch}
        for h in range(NUM_SLOTS):
            row[f'H{h+1}_retmse10'] = float(head_retmse[h])
            row[f'H{h+1}_kl'] = float(head_kl[h])
            row[f'H{h+1}_win_fraction'] = float(head_win_frac[h])
            row[f'H{h+1}_mean_responsibility'] = float(head_resp[h])
        row['pairwise_top10_overlap'] = pairwise_overlap_mean
        row['union_size'] = union_mean
        per_head_epoch_rows.append(row)

        payload = {'model_state_dict': model.state_dict(), 'slot_heads_state_dict': slot_heads.state_dict(),
                  'epoch': epoch, 'val_retmse10': val_retmse10, 'config': config}
        if val_retmse10 < best['val_retmse10']:
            best = {'val_retmse10': val_retmse10, 'epoch': epoch}
            torch.save(payload, ckpt_dir / 'checkpoint.pth')
        print(f'[expert_v5] epoch={epoch} val_retmse10(RR)={val_retmse10:.6f} val_agg={val_agg10:.6f} '
             f'head_retmse={[round(float(x),4) for x in head_retmse]} '
             f'head_win%={[round(float(x)*100,1) for x in head_win_frac]} '
             f'(best={best["epoch"]}:{best["val_retmse10"]:.6f})')
        if epoch - best['epoch'] >= cli.patience:
            print(f'[expert_v5] early stop at epoch {epoch} (best={best["epoch"]})')
            break

    bl = torch.load(ckpt_dir / 'checkpoint.pth', map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    slot_heads.load_state_dict(bl['slot_heads_state_dict'])
    model.eval()

    # ---- shared post-training evaluation (utils.expert_head_metrics) --
    # identical code path used later to evaluate the EXISTING canonical
    # Current-V5 checkpoint, so the two are compared byte-for-byte.
    final_test_metrics = evaluate_and_save_head_report(
        model, slot_heads, exp, args, cli, channels, device, train_loader, val_loader, test_loader,
        out_dir, best['epoch'], num_slots=NUM_SLOTS, tau_e=TAU_E)

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
    write_csv(out_dir / 'per_head_epoch_metrics.csv', per_head_epoch_rows)
    write_csv(out_dir / 'batch_order_hashes.csv', batch_order_hashes)
    (out_dir / 'checkpoint_fingerprints.json').write_text(json.dumps(
        {'init_hash': init_hash, 'best_epoch': best['epoch'], 'final_model_hash': state_hash(model),
         'batch_order_hashes': batch_order_hashes}, indent=2))
    vram = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
    print(f'[expert_v5] done. best_epoch={best["epoch"]} test_retMSE@10(RR)={final_test_metrics["retmse10"]:.6f} '
         f'test_agg_mse10={final_test_metrics["agg_mse10"]:.6f} max_vram_mb={vram:.1f} wall_seconds={time.time()-t0:.1f}')




if __name__ == '__main__':
    main()
