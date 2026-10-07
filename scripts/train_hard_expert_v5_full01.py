#!/usr/bin/env python3
"""TRACK-HARD-EXPERT-V5-FULL01 -- controlled follow-up to
TRACK-EXPERT-V5-FULL01 (Soft Expert-V5). Changes EXACTLY ONE thing
relative to the Soft trainer: the loss no longer soft-weights every
head by responsibility; instead only the query-wise future-best head
(by standalone Top-10 mean future-MSE) receives a direct KL gradient.

    Soft: L_soft = mean_q[ sum_h r_h(q) * KL_h(q) ],  r_h = softmax(-z(U)/tau_E)
    Hard: h*(q)  = argmin_h U_h(q)  (detached)
          L_hard = mean_q[ KL_{h*(q)}(q) ]

Everything else is held byte-for-byte identical to the canonical
Full-Candidate V5 architecture and to Soft Expert's own training
control: num_slots=5, full candidate N (fresh re-encode every
optimizer step, candidate-side AND query-side gradient ON), SlotHeads
architecture/init (slot_std=1e-3), Future-MSE teacher (tau_t), student
softmax (tau_s), optimizer/LR/batch_size/seed. The ONLY two
INTENTIONAL deviations from Soft Expert's own script, both per
explicit spec: (1) A1 channel-first scoring
(`compute_scores_full_grad_channel_first`) is used instead of the
legacy channel-last path -- bit-exact equivalent, applied per this
project's standing policy that new experiments default to A1; Soft
Expert itself predates that policy and is left untouched. (2) NO early
stopping -- exactly `train_epochs` (10) epochs are always run, every
epoch's checkpoint is saved (including epoch 0, the pre-training
diagnostic baseline), and the best epoch is chosen post-hoc.

No Router, router CE, calendar feature, load-balancing/diversity/
overlap/entropy/usage regularizer, auxiliary aggregate loss, tau_E
sweep, P100, or R100 is added anywhere in this file (spec section 6).
tau_E does not even appear here -- Hard assignment has no responsibility
temperature, only a hard argmin.

Reused UNMODIFIED: everything `train_expert_v5_full01.py` itself reuses
(see that file's own docstring), plus `hard_expert_loss`
(`utils/expert_head_metrics.py`, new primitive, Soft's own
`expert_weighted_loss`/`responsibility_from_utility` untouched) and
`compute_scores_full_grad_channel_first`
(`utils/full_candidate_bank.py`).
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as TF

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
from scripts.train_t_pure_multislot01 import hard_eval_decomposition
from utils.expert_head_metrics import (
    evaluate_and_save_head_report, hard_expert_loss, kl_per_head, per_head_future_utility,
    per_head_standalone_topk, winner_margin_stats,
)
from utils.full_candidate_bank import compute_scores_full_grad_channel_first

TOP_K = 10
NUM_SLOTS = 5


def hard_step(model, slot_heads, batch_x, batch_y, batch_start_idx, c, cand_mask, exp, args, cli):
    """One channel's worth of the Hard-Expert-V5 forward+loss (no
    backward). Mirrors `train_expert_v5_full01.expert_step` exactly,
    swapping soft responsibility for a detached hard argmin winner."""
    scores = compute_scores_full_grad_channel_first(model, slot_heads, batch_x, exp.memory_x, c)  # [B,5,N]
    memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
    query_future = batch_y[:, :, c]
    with torch.no_grad():
        u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
        d_raw = -u  # MSE, lower=better (spec section 2 sign check)
        p_t = normalized_teacher_prob(d_raw, cand_mask, cli.tau_t)
        head_topk_idx = per_head_standalone_topk(scores, cand_mask, k=cli.top_k)  # standalone, NOT round-robin
        U = per_head_future_utility(head_topk_idx, d_raw)  # [B,5]
        winner_idx, u_best, u_second, margin = winner_margin_stats(U)  # argmin, fully detached (no_grad scope)
    kl_vals, p_h = kl_per_head(p_t, scores, cand_mask, cli.tau_s)  # [B,5], grad-attached via scores
    loss_c = hard_expert_loss(kl_vals, winner_idx)
    return loss_c, scores, p_t, d_raw, memory_c, offset_c, query_future, winner_idx, margin, U


@torch.no_grad()
def hard_epoch_val_diagnostics(model, slot_heads, exp, args, cli, channels, device, val_loader):
    """Lightweight per-epoch validation pass (spec sections 9-11): NOT
    the expensive full per-head report (no overlap/union/spearman/
    entropy here -- those are computed once, at the end, on the
    SELECTED checkpoint only, via the shared `evaluate_and_save_head_report`,
    for parity with Soft Expert). Returns winner fraction per head,
    margin-distribution stats, the PRIMARY checkpoint criterion
    (val_hard_loss), the legacy Round-Robin val retmse10 diagnostic, and
    a flat per-(query,channel) winner tensor in a FIXED iteration order
    (shuffle=False val_loader, channels inner loop) so epoch-to-epoch
    winner identity is directly comparable for assignment-stability /
    transition-matrix tracking."""
    win_count = torch.zeros(NUM_SLOTS)
    margins, winner_flat = [], []
    hard_kl_sum, rr_sum, n = 0.0, 0.0, 0
    for batch_x, batch_y, batch_start_idx in val_loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        bsz = batch_x.size(0)
        for c in channels:
            scores = compute_scores_full_grad_channel_first(model, slot_heads, batch_x, exp.memory_x, c)
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
            d_raw = -u
            p_t = normalized_teacher_prob(d_raw, cand_mask, cli.tau_t)
            head_topk_idx = per_head_standalone_topk(scores, cand_mask, k=cli.top_k)
            U = per_head_future_utility(head_topk_idx, d_raw)
            winner, u_best, u_second, margin = winner_margin_stats(U)
            kl_vals, _ = kl_per_head(p_t, scores, cand_mask, cli.tau_s)
            hard_kl = kl_vals.gather(1, winner.unsqueeze(1)).squeeze(1)

            oracle_idx = stable_topk_indices(d_raw.masked_fill(~cand_mask, float('inf')), cli.top_k, largest=False)
            res_rr = hard_eval_decomposition(scores, cand_mask, memory_c, offset_c, query_future, d_raw,
                                             oracle_idx, cli.top_k)

            win_count += TF.one_hot(winner, NUM_SLOTS).float().sum(0).cpu()
            margins.append(margin.cpu())
            hard_kl_sum += float(hard_kl.sum())
            rr_sum += float(res_rr['model_ind_mse'].sum())
            winner_flat.append(winner.cpu())
        n += bsz
    denom = max(n * len(channels), 1)
    margin_cat = torch.cat(margins)
    q = torch.quantile(margin_cat, torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90]))
    return {
        'winner_fraction': (win_count / denom).tolist(),
        'max_winner_fraction': float((win_count / denom).max()),
        'val_hard_loss': hard_kl_sum / denom,
        'val_rr_retmse10': rr_sum / denom,
        'margin_mean': float(margin_cat.mean()), 'margin_p10': float(q[0]), 'margin_p25': float(q[1]),
        'margin_median': float(q[2]), 'margin_p75': float(q[3]), 'margin_p90': float(q[4]),
        'near_tie_fraction': float((margin_cat.abs() < 1e-4).float().mean()),
        'exact_tie_fraction': float((margin_cat == 0).float().mean()),
        'winner_flat': torch.cat(winner_flat),
    }


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
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--init_seed', type=int, default=0)
    ap.add_argument('--loader_seed', type=int, default=0)
    ap.add_argument('--out_dir', default='results/TRACK-HARD-EXPERT-V5-FULL01')
    ap.add_argument('--checkpoints', default='checkpoints/track_hard_expert_v5_full01')
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
    assert n_candidates == exp.memory_x.shape[0], '[ISSUE][ABORT] Full Candidate assertion failed'

    slot_heads = SlotHeads(d_model, n_slots=NUM_SLOTS, std=cli.slot_std).to(device)
    slot_heads_init_hash = state_hash(slot_heads)
    for p in model.parameters():
        p.requires_grad_(True)
    params = list(model.parameters()) + list(slot_heads.parameters())
    optimizer = torch.optim.Adam(params, lr=cli.learning_rate)
    assert not any('router' in n.lower() or 'gate' in n.lower() for n, _ in model.named_parameters()), \
        '[ISSUE][ABORT] no Router/gate parameters allowed in this track'
    assert not any('router' in n.lower() or 'gate' in n.lower() for n, _ in slot_heads.named_parameters()), \
        '[ISSUE][ABORT] no Router/gate parameters allowed in this track'

    train_gen = make_loader_generator(cli.loader_seed)
    _, train_loader = exp._get_data(flag='train', shuffle=True, generator=train_gen)
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)

    config = {'cell': cli.cell, 'arm': 'Hard-Expert-V5-Full', 'num_slots': NUM_SLOTS, 'slot_std': cli.slot_std,
             'tau_t': cli.tau_t, 'tau_s': cli.tau_s, 'top_k': cli.top_k,
             'batch_size': cli.batch_size, 'learning_rate': cli.learning_rate, 'train_epochs': cli.train_epochs,
             'early_stopping': False, 'init_seed': cli.init_seed, 'loader_seed': cli.loader_seed,
             'n_candidates': n_candidates, 'init_hash': init_hash, 'slot_heads_init_hash': slot_heads_init_hash,
             'candidate_scoring': 'A1 channel-first (compute_scores_full_grad_channel_first), '
                                  'bit-exact equivalent to legacy -- new-experiment standing policy',
             'loss': 'L_hard = mean_q[ KL(p_T(q) || p_{argmin_h U_h(q)}(q)) ], winner detached, no Router',
             'checkpoint_criterion': 'PRIMARY: min validation Hard objective loss (val_hard_loss), '
                                     'epochs 1-10 only (epoch0 is diagnostic-only). Legacy Round-Robin '
                                     'val retmse10 recorded as a DIAGNOSTIC ONLY, never used for selection.'}
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2))
    print(f'[hard_expert_v5] cell={cli.cell} N={n_candidates} init_hash={init_hash[:16]} '
         f'slot_heads_init_hash={slot_heads_init_hash[:16]}')

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
                loss_c, scores, *_ = hard_step(model, slot_heads, batch_x, batch_y, batch_start_idx, c,
                                               cand_mask, exp, args, cli)
                assert scores.shape[1] == NUM_SLOTS
                (loss_c / len(channels)).backward()
            assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters()), \
                '[ISSUE] query/candidate-side encoder grad is zero'
            optimizer.step()
        vram = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
        print(f'[hard_expert_v5] SMOKE PASS max_vram_mb={vram:.1f}')
        return

    def save_epoch_checkpoint(epoch, val_hard_loss):
        payload = {'model_state_dict': model.state_dict(), 'slot_heads_state_dict': slot_heads.state_dict(),
                  'epoch': epoch, 'val_hard_loss': val_hard_loss, 'config': config}
        torch.save(payload, ckpt_dir / f'checkpoint_epoch{epoch}.pth')
        return payload

    epoch_diag_rows, winner_flat_by_epoch = [], {}

    # ---- epoch 0: pre-training diagnostic baseline (spec section 8) ----
    model.eval()
    diag0 = hard_epoch_val_diagnostics(model, slot_heads, exp, args, cli, channels, device, val_loader)
    winner_flat_by_epoch[0] = diag0.pop('winner_flat')
    epoch_diag_rows.append({'epoch': 0, **diag0})
    save_epoch_checkpoint(0, diag0['val_hard_loss'])
    print(f'[hard_expert_v5] epoch=0 (pre-training, diagnostic only) val_hard_loss={diag0["val_hard_loss"]:.6f} '
         f'winner_fraction={[round(x,3) for x in diag0["winner_fraction"]]}')

    step_rows, batch_order_hashes = [], []
    best = {'val_hard_loss': float('inf'), 'epoch': -1}
    best_rr = {'val_rr_retmse10': float('inf'), 'epoch': -1}
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
                loss_c, *_ = hard_step(model, slot_heads, batch_x, batch_y, batch_start_idx, c, cand_mask,
                                       exp, args, cli)
                (loss_c / len(channels)).backward()
                batch_loss += float(loss_c.detach()) / len(channels)
            optimizer.step()
            global_step += 1
            step_rows.append({'global_step': global_step, 'epoch': epoch, 'batch_in_epoch': bi,
                              'train_loss': batch_loss})
        batch_order_hashes.append({'epoch': epoch, 'batch_order_sha256': batch_order_sha256(starts)})

        model.eval()
        diag = hard_epoch_val_diagnostics(model, slot_heads, exp, args, cli, channels, device, val_loader)
        winner_flat_by_epoch[epoch] = diag.pop('winner_flat')
        epoch_diag_rows.append({'epoch': epoch, **diag})
        save_epoch_checkpoint(epoch, diag['val_hard_loss'])
        if diag['val_hard_loss'] < best['val_hard_loss']:
            best = {'val_hard_loss': diag['val_hard_loss'], 'epoch': epoch}
        if diag['val_rr_retmse10'] < best_rr['val_rr_retmse10']:
            best_rr = {'val_rr_retmse10': diag['val_rr_retmse10'], 'epoch': epoch}
        print(f'[hard_expert_v5] epoch={epoch} val_hard_loss={diag["val_hard_loss"]:.6f} '
             f'val_rr_retmse10(diag)={diag["val_rr_retmse10"]:.6f} '
             f'winner%={[round(x*100,1) for x in diag["winner_fraction"]]} '
             f'max_win={diag["max_winner_fraction"]:.3f} margin_median={diag["margin_median"]:.6f} '
             f'(hard-best={best["epoch"]}:{best["val_hard_loss"]:.6f})')
        # NO early-stop break -- fixed train_epochs always, per spec section 8.

    # ---- assignment stability / transition matrix across epoch0..N (spec section 11) ----
    epochs_present = sorted(winner_flat_by_epoch)
    stability_rows = []
    transition_counts = torch.zeros(NUM_SLOTS, NUM_SLOTS)
    for i in range(1, len(epochs_present)):
        prev_e, cur_e = epochs_present[i - 1], epochs_present[i]
        prev_w, cur_w = winner_flat_by_epoch[prev_e], winner_flat_by_epoch[cur_e]
        stability = float((prev_w == cur_w).float().mean())
        stability_rows.append({'from_epoch': prev_e, 'to_epoch': cur_e, 'stability': stability})
        idx = prev_w * NUM_SLOTS + cur_w
        counts = torch.bincount(idx, minlength=NUM_SLOTS * NUM_SLOTS).float().reshape(NUM_SLOTS, NUM_SLOTS)
        transition_counts += counts
    transition_matrix = (transition_counts / transition_counts.sum(dim=1, keepdim=True).clamp_min(1)).tolist()

    # ---- PRIMARY checkpoint selection: min val_hard_loss, epochs 1-10 ONLY, test split never touched ----
    bl = torch.load(ckpt_dir / f'checkpoint_epoch{best["epoch"]}.pth', map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    slot_heads.load_state_dict(bl['slot_heads_state_dict'])
    model.eval()

    # shared post-training evaluation -- IDENTICAL code path to Soft
    # Expert and the Current-V5 standalone evaluator (byte-for-byte
    # comparable measurement, spec sections 10/15).
    final_test_metrics = evaluate_and_save_head_report(
        model, slot_heads, exp, args, cli, channels, device, train_loader, val_loader, test_loader,
        out_dir, best['epoch'], num_slots=NUM_SLOTS, tau_e=1.0)

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
    write_csv(out_dir / 'epoch_diagnostics.csv', epoch_diag_rows)
    write_csv(out_dir / 'batch_order_hashes.csv', batch_order_hashes)
    write_csv(out_dir / 'assignment_stability.csv', stability_rows)
    (out_dir / 'winner_transition_matrix.json').write_text(json.dumps({
        'matrix_rows_are_from_head_cols_are_to_head': transition_matrix,
        'note': 'row i, col j = P(winner_t = j | winner_{t-1} = i), aggregated over epoch0->1,...,9->10',
    }, indent=2))
    (out_dir / 'checkpoint_selection.json').write_text(json.dumps({
        'objective_aligned_best_epoch': best['epoch'], 'objective_aligned_best_val_hard_loss': best['val_hard_loss'],
        'legacy_rr_best_epoch': best_rr['epoch'], 'legacy_rr_best_val_rr_retmse10': best_rr['val_rr_retmse10'],
        'epochs_agree': best['epoch'] == best_rr['epoch'],
        'note': 'Round-Robin is NOT the Hard Expert training objective -- diagnostic only (spec section 13).',
    }, indent=2))
    (out_dir / 'checkpoint_fingerprints.json').write_text(json.dumps(
        {'init_hash': init_hash, 'slot_heads_init_hash': slot_heads_init_hash, 'best_epoch': best['epoch'],
         'final_model_hash': state_hash(model), 'batch_order_hashes': batch_order_hashes}, indent=2))
    vram = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
    print(f'[hard_expert_v5] done. objective_best_epoch={best["epoch"]} legacy_rr_best_epoch={best_rr["epoch"]} '
         f'test_retMSE@10(RR,diag)={final_test_metrics["retmse10"]:.6f} max_vram_mb={vram:.1f} '
         f'wall_seconds={time.time()-t0:.1f}')


if __name__ == '__main__':
    main()
