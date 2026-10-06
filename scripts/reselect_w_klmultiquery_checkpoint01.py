#!/usr/bin/env python3
"""TRACK-W-CHECKPOINT-CRITERION-CORRECTION01 -- Correction B, CASE A
re-selection for Original KL (S=0) / Multi-Query (S>=1). For a (cell, S)
whose FULL intended epoch trajectory already exists on disk (no early
-stop truncation under the OLD val-retmse10 criterion), this does NOT
retrain anything: it reloads EVERY saved `checkpoint_epoch{N}.pth`,
recomputes validation KL(p_T||mean_m softmax(s_m/tau_s)) via a single
forward-only pass (no backprop), picks the argmin epoch, and writes a
corrected checkpoint + report. Mirrors
`reselect_w_hostfree_checkpoint01.py`'s own CASE-A pattern exactly.

S=0 uses `compute_scores_true_original_kl` (`build_t2_true_original_kl_cache01`,
zero projection params); S>=1 uses `compute_scores_full_grad` +
`SlotHeads` (`train_t_pure_multislot01`), exactly as each S's own
original training script used.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_t2_true_original_kl_cache01 import compute_scores_true_original_kl
from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob
from scripts.train_j_shared_encoder_drift01 import build_model
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads, kl_loss_from_prob
from scripts.train_margutil01 import memory_value
from scripts.train_t_pure_multislot01 import compute_scores_full_grad
from scripts.train_factorial_e2e01 import individual_utility_memsafe


def state_hash_model(model):
    h = hashlib.sha256()
    for k in sorted(model.state_dict()):
        h.update(k.encode())
        h.update(model.state_dict()[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def compute_val_kl(exp, args, model, slot_heads, loader, channels, device, tau_t, tau_s, chunk_size):
    total_kl, n = 0.0, 0
    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        bsz = batch_x.size(0)
        for c in channels:
            if slot_heads is None:
                scores = compute_scores_true_original_kl(model, batch_x, exp.memory_x, c)
            else:
                scores = compute_scores_full_grad(model, slot_heads, batch_x, exp.memory_x, c)
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            u = individual_utility_memsafe(memory_c, offset_c, query_future, chunk_size)
            p_t = normalized_teacher_prob(-u, cand_mask, tau_t)
            s_masked = scores.masked_fill(~cand_mask.unsqueeze(1), float('-inf'))
            p_bar = torch.softmax(s_masked / tau_s, dim=-1).mean(dim=1)
            kl = kl_loss_from_prob(p_t, p_bar, cand_mask)
            total_kl += float(kl) * bsz
        n += bsz * len(channels)
    return total_kl / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True)
    ap.add_argument('--num_slots', type=int, required=True)
    ap.add_argument('--old_ckpt_dir', required=True, help='dir containing checkpoint_epoch{N}.pth')
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--patch_len', type=int, default=16)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--tau_t', type=float, default=0.1)
    ap.add_argument('--tau_s', type=float, default=0.1)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--init_seed', type=int, default=0)
    ap.add_argument('--out_results_root', default='results/TRACK-W-CHECKPOINT-CRITERION-CORRECTION01/klmq_stage1')
    ap.add_argument('--out_ckpt_root', default='checkpoints/track_w_checkpoint_criterion_correction01/klmq_stage1')
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    arm = f'S{cli.num_slots}'
    old_dir = Path(cli.old_ckpt_dir)
    epoch_files = sorted(old_dir.glob('checkpoint_epoch*.pth'),
                        key=lambda p: int(re.search(r'epoch(\d+)', p.name).group(1)))
    assert epoch_files, f'[ISSUE][ABORT] no checkpoint_epoch*.pth found in {old_dir}'
    print(f'[reselect_klmq] {cli.cell}/{arm}: {len(epoch_files)} epoch checkpoints found in {old_dir}')

    exp, args, model = build_model(cli, device)
    channels = list(range(int(args.enc_in)))
    d_model = int(args.d_model)
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)

    results = []
    for ef in epoch_files:
        ck = torch.load(ef, map_location=device)
        model.load_state_dict(ck['model_state_dict'])
        model.eval()
        slot_heads = None
        if cli.num_slots >= 1 and 'slot_heads_state_dict' in ck and ck['slot_heads_state_dict'] is not None:
            slot_heads = SlotHeads(d_model, n_slots=cli.num_slots).to(device)
            slot_heads.load_state_dict(ck['slot_heads_state_dict'])
            slot_heads.eval()
        val_kl = compute_val_kl(exp, args, model, slot_heads, val_loader, channels, device,
                                cli.tau_t, cli.tau_s, cli.chunk_size)
        epoch_n = int(ck['epoch'])
        print(f'[reselect_klmq] {cli.cell}/{arm}: epoch={epoch_n} val_kl={val_kl:.6f} '
             f'(old_val_metric={ck.get("val_metric", ck.get("val_retmse10"))})')
        results.append((epoch_n, val_kl, ef))

    new_epoch, new_val_kl, new_ef = min(results, key=lambda t: t[1])
    print(f'[reselect_klmq] {cli.cell}/{arm}: NEW best_epoch={new_epoch} (val_kl={new_val_kl:.6f})')

    # independent re-verification: reload the chosen epoch fresh and recompute once more
    ck = torch.load(new_ef, map_location=device)
    model.load_state_dict(ck['model_state_dict'])
    model.eval()
    slot_heads = None
    if 'slot_heads_state_dict' in ck and ck['slot_heads_state_dict'] is not None:
        slot_heads = SlotHeads(d_model, n_slots=cli.num_slots).to(device)
        slot_heads.load_state_dict(ck['slot_heads_state_dict'])
        slot_heads.eval()
    recomputed = compute_val_kl(exp, args, model, slot_heads, val_loader, channels, device,
                               cli.tau_t, cli.tau_s, cli.chunk_size)
    match = abs(recomputed - new_val_kl) < 1e-5
    print(f'[reselect_klmq] {cli.cell}/{arm}: recomputed val_kl={recomputed:.6f} vs {new_val_kl:.6f} match={match}')
    assert match, '[ISSUE][ABORT] recomputed val_kl does not match the selection pass'

    test_kl = compute_val_kl(exp, args, model, slot_heads, test_loader, channels, device,
                             cli.tau_t, cli.tau_s, cli.chunk_size)

    out_dir = Path(cli.out_results_root) / cli.cell
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt_dir = Path(cli.out_ckpt_root) / cli.cell / arm
    out_ckpt_dir.mkdir(parents=True, exist_ok=True)
    ck['correction_note'] = ('OLD criterion was min val_retmse10; CORRECTED to min val_KL '
                             '(training objective) -- TRACK-W Correction B, CASE A re-selection.')
    torch.save(ck, out_ckpt_dir / 'checkpoint.pth')

    old_epoch, old_val = min(((int(torch.load(ef, map_location='cpu')['epoch']),
                              torch.load(ef, map_location='cpu').get('val_metric',
                                        torch.load(ef, map_location='cpu').get('val_retmse10')))
                             for ef in epoch_files), key=lambda t: t[1])
    summary = {
        'exp': 'TRACK-W-CHECKPOINT-CRITERION-CORRECTION01', 'correction': 'B (Original KL / Multi-Query)',
        'case': 'A (re-selection, no retraining -- full epoch trajectory already existed)',
        'cell': cli.cell, 'arm': arm, 'num_slots': cli.num_slots,
        'old_best_epoch': old_epoch, 'old_best_val_metric': old_val,
        'new_best_epoch': new_epoch, 'new_best_val_kl': new_val_kl,
        'test_kl': test_kl, 'recomputed_match': match,
    }
    (out_dir / f'retrieval_metrics_{arm}.json').write_text(json.dumps(summary, indent=2, default=str))
    (out_dir / f'DONE_{arm}.marker').write_text(json.dumps({'done': True, 'best_epoch': new_epoch}))
    print(f'[reselect_klmq] done. {cli.cell}/{arm} CASE A corrected: old_epoch={old_epoch} -> new_epoch={new_epoch}, '
         f'test_kl={test_kl:.6f}')


if __name__ == '__main__':
    main()
