#!/usr/bin/env python3
"""TRACK-K-MULTISLOT-PREDICTIVE-RETRIEVAL01.

Builds on TRACK-J2's J1 (StopGrad-Key) architecture: single shared MLP
encoder, candidates re-encoded every step then `.detach()`-ed before any
score/loss use (zero candidate-side gradient, candidate space still
moves with the encoder). This track adds S=10 retrieval "slots" -- one
query representation, S independent linear projections, each producing
its OWN score row over the full candidate bank, computed via ONE batched
`Q @ K^T` (candidates encoded once per step, never per slot).

No Set Oracle, no combinatorial search, no oracle prefix/sequence
anywhere in this file. Candidate selection is always score-based
(future-blind); future is used ONLY for the teacher target and the
(K2-only) differentiable aggregate loss.

K1 (multislot_relevance): L = L_anchor + beta * L_overlap. No aggregate
future loss at all.
K2 (multislot_aggregate): L = L_anchor + lambda_agg * L_agg + beta * L_overlap.

Reused UNMODIFIED: `build_model`/`state_hash` (`train_j_shared_encoder_drift01`),
`encode_raw`/`arm_score`/`individual_utility_memsafe` (`train_factorial_e2e01`),
`normalized_teacher_prob` (`train_horizon_retrieval_expert01`), `memory_value`
(`train_margutil01`), `stable_topk_indices` (`RelationStage1`),
`recall_at_k`/`ndcg_at_k` (`train_patch_retrieval_expert01`),
`make_loader_generator`/`batch_order_sha256` (`rng_control01`).
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.rng_control01 import batch_order_sha256, make_loader_generator
from scripts.train_factorial_e2e01 import encode_raw, individual_utility_memsafe
from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob
from scripts.train_j_shared_encoder_drift01 import build_model, state_hash
from scripts.train_margutil01 import memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k

EPS = 1e-8
N_SLOTS = 10
L_SOFT = 32
BETA_DEFAULT = 0.05
LAMBDA_AGG_DEFAULT = 1.0
EXPECTED_INIT_HASH = 'b37fa4031f538e4b5f5c522ae22e7d03613e4ee66283d2637656c7c4f872372e'


class SlotHeads(nn.Module):
    """S independent D x D linear maps, W_m = I + eps_m, eps_m ~
    N(0, std^2) with a PER-SLOT fixed seed (1000+m) -- deterministic,
    never all-identical (symmetry-breaking), small enough that step-0
    retrieval stays close to the single-head baseline (checked by a
    dedicated unit test and by the step-0 overlap diagnostic logged in
    main())."""

    def __init__(self, d_model, n_slots=N_SLOTS, std=1e-3):
        super().__init__()
        w = torch.eye(d_model).unsqueeze(0).repeat(n_slots, 1, 1)
        for m in range(n_slots):
            g = torch.Generator().manual_seed(1000 + m)
            w[m] = w[m] + torch.randn(d_model, d_model, generator=g) * std
        self.W = nn.Parameter(w)  # [S, D_out, D_in]

    def forward(self, z_q):
        q = torch.einsum('bd,sod->bso', z_q, self.W)  # [B, S, D]
        return F.normalize(q, dim=-1)


def kl_loss_from_prob(p_t, p_s, valid_mask, eps=EPS):
    """KL(p_t || p_s) where p_s is an ALREADY-COMPUTED probability
    (e.g. a mean-of-softmaxes) rather than a raw score `kl_loss` would
    itself softmax -- same math as `train_horizon_retrieval_expert01.kl_loss`,
    just taking a probability as input since p_bar (mean of per-slot
    softmaxes) is not expressible as softmax(single_score/tau)."""
    p_t = p_t.detach()
    log_p_s = torch.log(p_s.clamp_min(eps))
    term = (p_t * (torch.log(p_t.clamp_min(eps)) - log_p_s)).masked_fill(~valid_mask, 0.0)
    return term.sum(-1).mean()


def slot_overlap_penalty(p_m):
    """p_m: [B, S, N] probability distributions. Mean pairwise cosine
    similarity between DISTRIBUTIONS (not parameters), off-diagonal only."""
    p_norm = F.normalize(p_m, dim=-1)  # [B, S, N]
    cos = torch.einsum('bsn,btn->bst', p_norm, p_norm)  # [B, S, S]
    s = p_m.size(1)
    iu = torch.triu_indices(s, s, offset=1)
    return cos[:, iu[0], iu[1]].mean()


def soft_aggregate_loss(scores_masked, memory_c, offset_c, query_future, tau_s, l_soft=L_SOFT):
    """K2 only. Never materializes [B,S,N,H] -- per-slot Top-L gather
    only, shape [B, L, H] at any point."""
    bsz, s, n = scores_masked.shape
    y_soft_sum = query_future.new_zeros(bsz, query_future.size(-1))
    for m in range(s):
        s_m = scores_masked[:, m, :]
        idx_m = stable_topk_indices(s_m, l_soft, largest=True)  # [B, L], future-blind
        gathered = s_m.gather(1, idx_m)
        w_m = torch.softmax(gathered / tau_s, dim=-1)
        y_gathered = memory_c[idx_m] + offset_c.view(-1, 1, 1)  # [B, L, H]
        y_soft_sum = y_soft_sum + (w_m.unsqueeze(-1) * y_gathered).sum(dim=1)
    y_soft = y_soft_sum / s
    return ((y_soft - query_future) ** 2).mean(-1).mean()


def hard_unique_selection(scores, cand_mask, s=N_SLOTS):
    """Deterministic greedy-unique selection, FIXED slot order 0->S-1,
    each slot's OWN score row, excluding all candidates already picked
    by earlier slots. Never uses future information."""
    bsz, _, n = scores.shape
    device = scores.device
    picked_mask = torch.zeros(bsz, n, dtype=torch.bool, device=device)
    picks = []
    for m in range(s):
        avail = cand_mask & ~picked_mask
        s_m = scores[:, m, :].masked_fill(~avail, float('-inf'))
        idx_m = s_m.argmax(dim=-1)
        picks.append(idx_m)
        picked_mask.scatter_(1, idx_m.unsqueeze(1), True)
    return torch.stack(picks, dim=1)  # [B, S]


def compute_scores(model, slot_heads, batch_x, memory_x, c):
    z_q = encode_raw(model, batch_x, c)
    q = slot_heads(z_q)  # [B, S, D]
    k_full = F.normalize(encode_raw(model, memory_x, c), dim=-1).detach()  # [N, D], stopgrad (J1-style)
    scores = torch.einsum('bsd,nd->bsn', q, k_full)  # [B, S, N]
    return scores


def hard_eval_decomposition(scores, cand_mask, memory_c, offset_c, query_future, d_raw, oracle_idx, top_k=N_SLOTS):
    model_idx = hard_unique_selection(scores, cand_mask, s=top_k)  # [B, S]
    model_ind_mse = d_raw.gather(1, model_idx).mean(-1)
    oracle_ind_mse = d_raw.gather(1, oracle_idx).mean(-1)
    recall10 = recall_at_k(model_idx, oracle_idx, top_k)
    ndcg10 = ndcg_at_k(model_idx, d_raw, cand_mask, top_k)
    y_sel = memory_c[model_idx] + offset_c.view(-1, 1, 1)
    e = y_sel - query_future.unsqueeze(1)
    individual_mse_i = (e ** 2).mean(-1)
    D_ = individual_mse_i.mean(-1) / top_k
    agg_pred = y_sel.mean(dim=1)
    agg_mse = ((agg_pred - query_future) ** 2).mean(-1)
    C_ = agg_mse - D_
    return dict(model_idx=model_idx, model_ind_mse=model_ind_mse, oracle_ind_mse=oracle_ind_mse,
               recall10=recall10, ndcg10=ndcg10, agg_mse=agg_mse, D=D_, C=C_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--arm', required=True, choices=('K1_multislot_relevance', 'K2_multislot_aggregate'))
    ap.add_argument('--cell', default='ETTh1_720')
    ap.add_argument('--pred_len', type=int, default=720)
    ap.add_argument('--seq_len', type=int, default=720)
    ap.add_argument('--patch_len', type=int, default=16)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--tau_t', type=float, default=0.1)
    ap.add_argument('--tau_s', type=float, default=0.1)
    ap.add_argument('--beta', type=float, default=BETA_DEFAULT)
    ap.add_argument('--lambda_agg', type=float, default=LAMBDA_AGG_DEFAULT)
    ap.add_argument('--l_soft', type=int, default=L_SOFT)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--train_epochs', type=int, default=10)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--init_seed', type=int, default=0)
    ap.add_argument('--loader_seed', type=int, default=0)
    ap.add_argument('--out_dir', default='results/TRACK-K-MULTISLOT-PREDICTIVE-RETRIEVAL01')
    ap.add_argument('--checkpoints', default='checkpoints/track_k_multislot_predictive_retrieval01')
    ap.add_argument('--limit_batches', type=int, default=0, help='SMOKE ONLY')
    ap.add_argument('--smoke_test', action='store_true')
    cli = ap.parse_args()
    use_agg = cli.arm == 'K2_multislot_aggregate'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(cli.out_dir) / cli.cell / cli.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints) / cli.cell / cli.arm
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    exp, args, model = build_model(cli, device)
    channels = list(range(int(args.enc_in)))
    assert state_hash(model) == EXPECTED_INIT_HASH, '[ISSUE][ABORT] init hash mismatch vs J0/J1/J2'
    d_model = int(args.d_model)

    slot_heads = SlotHeads(d_model, N_SLOTS).to(device)

    for p in model.parameters():
        p.requires_grad_(True)
    params = list(model.parameters()) + list(slot_heads.parameters())
    optimizer = torch.optim.Adam(params, lr=cli.learning_rate)

    train_gen = make_loader_generator(cli.loader_seed)
    _, train_loader = exp._get_data(flag='train', shuffle=True, generator=train_gen)
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)

    config = {'cell': cli.cell, 'arm': cli.arm, 'n_slots': N_SLOTS, 'l_soft': cli.l_soft, 'beta': cli.beta,
              'lambda_agg': cli.lambda_agg if use_agg else 0.0, 'tau_t': cli.tau_t, 'tau_s': cli.tau_s,
              'top_k': cli.top_k, 'batch_size': cli.batch_size, 'learning_rate': cli.learning_rate,
              'train_epochs': cli.train_epochs, 'patience': cli.patience, 'init_seed': cli.init_seed,
              'loader_seed': cli.loader_seed, 'n_candidates': int(exp.memory_x.size(0)),
              'checkpoint_criterion': 'min val HARD uniform_agg_mse10'}
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2))
    print(f'[track_k] arm={cli.arm} N={config["n_candidates"]} init_hash={state_hash(model)[:16]}')

    def eval_channel(batch_x, batch_y, batch_start_idx, c, collect_slot_diag=False):
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        scores = compute_scores(model, slot_heads, batch_x, exp.memory_x, c)
        memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
        query_future = batch_y[:, :, c]
        u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
        d_raw = -u
        oracle_idx = stable_topk_indices(d_raw.masked_fill(~cand_mask, float('inf')), cli.top_k, largest=False)
        res = hard_eval_decomposition(scores, cand_mask, memory_c, offset_c, query_future, d_raw, oracle_idx,
                                      cli.top_k)
        slot_diag = None
        if collect_slot_diag:
            with torch.no_grad():
                s_masked = scores.masked_fill(~cand_mask.unsqueeze(1), float('-inf'))
                top1 = s_masked.argmax(dim=-1)  # [B, S]
                p_m = torch.softmax(s_masked / cli.tau_s, dim=-1)
                overlap = float(slot_overlap_penalty(p_m))
                slot_diag = {'top1_idx': top1.cpu(), 'overlap': overlap}
        return res, scores, cand_mask, memory_c, offset_c, query_future, d_raw, oracle_idx, slot_diag

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
                l_anchor = kl_loss_from_prob(p_t, p_bar, cand_mask)
                l_overlap = slot_overlap_penalty(p_m)
                l = l_anchor + cli.beta * l_overlap
                if use_agg:
                    l_agg = soft_aggregate_loss(s_masked, memory_c, offset_c, query_future, cli.tau_s, cli.l_soft)
                    l = l + cli.lambda_agg * l_agg
                (l / len(channels)).backward()
            assert all(p.grad is not None for p in slot_heads.parameters())
            optimizer.step()
        vram = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
        print(f'[track_k] {cli.arm} SMOKE PASS max_vram_mb={vram:.1f}')
        return

    epoch_val_rows, step_rows, slot_rows = [], [], []
    best = {'val_agg': float('inf'), 'epoch': -1}
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
            batch_loss, batch_anchor, batch_overlap, batch_agg = 0.0, 0.0, 0.0, 0.0
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
                l_anchor = kl_loss_from_prob(p_t, p_bar, cand_mask)
                l_overlap = slot_overlap_penalty(p_m)
                l = l_anchor + cli.beta * l_overlap
                l_agg_val = 0.0
                if use_agg:
                    l_agg = soft_aggregate_loss(s_masked, memory_c, offset_c, query_future, cli.tau_s, cli.l_soft)
                    l = l + cli.lambda_agg * l_agg
                    l_agg_val = float(l_agg.detach())
                (l / len(channels)).backward()
                batch_loss += float(l.detach()) / len(channels)
                batch_anchor += float(l_anchor.detach()) / len(channels)
                batch_overlap += float(l_overlap.detach()) / len(channels)
                batch_agg += l_agg_val / len(channels)
            optimizer.step()
            global_step += 1
            step_rows.append({'global_step': global_step, 'epoch': epoch, 'batch_in_epoch': bi,
                              'train_loss': batch_loss, 'train_anchor': batch_anchor,
                              'train_overlap': batch_overlap, 'train_agg': batch_agg})
        batch_order_epoch1 = batch_order_sha256(starts) if epoch == 1 else None

        # full validation (hard inference)
        model.eval()
        val_sums, n = {}, 0
        slot_top1_all = {c: [] for c in channels}
        overlap_sum, overlap_n = 0.0, 0
        with torch.no_grad():
            for batch_x, batch_y, batch_start_idx in val_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                bsz = batch_x.size(0)
                per_ch = {}
                for c in channels:
                    res, scores, cand_mask, memory_c, offset_c, query_future, d_raw, oracle_idx, slot_diag = \
                        eval_channel(batch_x, batch_y, batch_start_idx, c, collect_slot_diag=True)
                    per_ch.setdefault('retmse10', []).append(res['model_ind_mse'].cpu())
                    per_ch.setdefault('agg_mse10', []).append(res['agg_mse'].cpu())
                    per_ch.setdefault('recall10', []).append(res['recall10'].cpu())
                    per_ch.setdefault('ndcg10', []).append(res['ndcg10'].cpu())
                    per_ch.setdefault('oracle_regret', []).append((res['model_ind_mse'] - res['oracle_ind_mse']).cpu())
                    slot_top1_all[c].append(slot_diag['top1_idx'])
                    overlap_sum += slot_diag['overlap']
                    overlap_n += 1
                for k_, vals in per_ch.items():
                    val_sums[k_] = val_sums.get(k_, 0.0) + torch.cat(vals).sum().item()
                n += bsz
        val_metrics = {k_: v / max(n * len(channels), 1) for k_, v in val_sums.items()}
        val_agg = val_metrics['agg_mse10']
        val_metrics['epoch'] = epoch
        val_metrics['slot_overlap'] = overlap_sum / max(overlap_n, 1)
        epoch_val_rows.append(val_metrics)
        payload = {'model_state_dict': model.state_dict(), 'slot_heads_state_dict': slot_heads.state_dict(),
                  'epoch': epoch, 'val_agg': val_agg, 'config': config}
        torch.save(payload, ckpt_dir / f'checkpoint_epoch{epoch}.pth')
        if val_agg < best['val_agg']:
            best = {'val_agg': val_agg, 'epoch': epoch}
            torch.save(payload, ckpt_dir / 'checkpoint.pth')
        print(f'[track_k] {cli.arm} epoch={epoch} done val_agg={val_agg:.6f} val_retmse10={val_metrics["retmse10"]:.6f} '
             f'slot_overlap={val_metrics["slot_overlap"]:.4f} (best={best["epoch"]}:{best["val_agg"]:.6f})')
        if epoch - best['epoch'] >= cli.patience:
            print(f'[track_k] {cli.arm} early stop at epoch {epoch} (best={best["epoch"]})')
            break

    bl = torch.load(ckpt_dir / 'checkpoint.pth', map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    slot_heads.load_state_dict(bl['slot_heads_state_dict'])
    model.eval()

    # final test (hard + soft-hard gap for K2)
    test_sums, n = {}, 0
    per_ch_slot_stats = {c: {'top1': []} for c in channels}
    with torch.no_grad():
        for batch_x, batch_y, batch_start_idx in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            bsz = batch_x.size(0)
            per_ch = {}
            for c in channels:
                res, scores, cand_mask, memory_c, offset_c, query_future, d_raw, oracle_idx, slot_diag = \
                    eval_channel(batch_x, batch_y, batch_start_idx, c, collect_slot_diag=True)
                per_ch.setdefault('retmse10', []).append(res['model_ind_mse'].cpu())
                per_ch.setdefault('agg_mse10_hard', []).append(res['agg_mse'].cpu())
                per_ch.setdefault('recall10', []).append(res['recall10'].cpu())
                per_ch.setdefault('ndcg10', []).append(res['ndcg10'].cpu())
                per_ch.setdefault('oracle_regret', []).append((res['model_ind_mse'] - res['oracle_ind_mse']).cpu())
                per_ch.setdefault('D', []).append(res['D'].cpu())
                per_ch.setdefault('C', []).append(res['C'].cpu())
                per_ch_slot_stats[c]['top1'].append(slot_diag['top1_idx'])
                if use_agg:
                    s_masked = scores.masked_fill(~cand_mask.unsqueeze(1), float('-inf'))
                    l_soft_agg = soft_aggregate_loss(s_masked, memory_c, offset_c, query_future, cli.tau_s, cli.l_soft)
                    per_ch.setdefault('agg_mse10_soft', []).append(torch.full((bsz,), float(l_soft_agg)))
            for k_, vals in per_ch.items():
                test_sums[k_] = test_sums.get(k_, 0.0) + torch.cat(vals).sum().item()
            n += bsz
    test_metrics = {k_: v / max(n * len(channels), 1) for k_, v in test_sums.items()}
    test_metrics['n_queries_seen'] = n

    # slot specialization diagnostics (top-1 identity overlap across slots, pooled over test)
    slot_rows = []
    for c in channels:
        top1_cat = torch.cat(per_ch_slot_stats[c]['top1'], dim=0)  # [n, S]
        for m in range(N_SLOTS):
            for mm in range(m + 1, N_SLOTS):
                overlap_frac = float((top1_cat[:, m] == top1_cat[:, mm]).float().mean())
                slot_rows.append({'channel': c, 'slot_a': m, 'slot_b': mm, 'top1_identity_overlap': overlap_frac})

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
    with open(out_dir_top / 'slot_metrics.csv', 'w', newline='') as fh:
        fieldnames = sorted({k for r in slot_rows for k in r})
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in slot_rows:
            w.writerow(r)
    if use_agg:
        gap = test_metrics['agg_mse10_hard'] - test_metrics.get('agg_mse10_soft', float('nan'))
        with open(out_dir_top / 'soft_hard_gap.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['agg_mse10_hard', 'agg_mse10_soft', 'gap'])
            w.writeheader()
            w.writerow({'agg_mse10_hard': test_metrics['agg_mse10_hard'],
                       'agg_mse10_soft': test_metrics.get('agg_mse10_soft'), 'gap': gap})
    (out_dir_top / 'final_test_metrics.json').write_text(json.dumps(
        {**test_metrics, 'best_epoch': best['epoch']}, indent=2))
    (out_dir_top / 'checkpoint_fingerprint.json').write_text(json.dumps(
        {'init_hash': EXPECTED_INIT_HASH, 'best_epoch': best['epoch'],
         'final_model_hash': state_hash(model), 'batch_order_epoch1': batch_order_epoch1}, indent=2))
    vram = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
    print(f'[track_k] done. {cli.arm} best_epoch={best["epoch"]} test_retMSE@10={test_metrics["retmse10"]:.6f} '
         f'test_agg_mse10_hard={test_metrics["agg_mse10_hard"]:.6f} max_vram_mb={vram:.1f} '
         f'wall_seconds={time.time()-t0:.1f}')


if __name__ == '__main__':
    main()
