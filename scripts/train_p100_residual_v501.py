#!/usr/bin/env python3
"""TRACK-P100-COARSE-TO-FINE-RETRIEVAL01 -- PHASE C: trainable
Residual-V5 scorer. Tests whether training V5's score as a residual
CORRECTION on top of the frozen-form coarse prior is more stable than
training it to rank candidates from scratch (canonical V5). Shared
-P100 candidate identity, fixed, exactly as every other Phase in this
track.

Architecture (spec section 18-20), the ONLY change from canonical V5:

    s_final_{h,i} = c_i + gamma * r_{h,i}
    r_{h,i}       = cosine(W_h z_q, z_i)         (= canonical V5's own per-head score, UNCHANGED math)
    c_i           = the SAME coarse delta-last-cosine score Phase A/B reuse, UNSCALED
                    (both c_i and r_{h,i} are already L2-normalized-vector cosine similarities
                    in [-1,1] -- no additional scale factor is introduced, so gamma=0 recovers
                    EXACTLY "coarse score only" for every head, with no head distinction, as required)
    gamma         ONE global scalar, shared by all 5 heads, init=0, trainable

Exactly ONE new trainable parameter vs canonical V5 (gamma); encoder
and SlotHeads architecture are otherwise untouched -- same `build_model`
+ `SlotHeads` construction, SAME init_seed/loader_seed as canonical
V5's own training run (`scripts/train_v_sharedtop100_01.py`, reused
for every other piece: `build_model`, `make_loader_generator`,
`normalized_teacher_prob`, `kl_loss_from_prob`, `encode_raw`,
`CandidatePoolCache`, `encode_pooled_candidates`/`gather_candidate_histories`).

Training loss is UNCHANGED canonical Future-MSE teacher KL:
`L = KL(p_teacher || p_mix)`, `p_mix = mean_h softmax(s_final_h/tau_s)`
-- no new auxiliary loss. Checkpoint selection deviates from
`train_v_sharedtop100_01.py`'s own historical round-robin-based
val_retmse10 by explicit spec instruction (section 21: "Checkpoint
selection: validation Mean-Mixture RetMSE@10") -- uses
`mean_mixture_topk_selection`, matching this track's own Phase A/B
convention (and the session's broader TRACK-V-MEANMIX-INFERENCE01
correction), not round-robin.

Controls (section 19): C0 is just the existing frozen canonical V5
artifact (no new run here -- read at report time). C1 (this script's
primary output) trains gamma from gamma=0 jointly with the full
encoder+SlotHeads. C2 (gamma FIXED=1, no training of gamma itself) is
run as a lightweight read-only/post-hoc diagnostic via
`--gamma_fixed 1.0 --epochs 0` on the SAME fresh init as C1 (one
forward pass worth of score computation, no backward/optimizer step)
-- per spec "C2는 가능하면 read-only/posthoc score diagnostic 또는
minimal control", this script treats "epoch 0" (fresh init, fixed
gamma, immediate val/test eval, no training loop) as exactly that
minimal control rather than a full second training run.
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
from scripts.diag_p100_coarse_v5_fusion01 import spearman_batched
from scripts.rng_control01 import batch_order_sha256, make_loader_generator
from scripts.train_factorial_e2e01 import encode_raw
from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob
from scripts.train_j_shared_encoder_drift01 import build_model, state_hash
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads, kl_loss_from_prob
from scripts.train_margutil01 import memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k
from scripts.train_retriever_pool01 import CandidatePoolCache
from utils.candidate_pool import compute_coarse_delta_last_cosine_scores, gather_candidate_values
from utils.mean_mixture_selection import mean_mixture_topk_selection

EPS = 1e-8
TOP_K = 10
N_SLOTS = 5


def eval_channel(model, slot_heads, gamma, exp, args, pool_caches, cli, device,
                 batch_x, batch_y, batch_start_idx, c, split):
    pool_idx_global = pool_caches[split].lookup(batch_start_idx, c, device)
    pooled_x = exp.memory_x[pool_idx_global]  # [B,100,L,C]
    bsz, pool_m = pooled_x.shape[0], pooled_x.shape[1]
    flat = pooled_x.reshape(bsz * pool_m, pooled_x.size(2), pooled_x.size(3))
    z_k = F.normalize(encode_raw(model, flat, c), dim=-1).reshape(bsz, pool_m, -1)

    z_q = encode_raw(model, batch_x, c)
    q = slot_heads(z_q)  # [B,5,D]
    r = torch.einsum('bsd,bmd->bsm', q, z_k)  # [B,5,100] residual (canonical V5's own raw score)

    coarse_full = compute_coarse_delta_last_cosine_scores(batch_x, exp.memory_x, c)
    c_i = coarse_full.gather(1, pool_idx_global)  # [B,100]
    s_final = c_i.unsqueeze(1) + gamma * r  # [B,5,100], same coarse term broadcast to every head

    valid_mask = torch.ones_like(c_i, dtype=torch.bool)  # P100 pool candidates are all valid by construction

    memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
    query_future = batch_y[:, :, c]
    pooled_memory_c = gather_candidate_values(memory_c, pool_idx_global)
    d_pool = ((pooled_memory_c + offset_c.view(-1, 1, 1) - query_future.unsqueeze(1)) ** 2).mean(dim=-1)

    p_t = normalized_teacher_prob(d_pool, valid_mask, cli.tau_t)
    p_heads = torch.softmax(s_final / cli.tau_s, dim=-1)
    p_mix = p_heads.mean(dim=1)
    kl = kl_loss_from_prob(p_t, p_mix, valid_mask)

    picks_local, _, _ = mean_mixture_topk_selection(s_final, valid_mask, cli.tau_s, k=cli.top_k)
    oracle_idx_local = stable_topk_indices(d_pool, cli.top_k, largest=False)
    individual_mse = d_pool.gather(1, picks_local)
    y_sel = pooled_memory_c.gather(1, picks_local.unsqueeze(-1).expand(-1, -1, pooled_memory_c.size(-1))
                                   ) + offset_c.view(-1, 1, 1)
    agg_mse = ((y_sel.mean(dim=1) - query_future) ** 2).mean(dim=-1)
    recall10 = recall_at_k(picks_local, oracle_idx_local, cli.top_k)
    ndcg10 = ndcg_at_k(picks_local, d_pool, valid_mask, cli.top_k)

    return dict(kl=kl, retmse10=individual_mse.mean(-1), agg_mse10=agg_mse,
               recall10=recall10, ndcg10=ndcg10, s_final=s_final, c_i=c_i, r=r,
               oracle_idx_local=oracle_idx_local, p_t=p_t)


def mechanism_diagnostics(res, tau_s, top_k):
    """Spec section 22: coarse/V5/final ranking overlap, Spearman with
    teacher, score scale, Head Top10 overlap, Union size, Union Oracle
    Recall -- all restricted to the SAME P100 pool, read-only."""
    s_final, c_i, r = res['s_final'], res['c_i'], res['r']
    oracle_idx, p_t = res['oracle_idx_local'], res['p_t']
    n_slots = s_final.size(1)

    p_heads_final = torch.softmax(s_final / tau_s, dim=-1)
    p_mix_final = p_heads_final.mean(dim=1)  # [B,100]
    r_mix = r.mean(dim=1)  # mean-head raw residual score, for coarse-vs-residual-only comparison

    top10_final = stable_topk_indices(p_mix_final, top_k, largest=True)
    top10_coarse = stable_topk_indices(c_i, top_k, largest=True)
    top10_r = stable_topk_indices(r_mix, top_k, largest=True)

    def _overlap(a, b):
        return float(((a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1).float().sum(-1) / top_k).mean())

    out = {
        'overlap_final_vs_coarse': _overlap(top10_final, top10_coarse),
        'overlap_final_vs_residual': _overlap(top10_final, top10_r),
        'overlap_coarse_vs_residual': _overlap(top10_coarse, top10_r),
        'spearman_final_vs_teacher': float(spearman_batched(p_mix_final, p_t).nanmean()),
        'spearman_coarse_vs_teacher': float(spearman_batched(
            torch.softmax(c_i, dim=-1), p_t).nanmean()),
        's_final_mean': float(s_final.mean()), 's_final_std': float(s_final.std()),
        'r_mean': float(r.mean()), 'r_std': float(r.std()),
    }

    head_top10 = [stable_topk_indices(p_heads_final[:, m, :], top_k, largest=True) for m in range(n_slots)]
    pair_overlaps = []
    for m in range(n_slots):
        for mm in range(m + 1, n_slots):
            pair_overlaps.append(torch.tensor(_overlap(head_top10[m], head_top10[mm])))
    out['head_top10_overlap_mean'] = float(torch.stack(pair_overlaps).mean())

    bsz, n = s_final.size(0), s_final.size(2)
    union_mask = torch.zeros(bsz, n, dtype=torch.bool, device=s_final.device)
    curve = []
    for m in range(n_slots):
        union_mask.scatter_(1, head_top10[m], True)
        curve.append(float(union_mask.gather(1, oracle_idx).float().mean(dim=-1).mean()))
    out['union_size_mean'] = float(union_mask.float().sum(dim=-1).mean())
    out['union_oracle_recall_by_n_heads'] = curve
    return out


def run_split(model, slot_heads, gamma, exp, args, pool_caches, cli, device, split, want_diag=False):
    channels = list(range(int(args.enc_in)))
    _, loader = exp._get_data(flag=split, shuffle=False)
    sums, n = {}, 0
    diag_rows = []
    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        bsz = batch_x.size(0)
        for c in channels:
            res = eval_channel(model, slot_heads, gamma, exp, args, pool_caches, cli, device,
                               batch_x, batch_y, batch_start_idx, c, split)
            for key_ in ('kl', 'retmse10', 'agg_mse10', 'recall10', 'ndcg10'):
                v = res[key_].detach()
                sums[key_] = sums.get(key_, 0.0) + float(v.sum())
            if want_diag:
                diag_rows.append(mechanism_diagnostics(res, cli.tau_s, cli.top_k))
        n += bsz
    n_eff = max(n * len(channels), 1)
    metrics = {k_: v / n_eff for k_, v in sums.items()}
    if want_diag and diag_rows:
        keys = [k for k in diag_rows[0] if k != 'union_oracle_recall_by_n_heads']
        metrics['diagnostics'] = {k: sum(r[k] for r in diag_rows) / len(diag_rows) for k in keys}
        curve_len = len(diag_rows[0]['union_oracle_recall_by_n_heads'])
        metrics['diagnostics']['union_oracle_recall_by_n_heads'] = [
            sum(r['union_oracle_recall_by_n_heads'][i] for r in diag_rows) / len(diag_rows)
            for i in range(curve_len)]
    return metrics


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
    ap.add_argument('--init_seed', type=int, default=0)
    ap.add_argument('--loader_seed', type=int, default=0)
    ap.add_argument('--candidate_mask', default='raft')
    ap.add_argument('--candidate_pool_size', type=int, default=100)
    ap.add_argument('--candidate_pool_cache', required=True)
    ap.add_argument('--gamma_fixed', type=float, default=None,
                    help='C2 control: fix gamma at this value, run 0 training epochs '
                         '(fresh init, immediate val/test eval only -- a read-only/minimal '
                         'control, not a second full training run, per spec section 19).')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--checkpoints', required=True)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    arm = 'C2_gamma_fixed' if cli.gamma_fixed is not None else 'C1_residual_v5'
    out_dir = Path(cli.out_dir) / cli.cell / arm
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints) / cli.cell / arm
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    exp, args, model = build_model(cli, device)
    init_hash = state_hash(model)
    d_model = int(args.d_model)
    slot_heads = SlotHeads(d_model, n_slots=N_SLOTS, std=cli.slot_std).to(device)
    gamma = nn.Parameter(torch.tensor(0.0 if cli.gamma_fixed is None else cli.gamma_fixed, device=device),
                         requires_grad=cli.gamma_fixed is None)

    expected_meta = {'seq_len': cli.seq_len, 'pred_len': cli.pred_len, 'candidate_mask': cli.candidate_mask,
                     'candidate_pool_size': cli.candidate_pool_size, 'candidate_pool_metric': 'delta_last_cosine'}
    cache_dir = Path(cli.candidate_pool_cache)
    pool_caches = {s: CandidatePoolCache(cache_dir / f'{s}.pt', expected_meta) for s in ('train', 'val', 'test')}

    for p in model.parameters():
        p.requires_grad_(True)
    params = list(model.parameters()) + list(slot_heads.parameters())
    if cli.gamma_fixed is None:
        params = params + [gamma]
    optimizer = torch.optim.Adam(params, lr=cli.learning_rate)

    config = {'exp': 'TRACK-P100-COARSE-TO-FINE-RETRIEVAL01-PHASE-C', 'cell': cli.cell, 'arm': arm,
             'gamma_init': 0.0 if cli.gamma_fixed is None else cli.gamma_fixed,
             'gamma_trainable': cli.gamma_fixed is None,
             'tau_t': cli.tau_t, 'tau_s': cli.tau_s, 'top_k': cli.top_k, 'batch_size': cli.batch_size,
             'learning_rate': cli.learning_rate, 'train_epochs': cli.train_epochs,
             'init_seed': cli.init_seed, 'loader_seed': cli.loader_seed, 'init_hash': init_hash}
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2, default=str))
    print(f'[residual_v5] arm={arm} cell={cli.cell} gamma_init={config["gamma_init"]} init_hash={init_hash[:16]}')

    if cli.gamma_fixed is not None:
        # C2 minimal control: fresh init, fixed gamma, NO training loop at all.
        model.eval(); slot_heads.eval()
        with torch.no_grad():
            val_metrics = run_split(model, slot_heads, gamma, exp, args, pool_caches, cli, device, 'val')
            test_metrics = run_split(model, slot_heads, gamma, exp, args, pool_caches, cli, device, 'test', want_diag=True)
        (out_dir / 'val_metrics.json').write_text(json.dumps(val_metrics, indent=2))
        (out_dir / 'final_test_metrics.json').write_text(json.dumps(test_metrics, indent=2, default=str))
        print(f'[residual_v5] {arm} (no training) val_agg_mse10={val_metrics["agg_mse10"]:.6f} '
             f'test_agg_mse10={test_metrics["agg_mse10"]:.6f}')
        return

    train_gen = make_loader_generator(cli.loader_seed)
    _, train_loader = exp._get_data(flag='train', shuffle=True, generator=train_gen)
    channels = list(range(int(args.enc_in)))

    epoch_rows, gamma_trajectory, batch_order_hashes = [], [], []
    best = {'val': float('inf'), 'epoch': -1}
    t0 = time.time()
    for epoch in range(1, cli.train_epochs + 1):
        model.train(); slot_heads.train()
        starts = []
        for bi, (batch_x, batch_y, batch_start_idx) in enumerate(train_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            starts.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx)
                         else torch.as_tensor(batch_start_idx))
            optimizer.zero_grad()
            for c in channels:
                res = eval_channel(model, slot_heads, gamma, exp, args, pool_caches, cli, device,
                                   batch_x, batch_y, batch_start_idx, c, 'train')
                (res['kl'] / len(channels)).backward()
            optimizer.step()
        batch_order_hashes.append({'epoch': epoch, 'batch_order_sha256': batch_order_sha256(starts)})

        model.eval(); slot_heads.eval()
        with torch.no_grad():
            val_metrics = run_split(model, slot_heads, gamma, exp, args, pool_caches, cli, device, 'val')
        val_metrics['epoch'] = epoch
        val_metrics['gamma'] = float(gamma.detach())
        epoch_rows.append(val_metrics)
        gamma_trajectory.append(float(gamma.detach()))

        payload = {'model_state_dict': model.state_dict(), 'slot_heads_state_dict': slot_heads.state_dict(),
                  'gamma': float(gamma.detach()), 'epoch': epoch,
                  'val_retmse10': val_metrics['retmse10'], 'config': config}
        torch.save(payload, ckpt_dir / f'checkpoint_epoch{epoch}.pth')
        if val_metrics['retmse10'] < best['val']:
            best = {'val': val_metrics['retmse10'], 'epoch': epoch}
            torch.save(payload, ckpt_dir / 'checkpoint_best_retmse.pth')
        print(f'[residual_v5] {arm} epoch={epoch} gamma={gamma_trajectory[-1]:.6f} '
             f'val_kl={val_metrics["kl"]:.6f} val_retmse10={val_metrics["retmse10"]:.6f} '
             f'val_agg_mse10={val_metrics["agg_mse10"]:.6f} (best_epoch={best["epoch"]}:{best["val"]:.6f})')

    bl = torch.load(ckpt_dir / 'checkpoint_best_retmse.pth', map_location=device)
    model.load_state_dict(bl['model_state_dict']); slot_heads.load_state_dict(bl['slot_heads_state_dict'])
    gamma = nn.Parameter(torch.tensor(bl['gamma'], device=device), requires_grad=False)
    model.eval(); slot_heads.eval()
    with torch.no_grad():
        test_metrics = run_split(model, slot_heads, gamma, exp, args, pool_caches, cli, device, 'test', want_diag=True)
    test_metrics['best_epoch'] = bl['epoch']
    test_metrics['gamma_at_best_epoch'] = bl['gamma']
    (out_dir / 'final_test_metrics.json').write_text(json.dumps(test_metrics, indent=2, default=str))

    with open(out_dir / 'val_metrics.csv', 'w', newline='') as fh:
        fieldnames = sorted({k for r in epoch_rows for k in r})
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in epoch_rows:
            w.writerow(r)
    (out_dir / 'gamma_trajectory.json').write_text(json.dumps(gamma_trajectory, indent=2))
    (out_dir / 'checkpoint_fingerprints.json').write_text(json.dumps(
        {'init_hash': init_hash, 'best_epoch': best['epoch'], 'batch_order_hashes': batch_order_hashes}, indent=2))
    vram = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
    print(f'[residual_v5] done. {arm} best_epoch={best["epoch"]} gamma_at_best={bl["gamma"]:.6f} '
         f'test_agg_mse10={test_metrics["agg_mse10"]:.6f} max_vram_mb={vram:.1f} wall_seconds={time.time()-t0:.1f}')


if __name__ == '__main__':
    main()
