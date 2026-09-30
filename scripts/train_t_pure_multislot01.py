#!/usr/bin/env python3
"""TRACK-T-PURE-MULTISLOT-VALIDATION01.

Question (PART 0): does giving Original KL retrieval multiple independent
query "slots" (S rankings instead of 1), with everything else held fixed
-- same shared encoder, same teacher, same KL objective, same FULL
gradient (candidate/key side NOT stopgrad'd) -- improve the retrieved
Top-10 set, on its own, without StopGrad-Key, overlap penalty, aggregate
loss, or relevance budget?

T0 (S=1) / T1 (S=2) / T2 (S=4) / T3 (S=10) are ALL the exact same code
path here, differing ONLY in `--num_slots`. Loss is ALWAYS exactly:

    L = KL(p_T || mean_m softmax(s_m / tau_s))

-- no overlap term, no aggregate-future term, no relevance-budget term
(unlike TRACK-K/TRACK-M, which add those on top of this same anchor
term). Candidate/key side is encoded fresh every step and NEVER
`.detach()`-ed (PART 11: full gradient, both branches -- this is what
makes T0 == Original KL rather than == J1/M2's StopGrad-Key ancestor).

Final retrieval Top-K is ALWAYS K=10 regardless of `--num_slots`, via
deterministic round-robin unique selection across slots (PART 5/6):
slot 0's best available, slot 1's best available, ..., slot (S-1)'s,
then slot 0's NEXT best available, etc., until 10 unique candidates are
picked. At S=1 this is mathematically the same greedy process as
ordinary Top-10 selection (verified by a dedicated unit test).

Reused UNMODIFIED: `build_model`/`state_hash` (`train_j_shared_encoder_drift01`),
`encode_raw`/`individual_utility_memsafe` (`train_factorial_e2e01`),
`normalized_teacher_prob` (`train_horizon_retrieval_expert01`),
`memory_value` (`train_margutil01`), `stable_topk_indices`
(`RelationStage1`), `recall_at_k`/`ndcg_at_k` (`train_patch_retrieval_expert01`),
`make_loader_generator`/`batch_order_sha256` (`rng_control01`),
`SlotHeads`/`kl_loss_from_prob`/`slot_overlap_penalty` (`train_k_multislot_predictive_retrieval01`
-- SlotHeads' own `n_slots` constructor argument is used as-is, so no
S-dependent code change is needed there; `slot_overlap_penalty` and
`kl_loss_from_prob` are used exactly as TRACK-K defined them, just with
the candidate side NOT detached upstream, in `compute_scores_full_grad`
below).

NOT reused (TRACK-K's `compute_scores` detaches the candidate side --
StopGrad-Key -- which this track must NOT have): a new
`compute_scores_full_grad` is defined here instead. NOT reused (TRACK-K's
`hard_unique_selection` picks exactly one candidate PER SLOT, i.e. only
S candidates total, not a fixed K=10 regardless of S): a new
`round_robin_topk_selection` is defined here instead, plus a
`hard_eval_decomposition` analog built on it.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.rng_control01 import batch_order_sha256, make_loader_generator
from scripts.train_factorial_e2e01 import encode_raw, individual_utility_memsafe
from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob
from scripts.train_j_shared_encoder_drift01 import build_model, state_hash
from scripts.train_k_multislot_predictive_retrieval01 import (
    SlotHeads, kl_loss_from_prob, slot_overlap_penalty,
)
from scripts.train_margutil01 import memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k

EPS = 1e-8
TOP_K = 10
VALID_NUM_SLOTS = (1, 2, 4, 10)


def compute_scores_full_grad(model, slot_heads, batch_x, memory_x, c):
    """PART 2/3/11: query AND candidate/key gradients both ON -- candidate
    bank is re-encoded every call, never `.detach()`-ed. The one intended
    architectural difference from TRACK-K/TRACK-M's `compute_scores` is
    exactly this missing `.detach()`."""
    z_q = encode_raw(model, batch_x, c)
    q = slot_heads(z_q)  # [B, S, D], already L2-normalized inside SlotHeads
    k_full = F.normalize(encode_raw(model, memory_x, c), dim=-1)  # [N, D], gradient ON
    scores = torch.einsum('bsd,nd->bsn', q, k_full)  # [B, S, N]
    return scores


def round_robin_topk_selection(scores, cand_mask, k=TOP_K):
    """PART 5/6: deterministic round-robin unique selection, slot order
    0->S-1 repeating, until exactly `k` unique candidates are picked --
    k is FIXED at 10 regardless of S (the number of slots). At S=1 this
    reduces exactly to greedy iterative argmax-excluding-already-picked
    over the single score row, i.e. ordinary Top-10 (verified by unit
    test 3)."""
    bsz, s, n = scores.shape
    device = scores.device
    picked_mask = torch.zeros(bsz, n, dtype=torch.bool, device=device)
    picks = []
    for t in range(k):
        m = t % s
        avail = cand_mask & ~picked_mask
        s_m = scores[:, m, :].masked_fill(~avail, float('-inf'))
        idx_m = s_m.argmax(dim=-1)
        picks.append(idx_m)
        picked_mask.scatter_(1, idx_m.unsqueeze(1), True)
    return torch.stack(picks, dim=1)  # [B, k]


def naive_round_robin_picks(scores, cand_mask, k=TOP_K):
    """PART 14 diagnostic ONLY (never used for training or selection):
    same slot-cycling order as `round_robin_topk_selection` but WITHOUT
    excluding already-picked candidates -- reveals how many of the k
    picks would coincide if forced-uniqueness were not applied."""
    bsz, s, n = scores.shape
    picks = []
    for t in range(k):
        m = t % s
        s_m = scores[:, m, :].masked_fill(~cand_mask, float('-inf'))
        idx_m = s_m.argmax(dim=-1)
        picks.append(idx_m)
    return torch.stack(picks, dim=1)  # [B, k], may contain duplicates


def hard_eval_decomposition(scores, cand_mask, memory_c, offset_c, query_future, d_raw, oracle_idx, top_k=TOP_K):
    model_idx = round_robin_topk_selection(scores, cand_mask, k=top_k)
    model_ind_mse = d_raw.gather(1, model_idx).mean(-1)
    oracle_ind_mse = d_raw.gather(1, oracle_idx).mean(-1)
    recall10 = recall_at_k(model_idx, oracle_idx, top_k)
    ndcg10 = ndcg_at_k(model_idx, d_raw, cand_mask, top_k)
    y_sel = memory_c[model_idx] + offset_c.view(-1, 1, 1)
    e = y_sel - query_future.unsqueeze(1)
    individual_mse_i = (e ** 2).mean(-1)
    D_ = individual_mse_i.mean(-1) / top_k  # PART 0 convention: same D formula as TRACK-J3/K/M
    agg_pred = y_sel.mean(dim=1)
    agg_mse = ((agg_pred - query_future) ** 2).mean(-1)
    C_ = agg_mse - D_
    return dict(model_idx=model_idx, model_ind_mse=model_ind_mse, oracle_ind_mse=oracle_ind_mse,
               recall10=recall10, ndcg10=ndcg10, agg_mse=agg_mse, D=D_, C=C_)


def spearman_batch(mean_score, d_raw, cand_mask):
    """PART 13: Spearman rank correlation between the model's own
    (slot-averaged) score and the true future-MSE relevance ranking
    (-d_raw), computed per query over its valid candidates only, then
    averaged. `scipy.stats.spearmanr`, consistent with this codebase's
    existing usage (`compute_o_aggregation_diagnostic01.py`)."""
    rhos = []
    score_np = mean_score.detach().cpu().numpy()
    rel_np = (-d_raw).detach().cpu().numpy()
    mask_np = cand_mask.detach().cpu().numpy()
    for b in range(score_np.shape[0]):
        m = mask_np[b]
        if m.sum() < 2:
            continue
        rho, _ = spearmanr(score_np[b, m], rel_np[b, m])
        if rho == rho:  # NaN-safe
            rhos.append(rho)
    return float(sum(rhos) / len(rhos)) if rhos else float('nan')


def slot_mechanism_diagnostics(scores, cand_mask, p_m, tau_s):
    """PART 14: mean pairwise slot-distribution cosine, slot Top-1/Top-10
    overlap, unique-candidates-before-forced-uniqueness, per-slot entropy.
    No overlap penalty is used as a LOSS anywhere in this track -- these
    are read-only diagnostics."""
    bsz, s, n = scores.shape
    out = {}
    if s >= 2:
        out['mean_pairwise_slot_cosine'] = float(slot_overlap_penalty(p_m))
    else:
        out['mean_pairwise_slot_cosine'] = float('nan')

    def _pair_overlap(pair_k):
        pair_k = min(pair_k, int(cand_mask.sum(dim=-1).min().item()))
        if s < 2 or pair_k < 1:
            return float('nan')
        tops = []
        for m in range(s):
            s_m = scores[:, m, :].masked_fill(~cand_mask, float('-inf'))
            idx_m = stable_topk_indices(s_m, pair_k, largest=True)
            tops.append(idx_m)
        overlaps = []
        for i in range(s):
            for j in range(i + 1, s):
                match = (tops[i].unsqueeze(-1) == tops[j].unsqueeze(-2)).any(-1).float().sum(-1)
                overlaps.append((match / pair_k).mean())
        return float(torch.stack(overlaps).mean()) if overlaps else float('nan')

    out['slot_top1_overlap'] = _pair_overlap(1)
    out['slot_top10_overlap'] = _pair_overlap(10)

    naive_picks = naive_round_robin_picks(scores, cand_mask, k=TOP_K)
    uniq = torch.tensor([len(set(naive_picks[b].tolist())) for b in range(naive_picks.size(0))],
                        dtype=torch.float32)
    out['unique_before_forced_uniqueness'] = float(uniq.mean())

    p_valid = p_m.clamp_min(EPS)
    ent = -(p_valid * p_valid.log()).sum(dim=-1)  # [B, S]
    out['per_slot_entropy_mean'] = float(ent.mean())
    return out


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
    ap.add_argument('--slot_std', type=float, default=1e-3,
                    help='SlotHeads symmetry-breaking perturbation std -- PART 7, SAME for every num_slots value')
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--train_epochs', type=int, default=10)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--init_seed', type=int, default=0)
    ap.add_argument('--loader_seed', type=int, default=0)
    ap.add_argument('--out_dir', default='results/TRACK-T-PURE-MULTISLOT-VALIDATION01')
    ap.add_argument('--checkpoints', default='checkpoints/track_t_pure_multislot_validation01')
    ap.add_argument('--limit_batches', type=int, default=0, help='SMOKE ONLY')
    ap.add_argument('--smoke_test', action='store_true')
    ap.add_argument('--skip_init_hash_check', action='store_true',
                    help='non-ETTh1_720 settings legitimately produce a different init hash')
    ap.add_argument('--expected_init_hash', default=None,
                    help='if set (and --skip_init_hash_check not given), assert state_hash(model) equals this')
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    arm = f'S{cli.num_slots}'
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    exp, args, model = build_model(cli, device)
    channels = list(range(int(args.enc_in)))
    init_hash = state_hash(model)
    if cli.expected_init_hash and not cli.skip_init_hash_check:
        assert init_hash == cli.expected_init_hash, \
            f'[ISSUE][ABORT] init hash mismatch: {init_hash} != {cli.expected_init_hash}'
    d_model = int(args.d_model)

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
              'init_seed': cli.init_seed, 'loader_seed': cli.loader_seed,
              'n_candidates': int(exp.memory_x.size(0)),
              'checkpoint_criterion': 'min val retmse10 (round-robin Top-10 individual MSE) -- '
                                      'IDENTICAL to Original-KL/J0\'s own criterion, PART 10'}
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2))
    print(f'[track_t] arm={arm} cell={cli.cell} N={config["n_candidates"]} init_hash={init_hash[:16]}')

    def eval_channel(batch_x, batch_y, batch_start_idx, c, want_diag=False):
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        scores = compute_scores_full_grad(model, slot_heads, batch_x, exp.memory_x, c)
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
                scores = compute_scores_full_grad(model, slot_heads, batch_x, exp.memory_x, c)
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
                '[ISSUE] candidate/query encoder grad is zero -- full-gradient violated'
            optimizer.step()
        vram = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
        print(f'[track_t] {arm} SMOKE PASS max_vram_mb={vram:.1f}')
        return

    epoch_val_rows, step_rows, batch_order_hashes = [], [], []
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
                scores = compute_scores_full_grad(model, slot_heads, batch_x, exp.memory_x, c)
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
            step_rows.append({'global_step': global_step, 'epoch': epoch, 'batch_in_epoch': bi,
                              'train_loss': batch_loss})
        batch_order_hashes.append({'epoch': epoch, 'batch_order_sha256': batch_order_sha256(starts)})

        model.eval()
        val_sums, n = {}, 0
        with torch.no_grad():
            for batch_x, batch_y, batch_start_idx in val_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                bsz = batch_x.size(0)
                per_ch = {}
                for c in channels:
                    res, _ = eval_channel(batch_x, batch_y, batch_start_idx, c, want_diag=False)
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
        torch.save(payload, ckpt_dir / f'checkpoint_epoch{epoch}.pth')
        if val_retmse10 < best['val_retmse10']:
            best = {'val_retmse10': val_retmse10, 'epoch': epoch}
            torch.save(payload, ckpt_dir / 'checkpoint.pth')
        print(f'[track_t] {arm} epoch={epoch} val_retmse10={val_retmse10:.6f} val_agg={val_metrics["agg_mse10"]:.6f} '
             f'(best={best["epoch"]}:{best["val_retmse10"]:.6f})')
        if epoch - best['epoch'] >= cli.patience:
            print(f'[track_t] {arm} early stop at epoch {epoch} (best={best["epoch"]})')
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
            bsz = batch_x.size(0)
            per_ch = {}
            for c in channels:
                res, diag = eval_channel(batch_x, batch_y, batch_start_idx, c, want_diag=True)
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

    with open(out_dir / 'train_metrics.csv', 'w', newline='') as fh:
        fieldnames = sorted({k for r in step_rows for k in r})
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in step_rows:
            w.writerow(r)
    with open(out_dir / 'val_metrics.csv', 'w', newline='') as fh:
        fieldnames = sorted({k for r in epoch_val_rows for k in r})
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in epoch_val_rows:
            w.writerow(r)
    with open(out_dir / 'batch_order_hashes.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['epoch', 'batch_order_sha256'])
        w.writeheader()
        for r in batch_order_hashes:
            w.writerow(r)
    (out_dir / 'final_test_metrics.json').write_text(json.dumps(test_metrics, indent=2, default=str))
    (out_dir / 'slot_mechanism_metrics.json').write_text(json.dumps(slot_mech_metrics, indent=2, default=str))
    (out_dir / 'checkpoint_fingerprints.json').write_text(json.dumps(
        {'init_hash': init_hash, 'best_epoch': best['epoch'], 'final_model_hash': state_hash(model),
         'batch_order_hashes': batch_order_hashes}, indent=2))
    vram = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
    print(f'[track_t] done. {arm} best_epoch={best["epoch"]} test_retMSE@10={test_metrics["retmse10"]:.6f} '
         f'test_agg_mse10={test_metrics["agg_mse10"]:.6f} max_vram_mb={vram:.1f} wall_seconds={time.time()-t0:.1f}')


if __name__ == '__main__':
    main()
