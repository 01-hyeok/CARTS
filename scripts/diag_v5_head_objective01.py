#!/usr/bin/env python3
"""TRACK-V5-HEAD-OBJECTIVE-AUDIT01 -- diagnostic-only script. NO new
training, NO checkpoint writes. Loads the EXISTING canonical sign-fixed
V5 checkpoint (`checkpoints/track_v_sharedtop100_signfix01/...`) and the
EXISTING Shared-P100 pool caches, and computes, in one forward pass per
split, everything PHASE B (reproduction/correctness), PHASE C (per-head
+ true Mean-Mixture quality), PHASE D (head-level error complementarity),
PHASE E2/E3 (31-subset head oracle), and PHASE G (soft/hard alignment)
need -- without ever re-running the model more than once per split.

Reused UNMODIFIED: `compute_scores`/`CandidatePoolCache`
(`train_retriever_pool01.py`), `build_model`/`state_hash`
(`train_j_shared_encoder_drift01.py`), `SlotHeads`
(`train_k_multislot_predictive_retrieval01.py`), `memory_value`
(`train_margutil01.py`), `gather_candidate_values`/`pooled_future_mse`
(`utils/candidate_pool.py`), `normalized_teacher_prob`
(`train_horizon_retrieval_expert01.py`), `mean_mixture_topk_selection`
(`utils/mean_mixture_selection.py`), `per_head_standalone_topk`/
`per_head_future_utility`/`kl_per_head`/`head_pairwise_overlap`
(`utils/expert_head_metrics.py`), `recall_at_k`/`ndcg_at_k`
(`train_patch_retrieval_expert01.py`), `stable_topk_indices`
(`models/RelationStage1.py`).

Verification mode (`--mode verify`): PHASE B2/B3 numeric checks on a
couple of batches, with hard asserts (abort, not warn, on failure).
Full-pass mode (`--mode full_pass`): PHASE C/D/E2/G per-(query,channel)
CSV dump, one row per query x channel, for the requested split.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob
from scripts.train_j_shared_encoder_drift01 import build_model, state_hash
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads, kl_loss_from_prob
from scripts.train_margutil01 import memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k
from scripts.train_retriever_pool01 import CandidatePoolCache, compute_scores
from utils.candidate_pool import CandidatePoolConfig, gather_candidate_values, pooled_future_mse
from utils.expert_head_metrics import head_pairwise_overlap, kl_per_head, per_head_future_utility, \
    per_head_standalone_topk
from utils.mean_mixture_selection import mean_mixture_topk_selection

EPS = 1e-8
TOP_K = 10
N_SLOTS = 5
ALL_HEADS = list(range(N_SLOTS))
ALL_SUBSETS = [c for r in range(1, N_SLOTS + 1) for c in itertools.combinations(ALL_HEADS, r)]  # 31


def load_v5(cli, device):
    exp, args, model = build_model(cli, device)
    bl = torch.load(cli.checkpoint, map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    slot_heads = SlotHeads(int(args.d_model), n_slots=N_SLOTS, std=cli.slot_std).to(device)
    slot_heads.load_state_dict(bl['slot_heads_state_dict'])
    model.eval()
    slot_heads.eval()
    return exp, args, model, slot_heads, bl


def get_pool_cache(cli, split):
    expected_meta = {'seq_len': cli.seq_len, 'pred_len': cli.pred_len, 'candidate_mask': cli.candidate_mask,
                     'candidate_pool_size': cli.candidate_pool_size, 'candidate_pool_metric': 'delta_last_cosine'}
    return CandidatePoolCache(Path(cli.candidate_pool_cache) / f'{split}.pt', expected_meta)


def subset_agg_mse(p_heads, subset, pooled_memory_c, offset_c, query_future, top_k=10):
    """p_heads: [B,5,M]. Returns (agg_mse[B], R_A[B,Horizon], idx_A[B,k])."""
    p_a = p_heads[:, list(subset), :].mean(dim=1)  # [B,M]
    idx_a = stable_topk_indices(p_a, top_k, largest=True)
    y_sel = pooled_memory_c.gather(1, idx_a.unsqueeze(-1).expand(-1, -1, pooled_memory_c.size(-1))) \
        + offset_c.view(-1, 1, 1)
    r_a = y_sel.mean(dim=1)
    agg_mse = ((r_a - query_future) ** 2).mean(dim=-1)
    return agg_mse, r_a, idx_a


def run_verify(cli, device):
    exp, args, model, slot_heads, bl = load_v5(cli, device)
    pool_cfg = CandidatePoolConfig(mode='coarse_topk', size=cli.candidate_pool_size)
    pool_cache = get_pool_cache(cli, 'test')
    channels = list(range(int(args.enc_in)))
    _, test_loader = exp._get_data(flag='test', shuffle=False)
    report = {'cell': cli.cell, 'checkpoint': cli.checkpoint, 'best_epoch': int(bl['epoch']),
             'init_hash_check_model_state_hash': state_hash(model)[:16]}
    n_checked = 0
    max_abs_diff_topk_vs_manual = 0.0
    rr_vs_mean_overlap_fracs = []
    for bi, (batch_x, batch_y, batch_start_idx) in enumerate(test_loader):
        if bi >= cli.n_verify_batches:
            break
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        for c in channels[:cli.n_verify_channels]:
            with torch.no_grad():
                scores, valid_mask, pool_idx_global = compute_scores(
                    pool_cfg, model, slot_heads, batch_x, exp, c, None, pool_cache, batch_start_idx, device)
                memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
                query_future = batch_y[:, :, c]
                pooled_memory_c = gather_candidate_values(memory_c, pool_idx_global)
                d_pool = pooled_future_mse(pooled_memory_c, offset_c, query_future)

                # ---- B2: Mean-Mixture manual recomputation vs canonical ----
                picks, p_heads, p_mix = mean_mixture_topk_selection(scores, valid_mask, cli.tau_s, cli.top_k)
                s_masked = scores.masked_fill(~valid_mask.unsqueeze(1), float('-inf'))
                p_heads_manual = torch.softmax(s_masked / cli.tau_s, dim=-1)
                p_mix_manual = p_heads_manual.mean(dim=1)
                assert torch.allclose(p_heads, p_heads_manual, atol=1e-6), '[ISSUE][ABORT] p_heads mismatch'
                assert torch.allclose(p_mix, p_mix_manual, atol=1e-6), '[ISSUE][ABORT] p_mix mismatch'
                row_sums = p_mix.sum(dim=-1)
                assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4), \
                    f'[ISSUE][ABORT] p_mix rows do not sum to 1: {row_sums.min()}..{row_sums.max()}'
                assert bool(valid_mask.all()), '[ISSUE][ABORT] Shared-P100 pool_valid_mask expected all-True by construction'
                assert scores.size(1) == N_SLOTS, f'[ISSUE][ABORT] expected 5 heads, got {scores.size(1)}'
                picks_manual = stable_topk_indices(p_mix_manual.masked_fill(~valid_mask, float('-inf')),
                                                   cli.top_k, largest=True)
                assert torch.equal(picks, picks_manual), '[ISSUE][ABORT] Top-10(p_mix) picks differ from manual stable-topk recomputation'
                max_abs_diff_topk_vs_manual = max(max_abs_diff_topk_vs_manual,
                                                  float((p_mix - p_mix_manual).abs().max()))

                # ---- B2: confirm Mean-Mixture selection is NOT round-robin ----
                from scripts.train_t_pure_multislot01 import round_robin_topk_selection
                rr_idx = round_robin_topk_selection(scores, valid_mask, k=cli.top_k)
                overlap = (picks.unsqueeze(-1) == rr_idx.unsqueeze(-2)).any(-1).float().mean(-1)
                rr_vs_mean_overlap_fracs.append(float(overlap.mean()))

                # ---- B3: future alignment ----
                recon_offset_check = offset_c  # query_offset per memory_value's own delta_last convention
                assert pooled_memory_c.shape[-1] == query_future.shape[-1], \
                    '[ISSUE][ABORT] candidate future horizon != query future horizon'
                # candidate absolute future for the TOP pick must be finite and same scale as query_future
                top1_future = pooled_memory_c.gather(
                    1, picks[:, :1].unsqueeze(-1).expand(-1, -1, pooled_memory_c.size(-1))
                ).squeeze(1) + offset_c.view(-1, 1)
                assert torch.isfinite(top1_future).all(), '[ISSUE][ABORT] non-finite reconstructed candidate future'
                scale_ratio = float((top1_future.std() / query_future.std().clamp_min(EPS)))
                report.setdefault('b3_future_scale_ratio_samples', []).append(scale_ratio)
                n_checked += 1
    report['n_query_channel_pairs_checked'] = n_checked
    report['max_abs_diff_p_mix_vs_manual'] = max_abs_diff_topk_vs_manual
    report['mean_mixture_vs_round_robin_top10_overlap_fraction'] = float(np.mean(rr_vs_mean_overlap_fracs))
    report['verdict'] = 'PASS -- all B2/B3 asserts held'
    out_dir = Path(cli.out_dir) / cli.cell
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'phase_b_verify.json').write_text(json.dumps(report, indent=2, default=str))
    print(f'[diag_v5] {cli.cell} PHASE B verify PASS. '
         f'mean_mixture_vs_round_robin_overlap={report["mean_mixture_vs_round_robin_top10_overlap_fraction"]:.4f} '
         f'max_abs_diff={max_abs_diff_topk_vs_manual:.2e}')


def run_full_pass(cli, device):
    exp, args, model, slot_heads, bl = load_v5(cli, device)
    pool_cfg = CandidatePoolConfig(mode='coarse_topk', size=cli.candidate_pool_size)
    pool_cache = get_pool_cache(cli, cli.split)
    channels = list(range(int(args.enc_in)))
    _, loader = exp._get_data(flag=cli.split, shuffle=False)

    pair_names = [f'H{m+1}-H{n+1}' for m, n in itertools.combinations(ALL_HEADS, 2)]
    subset_names = [''.join(str(h + 1) for h in s) for s in ALL_SUBSETS]
    rows = []
    t0 = time.time()
    n_batches = 0
    with torch.no_grad():
        for bi, (batch_x, batch_y, batch_start_idx) in enumerate(loader):
            if cli.limit_batches and bi >= cli.limit_batches:
                break
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            for c in channels:
                scores, valid_mask, pool_idx_global = compute_scores(
                    pool_cfg, model, slot_heads, batch_x, exp, c, None, pool_cache, batch_start_idx, device)
                memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
                query_future = batch_y[:, :, c]
                pooled_memory_c = gather_candidate_values(memory_c, pool_idx_global)
                d_pool = pooled_future_mse(pooled_memory_c, offset_c, query_future)
                oracle_idx = stable_topk_indices(d_pool, cli.top_k, largest=False)

                s_masked = scores.masked_fill(~valid_mask.unsqueeze(1), float('-inf'))
                p_heads = torch.softmax(s_masked / cli.tau_s, dim=-1)  # [B,5,M]

                # ---- PHASE C: per-head standalone quality + R_m (needed for PHASE D) ----
                head_topk_idx = per_head_standalone_topk(scores, valid_mask, k=cli.top_k)  # [B,5,k]
                bsz = scores.size(0)
                R_m = torch.zeros(bsz, N_SLOTS, query_future.size(-1), device=device)
                head_row = {}
                for h in ALL_HEADS:
                    idx_h = head_topk_idx[:, h, :]
                    ind_mse_i = d_pool.gather(1, idx_h)
                    y_sel = pooled_memory_c.gather(1, idx_h.unsqueeze(-1).expand(-1, -1, pooled_memory_c.size(-1))) \
                        + offset_c.view(-1, 1, 1)
                    r_h = y_sel.mean(dim=1)
                    R_m[:, h, :] = r_h
                    agg_mse_h = ((r_h - query_future) ** 2).mean(dim=-1)
                    D_h = ind_mse_i.mean(dim=-1) / cli.top_k
                    C_h = agg_mse_h - D_h
                    head_row[f'retmse10_H{h+1}'] = ind_mse_i.mean(dim=-1).cpu().numpy()
                    head_row[f'aggmse_H{h+1}'] = agg_mse_h.cpu().numpy()
                    head_row[f'D_H{h+1}'] = D_h.cpu().numpy()
                    head_row[f'C_H{h+1}'] = C_h.cpu().numpy()
                    head_row[f'recall10_H{h+1}'] = recall_at_k(idx_h, oracle_idx, cli.top_k).cpu().numpy()
                    head_row[f'ndcg10_H{h+1}'] = ndcg_at_k(idx_h, d_pool, valid_mask, cli.top_k).cpu().numpy()

                # ---- PHASE C: TRUE Mean-Mixture (NOT round-robin) ----
                mean_idx, _, p_mix = mean_mixture_topk_selection(scores, valid_mask, cli.tau_s, cli.top_k)
                y_sel_mix = pooled_memory_c.gather(1, mean_idx.unsqueeze(-1).expand(-1, -1, pooled_memory_c.size(-1))) \
                    + offset_c.view(-1, 1, 1)
                r_mix = y_sel_mix.mean(dim=1)
                ind_mse_mix = d_pool.gather(1, mean_idx)
                aggmse_mix = ((r_mix - query_future) ** 2).mean(dim=-1)
                D_mix = ind_mse_mix.mean(dim=-1) / cli.top_k
                C_mix = aggmse_mix - D_mix
                # independent cross-check of the D+C=Agg identity via direct pairwise sum
                e_sel = y_sel_mix - query_future.unsqueeze(1)  # [B,k,H]
                cross = torch.einsum('bih,bjh->bij', e_sel, e_sel) / e_sel.size(-1)
                c_direct = (cross.sum(dim=(-1, -2)) - cross.diagonal(dim1=-2, dim2=-1).sum(-1)) / (cli.top_k ** 2)
                assert torch.allclose(C_mix, c_direct, atol=1e-4), \
                    f'[ISSUE][ABORT] D+C identity failed: max diff {(C_mix - c_direct).abs().max()}'

                # ---- PHASE D: head-level error complementarity ----
                e_m = R_m - query_future.unsqueeze(1)  # [B,5,H]
                dot_mn = torch.einsum('bmh,bnh->bmn', e_m, e_m) / e_m.size(-1)  # [B,5,5]
                norm_m = e_m.norm(dim=-1)  # [B,5]
                pair_vals_dot, pair_vals_cos = {}, {}
                for (m, n), name in zip(itertools.combinations(ALL_HEADS, 2), pair_names):
                    d_ = dot_mn[:, m, n]
                    cos_ = d_ / (norm_m[:, m] * norm_m[:, n] + EPS)
                    pair_vals_dot[f'dot_{name}'] = d_.cpu().numpy()
                    pair_vals_cos[f'cos_{name}'] = cos_.cpu().numpy()
                overlaps, union_size = head_pairwise_overlap(head_topk_idx)
                overlap_row = {f'overlap_{pair_names[i]}': v.cpu().numpy()
                              for i, v in enumerate(overlaps.values())}

                # Within-Mean-Mixture-Set complementarity (already = C_mix, recorded explicitly for Section D3)
                c_within_mix = C_mix.cpu().numpy()

                # ---- PHASE E2/E3: 31-subset head oracle ----
                subset_row = {}
                for subset, sname in zip(ALL_SUBSETS, subset_names):
                    agg_mse_a, _, _ = subset_agg_mse(p_heads, subset, pooled_memory_c, offset_c, query_future,
                                                     cli.top_k)
                    subset_row[f'subset_aggmse_{sname}'] = agg_mse_a.cpu().numpy()

                # ---- PHASE G: soft vs hard alignment (full-pool soft expectation vs Mean-Mixture hard top-10) ----
                r_soft = torch.einsum('bm,bmh->bh', p_mix, pooled_memory_c) + offset_c.view(-1, 1)
                aggmse_soft = ((r_soft - query_future) ** 2).mean(dim=-1)

                base_cols = {
                    'query_start_idx': batch_start_idx.cpu().numpy() if torch.is_tensor(batch_start_idx)
                    else np.asarray(batch_start_idx), 'channel': np.full(bsz, c),
                    'oracle_ind_mse': d_pool.gather(1, oracle_idx).mean(-1).cpu().numpy(),
                    'retmse10_mix': ind_mse_mix.mean(dim=-1).cpu().numpy(), 'aggmse_mix': aggmse_mix.cpu().numpy(),
                    'D_mix': D_mix.cpu().numpy(), 'C_mix': c_within_mix, 'C_mix_subtraction': C_mix.cpu().numpy(),
                    'recall10_mix': recall_at_k(mean_idx, oracle_idx, cli.top_k).cpu().numpy(),
                    'ndcg10_mix': ndcg_at_k(mean_idx, d_pool, valid_mask, cli.top_k).cpu().numpy(),
                    'aggmse_soft': aggmse_soft.cpu().numpy(), 'union_size': union_size.cpu().numpy(),
                }
                df_cols = {**base_cols, **head_row, **pair_vals_dot, **pair_vals_cos, **overlap_row, **subset_row}
                rows.append(df_cols)
            n_batches += 1

    import pandas as pd
    all_dfs = [pd.DataFrame(r) for r in rows]
    df = pd.concat(all_dfs, ignore_index=True)
    out_dir = Path(cli.out_dir) / cli.cell
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'full_pass_{cli.split}.parquet'
    df.to_parquet(out_path)
    meta = {'cell': cli.cell, 'split': cli.split, 'checkpoint': cli.checkpoint, 'best_epoch': int(bl['epoch']),
           'n_query_channel_rows': len(df), 'n_batches': n_batches, 'wall_seconds': time.time() - t0,
           'subset_names': subset_names, 'pair_names': pair_names, 'tau_s': cli.tau_s}
    (out_dir / f'full_pass_{cli.split}_meta.json').write_text(json.dumps(meta, indent=2, default=str))
    print(f'[diag_v5] {cli.cell} [{cli.split}] full_pass DONE rows={len(df)} wall={time.time()-t0:.1f}s -> {out_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=('verify', 'full_pass'))
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--split', default='test', choices=('train', 'val', 'test'))
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--patch_len', type=int, default=16)
    ap.add_argument('--top_k', type=int, default=TOP_K)
    ap.add_argument('--tau_s', type=float, default=0.1)
    ap.add_argument('--slot_std', type=float, default=1e-3)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--init_seed', type=int, default=0)
    ap.add_argument('--candidate_mask', default='raft')
    ap.add_argument('--candidate_pool_size', type=int, default=100)
    ap.add_argument('--candidate_pool_cache', required=True)
    ap.add_argument('--n_verify_batches', type=int, default=2)
    ap.add_argument('--n_verify_channels', type=int, default=3)
    ap.add_argument('--limit_batches', type=int, default=0)
    ap.add_argument('--out_dir', required=True)
    cli = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if cli.mode == 'verify':
        run_verify(cli, device)
    else:
        run_full_pass(cli, device)


if __name__ == '__main__':
    main()
