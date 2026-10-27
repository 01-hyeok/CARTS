#!/usr/bin/env python3
"""TRACK-V5-HEAD-OBJECTIVE-AUDIT01 -- PHASE F. Diagnostic-only gradient
comparison at the EXISTING canonical V5 checkpoint: compares the
gradient actually used in training (`L0 = KL(p_T || p_bar)`, Mean
-Mixture-first) against an UNWEIGHTED per-head-KL alternative
(`L_head = mean_h KL(p_T || p_h)`), on the SAME batches, SAME teacher,
SAME candidate mask. No optimizer step, no checkpoint write, no new
training. Measures, per (batch, channel), each SlotHead's own W_m
gradient norm (the only genuinely per-head parameter -- the encoder
itself is shared across all 5 heads, so "head-specific encoder
gradient" is not a meaningful quantity; the shared ENCODER's total
gradient norm is reported once per loss, not per head) and the cosine
similarity between L0's and L_head's full gradient vector (encoder +
all 5 SlotHeads, flattened).
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob
from scripts.train_j_shared_encoder_drift01 import build_model
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_margutil01 import memory_value
from scripts.train_retriever_pool01 import CandidatePoolCache, compute_scores
from utils.candidate_pool import CandidatePoolConfig, gather_candidate_values, pooled_future_mse

EPS = 1e-8
N_SLOTS = 5


def flat_grad(params):
    gs = [p.grad.flatten() for p in params if p.grad is not None]
    return torch.cat(gs) if gs else torch.zeros(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--patch_len', type=int, default=16)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--tau_t', type=float, default=0.1)
    ap.add_argument('--tau_s', type=float, default=0.1)
    ap.add_argument('--slot_std', type=float, default=1e-3)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--init_seed', type=int, default=0)
    ap.add_argument('--candidate_mask', default='raft')
    ap.add_argument('--candidate_pool_size', type=int, default=100)
    ap.add_argument('--candidate_pool_cache', required=True)
    ap.add_argument('--n_batches', type=int, default=6)
    ap.add_argument('--out_dir', required=True)
    cli = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    exp, args, model = build_model(cli, device)
    bl = torch.load(cli.checkpoint, map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    slot_heads = SlotHeads(int(args.d_model), n_slots=N_SLOTS, std=cli.slot_std).to(device)
    slot_heads.load_state_dict(bl['slot_heads_state_dict'])
    model.train()
    slot_heads.train()
    for p in model.parameters():
        p.requires_grad_(True)
    for p in slot_heads.parameters():
        p.requires_grad_(True)

    pool_cfg = CandidatePoolConfig(mode='coarse_topk', size=cli.candidate_pool_size)
    expected_meta = {'seq_len': cli.seq_len, 'pred_len': cli.pred_len, 'candidate_mask': cli.candidate_mask,
                     'candidate_pool_size': cli.candidate_pool_size, 'candidate_pool_metric': 'delta_last_cosine'}
    pool_cache = CandidatePoolCache(Path(cli.candidate_pool_cache) / 'train.pt', expected_meta)
    _, train_loader = exp._get_data(flag='train', shuffle=False)
    channels = list(range(int(args.enc_in)))

    all_params = list(model.parameters()) + list(slot_heads.parameters())
    head_w_norms_L0 = torch.zeros(N_SLOTS)
    head_w_norms_Lhead = torch.zeros(N_SLOTS)
    encoder_norm_L0 = encoder_norm_Lhead = 0.0
    cos_sum = 0.0
    n_pairs = 0

    for bi, (batch_x, batch_y, batch_start_idx) in enumerate(train_loader):
        if bi >= cli.n_batches:
            break
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        for c in channels:
            scores, valid_mask, pool_idx_global = compute_scores(
                pool_cfg, model, slot_heads, batch_x, exp, c, None, pool_cache, batch_start_idx, device)
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            pooled_memory_c = gather_candidate_values(memory_c, pool_idx_global)
            with torch.no_grad():
                d_pool = pooled_future_mse(pooled_memory_c, offset_c, query_future)
                p_t = normalized_teacher_prob(d_pool, valid_mask, cli.tau_t).detach()

            s_masked = scores.masked_fill(~valid_mask.unsqueeze(1), float('-inf'))
            p_h = torch.softmax(s_masked / cli.tau_s, dim=-1)  # [B,5,M]
            p_bar = p_h.mean(dim=1)
            log_p_bar = torch.log(p_bar.clamp_min(EPS))
            loss_mix = (p_t * (torch.log(p_t.clamp_min(EPS)) - log_p_bar)).sum(-1).mean()

            model.zero_grad(set_to_none=True)
            slot_heads.zero_grad(set_to_none=True)
            loss_mix.backward(retain_graph=True)
            g0 = flat_grad(all_params).clone()
            for h in range(N_SLOTS):
                if slot_heads.W.grad is not None:
                    head_w_norms_L0[h] += float(slot_heads.W.grad[h].norm())
            encoder_norm_L0 += float(flat_grad(model.parameters()).norm())

            log_p_h = torch.log(p_h.clamp_min(EPS))
            kl_per_h = (p_t.unsqueeze(1) * (torch.log(p_t.clamp_min(EPS)).unsqueeze(1) - log_p_h)).sum(-1)  # [B,5]
            loss_head = kl_per_h.mean()

            model.zero_grad(set_to_none=True)
            slot_heads.zero_grad(set_to_none=True)
            loss_head.backward()
            g1 = flat_grad(all_params).clone()
            for h in range(N_SLOTS):
                if slot_heads.W.grad is not None:
                    head_w_norms_Lhead[h] += float(slot_heads.W.grad[h].norm())
            encoder_norm_Lhead += float(flat_grad(model.parameters()).norm())

            cos = float(torch.nn.functional.cosine_similarity(g0.unsqueeze(0), g1.unsqueeze(0)).item())
            cos_sum += cos
            n_pairs += 1

    report = {
        'cell': cli.cell, 'n_query_channel_pairs': n_pairs,
        'mean_gradient_cosine_L0_vs_Lhead': cos_sum / max(n_pairs, 1),
        'mean_encoder_grad_norm_L0': encoder_norm_L0 / max(n_pairs, 1),
        'mean_encoder_grad_norm_Lhead': encoder_norm_Lhead / max(n_pairs, 1),
        'mean_head_W_grad_norm_L0': (head_w_norms_L0 / max(n_pairs, 1)).tolist(),
        'mean_head_W_grad_norm_Lhead': (head_w_norms_Lhead / max(n_pairs, 1)).tolist(),
        'head_W_grad_norm_ratio_Lhead_over_L0': (
            (head_w_norms_Lhead / max(n_pairs, 1)) / (head_w_norms_L0 / max(n_pairs, 1)).clamp_min(EPS)
        ).tolist(),
    }
    out_dir = Path(cli.out_dir) / cli.cell
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'phase_f_gradient.json').write_text(json.dumps(report, indent=2, default=str))
    print(f'[diag_v5_grad] {cli.cell} DONE n_pairs={n_pairs} '
         f'cos(L0,Lhead)={report["mean_gradient_cosine_L0_vs_Lhead"]:.4f} '
         f'encoder_norm L0={report["mean_encoder_grad_norm_L0"]:.4f} Lhead={report["mean_encoder_grad_norm_Lhead"]:.4f}')


if __name__ == '__main__':
    main()
