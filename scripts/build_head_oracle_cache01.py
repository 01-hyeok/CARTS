#!/usr/bin/env python3
"""TRACK-V-CALENDAR-ROUTER01 -- per-head Head Oracle cache. For a FROZEN
best-retMSE V5 checkpoint (encoder + 5 SlotHeads), computes, for every
query/channel, each head's OWN independent Top-10-within-P100 retMSE
(`U_h`, spec section 7 -- individual future-MSE mean, NEVER the
aggregate), the hard oracle label `argmin_h U_h`, and the soft head
-teacher distribution (z-scored `U_h` across heads, `softmax(-z/tau_H)`,
spec section 8). Saves one `.pt` per split; train/val/test are built
identically here, but ONLY train's labels are used for Router training
-- val/test stay diagnostic-only (the calling orchestrator enforces
that, this builder has no opinion on how its output is used).
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.train_j_shared_encoder_drift01 import build_model
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_margutil01 import memory_value
from scripts.train_retriever_pool01 import CandidatePoolCache, compute_scores
from utils.candidate_pool import CandidatePoolConfig, gather_candidate_values, pooled_future_mse

TOP_K = 10
EPS = 1e-8
N_HEADS = 5


@torch.no_grad()
def build_split(exp, args, model, slot_heads, pool_cfg, pool_cache, split, device, tau_h):
    channels = list(range(int(args.enc_in)))
    _, loader = exp._get_data(flag=split, shuffle=False)
    all_start, all_U, all_hard = [], [], []

    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        bsz = batch_x.size(0)
        U_bc = torch.zeros(bsz, len(channels), N_HEADS, device=device)
        for c in channels:
            scores, valid_mask, pool_idx_global = compute_scores(
                pool_cfg, model, slot_heads, batch_x, exp, c, cand_mask, pool_cache, batch_start_idx, device)
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            pooled_memory_c = gather_candidate_values(memory_c, pool_idx_global)
            d_pool = pooled_future_mse(pooled_memory_c, offset_c, query_future)  # [B, M]
            for h in range(N_HEADS):
                top10_h = stable_topk_indices(scores[:, h, :], TOP_K, largest=True)
                U_h = d_pool.gather(1, top10_h).mean(dim=-1)  # [B]
                U_bc[:, c, h] = U_h
        all_start.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx)
                         else torch.as_tensor(batch_start_idx))
        all_U.append(U_bc.cpu())
        all_hard.append(U_bc.argmin(dim=-1).cpu())

    query_start_idx = torch.cat(all_start)
    U = torch.cat(all_U, dim=0)          # [Q, C, 5]
    hard_label = torch.cat(all_hard, dim=0)  # [Q, C]

    mean_u = U.mean(dim=-1, keepdim=True)
    std_u = U.std(dim=-1, keepdim=True).clamp_min(EPS)
    z = (U - mean_u) / std_u
    soft_teacher = torch.softmax(-z / tau_h, dim=-1)  # [Q, C, 5]

    oracle_fraction = torch.stack([(hard_label == h).float().mean() for h in range(N_HEADS)])
    return {
        'query_start_idx': query_start_idx, 'U': U, 'hard_label': hard_label,
        'soft_teacher': soft_teacher, 'channels': channels, 'split': split, 'tau_h': tau_h,
    }, oracle_fraction


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--candidate_mask', default='raft')
    ap.add_argument('--candidate_pool_size', type=int, default=100)
    ap.add_argument('--candidate_pool_cache', required=True)
    ap.add_argument('--v5_checkpoint', required=True, help='checkpoint_best_retmse.pth from train_v_sharedtop100_01.py')
    ap.add_argument('--tau_h', type=float, default=1.0)
    ap.add_argument('--out_dir', required=True)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pool_cfg = CandidatePoolConfig(mode='coarse_topk', size=cli.candidate_pool_size)
    cli.init_seed = 0
    cli.patch_len = 16
    cli.top_k = TOP_K
    cli.batch_size = 32
    exp, args, model = build_model(cli, device)
    d_model = int(args.d_model)
    slot_heads = SlotHeads(d_model, n_slots=N_HEADS).to(device)

    bl = torch.load(cli.v5_checkpoint, map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    slot_heads.load_state_dict(bl['slot_heads_state_dict'])
    model.eval()
    slot_heads.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in slot_heads.parameters():
        p.requires_grad_(False)

    expected_meta = {'seq_len': cli.seq_len, 'pred_len': cli.pred_len, 'candidate_mask': cli.candidate_mask,
                     'candidate_pool_size': cli.candidate_pool_size, 'candidate_pool_metric': 'delta_last_cosine'}
    cache_dir = Path(cli.candidate_pool_cache)
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for split in ('train', 'val', 'test'):
        pool_cache = CandidatePoolCache(cache_dir / f'{split}.pt', expected_meta)
        cache, oracle_fraction = build_split(exp, args, model, slot_heads, pool_cfg, pool_cache, split, device,
                                             cli.tau_h)
        torch.save(cache, out_dir / f'head_oracle_{split}.pt')
        frac = {f'H{h+1}': float(oracle_fraction[h]) for h in range(N_HEADS)}
        print(f'[head_oracle] {split}: n={cache["query_start_idx"].numel()} oracle_fraction={frac}')
        summary[split] = frac
    (out_dir / 'head_oracle_distribution.json').write_text(json.dumps(summary, indent=2))
    print(f'[head_oracle] wrote {out_dir / "head_oracle_distribution.json"}')


if __name__ == '__main__':
    main()
