#!/usr/bin/env python3
"""TRACK-AGGREGATE-ROUTER-FROZEN01 -- precomputes and caches, per split
(train/val/test), the ONLY two things the Router needs: `z_qc`
(frozen backbone's past-only query representation per channel,
`encode_raw_channel_first` output, BEFORE the per-head SlotHeads
projection -- i.e. head-agnostic) and `U_agg`/`U_ind` (per query,
channel, head, from `utils.head_utility_alignment_correction01`,
reused unmodified). Since the backbone (encoder + SlotHeads) is frozen
throughout Phase 2 (spec section 1), these quantities never change
across router-training epochs, so caching them ONCE makes router
training itself (a tiny MLP) nearly free -- no repeated backbone
forward passes.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_j_shared_encoder_drift01 import build_model
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_retriever_pool01 import CandidatePoolCache
from utils.full_candidate_bank import encode_raw_channel_first
from utils.head_utility_alignment_correction01 import per_query_channel_head_quantities

NUM_SLOTS = 5
TOP_K = 10


@torch.no_grad()
def compute_z_qc(model, channels, device, loader):
    """Returns z_qc: [N, C, d_model] -- head-agnostic frozen past
    representation, one vector per (query, channel)."""
    all_z = []
    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        z_c = [encode_raw_channel_first(model, batch_x, c) for c in channels]  # each [B, d]
        all_z.append(torch.stack(z_c, dim=1).cpu())  # [B, C, d]
    return torch.cat(all_z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--ds', required=True)
    ap.add_argument('--h', type=int, required=True)
    ap.add_argument('--backbone', required=True, choices=('V5', 'Hard'))
    ap.add_argument('--retriever_checkpoint', required=True)
    ap.add_argument('--candidate_pool_cache', required=True)
    ap.add_argument('--candidate_pool_size', type=int, default=100)
    ap.add_argument('--candidate_mask', default='raft')
    ap.add_argument('--patch_len', type=int, default=16)
    ap.add_argument('--tau_s', type=float, required=True)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', required=True)
    cli = ap.parse_args()
    cli.init_seed = cli.seed
    cli.pred_len = cli.h
    cli.seq_len = cli.h
    cli.top_k = TOP_K

    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp, args, model = build_model(cli, device)
    channels = list(range(int(args.enc_in)))
    d_model = int(args.d_model)

    bl = torch.load(cli.retriever_checkpoint, map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    slot_heads = SlotHeads(d_model, n_slots=NUM_SLOTS).to(device)
    slot_heads.load_state_dict(bl['slot_heads_state_dict'])
    model.eval(); slot_heads.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in slot_heads.parameters():
        p.requires_grad_(False)

    expected_meta = {'seq_len': cli.seq_len, 'pred_len': cli.pred_len, 'candidate_mask': cli.candidate_mask,
                     'candidate_pool_size': cli.candidate_pool_size, 'candidate_pool_metric': 'delta_last_cosine'}
    cache_dir = Path(cli.candidate_pool_cache)
    pool_caches = {s: CandidatePoolCache(cache_dir / f'{s}.pt', expected_meta) for s in ('train', 'val', 'test')}

    for split in ('train', 'val', 'test'):
        _, loader = exp._get_data(flag=split, shuffle=False)
        z_qc = compute_z_qc(model, channels, device, loader)
        util = per_query_channel_head_quantities(model, slot_heads, exp, args, channels, device, loader,
                                                  pool_caches[split])
        assert z_qc.size(0) == util['U_agg'].size(0), '[ISSUE][ABORT] z_qc/U_agg query-count mismatch'
        torch.save({'z_qc': z_qc, 'U_ind': util['U_ind'], 'U_agg': util['U_agg'],
                   'query_start_idx': util['query_start_idx']}, out_dir / f'{split}.pt')
        print(f'[precompute_router_features] {cli.ds}_{cli.h}/{cli.backbone}/{split}: '
             f'N={z_qc.size(0)} C={z_qc.size(1)} d={z_qc.size(2)}')

    (out_dir / 'config.json').write_text(json.dumps({
        'ds': cli.ds, 'h': cli.h, 'backbone': cli.backbone, 'd_model': d_model, 'num_channels': len(channels),
        'retriever_checkpoint': cli.retriever_checkpoint, 'wall_seconds': time.time() - t0,
    }, indent=2))
    print(f'[precompute_router_features] done. wall_seconds={time.time()-t0:.1f}')


if __name__ == '__main__':
    main()
