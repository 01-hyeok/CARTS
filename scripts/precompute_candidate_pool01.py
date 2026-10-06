#!/usr/bin/env python3
"""Precompute and cache the future-blind, parameter-free delta-last
-cosine coarse Top-M candidate pool for one (dataset, seq_len, pred_len,
candidate_mask) cell, for every channel and every split (train/val/test).

Used by `scripts/train_retriever_pool01.py` / `scripts/build_retrieval_cache_pool01.py`
when `--candidate_pool_mode coarse_topk` -- those scripts FAIL FAST with a
clear message if no matching cache is found, rather than silently
kicking off this (potentially expensive-ish, though far cheaper than a
learned-encoder pass over all N) precompute themselves.

Memory discipline (spec section 12): never materializes the full
`[Q, N]` coarse-score matrix. Queries are processed in chunks
(`--query_chunk_size`), and the per-chunk `[chunk, N]` score matrix is
discarded immediately after its chunk's Top-M is taken.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_margutil01 import build_experiment
from utils.candidate_pool import build_coarse_pool


def _idx_fingerprint(idx_tensor):
    h = hashlib.sha256()
    if torch.is_tensor(idx_tensor):
        idx_tensor = idx_tensor.detach().cpu().numpy()
    h.update(idx_tensor.tobytes())
    return h.hexdigest()


@torch.no_grad()
def precompute_split(exp, args, split, channels, pool_size, device, query_chunk_size):
    _, loader = exp._get_data(flag=split, shuffle=False)
    all_start, all_pool = [], []
    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        bsz = batch_x.size(0)
        pool_this_batch = torch.empty(bsz, len(channels), pool_size, dtype=torch.int64, device=device)
        for start in range(0, bsz, query_chunk_size):
            end = min(start + query_chunk_size, bsz)
            chunk_x = batch_x[start:end]
            chunk_mask = cand_mask[start:end]
            for c in channels:
                from utils.candidate_pool import compute_coarse_delta_last_cosine_scores
                scores = compute_coarse_delta_last_cosine_scores(chunk_x, exp.memory_x, c)
                pool_this_batch[start:end, c, :] = build_coarse_pool(scores, chunk_mask, pool_size)
        all_start.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx)
                         else torch.as_tensor(batch_start_idx))
        all_pool.append(pool_this_batch.cpu())
    query_start_idx = torch.cat(all_start)
    pool_idx_global = torch.cat(all_pool, dim=0)  # [Q, C, M]
    return query_start_idx, pool_idx_global


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--dataset', required=True, help='for metadata only (reference_ckpt determines the real data)')
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--candidate_pool_size', type=int, required=True)
    ap.add_argument('--candidate_pool_metric', default='delta_last_cosine', choices=('delta_last_cosine',))
    ap.add_argument('--candidate_mask', default='raft')
    ap.add_argument('--query_chunk_size', type=int, default=256)
    ap.add_argument('--out_dir', required=True)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.seq_len, 'batch_size': 32, 'seed': 0,
        'relation_encoder_type': 'mlp', 'relation_self_fill': 'linear', 'relation_input_space': 'delta_last',
        'relation_teacher_space': 'delta_last', 'relation_value_space': 'delta_last',
        'candidate_mask': cli.candidate_mask,
    })
    exp._ensure_memory()
    exp.memory_x = exp.memory_x.to(device)
    channels = list(range(int(args.enc_in)))

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    memory_x_fp = _idx_fingerprint(exp.memory_sampler.starts if hasattr(exp.memory_sampler, 'starts')
                                   else torch.arange(exp.memory_x.size(0)))
    for split in ('train', 'val', 'test'):
        query_start_idx, pool_idx_global = precompute_split(
            exp, args, split, channels, cli.candidate_pool_size, device, cli.query_chunk_size)
        query_fp = _idx_fingerprint(query_start_idx)
        meta = {
            'dataset': cli.dataset, 'seq_len': cli.seq_len, 'pred_len': cli.pred_len,
            'candidate_mask': cli.candidate_mask, 'candidate_pool_mode': 'coarse_topk',
            'candidate_pool_size': cli.candidate_pool_size, 'candidate_pool_metric': cli.candidate_pool_metric,
            'split': split, 'channel_count': len(channels),
            'n_memory_candidates': int(exp.memory_x.size(0)),
            'memory_starts_fingerprint': memory_x_fp, 'query_start_idx_fingerprint': query_fp,
        }
        payload = {
            'query_start_idx': query_start_idx,
            'pool_idx_global': pool_idx_global.to(torch.int32) if exp.memory_x.size(0) < 2**31 else pool_idx_global,
            'meta': meta,
        }
        torch.save(payload, out_dir / f'{split}.pt')
        print(f'[precompute_pool] {split}: Q={query_start_idx.numel()} '
             f'pool_idx_global.shape={tuple(pool_idx_global.shape)}')
    (out_dir / 'meta.json').write_text(json.dumps(meta, indent=2, default=str))
    print(f'[precompute_pool] done. wall_seconds={time.time()-t0:.1f}')


if __name__ == '__main__':
    main()
