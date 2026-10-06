#!/usr/bin/env python3
"""R0 (Raw Cosine) retrieval cache builder, restricted to the SAME
Shared Top-100 candidate pool every other arm in this experiment uses.
No training, no checkpoint -- re-scores the M pooled candidates with
the identical future-blind delta-last-cosine metric that built the pool
in the first place, takes Top-10, builds the Stage-2 cache. Same cache
schema as `build_retrieval_cache_pool01.py` so `train_r_stage2_lambda01.py`
is reused UNMODIFIED for Stage-2.
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
from scripts.train_factorial_e2e01 import individual_utility_memsafe
from scripts.train_j_shared_encoder_drift01 import build_model
from scripts.train_margutil01 import memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k
from scripts.train_retriever_pool01 import CandidatePoolCache
from utils.candidate_pool import (
    compute_coarse_delta_last_cosine_scores, gather_candidate_values, pooled_future_mse,
)

TOP_K = 10


@torch.no_grad()
def build_split(exp, args, pool_cache, split, device, chunk_size=4096):
    channels = list(range(int(args.enc_in)))
    _, loader = exp._get_data(flag=split, shuffle=False)
    all_start, all_R, all_D, all_C, all_Agg, all_Recall, all_NDCG, all_ret = [], [], [], [], [], [], [], []
    all_D_pq, all_C_pq = [], []

    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        bsz = batch_x.size(0)
        rel_out = torch.zeros(bsz, args.pred_len, len(channels), device=device)
        D_pc = torch.zeros(bsz, len(channels), device=device)
        C_pc = torch.zeros(bsz, len(channels), device=device)
        for c in channels:
            pool_idx_global = pool_cache.lookup(batch_start_idx, c, device)  # [B, M]
            coarse_scores_full = compute_coarse_delta_last_cosine_scores(batch_x, exp.memory_x, c)  # [B, N]
            coarse_scores_pool = coarse_scores_full.gather(1, pool_idx_global)  # [B, M] -- re-score within pool
            picks_local = stable_topk_indices(coarse_scores_pool, TOP_K, largest=True)

            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            pooled_memory_c = gather_candidate_values(memory_c, pool_idx_global)
            d_pool = pooled_future_mse(pooled_memory_c, offset_c, query_future)
            valid_mask = torch.ones_like(pool_idx_global, dtype=torch.bool)
            oracle_idx = stable_topk_indices(d_pool, TOP_K, largest=False)

            ind_mse_i = d_pool.gather(1, picks_local)
            y_sel = pooled_memory_c.gather(
                1, picks_local.unsqueeze(-1).expand(-1, -1, pooled_memory_c.size(-1))) + offset_c.view(-1, 1, 1)
            r_c = y_sel.mean(dim=1)
            rel_out[:, :, c] = r_c
            D_ = ind_mse_i.mean(dim=-1) / TOP_K
            agg_mse = ((r_c - query_future) ** 2).mean(dim=-1)
            C_ = agg_mse - D_
            recall = recall_at_k(picks_local, oracle_idx, TOP_K)
            ndcg = ndcg_at_k(picks_local, d_pool, valid_mask, TOP_K)
            all_D.append(D_.cpu()); all_C.append(C_.cpu()); all_Agg.append(agg_mse.cpu())
            all_Recall.append(recall.cpu()); all_NDCG.append(ndcg.cpu())
            all_ret.append(ind_mse_i.mean(dim=-1).cpu())
            D_pc[:, c] = D_
            C_pc[:, c] = C_

        all_start.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx)
                         else torch.as_tensor(batch_start_idx))
        all_R.append(rel_out.cpu())
        all_D_pq.append(D_pc.mean(dim=-1).cpu())
        all_C_pq.append(C_pc.mean(dim=-1).cpu())

    cache = {'query_start_idx': torch.cat(all_start), 'relation_outputs': torch.cat(all_R),
            'D_per_query': torch.cat(all_D_pq), 'C_per_query': torch.cat(all_C_pq),
            'channels': channels, 'pred_len': args.pred_len, 'split': split}
    D_cat, C_cat, Agg_cat = torch.cat(all_D), torch.cat(all_C), torch.cat(all_Agg)
    assert torch.allclose(D_cat + C_cat, Agg_cat, atol=1e-3)
    metrics = {'retmse10': float(torch.cat(all_ret).mean()), 'D': float(D_cat.mean()), 'C': float(C_cat.mean()),
              'agg_mse10': float(Agg_cat.mean()), 'recall10': float(torch.cat(all_Recall).mean()),
              'ndcg10': float(torch.cat(all_NDCG).mean())}
    return cache, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--candidate_mask', default='raft')
    ap.add_argument('--candidate_pool_size', type=int, default=100)
    ap.add_argument('--candidate_pool_cache', required=True)
    ap.add_argument('--out_dir', required=True)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cli.init_seed = cli.seed
    cli.patch_len = 16
    cli.top_k = TOP_K
    cli.batch_size = 32
    exp, args, model = build_model(cli, device)
    exp.memory_x = exp.memory_x.to(device)
    del model  # R0 never uses the learned encoder

    expected_meta = {'seq_len': cli.seq_len, 'pred_len': cli.pred_len, 'candidate_mask': cli.candidate_mask,
                     'candidate_pool_size': cli.candidate_pool_size, 'candidate_pool_metric': 'delta_last_cosine'}
    cache_dir = Path(cli.candidate_pool_cache)
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_metrics = None
    for split in ('train', 'val', 'test'):
        pool_cache = CandidatePoolCache(cache_dir / f'{split}.pt', expected_meta)
        cache, metrics = build_split(exp, args, pool_cache, split, device)
        torch.save(cache, out_dir / f'{split}.pt')
        print(f'[build_r0_cache] {split}: n={cache["query_start_idx"].numel()} '
             f'retmse10={metrics["retmse10"]:.6f} agg={metrics["agg_mse10"]:.6f}')
        if split == 'test':
            test_metrics = metrics
    (out_dir / 'stage1_metrics.json').write_text(json.dumps(test_metrics, indent=2))
    print(f'[build_r0_cache] wrote {out_dir / "stage1_metrics.json"}')


if __name__ == '__main__':
    main()
