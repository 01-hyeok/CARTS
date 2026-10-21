#!/usr/bin/env python3
"""P100-restricted counterpart of `build_r_retrieval_cache01.py`, for
the `cosine` (Raw Cosine, no training) and `original_kl` retrievers
only (the two this ad-hoc follow-up needs; `j1`/`m2` are out of
scope). The original script scores every candidate in the FULL memory
bank (`exp.memory_x`, restricted only by the future-blind `cand_mask`)
-- this script restricts scoring to the SAME fixed Shared-Top-100 (P100)
pool the main TRACK-P100-SOFTTOPK-QUALITY-AWARE01/
TRACK-P100-QA-CAUSAL-VALIDATION01 tracks use
(`precompute_candidate_pool01.py` output, read via
`CandidatePoolCache`), so these baselines are directly comparable
(same 100-candidate search space) rather than Full-memory-vs-P100
apples-to-oranges.

`original_kl`'s encoder is still trained on the FULL candidate universe
by `train_j_shared_encoder_drift01.py` (Stage1 training itself is
unaffected by this change -- P100 restriction is a SCORING-time
choice, not a training-time one, exactly like how the main SoftTopK E2E
trainers restrict `compute_scores` to `coarse_topk` mode without
retraining the encoder from scratch for each candidate-pool choice).
`cosine` needs no training at all, as before.

Output cache schema is IDENTICAL to the original script
(`{train,val,test}.pt`: `query_start_idx`, `relation_outputs`) so
`train_r_stage2_lambda01.py` consumes it unmodified.
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
from scripts.train_factorial_e2e01 import encode_raw
from scripts.train_j_shared_encoder_drift01 import build_model
from scripts.train_margutil01 import memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k
from scripts.train_retriever_pool01 import CandidatePoolCache
from utils.candidate_pool import (
    encode_pooled_candidates, gather_candidate_histories, gather_candidate_values, pooled_future_mse,
)

TOP_K = 10


def p100_cosine_scores(batch_x, pooled_x, c):
    """`pooled_x`: [B, M, L, C] (the P100-gathered candidate histories).
    Raw delta-last cosine, same convention as the Full-memory version,
    restricted to the M=100 pooled candidates."""
    q = batch_x[:, :, c] - batch_x[:, -1:, c]  # [B, L]
    k = pooled_x[:, :, :, c] - pooled_x[:, :, -1:, c]  # [B, M, L]
    q = torch.nn.functional.normalize(q, dim=-1)
    k = torch.nn.functional.normalize(k, dim=-1)
    return torch.einsum('bl,bml->bm', q, k)  # [B, M]


@torch.no_grad()
def build_split(retriever, exp, args, model, pool_cache, split, device):
    channels = list(range(int(args.enc_in)))
    _, loader = exp._get_data(flag=split, shuffle=False)
    all_start, all_R, all_D, all_C, all_Agg, all_Recall, all_NDCG, all_ret = [], [], [], [], [], [], [], []

    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        bsz = batch_x.size(0)
        rel_out = torch.zeros(bsz, args.pred_len, len(channels), device=device)
        for c in channels:
            pool_idx_global = pool_cache.lookup(batch_start_idx, c, device)  # [B, M]
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            pooled_memory_c = gather_candidate_values(memory_c, pool_idx_global)  # [B, M, H]
            query_future = batch_y[:, :, c]
            d_pool = pooled_future_mse(pooled_memory_c, offset_c, query_future)  # [B, M]

            if retriever == 'cosine':
                pooled_x = gather_candidate_histories(exp.memory_x, pool_idx_global)  # [B, M, L, C]
                s = p100_cosine_scores(batch_x, pooled_x, c)
            else:  # original_kl
                z_q = encode_raw(model, batch_x, c)
                pooled_x = gather_candidate_histories(exp.memory_x, pool_idx_global)
                z_k = encode_pooled_candidates(lambda x, cc: encode_raw(model, x, cc), pooled_x, c)
                # z_k is [B, M, D] (per-query candidates, unlike arm_score's [N, D]
                # full-memory assumption) -- same cosine convention, batched manually.
                z_q_n = torch.nn.functional.normalize(z_q, dim=-1)
                z_k_n = torch.nn.functional.normalize(z_k, dim=-1)
                s = torch.einsum('bd,bmd->bm', z_q_n, z_k_n)

            picks_local = stable_topk_indices(s, TOP_K, largest=True)  # [B, K], positions in [0, M)
            ind_mse_i = d_pool.gather(1, picks_local)
            y_sel = pooled_memory_c.gather(
                1, picks_local.unsqueeze(-1).expand(-1, -1, pooled_memory_c.size(-1))) + offset_c.view(-1, 1, 1)
            r_c = y_sel.mean(dim=1)
            rel_out[:, :, c] = r_c

            oracle_idx = stable_topk_indices(d_pool, TOP_K, largest=False)  # best-within-the-100, not global
            D_ = ind_mse_i.mean(dim=-1) / TOP_K
            agg_mse = ((r_c - query_future) ** 2).mean(dim=-1)
            C_ = agg_mse - D_
            valid_mask = torch.ones_like(d_pool, dtype=torch.bool)
            recall = recall_at_k(picks_local, oracle_idx, TOP_K)
            ndcg = ndcg_at_k(picks_local, d_pool, valid_mask, TOP_K)
            all_D.append(D_.cpu()); all_C.append(C_.cpu()); all_Agg.append(agg_mse.cpu())
            all_Recall.append(recall.cpu()); all_NDCG.append(ndcg.cpu())
            all_ret.append((ind_mse_i.mean(dim=-1)).cpu())

        all_start.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx)
                         else torch.as_tensor(batch_start_idx))
        all_R.append(rel_out.cpu())

    cache = {'query_start_idx': torch.cat(all_start), 'relation_outputs': torch.cat(all_R),
            'channels': channels, 'pred_len': args.pred_len, 'retriever': retriever, 'split': split,
            'candidate_pool_restricted': True, 'candidate_pool_size': TOP_K * 10}
    D_cat, C_cat, Agg_cat = torch.cat(all_D), torch.cat(all_C), torch.cat(all_Agg)
    assert torch.allclose(D_cat + C_cat, Agg_cat, atol=1e-3)
    metrics = {'retmse10': float(torch.cat(all_ret).mean()), 'D': float(D_cat.mean()),
              'C': float(C_cat.mean()), 'agg_mse10': float(Agg_cat.mean()),
              'recall10': float(torch.cat(all_Recall).mean()), 'ndcg10': float(torch.cat(all_NDCG).mean())}
    return cache, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--retriever', required=True, choices=('cosine', 'original_kl'))
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--retriever_checkpoint', default=None, help='required for original_kl')
    ap.add_argument('--candidate_pool_cache', required=True,
                    help='directory with train.pt/val.pt/test.pt from precompute_candidate_pool01.py')
    ap.add_argument('--candidate_pool_size', type=int, default=100)
    ap.add_argument('--out_dir', required=True)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cli.cell = f'_p100_cache_{cli.seed}'
    cli.num_query_views = 0
    cli.init_seed = cli.seed
    cli.loader_seed = cli.seed
    cli.slot_std = 1e-3
    cli.batch_size = 32
    cli.top_k = TOP_K
    cli.tau_t = 0.1
    cli.tau_s = 0.1
    cli.patch_len = 16
    exp, args, model = build_model(cli, device)

    if cli.retriever == 'original_kl':
        if not cli.retriever_checkpoint:
            raise SystemExit('[ISSUE][ABORT] --retriever_checkpoint required for original_kl')
        sd = torch.load(cli.retriever_checkpoint, map_location=device)
        model.load_state_dict(sd['model_state_dict'] if 'model_state_dict' in sd else sd)
    model.eval()

    expected_meta = {'seq_len': cli.seq_len, 'pred_len': cli.pred_len, 'candidate_mask': 'raft',
                     'candidate_pool_size': cli.candidate_pool_size, 'candidate_pool_metric': 'delta_last_cosine'}
    cache_dir = Path(cli.candidate_pool_cache)
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    for split in ('train', 'val', 'test'):
        pool_cache = CandidatePoolCache(cache_dir / f'{split}.pt', expected_meta)
        cache, metrics = build_split(cli.retriever, exp, args, model, pool_cache, split, device)
        torch.save(cache, out_dir / f'{split}.pt')
        all_metrics[split] = metrics
        print(f'[build_r_retrieval_cache_p100] retriever={cli.retriever} split={split} '
             f'retmse10={metrics["retmse10"]:.6f} recall10={metrics["recall10"]:.4f}')
    (out_dir / 'stage1_metrics.json').write_text(json.dumps(all_metrics, indent=2))


if __name__ == '__main__':
    main()
