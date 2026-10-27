#!/usr/bin/env python3
"""TRACK-V5-HEAD-LOSS-DUAL-ENCODER01 -- builds a per-arm Stage2 retrieval
cache (`query_start_idx`/`relation_outputs`, same schema as
`build_v_meanmix_cache_pool01.py`) from one of this track's trained
S0/S1/S2/D0/D1/D2 checkpoints, so that `scripts/train_r_stage2_lambda01.py`
(reused UNMODIFIED, per spec) can be run against it unchanged. Mean
-Mixture Top-10 Hard inference (`mean_mixture_topk_selection`) is used
for every arm, identically to every other Stage2 cache in this
project -- the loss used to TRAIN the retriever never changes the
INFERENCE rule used to build its Stage2 cache.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_j_shared_encoder_drift01 import build_model
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_margutil01 import memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k
from scripts.train_retriever_pool01 import CandidatePoolCache
from scripts.train_v5_head_loss_dual_encoder01 import compute_scores_dual
from models.RelationStage1 import stable_topk_indices
from utils.candidate_pool import gather_candidate_values, pooled_future_mse
from utils.mean_mixture_selection import mean_mixture_topk_selection

TOP_K = 10


@torch.no_grad()
def build_split(exp, args, model_q, model_k, slot_heads, pool_cache, split, device, tau_s):
    channels = list(range(int(args.enc_in)))
    _, loader = exp._get_data(flag=split, shuffle=False)
    all_start, all_R = [], []
    all_D, all_C, all_Agg, all_Recall, all_NDCG, all_ret = [], [], [], [], [], []

    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        bsz = batch_x.size(0)
        rel_out = torch.zeros(bsz, args.pred_len, len(channels), device=device)
        for c in channels:
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            scores, valid_mask, pool_idx_global = compute_scores_dual(
                model_q, model_k, slot_heads, batch_x, exp, c, pool_cache, batch_start_idx, device)
            pooled_memory_c = gather_candidate_values(memory_c, pool_idx_global)
            d_pool = pooled_future_mse(pooled_memory_c, offset_c, query_future)
            picks_t, _, _ = mean_mixture_topk_selection(scores, valid_mask, tau_s, k=TOP_K)
            ind_mse_i = d_pool.gather(1, picks_t)
            y_sel = pooled_memory_c.gather(
                1, picks_t.unsqueeze(-1).expand(-1, -1, pooled_memory_c.size(-1))) + offset_c.view(-1, 1, 1)
            r_c = y_sel.mean(dim=1)
            rel_out[:, :, c] = r_c
            oracle_idx = stable_topk_indices(d_pool, TOP_K, largest=False)
            D_ = ind_mse_i.mean(dim=-1) / TOP_K
            agg_mse = ((r_c - query_future) ** 2).mean(dim=-1)
            C_ = agg_mse - D_
            all_D.append(D_.cpu()); all_C.append(C_.cpu()); all_Agg.append(agg_mse.cpu())
            all_Recall.append(recall_at_k(picks_t, oracle_idx, TOP_K).cpu())
            all_NDCG.append(ndcg_at_k(picks_t, d_pool, valid_mask, TOP_K).cpu())
            all_ret.append(ind_mse_i.mean(dim=-1).cpu())
        all_start.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx)
                         else torch.as_tensor(batch_start_idx))
        all_R.append(rel_out.cpu())

    cache = {'query_start_idx': torch.cat(all_start), 'relation_outputs': torch.cat(all_R),
            'channels': channels, 'pred_len': args.pred_len, 'split': split}
    D_cat, C_cat, Agg_cat = torch.cat(all_D), torch.cat(all_C), torch.cat(all_Agg)
    assert torch.allclose(D_cat + C_cat, Agg_cat, atol=1e-3), '[ISSUE][ABORT] D+C!=Agg in cache build'
    metrics = {'retmse10': float(torch.cat(all_ret).mean()), 'D': float(D_cat.mean()), 'C': float(C_cat.mean()),
              'agg_mse10': float(Agg_cat.mean()), 'recall10': float(torch.cat(all_Recall).mean()),
              'ndcg10': float(torch.cat(all_NDCG).mean())}
    return cache, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--arm', required=True)
    ap.add_argument('--retriever_checkpoint', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--patch_len', type=int, default=16)
    ap.add_argument('--init_seed', type=int, default=0)
    ap.add_argument('--top_k', type=int, default=TOP_K)
    ap.add_argument('--tau_s', type=float, default=0.1)
    ap.add_argument('--slot_std', type=float, default=1e-3)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--candidate_mask', default='raft')
    ap.add_argument('--candidate_pool_size', type=int, default=100)
    ap.add_argument('--candidate_pool_cache', required=True)
    ap.add_argument('--out_dir', required=True)
    cli = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    exp, args, model_q = build_model(cli, device)
    bl = torch.load(cli.retriever_checkpoint, map_location=device)
    model_q.load_state_dict(bl['model_q_state_dict'])
    model_q.eval()
    shared = bool(bl['shared_encoder'])
    if shared:
        model_k = model_q
    else:
        import copy
        model_k = copy.deepcopy(model_q)
        model_k.load_state_dict(bl['model_k_state_dict'])
        model_k.eval()
    slot_heads = SlotHeads(int(args.d_model), n_slots=5, std=cli.slot_std).to(device)
    slot_heads.load_state_dict(bl['slot_heads_state_dict'])
    slot_heads.eval()

    expected_meta = {'seq_len': cli.seq_len, 'pred_len': cli.pred_len, 'candidate_mask': cli.candidate_mask,
                     'candidate_pool_size': cli.candidate_pool_size, 'candidate_pool_metric': 'delta_last_cosine'}
    cache_dir = Path(cli.candidate_pool_cache)
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_metrics = {}
    for split in ('train', 'val', 'test'):
        pool_cache = CandidatePoolCache(cache_dir / f'{split}.pt', expected_meta)
        cache, metrics = build_split(exp, args, model_q, model_k, slot_heads, pool_cache, split, device, cli.tau_s)
        torch.save(cache, out_dir / f'{split}.pt')
        all_metrics[split] = metrics
        print(f'[build_cache] arm={cli.arm} split={split} n={cache["query_start_idx"].size(0)} '
             f'retmse10={metrics["retmse10"]:.6f} agg_mse10={metrics["agg_mse10"]:.6f} best_epoch={bl["epoch"]}')
    (out_dir / 'stage1_metrics.json').write_text(json.dumps(
        {'arm': cli.arm, 'best_epoch': int(bl['epoch']), **all_metrics}, indent=2, default=str))


if __name__ == '__main__':
    main()
