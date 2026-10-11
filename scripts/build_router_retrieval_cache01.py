#!/usr/bin/env python3
"""TRACK-AGGREGATE-ROUTER-FROZEN01 -- builds a retrieval cache (same
schema as `build_v_meanmix_cache_pool01.py`: {train,val,test}.pt with
`relation_outputs`/`query_start_idx`/`D_per_query`/`C_per_query`) using
a TRAINED frozen Router's inference, so `scripts/eval_r_stage2_lambda_signfix01.py`
(== `train_r_stage2_lambda01.py`'s formula, reused unmodified) can be
run on it exactly as on any other P100 cache. `--routing_mode soft`
uses `soft_routed_topk` (p_route = sum_h pi_h * p_h, matching Phase 2's
BEST_ROUTING_MODE decision); `hard` is also supported for completeness/
comparison. Read-only w.r.t. the frozen backbone and the trained
router -- this script trains nothing.
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
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k
from scripts.train_retriever_pool01 import CandidatePoolCache
from utils.aggregate_router01 import AggregateRouter, hard_routed_topk, soft_routed_topk
from utils.candidate_pool import gather_candidate_values, pooled_future_mse
from utils.full_candidate_bank import compute_scores_pool_channel_first_grad, encode_raw_channel_first

NUM_SLOTS = 5
TOP_K = 10


@torch.no_grad()
def build_split(model, slot_heads, router, routing_mode, exp, args, channels, device, loader, pool_cache, tau_s,
                top_k=TOP_K):
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
            scores, pool_valid_mask, pool_idx_global = compute_scores_pool_channel_first_grad(
                model, slot_heads, batch_x, exp, c, pool_cache, batch_start_idx, device)
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            pooled_memory_c = gather_candidate_values(memory_c, pool_idx_global)
            d_pool = pooled_future_mse(pooled_memory_c, offset_c, query_future)
            p_h = torch.softmax(scores / tau_s, dim=-1)

            z_qc = encode_raw_channel_first(model, batch_x, c)
            router_logits = router(z_qc)
            if routing_mode == 'soft':
                picks, _ = soft_routed_topk(router_logits, p_h, k=top_k)
            else:
                picks, _ = hard_routed_topk(router_logits, p_h, k=top_k)

            ind_mse_i = d_pool.gather(1, picks)
            y_sel = pooled_memory_c.gather(1, picks.unsqueeze(-1).expand(-1, -1, pooled_memory_c.size(-1))) \
                + offset_c.view(-1, 1, 1)
            agg_pred = y_sel.mean(dim=1)
            rel_out[:, :, c] = agg_pred
            agg_mse = ((agg_pred - query_future) ** 2).mean(dim=-1)
            D_ = ind_mse_i.mean(dim=-1) / top_k
            C_ = agg_mse - D_
            oracle_idx = stable_topk_indices(d_pool, top_k, largest=False)
            recall10 = recall_at_k(picks, oracle_idx, top_k)
            ndcg10 = ndcg_at_k(picks, d_pool, pool_valid_mask, top_k)
            all_D.append(D_.cpu()); all_C.append(C_.cpu()); all_Agg.append(agg_mse.cpu())
            all_Recall.append(recall10.cpu()); all_NDCG.append(ndcg10.cpu())
            all_ret.append(ind_mse_i.mean(dim=-1).cpu())
            D_pc[:, c] = D_; C_pc[:, c] = C_
        all_start.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx) else torch.as_tensor(batch_start_idx))
        all_R.append(rel_out.cpu())
        all_D_pq.append(D_pc.mean(dim=-1).cpu())
        all_C_pq.append(C_pc.mean(dim=-1).cpu())

    cache = {'query_start_idx': torch.cat(all_start), 'relation_outputs': torch.cat(all_R),
            'D_per_query': torch.cat(all_D_pq), 'C_per_query': torch.cat(all_C_pq),
            'channels': channels, 'pred_len': args.pred_len, 'split': loader}
    metrics = {'retmse10': float(torch.cat(all_ret).mean()), 'D': float(torch.cat(all_D).mean()),
              'C': float(torch.cat(all_C).mean()), 'agg_mse10': float(torch.cat(all_Agg).mean()),
              'recall10': float(torch.cat(all_Recall).mean()), 'ndcg10': float(torch.cat(all_NDCG).mean())}
    return cache, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--h', type=int, required=True)
    ap.add_argument('--retriever_checkpoint', required=True)
    ap.add_argument('--router_checkpoint', required=True)
    ap.add_argument('--routing_mode', required=True, choices=('hard', 'soft'))
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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    router = AggregateRouter(d_model).to(device)
    r_bl = torch.load(cli.router_checkpoint, map_location=device)
    router.load_state_dict(r_bl['router_state_dict'])
    router.eval()

    expected_meta = {'seq_len': cli.seq_len, 'pred_len': cli.pred_len, 'candidate_mask': cli.candidate_mask,
                     'candidate_pool_size': cli.candidate_pool_size, 'candidate_pool_metric': 'delta_last_cosine'}
    cache_dir = Path(cli.candidate_pool_cache)
    pool_caches = {s: CandidatePoolCache(cache_dir / f'{s}.pt', expected_meta) for s in ('train', 'val', 'test')}

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_metrics = None
    for split in ('train', 'val', 'test'):
        _, loader = exp._get_data(flag=split, shuffle=False)
        cache, metrics = build_split(model, slot_heads, router, cli.routing_mode, exp, args, channels, device,
                                     loader, pool_caches[split], cli.tau_s)
        cache['split'] = split
        torch.save(cache, out_dir / f'{split}.pt')
        print(f'[build_router_cache] {split}: n={cache["query_start_idx"].numel()} agg={metrics["agg_mse10"]:.6f} '
             f'retmse10={metrics["retmse10"]:.6f}')
        if split == 'test':
            test_metrics = metrics
    (out_dir / 'stage1_metrics.json').write_text(json.dumps(test_metrics, indent=2))
    print(f'[build_router_cache] wrote {out_dir / "stage1_metrics.json"}')


if __name__ == '__main__':
    main()
