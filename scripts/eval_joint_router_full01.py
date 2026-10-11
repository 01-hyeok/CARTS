#!/usr/bin/env python3
"""TRACK-JOINT-AGGREGATE-ROUTED-MQ01 -- full real-backbone test
evaluation for a trained joint checkpoint (model+slot_heads+router all
jointly trained). Computes test Aggregate MSE / RetMSE@10 / Recall@10
/ NDCG@10 via Soft routing (the only routing mode this track trains or
evaluates, per Phase 2's decision), plus the Aggregate Oracle
reference and Oracle Gap Recovered % against the SAME validation-
selected fixed head used throughout this phase's training
diagnostics. Read-only w.r.t. the loaded checkpoint; no backward pass.
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
from scripts.train_joint_aggregate_routed_mq01 import per_head_aggregate_utility
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_margutil01 import memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k
from scripts.train_retriever_pool01 import CandidatePoolCache
from utils.aggregate_router01 import AggregateRouter
from utils.candidate_pool import gather_candidate_values, pooled_future_mse
from utils.full_candidate_bank import compute_scores_pool_channel_first_grad, encode_raw_channel_first

NUM_SLOTS = 5
TOP_K = 10


@torch.no_grad()
def full_test_pass(model, slot_heads, router, fixed_head_per_channel, exp, args, channels, device, loader,
                   pool_cache, tau_s, top_k=TOP_K):
    agg_sum, ret_sum, recall_sum, ndcg_sum, n = 0.0, 0.0, 0.0, 0.0, 0
    oracle_sum, fixed_sum = 0.0, 0.0
    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        bsz = batch_x.size(0)
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
            pi = torch.softmax(router_logits, dim=-1)
            p_R = (pi.unsqueeze(-1) * p_h).sum(dim=1)

            picks = p_R.topk(top_k, dim=-1).indices
            ind_mse_i = d_pool.gather(1, picks)
            y_sel = pooled_memory_c.gather(1, picks.unsqueeze(-1).expand(-1, -1, pooled_memory_c.size(-1))) \
                + offset_c.view(-1, 1, 1)
            agg_pred = y_sel.mean(dim=1)
            agg_mse = ((agg_pred - query_future) ** 2).mean(dim=-1)
            oracle_idx = stable_topk_indices(d_pool, top_k, largest=False)
            recall10 = recall_at_k(picks, oracle_idx, top_k)
            ndcg10 = ndcg_at_k(picks, d_pool, pool_valid_mask, top_k)

            agg_sum += float(agg_mse.sum()); ret_sum += float(ind_mse_i.mean(dim=-1).sum())
            recall_sum += float(recall10.sum()); ndcg_sum += float(ndcg10.sum())

            U_agg, _ = per_head_aggregate_utility(scores, pooled_memory_c, offset_c, query_future, d_pool,
                                                  top_k=top_k)
            oracle_sum += float(U_agg.min(dim=-1).values.sum())
            fh = int(fixed_head_per_channel[c])
            fh_idx = p_h[:, fh, :].topk(top_k, dim=-1).indices
            y_sel_fh = pooled_memory_c.gather(1, fh_idx.unsqueeze(-1).expand(-1, -1, pooled_memory_c.size(-1))) \
                + offset_c.view(-1, 1, 1)
            fixed_sum += float(((y_sel_fh.mean(dim=1) - query_future) ** 2).mean(dim=-1).sum())
        n += bsz
    denom = max(n * len(channels), 1)
    return {'agg_mse': agg_sum / denom, 'retmse10': ret_sum / denom, 'recall10': recall_sum / denom,
           'ndcg10': ndcg_sum / denom, 'aggregate_oracle_mse': oracle_sum / denom,
           'fixed_head_mse': fixed_sum / denom}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--patch_len', type=int, default=16)
    ap.add_argument('--tau_s', type=float, required=True)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--joint_checkpoint', required=True)
    ap.add_argument('--fixed_head_json', required=True)
    ap.add_argument('--candidate_pool_cache', required=True)
    ap.add_argument('--candidate_pool_size', type=int, default=100)
    ap.add_argument('--candidate_mask', default='raft')
    ap.add_argument('--out_dir', required=True)
    cli = ap.parse_args()
    cli.init_seed = cli.seed
    cli.top_k = TOP_K

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp, args, model = build_model(cli, device)
    channels = list(range(int(args.enc_in)))
    d_model = int(args.d_model)

    bl = torch.load(cli.joint_checkpoint, map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    slot_heads = SlotHeads(d_model, n_slots=NUM_SLOTS).to(device)
    slot_heads.load_state_dict(bl['slot_heads_state_dict'])
    router = AggregateRouter(d_model).to(device)
    router.load_state_dict(bl['router_state_dict'])
    model.eval(); slot_heads.eval(); router.eval()

    oh = json.load(open(cli.fixed_head_json))
    fixed_head_per_channel = torch.tensor(oh['per_channel']['fixed_heads_per_channel'], dtype=torch.long)

    expected_meta = {'seq_len': cli.seq_len, 'pred_len': cli.pred_len, 'candidate_mask': cli.candidate_mask,
                     'candidate_pool_size': cli.candidate_pool_size, 'candidate_pool_metric': 'delta_last_cosine'}
    pool_cache = CandidatePoolCache(Path(cli.candidate_pool_cache) / 'test.pt', expected_meta)
    _, test_loader = exp._get_data(flag='test', shuffle=False)

    results = full_test_pass(model, slot_heads, router, fixed_head_per_channel, exp, args, channels, device,
                             test_loader, pool_cache, cli.tau_s)
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'test_evaluation.json').write_text(json.dumps(results, indent=2))
    print(f'[joint_eval_full] {cli.cell} done. agg_mse={results["agg_mse"]:.5f} retmse10={results["retmse10"]:.4f} '
         f'oracle={results["aggregate_oracle_mse"]:.5f} fixed={results["fixed_head_mse"]:.5f}')


if __name__ == '__main__':
    main()
