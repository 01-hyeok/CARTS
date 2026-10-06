#!/usr/bin/env python3
"""TRACK-W-TIMESTAMP-FUSION01 -- retrieval cache builder for Phase 2's
promoted arms (T-V0=reused C1, T-V1, T-V2, T-V5 -- ALL real_time; Phase
1's C0/C2 never need a cache/Stage2 at all, per spec section 6, since
their own Stage1 `final_test_metrics.json` already has everything
Phase 1 needs). Same cache schema as `build_t_multislot_cache01.py`
(`{train,val,test}.pt`: `relation_outputs` [N,H,C], `query_start_idx`,
`D_per_query`, `C_per_query`), so `train_r_stage2_lambda01.py` is reused
UNMODIFIED for Stage2 here, exactly as every other track this session
has done.
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
from models.TimestampRelationEncoder import TimestampFusionEncoder
from scripts.train_factorial_e2e01 import individual_utility_memsafe
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_margutil01 import memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k
from scripts.train_t_pure_multislot01 import round_robin_topk_selection
from scripts.train_w_timestamp_common01 import build_timestamp_experiment
from scripts.train_w_timestamp_stage1_01 import compute_scores_slots, compute_scores_zero_proj

TOP_K = 10


@torch.no_grad()
def build_split(exp, args, time_encoder, slot_heads, split, device, chunk_size=4096):
    channels = list(range(int(args.enc_in)))
    _, loader = exp._get_data(flag=split, shuffle=False, include_time_mark=True)
    all_start, all_R, all_D, all_C, all_Agg, all_Recall, all_NDCG, all_ret = [], [], [], [], [], [], [], []
    all_D_pq, all_C_pq = [], []

    for batch in loader:
        batch_x, batch_y, batch_start_idx, batch_x_mark = batch
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        batch_x_mark = batch_x_mark.float().to(device)
        cand_mask, counts = exp._candidate_mask(batch_start_idx)
        bsz = batch_x.size(0)
        rel_out = torch.zeros(bsz, args.pred_len, len(channels), device=device)
        D_pc = torch.zeros(bsz, len(channels), device=device)
        C_pc = torch.zeros(bsz, len(channels), device=device)
        for c in channels:
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            u = individual_utility_memsafe(memory_c, offset_c, query_future, chunk_size)
            d_raw = -u

            if slot_heads is None:
                scores = compute_scores_zero_proj(time_encoder, batch_x, batch_x_mark, exp.memory_x,
                                                  exp.memory_x_mark, c)
            else:
                scores = compute_scores_slots(time_encoder, slot_heads, batch_x, batch_x_mark, exp.memory_x,
                                              exp.memory_x_mark, c)
            picks_t = round_robin_topk_selection(scores, cand_mask, k=TOP_K)

            ind_mse_i = d_raw.gather(1, picks_t)
            y_sel = memory_c[picks_t] + offset_c.view(-1, 1, 1)
            r_c = y_sel.mean(dim=1)
            rel_out[:, :, c] = r_c

            oracle_idx = stable_topk_indices(d_raw.masked_fill(~cand_mask, float('inf')), TOP_K, largest=False)
            D_ = ind_mse_i.mean(dim=-1) / TOP_K
            agg_mse = ((r_c - query_future) ** 2).mean(dim=-1)
            C_ = agg_mse - D_
            recall = recall_at_k(picks_t, oracle_idx, TOP_K)
            ndcg = ndcg_at_k(picks_t, d_raw, cand_mask, TOP_K)
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
    metrics = None
    if all_D:
        D_cat, C_cat, Agg_cat = torch.cat(all_D), torch.cat(all_C), torch.cat(all_Agg)
        assert torch.allclose(D_cat + C_cat, Agg_cat, atol=1e-3)
        metrics = {'retmse10': float(torch.cat(all_ret).mean()), 'D': float(D_cat.mean()),
                  'C': float(C_cat.mean()), 'agg_mse10': float(Agg_cat.mean()),
                  'recall10': float(torch.cat(all_Recall).mean()), 'ndcg10': float(torch.cat(all_NDCG).mean())}
    return cache, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--mode', required=True, choices=('zero_proj', 'slots'))
    ap.add_argument('--num_slots', type=int, default=1)
    ap.add_argument('--arm_name', required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--patch_len', type=int, default=16)
    ap.add_argument('--init_seed', type=int, default=0)
    ap.add_argument('--top_k', type=int, default=TOP_K)
    ap.add_argument('--time_proj_dim', type=int, default=32)
    ap.add_argument('--retriever_checkpoint', required=True)
    ap.add_argument('--out_dir', required=True)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp, args, time_feat_dim = build_timestamp_experiment(cli, device)
    d_model = int(args.d_model)
    d_ff = int(args.d_ff)

    time_encoder = TimestampFusionEncoder(
        seq_len=cli.seq_len, d_model=d_model, d_ff=d_ff, time_feat_dim=time_feat_dim,
        time_proj_dim=cli.time_proj_dim, dropout=float(args.dropout),
        retrieval_similarity=getattr(args, 'retrieval_similarity', 'cosine'),
    ).to(device)
    bl = torch.load(cli.retriever_checkpoint, map_location=device)
    time_encoder.load_state_dict(bl['time_encoder_state_dict'])
    time_encoder.eval()
    for p in time_encoder.parameters():
        p.requires_grad_(False)

    slot_heads = None
    if cli.mode == 'slots':
        slot_heads = SlotHeads(d_model, n_slots=cli.num_slots).to(device)
        slot_heads.load_state_dict(bl['slot_heads_state_dict'])
        slot_heads.eval()
        for p in slot_heads.parameters():
            p.requires_grad_(False)

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_metrics = None
    for split in ('train', 'val', 'test'):
        cache, metrics = build_split(exp, args, time_encoder, slot_heads, split, device)
        torch.save(cache, out_dir / f'{split}.pt')
        print(f'[build_w_ts_cache] {cli.arm_name}/{split}: n={cache["query_start_idx"].numel()}'
             + (f' retmse10={metrics["retmse10"]:.6f} agg={metrics["agg_mse10"]:.6f}' if metrics else ''))
        if split == 'test':
            test_metrics = metrics
    (out_dir / 'stage1_metrics.json').write_text(json.dumps(test_metrics, indent=2))
    print(f'[build_w_ts_cache] wrote {out_dir / "stage1_metrics.json"}')


if __name__ == '__main__':
    main()
