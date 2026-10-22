#!/usr/bin/env python3
"""TRACK-TRANSFORMER-PATCH-QUERY-HEADS01 -- retrieval cache builder
(for Stage-2) + full test-split diagnostics (Stage-1 retrieval quality,
head complementarity for n_slots=5 arms, attention diagnostics for
patch-aware arms). Same cache schema as `build_tqh_retrieval_cache01.py`
so `train_r_stage2_lambda01.py` is reused UNMODIFIED. n_slots=1 arms
use ordinary Top-K (== Mean-Mixture, proven); n_slots=5 arms use
Mean-Mixture (`utils.mean_mixture_selection`, reused unmodified).
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
from models.TemporalQueryHeads import (
    normalized_attention_entropy, pairwise_head_attention_cosine, query_attention_diversity,
)
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_margutil01 import memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k
from scripts.train_retriever_pool01 import CandidatePoolCache
from scripts.train_t_pure_multislot01 import round_robin_topk_selection, slot_mechanism_diagnostics
from scripts.train_temporal_query_heads01 import union_oracle_recall_curve
from scripts.train_transformer_patch_query_heads01 import (
    ARM_N_SLOTS, PATCH_AWARE_ARMS, build_transformer_model, compute_scores_arm,
)
from utils.candidate_pool import gather_candidate_values, pooled_future_mse
from utils.mean_mixture_selection import mean_mixture_topk_selection

TOP_K = 10


@torch.no_grad()
def build_split(arm, exp, args, model, slot_heads, pool_cache, split, device, tau_s, collect_diag=False):
    channels = list(range(int(args.enc_in)))
    _, loader = exp._get_data(flag=split, shuffle=False)
    n_slots = ARM_N_SLOTS[arm]
    all_start, all_R, all_D, all_C, all_Agg, all_Recall, all_NDCG, all_ret = [], [], [], [], [], [], [], []
    all_D_pq, all_C_pq = [], []
    diag = {'slot_top1': [], 'slot_top10': [], 'cosine': [], 'union_curves': [], 'union_size': [],
           'attn_entropy': [], 'attn_max_prob': [], 'attn_head_cosine': [], 'attn_query_div': []}

    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        bsz = batch_x.size(0)
        rel_out = torch.zeros(bsz, args.pred_len, len(channels), device=device)
        D_pc = torch.zeros(bsz, len(channels), device=device)
        C_pc = torch.zeros(bsz, len(channels), device=device)
        for c in channels:
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            scores, valid_mask, pool_idx_global, attn = compute_scores_arm(
                arm, model, slot_heads, batch_x, exp, c, pool_cache, batch_start_idx, device)
            pooled_memory_c = gather_candidate_values(memory_c, pool_idx_global)
            d_pool = pooled_future_mse(pooled_memory_c, offset_c, query_future)

            if n_slots == 1:
                picks_t = round_robin_topk_selection(scores, valid_mask, k=TOP_K)
            else:
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
            recall = recall_at_k(picks_t, oracle_idx, TOP_K)
            ndcg = ndcg_at_k(picks_t, d_pool, valid_mask, TOP_K)
            all_D.append(D_.cpu()); all_C.append(C_.cpu()); all_Agg.append(agg_mse.cpu())
            all_Recall.append(recall.cpu()); all_NDCG.append(ndcg.cpu())
            all_ret.append(ind_mse_i.mean(dim=-1).cpu())
            D_pc[:, c] = D_
            C_pc[:, c] = C_

            if collect_diag and n_slots > 1:
                s_masked = scores.masked_fill(~valid_mask.unsqueeze(1), float('-inf'))
                p_m = torch.softmax(s_masked / tau_s, dim=-1)
                mech = slot_mechanism_diagnostics(scores, valid_mask, p_m, tau_s)
                diag['slot_top1'].append(mech['slot_top1_overlap'])
                diag['slot_top10'].append(mech['slot_top10_overlap'])
                diag['cosine'].append(mech['mean_pairwise_slot_cosine'])
                curve, usize = union_oracle_recall_curve(scores, valid_mask, oracle_idx, TOP_K)
                diag['union_curves'].append(curve)
                diag['union_size'].append(usize)
                if arm in PATCH_AWARE_ARMS and attn is not None:
                    diag['attn_entropy'].append(float(normalized_attention_entropy(attn).mean()))
                    diag['attn_max_prob'].append(float(attn.max(dim=-1).values.mean()))
                    diag['attn_head_cosine'].append(pairwise_head_attention_cosine(attn))
                    qdiv = query_attention_diversity(attn)  # NaN when this batch has bsz<2
                    # (e.g. a trailing partial batch of size 1) -- skip rather than poison the mean
                    if qdiv == qdiv:
                        diag['attn_query_div'].append(qdiv)

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
    metrics = {'retmse10': float(torch.cat(all_ret).mean()), 'D': float(D_cat.mean()),
              'C': float(C_cat.mean()), 'agg_mse10': float(Agg_cat.mean()),
              'recall10': float(torch.cat(all_Recall).mean()), 'ndcg10': float(torch.cat(all_NDCG).mean())}
    if collect_diag and diag['slot_top1']:
        metrics['head_diag'] = {
            'slot_top1_overlap': sum(diag['slot_top1']) / len(diag['slot_top1']),
            'slot_top10_overlap': sum(diag['slot_top10']) / len(diag['slot_top10']),
            'mean_pairwise_slot_cosine': sum(diag['cosine']) / len(diag['cosine']),
            'union_size_mean': sum(diag['union_size']) / len(diag['union_size']),
            'union_oracle_recall_by_n_heads': torch.tensor(diag['union_curves']).mean(dim=0).tolist(),
        }
        if diag['attn_entropy']:
            metrics['attn_diag'] = {
                'attn_entropy_normalized_mean': sum(diag['attn_entropy']) / len(diag['attn_entropy']),
                'attn_max_prob_mean': sum(diag['attn_max_prob']) / len(diag['attn_max_prob']),
                'attn_pairwise_head_cosine': sum(diag['attn_head_cosine']) / len(diag['attn_head_cosine']),
                'attn_query_diversity': sum(diag['attn_query_div']) / len(diag['attn_query_div']),
            }
    return cache, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--arm', required=True, choices=('V1', 'V5', 'PH1', 'PH5'))
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--patch_len', type=int, required=True)
    ap.add_argument('--stride', type=int, default=None)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--tau_s', type=float, default=0.1)
    ap.add_argument('--candidate_pool_size', type=int, default=100)
    ap.add_argument('--candidate_pool_cache', required=True)
    ap.add_argument('--candidate_mask', default='raft')
    ap.add_argument('--expected_num_patches', type=int, default=6)
    ap.add_argument('--retriever_checkpoint', required=True)
    ap.add_argument('--out_dir', required=True)
    cli = ap.parse_args()
    if cli.stride is None:
        cli.stride = cli.patch_len

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cli.init_seed = cli.seed
    cli.top_k = TOP_K
    cli.batch_size = 32
    n_slots = ARM_N_SLOTS[cli.arm]

    exp, args, model = build_transformer_model(cli, device)
    exp.memory_x = exp.memory_x.to(device)
    model.eval()
    assert model.encoder.patch_embed.num_patches == cli.expected_num_patches
    for p in model.parameters():
        p.requires_grad_(False)

    bl = torch.load(cli.retriever_checkpoint, map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    slot_heads = SlotHeads(int(args.d_model), n_slots=n_slots).to(device)
    slot_heads.load_state_dict(bl['slot_heads_state_dict'])
    slot_heads.eval()
    for p in slot_heads.parameters():
        p.requires_grad_(False)

    expected_meta = {'seq_len': cli.seq_len, 'pred_len': cli.pred_len, 'candidate_mask': cli.candidate_mask,
                     'candidate_pool_size': cli.candidate_pool_size, 'candidate_pool_metric': 'delta_last_cosine'}
    cache_dir = Path(cli.candidate_pool_cache)
    pool_cache_by_split = {s: CandidatePoolCache(cache_dir / f'{s}.pt', expected_meta)
                          for s in ('train', 'val', 'test')}

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_metrics = None
    for split in ('train', 'val', 'test'):
        cache, metrics = build_split(cli.arm, exp, args, model, slot_heads, pool_cache_by_split[split], split,
                                     device, cli.tau_s, collect_diag=(split == 'test'))
        torch.save(cache, out_dir / f'{split}.pt')
        print(f'[build_tpqh_cache] {cli.arm}/{split}: n={cache["query_start_idx"].numel()} '
             f'retmse10={metrics["retmse10"]:.6f} agg={metrics["agg_mse10"]:.6f}')
        if split == 'test':
            test_metrics = metrics
    (out_dir / 'stage1_metrics.json').write_text(json.dumps(test_metrics, indent=2))
    print(f'[build_tpqh_cache] wrote {out_dir / "stage1_metrics.json"}')


if __name__ == '__main__':
    main()
