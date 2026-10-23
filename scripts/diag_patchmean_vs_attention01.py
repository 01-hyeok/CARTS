#!/usr/bin/env python3
"""TRACK-PATCH-REPRESENTATION-RESIDUAL01 -- PHASE A: PH1 attention vs
patch-mean, evaluation-only. No training, no new parameters, no new
checkpoint -- reads the EXISTING PH1 selected checkpoint (read-only)
and computes two queries from the SAME Transformer forward pass:

    q_attn = normalize(sum_p a_p h_p)   -- PH1's own temporal_attention_heads
    q_mean = normalize(mean_p h_p)      -- plain patch mean, no attention

for every query/channel on the TEST split, then scores both against
the SAME pooled P100 candidate embeddings (candidate branch untouched).
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from models.TemporalQueryHeads import temporal_attention_heads
from scripts.train_factorial_e2e01 import encode_raw
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_margutil01 import memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k
from scripts.train_retriever_pool01 import CandidatePoolCache
from scripts.train_t_pure_multislot01 import round_robin_topk_selection
from scripts.train_transformer_patch_query_heads01 import build_transformer_model, encode_raw_with_tokens
from utils.candidate_pool import (
    encode_pooled_candidates, gather_candidate_histories, gather_candidate_values, pooled_future_mse,
)

TOP_K = 10


def _dist(t):
    return {'mean': float(t.mean()), 'std': float(t.std()), 'median': float(t.median()),
           'p5': float(t.quantile(0.05)), 'p25': float(t.quantile(0.25)), 'p75': float(t.quantile(0.75)),
           'p95': float(t.quantile(0.95)), 'min': float(t.min()), 'max': float(t.max())}


@torch.no_grad()
def run_cell(cli, device):
    exp, args, model = build_transformer_model(cli, device)
    exp.memory_x = exp.memory_x.to(device)
    model.eval()
    assert model.encoder.patch_embed.num_patches == cli.expected_num_patches

    slot_heads = SlotHeads(int(args.d_model), n_slots=1).to(device)
    bl = torch.load(cli.ph1_checkpoint, map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    slot_heads.load_state_dict(bl['slot_heads_state_dict'])
    slot_heads.eval()

    expected_meta = {'seq_len': cli.seq_len, 'pred_len': cli.pred_len, 'candidate_mask': cli.candidate_mask,
                     'candidate_pool_size': cli.candidate_pool_size, 'candidate_pool_metric': 'delta_last_cosine'}
    pool_cache = CandidatePoolCache(Path(cli.candidate_pool_cache) / 'test.pt', expected_meta)

    channels = list(range(int(args.enc_in)))
    _, loader = exp._get_data(flag='test', shuffle=False)

    cos_vals, l2_vals, overlap_vals = [], [], []
    rows = {tag: {'ret': [], 'agg': [], 'recall': [], 'ndcg': [], 'D': [], 'C': []}
           for tag in ('attn', 'mean')}

    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        for c in channels:
            z_q, h_patches = encode_raw_with_tokens(model, batch_x, c)
            g = slot_heads(z_q)  # [B,1,D]
            q_attn, _ = temporal_attention_heads(g, h_patches)  # [B,1,D]
            q_mean = F.normalize(h_patches.mean(dim=1), dim=-1).unsqueeze(1)  # [B,1,D]

            qa = q_attn[:, 0, :]
            qm = q_mean[:, 0, :]
            cos_vals.append(F.cosine_similarity(qa, qm, dim=-1).cpu())
            l2_vals.append((qa - qm).norm(dim=-1).cpu())

            pool_idx_global = pool_cache.lookup(batch_start_idx, c, device)
            pooled_x = gather_candidate_histories(exp.memory_x, pool_idx_global)
            z_k = encode_pooled_candidates(lambda x, cc: encode_raw(model, x, cc), pooled_x, c)
            z_k = F.normalize(z_k, dim=-1)

            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            pooled_memory_c = gather_candidate_values(memory_c, pool_idx_global)
            d_pool = pooled_future_mse(pooled_memory_c, offset_c, query_future)
            oracle_idx = stable_topk_indices(d_pool, TOP_K, largest=False)
            valid_mask = torch.ones(pool_idx_global.shape, dtype=torch.bool, device=device)

            picks = {}
            for tag, q in (('attn', q_attn), ('mean', q_mean)):
                scores = torch.einsum('bsd,bmd->bsm', q, z_k)  # [B,1,M]
                idx = round_robin_topk_selection(scores, valid_mask, k=TOP_K)
                picks[tag] = idx
                ind_mse_i = d_pool.gather(1, idx)
                y_sel = pooled_memory_c.gather(
                    1, idx.unsqueeze(-1).expand(-1, -1, pooled_memory_c.size(-1))) + offset_c.view(-1, 1, 1)
                r_c = y_sel.mean(dim=1)
                agg_mse = ((r_c - query_future) ** 2).mean(dim=-1)
                D_ = ind_mse_i.mean(dim=-1) / TOP_K
                C_ = agg_mse - D_
                recall = recall_at_k(idx, oracle_idx, TOP_K)
                ndcg = ndcg_at_k(idx, d_pool, valid_mask, TOP_K)
                rows[tag]['ret'].append(ind_mse_i.mean(dim=-1).cpu())
                rows[tag]['agg'].append(agg_mse.cpu())
                rows[tag]['recall'].append(recall.cpu())
                rows[tag]['ndcg'].append(ndcg.cpu())
                rows[tag]['D'].append(D_.cpu())
                rows[tag]['C'].append(C_.cpu())

            ov = torch.tensor([len(set(picks['attn'][b].tolist()) & set(picks['mean'][b].tolist())) / TOP_K
                               for b in range(picks['attn'].size(0))])
            overlap_vals.append(ov)

    cos_all = torch.cat(cos_vals)
    l2_all = torch.cat(l2_vals)
    overlap_all = torch.cat(overlap_vals)

    out = {
        'cosine_distribution': _dist(cos_all),
        'l2_distance_distribution': _dist(l2_all),
        'top10_overlap_attn_vs_mean': {'mean': float(overlap_all.mean()), 'std': float(overlap_all.std())},
    }
    for tag in ('attn', 'mean'):
        out[f'retrieval_{tag}'] = {
            'retmse10': float(torch.cat(rows[tag]['ret']).mean()),
            'agg_mse10': float(torch.cat(rows[tag]['agg']).mean()),
            'recall10': float(torch.cat(rows[tag]['recall']).mean()),
            'ndcg10': float(torch.cat(rows[tag]['ndcg']).mean()),
            'D': float(torch.cat(rows[tag]['D']).mean()),
            'C': float(torch.cat(rows[tag]['C']).mean()),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--patch_len', type=int, required=True)
    ap.add_argument('--stride', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--candidate_pool_size', type=int, default=100)
    ap.add_argument('--candidate_pool_cache', required=True)
    ap.add_argument('--candidate_mask', default='raft')
    ap.add_argument('--expected_num_patches', type=int, default=6)
    ap.add_argument('--ph1_checkpoint', required=True)
    ap.add_argument('--out_dir', required=True)
    cli = ap.parse_args()
    if cli.stride is None:
        cli.stride = cli.patch_len
    cli.init_seed = cli.seed
    cli.top_k = TOP_K
    cli.batch_size = 32

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[audit] cell={cli.cell} PH1 checkpoint = {cli.ph1_checkpoint}')
    out = run_cell(cli, device)
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'{cli.cell}_phaseA.json').write_text(json.dumps(out, indent=2))
    print(f'[diag_patchmean_vs_attention] {cli.cell}: cosine_mean={out["cosine_distribution"]["mean"]:.4f} '
         f'top10_overlap_mean={out["top10_overlap_attn_vs_mean"]["mean"]:.4f} '
         f'retmse_attn={out["retrieval_attn"]["retmse10"]:.4f} retmse_mean={out["retrieval_mean"]["retmse10"]:.4f} '
         f'agg_attn={out["retrieval_attn"]["agg_mse10"]:.4f} agg_mean={out["retrieval_mean"]["agg_mse10"]:.4f}')


if __name__ == '__main__':
    main()
