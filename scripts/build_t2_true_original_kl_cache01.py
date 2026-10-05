#!/usr/bin/env python3
"""TRACK-T2-PROJECTION-MULTISLOT-DECOMPOSITION01 -- T0 (TRUE Original KL)
retrieval cache builder.

T0 has ZERO trainable query-projection parameters (no SlotHeads
anywhere) -- the checkpoint loaded here comes straight from
`train_j_shared_encoder_drift01.py`, run completely UNMODIFIED (PART 5:
"T0에서는 W_1 parameter 자체가 존재하면 안 된다" -- satisfied trivially
since that script never constructs one). Score is exactly
`arm_score(encode_raw(model, x, c), encode_raw(model, memory_x, c),
None)` -- plain cosine, both branches gradient-ON during training,
`.detach()`-free.

For Stage1-metric and Stage2-cache PARITY with T1/T2/T4/T10 (all built
via `train_t_pure_multislot01.py`'s `round_robin_topk_selection` +
`hard_eval_decomposition` + `spearman_batch`), T0's score matrix is
reshaped to `[B, 1, N]` (a size-1 "slot" axis) purely so the IDENTICAL
selection/decomposition code can be reused unmodified -- T0 still has
S=1 exactly, so `round_robin_topk_selection` reduces exactly to
ordinary Top-10 (TRACK-T's own unit test 3 already proves this
reduction is exact). No projection matrix is involved anywhere in this
file.
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

from scripts.train_factorial_e2e01 import arm_score, encode_raw, individual_utility_memsafe
from scripts.train_margutil01 import build_experiment, memory_value
from scripts.train_t_pure_multislot01 import hard_eval_decomposition, round_robin_topk_selection, spearman_batch

TOP_K = 10


def compute_scores_true_original_kl(model, batch_x, memory_x, c):
    """No SlotHeads, no projection matrix -- literally `arm_score` on the
    raw encoder output, reshaped with a size-1 slot axis for downstream
    code reuse only."""
    z_q = encode_raw(model, batch_x, c)
    z_k = encode_raw(model, memory_x, c)
    s = arm_score(z_q, z_k, None)  # [B, N], plain cosine
    return s.unsqueeze(1)  # [B, 1, N]


@torch.no_grad()
def build_split(exp, args, model, split, device, candidate_pool, chunk_size=4096):
    channels = list(range(int(args.enc_in)))
    _, loader = exp._get_data(flag=split, shuffle=False)
    all_start, all_R, all_D, all_C, all_Agg, all_Recall, all_NDCG, all_ret, all_spearman = \
        [], [], [], [], [], [], [], [], []
    all_D_pq, all_C_pq = [], []  # TRACK-V: per-query (channel-averaged) D/C for bootstrap
    compute_spearman = (split == 'test')  # spearman is only ever saved from test metrics -- skip elsewhere

    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        base_mask, counts = exp._candidate_mask(batch_start_idx)
        bsz = batch_x.size(0)
        rel_out = torch.zeros(bsz, args.pred_len, len(channels), device=device)
        D_pc = torch.zeros(bsz, len(channels), device=device)
        C_pc = torch.zeros(bsz, len(channels), device=device)
        for c in channels:
            cand_mask = candidate_pool.apply(base_mask, batch_start_idx, c, split)
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            u = individual_utility_memsafe(memory_c, offset_c, query_future, chunk_size)
            d_raw = -u

            scores = compute_scores_true_original_kl(model, batch_x, exp.memory_x, c)
            picks_t = round_robin_topk_selection(scores, cand_mask, k=TOP_K)

            ind_mse_i = d_raw.gather(1, picks_t)
            y_sel = memory_c[picks_t] + offset_c.view(-1, 1, 1)
            r_c = y_sel.mean(dim=1)
            rel_out[:, :, c] = r_c

            from models.RelationStage1 import stable_topk_indices
from scripts.shared_candidate_pool01 import SharedCandidatePool, VALID_POOL_MODES
            oracle_idx = stable_topk_indices(d_raw.masked_fill(~cand_mask, float('inf')), TOP_K, largest=False)
            D_ = ind_mse_i.mean(dim=-1) / TOP_K
            agg_mse = ((r_c - query_future) ** 2).mean(dim=-1)
            C_ = agg_mse - D_
            from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k
            recall = recall_at_k(picks_t, oracle_idx, TOP_K)
            ndcg = ndcg_at_k(picks_t, d_raw, cand_mask, TOP_K)
            all_D.append(D_.cpu()); all_C.append(C_.cpu()); all_Agg.append(agg_mse.cpu())
            all_Recall.append(recall.cpu()); all_NDCG.append(ndcg.cpu())
            all_ret.append(ind_mse_i.mean(dim=-1).cpu())
            if compute_spearman:
                all_spearman.append(spearman_batch(scores.mean(dim=1), d_raw, cand_mask))
            D_pc[:, c] = D_
            C_pc[:, c] = C_

        all_start.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx)
                         else torch.as_tensor(batch_start_idx))
        all_R.append(rel_out.cpu())
        all_D_pq.append(D_pc.mean(dim=-1).cpu())
        all_C_pq.append(C_pc.mean(dim=-1).cpu())

    cache = {'query_start_idx': torch.cat(all_start), 'relation_outputs': torch.cat(all_R),
            'D_per_query': torch.cat(all_D_pq), 'C_per_query': torch.cat(all_C_pq),
            'channels': channels, 'pred_len': args.pred_len, 'split': split, **candidate_pool.describe()}
    metrics = None
    if all_D:
        D_cat, C_cat, Agg_cat = torch.cat(all_D), torch.cat(all_C), torch.cat(all_Agg)
        assert torch.allclose(D_cat + C_cat, Agg_cat, atol=1e-3)
        metrics = {'retmse10': float(torch.cat(all_ret).mean()), 'D': float(D_cat.mean()),
                  'C': float(C_cat.mean()), 'agg_mse10': float(Agg_cat.mean()),
                  'recall10': float(torch.cat(all_Recall).mean()), 'ndcg10': float(torch.cat(all_NDCG).mean())}
        if all_spearman:
            metrics['spearman'] = float(sum(all_spearman) / len(all_spearman))
    return cache, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--retriever_checkpoint', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--candidate_pool_mode', choices=VALID_POOL_MODES, default='full')
    ap.add_argument('--candidate_pool_cache_dir', default=None)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.seq_len, 'batch_size': 32, 'seed': cli.seed, 'top_k': TOP_K,
        'relation_encoder_type': 'mlp', 'relation_self_fill': 'linear', 'relation_input_space': 'delta_last',
        'relation_teacher_space': 'delta_last', 'relation_value_space': 'delta_last', 'candidate_mask': 'raft',
        'patch_len': 16, 'stride': 16,
    })
    exp._ensure_memory()
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.to(device)
    model.eval()

    bl = torch.load(cli.retriever_checkpoint, map_location=device)
    assert 'slot_heads_state_dict' not in bl, \
        '[ISSUE][ABORT] T0 checkpoint must have NO slot_heads_state_dict -- this is not TRUE Original KL'
    model.load_state_dict(bl['model_state_dict'])
    for p in model.parameters():
        p.requires_grad_(False)

    candidate_pool = SharedCandidatePool(cli.candidate_pool_mode, cli.candidate_pool_cache_dir,
                                         int(exp.memory_x.size(0)), list(range(int(args.enc_in))), TOP_K)

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_metrics = None
    for split in ('train', 'val', 'test'):
        cache, metrics = build_split(exp, args, model, split, device, candidate_pool)
        torch.save(cache, out_dir / f'{split}.pt')
        print(f'[build_t2_cache] T0/{split}: n={cache["query_start_idx"].numel()}'
             + (f' retmse10={metrics["retmse10"]:.6f} agg={metrics["agg_mse10"]:.6f}' if metrics else ''))
        if split == 'test':
            test_metrics = metrics
    (out_dir / 'stage1_metrics.json').write_text(json.dumps(test_metrics, indent=2))
    print(f'[build_t2_cache] wrote {out_dir / "stage1_metrics.json"}')


if __name__ == '__main__':
    main()
