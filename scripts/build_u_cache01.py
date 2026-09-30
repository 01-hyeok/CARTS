#!/usr/bin/env python3
"""TRACK-U-ASYMMETRY-CAPACITY-DECOMPOSITION01 -- retrieval cache builder
for U0/U1/U2/U3. Ordinary (single-score-vector) Top-10 via
`stable_topk_indices` -- no multi-slot round-robin anywhere in this
track. Same cache schema as `build_r_retrieval_cache01.py`/
`build_t2_true_original_kl_cache01.py` so `train_r_stage2_lambda01.py`
is reused unmodified for Stage2.
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
from scripts.train_margutil01 import build_experiment, memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k
from scripts.train_u_asymmetry_capacity01 import SingleProjection, compute_scores, spearman_batch

TOP_K = 10


@torch.no_grad()
def build_split(exp, args, model, projection, arm, split, device, chunk_size=4096):
    channels = list(range(int(args.enc_in)))
    _, loader = exp._get_data(flag=split, shuffle=False)
    all_start, all_R, all_D, all_C, all_Agg, all_Recall, all_NDCG, all_ret, all_spearman = \
        [], [], [], [], [], [], [], [], []
    # Spearman is only ever saved from the TEST split's metrics (below) -- computing
    # it for train/val too is pure wasted work (scipy spearmanr per query, per
    # channel, per batch) that dominated wall-clock time with no effect on any
    # saved metric. Skipped for train/val.
    compute_spearman = (split == 'test')

    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, counts = exp._candidate_mask(batch_start_idx)
        bsz = batch_x.size(0)
        rel_out = torch.zeros(bsz, args.pred_len, len(channels), device=device)
        for c in channels:
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            u = individual_utility_memsafe(memory_c, offset_c, query_future, chunk_size)
            d_raw = -u

            scores = compute_scores(model, projection, batch_x, exp.memory_x, c, arm)
            s_masked = scores.masked_fill(~cand_mask, float('-inf'))
            picks_t = stable_topk_indices(s_masked, TOP_K, largest=True)

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
            if compute_spearman:
                all_spearman.append(spearman_batch(scores, d_raw, cand_mask))

        all_start.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx)
                         else torch.as_tensor(batch_start_idx))
        all_R.append(rel_out.cpu())

    cache = {'query_start_idx': torch.cat(all_start), 'relation_outputs': torch.cat(all_R),
            'channels': channels, 'pred_len': args.pred_len, 'split': split}
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
    ap.add_argument('--arm', required=True, choices=('U0', 'U1', 'U2', 'U3'))
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--retriever_checkpoint', required=True)
    ap.add_argument('--out_dir', required=True)
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
    model.load_state_dict(bl['model_state_dict'])
    projection = None
    if cli.arm != 'U0':
        assert 'projection_state_dict' in bl, f'[ISSUE][ABORT] {cli.arm} checkpoint missing projection_state_dict'
        projection = SingleProjection(int(args.d_model)).to(device)
        projection.load_state_dict(bl['projection_state_dict'])
        projection.eval()
        for p in projection.parameters():
            p.requires_grad_(False)
    else:
        assert 'projection_state_dict' not in bl, '[ISSUE][ABORT] U0 checkpoint must have NO projection_state_dict'
    for p in model.parameters():
        p.requires_grad_(False)

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_metrics = None
    for split in ('train', 'val', 'test'):
        cache, metrics = build_split(exp, args, model, projection, cli.arm, split, device)
        torch.save(cache, out_dir / f'{split}.pt')
        print(f'[build_u_cache] {cli.arm}/{split}: n={cache["query_start_idx"].numel()}'
             + (f' retmse10={metrics["retmse10"]:.6f} agg={metrics["agg_mse10"]:.6f}' if metrics else ''))
        if split == 'test':
            test_metrics = metrics
    (out_dir / 'stage1_metrics.json').write_text(json.dumps(test_metrics, indent=2))
    print(f'[build_u_cache] wrote {out_dir / "stage1_metrics.json"}')


if __name__ == '__main__':
    main()
