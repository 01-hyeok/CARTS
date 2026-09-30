#!/usr/bin/env python3
"""TRACK-R-FINAL-METHOD-GENERALIZATION01 -- generalized (dataset x
horizon x seed) retrieval cache builder for the three retriever arms:

  cosine: NO TRAINING. Cosine similarity computed directly on the raw
    delta-last input window (`x[:,:,c] - x[:,-1,c]`, the SAME
    `relation_input_space='delta_last'` convention used everywhere this
    session) -- no learned encoder at all. Future-blind by construction
    (uses only the input history). Documented choice (AUDIT.md): the
    spec's "raw cosine retrieval" is ambiguous between (a) an untrained
    encoder's embeddings or (b) direct cosine on raw windows; (b) is used
    here as the more standard, meaningful non-learned content-based
    baseline (an untrained encoder's cosine would be close to noise).
  j1: J1's own architecture (`train_j2_key_update_decomposition01.py`),
    single score vector, `arm_score`+`stable_topk_indices`, future-blind
    hard Top-10 -- UNMODIFIED selection code.
  m2: M2's own architecture (`train_m_relevance_constrained_multislot01.py`),
    10-slot scores, `compute_scores`+`hard_unique_selection`, future
    -blind hard Top-10 -- UNMODIFIED selection code.

All three produce the SAME Uniform aggregate `R = mean_i(Y_i)` and the
SAME cache schema (`{train,val,test}.pt`: `relation_outputs`
[N,H] absolute-space aggregate, `query_start_idx`), plus Stage1 metrics
(retMSE/D/C/AggMSE/Recall/NDCG, `Agg=D+C` identity asserted) saved to
`stage1_metrics.json`.
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
from scripts.train_factorial_e2e01 import arm_score, encode_raw, individual_utility_memsafe
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads, compute_scores, hard_unique_selection
from scripts.train_margutil01 import build_experiment, memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k

TOP_K = 10


def cosine_scores(batch_x, memory_x, c, cand_mask):
    """NO TRAINING, no learned encoder: cosine similarity directly on the
    raw delta-last window for channel c."""
    q = batch_x[:, :, c] - batch_x[:, -1:, c]  # [B, seq_len], delta-last
    k = memory_x[:, :, c] - memory_x[:, -1:, c]  # [N, seq_len]
    q = torch.nn.functional.normalize(q, dim=-1)
    k = torch.nn.functional.normalize(k, dim=-1)
    s = q @ k.t()
    return s.masked_fill(~cand_mask, float('-inf'))


@torch.no_grad()
def build_split(retriever, exp, args, model, slot_heads, split, device, chunk_size=4096):
    channels = list(range(int(args.enc_in)))
    _, loader = exp._get_data(flag=split, shuffle=False)
    all_start, all_R, all_D, all_C, all_Agg, all_Recall, all_NDCG, all_ret = [], [], [], [], [], [], [], []

    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, counts = exp._candidate_mask(batch_start_idx)
        bsz = batch_x.size(0)
        rel_out = torch.zeros(bsz, args.pred_len, len(channels), device=device)  # [B, H, C], matches batch_y layout
        for c in channels:
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            u = individual_utility_memsafe(memory_c, offset_c, query_future, chunk_size)
            d_raw = -u

            if retriever == 'cosine':
                s = cosine_scores(batch_x, exp.memory_x, c, cand_mask)
                picks_t = stable_topk_indices(s, TOP_K, largest=True)
            elif retriever in ('j1', 'original_kl'):
                z_q = encode_raw(model, batch_x, c)
                z_k = encode_raw(model, exp.memory_x, c)
                s = arm_score(z_q, z_k, None).masked_fill(~cand_mask, float('-inf'))
                picks_t = stable_topk_indices(s, TOP_K, largest=True)
            else:  # m2
                scores = compute_scores(model, slot_heads, batch_x, exp.memory_x, c)
                picks_t = hard_unique_selection(scores, cand_mask, s=TOP_K)

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
            all_ret.append((ind_mse_i.mean(dim=-1)).cpu())

        all_start.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx)
                         else torch.as_tensor(batch_start_idx))
        all_R.append(rel_out.cpu())

    cache = {'query_start_idx': torch.cat(all_start), 'relation_outputs': torch.cat(all_R),
            'channels': channels, 'pred_len': args.pred_len, 'retriever': retriever, 'split': split}
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
    ap.add_argument('--retriever', required=True, choices=('cosine', 'j1', 'm2', 'original_kl'))
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--retriever_checkpoint', default=None, help='required for j1/m2')
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
    slot_heads = None

    if cli.retriever in ('j1', 'm2', 'original_kl'):
        assert cli.retriever_checkpoint, '--retriever_checkpoint required for j1/m2'
        bl = torch.load(cli.retriever_checkpoint, map_location=device)
        model.load_state_dict(bl['model_state_dict'])
        if cli.retriever == 'm2':
            slot_heads = SlotHeads(int(args.d_model)).to(device)
            slot_heads.load_state_dict(bl['slot_heads_state_dict'])
            slot_heads.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_metrics = None
    for split in ('train', 'val', 'test'):
        cache, metrics = build_split(cli.retriever, exp, args, model, slot_heads, split, device)
        torch.save(cache, out_dir / f'{split}.pt')
        print(f'[build_r_cache] {cli.retriever}/{split}: n={cache["query_start_idx"].numel()}'
             + (f' retmse10={metrics["retmse10"]:.6f} agg={metrics["agg_mse10"]:.6f}' if metrics else ''))
        if split == 'test':
            test_metrics = metrics
    (out_dir / 'stage1_metrics.json').write_text(json.dumps(test_metrics, indent=2))
    print(f'[build_r_cache] wrote {out_dir / "stage1_metrics.json"}')


if __name__ == '__main__':
    main()
