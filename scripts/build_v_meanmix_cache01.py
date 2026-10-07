#!/usr/bin/env python3
"""TRACK-V-MEANMIX-INFERENCE01 -- retrieval cache builder that replaces
the canonical `round_robin_topk_selection` inference rule with
`mean_mixture_topk_selection` (`utils/mean_mixture_selection.py`) --
the SAME probability-mixture that V1/V2/V5's own training loss already
computes (`p_bar = mean_h softmax(scores_h/tau_s)`,
`L = KL(p_T || p_bar)`), so training and inference finally match.

NEVER retrains anything -- loads the EXISTING canonical
TRACK-V-MULTIQUERY-GENERALIZATION01 checkpoint read-only (V0 via
`train_j_shared_encoder_drift01.py`'s TRUE-Original-KL score path, no
SlotHeads; V1/V2/V5 via `train_t_pure_multislot01.py`'s
`compute_scores_full_grad` + SlotHeads). Full candidate N throughout,
same candidate mask, same teacher, same cache schema as the historical
cache builders (`relation_outputs`/`query_start_idx`/`D_per_query`/
`C_per_query`) so `train_r_stage2_lambda01.py` is reused UNMODIFIED.

Also computes, per channel, the Top-10 overlap fraction between the
new mean-mixture selection and the canonical round-robin selection on
the SAME scores (spec section 13) -- always 1.0 for V0/V1 by
construction (verified separately in `tests/test_v_meanmix_cache01.py`),
generally <1.0 for V2/V5.
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
from scripts.build_t2_true_original_kl_cache01 import compute_scores_true_original_kl
from scripts.train_factorial_e2e01 import individual_utility_memsafe
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_margutil01 import build_experiment, memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k
from scripts.train_t_pure_multislot01 import compute_scores_full_grad, round_robin_topk_selection
from utils.full_candidate_bank import compute_scores_full_grad_channel_first
from utils.mean_mixture_selection import mean_mixture_topk_selection

TOP_K = 10


@torch.no_grad()
def build_split(exp, args, model, slot_heads, arm, split, device, tau_s, chunk_size=4096):
    channels = list(range(int(args.enc_in)))
    _, loader = exp._get_data(flag=split, shuffle=False)
    all_start, all_R, all_D, all_C, all_Agg, all_Recall, all_NDCG, all_ret = [], [], [], [], [], [], [], []
    all_D_pq, all_C_pq, all_overlap = [], [], []

    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, counts = exp._candidate_mask(batch_start_idx)
        bsz = batch_x.size(0)
        rel_out = torch.zeros(bsz, args.pred_len, len(channels), device=device)
        D_pc = torch.zeros(bsz, len(channels), device=device)
        C_pc = torch.zeros(bsz, len(channels), device=device)
        overlap_pc = torch.zeros(bsz, len(channels), device=device)
        for c in channels:
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            u = individual_utility_memsafe(memory_c, offset_c, query_future, chunk_size)
            d_raw = -u

            if arm == 'V0':
                scores = compute_scores_true_original_kl(model, batch_x, exp.memory_x, c)
            else:
                scores = compute_scores_full_grad_channel_first(model, slot_heads, batch_x, exp.memory_x, c)

            picks_t, _, _ = mean_mixture_topk_selection(scores, cand_mask, tau_s, k=TOP_K)
            rr_picks = round_robin_topk_selection(scores, cand_mask, k=TOP_K)
            ov = torch.tensor([len(set(picks_t[b].tolist()) & set(rr_picks[b].tolist())) / TOP_K
                               for b in range(bsz)], device=device)
            overlap_pc[:, c] = ov

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
        all_overlap.append(overlap_pc.mean(dim=-1).cpu())

    cache = {'query_start_idx': torch.cat(all_start), 'relation_outputs': torch.cat(all_R),
            'D_per_query': torch.cat(all_D_pq), 'C_per_query': torch.cat(all_C_pq),
            'channels': channels, 'pred_len': args.pred_len, 'split': split}
    metrics = None
    if all_D:
        D_cat, C_cat, Agg_cat = torch.cat(all_D), torch.cat(all_C), torch.cat(all_Agg)
        assert torch.allclose(D_cat + C_cat, Agg_cat, atol=1e-3)
        metrics = {'retmse10': float(torch.cat(all_ret).mean()), 'D': float(D_cat.mean()),
                  'C': float(C_cat.mean()), 'agg_mse10': float(Agg_cat.mean()),
                  'recall10': float(torch.cat(all_Recall).mean()), 'ndcg10': float(torch.cat(all_NDCG).mean()),
                  'top10_overlap_vs_roundrobin': float(torch.cat(all_overlap).mean())}
    return cache, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=('V0', 'V1', 'V2', 'V5'))
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--retriever_checkpoint', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--tau_s', type=float, required=True,
                    help='MUST be read from the checkpoint arm\'s own historical config.json -- never a new default')
    ap.add_argument('--out_dir', required=True)
    cli = ap.parse_args()
    num_slots = {'V0': 1, 'V1': 1, 'V2': 2, 'V5': 5}[cli.arm]

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
    for p in model.parameters():
        p.requires_grad_(False)

    slot_heads = None
    if cli.arm == 'V0':
        assert 'slot_heads_state_dict' not in bl, \
            '[ISSUE][ABORT] V0 checkpoint must have NO slot_heads_state_dict'
    else:
        slot_heads = SlotHeads(int(args.d_model), n_slots=num_slots).to(device)
        slot_heads.load_state_dict(bl['slot_heads_state_dict'])
        slot_heads.eval()
        for p in slot_heads.parameters():
            p.requires_grad_(False)

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_metrics = None
    for split in ('train', 'val', 'test'):
        cache, metrics = build_split(exp, args, model, slot_heads, cli.arm, split, device, cli.tau_s)
        torch.save(cache, out_dir / f'{split}.pt')
        print(f'[build_v_meanmix] {cli.arm}/{split}: n={cache["query_start_idx"].numel()}'
             + (f' retmse10={metrics["retmse10"]:.6f} agg={metrics["agg_mse10"]:.6f} '
                f'overlap_vs_RR={metrics["top10_overlap_vs_roundrobin"]:.4f}' if metrics else ''))
        if split == 'test':
            test_metrics = metrics
    (out_dir / 'stage1_metrics.json').write_text(json.dumps(test_metrics, indent=2))
    print(f'[build_v_meanmix] wrote {out_dir / "stage1_metrics.json"}')


if __name__ == '__main__':
    main()
