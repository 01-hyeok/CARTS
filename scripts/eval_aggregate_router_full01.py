#!/usr/bin/env python3
"""TRACK-AGGREGATE-ROUTER-FROZEN01 -- single real-backbone-forward
evaluation pass (per split) that computes EVERY required baseline and
router-inference-mode metric together, to avoid redundant forward
passes: Mean-Mixture, Validation-selected Fixed-Best-Head,
Majority-Head (train-mode winner), Random-Router, R1-Hard, R1-Soft,
R2-Hard, R2-Soft, Aggregate-Oracle, Final-Fusion-Oracle. Loads the
TRAINED R1-best / R2-best router checkpoints (never trains here).
Read-only w.r.t. the frozen backbone; the only parameters loaded with
gradients enabled anywhere in this file are none -- everything runs
under `torch.no_grad()`.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from models.RelationStage2 import BaseForecastHead
from scripts.train_j_shared_encoder_drift01 import build_model
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_margutil01 import memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k
from scripts.train_retriever_pool01 import CandidatePoolCache
from scripts.train_t_pure_multislot01 import round_robin_topk_selection
from utils.aggregate_router01 import (
    AggregateRouter, hard_routed_topk, oracle_gap_recovered_pct, router_entropy, soft_routed_topk,
    top2_accuracy, winner_accuracy,
)
from utils.candidate_pool import gather_candidate_values, pooled_future_mse
from utils.full_candidate_bank import compute_scores_pool_channel_first_grad, encode_raw_channel_first
from utils.mean_mixture_selection import mean_mixture_topk_selection

NUM_SLOTS = 5
TOP_K = 10


@torch.no_grad()
def full_pass(model, slot_heads, base, lambda_arm, router_r1, router_r2, random_head_fixed, majority_head,
             fixed_head_per_channel, exp, args, channels, device, loader, pool_cache, tau_s, top_k=TOP_K):
    """One pass over `loader`. `fixed_head_per_channel`: [C] tensor
    (validation-selected, Aggregate criterion, per TRACK-HEAD-UTILITY-ALIGNMENT-CORRECTION01).
    `majority_head`: [C] tensor (train-split mode of the per-channel
    Individual winner). `random_head_fixed`: [C] tensor (one fixed
    random draw per channel, NOT re-randomized every query -- a
    deterministic random baseline).  Returns a dict of per-method
    accumulators."""
    methods = ('mean_mixture', 'fixed_head', 'majority_head', 'random_router', 'r1_hard', 'r1_soft',
              'r2_hard', 'r2_soft', 'aggregate_oracle')
    sums = {m: {'agg_mse': 0.0, 'retmse10': 0.0, 'recall10': 0.0, 'ndcg10': 0.0} for m in methods}
    final_oracle_sum, n = 0.0, 0
    router_metrics_sum = {'r1_winner_acc': 0.0, 'r1_top2_acc': 0.0, 'r1_entropy': 0.0, 'r1_max_head_usage_count': torch.zeros(5),
                          'r2_winner_acc': 0.0, 'r2_top2_acc': 0.0, 'r2_entropy': 0.0, 'r2_max_head_usage_count': torch.zeros(5)}

    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        bsz = batch_x.size(0)
        b_q_full = base(batch_x) + batch_x[:, -1:, :].detach()  # [B,pred_len,C]
        for c in channels:
            scores, pool_valid_mask, pool_idx_global = compute_scores_pool_channel_first_grad(
                model, slot_heads, batch_x, exp, c, pool_cache, batch_start_idx, device)
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            pooled_memory_c = gather_candidate_values(memory_c, pool_idx_global)
            d_pool = pooled_future_mse(pooled_memory_c, offset_c, query_future)
            p_h = torch.softmax(scores / tau_s, dim=-1)  # [B,5,M]
            oracle_idx = stable_topk_indices(d_pool, top_k, largest=False)

            def eval_picks(picks):
                ind_mse_i = d_pool.gather(1, picks)
                y_sel = pooled_memory_c.gather(1, picks.unsqueeze(-1).expand(-1, -1, pooled_memory_c.size(-1))) \
                    + offset_c.view(-1, 1, 1)
                agg_pred = y_sel.mean(dim=1)
                agg_mse = ((agg_pred - query_future) ** 2).mean(dim=-1)
                retmse10 = ind_mse_i.mean(dim=-1)
                recall10 = recall_at_k(picks, oracle_idx, top_k)
                ndcg10 = ndcg_at_k(picks, d_pool, pool_valid_mask, top_k)
                return agg_mse, retmse10, recall10, ndcg10, agg_pred

            z_qc = encode_raw_channel_first(model, batch_x, c)  # [B,d]

            # ---- Mean-Mixture ----
            mm_idx, _, _ = mean_mixture_topk_selection(scores, pool_valid_mask, tau_s, k=top_k)
            agg, ret, rec, ndcg, _ = eval_picks(mm_idx)
            for k_, v_ in zip(('agg_mse', 'retmse10', 'recall10', 'ndcg10'), (agg, ret, rec, ndcg)):
                sums['mean_mixture'][k_] += float(v_.sum())

            # ---- Fixed-Best-Head (val-selected, per channel) ----
            fh = int(fixed_head_per_channel[c])
            fh_idx = p_h[:, fh, :].topk(top_k, dim=-1).indices
            agg, ret, rec, ndcg, _ = eval_picks(fh_idx)
            for k_, v_ in zip(('agg_mse', 'retmse10', 'recall10', 'ndcg10'), (agg, ret, rec, ndcg)):
                sums['fixed_head'][k_] += float(v_.sum())

            # ---- Majority-Head (train-mode winner, per channel) ----
            mh = int(majority_head[c])
            mh_idx = p_h[:, mh, :].topk(top_k, dim=-1).indices
            agg, ret, rec, ndcg, _ = eval_picks(mh_idx)
            for k_, v_ in zip(('agg_mse', 'retmse10', 'recall10', 'ndcg10'), (agg, ret, rec, ndcg)):
                sums['majority_head'][k_] += float(v_.sum())

            # ---- Random-Router (fixed random head per channel) ----
            rh = int(random_head_fixed[c])
            rh_idx = p_h[:, rh, :].topk(top_k, dim=-1).indices
            agg, ret, rec, ndcg, _ = eval_picks(rh_idx)
            for k_, v_ in zip(('agg_mse', 'retmse10', 'recall10', 'ndcg10'), (agg, ret, rec, ndcg)):
                sums['random_router'][k_] += float(v_.sum())

            # ---- Aggregate Oracle (per query, per channel, argmin U_agg) ----
            head_topk_idx_allheads = torch.stack([p_h[:, h, :].topk(top_k, dim=-1).indices for h in range(NUM_SLOTS)], dim=1)
            u_agg_allheads = torch.zeros(bsz, NUM_SLOTS, device=device)
            r_h_allheads = torch.zeros(bsz, NUM_SLOTS, query_future.size(-1), device=device)
            for h in range(NUM_SLOTS):
                idx_h = head_topk_idx_allheads[:, h, :]
                y_sel_h = pooled_memory_c.gather(1, idx_h.unsqueeze(-1).expand(-1, -1, pooled_memory_c.size(-1))) \
                    + offset_c.view(-1, 1, 1)
                r_h = y_sel_h.mean(dim=1)
                r_h_allheads[:, h, :] = r_h
                u_agg_allheads[:, h] = ((r_h - query_future) ** 2).mean(dim=-1)
            oracle_winner = u_agg_allheads.argmin(dim=-1)
            oracle_idx_picks = head_topk_idx_allheads[torch.arange(bsz, device=device), oracle_winner]
            agg, ret, rec, ndcg, _ = eval_picks(oracle_idx_picks)
            for k_, v_ in zip(('agg_mse', 'retmse10', 'recall10', 'ndcg10'), (agg, ret, rec, ndcg)):
                sums['aggregate_oracle'][k_] += float(v_.sum())

            # ---- Final-Fusion Oracle (per query, argmin U_final over heads, this channel only fused) ----
            b_c = b_q_full[:, :, c]
            y_hat_h = b_c.unsqueeze(1) + lambda_arm * (r_h_allheads - b_c.unsqueeze(1))  # [B,5,pred_len]
            u_final_allheads = ((y_hat_h - query_future.unsqueeze(1)) ** 2).mean(dim=-1)
            final_oracle_sum += float(u_final_allheads.min(dim=-1).values.sum())

            # ---- R1 / R2 router (Hard + Soft inference) ----
            r1_logits = router_r1(z_qc)
            r2_logits = router_r2(z_qc)

            u_agg_star = u_agg_allheads.argmin(dim=-1)
            router_metrics_sum['r1_winner_acc'] += winner_accuracy(r1_logits, u_agg_star) * bsz
            router_metrics_sum['r1_top2_acc'] += top2_accuracy(r1_logits, u_agg_star) * bsz
            router_metrics_sum['r1_entropy'] += float(router_entropy(r1_logits).sum())
            router_metrics_sum['r1_max_head_usage_count'] += torch.bincount(r1_logits.argmax(-1).cpu(), minlength=5).float()
            router_metrics_sum['r2_winner_acc'] += winner_accuracy(r2_logits, u_agg_star) * bsz
            router_metrics_sum['r2_top2_acc'] += top2_accuracy(r2_logits, u_agg_star) * bsz
            router_metrics_sum['r2_entropy'] += float(router_entropy(r2_logits).sum())
            router_metrics_sum['r2_max_head_usage_count'] += torch.bincount(r2_logits.argmax(-1).cpu(), minlength=5).float()

            r1h_idx, _ = hard_routed_topk(r1_logits, p_h, k=top_k)
            agg, ret, rec, ndcg, _ = eval_picks(r1h_idx)
            for k_, v_ in zip(('agg_mse', 'retmse10', 'recall10', 'ndcg10'), (agg, ret, rec, ndcg)):
                sums['r1_hard'][k_] += float(v_.sum())
            r1s_idx, _ = soft_routed_topk(r1_logits, p_h, k=top_k)
            agg, ret, rec, ndcg, _ = eval_picks(r1s_idx)
            for k_, v_ in zip(('agg_mse', 'retmse10', 'recall10', 'ndcg10'), (agg, ret, rec, ndcg)):
                sums['r1_soft'][k_] += float(v_.sum())
            r2h_idx, _ = hard_routed_topk(r2_logits, p_h, k=top_k)
            agg, ret, rec, ndcg, _ = eval_picks(r2h_idx)
            for k_, v_ in zip(('agg_mse', 'retmse10', 'recall10', 'ndcg10'), (agg, ret, rec, ndcg)):
                sums['r2_hard'][k_] += float(v_.sum())
            r2s_idx, _ = soft_routed_topk(r2_logits, p_h, k=top_k)
            agg, ret, rec, ndcg, _ = eval_picks(r2s_idx)
            for k_, v_ in zip(('agg_mse', 'retmse10', 'recall10', 'ndcg10'), (agg, ret, rec, ndcg)):
                sums['r2_soft'][k_] += float(v_.sum())

        n += bsz

    denom = max(n * len(channels), 1)
    out = {m: {k_: v_ / denom for k_, v_ in d.items()} for m, d in sums.items()}
    out['final_fusion_oracle_mse'] = final_oracle_sum / denom
    out['router_metrics'] = {
        'r1_winner_acc': router_metrics_sum['r1_winner_acc'] / denom,
        'r1_top2_acc': router_metrics_sum['r1_top2_acc'] / denom,
        'r1_entropy': router_metrics_sum['r1_entropy'] / denom,
        'r1_max_head_usage': float(router_metrics_sum['r1_max_head_usage_count'].max() / (n * len(channels))),
        'r2_winner_acc': router_metrics_sum['r2_winner_acc'] / denom,
        'r2_top2_acc': router_metrics_sum['r2_top2_acc'] / denom,
        'r2_entropy': router_metrics_sum['r2_entropy'] / denom,
        'r2_max_head_usage': float(router_metrics_sum['r2_max_head_usage_count'].max() / (n * len(channels))),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--ds', required=True)
    ap.add_argument('--h', type=int, required=True)
    ap.add_argument('--backbone', required=True, choices=('V5', 'Hard'))
    ap.add_argument('--retriever_checkpoint', required=True)
    ap.add_argument('--base_checkpoint', required=True)
    ap.add_argument('--arm_stage2_metrics', required=True, help='frozen lambda source (ALLH01 stage2/metrics.json)')
    ap.add_argument('--candidate_pool_cache', required=True)
    ap.add_argument('--candidate_pool_size', type=int, default=100)
    ap.add_argument('--candidate_mask', default='raft')
    ap.add_argument('--patch_len', type=int, default=16)
    ap.add_argument('--tau_s', type=float, required=True)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--features_dir', required=True)
    ap.add_argument('--oracle_headroom_json', required=True, help='Phase1 oracle_headroom.json for fixed_heads_per_channel')
    ap.add_argument('--r1_checkpoint', required=True)
    ap.add_argument('--r2_checkpoint', required=True)
    ap.add_argument('--out_dir', required=True)
    cli = ap.parse_args()
    cli.init_seed = cli.seed
    cli.pred_len = cli.h
    cli.seq_len = cli.h
    cli.top_k = TOP_K

    t0 = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    base = BaseForecastHead(seq_len=cli.h, pred_len=cli.h, channels=int(args.enc_in),
                            mode='per_channel_linear').to(device)
    bbl = torch.load(cli.base_checkpoint, map_location=device)
    base.load_state_dict(bbl['model_state_dict'])
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    lambda_arm = float(json.load(open(cli.arm_stage2_metrics))['lambda'])

    router_r1 = AggregateRouter(d_model).to(device)
    r1_bl = torch.load(cli.r1_checkpoint, map_location=device)
    router_r1.load_state_dict(r1_bl['router_state_dict'])
    router_r1.eval()
    router_r2 = AggregateRouter(d_model).to(device)
    r2_bl = torch.load(cli.r2_checkpoint, map_location=device)
    router_r2.load_state_dict(r2_bl['router_state_dict'])
    router_r2.eval()

    oh = json.load(open(cli.oracle_headroom_json))
    fixed_head_per_channel = torch.tensor(oh['per_channel']['fixed_heads_per_channel'], dtype=torch.long)

    train_f = torch.load(Path(cli.features_dir) / 'train.pt', map_location='cpu')
    U_ind_tr = train_f['U_ind']  # [N,C,5]
    majority_head = U_ind_tr.argmin(dim=-1).mode(dim=0).values  # [C], mode over queries

    g = torch.Generator().manual_seed(cli.seed)
    random_head_fixed = torch.randint(0, NUM_SLOTS, (len(channels),), generator=g)

    expected_meta = {'seq_len': cli.seq_len, 'pred_len': cli.pred_len, 'candidate_mask': cli.candidate_mask,
                     'candidate_pool_size': cli.candidate_pool_size, 'candidate_pool_metric': 'delta_last_cosine'}
    cache_dir = Path(cli.candidate_pool_cache)
    pool_cache_test = CandidatePoolCache(cache_dir / 'test.pt', expected_meta)
    _, test_loader = exp._get_data(flag='test', shuffle=False)

    results = full_pass(model, slot_heads, base, lambda_arm, router_r1, router_r2, random_head_fixed,
                        majority_head, fixed_head_per_channel, exp, args, channels, device, test_loader,
                        pool_cache_test, cli.tau_s)

    oracle_gap_recovered = {}
    for name in ('r1_hard', 'r1_soft', 'r2_hard', 'r2_soft', 'mean_mixture'):
        oracle_gap_recovered[name] = oracle_gap_recovered_pct(
            results['fixed_head']['agg_mse'], results[name]['agg_mse'], results['aggregate_oracle']['agg_mse'])

    out = {'test_methods': results, 'oracle_gap_recovered_pct': oracle_gap_recovered,
          'lambda_arm_frozen': lambda_arm, 'fixed_head_per_channel': fixed_head_per_channel.tolist(),
          'majority_head_per_channel': majority_head.tolist(), 'random_head_per_channel': random_head_fixed.tolist(),
          'wall_seconds': time.time() - t0}
    (out_dir / 'full_evaluation.json').write_text(json.dumps(out, indent=2, default=str))
    print(f'[eval_router_full] {cli.ds}_{cli.h}/{cli.backbone} done. '
         f'mean_mixture_agg={results["mean_mixture"]["agg_mse"]:.5f} '
         f'r1_hard_agg={results["r1_hard"]["agg_mse"]:.5f} r1_soft_agg={results["r1_soft"]["agg_mse"]:.5f} '
         f'r2_hard_agg={results["r2_hard"]["agg_mse"]:.5f} r2_soft_agg={results["r2_soft"]["agg_mse"]:.5f} '
         f'fixed_agg={results["fixed_head"]["agg_mse"]:.5f} oracle_agg={results["aggregate_oracle"]["agg_mse"]:.5f} '
         f'wall={time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
