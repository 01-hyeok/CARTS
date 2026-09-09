#!/usr/bin/env python3
"""EXP-ORACLE-SCRATCH01 -- Stage-1 evaluation + Stage-2 retrieval-supply
cache builder. Loads ONE trained (frozen from here on) Stage-1 arm's best
checkpoint and, for train/val/test:

1. Recomputes the full per-split diagnostics (Oracle next-choice mean/median
   rank, rank fraction -- t=1 and t>=2 reported SEPARATELY, containment@1/5/10,
   Choice CE, continuation regret and greedy Set-Oracle aggregate MSE for the
   Set arm) using teacher-forced-by-oracle stepping (same convention as
   training/validation), i.e. still uses the query future -- diagnostic only.

2. Builds the retrieval-supply CACHE Stage-2 consumes: for every query, a
   FREE-RUNNING (no query future, no Oracle target, no teacher cache) Top-K
   retrieval using ONLY past `X_q` and the historical memory -- Individual is
   simply `argmax` over the base score (never uses the future at inference,
   already deterministic); Set repeats the same on-policy argmax decoding
   training used, but with no Oracle-target computation in the loop at all.
   `Y_ret[query] = sum_i alpha_i * Y_i` (alpha from the BASE score, softmax/
   tau_topk, matching production's own aggregation convention) is
   precomputed and cached per split so Stage-2 training never needs to
   re-run Stage-1 (`Exp_Stage1_Relation`/`RelationEncoder` are not even
   imported by the Stage-2 script -- see `tests/test_exp_oracle_scratch01.py`
   test_10).
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from layers.retrieval_metric import RetrievalMetric, oracle_rank_statistics
from models.SequentialSetRetriever import SetConditioner
from scripts.eval_firstanchor_diag import _ndcg_at_k
from scripts.train_margutil01 import build_experiment, memory_value
from scripts.train_oracle_choice01 import oracle_choice_step_loss
from scripts.train_oracle_scratch01 import base_score, encode_raw
from utils.dense_utility import candidate_weights, dense_utility
from utils.oracle_intervention import select_greedy_weighted_set


@torch.no_grad()
def free_running_topk(z_q, E, cand_mask, target, set_conditioner, metric, k):
    """NO query future, NO Oracle target -- pure inference. Individual:
    static top-k by base score. Set: repeated on-policy argmax, t=1
    bypassing the conditioner exactly as in training."""
    b_i = base_score(z_q, E, metric)
    if target == 'individual':
        neg_inf = torch.finfo(b_i.dtype).min / 4
        masked = b_i.masked_fill(~cand_mask, neg_inf)
        picks = masked.topk(k, dim=-1).indices
        return picks, b_i
    selected_mask = torch.zeros_like(cand_mask)
    picks = []
    for t in range(k):
        if t == 0:
            s_hat = b_i
        else:
            prefix = torch.stack(picks, dim=1)
            m = E[prefix].mean(dim=1)
            h_t = set_conditioner(z_q, m)
            s_hat = base_score(h_t, E, metric)
        valid_now = cand_mask & ~selected_mask
        s_masked = s_hat.masked_fill(~valid_now, float('-inf'))
        nxt = s_masked.argmax(dim=-1, keepdim=True)
        picks.append(nxt.squeeze(-1))
        selected_mask = selected_mask.scatter(1, nxt, True)
    return torch.stack(picks, dim=1), b_i


def load_arm(reference_ckpt, ckpt_path, device):
    ck = torch.load(ckpt_path, map_location='cpu')
    target, scorer = ck['target'], ck['scorer']
    overrides = {'is_training': 0, 'model_id': 'eval_scratch01', 'des': 'eval',
                 'checkpoints': '/tmp/exp_oracle_scratch01_eval', 'stage1_retrieval_metric': 'cosine',
                 'seed': int(ck['args'].get('seed', 0)), 'pred_len': int(ck['args']['pred_len']),
                 'seq_len': int(ck['args']['seq_len'])}
    exp, args = build_experiment(reference_ckpt, overrides)
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.load_state_dict(ck['model_state_dict'], strict=True)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    metric = None
    if scorer == 'asymmetric':
        metric = RetrievalMetric(kind='asymmetric', dim=int(args.d_model), output='cosine', layer_norm=False).to(device)
        metric.load_state_dict(ck['metric_state_dict'])
        metric.eval()
        for p in metric.parameters():
            p.requires_grad_(False)
    set_conditioner = None
    if target == 'set':
        set_conditioner = SetConditioner(int(args.d_model)).to(device)
        set_conditioner.load_state_dict(ck['set_conditioner_state_dict'])
        set_conditioner.eval()
        for p in set_conditioner.parameters():
            p.requires_grad_(False)
    return exp, args, model, set_conditioner, metric, target, scorer, ck


@torch.no_grad()
def diagnostics_for_split(exp, args, model, set_conditioner, metric, target, split, channels,
                           k, tau_topk, tau_choice, chunk_size, n_queries):
    _, loader = exp._get_data(flag=split, shuffle=False)
    device = exp.device
    step_rows = {t: {'rank': [], 'rank_frac': [], 'top1': [], 'top5': [], 'top10': [], 'ndcg10': []}
                 for t in range(k)}
    step_rows_t1 = step_rows[0]
    ce_sum, ce_n = 0.0, 0.0
    regret_sum, regret_n = 0.0, 0.0
    final_agg_se, final_agg_n = 0.0, 0.0
    greedy_oracle_se, greedy_oracle_n = 0.0, 0.0
    rows_seen = 0

    for batch_x, batch_y, batch_start_idx in loader:
        if rows_seen >= n_queries:
            break
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, counts = exp._candidate_mask(batch_start_idx)
        for c in channels:
            E = encode_raw(model, exp.memory_x, c)
            z_q = encode_raw(model, batch_x, c)
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            futures = memory_c + offset_c.view(-1, 1, 1)
            query_future = batch_y[:, :, c]
            b_i = base_score(z_q, E, metric)
            w_base = candidate_weights(b_i, cand_mask, tau_topk)

            if target == 'individual':
                d_i = ((futures - query_future.unsqueeze(1)) ** 2).mean(dim=-1)
                u_target_const = -d_i
                selected_mask = torch.zeros_like(cand_mask)
                for t in range(k):
                    valid_now = cand_mask & ~selected_mask
                    loss_t, diag_t = oracle_choice_step_loss(b_i, u_target_const, valid_now, tau_choice)
                    stats = oracle_rank_statistics(
                        b_i, u_target_const.masked_fill(~valid_now, float(torch.finfo(b_i.dtype).min / 4)).argmax(-1, keepdim=True),
                        valid_now, oracle_valid=valid_now.any(dim=-1))
                    ce_sum += float(loss_t.detach()) * batch_x.size(0)
                    ce_n += batch_x.size(0)
                    frac = float(stats['oracle_top10_rank_fraction'])
                    rank = float(stats['oracle_top10_mean_rank'])
                    if frac == frac:
                        step_rows[t]['rank_frac'].append(frac)
                        step_rows[t]['rank'].append(rank)
                        step_rows[t]['top1'].append(diag_t['top1_acc'])
                        step_rows[t]['top5'].append(diag_t['top5_acc'])
                        step_rows[t]['top10'].append(diag_t['top10_acc'])
                    nxt = u_target_const.masked_fill(~valid_now, float(torch.finfo(b_i.dtype).min / 4)).argmax(-1, keepdim=True)
                    selected_mask = selected_mask.scatter(1, nxt, True)
                picks = selected_mask.nonzero(as_tuple=False)  # unused downstream here
            else:
                selected_mask = torch.zeros_like(cand_mask)
                picks_list = []
                for t in range(k):
                    if t == 0:
                        s_hat = b_i
                    else:
                        prefix = torch.stack(picks_list, dim=1)
                        m = E[prefix].mean(dim=1)
                        h_t = set_conditioner(z_q, m)
                        s_hat = base_score(h_t, E, metric)
                    valid_now = cand_mask & ~selected_mask
                    prefix_now = torch.stack(picks_list, dim=1) if t > 0 else torch.zeros(
                        batch_x.size(0), 0, dtype=torch.long, device=device)
                    a_dense = dense_utility(prefix_now, w_base, futures, query_future, chunk_size=chunk_size)
                    u_target_t = -a_dense
                    loss_t, diag_t = oracle_choice_step_loss(s_hat, u_target_t, valid_now, tau_choice)
                    stats = oracle_rank_statistics(
                        s_hat, u_target_t.masked_fill(~valid_now, float(torch.finfo(s_hat.dtype).min / 4)).argmax(-1, keepdim=True),
                        valid_now, oracle_valid=valid_now.any(dim=-1))
                    ce_sum += float(loss_t.detach()) * batch_x.size(0)
                    ce_n += batch_x.size(0)
                    frac = float(stats['oracle_top10_rank_fraction'])
                    rank = float(stats['oracle_top10_mean_rank'])
                    if frac == frac:
                        step_rows[t]['rank_frac'].append(frac)
                        step_rows[t]['rank'].append(rank)
                        step_rows[t]['top1'].append(diag_t['top1_acc'])
                        step_rows[t]['top5'].append(diag_t['top5_acc'])
                        step_rows[t]['top10'].append(diag_t['top10_acc'])
                    s_masked = s_hat.masked_fill(~valid_now, float('-inf'))
                    nxt = s_masked.argmax(dim=-1, keepdim=True)
                    picks_list.append(nxt.squeeze(-1))
                    selected_mask = selected_mask.scatter(1, nxt, True)
                picks = torch.stack(picks_list, dim=1)

                # continuation regret at the final step + weighted aggregate MSE
                a_dense_final = dense_utility(picks, w_base, futures, query_future, chunk_size=chunk_size)
                w_pick = w_base.gather(1, picks)
                alpha = w_pick / w_pick.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                y_ret = (alpha.unsqueeze(-1) * futures.gather(
                    1, picks.unsqueeze(-1).expand(-1, -1, futures.size(-1)))).sum(dim=1)
                agg_se = (y_ret - query_future).pow(2).mean(dim=-1)
                final_agg_se += float(agg_se.sum())
                final_agg_n += batch_x.size(0)

                greedy_picks = select_greedy_weighted_set(futures, query_future, b_i.detach(), cand_mask, k, tau_topk)
                w_pick_g = w_base.gather(1, greedy_picks)
                alpha_g = w_pick_g / w_pick_g.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                y_ret_g = (alpha_g.unsqueeze(-1) * futures.gather(
                    1, greedy_picks.unsqueeze(-1).expand(-1, -1, futures.size(-1)))).sum(dim=1)
                greedy_oracle_se += float((y_ret_g - query_future).pow(2).mean(dim=-1).sum())
                greedy_oracle_n += batch_x.size(0)

        rows_seen += batch_x.size(0)

    def _avg(lst):
        return sum(lst) / len(lst) if lst else float('nan')

    per_step = {t: {kk: _avg(vv) for kk, vv in step_rows[t].items() if kk != 'ndcg10'} for t in range(k)}
    t1 = per_step[0]
    tgeq2_vals = {kk: _avg(sum((step_rows[t][kk] for t in range(1, k)), [])) for kk in ('rank', 'rank_frac', 'top1', 'top5', 'top10')}
    out = {
        'choice_ce_mean': ce_sum / max(ce_n, 1),
        't1': t1, 'tgeq2': tgeq2_vals, 'per_step': per_step,
        'n_rows_evaluated': rows_seen,
    }
    if target == 'set':
        out['final_weighted_aggregate_mse'] = final_agg_se / max(final_agg_n, 1)
        out['greedy_set_oracle_aggregate_mse'] = greedy_oracle_se / max(greedy_oracle_n, 1)
    return out


@torch.no_grad()
def build_retrieval_cache(exp, args, model, set_conditioner, metric, target, split, channels,
                           k, tau_topk, chunk_size, device):
    _, loader = exp._get_data(flag=split, shuffle=False)
    x_rows, yq_rows, yret_rows = [], [], []
    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, counts = exp._candidate_mask(batch_start_idx)
        yq_c, yret_c = [], []
        for c in channels:
            E = encode_raw(model, exp.memory_x, c)
            z_q = encode_raw(model, batch_x, c)
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            futures = memory_c + offset_c.view(-1, 1, 1)
            query_future = batch_y[:, :, c]
            picks, b_i = free_running_topk(z_q, E, cand_mask, target, set_conditioner, metric, k)
            w_base = candidate_weights(b_i, cand_mask, tau_topk)
            w_pick = w_base.gather(1, picks)
            alpha = w_pick / w_pick.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            y_ret = (alpha.unsqueeze(-1) * futures.gather(
                1, picks.unsqueeze(-1).expand(-1, -1, futures.size(-1)))).sum(dim=1)
            yq_c.append(query_future)
            yret_c.append(y_ret)
        x_rows.append(batch_x.cpu())
        yq_rows.append(torch.stack(yq_c, dim=-1).cpu())
        yret_rows.append(torch.stack(yret_c, dim=-1).cpu())
    return {'batch_x': torch.cat(x_rows, dim=0), 'Y_q': torch.cat(yq_rows, dim=0), 'Y_ret': torch.cat(yret_rows, dim=0)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--stage1_checkpoint', required=True)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--n_diag_queries', type=int, default=500)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--cache_dir', required=True)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp, args, model, set_conditioner, metric, target, scorer, ck = load_arm(cli.reference_ckpt, cli.stage1_checkpoint, device)
    exp._ensure_memory()
    channels = list(model.target_channels())
    k = int(cli.top_k)
    tau_topk = float(args.tau_topk)
    tau_choice = float(ck.get('tau_choice', tau_topk))

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cli.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    report = {'target': target, 'scorer': scorer, 'pred_len': int(args.pred_len),
              'best_epoch': ck.get('epoch'), 'val_rank_fraction_overall': ck.get('val_rank_fraction_overall')}
    for split in ('train', 'val', 'test'):
        print(f'[eval_oracle_scratch01] {target}/{scorer} split={split}: diagnostics...')
        diag = diagnostics_for_split(exp, args, model, set_conditioner, metric, target, split, channels,
                                      k, tau_topk, tau_choice, cli.chunk_size, cli.n_diag_queries)
        report[split] = diag
        print(f'  choice_ce={diag["choice_ce_mean"]:.4f} t1_rank_frac={diag["t1"]["rank_frac"]:.5f} '
              f'tgeq2_rank_frac={diag["tgeq2"]["rank_frac"]:.5f}')

        print(f'[eval_oracle_scratch01] {target}/{scorer} split={split}: retrieval cache (free-running)...')
        cache = build_retrieval_cache(exp, args, model, set_conditioner, metric, target, split, channels,
                                       k, tau_topk, cli.chunk_size, device)
        torch.save(cache, cache_dir / f'{split}.pt')
        print(f'  cached n={cache["Y_q"].size(0)} -> {cache_dir / f"{split}.pt"}')

    with open(out_dir / 'diagnostics.json', 'w') as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f'[eval_oracle_scratch01] diagnostics written to {out_dir / "diagnostics.json"}')


if __name__ == '__main__':
    main()
