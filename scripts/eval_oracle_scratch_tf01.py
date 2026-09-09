#!/usr/bin/env python3
"""EXP-ORACLE-SCRATCH-TF01 -- Stage-1 evaluation, split into TWO
deliberately-separate evaluations (spec sections 7-8), never mixed:

1. TEACHER-FORCED diagnostics (`diagnostics_for_split`): the prefix used
   to compute rank/Top-K/Choice-CE at every step is the ORACLE's own
   greedy sequence (Individual: its own step-invariant ordering; Set: the
   SAME Oracle-forced construction training used, `run_sequence_set_
   teacher_forced`'s own prefix). Reports t=1, t>=2, and full per-step
   (t=1..K) breakdowns separately -- never a single pooled average.

2. FREE-RUNNING retrieval-supply cache (`free_running_topk`/
   `build_retrieval_cache`, imported UNCHANGED from `scripts/eval_oracle_
   scratch01.py` -- that script's own free-running functions already do
   exactly what this spec's section 8 requires: Individual = static top-K
   by base score; Set = repeated on-policy argmax, no Oracle/future
   involvement at all). Also reports the free-running final weighted
   aggregate MSE, the TRUE greedy Set-Oracle aggregate MSE (diagnostic,
   uses the query future), and the gap between them.
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
from scripts.eval_oracle_scratch01 import build_retrieval_cache, free_running_topk
from scripts.train_margutil01 import build_experiment, memory_value
from scripts.train_oracle_choice01 import oracle_choice_step_loss
from scripts.train_oracle_scratch01 import base_score, encode_raw
from scripts.train_oracle_scratch_tf01 import run_sequence_set_teacher_forced
from utils.dense_utility import candidate_weights, dense_utility
from utils.oracle_intervention import select_greedy_weighted_set


def load_arm(reference_ckpt, ckpt_path, device):
    ck = torch.load(ckpt_path, map_location='cpu')
    target, scorer = ck['target'], ck['scorer']
    overrides = {'is_training': 0, 'model_id': 'eval_scratch_tf01', 'des': 'eval',
                 'checkpoints': '/tmp/exp_oracle_scratch_tf01_eval', 'stage1_retrieval_metric': 'cosine',
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
def teacher_forced_diagnostics_for_split(exp, args, model, set_conditioner, metric, target, split, channels,
                                          k, tau_topk, tau_choice, chunk_size, n_queries):
    _, loader = exp._get_data(flag=split, shuffle=False)
    device = exp.device
    step_rows = {t: {'rank': [], 'rank_frac': [], 'top1': [], 'top5': [], 'top10': []} for t in range(k)}
    ce_sum, ce_n = 0.0, 0.0
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

            if target == 'individual':
                d_i = ((futures - query_future.unsqueeze(1)) ** 2).mean(dim=-1)
                u_target_const = -d_i
                selected_mask = torch.zeros_like(cand_mask)
                neg_inf = torch.finfo(b_i.dtype).min / 4
                for t in range(k):
                    valid_now = cand_mask & ~selected_mask
                    loss_t, diag_t = oracle_choice_step_loss(b_i, u_target_const, valid_now, tau_choice)
                    i_star = u_target_const.masked_fill(~valid_now, neg_inf).argmax(-1, keepdim=True)
                    stats = oracle_rank_statistics(b_i, i_star, valid_now, oracle_valid=valid_now.any(dim=-1))
                    ce_sum += float(loss_t.detach()) * batch_x.size(0)
                    ce_n += batch_x.size(0)
                    frac = float(stats['oracle_top10_rank_fraction'])
                    if frac == frac:
                        step_rows[t]['rank_frac'].append(frac)
                        step_rows[t]['rank'].append(float(stats['oracle_top10_mean_rank']))
                        step_rows[t]['top1'].append(diag_t['top1_acc'])
                        step_rows[t]['top5'].append(diag_t['top5_acc'])
                        step_rows[t]['top10'].append(diag_t['top10_acc'])
                    selected_mask = selected_mask.scatter(1, i_star, True)
            else:
                w_base = candidate_weights(b_i, cand_mask, tau_topk)
                losses, diags, oracle_picks, a_dense_steps = run_sequence_set_teacher_forced(
                    z_q, E, cand_mask, set_conditioner, metric, w_base, futures, query_future,
                    tau_choice, k, chunk_size)
                selected_mask = torch.zeros_like(cand_mask)
                for t in range(k):
                    valid_now = cand_mask & ~selected_mask
                    if t == 0:
                        s_hat_t = base_score(z_q, E, metric)
                    else:
                        prefix = oracle_picks[:, :t]
                        m = E[prefix].mean(dim=1)
                        h_t = set_conditioner(z_q, m)
                        s_hat_t = base_score(h_t, E, metric)
                    u_target_t = -a_dense_steps[t]
                    stats = oracle_rank_statistics(
                        s_hat_t, oracle_picks[:, t:t + 1], valid_now, oracle_valid=valid_now.any(dim=-1))
                    frac = float(stats['oracle_top10_rank_fraction'])
                    ce_sum += float(losses[t].detach()) * batch_x.size(0)
                    ce_n += batch_x.size(0)
                    if frac == frac:
                        step_rows[t]['rank_frac'].append(frac)
                        step_rows[t]['rank'].append(float(stats['oracle_top10_mean_rank']))
                        step_rows[t]['top1'].append(diags[t]['top1_acc'])
                        step_rows[t]['top5'].append(diags[t]['top5_acc'])
                        step_rows[t]['top10'].append(diags[t]['top10_acc'])
                    nxt = oracle_picks[:, t:t + 1]
                    selected_mask = selected_mask.scatter(1, nxt, True)

        rows_seen += batch_x.size(0)

    def _avg(lst):
        return sum(lst) / len(lst) if lst else float('nan')

    per_step = {t: {kk: _avg(vv) for kk, vv in step_rows[t].items()} for t in range(k)}
    t1 = per_step[0]
    tgeq2_vals = {kk: _avg(sum((step_rows[t][kk] for t in range(1, k)), [])) for kk in ('rank', 'rank_frac', 'top1', 'top5', 'top10')}
    return {'choice_ce_mean': ce_sum / max(ce_n, 1), 't1': t1, 'tgeq2': tgeq2_vals, 'per_step': per_step,
            'n_rows_evaluated': rows_seen}


@torch.no_grad()
def free_running_eval_for_split(exp, args, model, set_conditioner, metric, target, split, channels,
                                 k, tau_topk, chunk_size, device):
    """Free-running (model argmax, no Oracle/future) final weighted
    aggregate MSE; for Set arms also the TRUE greedy Set-Oracle aggregate
    MSE (diagnostic, uses the query future -- reported ONLY as a
    reference ceiling, never fed back into any selection)."""
    _, loader = exp._get_data(flag=split, shuffle=False)
    final_se, final_n = 0.0, 0.0
    oracle_se, oracle_n = 0.0, 0.0
    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, counts = exp._candidate_mask(batch_start_idx)
        valid_query = counts.to(device) >= k
        if not bool(valid_query.any()):
            continue
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
            vq = valid_query
            final_se += float((y_ret - query_future).pow(2).mean(dim=-1)[vq].sum())
            final_n += int(vq.sum())

            if target == 'set':
                greedy_picks = select_greedy_weighted_set(futures, query_future, b_i.detach(), cand_mask, k, tau_topk)
                w_pick_g = w_base.gather(1, greedy_picks)
                alpha_g = w_pick_g / w_pick_g.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                y_ret_g = (alpha_g.unsqueeze(-1) * futures.gather(
                    1, greedy_picks.unsqueeze(-1).expand(-1, -1, futures.size(-1)))).sum(dim=1)
                oracle_se += float((y_ret_g - query_future).pow(2).mean(dim=-1)[vq].sum())
                oracle_n += int(vq.sum())

    out = {'free_running_final_aggregate_mse': final_se / max(final_n, 1)}
    if target == 'set':
        out['greedy_set_oracle_aggregate_mse'] = oracle_se / max(oracle_n, 1)
        out['oracle_gap'] = out['free_running_final_aggregate_mse'] - out['greedy_set_oracle_aggregate_mse']
    return out


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

    report = {'target': target, 'scorer': scorer, 'pred_len': int(args.pred_len), 'teacher_forced': True,
              'best_epoch': ck.get('epoch'), 'val_rank_fraction_overall': ck.get('val_rank_fraction_overall')}
    for split in ('train', 'val', 'test'):
        print(f'[eval_oracle_scratch_tf01] {target}/{scorer} split={split}: TEACHER-FORCED diagnostics...')
        diag = teacher_forced_diagnostics_for_split(exp, args, model, set_conditioner, metric, target, split,
                                                      channels, k, tau_topk, tau_choice, cli.chunk_size, cli.n_diag_queries)
        report[f'{split}_teacher_forced'] = diag
        print(f'  choice_ce={diag["choice_ce_mean"]:.4f} t1_rank_frac={diag["t1"]["rank_frac"]:.5f} '
              f'tgeq2_rank_frac={diag["tgeq2"]["rank_frac"]:.5f} t1_top10={diag["t1"]["top10"]:.4f} '
              f'tgeq2_top10={diag["tgeq2"]["top10"]:.4f}')

        print(f'[eval_oracle_scratch_tf01] {target}/{scorer} split={split}: FREE-RUNNING eval + cache...')
        fr = free_running_eval_for_split(exp, args, model, set_conditioner, metric, target, split, channels,
                                          k, tau_topk, cli.chunk_size, device)
        report[f'{split}_free_running'] = fr
        print(f'  {fr}')
        cache = build_retrieval_cache(exp, args, model, set_conditioner, metric, target, split, channels,
                                       k, tau_topk, cli.chunk_size, device)
        torch.save(cache, cache_dir / f'{split}.pt')
        print(f'  cached n={cache["Y_q"].size(0)} -> {cache_dir / f"{split}.pt"}')

    with open(out_dir / 'diagnostics.json', 'w') as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f'[eval_oracle_scratch_tf01] diagnostics written to {out_dir / "diagnostics.json"}')


if __name__ == '__main__':
    main()
