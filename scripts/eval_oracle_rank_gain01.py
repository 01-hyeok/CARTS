#!/usr/bin/env python3
"""EXP-ORACLE-RANK-GAIN01 evaluation. For ONE checkpoint (Epoch0 OR Best --
called twice per arm, with the EXACT SAME code path both times, so
Epoch0-vs-Best is a same-protocol comparison per spec section 14):

1. Per-step (t=1..K) diagnostics, walking the SAME prefix policy the arm
   was trained with (TF: Oracle-forced prefix; on-policy: this
   checkpoint's OWN argmax prefix, re-derived fresh at eval time -- never
   assumed from training) -- rank/rank_fraction/Top-1/5/10/Choice-CE, per
   step AND aggregated as t=1 vs t>=2.

2. FREE-RUNNING retrieval-supply cache + final aggregate MSE (`scripts.
   eval_oracle_scratch01.free_running_topk`/`build_retrieval_cache`,
   imported UNCHANGED -- always model-argmax, no future/Oracle
   involvement, identical regardless of training prefix policy) plus, for
   Set arms, the TRUE greedy Set-Oracle aggregate MSE (diagnostic only)
   and the gap between them.

Never mixes (1) and (2) into one number, per spec section 20.
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
from scripts.train_oracle_rank_gain01 import run_sequence_individual_onpolicy
from scripts.train_oracle_scratch01 import base_score, encode_raw, run_sequence_individual, run_sequence_set
from scripts.train_oracle_scratch_tf01 import run_sequence_set_teacher_forced
from utils.dense_utility import candidate_weights, dense_utility
from utils.oracle_intervention import select_greedy_weighted_set


def load_arm(reference_ckpt, ckpt_path, device):
    ck = torch.load(ckpt_path, map_location='cpu')
    target, prefix_policy, scorer = ck['target'], ck['prefix_policy'], ck['scorer']
    overrides = {'is_training': 0, 'model_id': 'eval_rank_gain01', 'des': 'eval',
                 'checkpoints': '/tmp/exp_oracle_rank_gain01_eval', 'stage1_retrieval_metric': 'cosine',
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
    return exp, args, model, set_conditioner, metric, target, prefix_policy, scorer, ck


@torch.no_grad()
def per_step_diagnostics_for_split(exp, args, model, set_conditioner, metric, target, prefix_policy,
                                    split, channels, k, tau_topk, tau_choice, chunk_size, n_queries):
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
                seq_fn = run_sequence_individual if prefix_policy == 'tf' else run_sequence_individual_onpolicy
                losses, diags, picks = seq_fn(b_i, u_target_const, cand_mask, tau_choice, k)
                selected_mask = torch.zeros_like(cand_mask)
                neg_inf = torch.finfo(b_i.dtype).min / 4
                for t in range(k):
                    valid_now = cand_mask & ~selected_mask
                    # The diagnostic question is always "where does the
                    # TRUE Oracle best (remaining) candidate sit in the
                    # model's OWN ranking `b_i`?" -- for Individual, that
                    # is `argmax(u_target_const)` restricted to `valid_now`,
                    # regardless of TF/on-policy (the model's own trajectory
                    # `picks` determines the MASK, never what counts as the
                    # oracle answer at that state).
                    oracle_idx = u_target_const.masked_fill(~valid_now, neg_inf).argmax(-1, keepdim=True)
                    stats = oracle_rank_statistics(b_i, oracle_idx, valid_now, oracle_valid=valid_now.any(dim=-1))
                    frac = float(stats['oracle_top10_rank_fraction'])
                    ce_sum += float(losses[t].detach()) * batch_x.size(0)
                    ce_n += batch_x.size(0)
                    if frac == frac:
                        step_rows[t]['rank_frac'].append(frac)
                        step_rows[t]['rank'].append(float(stats['oracle_top10_mean_rank']))
                        step_rows[t]['top1'].append(diags[t]['top1_acc'])
                        step_rows[t]['top5'].append(diags[t]['top5_acc'])
                        step_rows[t]['top10'].append(diags[t]['top10_acc'])
                    nxt = picks[:, t:t + 1]
                    selected_mask = selected_mask.scatter(1, nxt, True)
            else:
                w_base = candidate_weights(b_i, cand_mask, tau_topk)
                seq_fn = run_sequence_set_teacher_forced if prefix_policy == 'tf' else run_sequence_set
                losses, diags, picks, a_dense_steps = seq_fn(
                    z_q, E, cand_mask, set_conditioner, metric, w_base, futures, query_future,
                    tau_choice, k, chunk_size)
                selected_mask = torch.zeros_like(cand_mask)
                for t in range(k):
                    valid_now = cand_mask & ~selected_mask
                    if t == 0:
                        s_hat_t = base_score(z_q, E, metric)
                    else:
                        prefix = picks[:, :t]
                        m = E[prefix].mean(dim=1)
                        h_t = set_conditioner(z_q, m)
                        s_hat_t = base_score(h_t, E, metric)
                    u_target_t = -a_dense_steps[t]
                    # Same principle as the Individual branch: the Oracle
                    # answer at THIS state is `argmax(u_target_t)` over
                    # `valid_now`, recomputed fresh -- for TF, `picks[:,t]`
                    # already equals this by construction (harmless overlap);
                    # for on-policy, `picks[:,t]` is the MODEL's own choice,
                    # which must NOT be conflated with the Oracle's answer.
                    neg_inf_t = torch.finfo(u_target_t.dtype).min / 4
                    oracle_idx_t = u_target_t.masked_fill(~valid_now, neg_inf_t).argmax(-1, keepdim=True)
                    stats = oracle_rank_statistics(s_hat_t, oracle_idx_t, valid_now, oracle_valid=valid_now.any(dim=-1))
                    frac = float(stats['oracle_top10_rank_fraction'])
                    ce_sum += float(losses[t].detach()) * batch_x.size(0)
                    ce_n += batch_x.size(0)
                    if frac == frac:
                        step_rows[t]['rank_frac'].append(frac)
                        step_rows[t]['rank'].append(float(stats['oracle_top10_mean_rank']))
                        step_rows[t]['top1'].append(diags[t]['top1_acc'])
                        step_rows[t]['top5'].append(diags[t]['top5_acc'])
                        step_rows[t]['top10'].append(diags[t]['top10_acc'])
                    nxt = picks[:, t:t + 1]
                    selected_mask = selected_mask.scatter(1, nxt, True)

        rows_seen += batch_x.size(0)

    def _avg(lst):
        return sum(lst) / len(lst) if lst else float('nan')

    per_step = {t: {kk: _avg(vv) for kk, vv in step_rows[t].items()} for t in range(k)}
    t1 = per_step[0]
    tgeq2_vals = {kk: _avg(sum((step_rows[t][kk] for t in range(1, k)), [])) for kk in ('rank', 'rank_frac', 'top1', 'top5', 'top10')}
    overall_vals = {kk: _avg(sum((step_rows[t][kk] for t in range(k)), [])) for kk in ('rank', 'rank_frac', 'top1', 'top5', 'top10')}
    return {'choice_ce_mean': ce_sum / max(ce_n, 1), 't1': t1, 'tgeq2': tgeq2_vals, 'overall': overall_vals,
            'per_step': per_step, 'n_rows_evaluated': rows_seen}


@torch.no_grad()
def free_running_eval_for_split(exp, args, model, set_conditioner, metric, target, split, channels,
                                 k, tau_topk, chunk_size, device):
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
    ap.add_argument('--stage1_checkpoint', required=True, help='checkpoint.pth (Best) OR checkpoint_epoch0.pth')
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--n_diag_queries', type=int, default=500)
    ap.add_argument('--out', required=True, help='diagnostics json path')
    ap.add_argument('--cache_dir', default=None, help='if given, also build the free-running retrieval cache')
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp, args, model, set_conditioner, metric, target, prefix_policy, scorer, ck = load_arm(
        cli.reference_ckpt, cli.stage1_checkpoint, device)
    exp._ensure_memory()
    channels = list(model.target_channels())
    k = int(cli.top_k)
    tau_topk = float(args.tau_topk)
    tau_choice = float(ck.get('tau_choice', tau_topk))

    report = {'target': target, 'prefix_policy': prefix_policy, 'scorer': scorer, 'pred_len': int(args.pred_len),
              'epoch': ck.get('epoch'), 'val_rank_fraction_overall': ck.get('val_rank_fraction_overall')}
    for split in ('train', 'val', 'test'):
        print(f'[eval_oracle_rank_gain01] {target}/{prefix_policy}/{scorer} epoch={ck.get("epoch")} split={split}: per-step diagnostics...')
        diag = per_step_diagnostics_for_split(exp, args, model, set_conditioner, metric, target, prefix_policy,
                                               split, channels, k, tau_topk, tau_choice, cli.chunk_size, cli.n_diag_queries)
        report[f'{split}_diagnostics'] = diag
        print(f'  choice_ce={diag["choice_ce_mean"]:.4f} overall_rank_frac={diag["overall"]["rank_frac"]:.5f} '
              f't1_rank_frac={diag["t1"]["rank_frac"]:.5f} tgeq2_rank_frac={diag["tgeq2"]["rank_frac"]:.5f}')

        print(f'[eval_oracle_rank_gain01] {target}/{prefix_policy}/{scorer} split={split}: free-running eval...')
        fr = free_running_eval_for_split(exp, args, model, set_conditioner, metric, target, split, channels,
                                          k, tau_topk, cli.chunk_size, device)
        report[f'{split}_free_running'] = fr
        print(f'  {fr}')

        if cli.cache_dir:
            cache_dir = Path(cli.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache = build_retrieval_cache(exp, args, model, set_conditioner, metric, target, split, channels,
                                           k, tau_topk, cli.chunk_size, device)
            torch.save(cache, cache_dir / f'{split}.pt')
            print(f'  cached n={cache["Y_q"].size(0)} -> {cache_dir / f"{split}.pt"}')

    Path(cli.out).parent.mkdir(parents=True, exist_ok=True)
    with open(cli.out, 'w') as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f'[eval_oracle_rank_gain01] written to {cli.out}')


if __name__ == '__main__':
    main()
