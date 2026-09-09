#!/usr/bin/env python3
"""EXP-ORACLE-SCRATCH-TF01 evaluation.

Teacher-forced diagnostics are computed on the greedy Set-Oracle prefix, while
Stage-2 retrieval cache construction remains fully FREE-RUNNING and future-free.
This separation is intentional:

- teacher-forced rank/Top-k/Choice-CE => Oracle learnability
- free-running Y_ret cache / aggregate => deployable retrieval quality

Individual arms delegate to the parent evaluator unchanged because their
sequence is already deterministic Oracle-order masking.
"""
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.eval_oracle_scratch01 as base_eval
from scripts.train_margutil01 import memory_value
from scripts.train_oracle_scratch01 import base_score, encode_raw
from scripts.train_oracle_choice01 import oracle_choice_step_loss
from layers.retrieval_metric import oracle_rank_statistics
from utils.dense_utility import candidate_weights, dense_utility
from utils.oracle_intervention import select_greedy_weighted_set


@torch.no_grad()
def diagnostics_for_split_teacher_forcing(exp, args, model, set_conditioner, metric, target, split, channels,
                                           k, tau_topk, tau_choice, chunk_size, n_queries):
    if target != 'set':
        return base_eval._ORIGINAL_DIAGNOSTICS(
            exp, args, model, set_conditioner, metric, target, split, channels,
            k, tau_topk, tau_choice, chunk_size, n_queries
        )

    _, loader = exp._get_data(flag=split, shuffle=False)
    device = exp.device
    step_rows = {
        t: {'rank': [], 'rank_frac': [], 'top1': [], 'top5': [], 'top10': [], 'ndcg10': []}
        for t in range(k)
    }
    ce_sum, ce_n = 0.0, 0.0
    free_agg_se, free_agg_n = 0.0, 0.0
    greedy_oracle_se, greedy_oracle_n = 0.0, 0.0
    rows_seen = 0

    for batch_x, batch_y, batch_start_idx in loader:
        if rows_seen >= n_queries:
            break
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)

        for c in channels:
            E = encode_raw(model, exp.memory_x, c)
            z_q = encode_raw(model, batch_x, c)
            memory_c, offset_c = memory_value(
                args, batch_x, exp.memory_y, exp.memory_x_last, c
            )
            futures = memory_c + offset_c.view(-1, 1, 1)
            query_future = batch_y[:, :, c]
            b_i = base_score(z_q, E, metric)
            w_base = candidate_weights(b_i, cand_mask, tau_topk)

            # ---------- Teacher-forced Oracle-prefix diagnostics ----------
            selected_mask = torch.zeros_like(cand_mask)
            oracle_picks = []
            neg_inf = torch.finfo(b_i.dtype).min / 4

            for t in range(k):
                if t == 0:
                    s_hat = b_i
                else:
                    prefix = torch.stack(oracle_picks, dim=1)
                    m = E[prefix].mean(dim=1)
                    h_t = set_conditioner(z_q, m)
                    s_hat = base_score(h_t, E, metric)

                valid_now = cand_mask & ~selected_mask
                prefix_now = (
                    torch.stack(oracle_picks, dim=1)
                    if t > 0
                    else torch.zeros(batch_x.size(0), 0, dtype=torch.long, device=device)
                )
                a_dense = dense_utility(
                    prefix_now, w_base, futures, query_future, chunk_size=chunk_size
                )
                u_target_t = -a_dense

                loss_t, diag_t = oracle_choice_step_loss(
                    s_hat, u_target_t, valid_now, tau_choice
                )
                i_star = u_target_t.masked_fill(~valid_now, neg_inf).argmax(-1, keepdim=True)
                stats = oracle_rank_statistics(
                    s_hat, i_star, valid_now, oracle_valid=valid_now.any(dim=-1)
                )

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

                # TEACHER FORCING: next conditioning state is Oracle's own pick.
                oracle_picks.append(i_star.squeeze(-1))
                selected_mask = selected_mask.scatter(1, i_star, True)

            # ---------- Free-running deployable aggregate ----------
            free_picks, _ = base_eval.free_running_topk(
                z_q, E, cand_mask, target, set_conditioner, metric, k
            )
            w_pick = w_base.gather(1, free_picks)
            alpha = w_pick / w_pick.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            y_ret = (
                alpha.unsqueeze(-1)
                * futures.gather(1, free_picks.unsqueeze(-1).expand(-1, -1, futures.size(-1)))
            ).sum(dim=1)
            free_agg_se += float((y_ret - query_future).pow(2).mean(dim=-1).sum())
            free_agg_n += batch_x.size(0)

            # ---------- True greedy Set-Oracle upper bound ----------
            greedy_picks = select_greedy_weighted_set(
                futures, query_future, b_i.detach(), cand_mask, k, tau_topk
            )
            w_pick_g = w_base.gather(1, greedy_picks)
            alpha_g = w_pick_g / w_pick_g.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            y_ret_g = (
                alpha_g.unsqueeze(-1)
                * futures.gather(1, greedy_picks.unsqueeze(-1).expand(-1, -1, futures.size(-1)))
            ).sum(dim=1)
            greedy_oracle_se += float((y_ret_g - query_future).pow(2).mean(dim=-1).sum())
            greedy_oracle_n += batch_x.size(0)

        rows_seen += batch_x.size(0)

    def _avg(lst):
        return sum(lst) / len(lst) if lst else float('nan')

    per_step = {
        t: {kk: _avg(vv) for kk, vv in step_rows[t].items() if kk != 'ndcg10'}
        for t in range(k)
    }
    tgeq2 = {
        kk: _avg(sum((step_rows[t][kk] for t in range(1, k)), []))
        for kk in ('rank', 'rank_frac', 'top1', 'top5', 'top10')
    }
    free_mse = free_agg_se / max(free_agg_n, 1)
    oracle_mse = greedy_oracle_se / max(greedy_oracle_n, 1)

    return {
        'diagnostic_prefix_mode': 'teacher_forcing_oracle_prefix',
        'choice_ce_mean': ce_sum / max(ce_n, 1),
        't1': per_step[0],
        'tgeq2': tgeq2,
        'per_step': per_step,
        'n_rows_evaluated': rows_seen,
        # Keep the parent key name for compatibility; it means the model's
        # actual FREE-RUNNING selection quality in this evaluator.
        'final_weighted_aggregate_mse': free_mse,
        'free_running_final_weighted_aggregate_mse': free_mse,
        'greedy_set_oracle_aggregate_mse': oracle_mse,
        'free_running_to_oracle_gap': free_mse - oracle_mse,
    }


def main():
    print('[eval_oracle_scratch_tf01] diagnostics = TEACHER FORCING; cache = FREE RUNNING.')
    base_eval._ORIGINAL_DIAGNOSTICS = base_eval.diagnostics_for_split
    base_eval.diagnostics_for_split = diagnostics_for_split_teacher_forcing
    base_eval.main()


if __name__ == '__main__':
    main()
