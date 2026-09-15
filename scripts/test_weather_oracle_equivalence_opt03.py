#!/usr/bin/env python3
"""TRACK-A-WEATHER-OPT03 -- real Weather-data equivalence for the two paths
OPT03 actually modifies: Individual Oracle (A) and Oracle-Choice CE (C).
The Greedy Set Oracle (E) is unchanged OPT02 code, already validated on real
Weather data in `results/TRACK-A-WEATHER-OPT02/equivalence_H{96,720}.json`
and is not reverified here (would duplicate that work against an identical
implementation).

A: Top-K index agreement (argmax over valid candidates) + max/mean absolute
   distance (utility) difference, reference `individual_utility` vs
   `individual_oracle_utility_optimized`.
C: loss + all `diag` fields compared per call across the full K=10 greedy
   trajectory (not just isolated synthetic inputs) -- reference
   `oracle_choice_step_loss` vs `oracle_choice_step_loss_optimized`.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diag_set_difficulty01 import HostScorer
from scripts.train_factorial_e2e01 import arm_score, encode_raw, individual_utility
from scripts.train_margutil01 import memory_value
from scripts.train_oracle_choice01 import oracle_choice_step_loss
from scripts.test_set_oracle_equivalence import load_arm, load_scratch
from utils.dense_utility import candidate_weights
from utils.oracle_compute_optimized import (individual_oracle_utility_optimized,
                                            oracle_choice_step_loss_optimized,
                                            prepare_individual_query_static)


@torch.no_grad()
def run(arm_ckpt, stage2_host, out_path, n_queries=500, top_k=10, candidate_chunk_size=4096,
       scratch_reference_ckpt=None, scratch_pred_len=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if scratch_reference_ckpt:
        exp, args, model, sc = load_scratch(scratch_reference_ckpt, scratch_pred_len, device)
        ckpt_desc = f'SCRATCH(untrained, {scratch_reference_ckpt}, pred_len={scratch_pred_len})'
    else:
        exp, args, model, sc = load_arm(arm_ckpt, device)
        exp._ensure_memory()
        ckpt_desc = arm_ckpt
    host = HostScorer(stage2_host, device)

    _, loader = exp._get_data(flag='test', shuffle=False)
    channel = 0
    n_seen = 0
    a_max_diff, a_disagree, a_positions = 0.0, 0, 0
    a_tie_disagree, a_real_disagree, a_disagree_detail = 0, 0, []
    c_loss_max_diff, c_calls = 0.0, 0
    c_diag_max_diff = {}

    for batch_x, batch_y, batch_start_idx in loader:
        if n_seen >= n_queries:
            break
        batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)
        cand_mask, counts = exp._candidate_mask(batch_start_idx)
        valid_query = counts.to(device) >= top_k
        take = min(int(valid_query.sum()), n_queries - n_seen)
        if take <= 0:
            continue

        E = encode_raw(model, exp.memory_x, channel)
        z_q = encode_raw(model, batch_x, channel)
        memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, channel)
        futures = memory_c + offset_c.view(-1, 1, 1)
        q_future = batch_y[:, :, channel]
        host_scores = host.scores(batch_x, channel, cand_mask)
        w = candidate_weights(host_scores, cand_mask, host.tau_topk)

        cand_sq = prepare_individual_query_static(futures, candidate_chunk_size=candidate_chunk_size)

        selected = torch.zeros_like(cand_mask)
        picks = []
        for t in range(top_k):
            h_t = z_q if t == 0 else sc(z_q, E[torch.stack(picks, dim=1)].mean(dim=1))
            u_hat = arm_score(h_t, E, None)
            valid_now = cand_mask & ~selected

            u_ref = individual_utility(futures, q_future)
            u_opt = individual_oracle_utility_optimized(futures, q_future, cand_sq=cand_sq,
                                                         candidate_chunk_size=candidate_chunk_size)
            valid_take = valid_now[:take]
            diff = (u_ref - u_opt).abs()[:take][valid_take] if valid_take.any() else torch.zeros(0)
            if diff.numel() > 0:
                a_max_diff = max(a_max_diff, float(diff.max()))
            u_ref_m = u_ref.masked_fill(~valid_now, float('-inf'))
            u_opt_m = u_opt.masked_fill(~valid_now, float('-inf'))
            pick_ref_full = u_ref_m.argmax(dim=-1)
            pick_opt_take = u_opt_m.argmax(dim=-1)[:take]
            dis = pick_ref_full[:take] != pick_opt_take
            a_disagree += int(dis.sum())
            a_positions += take
            if bool(dis.any()):
                for bi in dis.nonzero(as_tuple=True)[0].tolist():
                    ref_u = float(u_ref_m[bi, pick_ref_full[bi]])
                    ref_u_at_opt = float(u_ref_m[bi, pick_opt_take[bi]])
                    is_tie = abs(ref_u - ref_u_at_opt) < 1e-5
                    if is_tie:
                        a_tie_disagree += 1
                    else:
                        a_real_disagree += 1
                    if len(a_disagree_detail) < 20:
                        a_disagree_detail.append({
                            't': t, 'row': bi, 'is_reference_tie': is_tie,
                            'ref_pick': int(pick_ref_full[bi]), 'opt_pick': int(pick_opt_take[bi]),
                            'ref_utility_at_ref_pick': ref_u, 'ref_utility_at_opt_pick': ref_u_at_opt,
                        })

            loss_ref, diag_ref = oracle_choice_step_loss(u_hat[:take], u_ref[:take], valid_now[:take],
                                                          host.tau_topk)
            loss_opt, diag_opt = oracle_choice_step_loss_optimized(u_hat[:take], u_ref[:take],
                                                                    valid_now[:take], host.tau_topk)
            c_loss_max_diff = max(c_loss_max_diff, float((loss_ref - loss_opt).abs()))
            c_calls += 1
            for k in diag_ref:
                a_, b_ = diag_ref[k], diag_opt[k]
                d = 0.0 if (a_ != a_ and b_ != b_) else abs(a_ - b_)
                c_diag_max_diff[k] = max(c_diag_max_diff.get(k, 0.0), d)

            picks.append(pick_ref_full)
            selected = selected.scatter(1, pick_ref_full.unsqueeze(-1), True)

        n_seen += take

    out = {
        'checkpoint': ckpt_desc, 'stage2_host': str(stage2_host),
        'candidate_chunk_size': candidate_chunk_size, 'n_queries_evaluated': n_seen,
        'individual_oracle': {
            'max_abs_utility_diff': a_max_diff,
            'selection_disagreements': a_disagree,
            'selection_disagreements_tie_with_reference': a_tie_disagree,
            'selection_disagreements_real_mismatch': a_real_disagree,
            'selection_positions_checked': a_positions,
            'selection_agreement_pct': 100.0 * (1 - a_disagree / max(a_positions, 1)),
            'non_tie_selection_agreement_pct': 100.0 * (1 - a_real_disagree / max(a_positions, 1)),
            'disagreement_detail': a_disagree_detail,
        },
        'oracle_choice_ce': {
            'max_abs_loss_diff': c_loss_max_diff,
            'max_abs_diag_diff': c_diag_max_diff,
            'calls_checked': c_calls,
        },
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm_checkpoint', default=None)
    ap.add_argument('--stage2_host', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_queries', type=int, default=500)
    ap.add_argument('--candidate_chunk_size', type=int, default=4096)
    ap.add_argument('--scratch_reference_ckpt', default=None)
    ap.add_argument('--scratch_pred_len', type=int, default=None)
    cli = ap.parse_args()
    run(cli.arm_checkpoint, cli.stage2_host, cli.out, n_queries=cli.n_queries,
        candidate_chunk_size=cli.candidate_chunk_size,
        scratch_reference_ckpt=cli.scratch_reference_ckpt, scratch_pred_len=cli.scratch_pred_len)


if __name__ == '__main__':
    main()
