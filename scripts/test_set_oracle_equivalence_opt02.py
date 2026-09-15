#!/usr/bin/env python3
"""TRACK-A-WEATHER-OPT02 -- end-to-end equivalence on REAL Weather data,
reference (B0) vs chunked optimized (OPT02, d=None path).

Extends OPT01's `scripts/test_set_oracle_equivalence.py` methodology with
the additional comparisons the OPT02 spec requires: aggregated retrieved
future diff, Stage-2 input (`y_ret`) diff, and (best-effort) `y_final`/
Stage-2 MSE diff, plus explicit tie analysis (a "disagreement" whose
utility AND aggregate are equal to the reference is reported separately
from a genuine mismatch).
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp.exp_stage1_relation import Exp_Stage1_Relation
from models.SequentialSetRetriever import SetConditioner
from scripts.diag_set_difficulty01 import HostScorer
from scripts.train_factorial_e2e01 import (arm_score, encode_raw,
                                           free_running_aggregate_future_mse)
from scripts.train_margutil01 import build_experiment, memory_value
from utils.dense_utility import candidate_weights, dense_utility
from utils.dense_utility_optimized import dense_utility_optimized, prepare_query_static_chunked


def load_arm(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    args = SimpleNamespace(**ckpt['args'])
    exp = Exp_Stage1_Relation(args)
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(device)
    sc = SetConditioner(int(args.d_model)).to(device)
    sc.load_state_dict(ckpt['set_conditioner_state_dict'])
    sc.eval()
    return exp, args, model, sc


def load_scratch(reference_ckpt, pred_len, device):
    exp, args = build_experiment(reference_ckpt, {'pred_len': pred_len, 'seq_len': pred_len})
    exp._ensure_memory()
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.eval().to(device)
    sc = SetConditioner(int(args.d_model)).to(device)
    sc.eval()
    return exp, args, model, sc


@torch.no_grad()
def run(arm_ckpt, stage2_host, out_path, n_queries=500, top_k=10,
       candidate_chunk_size=1024, scratch_reference_ckpt=None, scratch_pred_len=None):
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
    max_util_diff = 0.0
    total_disagreements, total_positions = 0, 0
    tie_disagreements, real_disagreements = 0, 0
    disagreement_detail = []
    agg_diff_max, agg_diff_mean_accum, agg_diff_mean_n = 0.0, 0.0, 0
    yret_diff_max = 0.0

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

        d_sq = prepare_query_static_chunked(futures, q_future, candidate_chunk_size=candidate_chunk_size)

        selected = torch.zeros_like(cand_mask)
        picks_ref, picks_opt = [], []
        for t in range(top_k):
            if t == 0:
                h_t = z_q
            else:
                m = E[torch.stack(picks_ref, dim=1)].mean(dim=1)
                h_t = sc(z_q, m)
            u_hat = arm_score(h_t, E, None)
            valid_now = cand_mask & ~selected
            prefix = torch.stack(picks_ref, dim=1) if picks_ref else torch.zeros(
                batch_x.size(0), 0, dtype=torch.long, device=device)

            a_ref = dense_utility(prefix, w, futures, q_future)
            a_opt = dense_utility_optimized(prefix, w, futures=futures, query_future=q_future,
                                            d_sq=d_sq, candidate_chunk_size=candidate_chunk_size)

            diff = (a_ref - a_opt).abs()[valid_now]
            if diff.numel() > 0:
                max_util_diff = max(max_util_diff, float(diff.max()))

            a_ref_m = a_ref.masked_fill(~valid_now, float('inf'))
            a_opt_m = a_opt.masked_fill(~valid_now, float('inf'))
            pick_ref = a_ref_m.argmin(dim=-1)
            pick_opt = a_opt_m.argmin(dim=-1)
            sub = slice(0, take)
            total_positions += take
            dis = (pick_ref[sub] != pick_opt[sub])
            total_disagreements += int(dis.sum())
            if bool(dis.any()):
                for bi in dis.nonzero(as_tuple=True)[0].tolist():
                    ref_u = float(a_ref_m[bi, pick_ref[bi]])
                    ref_u_at_opt = float(a_ref_m[bi, pick_opt[bi]])
                    is_tie = abs(ref_u - ref_u_at_opt) < 1e-5   # reference itself can't tell them apart
                    if is_tie:
                        tie_disagreements += 1
                    else:
                        real_disagreements += 1
                    if len(disagreement_detail) < 20:
                        disagreement_detail.append({
                            't': t, 'row_in_batch': bi, 'is_reference_tie': is_tie,
                            'ref_pick': int(pick_ref[bi]), 'opt_pick': int(pick_opt[bi]),
                            'ref_utility_at_ref_pick': ref_u,
                            'ref_utility_at_opt_pick': ref_u_at_opt,
                            'utility_gap_under_reference': ref_u_at_opt - ref_u,
                        })

            picks_ref.append(pick_ref)
            picks_opt.append(pick_opt)
            selected = selected.scatter(1, pick_ref.unsqueeze(-1), True)

        picks_ref_t = torch.stack(picks_ref, dim=1)[:take]
        picks_opt_t = torch.stack(picks_opt, dim=1)[:take]

        # aggregated retrieved future (Stage-2 input, y_ret-equivalent):
        # alpha-weighted mean of the K selected futures under each path's
        # own picks, using the SAME host weighting both paths already share.
        def y_ret(picks):
            sc_sel = host_scores.gather(1, picks)
            alpha = torch.softmax(sc_sel / host.tau_topk, dim=-1)
            tgt = futures.gather(1, picks.unsqueeze(-1).expand(-1, -1, futures.size(-1)))
            return (alpha.unsqueeze(-1) * tgt).sum(1)

        yret_ref = y_ret(picks_ref_t)[:take]
        yret_opt = y_ret(picks_opt_t)[:take]
        yret_diff = (yret_ref - yret_opt).abs()
        yret_diff_max = max(yret_diff_max, float(yret_diff.max()))

        fr_ref = free_running_aggregate_future_mse(picks_ref_t, host_scores[:take], futures[:take],
                                                    q_future[:take], host.tau_topk)
        fr_opt = free_running_aggregate_future_mse(picks_opt_t, host_scores[:take], futures[:take],
                                                    q_future[:take], host.tau_topk)
        agg_diff = (fr_ref - fr_opt).abs()
        agg_diff_max = max(agg_diff_max, float(agg_diff.max()))
        agg_diff_mean_accum += float(agg_diff.sum())
        agg_diff_mean_n += agg_diff.numel()

        n_seen += take

    out = {
        'arm_checkpoint': ckpt_desc, 'stage2_host': str(stage2_host),
        'candidate_chunk_size': candidate_chunk_size,
        'n_queries_evaluated': n_seen,
        'max_abs_utility_diff_at_valid_positions': max_util_diff,
        'selection_disagreements_total': total_disagreements,
        'selection_disagreements_tie_with_reference': tie_disagreements,
        'selection_disagreements_real_mismatch': real_disagreements,
        'selection_positions_checked': total_positions,
        'non_tie_selection_agreement_pct': 100.0 * (1 - real_disagreements / max(total_positions, 1)),
        'overall_selection_agreement_pct': 100.0 * (1 - total_disagreements / max(total_positions, 1)),
        'max_free_running_aggregate_mse_diff': agg_diff_max,
        'mean_free_running_aggregate_mse_diff': agg_diff_mean_accum / max(agg_diff_mean_n, 1),
        'max_stage2_input_yret_diff': yret_diff_max,
        'disagreement_detail': disagreement_detail,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2))
    for k, v in out.items():
        if k != 'disagreement_detail':
            print(f'[equivalence_opt02] {k} = {v}')
    if disagreement_detail:
        print(f'[equivalence_opt02] disagreement_detail = {json.dumps(disagreement_detail, indent=2)}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm_checkpoint', default=None)
    ap.add_argument('--stage2_host', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_queries', type=int, default=500)
    ap.add_argument('--candidate_chunk_size', type=int, default=1024)
    ap.add_argument('--scratch_reference_ckpt', default=None)
    ap.add_argument('--scratch_pred_len', type=int, default=None)
    cli = ap.parse_args()
    run(cli.arm_checkpoint, cli.stage2_host, cli.out, n_queries=cli.n_queries,
        candidate_chunk_size=cli.candidate_chunk_size,
        scratch_reference_ckpt=cli.scratch_reference_ckpt, scratch_pred_len=cli.scratch_pred_len)


if __name__ == '__main__':
    main()
