#!/usr/bin/env python3
"""TRACK-A-WEATHER-OPT01 -- end-to-end Set Oracle equivalence, on REAL
Weather data and a REAL trained checkpoint (not synthetic tensors).

Loads TRACK-A-FACTORIAL-E2E01's own `set_tf_cosine` Weather_96 checkpoint
(a Greedy Set Oracle, on-policy training, cosine score -- exactly the
production configuration this optimization targets), encodes real
candidates/queries with it, and walks a full K=10 free-running trajectory
TWICE -- once through the reference `dense_utility`, once through the
optimized `dense_utility_optimized` -- comparing at every step:

  - raw utility (masked to VALID positions, matching dense_utility's own
    documented contract -- see tests/test_dense_utility_optimized.py)
  - selected candidate index
  - resulting free-running aggregate MSE

No training. Read-only. Does not touch the running
TRACK-A-FACTORIAL-E2E01 / TRACK-A-ONPOLICY-RANKLOSS01 / TRACK-A-MULTIPOS-
CHOICE01 checkpoints or results -- only loads one of them for inference.
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
from scripts.train_factorial_e2e01 import arm_score, encode_raw
from scripts.train_margutil01 import memory_value
from utils.dense_utility import candidate_weights, dense_utility
from utils.dense_utility_optimized import dense_utility_optimized, prepare_query_static


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
    """For a cell TRACK-A-FACTORIAL-E2E01 has not yet trained a Set arm for
    (e.g. Weather_720 at the time of this benchmark): build a fresh,
    UNTRAINED scratch encoder via the exact same `build_experiment` +
    `Model(args).float()` path `train_factorial_e2e01.py` uses before
    training starts. The Oracle-equivalence and speed properties tested
    here do not depend on encoder quality -- only on real Weather-scale
    N/H and real candidate weight distributions -- so this is a real
    Weather_720 artifact test, not a synthetic one, while training that
    cell is still in progress elsewhere on the same GPU."""
    from scripts.train_margutil01 import build_experiment
    exp, args = build_experiment(reference_ckpt, {'pred_len': pred_len, 'seq_len': pred_len})
    exp._ensure_memory()
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.eval().to(device)
    sc = SetConditioner(int(args.d_model)).to(device)
    sc.eval()
    return exp, args, model, sc


@torch.no_grad()
def run(arm_ckpt, stage2_host, out_path, n_queries=500, top_k=10, chunk_size=4096,
        scratch_reference_ckpt=None, scratch_pred_len=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if scratch_reference_ckpt:
        exp, args, model, sc = load_scratch(scratch_reference_ckpt, scratch_pred_len, device)
        arm_ckpt = f'SCRATCH (untrained, {scratch_reference_ckpt}, pred_len={scratch_pred_len})'
    else:
        exp, args, model, sc = load_arm(arm_ckpt, device)
        exp._ensure_memory()
    host = HostScorer(stage2_host, device)

    _, loader = exp._get_data(flag='test', shuffle=False)
    channel = 0
    n_seen = 0
    max_util_diff = 0.0
    total_disagreements = 0
    total_positions = 0
    disagreement_detail = []
    agg_diff_max = 0.0
    step_timing_ref, step_timing_opt = [], []

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

        d, d_sq = prepare_query_static(futures, q_future)

        selected = torch.zeros_like(cand_mask)
        picks_ref, picks_opt = [], []
        agg_ref_prev, agg_opt_prev = None, None
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

            torch.cuda.synchronize() if device.type == 'cuda' else None
            t0 = time.perf_counter()
            a_ref = dense_utility(prefix, w, futures, q_future, chunk_size=chunk_size)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            step_timing_ref.append(time.perf_counter() - t0)

            torch.cuda.synchronize() if device.type == 'cuda' else None
            t0 = time.perf_counter()
            a_opt = dense_utility_optimized(prefix, w, d, d_sq, chunk_size=chunk_size)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            step_timing_opt.append(time.perf_counter() - t0)

            diff = (a_ref - a_opt).abs()[valid_now]
            if diff.numel() > 0:
                max_util_diff = max(max_util_diff, float(diff.max()))

            neg_inf = torch.finfo(u_hat.dtype).min / 4
            a_ref_m = a_ref.masked_fill(~valid_now, float('inf'))
            a_opt_m = a_opt.masked_fill(~valid_now, float('inf'))
            pick_ref = a_ref_m.argmin(dim=-1)
            pick_opt = a_opt_m.argmin(dim=-1)
            sub = slice(0, take)
            total_positions += take
            dis = (pick_ref[sub] != pick_opt[sub])
            total_disagreements += int(dis.sum())
            if bool(dis.any()) and len(disagreement_detail) < 20:
                for bi in dis.nonzero(as_tuple=True)[0].tolist():
                    ref_util = a_ref_m[bi, pick_ref[bi]].item()
                    ref_util_at_opt_pick = a_ref_m[bi, pick_opt[bi]].item()
                    opt_util = a_opt_m[bi, pick_opt[bi]].item()
                    opt_util_at_ref_pick = a_opt_m[bi, pick_ref[bi]].item()
                    disagreement_detail.append({
                        't': t, 'row_in_batch': bi,
                        'ref_pick': int(pick_ref[bi]), 'opt_pick': int(pick_opt[bi]),
                        'ref_utility_at_ref_pick': ref_util,
                        'ref_utility_at_opt_pick': ref_util_at_opt_pick,
                        'utility_gap_under_reference': ref_util_at_opt_pick - ref_util,
                        'opt_utility_at_opt_pick': opt_util,
                        'opt_utility_at_ref_pick': opt_util_at_ref_pick,
                        'utility_gap_under_optimized': opt_util_at_ref_pick - opt_util,
                    })

            picks_ref.append(pick_ref)
            picks_opt.append(pick_opt)
            selected = selected.scatter(1, pick_ref.unsqueeze(-1), True)

        picks_ref_t = torch.stack(picks_ref, dim=1)[:take]
        picks_opt_t = torch.stack(picks_opt, dim=1)[:take]
        from scripts.train_factorial_e2e01 import free_running_aggregate_future_mse
        fr_ref = free_running_aggregate_future_mse(picks_ref_t, host_scores[:take], futures[:take],
                                                    q_future[:take], host.tau_topk)
        fr_opt = free_running_aggregate_future_mse(picks_opt_t, host_scores[:take], futures[:take],
                                                    q_future[:take], host.tau_topk)
        agg_diff_max = max(agg_diff_max, float((fr_ref - fr_opt).abs().max()))
        n_seen += take

    out = {
        'arm_checkpoint': str(arm_ckpt), 'stage2_host': str(stage2_host),
        'n_queries_evaluated': n_seen,
        'max_abs_utility_diff_at_valid_positions': max_util_diff,
        'selection_disagreements': total_disagreements,
        'selection_positions_checked': total_positions,
        'selection_agreement_pct': 100.0 * (1 - total_disagreements / max(total_positions, 1)),
        'max_free_running_aggregate_mse_diff': agg_diff_max,
        'disagreement_detail': disagreement_detail,
        'mean_step_time_ms_reference': 1000 * sum(step_timing_ref) / max(len(step_timing_ref), 1),
        'mean_step_time_ms_optimized': 1000 * sum(step_timing_opt) / max(len(step_timing_opt), 1),
        'speedup_ratio': (sum(step_timing_ref) / max(sum(step_timing_opt), 1e-9)),
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=2))
    for k, v in out.items():
        print(f'[set_oracle_equivalence] {k} = {v}')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm_checkpoint', required=False, default=None)
    ap.add_argument('--stage2_host', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_queries', type=int, default=500)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--scratch_reference_ckpt', default=None,
                    help='if set, use a fresh UNTRAINED scratch encoder instead of --arm_checkpoint')
    ap.add_argument('--scratch_pred_len', type=int, default=None)
    cli = ap.parse_args()
    run(cli.arm_checkpoint, cli.stage2_host, cli.out, n_queries=cli.n_queries,
        chunk_size=cli.chunk_size, scratch_reference_ckpt=cli.scratch_reference_ckpt,
        scratch_pred_len=cli.scratch_pred_len)


if __name__ == '__main__':
    main()
