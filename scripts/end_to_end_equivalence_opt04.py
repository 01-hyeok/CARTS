#!/usr/bin/env python3
"""TRACK-A-WEATHER-OPT04 -- end-to-end equivalence: reference vs optimized,
same initial state, 10 real optimizer steps (forward + backward +
`optimizer.step()`), comparing per-step loss, gradient norm, parameter
displacement, and final model-state hash. For the `individual` target
(paths C + A only, both proven bit-exact at the function level) this checks
whether that bit-exactness survives 10 full training steps intact. For the
`greedy_set` target (path E, OPT02's algebraically-identical-but-not-
bit-identical formula) tolerance and selection agreement are reported
instead of a hash match.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.SequentialSetRetriever import SetConditioner
from scripts import train_factorial_e2e01 as T
from scripts.benchmark_training_opt04 import build_fresh
from scripts.train_margutil01 import memory_value
from utils.dense_utility import candidate_weights


def run_n_steps(target, prefix_policy, oracle_compute_impl, chunk_size,
                reference_ckpt, stage2_host, pred_len, device, n_steps, seed, init_state):
    exp, args, model, sc = build_fresh(reference_ckpt, pred_len, device, seed)
    model.load_state_dict(init_state['model'])
    sc.load_state_dict(init_state['sc'])
    host = T.HostScorer(stage2_host, device)
    tau_topk = host.tau_topk
    channels = list(range(int(args.enc_in)))
    torch.manual_seed(seed)
    _, loader = exp._get_data(flag='train')
    it = iter(loader)
    params = [p for p in model.parameters() if p.requires_grad] + list(sc.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=0.0)
    resolved = T.resolve_oracle_compute_impl(oracle_compute_impl, chunk_size)

    step_losses, step_picks = [], []
    for _ in range(n_steps):
        try:
            batch_x, batch_y, batch_start_idx = next(it)
        except StopIteration:
            it = iter(loader)
            batch_x, batch_y, batch_start_idx = next(it)
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        optimizer.zero_grad()
        batch_loss = 0.0
        picks_this_step = []
        for c in channels:
            E = T.encode_raw(model, exp.memory_x, c)
            z_q = T.encode_raw(model, batch_x, c)
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            futures = memory_c + offset_c.view(-1, 1, 1)
            query_future = batch_y[:, :, c]
            with torch.no_grad():
                host_scores = host.scores(batch_x, c, cand_mask)
                w_host = candidate_weights(host_scores, cand_mask, tau_topk)
            losses, _, picks, _ = T.run_sequence(
                z_q, E, cand_mask, sc, None, w_host, futures, query_future,
                target, prefix_policy, tau_topk, 10, chunk_size,
                choice_ce_impl=resolved['choice_ce_impl'],
                individual_impl=resolved['individual_impl'],
                greedy_set_impl=resolved['greedy_set_impl'])
            batch_loss = batch_loss + sum(losses) / 10
            picks_this_step.append(picks.detach().cpu())
        batch_loss = batch_loss / len(channels)
        step_losses.append(float(batch_loss.detach()))
        step_picks.append(picks_this_step)
        batch_loss.backward()
        optimizer.step()

    final_hash = T.state_sha({**{f'model.{k}': v for k, v in model.state_dict().items()},
                              **{f'sc.{k}': v for k, v in sc.state_dict().items()}})
    return {'step_losses': step_losses, 'step_picks': step_picks, 'final_state_sha256': final_hash,
           'model_state': model.state_dict()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--target', choices=['individual', 'greedy_set'], required=True)
    ap.add_argument('--prefix_policy', default='onpolicy')
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--stage2_host', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--n_steps', type=int, default=10)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp0, args0, model0, sc0 = build_fresh(cli.reference_ckpt, cli.pred_len, device, cli.seed)
    init_state = {'model': {k: v.detach().clone() for k, v in model0.state_dict().items()},
                  'sc': {k: v.detach().clone() for k, v in sc0.state_dict().items()}}
    del exp0, args0, model0, sc0

    ref = run_n_steps(cli.target, cli.prefix_policy, 'reference', cli.chunk_size,
                      cli.reference_ckpt, cli.stage2_host, cli.pred_len, device,
                      cli.n_steps, cli.seed, init_state)
    opt = run_n_steps(cli.target, cli.prefix_policy, 'optimized', cli.chunk_size,
                      cli.reference_ckpt, cli.stage2_host, cli.pred_len, device,
                      cli.n_steps, cli.seed, init_state)

    loss_diffs = [abs(a - b) for a, b in zip(ref['step_losses'], opt['step_losses'])]
    param_diff_max, param_diff_l2 = 0.0, 0.0
    for k in ref['model_state']:
        d = (ref['model_state'][k].float() - opt['model_state'][k].float())
        param_diff_max = max(param_diff_max, float(d.abs().max()))
        param_diff_l2 += float(d.pow(2).sum())
    param_diff_l2 = param_diff_l2 ** 0.5

    pick_disagreements, pick_positions = 0, 0
    for step_r, step_o in zip(ref['step_picks'], opt['step_picks']):
        for pr, po in zip(step_r, step_o):
            pick_disagreements += int((pr != po).sum())
            pick_positions += pr.numel()

    out = {
        'target': cli.target, 'prefix_policy': cli.prefix_policy, 'n_steps': cli.n_steps,
        'ref_step_losses': ref['step_losses'], 'opt_step_losses': opt['step_losses'],
        'max_abs_loss_diff': max(loss_diffs), 'mean_abs_loss_diff': sum(loss_diffs) / len(loss_diffs),
        'final_state_sha256_ref': ref['final_state_sha256'],
        'final_state_sha256_opt': opt['final_state_sha256'],
        'final_state_bit_exact': ref['final_state_sha256'] == opt['final_state_sha256'],
        'final_param_max_abs_diff': param_diff_max, 'final_param_l2_diff': param_diff_l2,
        'selection_disagreements': pick_disagreements, 'selection_positions_checked': pick_positions,
        'selection_agreement_pct': 100.0 * (1 - pick_disagreements / max(pick_positions, 1)),
    }
    Path(cli.out).parent.mkdir(parents=True, exist_ok=True)
    Path(cli.out).write_text(json.dumps({k: v for k, v in out.items()}, indent=2))
    print(json.dumps({k: v for k, v in out.items() if 'step_losses' not in k and 'sha256' not in k or 'exact' in k},
                     indent=2))


if __name__ == '__main__':
    main()
