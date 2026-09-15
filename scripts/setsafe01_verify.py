#!/usr/bin/env python3
"""TRACK-A-WEATHER-SETSAFE01 -- equivalence verification for
`--oracle_compute_impl safe` (C optimized + A cached, E always reference)
before launching the remaining Weather H720 Set-Oracle arms.

Covers BOTH oracle types (individual, greedy_set), both prefix policies
(tf, onpolicy), and both scorers (cosine, asymmetric) -- 8 combinations.
Compares `--oracle_compute_impl reference` vs `safe`, same initial model
state (model + SetConditioner + RetrievalMetric, when scorer=asymmetric),
same batches, same seed, real forward + backward + `optimizer.step()`,
for `n_steps` consecutive optimizer steps. Checks, every step: Oracle
target index, model selection, loss, and at the end: full Adam optimizer
state (exp_avg/exp_avg_sq per parameter) and final model-state hash.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from layers.retrieval_metric import RetrievalMetric, cosine_init_deviation
from models.SequentialSetRetriever import SetConditioner
from scripts import train_factorial_e2e01 as T
from scripts.train_margutil01 import build_experiment, memory_value
from utils.dense_utility import candidate_weights


def build_fresh_full(reference_ckpt, pred_len, device, seed, scorer):
    torch.manual_seed(seed)
    exp, args = build_experiment(reference_ckpt, {'pred_len': pred_len, 'seq_len': pred_len,
                                                  'batch_size': 32, 'seed': seed, 'top_k': 10})
    exp._ensure_memory()
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(True)
    d_model = int(args.d_model)
    sc = SetConditioner(d_model).to(device)
    metric = None
    if scorer == 'asymmetric':
        metric = RetrievalMetric(kind='asymmetric', dim=d_model, output='cosine',
                                 layer_norm=False).to(device)
        dev = cosine_init_deviation(metric)
        assert dev < 1e-5, f'asymmetric metric not identity at init: {dev}'
    return exp, args, model, sc, metric


def clone_state(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def optimizer_state_sha(optimizer):
    import hashlib
    h = hashlib.sha256()
    for group in optimizer.param_groups:
        for p in group['params']:
            st = optimizer.state.get(p, {})
            for k in sorted(st):
                v = st[k]
                if torch.is_tensor(v):
                    h.update(k.encode())
                    h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def run_n_steps(target, prefix_policy, scorer, oracle_compute_impl, chunk_size,
                reference_ckpt, stage2_host, pred_len, device, n_steps, seed, init_state):
    exp, args, model, sc, metric = build_fresh_full(reference_ckpt, pred_len, device, seed, scorer)
    model.load_state_dict(init_state['model'])
    sc.load_state_dict(init_state['sc'])
    if metric is not None and init_state.get('metric') is not None:
        metric.load_state_dict(init_state['metric'])
    host = T.HostScorer(stage2_host, device)
    tau_topk = host.tau_topk
    channels = list(range(int(args.enc_in)))
    torch.manual_seed(seed)
    _, loader = exp._get_data(flag='train')
    it = iter(loader)
    params = ([p for p in model.parameters() if p.requires_grad] + list(sc.parameters())
             + (list(metric.parameters()) if metric is not None else []))
    optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=0.0)
    resolved = T.resolve_oracle_compute_impl(oracle_compute_impl, chunk_size)

    step_losses, step_oracle_idx, step_model_idx = [], [], []
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
        oidx_this, midx_this = [], []
        for c in channels:
            E = T.encode_raw(model, exp.memory_x, c)
            z_q = T.encode_raw(model, batch_x, c)
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            futures = memory_c + offset_c.view(-1, 1, 1)
            query_future = batch_y[:, :, c]
            with torch.no_grad():
                host_scores = host.scores(batch_x, c, cand_mask)
                w_host = candidate_weights(host_scores, cand_mask, tau_topk)
            losses, _, picks, steps = T.run_sequence(
                z_q, E, cand_mask, sc, metric, w_host, futures, query_future,
                target, prefix_policy, tau_topk, 10, chunk_size,
                choice_ce_impl=resolved['choice_ce_impl'],
                individual_impl=resolved['individual_impl'],
                greedy_set_impl=resolved['greedy_set_impl'])
            batch_loss = batch_loss + sum(losses) / 10
            oidx_this.append(torch.stack([s['oracle_idx'] for s in steps], dim=1).detach().cpu())
            midx_this.append(picks.detach().cpu())
        batch_loss = batch_loss / len(channels)
        step_losses.append(float(batch_loss.detach()))
        step_oracle_idx.append(oidx_this)
        step_model_idx.append(midx_this)
        batch_loss.backward()
        optimizer.step()

    final_state = {**{f'model.{k}': v for k, v in model.state_dict().items()},
                   **{f'sc.{k}': v for k, v in sc.state_dict().items()}}
    if metric is not None:
        final_state.update({f'metric.{k}': v for k, v in metric.state_dict().items()})
    return {
        'step_losses': step_losses, 'step_oracle_idx': step_oracle_idx, 'step_model_idx': step_model_idx,
        'final_state_sha256': T.state_sha(final_state),
        'optimizer_state_sha256': optimizer_state_sha(optimizer),
        'model_state': model.state_dict(),
    }


def compare(target, prefix_policy, scorer, reference_ckpt, stage2_host, pred_len, device,
           n_steps, chunk_size, seed):
    exp0, args0, model0, sc0, metric0 = build_fresh_full(reference_ckpt, pred_len, device, seed, scorer)
    init_state = {'model': clone_state(model0), 'sc': clone_state(sc0),
                  'metric': clone_state(metric0) if metric0 is not None else None}
    del exp0, args0, model0, sc0, metric0

    ref = run_n_steps(target, prefix_policy, scorer, 'reference', chunk_size,
                      reference_ckpt, stage2_host, pred_len, device, n_steps, seed, init_state)
    safe = run_n_steps(target, prefix_policy, scorer, 'safe', chunk_size,
                       reference_ckpt, stage2_host, pred_len, device, n_steps, seed, init_state)

    loss_diffs = [abs(a - b) for a, b in zip(ref['step_losses'], safe['step_losses'])]
    oracle_dis, oracle_pos = 0, 0
    for step_r, step_s in zip(ref['step_oracle_idx'], safe['step_oracle_idx']):
        for cr, cs_ in zip(step_r, step_s):
            oracle_dis += int((cr != cs_).sum())
            oracle_pos += cr.numel()
    model_dis, model_pos = 0, 0
    for step_r, step_s in zip(ref['step_model_idx'], safe['step_model_idx']):
        for cr, cs_ in zip(step_r, step_s):
            model_dis += int((cr != cs_).sum())
            model_pos += cr.numel()
    param_diff_max = 0.0
    for k in ref['model_state']:
        d = (ref['model_state'][k].float() - safe['model_state'][k].float())
        param_diff_max = max(param_diff_max, float(d.abs().max()))

    out = {
        'target': target, 'prefix_policy': prefix_policy, 'scorer': scorer, 'n_steps': n_steps,
        'max_abs_loss_diff': max(loss_diffs), 'mean_abs_loss_diff': sum(loss_diffs) / len(loss_diffs),
        'oracle_target_index_agreement_pct': 100.0 * (1 - oracle_dis / max(oracle_pos, 1)),
        'model_selection_agreement_pct': 100.0 * (1 - model_dis / max(model_pos, 1)),
        'final_param_max_abs_diff': param_diff_max,
        'final_state_bit_exact': ref['final_state_sha256'] == safe['final_state_sha256'],
        'optimizer_state_bit_exact': ref['optimizer_state_sha256'] == safe['optimizer_state_sha256'],
        'final_state_sha256_ref': ref['final_state_sha256'], 'final_state_sha256_safe': safe['final_state_sha256'],
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--targets', default='individual,greedy_set')
    ap.add_argument('--prefix_policies', default='tf,onpolicy')
    ap.add_argument('--scorers', default='cosine,asymmetric')
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--stage2_host', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--n_steps', type=int, default=30)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = []
    for target in cli.targets.split(','):
        for pp in cli.prefix_policies.split(','):
            for sco in cli.scorers.split(','):
                r = compare(target, pp, sco, cli.reference_ckpt, cli.stage2_host, cli.pred_len,
                           device, cli.n_steps, cli.chunk_size, cli.seed)
                results.append(r)
                print(f"[setsafe01_verify] {target}/{pp}/{sco}: "
                     f"oracle_agree={r['oracle_target_index_agreement_pct']:.4f}% "
                     f"model_agree={r['model_selection_agreement_pct']:.4f}% "
                     f"loss_diff_max={r['max_abs_loss_diff']:.2e} "
                     f"final_state_bit_exact={r['final_state_bit_exact']} "
                     f"optimizer_state_bit_exact={r['optimizer_state_bit_exact']}")

    Path(cli.out).parent.mkdir(parents=True, exist_ok=True)
    Path(cli.out).write_text(json.dumps(results, indent=2))
    print(f'[setsafe01_verify] wrote {cli.out}')


if __name__ == '__main__':
    main()
