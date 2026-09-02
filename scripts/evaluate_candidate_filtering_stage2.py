#!/usr/bin/env python3
"""STEP 3 -- does a *learned* candidate filter improve the actual forecast?

STEP 1 gives the Oracle upper bound, STEP 2 shows the label is learnable. This
closes the loop with the classifier's own probabilities, so nothing here touches
a query future.

    soft   w~ = w * p        renormalized, every candidate kept but down-weighted
    hard   w~ = w * 1[p>0.5] renormalized over the survivors
    oracle w~ = w * 1[U>0]   the STEP 1 upper bound, for reference only

Reports the chain the hypothesis predicts: filter quality -> positive utility
rate in what survives -> forecast MSE.
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.retrieval_diagnostics import append_row, load_stage2, mse_mae, unwrap  # noqa: E402
from scripts.analyze_residual_oracle import prepare  # noqa: E402
from scripts.train_utility_classifier import PairClassifier  # noqa: E402

EPS = 1e-8
COLUMNS = [
    'dataset', 'pred_len', 'pool_m', 'delta', 'alpha', 'threshold',
    'base_mse', 'base_mae', 'current_mse', 'current_mae',
    'soft_filter_mse', 'soft_filter_mae', 'hard_filter_mse', 'hard_filter_mae',
    'oracle_filter_mse', 'oracle_filter_mae',
    'global_utility_oracle_mse', 'global_utility_oracle_mae',
    'positive_rate_before', 'positive_rate_after_hard',
    'mean_utility_before', 'mean_utility_after_hard',
    'retained_candidates', 'candidate_gate_mean', 'checkpoint',
]


@torch.no_grad()
def analyse(stage2_ckpt, classifier_ckpt, threshold=0.5, top_k=10, tau=0.1,
            max_batches=0, chunk=128):
    experiment, args = load_stage2(stage2_ckpt)
    model = unwrap(experiment.model)
    experiment._ensure_memory()
    experiment._build_key_bank(force=True)

    bundle = torch.load(classifier_ckpt, map_location='cpu')
    classifier = PairClassifier(bundle['dim'])
    classifier.load_state_dict(bundle['state_dict'])
    classifier = classifier.to(experiment.device).eval()
    pool_m, alpha = bundle['pool_m'], bundle['alpha']

    data = prepare(experiment, 'test', max_batches)
    channels = data['query_y'].size(-1)
    n_query = data['query_y'].size(0)
    zeros = torch.zeros_like(data['query_base'])
    corrections = {k: zeros.clone() for k in
                   ('current', 'soft', 'hard', 'oracle', 'global')}
    acc = {k: [] for k in ('pos_before', 'pos_after', 'u_before', 'u_after',
                           'retained', 'gate')}

    for start in range(0, n_query, chunk):
        stop = min(start + chunk, n_query)
        cand_mask, _ = experiment._candidate_mask(data['query_start'][start:stop])
        cand_mask = cand_mask.to(experiment.device)
        for c in range(channels):
            sources = model.source_channels(c)
            if c not in sources or experiment.key_bank is None:
                continue
            slot = sources.index(c)
            z_q = model._branch_embedding(data['query_x'][start:stop], c, c)
            z_bank = experiment.key_bank[c, slot].to(z_q.device, z_q.dtype)
            scores = torch.matmul(z_q, z_bank.transpose(0, 1)).masked_fill(
                ~cand_mask, float('-inf'))
            width = min(pool_m, z_bank.size(0))
            pool = scores.topk(width, dim=-1).indices
            pool_valid = cand_mask.gather(1, pool)

            q_res = data['query_residual'][start:stop, :, c]
            k_res = data['memory_residual'][:, :, c]
            horizon = float(k_res.size(-1))
            utility = (
                2.0 * alpha * torch.matmul(q_res, k_res.transpose(0, 1)) / horizon
                - (alpha ** 2) * k_res.square().mean(-1).unsqueeze(0)
            ).gather(1, pool)

            weights = torch.softmax(
                scores.gather(1, pool).masked_fill(~pool_valid, float('-inf')) / tau, dim=-1)
            residual = k_res[pool]
            probability = torch.sigmoid(classifier(z_q.unsqueeze(1), z_bank[pool]))

            def mix(mask_or_weight):
                w = weights * mask_or_weight
                w = w / w.sum(-1, keepdim=True).clamp_min(EPS)
                keep = (mask_or_weight.sum(-1, keepdim=True) > 0).float()
                return (residual * w.unsqueeze(-1)).sum(1) * keep

            corrections['current'][start:stop, :, c] = (
                residual * weights.unsqueeze(-1)).sum(1)
            corrections['soft'][start:stop, :, c] = mix(probability * pool_valid)
            hard_mask = ((probability > threshold) & pool_valid).float()
            corrections['hard'][start:stop, :, c] = mix(hard_mask)
            oracle_mask = ((utility > 0) & pool_valid).float()
            corrections['oracle'][start:stop, :, c] = mix(oracle_mask)

            g_idx = (
                2.0 * alpha * torch.matmul(q_res, k_res.transpose(0, 1)) / horizon
                - (alpha ** 2) * k_res.square().mean(-1).unsqueeze(0)
            ).masked_fill(~cand_mask, float('-inf')).topk(
                min(top_k, k_res.size(0)), dim=-1).indices
            corrections['global'][start:stop, :, c] = k_res[g_idx].mean(1)

            acc['pos_before'].append(
                (oracle_mask.sum(-1) / pool_valid.sum(-1).clamp_min(1)).cpu())
            survivors = hard_mask.sum(-1).clamp_min(1)
            acc['pos_after'].append(
                ((oracle_mask * hard_mask).sum(-1) / survivors).cpu())
            acc['u_before'].append(
                ((utility * pool_valid).sum(-1) / pool_valid.sum(-1).clamp_min(1)).cpu())
            acc['u_after'].append(((utility * hard_mask).sum(-1) / survivors).cpu())
            acc['retained'].append(hard_mask.sum(-1).cpu())
            acc['gate'].append(probability.mean(-1).cpu())

    base, true = data['query_base'], data['query_y']
    out = {name: mse_mae(base + alpha * value, true)
           for name, value in corrections.items()}
    base_mse, base_mae = mse_mae(base, true)
    mean = lambda key: float(torch.cat(acc[key]).mean())
    row = {
        'dataset': args.data, 'pred_len': int(args.pred_len),
        'pool_m': pool_m, 'delta': bundle['delta'], 'alpha': alpha,
        'threshold': threshold,
        'base_mse': base_mse, 'base_mae': base_mae,
        'current_mse': out['current'][0], 'current_mae': out['current'][1],
        'soft_filter_mse': out['soft'][0], 'soft_filter_mae': out['soft'][1],
        'hard_filter_mse': out['hard'][0], 'hard_filter_mae': out['hard'][1],
        'oracle_filter_mse': out['oracle'][0], 'oracle_filter_mae': out['oracle'][1],
        'global_utility_oracle_mse': out['global'][0],
        'global_utility_oracle_mae': out['global'][1],
        'positive_rate_before': mean('pos_before'),
        'positive_rate_after_hard': mean('pos_after'),
        'mean_utility_before': mean('u_before'),
        'mean_utility_after_hard': mean('u_after'),
        'retained_candidates': mean('retained'),
        'candidate_gate_mean': mean('gate'),
        'checkpoint': stage2_ckpt,
    }
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--classifier', required=True)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--max_batches', type=int, default=0)
    parser.add_argument('--csv', default='')
    args = parser.parse_args()

    row = analyse(args.checkpoint, args.classifier, args.threshold,
                  max_batches=args.max_batches)
    print(f"=== {row['dataset']}/{row['pred_len']} learned filtering ===")
    for key in COLUMNS:
        if key == 'checkpoint':
            continue
        value = row[key]
        print(f'  {key}: {value:.6f}' if isinstance(value, float) else f'  {key}: {value}')
    if args.csv:
        append_row(args.csv, row, COLUMNS)
        print(f'appended to {args.csv}')


if __name__ == '__main__':
    main()
