#!/usr/bin/env python3
"""PHASE 0 -- is there anything inside Top-M worth reranking for?

No training. The frozen full-bank retriever produces a broad Top-M; an oracle
then reorders it by *measured* downstream utility and the resulting Top-K goes
back through the unmodified production Stage-2. If that oracle cannot beat plain
CARTS Top-K, a learned reranker has nothing to recover and the study stops here.

Two rules keep this honest:

  * a candidate set is scored by running the real `forward` with the memory
    masked down to that set, so the aggregation, the mixer, the gate and the
    offset restore are the production ones. Nothing is reconstructed by hand.
  * "oracle individual Top-K" is the top K by single-candidate utility. It is
    not guaranteed to be the best *set* -- ten individually strong candidates
    can be redundant -- which is what the optional greedy set oracle is for.

Query futures build the utility labels and the metrics, and never reach a
selection input that a deployable model would have to reproduce.
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_set_level_utility import selected_candidates  # noqa: E402
from utils.retrieval_diagnostics import append_row, unwrap  # noqa: E402
from utils.utility_teacher import _dummy_cache, load_stage2_reference  # noqa: E402

COLUMNS = [
    'dataset', 'pred_len', 'pool_m', 'top_k', 'split', 'queries', 'channels',
    'original_topk_mse', 'original_topk_mae', 'original_set_utility',
    'oracle_individual_mse', 'oracle_individual_mae', 'oracle_set_utility',
    'oracle_best_single_mse', 'oracle_best_single_mae',
    'greedy_set_oracle_mse', 'greedy_queries',
    'base_mse', 'oracle_rerank_gain', 'oracle_rerank_gain_pct',
    'pool_best_utility', 'pool_positive_rate', 'checkpoint',
]


def restricted_mask(cand_mask, ids):
    """Validity mask that admits only `ids`, intersected with the real mask.

    Restricting the mask rather than injecting values is what keeps the
    aggregation production: the model still runs its own Top-K retrieval and its
    own softmax weights, it simply has nothing else to choose from.
    """
    mask = torch.zeros_like(cand_mask)
    mask.scatter_(1, ids, True)
    return mask & cand_mask


@torch.no_grad()
def forecast_with_mask(experiment, model, batch_x, mask, memory_y, top_k):
    previous = model.top_k
    model.top_k = top_k
    try:
        return experiment.model(
            batch_x=batch_x, memory_y=memory_y, valid_mask=mask,
            key_bank=experiment.key_bank, memory_x_last=experiment.memory_x_last,
        )[0]
    finally:
        model.top_k = previous


@torch.no_grad()
def channelwise_selection_forecast(experiment, model, batch_x, cand_mask,
                                   chosen_ids, memory_y, top_k):
    """Production forecast where each channel keeps only its own chosen ids.

    `forward` takes one [B, N] mask for every channel, so a per-channel
    selection needs one forward per channel; column c of pass c is the only part
    kept. Slower than a single pass and the only way to stay on the production
    path.
    """
    columns = []
    for c in range(chosen_ids.size(1)):
        mask = restricted_mask(cand_mask, chosen_ids[:, c])
        y = forecast_with_mask(experiment, model, batch_x, mask, memory_y, top_k)
        columns.append(y[:, :, c])
    return torch.stack(columns, dim=-1)


@torch.no_grad()
def greedy_set_oracle(experiment, model, batch_x, batch_y, cand_mask, pool_ids,
                      memory_y, top_k):
    """Greedy set construction against the real set MSE, one channel at a time.

    Cost is top_k * M forwards per channel, so this only ever runs on a small
    query subset and is reported as a diagnostic.
    """
    channels = pool_ids.size(1)
    width = pool_ids.size(2)
    columns = []
    for c in range(channels):
        chosen = []
        best_column = None
        for _ in range(top_k):
            best = (None, None, float('inf'))
            for slot in range(width):
                if slot in chosen:
                    continue
                trial = chosen + [slot]
                ids = pool_ids[:, c][:, trial]
                mask = restricted_mask(cand_mask, ids)
                y = forecast_with_mask(experiment, model, batch_x, mask,
                                       memory_y, len(trial))
                mse = float((y[:, :, c] - batch_y[:, :, c]).square().mean())
                if mse < best[2]:
                    best = (slot, y[:, :, c], mse)
            if best[0] is None:
                break
            chosen.append(best[0])
            best_column = best[1]
        columns.append(best_column)
    return torch.stack(columns, dim=-1)


def mse_mae_per_channel(prediction, target):
    if prediction.shape != target.shape:
        raise ValueError(
            f'forecast shape {tuple(prediction.shape)} != target {tuple(target.shape)}'
        )
    difference = prediction - target
    return difference.square().mean(dim=1), difference.abs().mean(dim=1)


@torch.no_grad()
def analyse(checkpoint, pool_m, top_k=10, max_queries=512, split='test',
            candidate_chunk=10, greedy_queries=0):
    experiment, args = load_stage2_reference(checkpoint)
    model = unwrap(experiment.model)
    experiment._ensure_memory()
    experiment._build_key_bank(force=True)
    _, loader = experiment._get_data(flag=split, shuffle=False)

    device = experiment.device
    memory_y = experiment.memory_y.to(device)
    memory_x_last = experiment.memory_x_last.to(device)

    totals = {name: [] for name in (
        'original_mse', 'original_mae', 'oracle_mse', 'oracle_mae',
        'single_mse', 'single_mae', 'base', 'best_utility', 'positive',
        'greedy_mse',
    )}
    seen = 0
    greedy_seen = 0
    for batch_x, batch_y, batch_start_idx in loader:
        if max_queries and seen >= max_queries:
            break
        batch_x, batch_y, batch_start_idx = experiment._move_batch(
            batch_x, batch_y, batch_start_idx)
        if max_queries and seen + batch_x.size(0) > max_queries:
            keep = max_queries - seen
            batch_x, batch_y, batch_start_idx = (
                batch_x[:keep], batch_y[:keep], batch_start_idx[:keep])
        seen += batch_x.size(0)

        cand_mask, _ = experiment._candidate_mask(batch_start_idx)
        pool_ids = selected_candidates(model, experiment.key_bank, batch_x,
                                       cand_mask, pool_m)          # [B, C, M]

        # A: plain CARTS. The full mask, the model's own Top-K.
        original = forecast_with_mask(experiment, model, batch_x, cand_mask,
                                      memory_y, top_k)
        original_mse, original_mae = mse_mae_per_channel(original, batch_y)

        utility, base_mse = model.evaluate_candidate_correction(
            batch_x=batch_x, batch_y=batch_y, candidate_indices=pool_ids,
            memory_y=memory_y, valid_mask=cand_mask, key_bank=None,
            memory_x_last=memory_x_last,
            retrieval_cache=_dummy_cache(model, batch_x.size(0), device, batch_x.dtype),
            candidate_chunk=candidate_chunk,
        )                                                          # [B, M, C]
        utility = utility.permute(0, 2, 1)                         # [B, C, M]

        width = min(top_k, utility.size(-1))
        best = utility.topk(width, dim=-1).indices                 # [B, C, k]
        oracle_ids = pool_ids.gather(2, best)
        oracle = channelwise_selection_forecast(
            experiment, model, batch_x, cand_mask, oracle_ids, memory_y, width)
        oracle_mse, oracle_mae = mse_mae_per_channel(oracle, batch_y)

        single_ids = pool_ids.gather(2, utility.argmax(-1, keepdim=True))
        single = channelwise_selection_forecast(
            experiment, model, batch_x, cand_mask, single_ids, memory_y, 1)
        single_mse, single_mae = mse_mae_per_channel(single, batch_y)

        totals['original_mse'].append(original_mse.cpu())
        totals['original_mae'].append(original_mae.cpu())
        totals['oracle_mse'].append(oracle_mse.cpu())
        totals['oracle_mae'].append(oracle_mae.cpu())
        totals['single_mse'].append(single_mse.cpu())
        totals['single_mae'].append(single_mae.cpu())
        totals['base'].append(base_mse.cpu())
        totals['best_utility'].append(utility.max(-1).values.cpu())
        totals['positive'].append((utility > 0).float().mean(-1).cpu())

        if greedy_queries and greedy_seen < greedy_queries:
            keep = min(batch_x.size(0), greedy_queries - greedy_seen)
            greedy = greedy_set_oracle(
                experiment, model, batch_x[:keep], batch_y[:keep],
                cand_mask[:keep], pool_ids[:keep], memory_y, width)
            greedy_mse, _ = mse_mae_per_channel(greedy, batch_y[:keep])
            totals['greedy_mse'].append(greedy_mse.cpu())
            greedy_seen += keep

    stacked = {name: torch.cat(value) for name, value in totals.items() if value}
    mean = lambda name: float(stacked[name].mean()) if name in stacked else float('nan')
    original_mse = mean('original_mse')
    oracle_mse = mean('oracle_mse')
    base = mean('base')
    row = {
        'dataset': args.data, 'pred_len': int(args.pred_len), 'pool_m': pool_m,
        'top_k': top_k, 'split': split, 'queries': seen,
        'channels': int(stacked['original_mse'].size(-1)),
        'original_topk_mse': original_mse, 'original_topk_mae': mean('original_mae'),
        'original_set_utility': base - original_mse,
        'oracle_individual_mse': oracle_mse, 'oracle_individual_mae': mean('oracle_mae'),
        'oracle_set_utility': base - oracle_mse,
        'oracle_best_single_mse': mean('single_mse'),
        'oracle_best_single_mae': mean('single_mae'),
        'greedy_set_oracle_mse': mean('greedy_mse'),
        'greedy_queries': greedy_seen,
        'base_mse': base,
        'oracle_rerank_gain': original_mse - oracle_mse,
        'oracle_rerank_gain_pct': 100.0 * (original_mse - oracle_mse) / max(original_mse, 1e-12),
        'pool_best_utility': mean('best_utility'),
        'pool_positive_rate': mean('positive'),
        'checkpoint': checkpoint,
    }
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--pool_m', type=int, default=100)
    p.add_argument('--top_k', type=int, default=10)
    p.add_argument('--max_queries', type=int, default=512)
    p.add_argument('--candidate_chunk', type=int, default=10)
    p.add_argument('--greedy_queries', type=int, default=0)
    p.add_argument('--split', default='test')
    p.add_argument('--csv', default='')
    a = p.parse_args()

    row = analyse(a.checkpoint, a.pool_m, a.top_k, a.max_queries, a.split,
                  a.candidate_chunk, a.greedy_queries)
    print(
        f"[{row['dataset']}/{row['pred_len']} M={row['pool_m']}] "
        f"original={row['original_topk_mse']:.4f}  "
        f"oracle_individual={row['oracle_individual_mse']:.4f}  "
        f"oracle_best_single={row['oracle_best_single_mse']:.4f}  "
        f"gain={row['oracle_rerank_gain']:+.4f} ({row['oracle_rerank_gain_pct']:+.1f}%)  "
        f"base={row['base_mse']:.4f}"
    )
    if a.csv:
        append_row(a.csv, row, COLUMNS)


if __name__ == '__main__':
    main()
