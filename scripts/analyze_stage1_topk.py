#!/usr/bin/env python3
"""What did the Stage-1 encoder actually learn to retrieve?

Aggregate recall says how often the encoder agrees with the future Oracle. It
cannot say *why* it disagrees. This loads a trained Stage-1 checkpoint and, over
real test queries against the full candidate bank, separates the two candidate
explanations:

  1. the encoder fails at its own job -- its Top-K is not even past-similar
  2. the encoder does its job -- its Top-K is past-similar, but past similarity
     does not carry future similarity in this data

It also checks the shortcuts a retrieval encoder can fall into: collapsing onto
a handful of candidates for every query, or just returning temporally adjacent
windows.
"""

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp.exp_stage1_relation import Exp_Stage1_Relation  # noqa: E402
from models.RelationStage1 import transform_relation_history  # noqa: E402


def load_experiment(checkpoint_path, device_override=None):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'args' not in checkpoint:
        raise ValueError(f'checkpoint has no saved args: {checkpoint_path}')
    args = SimpleNamespace(**checkpoint['args'])
    args.num_workers = 0
    if device_override is not None:
        args.use_gpu = device_override != 'cpu'
    experiment = Exp_Stage1_Relation(args)
    state = checkpoint.get('model_state_dict', checkpoint)
    experiment.model.load_state_dict(state)
    experiment.model.eval()
    return experiment, args


@torch.no_grad()
def analyse(experiment, args, top_k, max_queries, split='test'):
    experiment._ensure_memory()
    experiment._build_key_bank(log=False)
    model = experiment.model.module if hasattr(experiment.model, 'module') else experiment.model
    device = experiment.device

    memory_x = torch.from_numpy(experiment.memory_x_np).float().to(device)
    memory_y = experiment.memory_y
    memory_x_last = experiment.memory_x_last
    starts = getattr(experiment.memory_sampler, 'starts', None)
    memory_starts = (
        torch.as_tensor(starts, dtype=torch.long, device=device)
        if starts is not None else None
    )
    num_candidates = memory_x.size(0)

    # Past distance lives in the space the encoder actually sees.
    memory_past = transform_relation_history(memory_x, model.relation_input_space)

    # Always the full bank with no Oracle injection, whichever split: the point
    # is to compare fit on seen queries against generalization to unseen ones
    # under one identical evaluation protocol.
    _, test_loader = experiment._get_data(flag=split, shuffle=False)
    targets = model.target_channels()

    totals = {}
    seen_queries = 0
    picked_counter = torch.zeros(num_candidates, device=device)
    oracle_counter = torch.zeros(num_candidates, device=device)
    corr_num = corr_den_a = corr_den_b = 0.0
    corr_rows = 0

    for batch_x, batch_y, batch_start_idx in test_loader:
        if seen_queries >= max_queries:
            break
        batch_x, batch_y, batch_start_idx = experiment._move_batch(
            batch_x, batch_y, batch_start_idx
        )
        cand_mask, _ = experiment._candidate_mask(batch_start_idx)
        query_past = transform_relation_history(batch_x, model.relation_input_space)

        for channel in targets:
            source_slot = model.source_slot(channel, channel)
            z_q = model.encoder(model._relation_tensor(batch_x, channel, channel))
            z_k = experiment.key_bank[channel, source_slot].to(
                device=device, dtype=z_q.dtype
            )
            scores = torch.matmul(z_q, z_k.transpose(0, 1))
            scores = scores.masked_fill(~cand_mask, float('-inf'))

            future_mse = model._future_mse(
                batch_x, batch_y, memory_y, memory_x_last, channel, channel
            ).masked_fill(~cand_mask, float('inf'))

            # Past distance between the query window and every candidate window,
            # in the encoder's own input space.
            q_past = query_past[:, :, channel]
            k_past = memory_past[:, :, channel]
            past_mse = (
                q_past.square().mean(-1, keepdim=True)
                + k_past.square().mean(-1).unsqueeze(0)
                - 2.0 * torch.matmul(q_past, k_past.transpose(0, 1)) / q_past.size(-1)
            ).clamp_min(0.0).masked_fill(~cand_mask, float('inf'))

            k = min(top_k, num_candidates)
            student_idx = scores.topk(k, dim=-1).indices
            oracle_idx = future_mse.topk(k, dim=-1, largest=False).indices
            past_idx = past_mse.topk(k, dim=-1, largest=False).indices

            def gather_mean(values, idx):
                return values.gather(1, idx).mean(dim=-1)

            row = {
                'student_future_mse': gather_mean(future_mse, student_idx),
                'oracle_future_mse': gather_mean(future_mse, oracle_idx),
                'past_topk_future_mse': gather_mean(future_mse, past_idx),
                'student_past_mse': gather_mean(past_mse, student_idx),
                'oracle_past_mse': gather_mean(past_mse, oracle_idx),
                'best_past_mse': gather_mean(past_mse, past_idx),
                'random_future_mse': future_mse.masked_fill(~cand_mask, 0.0).sum(-1)
                / cand_mask.sum(-1).clamp_min(1),
            }
            overlap = (
                student_idx.unsqueeze(-1) == past_idx.unsqueeze(-2)
            ).any(-1).float().mean(-1)
            row['student_vs_past_topk_overlap'] = overlap

            for depth in (1, 5, 10):
                width = min(depth, num_candidates)
                s_idx = scores.topk(width, dim=-1).indices
                o_idx = future_mse.topk(width, dim=-1, largest=False).indices
                row[f'oracle_recall_at_{depth}'] = (
                    s_idx.unsqueeze(-1) == o_idx.unsqueeze(-2)
                ).any(-1).float().mean(-1)
                p_idx = past_mse.topk(width, dim=-1, largest=False).indices
                row[f'past_nn_recall_at_{depth}'] = (
                    p_idx.unsqueeze(-1) == o_idx.unsqueeze(-2)
                ).any(-1).float().mean(-1)

            if memory_starts is not None:
                query_start = batch_start_idx.to(device).unsqueeze(1).float()
                gap = (memory_starts.unsqueeze(0).float() - query_start).abs()
                row['student_temporal_gap'] = gather_mean(gap, student_idx)
                row['oracle_temporal_gap'] = gather_mean(gap, oracle_idx)
                row['random_temporal_gap'] = (
                    gap.masked_fill(~cand_mask, 0.0).sum(-1) / cand_mask.sum(-1).clamp_min(1)
                )

            for key, value in row.items():
                finite = value[torch.isfinite(value)]
                if finite.numel():
                    totals.setdefault(key, []).append(finite.mean().cpu())

            picked_counter.scatter_add_(
                0, student_idx.reshape(-1),
                torch.ones(student_idx.numel(), device=device)
            )
            oracle_counter.scatter_add_(
                0, oracle_idx.reshape(-1),
                torch.ones(oracle_idx.numel(), device=device)
            )

            # Pearson correlation between past distance and future distance over
            # all valid (query, candidate) pairs: the ceiling on any past-only
            # retriever.
            valid = cand_mask & torch.isfinite(past_mse) & torch.isfinite(future_mse)
            if bool(valid.any()):
                a = past_mse[valid].double()
                b = future_mse[valid].double()
                a = a - a.mean()
                b = b - b.mean()
                corr_num += float((a * b).sum())
                corr_den_a += float((a * a).sum())
                corr_den_b += float((b * b).sum())
                corr_rows += 1

        seen_queries += batch_x.size(0)

    metrics = {key: float(torch.stack(v).mean()) for key, v in totals.items()}
    metrics['queries'] = seen_queries
    metrics['candidates'] = num_candidates
    metrics['top_k'] = int(top_k)
    if corr_rows and corr_den_a > 0 and corr_den_b > 0:
        metrics['corr_past_vs_future_mse'] = corr_num / (corr_den_a ** 0.5 * corr_den_b ** 0.5)

    picked = int((picked_counter > 0).sum())
    metrics['distinct_candidates_retrieved'] = picked
    metrics['distinct_candidate_fraction'] = picked / num_candidates
    metrics['distinct_candidates_oracle'] = int((oracle_counter > 0).sum())
    share = picked_counter / picked_counter.sum().clamp_min(1)
    metrics['retrieval_top1pct_share'] = float(
        share.topk(max(1, num_candidates // 100)).values.sum()
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='Stage-1 checkpoint.pth')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--max_queries', type=int, default=512)
    parser.add_argument('--device', default=None)
    parser.add_argument('--out', default='', help='Optional JSON output path')
    parser.add_argument('--label', default='', help='Label printed with the result')
    args = parser.parse_args()

    experiment, saved = load_experiment(args.checkpoint, args.device)
    metrics = analyse(experiment, saved, args.top_k, args.max_queries, args.split)
    metrics['split'] = args.split
    metrics['checkpoint'] = args.checkpoint
    metrics['label'] = args.label

    print(f'=== {args.label or args.checkpoint} ===')
    for key in sorted(metrics):
        value = metrics[key]
        print(f'  {key}: {value:.6f}' if isinstance(value, float) else f'  {key}: {value}')
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(metrics, indent=2))
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
