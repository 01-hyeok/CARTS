#!/usr/bin/env python3
"""Stage-1 retrieval diagnostics: raw past-NN vs the learned encoder.

One pass over a split produces every number the current investigation needs:

  1. retrieval quality      recall@k, retrieved/oracle future MSE, regret, NDCG, Spearman
  2. shortlist quality      coverage@M -- how much of the future Oracle Top-10 a
                            raw Top-M shortlist contains, which decides whether a
                            "raw retriever -> learned reranker" split is viable
  3. target sharpness       margin at the Top-10 boundary and how many candidates
                            sit within 1/5/10% of Oracle quality, which decides
                            whether exact Top-10 identity is a fair target at all
  4. oracle gap recovery    (random - retrieved) / (random - oracle): 0 = random,
                            1 = Oracle. Recall@10 near zero does not mean the
                            retrieved candidates are near-random.

Retrievers share the evaluation protocol exactly -- full candidate bank, no
Oracle injection -- so learned and encoder-free rankings are directly comparable.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp.exp_stage1_relation import Exp_Stage1_Relation  # noqa: E402
from models.RelationStage1 import (  # noqa: E402
    _student_retrieval_metrics,
    transform_relation_history,
)

COVERAGE_DEPTHS = (10, 50, 100, 200, 500)
NEAR_TIE_TOLERANCES = (0.01, 0.05, 0.10)
RETRIEVERS = ('learned', 'raw_l2', 'raw_cos', 'random')


def load_experiment(checkpoint_path, load_weights=True):
    """Build the Stage-1 experiment from a checkpoint's own saved args.

    Encoder-free retrievers still load a checkpoint: it is the record of the
    dataset, horizon and candidate-mask configuration the learned arm ran under,
    so every retriever is scored against an identical candidate pool.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'args' not in checkpoint:
        raise ValueError(f'checkpoint has no saved args: {checkpoint_path}')
    args = SimpleNamespace(**checkpoint['args'])
    args.num_workers = 0
    experiment = Exp_Stage1_Relation(args)
    if load_weights:
        experiment.model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    experiment.model.eval()
    return experiment, args


def _pairwise_mse(query, memory):
    """[B, N] mean squared distance between every query and candidate row."""
    return (
        query.square().mean(-1, keepdim=True)
        + memory.square().mean(-1).unsqueeze(0)
        - 2.0 * torch.matmul(query, memory.transpose(0, 1)) / query.size(-1)
    ).clamp_min(0.0)


@torch.no_grad()
def _coverage_at_m(scores, future_mse, valid_mask, oracle_k=10):
    """Fraction of the global Oracle Top-K that a Top-M shortlist contains."""
    num_cand = scores.size(-1)
    oracle_k = min(oracle_k, num_cand)
    oracle_idx = future_mse.topk(oracle_k, dim=-1, largest=False).indices
    oracle_valid = valid_mask.gather(1, oracle_idx)
    denominator = oracle_valid.sum(-1).clamp_min(1).float()
    out = {}
    for depth in COVERAGE_DEPTHS:
        width = min(depth, num_cand)
        shortlist = scores.topk(width, dim=-1, largest=True).indices
        hit = (
            (oracle_idx.unsqueeze(-1) == shortlist.unsqueeze(-2)).any(-1) & oracle_valid
        )
        out[f'coverage_at_{depth}'] = hit.sum(-1).float() / denominator
    return out


@torch.no_grad()
def _target_sharpness(future_mse, valid_mask, oracle_k=10, eps=1e-8):
    """How sharply the Oracle Top-K is separated from the rest.

    A large near-tie population means exact Top-K identity punishes a retriever
    for picking candidates that are, in future-MSE terms, interchangeable.
    """
    num_cand = future_mse.size(-1)
    depth = min(oracle_k + 1, num_cand)
    sorted_mse = future_mse.topk(depth, dim=-1, largest=False).values
    d_k = sorted_mse[:, min(oracle_k, depth) - 1]
    out = {}
    if depth > oracle_k:
        d_next = sorted_mse[:, oracle_k]
        out[f'oracle_margin_{oracle_k}'] = d_next - d_k
        out[f'oracle_relative_margin_{oracle_k}'] = (d_next - d_k) / (d_k.abs() + eps)

    oracle_mean = sorted_mse[:, :min(oracle_k, depth)].mean(-1)
    finite = torch.isfinite(future_mse) & valid_mask
    for tol in NEAR_TIE_TOLERANCES:
        pct = int(round(tol * 100))
        for name, reference in (('cutoff', d_k), ('oraclemean', oracle_mean)):
            threshold = reference.abs() * (1.0 + tol)
            within = (future_mse <= threshold.unsqueeze(1)) & finite
            out[f'near_oracle_count_{pct}pct_{name}'] = within.sum(-1).float()
    return out


@torch.no_grad()
def run(experiment, retriever, split, max_queries, input_space=None, top_k=10, tau=0.1):
    experiment._ensure_memory()
    model = experiment.model.module if hasattr(experiment.model, 'module') else experiment.model
    device = experiment.device
    if retriever == 'learned':
        experiment._build_key_bank(log=False)

    space = input_space or model.relation_input_space
    memory_x = torch.from_numpy(experiment.memory_x_np).float().to(device)
    memory_past = transform_relation_history(memory_x, space)
    memory_y, memory_x_last = experiment.memory_y, experiment.memory_x_last

    _, loader = experiment._get_data(flag=split, shuffle=False)
    targets = model.target_channels()
    generator = torch.Generator(device='cpu').manual_seed(0)

    totals, seen = {}, 0
    for batch_x, batch_y, batch_start_idx in loader:
        if seen >= max_queries:
            break
        batch_x, batch_y, batch_start_idx = experiment._move_batch(
            batch_x, batch_y, batch_start_idx
        )
        cand_mask, _ = experiment._candidate_mask(batch_start_idx)
        query_past = transform_relation_history(batch_x, space)

        for channel in targets:
            future_mse = model._future_mse(
                batch_x, batch_y, memory_y, memory_x_last, channel, channel
            )
            q, k = query_past[:, :, channel], memory_past[:, :, channel]

            if retriever == 'learned':
                slot = model.source_slot(channel, channel)
                z_q = model.encoder(model._relation_tensor(batch_x, channel, channel))
                z_k = experiment.key_bank[channel, slot].to(device=device, dtype=z_q.dtype)
                if model.encoder.retrieval_similarity == 'l2':
                    # The encoder skipped its normalization, so the embeddings are
                    # raw and must be scored the same way Stage-1 trained them.
                    scores = -_pairwise_mse(z_q.float(), z_k.float())
                else:
                    scores = torch.matmul(z_q, z_k.transpose(0, 1))
            elif retriever == 'raw_l2':
                scores = -_pairwise_mse(q, k)
            elif retriever == 'raw_cos':
                scores = torch.matmul(
                    torch.nn.functional.normalize(q, dim=-1),
                    torch.nn.functional.normalize(k, dim=-1).transpose(0, 1),
                )
            elif retriever == 'random':
                scores = torch.rand(
                    cand_mask.shape, generator=generator
                ).to(device)
            else:
                raise ValueError(f'unknown retriever: {retriever}')

            masked_scores = scores.masked_fill(~cand_mask, float('-inf'))
            masked_future = future_mse.masked_fill(~cand_mask, float('inf'))
            prob = torch.softmax(masked_scores.float() / tau, dim=-1)
            prob = torch.nan_to_num(prob, nan=0.0)

            row = dict(_student_retrieval_metrics(
                masked_scores.float(), prob, future_mse.float(), cand_mask
            ))
            row.update({
                key: value.mean()
                for key, value in _coverage_at_m(
                    masked_scores.float(), masked_future.float(), cand_mask, top_k
                ).items()
            })
            row.update({
                key: value[torch.isfinite(value)].mean()
                for key, value in _target_sharpness(
                    masked_future.float(), cand_mask, top_k
                ).items()
            })
            random_mse = (
                future_mse.masked_fill(~cand_mask, 0.0).sum(-1)
                / cand_mask.sum(-1).clamp_min(1)
            )
            row['random_future_mse'] = random_mse.mean()

            for key, value in row.items():
                value = value.detach().float()
                if torch.isfinite(value).all():
                    totals.setdefault(key, []).append(value.mean().cpu())

        seen += batch_x.size(0)

    metrics = {key: float(torch.stack(v).mean()) for key, v in totals.items()}
    # Recall@10 near zero does not imply near-random retrieval: this says how
    # much of the achievable random->Oracle improvement was actually captured.
    gap = metrics['random_future_mse'] - metrics[f'oracle_future_mse_at_{top_k}']
    metrics[f'oracle_gap_recovery_at_{top_k}'] = (
        (metrics['random_future_mse'] - metrics[f'retrieved_future_mse_at_{top_k}'])
        / (gap + 1e-8)
    )
    metrics.update({'queries': seen, 'candidates': int(memory_x.size(0)), 'split': split})
    return metrics


CSV_COLUMNS = [
    'dataset', 'pred_len', 'retriever', 'input_space', 'split',
    'oracle_recall_at_1', 'oracle_recall_at_5', 'oracle_recall_at_10',
    'retrieved_future_mse_at_10', 'oracle_future_mse_at_10',
    'retrieval_regret_at_10', 'random_future_mse', 'oracle_gap_recovery_at_10',
    'ndcg_at_10', 'spearman_score_vs_negative_mse',
    *[f'coverage_at_{m}' for m in COVERAGE_DEPTHS],
    'oracle_margin_10', 'oracle_relative_margin_10',
    *[
        f'near_oracle_count_{int(t * 100)}pct_{ref}'
        for t in NEAR_TIE_TOLERANCES for ref in ('cutoff', 'oraclemean')
    ],
    'retrieval_similarity', 'queries', 'candidates', 'checkpoint',
]


def append_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, 'a', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction='ignore')
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True,
                        help='Stage-1 checkpoint; supplies the data/candidate configuration')
    parser.add_argument('--retriever', default='raw_l2', choices=RETRIEVERS)
    parser.add_argument('--input_space', default=None,
                        choices=[None, 'absolute', 'delta_last'],
                        help='raw retrievers only; defaults to the checkpoint setting')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--max_queries', type=int, default=512)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--csv', default='', help='append one row to this CSV')
    parser.add_argument('--json', default='', help='also write the full metric dict here')
    args = parser.parse_args()

    experiment, saved = load_experiment(
        args.checkpoint, load_weights=args.retriever == 'learned'
    )
    metrics = run(
        experiment, args.retriever, args.split, args.max_queries,
        input_space=args.input_space, top_k=args.top_k,
    )
    metrics.update({
        'dataset': saved.data,
        'pred_len': int(saved.pred_len),
        'retriever': args.retriever,
        'input_space': args.input_space or saved.relation_input_space,
        'retrieval_similarity': getattr(saved, 'retrieval_similarity', 'cosine'),
        'checkpoint': args.checkpoint,
    })

    label = f"{metrics['dataset']}/{metrics['pred_len']} {args.retriever} [{args.split}]"
    print(f'=== {label} ===')
    for key in sorted(metrics):
        value = metrics[key]
        print(f'  {key}: {value:.6f}' if isinstance(value, float) else f'  {key}: {value}')
    if args.csv:
        append_csv(args.csv, metrics)
        print(f'appended to {args.csv}')
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
