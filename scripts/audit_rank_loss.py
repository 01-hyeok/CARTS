#!/usr/bin/env python3
"""Is the ranking loss aimed at the thing it was meant to fix?

The formulas check out against hand computation. This asks the design question
instead: the loss exists to decompress the Top-10 that Stage-2 weights, so how
much of it is actually about pairs *inside* that Top-10, and is the margin on the
same scale as the gaps it is supposed to widen?

Measured on a trained checkpoint, no training. Same mining and same score
computation the training path uses.
"""

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.rank_losses import mine_ranking_candidates, ranking_loss  # noqa: E402
from utils.retrieval_diagnostics import append_row  # noqa: E402

COLUMNS = [
    'dataset', 'pred_len', 'split', 'queries', 'top_k', 'margin',
    'pool_size', 'pairs_total', 'pairs_inside_topk', 'fraction_inside_topk',
    'gap_inside_topk_mean', 'gap_outside_topk_mean', 'gap_ratio_outside_inside',
    'margin_satisfied_inside', 'margin_satisfied_outside',
    'loss_share_inside_margin', 'loss_share_inside_ranknet',
    'ranknet_grad_inside_mean', 'ranknet_grad_outside_mean',
    'topk_cosine_spread', 'mined_cosine_spread', 'checkpoint',
]


def load_stage1(checkpoint_path):
    from exp.exp_stage1_relation import Exp_Stage1_Relation

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    args = SimpleNamespace(**checkpoint['args'])
    args.num_workers = 0
    experiment = Exp_Stage1_Relation(args)
    experiment.model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    experiment.model.eval()
    return experiment, args


@torch.no_grad()
def audit(checkpoint, residual_cache, split='test', max_queries=256, top_k=10,
          margin=0.05, top_p=10, hard=30, random_negatives=10):
    from scripts.precompute_residual_teacher import load

    experiment, args = load_stage1(checkpoint)
    model = experiment.model.module if hasattr(experiment.model, 'module') else experiment.model
    experiment._ensure_memory()
    experiment._build_key_bank(log=False)
    device = experiment.device

    cache = load(residual_cache)
    memory_residual = cache['memory_residual'].to(device)
    part = cache['splits'][split if split in cache['splits'] else 'val']

    _, loader = experiment._get_data(flag=split, shuffle=False)
    totals, seen = {}, 0

    for batch_x, batch_y, batch_start_idx in loader:
        if seen >= max_queries:
            break
        batch_x, batch_y, batch_start_idx = experiment._move_batch(
            batch_x, batch_y, batch_start_idx)
        cand_mask, _ = experiment._candidate_mask(batch_start_idx)
        rows = [part['start_to_row'][int(v)] for v in batch_start_idx.cpu().tolist()]
        query_residual = part['query_residual'].index_select(
            0, torch.tensor(rows, dtype=torch.long)).to(device)
        seen += batch_x.size(0)

        for c in model.target_channels():
            slot = model.source_slot(c, c)
            z_q = model.encoder(model._relation_tensor(batch_x, c, c))
            z_mem = experiment.key_bank[c, slot].to(z_q.device, z_q.dtype)
            scores = torch.matmul(z_q, z_mem.transpose(0, 1))
            floor = torch.finfo(scores.dtype).min / 4

            q = query_residual[:, :, c]
            k = memory_residual[:, :, c]
            teacher = -(
                q.square().mean(-1, keepdim=True) + k.square().mean(-1).unsqueeze(0)
                - 2.0 * torch.matmul(q, k.transpose(0, 1)) / q.size(-1)
            ).clamp_min(0.0)

            mined, _ = mine_ranking_candidates(
                teacher, scores, cand_mask, top_p=top_p,
                hard_negatives=hard, random_negatives=random_negatives)
            student_top = scores.masked_fill(~cand_mask, floor).topk(top_k, -1).indices

            mined_scores = scores.gather(1, mined)
            mined_teacher = teacher.gather(1, mined)
            # Which mined slots are also in the Top-K Stage-2 actually weights.
            in_topk = (mined.unsqueeze(-1) == student_top.unsqueeze(-2)).any(-1)

            gap_t = mined_teacher.unsqueeze(-1) - mined_teacher.unsqueeze(-2)
            gap_s = mined_scores.unsqueeze(-1) - mined_scores.unsqueeze(-2)
            upper = torch.triu(torch.ones_like(gap_t[0]), diagonal=1).bool().unsqueeze(0)
            keep = upper & (gap_t.abs() > 1e-6)
            both_inside = in_topk.unsqueeze(-1) & in_topk.unsqueeze(-2) & keep
            outside = keep & ~both_inside

            agreement = torch.sign(gap_t) * gap_s
            margin_term = (margin - agreement).clamp_min(0.0)
            ranknet_term = torch.nn.functional.softplus(-agreement)
            # d/d(agreement) of softplus(-a) is -sigmoid(-a): the per-pair push.
            ranknet_grad = torch.sigmoid(-agreement)

            def mean_over(mask, values):
                total = mask.float().sum().clamp_min(1.0)
                return (values * mask.float()).sum() / total

            topk_scores = scores.gather(1, student_top)
            batch = {
                'pairs_total': keep.float().sum(dim=(-2, -1)),
                'pairs_inside_topk': both_inside.float().sum(dim=(-2, -1)),
                'fraction_inside_topk': both_inside.float().sum(dim=(-2, -1))
                / keep.float().sum(dim=(-2, -1)).clamp_min(1.0),
                'gap_inside_topk_mean': mean_over(both_inside, gap_s.abs()).expand(1),
                'gap_outside_topk_mean': mean_over(outside, gap_s.abs()).expand(1),
                'margin_satisfied_inside': mean_over(
                    both_inside, (agreement >= margin).float()).expand(1),
                'margin_satisfied_outside': mean_over(
                    outside, (agreement >= margin).float()).expand(1),
                'loss_share_inside_margin': (
                    (margin_term * both_inside.float()).sum()
                    / (margin_term * keep.float()).sum().clamp_min(1e-12)).expand(1),
                'loss_share_inside_ranknet': (
                    (ranknet_term * both_inside.float()).sum()
                    / (ranknet_term * keep.float()).sum().clamp_min(1e-12)).expand(1),
                'ranknet_grad_inside_mean': mean_over(both_inside, ranknet_grad).expand(1),
                'ranknet_grad_outside_mean': mean_over(outside, ranknet_grad).expand(1),
                'topk_cosine_spread': (topk_scores.max(-1).values - topk_scores.min(-1).values),
                'mined_cosine_spread': (mined_scores.max(-1).values - mined_scores.min(-1).values),
            }
            for key, value in batch.items():
                totals.setdefault(key, []).append(value.detach().float().cpu().reshape(-1))

    row = {key: float(torch.cat(value).mean()) for key, value in totals.items()}
    row['gap_ratio_outside_inside'] = (
        row['gap_outside_topk_mean'] / max(row['gap_inside_topk_mean'], 1e-12))
    row.update({
        'dataset': args.data, 'pred_len': int(args.pred_len), 'split': split,
        'queries': seen, 'top_k': top_k, 'margin': margin,
        'pool_size': top_p + hard + random_negatives, 'checkpoint': checkpoint,
    })
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--residual_cache', required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--max_queries', type=int, default=256)
    parser.add_argument('--margin', type=float, default=0.05)
    parser.add_argument('--csv', default='')
    args = parser.parse_args()

    row = audit(args.checkpoint, args.residual_cache, args.split,
                args.max_queries, margin=args.margin)
    print(f"=== {row['dataset']}/{row['pred_len']} [{row['split']}] rank-loss audit ===")
    for key in COLUMNS:
        if key == 'checkpoint':
            continue
        value = row[key]
        print(f'  {key}: {value:.6f}' if isinstance(value, float) else f'  {key}: {value}')
    if args.csv:
        append_row(args.csv, row, COLUMNS)


if __name__ == '__main__':
    main()
