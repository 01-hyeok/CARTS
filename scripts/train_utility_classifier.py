#!/usr/bin/env python3
"""STEP 2 -- can "this candidate helps" be predicted from past information alone?

STEP 1 shows dropping the harmful candidates recovers most of the Oracle gap.
That only matters if the drop decision is learnable without the query future.

    target   y = 1[(E_base - E_k) / E_base > delta]
    input    pair features over past-only embeddings:
             [z_q, z_k, |z_q - z_k|, z_q * z_k]

Leakage rule: query futures build the *label* and nothing else. The classifier
sees only past embeddings, so the same forward pass is valid at inference.

Reported per split so a weak result separates into "cannot fit" and "cannot
generalize". PR-AUC is the headline because positives are the minority.
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.retrieval_diagnostics import append_row, load_stage2, unwrap  # noqa: E402
from scripts.analyze_residual_oracle import _pair_mse, prepare  # noqa: E402

EPS = 1e-8
COLUMNS = [
    'dataset', 'pred_len', 'delta', 'pool_m', 'encoder_mode', 'split',
    'positive_prevalence', 'pr_auc', 'roc_auc', 'precision', 'recall', 'f1',
    'precision_at_5', 'precision_at_10', 'precision_at_20',
    'mean_utility_at_5', 'mean_utility_at_10', 'mean_utility_at_20',
    'positive_utility_rate_at_10', 'pairs', 'checkpoint',
]


class PairClassifier(nn.Module):
    """[z_q, z_k, |z_q-z_k|, z_q*z_k] -> P(useful).

    A pair head instead of another global metric: the previous diagnosis showed
    a single cosine space cannot express the ordering, and "is this pair good"
    does not require one.
    """

    def __init__(self, dim, hidden=256, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 * dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, z_q, z_k):
        # z_q [B, 1, D] broadcast against z_k [B, M, D]
        z_q = z_q.expand_as(z_k)
        return self.net(torch.cat(
            [z_q, z_k, (z_q - z_k).abs(), z_q * z_k], dim=-1
        )).squeeze(-1)


@torch.no_grad()
def build_pairs(experiment, model, split, pool_m, delta, alpha, max_batches=0, chunk=128):
    """Pair embeddings, labels and utilities for one split."""
    data = prepare(experiment, split, max_batches)
    device = experiment.device
    channels = data['query_y'].size(-1)
    n_query = data['query_y'].size(0)
    zq_all, zk_all, y_all, u_all = [], [], [], []

    for start in range(0, n_query, chunk):
        stop = min(start + chunk, n_query)
        cand_mask, _ = experiment._candidate_mask(data['query_start'][start:stop])
        cand_mask = cand_mask.to(device)
        for c in range(channels):
            sources = model.source_channels(c)
            if c not in sources or experiment.key_bank is None:
                continue
            slot = sources.index(c)
            z_q = model._branch_embedding(data['query_x'][start:stop], c, c)
            z_k_bank = experiment.key_bank[c, slot].to(z_q.device, z_q.dtype)
            scores = torch.matmul(z_q, z_k_bank.transpose(0, 1)).masked_fill(
                ~cand_mask, float('-inf')
            )
            width = min(pool_m, z_k_bank.size(0))
            pool = scores.topk(width, dim=-1).indices

            q_res = data['query_residual'][start:stop, :, c]
            k_res = data['memory_residual'][:, :, c]
            horizon = float(k_res.size(-1))
            utility = (
                2.0 * alpha * torch.matmul(q_res, k_res.transpose(0, 1)) / horizon
                - (alpha ** 2) * k_res.square().mean(-1).unsqueeze(0)
            ).gather(1, pool)
            e_base = q_res.square().mean(-1, keepdim=True)
            label = ((utility / (e_base + EPS)) > delta).float()

            zq_all.append(z_q.unsqueeze(1).detach().cpu())
            zk_all.append(z_k_bank[pool].detach().cpu())
            y_all.append(label.cpu())
            u_all.append(utility.cpu())

    return (torch.cat(zq_all), torch.cat(zk_all), torch.cat(y_all), torch.cat(u_all))


def _auc(labels, scores):
    """ROC-AUC and PR-AUC from sorted scores; no sklearn dependency."""
    order = scores.argsort(descending=True)
    y = labels[order]
    positives, negatives = float(y.sum()), float((1 - y).sum())
    if positives == 0 or negatives == 0:
        return float('nan'), float('nan')
    tp = torch.cumsum(y, 0)
    fp = torch.cumsum(1 - y, 0)
    tpr, fpr = tp / positives, fp / negatives
    roc = float(torch.trapz(tpr, fpr))
    precision = tp / torch.arange(1, len(y) + 1, dtype=torch.float32)
    recall = tpr
    pr = float(torch.trapz(precision, recall))
    return roc, pr


def evaluate(classifier, zq, zk, y, u, top_ks=(5, 10, 20), batch=512):
    classifier.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, zq.size(0), batch):
            logits.append(classifier(
                zq[start:start + batch], zk[start:start + batch]
            ).cpu())
    logits = torch.cat(logits)
    flat_y, flat_s = y.reshape(-1), logits.reshape(-1)
    roc, pr = _auc(flat_y, flat_s)
    pred = (flat_s > 0).float()
    tp = float((pred * flat_y).sum())
    precision = tp / max(float(pred.sum()), 1.0)
    recall = tp / max(float(flat_y.sum()), 1.0)
    out = {
        'positive_prevalence': float(flat_y.mean()),
        'pr_auc': pr, 'roc_auc': roc,
        'precision': precision, 'recall': recall,
        'f1': 2 * precision * recall / max(precision + recall, EPS),
        'pairs': int(flat_y.numel()),
    }
    for k in top_ks:
        width = min(k, logits.size(1))
        idx = logits.topk(width, dim=-1).indices
        out[f'precision_at_{k}'] = float(y.gather(1, idx).mean())
        out[f'mean_utility_at_{k}'] = float(u.gather(1, idx).mean())
        if k == 10:
            out['positive_utility_rate_at_10'] = float((u.gather(1, idx) > 0).float().mean())
    return out, logits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--delta', type=float, default=0.0)
    parser.add_argument('--pool_m', type=int, default=100)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--encoder_mode', default='frozen', choices=['frozen'])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--max_batches', type=int, default=0)
    parser.add_argument('--csv', default='')
    parser.add_argument('--save', default='')
    args = parser.parse_args()

    experiment, saved = load_stage2(args.checkpoint)
    model = unwrap(experiment.model)
    experiment._ensure_memory()
    experiment._build_key_bank(force=True)

    splits = {}
    for split in ('train', 'val', 'test'):
        splits[split] = build_pairs(
            experiment, model, split, args.pool_m, args.delta, args.alpha, args.max_batches
        )
        print(f'  {split}: pairs={splits[split][2].numel()} '
              f'positives={float(splits[split][2].mean()):.4f}')

    zq, zk, y, _ = splits['train']
    device = experiment.device
    classifier = PairClassifier(zq.size(-1)).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    positives = float(y.mean())
    pos_weight = torch.tensor([(1 - positives) / max(positives, EPS)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(args.epochs):
        classifier.train()
        permutation = torch.randperm(zq.size(0))
        total = 0.0
        for start in range(0, zq.size(0), args.batch):
            sel = permutation[start:start + args.batch]
            optimizer.zero_grad()
            logits = classifier(zq[sel].to(device), zk[sel].to(device))
            loss = criterion(logits, y[sel].to(device))
            loss.backward(); optimizer.step()
            total += float(loss) * len(sel)
        print(f'  epoch {epoch + 1}/{args.epochs} loss={total / zq.size(0):.5f}')

    classifier_cpu = classifier.to('cpu')
    for split, (a, b, c, d) in splits.items():
        metrics, _ = evaluate(classifier_cpu, a, b, c, d)
        metrics.update({
            'dataset': saved.data, 'pred_len': int(saved.pred_len),
            'delta': args.delta, 'pool_m': args.pool_m,
            'encoder_mode': args.encoder_mode, 'split': split,
            'checkpoint': args.checkpoint,
        })
        print(f"[{split}] PR-AUC={metrics['pr_auc']:.4f} prevalence={metrics['positive_prevalence']:.4f} "
              f"P@10={metrics['precision_at_10']:.4f} meanU@10={metrics['mean_utility_at_10']:+.4f}")
        if args.csv:
            append_row(args.csv, metrics, COLUMNS)
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'state_dict': classifier_cpu.state_dict(),
                    'dim': zq.size(-1), 'delta': args.delta,
                    'pool_m': args.pool_m, 'alpha': args.alpha}, args.save)
        print(f'saved classifier to {args.save}')


if __name__ == '__main__':
    main()
