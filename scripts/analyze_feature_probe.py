#!/usr/bin/env python3
"""OBSERVABILITY 3/4 -- which observable features separate useful candidates,
and does any of it survive a permutation control?

Small probes on purpose. The question is not how well utility can be predicted
but *where the signal lives*, so a linear probe and a two-layer MLP are enough
and a stronger model would only blur the reading.

Feature groups
    A query past statistics        B candidate past statistics
    C query-candidate relation     D candidate residual statistics
    E everything

Permutation controls answer the question a high AUROC cannot: a probe that
scores well on group D might only have learned "some candidates are useful to
everybody". Shuffling candidate residuals across candidates, queries across
queries, or breaking the pairing entirely, destroys query-specific
correspondence while leaving the marginal statistics intact. Real >> shuffled is
the only evidence of a genuine query-candidate signal.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_utility_reranker import batched_residual, load_cache  # noqa: E402
from utils.retrieval_diagnostics import append_row  # noqa: E402

PROBE_COLUMNS = [
    'dataset', 'pred_len', 'pool_m', 'feature_group', 'model', 'label',
    'features', 'train_rows', 'test_rows', 'prevalence', 'auroc', 'pr_auc',
    'precision_at_10', 'utility_spearman',
]
PERM_COLUMNS = [
    'dataset', 'pred_len', 'pool_m', 'feature_group', 'model', 'label',
    'shuffle', 'seeds', 'auroc', 'auroc_std', 'pr_auc', 'pr_auc_std',
    'real_auroc', 'excess_auroc',
]


def window_statistics(window):
    """[N, L] -> [N, F] hand-picked descriptors, no learning involved."""
    length = window.size(-1)
    steps = torch.arange(length, dtype=window.dtype, device=window.device)
    steps = (steps - steps.mean()) / steps.std().clamp_min(1e-8)
    delta = window[:, 1:] - window[:, :-1]
    centered = window - window.mean(-1, keepdim=True)
    spectrum = torch.fft.rfft(centered, dim=-1).abs()
    cut = max(1, spectrum.size(-1) // 4)
    lagged = (centered[:, 1:] * centered[:, :-1]).mean(-1)
    return torch.stack([
        window.mean(-1), window.std(-1), window[:, -1], window[:, -1] - window[:, 0],
        (centered * steps).mean(-1),                        # linear slope
        delta.mean(-1), delta.std(-1),
        lagged / centered.square().mean(-1).clamp_min(1e-8),  # lag-1 autocorrelation
        spectrum[:, :cut].square().mean(-1),                 # low-frequency energy
        spectrum[:, cut:].square().mean(-1),                 # high-frequency energy
    ], dim=-1)


def relation_features(query, candidate):
    """[N, L] vs [N, L] -> pairwise relation descriptors."""
    delta_q = query[:, 1:] - query[:, :-1]
    delta_k = candidate[:, 1:] - candidate[:, :-1]
    normalise = lambda x: x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    centre = lambda x: x - x.mean(-1, keepdim=True)
    slope = lambda x: (centre(x) * torch.linspace(
        -1, 1, x.size(-1), device=x.device, dtype=x.dtype)).mean(-1)
    return torch.stack([
        (query - candidate).square().mean(-1),
        (delta_q - delta_k).square().mean(-1),
        (normalise(query) * normalise(candidate)).sum(-1),
        (normalise(centre(query)) * normalise(centre(candidate))).sum(-1),
        slope(query) - slope(candidate),
        query.std(-1) / candidate.std(-1).clamp_min(1e-8),
        query.mean(-1) - candidate.mean(-1),
        query[:, -1] - candidate[:, -1],
    ], dim=-1)


def build_features(cache, memory_residual, channels, device, max_pairs=200000,
                   seed=0):
    """Flatten (query, channel, candidate) into rows with grouped features."""
    queries, _, width = cache['ids'].shape
    horizon = memory_residual.size(1)
    residual = batched_residual(memory_residual, cache['ids'], channels,
                                torch.device('cpu'))                    # [Q*C, M, H]
    past = cache['query_x'].permute(0, 2, 1).reshape(-1, cache['query_x'].size(1))
    utility = cache['utility'].reshape(-1, width)
    valid = cache['valid'].reshape(-1, width)

    rows = queries * channels
    generator = torch.Generator().manual_seed(seed)
    row_index = torch.arange(rows).repeat_interleave(width)
    candidate_index = torch.arange(width).repeat(rows)
    keep = valid.reshape(-1)
    row_index, candidate_index = row_index[keep], candidate_index[keep]
    if row_index.numel() > max_pairs:
        pick = torch.randperm(row_index.numel(), generator=generator)[:max_pairs]
        row_index, candidate_index = row_index[pick], candidate_index[pick]

    query_past = past[row_index]
    candidate_residual = residual[row_index, candidate_index]
    # Candidate pasts are not stored; the residual and the memory index stand in
    # for the candidate's own statistics, which is what group B is here.
    groups = {
        'A_query_past': window_statistics(query_past),
        'B_candidate': window_statistics(candidate_residual),
        'C_relation': relation_features(query_past[:, -horizon:], candidate_residual),
        'D_candidate_residual': torch.stack([
            candidate_residual.mean(-1), candidate_residual.std(-1),
            candidate_residual.norm(dim=-1),
            (candidate_residual * torch.linspace(-1, 1, horizon)).mean(-1),
            candidate_residual[:, -1],
            candidate_residual.square().mean(-1),
        ], dim=-1),
    }
    groups['E_combined'] = torch.cat(list(groups.values()), dim=-1)
    labels = utility[row_index, candidate_index]
    return groups, labels, row_index


def probe(features, labels, split_at, model_name, seed=0):
    """AUROC / PR-AUC / precision@10 for one feature group."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    x = features.numpy()
    y = (labels.numpy() > 0).astype(np.int64)
    x_train, x_test = x[:split_at], x[split_at:]
    y_train, y_test = y[:split_at], y[split_at:]
    if y_train.min() == y_train.max() or y_test.min() == y_test.max():
        return None
    scaler = StandardScaler().fit(x_train)
    x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
    if model_name == 'linear':
        model = LogisticRegression(max_iter=500, n_jobs=1)
    else:
        model = MLPClassifier(hidden_layer_sizes=(64,), max_iter=120,
                              random_state=seed, early_stopping=True)
    model.fit(x_train, y_train)
    score = model.predict_proba(x_test)[:, 1]
    order = np.argsort(-score)
    depth = max(1, int(0.10 * len(order)))
    return {
        'auroc': float(roc_auc_score(y_test, score)),
        'pr_auc': float(average_precision_score(y_test, score)),
        'precision_at_10': float(y_test[order[:depth]].mean()),
        'prevalence': float(y_test.mean()),
        'features': features.size(-1),
        'train_rows': int(split_at), 'test_rows': int(len(y_test)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache_dir', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--pred_len', type=int, default=96)
    p.add_argument('--pool_m', type=int, default=100)
    p.add_argument('--max_pairs', type=int, default=120000)
    p.add_argument('--models', default='linear,mlp')
    p.add_argument('--seeds', type=int, default=10)
    p.add_argument('--probe_csv', default='')
    p.add_argument('--permutation_csv', default='')
    a = p.parse_args()

    stem = f'{a.dataset}_{a.pred_len}_top{a.pool_m}'
    train_cache = load_cache(Path(a.cache_dir) / f'{stem}_train.pt')
    test_cache = load_cache(Path(a.cache_dir) / f'{stem}_test.pt')
    memory_residual = train_cache['memory_residual']
    channels = train_cache['ids'].size(1)
    device = torch.device('cpu')

    train_groups, train_labels, _ = build_features(
        train_cache, memory_residual, channels, device, a.max_pairs)
    test_groups, test_labels, _ = build_features(
        test_cache, memory_residual, channels, device, a.max_pairs // 2)

    split_at = train_labels.numel()
    real = {}
    for name in train_groups:
        features = torch.cat([train_groups[name], test_groups[name]])
        labels = torch.cat([train_labels, test_labels])
        for model_name in a.models.split(','):
            result = probe(features, labels, split_at, model_name)
            if result is None:
                continue
            real[(name, model_name)] = result
            print(f"[{a.dataset} M={a.pool_m} {name}/{model_name}] "
                  f"AUROC={result['auroc']:.3f} PR-AUC={result['pr_auc']:.3f} "
                  f"prev={result['prevalence']:.3f}")
            if a.probe_csv:
                append_row(a.probe_csv, {
                    'dataset': a.dataset, 'pred_len': a.pred_len,
                    'pool_m': a.pool_m, 'feature_group': name,
                    'model': model_name, 'label': 'utility_positive',
                    'utility_spearman': float('nan'), **result,
                }, PROBE_COLUMNS)

    # Permutation controls, on the combined group and the candidate-residual
    # group -- the two that could most easily look good for the wrong reason.
    for name in ('D_candidate_residual', 'E_combined'):
        for model_name in a.models.split(','):
            if (name, model_name) not in real:
                continue
            for shuffle in ('pair',):
                scores = []
                for seed in range(a.seeds):
                    # Permuting the labels breaks the pairing while leaving both
                    # marginals intact, which is exactly the null the spec wants.
                    generator = torch.Generator().manual_seed(seed)
                    train_perm = torch.randperm(train_labels.numel(), generator=generator)
                    test_perm = torch.randperm(test_labels.numel(), generator=generator)
                    features = torch.cat([train_groups[name], test_groups[name]])
                    labels = torch.cat([train_labels[train_perm], test_labels[test_perm]])
                    result = probe(features, labels, split_at, model_name, seed)
                    if result:
                        scores.append((result['auroc'], result['pr_auc']))
                if not scores:
                    continue
                auroc = [s[0] for s in scores]
                pr = [s[1] for s in scores]
                mean = lambda v: float(np.mean(v))
                std = lambda v: float(np.std(v))
                print(f"  [control {shuffle}] {name}/{model_name} "
                      f"AUROC={mean(auroc):.3f}±{std(auroc):.3f} "
                      f"(real {real[(name, model_name)]['auroc']:.3f})")
                if a.permutation_csv:
                    append_row(a.permutation_csv, {
                        'dataset': a.dataset, 'pred_len': a.pred_len,
                        'pool_m': a.pool_m, 'feature_group': name,
                        'model': model_name, 'label': 'utility_positive',
                        'shuffle': shuffle, 'seeds': len(scores),
                        'auroc': mean(auroc), 'auroc_std': std(auroc),
                        'pr_auc': mean(pr), 'pr_auc_std': std(pr),
                        'real_auroc': real[(name, model_name)]['auroc'],
                        'excess_auroc': real[(name, model_name)]['auroc'] - mean(auroc),
                    }, PERM_COLUMNS)


if __name__ == '__main__':
    main()
