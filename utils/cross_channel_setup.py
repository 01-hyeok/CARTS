"""Common setup for the cross-channel arms.

Both the ResDirect arms and the ResSel arms need the identical starting point:
one frozen Stage-2 checkpoint (for the base forecast, the memory bank and --
for the selection arms -- the candidate pool), the same split tensors, and the
same train-derived source-channel sets. Building that in one place is what
makes "cross-channel ResSel vs cross-channel ResDirect" a comparison of heads
rather than a comparison of pipelines.

Leakage rules carried over from the modules this composes:
  * base forecasts and candidate residuals come from pasts only
  * query futures appear only in training targets and oracle metrics
  * the source-channel sets are computed on the train split alone
"""

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.cross_channel_context import build_source_index  # noqa: E402
from utils.retrieval_diagnostics import load_stage2, unwrap  # noqa: E402

SPLITS = ('train', 'val', 'test')


def load_context(checkpoint, topk, source_mode='pearson_topk', max_batches=0,
                 metrics_root='./metrics', need_pool=False, pool_m=100,
                 alpha=1.0, splits=SPLITS):
    """Frozen experiment, per-split tensors, source sets and (optional) pools."""
    from scripts.analyze_residual_oracle import prepare
    from utils.utility_selection import build_selection_cache

    experiment, saved = load_stage2(checkpoint)
    model = unwrap(experiment.model)
    experiment._ensure_memory()
    if need_pool:
        experiment._build_key_bank(force=True)

    source_index, correlations, channel_names = build_source_index(
        experiment, saved, topk, source_mode, metrics_root
    )

    data, caches = {}, {}
    for split in splits:
        data[split] = prepare(experiment, split, max_batches)
        if need_pool:
            caches[split] = build_selection_cache(
                experiment, model, data[split], pool_m, alpha
            )
    return {
        'experiment': experiment, 'saved': saved, 'model': model,
        'data': data, 'caches': caches, 'source_index': source_index,
        'correlations': correlations, 'channel_names': channel_names,
    }


def source_rows(source_index, correlations, channel_names, dataset, pred_len,
                topk):
    """One row per (target, source) pair, for metrics/.../source_channels.csv."""
    rows = []
    for target in range(source_index.size(0)):
        for rank in range(source_index.size(1)):
            source = int(source_index[target, rank])
            rows.append({
                'dataset': dataset, 'pred_len': int(pred_len), 'topk': int(topk),
                'target_index': target, 'target_channel': channel_names[target],
                'source_rank': rank, 'source_index': source,
                'source_channel': channel_names[source],
                'pearson': float(correlations[target, rank]),
                'abs_pearson': abs(float(correlations[target, rank])),
            })
    return rows


def encode_candidates(context_encoder, memory_x, index, channel,
                      candidate_context=False):
    """Embeddings for pooled candidates, encoding each window only once.

    index is [B, M] into the memory bank and repeats heavily across a batch, so
    the unique set is encoded and scattered back. Gradients still reach the
    shared encoder -- this is a de-duplication, not a detach.
    """
    flat = index.reshape(-1)
    unique, inverse = torch.unique(flat, return_inverse=True)
    windows = memory_x[unique]                                  # [U, L, C]
    if candidate_context:
        z_unique = context_encoder(windows, channel)
    else:
        z_unique = context_encoder.encoder(windows[:, :, channel])
    return z_unique[inverse].reshape(index.size(0), index.size(1), -1)


def residual_stats(predicted, target):
    """MSE plus the shape-only diagnostics: correlation, cosine, norm error."""
    flat_p = predicted.reshape(-1).double()
    flat_t = target.reshape(-1).double()
    centered_p = flat_p - flat_p.mean()
    centered_t = flat_t - flat_t.mean()
    corr = float(
        (centered_p * centered_t).sum()
        / (centered_p.norm() * centered_t.norm()).clamp_min(1e-12)
    )
    cosine = float(
        torch.nn.functional.cosine_similarity(
            predicted.reshape(predicted.size(0), -1).double(),
            target.reshape(target.size(0), -1).double(), dim=-1
        ).mean()
    )
    norm_p = predicted.reshape(predicted.size(0), -1).norm(dim=-1)
    norm_t = target.reshape(target.size(0), -1).norm(dim=-1)
    norm_error = float((norm_p - norm_t).abs().mean() / norm_t.abs().mean().clamp_min(1e-12))
    return {
        'residual_pred_mse': float((predicted - target).square().mean()),
        'residual_pred_corr': corr,
        'residual_pred_cosine': cosine,
        'residual_norm_error': norm_error,
    }


ATTENTION_COLUMNS = [
    'dataset', 'pred_len', 'arm', 'target_index', 'target_channel',
    'source_rank', 'source_index', 'source_channel', 'pearson',
    'mean_attention', 'std_attention', 'uniform_attention',
    'attention_matches_top_pearson',
]

SOURCE_COLUMNS = [
    'dataset', 'pred_len', 'topk', 'target_index', 'target_channel',
    'source_rank', 'source_index', 'source_channel', 'pearson', 'abs_pearson',
]


@torch.no_grad()
def write_attention_rows(context_encoder, x, context, dataset, pred_len, arm,
                         csv_path, chunk=512):
    """Per-(target, source) attention summary; diagnostic, never a conclusion.

    Also records whether the most-attended source is the one abs-Pearson ranked
    first, which is the only claim here that could be checked independently.
    """
    from utils.retrieval_diagnostics import append_row

    if context_encoder.mixer is None:
        return
    names = context['channel_names']
    channels = context_encoder.source_index.size(0)
    for target in range(channels):
        collected = []
        for start in range(0, x.size(0), chunk):
            _, attention = context_encoder(
                x[start:start + chunk], target, return_attention=True
            )
            if attention is None:
                break
            collected.append(attention.float().cpu())
        if not collected:
            continue
        attention = torch.cat(collected)
        top_attended = int(attention.mean(0).argmax())
        for rank in range(attention.size(-1)):
            source = int(context_encoder.source_index[target, rank])
            append_row(csv_path, {
                'dataset': dataset, 'pred_len': int(pred_len), 'arm': arm,
                'target_index': target, 'target_channel': names[target],
                'source_rank': rank, 'source_index': source,
                'source_channel': names[source],
                'pearson': float(context['correlations'][target, rank]),
                'mean_attention': float(attention[:, rank].mean()),
                'std_attention': float(attention[:, rank].std()),
                'uniform_attention': 1.0 / attention.size(-1),
                'attention_matches_top_pearson': int(top_attended == 0),
            }, ATTENTION_COLUMNS)


def write_source_rows(context, dataset, pred_len, topk, csv_path):
    from utils.retrieval_diagnostics import append_row

    for row in source_rows(context['source_index'], context['correlations'],
                           context['channel_names'], dataset, pred_len, topk):
        append_row(csv_path, row, SOURCE_COLUMNS)
