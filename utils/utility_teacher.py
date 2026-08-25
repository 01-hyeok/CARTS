"""Precomputed retrieval teachers keyed by actual downstream forecast utility.

Stage-1 has always been trained against Future-MSE: retrieve the candidates whose
futures most resemble the query's. The alignment diagnostic showed that target is
only weakly related to what actually helps Stage-2 (mean Spearman 0.178). This
module builds the alternatives so the teacher can be swapped without touching the
Stage-1 training loop.

Three scores per (query, candidate, target channel):

    future    -MSE(Y_q, Y_k)                       the existing target
    residual  -MSE(R_q, R_k), R = Y - base         a cheap surrogate for utility
    utility   MSE(Y_q, no-ret) - MSE(Y_q, given k) the thing we actually want

Utility is produced by `RelationStage2.evaluate_candidate_correction`, which
injects one candidate and calls the production forward. Nothing here reconstructs
a base forecast or a fusion by hand -- that reconstruction is what invalidated
the previous generation of these diagnostics.

Two things worth stating plainly about what this measures:

* Utility is defined against a *reference* Stage-2, which was itself trained on a
  Future-KL Stage-1. It is a fixed vantage point, not a neutral one. Every arm
  must share the same reference, and the comparison is between teachers, not
  between reference models.
* The candidate pool is mined with the baseline encoder's own key bank. That
  handicaps the new teachers rather than flattering them: they are scored on the
  pool the incumbent chose.

Leakage: query futures enter only through the teacher targets, which are training
supervision and evaluation labels. No score a retriever consumes at inference
time is derived from them.
"""

from pathlib import Path

import torch

CACHE_VERSION = 2


def load_stage2_reference(checkpoint, stage1_path=''):
    """A fixed Stage-2 with, optionally, a different Stage-1 encoder installed.

    Swapping only the encoder is how a retrieval method gets compared against
    another without also changing the forecaster underneath it -- exactly what
    Stage-2 training itself does at init with a frozen encoder.
    """
    from utils.retrieval_diagnostics import load_stage2, unwrap

    experiment, args = load_stage2(checkpoint)
    if stage1_path:
        unwrap(experiment.model).load_stage1_checkpoint(stage1_path, strict=True)
        experiment.model.eval()
        experiment.key_bank = None
        print(f'[reference] installed Stage-1 encoder from {stage1_path}')
    experiment.reference_checkpoint = checkpoint
    return experiment, args


def _dummy_cache(model, bsz, device, dtype):
    """Zero retrieval branch shaped as `build_retrieval_cache` writes it."""
    slots = model.num_source_slots()
    query_dim = (
        model.relation_emb_dim
        if getattr(model.relation_mixer, 'input_mode', '') == 'retrieved_plus_query'
        else 0
    )
    return {
        'relation_outputs': torch.zeros(bsz, model.channels, slots, model.pred_len,
                                        device=device, dtype=dtype),
        'relation_query_embs': torch.zeros(bsz, model.channels, slots, query_dim,
                                           device=device, dtype=dtype),
    }


@torch.no_grad()
def base_forecast_via_forward(model, x, memory_y, memory_x_last, chunk=512):
    """Base forecast read off `forward`'s own second output.

    Deliberately not `model.base_head(x)`: that returns a delta-space tensor, and
    restoring the offset outside the model is exactly the mistake this file
    exists to avoid.
    """
    outputs = []
    for start in range(0, x.size(0), chunk):
        window = x[start:start + chunk]
        outputs.append(model(
            batch_x=window,
            memory_y=memory_y,
            valid_mask=torch.ones(window.size(0), memory_y.size(0),
                                  dtype=torch.bool, device=window.device),
            key_bank=None,
            memory_x_last=memory_x_last,
            retrieval_cache=_dummy_cache(model, window.size(0), window.device, window.dtype),
        )[1])
    return torch.cat(outputs)


@torch.no_grad()
def branch_scores(model, key_bank, batch_x, target_channel):
    """Student cosine scores of one target channel's self branch, over the bank."""
    sources = model.source_channels(target_channel)
    if target_channel not in sources:
        raise ValueError(
            f'channel {target_channel} does not retrieve from itself; this pipeline '
            'is defined for the self-only protocol (source_mode=auto, relation_top_n=1)'
        )
    slot = sources.index(target_channel)
    z_q = model._branch_embedding(batch_x, target_channel, target_channel)
    z_bank = key_bank[target_channel, slot].to(z_q.device, z_q.dtype)
    return torch.matmul(z_q, z_bank.transpose(0, 1))


def _pairwise_neg_mse(query, candidate):
    """-MSE between paired rows. query [B, K, H], candidate [B, K, H] -> [B, K]."""
    if query.shape != candidate.shape:
        raise ValueError(f'shape mismatch: {tuple(query.shape)} vs {tuple(candidate.shape)}')
    return -(query - candidate).square().mean(-1)


@torch.no_grad()
def build_teacher_cache(experiment, split, pool_m=100, candidate_chunk=25,
                        max_queries=0, batch_limit=0):
    """Mine a fixed candidate pool per (query, channel) and score it three ways.

    The pool comes from the baseline encoder's key bank, so every teacher arm
    later ranks *the same* candidate ids. That is what keeps "subset cost" and
    "teacher effect" separable, per the ablation design.
    """
    model = experiment.model.module if hasattr(experiment.model, 'module') else experiment.model
    model.eval()
    experiment._ensure_memory()
    experiment._build_key_bank(force=True)
    if experiment.key_bank is None:
        raise RuntimeError('teacher precomputation needs a Stage-1 key bank')

    device = experiment.device
    memory_y = experiment.memory_y.to(device)
    memory_x_last = experiment.memory_x_last.to(device)
    memory_x = torch.from_numpy(experiment.memory_bank.memory_x).float().to(device)
    candidate_base = base_forecast_via_forward(model, memory_x, memory_y, memory_x_last)
    candidate_residual = memory_y - candidate_base
    del memory_x, candidate_base

    _, loader = experiment._get_data(flag=split, shuffle=False)
    channels = model.channels
    parts = {key: [] for key in
             ('pool', 'student', 'future', 'residual', 'utility', 'valid', 'base_mse', 'start')}
    seen = 0
    for index, (batch_x, batch_y, batch_start_idx) in enumerate(loader):
        if batch_limit and index >= batch_limit:
            break
        if max_queries and seen >= max_queries:
            break
        batch_x, batch_y, batch_start_idx = experiment._move_batch(batch_x, batch_y, batch_start_idx)
        if max_queries and seen + batch_x.size(0) > max_queries:
            keep = max_queries - seen
            batch_x, batch_y, batch_start_idx = batch_x[:keep], batch_y[:keep], batch_start_idx[:keep]
        seen += batch_x.size(0)
        bsz = batch_x.size(0)

        cand_mask, _ = experiment._candidate_mask(batch_start_idx)
        width = min(pool_m, int(cand_mask.sum(-1).min()))
        if width < 1:
            raise RuntimeError('a query in this split has no valid candidate')

        pool, student = [], []
        for c in range(channels):
            scores = branch_scores(model, experiment.key_bank, batch_x, c)
            scores = scores.masked_fill(~cand_mask, float('-inf'))
            top = scores.topk(width, dim=-1)
            pool.append(top.indices)
            student.append(top.values)
        pool = torch.stack(pool, dim=1)                       # [B, C, M]
        student = torch.stack(student, dim=1)

        query_base = base_forecast_via_forward(model, batch_x, memory_y, memory_x_last)
        query_residual = batch_y - query_base

        future, residual = [], []
        for c in range(channels):
            index_c = pool[:, c]
            future.append(_pairwise_neg_mse(
                batch_y[:, :, c].unsqueeze(1).expand(-1, width, -1),
                memory_y[:, :, c].index_select(0, index_c.reshape(-1)).view(bsz, width, -1),
            ))
            residual.append(_pairwise_neg_mse(
                query_residual[:, :, c].unsqueeze(1).expand(-1, width, -1),
                candidate_residual[:, :, c].index_select(0, index_c.reshape(-1)).view(bsz, width, -1),
            ))
        future = torch.stack(future, dim=1)
        residual = torch.stack(residual, dim=1)

        cache = _dummy_cache(model, bsz, device, batch_x.dtype)
        utility, base_mse = model.evaluate_candidate_correction(
            batch_x=batch_x, batch_y=batch_y, candidate_indices=pool,
            memory_y=memory_y, valid_mask=cand_mask, key_bank=None,
            memory_x_last=memory_x_last, retrieval_cache=cache,
            candidate_chunk=candidate_chunk,
        )
        utility = utility.permute(0, 2, 1)                     # [B, C, M]
        if utility.shape != future.shape:
            raise ValueError(
                f'utility shape {tuple(utility.shape)} != score shape {tuple(future.shape)}'
            )

        parts['pool'].append(pool.cpu())
        parts['student'].append(student.cpu())
        parts['future'].append(future.cpu())
        parts['residual'].append(residual.cpu())
        parts['utility'].append(utility.cpu())
        parts['valid'].append(torch.isfinite(student).cpu())
        parts['base_mse'].append(base_mse.cpu())
        parts['start'].append(batch_start_idx.cpu())

    built = {key: torch.cat(value) for key, value in parts.items()}
    built['pool'] = built['pool'].long()
    starts = built.pop('start').tolist()
    built['starts'] = torch.tensor(starts, dtype=torch.long)
    built['start_to_row'] = {int(start): row for row, start in enumerate(starts)}
    built['meta'] = {
        'version': CACHE_VERSION, 'split': split, 'pool_m': pool_m,
        'dataset': experiment.args.data, 'pred_len': int(experiment.args.pred_len),
        'seq_len': int(experiment.args.seq_len), 'channels': channels,
        'queries': len(starts), 'reference_stage2': getattr(experiment, 'reference_checkpoint', ''),
    }
    return built


def save_cache(cache, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, path)
    return path


def load_cache(path):
    cache = torch.load(path, map_location='cpu')
    version = cache.get('meta', {}).get('version')
    if version != CACHE_VERSION:
        raise ValueError(
            f'teacher cache at {path} is version {version}, expected {CACHE_VERSION}; '
            'rebuild it rather than mixing formats'
        )
    return cache


def rows_for_starts(cache, batch_start_idx):
    """Map a batch's window ids onto cache rows, or None if any are missing."""
    try:
        rows = [cache['start_to_row'][int(value)] for value in batch_start_idx.cpu().tolist()]
    except KeyError:
        return None
    return torch.tensor(rows, dtype=torch.long)


def teacher_scores(cache, name):
    """The score matrix a teacher ranks by, higher meaning better."""
    if name not in ('future', 'residual', 'utility'):
        raise ValueError(f'unknown teacher: {name}')
    return cache[name]
