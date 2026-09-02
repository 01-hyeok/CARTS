#!/usr/bin/env python3
"""PHASE 1 -- freeze one shortlist so every reranker arm sees the same thing.

Per query and target channel this stores the frozen retriever's Top-M ids, the
retriever scores that produced them, the frozen Stage-1 query embedding, and the
measured downstream utility of every shortlisted candidate.

The utility labels come from `evaluate_candidate_correction`, i.e. the
production forward with one candidate injected -- the same helper Phase 0 used,
so learned arms and the oracle are scored on one definition.

Leakage: `utility` is a label. It is written to the cache because training and
evaluation need it; nothing in the reranker's forward signature can read it.
Train, val and test are cached to separate files and never merged.
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_forecast_utility_alignment import base_forecast_via_forward  # noqa: E402
from scripts.analyze_set_level_utility import selected_candidates  # noqa: E402
from utils.retrieval_diagnostics import unwrap  # noqa: E402
from utils.utility_teacher import _dummy_cache, load_stage2_reference  # noqa: E402


@torch.no_grad()
def retriever_scores_for(model, key_bank, batch_x, ids):
    """The frozen retriever's own score for the shortlisted ids, [B, C, M].

    Recomputed from the same embeddings `selected_candidates` ranks with, so the
    stored score is the one that produced the ordering rather than a proxy.
    """
    out = []
    for slot_i, c in enumerate(model.target_channels()):
        sources = model.source_channels(c)
        slot = sources.index(c)
        z_q = model._branch_embedding(batch_x, c, c)                      # [B, d]
        z_mem = model._branch_memory(key_bank, c, slot, c, z_q.dtype, batch_x.device)
        picked = z_mem.index_select(0, ids[:, slot_i].reshape(-1)).view(
            ids.size(0), ids.size(2), -1)                                 # [B, M, d]
        if model.retrieval_similarity == 'l2':
            score = -(z_q.unsqueeze(1) - picked).square().mean(-1)
        else:
            score = (z_q.unsqueeze(1) * picked).sum(-1)
        out.append(score)
    return torch.stack(out, dim=1)


@torch.no_grad()
def query_embeddings(model, batch_x):
    return torch.stack(
        [model._branch_embedding(batch_x, c, c) for c in model.target_channels()],
        dim=1,
    )                                                                     # [B, C, d]


@torch.no_grad()
def build(checkpoint, split, pool_m, max_queries, candidate_chunk=10):
    experiment, args = load_stage2_reference(checkpoint)
    model = unwrap(experiment.model)
    experiment._ensure_memory()
    experiment._build_key_bank(force=True)
    _, loader = experiment._get_data(flag=split, shuffle=False)

    device = experiment.device
    memory_y = experiment.memory_y.to(device)
    memory_x_last = experiment.memory_x_last.to(device)
    memory_x = torch.from_numpy(experiment.memory_bank.memory_x).float().to(device)

    # Candidate residuals, once: the canonical base forecast is forward's own
    # second output, never `base_head`, whose delta space is what earlier
    # diagnostics got wrong.
    memory_base = base_forecast_via_forward(model, memory_x, memory_y, memory_x_last)
    memory_residual = (memory_y - memory_base).cpu()

    # query_x / query_residual / query_future are stored for the observability
    # feature ladder. The residual and the future are ORACLE-ONLY columns: the
    # deployable arms may not read them, and the report marks which arm does.
    store = {name: [] for name in ('ids', 'scores', 'valid', 'z_query',
                                   'utility', 'base_mse', 'query_start',
                                   'query_x', 'query_residual', 'query_future')}
    seen = 0
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
        ids = selected_candidates(model, experiment.key_bank, batch_x, cand_mask, pool_m)
        scores = retriever_scores_for(model, experiment.key_bank, batch_x, ids)
        utility, base_mse = model.evaluate_candidate_correction(
            batch_x=batch_x, batch_y=batch_y, candidate_indices=ids,
            memory_y=memory_y, valid_mask=cand_mask, key_bank=None,
            memory_x_last=memory_x_last,
            retrieval_cache=_dummy_cache(model, batch_x.size(0), device, batch_x.dtype),
            candidate_chunk=candidate_chunk,
        )
        valid = cand_mask.unsqueeze(1).expand(-1, ids.size(1), -1).gather(2, ids)

        store['ids'].append(ids.cpu().int())
        store['scores'].append(scores.float().cpu())
        store['valid'].append(valid.cpu())
        store['z_query'].append(query_embeddings(model, batch_x).float().cpu())
        store['utility'].append(utility.permute(0, 2, 1).float().cpu())    # [B, C, M]
        store['base_mse'].append(base_mse.float().cpu())
        store['query_start'].append(batch_start_idx.cpu())
        query_base = base_forecast_via_forward(model, batch_x, memory_y, memory_x_last)
        store['query_x'].append(batch_x.float().cpu())
        store['query_residual'].append((batch_y - query_base).float().cpu())
        store['query_future'].append(batch_y.float().cpu())

    cache = {name: torch.cat(value) for name, value in store.items()}
    cache.update({
        'dataset': args.data, 'pred_len': int(args.pred_len), 'split': split,
        'pool_m': int(pool_m), 'queries': seen, 'checkpoint': checkpoint,
        'memory_residual': memory_residual,
    })
    expected = (seen, len(model.target_channels()), min(pool_m, memory_y.size(0)))
    if tuple(cache['ids'].shape) != expected:
        raise ValueError(f'shortlist shape {tuple(cache["ids"].shape)} != {expected}')
    return cache


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split', required=True, choices=['train', 'val', 'test'])
    p.add_argument('--pool_m', type=int, default=100)
    p.add_argument('--max_queries', type=int, default=0)
    p.add_argument('--candidate_chunk', type=int, default=10)
    p.add_argument('--out', required=True)
    a = p.parse_args()

    cache = build(a.checkpoint, a.split, a.pool_m, a.max_queries, a.candidate_chunk)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, a.out)
    print(
        f"[{cache['dataset']}/{cache['pred_len']} {a.split} M={cache['pool_m']}] "
        f"queries={cache['queries']} ids={tuple(cache['ids'].shape)} "
        f"positive_rate={float((cache['utility'] > 0).float().mean()):.3f} -> {a.out}"
    )


if __name__ == '__main__':
    main()
