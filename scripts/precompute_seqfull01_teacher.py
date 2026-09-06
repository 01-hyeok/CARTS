#!/usr/bin/env python3
"""EXP-SEQFULL01: offline, fixed Full-Memory Weighted Set Oracle teacher.

Reuses utils/oracle_intervention.py::select_greedy_weighted_set (the exact
greedy construction validated in EXP-1/EXP-2) with the candidate support set
to the WHOLE valid memory bank (no Top-M pool) -- built the same way EXP-1/2's
"FULL" arm was: skip the cosine-topk in build_common_support entirely and use
every valid candidate as the support.

The aggregation score used for the softmax weights inside the greedy search is
the frozen B0 Stage-2 checkpoint's own retrieval score (plain cosine for the
S0_wce checkpoint) -- fixed once, offline, never the student's own changing
score. Only the ORDERED candidate-id sequence (i_1*, ..., i_10*) is cached;
the values/futures used to build it are not, since they are cheap to recompute
and would otherwise dominate the cache size.

Leakage note: query_target (Y_q) is used here, inside the teacher only. It
never reaches anything the student's forward pass consumes.
"""
import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.oracle_intervention import build_common_support, select_greedy_weighted_set
from utils.retrieval_diagnostics import load_stage2

CACHE_VERSION = 1


@torch.no_grad()
def build_split_sequences(experiment, split, k, tau):
    model = experiment.model.module if hasattr(experiment.model, 'module') else experiment.model
    device = experiment.device
    memory_y = experiment.memory_y.to(device)
    memory_x_last = experiment.memory_x_last.to(device)
    key_bank = experiment.key_bank

    _, loader = experiment._get_data(flag=split, shuffle=False)
    per_channel = {int(c): [] for c in model.target_channels()}
    starts = []
    dropped = 0

    for batch_x, batch_y, batch_start_idx in loader:
        batch_x, batch_y, batch_start_idx = experiment._move_batch(
            batch_x, batch_y, batch_start_idx)
        cand_mask, counts = experiment._candidate_mask(batch_start_idx)
        valid_query = counts.to(batch_x.device) >= k
        dropped += int((counts.to(batch_x.device) < k).sum())
        starts.extend(int(v) for v in batch_start_idx.cpu().tolist())

        for c in model.target_channels():
            # relation_top_n=1 self-only: the single source is the target
            # itself, so source_slot is always 0 and r == c.
            source_slots = model.source_channels(c)
            if len(source_slots) != 1 or int(source_slots[0]) != int(c):
                raise ValueError(
                    f'EXP-SEQFULL01 is self-only; channel {c} has sources '
                    f'{source_slots}, expected [{c}]')
            slot = 0
            z_q = model._branch_embedding(batch_x, c, c)
            z_mem = model._branch_memory(key_bank, c, slot, c, z_q.dtype, batch_x.device)
            cosine = torch.matmul(z_q, z_mem.transpose(0, 1))
            score_fn = model._retrieval_score_fn()
            learned_full = score_fn(z_q, z_mem) if score_fn is not None else cosine

            # FULL memory support: width = N, so build_common_support's topk
            # degenerates to "every valid candidate", exactly EXP-1/2's FULL arm.
            n_total = cosine.size(-1)
            pool_idx, pool_valid = build_common_support(cosine, cand_mask, n_total, k)
            learned = learned_full.gather(1, pool_idx)

            memory_c, offset_c = model._memory_value(batch_x, memory_y, memory_x_last, c)
            tgt = memory_c[pool_idx] + offset_c.view(-1, 1, 1)
            q_tgt = batch_y[:, :, c]

            teacher_idx_local = select_greedy_weighted_set(
                tgt, q_tgt, learned, pool_valid, k, tau)
            teacher_idx_global = pool_idx.gather(1, teacher_idx_local)
            # Invalid for any query with < k valid candidates -- excluded from
            # training/eval on this channel, matching the oracle_intervention
            # convention (never padded).
            teacher_idx_global = torch.where(
                valid_query.unsqueeze(-1), teacher_idx_global,
                torch.full_like(teacher_idx_global, -1))
            per_channel[int(c)].append(teacher_idx_global.cpu())

    out = {
        int(c): torch.cat(parts, dim=0) for c, parts in per_channel.items()
    }
    starts_t = torch.tensor(starts, dtype=torch.long)
    print(f'  {split}: {len(starts)} queries, dropped(<k valid)={dropped}')
    return {
        'teacher_idx': out,                    # {channel: [Q, K] global candidate ids, -1 = invalid query}
        'starts': starts_t,
        'start_to_row': {s: i for i, s in enumerate(starts)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True,
                         help='Frozen B0 Stage-2 checkpoint providing the fixed reference score')
    parser.add_argument('--out_dir', default='./cache/seqfull01_teacher')
    parser.add_argument('--splits', default='train,val,test')
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    experiment, saved = load_stage2(args.checkpoint)
    experiment._ensure_memory()
    experiment._build_key_bank()

    target = Path(args.out_dir) / f'{saved.data}_pred{saved.pred_len}.pt'
    if target.exists() and not args.force:
        print(f'[skip] {target} already exists')
        return

    print(f'[teacher] reference checkpoint={args.checkpoint}')
    print(f'[teacher] k={args.top_k} tau={args.tau}')
    cache = {'splits': {}, 'meta': {
        'version': CACHE_VERSION,
        'dataset': saved.data,
        'pred_len': int(saved.pred_len),
        'seq_len': int(saved.seq_len),
        'top_k': int(args.top_k),
        'tau': float(args.tau),
        'reference_stage2_checkpoint': args.checkpoint,
    }}
    for split in [s.strip() for s in args.splits.split(',')]:
        cache['splits'][split] = build_split_sequences(
            experiment, split, args.top_k, args.tau)

    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, target)
    print(f'[done] {target}')


def load(path):
    cache = torch.load(path, map_location='cpu')
    if cache.get('meta', {}).get('version') != CACHE_VERSION:
        raise ValueError(f'seqfull01 teacher cache at {path} has the wrong version; rebuild it')
    return cache


if __name__ == '__main__':
    main()
