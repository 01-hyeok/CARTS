#!/usr/bin/env python3
"""ROUTER-ORACLE-HEADROOM01 -- Phase 1: does per-query best retriever differ
across three retrieval experts?

R0 = raw_absolute_cosine   -- transform_relation_history(x, 'absolute'), cosine
R1 = delta_last_cosine     -- transform_relation_history(x, 'delta_last'), cosine
R2 = future_aligned_individual -- the TRAINED Individual-Oracle/TF/Hard-CE/
     Cosine scratch encoder's free-running K-step policy: EXACTLY
     `scripts.train_factorial_e2e01.run_sequence(..., free_running=True)`,
     reused unmodified (not reimplemented) so it is structurally impossible
     for this diagnostic to leak future information into R2's selection --
     `free_running=True` never reads `futures`/`query_future` inside
     `run_sequence` at all (see that function's own branch: the `if
     free_running:` block only touches `u_hat`, never `u_target`).

All three retrievers are evaluated in a SINGLE pass over one `shuffle=False`
loader per split, computing R0/R1/R2 for the SAME batch/channel inside the
SAME loop iteration -- so query order and `cand_mask` identity across the
three experts holds by construction (`exp._candidate_mask(batch_start_idx)`
is called exactly once per batch and reused by all three), not by a
post-hoc assertion over separately-generated caches.

Retrieved-future VALUE reconstruction (retrieval INPUT space vs VALUE space
kept separate per spec section 34) reuses
`scripts.train_margutil01.memory_value()` unmodified for all three experts --
the same production convention Stage-1/Stage-2/Factorial already use
(`futures = memory_c + offset_c.view(-1, 1, 1)`).

Primary Phase-1 utility is HOST-INDEPENDENT (spec section 9): uniform
(1/K)-weighted aggregate MSE over the K selected futures, per query,
averaged over channels. This never touches Stage-2 host weighting, so a
missing Solar host cannot block Phase 1.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import transform_relation_history
from models.SequentialSetRetriever import SetConditioner
from scripts.train_factorial_e2e01 import encode_raw, run_sequence
from scripts.train_margutil01 import build_experiment, memory_value

EXPERTS = ('raw', 'delta', 'learned')
TOP_K = 10


def _query_id(batch_start_idx):
    """Row identifier: the dataset's own start-index into the raw series --
    unique per query, deterministic, independent of any retriever."""
    return (batch_start_idx if torch.is_tensor(batch_start_idx)
           else torch.as_tensor(batch_start_idx)).cpu()


def raw_cosine_scores(q, k):
    return torch.matmul(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(0, 1))


@torch.no_grad()
def run_split(exp, args, model, set_conditioner, learned_ckpt_sha, split, channels,
             top_k=TOP_K, chunk_size=4096, limit_batches=0, tau_choice=0.1):
    exp._ensure_memory()
    device = exp.device
    memory_x = torch.from_numpy(exp.memory_x_np).float().to(device)
    memory_abs = transform_relation_history(memory_x, 'absolute')
    memory_delta = transform_relation_history(memory_x, 'delta_last')
    memory_y, memory_x_last = exp.memory_y, exp.memory_x_last

    _, loader = exp._get_data(flag=split, shuffle=False)

    rows = []
    n_batches = 0
    for batch_x, batch_y, batch_start_idx in loader:
        if limit_batches and n_batches >= limit_batches:
            break
        n_batches += 1
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        qid = _query_id(batch_start_idx)
        bsz = batch_x.size(0)

        query_abs = transform_relation_history(batch_x, 'absolute')
        query_delta = transform_relation_history(batch_x, 'delta_last')

        neg_inf = float('-inf')
        per_channel = {e: {'uniform_mse': [], 'cand_mse': [], 'picks': []} for e in EXPERTS}
        query_future_by_c = {}

        for c in channels:
            memory_c, offset_c = memory_value(args, batch_x, memory_y, memory_x_last, c)
            query_future = batch_y[:, :, c]
            query_future_by_c[c] = query_future

            scores_raw = raw_cosine_scores(query_abs[:, :, c], memory_abs[:, :, c])
            scores_delta = raw_cosine_scores(query_delta[:, :, c], memory_delta[:, :, c])
            picks_raw = scores_raw.masked_fill(~cand_mask, neg_inf).topk(top_k, dim=-1).indices
            picks_delta = scores_delta.masked_fill(~cand_mask, neg_inf).topk(top_k, dim=-1).indices

            z_q = encode_raw(model, batch_x, c)
            E = encode_raw(model, memory_x, c)
            _, _, picks_learned, _ = run_sequence(
                z_q, E, cand_mask, set_conditioner, None, None,
                None, None, 'individual', 'tf', tau_choice, top_k, chunk_size,
                free_running=True)

            picks_by_expert = {'raw': picks_raw, 'delta': picks_delta, 'learned': picks_learned}
            for e, picks in picks_by_expert.items():
                V = memory_c[picks] + offset_c.view(-1, 1, 1)  # [B, K, pred_len]
                uniform_agg = V.mean(dim=1)  # [B, pred_len]
                uniform_mse = ((uniform_agg - query_future) ** 2).mean(-1)  # [B]
                cand_mse = ((V - query_future.unsqueeze(1)) ** 2).mean(-1).mean(-1)  # [B]
                per_channel[e]['uniform_mse'].append(uniform_mse.cpu())
                per_channel[e]['cand_mse'].append(cand_mse.cpu())
                per_channel[e]['picks'].append(picks.cpu())

        for b in range(bsz):
            row = {'query_id': int(qid[b]), 'split': split}
            for e in EXPERTS:
                um = torch.stack([per_channel[e]['uniform_mse'][ci][b] for ci in range(len(channels))]).mean()
                cm = torch.stack([per_channel[e]['cand_mse'][ci][b] for ci in range(len(channels))]).mean()
                row[f'{e}_uniform_mse'] = float(um)
                row[f'{e}_candidate_mse'] = float(cm)
            rows.append(row)

    return rows


def query_order_hash(rows):
    ids = [r['query_id'] for r in rows]
    h = hashlib.sha256()
    for i in ids:
        h.update(int(i).to_bytes(8, 'big', signed=True))
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--learned_ckpt', required=True,
                    help='TRACK-A-TF-ORACLE-LEARNABILITY01 or track_a_factorial_e2e '
                         'individual_tf_cosine checkpoint (weights ARE loaded from this)')
    ap.add_argument('--reference_ckpt', default=None,
                    help='Stage-1 checkpoint for data/config args; defaults to --learned_ckpt')
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--out_dir', default='results/ROUTER-ORACLE-HEADROOM01')
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--limit_batches', type=int, default=0, help='SMOKE ONLY')
    ap.add_argument('--splits', default='train,val,test')
    cli = ap.parse_args()

    ref = cli.reference_ckpt or cli.learned_ckpt
    learned_blob = torch.load(cli.learned_ckpt, map_location='cpu')
    fp = learned_blob.get('fingerprint', {})
    problems = []
    if fp.get('axis_oracle') != 'individual':
        problems.append(f"axis_oracle={fp.get('axis_oracle')!r} != 'individual'")
    if fp.get('axis_prefix') != 'tf':
        problems.append(f"axis_prefix={fp.get('axis_prefix')!r} != 'tf'")
    if fp.get('axis_score') != 'cosine':
        problems.append(f"axis_score={fp.get('axis_score')!r} != 'cosine'")
    if int(fp.get('top_k', -1)) != int(cli.top_k):
        problems.append(f"top_k={fp.get('top_k')} != {cli.top_k}")
    if problems:
        raise SystemExit(f'[ISSUE][ABORT] {cli.cell}: learned checkpoint fingerprint invalid: '
                         + '; '.join(problems))

    exp, args = build_experiment(ref, {
        'pred_len': cli.pred_len, 'seq_len': cli.pred_len, 'batch_size': 32,
        'seed': 0, 'top_k': cli.top_k, 'tau_topk': 0.1,
    })
    device = exp.device
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.to(device)
    model.load_state_dict(learned_blob['model_state_dict'])
    model.eval()
    d_model = int(args.d_model)
    set_conditioner = SetConditioner(d_model).to(device)
    set_conditioner.load_state_dict(learned_blob['set_conditioner_state_dict'])
    set_conditioner.eval()

    learned_ckpt_sha = hashlib.sha256(Path(cli.learned_ckpt).read_bytes()).hexdigest()

    cell_dir = Path(cli.out_dir) / cli.cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    channels = list(range(int(args.enc_in)))

    all_rows = {}
    for split in cli.splits.split(','):
        rows = run_split(exp, args, model, set_conditioner, learned_ckpt_sha, split,
                         channels, top_k=cli.top_k, chunk_size=cli.chunk_size,
                         limit_batches=cli.limit_batches)
        all_rows[split] = rows
        import csv
        keys = sorted({k for r in rows for k in r})
        with open(cell_dir / f'query_utility_{split}.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        (cell_dir / f'query_order_hash_{split}.json').write_text(json.dumps({
            'split': split, 'n_queries': len(rows), 'hash': query_order_hash(rows),
        }, indent=2))
        print(f'[router_oracle_headroom01] {cli.cell}/{split}: {len(rows)} queries written')

    (cell_dir / 'config_fingerprint.json').write_text(json.dumps({
        'exp': 'ROUTER-ORACLE-HEADROOM01', 'cell': cli.cell, 'pred_len': cli.pred_len,
        'top_k': cli.top_k, 'channels': channels, 'experts': list(EXPERTS),
        'learned_ckpt': cli.learned_ckpt, 'learned_ckpt_sha256': learned_ckpt_sha,
        'learned_fingerprint': fp, 'reference_ckpt': ref,
    }, indent=2))
    (cell_dir / 'checkpoint_provenance.json').write_text(json.dumps({
        'learned_ckpt': cli.learned_ckpt, 'learned_ckpt_sha256': learned_ckpt_sha,
        'learned_fingerprint': fp,
    }, indent=2))
    print(f'[router_oracle_headroom01] {cli.cell} done.')


if __name__ == '__main__':
    main()
