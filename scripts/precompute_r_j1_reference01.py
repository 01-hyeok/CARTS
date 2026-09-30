#!/usr/bin/env python3
"""TRACK-R-FINAL-METHOD-GENERALIZATION01 -- generalized (dataset x
horizon x seed) J1 train-relevance reference `T_J1(q,c)` precompute,
parameterized version of `precompute_m_j1_reference01.py`. Train split
only (PART 13: NO val/test future), using the SAME setting's own frozen
J1 checkpoint (not a different horizon/seed's).
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.train_factorial_e2e01 import arm_score, encode_raw, individual_utility_memsafe
from scripts.train_margutil01 import build_experiment, memory_value

TOP_K = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--j1_checkpoint', required=True)
    ap.add_argument('--out_dir', required=True)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.seq_len, 'batch_size': 32, 'seed': cli.seed, 'top_k': TOP_K,
        'relation_encoder_type': 'mlp', 'relation_self_fill': 'linear', 'relation_input_space': 'delta_last',
        'relation_teacher_space': 'delta_last', 'relation_value_space': 'delta_last', 'candidate_mask': 'raft',
        'patch_len': 16, 'stride': 16,
    })
    exp._ensure_memory()
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.to(device)

    bl = torch.load(cli.j1_checkpoint, map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    channels = list(range(int(args.enc_in)))
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit = {'j1_checkpoint': str(cli.j1_checkpoint), 'j1_best_epoch': bl.get('epoch'),
             'n_candidates': int(exp.memory_x.size(0)), 'top_k': TOP_K, 'channels': len(channels),
             'split_used': 'train ONLY -- val/test never touched'}

    _, train_loader = exp._get_data(flag='train', shuffle=False)
    rows, n_queries = [], 0
    with torch.no_grad():
        for batch_x, batch_y, batch_start_idx in train_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            cand_mask, _ = exp._candidate_mask(batch_start_idx)
            bsz = batch_x.size(0)
            for c in channels:
                z_q = encode_raw(model, batch_x, c)
                z_k = encode_raw(model, exp.memory_x, c)
                s = arm_score(z_q, z_k, None).masked_fill(~cand_mask, float('-inf'))
                memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
                query_future = batch_y[:, :, c]
                u = individual_utility_memsafe(memory_c, offset_c, query_future, 4096)
                d_raw = -u
                model_idx = stable_topk_indices(s, TOP_K, largest=True)
                t_j1 = d_raw.gather(1, model_idx).mean(-1)
                for b in range(bsz):
                    rows.append({'query_start_idx': int(batch_start_idx[b]), 'channel': c, 'T_J1': float(t_j1[b])})
            n_queries += bsz

    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / 'train_reference.parquet', index=False)
    audit['n_train_queries'] = n_queries
    audit['n_rows'] = len(df)
    audit['T_J1_mean'] = float(df['T_J1'].mean())
    (out_dir / 'audit.json').write_text(json.dumps(audit, indent=2))
    print(f'[precompute_r_j1_ref] wrote {len(df)} rows, mean T_J1={audit["T_J1_mean"]:.6f}')

    # PART 14: J1's full-validation baseline (retmse10/agg_mse10), needed for M2's
    # constrained (feasibility-gated) checkpoint selection -- same formula as the train loop above.
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    val_sums, n_val = {}, 0
    with torch.no_grad():
        for batch_x, batch_y, batch_start_idx in val_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            cand_mask, _ = exp._candidate_mask(batch_start_idx)
            bsz = batch_x.size(0)
            per_ch = {}
            for c in channels:
                z_q = encode_raw(model, batch_x, c)
                z_k = encode_raw(model, exp.memory_x, c)
                s = arm_score(z_q, z_k, None).masked_fill(~cand_mask, float('-inf'))
                memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
                query_future = batch_y[:, :, c]
                u = individual_utility_memsafe(memory_c, offset_c, query_future, 4096)
                d_raw = -u
                model_idx = stable_topk_indices(s, TOP_K, largest=True)
                model_ind_mse = d_raw.gather(1, model_idx).mean(-1)
                y_sel = memory_c[model_idx] + offset_c.view(-1, 1, 1)
                agg_pred = y_sel.mean(dim=1)
                agg_mse = ((agg_pred - query_future) ** 2).mean(-1)
                per_ch.setdefault('retmse10', []).append(model_ind_mse.cpu())
                per_ch.setdefault('agg_mse10', []).append(agg_mse.cpu())
            for k_, vals in per_ch.items():
                val_sums[k_] = val_sums.get(k_, 0.0) + torch.cat(vals).sum().item()
            n_val += bsz
    val_baseline = {k_: v / max(n_val * len(channels), 1) for k_, v in val_sums.items()}
    val_baseline['n_queries_seen'] = n_val
    (out_dir / 'validation_baseline.json').write_text(json.dumps(val_baseline, indent=2))
    print(f'[precompute_r_j1_ref] J1 validation baseline: retmse10={val_baseline["retmse10"]:.6f} '
         f'agg_mse10={val_baseline["agg_mse10"]:.6f} (n={n_val})')


if __name__ == '__main__':
    main()
