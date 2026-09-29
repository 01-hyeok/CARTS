#!/usr/bin/env python3
"""TRACK-M-RELEVANCE-CONSTRAINED-MULTISLOT01 -- precompute J1's frozen
train-split individual-relevance reference T_J1(q,c).

NO TRAINING. J1's checkpoint (`checkpoints/track_j2_key_update_decomposition01/
ETTh1_720/J1_stopgrad_key/checkpoint.pth`) is loaded frozen. For every
TRAIN query_start_idx x channel, computes J1's own future-blind hard
Top-10 (student score only, exactly J1's own evaluation protocol) and
the mean raw future-MSE of those 10 picks against that query's OWN true
future -- i.e. `T_J1(q,c) = J1's own individual retMSE`, just measured on
train queries instead of val/test. This is a one-time precompute, looked
up by (query_start_idx, channel) during M1/M2 training -- never
recomputed inside the training loop.

Train-only: uses `exp._get_data(flag='train', shuffle=False)` exclusively;
val/test futures never touch this file. Not a Set Oracle -- Top-10
selection uses ONLY J1's own frozen student score, never future
information.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.eval_l_aligned_population01 import macro_summary, run_population
from scripts.train_factorial_e2e01 import arm_score, encode_raw, individual_utility_memsafe
from scripts.train_j_shared_encoder_drift01 import build_model, state_hash
from scripts.train_margutil01 import memory_value

TOP_K = 10
OUT_DIR = REPO_ROOT / 'results/TRACK-M-RELEVANCE-CONSTRAINED-MULTISLOT01/ETTh1_720/j1_reference'
J1_CKPT = REPO_ROOT / 'checkpoints/track_j2_key_update_decomposition01/ETTh1_720/J1_stopgrad_key/checkpoint.pth'
EXPECTED_INIT_HASH = 'b37fa4031f538e4b5f5c522ae22e7d03613e4ee66283d2637656c7c4f872372e'


class _Cli:
    reference_ckpt = ('checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/'
                      'stage1_carts_softset_ETTh1_720_S0_wce_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_'
                      'nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth')
    cell = 'ETTh1_720'
    pred_len = 720
    seq_len = 720
    patch_len = 16
    top_k = 10
    tau_t = 0.1
    tau_s = 0.1
    batch_size = 32
    learning_rate = 1e-3
    chunk_size = 4096
    init_seed = 0
    loader_seed = 0


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cli = _Cli()
    exp, args, base_model = build_model(cli, device)
    channels = list(range(int(args.enc_in)))
    assert state_hash(base_model) == EXPECTED_INIT_HASH

    bl = torch.load(J1_CKPT, map_location=device)
    m = base_model
    m.load_state_dict(bl['model_state_dict'])
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)

    audit = {'j1_checkpoint': str(J1_CKPT.relative_to(REPO_ROOT)), 'j1_best_epoch': bl.get('epoch'),
             'n_candidates': int(exp.memory_x.size(0)), 'top_k': TOP_K, 'channels': len(channels),
             'split_used': 'train ONLY (flag="train", shuffle=False) -- val/test never touched'}

    _, train_loader = exp._get_data(flag='train', shuffle=False)
    rows = []
    n_queries = 0
    with torch.no_grad():
        for batch_x, batch_y, batch_start_idx in train_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            cand_mask, _ = exp._candidate_mask(batch_start_idx)
            bsz = batch_x.size(0)
            for c in channels:
                z_q = encode_raw(m, batch_x, c)
                z_k = encode_raw(m, exp.memory_x, c)
                s = arm_score(z_q, z_k, None).masked_fill(~cand_mask, float('-inf'))  # J1's own score, future-blind
                memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
                query_future = batch_y[:, :, c]
                u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
                d_raw = -u
                model_idx = stable_topk_indices(s, TOP_K, largest=True)
                t_j1 = d_raw.gather(1, model_idx).mean(-1)  # J1's own hard Top-10, mean raw future-MSE
                for b in range(bsz):
                    rows.append({'query_start_idx': int(batch_start_idx[b]), 'channel': c,
                                'T_J1': float(t_j1[b])})
            n_queries += bsz

    df = pd.DataFrame(rows)
    df.to_parquet(OUT_DIR / 'train_reference.parquet', index=False)
    audit['n_train_queries'] = n_queries
    audit['n_rows'] = len(df)
    audit['T_J1_mean'] = float(df['T_J1'].mean())
    (OUT_DIR / 'audit.json').write_text(json.dumps(audit, indent=2))
    print(f'[precompute_m] wrote {len(df)} rows (n_train_queries={n_queries} x {len(channels)} channels), '
         f'mean T_J1={audit["T_J1_mean"]:.6f}')

    # J1's full-validation baseline (needed for M1/M2's checkpoint-selection constraint),
    # computed via TRACK-L's own unified evaluator (`run_population`/`macro_summary`) on
    # the non-multislot (single-score-vector) J1 path -- Part 8 explicitly requires the
    # "current TRACK-L unified evaluator", not a parallel re-implementation.
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    val_df = run_population('J1', m, None, exp, args, val_loader, channels, device, cli.chunk_size)
    val_summary = macro_summary(val_df)
    val_baseline = {'retmse10': val_summary['retmse10'], 'agg_mse10': val_summary['agg_mse10'],
                    'D': val_summary['D'], 'C': val_summary['C'], 'recall10': val_summary['recall10'],
                    'n_queries_seen': int(val_df['query_start_idx'].nunique())}
    (OUT_DIR / 'validation_baseline.json').write_text(json.dumps(val_baseline, indent=2))
    print(f'[precompute_m] J1 validation baseline (TRACK-L evaluator): '
         f'retmse10={val_baseline["retmse10"]:.6f} agg_mse10={val_baseline["agg_mse10"]:.6f} '
         f'(n_queries={val_baseline["n_queries_seen"]})')


if __name__ == '__main__':
    main()
