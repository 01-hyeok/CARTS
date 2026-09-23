#!/usr/bin/env python3
"""Patch-level MoE feasibility diagnostic (query-level), Stage 0/4/5/6/7 of
the patch_moe_feasibility spec.

Loads the 4 ALREADY-TRAINED TRACK-A-PATCH-RETRIEVAL-EXPERT01 checkpoints for
one cell (no retraining) and re-evaluates them on a given split, this time
saving PER-QUERY results (not just the aggregate means already in
`retrieval_metrics_*.json`) -- retMSE@10 per query/channel/arm and each
arm's Top-10 candidate id set per query/channel, from which winner
distribution, scale-selection oracle headroom, and pairwise Top-10 overlap
are all computed directly (no new retrieval math -- same
`arm_score`/`stable_topk_indices`/`individual_utility_memsafe` already used
by Stage-1 training/eval, reused unmodified).

retMSE@10(q) definition reused verbatim from
`train_patch_retrieval_expert01.py::eval_epoch`'s
`model_top10_individual_mse`: mean over the model's own (future-blind) hard
Top-10 picks of each picked candidate's OWN individual future MSE. This is
NOT the uniform-aggregate reconstruction MSE (`global_h720_mse`) -- the two
are different, documented metrics already present in this repo; this
script only adds the per-query breakdown of the first one.
"""
import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.train_factorial_e2e01 import arm_score, encode_raw, individual_utility_memsafe
from scripts.train_margutil01 import build_experiment, memory_value

ARMS = ('native_p16', 'p24', 'p48', 'p120')
PATCH_LEN = {'native_p16': 16, 'p24': 24, 'p48': 48, 'p120': 120}


@torch.no_grad()
def rows_for_arm_split(arm_name, cell, split, checkpoints_root, reference_ckpt, pred_len,
                       top_k, chunk_size, device):
    """Same computation as diag_patch_retrieval_expert01_posthoc.py's
    `rows_for_arm`, generalized over `split` (that script hardcodes 'val').
    Returns (ind_mse [N,C], picks [N,C,K], starts [N], channels list)."""
    ckpt_path = f'{checkpoints_root}/{cell}/{arm_name}/checkpoint.pth'
    ckpt = torch.load(ckpt_path, map_location='cpu')
    patch_len = PATCH_LEN[arm_name]
    exp, args = build_experiment(reference_ckpt, {
        'pred_len': pred_len, 'seq_len': pred_len, 'batch_size': 32, 'seed': 0,
        'patch_len': patch_len, 'stride': patch_len,
        'relation_encoder_type': 'transformer', 'relation_self_fill': 'zero',
    })
    exp._ensure_memory()
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    channels = list(range(int(args.enc_in)))
    memory_y, memory_x_last = exp.memory_y, exp.memory_x_last

    _, loader = exp._get_data(flag=split, shuffle=False)
    ind_mse_rows, picks_rows, starts = [], [], []
    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        bsz = batch_x.size(0)
        ind_mse_c = torch.zeros(bsz, len(channels))
        picks_c = torch.zeros(bsz, len(channels), top_k, dtype=torch.long)
        for c in channels:
            z_q = encode_raw(model, batch_x, c)
            E = encode_raw(model, exp.memory_x, c)
            memory_c, offset_c = memory_value(args, batch_x, memory_y, memory_x_last, c)
            query_future = batch_y[:, :, c]
            u = individual_utility_memsafe(memory_c, offset_c, query_future, chunk_size)
            d = -u
            s = arm_score(z_q, E, None).masked_fill(~cand_mask, float('-inf'))
            picks = stable_topk_indices(s, top_k, largest=True)
            ind_mse_c[:, c] = d.gather(1, picks).mean(-1).cpu()
            picks_c[:, c, :] = picks.cpu()
        ind_mse_rows.append(ind_mse_c)
        picks_rows.append(picks_c)
        starts.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx)
                      else torch.as_tensor(batch_start_idx))
    return torch.cat(ind_mse_rows), torch.cat(picks_rows), torch.cat(starts), channels


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = values.numel()
    idx = torch.randint(0, n, (n_boot, n), generator=g)
    boot_means = values[idx].mean(dim=1)
    lo = torch.quantile(boot_means, alpha / 2)
    hi = torch.quantile(boot_means, 1 - alpha / 2)
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True)
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--pred_len', type=int, default=720)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--checkpoints_root', default='checkpoints/track_a_patch_retrieval_expert01')
    ap.add_argument('--out_dir', default='reports/patch_moe_feasibility')
    ap.add_argument('--split', default='val', choices=['val', 'test', 'train'])
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(cli.out_dir) / cli.cell / cli.split
    out_dir.mkdir(parents=True, exist_ok=True)

    per_arm_ind_mse, per_arm_picks, starts_ref, channels_ref = {}, {}, None, None
    for arm in ARMS:
        ind_mse, picks, starts, channels = rows_for_arm_split(
            arm, cli.cell, cli.split, cli.checkpoints_root, cli.reference_ckpt,
            cli.pred_len, cli.top_k, cli.chunk_size, device)
        per_arm_ind_mse[arm] = ind_mse
        per_arm_picks[arm] = picks
        if starts_ref is None:
            starts_ref, channels_ref = starts, channels
        else:
            assert torch.equal(starts, starts_ref), f'{arm}: {cli.split} batch order/content mismatch vs other arms'
            assert channels == channels_ref, f'{arm}: channel list mismatch'
        print(f'[patch_moe_feasibility] {cli.cell}/{cli.split}/{arm}: '
             f'mean retMSE@10={float(ind_mse.mean()):.6f} (n_rows={ind_mse.size(0)})')

    n_rows, n_ch = starts_ref.size(0), len(channels_ref)
    stacked = torch.stack([per_arm_ind_mse[a] for a in ARMS], dim=0)  # [A, N, C]

    # ---- per-query flat CSV (query x channel x arm rows) ----
    import csv
    with open(out_dir / 'per_query_patch_metrics.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['query_start_idx', 'channel'] + [f'retmse10_{a}' for a in ARMS] + ['winner_arm', 'winner_retmse10'])
        winner_idx_flat = stacked.argmin(dim=0)
        for qi in range(n_rows):
            start_val = int(starts_ref[qi])
            for c in range(n_ch):
                row_vals = [float(stacked[ai, qi, c]) for ai in range(len(ARMS))]
                w_arm = ARMS[int(winner_idx_flat[qi, c])]
                w.writerow([start_val, c] + row_vals + [w_arm, min(row_vals)])

    # ---- Stage 4: per-arm mean, best fixed, paired diff + bootstrap CI ----
    fixed_means = {a: float(per_arm_ind_mse[a].mean()) for a in ARMS}
    best_fixed_arm = min(fixed_means, key=fixed_means.get)
    best_fixed_mean = fixed_means[best_fixed_arm]
    runner_up = sorted(fixed_means.items(), key=lambda kv: kv[1])[1]

    # ---- Stage 5: winner distribution ----
    winner_idx = stacked.argmin(dim=0)  # [N, C]
    winner_share = {a: float((winner_idx == i).float().mean()) for i, a in enumerate(ARMS)}
    # ties: rows where min value achieved by >1 arm within 1e-9
    min_val = stacked.min(dim=0).values
    tie_mask = (torch.abs(stacked - min_val.unsqueeze(0)) < 1e-9).sum(dim=0) > 1
    tie_rate = float(tie_mask.float().mean())
    # margin between best and 2nd-best per row
    sorted_vals, _ = stacked.sort(dim=0)
    margin = (sorted_vals[1] - sorted_vals[0])  # [N, C]
    winner_improve = {}
    for i, a in enumerate(ARMS):
        mask = winner_idx == i
        winner_improve[a] = float(margin[mask].mean()) if bool(mask.any()) else float('nan')

    # ---- Stage 6: scale-selection oracle ----
    oracle_mean = float(min_val.mean())
    oracle_gain_pct = (best_fixed_mean - oracle_mean) / best_fixed_mean * 100.0
    paired_diff = (per_arm_ind_mse[best_fixed_arm] - min_val).flatten()
    ci_lo, ci_hi = bootstrap_ci(paired_diff)

    # ---- Stage 7: pairwise Top-10 Jaccard overlap + top-1 agreement ----
    overlaps, top1_agree = {}, {}
    for a, b in combinations(ARMS, 2):
        pa, pb = per_arm_picks[a], per_arm_picks[b]  # [N,C,K]
        inter = (pa.unsqueeze(-1) == pb.unsqueeze(-2)).any(-1).float().sum(-1)  # [N,C]
        union = float(cli.top_k) * 2 - inter
        jaccard = (inter / union.clamp_min(1e-9))
        overlaps[f'{a}_vs_{b}'] = float(jaccard.mean())
        top1_agree[f'{a}_vs_{b}'] = float((pa[:, :, 0] == pb[:, :, 0]).float().mean())

    result = {
        'cell': cli.cell, 'split': cli.split, 'n_queries': n_rows, 'n_channels': n_ch, 'top_k': cli.top_k,
        'per_arm_mean_retmse10': fixed_means,
        'best_fixed_arm': best_fixed_arm, 'best_fixed_mean': best_fixed_mean,
        'runner_up_arm': runner_up[0], 'runner_up_mean': runner_up[1],
        'winner_share': winner_share, 'tie_rate': tie_rate,
        'winner_improve_margin_mean': winner_improve,
        'scale_oracle_mean': oracle_mean, 'oracle_gain_pct': oracle_gain_pct,
        'paired_diff_mean': float(paired_diff.mean()),
        'paired_diff_95ci': [ci_lo, ci_hi],
        'pairwise_top10_jaccard': overlaps, 'pairwise_top1_agreement': top1_agree,
        'max_pairwise_jaccard': max(overlaps.values()), 'min_pairwise_jaccard': min(overlaps.values()),
    }
    (out_dir / 'summary.json').write_text(json.dumps(result, indent=2))
    print(f"[patch_moe_feasibility] {cli.cell}/{cli.split}: best_fixed={best_fixed_arm}({best_fixed_mean:.6f}) "
         f"oracle={oracle_mean:.6f} gain={oracle_gain_pct:.2f}% "
         f"winner_share={ {k: round(v,3) for k,v in winner_share.items()} } "
         f"max_jaccard={result['max_pairwise_jaccard']:.4f}")


if __name__ == '__main__':
    main()
