#!/usr/bin/env python3
"""TRACK-A-PATCH-RETRIEVAL-EXPERT01 -- Phase A post-hoc pass/fail gate.

Loads all 4 trained patch-size arms for a cell (native_p16/p24/p48/p120),
recomputes each arm's per-query/channel model_top10_individual_mse on the
SAME split (val, matching the checkpoint-selection convention -- never
test), then asks: does a per-query/channel POST-HOC pick of the best-
performing arm beat the single best FIXED arm?

This post-hoc pick uses each query's own future MSE only to SELECT among
already-computed, already-frozen arm results -- it is a future-aware
Oracle diagnostic, exactly analogous to TRACK-A-HORIZON-RETRIEVAL-
HEADROOM01's block-Oracle diagnostic. It is NEVER a claim about a usable
router (no router is built or trained here) and is not reported as
deployable performance anywhere in this script's output.

Per spec section 3's explicit requirements:
  - improvement must be >= 2% relative on val
  - improvement must not be concentrated in a few exceptional queries
    (checked here via a trimmed-mean re-check dropping the top 5% of
    per-row improvements, plus the winner-share histogram across arms)
  - different patch sizes must actually retrieve DIFFERENT good candidates
    (checked via pairwise Top-10 overlap between arms -- if overlap is
    near 100%, the "improvement" would just be selection noise, not a
    real patch-diversity signal)
  - temporal-overlap dependency between queries (pred_len=720, stride=1
    sliding windows share almost their entire input/future span) is
    accounted for by ALSO reporting the same statistic on a temporally-
    thinned subset (every 720th query, i.e. non-overlapping windows) as a
    conservative robustness check, since the full val set's per-row
    "independent samples" assumption is not literally true.
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
def rows_for_arm(arm_name, cell, checkpoints_root, reference_ckpt, pred_len, top_k, chunk_size, device):
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

    _, loader = exp._get_data(flag='val', shuffle=False)
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
    return torch.cat(ind_mse_rows), torch.cat(picks_rows), torch.cat(starts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True)
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--pred_len', type=int, default=720)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--checkpoints_root', default='checkpoints/track_a_patch_retrieval_expert01')
    ap.add_argument('--out_dir', default='results/TRACK-A-PATCH-RETRIEVAL-EXPERT01')
    ap.add_argument('--thin_stride', type=int, default=720,
                    help='non-overlapping-window robustness subset stride')
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    per_arm_ind_mse, per_arm_picks, starts_ref = {}, {}, None
    for arm in ARMS:
        ind_mse, picks, starts = rows_for_arm(arm, cli.cell, cli.checkpoints_root, cli.reference_ckpt,
                                              cli.pred_len, cli.top_k, cli.chunk_size, device)
        per_arm_ind_mse[arm] = ind_mse  # [N, C]
        per_arm_picks[arm] = picks      # [N, C, K]
        if starts_ref is None:
            starts_ref = starts
        else:
            assert torch.equal(starts, starts_ref), f'{arm}: val batch order/content mismatch vs other arms'
        print(f'[posthoc] {cli.cell}/{arm}: val mean individual_top10_mse='
             f'{float(ind_mse.mean()):.6f}')

    stacked = torch.stack([per_arm_ind_mse[a] for a in ARMS], dim=0)  # [A, N, C]
    n_arms, n_rows, n_ch = stacked.shape

    fixed_means = {a: float(per_arm_ind_mse[a].mean()) for a in ARMS}
    best_fixed_arm = min(fixed_means, key=fixed_means.get)
    best_fixed_mean = fixed_means[best_fixed_arm]

    winner_idx = stacked.argmin(dim=0)          # [N, C] which arm index wins per row
    posthoc_best = stacked.min(dim=0).values     # [N, C]
    posthoc_mean = float(posthoc_best.mean())

    rel_improvement = (best_fixed_mean - posthoc_mean) / best_fixed_mean

    # not-concentrated-in-a-few-queries check: per (query,channel) improvement
    # over the best fixed arm, trimmed-mean dropping the top 5% of individual
    # improvements, re-expressed as a relative improvement of the trimmed mean.
    per_row_improve = (stacked[ARMS.index(best_fixed_arm)] - posthoc_best).flatten()
    k_drop = max(int(0.05 * per_row_improve.numel()), 1)
    trimmed_sorted = per_row_improve.sort(descending=True).values[k_drop:]
    trimmed_posthoc_mean = best_fixed_mean - float(trimmed_sorted.mean())
    trimmed_rel_improvement = (best_fixed_mean - trimmed_posthoc_mean) / best_fixed_mean

    winner_share = {a: float((winner_idx == i).float().mean()) for i, a in enumerate(ARMS)}

    # pairwise Top-10 overlap between arms (mean over rows/channels)
    overlaps = {}
    for a, b in combinations(ARMS, 2):
        pa, pb = per_arm_picks[a], per_arm_picks[b]  # [N, C, K]
        ov = (pa.unsqueeze(-1) == pb.unsqueeze(-2)).any(-1).float().sum(-1) / cli.top_k
        overlaps[f'{a}_vs_{b}'] = float(ov.mean())

    # temporal-overlap robustness: non-overlapping-window thinned subset
    thin_idx = torch.arange(0, n_rows, cli.thin_stride)
    thin_stacked = stacked[:, thin_idx, :]
    thin_fixed_mean = float(thin_stacked[ARMS.index(best_fixed_arm)].mean())
    thin_posthoc_mean = float(thin_stacked.min(dim=0).values.mean())
    thin_rel_improvement = ((thin_fixed_mean - thin_posthoc_mean) / thin_fixed_mean
                            if thin_fixed_mean != 0 else float('nan'))
    n_thin = thin_idx.numel()

    passes_2pct = rel_improvement >= 0.02
    not_concentrated = trimmed_rel_improvement >= 0.02 * 0.5  # trimmed effect retains at least half
    max_overlap = max(overlaps.values())
    genuinely_different_candidates = max_overlap < 0.95  # not near-total overlap

    verdict = 'PROCEED_TO_PHASE_B' if (passes_2pct and not_concentrated and
                                       genuinely_different_candidates) else 'DO_NOT_PROCEED_TO_PHASE_B'

    result = {
        'cell': cli.cell, 'n_val_rows': n_rows, 'n_channels': n_ch, 'top_k': cli.top_k,
        'method': 'FUTURE-AWARE POST-HOC ORACLE DIAGNOSTIC -- picks the best already-frozen '
                 'arm per query/channel using that query\'s own future; NOT a trained or usable '
                 'router; never claim as deployable performance.',
        'per_arm_val_mean_individual_top10_mse': fixed_means,
        'best_fixed_arm': best_fixed_arm, 'best_fixed_mean': best_fixed_mean,
        'posthoc_best_mean': posthoc_mean, 'relative_improvement': rel_improvement,
        'trimmed_top5pct_dropped_posthoc_mean': trimmed_posthoc_mean,
        'trimmed_relative_improvement': trimmed_rel_improvement,
        'winner_arm_share': winner_share,
        'pairwise_top10_overlap': overlaps, 'max_pairwise_overlap': max_overlap,
        'thinned_robustness_check': {
            'thin_stride': cli.thin_stride, 'n_rows_used': int(n_thin),
            'note': 'non-overlapping-window subset (temporal dependency caveat); NOT a formal CI',
            'best_fixed_mean': thin_fixed_mean, 'posthoc_best_mean': thin_posthoc_mean,
            'relative_improvement': thin_rel_improvement,
        },
        'gate_criteria': {
            'relative_improvement_ge_2pct': bool(passes_2pct),
            'not_concentrated_in_few_queries_trimmed_retains_half': bool(not_concentrated),
            'arms_retrieve_genuinely_different_candidates_max_overlap_lt_0.95': bool(genuinely_different_candidates),
        },
        'verdict': verdict,
    }

    out_dir = Path(cli.out_dir) / cli.cell
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'phaseA_posthoc_oracle_gate.json').write_text(json.dumps(result, indent=2))
    print(f"[posthoc] {cli.cell}: best_fixed={best_fixed_arm}({best_fixed_mean:.6f}) "
         f"posthoc={posthoc_mean:.6f} rel_improve={rel_improvement:.4f} "
         f"trimmed_rel_improve={trimmed_rel_improvement:.4f} max_overlap={max_overlap:.4f} "
         f"thinned_rel_improve={thin_rel_improvement:.4f} (n={n_thin}) verdict={verdict}")


if __name__ == '__main__':
    main()
