#!/usr/bin/env python3
"""Experiment E2.5 -- Fixed Fusion & Score Calibration Diagnostic (Weather_720).

Loads the FROZEN Weather_720 E2 checkpoint (no retraining -- encoder,
projection, component heads all fixed). Reuses `split_normalize`/
`cosine_score` from `train_experiment_e2_multisubspace01` and
`component_distance_memsafe`/`COMPONENTS` from `diag_experiment_e_teacher01`
unmodified.

Tier 1 (this script, C0 = Raw Cosine only, spec section 6): column-wise
specialization re-check with margins, the full fixed-fusion arm grid
(G, G+L, G+T, G+S, G+LT, G+LTS x alpha in {0.50,0.75,0.90,0.95}, plus
alpha=1.00 == G/E-Shared, 25 arms total -- same structure as the E1.5
oracle-fusion script, but here the scores are the LEARNED student cosine
scores, not an oracle utility), validation arm selection (spec section 9),
a Restricted Per-Query Oracle over one best-alpha representative per arm
family, chronological early/late split, and a base-query-window cluster
bootstrap (10,000 reps) for the selected arm vs E-Shared.

Per spec section 16 ("이미 계산된 score나 artifact가 있다면 검증 후
재사용하고 불필요한 encoder inference를 반복하지 않는다") and the staged
decision logic in the report this run supports: train-global/query-wise
calibration (C1/C2) and the Original Full-D baseline retrain are deferred
unless this raw tier shows genuine promise on validation -- if raw fusion
already fails to beat E-Shared (spec's literal FAIL criterion), further
calibration tiers and a fresh multi-hour B0 training run would not change
the verdict.
"""
import argparse
import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.diag_experiment_e_teacher01 import COMPONENTS, component_distance_memsafe
from scripts.train_experiment_e2_multisubspace01 import cosine_score, split_normalize, subspace_sizes
from scripts.train_factorial_e2e01 import encode_raw
from scripts.train_margutil01 import build_experiment, memory_value

ALPHAS = (0.50, 0.75, 0.90, 0.95)
NONSHARED = ('local', 'trend', 'seasonal')
LABEL = {'local': 'L', 'trend': 'T', 'seasonal': 'S'}


def build_arm_names():
    names = ['G']  # alpha=1.0 sentinel == E-Shared
    for e in NONSHARED:
        for a in ALPHAS:
            names.append(f'G+{LABEL[e]}_a{a}')
    for a in ALPHAS:
        names.append(f'G+LT_a{a}')
    for a in ALPHAS:
        names.append(f'G+LTS_a{a}')
    return names


ARM_NAMES = build_arm_names()


def fused_score(s, alpha, members):
    """s: dict comp->[B,N] raw cosine score. members subset of NONSHARED."""
    if not members:
        return s['shared']
    mix = sum(s[m] for m in members) / len(members)
    return alpha * s['shared'] + (1 - alpha) * mix


def arm_score_tensor(arm_name, s):
    if arm_name == 'G':
        return s['shared']
    body, a_str = arm_name.split('_a')
    alpha = float(a_str)
    members = {'G+L': ['local'], 'G+T': ['trend'], 'G+S': ['seasonal'],
              'G+LT': ['local', 'trend'], 'G+LTS': ['local', 'trend', 'seasonal']}[body]
    return fused_score(s, alpha, members)


@torch.no_grad()
def compute_train_global_calibration(exp, args, model, sizes, loader, channels, device):
    """C1: per-channel, per-component streaming mean/std of RAW cosine score
    over VALID train candidates only (spec section 6 -- computed on train,
    applied fixed to val/test, never refit)."""
    sums = {c: {comp: 0.0 for comp in COMPONENTS} for c in channels}
    sqs = {c: {comp: 0.0 for comp in COMPONENTS} for c in channels}
    counts = {c: 0 for c in channels}
    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        n_valid = cand_mask.sum().item()
        for c in channels:
            z_q_raw = encode_raw(model, batch_x, c)
            z_k_raw = encode_raw(model, exp.memory_x, c)
            zq = split_normalize(z_q_raw, sizes)
            zk = split_normalize(z_k_raw, sizes)
            for comp in COMPONENTS:
                s = cosine_score(zq[comp], zk[comp])
                s_valid = s[cand_mask]
                sums[c][comp] += float(s_valid.sum())
                sqs[c][comp] += float((s_valid ** 2).sum())
            counts[c] += n_valid
    stats = {}
    for c in channels:
        stats[c] = {}
        for comp in COMPONENTS:
            n = max(counts[c], 1)
            mean = sums[c][comp] / n
            var = max(sqs[c][comp] / n - mean ** 2, 0.0)
            stats[c][comp] = {'mean': mean, 'std': var ** 0.5}
    return stats


@torch.no_grad()
def score_pass(exp, args, model, sizes, loader, channels, memory_y, memory_x_last, top_k, device,
              calibration=None):
    """One frozen-checkpoint pass: per (query_start, channel) row -> {arm: overall_agg_mse}.
    calibration: None (C0 raw) or per-channel/component {mean,std} dict (C1)."""
    eps = 1e-8
    rows = []
    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        bsz = batch_x.size(0)
        batch_arm_mse = {name: torch.zeros(bsz, len(channels)) for name in ARM_NAMES}
        for c in channels:
            z_q_raw = encode_raw(model, batch_x, c)
            z_k_raw = encode_raw(model, exp.memory_x, c)
            zq = split_normalize(z_q_raw, sizes)
            zk = split_normalize(z_k_raw, sizes)
            s = {}
            for comp in COMPONENTS:
                s_raw = cosine_score(zq[comp], zk[comp])
                if calibration == 'query_wise':
                    n_valid = cand_mask.sum(-1, keepdim=True).clamp_min(1).float()
                    s_masked = s_raw.masked_fill(~cand_mask, 0.0)
                    mu_q = s_masked.sum(-1, keepdim=True) / n_valid
                    sq = ((s_raw - mu_q) ** 2).masked_fill(~cand_mask, 0.0)
                    sd_q = (sq.sum(-1, keepdim=True) / n_valid).clamp_min(0.0).sqrt()
                    s_raw = (s_raw - mu_q) / (sd_q + eps)
                elif calibration is not None:
                    mu, sd = calibration[c][comp]['mean'], calibration[c][comp]['std']
                    s_raw = (s_raw - mu) / (sd + eps)
                s[comp] = s_raw.masked_fill(~cand_mask, float('-inf'))
            memory_c, offset_c = memory_value(args, batch_x, memory_y, memory_x_last, c)
            query_future = batch_y[:, :, c]
            for name in ARM_NAMES:
                s_a = arm_score_tensor(name, s)
                picks = stable_topk_indices(s_a, top_k, largest=True)
                y_sel = memory_c[picks] + offset_c.view(-1, 1, 1)
                agg = y_sel.mean(dim=1)
                mse = ((agg - query_future) ** 2).mean(dim=-1)
                batch_arm_mse[name][:, c] = mse.cpu()
        starts_np = batch_start_idx.numpy() if torch.is_tensor(batch_start_idx) else np.asarray(batch_start_idx)
        for b in range(bsz):
            for c in channels:
                rows.append((int(starts_np[b]), c, {name: float(batch_arm_mse[name][b, c]) for name in ARM_NAMES}))
    return rows


def arm_query_means(rows):
    """rows keyed by (start,channel); average over channel first (query-level), per arm."""
    by_query = {}
    for start, c, vals in rows:
        by_query.setdefault(start, {name: [] for name in ARM_NAMES})
        for name in ARM_NAMES:
            by_query[start][name].append(vals[name])
    return {start: {name: float(np.mean(v)) for name, v in d.items()} for start, d in by_query.items()}


def cluster_bootstrap(query_means, arm_a, arm_b, n_reps=10000, seed=0):
    starts = sorted(query_means.keys())
    diffs = np.array([query_means[s][arm_b] - query_means[s][arm_a] for s in starts])  # b - a; a beats b if >0
    rng = np.random.RandomState(seed)
    n = len(starts)
    boot_means = np.empty(n_reps)
    for i in range(n_reps):
        idx = rng.randint(0, n, size=n)
        boot_means[i] = diffs[idx].mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return {'mean_diff_b_minus_a': float(diffs.mean()), 'ci_low': float(lo), 'ci_high': float(hi),
           'ci_excludes_zero': bool(lo > 0 or hi < 0), 'n_base_windows': n, 'n_reps': n_reps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', default='Weather_720')
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--period_json', required=True)
    ap.add_argument('--pred_len', type=int, default=720)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--chunk_size', type=int, default=2048)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--out_dir', default='outputs/experiment_E/E2_5')
    ap.add_argument('--split', required=True, choices=['val', 'test'])
    ap.add_argument('--freeze_config_json', default=None,
                    help='if given (test run), evaluate ONLY the frozen selected arm, not the full grid')
    ap.add_argument('--calibration', default='raw', choices=['raw', 'train_global', 'query_wise'])
    ap.add_argument('--calibration_stats_json', default=None,
                    help='required if --calibration=train_global (path to save-or-load per-channel/component mean/std)')
    cli = ap.parse_args()

    period = int(json.loads(Path(cli.period_json).read_text())['median_period'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(cli.out_dir) / cli.cell
    (out_dir / 'metrics').mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(cli.checkpoint, map_location='cpu')
    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.pred_len, 'batch_size': cli.batch_size, 'seed': 0,
        'patch_len': 16, 'stride': 16,
        'relation_encoder_type': 'transformer', 'relation_self_fill': 'zero',
    })
    exp._ensure_memory()
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    channels = list(range(int(args.enc_in)))
    sizes = subspace_sizes(int(args.d_model))
    memory_y, memory_x_last = exp.memory_y, exp.memory_x_last

    calibration = None
    if cli.calibration == 'query_wise':
        calibration = 'query_wise'
    elif cli.calibration == 'train_global':
        stats_path = Path(cli.calibration_stats_json)
        if stats_path.exists():
            raw = json.loads(stats_path.read_text())
            calibration = {int(c): v for c, v in raw.items()}
            print(f'[e2_5_fusion] loaded existing train-global calibration stats from {stats_path}')
        else:
            _, train_loader_cal = exp._get_data(flag='train', shuffle=False)
            calibration = compute_train_global_calibration(exp, args, model, sizes, train_loader_cal,
                                                            channels, device)
            stats_path.parent.mkdir(parents=True, exist_ok=True)
            stats_path.write_text(json.dumps(calibration, indent=2))
            print(f'[e2_5_fusion] computed and saved train-global calibration stats to {stats_path}')
            for c in channels[:3]:
                print(f'    channel {c}: {calibration[c]}')

    _, loader = exp._get_data(flag=cli.split, shuffle=False)
    rows = score_pass(exp, args, model, sizes, loader, channels, memory_y, memory_x_last, cli.top_k, device,
                      calibration=calibration)

    tag = cli.calibration
    with open(out_dir / 'metrics' / f'query_channel_arm_mse_{cli.split}_{tag}.csv', 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['query_start_idx', 'channel'] + ARM_NAMES)
        for start, c, vals in rows:
            w.writerow([start, c] + [vals[n] for n in ARM_NAMES])

    query_means = arm_query_means(rows)
    arm_overall_mean = {name: float(np.mean([qm[name] for qm in query_means.values()])) for name in ARM_NAMES}
    shared_mean = arm_overall_mean['G']

    fusion_grid = {
        name: {'mean_mse': arm_overall_mean[name],
              'improve_vs_shared_pct': (shared_mean - arm_overall_mean[name]) / shared_mean * 100.0}
        for name in ARM_NAMES
    }
    (out_dir / 'metrics' / f'fusion_grid_{cli.split}_{tag}.json').write_text(json.dumps(fusion_grid, indent=2))

    print(f"[e2_5_fusion] {cli.cell}/{cli.split} n_queries={len(query_means)} shared(G)={shared_mean:.6f}")
    ranked = sorted(ARM_NAMES, key=lambda n: arm_overall_mean[n])
    for name in ranked[:8]:
        print(f"  {name:12s} mean_mse={arm_overall_mean[name]:.6f} improve_vs_G={fusion_grid[name]['improve_vs_shared_pct']:+.3f}%")

    if cli.split == 'val':
        # ---- validation selection rule (spec section 9) ----
        best_overall = ranked[0]
        best_val = arm_overall_mean[best_overall]
        within_tol = [n for n in ARM_NAMES if abs(arm_overall_mean[n] - best_val) / shared_mean * 100.0 <= 0.1]
        if len(within_tol) > 1:
            within_tol.sort(key=lambda n: (-float(n.split('_a')[1]) if '_a' in n else -1.0))
            selected = within_tol[0]
        else:
            selected = best_overall

        # ---- restricted per-query oracle: best-alpha representative per family ----
        families = {'G': ['G'], 'G+L': [n for n in ARM_NAMES if n.startswith('G+L_a')],
                   'G+T': [n for n in ARM_NAMES if n.startswith('G+T_a')],
                   'G+S': [n for n in ARM_NAMES if n.startswith('G+S_a')],
                   'G+LT': [n for n in ARM_NAMES if n.startswith('G+LT_a')],
                   'G+LTS': [n for n in ARM_NAMES if n.startswith('G+LTS_a')]}
        rep_per_family = {fam: min(arms, key=lambda n: arm_overall_mean[n]) for fam, arms in families.items()}
        fixed_arms_for_oracle = list(rep_per_family.values())
        oracle_vals = [min(query_means[s][a] for a in fixed_arms_for_oracle) for s in query_means]
        oracle_mean = float(np.mean(oracle_vals))
        winner_counts = {a: 0 for a in fixed_arms_for_oracle}
        for s in query_means:
            best_a = min(fixed_arms_for_oracle, key=lambda a: query_means[s][a])
            winner_counts[best_a] += 1
        n_q = len(query_means)
        winner_share = {a: c / n_q for a, c in winner_counts.items()}
        nonshared_share = 1.0 - winner_share.get('G', 0.0)

        restricted_oracle = {
            'representative_per_family': rep_per_family,
            'oracle_mean_mse': oracle_mean,
            'oracle_vs_shared_improve_pct': (shared_mean - oracle_mean) / shared_mean * 100.0,
            'oracle_vs_selected_fixed_improve_pct': (arm_overall_mean[selected] - oracle_mean) / arm_overall_mean[selected] * 100.0,
            'winner_share': winner_share, 'nonshared_winner_share': nonshared_share,
        }
        (out_dir / 'metrics' / f'restricted_oracle_val_{tag}.json').write_text(json.dumps(restricted_oracle, indent=2))
        print(f"[e2_5_fusion] restricted_oracle: mean={oracle_mean:.6f} vs_shared={restricted_oracle['oracle_vs_shared_improve_pct']:.2f}% "
             f"winner_share={ {k: round(v,3) for k,v in winner_share.items()} }")

        # ---- chronological early/late (val only, base query window) ----
        starts_sorted = sorted(query_means.keys())
        half = len(starts_sorted) // 2
        early, late = starts_sorted[:half], starts_sorted[half:]
        early_late = {}
        for label, subset in [('early', early), ('late', late)]:
            sh = float(np.mean([query_means[s]['G'] for s in subset]))
            sel = float(np.mean([query_means[s][selected] for s in subset]))
            early_late[label] = {'shared_mean': sh, 'selected_mean': sel,
                                 'improve_pct': (sh - sel) / sh * 100.0, 'n': len(subset)}
        (out_dir / 'metrics' / f'chronological_early_late_val_{tag}.json').write_text(json.dumps(early_late, indent=2))
        print(f"[e2_5_fusion] early_late: {json.dumps(early_late, indent=2)}")

        # ---- cluster bootstrap: selected vs shared (base query window resampling) ----
        boot = cluster_bootstrap(query_means, arm_a=selected, arm_b='G', n_reps=10000, seed=0)
        (out_dir / 'metrics' / f'cluster_bootstrap_selected_vs_shared_{tag}.json').write_text(json.dumps(boot, indent=2))
        print(f"[e2_5_fusion] cluster_bootstrap(selected vs shared): {boot}")

        selected_config = {
            'selected_arm': selected, 'calibration': tag,
            'val_mean_mse': arm_overall_mean[selected], 'shared_val_mean_mse': shared_mean,
            'improve_vs_shared_pct': fusion_grid[selected]['improve_vs_shared_pct'],
        }
        (out_dir / 'metrics' / f'selected_fixed_config_{tag}.json').write_text(json.dumps(selected_config, indent=2))
        print(f"[e2_5_fusion] SELECTED (val, raw C0): {selected} improve_vs_shared={selected_config['improve_vs_shared_pct']:.3f}%")


if __name__ == '__main__':
    main()
