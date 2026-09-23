#!/usr/bin/env python3
"""Patch-level MoE feasibility, Stage 9: past-only router probe.

Trains a tiny MLP router that sees ONLY the query's own past window
(hand-crafted features: mean/std/trend/recent-change/FFT-band-energy/
ACF-at-a-few-lags -- no learned encoder, no candidate information, no
future information anywhere) and predicts which of the 4
already-trained TRACK-A-PATCH-RETRIEVAL-EXPERT01 patch arms will have the
lowest retMSE@10 for that query/channel. The teacher signal
(`r_T(P|q) = softmax(-z(u_P(q))/tau_R)`) is built from TRAIN-split
per-query retMSE@10 already computed by
`diag_patch_moe_feasibility01.py --split train` (reused, not recomputed).

No retraining of any patch arm. No candidate-bank access. No GPU retrieval
math -- this whole script only needs the per-query CSVs already on disk
plus the raw past window (for feature extraction) and is intentionally
CPU-cheap.

Evaluation (val/test, using the SAME per-query CSVs from
diag_patch_moe_feasibility01.py):
  - hard winner Top-1 accuracy (router argmax vs true argmin_P u_P(q))
  - majority-baseline accuracy (always predict the globally most-frequent
    winner arm) for comparison
  - macro F1, confusion matrix
  - router-selected retMSE@10 (router's argmax arm's ALREADY-COMPUTED
    retMSE@10 for that query -- no new retrieval, just an index lookup)
  - recovered headroom = (U_fixed - U_router) / (U_fixed - U_oracle)
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_margutil01 import build_experiment

ARMS = ('native_p16', 'p24', 'p48', 'p120')
TAU_R = 0.5  # pre-specified, not tuned on val/test


def _load_per_query_csv(path):
    """Returns dict: (query_start_idx, channel) -> {arm: retmse10}."""
    rows = {}
    with open(path) as fh:
        r = csv.DictReader(fh)
        for row in r:
            key = (int(row['query_start_idx']), int(row['channel']))
            rows[key] = {a: float(row[f'retmse10_{a}']) for a in ARMS}
    return rows


@torch.no_grad()
def _extract_features(x_c):
    """x_c: [B, L] raw past window for one channel. Returns [B, F] hand-crafted,
    PAST-ONLY features -- no future, no candidate, no learned encoder."""
    mean = x_c.mean(dim=-1, keepdim=True)
    std = x_c.std(dim=-1, keepdim=True)
    n = x_c.size(-1)
    t = torch.linspace(0, 1, n, device=x_c.device).unsqueeze(0)
    t_c = t - t.mean()
    x_c_c = x_c - mean
    trend = (t_c * x_c_c).sum(-1, keepdim=True) / (t_c.pow(2).sum() + 1e-8)
    recent_change = (x_c[:, -1:] - x_c[:, -min(n, 24):].mean(dim=-1, keepdim=True))
    last_val = x_c[:, -1:]
    # FFT band energies (log-magnitude, first few bins = low-freq trend/season)
    fft = torch.fft.rfft(x_c_c, dim=-1)
    mag = fft.abs()
    n_bands = 6
    band_size = max(mag.size(-1) // n_bands, 1)
    bands = [mag[:, i * band_size:(i + 1) * band_size].mean(dim=-1, keepdim=True).clamp_min(1e-8).log()
            for i in range(n_bands)]
    # ACF at a few lags (normalized)
    acf_lags = [1, 4, 12, 24] if n > 24 else [1, 2, 4]
    denom = (x_c_c.pow(2).sum(-1, keepdim=True) + 1e-8)
    acfs = []
    for lag in acf_lags:
        if lag < n:
            acfs.append((x_c_c[:, :-lag] * x_c_c[:, lag:]).sum(-1, keepdim=True) / denom)
        else:
            acfs.append(torch.zeros_like(mean))
    feats = torch.cat([mean, std, trend, recent_change, last_val] + bands + acfs, dim=-1)
    return feats


class Router(nn.Module):
    def __init__(self, in_dim, n_arms=4, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, n_arms))

    def forward(self, x):
        return self.net(x)


def _materialize_past(exp, split, device):
    _, loader = exp._get_data(flag=split, shuffle=False)
    xs, starts = [], []
    for bx, by, bstart in loader:
        xs.append(bx)
        starts.append(bstart if torch.is_tensor(bstart) else torch.as_tensor(bstart))
    return torch.cat(xs, 0).to(device), torch.cat(starts, 0)


def build_dataset(x_all, starts, per_query, channels, device):
    """Returns (features [N*C, F], teacher_u [N*C, A], start_idx [N*C], channel [N*C])."""
    feats_by_c = {c: _extract_features(x_all[:, :, c]) for c in channels}
    feats, u_rows, start_rows, ch_rows = [], [], [], []
    for qi in range(x_all.size(0)):
        s = int(starts[qi])
        for c in channels:
            key = (s, c)
            if key not in per_query:
                continue
            u_rows.append([per_query[key][a] for a in ARMS])
            feats.append(feats_by_c[c][qi])
            start_rows.append(s)
            ch_rows.append(c)
    return (torch.stack(feats), torch.tensor(u_rows, dtype=torch.float32),
           torch.tensor(start_rows), torch.tensor(ch_rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True)
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--pred_len', type=int, default=720)
    ap.add_argument('--feasibility_dir', default='reports/patch_moe_feasibility')
    ap.add_argument('--out_dir', default='reports/patch_moe_feasibility')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=0)
    cli = ap.parse_args()

    torch.manual_seed(cli.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(cli.out_dir) / cli.cell
    out_dir.mkdir(parents=True, exist_ok=True)

    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.pred_len, 'batch_size': 32, 'seed': 0,
    })
    channels = list(range(int(args.enc_in)))

    splits_x, splits_u = {}, {}
    for split in ('train', 'val', 'test'):
        csv_path = Path(cli.feasibility_dir) / cli.cell / split / 'per_query_patch_metrics.csv'
        if not csv_path.exists():
            raise SystemExit(f'[ISSUE][ABORT] missing {csv_path} -- run diag_patch_moe_feasibility01.py '
                             f'--split {split} first (this router probe does not recompute retrieval).')
        per_query = _load_per_query_csv(csv_path)
        x_all, starts = _materialize_past(exp, split, device)
        feats, u, s_idx, ch_idx = build_dataset(x_all, starts, per_query, channels, device)
        splits_x[split] = feats
        splits_u[split] = u
        print(f'[router_probe] {split}: {feats.size(0)} rows, feature_dim={feats.size(1)}')

    in_dim = splits_x['train'].size(1)
    router = Router(in_dim, n_arms=len(ARMS)).to(device)
    opt = torch.optim.Adam(router.parameters(), lr=cli.lr)

    # teacher: z-score u_P(q) across the 4 arms (per row), softmax(-z/tau_R)
    def teacher_dist(u):
        mean = u.mean(dim=-1, keepdim=True)
        std = u.std(dim=-1, keepdim=True).clamp_min(1e-6)
        z = (u - mean) / std
        return torch.softmax(-z / TAU_R, dim=-1)

    x_tr = splits_x['train'].to(device)
    p_t_tr = teacher_dist(splits_u['train'].to(device))

    for epoch in range(cli.epochs):
        router.train()
        opt.zero_grad()
        logits = router(x_tr)
        log_p_s = F.log_softmax(logits, dim=-1)
        loss = -(p_t_tr.detach() * log_p_s).sum(-1).mean()
        loss.backward()
        opt.step()
        if epoch % 50 == 0 or epoch == cli.epochs - 1:
            print(f'[router_probe] epoch {epoch} train_listwise_ce={float(loss):.5f}')

    router.eval()
    results = {}
    for split in ('val', 'test'):
        x = splits_x[split].to(device)
        u = splits_u[split].to(device)
        with torch.no_grad():
            logits = router(x)
            pred_arm = logits.argmax(dim=-1)
            true_arm = u.argmin(dim=-1)
            hard_acc = float((pred_arm == true_arm).float().mean())
            majority_arm = int(true_arm.mode().values) if true_arm.numel() else 0
            majority_acc = float((true_arm == majority_arm).float().mean())

            # macro F1
            n_arms = len(ARMS)
            f1s = []
            confusion = torch.zeros(n_arms, n_arms, dtype=torch.long)
            for p, t in zip(pred_arm.tolist(), true_arm.tolist()):
                confusion[t, p] += 1
            for a in range(n_arms):
                tp = confusion[a, a].item()
                fp = confusion[:, a].sum().item() - tp
                fn = confusion[a, :].sum().item() - tp
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                f1s.append(f1)
            macro_f1 = sum(f1s) / len(f1s)

            router_retmse = float(u.gather(1, pred_arm.unsqueeze(-1)).mean())
            best_fixed_col = u.mean(dim=0).argmin()
            best_fixed_mean = float(u[:, best_fixed_col].mean())
            oracle_mean = float(u.min(dim=-1).values.mean())
            recovered = ((best_fixed_mean - router_retmse) / (best_fixed_mean - oracle_mean)
                        if best_fixed_mean != oracle_mean else float('nan'))

        results[split] = {
            'n_rows': int(x.size(0)),
            'hard_top1_acc': hard_acc, 'majority_baseline_acc': majority_acc,
            'macro_f1': macro_f1, 'confusion_matrix': confusion.tolist(),
            'router_selected_retmse10': router_retmse,
            'best_fixed_arm': ARMS[int(best_fixed_col)], 'best_fixed_mean': best_fixed_mean,
            'scale_oracle_mean': oracle_mean,
            'router_improve_vs_best_fixed_pct': (best_fixed_mean - router_retmse) / best_fixed_mean * 100.0,
            'recovered_headroom_frac': recovered,
        }
        print(f"[router_probe] {split}: hard_acc={hard_acc:.4f} (majority={majority_acc:.4f}) "
             f"macro_f1={macro_f1:.4f} router_retmse10={router_retmse:.6f} "
             f"best_fixed={best_fixed_mean:.6f} oracle={oracle_mean:.6f} "
             f"improve={results[split]['router_improve_vs_best_fixed_pct']:.2f}% "
             f"recovered_headroom={recovered * 100:.1f}%")

    (out_dir / 'router_probe_summary.json').write_text(json.dumps(results, indent=2))
    print(f'[router_probe] done. written to {out_dir / "router_probe_summary.json"}')


if __name__ == '__main__':
    main()
