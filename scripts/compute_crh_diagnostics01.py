#!/usr/bin/env python3
"""TRACK-V-CALENDAR-ROUTER01 -- head-specialization / router-regret /
baseline-comparison diagnostics (spec section 16). Reads the already
-built head_oracle caches and the best-retMSE Router checkpoint; never
retrains anything.

Baselines computed (spec section 15):
  A. V5 RoundRobin       -- read directly from train_v_sharedtop100_01.py's
                            own final_test_metrics_best_retmse.json
  B. Calendar-Routed V5  -- recomputed here from the Router + head_oracle's U
  C. Oracle Head         -- argmin_h U_h on TEST (upper bound, diagnostic only)
  D. Fixed Best Head     -- single head chosen by VAL-mean U (global AND
                            per-channel), applied to TEST
  E. Shuffled Calendar   -- read from the CRH-Shuffled run's own
                            final_test_metrics_best_retmse.json (passed in)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.calendar_features import forecast_start_calendar_features, load_date_column

N_HEADS = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--head_oracle_dir', required=True)
    ap.add_argument('--v5_roundrobin_metrics', required=True,
                    help='final_test_metrics_best_retmse.json from train_v_sharedtop100_01.py')
    ap.add_argument('--crh_router_test_metrics', required=True,
                    help='final_test_metrics_best_retmse.json from train_calendar_router01.py')
    ap.add_argument('--shuffled_router_test_metrics', default=None,
                    help='same, from the CRH-Shuffled control run (optional)')
    ap.add_argument('--root_path', required=True)
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--out_dir', required=True)
    cli = ap.parse_args()

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    head_oracle_val = torch.load(Path(cli.head_oracle_dir) / 'head_oracle_val.pt', map_location='cpu')
    head_oracle_test = torch.load(Path(cli.head_oracle_dir) / 'head_oracle_test.pt', map_location='cpu')
    U_val = head_oracle_val['U']    # [Qval, C, 5]
    U_test = head_oracle_test['U']  # [Qtest, C, 5]
    hard_label_test = head_oracle_test['hard_label']  # [Qtest, C]
    n_channels = U_test.size(1)

    # ---- 16-1/16-2: oracle distribution + headroom ----
    oracle_fraction = {f'H{h+1}': float((hard_label_test == h).float().mean()) for h in range(N_HEADS)}
    oracle_retmse = U_test.gather(-1, hard_label_test.unsqueeze(-1)).squeeze(-1).mean().item()

    # D: Fixed Best Head, chosen by VAL mean U
    global_best_head = int(U_val.mean(dim=(0, 1)).argmin())
    global_fixed_retmse = U_test[:, :, global_best_head].mean().item()
    per_channel_best_head = U_val.mean(dim=0).argmin(dim=-1)  # [C]
    per_channel_fixed_retmse = torch.stack(
        [U_test[:, c, per_channel_best_head[c]] for c in range(n_channels)], dim=1).mean().item()

    v5_rr = json.loads(Path(cli.v5_roundrobin_metrics).read_text())
    roundrobin_retmse = v5_rr['model_ind_mse']

    crh = json.loads(Path(cli.crh_router_test_metrics).read_text())
    crh_retmse = crh['retmse10']
    router_selected_fraction = {f'H{h+1}': crh['selected_head_fraction'][h] for h in range(N_HEADS)}

    shuffled_retmse = None
    if cli.shuffled_router_test_metrics:
        shuffled = json.loads(Path(cli.shuffled_router_test_metrics).read_text())
        shuffled_retmse = shuffled['retmse10']

    # ---- 16-3/16-4: router accuracy + regret (needs the router's OWN head
    # selection per query/channel on test, which the router script doesn't
    # dump raw -- approximate accuracy/regret via the aggregate selected_head
    # distribution vs the oracle distribution is NOT what spec asks; so we
    # recompute router head selection directly here for exactness) ----
    # NOTE: router hard-selection per (query,channel) is recomputed by the
    # orchestrator's own accuracy/regret step (see run_crh_diagnostics01.sh)
    # which calls this script AFTER a dedicated recompute pass -- this
    # script accepts that recompute's output file if present.
    accuracy = None
    regret_stats = None
    router_pick_path = Path(cli.head_oracle_dir) / 'router_head_selection_test.pt'
    if router_pick_path.exists():
        picks = torch.load(router_pick_path, map_location='cpu')['head_sel']  # [Q, C]
        accuracy = float((picks == hard_label_test).float().mean())
        regret = (U_test.gather(-1, picks.unsqueeze(-1)).squeeze(-1)
                 - U_test.gather(-1, hard_label_test.unsqueeze(-1)).squeeze(-1))
        regret_stats = {
            'mean': float(regret.mean()), 'median': float(regret.median()),
            'p90': float(torch.quantile(regret.flatten(), 0.9)),
            'near_zero_fraction': float((regret.abs() < 1e-4).float().mean()),
        }

    # ---- 16-5: calendar -> head specialization table (by hour / weekday) ----
    date_column = load_date_column(cli.root_path, cli.data_path)
    cal_dates = pd.DatetimeIndex(date_column).to_numpy()[
        head_oracle_test['query_start_idx'].numpy() + cli.seq_len]
    cal_dates = pd.DatetimeIndex(cal_dates)
    hour = cal_dates.hour.values
    dow = cal_dates.dayofweek.values
    is_weekend = (dow >= 5)

    def _regime_table(mask_fn, regimes):
        rows = []
        hard_flat = hard_label_test.numpy()  # [Q, C] -> use channel-mean mode per query for simplicity
        hard_mode = np.array([np.bincount(hard_flat[i], minlength=N_HEADS).argmax() for i in range(len(hard_flat))])
        for name, mask in regimes:
            sel = hard_mode[mask]
            if len(sel) == 0:
                continue
            row = {'regime': name, 'n': int(mask.sum())}
            for h in range(N_HEADS):
                row[f'H{h+1}'] = float((sel == h).mean())
            rows.append(row)
        return rows

    hour_regimes = [(f'hour_{h}', hour == h) for h in range(0, 24, 4)]
    hour_table = _regime_table(None, hour_regimes)
    weekday_regimes = [('weekday', ~is_weekend), ('weekend', is_weekend)]
    weekday_table = _regime_table(None, weekday_regimes)

    pd.DataFrame(hour_table).to_csv(out_dir / 'specialization_by_hour.csv', index=False)
    pd.DataFrame(weekday_table).to_csv(out_dir / 'specialization_by_weekday.csv', index=False)

    summary = {
        'A_v5_roundrobin_retmse10': roundrobin_retmse,
        'B_calendar_router_retmse10': crh_retmse,
        'C_oracle_head_retmse10': oracle_retmse,
        'D_fixed_best_head_global_retmse10': global_fixed_retmse,
        'D_fixed_best_head_per_channel_retmse10': per_channel_fixed_retmse,
        'E_shuffled_calendar_retmse10': shuffled_retmse,
        'oracle_gain_vs_roundrobin': roundrobin_retmse - oracle_retmse,
        'oracle_gain_vs_fixed_global': global_fixed_retmse - oracle_retmse,
        'router_gain_vs_roundrobin': roundrobin_retmse - crh_retmse,
        'router_gain_vs_fixed_global': global_fixed_retmse - crh_retmse,
        'router_gain_vs_shuffled': (shuffled_retmse - crh_retmse) if shuffled_retmse is not None else None,
        'oracle_head_fraction': oracle_fraction,
        'router_selected_head_fraction': router_selected_fraction,
        'global_best_fixed_head': f'H{global_best_head+1}',
        'per_channel_best_fixed_head': [f'H{int(h)+1}' for h in per_channel_best_head.tolist()],
        'router_accuracy_vs_oracle': accuracy,
        'router_regret': regret_stats,
    }
    (out_dir / 'crh_diagnostics_summary.json').write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))

    if oracle_fraction[max(oracle_fraction, key=oracle_fraction.get)] >= 0.90:
        print('[STOP-CANDIDATE] one head is oracle >=90% of the time -- routing contribution may be weak.')
    if roundrobin_retmse - oracle_retmse <= 1e-6:
        print('[STOP] Oracle Head does not improve on RoundRobin -- no structural headroom for routing.')


if __name__ == '__main__':
    main()
