#!/usr/bin/env python3
"""Dumps the Router's raw per-(query,channel) head selection on one
split (test), for `compute_crh_diagnostics01.py`'s accuracy/regret
computation. Forward-only, no training."""
import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.CalendarRouter import CalendarRouter
from scripts.train_j_shared_encoder_drift01 import build_model
from utils.calendar_features import forecast_start_calendar_features, load_date_column

N_HEADS = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--router_checkpoint', required=True)
    ap.add_argument('--head_oracle_test', required=True)
    ap.add_argument('--root_path', required=True)
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--out_path', required=True)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cli.init_seed = 0
    cli.patch_len = 16
    cli.top_k = 10
    cli.batch_size = 32
    exp, args, model = build_model(cli, device)
    n_channels = int(args.enc_in)

    router = CalendarRouter(n_channels=n_channels, n_heads=N_HEADS).to(device)
    bl = torch.load(cli.router_checkpoint, map_location=device)
    router.load_state_dict(bl['router_state_dict'])
    router.eval()

    head_oracle_test = torch.load(cli.head_oracle_test, map_location='cpu')
    starts = head_oracle_test['query_start_idx']
    date_column = load_date_column(cli.root_path, cli.data_path)
    cal = forecast_start_calendar_features(date_column, starts, cli.seq_len).to(device)

    with torch.no_grad():
        head_sel = torch.zeros(len(starts), n_channels, dtype=torch.long)
        for c in range(n_channels):
            p_r = router(cal, c)
            head_sel[:, c] = p_r.argmax(dim=-1).cpu()

    Path(cli.out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({'query_start_idx': starts, 'head_sel': head_sel}, cli.out_path)
    print(f'[router_head_selection] wrote {cli.out_path} shape={tuple(head_sel.shape)}')


if __name__ == '__main__':
    main()
