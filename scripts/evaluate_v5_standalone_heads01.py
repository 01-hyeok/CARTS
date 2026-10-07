#!/usr/bin/env python3
"""TRACK-EXPERT-V5-FULL01 -- applies the EXACT SAME per-head diagnostic
suite (`utils.expert_head_metrics.evaluate_and_save_head_report`) to
the EXISTING canonical Current-V5 checkpoint (`train_t_pure_multislot01.py
--num_slots 5`, Full-candidate, reused UNMODIFIED). Never retrains
anything; loads the checkpoint, runs the shared evaluator, writes its
output into a separate `current_v5_standalone/` subdirectory so the
Expert-V5 comparison (spec section 11) is between two reports produced
by byte-identical measurement code.
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_j_shared_encoder_drift01 import build_model
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from utils.expert_head_metrics import evaluate_and_save_head_report

NUM_SLOTS = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--patch_len', type=int, default=16)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--tau_t', type=float, default=0.1)
    ap.add_argument('--tau_s', type=float, default=0.1)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--init_seed', type=int, default=0)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--current_v5_checkpoint', required=True,
                    help='canonical Full-candidate V5 checkpoint, e.g. '
                         'checkpoints/track_v_multiquery_generalization01/<Dataset>/H<H>/seed0/V5/stage1/checkpoint.pth')
    ap.add_argument('--out_dir', required=True)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp, args, model = build_model(cli, device)
    channels = list(range(int(args.enc_in)))
    d_model = int(args.d_model)
    slot_heads = SlotHeads(d_model, n_slots=NUM_SLOTS).to(device)

    bl = torch.load(cli.current_v5_checkpoint, map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    slot_heads.load_state_dict(bl['slot_heads_state_dict'])
    model.eval()
    best_epoch = bl.get('epoch', bl.get('val_retmse10', None))

    _, train_loader = exp._get_data(flag='train', shuffle=False)
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)

    out_dir = Path(cli.out_dir)
    final_test_metrics = evaluate_and_save_head_report(
        model, slot_heads, exp, args, cli, channels, device, train_loader, val_loader, test_loader,
        out_dir, best_epoch, num_slots=NUM_SLOTS, tau_e=1.0)
    print(f'[eval_v5_standalone] cell={cli.cell} ckpt={cli.current_v5_checkpoint} '
         f'test_retMSE@10(RR)={final_test_metrics["retmse10"]:.6f} wrote {out_dir}')


if __name__ == '__main__':
    main()
