#!/usr/bin/env python3
"""TRACK-HARD-EXPERT-V5-P100-ALLH01 -- loads a selected P100 checkpoint
(model + 5-slot SlotHeads, from ANY of the V5/Soft/Hard arms -- the
per-head diagnostic report does not care how the checkpoint was
trained) and runs `evaluate_and_save_head_report_pool`
(`utils/expert_head_metrics_pool01.py`) against it, writing the same
per-head/specialization/oracle-vs-fixed files
`evaluate_and_save_head_report` writes for Full-memory arms, restricted
to the Shared-Top-100 pool throughout."""
import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.rng_control01 import make_loader_generator
from scripts.train_j_shared_encoder_drift01 import build_model
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_retriever_pool01 import CandidatePoolCache
from utils.expert_head_metrics_pool01 import evaluate_and_save_head_report_pool

NUM_SLOTS = 5
TOP_K = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--retriever_checkpoint', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--patch_len', type=int, default=16)
    ap.add_argument('--tau_t', type=float, default=0.1)
    ap.add_argument('--tau_s', type=float, default=0.1)
    ap.add_argument('--candidate_pool_size', type=int, default=100)
    ap.add_argument('--candidate_mask', default='raft')
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--loader_seed', type=int, default=0)
    ap.add_argument('--candidate_pool_cache', required=True)
    ap.add_argument('--best_epoch', type=int, required=True)
    ap.add_argument('--out_dir', required=True)
    cli = ap.parse_args()
    cli.top_k = TOP_K
    cli.init_seed = cli.seed

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp, args, model = build_model(cli, device)
    channels = list(range(int(args.enc_in)))
    d_model = int(args.d_model)

    bl = torch.load(cli.retriever_checkpoint, map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    slot_heads = SlotHeads(d_model, n_slots=NUM_SLOTS).to(device)
    slot_heads.load_state_dict(bl['slot_heads_state_dict'])
    model.eval()
    slot_heads.eval()

    expected_meta = {'seq_len': cli.seq_len, 'pred_len': cli.pred_len, 'candidate_mask': cli.candidate_mask,
                     'candidate_pool_size': cli.candidate_pool_size, 'candidate_pool_metric': 'delta_last_cosine'}
    cache_dir = Path(cli.candidate_pool_cache)
    pool_caches = {s: CandidatePoolCache(cache_dir / f'{s}.pt', expected_meta) for s in ('train', 'val', 'test')}

    train_gen = make_loader_generator(cli.loader_seed)
    _, train_loader = exp._get_data(flag='train', shuffle=True, generator=train_gen)
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)

    final_test_metrics = evaluate_and_save_head_report_pool(
        model, slot_heads, exp, args, cli, channels, device, train_loader, val_loader, test_loader,
        pool_caches, cli.out_dir, cli.best_epoch, num_slots=NUM_SLOTS, tau_e=1.0)
    print(f'[specialization_report_pool] wrote {cli.out_dir}: {final_test_metrics}')


if __name__ == '__main__':
    main()
