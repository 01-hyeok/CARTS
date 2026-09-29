#!/usr/bin/env python3
"""TRACK-H-DIRECT-SET-UTILITY01 -- pre-training N_eff diagnostic (spec
section 9). Computes median full-memory softmax effective-candidate-number
N_eff = 1/sum(w_i^2) at tau in {0.1, 0.05, 0.02} on the FROZEN p120
checkpoint's own cosine score (no training), val split, all channels, for
one dataset. Read-only diagnostic -- writes nothing, trains nothing."""
import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_factorial_e2e01 import arm_score, encode_raw
from scripts.train_margutil01 import build_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--taus', default='0.1,0.05,0.02')
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(cli.checkpoint, map_location='cpu')
    torch.manual_seed(0)
    exp, args = build_experiment(cli.checkpoint, {
        'pred_len': 720, 'seq_len': 720, 'batch_size': 32, 'seed': 0,
        'patch_len': 120, 'stride': 120,
        'relation_encoder_type': 'transformer', 'relation_self_fill': 'zero',
    })
    exp._ensure_memory()
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    channels = list(range(int(args.enc_in)))
    n_candidates = int(exp.memory_x.size(0))

    taus = [float(t) for t in cli.taus.split(',')]
    _, val_loader = exp._get_data(flag='val', shuffle=False)

    with torch.no_grad():
        cand_reps = {c: encode_raw(model, exp.memory_x, c) for c in channels}
        neffs = {t: [] for t in taus}
        for bi, (batch_x, batch_y, batch_start_idx) in enumerate(val_loader):
            batch_x = batch_x.float().to(device)
            cand_mask, _ = exp._candidate_mask(batch_start_idx)
            for c in channels:
                z_q = encode_raw(model, batch_x, c)
                s = arm_score(z_q, cand_reps[c], None).masked_fill(~cand_mask, float('-inf'))
                for t in taus:
                    w = torch.softmax(s / t, dim=-1)
                    neff = 1.0 / (w ** 2).sum(-1).clamp_min(1e-12)
                    neffs[t].append(neff.cpu())
            if bi >= 20:  # enough queries for a stable median, keep it cheap
                break

    print(f'[diag_h_neff] cell={cli.cell} n_candidates={n_candidates}')
    for t in taus:
        vals = torch.cat(neffs[t])
        print(f'  tau={t}: median N_eff={vals.median().item():.1f} mean={vals.mean().item():.1f} '
             f'p90={vals.quantile(0.9).item():.1f} (n_queries_seen={vals.numel()})')


if __name__ == '__main__':
    main()
