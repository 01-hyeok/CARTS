#!/usr/bin/env python3
"""One-off offline check (spec S4): does alpha=0.10 produce a reasonable
positive-set size on the REAL Set Oracle utility distribution, before
committing to it for TRACK-A-MULTIPOS-CHOICE01? Read-only, no training,
reuses TRACK-A-SET-DIFFICULTY01's own checkpoint and utility computation.

Kept in the repository (not a scratch file) because
`scripts/train_multipos_choice01.py`'s module docstring documents the
alpha-selection decision by name-referencing this script -- it is the
reproducible provenance for why ALPHA=0.50 was chosen instead of the
spec's suggested default of 0.10. Pass ALPHA_OVERRIDE=<value> to sweep.

Result of the actual sweep run before training (20 test batches, channel 0,
both horizons, TRACK-A-FACTORIAL-E2E01's own `set_onpolicy_cosine`
checkpoints):

    alpha=0.10 -> median positive count = 1 at every step, both horizons
    alpha=0.20 -> median = 1
    alpha=0.30 -> median = 2
    alpha=0.40 -> median = 2, mean 2.3-2.6
    alpha=0.50 -> median = 3, mean ~3.0, p90<=5, max<=9 (both horizons agree)
    alpha=0.60 -> median = 4, mean ~3.6-4.1

ALPHA=0.50 was fixed as the value closest to the centre of the spec's
target band (median 2-5), before any TRACK-A-MULTIPOS-CHOICE01 training.
"""
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diag_set_difficulty01 import HostScorer, load_arm, resolve_checkpoint
from scripts.train_factorial_e2e01 import encode_raw, greedy_set_utility
from scripts.train_margutil01 import memory_value
from utils.dense_utility import candidate_weights
from models.SequentialSetRetriever import SetConditioner

ALPHA = float(__import__("os").environ.get("ALPHA_OVERRIDE", 0.10))


@torch.no_grad()
def check(cell, stage2_host, n_batches=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path, fp = resolve_checkpoint('results/track_a_factorial_e2e', cell, 'set_onpolicy_cosine')
    exp, args, model, sc, ckpt = load_arm(ckpt_path, device)
    exp._ensure_memory()
    host = HostScorer(stage2_host, device)
    _, loader = exp._get_data(flag='test', shuffle=False)
    channels = list(range(int(args.enc_in)))

    counts_by_t = {t: [] for t in range(2, 11)}
    for bi, (batch_x, batch_y, batch_start_idx) in enumerate(loader):
        if bi >= n_batches:
            break
        batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)
        cand_mask, counts = exp._candidate_mask(batch_start_idx)
        valid_query = counts.to(device) >= 10
        for c in channels[:1]:  # channel 0 sufficient for a scale check
            E = encode_raw(model, exp.memory_x, c)
            z_q = encode_raw(model, batch_x, c)
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            futures = memory_c + offset_c.view(-1, 1, 1)
            q_future = batch_y[:, :, c]
            host_scores = host.scores(batch_x, c, cand_mask)
            w_host = candidate_weights(host_scores, cand_mask, host.tau_topk)

            from scripts.train_factorial_e2e01 import arm_score
            selected = torch.zeros_like(cand_mask)
            picks = []
            for t in range(1, 11):
                if t == 1:
                    h_t = z_q
                else:
                    m = E[torch.stack(picks, dim=1)].mean(dim=1)
                    h_t = sc(z_q, m)
                u_hat = arm_score(h_t, E, None)
                valid_now = cand_mask & ~selected
                prefix = torch.stack(picks, dim=1) if picks else torch.zeros(
                    batch_x.size(0), 0, dtype=torch.long, device=device)
                u_target = greedy_set_utility(prefix, w_host, futures, q_future, None)
                neg_inf = torch.finfo(u_target.dtype).min / 4
                masked = u_target.masked_fill(~valid_now, neg_inf)
                model_idx = u_hat.masked_fill(~valid_now, neg_inf).argmax(dim=-1)

                if t >= 2:
                    n_valid = valid_now.sum(dim=-1)
                    k_eff = min(10, int(n_valid.min().clamp_min(1)))
                    top_vals, _ = masked.topk(k_eff, dim=-1)
                    u1 = top_vals[:, 0]
                    u10 = top_vals[:, -1]
                    rng = (u1 - u10).clamp_min(1e-12)
                    threshold = u1 - ALPHA * rng
                    pos = (masked >= threshold.unsqueeze(-1)) & valid_now
                    pc = pos.float().sum(dim=-1)[valid_query]
                    counts_by_t[t].extend(pc.tolist())

                picks.append(model_idx)
                selected = selected.scatter(1, model_idx.unsqueeze(-1), True)

    print(f'=== {cell} alpha={ALPHA} positive-count check (channel 0, {n_batches} batches) ===')
    import statistics as st
    for t in range(2, 11):
        c = counts_by_t[t]
        if not c:
            continue
        c_sorted = sorted(c)
        p90 = c_sorted[int(0.9 * len(c_sorted))]
        print(f'  t={t:>2} median={st.median(c):.1f} mean={st.mean(c):.2f} '
              f'p90={p90:.1f} max={max(c):.1f} n={len(c)}')


if __name__ == '__main__':
    check('ETTh1_96',
          'checkpoints/stage2/ETTh1/seq96_pred96/stage2_carts_softset_s2_ETTh1_96_S0_wce_RelationStage2_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth')
    check('ETTh1_720',
          'checkpoints/stage2/ETTh1/seq720_pred720/stage2_carts_softset_s2_ETTh1_720_S0_wce_RelationStage2_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth')
