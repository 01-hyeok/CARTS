#!/usr/bin/env python3
"""TRACK-P100-COARSE-TO-FINE-RETRIEVAL01 -- one-time TRAIN-only tau_c
calibration for the coarse delta-last-cosine score's softmax
temperature. No canonical tau_c exists anywhere in the repo (grep
confirmed: `precompute_candidate_pool01.py` only ever uses the coarse
score for a raw, temperature-free Top-M argsort when BUILDING the
pool -- it never turns it into a softmax distribution).

IMPORTANT correction recorded here, not hidden: the FIRST calibration
attempt in this file picked tau_c by minimizing coarse-only AggMSE@10
on TRAIN. That criterion turned out to be degenerate and was
discarded before being used for anything -- argmax/top-k of
softmax(c_i/tau) is RANK-INVARIANT to tau for any tau>0 (softmax is a
strictly monotonic transform of a single score vector), so the
coarse-only Top-10 (and therefore its AggMSE) is identical for every
tau_c tested, confirmed by direct re-run (AggMSE constant to 6+
decimal places across tau_c in [0.02, 2.0] on all 4 cells). tau_c only
starts to matter once fused with V5's score in LOG space (Phase A2),
where its role is to set the RELATIVE SCALE of log p_c against log
p_v5 -- so the quantity actually worth calibrating is DISTRIBUTION
SHAPE (entropy), not AggMSE.

Corrected criterion: pick tau_c so that the coarse distribution
p_c=softmax(c_i/tau_c) has, on TRAIN, the SAME average entropy (over
the 100 P100 candidates) as the canonical V5 Mean-Mixture distribution
p_v5=mean_h softmax(scores_h/tau_s) already has at its own
production tau_s=0.1 -- so that a log-domain linear fusion at
intermediate beta is a genuine blend of two comparably-peaked
distributions, rather than one trivially dominating the other's log
-magnitude regardless of beta. One tau_c, picked once, shared by
every cell and every later Phase (never re-swept).
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_factorial_e2e01 import encode_raw
from scripts.train_j_shared_encoder_drift01 import build_model
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_retriever_pool01 import CandidatePoolCache
from utils.candidate_pool import compute_coarse_delta_last_cosine_scores

TAU_S = 0.1
N_SLOTS = 5
CANDIDATE_TAUS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0)

CELLS = [
    dict(cell='ETTh1_96', seq_len=96, pred_len=96,
         reference_ckpt='checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth',
         v5_checkpoint='checkpoints/track_v_sharedtop100_signfix01/ETTh1/H96/seed0/pool_top100/ETTh1_96/V5/checkpoint_epoch10.pth',
         pool='results/TRACK-V-MULTIQUERY-GENERALIZATION01/ETTh1/H96/seed0/pool_top100/shared_candidate_pool'),
    dict(cell='ETTh1_720', seq_len=720, pred_len=720,
         reference_ckpt='checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/stage1_carts_softset_ETTh1_720_S0_wce_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth',
         v5_checkpoint='checkpoints/track_v_sharedtop100_signfix01/ETTh1/H720/seed0/pool_top100/ETTh1_720/V5/checkpoint_epoch9.pth',
         pool='results/TRACK-V-MULTIQUERY-GENERALIZATION01/ETTh1/H720/seed0/pool_top100/shared_candidate_pool'),
    dict(cell='Weather_96', seq_len=96, pred_len=96,
         reference_ckpt='checkpoints/soft_set_mse/stage1/custom/seq96_pred96/stage1_carts_softset_Weather_96_S0_wce_RelationStage1_custom_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl96_pl96_0/checkpoint.pth',
         v5_checkpoint='checkpoints/track_v_sharedtop100_signfix01/Weather/H96/seed0/pool_top100/Weather_96/V5/checkpoint_epoch4.pth',
         pool='results/TRACK-V-MULTIQUERY-GENERALIZATION01/Weather/H96/seed0/pool_top100/shared_candidate_pool'),
    dict(cell='Weather_720', seq_len=720, pred_len=720,
         reference_ckpt='checkpoints/soft_set_mse/stage1/custom/seq720_pred720/stage1_carts_softset_Weather_720_S0_wce_RelationStage1_custom_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl720_pl720_0/checkpoint.pth',
         v5_checkpoint='checkpoints/track_v_sharedtop100_signfix01/Weather/H720/seed0/pool_top100/Weather_720/V5/checkpoint_epoch6.pth',
         pool='results/TRACK-V-MULTIQUERY-GENERALIZATION01/Weather/H720/seed0/pool_top100/shared_candidate_pool'),
]


class _Cli:
    pass


def _entropy(p, eps=1e-12):
    """p: [...,100], rows sum to 1. Returns [...] entropy in nats."""
    return -(p * (p + eps).log()).sum(dim=-1)


@torch.no_grad()
def cell_entropies(cell_cfg, device):
    cli = _Cli()
    cli.reference_ckpt = cell_cfg['reference_ckpt']
    cli.pred_len = cell_cfg['pred_len']
    cli.seq_len = cell_cfg['seq_len']
    cli.batch_size = 32
    cli.top_k = 10
    cli.init_seed = 0
    cli.patch_len = 16

    exp, args, model = build_model(cli, device)
    exp.memory_x = exp.memory_x.to(device)
    model.eval()
    slot_heads = SlotHeads(int(args.d_model), n_slots=N_SLOTS).to(device)
    bl = torch.load(cell_cfg['v5_checkpoint'], map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    slot_heads.load_state_dict(bl['slot_heads_state_dict'])
    slot_heads.eval()

    channels = list(range(int(args.enc_in)))
    expected_meta = {'seq_len': cli.seq_len, 'pred_len': cli.pred_len, 'candidate_mask': 'raft',
                     'candidate_pool_size': 100, 'candidate_pool_metric': 'delta_last_cosine'}
    pool_cache = CandidatePoolCache(Path(cell_cfg['pool']) / 'train.pt', expected_meta)

    _, loader = exp._get_data(flag='train', shuffle=False)

    v5_entropies = []
    coarse_entropies = {tau: [] for tau in CANDIDATE_TAUS}

    for c in channels:
        k_full = F.normalize(encode_raw(model, exp.memory_x, c), dim=-1)  # [N,D]
        for batch_x, batch_y, batch_start_idx in loader:
            batch_x = batch_x.float().to(device)
            pool_idx_global = pool_cache.lookup(batch_start_idx, c, device)  # [B,100]

            coarse_full = compute_coarse_delta_last_cosine_scores(batch_x, exp.memory_x, c)  # [B,N]
            c_i = coarse_full.gather(1, pool_idx_global)  # [B,100]
            for tau in CANDIDATE_TAUS:
                p_c = F.softmax(c_i / tau, dim=-1)
                coarse_entropies[tau].append(_entropy(p_c).cpu())

            z_q = encode_raw(model, batch_x, c)
            q = slot_heads(z_q)  # [B,5,D]
            k_pool = k_full[pool_idx_global]  # [B,100,D]
            scores = torch.einsum('bsd,bmd->bsm', q, k_pool)  # [B,5,100]
            p_heads = F.softmax(scores / TAU_S, dim=-1)
            p_mix = p_heads.mean(dim=1)  # [B,100]
            v5_entropies.append(_entropy(p_mix).cpu())

    v5_entropy_mean = float(torch.cat(v5_entropies).mean())
    coarse_entropy_means = {tau: float(torch.cat(v).mean()) for tau, v in coarse_entropies.items()}
    return v5_entropy_mean, coarse_entropy_means


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    v5_target_per_cell = {}
    coarse_per_cell = {}
    for cfg in CELLS:
        print(f'[calibrate_tau_c] running TRAIN-only entropy calibration for {cfg["cell"]} ...')
        v5_ent, coarse_ents = cell_entropies(cfg, device)
        v5_target_per_cell[cfg['cell']] = v5_ent
        coarse_per_cell[cfg['cell']] = coarse_ents
        print(f'[calibrate_tau_c] {cfg["cell"]}: V5 Mean-Mixture entropy (tau_s=0.1) = {v5_ent:.4f}; '
             f'coarse entropy by tau_c = { {k: round(v,4) for k,v in coarse_ents.items()} }')

    v5_target_avg = sum(v5_target_per_cell.values()) / len(v5_target_per_cell)
    avg_coarse_by_tau = {tau: sum(coarse_per_cell[c][tau] for c in coarse_per_cell) / len(coarse_per_cell)
                        for tau in CANDIDATE_TAUS}
    best_tau = min(avg_coarse_by_tau, key=lambda t: abs(avg_coarse_by_tau[t] - v5_target_avg))

    print(f'[calibrate_tau_c] cross-cell V5 target entropy (tau_s=0.1) = {v5_target_avg:.4f}')
    print(f'[calibrate_tau_c] cross-cell coarse entropy by tau_c: { {k: round(v,4) for k,v in avg_coarse_by_tau.items()} }')
    print(f'[calibrate_tau_c] SELECTED tau_c = {best_tau} '
         f'(closest cross-cell TRAIN-only coarse entropy to V5\'s own entropy)')

    out = {'criterion': 'match average P100 softmax entropy (nats) on TRAIN to canonical V5 Mean-Mixture '
                       '(tau_s=0.1) entropy, averaged across all 4 cells',
          'v5_entropy_per_cell': v5_target_per_cell, 'v5_entropy_cross_cell_avg': v5_target_avg,
          'coarse_entropy_per_cell_per_tau': coarse_per_cell,
          'coarse_entropy_cross_cell_avg_by_tau': avg_coarse_by_tau,
          'selected_tau_c': best_tau,
          'note_discarded_criterion': 'AggMSE@10 of coarse-only Top-10 is rank-invariant to tau_c for any '
                                      'tau_c>0 (softmax of one score vector is a monotonic transform); '
                                      'confirmed constant across tau_c in [0.02,2.0] on all 4 cells before '
                                      'this entropy-based criterion was adopted instead.'}
    out_dir = Path('results/TRACK-P100-COARSE-TO-FINE-RETRIEVAL01')
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'tau_c_calibration.json').write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
