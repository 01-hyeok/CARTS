"""TRACK-V-MEANMIX-INFERENCE01 -- integration test (spec section 10,
test 9): loads the REAL ETTh1_96 V0 and V1 checkpoints and verifies
old(round-robin)/new(mean-mixture) Top-10 agreement is exactly 100%,
as required before any 32-cell run. Skips cleanly if the checkpoints
are unavailable."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.build_t2_true_original_kl_cache01 import compute_scores_true_original_kl
from scripts.train_j_shared_encoder_drift01 import build_model
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_margutil01 import build_experiment
from scripts.train_t_pure_multislot01 import compute_scores_full_grad, round_robin_topk_selection
from utils.mean_mixture_selection import mean_mixture_topk_selection

REF = (REPO_ROOT / 'checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/'
      'stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_'
      'df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth')
V0_CKPT = (REPO_ROOT / 'checkpoints/track_v_multiquery_generalization01/ETTh1/H96/seed0/V0/stage1/'
          'ETTh1_96/checkpoint.pth')
V1_CKPT = (REPO_ROOT / 'checkpoints/track_v_multiquery_generalization01/ETTh1/H96/seed0/V1/stage1/'
          'checkpoint.pth')
TOP_K = 10
TAU_S = 0.1


def _skip_if_missing():
    if not (REF.exists() and V0_CKPT.exists() and V1_CKPT.exists()):
        pytest.skip('ETTh1_96 V0/V1 checkpoints not available in this environment')


def _first_batch(exp, device):
    _, loader = exp._get_data(flag='val', shuffle=False)
    batch_x, batch_y, batch_start_idx = next(iter(loader))
    return batch_x.float().to(device), batch_y.float().to(device), batch_start_idx


def test_v0_old_new_top10_agreement_100pct():
    _skip_if_missing()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cli = SimpleNamespace(reference_ckpt=str(REF), pred_len=96, seq_len=96, batch_size=32,
                          init_seed=0, top_k=TOP_K, patch_len=16)
    exp, args, model = build_model(cli, device)
    bl = torch.load(V0_CKPT, map_location=device)
    assert 'slot_heads_state_dict' not in bl
    model.load_state_dict(bl['model_state_dict'])
    model.eval()
    batch_x, batch_y, batch_start_idx = _first_batch(exp, device)
    cand_mask, _ = exp._candidate_mask(batch_start_idx)
    with torch.no_grad():
        scores = compute_scores_true_original_kl(model, batch_x, exp.memory_x, 0)
        rr_idx = round_robin_topk_selection(scores, cand_mask, k=TOP_K)
        mm_idx, _, _ = mean_mixture_topk_selection(scores, cand_mask, TAU_S, k=TOP_K)
    agreement = torch.tensor([
        len(set(rr_idx[b].tolist()) & set(mm_idx[b].tolist())) / TOP_K for b in range(scores.size(0))
    ]).mean()
    assert float(agreement) == 1.0, f'V0 old/new Top10 agreement must be 100%, got {float(agreement)*100:.2f}%'


def test_v1_old_new_top10_agreement_100pct():
    _skip_if_missing()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cli = SimpleNamespace(reference_ckpt=str(REF), pred_len=96, seq_len=96, batch_size=32,
                          init_seed=0, top_k=TOP_K, patch_len=16)
    exp, args, model = build_model(cli, device)
    slot_heads = SlotHeads(int(args.d_model), n_slots=1).to(device)
    bl = torch.load(V1_CKPT, map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    slot_heads.load_state_dict(bl['slot_heads_state_dict'])
    model.eval(); slot_heads.eval()
    batch_x, batch_y, batch_start_idx = _first_batch(exp, device)
    cand_mask, _ = exp._candidate_mask(batch_start_idx)
    with torch.no_grad():
        scores = compute_scores_full_grad(model, slot_heads, batch_x, exp.memory_x, 0)
        rr_idx = round_robin_topk_selection(scores, cand_mask, k=TOP_K)
        mm_idx, _, _ = mean_mixture_topk_selection(scores, cand_mask, TAU_S, k=TOP_K)
    agreement = torch.tensor([
        len(set(rr_idx[b].tolist()) & set(mm_idx[b].tolist())) / TOP_K for b in range(scores.size(0))
    ]).mean()
    assert float(agreement) == 1.0, f'V1 old/new Top10 agreement must be 100%, got {float(agreement)*100:.2f}%'
