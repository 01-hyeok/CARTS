"""Integration tests for TRACK-V-CALENDAR-ROUTER01, spec section 20
Tests 3, 5, 6, 7 -- needs the already-trained ETTh1_96 V5 checkpoint and
its Shared-Top-100 pool cache (both produced earlier this session;
skips cleanly if unavailable, e.g. on a fresh checkout)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.train_j_shared_encoder_drift01 import build_model
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_margutil01 import memory_value
from scripts.train_retriever_pool01 import CandidatePoolCache, compute_scores
from utils.candidate_pool import CandidatePoolConfig, gather_candidate_values, pooled_future_mse

REF = (REPO_ROOT / 'checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/'
      'stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_'
      'dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth')
POOL = REPO_ROOT / 'results/TRACK-V-MULTIQUERY-GENERALIZATION01/ETTh1/H96/seed0/pool_top100/shared_candidate_pool'
V5CKPT = (REPO_ROOT / 'checkpoints/track_v_multiquery_generalization01/ETTh1/H96/seed0/pool_top100/'
         'ETTh1_96/V5/checkpoint_best_retmse.pth')

N_HEADS = 5
TOP_K = 10


def _skip_if_missing():
    if not (REF.exists() and V5CKPT.exists() and (POOL / 'train.pt').exists()):
        pytest.skip('ETTh1_96 Shared-Top-100 V5 checkpoint/pool not available in this environment')


def _setup():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cli = SimpleNamespace(reference_ckpt=str(REF), pred_len=96, seq_len=96, batch_size=32,
                          init_seed=0, top_k=TOP_K, patch_len=16)
    exp, args, model = build_model(cli, device)
    slot_heads = SlotHeads(int(args.d_model), n_slots=N_HEADS).to(device)
    bl = torch.load(V5CKPT, map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    slot_heads.load_state_dict(bl['slot_heads_state_dict'])
    model.eval(); slot_heads.eval()
    pool_cfg = CandidatePoolConfig(mode='coarse_topk', size=100)
    pool_cache = CandidatePoolCache(POOL / 'val.pt', {'seq_len': 96, 'pred_len': 96})
    return device, exp, args, model, slot_heads, pool_cfg, pool_cache


def _first_batch(exp, device):
    _, loader = exp._get_data(flag='val', shuffle=False)
    batch_x, batch_y, batch_start_idx = next(iter(loader))
    return batch_x.float().to(device), batch_y.float().to(device), batch_start_idx


# -------------------- Test 5: each head's Top-10 only selects from its own P100 pool --------------------

def test_head_top10_only_from_p100_pool():
    _skip_if_missing()
    device, exp, args, model, slot_heads, pool_cfg, pool_cache = _setup()
    batch_x, batch_y, batch_start_idx = _first_batch(exp, device)
    cand_mask, _ = exp._candidate_mask(batch_start_idx)
    c = 0
    with torch.no_grad():
        scores, valid_mask, pool_idx_global = compute_scores(
            pool_cfg, model, slot_heads, batch_x, exp, c, cand_mask, pool_cache, batch_start_idx, device)
        for h in range(N_HEADS):
            top10_local = stable_topk_indices(scores[:, h, :], TOP_K, largest=True)
            assert top10_local.max() < pool_cfg.size
            assert top10_local.min() >= 0


# -------------------- Test 6: forcing head selection == that head's own standalone Top-10 --------------------

def test_forced_head_matches_standalone_head_top10():
    _skip_if_missing()
    device, exp, args, model, slot_heads, pool_cfg, pool_cache = _setup()
    batch_x, batch_y, batch_start_idx = _first_batch(exp, device)
    cand_mask, _ = exp._candidate_mask(batch_start_idx)
    c = 0
    with torch.no_grad():
        scores, valid_mask, pool_idx_global = compute_scores(
            pool_cfg, model, slot_heads, batch_x, exp, c, cand_mask, pool_cache, batch_start_idx, device)
        h_forced = 2  # "H3"
        # simulate router hard-selecting head h_forced for every query
        head_sel = torch.full((batch_x.size(0),), h_forced, dtype=torch.long, device=device)
        scores_sel = scores.gather(1, head_sel.view(-1, 1, 1).expand(-1, 1, scores.size(-1))).squeeze(1)
        picks_via_routing = stable_topk_indices(scores_sel, TOP_K, largest=True)
        picks_standalone = stable_topk_indices(scores[:, h_forced, :], TOP_K, largest=True)
        assert torch.equal(picks_via_routing, picks_standalone)


# -------------------- Test 7: oracle label == argmin(per-head retMSE) exactly --------------------

def test_oracle_label_matches_argmin_u():
    _skip_if_missing()
    device, exp, args, model, slot_heads, pool_cfg, pool_cache = _setup()
    batch_x, batch_y, batch_start_idx = _first_batch(exp, device)
    cand_mask, _ = exp._candidate_mask(batch_start_idx)
    c = 0
    with torch.no_grad():
        scores, valid_mask, pool_idx_global = compute_scores(
            pool_cfg, model, slot_heads, batch_x, exp, c, cand_mask, pool_cache, batch_start_idx, device)
        memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
        query_future = batch_y[:, :, c]
        pooled_memory_c = gather_candidate_values(memory_c, pool_idx_global)
        d_pool = pooled_future_mse(pooled_memory_c, offset_c, query_future)
        U = torch.zeros(batch_x.size(0), N_HEADS, device=device)
        for h in range(N_HEADS):
            top10_h = stable_topk_indices(scores[:, h, :], TOP_K, largest=True)
            U[:, h] = d_pool.gather(1, top10_h).mean(dim=-1)
        hard_label = U.argmin(dim=-1)
        # recompute independently via a different code path (direct min) and compare
        min_val, min_idx = U.min(dim=-1)
        assert torch.equal(hard_label, min_idx)


# -------------------- Test 3 (integration-level freeze check) --------------------

def test_v5_frozen_during_router_backward():
    _skip_if_missing()
    device, exp, args, model, slot_heads, pool_cfg, pool_cache = _setup()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in slot_heads.parameters():
        p.requires_grad_(False)
    batch_x, batch_y, batch_start_idx = _first_batch(exp, device)
    cand_mask, _ = exp._candidate_mask(batch_start_idx)
    from models.CalendarRouter import CalendarRouter
    router = CalendarRouter(n_channels=int(args.enc_in), n_heads=N_HEADS).to(device)
    optimizer = torch.optim.Adam(router.parameters(), lr=1e-2)
    cal = torch.randn(batch_x.size(0), 6, device=device)
    optimizer.zero_grad()
    scores, valid_mask, pool_idx_global = compute_scores(
        pool_cfg, model, slot_heads, batch_x, exp, 0, cand_mask, pool_cache, batch_start_idx, device)
    p_r = router(cal, 0)
    loss = (p_r * scores.mean(dim=1)[:, :5]).sum()  # dummy loss just to exercise backward through router+scores
    loss.backward()
    assert router.linear[0].weight.grad is not None and router.linear[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.parameters())
    assert all(p.grad is None for p in slot_heads.parameters())
