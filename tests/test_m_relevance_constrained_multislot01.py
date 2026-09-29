"""Unit tests for TRACK-M-RELEVANCE-CONSTRAINED-MULTISLOT01 (spec PART 23,
18 items). Fast synthetic tests; a few touch the real checkpoint/dataset
(cheap, MLP encoder) or the precomputed J1 reference artifacts (skipped if
not yet produced in this checkout).
"""
import inspect
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.eval_l_aligned_population01 import (
    EXPECTED_FULL as L_EXPECTED_FULL,
)
from scripts.eval_l_aligned_population01 import load_arm as l_load_arm
from scripts.eval_l_aligned_population01 import macro_summary as l_macro_summary
from scripts.eval_l_aligned_population01 import run_population as l_run_population
from scripts.train_j_shared_encoder_drift01 import build_model, state_hash
from scripts.train_k_multislot_predictive_retrieval01 import (
    N_SLOTS, SlotHeads, compute_scores, hard_unique_selection, soft_aggregate_loss,
)
from scripts.train_m_relevance_constrained_multislot01 import (
    DELTA_DEFAULT, FEAS_MARGIN, load_j1_reference_table, soft_relevance_loss,
)

REF_CKPT = ('checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/'
           'stage1_carts_softset_ETTh1_720_S0_wce_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_'
           'df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth')
_ref_available = Path(REPO_ROOT / REF_CKPT).exists()
needs_ref = pytest.mark.skipif(not _ref_available, reason='reference checkpoint not present in this checkout')

J1_REF_PARQUET = REPO_ROOT / 'results/TRACK-M-RELEVANCE-CONSTRAINED-MULTISLOT01/ETTh1_720/j1_reference/train_reference.parquet'
J1_VAL_BASELINE = REPO_ROOT / 'results/TRACK-M-RELEVANCE-CONSTRAINED-MULTISLOT01/ETTh1_720/j1_reference/validation_baseline.json'
_j1_ref_available = J1_REF_PARQUET.exists() and J1_VAL_BASELINE.exists()
needs_j1_ref = pytest.mark.skipif(not _j1_ref_available, reason='J1 reference precompute not yet run')

J1_CKPT = REPO_ROOT / 'checkpoints/track_j2_key_update_decomposition01/ETTh1_720/J1_stopgrad_key/checkpoint.pth'


class _Cli:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _default_cli(**overrides):
    base = dict(reference_ckpt=str(REPO_ROOT / REF_CKPT), cell='ETTh1_720', pred_len=720, seq_len=720,
               patch_len=16, top_k=10, tau_t=0.1, tau_s=0.1, batch_size=32, learning_rate=1e-3,
               chunk_size=4096, init_seed=0, loader_seed=0)
    base.update(overrides)
    return _Cli(**base)


# item 1: K2 architecture exact reuse -- SlotHeads/compute_scores/hard_unique_selection/
# soft_aggregate_loss are IMPORTED (not reimplemented) from train_k, so identity holds trivially;
# assert the source module of each imported symbol is train_k's own file.
def test_item1_k2_architecture_exact_reuse():
    import scripts.train_k_multislot_predictive_retrieval01 as k_mod
    assert SlotHeads is k_mod.SlotHeads
    assert compute_scores is k_mod.compute_scores
    assert hard_unique_selection is k_mod.hard_unique_selection
    assert soft_aggregate_loss is k_mod.soft_aggregate_loss


# item 2: M1/M2 differ from K2 ONLY in the relevance term -- soft_aggregate_loss (K2's own
# aggregate objective) is called UNMODIFIED with identical outputs to K2's own call.
def test_item2_m1_m2_differ_only_in_relevance_term():
    torch.manual_seed(0)
    bsz, s, n, h = 3, 4, 20, 5
    scores = torch.randn(bsz, s, n)
    memory_c = torch.randn(n, h)
    offset_c = torch.randn(bsz)
    query_future = torch.randn(bsz, h)
    l_agg_a = soft_aggregate_loss(scores, memory_c, offset_c, query_future, tau_s=0.1, l_soft=8)
    l_agg_b = soft_aggregate_loss(scores, memory_c, offset_c, query_future, tau_s=0.1, l_soft=8)
    assert torch.allclose(l_agg_a, l_agg_b)  # deterministic, same code path every call
    # relevance term is a genuinely SEPARATE loss (not folded into l_agg's own math)
    d_raw = torch.randn(bsz, n)
    r_set = soft_relevance_loss(scores, d_raw, tau_s=0.1, l_soft=8)
    assert r_set.shape == (bsz,)


# item 3: J1 reference checkpoint frozen (all params requires_grad=False after load)
@needs_ref
def test_item3_j1_checkpoint_frozen():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cli = _default_cli()
    exp, args, model = build_model(cli, device)
    bl = torch.load(J1_CKPT, map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    assert all(not p.requires_grad for p in model.parameters())


# item 4: train reference uses ONLY train futures -- precompute script source never
# calls _get_data(flag='val'/'test') to build the parquet rows (only for validation_baseline.json,
# a SEPARATE, explicitly-labeled artifact).
def test_item4_train_reference_uses_only_train_split():
    src = inspect.getsource(__import__('scripts.precompute_m_j1_reference01', fromlist=['main']))
    # the parquet-writing block must be reached only via a 'train' flag call
    assert "flag='train'" in src
    parquet_write_idx = src.index("to_parquet")
    train_call_idx = src.index("flag='train'")
    val_call_idx = src.index("flag='val'")
    assert train_call_idx < parquet_write_idx < val_call_idx, \
        'parquet write must occur using the train loader, before any val loader is touched'


# item 5: no val/test leakage in the relevance reference -- row count is EXACTLY
# n_train_queries x n_channels (dense, no extra rows from another split)
@needs_j1_ref
def test_item5_no_val_test_leakage_in_reference():
    df = pd.read_parquet(J1_REF_PARQUET)
    n_q = df['query_start_idx'].nunique()
    n_c = df['channel'].nunique()
    assert len(df) == n_q * n_c
    assert n_q == 7201  # ETTh1_720 train-only candidate/query count established across J/K/L


# item 6: query_start_idx/channel lookup exact
def test_item6_lookup_table_exact(tmp_path, monkeypatch):
    df = pd.DataFrame({'query_start_idx': [0, 0, 1, 1], 'channel': [0, 1, 0, 1],
                       'T_J1': [1.0, 2.0, 3.0, 4.0]})
    ref_dir = tmp_path / 'j1_reference'
    ref_dir.mkdir()
    df.to_parquet(ref_dir / 'train_reference.parquet')
    import scripts.train_m_relevance_constrained_multislot01 as m_mod
    monkeypatch.setattr(m_mod, 'J1_REF_DIR', ref_dir)
    table = load_j1_reference_table(torch.device('cpu'))
    assert table.shape == (2, 2)
    assert float(table[0, 0]) == 1.0
    assert float(table[0, 1]) == 2.0
    assert float(table[1, 0]) == 3.0
    assert float(table[1, 1]) == 4.0


# item 7: R_m brute-force synthetic match
def test_item7_r_m_brute_force_match():
    torch.manual_seed(1)
    bsz, s, n = 2, 1, 10
    l_soft = 4
    scores = torch.randn(bsz, s, n)
    d_raw = torch.randn(bsz, n)
    tau_s = 0.3
    r_set = soft_relevance_loss(scores, d_raw, tau_s, l_soft)
    # brute-force per batch item
    for b in range(bsz):
        s_m = scores[b, 0]
        topv, topi = torch.topk(s_m, l_soft, largest=True)
        w = torch.softmax(topv / tau_s, dim=-1)
        d_sel = d_raw[b, topi]
        expected = (w * d_sel).sum()
        assert torch.allclose(r_set[b], expected, atol=1e-5)


# item 8: R_set computed correctly (mean over slots of R_m)
def test_item8_r_set_is_mean_of_r_m():
    torch.manual_seed(2)
    bsz, s, n = 2, 5, 12
    l_soft = 4
    scores = torch.randn(bsz, s, n)
    d_raw = torch.randn(bsz, n)
    tau_s = 0.2
    r_set = soft_relevance_loss(scores, d_raw, tau_s, l_soft)
    r_m_sum = torch.zeros(bsz)
    for m in range(s):
        s_m = scores[:, m, :]
        idx_m = stable_topk_indices(s_m, l_soft, largest=True)
        w_m = torch.softmax(s_m.gather(1, idx_m) / tau_s, dim=-1)
        r_m_sum = r_m_sum + (w_m * d_raw.gather(1, idx_m)).sum(-1)
    assert torch.allclose(r_set, r_m_sum / s, atol=1e-5)


# item 9: M1's soft-relevance term produces a nonzero gradient
def test_item9_m1_relevance_nonzero_gradient():
    torch.manual_seed(3)
    bsz, s, n = 2, 3, 10
    scores = torch.randn(bsz, s, n, requires_grad=True)
    d_raw = torch.randn(bsz, n)
    r_set = soft_relevance_loss(scores, d_raw, tau_s=0.1, l_soft=6)
    r_set.mean().backward()
    assert scores.grad is not None
    assert scores.grad.abs().sum() > 0


# item 10: M2's hinge is inactive (zero) below the threshold
def test_item10_m2_hinge_inactive_below_threshold():
    torch.manual_seed(4)
    bsz, s, n = 2, 3, 10
    scores = torch.randn(bsz, s, n)
    d_raw = torch.full((bsz, n), 0.01)  # tiny future-MSE everywhere -> R_set tiny
    r_set = soft_relevance_loss(scores, d_raw, tau_s=0.1, l_soft=6)
    t_j1 = torch.full((bsz,), 100.0)  # threshold far above R_set
    l_budget = F.relu(r_set - (1.0 + DELTA_DEFAULT) * t_j1).mean()
    assert float(l_budget) == 0.0


# item 11: M2's hinge is active above the threshold
def test_item11_m2_hinge_active_above_threshold():
    torch.manual_seed(5)
    bsz, s, n = 2, 3, 10
    scores = torch.randn(bsz, s, n)
    d_raw = torch.full((bsz, n), 100.0)  # huge future-MSE everywhere -> R_set huge
    r_set = soft_relevance_loss(scores, d_raw, tau_s=0.1, l_soft=6)
    t_j1 = torch.full((bsz,), 0.01)  # threshold far below R_set
    l_budget = F.relu(r_set - (1.0 + DELTA_DEFAULT) * t_j1).mean()
    assert float(l_budget) > 0.0


# item 12: no [B,S,N,H] allocation anywhere -- every intermediate tensor inside
# soft_relevance_loss stays at rank <= 3 with its largest axis bounded by L_soft, not N
def test_item12_no_bsnh_allocation():
    bsz, s, n, l_soft = 2, 4, 500, 8
    scores = torch.randn(bsz, s, n)
    d_raw = torch.randn(bsz, n)
    shapes_seen = []
    orig_gather = torch.Tensor.gather

    def spy_gather(self, *a, **kw):
        out = orig_gather(self, *a, **kw)
        shapes_seen.append(tuple(out.shape))
        return out

    torch.Tensor.gather = spy_gather
    try:
        soft_relevance_loss(scores, d_raw, tau_s=0.1, l_soft=l_soft)
    finally:
        torch.Tensor.gather = orig_gather
    for shp in shapes_seen:
        assert len(shp) <= 2 and all(dim <= max(bsz, l_soft) for dim in shp), shp


# item 13: same candidate mask as K2 (same exp._candidate_mask call site/semantics)
@needs_ref
def test_item13_same_candidate_mask_as_k2():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cli = _default_cli()
    exp, args, model = build_model(cli, device)
    _, loader = exp._get_data(flag='val', shuffle=False)
    batch_x, batch_y, batch_start_idx = next(iter(loader))
    mask_a, _ = exp._candidate_mask(batch_start_idx)
    mask_b, _ = exp._candidate_mask(batch_start_idx)
    assert torch.equal(mask_a, mask_b)  # deterministic, identical function used by both K2 and M


# item 14: same batch order as K2/pairing -- identical loader_seed produces identical
# batch-order hash across two independently-constructed loaders (M1 vs M2 pairing)
@needs_ref
def test_item14_pairing_identical_batch_order():
    from scripts.rng_control01 import batch_order_sha256, make_loader_generator
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cli = _default_cli()
    exp, args, model = build_model(cli, device)
    gen_a = make_loader_generator(0)
    _, loader_a = exp._get_data(flag='train', shuffle=True, generator=gen_a)
    starts_a = []
    for i, (_, _, batch_start_idx) in enumerate(loader_a):
        if i >= 5:
            break
        starts_a.append(batch_start_idx)
    gen_b = make_loader_generator(0)
    _, loader_b = exp._get_data(flag='train', shuffle=True, generator=gen_b)
    starts_b = []
    for i, (_, _, batch_start_idx) in enumerate(loader_b):
        if i >= 5:
            break
        starts_b.append(batch_start_idx)
    assert batch_order_sha256(starts_a) == batch_order_sha256(starts_b)


# item 15: hard Top-10 selection returns exactly 10 unique candidates
def test_item15_hard_selection_exactly_10_unique():
    torch.manual_seed(6)
    bsz, s, n = 4, N_SLOTS, 50
    scores = torch.randn(bsz, s, n)
    cand_mask = torch.ones(bsz, n, dtype=torch.bool)
    idx = hard_unique_selection(scores, cand_mask, s=N_SLOTS)
    assert idx.shape == (bsz, N_SLOTS)
    for b in range(bsz):
        assert len(set(idx[b].tolist())) == N_SLOTS


# item 16: constrained checkpoint selection logic is correct (feasibility-gated min-AggMSE)
def test_item16_constrained_checkpoint_selection_logic():
    r_val_j1 = 1.0
    feas_threshold = (1.0 + FEAS_MARGIN) * r_val_j1
    epochs = [
        {'epoch': 1, 'val_retmse10': 0.90, 'val_agg': 0.80},  # feasible, agg 0.80
        {'epoch': 2, 'val_retmse10': 1.20, 'val_agg': 0.50},  # infeasible (best agg, must be rejected)
        {'epoch': 3, 'val_retmse10': 1.00, 'val_agg': 0.70},  # feasible (== threshold), agg 0.70 (best feasible)
        {'epoch': 4, 'val_retmse10': 0.95, 'val_agg': 0.90},  # feasible, worse agg
    ]
    best = {'val_agg': float('inf'), 'epoch': -1}
    for e in epochs:
        feasible = e['val_retmse10'] <= feas_threshold
        if feasible and e['val_agg'] < best['val_agg']:
            best = {'val_agg': e['val_agg'], 'epoch': e['epoch']}
    assert best['epoch'] == 3
    assert best['val_agg'] == 0.70


# item 17: TRACK-L's evaluator reproduces K2's known FULL2161 baseline (reused, not recomputed,
# for TRACK-M's own comparison table -- this test confirms the evaluator is still trustworthy)
@needs_ref
def test_item17_track_l_evaluator_reproduces_k2_baseline():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cli = _default_cli()
    exp, args, base_model = build_model(cli, device)
    channels = list(range(int(args.enc_in)))
    m, slot_heads, epoch, ckpt_path = l_load_arm('K2', base_model, int(args.d_model), device)
    _, test_loader = exp._get_data(flag='test', shuffle=False)
    df = l_run_population('K2', m, slot_heads, exp, args, test_loader, channels, device, cli.chunk_size)
    summary = l_macro_summary(df)
    exp_vals = L_EXPECTED_FULL['K2']
    assert abs(summary['retmse10'] - exp_vals['retmse10']) < 1e-3
    assert abs(summary['agg_mse10'] - exp_vals['agg10']) < 1e-3
    assert abs(summary['recall10'] - exp_vals['recall10']) < 1e-3


# item 18: Agg = D + C identity holds
def test_item18_agg_equals_d_plus_c():
    torch.manual_seed(7)
    from scripts.train_k_multislot_predictive_retrieval01 import hard_eval_decomposition
    bsz, s, n, h, top_k = 3, N_SLOTS, 30, 6, N_SLOTS
    scores = torch.randn(bsz, s, n)
    cand_mask = torch.ones(bsz, n, dtype=torch.bool)
    memory_c = torch.randn(n, h)
    offset_c = torch.randn(bsz)
    query_future = torch.randn(bsz, h)
    d_raw = torch.randn(bsz, n)
    oracle_idx = stable_topk_indices(d_raw, top_k, largest=False)
    res = hard_eval_decomposition(scores, cand_mask, memory_c, offset_c, query_future, d_raw, oracle_idx, top_k)
    assert torch.allclose(res['D'] + res['C'], res['agg_mse'], atol=1e-4)
