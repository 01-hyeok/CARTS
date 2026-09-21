"""TRACK-A-HORIZON-RETRIEVAL-HEADROOM01 sanity tests (spec section 21)."""
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.diag_horizon_retrieval_headroom01 import (BLOCKS, _gather_mean,
                                                        _gather_mean_masked, _mse, run_split)
from scripts.train_factorial_e2e01 import individual_utility, individual_utility_memsafe
from scripts.train_margutil01 import build_experiment, memory_value

REF_CKPT = ('checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/'
           'stage1_carts_softset_ETTh1_720_S0_wce_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_'
           'dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl720_pl720_0/'
           'checkpoint.pth')


def _available():
    return (REPO_ROOT / REF_CKPT).exists()


pytestmark_real = pytest.mark.skipif(not _available(), reason='ETTh1_720 S0_wce checkpoint not present')


# ---- T1: block boundaries exact ----
def test_t1_block_boundaries_exact():
    assert BLOCKS == {'block1': (0, 96), 'block2': (96, 336), 'block3': (336, 720)}
    lo1, hi1 = BLOCKS['block1']; lo2, hi2 = BLOCKS['block2']; lo3, hi3 = BLOCKS['block3']
    assert (hi1 - lo1, hi2 - lo2, hi3 - lo3) == (96, 240, 384)
    assert lo1 == 0 and hi1 == lo2 and hi2 == lo3 and hi3 == 720


# ---- T2: global MSE matches direct reference ----
def test_t2_global_mse_matches_direct_reference():
    torch.manual_seed(0)
    n, h, b = 50, 720, 4
    memory_c = torch.randn(n, h)
    offset_c = torch.randn(b)
    query_future = torch.randn(b, h)
    u_memsafe = individual_utility_memsafe(memory_c, offset_c, query_future, chunk_size=17)
    futures = memory_c + offset_c.view(-1, 1, 1)
    u_ref = individual_utility(futures, query_future)
    assert torch.allclose(u_memsafe, u_ref, atol=1e-5)


# ---- T3: block MSE matches direct slicing reference ----
def test_t3_block_mse_matches_direct_slicing():
    torch.manual_seed(1)
    n, h, b = 50, 720, 4
    memory_c = torch.randn(n, h)
    offset_c = torch.randn(b)
    query_future = torch.randn(b, h)
    for name, (lo, hi) in BLOCKS.items():
        u_sliced = individual_utility_memsafe(memory_c[:, lo:hi], offset_c, query_future[:, lo:hi],
                                              chunk_size=13)
        futures_full = memory_c + offset_c.view(-1, 1, 1)
        direct = -((futures_full[:, :, lo:hi] - query_future[:, lo:hi].unsqueeze(1)) ** 2).mean(-1)
        assert torch.allclose(u_sliced, direct, atol=1e-5), name


# ---- T4/T5: candidate mask identical + no invalid candidate in Oracle Top-K ----
@pytest.mark.skipif(not _available(), reason='ETTh1_720 S0_wce checkpoint not present')
def test_t4_t5_mask_identity_and_no_invalid_topk():
    exp, args = build_experiment(REF_CKPT, {
        'pred_len': 720, 'seq_len': 720, 'batch_size': 8, 'seed': 0, 'top_k': 10, 'tau_topk': 0.1})
    exp._ensure_memory()
    _, loader = exp._get_data(flag='val', shuffle=False)
    batch_x, batch_y, batch_start_idx = next(iter(loader))
    batch_x = batch_x.float().to(exp.device)
    batch_y = batch_y.float().to(exp.device)
    cand_mask, _ = exp._candidate_mask(batch_start_idx)
    cand_mask2, _ = exp._candidate_mask(batch_start_idx)
    assert torch.equal(cand_mask, cand_mask2)

    c = 0
    memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
    query_future = batch_y[:, :, c]
    u_g = individual_utility_memsafe(memory_c, offset_c, query_future, 4096).masked_fill(
        ~cand_mask, float('-inf'))
    s_g10 = stable_topk_indices(u_g, 10, largest=True)
    assert cand_mask.gather(1, s_g10).all()
    for name, (lo, hi) in BLOCKS.items():
        u_b = individual_utility_memsafe(memory_c[:, lo:hi], offset_c, query_future[:, lo:hi],
                                         4096).masked_fill(~cand_mask, float('-inf'))
        s_b = stable_topk_indices(u_b, 10, largest=True)
        assert cand_mask.gather(1, s_b).all(), name


# ---- T6: stable Top-K tie handling ----
def test_t6_stable_topk_tie_handling():
    values = torch.tensor([[1.0, 1.0, 1.0, 0.5, 2.0]])
    idx = stable_topk_indices(values, 3, largest=True)
    # ties at 1.0 broken by ascending candidate index -> 4 (2.0), 0, 1
    assert idx.tolist() == [[4, 0, 1]]


# ---- T7: uniform aggregate reconstruction matches manual toy example ----
def test_t7_uniform_aggregate_matches_manual():
    memory_c = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # [N=3, H=2]
    offset_c = torch.tensor([10.0, 20.0])  # [B=2]
    picks = torch.tensor([[0, 1], [1, 2]])  # [B=2, K=2]
    out = _gather_mean(memory_c, offset_c, picks, 0, 2)
    manual_row0 = ((memory_c[0] + 10.0) + (memory_c[1] + 10.0)) / 2
    manual_row1 = ((memory_c[1] + 20.0) + (memory_c[2] + 20.0)) / 2
    assert torch.allclose(out[0], manual_row0)
    assert torch.allclose(out[1], manual_row1)


# ---- T8: block concatenation length == 720 ----
def test_t8_block_concat_length_720():
    lengths = [hi - lo for lo, hi in BLOCKS.values()]
    assert sum(lengths) == 720


# ---- T9: synthetic short/mid/long best candidates selected correctly ----
def test_t9_block_oracle_selects_designed_winners():
    h = 720
    memory_c = torch.zeros(3, h)  # candidates A=0, B=1, C=2
    query_future = torch.zeros(1, h)
    # A is exact match on block1, far elsewhere; B exact on block2; C exact on block3
    memory_c[0, 0:96] = 0.0; memory_c[0, 96:] = 100.0
    memory_c[1, 96:336] = 0.0; memory_c[1, :96] = 100.0; memory_c[1, 336:] = 100.0
    memory_c[2, 336:720] = 0.0; memory_c[2, :336] = 100.0
    offset_c = torch.zeros(1)
    for name, (lo, hi), expected in zip(BLOCKS.keys(), BLOCKS.values(), (0, 1, 2)):
        u = individual_utility_memsafe(memory_c[:, lo:hi], offset_c, query_future[:, lo:hi], None)
        best = stable_topk_indices(u, 1, largest=True)
        assert int(best[0, 0]) == expected, name


# ---- T10: identical block relevance -> Global Top-K == Block Top-K, headroom~=0 ----
def test_t10_identical_relevance_zero_headroom():
    torch.manual_seed(2)
    n, h = 20, 720
    # Each candidate's error is a CONSTANT bias across the whole horizon (not
    # per-timestep noise), so every block-length MSE ranks candidates
    # identically to the global MSE -- a genuine "identical relevance" case.
    per_candidate_bias = torch.randn(n, 1)
    memory_c = per_candidate_bias.expand(n, h).clone()
    query_future = torch.zeros(1, h)
    offset_c = torch.zeros(1)
    u_g = individual_utility_memsafe(memory_c, offset_c, query_future, None)
    s_g = stable_topk_indices(u_g, 5, largest=True)
    for lo, hi in BLOCKS.values():
        u_b = individual_utility_memsafe(memory_c[:, lo:hi], offset_c, query_future[:, lo:hi], None)
        s_b = stable_topk_indices(u_b, 5, largest=True)
        assert torch.equal(s_g.sort().values, s_b.sort().values)


# ---- T11: Top-30 / budget-matched control ----
def test_t11_budget_matched_control():
    torch.manual_seed(3)
    n, h, b = 40, 720, 2
    memory_c = torch.randn(n, h)
    offset_c = torch.randn(b)
    query_future = torch.randn(b, h)
    u_g = individual_utility_memsafe(memory_c, offset_c, query_future, None)
    s_g30 = stable_topk_indices(u_g, 30, largest=True)
    counts = torch.tensor([10, 20])
    out = _gather_mean_masked(memory_c, offset_c, s_g30, counts, 0, h)
    for row in range(b):
        k = int(counts[row])
        manual = (memory_c[s_g30[row, :k]].mean(dim=0) + offset_c[row])
        assert torch.allclose(out[row], manual, atol=1e-5)


# ---- T12: delta_last reconstruction matches memory_value() production semantics ----
@pytest.mark.skipif(not _available(), reason='ETTh1_720 S0_wce checkpoint not present')
def test_t12_memory_value_delta_last_matches_production():
    exp, args = build_experiment(REF_CKPT, {
        'pred_len': 720, 'seq_len': 720, 'batch_size': 4, 'seed': 0, 'top_k': 10, 'tau_topk': 0.1})
    exp._ensure_memory()
    _, loader = exp._get_data(flag='val', shuffle=False)
    batch_x, batch_y, batch_start_idx = next(iter(loader))
    batch_x = batch_x.float().to(exp.device)
    c = 0
    memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
    if getattr(args, 'relation_value_space', None) == 'delta_last':
        manual = exp.memory_y[:, :, c] - exp.memory_x_last[:, c].unsqueeze(-1)
        assert torch.allclose(memory_c.cpu(), manual.cpu(), atol=1e-5)
        assert torch.allclose(offset_c.cpu(), batch_x[:, -1, c].detach().cpu(), atol=1e-5)
    else:
        assert torch.allclose(memory_c.cpu(), exp.memory_y[:, :, c].cpu(), atol=1e-5)
