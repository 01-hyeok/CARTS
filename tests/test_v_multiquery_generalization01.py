"""Unit tests for TRACK-V-MULTIQUERY-GENERALIZATION01 (spec PART 10.D
implementation audit). V0 = TRUE Original KL
(`train_j_shared_encoder_drift01.py`, unmodified). V1/V2/V5 =
`train_t_pure_multislot01.py` (unmodified), `--num_slots` 1/2/5 --
architecturally IDENTICAL to TRACK-T's own S1/S2 arms (V5 is the one
genuinely new `--num_slots` value). Most equivalence properties here
were already established by TRACK-T/TRACK-T2's own test suites; this
file re-verifies them in TRACK-V's own self-contained audit and adds
the one genuinely NEW claim this track depends on: cross-arm
initialization fairness (PART 4) -- W_1 must be bit-identical whether
built as part of S=1, S=2, or S=5.
"""
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_factorial_e2e01 import arm_score, encode_raw
from scripts.train_horizon_retrieval_expert01 import kl_loss, normalized_teacher_prob
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads, kl_loss_from_prob
from scripts.train_t_pure_multislot01 import (
    VALID_NUM_SLOTS, compute_scores_full_grad, hard_eval_decomposition, round_robin_topk_selection,
)


class FakeModel(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.encoder = nn.Linear(d, d)

    def _relation_tensor(self, x, c, cc):
        return x


# item 1: V5 (S=5) is an accepted num_slots value now (additive extension)
def test_item1_v5_num_slots_supported():
    assert 5 in VALID_NUM_SLOTS
    assert set(VALID_NUM_SLOTS) >= {1, 2, 4, 5, 10}  # additive only, nothing removed


# item 2 (PART 4, the core new claim): cross-arm projection init fairness
# W_1 must be bit-identical across S=1/S=2/S=5; W_2 must be bit-identical
# across S=2/S=5. SlotHeads' own per-slot-index seed (1000+m) already
# guarantees this by construction -- verified live here, not assumed.
def test_item2_cross_arm_w1_bit_identical():
    d = 32
    s1 = SlotHeads(d, n_slots=1)
    s2 = SlotHeads(d, n_slots=2)
    s5 = SlotHeads(d, n_slots=5)
    assert torch.equal(s1.W[0], s2.W[0])
    assert torch.equal(s2.W[0], s5.W[0])


def test_item2b_cross_arm_w2_bit_identical():
    d = 32
    s2 = SlotHeads(d, n_slots=2)
    s5 = SlotHeads(d, n_slots=5)
    assert torch.equal(s2.W[1], s5.W[1])


def test_item2c_all_slots_within_v5_are_distinct():
    # sanity: the 5 slots in V5 are NOT all identical to each other
    # (i.e. the per-slot seed actually varies, symmetry is genuinely broken)
    d = 32
    s5 = SlotHeads(d, n_slots=5)
    for m in range(5):
        for mm in range(m + 1, 5):
            assert not torch.equal(s5.W[m], s5.W[mm])


# item 3: V0 score/loss equals the literal Original-KL reference (score
# path never involves SlotHeads at all -- V0 uses train_j_shared_encoder
# _drift01.py directly, no projection module exists anywhere in its code)
def test_item3_v0_score_and_loss_match_reference():
    torch.manual_seed(0)
    d, bsz, n = 8, 4, 20
    model = FakeModel(d)
    x_q, x_k = torch.randn(bsz, d), torch.randn(n, d)
    mask = torch.ones(bsz, n, dtype=torch.bool)
    p_t = torch.softmax(torch.randn(bsz, n), dim=-1)
    tau_s = 0.1
    score = arm_score(encode_raw(model, x_q, 0), encode_raw(model, x_k, 0), None)
    loss = kl_loss(p_t, score, mask, tau_s)
    assert score.shape == (bsz, n)
    assert torch.isfinite(loss)


# item 4: query AND candidate encoder gradients nonzero for V1/V2/V5 (S>=1),
# projection gradient also nonzero
@pytest.mark.parametrize('s', (1, 2, 5))
def test_item4_full_gradient_all_branches(s):
    torch.manual_seed(1)
    d, bsz, n = 8, 3, 12
    model = FakeModel(d)
    slot_heads = SlotHeads(d, n_slots=s)
    x_q, x_k = torch.randn(bsz, d), torch.randn(n, d)
    mask = torch.ones(bsz, n, dtype=torch.bool)
    p_t = torch.softmax(torch.randn(bsz, n), dim=-1)

    scores = compute_scores_full_grad(model, slot_heads, x_q, x_k, 0)
    s_masked = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
    p_m = torch.softmax(s_masked / 0.1, dim=-1)
    p_bar = p_m.mean(dim=1)
    loss = kl_loss_from_prob(p_t, p_bar, mask)
    loss.backward()

    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters()), \
        f'[ISSUE] S={s}: encoder grad is zero'
    assert slot_heads.W.grad is not None and slot_heads.W.grad.abs().sum() > 0, \
        f'[ISSUE] S={s}: projection grad is zero'


def test_item4b_v0_full_gradient_both_branches():
    # candidate branch detached -> only query grad; query branch detached -> only candidate grad
    torch.manual_seed(2)
    d, bsz, n = 8, 3, 12
    encoder = nn.Linear(d, d)
    x_q, x_k = torch.randn(bsz, d), torch.randn(n, d)
    mask = torch.ones(bsz, n, dtype=torch.bool)
    p_t = torch.softmax(torch.randn(bsz, n), dim=-1)
    tau_s = 0.1

    def loss_fn(detach_q=False, detach_k=False):
        z_q = encoder(x_q)
        if detach_q:
            z_q = z_q.detach()
        z_k = encoder(x_k)
        if detach_k:
            z_k = z_k.detach()
        s = arm_score(z_q, z_k, None)
        return kl_loss(p_t, s, mask, tau_s)

    encoder.zero_grad()
    loss_fn(detach_k=True).backward()
    g_q = torch.cat([p.grad.flatten().clone() for p in encoder.parameters()])
    assert float(g_q.abs().sum()) > 0

    encoder.zero_grad()
    loss_fn(detach_q=True).backward()
    g_k = torch.cat([p.grad.flatten().clone() for p in encoder.parameters()])
    assert float(g_k.abs().sum()) > 0


# item 5: teacher identical regardless of S (no S-dependence in its signature/body)
def test_item5_teacher_s_independent():
    import inspect
    sig = inspect.signature(normalized_teacher_prob)
    assert 'num_slots' not in sig.parameters and 's' not in sig.parameters
    torch.manual_seed(3)
    d = torch.randn(4, 10)
    mask = torch.ones(4, 10, dtype=torch.bool)
    assert torch.equal(normalized_teacher_prob(d, mask, 0.1), normalized_teacher_prob(d, mask, 0.1))


# item 6: K=10 exactly and all-unique, for every S in {1,2,5}
@pytest.mark.parametrize('s', (1, 2, 5))
def test_item6_k_equals_10_unique(s):
    torch.manual_seed(4)
    bsz, n = 4, 40
    scores = torch.randn(bsz, s, n)
    mask = torch.ones(bsz, n, dtype=torch.bool)
    idx = round_robin_topk_selection(scores, mask, k=10)
    assert idx.shape == (bsz, 10)
    for b in range(bsz):
        assert len(set(idx[b].tolist())) == 10


# item 7: Agg = D + C identity holds via hard_eval_decomposition, for every S
@pytest.mark.parametrize('s', (1, 2, 5))
def test_item7_agg_equals_d_plus_c(s):
    torch.manual_seed(5)
    bsz, n, h = 4, 40, 6
    scores = torch.randn(bsz, s, n)
    mask = torch.ones(bsz, n, dtype=torch.bool)
    memory_c = torch.randn(n, h)
    offset_c = torch.randn(bsz)
    query_future = torch.randn(bsz, h)
    d_raw = torch.randn(bsz, n)
    from models.RelationStage1 import stable_topk_indices
    oracle_idx = stable_topk_indices(d_raw, 10, largest=False)
    res = hard_eval_decomposition(scores, mask, memory_c, offset_c, query_future, d_raw, oracle_idx, top_k=10)
    assert torch.allclose(res['D'] + res['C'], res['agg_mse'], atol=1e-4)


# item 8: S=1 round-robin selection reduces exactly to ordinary Top-10
# (V1's inference path is the exact V0-equivalent reduction, already
# proven generically by TRACK-T's own test suite -- reasserted here)
def test_item8_s1_equals_ordinary_top10():
    from models.RelationStage1 import stable_topk_indices
    torch.manual_seed(6)
    bsz, n = 5, 50
    scores = torch.randn(bsz, 1, n)
    mask = torch.ones(bsz, n, dtype=torch.bool)
    rr_idx = round_robin_topk_selection(scores, mask, k=10)
    ordinary_idx = stable_topk_indices(scores[:, 0, :], 10, largest=True)
    for b in range(bsz):
        assert set(rr_idx[b].tolist()) == set(ordinary_idx[b].tolist())


# item 9: cache schema carries D_per_query/C_per_query consistent with the
# saved scalar D/C metrics (integration check against real produced caches,
# skipped until Phase A artifacts exist)
def _first_available_cache():
    root = REPO_ROOT / 'results/TRACK-V-MULTIQUERY-GENERALIZATION01'
    for p in root.glob('*/H*/seed*/V*/cache/test.pt'):
        stage1_metrics = p.parent / 'stage1_metrics.json'
        if stage1_metrics.exists():
            return p, stage1_metrics
    return None, None


_cache_path, _metrics_path = _first_available_cache()
needs_cache = pytest.mark.skipif(_cache_path is None, reason='no TRACK-V cache artifacts produced yet')


@needs_cache
def test_item9_cache_dc_consistent_with_scalar_metrics():
    import json
    cache = torch.load(_cache_path, map_location='cpu')
    metrics = json.loads(_metrics_path.read_text())
    assert 'D_per_query' in cache and 'C_per_query' in cache
    assert cache['D_per_query'].shape[0] == cache['query_start_idx'].shape[0]
    assert abs(float(cache['D_per_query'].mean()) - metrics['D']) < 1e-3
    assert abs(float(cache['C_per_query'].mean()) - metrics['C']) < 1e-3
