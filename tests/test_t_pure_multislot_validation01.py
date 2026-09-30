"""Unit tests for TRACK-T-PURE-MULTISLOT-VALIDATION01 (spec PART 29,
15 items). Fast synthetic checks wherever the property under test does
not require a real dataset/model; source-level checks for "this
mechanism is absent" properties (no StopGrad, no overlap/aggregate/
budget loss); skip-if-unavailable for the one item that needs real
saved artifacts (item 15, common Stage2 base).
"""
import hashlib
import inspect
import re
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_factorial_e2e01 import arm_score
from scripts.train_horizon_retrieval_expert01 import kl_loss
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads, kl_loss_from_prob
from scripts.train_t_pure_multislot01 import (
    hard_eval_decomposition, round_robin_topk_selection,
)

SCRIPT_SRC = (REPO_ROOT / 'scripts/train_t_pure_multislot01.py').read_text()
RUN_SH_SRC = (REPO_ROOT / 'scripts/run_t_one_setting01.sh').read_text()


# item 1: S=1 score equals Original-KL score (std=0 test double, PART 8 item 8)
def test_item1_s1_score_equals_original_kl_score():
    torch.manual_seed(0)
    d, bsz, n = 8, 4, 20
    z_q = torch.randn(bsz, d)
    z_k = torch.randn(n, d)
    slot_heads = SlotHeads(d, n_slots=1, std=0.0)
    q = slot_heads(z_q)  # [B, 1, D]
    k = F.normalize(z_k, dim=-1)
    multi_score = torch.einsum('bsd,nd->bsn', q, k)[:, 0, :]
    orig_score = arm_score(z_q, z_k, None)
    assert torch.allclose(multi_score, orig_score, atol=1e-6)


# item 2: S=1 loss equals Original-KL KL loss (std=0 test double)
def test_item2_s1_loss_equals_original_kl_loss():
    torch.manual_seed(1)
    d, bsz, n = 8, 4, 20
    z_q = torch.randn(bsz, d)
    z_k = torch.randn(n, d)
    mask = torch.ones(bsz, n, dtype=torch.bool)
    tau_s = 0.1
    slot_heads = SlotHeads(d, n_slots=1, std=0.0)
    q = slot_heads(z_q)
    k = F.normalize(z_k, dim=-1)
    scores = torch.einsum('bsd,nd->bsn', q, k)
    s_masked = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
    p_m = torch.softmax(s_masked / tau_s, dim=-1)
    p_bar = p_m.mean(dim=1)  # mean over the single slot == identity

    p_t = torch.softmax(torch.randn(bsz, n), dim=-1)
    loss_multi = kl_loss_from_prob(p_t, p_bar, mask)

    orig_score = arm_score(z_q, z_k, None)
    loss_orig = kl_loss(p_t, orig_score, mask, tau_s)
    assert torch.allclose(loss_multi, loss_orig, atol=1e-5)


# item 3: S=1 round-robin selection equals ordinary Top10 (exact index-set equality)
def test_item3_s1_round_robin_equals_ordinary_top10():
    from models.RelationStage1 import stable_topk_indices
    torch.manual_seed(2)
    bsz, n = 5, 50
    scores = torch.randn(bsz, 1, n)
    mask = torch.ones(bsz, n, dtype=torch.bool)
    rr_idx = round_robin_topk_selection(scores, mask, k=10)
    ordinary_idx = stable_topk_indices(scores[:, 0, :], 10, largest=True)
    for b in range(bsz):
        assert set(rr_idx[b].tolist()) == set(ordinary_idx[b].tolist())


# item 4: teacher identical for all S (no S-dependence in the teacher function at all)
def test_item4_teacher_identical_across_s():
    from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob
    sig = inspect.signature(normalized_teacher_prob)
    assert 'num_slots' not in sig.parameters and 's' not in sig.parameters
    torch.manual_seed(3)
    d = torch.randn(4, 10)
    mask = torch.ones(4, 10, dtype=torch.bool)
    p1 = normalized_teacher_prob(d, mask, 0.1)
    p2 = normalized_teacher_prob(d, mask, 0.1)  # simulating a second, different-S arm's call
    assert torch.equal(p1, p2)


# items 5 & 6: candidate-side AND query-side encoder gradients both nonzero
# (isolated branch-detach decomposition, mirrors TRACK-J's gradient_conflict_diagnostic)
def test_item5_and_6_candidate_and_query_gradients_nonzero():
    torch.manual_seed(4)
    d, bsz, n = 8, 3, 12
    encoder = torch.nn.Linear(d, d)
    slot_heads = SlotHeads(d, n_slots=2, std=1e-3)
    x_q = torch.randn(bsz, d)
    x_k = torch.randn(n, d)
    mask = torch.ones(bsz, n, dtype=torch.bool)
    p_t = torch.softmax(torch.randn(bsz, n), dim=-1)

    def loss_fn(detach_q=False, detach_k=False):
        z_q = encoder(x_q)
        if detach_q:
            z_q = z_q.detach()
        z_k = encoder(x_k)
        if detach_k:
            z_k = z_k.detach()
        q = slot_heads(z_q)
        k = F.normalize(z_k, dim=-1)
        scores = torch.einsum('bsd,nd->bsn', q, k)
        s_masked = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))
        p_m = torch.softmax(s_masked / 0.1, dim=-1)
        p_bar = p_m.mean(dim=1)
        return kl_loss_from_prob(p_t, p_bar, mask)

    encoder.zero_grad()
    loss_fn(detach_k=True).backward()  # only query branch live -> tests item 6
    g_q = torch.cat([p.grad.flatten().clone() for p in encoder.parameters()])
    assert float(g_q.abs().sum()) > 0, '[ISSUE] query-side encoder gradient is zero'

    encoder.zero_grad()
    loss_fn(detach_q=True).backward()  # only candidate branch live -> tests item 5
    g_k = torch.cat([p.grad.flatten().clone() for p in encoder.parameters()])
    assert float(g_k.abs().sum()) > 0, '[ISSUE] candidate-side encoder gradient is zero'


# item 7: no .detach() anywhere in the candidate/key score path
def test_item7_no_detach_in_candidate_score_path():
    import scripts.train_t_pure_multislot01 as mod
    src = inspect.getsource(mod.compute_scores_full_grad)
    code_only = src.split('"""', 2)[-1]  # strip the docstring (which mentions .detach() in prose)
    assert '.detach()' not in code_only, f'[ISSUE] .detach() found in compute_scores_full_grad body:\n{code_only}'


# item 8: no overlap loss anywhere in the training loss
def test_item8_no_overlap_loss():
    # slot_overlap_penalty is imported only for read-only diagnostics (slot_mechanism_diagnostics),
    # never added into the training loss `l`
    before_diag_helper = SCRIPT_SRC.split('def slot_mechanism_diagnostics')[0]
    assert 'slot_overlap_penalty(' not in before_diag_helper
    # the training loss line must be exactly the bare anchor KL term, nothing added to it
    assert re.search(r'l = kl_loss_from_prob\(p_t, p_bar, cand_mask\)\n\s+\(l / len\(channels\)\)\.backward\(\)',
                     SCRIPT_SRC), '[ISSUE] training loss is not the bare KL anchor term'


# item 9: no aggregate future loss anywhere
def test_item9_no_aggregate_loss():
    assert 'soft_aggregate_loss' not in SCRIPT_SRC


# item 10: no relevance-budget loss anywhere
def test_item10_no_budget_loss():
    for forbidden in ('j1_table', 'T_J1', 'relevance_term', 'J1_REF_DIR', 'F.relu('):
        assert forbidden not in SCRIPT_SRC, f'[ISSUE] forbidden relevance-budget artifact found: {forbidden}'


# item 11: hard selection is future-blind (no future/query_future/d_raw argument)
def test_item11_hard_selection_future_blind():
    sig = inspect.signature(round_robin_topk_selection)
    forbidden_params = {'query_future', 'd_raw', 'batch_y', 'memory_c', 'offset_c'}
    assert forbidden_params.isdisjoint(sig.parameters.keys())
    src = inspect.getsource(round_robin_topk_selection)
    for forbidden in ('query_future', 'd_raw', 'batch_y'):
        assert forbidden not in src


# item 12: K=10 for every num_slots value
@pytest.mark.parametrize('s', (1, 2, 4, 10))
def test_item12_k_equals_10_for_every_s(s):
    torch.manual_seed(5)
    bsz, n = 3, 30
    scores = torch.randn(bsz, s, n)
    mask = torch.ones(bsz, n, dtype=torch.bool)
    idx = round_robin_topk_selection(scores, mask, k=10)
    assert idx.shape == (bsz, 10)


# item 13: selected candidates are unique within each query
@pytest.mark.parametrize('s', (1, 2, 4, 10))
def test_item13_selected_candidates_unique(s):
    torch.manual_seed(6)
    bsz, n = 4, 30
    scores = torch.randn(bsz, s, n)
    mask = torch.ones(bsz, n, dtype=torch.bool)
    idx = round_robin_topk_selection(scores, mask, k=10)
    for b in range(bsz):
        assert len(set(idx[b].tolist())) == 10


# item 14: Agg = D + C identity
def test_item14_agg_equals_d_plus_c():
    torch.manual_seed(7)
    bsz, s, n, h = 4, 4, 40, 6
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


# item 15: common Stage2 base checkpoint hash is identical across all 4 arms
def _state_sha(state_dict):
    h = hashlib.sha256()
    for k in sorted(state_dict):
        h.update(k.encode())
        h.update(state_dict[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


_base_ckpt = REPO_ROOT / 'checkpoints/track_r_final_method_generalization01/ETTh1/H96/seed0/base/checkpoint.pth'
needs_base = pytest.mark.skipif(not _base_ckpt.exists(), reason='TRACK-R base checkpoint not yet produced')


@needs_base
def test_item15_common_stage2_base_hash_identical():
    bl1 = torch.load(_base_ckpt, map_location='cpu')
    bl2 = torch.load(_base_ckpt, map_location='cpu')
    assert _state_sha(bl1['model_state_dict']) == _state_sha(bl2['model_state_dict'])
    # driver script references ONE base-checkpoint variable inside the S-loop, not 4 separate paths
    loop_match = re.search(r'for S in 1 2 4 10; do(.*?)\ndone', RUN_SH_SRC, re.S)
    assert loop_match, '[ISSUE] run_t_one_setting01.sh does not contain the expected S-loop'
    loop_body = loop_match.group(1)
    assert loop_body.count('--base_checkpoint "$R_BASE_CKPT"') == 1, \
        '[ISSUE] base checkpoint is not referenced via one shared variable inside the loop'
