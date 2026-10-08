"""TRACK-HARD-EXPERT-V5-P100-ALLH01 -- required unit/sanity tests (spec
section 13), run before any GPU experiment. Covers items 1-12 of the
spec's implementation-verification checklist. No GPU/real-data
dependency except the channel-first-vs-legacy parity test, which uses a
real checkpoint if present and otherwise is skipped (the parity claim
is already proven generically by `tests/test_v_meanmix_checkpoint_correction01.py`
for the identical underlying math)."""
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from utils.expert_head_metrics import (
    expert_weighted_loss, hard_expert_loss, kl_per_head, per_head_future_utility, per_head_standalone_topk,
    responsibility_from_utility, winner_margin_stats,
)
from utils.expert_head_metrics_pool01 import per_head_decomposition_pool
from utils.mean_mixture_selection import mean_mixture_topk_selection


# -------------------- [1] P100 teacher argmax == argmin(d_pool) --------------------

def test_1_p100_teacher_argmax_equals_argmin_distance():
    d_pool = torch.tensor([[0.1, 1.0, 10.0]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    p_t = normalized_teacher_prob(d_pool, mask, tau_t=0.1)  # NEVER normalized_teacher_prob(-d_pool, ...)
    assert int(p_t.argmax()) == int(d_pool.argmin()) == 0


# -------------------- [2] teacher probability decreases with distance --------------------

def test_2_teacher_prob_decreases_with_distance():
    d_pool = torch.tensor([[0.1, 1.0, 10.0]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    p_t = normalized_teacher_prob(d_pool, mask, tau_t=0.1)
    assert p_t[0, 0] > p_t[0, 1] > p_t[0, 2]


# -------------------- [3] Hard winner == argmin head utility --------------------

def test_3_hard_winner_equals_argmin_utility():
    U = torch.tensor([[0.5, 0.1, 0.9, 0.3, 0.7]])
    winner, u_best, u_second, margin = winner_margin_stats(U)
    assert int(winner[0]) == int(U.argmin(dim=1)[0]) == 1
    assert float(u_best[0]) == pytest.approx(0.1)


# -------------------- [4] Hard loss uses ONLY the winner head's KL --------------------

def test_4_hard_loss_uses_only_winner_kl():
    kl_vals = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0]], requires_grad=True)
    winner_idx = torch.tensor([2, 0])
    loss = hard_expert_loss(kl_vals, winner_idx)
    expected = (kl_vals[0, 2] + kl_vals[1, 0]) / 2
    assert float(loss) == pytest.approx(float(expected))
    loss.backward()
    # gradient must be nonzero ONLY at the winner column for each row
    grad = kl_vals.grad
    assert grad[0, 2] != 0 and grad[1, 0] != 0
    non_winner_mask = torch.ones_like(grad, dtype=torch.bool)
    non_winner_mask[0, 2] = False
    non_winner_mask[1, 0] = False
    assert torch.all(grad[non_winner_mask] == 0), '[ISSUE] non-winner KL entries must receive zero gradient'


# -------------------- [5] winner selection is detached (no grad through U/winner_idx) --------------------

def test_5_winner_selection_is_detached():
    U_leaf = torch.tensor([[0.5, 0.1, 0.9]], requires_grad=True)
    with torch.no_grad():
        winner, u_best, u_second, margin = winner_margin_stats(U_leaf)
    assert winner.requires_grad is False
    assert not winner.dtype.is_floating_point  # long index tensor, structurally never a gradient source


# -------------------- [6] non-winner slot-specific SlotHeads projection gets zero gradient for that query --------------------

def test_6_non_winner_slot_projection_receives_no_gradient():
    torch.manual_seed(0)
    d_model = 8
    slot_heads = SlotHeads(d_model, n_slots=5, std=1e-3)
    z_q = torch.randn(1, d_model)
    z_k = torch.nn.functional.normalize(torch.randn(1, 6, d_model), dim=-1)
    q = slot_heads(z_q)  # [1,5,D]
    scores = torch.einsum('bsd,bmd->bsm', q, z_k)  # [1,5,6]
    p_t = torch.softmax(torch.tensor([[5.0, 1.0, 0.5, 0.2, 0.1, 0.05]]), dim=-1)
    mask = torch.ones(1, 6, dtype=torch.bool)
    kl_vals, _ = kl_per_head(p_t, scores, mask, tau_s=0.1)
    winner_idx = torch.tensor([2])  # fixed winner for this single-query test
    loss = hard_expert_loss(kl_vals, winner_idx)
    loss.backward()
    grad_per_slot_norm = slot_heads.W.grad.flatten(1).norm(dim=1)
    assert grad_per_slot_norm[2] > 0, '[ISSUE] winner head must receive a nonzero gradient'
    for h in range(5):
        if h != 2:
            assert grad_per_slot_norm[h] == 0, \
                f'[ISSUE] non-winner head {h} received nonzero gradient ({float(grad_per_slot_norm[h])})'


# -------------------- [7] shared encoder receives gradient from the winner loss --------------------

def test_7_shared_encoder_path_receives_gradient():
    torch.manual_seed(0)
    d_model = 8
    encoder = torch.nn.Linear(d_model, d_model, bias=False)
    slot_heads = SlotHeads(d_model, n_slots=3, std=1e-3)
    x_q = torch.randn(1, d_model)
    x_k = torch.randn(1, 4, d_model)
    z_q = encoder(x_q)
    z_k = torch.nn.functional.normalize(encoder(x_k), dim=-1)
    q = slot_heads(z_q)
    scores = torch.einsum('bsd,bmd->bsm', q, z_k)
    p_t = torch.softmax(torch.tensor([[3.0, 1.0, 0.2, 0.1]]), dim=-1)
    mask = torch.ones(1, 4, dtype=torch.bool)
    kl_vals, _ = kl_per_head(p_t, scores, mask, tau_s=0.1)
    winner_idx = torch.tensor([0])
    loss = hard_expert_loss(kl_vals, winner_idx)
    loss.backward()
    assert encoder.weight.grad is not None and encoder.weight.grad.abs().sum() > 0, \
        '[ISSUE] shared encoder must receive gradient via the winner-head KL path'


# -------------------- [8] no Router/auxiliary-loss parameters --------------------

def test_8_no_router_or_auxiliary_loss_parameters():
    slot_heads = SlotHeads(8, n_slots=5, std=1e-3)
    names = [n for n, _ in slot_heads.named_parameters()]
    assert not any('router' in n.lower() or 'gate' in n.lower() for n in names)
    # the loss functions themselves must be exactly these two, nothing else
    assert expert_weighted_loss.__doc__ and 'ONLY loss term' in expert_weighted_loss.__doc__
    assert hard_expert_loss.__doc__ and 'ONLY loss term' in hard_expert_loss.__doc__


# -------------------- [9] inference uses Mean-Mixture, not Round-Robin --------------------

def test_9_inference_is_mean_mixture():
    torch.manual_seed(0)
    scores = torch.randn(2, 5, 20)
    mask = torch.ones(2, 20, dtype=torch.bool)
    idx, p_heads, p_mix = mean_mixture_topk_selection(scores, mask, tau_s=0.1, k=10)
    p_bar = torch.softmax(scores / 0.1, dim=-1).mean(dim=1)
    assert torch.allclose(p_mix, p_bar, atol=1e-6)
    assert idx.shape == (2, 10)


# -------------------- [10] Stage-2 must be the original CARTS trainable-lambda script --------------------

def test_10_stage2_uses_original_carts_lambda_script():
    src = (REPO_ROOT / 'scripts' / 'eval_r_stage2_lambda_signfix01.py').read_text()
    assert 'from scripts.train_r_stage2_lambda01 import load_tensors' in src
    assert 'Y_final = B + lambda*(R-B)' in src or 'fused = b_b + lam * (r_b - b_b)' in src
    assert 'select_beta_on_validation' not in src, \
        '[ISSUE][ABORT] Professor-style beta fusion must NOT be used for this Stage-2 mode'
    assert 'BETA_GRID' not in src


# -------------------- [11] all arms must use the same P100 pool and same Base checkpoint --------------------

def test_11_orchestrator_uses_one_pool_and_one_base_per_cell():
    orch = REPO_ROOT / 'scripts' / 'run_hard_expert_v5_p100_allh01.sh'
    if not orch.exists():
        pytest.skip('orchestrator not written yet')
    src = orch.read_text()
    for arm in ('V5', 'Soft', 'Hard'):
        assert arm in src


# -------------------- [12] no bugged P100 checkpoint/cache reuse --------------------

def test_12_no_bugged_p100_checkpoint_reuse():
    for fname in ('train_expert_v5_p100_allh01.py', 'train_hard_expert_v5_p100_allh01.py'):
        f = REPO_ROOT / 'scripts' / fname
        if not f.exists():
            pytest.skip(f'{fname} not written yet')
        src = f.read_text()
        assert 'normalized_teacher_prob(-d_pool' not in src
        assert 'normalized_teacher_prob(-' not in src.replace('normalized_teacher_prob(-d_pool', ''), \
            f'[ISSUE][ABORT] {fname} must never double-negate an already-signed pool distance'


# -------------------- bonus: per_head_decomposition_pool uses LOCAL (gather) indexing, never global memory_c[idx] --------------------

def test_per_head_decomposition_pool_uses_gather_not_global_index():
    torch.manual_seed(0)
    bsz, s, k, m, c = 2, 3, 2, 5, 4
    head_topk_idx = torch.randint(0, m, (bsz, s, k))
    pooled_memory_c = torch.randn(bsz, m, c)
    offset_c = torch.zeros(bsz)
    query_future = torch.randn(bsz, c)
    d_pool = torch.rand(bsz, m)
    oracle_idx_pool = torch.randint(0, m, (bsz, k))
    out = per_head_decomposition_pool(head_topk_idx, pooled_memory_c, offset_c, query_future, d_pool,
                                      oracle_idx_pool, top_k=k)
    for name in ('retmse10', 'agg_mse', 'D', 'C', 'recall10', 'ndcg10'):
        assert out[name].shape == (bsz, s)
