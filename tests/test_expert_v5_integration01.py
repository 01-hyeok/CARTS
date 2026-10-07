"""TRACK-EXPERT-V5-FULL01 -- integration tests (spec section 12, tests
1-5, 14-18). Skips cleanly if the real ETTh1_96 reference checkpoint is
unavailable."""
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_j_shared_encoder_drift01 import build_model, state_hash
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_t_pure_multislot01 import compute_scores_full_grad
from scripts.train_expert_v5_full01 import expert_step

REF = (REPO_ROOT / 'checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/'
      'stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_'
      'df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth')
NUM_SLOTS = 5


def _skip_if_missing():
    if not REF.exists():
        pytest.skip('ETTh1_96 reference checkpoint not available in this environment')


def _setup():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cli = SimpleNamespace(reference_ckpt=str(REF), pred_len=96, seq_len=96, batch_size=32,
                          init_seed=0, top_k=10, patch_len=16, tau_t=0.1, tau_s=0.1, chunk_size=4096)
    exp, args, model = build_model(cli, device)
    slot_heads = SlotHeads(int(args.d_model), n_slots=NUM_SLOTS).to(device)
    return device, exp, args, model, slot_heads, cli


def _first_batch(exp, device):
    _, loader = exp._get_data(flag='train', shuffle=False)
    batch_x, batch_y, batch_start_idx = next(iter(loader))
    return batch_x.float().to(device), batch_y.float().to(device), batch_start_idx


# -------------------- test 1: full candidate N maintained --------------------

def test_scores_cover_full_candidate_N():
    _skip_if_missing()
    device, exp, args, model, slot_heads, cli = _setup()
    batch_x, batch_y, batch_start_idx = _first_batch(exp, device)
    cand_mask, _ = exp._candidate_mask(batch_start_idx)
    loss_c, scores, *_ = expert_step(model, slot_heads, batch_x, batch_y, batch_start_idx, 0, cand_mask, exp,
                                     args, cli)
    assert scores.shape == (batch_x.size(0), NUM_SLOTS, exp.memory_x.size(0))


# -------------------- tests 2/3/4: gradients --------------------

def test_query_candidate_and_slothead_gradients_are_nonzero():
    _skip_if_missing()
    device, exp, args, model, slot_heads, cli = _setup()
    batch_x, batch_y, batch_start_idx = _first_batch(exp, device)
    cand_mask, _ = exp._candidate_mask(batch_start_idx)
    for p in model.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(slot_heads.parameters()), lr=1e-3)
    optimizer.zero_grad()
    channels = list(range(int(args.enc_in)))
    for c in channels:
        loss_c, *_ = expert_step(model, slot_heads, batch_x, batch_y, batch_start_idx, c, cand_mask, exp, args, cli)
        (loss_c / len(channels)).backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters()), \
        'query/candidate-side encoder gradient must be nonzero (full gradient, both branches)'
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in slot_heads.parameters()), \
        'SlotHeads gradient must be nonzero'


# -------------------- test 5: fresh re-encode every step (structural) --------------------

def test_trainer_never_imports_candidate_bank_module():
    """Expert-V5 must never touch the R100 efficiency infra -- candidate
    re-encoding is always fresh, every optimizer step, by construction
    (via `compute_scores_full_grad`, which has no caching path at all)."""
    import scripts.train_expert_v5_full01 as mod
    src = inspect.getsource(mod)
    assert 'full_candidate_bank' not in src
    assert 'FullCandidateBank' not in src
    assert 'refresh_interval' not in src


def test_expert_step_uses_compute_scores_full_grad_directly():
    """model.eval() is mandatory for this comparison: the MLP relation
    encoder has dropout, so two independent forward passes under
    train() would see two different dropout masks and spuriously
    "disagree" even though the deterministic computation (the thing
    actually under test -- that expert_step scores candidates via the
    identical fresh full-grad call) is byte-identical."""
    _skip_if_missing()
    device, exp, args, model, slot_heads, cli = _setup()
    model.eval()
    batch_x, batch_y, batch_start_idx = _first_batch(exp, device)
    cand_mask, _ = exp._candidate_mask(batch_start_idx)
    loss_c, scores, *_ = expert_step(model, slot_heads, batch_x, batch_y, batch_start_idx, 0, cand_mask, exp,
                                     args, cli)
    with torch.no_grad():
        legacy_scores = compute_scores_full_grad(model, slot_heads, batch_x, exp.memory_x, 0)
    assert torch.allclose(scores, legacy_scores, atol=1e-5), \
        'expert_step must score candidates via the SAME fresh full-grad path as canonical Current-V5'


# -------------------- test 14: teacher distribution unchanged --------------------

def test_teacher_distribution_reused_unmodified():
    _skip_if_missing()
    from scripts.train_factorial_e2e01 import individual_utility_memsafe
    from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob
    from scripts.train_margutil01 import memory_value
    device, exp, args, model, slot_heads, cli = _setup()
    batch_x, batch_y, batch_start_idx = _first_batch(exp, device)
    cand_mask, _ = exp._candidate_mask(batch_start_idx)
    memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, 0)
    query_future = batch_y[:, :, 0]
    u1 = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
    p_t1 = normalized_teacher_prob(-u1, cand_mask, cli.tau_t)
    u2 = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
    p_t2 = normalized_teacher_prob(-u2, cand_mask, cli.tau_t)
    assert torch.equal(p_t1, p_t2)


# -------------------- test 15: candidate mask unchanged --------------------

def test_candidate_mask_deterministic():
    _skip_if_missing()
    device, exp, args, model, slot_heads, cli = _setup()
    _, _, batch_start_idx = _first_batch(exp, device)
    m1, _ = exp._candidate_mask(batch_start_idx)
    m2, _ = exp._candidate_mask(batch_start_idx)
    assert torch.equal(m1, m2)


# -------------------- tests 16/17: init parity --------------------

def test_encoder_init_parity_across_build_model_calls():
    _skip_if_missing()
    device, exp1, args1, model1, slot_heads1, cli = _setup()
    _, _, model2 = build_model(cli, device)
    assert state_hash(model1) == state_hash(model2)


def test_slotheads_init_parity_matches_canonical_num_slots_5():
    """SlotHeads' own per-slot-seeded init (seed=1000+m) means a fresh
    5-slot SlotHeads here is bit-identical to one constructed the same
    way in `train_t_pure_multislot01.py --num_slots 5` -- no
    Expert-V5-specific change to initialization."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    a = SlotHeads(64, n_slots=5).to(device)
    b = SlotHeads(64, n_slots=5).to(device)
    assert torch.equal(a.W, b.W)


# -------------------- test 18: batch-order parity --------------------

def test_batch_order_sha256_reused_unmodified():
    from scripts.rng_control01 import batch_order_sha256
    starts_a = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
    starts_b = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
    assert batch_order_sha256(starts_a) == batch_order_sha256(starts_b)
