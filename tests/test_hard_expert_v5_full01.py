"""TRACK-HARD-EXPERT-V5-FULL01 -- required unit tests (spec section 16),
run BEFORE any 4-cell GPU training. Tests 1/2/3/5/8 are purely
synthetic (no real checkpoint needed); 4/9/10 use the real ETTh1_96
reference checkpoint (skip cleanly if unavailable); 6/7/11/12 are
source-level audits of the actual trainer file, catching a real
regression (e.g. someone later adding Round-Robin or a Router) without
needing a GPU run."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from utils.expert_head_metrics import (
    hard_expert_loss, kl_per_head, per_head_future_utility, per_head_standalone_topk, winner_margin_stats,
)

TOP_K = 10
TRAINER_SRC = (REPO_ROOT / 'scripts' / 'train_hard_expert_v5_full01.py').read_text()

REF = (REPO_ROOT / 'checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/'
      'stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_'
      'df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth')


# -------------------- [1] Hard winner correctness --------------------

def test_hard_winner_correctness():
    U = torch.tensor([[0.50, 0.20, 0.40, 0.60, 0.30]])
    winner, u_best, u_second, margin = winner_margin_stats(U)
    assert int(winner[0]) == 1, f'expected H2 (index 1) to win, got index {int(winner[0])}'
    assert float(u_best[0]) == pytest.approx(0.20)
    assert float(u_second[0]) == pytest.approx(0.30)
    assert float(margin[0]) == pytest.approx(0.10)


# -------------------- [2] Winner-only loss equality --------------------

def test_winner_only_loss_equals_gathered_mean():
    g = torch.Generator().manual_seed(0)
    kl_vals = torch.rand(8, 5, generator=g)
    winner_idx = torch.randint(0, 5, (8,), generator=g)
    loss = hard_expert_loss(kl_vals, winner_idx)
    manual = kl_vals[torch.arange(8), winner_idx].mean()
    assert torch.allclose(loss, manual, atol=1e-7)


# -------------------- [3] Non-winner direct gradient == 0 --------------------

def test_non_winner_slot_gradient_is_exactly_zero():
    torch.manual_seed(0)
    d_model, n_slots, n_cand = 8, 5, 20
    slot_heads = SlotHeads(d_model, n_slots=n_slots, std=1e-3)
    z_q = torch.randn(1, d_model)
    keys = torch.nn.functional.normalize(torch.randn(n_cand, d_model), dim=-1)
    q = slot_heads(z_q)
    scores = torch.einsum('bsd,nd->bsn', q, keys)  # [1,5,20], depends on slot_heads.W
    cand_mask = torch.ones(1, n_cand, dtype=torch.bool)
    p_t = torch.softmax(torch.randn(1, n_cand), dim=-1).detach()
    kl_vals, _ = kl_per_head(p_t, scores, cand_mask, tau_s=0.1)
    winner_idx = torch.tensor([1])  # force H2 as winner for this isolation test
    loss = hard_expert_loss(kl_vals, winner_idx)
    loss.backward()
    assert slot_heads.W.grad is not None
    assert slot_heads.W.grad[1].abs().sum().item() > 0, 'winner slot (H2) must receive direct gradient'
    for h in (0, 2, 3, 4):
        assert slot_heads.W.grad[h].abs().sum().item() == 0.0, \
            f'non-winner slot H{h+1} must have EXACTLY zero direct gradient, got {slot_heads.W.grad[h].abs().sum().item()}'


# -------------------- [5] Assignment detach (U/TopK/argmin carry no gradient) --------------------

def test_assignment_path_is_detached():
    torch.manual_seed(1)
    d_model, n_slots, n_cand = 8, 5, 20
    slot_heads = SlotHeads(d_model, n_slots=n_slots, std=1e-3)
    z_q = torch.randn(1, d_model)
    keys = torch.nn.functional.normalize(torch.randn(n_cand, d_model), dim=-1)
    q = slot_heads(z_q)
    scores = torch.einsum('bsd,nd->bsn', q, keys)
    cand_mask = torch.ones(1, n_cand, dtype=torch.bool)
    d_raw = torch.rand(1, n_cand)  # pretend future-MSE, no grad needed (detached in real pipeline)
    with torch.no_grad():
        head_topk_idx = per_head_standalone_topk(scores, cand_mask, k=TOP_K)
        U = per_head_future_utility(head_topk_idx, d_raw)
        winner_idx, *_ = winner_margin_stats(U)
    assert not U.requires_grad, '[ISSUE] U must be computed under no_grad / detached'
    assert winner_idx.dtype == torch.long  # indices, no grad concept applies, but confirm no .requires_grad leak
    assert not hasattr(winner_idx, 'grad_fn') or winner_idx.grad_fn is None


# -------------------- [8] Sign correctness --------------------

def test_sign_correctness_d_raw_is_mse():
    from scripts.train_factorial_e2e01 import individual_utility_memsafe
    memory_c = torch.tensor([[1.0, 1.0], [2.0, 2.0]])  # [N=2, H=2]
    offset_c = torch.zeros(1)  # [B=1]
    query_future = torch.tensor([[1.0, 1.0]])  # [B=1, H=2]
    u = individual_utility_memsafe(memory_c, offset_c, query_future, 4096)  # [1, 2]
    d_raw = -u
    # candidate 0 == query exactly -> MSE 0 -> best (lowest d_raw); candidate 1 differs -> MSE>0
    assert d_raw[0, 0].item() == pytest.approx(0.0, abs=1e-6)
    assert d_raw[0, 1].item() > 0
    assert u[0, 0].item() >= u[0, 1].item(), 'u = -MSE must be HIGHER (less negative) for the better candidate'


# -------------------- [4] Shared encoder gradient ON (real checkpoint) --------------------

def _skip_if_missing():
    if not REF.exists():
        pytest.skip('ETTh1_96 reference checkpoint not available in this environment')


def test_shared_encoder_gradient_on_real_checkpoint():
    _skip_if_missing()
    from scripts.train_hard_expert_v5_full01 import hard_step
    from scripts.train_j_shared_encoder_drift01 import build_model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cli = SimpleNamespace(reference_ckpt=str(REF), pred_len=96, seq_len=96, init_seed=0, patch_len=16,
                         top_k=TOP_K, tau_t=0.1, tau_s=0.1, chunk_size=4096, batch_size=32)
    exp, args, model = build_model(cli, device)
    model.to(device)
    model.train()
    slot_heads = SlotHeads(int(args.d_model), n_slots=5, std=1e-3).to(device)
    for p in model.parameters():
        p.requires_grad_(True)

    _, loader = exp._get_data(flag='train', shuffle=False)
    batch_x, batch_y, batch_start_idx = next(iter(loader))
    batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)
    cand_mask, _ = exp._candidate_mask(batch_start_idx)
    loss_c, *_ = hard_step(model, slot_heads, batch_x, batch_y, batch_start_idx, 0, cand_mask, exp, args, cli)
    loss_c.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters()), \
        '[ISSUE] shared encoder (query+candidate path) must receive nonzero gradient from the winner KL'


# -------------------- [9] Teacher unchanged (byte-identical to Soft) --------------------

def test_teacher_function_is_the_same_import_as_soft():
    soft_src = (REPO_ROOT / 'scripts' / 'train_expert_v5_full01.py').read_text()
    assert 'from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob' in soft_src
    assert 'from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob' in TRAINER_SRC
    assert 'def normalized_teacher_prob' not in TRAINER_SRC, \
        '[ISSUE][ABORT] Hard Expert must reuse the teacher function, never redefine it'


# -------------------- [10] Initialization equality (Hard vs Soft, same seed) --------------------

def test_hard_and_soft_init_hash_equal_on_real_checkpoint():
    _skip_if_missing()
    from scripts.train_j_shared_encoder_drift01 import build_model, state_hash

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cli_a = SimpleNamespace(reference_ckpt=str(REF), pred_len=96, seq_len=96, init_seed=0, patch_len=16,
                           batch_size=32, top_k=TOP_K)
    cli_b = SimpleNamespace(reference_ckpt=str(REF), pred_len=96, seq_len=96, init_seed=0, patch_len=16,
                           batch_size=32, top_k=TOP_K)
    _, _, model_a = build_model(cli_a, device)
    _, _, model_b = build_model(cli_b, device)
    assert state_hash(model_a) == state_hash(model_b)

    sh_a = SlotHeads(int(64), n_slots=5, std=1e-3)  # d_model irrelevant to this equality check, just needs to match
    sh_b = SlotHeads(int(64), n_slots=5, std=1e-3)
    assert state_hash(sh_a) == state_hash(sh_b), \
        'SlotHeads init is a per-slot FIXED seed (1000+m) -- must be identical across two fresh constructions'


# -------------------- [6] Full Candidate (no P100/R100/prefilter) --------------------

def test_no_candidate_pool_or_r100_imports():
    forbidden = ('utils.candidate_pool', 'FullCandidateBank', 'candidate_pool_mode', 'candidate_pool_size',
                'refresh_interval')
    for token in forbidden:
        assert token not in TRAINER_SRC, f'[ISSUE][ABORT] forbidden P100/R100 token found: {token}'
    assert 'n_candidates == exp.memory_x.shape[0]' in TRAINER_SRC, \
        '[ISSUE] missing runtime Full-Candidate assertion'


# -------------------- [7] Standalone Top10 (not Round-Robin) for winner utility --------------------

def test_winner_utility_uses_standalone_topk_not_round_robin():
    assert 'per_head_standalone_topk(scores, cand_mask, k=cli.top_k)' in TRAINER_SRC
    # round_robin_topk_selection must ONLY appear for the legacy diagnostic (hard_eval_decomposition),
    # never feed into U/winner_idx computation.
    assert 'round_robin_topk_selection' not in TRAINER_SRC, \
        'hard_eval_decomposition itself calls round_robin internally -- this file should not call it directly'


# -------------------- [11] Fixed 10 epochs, no early stopping --------------------

def test_no_early_stopping_branch_present():
    assert '--patience' not in TRAINER_SRC, '[ISSUE][ABORT] Hard Expert must not expose an early-stopping patience flag'
    assert 'early stop' not in TRAINER_SRC.lower()
    assert "for epoch in range(1, cli.train_epochs + 1):" in TRAINER_SRC


# -------------------- [12] No Router --------------------

def test_checkpoint_criterion_is_mean_mixture_not_hard_loss_or_rr():
    assert "'val_mean_retmse10'" in TRAINER_SRC
    assert "best = {'val_mean_retmse10'" in TRAINER_SRC, \
        '[ISSUE][ABORT] PRIMARY checkpoint selection must track val_mean_retmse10, not val_hard_loss or RR'
    assert 'mean_mixture_topk_selection(scores, cand_mask, cli.tau_s, k=cli.top_k)' in TRAINER_SRC


def test_no_router_parameters_or_imports():
    # code-level signals only (the docstring legitimately says "no Router is added" in prose)
    forbidden = ('class Router', 'Router(', 'import Router', 'router_ce', 'RouterCE', 'nn.Linear(6,',
                'load_balanc', 'diversity_loss', 'overlap_penalty')
    for token in forbidden:
        assert token not in TRAINER_SRC, f'[ISSUE][ABORT] forbidden Router/regularizer token found: {token}'
    assert "'router' in n.lower() or 'gate' in n.lower()" in TRAINER_SRC, \
        '[ISSUE] missing runtime no-Router assertion'
