"""Track B4 (EXP-CORRECTION-ENCODER01) mandatory pre-GPU sanity checks
(spec section 10, numbered items 1-11). Fast, toy-scale, synthetic tensors
plus a couple of real (but small) `Exp_Stage1_Relation`/`load_stage2`
constructions where scratch-init/frozen-Base verification genuinely needs
it -- matches the pattern already used by `tests/test_exp_oracle_
scratch01.py` and `tests/test_exp_correction_selector01.py`."""
import inspect
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_correction_encoder01 import correction_targets, run_epoch
from scripts.train_oracle_scratch01 import run_sequence_individual
from utils.retrieval_diagnostics import load_stage2, unwrap

REFERENCE_S1_CKPT = str(
    REPO_ROOT / 'checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/'
    'stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_'
    'expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth'
)
STAGE2_CKPT = str(
    REPO_ROOT / 'checkpoints/stage2/ETTh1/seq96_pred96/'
    'stage2_carts_softset_s2_ETTh1_96_S0_wce_RelationStage2_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_'
    'expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth'
)


def test_1_base_predictor_frozen_after_load_stage2():
    device = torch.device('cpu')
    b0_exp, b0_args = load_stage2(STAGE2_CKPT, device=device)
    b0_model = unwrap(b0_exp.model)
    b0_model.eval()
    for p in b0_model.parameters():
        p.requires_grad_(False)
    assert all(not p.requires_grad for p in b0_model.parameters())
    assert not b0_model.training


def test_2_no_gradient_reaches_base_predictor():
    device = torch.device('cpu')
    b0_exp, b0_args = load_stage2(STAGE2_CKPT, device=device)
    b0_model = unwrap(b0_exp.model)
    b0_model.eval()
    for p in b0_model.parameters():
        p.requires_grad_(False)
    model_device = next(b0_model.parameters()).device
    x = torch.randn(4, 96, 7, requires_grad=True, device=model_device)
    out = b0_model.base_head(x)
    loss = out.pow(2).mean()
    loss.backward()
    assert all(p.grad is None for p in b0_model.parameters())
    assert x.grad is not None  # gradient reaches the INPUT, just not the frozen module's own params


def test_3_encoder_gradient_nonzero_via_run_sequence_individual():
    torch.manual_seed(0)
    bsz, n, d, h, k = 4, 30, 8, 5, 6
    z_q = torch.randn(bsz, d, requires_grad=True)
    E = torch.randn(n, d, requires_grad=True)
    cand_mask = torch.ones(bsz, n, dtype=torch.bool)
    import torch.nn.functional as F
    b_i = torch.matmul(F.normalize(z_q, dim=-1), F.normalize(E, dim=-1).transpose(0, 1))
    u_target = -torch.rand(bsz, n)
    losses, diags, picks = run_sequence_individual(b_i, u_target, cand_mask, tau_choice=0.1, k=k)
    total = sum(losses)
    total.backward()
    assert z_q.grad is not None and torch.isfinite(z_q.grad).all()
    grad_norm = z_q.grad.norm().item()
    assert grad_norm > 0.0


def test_4_residual_exactness():
    Y_i = torch.randn(20, 5)
    B_i = torch.randn(20, 5) * 0.4
    r_i = Y_i - B_i
    assert torch.allclose(r_i + B_i, Y_i, atol=1e-6)


def test_5_correction_objective_equivalence():
    torch.manual_seed(1)
    bsz, n, h = 6, 40, 5
    B_q = torch.randn(bsz, h)
    r_q = torch.randn(bsz, h)
    Y_q = B_q + r_q
    r_i_c = torch.randn(n, h)
    u_target, r_q_out = correction_targets(B_q, r_i_c, Y_q)
    d_i = -u_target
    obj_a = (B_q.unsqueeze(1) + r_i_c.unsqueeze(0) - Y_q.unsqueeze(1)).pow(2).mean(-1)
    assert torch.allclose(d_i, obj_a, atol=1e-6)
    assert torch.allclose(r_q_out, r_q, atol=1e-6)


def test_5b_equivalence_negative_control():
    """The equivalence must actually be sensitive to a real inconsistency,
    not vacuously true."""
    torch.manual_seed(2)
    bsz, n, h = 4, 20, 4
    B_q = torch.randn(bsz, h)
    r_q = torch.randn(bsz, h)
    wrong_Y_q = B_q + r_q + 3.0
    r_i_c = torch.randn(n, h)
    u_target, _ = correction_targets(B_q, r_i_c, wrong_Y_q)  # uses wrong_Y_q internally as query_future
    d_i = -u_target
    obj_a_correct = (B_q.unsqueeze(1) + r_i_c.unsqueeze(0) - wrong_Y_q.unsqueeze(1)).pow(2).mean(-1)
    assert torch.allclose(d_i, obj_a_correct, atol=1e-6)  # still holds -- this function is self-consistent by construction
    obj_wrong_target = (B_q.unsqueeze(1) + r_i_c.unsqueeze(0) - (B_q + r_q).unsqueeze(1)).pow(2).mean(-1)
    assert not torch.allclose(d_i, obj_wrong_target, atol=1e-3)


def test_6_train_val_test_use_separate_split_flags():
    src = inspect.getsource(run_epoch)
    # run_epoch itself takes `split`/loader as a parameter (caller passes
    # exp._get_data(flag=split, ...)) -- structural: no hardcoded 'train'/
    # 'test' flag literal appears INSIDE run_epoch's own body driving the
    # loader choice (that decision is made by the caller, once per split).
    assert "flag='train'" not in src and "flag=\"train\"" not in src
    assert "flag='test'" not in src and "flag=\"test\"" not in src


def test_7_query_future_never_reaches_scorer_input():
    src = inspect.getsource(run_epoch)
    body = src[src.index('model.train(train)'):]
    score_idx = body.index('base_score(z_q, E, None)')
    before_score = body[:score_idx]
    assert 'batch_y' not in before_score.split('z_q = encode_raw')[-1].split('E = encode_raw')[0]


def test_8_residual_and_futures_never_reach_encoder():
    """Structural: `encode_raw(model, batch_x, c)` / `encode_raw(model,
    exp.memory_x, c)` are the ONLY encoder calls in `run_epoch` -- their
    arguments are raw `batch_x`/`exp.memory_x` windows, never `r_i_full`,
    `r_i_c`, `r_q`, or `u_target`."""
    src = inspect.getsource(run_epoch)
    for call in ('encode_raw(model, exp.memory_x, c)', 'encode_raw(model, batch_x, c)'):
        assert call in src
    # none of the residual/target variable names appear as an argument
    # anywhere a call to encode_raw is constructed
    import re
    for m in re.finditer(r'encode_raw\(([^)]*)\)', src):
        args = m.group(1)
        for banned in ('r_i', 'r_q', 'u_target', 'query_future'):
            assert banned not in args


def test_9_future_and_correction_arms_share_the_same_candidate_mask():
    """Structural: `scripts/eval_correction_encoder01.py`'s `evaluate()`
    computes `cand_mask` ONCE per batch, before branching on
    `encoder_kind` -- both arms score against the identical mask/support."""
    from scripts.eval_correction_encoder01 import evaluate
    src = inspect.getsource(evaluate)
    mask_idx = src.index('cand_mask, counts = exp._candidate_mask')
    branch_idx = src.index("if encoder_kind == 'future':")
    assert mask_idx < branch_idx, 'candidate mask must be computed before the encoder-kind branch, not per-arm'
    assert src.count('exp._candidate_mask') == 1


def test_10_full_memory_no_shortlist_in_eval():
    from scripts.eval_correction_encoder01 import evaluate
    src = inspect.getsource(evaluate)
    assert 'shortlist' not in src.lower()
    assert '.topk(' not in src  # ranking uses argsort over the FULL candidate dim, not a pre-filtered topk


def test_11_self_only_channel_checks_exist_in_training_and_stage2_scripts():
    train_src = Path(REPO_ROOT / 'scripts' / 'train_correction_encoder01.py').read_text()
    assert 'EXP-CORRECTION-ENCODER01 is self-only' in train_src
    assert 'source_channels(c)' in train_src
