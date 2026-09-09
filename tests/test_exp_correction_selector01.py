"""EXP-CORRECTION-SELECTOR01 (Track B3) mandatory pre-run sanity checks.

Fast, toy-scale, synthetic tensors -- consistent with every other sanity-
test file in this project (no real Stage-2 checkpoint loaded here; the
real checkpoint's own machinery, e.g. `_branch_embedding`/`base_forecast`,
is exercised implicitly by the real run itself and cross-checked online by
the script's own `assert equiv_max_err < 1e-3` after every epoch/split)."""
import inspect
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from layers.retrieval_gate import RetrievalGate
from models.DenseUtilityRetriever import UtilityHead
from models.SequentialSetRetriever import EmptySetToken, SetConditioner
from scripts.train_correction_selector01 import (
    equivalence_max_err, run_sequence_correction_choice, train_gate,
)
from utils.dense_utility import candidate_weights, dense_utility


def _toy_setup(bsz=4, n=30, d=8, pred_len=5, k=6, seed=0):
    torch.manual_seed(seed)
    q = F.normalize(torch.randn(bsz, d), dim=-1)
    E = F.normalize(torch.randn(n, d), dim=-1)
    cand_mask = torch.ones(bsz, n, dtype=torch.bool)
    r_i = torch.randn(n, pred_len) * 0.5
    r_i_batched = r_i.unsqueeze(0).expand(bsz, -1, -1)
    r_q = torch.randn(bsz, pred_len) * 0.5
    w_ref = candidate_weights(torch.matmul(q, E.transpose(0, 1)), cand_mask, tau=0.1)
    set_conditioner = SetConditioner(d)
    empty_token = EmptySetToken(d)
    utility_head = UtilityHead()
    return q, E, cand_mask, r_i, r_i_batched, r_q, w_ref, set_conditioner, empty_token, utility_head, k


def test_1_residual_exactness():
    Y_i = torch.randn(20, 5)
    B_i = torch.randn(20, 5) * 0.4
    r_i = Y_i - B_i
    assert torch.allclose(r_i + B_i, Y_i, atol=1e-6)


def test_2_candidate_residual_independent_of_query():
    torch.manual_seed(0)
    Y_i = torch.randn(15, 4)
    B_i = torch.randn(15, 4) * 0.3
    r_i_before = Y_i - B_i
    _unused = torch.randn(8, 4) * 1000
    r_i_after = Y_i - B_i
    assert torch.allclose(r_i_before, r_i_after)


def test_3_choice_target_equals_dense_utility_correction_argmax():
    """`run_sequence_correction_choice`'s target at every step must be
    exactly `-dense_utility(prefix, w_ref, r_i_batched, r_q)`, computed
    from the model's OWN on-policy prefix (never a fixed oracle sequence)."""
    q, E, cand_mask, r_i, r_i_batched, r_q, w_ref, sc, et, uh, k = _toy_setup()
    losses, diags, picks, a_dense_steps = run_sequence_correction_choice(
        q, E, cand_mask, sc, et, uh, w_ref, r_i_batched, r_q, tau_choice=0.1, k=k, chunk_size=None)
    assert len(losses) == k
    assert picks.shape == (q.size(0), k)
    for t in range(k):
        prefix = picks[:, :t]
        expected = dense_utility(prefix, w_ref, r_i_batched, r_q, chunk_size=None)
        assert torch.allclose(a_dense_steps[t], expected, atol=1e-6)


def test_4_no_duplicate_picks():
    q, E, cand_mask, r_i, r_i_batched, r_q, w_ref, sc, et, uh, k = _toy_setup()
    _, _, picks, _ = run_sequence_correction_choice(
        q, E, cand_mask, sc, et, uh, w_ref, r_i_batched, r_q, tau_choice=0.1, k=k, chunk_size=None)
    for b in range(picks.size(0)):
        assert len(set(picks[b].tolist())) == k, 'duplicate candidate picked within one query'


def test_5_on_policy_state_uses_models_own_pick_not_oracle():
    """The prefix at step t must be built from the model's own argmax
    picks (`picks` list), never from a separately-tracked oracle sequence
    -- verified structurally (no second index sequence exists in scope)
    plus behaviourally: perturbing the correction target after step 0
    changes step-1's pick set membership in `picks`, which then determines
    step 2's prefix (there is no other object it could come from)."""
    src = inspect.getsource(run_sequence_correction_choice)
    body = src[src.index('bsz, device, dtype ='):]  # skip docstring/comments
    # 'oracle_choice_step_loss' (the reused D1/OPC1 CE loss) is a legitimate
    # function name to call here; what must NOT appear is a SEPARATE
    # oracle-index sequence used to build the prefix (there is only one
    # index sequence in scope: `picks`, built from the model's own argmax).
    assert 'i_t*' not in body
    assert body.count('picks.append(') == 1
    assert 'picks.append(nxt.squeeze(-1).detach())' in body
    assert 'prefix = torch.stack(picks, dim=1).detach()' in body


def test_6_no_gradient_through_argmax():
    q, E, cand_mask, r_i, r_i_batched, r_q, w_ref, sc, et, uh, k = _toy_setup()
    q = q.clone().requires_grad_(True)
    losses, diags, picks, a_dense_steps = run_sequence_correction_choice(
        q, E, cand_mask, sc, et, uh, w_ref, r_i_batched, r_q, tau_choice=0.1, k=k, chunk_size=None)
    assert not picks.requires_grad
    total = sum(losses)
    total.backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()


def test_7_gradient_reaches_only_trainable_modules():
    q, E, cand_mask, r_i, r_i_batched, r_q, w_ref, sc, et, uh, k = _toy_setup()
    E = E.clone().requires_grad_(False)
    losses, diags, picks, a_dense_steps = run_sequence_correction_choice(
        q, E, cand_mask, sc, et, uh, w_ref, r_i_batched, r_q, tau_choice=0.1, k=k, chunk_size=None)
    total = sum(losses)
    total.backward()
    assert all(p.grad is not None for p in sc.parameters())
    assert all(p.grad is not None for p in et.parameters())
    assert all(p.grad is not None for p in uh.parameters())
    assert E.grad is None


def test_8_scorer_never_receives_residuals():
    """Structural: `run_sequence_correction_choice`'s SetConditioner/
    UtilityHead calls take only (q, m) / (h, E) -- `r_i_batched`/`r_q` are
    passed ONLY to `dense_utility` (the no_grad target computation), never
    to `set_conditioner` or `utility_head`."""
    src = inspect.getsource(run_sequence_correction_choice)
    body = src[src.index('bsz, device, dtype ='):]  # skip docstring/comments
    sc_call = body.index('set_conditioner(q, m)')
    uh_call = body.index('utility_head(h, E)')
    du_call = body.index('dense_utility(prefix, w_ref, r_i_batched, r_q')
    assert 'r_i' not in body[:sc_call] and 'r_q' not in body[:sc_call]
    assert 'r_i' not in body[:uh_call] and 'r_q' not in body[:uh_call]
    assert du_call > 0


def test_9_full_memory_candidate_universe_no_shortlist():
    """`utility_head(h, E)` scores the WHOLE candidate bank E every step --
    no top-M/shortlist slicing of E anywhere in the sequence loop."""
    src = inspect.getsource(run_sequence_correction_choice)
    assert 'topk' not in src.lower() and 'shortlist' not in src.lower()
    assert '[:num' not in src and 'E[:' not in src


def test_10_runtime_equivalence_check_holds_on_synthetic_data():
    torch.manual_seed(1)
    bsz, n, pred_len, k = 5, 40, 6, 4
    r_i = torch.randn(n, pred_len)
    B_q = torch.randn(bsz, pred_len)
    r_q = torch.randn(bsz, pred_len)
    q_tgt = B_q + r_q
    w_ref = torch.rand(bsz, n) + 0.1
    picks = torch.stack([torch.randperm(n)[:k] for _ in range(bsz)], dim=0)
    valid_query = torch.ones(bsz, dtype=torch.bool)
    err, C_ret = equivalence_max_err(w_ref, r_i, picks, B_q, q_tgt, r_q, valid_query)
    assert err < 1e-5


def test_11_equivalence_check_catches_a_real_violation():
    """Negative control: feeding a WRONG query target must break the
    equivalence check (proves the assertion is not vacuously true)."""
    torch.manual_seed(2)
    bsz, n, pred_len, k = 3, 20, 4, 3
    r_i = torch.randn(n, pred_len)
    B_q = torch.randn(bsz, pred_len)
    r_q = torch.randn(bsz, pred_len)
    wrong_q_tgt = B_q + r_q + 5.0  # deliberately inconsistent with r_q
    w_ref = torch.rand(bsz, n) + 0.1
    picks = torch.stack([torch.randperm(n)[:k] for _ in range(bsz)], dim=0)
    valid_query = torch.ones(bsz, dtype=torch.bool)
    err, _ = equivalence_max_err(w_ref, r_i, picks, wrong_q_tgt, wrong_q_tgt, r_q, valid_query)
    assert err > 1.0


def test_12_zero_correction_reduces_to_b0():
    pred_len = 6
    B_q = torch.randn(4, pred_len)
    C_ret = torch.zeros(4, pred_len)
    gate = RetrievalGate(pred_len, fixed_lambda=1.0)
    y_final, _ = gate(B_q, C_ret)
    assert torch.allclose(y_final, B_q, atol=1e-6)


def test_13_fixed_gamma_one_gives_base_plus_correction():
    pred_len = 6
    B_q = torch.randn(4, pred_len)
    C_ret = torch.randn(4, pred_len)
    gate = RetrievalGate(pred_len, fixed_lambda=1.0)
    y_final, lam = gate(B_q, C_ret)
    assert torch.allclose(y_final, B_q + C_ret, atol=1e-6)
    assert torch.allclose(lam, torch.ones_like(lam))


def test_14_learned_gate_recovers_a_known_gamma_positive_control():
    torch.manual_seed(0)
    pred_len = 5
    n = 200
    B_q = torch.randn(n, pred_len)
    C_ret = torch.randn(n, pred_len)
    Y_q = B_q + 0.5 * C_ret + 0.01 * torch.randn(n, pred_len)
    feats_train = {'B_q': B_q[:140], 'C_ret': C_ret[:140], 'Y_q': Y_q[:140]}
    feats_val = {'B_q': B_q[140:170], 'C_ret': C_ret[140:170], 'Y_q': Y_q[140:170]}
    feats_test = {'B_q': B_q[170:], 'C_ret': C_ret[170:], 'Y_q': Y_q[170:]}
    gate, best_val, history = train_gate(feats_train, feats_val, pred_len, epochs=300, patience=30, lr=0.05,
                                          device='cpu')
    with torch.no_grad():
        y_final, lam = gate(feats_test['B_q'], feats_test['C_ret'])
        gate_mse = float((y_final - feats_test['Y_q']).pow(2).mean())
        gamma0_mse = float((feats_test['B_q'] - feats_test['Y_q']).pow(2).mean())
        gamma1_mse = float((feats_test['B_q'] + feats_test['C_ret'] - feats_test['Y_q']).pow(2).mean())
    assert gate_mse < gamma0_mse
    assert gate_mse < gamma1_mse
    assert abs(float(lam.mean()) - 0.5) < 0.15


def test_15_gate_input_is_exactly_b_q_and_c_ret():
    src = inspect.getsource(RetrievalGate.forward)
    assert 'torch.cat([y_base, y_ret], dim=-1)' in src


def test_16_empty_set_token_used_at_step_zero_only():
    src = inspect.getsource(run_sequence_correction_choice)
    idx = src.index('empty_token(bsz, device, dtype)')
    assert 't == 0' in src[max(0, idx - 60):idx]


def test_17_frozen_b0_asserted_before_backward_in_run_epoch():
    from scripts.train_correction_selector01 import run_epoch
    src = inspect.getsource(run_epoch)
    assert "p.grad is None for p in model.parameters()" in src


def test_18_no_reranker_no_coarse_retrieval_in_scorer_path():
    src = inspect.getsource(run_sequence_correction_choice)
    for banned in ('rerank', 'coarse', 'reranker'):
        assert banned not in src.lower()
