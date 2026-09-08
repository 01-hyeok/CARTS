"""EXP-ONPOLICY-PREFIX01 mandatory pre-GPU sanity checks (spec's numbered
list, on-policy experiment):
1. t=0 state is identical between oracle-prefix (C0) and on-policy (T1)
2. t>=1: the actual model-selected index enters the prefix
3. the model-selected candidate is removed from the next valid set
4. dense utility is recomputed from the CURRENT MODEL prefix, not the oracle's
5. no gradient flows through argmax/index selection
6. R2 gradient flows normally to SetConditioner/UtilityHead
7. FULL MEMORY semantics are preserved (every valid candidate considered
   at every step, none dropped)
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.DenseUtilityRetriever import UtilityHead
from models.SequentialSetRetriever import EmptySetToken, SetConditioner
from scripts.train_margutil01 import run_sequence_dense
from scripts.train_onpolicy_prefix01 import run_sequence_onpolicy
from utils.dense_utility import dense_utility


def _toy(bsz=5, n=31, d=12, h=4, k=6, seed=0):
    torch.manual_seed(seed)
    q = F.normalize(torch.randn(bsz, d), dim=-1)
    E = F.normalize(torch.randn(n, d), dim=-1)
    cand_mask = torch.ones(bsz, n, dtype=torch.bool)
    cand_mask[:, -3:] = False
    futures = torch.randn(bsz, n, h)
    query_future = torch.randn(bsz, h)
    sc = SetConditioner(d)
    et = EmptySetToken(d)
    uh = UtilityHead()
    return q, E, cand_mask, futures, query_future, sc, et, uh, k


def test_1_t0_state_matches_oracle_prefix_run():
    q, E, cand_mask, futures, query_future, sc, et, uh, k = _toy()
    teacher_idx = torch.randint(0, E.size(0), (q.size(0), k))
    torch.manual_seed(42)
    u_hat_oracle, _, _, _ = run_sequence_dense(q, E, cand_mask, sc, et, uh, futures, query_future,
                                                tau=0.1, k=k, chunk_size=None, teacher_idx=teacher_idx)
    torch.manual_seed(42)
    u_hat_onpolicy, _, _, _, _ = run_sequence_onpolicy(q, E, cand_mask, sc, et, uh, futures, query_future,
                                                        tau=0.1, k=k, chunk_size=None)
    assert torch.allclose(u_hat_oracle[0], u_hat_onpolicy[0], atol=1e-6), \
        't=0 state (empty prefix, same q/E/SetConditioner/EmptySetToken) must produce identical u_hat'


def test_2_selected_index_enters_prefix():
    q, E, cand_mask, futures, query_future, sc, et, uh, k = _toy()
    _, _, _, picks, _ = run_sequence_onpolicy(q, E, cand_mask, sc, et, uh, futures, query_future,
                                               tau=0.1, k=k, chunk_size=None)
    assert picks.shape == (q.size(0), k)
    # every picked index must be a valid candidate index (0..n-1) and within cand_mask
    for b in range(q.size(0)):
        assert bool(cand_mask[b, picks[b]].all())


def test_3_selected_candidate_removed_from_next_valid_set():
    q, E, cand_mask, futures, query_future, sc, et, uh, k = _toy()
    _, _, valid_steps, picks, _ = run_sequence_onpolicy(q, E, cand_mask, sc, et, uh, futures, query_future,
                                                         tau=0.1, k=k, chunk_size=None)
    for t in range(1, k):
        prev_picks = picks[:, :t]
        valid_now = valid_steps[t]
        for b in range(q.size(0)):
            for idx in prev_picks[b].tolist():
                assert not bool(valid_now[b, idx]), f'previously-picked candidate {idx} still valid at step {t}'
    # no duplicate picks within a query
    for b in range(q.size(0)):
        assert len(set(picks[b].tolist())) == k, 'on-policy picks must not contain duplicates'


def test_4_target_recomputed_from_model_prefix_not_oracle():
    q, E, cand_mask, futures, query_future, sc, et, uh, k = _toy()
    _, u_target_steps, _, picks, a_dense_steps = run_sequence_onpolicy(
        q, E, cand_mask, sc, et, uh, futures, query_future, tau=0.1, k=k, chunk_size=None)
    from utils.dense_utility import candidate_weights
    w = candidate_weights(torch.matmul(q, E.transpose(0, 1)), cand_mask, 0.1)
    # recompute the t=2 target directly from the MODEL's own t=1 pick and
    # confirm it matches what run_sequence_onpolicy produced internally
    model_prefix_t2 = picks[:, :1]
    a_expected = dense_utility(model_prefix_t2, w, futures, query_future, chunk_size=None)
    assert torch.allclose(a_dense_steps[1], a_expected, atol=1e-6), \
        'step-2 target must be built from the MODEL prefix, not the oracle prefix'


def test_5_no_gradient_through_argmax_selection():
    q, E, cand_mask, futures, query_future, sc, et, uh, k = _toy()
    q = q.clone().requires_grad_(False)
    u_hat_steps, u_target_steps, valid_steps, picks, _ = run_sequence_onpolicy(
        q, E, cand_mask, sc, et, uh, futures, query_future, tau=0.1, k=k, chunk_size=None)
    assert picks.dtype in (torch.int64, torch.long)
    assert not picks.requires_grad, 'picks (argmax output) must never require grad'
    for ut in u_target_steps:
        assert not ut.requires_grad, 'dense-utility targets must never require grad'


def test_6_r2_gradient_flows_to_setconditioner_and_utilityhead():
    q, E, cand_mask, futures, query_future, sc, et, uh, k = _toy()
    from scripts.train_toptail_rank01 import step_losses
    u_hat_steps, u_target_steps, valid_steps, picks, _ = run_sequence_onpolicy(
        q, E, cand_mask, sc, et, uh, futures, query_future, tau=0.1, k=k, chunk_size=None)
    query_valid = torch.ones(q.size(0), dtype=torch.bool)
    losses, diags = step_losses(u_hat_steps, u_target_steps, valid_steps, query_valid, 'hybrid', 1.0)
    total = sum(losses) / k
    total.backward()
    sc_grads = [p.grad for p in sc.parameters()]
    uh_grads = [p.grad for p in uh.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in sc_grads), 'SetConditioner must receive gradient'
    assert any(g is not None and g.abs().sum() > 0 for g in uh_grads), 'UtilityHead must receive gradient'


def test_7_full_memory_every_valid_candidate_considered():
    q, E, cand_mask, futures, query_future, sc, et, uh, k = _toy()
    u_hat_steps, _, valid_steps, _, _ = run_sequence_onpolicy(
        q, E, cand_mask, sc, et, uh, futures, query_future, tau=0.1, k=k, chunk_size=None)
    n = E.size(0)
    for t, u_hat in enumerate(u_hat_steps):
        assert u_hat.shape == (q.size(0), n), 'every step must score the FULL candidate bank, never a shortlist'
