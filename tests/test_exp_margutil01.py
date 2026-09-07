"""EXP-MARGUTIL01: Full-Memory Set-Conditioned Dense Marginal Utility.

Correctness/invariant tests for utils/dense_utility.py, models/
DenseUtilityRetriever.py, and scripts/train_margutil01.py's run_sequence_dense.
"""
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.DenseUtilityRetriever import UtilityHead
from models.SequentialSetRetriever import EmptySetToken, SetConditioner
from scripts.train_margutil01 import run_sequence_dense
from utils.dense_utility import candidate_weights, dense_utility, normalize_utility
from utils.oracle_intervention import select_greedy_weighted_set


def _toy(bsz=3, n=17, h=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    futures = torch.randn(bsz, n, h, generator=g)
    query_future = torch.randn(bsz, h, generator=g)
    scores = torch.randn(bsz, n, generator=g)
    valid = torch.ones(bsz, n, dtype=torch.bool)
    valid[:, -2:] = False  # a couple of structurally invalid candidates
    return futures, query_future, scores, valid


def _brute_force_a(prefix_idx, futures, query_future, w):
    """O(P) direct recomputation of A_weighted(S+i) for every i, no closed
    form -- an independent check against the vectorised incremental version."""
    bsz, n, h = futures.shape
    out = torch.empty(bsz, n)
    for b in range(bsz):
        prefix = prefix_idx[b].tolist()
        for i in range(n):
            s = prefix + [i]
            ws = w[b, s]
            ys = futures[b, s]
            agg = (ws.unsqueeze(-1) * ys).sum(0) / ws.sum().clamp_min(1e-12)
            out[b, i] = (agg - query_future[b]).pow(2).mean()
    return out


def test_dense_utility_matches_brute_force_including_empty_prefix():
    futures, query_future, scores, valid = _toy()
    tau = 0.1
    w = candidate_weights(scores, valid, tau)
    for prefix_len in (0, 1, 3):
        prefix_idx = torch.randint(0, futures.size(1), (futures.size(0), prefix_len))
        a_vec = dense_utility(prefix_idx, w, futures, query_future)
        a_bf = _brute_force_a(prefix_idx, futures, query_future, w)
        assert torch.allclose(a_vec, a_bf, atol=1e-5), f'mismatch at prefix_len={prefix_len}'


def test_t1_does_not_require_a_empty_and_uses_prefix_zero_sums():
    """t=1 (empty prefix): Z_S=M_S=0 falls straight out of summing over an
    empty selection -- no special-cased A(empty) constant anywhere. Uses
    tau=1.0 (not the production 0.1) so the identity check below is not
    swamped by legitimate float32 exp() underflow at extreme score spread --
    that underflow-safety clamp is exercised separately and is not what this
    test is checking."""
    futures, query_future, scores, valid = _toy()
    w = candidate_weights(scores, valid, tau=1.0)
    empty_prefix = torch.zeros(futures.size(0), 0, dtype=torch.long)
    a0 = dense_utility(empty_prefix, w, futures, query_future)
    # With an empty prefix, aggregate(S+{i}) reduces exactly to Y_i itself,
    # for every VALID candidate (invalid ones have w_i=0 -> denom clamped to
    # eps -> not a meaningful comparison, excluded here by design).
    expected = (futures - query_future.unsqueeze(1)).pow(2).mean(-1)
    assert torch.allclose(a0[valid], expected[valid], atol=1e-4)


def test_argmin_utility_matches_greedy_oracle_next_pick():
    """u_i = -A_i, so argmax(u) at every step must reproduce
    select_greedy_weighted_set's own greedy next-candidate choice exactly --
    same math, dense vs argmin-only."""
    futures, query_future, scores, valid = _toy(n=23)
    tau = 0.1
    oracle_seq = select_greedy_weighted_set(futures, query_future, scores, valid, k=4, tau=tau)
    w = candidate_weights(scores, valid, tau)
    for t in range(4):
        prefix = oracle_seq[:, :t]
        a_dense = dense_utility(prefix, w, futures, query_future)
        taken = torch.zeros_like(valid)
        for tt in range(t):
            taken.scatter_(1, oracle_seq[:, tt:tt + 1], True)
        a_dense = a_dense.masked_fill(taken | ~valid, float('inf'))
        pred_next = a_dense.argmin(dim=-1)
        assert torch.equal(pred_next, oracle_seq[:, t]), f'step {t} disagrees with oracle'


def test_chunked_matches_unchunked():
    futures, query_future, scores, valid = _toy(n=31)
    w = candidate_weights(scores, valid, tau=0.1)
    prefix = torch.randint(0, 31, (futures.size(0), 2))
    full = dense_utility(prefix, w, futures, query_future, chunk_size=None)
    chunked = dense_utility(prefix, w, futures, query_future, chunk_size=4)
    assert torch.allclose(full, chunked, atol=1e-6)


def test_normalization_finite_and_excludes_invalid():
    futures, query_future, scores, valid = _toy()
    w = candidate_weights(scores, valid, tau=0.1)
    prefix = torch.zeros(futures.size(0), 0, dtype=torch.long)
    a = dense_utility(prefix, w, futures, query_future)
    u = -a
    u_norm = normalize_utility(u, valid)
    assert torch.isfinite(u_norm).all()
    # mean/std computed only over valid positions: corrupting an invalid
    # position must not move the normalisation of the valid ones.
    u_corrupt = u.clone()
    u_corrupt[:, -1] = 1e9  # last column is invalid per _toy()
    u_norm_corrupt = normalize_utility(u_corrupt, valid)
    assert torch.allclose(u_norm[valid], u_norm_corrupt[valid], atol=1e-4)


def test_candidate_weights_zero_at_invalid_positions():
    _, _, scores, valid = _toy()
    w = candidate_weights(scores, valid, tau=0.1)
    assert torch.equal(w[~valid], torch.zeros_like(w[~valid]))
    assert (w[valid] > 0).all()


def test_run_sequence_free_running_has_no_duplicates_or_invalid_picks():
    bsz, n, d, k = 4, 40, 8, 6
    futures, query_future, scores, valid = _toy(bsz=bsz, n=n, h=5)
    q = torch.nn.functional.normalize(torch.randn(bsz, d), dim=-1)
    E = torch.nn.functional.normalize(torch.randn(n, d), dim=-1)
    set_conditioner = SetConditioner(d)
    empty_token = EmptySetToken(d)
    utility_head = UtilityHead()
    _, _, _, picks = run_sequence_dense(
        q, E, valid, set_conditioner, empty_token, utility_head,
        futures, query_future, tau=0.1, k=k, chunk_size=None, teacher_idx=None)
    for b in range(bsz):
        row = picks[b].tolist()
        assert len(set(row)) == len(row), 'duplicate pick'
        assert all(valid[b, i] for i in row), 'invalid candidate picked'


def test_run_sequence_teacher_forced_prefix_matches_oracle_exactly():
    """Teacher-forced picks must equal teacher_idx verbatim -- the set state
    at every step is built from the oracle prefix, never the model's own
    (possibly wrong) earlier picks, by construction."""
    bsz, n, d, k = 3, 25, 8, 5
    futures, query_future, scores, valid = _toy(bsz=bsz, n=n, h=5)
    teacher_idx = select_greedy_weighted_set(futures, query_future, scores, valid, k=k, tau=0.1)
    q = torch.nn.functional.normalize(torch.randn(bsz, d), dim=-1)
    E = torch.nn.functional.normalize(torch.randn(n, d), dim=-1)
    set_conditioner = SetConditioner(d)
    empty_token = EmptySetToken(d)
    utility_head = UtilityHead()
    _, u_target_steps, valid_steps, picks = run_sequence_dense(
        q, E, valid, set_conditioner, empty_token, utility_head,
        futures, query_future, tau=0.1, k=k, chunk_size=None, teacher_idx=teacher_idx)
    assert torch.equal(picks, teacher_idx)
    for u_target in u_target_steps:
        assert torch.isfinite(u_target).all()


def test_utility_head_forward_never_receives_query_future():
    """Leakage audit: UtilityHead.forward's signature is (h_t, candidate_
    embeddings) only -- structurally cannot see Y_q, by inspection of the
    call site in run_sequence_dense (only h and E are passed)."""
    import inspect
    sig = inspect.signature(UtilityHead.forward)
    params = list(sig.parameters)
    assert params == ['self', 'h_t', 'candidate_embeddings']


def test_set_conditioner_forward_never_receives_query_future():
    import inspect
    sig = inspect.signature(SetConditioner.forward)
    params = list(sig.parameters)
    assert params == ['self', 'q', 'm']


def test_gradient_flows_to_set_conditioner_and_utility_head_not_encoder():
    bsz, n, d, k = 2, 15, 8, 3
    futures, query_future, scores, valid = _toy(bsz=bsz, n=n, h=4)
    teacher_idx = select_greedy_weighted_set(futures, query_future, scores, valid, k=k, tau=0.1)
    q = torch.nn.functional.normalize(torch.randn(bsz, d), dim=-1)
    E = torch.nn.functional.normalize(torch.randn(n, d), dim=-1)  # frozen, no grad tracked
    set_conditioner = SetConditioner(d)
    empty_token = EmptySetToken(d)
    utility_head = UtilityHead()
    u_hat_steps, u_target_steps, valid_steps, _ = run_sequence_dense(
        q, E, valid, set_conditioner, empty_token, utility_head,
        futures, query_future, tau=0.1, k=k, chunk_size=None, teacher_idx=teacher_idx)
    loss = 0.0
    for u_hat, u_target, vmask in zip(u_hat_steps, u_target_steps, valid_steps):
        u_norm = normalize_utility(u_target, vmask)
        diff = torch.nn.functional.smooth_l1_loss(u_hat, u_norm, reduction='none')
        loss = loss + (diff * vmask.float()).sum() / vmask.float().sum().clamp_min(1.0)
    loss.backward()
    sc_grad_norm = sum(p.grad.norm().item() for p in set_conditioner.parameters() if p.grad is not None)
    uh_grad_norm = sum(p.grad.norm().item() for p in utility_head.parameters() if p.grad is not None)
    assert sc_grad_norm > 0
    assert uh_grad_norm > 0
    assert E.grad is None  # E was never requires_grad -- nothing to check but confirms no crash
