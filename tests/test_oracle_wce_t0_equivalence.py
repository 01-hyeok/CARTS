"""EXP-ORACLE-WCE-CONTROL01 -- t=0 Individual==Set equivalence gate (spec
section 8). At t=0 the prefix S_{t-1} is EMPTY for both Oracles, so
Set's `Aggregate(S_{t-1} U {i}) = Aggregate({i})` reduces to the
singleton candidate's own value -- independent of the host weighting
`w_host` (which only matters once the aggregate set has >=2 members, at
t>=1). Verified directly against the REAL reference implementations
(`train_factorial_e2e01.individual_utility` / `greedy_set_utility`) under
a REALISTIC (non-degenerate, softmax-normalized) host weight -- not a
contrived delta weighting -- confirming t=0 equivalence holds generally,
not just in a special case.
"""
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_factorial_e2e01 import greedy_set_utility, individual_utility
from utils.oracle_utility_wce import wce_step_loss

ATOL, RTOL = 1e-6, 1e-5


def test_t0_distance_identical_under_realistic_host_weight():
    """PASS/[ISSUE] gate: t=0 Individual distance must equal Set distance
    under a REALISTIC (softmax-normalized, non-delta) host weight -- the
    actual weight used in real training, not a special case."""
    bsz, n_cand, pred_len = 4, 12, 6
    torch.manual_seed(3)
    futures = torch.randn(bsz, n_cand, pred_len)
    query_future = torch.randn(bsz, pred_len)
    empty_prefix = torch.zeros(bsz, 0, dtype=torch.long)
    w_host = torch.softmax(torch.randn(bsz, n_cand), dim=-1)

    d_ind = -individual_utility(futures, query_future)
    u_set = greedy_set_utility(empty_prefix, w_host, futures, query_future, chunk_size=4096)
    d_set = -u_set

    max_abs_diff = float((d_ind - d_set).abs().max())
    ok = torch.allclose(d_ind, d_set, atol=ATOL, rtol=RTOL)
    assert ok, (
        f'[ISSUE] t=0 Individual distance != Set distance under a realistic host weight.\n'
        f'FIRST DIVERGENCE:\nTENSOR: distance (d_ind vs d_set)\nSTEP: t=0\n'
        f'MAX_ABS_DIFF: {max_abs_diff}\n'
        f'LIKELY CAUSE: Aggregate(S_{{-1}} U {{i}}) with S_{{-1}}=empty should reduce to the '
        f'singleton candidate value regardless of host weighting -- if this fails, either the '
        f'aggregation formula or the host-weight application changed.')
    print(f'[t0_equivalence] PASS, max_abs_diff={max_abs_diff:.2e} (atol={ATOL})')


def test_t0_topk_indices_identical():
    bsz, n_cand, pred_len = 3, 10, 5
    torch.manual_seed(4)
    futures = torch.randn(bsz, n_cand, pred_len)
    query_future = torch.randn(bsz, pred_len)
    empty_prefix = torch.zeros(bsz, 0, dtype=torch.long)
    w_host = torch.softmax(torch.randn(bsz, n_cand), dim=-1)

    d_ind = -individual_utility(futures, query_future)
    d_set = -greedy_set_utility(empty_prefix, w_host, futures, query_future, chunk_size=4096)

    idx_ind = d_ind.topk(5, dim=-1, largest=False).indices
    idx_set = d_set.topk(5, dim=-1, largest=False).indices
    assert torch.equal(idx_ind, idx_set), (
        '[ISSUE] t=0 Top-M indices differ between Individual and Set Oracle -- exact index '
        'equality required, not approximate.')


def test_t0_teacher_weights_identical():
    from models.RelationStage1 import prepare_topk_coverage_targets
    bsz, n_cand, pred_len = 3, 10, 5
    torch.manual_seed(5)
    futures = torch.randn(bsz, n_cand, pred_len)
    query_future = torch.randn(bsz, pred_len)
    empty_prefix = torch.zeros(bsz, 0, dtype=torch.long)
    w_host = torch.softmax(torch.randn(bsz, n_cand), dim=-1)
    valid_now = torch.ones(bsz, n_cand, dtype=torch.bool)

    d_ind = -individual_utility(futures, query_future)
    d_set = -greedy_set_utility(empty_prefix, w_host, futures, query_future, chunk_size=4096)

    t_ind = prepare_topk_coverage_targets(d_ind, valid_now, top_k=10)
    t_set = prepare_topk_coverage_targets(d_set, valid_now, top_k=10)
    assert torch.equal(t_ind['oracle_indices'], t_set['oracle_indices'])
    assert torch.allclose(t_ind['oracle_mse'], t_set['oracle_mse'], atol=ATOL, rtol=RTOL)


def test_t0_wce_loss_identical_on_same_student_logits():
    bsz, n_cand, pred_len = 3, 10, 5
    torch.manual_seed(6)
    futures = torch.randn(bsz, n_cand, pred_len)
    query_future = torch.randn(bsz, pred_len)
    empty_prefix = torch.zeros(bsz, 0, dtype=torch.long)
    w_host = torch.softmax(torch.randn(bsz, n_cand), dim=-1)
    valid_now = torch.ones(bsz, n_cand, dtype=torch.bool)

    u_ind = individual_utility(futures, query_future)
    u_set = greedy_set_utility(empty_prefix, w_host, futures, query_future, chunk_size=4096)
    assert torch.allclose(u_ind, u_set, atol=ATOL, rtol=RTOL)

    u_hat1 = torch.randn(bsz, n_cand, requires_grad=True)
    u_hat2 = u_hat1.detach().clone().requires_grad_(True)
    l1, _ = wce_step_loss(u_hat1, u_ind, valid_now, tau=0.1, top_k_oracle=10)
    l2, _ = wce_step_loss(u_hat2, u_set, valid_now, tau=0.1, top_k_oracle=10)
    l1.backward(); l2.backward()
    assert torch.allclose(l1, l2, atol=ATOL, rtol=RTOL), f'{float(l1)} vs {float(l2)}'
    assert torch.allclose(u_hat1.grad, u_hat2.grad, atol=ATOL, rtol=RTOL)
