"""EXP-STRONG-SCORER-DIAG01 OOM fix: streaming (chunked, incremental-backward)
training path must be mathematically equivalent to the original
hold-the-whole-graph implementation -- same scores, same mined hard
negatives, same loss values, same parameter gradients."""
import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.DenseUtilityRetriever import StrongResidualPairScorer
from models.SequentialSetRetriever import EmptySetToken, SetConditioner
from scripts.train_toptail_rank01 import (mine_pairs, pairwise_step_loss,
                                           strong_pair_step)
from utils.dense_utility import normalize_utility


def _toy(bsz=4, n=97, d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = F.normalize(torch.randn(bsz, d, generator=g), dim=-1)
    m = F.normalize(torch.randn(bsz, d, generator=g), dim=-1)
    E = F.normalize(torch.randn(n, d, generator=g), dim=-1)
    u_target = torch.randn(bsz, n, generator=g)
    valid = torch.ones(bsz, n, dtype=torch.bool)
    valid[:, -3:] = False
    return q, m, E, u_target, valid


def _build_pair(d):
    torch.manual_seed(0)
    sc_a = SetConditioner(d)
    sc_b = copy.deepcopy(sc_a)
    head_a = StrongResidualPairScorer(dim=d, chunk_size=1000)  # effectively unchunked
    head_b = copy.deepcopy(head_a)
    head_b.chunk_size = 11  # deliberately awkward chunk size for the streaming path
    return sc_a, head_a, sc_b, head_b


def test_mined_pairs_match_between_nograd_and_withgrad_pass():
    q, m, E, u_target, valid = _toy()
    d = E.size(-1)
    sc, head, _, _ = _build_pair(d)
    h = sc(q, m)
    u_hat = head(h, E)
    pos_ref, neg_ref, negvalid_ref = mine_pairs(u_target, u_hat.detach(), valid)
    with torch.no_grad():
        h2 = sc(q, m)
        u_hat2 = head(h2, E)
    pos2, neg2, negvalid2 = mine_pairs(u_target, u_hat2, valid)
    assert torch.equal(pos_ref, pos2)
    assert torch.equal(neg_ref, neg2)
    assert torch.equal(negvalid_ref, negvalid2)


def test_streaming_gradients_match_unchunked_reference():
    """Old (unchunked, single backward) vs new (chunked, incremental
    backward) must produce the SAME gradient on every SetConditioner and
    scorer parameter, within float tolerance."""
    q, m, E, u_target, valid = _toy(bsz=6, n=133, d=16)
    d = E.size(-1)
    sc_a, head_a, sc_b, head_b = _build_pair(d)
    lambda_rank = 1.0

    # ---- (a) reference: single full-memory forward, one backward ----
    h_a = sc_a(q, m)
    u_hat_a = head_a(h_a, E)
    u_norm = normalize_utility(u_target, valid)
    diff = F.smooth_l1_loss(u_hat_a, u_norm, reduction='none')
    vf = valid.float()
    per_query_smooth = (diff * vf).sum(dim=-1) / vf.sum(dim=-1).clamp_min(1.0)
    smoothl1_ref = per_query_smooth.mean()
    pair_loss_ref, _ = pairwise_step_loss(u_hat_a, u_target, valid)
    total_ref = smoothl1_ref + lambda_rank * pair_loss_ref
    total_ref.backward()

    # ---- (b) streaming: mined from no-grad pass, chunked SmoothL1 ----
    strong_pair_step(q, (q, lambda: m), E, valid, sc_b, head_b, u_target, valid,
                      tau=0.1, lambda_rank=lambda_rank, grad_scale=1.0,
                      chunk_size=17, train=True)

    for (na, pa), (nb, pb) in zip(sc_a.named_parameters(), sc_b.named_parameters()):
        assert na == nb
        ga = pa.grad if pa.grad is not None else torch.zeros_like(pa)
        gb = pb.grad if pb.grad is not None else torch.zeros_like(pb)
        assert torch.allclose(ga, gb, atol=1e-4, rtol=1e-3), f'SetConditioner.{na} gradient mismatch'

    for (na, pa), (nb, pb) in zip(head_a.named_parameters(), head_b.named_parameters()):
        assert na == nb
        ga = pa.grad if pa.grad is not None else torch.zeros_like(pa)
        gb = pb.grad if pb.grad is not None else torch.zeros_like(pb)
        assert torch.allclose(ga, gb, atol=1e-4, rtol=1e-3), f'ScorerHead.{na} gradient mismatch'


def test_streaming_with_trainable_m_source_does_not_double_backward():
    """Regression test for the exact bug caught on real data: at t=1, `m`
    comes from the TRAINABLE `EmptySetToken`. If `m` were precomputed once
    and reused across the pairwise forward + multiple SmoothL1 chunk
    forwards, the second `.backward()` through EmptySetToken's shared graph
    segment would raise. `m_fn` must be a fresh callable so each forward's
    graph (including through EmptySetToken) is independent."""
    q, _, E, u_target, valid = _toy(bsz=5, n=211, d=16)
    d = E.size(-1)
    sc = SetConditioner(d)
    head = StrongResidualPairScorer(dim=d, chunk_size=1000)
    empty_token = EmptySetToken(d)

    def m_fn():
        return empty_token(q.size(0), q.device, q.dtype)

    # Must not raise "Trying to backward through the graph a second time".
    diag = strong_pair_step(q, (q, m_fn), E, valid, sc, head, u_target, valid,
                             tau=0.1, lambda_rank=1.0, grad_scale=1.0,
                             chunk_size=23, train=True)
    assert diag['smoothl1_loss'] == diag['smoothl1_loss']
    assert empty_token.empty.grad is not None
    assert float(empty_token.empty.grad.abs().sum()) > 0


def test_streaming_never_calls_optimizer_step_itself():
    """strong_pair_step must only call .backward(), never .step() -- the
    caller (run_epoch) owns exactly one optimizer.step() per batch."""
    import ast
    import inspect
    src = inspect.getsource(strong_pair_step)
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    step_calls = [c for c in calls if isinstance(c.func, ast.Attribute) and c.func.attr == 'step']
    assert not step_calls, 'strong_pair_step must never call .step() itself'


def test_streaming_peak_memory_does_not_scale_with_full_candidate_count():
    """A rough smoke check: streaming with a small chunk_size on a larger N
    must not raise and must produce finite gradients (the actual GPU-memory
    claim is validated on real hardware separately; this at least exercises
    many chunk iterations without correctness regressions)."""
    q, m, E, u_target, valid = _toy(bsz=3, n=2503, d=16)
    d = E.size(-1)
    sc = SetConditioner(d)
    head = StrongResidualPairScorer(dim=d, chunk_size=997)
    diag = strong_pair_step(q, (q, lambda: m), E, valid, sc, head, u_target, valid,
                             tau=0.1, lambda_rank=1.0, grad_scale=1.0,
                             chunk_size=64, train=True)
    assert diag['smoothl1_loss'] == diag['smoothl1_loss']  # not NaN
    grads = [p.grad for p in list(sc.parameters()) + list(head.parameters()) if p.grad is not None]
    assert all(torch.isfinite(g).all() for g in grads)
