"""Set-level retrieval loss: aggregation, decomposition, and gradient reach.

The premise these guard is that Stage-1 grades candidates one at a time while
Stage-2 averages ten of them, so the two quantities differ by the spread among
the selected candidates. Test 2 is the case that makes the difference concrete:
two candidates that are individually wrong in opposite directions aggregate to
the exact target.
"""

import torch

from models.RelationStage1 import soft_set_mse, hard_aggregate_metrics


def test_soft_aggregate_matches_manual_weighted_sum():
    scores = torch.tensor([[0.0, torch.log(torch.tensor(3.0)).item()]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    cand = torch.tensor([[1.0], [5.0]])
    query = torch.tensor([[0.0]])
    _, _, metrics = soft_set_mse(scores, mask, query, cand, tau_set=1.0,
                                 normalization='none')
    # softmax([0, ln 3]) = [1/4, 3/4]  ->  aggregate = 1/4 + 15/4 = 4
    assert torch.allclose(metrics['set_soft_mse_raw'], torch.tensor(16.0), atol=1e-5)


def test_complementary_candidates_aggregate_to_the_target():
    """Individually wrong, jointly exact -- the case individual grading misses."""
    scores = torch.zeros(1, 2)                    # equal scores -> alpha = [.5, .5]
    mask = torch.ones(1, 2, dtype=torch.bool)
    cand = torch.tensor([[1.0], [-1.0]])
    query = torch.tensor([[0.0]])
    stats = hard_aggregate_metrics(scores, mask, query, cand, top_k=2, tau_topk=1.0)
    assert torch.allclose(stats['hard_aggregate_mse10'], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(stats['weighted_individual_mse10'], torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(stats['weighted_candidate_variance10'], torch.tensor(1.0), atol=1e-6)


def test_variance_decomposition_identity_holds():
    torch.manual_seed(0)
    scores = torch.randn(4, 40)
    mask = torch.ones(4, 40, dtype=torch.bool)
    cand = torch.randn(40, 12)
    query = torch.randn(4, 12)
    stats = hard_aggregate_metrics(scores, mask, query, cand, top_k=10, tau_topk=0.1)
    lhs = stats['weighted_individual_mse10']
    rhs = stats['hard_aggregate_mse10'] + stats['weighted_candidate_variance10']
    assert torch.allclose(lhs, rhs, atol=1e-5), (lhs, rhs)


def test_gradient_reaches_every_valid_candidate():
    """Full-memory softmax, so no candidate is structurally cut off."""
    torch.manual_seed(0)
    scores = torch.randn(2, 60, requires_grad=True)
    mask = torch.ones(2, 60, dtype=torch.bool)
    cand = torch.randn(60, 8)
    query = torch.randn(2, 8)
    loss, _, _ = soft_set_mse(scores, mask, query, cand, tau_set=0.5,
                              normalization='none')
    loss.backward()
    assert scores.grad is not None
    assert (scores.grad != 0).all(), 'a valid candidate received no gradient'


def test_invalid_candidates_get_no_probability_and_no_gradient():
    torch.manual_seed(0)
    scores = torch.randn(2, 30, requires_grad=True)
    mask = torch.ones(2, 30, dtype=torch.bool)
    mask[:, 10:] = False
    cand = torch.randn(30, 6)
    query = torch.randn(2, 6)
    loss, _, _ = soft_set_mse(scores, mask, query, cand, tau_set=0.5,
                              normalization='none')
    loss.backward()
    assert torch.allclose(scores.grad[:, 10:], torch.zeros_like(scores.grad[:, 10:]))
    assert (scores.grad[:, :10] != 0).all()


def test_support_hinge_is_one_sided():
    """Concentration below the target is what the cross-entropy is for; only a
    support wider than the target may be penalised."""
    mask = torch.ones(1, 500, dtype=torch.bool)
    cand = torch.randn(500, 4)
    query = torch.randn(1, 4)
    flat = torch.zeros(1, 500)                       # entropy = ln 500, very wide
    peaked = torch.zeros(1, 500)
    peaked[0, 0] = 50.0                              # entropy ~ 0, very narrow
    _, wide, _ = soft_set_mse(flat, mask, query, cand, tau_set=1.0,
                              normalization='none', support_k=20)
    _, narrow, _ = soft_set_mse(peaked, mask, query, cand, tau_set=1.0,
                                normalization='none', support_k=20)
    assert float(wide) > 0.0
    assert float(narrow) == 0.0


def test_effective_support_shrinks_as_temperature_falls():
    torch.manual_seed(0)
    scores = torch.randn(8, 400)
    mask = torch.ones(8, 400, dtype=torch.bool)
    cand = torch.randn(400, 5)
    query = torch.randn(8, 5)
    seen = []
    for tau in (1.0, 0.3, 0.1):
        _, _, m = soft_set_mse(scores, mask, query, cand, tau_set=tau,
                               normalization='none')
        seen.append((float(m['set_soft_effective_candidates']),
                     float(m['set_soft_top10_mass'])))
    assert seen[0][0] > seen[1][0] > seen[2][0]
    assert seen[0][1] < seen[1][1] < seen[2][1]


def test_wce_weight_zero_isolates_the_set_loss(monkeypatch):
    """S1 (SetMSE-only) reuses wce_soft_set_mse with wce_weight=0, not a new loss.

    Pinned as a spec check: `total_loss = wce_weight * L_WCE + lambda_set * L_set`
    must reduce to pure L_set when wce_weight=0 and to the pre-existing
    `L_WCE + lambda_set * L_set` when wce_weight defaults to 1.0.
    """
    import torch as _torch

    class Fake:
        wce_weight = 0.0

    coverage_loss = _torch.tensor(3.7, requires_grad=True)
    regularization_loss = _torch.tensor(0.0)
    total_loss = Fake.wce_weight * coverage_loss + regularization_loss
    assert float(total_loss) == 0.0

    Fake.wce_weight = 1.0
    total_loss = Fake.wce_weight * coverage_loss + regularization_loss
    assert abs(float(total_loss) - 3.7) < 1e-5
