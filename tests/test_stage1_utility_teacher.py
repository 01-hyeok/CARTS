"""The teacher swap must change the target and nothing else.

Two failure modes matter here. One is silent: an arm that quietly ranks a
different candidate pool than its comparison arm, which would make "teacher
effect" and "subset effect" the same number. The other is scale: utility and
Future-MSE differ by an order of magnitude, so a shared tau would compare
sharpness rather than targets unless the scores are normalized first.
"""

import pytest
import torch

from models.RelationStage1 import normalize_teacher_scores, utility_teacher_loss


def _batch(bsz=4, pool=6, seed=0):
    torch.manual_seed(seed)
    return {
        'student_scores': torch.randn(bsz, pool, requires_grad=True),
        'teacher_scores': torch.randn(bsz, pool),
        'utility': torch.randn(bsz, pool) * 0.1,
        'valid_mask': torch.ones(bsz, pool, dtype=torch.bool),
    }


def _loss(batch, **kwargs):
    options = dict(tau_student=0.1, tau_teacher=0.05)
    options.update(kwargs)
    return utility_teacher_loss(**batch, **options)


def test_scale_normalization_equalizes_teacher_sharpness():
    """Utility values are ~10x smaller than Future-MSE gaps, so an unnormalized
    shared tau would make the utility teacher flat by accident."""
    valid = torch.ones(3, 20, dtype=torch.bool)
    torch.manual_seed(1)
    small = torch.randn(3, 20) * 0.01
    large = small * 100.0

    def entropy(scores):
        probability = torch.softmax(
            normalize_teacher_scores(scores, valid, 'per_query_scale') / 0.05, dim=-1)
        return -(probability * probability.log()).sum(-1)

    torch.testing.assert_close(entropy(small), entropy(large))
    raw = lambda s: -(torch.softmax(s / 0.05, -1) * torch.softmax(s / 0.05, -1).log()).sum(-1)
    assert (raw(small) - raw(large)).abs().max() > 0.5


def test_normalization_preserves_the_sign_of_utility():
    """The NULL action's score is a utility of exactly zero, so a normalization
    that recentred the scores would destroy what abstention means."""
    valid = torch.tensor([[True, True, True, False]])
    scores = torch.tensor([[-2.0, 0.5, 3.0, 99.0]])
    out = normalize_teacher_scores(scores, valid, 'per_query_scale')
    assert torch.sign(out[0, :3]).tolist() == [-1.0, 1.0, 1.0]


def test_kl_and_expected_utility_both_reach_the_encoder():
    for objective in ('kl', 'expected_utility'):
        batch = _batch()
        loss, metrics = _loss(batch, objective=objective)
        loss.backward()
        assert torch.isfinite(loss)
        assert batch['student_scores'].grad.abs().sum() > 0
        assert metrics['utility_teacher_loss'].isfinite()


def test_expected_utility_prefers_the_useful_candidate():
    """Its whole point: putting mass on a candidate with positive measured gain
    must lower the loss relative to spreading mass onto a harmful one."""
    utility = torch.tensor([[0.5, -0.5]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    good = utility_teacher_loss(
        torch.tensor([[3.0, -3.0]]), utility, utility, valid,
        tau_student=1.0, tau_teacher=1.0, objective='expected_utility')[0]
    bad = utility_teacher_loss(
        torch.tensor([[-3.0, 3.0]]), utility, utility, valid,
        tau_student=1.0, tau_teacher=1.0, objective='expected_utility')[0]
    assert good < bad


def test_null_action_absorbs_mass_when_every_candidate_is_harmful():
    """A softmax retriever without NULL must always pick someone; with NULL the
    teacher should put most of its mass on abstaining."""
    utility = torch.tensor([[-0.4, -0.3, -0.5]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    _, metrics = utility_teacher_loss(
        torch.zeros(1, 3), utility, utility, valid,
        tau_student=0.1, tau_teacher=0.05, objective='kl',
        null_logit=torch.zeros(1, 1))
    assert float(metrics['utility_teacher_null_probability']) > 0.9


def test_null_head_gradient_flows_from_the_loss():
    head = torch.nn.Linear(5, 1)
    z_q = torch.randn(4, 5)
    batch = _batch()
    loss, _ = _loss(batch, objective='kl', null_logit=head(z_q))
    loss.backward()
    assert head.weight.grad.abs().sum() > 0


def test_mismatched_teacher_shape_is_rejected():
    batch = _batch()
    batch['utility'] = batch['utility'][:, :-1]
    with pytest.raises(ValueError, match='utility shape'):
        _loss(batch)


def test_unknown_objective_and_normalization_are_rejected():
    with pytest.raises(ValueError, match='Unsupported teacher objective'):
        _loss(_batch(), objective='hinge')
    with pytest.raises(ValueError, match='Unsupported teacher normalization'):
        normalize_teacher_scores(torch.zeros(1, 3), torch.ones(1, 3, dtype=torch.bool), 'zscore')
