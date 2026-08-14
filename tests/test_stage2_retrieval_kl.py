from types import SimpleNamespace

import torch
import torch.nn as nn

from models.RelationStage2 import Model
from utils.retrieval_ops import retrieve_relation_future


def _kl_model(tau_teacher=0.1, tau_student=0.1, weight=1.0):
    model = object.__new__(Model)
    nn.Module.__init__(model)
    model.tau_teacher = tau_teacher
    model.tau_student = tau_student
    model.retrieval_kl_weight = weight
    return model


def test_retrieval_kl_reaches_candidates_outside_student_topk():
    """The whole reason the KL exists: Top-K is not differentiable, so the
    forecasting loss can only reweight already-selected candidates. The KL has
    to push on the ones the student ranked out."""
    model = _kl_model()
    # Candidate 3 is the teacher's best future but the student scores it worst.
    student_scores = torch.tensor([[0.9, 0.8, 0.7, -0.9]], requires_grad=True)
    future_mse = torch.tensor([[5.0, 4.0, 3.0, 0.01]])
    valid_mask = torch.ones(1, 4, dtype=torch.bool)

    _, _, top_idx, _, _ = retrieve_relation_future(
        z_q=torch.eye(1, 4),
        z_mem=torch.eye(4),
        memory_value_c=torch.zeros(4, 2),
        valid_mask=valid_mask,
        top_k=2,
        tau_topk=0.1,
    )
    del top_idx

    kl = model._retrieval_kl(student_scores, future_mse, valid_mask)
    kl.sum().backward()

    grad = student_scores.grad[0]
    # Index 3 sits outside any Top-2 the student would pick, yet must move.
    assert grad[3] != 0.0
    # Teacher favours it, so the KL pushes its score up.
    assert grad[3] < 0.0
    assert torch.isfinite(grad).all()


def test_retrieval_kl_ignores_invalid_candidates():
    model = _kl_model()
    student_scores = torch.tensor([[0.5, 0.4, 0.3]], requires_grad=True)
    future_mse = torch.tensor([[1.0, 2.0, float('inf')]])
    valid_mask = torch.tensor([[True, True, False]])

    kl = model._retrieval_kl(student_scores, future_mse, valid_mask)
    kl.sum().backward()

    assert torch.isfinite(kl).all()
    assert student_scores.grad[0, 2].abs() < 1e-6


def test_retrieval_kl_is_zero_when_student_matches_teacher():
    model = _kl_model(tau_teacher=1.0, tau_student=1.0)
    future_mse = torch.tensor([[1.0, 2.0, 3.0]])
    valid_mask = torch.ones(1, 3, dtype=torch.bool)
    # student logits == teacher logits => identical distributions
    student_scores = -future_mse.clone().requires_grad_(True)

    kl = model._retrieval_kl(student_scores, future_mse, valid_mask)

    torch.testing.assert_close(kl, torch.zeros_like(kl), atol=1e-6, rtol=0)


def test_relation_future_mse_masks_invalid_and_keeps_all_candidates():
    model = object.__new__(Model)
    nn.Module.__init__(model)
    model.relation_value_space = 'absolute'
    model.pred_len = 2
    model._memory_value = lambda x, my, mxl, c: (my[..., c], torch.zeros(x.size(0)))

    batch_x = torch.zeros(1, 3, 2)
    memory_y = torch.tensor([
        [[0.0, 0.0], [0.0, 0.0]],
        [[9.0, 9.0], [9.0, 9.0]],
    ]).transpose(1, 2)
    target_y = torch.zeros(1, 2, 2)
    valid_mask = torch.tensor([[True, False]])

    mse = model._relation_future_mse(
        batch_x, memory_y, None, valid_mask, target_y, 0, 1
    )

    assert mse.shape == (1, 2)
    assert torch.isfinite(mse[0, 0])
    assert torch.isinf(mse[0, 1])
