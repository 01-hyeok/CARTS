import torch
import torch.nn as nn

from models.RelationStage2 import Model


def _oracle_model(top_k=2, tau_topk=1.0, value_space='absolute'):
    model = object.__new__(Model)
    nn.Module.__init__(model)
    model.top_k = top_k
    model.tau_topk = tau_topk
    model.relation_value_space = value_space
    return model


def test_candidate_oracle_selects_ground_truth_topk_from_full_memory():
    model = _oracle_model(top_k=2)
    memory_values = torch.tensor([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
    ])
    target = torch.tensor([[1.4, 1.4]])
    valid_mask = torch.tensor([[True, True, True, True]])

    indices, valid, values = model._oracle_topk_candidates(
        memory_value_c=memory_values,
        valid_mask=valid_mask,
        oracle_target_c=target,
        query_offset=torch.zeros(1),
    )

    assert indices.tolist() == [[1, 2]]
    assert valid.tolist() == [[True, True]]
    torch.testing.assert_close(values, memory_values[indices])


def test_candidate_oracle_keeps_k_candidates_and_uses_encoder_alpha():
    model = _oracle_model(top_k=2, tau_topk=1.0)
    oracle_idx = torch.tensor([[1, 2]])
    oracle_valid = torch.tensor([[True, True]])
    oracle_values = torch.tensor([[[1.0, 1.0], [2.0, 2.0]]])
    scores = torch.tensor([[10.0, 0.0, 1.0, -10.0]])

    retrieved = model._weight_oracle_candidates(
        scores=scores,
        oracle_idx=oracle_idx,
        oracle_valid=oracle_valid,
        oracle_values=oracle_values,
    )

    alpha = torch.softmax(torch.tensor([[0.0, 1.0]]), dim=-1)
    expected = (alpha.unsqueeze(-1) * oracle_values).sum(dim=1)
    torch.testing.assert_close(retrieved, expected)


def test_candidate_oracle_respects_valid_mask_and_delta_space():
    model = _oracle_model(top_k=2, value_space='delta_last')
    memory_deltas = torch.tensor([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
    ])
    query_offset = torch.tensor([10.0])
    absolute_target = torch.tensor([[11.2, 11.2]])
    valid_mask = torch.tensor([[True, False, True]])

    indices, valid, _ = model._oracle_topk_candidates(
        memory_value_c=memory_deltas,
        valid_mask=valid_mask,
        oracle_target_c=absolute_target,
        query_offset=query_offset,
    )

    assert indices.tolist() == [[2, 0]]
    assert valid.tolist() == [[True, True]]
