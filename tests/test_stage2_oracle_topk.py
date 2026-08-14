from types import SimpleNamespace

import torch
import torch.nn as nn

from exp.exp_stage2_relation import Exp_Stage2_Relation
from layers.relation_mixer import RelationMixer
from models.RelationStage2 import Model


def _oracle_model(top_k=2, tau_topk=1.0, value_space='absolute'):
    model = object.__new__(Model)
    nn.Module.__init__(model)
    model.top_k = top_k
    model.tau_topk = tau_topk
    model.relation_value_space = value_space
    return model


def test_identity_retrieval_uses_raw_target_source_relation_without_encoder():
    model = _oracle_model()
    model.retrieval_backbone = 'identity'
    model.relation_input_space = 'absolute'
    model.stage1_encoder = None
    history = torch.tensor([[
        [3.0, 0.0],
        [4.0, 0.0],
    ]])

    self_embedding = model._branch_embedding(history, 0, 0)
    cross_embedding = model._branch_embedding(history, 0, 1)

    torch.testing.assert_close(
        self_embedding,
        torch.tensor([[0.42426407, 0.56568545, 0.42426407, 0.56568545]]),
    )
    torch.testing.assert_close(
        cross_embedding,
        torch.tensor([[0.6, 0.8, 0.0, 0.0]]),
    )
    assert model.stage1_encoder is None


def test_identity_retrieval_uses_diff1_for_direct_baseline():
    model = _oracle_model()
    model.retrieval_backbone = 'identity'
    model.relation_input_space = 'diff1'
    model.retrieval_similarity = 'cosine'
    model.stage1_encoder = None
    history = torch.tensor([[[10.0], [12.0], [15.0], [19.0]]])

    embedding = model._branch_embedding(history, 0, 0)
    expected = torch.nn.functional.normalize(
        torch.tensor([[2.0, 3.0, 4.0, 2.0, 3.0, 4.0]]),
        dim=-1,
    )
    torch.testing.assert_close(embedding, expected)


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

    indices, valid, values, mse = model._oracle_topk_candidates(
        memory_value_c=memory_values,
        valid_mask=valid_mask,
        oracle_target_c=target,
        query_offset=torch.zeros(1),
    )

    assert indices.tolist() == [[1, 2]]
    assert valid.tolist() == [[True, True]]
    torch.testing.assert_close(values, memory_values[indices])
    torch.testing.assert_close(mse, torch.tensor([[0.16, 0.36]]))


def test_candidate_oracle_keeps_k_candidates_and_uses_encoder_alpha():
    model = _oracle_model(top_k=2, tau_topk=1.0)
    oracle_idx = torch.tensor([[1, 2]])
    oracle_valid = torch.tensor([[True, True]])
    oracle_values = torch.tensor([[[1.0, 1.0], [2.0, 2.0]]])
    scores = torch.tensor([[10.0, 0.0, 1.0, -10.0]])

    retrieved, alpha = model._weight_oracle_candidates(
        scores=scores,
        oracle_idx=oracle_idx,
        oracle_valid=oracle_valid,
        oracle_values=oracle_values,
    )

    expected_alpha = torch.softmax(torch.tensor([[0.0, 1.0]]), dim=-1)
    expected = (expected_alpha.unsqueeze(-1) * oracle_values).sum(dim=1)
    torch.testing.assert_close(alpha, expected_alpha)
    torch.testing.assert_close(retrieved, expected)


def test_full_oracle_uses_negative_future_mse_alpha():
    model = _oracle_model(top_k=2, tau_topk=0.5)
    oracle_valid = torch.tensor([[True, True]])
    oracle_values = torch.tensor([[[1.0, 1.0], [3.0, 3.0]]])
    oracle_mse = torch.tensor([[0.25, 1.0]])

    retrieved, alpha = model._weight_full_oracle_candidates(
        oracle_mse=oracle_mse,
        oracle_valid=oracle_valid,
        oracle_values=oracle_values,
    )

    expected_alpha = torch.softmax(-oracle_mse / 0.5, dim=-1)
    expected = (expected_alpha.unsqueeze(-1) * oracle_values).sum(dim=1)
    torch.testing.assert_close(alpha, expected_alpha)
    torch.testing.assert_close(retrieved, expected)


def test_identical_oracle_branches_produce_uniform_mixer_weights():
    mixer = RelationMixer(pred_len=2, hidden_dim=4, input_mode='retrieved')
    retrieved = torch.tensor([[1.25, 2.5], [3.0, 4.0]])
    relation_outputs = retrieved.unsqueeze(1).expand(-1, 3, -1)

    mixed, beta, scores = mixer(relation_outputs)

    torch.testing.assert_close(mixed, retrieved)
    torch.testing.assert_close(beta, torch.full_like(beta, 1.0 / 3.0))
    torch.testing.assert_close(scores, scores[:, :1].expand_as(scores))


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

    indices, valid, _, _ = model._oracle_topk_candidates(
        memory_value_c=memory_deltas,
        valid_mask=valid_mask,
        oracle_target_c=absolute_target,
        query_offset=query_offset,
    )

    assert indices.tolist() == [[2, 0]]
    assert valid.tolist() == [[True, True]]


def test_relation_oracle_topk_uses_concatenated_target_source_futures():
    model = _oracle_model(top_k=1, value_space='absolute')
    batch_x = torch.zeros(1, 2, 2)
    oracle_target_y = torch.tensor([[
        [1.0, 10.0],
        [1.0, 10.0],
    ]])
    memory_y = torch.tensor([
        [[1.0, 0.0], [1.0, 0.0]],
        [[2.0, 10.0], [2.0, 10.0]],
    ])
    valid_mask = torch.ones(1, 2, dtype=torch.bool)

    self_indices, _, _, _ = model._relation_oracle_topk_candidates(
        batch_x=batch_x,
        memory_y=memory_y,
        memory_x_last=None,
        valid_mask=valid_mask,
        oracle_target_y=oracle_target_y,
        target_channel=0,
        source_channel=0,
    )
    cross_indices, _, _, _ = model._relation_oracle_topk_candidates(
        batch_x=batch_x,
        memory_y=memory_y,
        memory_x_last=None,
        valid_mask=valid_mask,
        oracle_target_y=oracle_target_y,
        target_channel=0,
        source_channel=1,
    )

    assert self_indices.tolist() == [[0]]
    assert cross_indices.tolist() == [[1]]


def test_oracle_evaluation_cache_allows_frozen_relation_graph_backbone_on_test():
    exp = object.__new__(Exp_Stage2_Relation)
    exp.args = SimpleNamespace(
        disable_retrieval=0,
        freeze_stage1_encoder=1,
        oracle_candidate_eval=1,
        source_mode='auto',
        stage2_retrieval_backbone='stage1',
        stage2_oracle_train_mode='none',
    )

    assert not exp._use_retrieval_cache()
    assert exp._use_oracle_evaluation_cache('test')
    assert exp._use_retrieval_cache_for_split('test')
    assert not exp._use_retrieval_cache_for_split('train')


def test_no_retrieval_source_indices_do_not_require_relation_graph():
    model = _oracle_model()
    model.channels = 3
    model.disable_retrieval = True
    model.relation_sources = None

    indices = model.source_index_tensor(torch.device('cpu'))

    torch.testing.assert_close(
        indices,
        torch.tensor([
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
        ]),
    )


def test_no_retrieval_checkpoint_has_no_active_relation_order():
    exp = object.__new__(Exp_Stage2_Relation)
    exp.args = SimpleNamespace(disable_retrieval=1)

    assert exp._active_relation_order() is None
