import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.RelationStage1 import (
    Model,
    RelationEncoder,
    build_direct_relation_embedding,
    build_relation_encoder_input,
    expected_future_mse_loss,
    prepare_topk_coverage_targets,
    prepare_query_conditioned_rnc_targets,
    relation_bank_collapse_metrics,
    relation_sequence_length,
    relation_variance_covariance_loss,
    transform_relation_history,
    topk_coverage_loss,
    query_conditioned_rnc_loss,
)
from models import RelationStage1 as relation_stage1_module
from utils.relation_graph import load_or_build_relation_graph, relation_graph_enabled


def test_teacher_student_distribution_metrics_identical_and_misaligned():
    valid = torch.tensor([
        [True, True, True, False],
        [True, True, False, False],
    ])
    teacher = torch.tensor([
        [0.7, 0.2, 0.1, 0.0],
        [0.8, 0.2, 0.0, 0.0],
    ])
    identical = relation_stage1_module._teacher_student_distribution_metrics(
        teacher,
        teacher,
        valid,
    )
    assert torch.allclose(
        identical['teacher_student_js_divergence'],
        torch.tensor(0.0),
        atol=1e-7,
    )
    assert torch.allclose(
        identical['teacher_student_total_variation'],
        torch.tensor(0.0),
        atol=1e-7,
    )
    assert torch.allclose(
        identical['teacher_student_hellinger_distance'],
        torch.tensor(0.0),
        atol=1e-7,
    )
    assert torch.allclose(
        identical['teacher_student_probability_cosine'],
        torch.tensor(1.0),
        atol=1e-7,
    )
    assert identical['teacher_student_topk_overlap_at_1'] == 1.0

    reversed_student = torch.tensor([
        [0.1, 0.2, 0.7, 0.0],
        [0.2, 0.8, 0.0, 0.0],
    ])
    misaligned = relation_stage1_module._teacher_student_distribution_metrics(
        teacher,
        reversed_student,
        valid,
    )
    assert misaligned['teacher_student_js_divergence'] > 0.0
    assert misaligned['teacher_student_total_variation'] > 0.0
    assert misaligned['teacher_student_hellinger_distance'] > 0.0
    assert misaligned['teacher_student_probability_cosine'] < 1.0
    assert misaligned['teacher_student_topk_overlap_at_1'] == 0.0
    # k is clipped to each row's valid candidate count, never padded by invalids.
    assert misaligned['teacher_student_topk_overlap_at_5'] == 1.0


def test_ranking_source_metrics_compare_all_student_teacher_oracle_pairs():
    student = torch.arange(12, 0, -1, dtype=torch.float32).unsqueeze(0)
    future_mse = torch.arange(12, dtype=torch.float32).unsqueeze(0)
    future_cosine = torch.arange(12, dtype=torch.float32).unsqueeze(0)
    teacher = future_cosine.clone()
    valid = torch.ones_like(student, dtype=torch.bool)

    metrics = relation_stage1_module._ranking_source_topk_metrics(
        student,
        teacher,
        future_mse,
        future_cosine,
        valid,
    )

    assert metrics['oracle_mse_student_topk_overlap_at_1'] == 1.0
    assert metrics['teacher_oracle_cos_topk_overlap_at_1'] == 1.0
    assert metrics['teacher_student_topk_overlap_at_1'] == 0.0
    assert metrics['teacher_oracle_mse_topk_overlap_at_1'] == 0.0
    assert metrics['oracle_cos_student_topk_overlap_at_1'] == 0.0
    assert metrics['oracle_mse_oracle_cos_topk_overlap_at_1'] == 0.0
    assert torch.allclose(
        metrics['student_oracle_recall_at_10'],
        metrics['oracle_mse_student_topk_overlap_at_10'],
    )
    assert torch.allclose(
        metrics['student_oracle_cos_recall_at_10'],
        metrics['oracle_cos_student_topk_overlap_at_10'],
    )


def test_stage1_oracle_mse_is_branch_specific_target_source_concat():
    model = object.__new__(Model)
    model.teacher_mse_space = 'normalized'
    model.relation_teacher_space = 'absolute'
    query_x = torch.zeros(1, 2, 2)
    query_y = torch.tensor([[
        [1.0, 10.0],
        [1.0, 10.0],
    ]])
    memory_y = torch.tensor([
        [[1.0, 0.0], [1.0, 0.0]],
        [[2.0, 10.0], [2.0, 10.0]],
    ])

    self_mse = model._future_mse(
        query_x, query_y, memory_y, None, 0, 0
    )
    cross_mse = model._future_mse(
        query_x, query_y, memory_y, None, 0, 1
    )

    assert self_mse.argmin(dim=-1).tolist() == [0]
    assert cross_mse.argmin(dim=-1).tolist() == [1]


def test_linear_relation_mode_bypasses_self_and_projects_cross_before_l_to_dff_mlp():
    configs = SimpleNamespace(
        relation_encoder_type='mlp',
        relation_pooling='cls',
        relation_self_fill='linear',
        seq_len=4,
        d_model=8,
        d_ff=16,
        dropout=0.0,
    )
    encoder = RelationEncoder(configs)
    assert encoder.encoder[0].in_features == configs.seq_len

    projection = torch.nn.Linear(2 * configs.seq_len, configs.seq_len)
    x = torch.randn(3, configs.seq_len, 2)
    self_relation = build_relation_encoder_input(
        x,
        target_channel=0,
        source_channel=0,
        shared_cross_projection=projection,
    )
    cross_relation = build_relation_encoder_input(
        x,
        target_channel=0,
        source_channel=1,
        shared_cross_projection=projection,
    )

    expected_cross = projection(torch.cat([x[..., 0], x[..., 1]], dim=-1))
    assert self_relation.shape == (3, 1, configs.seq_len)
    assert cross_relation.shape == (3, 1, configs.seq_len)
    assert torch.allclose(self_relation[:, 0], x[..., 0])
    assert torch.allclose(cross_relation[:, 0], expected_cross)
    assert encoder(self_relation).shape == (3, configs.d_model)


def test_diff1_changes_relation_length_and_values_without_touching_future_space():
    x = torch.tensor([[[10.0], [12.0], [15.0], [19.0]]])
    transformed = transform_relation_history(x, 'diff1')

    assert relation_sequence_length(4, 'diff1') == 3
    torch.testing.assert_close(
        transformed,
        torch.tensor([[[2.0], [3.0], [4.0]]]),
    )


def test_diff1_linear_encoder_and_cross_projection_use_l_minus_one_width():
    configs = SimpleNamespace(
        relation_encoder_type='mlp',
        relation_pooling='cls',
        relation_self_fill='linear',
        relation_input_space='diff1',
        seq_len=4,
        d_model=8,
        d_ff=16,
        dropout=0.0,
    )
    encoder = RelationEncoder(configs)
    projection = torch.nn.Linear(2 * 3, 3)
    x = torch.tensor([[[1.0, 5.0], [2.0, 7.0], [4.0, 8.0], [7.0, 12.0]]])
    cross = build_relation_encoder_input(
        x,
        target_channel=0,
        source_channel=1,
        relation_input_space='diff1',
        shared_cross_projection=projection,
        self_fill='linear',
    )

    assert encoder.encoder[0].in_features == 3
    assert projection.in_features == 6
    assert projection.out_features == 3
    assert cross.shape == (1, 1, 3)
    assert encoder(cross).shape == (1, 8)


def test_diff1_direct_embedding_is_cosine_ready_and_duplicates_self_role():
    x = torch.tensor([[[10.0], [12.0], [15.0], [19.0]]])
    embedding = build_direct_relation_embedding(
        x,
        target_channel=0,
        source_channel=0,
        relation_input_space='diff1',
    )
    expected = torch.nn.functional.normalize(
        torch.tensor([[2.0, 3.0, 4.0, 2.0, 3.0, 4.0]]),
        dim=-1,
    )
    torch.testing.assert_close(embedding, expected)


def test_ema_input_teacher_embeds_diff1_past_and_stays_gradient_free():
    config = _model_config()
    config.relation_input_space = 'diff1'
    config.stage1_teacher_mode = 'ema_input'
    config.pred_len = 2
    model = Model(config)
    memory_x = torch.tensor([
        [[0.0], [1.0], [3.0], [6.0]],
        [[2.0], [3.0], [5.0], [8.0]],
    ])
    query_x = memory_x[:1]
    query_y = torch.zeros(1, config.pred_len, config.enc_in)

    teacher_bank = model.build_teacher_embedding_bank(
        memory_x,
        device=torch.device('cpu'),
    )
    scores = model._teacher_embedding_scores(
        query_x,
        query_y,
        teacher_bank,
        target_channel=0,
        source_channel=0,
    )

    assert teacher_bank.shape == (1, 1, 2, config.d_model)
    assert scores.shape == (1, 2)
    assert all(not parameter.requires_grad for parameter in model.teacher_encoder.parameters())
    assert all(
        not parameter.requires_grad
        for parameter in model.teacher_shared_cross_projection.parameters()
    )


def _explicit_rnc(scores, future_mse, valid_mask, temperature):
    query_losses = []
    for row_scores, row_future, row_valid in zip(scores, future_mse, valid_mask):
        logits = row_scores[row_valid] / temperature
        distances = row_future[row_valid]
        if logits.numel() < 2:
            continue
        denominator_mask = distances[None, :] >= distances[:, None]
        anchor_mask = denominator_mask.sum(dim=1) > 1
        pairwise_logits = logits[None, :].expand(logits.numel(), -1)
        denominator = torch.logsumexp(
            pairwise_logits.masked_fill(~denominator_mask, float('-inf')),
            dim=1,
        )
        query_losses.append((denominator[anchor_mask] - logits[anchor_mask]).mean())
    if not query_losses:
        return scores.sum() * 0.0
    return torch.stack(query_losses).mean()


def _expected_from_logits(logits, future_mse, valid_mask, normalization='mean'):
    masked_logits = logits.masked_fill(~valid_mask, float('-inf'))
    probability = torch.softmax(masked_logits, dim=-1)
    return expected_future_mse_loss(
        probability,
        future_mse,
        valid_mask,
        normalization=normalization,
    )[0]


def _model_config(loss_mode='kl', expected_mse_weight=0.1):
    return SimpleNamespace(
        seq_len=4,
        pred_len=4,
        enc_in=1,
        tau_student=0.2,
        tau_teacher=0.3,
        teacher_mse_space='normalized',
        stage1_teacher_mode='mse',
        relation_input_space='absolute',
        relation_teacher_space='absolute',
        source_mode='all',
        relation_graph_threshold=21,
        target_mode='all',
        target_channel=None,
        stage1_key_chunk_size=16,
        stage1_loss_mode=loss_mode,
        stage1_use_rank_loss=0,
        stage1_rank_weight=0.1,
        stage1_rank_margin=0.1,
        stage1_rank_min_mse_gap=0.0,
        stage1_rank_top_k=2,
        stage1_coverage_top_k=-1,
        top_k=2,
        rnc_temperature=0.2,
        rnc_tie_epsilon=0.0,
        expected_mse_weight=expected_mse_weight,
        stage1_variance_weight=0.0,
        stage1_covariance_weight=0.0,
        stage1_variance_target=1.0,
        expected_mse_normalization='mean',
        relation_encoder_type='mlp',
        relation_pooling='cls',
        relation_self_fill='repeat',
        d_model=8,
        d_ff=16,
        dropout=0.0,
    )


def _model_batch():
    query_x = torch.tensor([
        [[0.0], [0.2], [0.4], [0.6]],
        [[0.1], [0.3], [0.2], [0.5]],
    ])
    query_y = torch.tensor([
        [[0.7], [0.8], [0.9], [1.0]],
        [[0.4], [0.3], [0.5], [0.6]],
    ])
    memory_y = torch.tensor([
        [[0.7], [0.8], [0.9], [1.0]],
        [[0.4], [0.5], [0.5], [0.7]],
        [[-0.4], [-0.2], [0.0], [0.1]],
    ])
    cand_mask = torch.tensor([[True, True, True], [True, True, True]])
    key_bank = torch.nn.functional.normalize(
        torch.tensor([[[
            [1.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0],
        ]]]),
        dim=-1,
    )
    return query_x, query_y, cand_mask, memory_y, key_bank


def test_rnc_order_and_score_direction():
    future = torch.tensor([[0.1, 0.2, 0.4]])
    valid = torch.ones_like(future, dtype=torch.bool)
    aligned = torch.tensor([[1.0, 0.2, -0.7]])
    reversed_scores = torch.flip(aligned, dims=[1])
    aligned_loss, _ = query_conditioned_rnc_loss(aligned, future, valid)
    reversed_loss, _ = query_conditioned_rnc_loss(reversed_scores, future, valid)
    assert aligned_loss < reversed_loss

    base = torch.tensor([[0.3, 0.1, -0.2]])
    better_up = base.clone()
    better_up[0, 0] += 0.5
    worse_up = base.clone()
    worse_up[0, 2] += 0.5
    base_loss, _ = query_conditioned_rnc_loss(base, future, valid)
    better_loss, _ = query_conditioned_rnc_loss(better_up, future, valid)
    worse_loss, _ = query_conditioned_rnc_loss(worse_up, future, valid)
    assert better_loss < base_loss
    assert worse_loss > base_loss


def test_rnc_invalid_candidate_ties_and_single_candidate():
    future = torch.tensor([[0.1, 0.2, 0.2, 0.8], [0.3, 9.0, 9.0, 9.0]])
    valid = torch.tensor([[True, True, True, False], [True, False, False, False]])
    scores = torch.tensor([[0.4, 0.3, 0.2, -10.0], [0.1, 0.0, 0.0, 0.0]])
    loss, _, debug = query_conditioned_rnc_loss(
        scores, future, valid, return_debug=True
    )
    changed = scores.clone()
    changed[~valid] = 1000.0
    changed_loss, _ = query_conditioned_rnc_loss(changed, future, valid)
    assert torch.allclose(loss, changed_loss)
    assert torch.isfinite(loss)
    assert debug[0]['denominator_starts'][1].item() == debug[0]['denominator_starts'][2].item()

    near_tie_future = torch.tensor([[0.1, 0.2, 0.2005, 0.8]])
    _, _, near_tie_debug = query_conditioned_rnc_loss(
        scores[:1],
        near_tie_future,
        torch.ones_like(near_tie_future, dtype=torch.bool),
        tie_epsilon=0.001,
        return_debug=True,
    )
    assert (
        near_tie_debug[0]['denominator_starts'][1].item()
        == near_tie_debug[0]['denominator_starts'][2].item()
    )


def test_rnc_suffix_matches_explicit_pairwise_reference():
    scores = torch.tensor([
        [0.3, -0.1, 0.7, 0.2],
        [-0.4, 0.6, 0.1, 0.0],
    ], requires_grad=True)
    future = torch.tensor([
        [0.4, 0.1, 0.8, 0.2],
        [0.3, 0.3, 0.7, 0.9],
    ])
    valid = torch.tensor([
        [True, True, True, True],
        [True, True, True, False],
    ])
    actual, _ = query_conditioned_rnc_loss(scores, future, valid, temperature=0.4)
    expected = _explicit_rnc(scores, future, valid, temperature=0.4)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    actual.backward()
    assert scores.grad is not None
    assert scores.grad.abs().sum() > 0


def test_rnc_prepared_target_order_matches_direct_loss():
    scores = torch.tensor([
        [0.3, -0.1, 0.7, 0.2],
        [-0.4, 0.6, 0.1, 0.0],
    ])
    future = torch.tensor([
        [0.4, 0.1, 0.8, 0.2],
        [0.3, 0.3, 0.7, 0.9],
    ])
    valid = torch.tensor([
        [True, True, True, True],
        [True, True, True, False],
    ])
    prepared = prepare_query_conditioned_rnc_targets(future, valid)
    direct, _ = query_conditioned_rnc_loss(scores, future, valid, temperature=0.4)
    cached, _ = query_conditioned_rnc_loss(
        scores,
        future,
        valid,
        temperature=0.4,
        prepared_targets=prepared,
    )
    assert torch.allclose(cached, direct, atol=1e-6, rtol=1e-6)


def test_expected_mse_probability_and_logit_directions():
    future = torch.tensor([[0.1, 0.5, 1.0]])
    valid = torch.ones_like(future, dtype=torch.bool)
    good_probability = torch.tensor([[0.8, 0.15, 0.05]])
    bad_probability = torch.tensor([[0.05, 0.15, 0.8]])
    good_loss, _ = expected_future_mse_loss(good_probability, future, valid)
    bad_loss, _ = expected_future_mse_loss(bad_probability, future, valid)
    assert good_loss < bad_loss

    base_logits = torch.tensor([[0.0, 0.0, 0.0]])
    good_up = base_logits.clone()
    good_up[0, 0] += 1.0
    bad_up = base_logits.clone()
    bad_up[0, 2] += 1.0
    assert _expected_from_logits(good_up, future, valid) < _expected_from_logits(
        base_logits, future, valid
    )
    assert _expected_from_logits(bad_up, future, valid) > _expected_from_logits(
        base_logits, future, valid
    )


def test_expected_mse_mask_and_normalizations():
    valid = torch.tensor([[True, True, False]])
    logits = torch.tensor([[0.1, -0.2, -20.0]])
    future = torch.tensor([[0.2, 0.8, 10.0]])
    baseline = _expected_from_logits(logits, future, valid)
    changed_logits = logits.clone()
    changed_logits[0, 2] = 1000.0
    changed_future = future.clone()
    changed_future[0, 2] = 1e9
    changed = _expected_from_logits(changed_logits, changed_future, valid)
    assert torch.allclose(baseline, changed)

    for normalization in ('none', 'mean', 'median'):
        loss = _expected_from_logits(logits, future, valid, normalization)
        assert torch.isfinite(loss)


def test_topk_coverage_oracle_indices_follow_lowest_future_mse():
    future = torch.tensor([[0.4, 0.1, 0.3, 0.2]])
    valid = torch.ones_like(future, dtype=torch.bool)
    targets = prepare_topk_coverage_targets(future, valid, top_k=2)
    assert targets['oracle_indices'].tolist() == [[1, 3]]
    assert torch.allclose(targets['oracle_mse'], torch.tensor([[0.1, 0.2]]))
    assert targets['effective_k'].tolist() == [2]
    assert targets['active_query'].tolist() == [True]


def test_topk_coverage_excludes_invalid_low_mse_candidate():
    future = torch.tensor([[0.0, 0.2, 0.3]])
    valid = torch.tensor([[False, True, True]])
    targets = prepare_topk_coverage_targets(future, valid, top_k=1)
    assert targets['oracle_indices'].item() == 1
    assert targets['oracle_mse'].item() == future[0, 1].item()


def test_topk_coverage_handles_fewer_valid_candidates_than_k():
    future = torch.tensor([[0.5, 0.1, 9.0, 0.3, 8.0]])
    valid = torch.tensor([[True, True, False, True, False]])
    targets = prepare_topk_coverage_targets(future, valid, top_k=10)
    logits = torch.tensor([[0.2, 0.4, -100.0, 0.1, -100.0]])
    log_prob = torch.log_softmax(logits.masked_fill(~valid, -1e4), dim=-1)
    loss, metrics = topk_coverage_loss(log_prob, targets)
    assert torch.isfinite(loss)
    assert targets['effective_k'].item() == 3
    assert metrics['coverage_effective_k'].item() == 3
    assert targets['oracle_valid'].sum().item() == 3


def test_topk_coverage_excludes_query_with_no_valid_candidate():
    future = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.1, 0.2]])
    valid = torch.tensor([
        [False, False, False],
        [True, True, False],
    ])
    targets = prepare_topk_coverage_targets(future, valid, top_k=2)
    logits = torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.5, -100.0]])
    log_prob = torch.log_softmax(logits.masked_fill(~valid, -1e4), dim=-1)
    loss, _ = topk_coverage_loss(log_prob, targets)
    expected = -log_prob[1, :2].mean()
    assert targets['active_query'].tolist() == [False, True]
    assert torch.allclose(loss, expected)


def test_topk_coverage_loss_decreases_when_positive_logits_increase():
    future = torch.tensor([[0.1, 0.2, 0.8, 1.0]])
    valid = torch.ones_like(future, dtype=torch.bool)
    targets = prepare_topk_coverage_targets(future, valid, top_k=2)
    base_logits = torch.zeros_like(future)
    better_logits = base_logits.clone()
    better_logits[0, :2] = 2.0
    base_loss, _ = topk_coverage_loss(
        torch.log_softmax(base_logits, dim=-1), targets
    )
    better_loss, _ = topk_coverage_loss(
        torch.log_softmax(better_logits, dim=-1), targets
    )
    assert better_loss < base_loss


def test_topk_coverage_requires_every_oracle_positive_not_only_set_mass():
    future = torch.tensor([[0.1, 0.2, 0.3, 1.0]])
    valid = torch.ones_like(future, dtype=torch.bool)
    targets = prepare_topk_coverage_targets(future, valid, top_k=3)
    one_positive_only = torch.tensor([[6.0, -6.0, -6.0, 0.0]])
    all_positives = torch.tensor([[6.0, 6.0, 6.0, 0.0]])
    one_loss, _ = topk_coverage_loss(
        torch.log_softmax(one_positive_only, dim=-1), targets
    )
    all_loss, _ = topk_coverage_loss(
        torch.log_softmax(all_positives, dim=-1), targets
    )
    assert all_loss < one_loss


def test_topk_coverage_has_finite_nonzero_gradient():
    future = torch.tensor([[0.1, 0.2, 0.8, 1.0]])
    valid = torch.ones_like(future, dtype=torch.bool)
    targets = prepare_topk_coverage_targets(future, valid, top_k=2)
    logits = torch.tensor([[0.1, -0.2, 0.3, 0.0]], requires_grad=True)
    loss, _ = topk_coverage_loss(torch.log_softmax(logits, dim=-1), targets)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_topk_coverage_reuses_target_indices_across_relations(monkeypatch):
    config = _model_config(loss_mode='topk_coverage')
    config.enc_in = 2
    config.target_mode = 'single'
    config.target_channel = 0
    config.stage1_coverage_top_k = 2
    model = Model(config).eval()

    query_x = torch.tensor([[
        [0.0, 1.0], [0.2, 0.8], [0.4, 0.7], [0.6, 0.5],
    ]])
    query_y = torch.tensor([[
        [0.7, 0.4], [0.8, 0.3], [0.9, 0.2], [1.0, 0.1],
    ]])
    memory_y = torch.tensor([
        [[0.7, 8.0], [0.8, 8.0], [0.9, 8.0], [1.0, 8.0]],
        [[0.5, 0.4], [0.6, 0.3], [0.7, 0.2], [0.8, 0.1]],
        [[-0.5, 0.4], [-0.4, 0.3], [-0.3, 0.2], [-0.2, 0.1]],
    ])
    cand_mask = torch.ones(1, 3, dtype=torch.bool)
    key_bank = torch.nn.functional.normalize(
        torch.randn(2, 2, 3, config.d_model), dim=-1
    )

    target_pointers = []
    original_loss = relation_stage1_module.topk_coverage_loss

    def record_target_pointer(student_log_prob, targets):
        target_pointers.append(targets['oracle_indices'].data_ptr())
        return original_loss(student_log_prob, targets)

    monkeypatch.setattr(
        relation_stage1_module, 'topk_coverage_loss', record_target_pointer
    )
    loss, metrics = model(
        query_x,
        query_y,
        cand_mask,
        memory_y,
        key_bank,
        compute_detailed_metrics=False,
    )
    assert torch.isfinite(loss)
    assert len(target_pointers) == 2
    assert target_pointers[0] == target_pointers[1]
    assert metrics['topk_coverage_loss'] > 0
    assert metrics['kl_loss'] == 0


def test_topk_coverage_does_not_require_or_call_ema_teacher(monkeypatch):
    config = _model_config(loss_mode='topk_coverage')
    config.stage1_teacher_mode = 'ema_target'
    model = Model(config).eval()
    query_x, query_y, cand_mask, memory_y, key_bank = _model_batch()

    def forbidden(*args, **kwargs):
        raise AssertionError('teacher and non-coverage loss paths must not run')

    monkeypatch.setattr(model, '_teacher_logits', forbidden)
    monkeypatch.setattr(model, '_teacher_embedding_scores', forbidden)
    monkeypatch.setattr(model, '_teacher_embedding_logits', forbidden)
    monkeypatch.setattr(relation_stage1_module, 'query_conditioned_rnc_loss', forbidden)
    monkeypatch.setattr(relation_stage1_module, 'expected_future_mse_loss', forbidden)
    monkeypatch.setattr(relation_stage1_module, 'future_aware_topk_ranking_loss', forbidden)

    assert not model.requires_ema_teacher_bank()
    loss, metrics = model(
        query_x,
        query_y,
        cand_mask,
        memory_y,
        key_bank,
        teacher_key_bank=None,
    )
    assert torch.isfinite(loss)
    assert metrics['topk_coverage_loss'] > 0
    assert metrics['kl_loss'] == 0
    assert metrics['rnc_loss'] == 0
    assert metrics['expected_mse_loss'] == 0
    assert 'teacher_entropy' not in metrics


def test_kl_mode_matches_existing_formula_and_zero_weight_expected_mode():
    torch.manual_seed(7)
    model = Model(_model_config(loss_mode='kl')).eval()
    query_x, query_y, cand_mask, memory_y, key_bank = _model_batch()

    loss, _ = model(query_x, query_y, cand_mask, memory_y, key_bank)
    future_logits, _ = model._teacher_logits(
        query_x, query_y, memory_y, None, 0, 0
    )
    teacher_prob = torch.softmax(future_logits, dim=-1)
    query_embedding = model.encoder(model._relation_tensor(query_x, 0, 0))
    scores = query_embedding @ key_bank[0, 0].transpose(0, 1)
    student_log_prob = torch.log_softmax(scores / model.tau_student, dim=-1)
    explicit_kl = (
        teacher_prob * (torch.log(teacher_prob + model.eps) - student_log_prob)
    ).sum(dim=-1).mean()
    assert torch.allclose(loss, explicit_kl, atol=1e-7, rtol=1e-6)

    model.loss_mode = 'kl_expected_mse'
    model.expected_mse_weight = 0.0
    zero_weight_loss, _ = model(query_x, query_y, cand_mask, memory_y, key_bank)
    assert torch.allclose(zero_weight_loss, loss, atol=1e-7, rtol=1e-6)

    model.expected_mse_weight = 0.5
    half_weight_loss, half_weight_metrics = model(
        query_x, query_y, cand_mask, memory_y, key_bank
    )
    expected_half_mix = 0.5 * (
        half_weight_metrics['kl_loss'] + half_weight_metrics['expected_mse_loss']
    )
    assert torch.allclose(half_weight_loss, expected_half_mix, atol=1e-7, rtol=1e-6)


def test_legacy_rank_flag_maps_default_kl_mode_to_kl_rank():
    config = _model_config(loss_mode='kl')
    config.stage1_use_rank_loss = 1
    model = Model(config)
    assert model.loss_mode == 'kl_rank'
    assert model.use_rank_loss


def test_rnc_model_backward_reaches_query_encoder():
    torch.manual_seed(11)
    model = Model(_model_config(loss_mode='rnc')).train()
    query_x, query_y, cand_mask, memory_y, key_bank = _model_batch()
    loss, _ = model(query_x, query_y, cand_mask, memory_y, key_bank)
    loss.backward()
    encoder_grad = sum(
        parameter.grad.abs().sum()
        for parameter in model.encoder.parameters()
        if parameter.grad is not None
    )
    assert encoder_grad > 0


def test_rnc_forward_does_not_require_or_call_teacher_paths(monkeypatch):
    config = _model_config(loss_mode='rnc')
    config.stage1_teacher_mode = 'ema_target'
    model = Model(config).eval()
    query_x, query_y, cand_mask, memory_y, key_bank = _model_batch()

    def forbidden(*args, **kwargs):
        raise AssertionError('teacher, rank, and expected-MSE paths must not run in RnC mode')

    monkeypatch.setattr(model, '_teacher_logits', forbidden)
    monkeypatch.setattr(model, '_teacher_embedding_scores', forbidden)
    monkeypatch.setattr(model, '_teacher_embedding_logits', forbidden)
    monkeypatch.setattr(relation_stage1_module, 'future_aware_topk_ranking_loss', forbidden)
    monkeypatch.setattr(relation_stage1_module, 'expected_future_mse_loss', forbidden)

    loss, metrics = model(
        query_x,
        query_y,
        cand_mask,
        memory_y,
        key_bank,
        teacher_key_bank=None,
    )
    assert torch.isfinite(loss)
    assert metrics['rnc_loss'] > 0
    assert metrics['kl_loss'] == 0
    assert 'teacher_entropy' not in metrics


def test_ema_rnc_uses_raw_future_cosine_order_without_other_losses(monkeypatch):
    config = _model_config(loss_mode='rnc')
    config.stage1_teacher_mode = 'ema_target'
    config.rnc_quality_source = 'ema_cosine'
    model = Model(config).eval()
    query_x, query_y, cand_mask, memory_y, key_bank = _model_batch()
    teacher_key_bank = torch.nn.functional.normalize(key_bank, dim=-1)

    def forbidden(*args, **kwargs):
        raise AssertionError('KL, rank, expected-MSE, and teacher softmax must not run')

    monkeypatch.setattr(model, '_teacher_logits', forbidden)
    monkeypatch.setattr(model, '_teacher_embedding_logits', forbidden)
    monkeypatch.setattr(relation_stage1_module, 'future_aware_topk_ranking_loss', forbidden)
    monkeypatch.setattr(relation_stage1_module, 'expected_future_mse_loss', forbidden)

    loss, metrics = model(
        query_x,
        query_y,
        cand_mask,
        memory_y,
        key_bank,
        teacher_key_bank=teacher_key_bank,
    )
    assert torch.isfinite(loss)
    assert metrics['rnc_loss'] > 0
    assert metrics['kl_loss'] == 0
    assert metrics['expected_mse_loss'] == 0
    assert 'teacher_entropy' not in metrics


def test_ema_teacher_bank_has_one_distribution_key_bank_per_relation():
    config = _model_config(loss_mode='kl')
    config.enc_in = 2
    config.stage1_teacher_mode = 'ema_target'
    model = Model(config).eval()
    memory_y = torch.tensor([
        [[0.0, 1.0], [0.1, 0.9], [0.2, 0.7], [0.3, 0.4]],
        [[0.3, 0.2], [0.4, 0.4], [0.5, 0.6], [0.6, 0.8]],
        [[0.8, 0.3], [0.7, 0.2], [0.5, 0.1], [0.2, 0.0]],
    ])

    teacher_key_bank = model.build_teacher_embedding_bank(memory_y, 'cpu')

    assert teacher_key_bank.shape == (2, 2, 3, config.d_model)


def test_ema_forward_builds_a_teacher_distribution_for_each_relation(monkeypatch):
    config = _model_config(loss_mode='kl')
    config.enc_in = 2
    config.target_mode = 'single'
    config.target_channel = 0
    config.stage1_teacher_mode = 'ema_target'
    model = Model(config).eval()
    query_x = torch.randn(2, config.seq_len, config.enc_in)
    query_y = torch.randn(2, config.pred_len, config.enc_in)
    memory_y = torch.randn(3, config.pred_len, config.enc_in)
    cand_mask = torch.ones(2, 3, dtype=torch.bool)
    key_bank = torch.nn.functional.normalize(
        torch.randn(2, 2, 3, config.d_model), dim=-1
    )
    teacher_key_bank = torch.nn.functional.normalize(
        torch.randn(2, 2, 3, config.d_model), dim=-1
    )
    calls = []

    def relation_teacher_logits(
        query_x_arg,
        query_y_arg,
        teacher_key_bank_arg,
        target_channel,
        source_channel,
        source_slot,
    ):
        calls.append((target_channel, source_channel, source_slot))
        logits = query_x_arg.new_zeros(query_x_arg.size(0), cand_mask.size(1))
        logits[:, source_slot] = 1.0
        return logits

    monkeypatch.setattr(model, '_teacher_embedding_logits', relation_teacher_logits)
    loss, _ = model(
        query_x,
        query_y,
        cand_mask,
        memory_y,
        key_bank,
        teacher_key_bank=teacher_key_bank,
        compute_detailed_metrics=False,
    )

    assert torch.isfinite(loss)
    assert calls == [(0, 0, 0), (0, 1, 1)]


def test_relation_bank_collapse_metrics_separate_collapsed_and_diverse_banks():
    collapsed = torch.ones(1, 1, 8, 4)
    diverse = torch.eye(4).repeat(2, 1).reshape(1, 1, 8, 4)

    collapsed_metrics = relation_bank_collapse_metrics(
        collapsed,
        sample_size=8,
    )
    diverse_metrics = relation_bank_collapse_metrics(
        diverse,
        sample_size=8,
    )

    assert torch.allclose(
        collapsed_metrics['pairwise_cosine_mean'],
        torch.tensor(1.0),
    )
    assert collapsed_metrics['embedding_variance_mean'] == 0
    assert collapsed_metrics['dead_dimension_fraction_mean'] == 1
    assert collapsed_metrics['effective_rank_mean'] == 0
    assert diverse_metrics['pairwise_cosine_mean'] < 1
    assert diverse_metrics['embedding_variance_mean'] > 0
    assert diverse_metrics['dead_dimension_fraction_mean'] == 0
    assert diverse_metrics['effective_rank_mean'] > 1


def test_relation_variance_covariance_loss_penalizes_collapsed_embeddings():
    collapsed = torch.ones(8, 4, requires_grad=True)
    diverse = torch.eye(4).repeat(2, 1)

    collapsed_variance, collapsed_covariance, collapsed_std = (
        relation_variance_covariance_loss(collapsed)
    )
    diverse_variance, _, diverse_std = relation_variance_covariance_loss(
        diverse,
        variance_target=0.4,
    )

    assert collapsed_variance > diverse_variance
    assert collapsed_covariance == 0
    assert collapsed_std < diverse_std
    collapsed_variance.backward()
    assert collapsed.grad is not None


def test_kl_loss_includes_variance_and_covariance_regularization():
    config = _model_config(loss_mode='kl')
    config.stage1_variance_weight = 1.0
    config.stage1_covariance_weight = 0.01
    model = Model(config)
    query_x, query_y, cand_mask, memory_y, key_bank = _model_batch()

    loss, metrics = model(
        query_x,
        query_y,
        cand_mask,
        memory_y,
        key_bank,
        compute_detailed_metrics=False,
    )

    expected = (
        metrics['stage1_loss_kl']
        + metrics['stage1_loss_variance_weighted']
        + metrics['stage1_loss_covariance_weighted']
    )
    assert torch.allclose(loss.detach(), expected)
    assert metrics['stage1_loss_variance'] >= 0
    assert metrics['stage1_loss_covariance'] >= 0


def test_auto_relation_graph_uses_self_plus_top_two_for_small_channel_count(tmp_path):
    base = np.arange(12, dtype=np.float64)
    train_dataset = SimpleNamespace(
        data_x=np.stack([
            base,
            2.0 * base,
            -base,
            np.tile([0.0, 1.0, -1.0], 4),
        ], axis=1),
        channel_names=['target', 'positive', 'negative', 'weak'],
    )
    args = SimpleNamespace(
        enc_in=4,
        source_mode='auto',
        relation_top_n=3,
        relation_graph_threshold=21,
        relation_graph_path='',
        metrics_csv_dir=str(tmp_path),
        data_path='small.csv',
    )

    assert relation_graph_enabled(args)
    graph = load_or_build_relation_graph(train_dataset, args)

    assert graph['top_n'] == 3
    assert graph['sources'][0] == [0, 1, 2]
    assert all(len(sources) == 3 for sources in graph['sources'])
    assert all(sources[0] == target for target, sources in enumerate(graph['sources']))
