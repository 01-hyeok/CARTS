import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.RelationStage1 import (
    Model,
    expected_future_mse_loss,
    prepare_query_conditioned_rnc_targets,
    query_conditioned_rnc_loss,
)
from models import RelationStage1 as relation_stage1_module


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
        top_k=2,
        rnc_temperature=0.2,
        rnc_tie_epsilon=0.0,
        expected_mse_weight=expected_mse_weight,
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


def test_kl_mode_matches_existing_formula_and_zero_weight_expected_mode():
    torch.manual_seed(7)
    model = Model(_model_config(loss_mode='kl')).eval()
    query_x, query_y, cand_mask, memory_y, key_bank = _model_batch()

    loss, _ = model(query_x, query_y, cand_mask, memory_y, key_bank)
    future_logits, _ = model._teacher_logits(query_x, query_y, memory_y, None, 0)
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
    teacher_key_bank = torch.nn.functional.normalize(key_bank[:, 0], dim=-1)

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
