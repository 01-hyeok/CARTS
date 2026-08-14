import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.RelationStage1 import (
    Model,
    RelationEncoder,
    build_direct_relation_embedding,
    build_relation_encoder_input,
    relation_feature_count,
    relation_sequence_length,
    transform_relation_features,
    transform_relation_history,
)


def _model_config(**overrides):
    base = dict(
        seq_len=8,
        pred_len=8,
        enc_in=1,
        tau_student=0.2,
        tau_teacher=0.3,
        teacher_mse_space='normalized',
        stage1_teacher_mode='mse',
        relation_input_space='delta_last_diff1',
        relation_teacher_space='absolute',
        source_mode='all',
        relation_graph_threshold=21,
        target_mode='all',
        target_channel=None,
        stage1_key_chunk_size=16,
        stage1_loss_mode='kl',
        stage1_use_rank_loss=0,
        stage1_rank_weight=0.1,
        stage1_rank_margin=0.1,
        stage1_rank_min_mse_gap=0.0,
        stage1_rank_top_k=2,
        stage1_coverage_top_k=-1,
        top_k=2,
        rnc_temperature=0.2,
        rnc_tie_epsilon=0.0,
        expected_mse_weight=0.0,
        stage1_variance_weight=0.0,
        stage1_covariance_weight=0.0,
        stage1_variance_target=1.0,
        expected_mse_normalization='mean',
        relation_encoder_type='tcn',
        relation_pooling='last',
        relation_self_fill='linear',
        relation_tcn_layers=2,
        relation_tcn_kernel_size=3,
        relation_tcn_channels=0,
        relation_tcn_dropout=-1.0,
        d_model=8,
        d_ff=16,
        dropout=0.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _configs(**overrides):
    base = dict(
        relation_encoder_type='tcn',
        relation_pooling='last',
        relation_self_fill='linear',
        relation_input_space='delta_last_diff1',
        relation_tcn_layers=2,
        relation_tcn_kernel_size=3,
        relation_tcn_channels=0,
        relation_tcn_dropout=-1.0,
        seq_len=8,
        d_model=8,
        d_ff=16,
        n_heads=2,
        e_layers=1,
        patch_len=4,
        stride=4,
        dropout=0.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_two_channel_space_stacks_delta_last_and_diff1_on_a_shared_length():
    x = torch.tensor([[[10.0], [12.0], [15.0], [19.0]]])
    views = transform_relation_features(x, 'delta_last_diff1')

    assert relation_feature_count('delta_last_diff1') == 2
    # diff1 is the shorter view (L-1), so delta_last keeps its trailing L-1 steps.
    assert relation_sequence_length(4, 'delta_last_diff1') == 3
    assert len(views) == 2
    torch.testing.assert_close(views[0], torch.tensor([[[-7.0], [-4.0], [0.0]]]))
    torch.testing.assert_close(views[1], torch.tensor([[[2.0], [3.0], [4.0]]]))


def test_single_feature_spaces_are_unchanged_by_the_feature_refactor():
    x = torch.randn(3, 6, 2)
    for space in ('absolute', 'delta_last', 'diff1'):
        views = transform_relation_features(x, space)
        assert len(views) == 1
        torch.testing.assert_close(views[0], transform_relation_history(x, space))


def test_transform_relation_history_rejects_multi_feature_spaces():
    with pytest.raises(ValueError, match='transform_relation_features'):
        transform_relation_history(torch.randn(2, 5, 1), 'delta_last_diff1')


def test_linear_relation_input_keeps_feature_rows_and_shares_the_2l_to_l_projection():
    configs = _configs()
    relation_len = relation_sequence_length(configs.seq_len, configs.relation_input_space)
    encoder = RelationEncoder(configs)
    projection = torch.nn.Linear(2 * relation_len, relation_len)
    x = torch.randn(4, configs.seq_len, 2)

    self_relation = build_relation_encoder_input(
        x, 0, 0,
        relation_input_space=configs.relation_input_space,
        shared_cross_projection=projection,
        self_fill='linear',
    )
    cross_relation = build_relation_encoder_input(
        x, 0, 1,
        relation_input_space=configs.relation_input_space,
        shared_cross_projection=projection,
        self_fill='linear',
    )

    views = transform_relation_features(x, configs.relation_input_space)
    expected_self = torch.stack([views[0][..., 0], views[1][..., 0]], dim=1)
    # The same 2L -> L weights are applied to each feature row.
    expected_cross = torch.stack([
        projection(torch.cat([view[..., 0], view[..., 1]], dim=-1)) for view in views
    ], dim=1)

    assert self_relation.shape == (4, 2, relation_len)
    assert cross_relation.shape == (4, 2, relation_len)
    torch.testing.assert_close(self_relation, expected_self)
    torch.testing.assert_close(cross_relation, expected_cross)
    assert encoder.encoder.in_channels == 2
    assert encoder(self_relation).shape == (4, configs.d_model)
    assert encoder(cross_relation).shape == (4, configs.d_model)


def test_repeat_self_fill_doubles_rows_per_feature_role_major():
    configs = _configs(relation_self_fill='repeat')
    relation_len = relation_sequence_length(configs.seq_len, configs.relation_input_space)
    encoder = RelationEncoder(configs)
    x = torch.randn(2, configs.seq_len, 2)

    cross = build_relation_encoder_input(
        x, 0, 1,
        relation_input_space=configs.relation_input_space,
        self_fill='repeat',
    )
    views = transform_relation_features(x, configs.relation_input_space)

    assert cross.shape == (2, 4, relation_len)
    # Role-major: target's two features first, then the source's two.
    torch.testing.assert_close(cross[:, 0], views[0][..., 0])
    torch.testing.assert_close(cross[:, 1], views[1][..., 0])
    torch.testing.assert_close(cross[:, 2], views[0][..., 1])
    torch.testing.assert_close(cross[:, 3], views[1][..., 1])

    self_relation = build_relation_encoder_input(
        x, 0, 0,
        relation_input_space=configs.relation_input_space,
        self_fill='repeat',
    )
    assert self_relation.shape == (2, 2, relation_len)
    rows = encoder._prepare_rows(self_relation)
    assert rows.shape == (2, 4, relation_len)
    torch.testing.assert_close(rows[:, :2], rows[:, 2:])
    assert encoder.encoder.in_channels == 4
    assert encoder(self_relation).shape == (2, configs.d_model)


def test_mlp_encoder_accepts_the_same_two_channel_input():
    configs = _configs(relation_encoder_type='mlp', relation_pooling='cls')
    relation_len = relation_sequence_length(configs.seq_len, configs.relation_input_space)
    encoder = RelationEncoder(configs)
    x = torch.randn(2, configs.seq_len, 2)

    # The MLP flattens both feature rows, so the 2-channel input can be ablated
    # against the TCN with the encoder held fixed.
    assert encoder.encoder[0].in_features == 2 * relation_len
    self_relation = build_relation_encoder_input(
        x, 0, 0,
        relation_input_space=configs.relation_input_space,
        self_fill='linear',
    )
    assert encoder(self_relation).shape == (2, configs.d_model)


def test_tcn_is_causal_so_later_steps_cannot_change_earlier_activations():
    configs = _configs(relation_self_fill='linear')
    encoder = RelationEncoder(configs).eval()
    relation_len = relation_sequence_length(configs.seq_len, configs.relation_input_space)

    a = torch.randn(1, 2, relation_len)
    b = a.clone()
    b[..., -1] += 5.0
    with torch.no_grad():
        blocks = encoder.encoder.blocks
        ha, hb = blocks(a), blocks(b)
    # Only the final position may react to a change at the final input step.
    torch.testing.assert_close(ha[..., :-1], hb[..., :-1])
    assert not torch.allclose(ha[..., -1], hb[..., -1])


def test_tcn_receptive_field_matches_the_dilation_schedule():
    encoder = RelationEncoder(_configs(relation_tcn_layers=4, relation_tcn_kernel_size=3))
    # 2 convs per block, dilations 1,2,4,8: 1 + 2*(3-1)*(2^4-1) = 61
    assert encoder.encoder.receptive_field == 61


def test_direct_embedding_concatenates_both_features_per_role():
    x = torch.tensor([[[10.0], [12.0], [15.0], [19.0]]])
    embedding = build_direct_relation_embedding(
        x, 0, 0, relation_input_space='delta_last_diff1',
    )
    expected = torch.nn.functional.normalize(
        torch.tensor([[-7.0, -4.0, 0.0, 2.0, 3.0, 4.0, -7.0, -4.0, 0.0, 2.0, 3.0, 4.0]]),
        dim=-1,
    )
    torch.testing.assert_close(embedding, expected)


def test_transformer_encoder_rejects_multi_feature_input_space():
    with pytest.raises(ValueError, match='transformer only supports single-feature'):
        RelationEncoder(_configs(relation_encoder_type='transformer', relation_pooling='cls'))


def test_tcn_encoder_rejects_cls_pooling():
    with pytest.raises(ValueError, match='relation_pooling for tcn'):
        RelationEncoder(_configs(relation_pooling='cls'))


def test_stage1_model_trains_end_to_end_with_the_tcn_two_channel_encoder():
    torch.manual_seed(0)
    config = _model_config()
    model = Model(config)
    query_x = torch.randn(2, config.seq_len, config.enc_in)
    query_y = torch.randn(2, config.pred_len, config.enc_in)
    memory_y = torch.randn(3, config.pred_len, config.enc_in)
    memory_x_last = torch.randn(3, config.enc_in)
    cand_mask = torch.ones(2, 3, dtype=torch.bool)

    key_bank = model.build_embedding_bank(
        torch.randn(3, config.seq_len, config.enc_in),
        device=torch.device('cpu'),
    )
    assert key_bank.shape == (config.enc_in, 1, 3, config.d_model)

    loss, metrics = model(
        query_x, query_y, cand_mask, memory_y, key_bank,
        memory_x_last=memory_x_last,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert metrics
    grads = [
        parameter.grad
        for parameter in model.encoder.encoder.parameters()
        if parameter.grad is not None
    ]
    assert grads and any(grad.abs().sum() > 0 for grad in grads)


def test_ema_input_teacher_supports_the_two_channel_space_but_ema_target_does_not():
    model = Model(_model_config(stage1_teacher_mode='ema_input'))
    teacher_bank = model.build_teacher_embedding_bank(
        torch.randn(3, 8, 1), device=torch.device('cpu'),
    )
    assert teacher_bank.shape == (1, 1, 3, 8)

    with pytest.raises(ValueError, match='ema_target'):
        Model(_model_config(stage1_teacher_mode='ema_target'))


def test_encoder_rejects_a_row_count_that_is_not_a_role_multiple_of_the_features():
    encoder = RelationEncoder(_configs(relation_self_fill='repeat'))
    relation_len = relation_sequence_length(8, 'delta_last_diff1')
    with pytest.raises(ValueError, match='relation row count must be 2'):
        encoder(torch.randn(2, 3, relation_len))
