"""Regression guard for the candidate-utility helpers on RelationStage2.

The earlier residual diagnostics were invalidated by an offset bug: analysis code
restated the delta_last convention by hand and got it wrong in two places at
once. These tests exist so that never silently repeats -- the helpers are pinned
to the production `forward`, and the two specific ways of getting the offset
wrong are asserted to be *detectably* different rather than close enough to pass
unnoticed.
"""

from types import SimpleNamespace

import pytest
import torch

from models.RelationStage2 import Model

CONFIG = dict(
    seq_len=16, pred_len=8, enc_in=3, d_model=16,
    top_k=2, tau_topk=0.1, memory_chunk_size=64,
    source_mode='auto', target_mode='all', target_channel=0, relation_top_n=1,
    fusion_mode='residual', gate_mode='scalar', gate_hidden=16,
    base_head_mode='per_channel_linear', fixed_lambda=-1.0,
    relation_mixer_input='retrieved', relation_mixer_hidden=16,
    relation_value_space='delta_last', relation_input_space='absolute',
    retrieval_similarity='cosine', retrieval_soft_all=0,
    retrieval_kl_weight=0.0, retrieval_kl_teacher='future_mse',
    tau_student=0.1, tau_teacher=0.1,
    freeze_stage1_encoder=1, freeze_stage='stage2', disable_retrieval=0,
    retrieval_backbone='identity', stage2_retrieval_backbone='identity',
    stage2_relation_fusion='gate', relation_graph_threshold=0.0,
)


@pytest.fixture
def fixture():
    torch.manual_seed(0)
    model = Model(SimpleNamespace(**CONFIG)).eval()
    # self-only protocol: each target channel retrieves from itself
    model.relation_sources = [[c] for c in range(CONFIG['enc_in'])]
    channels, horizon = CONFIG['enc_in'], CONFIG['pred_len']
    batch_x = torch.randn(4, CONFIG['seq_len'], channels)
    batch_y = torch.randn(4, horizon, channels)
    memory_x_last = torch.randn(12, channels)
    memory_y = torch.randn(12, horizon, channels)
    cache = {
        'relation_outputs': torch.randn(4, channels, model.num_source_slots(), horizon),
        'relation_query_embs': torch.zeros(4, channels, model.num_source_slots(), 0),
    }
    kwargs = dict(
        memory_y=memory_y, valid_mask=torch.ones(4, 12, dtype=torch.bool),
        key_bank=None, memory_x_last=memory_x_last,
    )
    return model, batch_x, batch_y, cache, kwargs


def _final(model, batch_x, cache, kwargs):
    with torch.no_grad():
        return model(batch_x=batch_x, retrieval_cache=cache, **kwargs)[0]


def test_helper_reproduces_production_forward_exactly(fixture):
    """Feeding the cache's own values back through the helper must be a no-op."""
    model, batch_x, _, cache, kwargs = fixture
    with torch.no_grad():
        helper = model.forward_from_retrieval_values(
            cache['relation_outputs'], batch_x=batch_x, retrieval_cache=cache, **kwargs
        )[0]
    torch.testing.assert_close(helper, _final(model, batch_x, cache, kwargs), rtol=0, atol=0)


def test_double_offset_is_detectably_wrong(fixture):
    """Restoring the offset a second time -- the exact bug that invalidated the
    residual diagnostics -- must change the answer, not hide inside tolerance."""
    model, batch_x, batch_y, cache, kwargs = fixture
    correct = _final(model, batch_x, cache, kwargs)
    doubled = correct + batch_x[:, -1:, :]
    assert not torch.allclose(doubled, correct, atol=1e-4)
    assert (doubled - batch_y).square().mean() != (correct - batch_y).square().mean()


def test_missing_offset_is_detectably_wrong(fixture):
    """Calling base_head alone leaves the forecast in delta space."""
    model, batch_x, _, cache, kwargs = fixture
    with torch.no_grad():
        raw = model.base_head(batch_x)
        restored = model(batch_x=batch_x, retrieval_cache=cache, **kwargs)[1]
    assert not torch.allclose(raw, restored, atol=1e-4)
    torch.testing.assert_close(raw + batch_x[:, -1:, :], restored)


def test_wrong_relation_output_shape_is_rejected(fixture):
    model, batch_x, _, cache, kwargs = fixture
    with pytest.raises(ValueError, match='relation_outputs shape'):
        model.forward_from_retrieval_values(
            cache['relation_outputs'][:, :, :, :-1],
            batch_x=batch_x, retrieval_cache=cache, **kwargs,
        )


def test_utility_never_broadcasts_query_against_candidate(fixture):
    """[B, K, C] and [B, C] must stay distinct; a silent broadcast between them
    is how a per-channel tensor turns into a plausible-looking wrong number."""
    model, batch_x, batch_y, cache, kwargs = fixture
    indices = torch.tensor([0, 3, 7, 9, 11])
    utility, base_mse = model.evaluate_candidate_correction(
        batch_x=batch_x, batch_y=batch_y, candidate_indices=indices,
        retrieval_cache=cache, candidate_chunk=2, **kwargs
    )
    assert utility.shape == (batch_x.size(0), indices.numel(), CONFIG['enc_in'])
    assert base_mse.shape == (batch_x.size(0), CONFIG['enc_in'])
    with pytest.raises(RuntimeError, match='must match'):
        utility - base_mse


def test_utility_equals_base_minus_single_candidate_forward(fixture):
    """The reported utility must be reproducible by running forward on one
    candidate by hand -- utility is a measurement of the real path, not algebra."""
    model, batch_x, batch_y, cache, kwargs = fixture
    indices = torch.tensor([2, 5])
    utility, base_mse = model.evaluate_candidate_correction(
        batch_x=batch_x, batch_y=batch_y, candidate_indices=indices,
        retrieval_cache=cache, candidate_chunk=1, **kwargs
    )
    values = model.relation_values_for_candidates(
        batch_x, kwargs['memory_y'], kwargs['memory_x_last'], indices
    )
    for position in range(indices.numel()):
        branch = (
            values[:, :, position, :]
            .unsqueeze(2).expand(-1, -1, model.num_source_slots(), -1)
            .contiguous()
        )
        with torch.no_grad():
            y_final, y_base = model.forward_from_retrieval_values(
                branch, batch_x=batch_x, retrieval_cache=cache, **kwargs
            )[:2]
        assert y_final.shape == batch_y.shape
        expected = ((y_base - batch_y).square().mean(1)
                    - (y_final - batch_y).square().mean(1))
        torch.testing.assert_close(utility[:, position], expected)
        torch.testing.assert_close(base_mse, (y_base - batch_y).square().mean(1))


def test_per_query_and_per_channel_pools_select_their_own_candidates(fixture):
    """Pools are mined per query and per channel, so a shared-pool result must be
    reproducible by spelling the same pool out in the wider index shapes."""
    model, batch_x, batch_y, cache, kwargs = fixture
    shared = torch.tensor([1, 6, 8])
    reference = model.evaluate_candidate_correction(
        batch_x=batch_x, batch_y=batch_y, candidate_indices=shared,
        retrieval_cache=cache, candidate_chunk=2, **kwargs
    )[0]
    per_query = shared.unsqueeze(0).expand(batch_x.size(0), -1)
    per_channel = per_query.unsqueeze(1).expand(-1, CONFIG['enc_in'], -1)
    for spelling in (per_query, per_channel):
        torch.testing.assert_close(
            model.evaluate_candidate_correction(
                batch_x=batch_x, batch_y=batch_y, candidate_indices=spelling,
                retrieval_cache=cache, candidate_chunk=2, **kwargs
            )[0],
            reference, rtol=0, atol=0,
        )

    # A genuinely per-channel pool must differ from the shared one, or the extra
    # index dimension is being silently dropped.
    varied = per_channel.clone()
    varied[:, 0, 0] = 4
    varied[:, 1, 2] = 10
    assert not torch.allclose(
        model.evaluate_candidate_correction(
            batch_x=batch_x, batch_y=batch_y, candidate_indices=varied,
            retrieval_cache=cache, candidate_chunk=2, **kwargs
        )[0],
        reference,
    )


def test_candidate_index_shape_is_validated(fixture):
    model, batch_x, batch_y, cache, kwargs = fixture
    bad = torch.zeros(batch_x.size(0) + 1, 3, dtype=torch.long)
    with pytest.raises(ValueError, match='candidate_indices batch'):
        model.evaluate_candidate_correction(
            batch_x=batch_x, batch_y=batch_y, candidate_indices=bad,
            retrieval_cache=cache, **kwargs
        )


def test_single_candidate_set_equals_injected_candidate(fixture):
    """With top_k=1 the model's own retrieval collapses to one candidate, so the
    set-level number and the injected-candidate number must be the same forecast.

    This is the join between the two measurement paths in the bottleneck study:
    if they ever disagree here, one of them is not the production path.
    """
    model, batch_x, batch_y, cache, kwargs = fixture
    model.top_k = 1
    chosen = torch.tensor([[5]] * batch_x.size(0))

    values = model.relation_values_for_candidates(
        batch_x, kwargs['memory_y'], kwargs['memory_x_last'], chosen
    )
    branch = values[:, :, 0, :].unsqueeze(2).expand(-1, -1, model.num_source_slots(), -1)
    with torch.no_grad():
        injected = model.forward_from_retrieval_values(
            branch.contiguous(), batch_x=batch_x, retrieval_cache=cache, **kwargs
        )[0]

    utility, base_mse = model.evaluate_candidate_correction(
        batch_x=batch_x, batch_y=batch_y, candidate_indices=chosen,
        retrieval_cache=cache, candidate_chunk=1, **kwargs
    )
    set_utility = base_mse - (injected - batch_y).square().mean(dim=1)
    torch.testing.assert_close(utility[:, 0], set_utility)
