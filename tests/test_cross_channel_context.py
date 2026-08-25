"""Checks for the cross-channel context arms.

The comparison these tests protect is D vs B: cross-channel ResSel against
cross-channel ResDirect. That only means anything if both arms see the same
source channels, neither sees a query future, and the target-only arms are
genuinely the same model with the mixer removed.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.cross_channel_context import (
    ContextEncoder, CrossChannelMixer, build_source_index,
)
from models.utility_pair_scorer import UtilityPairScorer
from scripts.train_cross_channel_resdirect import CrossChannelResDirect
from scripts.train_cross_channel_ressel import CrossChannelSelector
from utils.cross_channel_setup import encode_candidates
from utils.utility_selection import forecast_from_selection

CHANNELS = 7
SEQ_LEN = 16
PRED_LEN = 8
SOURCE_INDEX = torch.tensor(
    [[(c + offset) % CHANNELS for offset in range(1, 4)] for c in range(CHANNELS)]
)


def _encoder(use_context=True, scale_init=1e-2):
    torch.manual_seed(0)
    return ContextEncoder(
        SEQ_LEN, SOURCE_INDEX, d_model=8, d_ff=16, dropout=0.0,
        use_cross_channel_context=use_context, scale_init=scale_init,
    )


def test_context_off_is_exactly_target_only():
    """(1) use_cross_channel_context=0 reproduces the plain encoder."""
    encoder = _encoder(use_context=False)
    x = torch.randn(4, SEQ_LEN, CHANNELS)
    for channel in range(CHANNELS):
        expected = encoder.encoder(x[:, :, channel])
        assert torch.equal(encoder(x, channel), expected)
    assert encoder.mixer is None


def test_context_off_ignores_other_channels():
    """(1b) with the mixer off, other channels cannot reach the output."""
    encoder = _encoder(use_context=False)
    x = torch.randn(4, SEQ_LEN, CHANNELS)
    polluted = x.clone()
    polluted[:, :, 1:] = torch.randn_like(polluted[:, :, 1:])
    assert torch.equal(encoder(x, 0), encoder(polluted, 0))


def test_source_index_excludes_self_and_uses_train_split(tmp_path):
    """(2) + (3) self is never a source, and only the train split is read."""
    generator = torch.Generator().manual_seed(0)
    series = torch.randn(400, CHANNELS, generator=generator)
    # Channel 0 is built to correlate with 3, which must therefore be its top source.
    series[:, 3] = series[:, 0] * 0.95 + 0.05 * torch.randn(400, generator=generator)

    class _Dataset:
        data_x = series.numpy()
        channel_names = [f'ch{i}' for i in range(CHANNELS)]

    class _Experiment:
        def _get_data(self, flag, shuffle=False):
            assert flag == 'train', f'source selection read the {flag} split'
            return _Dataset(), None

    saved = SimpleNamespace(enc_in=CHANNELS, data_path='ETTh1.csv')
    index, correlations, names = build_source_index(
        _Experiment(), saved, topk=5, metrics_root=str(tmp_path)
    )
    assert index.shape == (CHANNELS, 5)
    for target in range(CHANNELS):
        assert target not in index[target].tolist()
    assert int(index[0, 0]) == 3
    assert correlations.shape == index.shape
    assert len(names) == CHANNELS


def test_source_index_is_a_buffer_that_survives_a_round_trip(tmp_path):
    """(10) source metadata travels with the checkpoint."""
    model = CrossChannelResDirect(
        SEQ_LEN, PRED_LEN, SOURCE_INDEX, d_model=8, d_ff=16, hidden=8,
        dropout=0.0, use_cross_channel_context=True,
    )
    path = tmp_path / 'arm.pt'
    torch.save({'state_dict': model.state_dict(), 'source_index': SOURCE_INDEX}, path)
    saved = torch.load(path)
    assert torch.equal(saved['source_index'], SOURCE_INDEX)
    assert torch.equal(saved['state_dict']['context.source_index'], SOURCE_INDEX)

    restored = CrossChannelResDirect(
        SEQ_LEN, PRED_LEN, torch.zeros_like(SOURCE_INDEX), d_model=8, d_ff=16,
        hidden=8, dropout=0.0, use_cross_channel_context=True,
    )
    restored.load_state_dict(saved['state_dict'])
    assert torch.equal(restored.context.source_index, SOURCE_INDEX)


def test_attention_shape_and_normalisation():
    """(4) attention is [B, K] and sums to one over the sources."""
    mixer = CrossChannelMixer(d_model=8, scale_init=1e-2)
    z_target = torch.randn(5, 8)
    z_sources = torch.randn(5, 3, 8)
    z_ctx, attention = mixer(z_target, z_sources, return_attention=True)
    assert z_ctx.shape == (5, 8)
    assert attention.shape == (5, 3)
    assert torch.allclose(attention.sum(-1), torch.ones(5), atol=1e-6)


def test_gamma_init_and_near_identity_start():
    """(5) gamma starts at the requested value, so the arm starts at baseline."""
    mixer = CrossChannelMixer(d_model=8, scale_init=1e-2)
    assert mixer.gamma.requires_grad
    assert float(mixer.gamma) == pytest.approx(1e-2)

    encoder = _encoder(use_context=True, scale_init=0.0)
    x = torch.randn(4, SEQ_LEN, CHANNELS)
    # gamma=0 is the exact target-only limit of the same weights.
    assert torch.allclose(encoder(x, 0), encoder.encoder(x[:, :, 0]), atol=1e-6)

    channel_wise = CrossChannelMixer(8, 1e-2, channel_wise_scale=True,
                                     num_targets=CHANNELS)
    assert channel_wise.gamma.shape == (CHANNELS,)


def test_candidate_encoder_matches_plain_encoder_when_context_off():
    """(6) candidate-side context off == the target-only candidate encoder."""
    encoder = _encoder(use_context=True)
    memory_x = torch.randn(20, SEQ_LEN, CHANNELS)
    index = torch.tensor([[0, 5, 5, 9], [3, 3, 1, 0]])
    got = encode_candidates(encoder, memory_x, index, channel=2,
                            candidate_context=False)
    expected = encoder.encoder(memory_x[index][:, :, :, 2])
    assert got.shape == (2, 4, 8)
    assert torch.allclose(got, expected, atol=1e-6)

    with_context = encode_candidates(encoder, memory_x, index, channel=2,
                                     candidate_context=True)
    assert not torch.allclose(with_context, expected, atol=1e-6)


def test_candidate_dedup_matches_one_by_one_encoding():
    """The unique-index shortcut must not change any embedding."""
    encoder = _encoder(use_context=False)
    memory_x = torch.randn(12, SEQ_LEN, CHANNELS)
    index = torch.tensor([[4, 4, 7], [7, 1, 4]])
    got = encode_candidates(encoder, memory_x, index, channel=1)
    for row in range(index.size(0)):
        for slot in range(index.size(1)):
            one = encoder.encoder(memory_x[index[row, slot]][:, 1].unsqueeze(0))
            assert torch.allclose(got[row, slot], one.squeeze(0), atol=1e-6)


def test_no_query_future_reaches_the_model():
    """(7) forwards take pasts only; a poisoned future cannot change them."""
    torch.manual_seed(0)
    direct = CrossChannelResDirect(
        SEQ_LEN, PRED_LEN, SOURCE_INDEX, d_model=8, d_ff=16, hidden=8,
        dropout=0.0, use_cross_channel_context=True,
    )
    selector = CrossChannelSelector(
        SEQ_LEN, SOURCE_INDEX, horizon=0, d_model=8, d_ff=16, hidden=8,
        dropout=0.0, use_cross_channel_context=True,
    )
    x = torch.randn(4, SEQ_LEN, CHANNELS)
    memory_x = torch.randn(10, SEQ_LEN, CHANNELS)
    index = torch.tensor([[0, 1, 2]] * 4)
    # Neither signature accepts a future; these are the complete input sets.
    assert torch.isfinite(direct(x)).all()
    assert torch.isfinite(selector(x, memory_x, index, channel=0)).all()

    import inspect
    for signature in (inspect.signature(direct.forward),
                      inspect.signature(selector.forward)):
        names = set(signature.parameters)
        assert not names & {'y', 'query_y', 'future', 'query_residual'}


def test_invalid_candidates_never_enter_the_forecast():
    """(8) the temporal validity mask still gates selection."""
    torch.manual_seed(0)
    n_query, n_memory, horizon, pool = 3, 6, PRED_LEN, 4
    data = {
        'query_base': torch.zeros(n_query, horizon, CHANNELS),
        'memory_residual': torch.zeros(n_memory, horizon, CHANNELS),
    }
    data['memory_residual'][2] = 99.0          # the poisoned, invalid candidate
    data['memory_residual'][5] = 1.0
    cache = {
        'targets': [0],
        'pool_idx': torch.tensor([[[2, 5, 0, 1]]] * n_query),
        'valid': torch.tensor([[[False, True, True, True]]] * n_query),
        'utility': torch.zeros(n_query, 1, pool),
        'alpha': 1.0,
    }
    # The invalid slot is given the winning score on purpose.
    scores = torch.tensor([[[10.0, 1.0, 0.0, -1.0]]] * n_query)

    class _Experiment:
        device = torch.device('cpu')

    prediction = forecast_from_selection(_Experiment(), data, cache, scores, 1)
    assert torch.equal(
        prediction[:, :, 0], torch.ones(n_query, horizon)
    ), 'an invalid candidate was selected'


def test_resdirect_and_ressel_share_encoder_shape_and_sources():
    """(9) both arms are the same encoder over the same source set."""
    torch.manual_seed(0)
    direct = CrossChannelResDirect(
        SEQ_LEN, PRED_LEN, SOURCE_INDEX, d_model=8, d_ff=16, hidden=8,
        dropout=0.0, use_cross_channel_context=True,
    )
    torch.manual_seed(0)
    selector = CrossChannelSelector(
        SEQ_LEN, SOURCE_INDEX, horizon=0, d_model=8, d_ff=16, hidden=8,
        dropout=0.0, use_cross_channel_context=True,
    )
    assert torch.equal(direct.context.source_index, selector.context.source_index)
    direct_encoder = dict(direct.context.encoder.named_parameters())
    selector_encoder = dict(selector.context.encoder.named_parameters())
    assert direct_encoder.keys() == selector_encoder.keys()
    for name, parameter in direct_encoder.items():
        assert parameter.shape == selector_encoder[name].shape
        assert torch.equal(parameter, selector_encoder[name])
    # The context stack is identical; only the heads may differ in size.
    context_params = lambda m: sum(p.numel() for p in m.context.parameters())
    assert context_params(direct) == context_params(selector)


def test_candidate_cross_channel_requires_the_mixer():
    with pytest.raises(ValueError):
        CrossChannelSelector(
            SEQ_LEN, SOURCE_INDEX, use_cross_channel_context=False,
            candidate_cross_channel=True,
        )


def test_residual_branch_is_only_built_when_asked():
    plain = UtilityPairScorer(dim=8, horizon=0)
    assert plain.residual_proj is None
    with_residual = UtilityPairScorer(dim=8, horizon=PRED_LEN)
    z_q = torch.randn(2, 1, 8)
    z_k = torch.randn(2, 5, 8)
    assert with_residual(z_q, z_k, torch.randn(2, 5, PRED_LEN)).shape == (2, 5)
    with pytest.raises(ValueError):
        with_residual(z_q, z_k)
