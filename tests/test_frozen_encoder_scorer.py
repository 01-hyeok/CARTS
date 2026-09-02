"""Freezing the encoder so the ranking objective can be judged on its own.

Rank-only fine-tuning drove effective rank from 16 to 1 within ten steps, which
leaves two readings open: the supervision is unusable, or updating the encoder
with it was. Holding the encoder fixed and training only the scorer separates
them -- but only if the scorer starts exactly at cosine and the encoder really
does not move, which is what these pin.
"""

import torch

from layers.retrieval_metric import RetrievalMetric, cosine_init_deviation


def _metric():
    # output=cosine with no layer norm is the configuration whose identity
    # initialisation is exactly cosine; the defaults are not.
    return RetrievalMetric(kind='asymmetric', dim=16, scaled_dot=True,
                           layer_norm=False, output='cosine')


def test_asymmetric_starts_exactly_at_cosine():
    dev = cosine_init_deviation(_metric(), samples=64,
                                generator=torch.Generator().manual_seed(0))
    assert float(dev) == 0.0, 'a step-0 difference would be the initialisation'


def test_scorer_has_trainable_parameters_and_they_move_the_score():
    metric = _metric()
    params = [p for p in metric.parameters() if p.requires_grad]
    assert params
    torch.manual_seed(0)
    z_q, z_k = torch.randn(4, 16), torch.randn(20, 16)
    before = metric.score(z_q, z_k)
    with torch.no_grad():
        params[0].add_(0.1 * torch.randn_like(params[0]))
    assert not torch.allclose(before, metric.score(z_q, z_k))


def test_freeze_leaves_only_the_metric_trainable():
    from models.RelationStage1 import Model
    from tests.test_stage1_full_memory_metric import _config

    cfg = _config()
    cfg.stage1_retrieval_metric = 'asymmetric'
    cfg.stage1_metric_output = 'cosine'
    cfg.stage1_metric_layer_norm = 0
    cfg.stage1_freeze_encoder = 1
    model = Model(cfg)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable, 'nothing left to train'
    assert not any(n.startswith('encoder.') for n in trainable)
    assert any(n.startswith('retrieval_metric.') for n in trainable)


def test_freezing_without_a_metric_is_rejected_not_silently_a_no_op():
    from models.RelationStage1 import Model
    from tests.test_stage1_full_memory_metric import _config
    import pytest

    cfg = _config()
    cfg.stage1_retrieval_metric = 'cosine'      # no learnable scorer exists
    cfg.stage1_freeze_encoder = 1
    with pytest.raises(ValueError):
        Model(cfg)
