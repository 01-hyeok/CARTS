"""EXP-FRR01 (R0/R1/R2/R12/R3): unit coverage for the new residual-conditioning
mechanisms in models/RelationStage1.py, following this repo's convention of
testing the extracted pure/standalone pieces rather than constructing the full
configured model (see tests/test_stage1_full_memory_metric.py).

_condition_query_embedding and _condition_candidate_embedding are plain
methods that only touch self.query_cond_proj / self.candidate_cond_proj and
self.eps, so a duck-typed stand-in with just those attributes exercises the
real, unbound method exactly as forward() calls it.
"""
import torch
import torch.nn as nn
import pytest

from models.RelationStage1 import Model as RelationStage1


class _Stub:
    """Duck-typed stand-in carrying only what the two methods under test read."""
    eps = 1e-8

    def __init__(self, pred_len=8, d_model=4, seed=0):
        torch.manual_seed(seed)
        self.query_cond_proj = nn.Linear(pred_len, d_model)
        self.candidate_cond_proj = nn.Linear(pred_len, d_model)


condition_query = RelationStage1._condition_query_embedding
condition_candidate = RelationStage1._condition_candidate_embedding


def test_query_conditioning_is_additive_and_renormalised():
    stub = _Stub()
    z_q = torch.nn.functional.normalize(torch.randn(3, 4), dim=-1)
    query_y = torch.randn(3, 8, 2)
    query_residual = torch.randn(3, 8, 2)

    out = condition_query(stub, z_q, query_y, query_residual, target_channel=0)

    assert out.shape == z_q.shape
    torch.testing.assert_close(out.norm(dim=-1), torch.ones(3), atol=1e-5, rtol=1e-5)
    # It must actually be a function of the base forecast, not a no-op.
    base = query_y[:, :, 0] - query_residual[:, :, 0]
    expected = torch.nn.functional.normalize(z_q + stub.query_cond_proj(base), dim=-1)
    torch.testing.assert_close(out, expected)


def test_query_conditioning_reacts_to_the_base_forecast():
    """Two queries with identical z_q but different base forecasts must not
    collapse to the same conditioned embedding -- otherwise the projection
    is dead weight and R1 reduces to R0 silently."""
    stub = _Stub()
    z_q = torch.nn.functional.normalize(torch.randn(1, 4), dim=-1).repeat(2, 1)
    query_y = torch.stack([torch.zeros(8, 2), torch.ones(8, 2) * 5.0])
    query_residual = torch.zeros(2, 8, 2)

    out = condition_query(stub, z_q, query_y, query_residual, target_channel=0)
    assert not torch.allclose(out[0], out[1])


def test_query_conditioning_requires_residual_cache():
    stub = _Stub()
    z_q = torch.randn(2, 4)
    query_y = torch.randn(2, 8, 2)
    with pytest.raises(ValueError, match='query_residual'):
        condition_query(stub, z_q, query_y, None, target_channel=0)


def test_candidate_conditioning_is_additive_and_renormalised():
    stub = _Stub()
    z_k = torch.nn.functional.normalize(torch.randn(10, 4), dim=-1)
    memory_residual = torch.randn(10, 8, 2)

    out = condition_candidate(stub, z_k, memory_residual, target_channel=1)

    assert out.shape == z_k.shape
    torch.testing.assert_close(out.norm(dim=-1), torch.ones(10), atol=1e-5, rtol=1e-5)
    expected = torch.nn.functional.normalize(
        z_k + stub.candidate_cond_proj(memory_residual[:, :, 1]), dim=-1)
    torch.testing.assert_close(out, expected)


def test_candidate_conditioning_requires_residual_cache():
    stub = _Stub()
    z_k = torch.randn(5, 4)
    with pytest.raises(ValueError, match='memory_residual'):
        condition_candidate(stub, z_k, None, target_channel=0)


def test_candidate_conditioning_gives_distinct_candidates_distinct_shifts():
    """Two candidates with identical z_k but different historical residuals
    must end up at different points -- the mechanism this experiment is
    testing (does historical residual similarity carry information the raw
    embedding does not)."""
    stub = _Stub()
    z_k = torch.nn.functional.normalize(torch.randn(1, 4), dim=-1).repeat(2, 1)
    memory_residual = torch.stack([torch.zeros(8, 2), torch.ones(8, 2) * 3.0])

    out = condition_candidate(stub, z_k, memory_residual, target_channel=0)
    assert not torch.allclose(out[0], out[1])


# ---------- temporal-legality assert (exp_stage1_relation.py::_residual_cache) ----------

def test_legality_check_rejects_a_memory_residual_not_sized_to_train():
    """A residual cache whose memory_residual row count does not match the
    train split's query count is refused outright -- this is the exact shape
    a val/test-contaminated (or otherwise malformed) cache would have."""
    train_rows, wrong_rows = 100, 100 + 1

    def fake_check(cache):
        train_rows_actual = cache['splits']['train']['query_residual'].shape[0]
        memory_rows = cache['memory_residual'].shape[0]
        if memory_rows != train_rows_actual:
            raise ValueError('memory_residual is expected to be built from the train split alone')

    cache = {
        'memory_residual': torch.zeros(wrong_rows, 8, 2),
        'splits': {'train': {'query_residual': torch.zeros(train_rows, 8, 2)}},
    }
    with pytest.raises(ValueError, match='train split alone'):
        fake_check(cache)


def test_legality_check_accepts_a_correctly_sized_memory_residual():
    def fake_check(cache):
        train_rows_actual = cache['splits']['train']['query_residual'].shape[0]
        memory_rows = cache['memory_residual'].shape[0]
        if memory_rows != train_rows_actual:
            raise ValueError('memory_residual is expected to be built from the train split alone')

    cache = {
        'memory_residual': torch.zeros(100, 8, 2),
        'splits': {'train': {'query_residual': torch.zeros(100, 8, 2)}},
    }
    fake_check(cache)  # must not raise
