"""Independent mathematical and leakage gates; all fixtures synthetic, CPU-only."""
import numpy as np
import pytest
import torch
from diagnostic_core import *


def fixture():
    torch.manual_seed(1)
    d = torch.rand(7, 100, dtype=torch.float64)
    s = torch.randn(7, 100, dtype=torch.float64)
    return s, d


def test_exchange_identity_and_missed_counts():
    s, d = fixture();m = support_metrics(s, d)
    for q in range(7):
        picked = set(m['selected'][q].tolist());oracle = set(m['oracle'][q].tolist())
        independent = (sum(float(d[q, i]) for i in picked-oracle)-
                       sum(float(d[q, i]) for i in oracle-picked))/10
        assert abs(independent-float(m['ranking_regret'][q])) < 1e-12
        assert len(picked-oracle) == len(oracle-picked)
    assert torch.equal(m['boundary_error'], 1-m['recall'])


def test_large_float32_ties_and_small_exchange_accumulate_in_double():
    d=torch.full((4,100),120.56980895996094,dtype=torch.float32)
    d[1]+=torch.arange(100,dtype=torch.float32)*1e-5
    d[2]*=1e4;d[3]=torch.arange(100,dtype=torch.float32)+1e6
    torch.manual_seed(143);score=torch.randn_like(d)
    m=support_metrics(score,d)
    assert m['retmse'].dtype==torch.float64
    assert float(m['exchange_regret'][0])==float(m['ranking_regret'][0])==0
    torch.testing.assert_close(m['exchange_regret'],m['ranking_regret'],atol=1e-9,rtol=1e-12)


def test_oracle_optimal_ret_and_ndcg():
    _, d = fixture();m = support_metrics(-d, d)
    assert torch.equal(m['ranking_regret'], torch.zeros(7, dtype=torch.float64))
    assert torch.equal(m['ndcg_inverse_mse'], torch.ones(7, dtype=torch.float64))


def test_equal_distance_ndcg_and_stable_ties():
    d = torch.ones(2, 100);s = torch.zeros_like(d)
    m = support_metrics(s, d)
    assert torch.equal(m['selected'], torch.arange(10).expand(2, 10))
    assert torch.equal(m['ndcg_inverse_mse'], torch.ones(2))
    assert torch.equal(m['oracle_boundary_gap'], torch.zeros(2))


def test_mask_and_unique_ids():
    s, d = fixture();mask = torch.ones_like(s, dtype=torch.bool);mask[:, :17] = False
    ids = torch.arange(100).expand(7, 100)
    p, actual = selected_ids(s, ids, mask)
    assert (actual >= 17).all()
    assert all(len(set(x)) == 10 for x in actual.tolist())
    assert (support_metrics(s, d, mask)['ranking_regret'] >= 0).all()


def test_insufficient_support_fails():
    with pytest.raises(AssertionError):
        selected_ids(torch.ones(1, 9), torch.arange(9)[None], torch.ones(1, 9, dtype=torch.bool))


def test_duplicate_ids_fail():
    with pytest.raises(AssertionError):
        selected_ids(torch.ones(1, 100), torch.zeros(1, 100, dtype=torch.long), torch.ones(1, 100, dtype=torch.bool))


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_nonfinite_fail(bad):
    s, d = fixture();s[0, 0] = bad
    with pytest.raises(AssertionError):support_metrics(s, d)


def test_p100_regret_and_aggregate_share():
    m = regret_decomposition([1, 1], [1, 5], [2, 6], 'P100')
    assert m['pool_share'] == pytest.approx(4/6)
    assert m['ranking_share'] == pytest.approx(2/6)
    assert np.array_equal(m['pool_regret']+m['ranking_regret'], m['total_regret'])


def test_full_selection_may_beat_p100_oracle():
    m = regret_decomposition([1], [5], [2], 'FULL')
    assert m['full_model_regret'][0] == 1
    assert 'ranking_regret' not in m


def test_wrong_p100_inequality_fails():
    with pytest.raises(AssertionError):regret_decomposition([1], [5], [2], 'P100')


def test_zero_regret_share_undefined():
    m = regret_decomposition([0], [0], [0], 'P100')
    assert m['pool_share'] is None and m['ranking_share'] is None


@pytest.mark.parametrize('chunk', [1, 7, 23, 100])
def test_stream_global_normalization(chunk):
    torch.manual_seed(3);score = torch.randn(4, 5, 100, dtype=torch.float64)*20
    mask = torch.rand(4, 100) > .3;mask[:, 0] = True
    normalizer = GlobalHeadNormalizer()
    for a in range(0, 100, chunk):normalizer.update(score[..., a:a+chunk], mask[:, a:a+chunk])
    got = torch.cat([normalizer.probability(score[..., a:a+chunk], mask[:, a:a+chunk])
                     for a in range(0, 100, chunk)], -1)
    expected = mean_mixture(score, mask)
    assert torch.allclose(got, expected, atol=5e-13, rtol=5e-13)
    assert torch.allclose(got.sum(-1), torch.ones(4, dtype=torch.float64))


def test_chunkwise_softmax_counterexample():
    s = torch.tensor([[[20., 19., -10., -11.]]], dtype=torch.float64)
    mask = torch.ones(1, 4, dtype=torch.bool)
    wrong = torch.cat([mean_mixture(s[..., :2], mask[:, :2]), mean_mixture(s[..., 2:], mask[:, 2:])], -1)
    assert not torch.allclose(wrong, mean_mixture(s, mask))
    assert wrong.sum().item() == pytest.approx(2)


def test_head_duplicate_and_permutation():
    torch.manual_seed(4);s = torch.randn(2, 1, 100, dtype=torch.float64);m = torch.ones(2, 100, dtype=torch.bool)
    assert torch.allclose(mean_mixture(s, m), mean_mixture(s.repeat(1, 5, 1), m), atol=1e-15)
    s = torch.randn(2, 5, 100, dtype=torch.float64)
    assert torch.allclose(mean_mixture(s, m), mean_mixture(s[:, [3, 2, 0, 4, 1]], m), atol=1e-15)


@pytest.mark.parametrize('chunk', [1, 9, 20, 100])
def test_stream_oracle_exact_and_ties(chunk):
    d = torch.arange(100, dtype=torch.float64).remainder(13).expand(3, 100).clone()
    ids = torch.arange(100).expand(3, 100);mask = torch.ones_like(ids, dtype=torch.bool);mask[:, :3] = False
    stream = StreamingTopK()
    for a in range(0, 100, chunk):stream.update(d[:, a:a+chunk], ids[:, a:a+chunk], mask[:, a:a+chunk])
    got_d, got_ids = stream.finalize()
    expected = stable_order(d, mask, False)[:, :10]
    assert torch.equal(got_ids, expected)
    assert torch.equal(got_d, d.gather(1, expected))


@pytest.mark.parametrize('h', [96, 192, 336, 720])
def test_paa_all_horizons_and_future_free_signature(h):
    q = torch.arange(h, dtype=torch.float64)[None]
    c = q[:, None, :].expand(1, 100, h)
    assert paa16(q).shape == (1, 16)
    scores = past_scores(q, c)
    for s in scores.values():assert torch.allclose(s, torch.ones_like(s), atol=1e-12)
    with pytest.raises(TypeError):past_scores(q, c, torch.zeros(1, h))


def test_constant_history_finite():
    s = past_scores(torch.ones(2, 96), torch.ones(2, 100, 96))
    assert all(torch.isfinite(v).all() for v in s.values())
    assert torch.equal(s['Pearson'], torch.zeros(2, 100))


def test_random_batch_invariance():
    a = fixed_random_scores([10, 11, 12, 13], 7, 100)
    b = np.concatenate([fixed_random_scores([10, 11], 7, 100), fixed_random_scores([12, 13], 7, 100)])
    assert np.array_equal(a, b)
    assert not np.array_equal(a, fixed_random_scores([10, 11, 12, 13], 8, 100))


def test_future_dispersion_identity():
    torch.manual_seed(6);v = torch.randn(4, 20, 96, dtype=torch.float64)
    direct = torch.stack([(v[:, a]-v[:, b]).square().mean(-1) for a in range(20) for b in range(a+1, 20)], 1).mean(1)
    assert torch.allclose(direct, pair_mse_from_dispersion(v), atol=1e-12)
    assert torch.isnan(pair_mse_from_dispersion(v[:, :1])).all()


def test_bootstrap_channel_joint_and_seed():
    values = np.c_[np.arange(20), -np.arange(20)]
    result = bootstrap_query_channel_mean(values, 4, 25)
    assert result['ci95'] == [0., 0.]
    assert result == bootstrap_query_channel_mean(values, 4, 25)


def test_ridge_train_val_guards_and_scaler():
    rng = np.random.default_rng(7);x = rng.normal(size=(100, 3));y = 2*x[:, 0]-x[:, 1]
    p = TrainOnlyRidgeProbe()
    with pytest.raises(AssertionError):p.fit(x, y, 'test', True)
    with pytest.raises(AssertionError):p.fit(x, y, 'train', False)
    p.fit(x, y, 'train', True)
    old = p.mean.copy()
    with pytest.raises(AssertionError):p.choose(x, y, 'test')
    p.choose(x+1, y+1, 'val')
    assert np.array_equal(old, p.mean)
    assert np.square(p.predict(x)-y).mean() < .01
