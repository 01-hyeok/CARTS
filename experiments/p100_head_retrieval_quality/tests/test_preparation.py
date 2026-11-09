import copy
import numpy as np
import pytest
import torch
from torch import nn
from preparation import (initialize_common_encoder, seeded_slot_heads, state_hash,
    make_training_loader, batch_order_hash, WeightedMean, future_distances,
    oracle_ids, retrieval_metrics, last_repeat_metrics, last_skill, random_topk,
    mixture_ids, validate_pool, require_teacher_resolution)


def encoder_factory():
    return nn.Sequential(nn.Linear(8, 6), nn.GELU(), nn.Dropout(.1), nn.Linear(6, 4))


def test_common_encoder_initialization_equal_across_arms():
    hashes = []
    for heads in (0, 1, 2, 5):
        _, state, h = initialize_common_encoder(encoder_factory, seed=1)
        hashes.append(h)
        if heads:
            seeded_slot_heads(4, heads, seed=1)
    assert len(set(hashes)) == 1


def test_common_encoder_seed_changes_and_rng_isolation():
    before = torch.get_rng_state().clone()
    _, state0, hash0 = initialize_common_encoder(encoder_factory, 0)
    _, state1, hash1 = initialize_common_encoder(encoder_factory, 1)
    assert hash0 != hash1
    assert torch.equal(torch.get_rng_state(), before)
    assert all(v.device.type == 'cpu' for v in state0.values())


def test_seed0_head_initialization_matches_canonical():
    from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
    assert torch.equal(seeded_slot_heads(4, 5, 0).W, SlotHeads(4, n_slots=5).W)


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_common_head_prefix_across_head_counts(seed):
    h1, h2, h5 = [seeded_slot_heads(4, n, seed).W for n in (1, 2, 5)]
    assert torch.equal(h1[0], h5[0]) and torch.equal(h2, h5[:2])


def test_head_seeds_differ_and_do_not_consume_global_rng():
    before = torch.get_rng_state().clone()
    hashes = [state_hash(seeded_slot_heads(4, 5, seed)) for seed in (0, 1, 2)]
    assert len(set(hashes)) == 3
    assert torch.equal(before, torch.get_rng_state())


def order(seed, heads):
    if heads:
        seeded_slot_heads(4, heads, seed)
    loader = make_training_loader(torch.arange(37), 8, seed)
    return [[batch.tolist() for batch in loader] for _ in range(2)]


def test_batch_order_shared_across_arms_and_complete_epochs():
    orders = [order(1, h) for h in (0, 1, 2, 5)]
    assert orders.count(orders[0]) == 4
    for epoch in orders[0]:
        assert sorted(sum(epoch, [])) == list(range(37))
        assert list(map(len, epoch)) == [8, 8, 8, 8, 5]
    assert orders[0][0] != orders[0][1]


def test_batch_order_seed_and_boundaries_recorded():
    assert order(0, 0) != order(1, 0)
    assert batch_order_hash([[1, 2], [3]]) != batch_order_hash([[1], [2, 3]])
    assert batch_order_hash(order(0, 0)[0]) == batch_order_hash(order(0, 5)[0])


def test_weighted_evaluation_not_mean_of_batch_means():
    scalar = WeightedMean()
    scalar.add_batch_mean(2., 6)
    scalar.add_batch_mean(8., 2)
    vector = WeightedMean()
    vector.add_values(np.array([[2., 2.], [2., 2.], [2., 2.]]))
    vector.add_values(np.array([[8., 8.]]))
    assert scalar.mean == vector.mean == 3.5
    assert scalar.mean != (2. + 8.) / 2


def test_evaluation_rejects_nonfinite_and_empty():
    with pytest.raises(ValueError):
        WeightedMean().add_values([np.nan])
    with pytest.raises(ValueError):
        _ = WeightedMean().mean


def test_delta_last_alignment_and_oracle_retmse_bound():
    future = torch.tensor([[[11., 12.], [4., 6.], [0., 0.]]])
    last = torch.tensor([[10., 3., 0.]])
    qlast = torch.tensor([5.])
    target = torch.tensor([[6., 7.]])
    d = future_distances(future, last, qlast, target)
    assert torch.allclose(d, torch.tensor([[0., .5, 2.5]]))
    oracle = oracle_ids(d, 2)
    m = retrieval_metrics(future, last, qlast, target, torch.tensor([[1, 2]]))
    assert (m['regret'] >= 0).all()
    om = retrieval_metrics(future, last, qlast, target, oracle)
    assert (om['recall'] == 1).all() and (om['regret'] == 0).all()


def test_individual_oracle_is_not_set_oracle():
    future = torch.tensor([[[1.], [1.1], [-2.]]], dtype=torch.float64)
    last, qlast, truth = torch.zeros(1, 3), torch.zeros(1), torch.zeros(1, 1)
    d = future_distances(future, last, qlast, truth)
    o = retrieval_metrics(future, last, qlast, truth, oracle_ids(d, 2))
    a = retrieval_metrics(future, last, qlast, truth, torch.tensor([[0, 2]]))
    assert o['retmse'] < a['retmse']
    assert o['agg_mse'] > a['agg_mse']
    assert torch.allclose(o['agg_mae'], torch.tensor([1.05], dtype=torch.float64))


def test_oracle_ties_deterministic_and_selection_unique():
    assert oracle_ids(torch.zeros(2, 8), 3).tolist() == [[0, 1, 2], [0, 1, 2]]
    with pytest.raises(ValueError, match='Duplicate'):
        retrieval_metrics(torch.zeros(1, 3, 2), torch.zeros(1, 3), torch.zeros(1), torch.zeros(1, 2), torch.tensor([[0, 0]]))


def test_last_repeat_exact_and_skill_ratio_of_means():
    m = last_repeat_metrics(torch.tensor([[1., 2., 3.], [8., 6., 4.]]), torch.tensor([[3., 5.], [2., 4.]]))
    assert m['prediction'].tolist() == [[3., 3.], [4., 4.]]
    assert m['mse'].tolist() == [2., 2.] and m['mae'].tolist() == [1., 1.]
    assert last_skill([1., 9.], [1., 3.]) == -1.5
    assert last_skill([0., 0.], [0., 0.]) is None


def test_random_topk_reproducible_without_replacement():
    a, b = random_topk(4, 100, 10, 901), random_topk(4, 100, 10, 901)
    assert torch.equal(a, b)
    assert not torch.equal(a, random_topk(4, 100, 10, 902))
    assert all(len(set(row.tolist())) == 10 for row in a)


def test_true_mixture_is_not_mean_logits():
    scores = torch.tensor([[[4., 3., 0.], [0., 3., 4.]]])
    ids, heads, mix = mixture_ids(scores, torch.ones(1, 3, dtype=torch.bool), 1., 1)
    assert int(ids[0, 0]) != int(scores.mean(1).argmax(-1)[0])
    assert torch.allclose(mix, heads.mean(1))


def test_head_duplication_permutation_and_mask():
    scores = torch.tensor([[[3., 1., 2., -1.]]])
    mask = torch.tensor([[True, True, False, True]])
    one, _, _ = mixture_ids(scores, mask, .1, 2)
    duplicated, _, _ = mixture_ids(scores.repeat(1, 5, 1), mask, .1, 2)
    assert torch.equal(one, duplicated) and 2 not in duplicated.tolist()[0]
    many = torch.cat([scores, scores * .5, scores.flip(-1)], dim=1)
    a, _, pa = mixture_ids(many, mask, .3, 2)
    b, _, pb = mixture_ids(many[:, [2, 0, 1]], mask, .3, 2)
    assert torch.equal(a, b) and torch.allclose(pa, pb)


def test_raft_training_is_nonoverlap_not_strict_causal():
    assert validate_pool(np.array([[[0, 2]]]), [20], [0, 10, 40], 4, 4, 'train', pool_size=2)
    with pytest.raises(ValueError, match='Temporal'):
        validate_pool(np.array([[[0, 2]]]), [20], [0, 10, 40], 4, 4, 'val', pool_size=2)


def test_validation_bank_known_at_forecast_origin():
    assert validate_pool(np.array([[[0, 1]]]), [20], [0, 16], 4, 4, 'test', pool_size=2)
    with pytest.raises(ValueError, match='Temporal'):
        validate_pool(np.array([[[0, 1]]]), [20], [0, 17], 4, 4, 'test', pool_size=2)


def test_pool_duplicate_and_wrong_size_fail():
    with pytest.raises(ValueError, match='Duplicate'):
        validate_pool(np.array([[[0, 0]]]), [20], [0, 10], 4, 4, 'train', pool_size=2)
    with pytest.raises(ValueError, match='shape'):
        validate_pool(np.array([[[0, 1]]]), [20], [0, 10], 4, 4, 'train')


def test_unresolved_teacher_blocks_even_if_mode_prepopulated():
    for mode in (None, 'zscore_future_mse', 'raw_future_mse'):
        with pytest.raises(RuntimeError, match='unresolved'):
            require_teacher_resolution({'teacher': {'status': 'PENDING_USER_DECISION', 'mode': mode}})


@pytest.mark.parametrize('mode', ['zscore_future_mse', 'raw_future_mse'])
def test_gate_requires_explicit_decision_reference(mode):
    with pytest.raises(RuntimeError, match='reference'):
        require_teacher_resolution({'teacher': {'status': 'USER_CONFIRMED', 'mode': mode}})
    assert require_teacher_resolution({'teacher': {'status': 'USER_CONFIRMED', 'mode': mode, 'decision_reference': 'synthetic-unit-test-only'}}) == mode
