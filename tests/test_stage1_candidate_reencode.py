"""Stage-1 candidate-side gradient recovery.

Covers the three arms of the experiment: the untouched full-bank KL baseline,
the selected-100 detached control, and the selected-100 re-encode arm whose
whole point is that the candidate side carries gradient.
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.RelationStage1 import Model, select_training_candidates
from tests.test_stage1_new_losses import _model_config


def _subset_config(mode, top_m=4, inject_k=2, enc_in=1):
    config = _model_config(loss_mode='kl')
    config.enc_in = enc_in
    config.stage1_candidate_subset_mode = mode
    config.stage1_candidate_mine_top_m = top_m
    config.stage1_candidate_oracle_inject_k = inject_k
    config.relation_input_space = 'delta_last'
    config.relation_teacher_space = 'delta_last'
    return config


def _batch(num_candidates=8, bsz=3, seq=4, ch=1, seed=3):
    torch.manual_seed(seed)
    query_x = torch.randn(bsz, seq, ch)
    query_y = torch.randn(bsz, seq, ch)
    memory_x = torch.randn(num_candidates, seq, ch)
    memory_y = torch.randn(num_candidates, seq, ch)
    memory_x_last = memory_x[:, -1, :]
    cand_mask = torch.ones(bsz, num_candidates, dtype=torch.bool)
    return query_x, query_y, memory_x, memory_y, memory_x_last, cand_mask


def _bank(model, memory_x):
    return model.build_embedding_bank(memory_x.numpy(), torch.device('cpu'))


def _forward(model, mode_train, **kw):
    model.train(mode_train)
    return model(**kw)


# --------------------------------------------------------------------------
# selection logic
# --------------------------------------------------------------------------

def test_mining_takes_highest_cosine_first():
    scores = torch.tensor([[0.1, 0.9, 0.5, 0.7, 0.3]])
    # Oracle already sits inside the mined set, so nothing is injected.
    future = torch.tensor([[9.0, 0.0, 8.0, 1.0, 7.0]])
    valid = torch.ones(1, 5, dtype=torch.bool)
    selected, _ = select_training_candidates(scores, future, valid, top_m=3, oracle_k=2)
    assert selected.tolist() == [[1, 3, 2]]


def test_missing_global_oracle_replaces_worst_ranked_non_oracle():
    scores = torch.tensor([[0.9, 0.8, 0.7, 0.1, 0.05]])
    # Global Oracle Top-2 is {4, 3}: both outside Bank Top-3 = {0, 1, 2}.
    future = torch.tensor([[9.0, 8.0, 7.0, 1.0, 0.0]])
    valid = torch.ones(1, 5, dtype=torch.bool)
    selected, stats = select_training_candidates(
        scores, future, valid, top_m=3, oracle_k=2
    )
    assert stats['oracle_missing_count_before_injection'].tolist() == [2.0]
    row = selected[0].tolist()
    assert row[0] == 0, 'the best bank rank must survive injection'
    assert set(row) == {0, 4, 3}
    assert len(set(row)) == 3


def test_partial_overlap_keeps_exactly_top_m_without_duplicates():
    torch.manual_seed(0)
    bsz, num_cand, top_m, oracle_k = 6, 40, 12, 5
    scores = torch.randn(bsz, num_cand)
    future = torch.rand(bsz, num_cand)
    valid = torch.rand(bsz, num_cand) > 0.15
    valid[:, :top_m + oracle_k] = True
    selected, _ = select_training_candidates(scores, future, valid, top_m, oracle_k)

    assert selected.shape == (bsz, top_m)
    oracle = torch.topk(
        future.masked_fill(~valid, float('inf')), k=oracle_k, dim=-1, largest=False
    ).indices
    for row in range(bsz):
        row_selected = selected[row].tolist()
        assert len(set(row_selected)) == top_m, 'duplicate candidate selected'
        assert set(oracle[row].tolist()).issubset(row_selected), 'oracle missing'


def test_invalid_candidates_are_never_injected():
    scores = torch.tensor([[0.9, 0.8, 0.7, 0.6]])
    future = torch.tensor([[5.0, 6.0, 7.0, 8.0]])
    valid = torch.tensor([[True, True, False, False]])
    selected, stats = select_training_candidates(
        scores, future, valid, top_m=3, oracle_k=3
    )
    assert stats['oracle_valid_count'].tolist() == [2.0]
    assert stats['oracle_missing_count_before_injection'].tolist() == [0.0]
    assert {0, 1}.issubset(selected[0].tolist())


def test_oracle_k_larger_than_top_m_is_rejected():
    scores = torch.zeros(1, 5)
    future = torch.zeros(1, 5)
    valid = torch.ones(1, 5, dtype=torch.bool)
    try:
        select_training_candidates(scores, future, valid, top_m=2, oracle_k=4)
    except ValueError:
        return
    raise AssertionError('oracle_k > top_m must raise')


# --------------------------------------------------------------------------
# arm behaviour
# --------------------------------------------------------------------------

def test_subset_mode_none_leaves_the_full_bank_arm_untouched():
    query_x, query_y, memory_x, memory_y, memory_x_last, cand_mask = _batch()
    legacy = _subset_config('none')
    del legacy.stage1_candidate_subset_mode
    baseline = Model(legacy)
    subset_off = Model(_subset_config('none'))
    subset_off.load_state_dict(baseline.state_dict())

    kw = dict(
        query_x=query_x, query_y=query_y, cand_mask=cand_mask,
        memory_y=memory_y, memory_x_last=memory_x_last,
        candidate_x=memory_x, compute_detailed_metrics=False,
    )
    loss_a, _ = _forward(baseline, True, key_bank=_bank(baseline, memory_x), **kw)
    loss_b, _ = _forward(subset_off, True, key_bank=_bank(subset_off, memory_x), **kw)
    assert torch.allclose(loss_a, loss_b, atol=1e-7)
    assert not subset_off.candidate_subset_active()


def test_both_selected_arms_use_the_same_candidate_set():
    """Only the gradient path may differ between arm B and arm C."""
    query_x, query_y, memory_x, memory_y, memory_x_last, cand_mask = _batch()
    detached = Model(_subset_config('selected_detached'))
    reencode = Model(_subset_config('selected_reencode'))
    reencode.load_state_dict(detached.state_dict())

    captured = []
    import models.RelationStage1 as module
    original = module.select_training_candidates

    def record(*args, **kwargs):
        selected, stats = original(*args, **kwargs)
        captured.append(selected.clone())
        return selected, stats

    module.select_training_candidates = record
    try:
        kw = dict(
            query_x=query_x, query_y=query_y, cand_mask=cand_mask,
            memory_y=memory_y, memory_x_last=memory_x_last,
            candidate_x=memory_x, compute_detailed_metrics=False,
        )
        _forward(detached, True, key_bank=_bank(detached, memory_x), **kw)
        _forward(reencode, True, key_bank=_bank(reencode, memory_x), **kw)
    finally:
        module.select_training_candidates = original

    assert len(captured) == 2
    assert torch.equal(captured[0], captured[1])


def test_detached_arm_gives_the_candidate_side_no_gradient():
    query_x, query_y, memory_x, memory_y, memory_x_last, cand_mask = _batch()
    model = Model(_subset_config('selected_detached'))
    memory_x = memory_x.clone().requires_grad_(True)

    loss, _ = _forward(
        model, True,
        query_x=query_x, query_y=query_y, cand_mask=cand_mask,
        memory_y=memory_y, key_bank=_bank(model, memory_x.detach()),
        memory_x_last=memory_x_last, candidate_x=memory_x,
        compute_detailed_metrics=False,
    )
    loss.backward()
    assert memory_x.grad is None or memory_x.grad.abs().sum() == 0


def test_reencode_arm_sends_gradient_through_the_candidate_side():
    query_x, query_y, memory_x, memory_y, memory_x_last, cand_mask = _batch()
    model = Model(_subset_config('selected_reencode'))
    memory_x = memory_x.clone().requires_grad_(True)

    captured = {}
    original = model._reencode_selected_candidates

    def spy(candidate_x, selected, c, r):
        z_k, count = original(candidate_x, selected, c, r)
        z_k.retain_grad()
        captured['z_k'] = z_k
        return z_k, count

    model._reencode_selected_candidates = spy
    loss, _ = _forward(
        model, True,
        query_x=query_x, query_y=query_y, cand_mask=cand_mask,
        memory_y=memory_y, key_bank=_bank(model, memory_x.detach()),
        memory_x_last=memory_x_last, candidate_x=memory_x,
        compute_detailed_metrics=False,
    )
    assert captured['z_k'].requires_grad
    loss.backward()

    assert captured['z_k'].grad is not None
    assert captured['z_k'].grad.abs().sum() > 0, 'candidate embeddings got no gradient'
    assert memory_x.grad is not None and memory_x.grad.abs().sum() > 0
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.encoder.parameters()
    )


def test_selected_distributions_are_shaped_by_top_m():
    query_x, query_y, memory_x, memory_y, memory_x_last, cand_mask = _batch(
        num_candidates=12
    )
    top_m = 5
    for mode in ('selected_detached', 'selected_reencode'):
        model = Model(_subset_config(mode, top_m=top_m, inject_k=2))
        captured = {}
        original_kl_source = model._teacher_logits

        def spy(*args, **kwargs):
            logits, future = original_kl_source(*args, **kwargs)
            captured['full'] = logits.shape
            return logits, future

        model._teacher_logits = spy
        _, metrics = _forward(
            model, True,
            query_x=query_x, query_y=query_y, cand_mask=cand_mask,
            memory_y=memory_y, key_bank=_bank(model, memory_x),
            memory_x_last=memory_x_last, candidate_x=memory_x,
            compute_detailed_metrics=True,
        )
        assert captured['full'] == (query_x.size(0), 12), 'mining must see the full pool'
        # student_entropy is computed on the selected columns only.
        assert metrics['student_effective_candidates'] <= top_m + 1e-4


def test_validation_never_subsets_or_injects():
    query_x, query_y, memory_x, memory_y, memory_x_last, cand_mask = _batch()
    model = Model(_subset_config('selected_reencode'))
    bank = _bank(model, memory_x)

    import models.RelationStage1 as module
    original = module.select_training_candidates

    def forbidden(*args, **kwargs):
        raise AssertionError('validation must not mine or inject candidates')

    module.select_training_candidates = forbidden
    try:
        with torch.no_grad():
            _forward(
                model, False,
                query_x=query_x, query_y=query_y, cand_mask=cand_mask,
                memory_y=memory_y, key_bank=bank,
                memory_x_last=memory_x_last, candidate_x=memory_x,
                compute_detailed_metrics=True,
            )
    finally:
        module.select_training_candidates = original
    assert not model.candidate_subset_active()


def test_mining_diagnostics_are_reported():
    query_x, query_y, memory_x, memory_y, memory_x_last, cand_mask = _batch(
        num_candidates=12
    )
    model = Model(_subset_config('selected_reencode', top_m=6, inject_k=3))
    _, metrics = _forward(
        model, True,
        query_x=query_x, query_y=query_y, cand_mask=cand_mask,
        memory_y=memory_y, key_bank=_bank(model, memory_x),
        memory_x_last=memory_x_last, candidate_x=memory_x,
        compute_detailed_metrics=False,
    )
    for key in (
        'bank_oracle_recall_at_3',
        'bank_oracle_recall_at_6',
        'oracle_count_in_bank_top_m',
        'oracle_missing_count_before_injection',
        'candidate_unique_encoded',
    ):
        assert key in metrics, key


def test_multi_channel_branches_each_start_from_the_full_candidate_pool():
    """Regression: the subset must not leak from one relation branch to the next.

    With several channels the forward runs many (target, source) branches. If a
    branch narrows the shared cand_mask in place, the next branch mines a
    100-column mask against an 8449-column bank and the shapes blow up.
    """
    channels, num_cand = 3, 16
    query_x, query_y, memory_x, memory_y, memory_x_last, cand_mask = _batch(
        num_candidates=num_cand, ch=channels
    )
    for mode in ('selected_detached', 'selected_reencode'):
        model = Model(_subset_config(mode, top_m=6, inject_k=3, enc_in=channels))
        loss, metrics = _forward(
            model, True,
            query_x=query_x, query_y=query_y, cand_mask=cand_mask,
            memory_y=memory_y, key_bank=_bank(model, memory_x),
            memory_x_last=memory_x_last, candidate_x=memory_x,
            compute_detailed_metrics=True,
        )
        assert torch.isfinite(loss), mode
        # channels * channels branches all contributed, none of them crashed.
        assert metrics['student_effective_candidates'] <= 6 + 1e-4, mode


def test_subset_mode_rejects_unsupported_objectives():
    """kl and topk_coverage are supported; anything else must be refused."""
    for loss_mode, teacher_mode in (('rnc', 'mse'), ('kl_infonce', 'mse'), ('kl', 'ema_input')):
        config = _subset_config('selected_reencode')
        config.stage1_loss_mode = loss_mode
        config.stage1_teacher_mode = teacher_mode
        try:
            Model(config)
        except ValueError:
            continue
        raise AssertionError(f'{loss_mode}/{teacher_mode} must be rejected')


# --------------------------------------------------------------------------
# topk_coverage objective on the selected subset
# --------------------------------------------------------------------------

def _coverage_subset_config(mode, top_m=6, inject_k=3, enc_in=1):
    config = _subset_config(mode, top_m=top_m, inject_k=inject_k, enc_in=enc_in)
    config.stage1_loss_mode = 'topk_coverage'
    config.stage1_coverage_top_k = inject_k
    return config


def test_coverage_objective_is_accepted_with_the_candidate_subset():
    query_x, query_y, memory_x, memory_y, memory_x_last, cand_mask = _batch(
        num_candidates=12
    )
    for mode in ('selected_detached', 'selected_reencode'):
        model = Model(_coverage_subset_config(mode))
        loss, metrics = _forward(
            model, True,
            query_x=query_x, query_y=query_y, cand_mask=cand_mask,
            memory_y=memory_y, key_bank=_bank(model, memory_x),
            memory_x_last=memory_x_last, candidate_x=memory_x,
            compute_detailed_metrics=True,
        )
        assert torch.isfinite(loss), mode
        assert metrics['topk_coverage_loss'] > 0, mode
        assert metrics['kl_loss'] == 0, mode
        # Injection guarantees every Oracle positive survived mining, so the
        # coverage target must be full width on the selected columns.
        assert float(metrics['coverage_effective_k']) == 3.0, mode


def test_coverage_targets_are_recomputed_on_the_selected_columns():
    """Full-pool Oracle indices are meaningless after gathering; recompute."""
    query_x, query_y, memory_x, memory_y, memory_x_last, cand_mask = _batch(
        num_candidates=12
    )
    model = Model(_coverage_subset_config('selected_reencode'))

    seen = []
    import models.RelationStage1 as module
    original = module.prepare_topk_coverage_targets

    def record(future_mse, valid_mask, top_k):
        seen.append(tuple(future_mse.shape))
        return original(future_mse, valid_mask, top_k)

    module.prepare_topk_coverage_targets = record
    try:
        _forward(
            model, True,
            query_x=query_x, query_y=query_y, cand_mask=cand_mask,
            memory_y=memory_y, key_bank=_bank(model, memory_x),
            memory_x_last=memory_x_last, candidate_x=memory_x,
            compute_detailed_metrics=False,
        )
    finally:
        module.prepare_topk_coverage_targets = original

    bsz = query_x.size(0)
    assert (bsz, 12) in seen, 'mining must prepare Oracle targets on the full pool'
    assert (bsz, 6) in seen, 'the loss must use targets on the selected columns'


def test_coverage_reencode_sends_gradient_through_candidates():
    query_x, query_y, memory_x, memory_y, memory_x_last, cand_mask = _batch(
        num_candidates=12
    )
    model = Model(_coverage_subset_config('selected_reencode'))
    memory_x = memory_x.clone().requires_grad_(True)

    loss, _ = _forward(
        model, True,
        query_x=query_x, query_y=query_y, cand_mask=cand_mask,
        memory_y=memory_y, key_bank=_bank(model, memory_x.detach()),
        memory_x_last=memory_x_last, candidate_x=memory_x,
        compute_detailed_metrics=False,
    )
    loss.backward()
    assert memory_x.grad is not None and memory_x.grad.abs().sum() > 0


def test_coverage_detached_gives_candidates_no_gradient():
    query_x, query_y, memory_x, memory_y, memory_x_last, cand_mask = _batch(
        num_candidates=12
    )
    model = Model(_coverage_subset_config('selected_detached'))
    memory_x = memory_x.clone().requires_grad_(True)

    loss, _ = _forward(
        model, True,
        query_x=query_x, query_y=query_y, cand_mask=cand_mask,
        memory_y=memory_y, key_bank=_bank(model, memory_x.detach()),
        memory_x_last=memory_x_last, candidate_x=memory_x,
        compute_detailed_metrics=False,
    )
    loss.backward()
    assert memory_x.grad is None or memory_x.grad.abs().sum() == 0


def test_coverage_multi_channel_branches():
    channels = 3
    query_x, query_y, memory_x, memory_y, memory_x_last, cand_mask = _batch(
        num_candidates=16, ch=channels
    )
    model = Model(_coverage_subset_config('selected_reencode', enc_in=channels))
    loss, metrics = _forward(
        model, True,
        query_x=query_x, query_y=query_y, cand_mask=cand_mask,
        memory_y=memory_y, key_bank=_bank(model, memory_x),
        memory_x_last=memory_x_last, candidate_x=memory_x,
        compute_detailed_metrics=True,
    )
    assert torch.isfinite(loss)
    assert 'student_oracle_recall_at_10' in metrics
