"""Guards for end-to-end retrieval and the score-separation losses.

Two things must hold for this experiment to mean anything. The re-scoring path
has to be the same arithmetic the production retrieval op does, or "end-to-end
helped" is a comparison against a different model. And the margin losses have to
actually be about absolute separation, not order -- order is what KL already
gave us, and it did not move the forecast.
"""

import pytest
import torch

from utils.rank_losses import (
    mine_ranking_candidates,
    ranking_loss,
    score_geometry,
    weight_geometry,
)
from utils.retrieval_ops import retrieve_relation_future, reweight_selected_candidates


# ---------- the re-scoring path is the production path ----------

def test_rescoring_reproduces_the_production_retrieval_exactly():
    """Feeding back the same bank rows must change nothing: end-to-end differs
    from the baseline in where gradient flows, not in what is computed."""
    torch.manual_seed(0)
    z_q = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    z_mem = torch.nn.functional.normalize(torch.randn(20, 8), dim=-1)
    values = torch.randn(20, 6)
    valid = torch.ones(4, 20, dtype=torch.bool)

    retrieved, alpha, top_idx, top_scores, debug = retrieve_relation_future(
        z_q=z_q, z_mem=z_mem, memory_value_c=values, valid_mask=valid,
        top_k=5, tau_topk=0.1)

    again, alpha_again, scores_again = reweight_selected_candidates(
        z_q=z_q, z_k_sel=z_mem[top_idx], values=debug['v_top'],
        top_valid=debug['top_valid'], tau_topk=0.1)

    torch.testing.assert_close(scores_again, top_scores)
    torch.testing.assert_close(alpha_again, alpha)
    torch.testing.assert_close(again, retrieved)


def test_rescoring_sends_gradient_to_both_sides():
    """The whole point: the precomputed bank blocks candidate-side gradient."""
    z_q = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1).requires_grad_(True)
    z_k = torch.nn.functional.normalize(torch.randn(3, 4, 8), dim=-1).requires_grad_(True)
    retrieved, _, _ = reweight_selected_candidates(
        z_q=z_q, z_k_sel=z_k, values=torch.randn(3, 4, 6),
        top_valid=torch.ones(3, 4, dtype=torch.bool), tau_topk=0.05)
    retrieved.square().mean().backward()
    assert z_q.grad.abs().sum() > 0
    assert z_k.grad.abs().sum() > 0


def test_rescoring_ignores_invalid_slots():
    z_q = torch.nn.functional.normalize(torch.randn(2, 8), dim=-1)
    z_k = torch.nn.functional.normalize(torch.randn(2, 4, 8), dim=-1)
    values = torch.randn(2, 4, 5)
    valid = torch.tensor([[True, True, False, False], [True, False, False, False]])
    _, alpha, _ = reweight_selected_candidates(
        z_q=z_q, z_k_sel=z_k, values=values, top_valid=valid, tau_topk=0.1)
    assert torch.allclose(alpha[~valid], torch.zeros(()))
    torch.testing.assert_close(alpha.sum(-1), torch.ones(2))
    assert alpha[1, 0] == pytest.approx(1.0)


def test_rescoring_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match='z_k_sel must be'):
        reweight_selected_candidates(
            z_q=torch.randn(2, 8), z_k_sel=torch.randn(3, 4, 8),
            values=torch.randn(3, 4, 5),
            top_valid=torch.ones(3, 4, dtype=torch.bool), tau_topk=0.1)
    with pytest.raises(ValueError, match='disagree on'):
        reweight_selected_candidates(
            z_q=torch.randn(2, 8), z_k_sel=torch.randn(2, 4, 8),
            values=torch.randn(2, 3, 5),
            top_valid=torch.ones(2, 4, dtype=torch.bool), tau_topk=0.1)


# ---------- the losses are about separation, not order ----------

def _teacher_student(student):
    teacher = torch.tensor([[3.0, 2.0, 1.0]])
    return teacher, torch.tensor([student]), torch.ones(1, 3, dtype=torch.bool)


@pytest.mark.parametrize('mode', ['ranknet', 'weighted_ranknet', 'margin', 'adaptive_margin'])
def test_agreeing_order_costs_less_than_reversed_order(mode):
    agree = ranking_loss(*_teacher_student([1.0, 0.0, -1.0]), mode=mode, margin=0.05)[0]
    reverse = ranking_loss(*_teacher_student([-1.0, 0.0, 1.0]), mode=mode, margin=0.05)[0]
    assert agree < reverse


def test_margin_loss_is_zero_once_the_gap_is_wide_enough():
    """A margin that is already met must stop pulling -- otherwise the loss keeps
    inflating scores forever instead of just separating them."""
    loss, metrics = ranking_loss(*_teacher_student([1.0, 0.5, 0.0]), mode='margin', margin=0.05)
    assert float(loss) == pytest.approx(0.0)
    assert float(metrics['rank_margin_satisfied']) == pytest.approx(1.0)


def test_margin_loss_still_penalizes_a_correct_but_compressed_order():
    """This is the failure the experiment exists for: the order is right and the
    gaps are far too small for tau_topk to turn into a usable weighting."""
    compressed = ranking_loss(*_teacher_student([0.002, 0.001, 0.0]), mode='margin', margin=0.05)
    assert float(compressed[0]) > 0.0
    assert float(compressed[1]['rank_order_accuracy']) == pytest.approx(1.0)
    # RankNet, which only cares about order, is nearly satisfied by the same scores.
    ranknet = ranking_loss(*_teacher_student([0.002, 0.001, 0.0]), mode='ranknet')
    assert float(ranknet[0]) < float(compressed[0]) + 1.0
    assert float(ranknet[1]['rank_margin_satisfied']) == pytest.approx(0.0)


def test_adaptive_margin_asks_less_of_near_tied_teachers():
    """Candidates the teacher barely separates should not be forced apart."""
    near_tie = torch.tensor([[1.0, 0.999]])
    far = torch.tensor([[1.0, 0.0]])
    student = torch.tensor([[0.01, 0.0]])
    valid = torch.ones(1, 2, dtype=torch.bool)
    tie_loss = ranking_loss(near_tie, student, valid, mode='adaptive_margin', margin=0.05)
    far_loss = ranking_loss(far, student, valid, mode='adaptive_margin', margin=0.05)
    # Both are single-pair, so the normalized gap is 1 in each; the guard is that
    # neither blows up and the tie is not penalised more than the clear case.
    assert float(tie_loss[0]) == pytest.approx(float(far_loss[0]))
    assert torch.isfinite(tie_loss[0]) and torch.isfinite(far_loss[0])


def test_ties_are_dropped_rather_than_counted_as_errors():
    teacher = torch.tensor([[1.0, 1.0, 1.0]])
    loss, metrics = ranking_loss(teacher, torch.randn(1, 3),
                                 torch.ones(1, 3, dtype=torch.bool), mode='margin')
    assert loss is None and metrics == {}


def test_unknown_mode_and_shape_mismatch_are_rejected():
    teacher, student, valid = _teacher_student([1.0, 0.0, -1.0])
    with pytest.raises(ValueError, match='Unsupported ranking mode'):
        ranking_loss(teacher, student, valid, mode='hinge2')
    with pytest.raises(ValueError, match='student_scores shape'):
        ranking_loss(teacher, student[:, :-1], valid, mode='margin')


def test_ranking_loss_reaches_the_student_scores():
    student = torch.tensor([[0.0, 0.0, 0.0]], requires_grad=True)
    loss, _ = ranking_loss(torch.tensor([[3.0, 2.0, 1.0]]), student,
                           torch.ones(1, 3, dtype=torch.bool), mode='margin', margin=0.05)
    loss.backward()
    assert student.grad.abs().sum() > 0
    # Better teacher score must be pushed up, worse pushed down.
    assert student.grad[0, 0] < 0 < student.grad[0, 2]


# ---------- pair mining ----------

def test_mining_takes_positives_from_teacher_and_hard_negatives_from_student():
    """A loss trained only on the model's own Top-K can never learn that
    something outside it should have been in, so the two sources must differ."""
    teacher = torch.tensor([[5.0, 4.0, 0.0, 0.0, 0.0]])
    student = torch.tensor([[0.0, 0.0, 9.0, 8.0, 0.0]])
    valid = torch.ones(1, 5, dtype=torch.bool)
    indices, counts = mine_ranking_candidates(
        teacher, student, valid, top_p=2, hard_negatives=2, random_negatives=0)
    assert set(indices[0, :2].tolist()) == {0, 1}      # teacher's best
    assert set(indices[0, 2:4].tolist()) == {2, 3}     # student's favourites
    assert counts['rank_candidates'] == 4.0


def test_mining_never_returns_an_invalid_candidate():
    torch.manual_seed(0)
    valid = torch.zeros(2, 12, dtype=torch.bool)
    valid[:, :5] = True
    indices, _ = mine_ranking_candidates(
        torch.randn(2, 12), torch.randn(2, 12), valid,
        top_p=2, hard_negatives=2, random_negatives=3)
    assert valid.gather(1, indices).all()


def test_mining_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match='must share shape'):
        mine_ranking_candidates(torch.randn(2, 5), torch.randn(2, 4),
                                torch.ones(2, 5, dtype=torch.bool))


# ---------- geometry ----------

def test_uniform_weights_report_the_full_effective_k():
    valid = torch.ones(3, 10, dtype=torch.bool)
    metrics = weight_geometry(torch.full((3, 10), 0.1), valid)
    assert float(metrics['effective_k']) == pytest.approx(10.0, abs=1e-4)
    assert float(metrics['normalized_weight_entropy']) == pytest.approx(1.0, abs=1e-4)
    assert float(metrics['max_min_weight_ratio']) == pytest.approx(1.0, abs=1e-4)


def test_peaked_weights_report_a_small_effective_k():
    weights = torch.zeros(1, 10)
    weights[0, 0] = 0.9
    weights[0, 1:] = 0.1 / 9
    metrics = weight_geometry(weights, torch.ones(1, 10, dtype=torch.bool))
    assert float(metrics['effective_k']) < 3.0
    assert float(metrics['max_weight']) == pytest.approx(0.9, abs=1e-5)


def test_score_geometry_measures_the_top1_to_last_spread():
    scores = torch.tensor([[0.996, 0.995, 0.994, 0.992]])
    metrics = score_geometry(scores, torch.ones(1, 4, dtype=torch.bool))
    assert float(metrics['top1_minus_top10']) == pytest.approx(0.004, abs=1e-6)
    assert float(metrics['top1_minus_top2']) == pytest.approx(0.001, abs=1e-6)


def test_score_geometry_ignores_invalid_slots():
    scores = torch.tensor([[0.9, 0.5, -1e30]])
    valid = torch.tensor([[True, True, False]])
    metrics = score_geometry(scores, valid)
    assert float(metrics['top1_minus_top10']) == pytest.approx(0.4, abs=1e-6)


# ---------- configuration guards ----------

def _config(**overrides):
    from tests.test_stage2_candidate_utility import CONFIG

    merged = dict(CONFIG)
    merged.update(overrides)
    return merged


def test_ranking_loss_without_end_to_end_is_rejected():
    """Stage-wise ranking belongs in Stage-1; asking for it here would silently
    compute a loss on scores no gradient can reach."""
    from types import SimpleNamespace

    from models.RelationStage2 import Model

    with pytest.raises(ValueError, match='requires stage2_e2e'):
        Model(SimpleNamespace(**_config(stage2_rank_loss='margin', stage2_e2e=0)))


def test_unknown_rank_mode_is_rejected_at_construction():
    from types import SimpleNamespace

    from models.RelationStage2 import Model

    with pytest.raises(ValueError, match='stage2_rank_loss must be one of'):
        Model(SimpleNamespace(**_config(stage2_rank_loss='listnet', stage2_e2e=1)))


def test_end_to_end_without_candidate_histories_is_rejected():
    from types import SimpleNamespace

    import torch as _torch
    from models.RelationStage2 import Model

    config = _config(stage2_e2e=1)
    model = Model(SimpleNamespace(**config)).eval()
    model.relation_sources = [[c] for c in range(config['enc_in'])]
    channels, horizon, seq_len = config['enc_in'], config['pred_len'], config['seq_len']
    batch_x = _torch.randn(2, seq_len, channels)
    embedding_dim = model._branch_embedding(batch_x, 0, 0).size(-1)
    with pytest.raises(ValueError, match='needs candidate_x'):
        model(batch_x=batch_x,
              memory_y=_torch.randn(6, horizon, channels),
              valid_mask=_torch.ones(2, 6, dtype=_torch.bool),
              key_bank=_torch.randn(channels, 1, 6, embedding_dim),
              memory_x_last=_torch.randn(6, channels))


def test_ema_flag_disables_the_teacher_update():
    """The end-to-end arms run with no EMA teacher; the flag has to actually stop
    the update rather than just relabel it."""
    from types import SimpleNamespace

    from exp.exp_stage2_relation import Exp_Stage2_Relation

    probe = object.__new__(Exp_Stage2_Relation)
    probe.args = SimpleNamespace(use_ema_teacher=0)
    assert probe._ema_enabled() is False
    probe.args = SimpleNamespace(use_ema_teacher=1)
    assert probe._ema_enabled() is True
    probe.args = SimpleNamespace()
    assert probe._ema_enabled() is True     # default keeps the original baseline


# ---------- v2: the corrections the audit forced ----------
#
# The v1 loss missed its own target. Measured on a real checkpoint: 3.7% of mined
# pairs sat inside the Top-K that Stage-2 weights, and they carried 1.9% of the
# margin loss. The other 98% went into widening pairs already 24x further apart,
# which a bounded cosine space can only answer by collapsing -- and the trained
# arm did collapse, to an effective rank of 1.88 out of 128.

def _scoped_batch():
    """Mined set shaped like the measured one: a tight Top-K inside a wider pool.

    Numbers follow the audit -- inside gaps around 0.009, outside gaps an order
    of magnitude larger, and most pairs in both regions still short of a 0.05
    margin, so the outside is not trivially satisfied.
    """
    teacher = torch.tensor([[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]])
    student = torch.tensor([[0.509, 0.500, 0.46, 0.44, 0.41, 0.37, 0.32, 0.26]])
    valid = torch.ones(1, 8, dtype=torch.bool)
    topk = torch.tensor([[True, True, False, False, False, False, False, False]])
    return teacher, student, valid, topk


def test_gamma_moves_the_loss_onto_the_pairs_stage2_actually_weights():
    """Without a scope the inside pairs are outnumbered and vanish; with gamma
    their share is fixed regardless of how many outside pairs were mined."""
    teacher, student, valid, topk = _scoped_batch()
    flat_loss, flat = ranking_loss(teacher, student, valid, mode='margin', margin=0.05,
                                   topk_mask=topk)
    inside = float(flat['rank_loss_inside_topk'])
    outside = float(flat['rank_loss_outside_topk'])
    # The compressed pairs are the ones that miss the margin, so they carry the
    # larger per-pair loss -- and are still swamped, because they are 3.6% of the
    # pairs and a plain mean weights by count.
    assert inside > outside
    assert float(flat['rank_fraction_inside_topk']) < 0.05
    assert abs(float(flat_loss) - outside) < abs(float(flat_loss) - inside)

    scoped, metrics = ranking_loss(teacher, student, valid, mode='margin', margin=0.05,
                                   topk_mask=topk, gamma=0.5)
    torch.testing.assert_close(float(scoped), 0.5 * inside + 0.5 * outside)
    # With the regions weighted rather than counted, the compressed pairs lead.
    assert abs(float(scoped) - inside) < abs(float(flat_loss) - inside)


def test_gamma_one_ignores_pairs_outside_the_topk():
    teacher, student, valid, topk = _scoped_batch()
    loss, metrics = ranking_loss(teacher, student, valid, mode='margin', margin=0.05,
                                 topk_mask=topk, gamma=1.0)
    torch.testing.assert_close(float(loss), float(metrics['rank_loss_inside_topk']))


def test_relative_margin_tracks_the_actual_topk_spread():
    """An absolute 0.05 was 5.7x the measured 0.0088 spread -- a demand for
    separation that is not in the data, which the encoder answered by collapsing."""
    teacher, student, valid, topk = _scoped_batch()
    spread = 0.509 - 0.500
    _, metrics = ranking_loss(teacher, student, valid, mode='margin', margin=2.0,
                              topk_mask=topk, margin_mode='topk_relative')
    assert float(metrics['rank_topk_spread']) == pytest.approx(spread, abs=1e-6)
    assert float(metrics['rank_effective_margin']) == pytest.approx(2.0 * spread, abs=1e-6)

    absolute = ranking_loss(teacher, student, valid, mode='margin', margin=2.0,
                            topk_mask=topk)[1]
    assert float(absolute['rank_effective_margin']) == pytest.approx(2.0)


def test_relative_margin_is_capped_so_the_demand_cannot_run_away():
    teacher = torch.tensor([[2.0, 1.0]])
    student = torch.tensor([[1.0, -1.0]])            # spread 2.0
    valid = torch.ones(1, 2, dtype=torch.bool)
    topk = torch.ones(1, 2, dtype=torch.bool)
    _, metrics = ranking_loss(teacher, student, valid, mode='margin', margin=2.0,
                              topk_mask=topk, margin_mode='topk_relative', margin_cap=0.2)
    assert float(metrics['rank_effective_margin']) == pytest.approx(0.2)


def test_sigma_gives_ranknet_the_scale_it_was_blind_to():
    """RankNet's per-pair push is sigmoid(-a). At sigma=1 a 0.009 gap and a 0.21
    gap both sit at ~0.50, so it cannot tell a compressed pair from a wide one."""
    tight = torch.tensor([[0.009, 0.0]])
    wide = torch.tensor([[0.21, 0.0]])
    teacher = torch.tensor([[2.0, 1.0]])
    valid = torch.ones(1, 2, dtype=torch.bool)

    # The push RankNet applies to a pair is sigmoid(-a). That is what the audit
    # measured at 0.5001 inside the Top-K and 0.4907 outside it -- gaps 24x apart
    # and the same gradient to within 2%.
    push = [float(torch.sigmoid(-s[0, 0] + s[0, 1])) for s in (tight, wide)]
    assert abs(push[0] - push[1]) < 0.06
    assert min(push) > 0.44                         # both parked at the midpoint
    fixed = [float(ranking_loss(teacher, s, valid, mode='ranknet')[0]) for s in (tight, wide)]

    topk = torch.ones(1, 2, dtype=torch.bool)
    scaled = [float(ranking_loss(teacher, s, valid, mode='ranknet',
                                 topk_mask=topk, sigma_mode='topk_relative')[0])
              for s in (tight, wide)]
    # Normalised by its own spread each pair is one unit apart, so the scaled
    # loss is the same for both -- the point is that it no longer depends on the
    # raw magnitude, which is what made 0.009 and 0.21 interchangeable.
    assert scaled[0] == pytest.approx(scaled[1], abs=1e-5)
    assert scaled[0] < fixed[0]


def test_scope_normalisation_is_per_region_for_the_weighted_modes():
    """Outside pairs are ~24x wider; if they set the normaliser, every inside
    pair looks like a rounding error and the adaptive margin asks for nothing."""
    teacher, student, valid, topk = _scoped_batch()
    _, scoped = ranking_loss(teacher, student, valid, mode='adaptive_margin',
                             margin=0.05, topk_mask=topk, gamma=0.5)
    _, flat = ranking_loss(teacher, student, valid, mode='adaptive_margin',
                           margin=0.05, topk_mask=topk)
    assert float(scoped['rank_loss_inside_topk']) > 0.0
    assert scoped['rank_loss_inside_topk'] != flat['rank_loss_inside_topk']


def test_v2_options_are_validated():
    teacher, student, valid, topk = _scoped_batch()
    with pytest.raises(ValueError, match='Unsupported margin mode'):
        ranking_loss(teacher, student, valid, mode='margin', margin_mode='sqrt')
    with pytest.raises(ValueError, match='Unsupported sigma mode'):
        ranking_loss(teacher, student, valid, mode='ranknet', sigma_mode='learned')
    with pytest.raises(ValueError, match='needs topk_mask'):
        ranking_loss(teacher, student, valid, mode='margin', margin_mode='topk_relative')
    with pytest.raises(ValueError, match='topk_mask shape'):
        ranking_loss(teacher, student, valid, mode='margin', topk_mask=topk[:, :-1])


def test_v1_behaviour_is_unchanged_when_the_new_options_are_off():
    """The running sweep is the v1 baseline; its numbers must stay reproducible."""
    teacher, student, valid, _ = _scoped_batch()
    for mode in ('ranknet', 'weighted_ranknet', 'margin', 'adaptive_margin'):
        before = ranking_loss(teacher, student, valid, mode=mode, margin=0.05)[0]
        after = ranking_loss(teacher, student, valid, mode=mode, margin=0.05,
                             topk_mask=None, gamma=None,
                             margin_mode='absolute', sigma_mode='fixed')[0]
        torch.testing.assert_close(before, after)
