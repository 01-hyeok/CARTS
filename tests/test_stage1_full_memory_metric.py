"""Guards for full-memory learnable retrieval metrics.

The previous attempt trained a pair scorer on 228 mined candidates and evaluated
it over 8449, and lost at short horizons. These metrics stay bilinear so the two
supports can be the same, and the single thing that must never regress is that
the softmax denominator really is the whole memory -- not a subset that quietly
reappeared.
"""

import pytest
import torch

from layers.retrieval_metric import (
    METRICS,
    EXPRESSIVENESS_LADDER,
    cosine_init_deviation,
    RetrievalMetric,
    oracle_rank_statistics,
    score_separation_metrics,
)


# ---------- the metrics stay a matrix multiplication ----------

@pytest.mark.parametrize('kind', METRICS)
def test_scoring_a_whole_bank_is_one_matmul_shaped_result(kind):
    metric = RetrievalMetric(kind=kind, dim=16)
    scores = metric.score(torch.randn(4, 16), torch.randn(500, 16))
    assert scores.shape == (4, 500)


@pytest.mark.parametrize('kind', METRICS)
def test_per_query_candidates_agree_with_bank_scoring(kind):
    """The subset path re-scores [B, M, D]; it must give the same numbers the
    bank path would for those candidates, or the scatter would mix two scores."""
    torch.manual_seed(0)
    metric = RetrievalMetric(kind=kind, dim=16).eval()
    z_q = torch.randn(3, 16)
    bank = torch.randn(40, 16)
    index = torch.tensor([[1, 7, 9], [0, 2, 5], [3, 4, 8]])
    with torch.no_grad():
        whole = metric.score(z_q, bank).gather(1, index)
        per_query = metric.score(z_q, bank[index])
    torch.testing.assert_close(per_query, whole)


def test_bilinear_and_asymmetric_start_at_the_incumbent_ranking():
    """Identity initialisation: the first step must rank like the dot product it
    replaces, not like a random projection."""
    torch.manual_seed(0)
    z_q, bank = torch.randn(3, 16), torch.randn(20, 16)
    plain = torch.matmul(z_q, bank.transpose(0, 1))
    for kind in ('bilinear', 'mahalanobis', 'asymmetric'):
        metric = RetrievalMetric(kind=kind, dim=16, layer_norm=False, scaled_dot=False).eval()
        with torch.no_grad():
            torch.testing.assert_close(metric.score(z_q, bank), plain, rtol=1e-5, atol=1e-5)


def test_scaled_dot_divides_by_sqrt_dim():
    """A tau tuned for cosine assumes scores in [-1, 1]; unnormalised dot
    products grow with width, so the scale is removed rather than the magnitude."""
    torch.manual_seed(0)
    z_q, bank = torch.randn(2, 64), torch.randn(10, 64)
    scaled = RetrievalMetric('bilinear', dim=64, layer_norm=False, scaled_dot=True).eval()
    plain = RetrievalMetric('bilinear', dim=64, layer_norm=False, scaled_dot=False).eval()
    with torch.no_grad():
        torch.testing.assert_close(scaled.score(z_q, bank) * (64 ** 0.5), plain.score(z_q, bank))


@pytest.mark.parametrize('kind', ['bilinear', 'mahalanobis', 'asymmetric'])
def test_metric_parameters_and_both_embedding_sides_receive_gradient(kind):
    metric = RetrievalMetric(kind=kind, dim=16)
    z_q = torch.randn(3, 16, requires_grad=True)
    bank = torch.randn(20, 16, requires_grad=True)
    metric.score(z_q, bank).sum().backward()
    assert z_q.grad.abs().sum() > 0
    assert bank.grad.abs().sum() > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in metric.parameters())


def test_asymmetric_learns_two_different_spaces():
    """A shared geometry is what this relaxes, so the two projections must be
    able to diverge. The perturbation has to be a rotation, not a rescaling: a
    uniform scale on one side is invisible under LayerNorm (and under the
    renormalised output), so scaling alone would not prove the spaces differ.
    """
    torch.manual_seed(0)
    metric = RetrievalMetric('asymmetric', dim=8)
    assert metric.query_projection is not None and metric.key_projection is not None
    assert metric.shared is False
    with torch.no_grad():
        metric.key_projection.weight.add_(torch.randn(8, 8) * 0.5)
    z_q = torch.randn(2, 8)
    with torch.no_grad():
        assert not torch.allclose(metric.project_query(z_q), metric.project_key(z_q))


def test_bilinear_folds_its_matrix_into_the_query_side_only():
    """So the bank can be pre-projected once instead of per query."""
    metric = RetrievalMetric('bilinear', dim=8)
    z = torch.randn(4, 8)
    torch.testing.assert_close(metric.project_key(z), z)


def test_unknown_metric_and_output_are_rejected():
    with pytest.raises(ValueError, match='Unsupported retrieval metric'):
        RetrievalMetric(kind='poincare', dim=8)
    with pytest.raises(ValueError, match='Unsupported metric output'):
        RetrievalMetric(kind='bilinear', dim=8, output='l2')


# ---------- Oracle rank diagnostics ----------

def test_oracle_rank_is_one_when_the_model_already_ranks_it_first():
    scores = torch.tensor([[9.0, 1.0, 0.5, 0.1]])
    valid = torch.ones(1, 4, dtype=torch.bool)
    stats = oracle_rank_statistics(scores, torch.tensor([[0]]), valid)
    assert float(stats['oracle_top10_mean_rank']) == pytest.approx(1.0)


def test_oracle_rank_shows_movement_recall_cannot_see():
    """A candidate going from rank 400 to rank 40 leaves Recall@10 at zero."""
    torch.manual_seed(0)
    valid = torch.ones(1, 500, dtype=torch.bool)
    oracle = torch.tensor([[399]])
    far = torch.linspace(1, 0, 500).unsqueeze(0)          # oracle sits at rank 400
    near = far.clone(); near[0, 399] = far[0, 39]         # moved to rank ~40
    assert float(oracle_rank_statistics(near, oracle, valid)['oracle_top10_mean_rank']) < \
           float(oracle_rank_statistics(far, oracle, valid)['oracle_top10_mean_rank'])


def test_invalid_candidates_never_occupy_a_rank():
    scores = torch.tensor([[0.1, 9.0, 0.2]])
    valid = torch.tensor([[True, False, True]])
    stats = oracle_rank_statistics(scores, torch.tensor([[0]]), valid)
    # Candidate 1 scores highest but is masked, so candidate 2 (0.2) is rank 1
    # and the Oracle at 0.1 is rank 2 -- not rank 3.
    assert float(stats['oracle_top10_mean_rank']) == pytest.approx(2.0)


def test_padded_oracle_slots_are_excluded():
    scores = torch.randn(2, 20)
    valid = torch.ones(2, 20, dtype=torch.bool)
    oracle = torch.tensor([[0, 1, 2], [3, 4, 5]])
    oracle_valid = torch.tensor([[True, True, False], [True, False, False]])
    stats = oracle_rank_statistics(scores, oracle, valid, oracle_valid)
    assert torch.isfinite(stats['oracle_top10_mean_rank'])


def test_score_separation_reports_the_oracle_advantage():
    scores = torch.tensor([[5.0, 5.0, 0.0, 0.0, 0.0]])
    valid = torch.ones(1, 5, dtype=torch.bool)
    stats = score_separation_metrics(scores, torch.tensor([[0, 1]]), valid, top_k=2)
    assert float(stats['oracle_top10_score_mean']) == pytest.approx(5.0)
    assert float(stats['non_oracle_score_mean']) == pytest.approx(0.0)
    assert float(stats['oracle_vs_nonoracle_score_gap']) == pytest.approx(5.0)


# ---------- the invariant that must not regress ----------

def _config(**overrides):
    from types import SimpleNamespace

    base = dict(
        seq_len=16, pred_len=8, enc_in=3, d_model=16, top_k=5,
        tau_student=0.1, tau_teacher=0.1, teacher_mse_space='normalized',
        source_mode='all', target_mode='all', target_channel=0,
        relation_input_space='delta_last', relation_teacher_space='delta_last',
        relation_encoder_type='mlp', relation_self_fill='linear',
        stage1_loss_mode='weighted_topk_ce', stage1_teacher_mode='mse',
        stage1_coverage_top_k=3, stage1_retrieval_metric='bilinear',
        n_heads=2, e_layers=1, d_ff=32, patch_len=8, stride=8, dropout=0.1,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.mark.parametrize('metric', ['cosine', 'mahalanobis', 'asymmetric', 'bilinear'])
def test_student_logits_span_the_whole_memory(metric):
    """The invariant. Training must not quietly shrink back to a mined subset --
    that mismatch is what the previous experiment is suspected to have failed on.
    """
    from models.RelationStage1 import Model

    n_memory = 24
    model = Model(_config(stage1_retrieval_metric=metric))
    model.relation_sources = [[c] for c in range(3)]
    model.train()

    captured = {}
    original = torch.log_softmax

    def spy(tensor, dim=-1, **kwargs):
        captured.setdefault('shape', tuple(tensor.shape))
        return original(tensor, dim=dim, **kwargs)

    torch.log_softmax = spy
    try:
        model(query_x=torch.randn(2, 16, 3), query_y=torch.randn(2, 8, 3),
              cand_mask=torch.ones(2, n_memory, dtype=torch.bool),
              memory_y=torch.randn(n_memory, 8, 3),
              key_bank=torch.randn(3, 3, n_memory, 16),
              memory_x_last=torch.randn(n_memory, 3))
    finally:
        torch.log_softmax = original
    assert captured['shape'] == (2, n_memory), (
        f'softmax denominator covered {captured["shape"]}, not the full memory')


def test_cosine_baseline_builds_no_metric_module():
    """Backward compatibility: the incumbent path is untouched."""
    from models.RelationStage1 import Model

    assert Model(_config(stage1_retrieval_metric='cosine')).retrieval_metric is None


def test_metric_parameters_are_registered_for_the_optimizer():
    from models.RelationStage1 import Model

    names = {n for n, p in Model(_config()).named_parameters() if p.requires_grad}
    assert any(n.startswith('retrieval_metric.') for n in names)
    assert any(n.startswith('encoder.') for n in names)


def test_unknown_gradient_mode_is_rejected():
    from models.RelationStage1 import Model

    with pytest.raises(ValueError, match='full_memory_gradient_mode'):
        Model(_config(stage1_full_memory_gradient_mode='half'))


# ---------------------------------------------------------------------------
# The two properties the arm comparison rests on. Both were wrong in the first
# draft of this experiment, so they are pinned rather than asserted in prose.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('kind', ['mahalanobis', 'asymmetric', 'bilinear'])
def test_identity_init_is_exactly_cosine_under_normalised_output(kind):
    """Arm 1 of the ladder must start where arm 0 sits.

    Identity-initialised projections are not enough on their own: the score also
    has to be renormalised afterwards. Under `output='dot'` the same identity
    weights produce an unnormalised dot product, which ranks differently from the
    baseline before a single gradient step -- so an epoch-1 gap would measure the
    initialisation instead of the learning.
    """
    torch.manual_seed(0)
    metric = RetrievalMetric(
        kind=kind, dim=32, layer_norm=False, scaled_dot=True, output='cosine')
    assert cosine_init_deviation(metric) == pytest.approx(0.0, abs=1e-6)

    z_q = torch.randn(5, 32)
    z_k = torch.randn(11, 32)
    expected = torch.matmul(
        torch.nn.functional.normalize(z_q, dim=-1),
        torch.nn.functional.normalize(z_k, dim=-1).T)
    assert torch.allclose(metric.score(z_q, z_k), expected, atol=1e-6)


@pytest.mark.parametrize('kind', ['mahalanobis', 'asymmetric', 'bilinear'])
def test_unnormalised_output_does_not_start_at_cosine(kind):
    """The negative control for the test above: this is the configuration the
    first draft of the experiment would have run, and it does not hold."""
    torch.manual_seed(0)
    metric = RetrievalMetric(
        kind=kind, dim=32, layer_norm=True, scaled_dot=True, output='dot')
    assert cosine_init_deviation(metric) > 1e-2


def test_mahalanobis_is_symmetric_and_asymmetric_is_not():
    """The ladder needs its two learned rungs to be genuinely different families.

    `bilinear` (z_q^T W z_i, W free) and `asymmetric` ((W_q z_q)^T (W_k z_i))
    span the same set of functions, because W_q^T W_k ranges over every D x D
    matrix -- pairing those two would have compared parameterisations, not
    expressive power. Mahalanobis ties both sides to one projection, so its W is
    symmetric PSD and swapping the arguments cannot change the score.
    """
    torch.manual_seed(0)
    z_a = torch.randn(4, 16)
    z_b = torch.randn(4, 16)

    shared = RetrievalMetric('mahalanobis', dim=16, layer_norm=False, output='cosine').eval()
    for parameter in shared.parameters():
        parameter.data.add_(torch.randn_like(parameter) * 0.3)
    assert torch.allclose(shared.score(z_a, z_b), shared.score(z_b, z_a).T, atol=1e-6)

    free = RetrievalMetric('asymmetric', dim=16, layer_norm=False, output='cosine').eval()
    with torch.no_grad():
        free.query_projection.weight.add_(torch.randn(16, 16) * 0.3)
        free.key_projection.weight.add_(torch.randn(16, 16) * 0.3)
    assert not torch.allclose(free.score(z_a, z_b), free.score(z_b, z_a).T, atol=1e-3)


def test_mahalanobis_uses_one_projection_for_both_sides():
    metric = RetrievalMetric('mahalanobis', dim=8, layer_norm=False, output='cosine')
    assert metric.shared is True
    assert metric.key_projection is None
    with torch.no_grad():
        metric.query_projection.weight.mul_(2.0)
    z = torch.randn(3, 8)
    # Both sides projected then renormalised -> the shared scaling cancels.
    assert torch.allclose(metric.score(z, z), metric.score(z, z).T, atol=1e-6)
    assert torch.allclose(
        metric.project_key(z), metric.project_query(z), atol=1e-6)


def test_normalised_output_removes_the_candidate_norm_confound():
    """Under `output='dot'` bilinear leaves the key unprojected and unnormalised,
    so a candidate can outrank another by being longer rather than by pointing
    anywhere useful -- something the cosine baseline structurally cannot do.
    Renormalising removes that channel, which is what makes the arms comparable.
    """
    torch.manual_seed(0)
    z_q = torch.randn(1, 16)
    z_k = torch.randn(2, 16)
    stretched = z_k.clone()
    stretched[1] *= 8.0

    raw = RetrievalMetric('bilinear', dim=16, layer_norm=False, output='dot').eval()
    grew = (raw.score(z_q, stretched) - raw.score(z_q, z_k)).abs().max()
    assert grew > 1e-3

    normalised = RetrievalMetric('bilinear', dim=16, layer_norm=False, output='cosine').eval()
    assert torch.allclose(
        normalised.score(z_q, stretched), normalised.score(z_q, z_k), atol=1e-6)


# ---------------------------------------------------------------------------
# The pair scorer over the whole memory. The earlier pairwise runs trained on a
# mined 228 and were evaluated over 8449, so a loss there could not be read as
# "the learned score is worse" rather than "it never saw what it was tested on".
# Materialising every pair was assumed too expensive; measured, it is ~2.4 GiB
# per target channel at N=8449, so the subset was not actually required.
# ---------------------------------------------------------------------------

def _pairwise_full_config(**overrides):
    base = dict(
        stage1_retrieval_metric='cosine',
        stage1_retrieval_score='pairwise_mlp',
        stage1_pairwise_feature='pair4',
        stage1_full_memory_gradient_mode='full_online',
    )
    base.update(overrides)
    return _config(**base)


def test_pairwise_full_memory_scores_every_candidate_in_the_softmax():
    """Same invariant as the metric arms: the denominator is the whole memory."""
    from models.RelationStage1 import Model

    n_memory = 24
    model = Model(_pairwise_full_config())
    assert model.pairwise_scorer is not None
    assert model.candidate_subset_active() is False, (
        'full-memory pairwise must not fall back to mining')
    model.relation_sources = [[c] for c in range(3)]
    model.train()

    captured = {}
    original = torch.log_softmax

    def spy(tensor, dim=-1, **kwargs):
        captured.setdefault('shape', tuple(tensor.shape))
        return original(tensor, dim=dim, **kwargs)

    torch.log_softmax = spy
    try:
        model(query_x=torch.randn(2, 16, 3), query_y=torch.randn(2, 8, 3),
              cand_mask=torch.ones(2, n_memory, dtype=torch.bool),
              memory_y=torch.randn(n_memory, 8, 3),
              candidate_x=torch.randn(n_memory, 16, 3),
              key_bank=torch.randn(3, 3, n_memory, 16),
              memory_x_last=torch.randn(n_memory, 3))
    finally:
        torch.log_softmax = original
    assert captured['shape'] == (2, n_memory), (
        f'pairwise softmax covered {captured["shape"]}, not the full memory')


def test_pairwise_full_memory_gives_the_encoder_candidate_side_gradient():
    """The point of full_online: candidates are re-encoded live, so the encoder
    is trained through the candidate branch and not only through the query."""
    from models.RelationStage1 import Model

    n_memory = 24
    torch.manual_seed(0)
    model = Model(_pairwise_full_config())
    model.relation_sources = [[c] for c in range(3)]
    model.train()
    # One fixed batch for both passes: the comparison below is only meaningful if
    # the sole difference between them is whether the candidate branch is cut.
    batch = dict(
        query_x=torch.randn(2, 16, 3), query_y=torch.randn(2, 8, 3),
        cand_mask=torch.ones(2, n_memory, dtype=torch.bool),
        memory_y=torch.randn(n_memory, 8, 3),
        candidate_x=torch.randn(n_memory, 16, 3),
        key_bank=torch.randn(3, 3, n_memory, 16),
        memory_x_last=torch.randn(n_memory, 3))
    # Dropout is active in train mode, so the two passes must be given the same
    # mask or the comparison measures sampling noise instead of the candidate
    # branch -- with this model the noise is larger than the effect.
    torch.manual_seed(1234)
    loss, metrics = model(**batch)[:2]
    loss.backward()
    assert float(metrics['full_memory_reencoded']) == pytest.approx(n_memory), (
        'full_online must re-encode every candidate, not a subset')

    scorer_grad = sum(
        p.grad.abs().sum() for p in model.pairwise_scorer.parameters() if p.grad is not None)
    live = sum(p.grad.abs().sum() for p in model.encoder.parameters() if p.grad is not None)
    assert float(scorer_grad) > 0, 'pair scorer received no gradient'
    assert float(live) > 0, 'encoder received no gradient'

    # A nonzero encoder gradient is not the claim -- the query branch alone would
    # produce one. Cut the candidate branch and re-run: if the total is unchanged,
    # candidates were re-encoded and their gradient discarded.
    model.zero_grad()
    original = model.encoder.forward

    def detach_candidates(relation_tensor, **kwargs):
        out = original(relation_tensor, **kwargs)
        if relation_tensor.size(0) != n_memory:
            return out
        return tuple(o.detach() for o in out) if isinstance(out, tuple) else out.detach()

    model.encoder.forward = detach_candidates
    try:
        torch.manual_seed(1234)
        detached_loss = model(**batch)[0]
        detached_loss.backward()
    finally:
        model.encoder.forward = original
    query_only = sum(
        p.grad.abs().sum() for p in model.encoder.parameters() if p.grad is not None)
    # Detaching changes only the backward pass, never the value.
    assert float(detached_loss) == pytest.approx(float(loss), rel=1e-6)
    assert float(query_only) < float(live), (
        'detaching the candidate branch left the encoder gradient unchanged, so the '
        'candidate side was not training the encoder')


def test_pairwise_full_memory_needs_the_candidate_histories():
    """full_online re-encodes from raw candidate pasts, so omitting memory_x is a
    configuration error rather than a silent fallback to the detached bank."""
    from models.RelationStage1 import Model

    model = Model(_pairwise_full_config())
    model.relation_sources = [[c] for c in range(3)]
    model.train()
    with pytest.raises(ValueError, match='candidate_x'):
        model(query_x=torch.randn(2, 16, 3), query_y=torch.randn(2, 8, 3),
              cand_mask=torch.ones(2, 24, dtype=torch.bool),
              memory_y=torch.randn(24, 8, 3),
              key_bank=torch.randn(3, 3, 24, 16),
              memory_x_last=torch.randn(24, 3))


@pytest.mark.parametrize('loss_mode', ['kl', 'topk_coverage', 'weighted_topk_ce',
                                       'kl_expected_mse'])
def test_full_online_is_reported_for_every_loss_mode(loss_mode):
    """A loss comparison is only about the loss if every arm re-encodes.

    The gradient path is loss-agnostic, but the metric that proves it ran was
    merged inside the coverage branch alone, so the KL arms trained under
    full_online while reporting nothing. A blank `full_memory_reencoded` reads as
    "this arm used the detached bank" -- the opposite of true, and enough to make
    a loss sweep look like a gradient-mode sweep.
    """
    from models.RelationStage1 import Model

    n_memory = 24
    torch.manual_seed(0)
    model = Model(_config(stage1_loss_mode=loss_mode,
                          stage1_retrieval_metric='cosine',
                          stage1_full_memory_gradient_mode='full_online'))
    model.relation_sources = [[c] for c in range(3)]
    model.train()
    _, metrics = model(
        query_x=torch.randn(2, 16, 3), query_y=torch.randn(2, 8, 3),
        cand_mask=torch.ones(2, n_memory, dtype=torch.bool),
        memory_y=torch.randn(n_memory, 8, 3),
        candidate_x=torch.randn(n_memory, 16, 3),
        key_bank=torch.randn(3, 3, n_memory, 16),
        memory_x_last=torch.randn(n_memory, 3))[:2]
    assert 'full_memory_reencoded' in metrics, (
        f'{loss_mode} ran under full_online without reporting it')
    assert float(metrics['full_memory_reencoded']) == pytest.approx(n_memory)
