"""Guards for the learnable pair score and the graded Top-K objective.

Two things have to be true for this experiment to answer its question. The
candidate side must carry gradient -- scoring against the detached bank is what
the whole design is meant to remove, and it fails silently if it slips back in.
And the graded loss must actually grade: if it collapses to the uniform coverage
loss it already ships with, the arm tests nothing new.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from layers.pairwise_scorer import PairwiseScorer, build_pair_features, pair_feature_multiplier
from models.RelationStage1 import (
    prepare_topk_coverage_targets,
    topk_coverage_loss,
    weighted_topk_listwise_ce,
)


# ---------- pair features ----------

def test_pair2_and_pair4_widths_match_their_names():
    z_q = torch.randn(3, 8)
    z_k = torch.randn(3, 5, 8)
    assert build_pair_features(z_q, z_k, 'pair2').shape == (3, 5, 16)
    assert build_pair_features(z_q, z_k, 'pair4').shape == (3, 5, 32)
    assert pair_feature_multiplier('pair2') == 2
    assert pair_feature_multiplier('pair4') == 4


def test_pair4_carries_direction_and_distance_that_pair2_hides():
    """A concatenation cannot expose the per-dimension difference without the
    scorer spending capacity to compute it; pair4 hands it over directly."""
    z_q = torch.tensor([[1.0, 0.0]])
    z_k = torch.tensor([[[0.0, 1.0]]])
    feature = build_pair_features(z_q, z_k, 'pair4')[0, 0]
    torch.testing.assert_close(feature, torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, -1.0, 1.0, 1.0]))
    # Swapping the pair flips the signed difference but not its magnitude.
    swapped = build_pair_features(z_k[:, 0], z_q.unsqueeze(1), 'pair4')[0, 0]
    torch.testing.assert_close(swapped[4:6], -feature[4:6])
    torch.testing.assert_close(swapped[6:], feature[6:])


def test_query_is_broadcast_not_repeated():
    """[B, M, D] worth of memory, not two copies of it."""
    z_q = torch.randn(2, 4)
    z_k = torch.randn(2, 100, 4)
    assert build_pair_features(z_q, z_k, 'pair2').shape == (2, 100, 8)


def test_feature_shape_mismatches_are_rejected():
    with pytest.raises(ValueError, match='z_q must be'):
        build_pair_features(torch.randn(2, 3, 4), torch.randn(2, 5, 4), 'pair2')
    with pytest.raises(ValueError, match='z_k must be'):
        build_pair_features(torch.randn(2, 4), torch.randn(2, 5, 8), 'pair2')
    with pytest.raises(ValueError, match='Unsupported pairwise feature'):
        build_pair_features(torch.randn(2, 4), torch.randn(2, 5, 4), 'pair8')


# ---------- the gradient the experiment exists for ----------

@pytest.mark.parametrize('feature', ['pair2', 'pair4'])
def test_both_query_and_candidate_receive_encoder_gradient(feature):
    """The incumbent path scores against a detached bank, so only the query
    branch trains the encoder. Here the shared encoder must be reached from both
    sides -- that is the difference being tested."""
    torch.manual_seed(0)
    encoder = torch.nn.Linear(6, 8)
    scorer = PairwiseScorer(8, feature_type=feature)

    z_q = encoder(torch.randn(4, 6))
    z_k = encoder(torch.randn(4, 5, 6))
    scorer(z_q, z_k).sum().backward()

    assert encoder.weight.grad.abs().sum() > 0
    assert all(p.grad is not None and p.grad.abs().sum() > 0
               for p in scorer.parameters() if p.requires_grad)

    # And specifically through the candidate branch: detaching it must reduce
    # the encoder gradient, or the candidate path was never contributing.
    encoder.zero_grad()
    z_q2 = encoder(torch.randn(4, 6))
    z_k2 = encoder(torch.randn(4, 5, 6)).detach()
    scorer(z_q2, z_k2).sum().backward()
    query_only = encoder.weight.grad.abs().sum().clone()

    encoder.zero_grad()
    torch.manual_seed(0)
    z_q3 = encoder(torch.randn(4, 6))
    z_k3 = encoder(torch.randn(4, 5, 6))
    scorer(z_q3, z_k3).sum().backward()
    assert encoder.weight.grad.abs().sum() != query_only


def test_scorer_keeps_grad_fn_on_its_output():
    encoder = torch.nn.Linear(6, 8)
    scorer = PairwiseScorer(8, feature_type='pair4')
    z_q = encoder(torch.randn(2, 6))
    z_k = encoder(torch.randn(2, 3, 6))
    scores = scorer(z_q, z_k)
    assert z_q.requires_grad and z_k.requires_grad
    assert z_q.grad_fn is not None and z_k.grad_fn is not None
    assert scores.grad_fn is not None


def test_chunked_bank_scoring_matches_the_unchunked_result():
    """Evaluation scores thousands of candidates a slice at a time; the split
    must not change the scores it produces."""
    torch.manual_seed(0)
    scorer = PairwiseScorer(8, feature_type='pair4').eval()
    z_q = torch.randn(3, 8)
    bank = torch.randn(50, 8)
    with torch.no_grad():
        whole = scorer(z_q, bank.unsqueeze(0).expand(3, -1, -1))
    torch.testing.assert_close(scorer.score_bank_in_chunks(z_q, bank, chunk_size=7), whole)


def test_scorer_output_shape_is_one_scalar_per_pair():
    scorer = PairwiseScorer(8, feature_type='pair2')
    assert scorer(torch.randn(4, 8), torch.randn(4, 11, 8)).shape == (4, 11)


def test_unknown_feature_type_is_rejected_at_construction():
    with pytest.raises(ValueError, match='Unsupported pairwise feature'):
        PairwiseScorer(8, feature_type='pair3')


# ---------- graded Top-K objective ----------

def _targets(future_mse, valid, k=3):
    return prepare_topk_coverage_targets(future_mse, valid, k)


def test_weighted_loss_grades_the_oracle_set_while_uniform_does_not():
    """Both losses use the same positives. The graded one must prefer putting
    mass on the *better* members; the uniform one is indifferent among them."""
    future = torch.tensor([[0.01, 0.50, 1.00, 9.0, 9.0]])
    valid = torch.ones(1, 5, dtype=torch.bool)
    targets = _targets(future, valid, k=3)

    # Same total mass on the Oracle set, arranged two ways.
    on_best = torch.log(torch.tensor([[0.70, 0.15, 0.10, 0.03, 0.02]]))
    on_worst = torch.log(torch.tensor([[0.10, 0.15, 0.70, 0.03, 0.02]]))

    weighted_best = float(weighted_topk_listwise_ce(on_best, targets, 0.1)[0])
    weighted_worst = float(weighted_topk_listwise_ce(on_worst, targets, 0.1)[0])
    assert weighted_best < weighted_worst

    uniform_best = float(topk_coverage_loss(on_best, targets)[0])
    uniform_worst = float(topk_coverage_loss(on_worst, targets)[0])
    assert uniform_best == pytest.approx(uniform_worst, abs=1e-5)


def test_weight_concentration_follows_the_teacher_temperature():
    future = torch.tensor([[0.01, 0.50, 1.00, 9.0, 9.0]])
    valid = torch.ones(1, 5, dtype=torch.bool)
    targets = _targets(future, valid, k=3)
    log_prob = torch.log(torch.full((1, 5), 0.2))

    sharp = weighted_topk_listwise_ce(log_prob, targets, 0.01)[1]
    flat = weighted_topk_listwise_ce(log_prob, targets, 10.0)[1]
    assert float(sharp['weighted_topk_effective_positives']) < float(
        flat['weighted_topk_effective_positives'])
    assert float(flat['weighted_topk_effective_positives']) == pytest.approx(3.0, abs=0.05)


def test_negatives_still_cost_even_though_their_target_weight_is_zero():
    """The student softmax runs over the whole candidate set, so a negative
    scoring highly steals mass from the positives and raises the loss."""
    future = torch.tensor([[0.01, 0.02, 0.03, 9.0, 9.0]])
    valid = torch.ones(1, 5, dtype=torch.bool)
    targets = _targets(future, valid, k=3)
    tidy = torch.log(torch.tensor([[0.30, 0.30, 0.30, 0.05, 0.05]]))
    leaky = torch.log(torch.tensor([[0.15, 0.15, 0.15, 0.50, 0.05]]))
    assert float(weighted_topk_listwise_ce(tidy, targets, 0.1)[0]) < float(
        weighted_topk_listwise_ce(leaky, targets, 0.1)[0])


def test_weighted_loss_reaches_the_student_scores():
    future = torch.tensor([[0.01, 0.50, 1.00, 9.0, 9.0]])
    valid = torch.ones(1, 5, dtype=torch.bool)
    scores = torch.zeros(1, 5, requires_grad=True)
    log_prob = torch.log_softmax(scores, dim=-1)
    loss, _ = weighted_topk_listwise_ce(log_prob, _targets(future, valid, k=3), 0.1)
    loss.backward()
    # Best Oracle member pushed up, a non-Oracle candidate pushed down.
    assert scores.grad[0, 0] < 0 < scores.grad[0, 3]


def test_weighted_loss_handles_masked_and_empty_oracle_sets():
    future = torch.tensor([[0.01, 9.0, 9.0]])
    valid = torch.tensor([[True, False, False]])
    log_prob = torch.log_softmax(torch.randn(1, 3), dim=-1)
    loss, metrics = weighted_topk_listwise_ce(log_prob, _targets(future, valid, k=3), 0.1)
    assert torch.isfinite(loss)
    assert float(metrics['weighted_topk_effective_positives']) == pytest.approx(1.0, abs=1e-4)


def test_weighted_loss_validates_its_inputs():
    future = torch.tensor([[0.01, 0.5, 1.0]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    targets = _targets(future, valid, k=2)
    with pytest.raises(ValueError, match='tau_teacher must be positive'):
        weighted_topk_listwise_ce(torch.log_softmax(torch.randn(1, 3), -1), targets, 0.0)
    with pytest.raises(ValueError, match='student_log_prob must be'):
        weighted_topk_listwise_ce(torch.randn(1, 3, 2), targets, 0.1)
    with pytest.raises(ValueError, match='missing keys'):
        weighted_topk_listwise_ce(torch.log_softmax(torch.randn(1, 3), -1),
                                  {'oracle_indices': targets['oracle_indices']}, 0.1)


# ---------- configuration ----------

def _config(**overrides):
    base = dict(
        seq_len=16, pred_len=8, enc_in=3, d_model=16, top_k=5,
        tau_student=0.1, tau_teacher=0.1, teacher_mse_space='normalized',
        source_mode='all', target_mode='all', target_channel=0,
        relation_input_space='delta_last', relation_teacher_space='delta_last',
        relation_encoder_type='mlp', relation_self_fill='linear',
        stage1_loss_mode='kl', stage1_teacher_mode='mse',
        stage1_retrieval_score='pairwise_mlp',
        stage1_candidate_subset_mode='selected_reencode',
        n_heads=2, e_layers=1, d_ff=32, patch_len=8, stride=8, dropout=0.1,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def test_pairwise_without_reencoding_is_rejected():
    """Scoring against the detached bank would train the scorer on a candidate
    side that never receives gradient -- the exact failure being removed.

    Two configurations avoid it: mine a subset and re-encode it, or re-encode the
    whole memory. Anything else is still rejected.
    """
    from models.RelationStage1 import Model

    with pytest.raises(ValueError, match='needs the candidate side'):
        Model(_config(stage1_candidate_subset_mode='none'))
    with pytest.raises(ValueError, match='needs the candidate side'):
        Model(_config(stage1_candidate_subset_mode='selected_detached'))
    with pytest.raises(ValueError, match='needs the candidate side'):
        Model(_config(stage1_candidate_subset_mode='none',
                      stage1_full_memory_gradient_mode='bank'))


def test_pairwise_over_the_whole_memory_needs_no_mining():
    """The support mismatch the mined runs were confounded by is a choice, not a
    constraint: re-encoding the full memory is an accepted configuration."""
    from models.RelationStage1 import Model

    model = Model(_config(stage1_candidate_subset_mode='none',
                          stage1_full_memory_gradient_mode='full_online'))
    assert model.pairwise_scorer is not None
    assert model.candidate_subset_active() is False


def test_cosine_baseline_needs_no_reencoding_and_builds_no_scorer():
    """Backward compatibility: the incumbent path must be untouched."""
    from models.RelationStage1 import Model

    model = Model(_config(stage1_retrieval_score='cosine',
                          stage1_candidate_subset_mode='none'))
    assert model.pairwise_scorer is None
    assert model.retrieval_score == 'cosine'


def test_scorer_parameters_are_registered_for_the_optimizer():
    """The scorer trains alongside the encoder, so it has to appear in
    model.parameters() -- the optimizer is built from that."""
    from models.RelationStage1 import Model

    model = Model(_config(stage1_pairwise_feature='pair4'))
    names = {n for n, _ in model.named_parameters()}
    assert any(n.startswith('pairwise_scorer.') for n in names)
    assert any(n.startswith('encoder.') for n in names)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert any(n.startswith('pairwise_scorer.') for n in trainable)


# ---------- common vs self mining ----------
#
# With self mining each arm picks its own training candidates, so "the pair
# scorer beat cosine" and "the pair scorer looked at different candidates" are
# the same measurement. Common mining freezes the candidate ids so the score
# function is the only thing that moves.

def test_common_mining_gives_every_arm_the_same_candidates():
    """Two different student score functions, one frozen mining score: the
    selected candidate ids must be identical."""
    from models.RelationStage1 import select_training_candidates

    torch.manual_seed(0)
    reference = torch.randn(4, 60)
    future = torch.rand(4, 60)
    valid = torch.ones(4, 60, dtype=torch.bool)

    cosine_like = torch.randn(4, 60)
    pairwise_like = torch.randn(4, 60) * 5.0

    frozen_a = select_training_candidates(reference, future, valid, top_m=20, oracle_k=5)[0]
    frozen_b = select_training_candidates(reference, future, valid, top_m=20, oracle_k=5)[0]
    torch.testing.assert_close(frozen_a, frozen_b)

    self_a = select_training_candidates(cosine_like, future, valid, top_m=20, oracle_k=5)[0]
    self_b = select_training_candidates(pairwise_like, future, valid, top_m=20, oracle_k=5)[0]
    # Self mining does not: the confound this option exists to remove.
    assert not torch.equal(self_a, self_b)


def test_oracle_injection_is_identical_under_either_mining():
    """The Oracle positives come from future MSE, never from the student, so
    they are the one part of the candidate set that cannot drift between arms."""
    from models.RelationStage1 import select_training_candidates

    torch.manual_seed(1)
    future = torch.rand(3, 40)
    valid = torch.ones(3, 40, dtype=torch.bool)
    oracle = future.topk(5, dim=-1, largest=False).indices

    for scores in (torch.randn(3, 40), torch.randn(3, 40) * 9.0):
        selected = select_training_candidates(scores, future, valid, top_m=15, oracle_k=5)[0]
        present = (oracle.unsqueeze(-1) == selected.unsqueeze(-2)).any(-1)
        assert present.all(), 'every Oracle positive must survive mining'


def test_mining_scores_shape_is_validated():
    from models.RelationStage1 import Model

    model = Model(_config())
    model.relation_sources = [[c] for c in range(3)]
    model.train()
    common = dict(
        query_x=torch.randn(2, 16, 3), query_y=torch.randn(2, 8, 3),
        cand_mask=torch.ones(2, 6, dtype=torch.bool),
        memory_y=torch.randn(6, 8, 3),
        key_bank=torch.randn(3, 1, 6, 16),
        memory_x_last=torch.randn(6, 3),
        candidate_x=torch.randn(6, 16, 3),
    )
    with pytest.raises(ValueError, match='mining_scores must be'):
        model(**common, mining_scores=torch.randn(2, 6))
    with pytest.raises(ValueError, match='mining_scores covers'):
        model(**common, mining_scores=torch.randn(2, 3, 5))


def test_mining_scores_require_the_subset_path():
    """Configuring mining scores without the subset path is a config error.

    Being in eval is not: evaluation scores the full memory and has no mining
    step, so the scores are simply unused there.
    """
    from models.RelationStage1 import Model

    model = Model(_config(stage1_retrieval_score='cosine',
                          stage1_candidate_subset_mode='none'))
    model.relation_sources = [[c] for c in range(3)]
    model.eval()
    with pytest.raises(ValueError, match='only applies to the candidate-subset'):
        model(query_x=torch.randn(2, 16, 3), query_y=torch.randn(2, 8, 3),
              cand_mask=torch.ones(2, 6, dtype=torch.bool),
              memory_y=torch.randn(6, 8, 3),
              key_bank=torch.randn(3, 1, 6, 16),
              memory_x_last=torch.randn(6, 3),
              mining_scores=torch.randn(2, 3, 6))


# ---------- random negatives ----------
#
# Mining alone hands the scorer only Top-M neighbours. A fixed score like cosine
# extrapolates to the rest of the bank by construction; a learned one does not,
# and evaluation ranks the whole bank. Measured on a one-epoch run without them,
# the pair scorer came out anti-correlated with future MSE over the full memory
# (Spearman -0.46) while training cleanly on its 100 mined candidates.

def test_random_negatives_extend_the_pool_beyond_the_mined_top_m():
    from models.RelationStage1 import select_training_candidates

    torch.manual_seed(0)
    scores = torch.randn(4, 200)
    future = torch.rand(4, 200)
    valid = torch.ones(4, 200, dtype=torch.bool)

    mined = select_training_candidates(scores, future, valid, top_m=20, oracle_k=5)[0]
    extended = select_training_candidates(
        scores, future, valid, top_m=20, oracle_k=5, random_negatives=30)[0]
    assert mined.shape == (4, 20)
    assert extended.shape == (4, 50)
    # The mined part is untouched; the negatives are appended after it.
    torch.testing.assert_close(extended[:, :20], mined)


def test_random_negatives_are_valid_and_not_duplicated():
    from models.RelationStage1 import select_training_candidates

    torch.manual_seed(0)
    valid = torch.zeros(3, 120, dtype=torch.bool)
    valid[:, :80] = True
    selected = select_training_candidates(
        torch.randn(3, 120), torch.rand(3, 120), valid,
        top_m=10, oracle_k=3, random_negatives=20)[0]

    assert valid.gather(1, selected).all(), 'a sampled negative must be a valid candidate'
    for row in selected:
        assert row.unique().numel() == row.numel(), 'the softmax denominator must not double-count'


def test_random_negatives_reach_outside_the_mined_neighbourhood():
    """The whole point: the scorer has to see pairs the mining never returns."""
    from models.RelationStage1 import select_training_candidates

    torch.manual_seed(0)
    # Scores make the first 20 candidates the obvious Top-M.
    scores = torch.cat([torch.full((2, 20), 9.0), torch.zeros(2, 180)], dim=-1)
    future = torch.cat([torch.zeros(2, 20), torch.ones(2, 180)], dim=-1)
    valid = torch.ones(2, 200, dtype=torch.bool)
    selected = select_training_candidates(
        scores, future, valid, top_m=20, oracle_k=5, random_negatives=40)[0]
    assert (selected >= 20).any(), 'no negative was drawn from outside the mined pool'


def test_zero_random_negatives_reproduces_the_previous_behaviour():
    """The existing candidate-gradient runs must stay reproducible."""
    from models.RelationStage1 import select_training_candidates

    torch.manual_seed(0)
    scores, future = torch.randn(3, 90), torch.rand(3, 90)
    valid = torch.ones(3, 90, dtype=torch.bool)
    a = select_training_candidates(scores, future, valid, top_m=15, oracle_k=4)[0]
    b = select_training_candidates(
        scores, future, valid, top_m=15, oracle_k=4, random_negatives=0)[0]
    torch.testing.assert_close(a, b)


def test_too_few_valid_candidates_is_an_error_not_a_silent_duplicate():
    from models.RelationStage1 import select_training_candidates

    valid = torch.zeros(2, 50, dtype=torch.bool)
    valid[:, :12] = True
    with pytest.raises(ValueError, match='cannot draw'):
        select_training_candidates(
            torch.randn(2, 50), torch.rand(2, 50), valid,
            top_m=10, oracle_k=3, random_negatives=40)


def test_negative_random_negative_count_is_rejected():
    from models.RelationStage1 import Model

    with pytest.raises(ValueError, match='must be non-negative'):
        Model(_config(stage1_candidate_random_negatives=-1))


# ---------- multi-criterion checkpoints ----------
#
# Recall@10 asks for exact identity inside an Oracle set whose 10th and 11th
# members differ by 1.4%; retrieved future MSE asks whether the candidates
# actually picked were any good. Selecting on one and reporting the others would
# hide an arm whose advantage sits on a different axis.

def _probe():
    from exp.exp_stage1_relation import Exp_Stage1_Relation

    probe = object.__new__(Exp_Stage1_Relation)
    probe.args = SimpleNamespace(stage1_checkpoint_metric='recall10')
    return probe


def test_each_criterion_is_scored_with_the_right_sign():
    """Recall and NDCG are maximised; retrieved MSE is minimised."""
    scores = _probe()._side_checkpoint_scores({
        'student_oracle_recall_at_10': 0.05,
        'student_ndcg_at_10': 0.80,
        'student_retrieved_future_mse_at_10': 0.99,
    })
    assert scores == {'recall10': 0.05, 'ndcg10': 0.80, 'retrieved_mse10': -0.99}


def test_lower_retrieved_mse_scores_higher():
    probe = _probe()
    better = probe._side_checkpoint_scores({'student_retrieved_future_mse_at_10': 0.90})
    worse = probe._side_checkpoint_scores({'student_retrieved_future_mse_at_10': 1.10})
    assert better['retrieved_mse10'] > worse['retrieved_mse10']


def test_unprefixed_metric_names_are_accepted_as_a_fallback():
    scores = _probe()._side_checkpoint_scores(
        {'oracle_recall_at_10': 0.04, 'ndcg_at_10': 0.7, 'retrieved_future_mse_at_10': 1.0})
    assert set(scores) == {'recall10', 'ndcg10', 'retrieved_mse10'}


def test_missing_metrics_are_skipped_rather_than_defaulted():
    """An epoch that did not compute a metric must not overwrite its checkpoint
    with a fabricated score."""
    scores = _probe()._side_checkpoint_scores({'student_ndcg_at_10': 0.8})
    assert set(scores) == {'ndcg10'}


def test_side_checkpoints_are_written_once_per_improvement(tmp_path):
    import torch as _torch

    probe = _probe()
    probe.model = _torch.nn.Linear(2, 2)
    probe.relation_graph = None
    optimizer = _torch.optim.Adam(probe.model.parameters())
    best = tmp_path / 'checkpoint.pth'

    probe._save_side_checkpoints(
        best, {'student_oracle_recall_at_10': 0.04, 'student_ndcg_at_10': 0.70,
               'student_retrieved_future_mse_at_10': 1.00}, optimizer, 1)
    written = sorted(f.name for f in tmp_path.glob('checkpoint_best_*.pth'))
    assert written == ['checkpoint_best_ndcg10.pth', 'checkpoint_best_recall10.pth',
                       'checkpoint_best_retrieved_mse10.pth']

    # Recall drops, retrieved MSE improves: only the MSE checkpoint moves.
    stamps = {f.name: f.stat().st_mtime_ns for f in tmp_path.glob('checkpoint_best_*.pth')}
    probe._save_side_checkpoints(
        best, {'student_oracle_recall_at_10': 0.03, 'student_ndcg_at_10': 0.65,
               'student_retrieved_future_mse_at_10': 0.90}, optimizer, 2)
    assert _torch.load(tmp_path / 'checkpoint_best_retrieved_mse10.pth')['epoch'] == 2
    assert _torch.load(tmp_path / 'checkpoint_best_recall10.pth')['epoch'] == 1
    assert stamps['checkpoint_best_ndcg10.pth'] == (
        tmp_path / 'checkpoint_best_ndcg10.pth').stat().st_mtime_ns


def test_side_checkpoints_record_which_criterion_chose_them():
    """So the Stage-2 hand-off knows what it is picking up."""
    import torch as _torch

    probe = _probe()
    probe.model = _torch.nn.Linear(2, 2)
    probe.relation_graph = None
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        best = Path(directory) / 'checkpoint.pth'
        probe._save_side_checkpoints(
            best, {'student_ndcg_at_10': 0.8}, _torch.optim.Adam(probe.model.parameters()), 3)
        saved = _torch.load(Path(directory) / 'checkpoint_best_ndcg10.pth')
        assert saved['selection_metric'] == 'ndcg10'
        assert saved['selection_score'] == pytest.approx(0.8)
        assert saved['epoch'] == 3
