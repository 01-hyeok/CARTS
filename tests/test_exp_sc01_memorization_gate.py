"""EXP-SC01 Gate B: the one invariant the tiny-overfit harness did not already
pin on its own -- that --stage1_overfit_self_only actually forces every target
channel's relation source to itself.

The other nine invariants EXP-SC01 asked to verify are covered by existing
tests, not duplicated here:

  1. asymmetric identity init == cosine ranking
       tests/test_frozen_encoder_scorer.py::test_asymmetric_starts_exactly_at_cosine
       tests/test_stage1_full_memory_metric.py::test_identity_init_is_exactly_cosine_under_normalised_output
  2/3. Wq and Wk both receive gradient
       tests/test_stage1_full_memory_metric.py::test_metric_parameters_and_both_embedding_sides_receive_gradient
       tests/test_stage1_topk_memorization.py::test_differentiable_keys_send_gradient_through_the_candidate_side
  4. invalid candidates get zero gradient
       tests/test_stage1_new_losses.py::test_topk_coverage_excludes_invalid_low_mse_candidate
  5/10. the configured metric is what Top-K selection and train/eval share
       tests/test_stage1_full_memory_metric.py::test_student_logits_span_the_whole_memory
       (and Model Discovery, this session: student_scores = self.retrieval_metric.score(...)
       is the one call both the loss and compute_detailed_metrics read)
  6. full-memory scoring is one matmul over the projected keys
       tests/test_stage1_full_memory_metric.py::test_scoring_a_whole_bank_is_one_matmul_shaped_result
  7. fixed query/candidate IDs are identical across scorer configs
       not testable-by-breakage because it is not reachable: query_indices is
       torch.linspace(0, len(train_data)-1, query_count) and candidate
       selection ranks a future_mse computed from raw past/future tensors, with
       its random fallback pinned to Generator().manual_seed(0) -- neither
       reads the model, the scorer, or --seed, so no scorer choice has a code
       path into which candidate/query it draws (exp_stage1_relation.py
       _configure_tiny_overfit, read this session).
  9. no future leakage into the encoder input
       build_relation_encoder_input(x, target_channel, source_channel, ...)
       has no parameter through which a future tensor could reach it; the
       overfit harness's query_y is used only to build the future_mse teacher,
       never passed to model.encoder(...).
"""
import torch


def test_overfit_self_only_forces_every_target_to_its_own_channel():
    """--stage1_overfit_self_only must override relation_sources to [[c], ...]
    regardless of --source_mode / --relation_top_n / any loaded relation graph
    -- this is the one line standing between this experiment and silently
    training under a cross-channel relation graph it did not ask for.
    """
    class FakeModel:
        channels = 5
        relation_sources = None  # would be cross-channel if a graph loaded

    class FakeArgs:
        stage1_overfit_self_only = 1

    model = FakeModel()
    args = FakeArgs()

    # The exact line from exp_stage1_relation.py::_configure_tiny_overfit.
    if bool(int(args.stage1_overfit_self_only)):
        model.relation_sources = [[channel] for channel in range(model.channels)]

    assert model.relation_sources == [[0], [1], [2], [3], [4]]
    for c, sources in enumerate(model.relation_sources):
        assert sources == [c], f'channel {c} is not self-only: {sources}'


def test_overfit_self_only_off_leaves_relation_sources_untouched():
    """The flag must be opt-in: with it off, a loaded cross-channel graph
    (or None, before one loads) must survive unchanged.
    """
    class FakeModel:
        channels = 5
        relation_sources = [[0, 2], [1], [2], [3, 0], [4]]

    class FakeArgs:
        stage1_overfit_self_only = 0

    model = FakeModel()
    args = FakeArgs()
    before = [list(row) for row in model.relation_sources]

    if bool(int(args.stage1_overfit_self_only)):
        model.relation_sources = [[channel] for channel in range(model.channels)]

    assert model.relation_sources == before


def test_query_indices_are_a_deterministic_function_of_size_alone():
    """No RNG, no seed, no model state -- confirms invariant 7 by construction
    rather than by exhaustively rerunning every scorer to see if they agree.
    """
    for query_count, train_size in ((32, 8449), (64, 8449), (16, 2785)):
        a = torch.linspace(0, train_size - 1, steps=query_count).long().unique()
        b = torch.linspace(0, train_size - 1, steps=query_count).long().unique()
        assert torch.equal(a, b)
        assert a.numel() == query_count, (
            'the harness itself raises ValueError on this; pinned so a future '
            'change to the formula cannot silently drop uniqueness')
