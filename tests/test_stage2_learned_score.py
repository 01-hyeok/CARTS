"""Stage-2 must retrieve with the comparison Stage-1 was trained with.

Stage-2 scored candidates with a plain dot product and loaded only the encoder
from the Stage-1 checkpoint. A Stage-1 arm that learned an asymmetric metric or a
pair scorer therefore had that half of its retriever dropped at the bridge, and
its embeddings judged by a function they were never shaped for -- silently, since
the extra weights simply did not match any prefix.

These pin the three things that has to be true: the comparison is used, it is
loaded, and a missing half is an error rather than a random initialisation.
"""

from pathlib import Path

import pytest
import torch

from utils.retrieval_ops import retrieve_relation_future, reweight_selected_candidates


def _stage2_config(**overrides):
    """Build a Stage-2 config from run.py's own defaults.

    Hand-written namespaces drift: Stage-2 reads dozens of fields and a missing
    one surfaces as an AttributeError deep in __init__ rather than as a clear
    test failure. Taking the real parser's defaults keeps this in step with the
    CLI without restating it.
    """
    import argparse
    import re
    source = Path(__file__).resolve().parents[1] / 'run.py'
    lines = source.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if 'argparse.ArgumentParser' in l)
    end = next(i for i, l in enumerate(lines) if 'args = parser.parse_args()' in l)
    block = '\n'.join(l[4:] if l.startswith('    ') else l for l in lines[start:end])
    scope = {'argparse': argparse}
    exec(block, scope)
    args = scope['parser'].parse_args([
        '--task_name', 'stage2_relation', '--data', 'ETTh1',
        '--seq_len', '16', '--label_len', '0', '--pred_len', '8', '--enc_in', '3',
        '--d_model', '16', '--d_ff', '32', '--n_heads', '2', '--e_layers', '1',
        '--patch_len', '8', '--stride', '8', '--top_k', '3',
        '--source_mode', 'all', '--target_mode', 'all', '--target_channel', '0',
        '--relation_input_space', 'delta_last', '--relation_teacher_space', 'delta_last',
        '--relation_encoder_type', 'mlp', '--relation_self_fill', 'linear',
    ])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


# ---------- the shared ops honour an injected comparison ----------

def test_retrieve_uses_score_fn_instead_of_the_dot_product():
    torch.manual_seed(0)
    z_q = torch.randn(2, 4)
    z_mem = torch.randn(6, 4)
    values = torch.randn(6, 5)
    mask = torch.ones(2, 6, dtype=torch.bool)

    # A comparison that reverses the dot product's ordering: if it is ignored,
    # the selected candidates stay the ones cosine would have picked.
    flipped = lambda q, k: -torch.matmul(q, k.transpose(0, 1))
    _, _, idx_default, _, _ = retrieve_relation_future(
        z_q, z_mem, values, mask, top_k=3, tau_topk=0.1)
    _, _, idx_scored, _, _ = retrieve_relation_future(
        z_q, z_mem, values, mask, top_k=3, tau_topk=0.1, score_fn=flipped)
    assert not torch.equal(idx_default, idx_scored)


def test_reweight_uses_score_fn():
    torch.manual_seed(0)
    z_q = torch.randn(2, 4)
    z_k = torch.randn(2, 3, 4)
    values = torch.randn(2, 3, 5)
    valid = torch.ones(2, 3, dtype=torch.bool)
    flipped = lambda q, k: -(q.unsqueeze(1) * k).sum(-1)
    _, a_default, _ = reweight_selected_candidates(z_q, z_k, values, valid, 0.1)
    _, a_scored, _ = reweight_selected_candidates(
        z_q, z_k, values, valid, 0.1, score_fn=flipped)
    assert not torch.allclose(a_default, a_scored)


def test_score_fn_shape_mismatch_is_rejected():
    z_q, z_mem = torch.randn(2, 4), torch.randn(6, 4)
    with pytest.raises(ValueError, match='score_fn returned'):
        retrieve_relation_future(
            z_q, z_mem, torch.randn(6, 5), torch.ones(2, 6, dtype=torch.bool),
            top_k=3, tau_topk=0.1, score_fn=lambda q, k: torch.randn(2, 3))


def test_default_path_is_unchanged_without_score_fn():
    """Every pre-existing arm passes no score_fn and must be bit-identical."""
    torch.manual_seed(0)
    args = (torch.randn(2, 4), torch.randn(6, 4), torch.randn(6, 5),
            torch.ones(2, 6, dtype=torch.bool))
    a = retrieve_relation_future(*args, top_k=3, tau_topk=0.1)
    b = retrieve_relation_future(*args, top_k=3, tau_topk=0.1, score_fn=None)
    for x, y in zip(a[:4], b[:4]):
        assert torch.equal(x, y)


# ---------- Stage-2 builds and uses them ----------

@pytest.mark.parametrize('kind', ['mahalanobis', 'asymmetric'])
def test_stage2_builds_the_metric_and_routes_scoring_through_it(kind):
    from models.RelationStage2 import Model

    model = Model(_stage2_config(stage1_retrieval_metric=kind))
    assert model.retrieval_metric is not None
    fn = model._retrieval_score_fn()
    assert fn is not None
    z_q, z_k = torch.randn(2, 16), torch.randn(5, 16)
    assert fn(z_q, z_k).shape == (2, 5)


def test_stage2_builds_the_pair_scorer():
    from models.RelationStage2 import Model

    model = Model(_stage2_config(stage1_retrieval_score='pairwise_mlp',
                                 stage1_pairwise_feature='pair2'))
    assert model.pairwise_scorer is not None
    fn = model._retrieval_score_fn()
    model.eval()
    assert fn(torch.randn(2, 16), torch.randn(5, 16)).shape == (2, 5)


def test_cosine_stage2_builds_nothing_and_scores_with_the_dot_product():
    """Backward compatibility: the incumbent path is untouched."""
    from models.RelationStage2 import Model

    model = Model(_stage2_config())
    assert model.retrieval_metric is None
    assert model.pairwise_scorer is None
    assert model._retrieval_score_fn() is None


def test_two_comparisons_at_once_is_rejected():
    from models.RelationStage2 import Model

    with pytest.raises(ValueError, match='pick one'):
        Model(_stage2_config(stage1_retrieval_metric='asymmetric',
                             stage1_retrieval_score='pairwise_mlp'))


def test_missing_comparison_weights_are_an_error_not_a_random_init(tmp_path):
    """The failure this exists to prevent: a Stage-1 checkpoint without the
    comparison would leave a randomly initialised metric scoring trained
    embeddings, and nothing would say so."""
    from models.RelationStage2 import Model

    cfg = _stage2_config(stage1_retrieval_metric='asymmetric')
    model = Model(cfg)
    path = tmp_path / 'stage1.pth'
    # Config has to match, or the loader rejects the checkpoint before it ever
    # looks for the comparison weights -- which is a different failure.
    torch.save({'model_state_dict': {'encoder.dummy': torch.zeros(1),
                                     'shared_cross_projection.dummy': torch.zeros(1)},
                'args': {k: getattr(cfg, k) for k in (
                    'relation_encoder_type', 'relation_input_space', 'seq_len',
                    'pred_len', 'enc_in', 'source_mode', 'relation_graph_threshold',
                    'relation_top_n', 'relation_pooling', 'relation_self_fill')}}, path)
    with pytest.raises(RuntimeError, match='retrieval_metric.*no retrieval_metric'):
        model.load_stage1_checkpoint(str(path), strict=False)


# ---------------------------------------------------------------------------
# Joint training reads the key bank for selection, and the bank is built once
# per epoch. While the encoder moves inside that epoch, selection is made with
# embeddings it has already left behind.
# ---------------------------------------------------------------------------

def test_e2e_full_online_reencodes_every_step_not_the_bank():
    """Selection must see the live encoder, not the epoch snapshot.

    A bank frozen for an epoch means the candidate that would now rank first may
    never be looked at -- only the Top-K chosen by stale scores get re-encoded.
    With the flag on, a bank of pure noise must not change what is retrieved.
    """
    from models.RelationStage2 import Model

    torch.manual_seed(0)
    n_memory = 12
    cfg = _stage2_config(stage2_e2e=1, stage2_e2e_full_online=1,
                         freeze_stage1_encoder=0, top_k=3)
    model = Model(cfg)
    model.relation_sources = [[c] for c in range(3)]
    model.train()

    batch = dict(
        batch_x=torch.randn(2, 16, 3),
        memory_y=torch.randn(n_memory, 8, 3),
        valid_mask=torch.ones(2, n_memory, dtype=torch.bool),
        memory_x_last=torch.randn(n_memory, 3),
        candidate_x=torch.randn(n_memory, 16, 3),
    )
    # Two genuinely different banks. Scaling one bank would not do: a uniform
    # factor leaves the dot-product ordering, and therefore the Top-K, unchanged.
    bank_a = torch.randn(3, 3, n_memory, cfg.d_model)
    bank_b = torch.randn(3, 3, n_memory, cfg.d_model)
    torch.manual_seed(7)
    a = model(key_bank=bank_a, **batch)[0]
    torch.manual_seed(7)
    b = model(key_bank=bank_b, **batch)[0]
    assert torch.allclose(a, b, atol=1e-5), (
        'the key bank still influenced the forecast, so selection did not come '
        'from the live encoder')


def test_bank_is_used_when_full_online_is_off():
    """The negative control: without the flag the bank is what selection reads,
    so changing it must change the output."""
    from models.RelationStage2 import Model

    torch.manual_seed(0)
    n_memory = 12
    cfg = _stage2_config(stage2_e2e=1, stage2_e2e_full_online=0,
                         freeze_stage1_encoder=0, top_k=3)
    model = Model(cfg)
    model.relation_sources = [[c] for c in range(3)]
    model.train()
    batch = dict(
        batch_x=torch.randn(2, 16, 3),
        memory_y=torch.randn(n_memory, 8, 3),
        valid_mask=torch.ones(2, n_memory, dtype=torch.bool),
        memory_x_last=torch.randn(n_memory, 3),
        candidate_x=torch.randn(n_memory, 16, 3),
    )
    bank_a = torch.randn(3, 3, n_memory, cfg.d_model)
    bank_b = torch.randn(3, 3, n_memory, cfg.d_model)
    torch.manual_seed(7)
    a = model(key_bank=bank_a, **batch)[0]
    torch.manual_seed(7)
    b = model(key_bank=bank_b, **batch)[0]
    assert not torch.allclose(a, b, atol=1e-5)


def test_full_online_needs_candidate_histories():
    from models.RelationStage2 import Model

    cfg = _stage2_config(stage2_e2e=1, stage2_e2e_full_online=1,
                         freeze_stage1_encoder=0, top_k=3)
    model = Model(cfg)
    model.relation_sources = [[c] for c in range(3)]
    model.train()
    with pytest.raises(ValueError, match='candidate_x'):
        model(batch_x=torch.randn(2, 16, 3), memory_y=torch.randn(12, 8, 3),
              valid_mask=torch.ones(2, 12, dtype=torch.bool),
              memory_x_last=torch.randn(12, 3),
              key_bank=torch.randn(3, 3, 12, cfg.d_model))


# ---------------------------------------------------------------------------
# Top-K membership, not just the name of a function.
#
# Six sweeps were interpreted before anyone noticed the configured comparison
# never reached the selection call, because every check available at the time
# asked whether the module existed rather than whether it chose the candidates.
# These pin the indices forward() actually retrieved against the indices each
# scorer's own Top-K returns on the same inputs.
# ---------------------------------------------------------------------------

def _capture_selection(model, batch, monkeypatch):
    """Record the (z_q, z_mem, top_idx) forward() really selected with."""
    import models.RelationStage2 as m2
    real = m2.retrieve_relation_future
    seen = []

    def spy(**kw):
        out = real(**kw)
        seen.append({'z_q': kw['z_q'].detach(), 'z_mem': kw['z_mem'].detach(),
                     'valid_mask': kw['valid_mask'], 'top_idx': out[2].detach()})
        return out

    monkeypatch.setattr(m2, 'retrieve_relation_future', spy)
    model(**batch)
    assert seen, 'forward() never reached the retrieval selection call'
    return seen[0]


def _manual_topk(scores, valid_mask, k):
    masked = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min / 4)
    return masked.topk(k, dim=-1).indices


def _selection_case(monkeypatch, top_k=3, n_memory=12, **cfg_overrides):
    from models.RelationStage2 import Model

    torch.manual_seed(0)
    cfg = _stage2_config(freeze_stage1_encoder=1, top_k=top_k, **cfg_overrides)
    model = Model(cfg)
    model.relation_sources = [[c] for c in range(3)]
    model.eval()
    batch = dict(
        batch_x=torch.randn(2, 16, 3), memory_y=torch.randn(n_memory, 8, 3),
        valid_mask=torch.ones(2, n_memory, dtype=torch.bool),
        memory_x_last=torch.randn(n_memory, 3),
        candidate_x=torch.randn(n_memory, 16, 3),
        key_bank=torch.randn(3, 3, n_memory, cfg.d_model),
    )
    return model, _capture_selection(model, batch, monkeypatch)


def test_A_cosine_selection_matches_manual_cosine_topk(monkeypatch):
    model, got = _selection_case(monkeypatch)
    assert model._retrieval_score_fn() is None
    manual = _manual_topk(torch.matmul(got['z_q'], got['z_mem'].T),
                          got['valid_mask'], got['top_idx'].size(-1))
    assert torch.equal(got['top_idx'], manual)


def test_B_asymmetric_selection_matches_manual_asymmetric_topk(monkeypatch):
    """Rotate the projections first.

    At identity initialisation the asymmetric score *is* cosine by construction,
    so a default-init model satisfies this assertion whether or not the metric is
    wired in -- the test would pass against the bug it exists to catch.
    """
    from models.RelationStage2 import Model

    torch.manual_seed(0)
    n_memory = 12
    cfg = _stage2_config(freeze_stage1_encoder=1, top_k=3,
                         stage1_retrieval_metric='asymmetric',
                         stage1_metric_output='cosine', stage1_metric_layer_norm=0)
    model = Model(cfg)
    model.relation_sources = [[c] for c in range(3)]
    model.eval()
    with torch.no_grad():
        model.retrieval_metric.query_projection.weight.copy_(
            torch.linalg.qr(torch.randn(cfg.d_model, cfg.d_model))[0])
        model.retrieval_metric.key_projection.weight.copy_(
            torch.linalg.qr(torch.randn(cfg.d_model, cfg.d_model))[0])
    batch = dict(
        batch_x=torch.randn(2, 16, 3), memory_y=torch.randn(n_memory, 8, 3),
        valid_mask=torch.ones(2, n_memory, dtype=torch.bool),
        memory_x_last=torch.randn(n_memory, 3),
        candidate_x=torch.randn(n_memory, 16, 3),
        key_bank=torch.randn(3, 3, n_memory, cfg.d_model),
    )
    got = _capture_selection(model, batch, monkeypatch)
    with torch.no_grad():
        scores = model.retrieval_metric.score(got['z_q'], got['z_mem'])
    manual = _manual_topk(scores, got['valid_mask'], got['top_idx'].size(-1))
    assert torch.equal(got['top_idx'], manual)


def test_C_pair2_selection_matches_manual_pair_topk(monkeypatch):
    model, got = _selection_case(monkeypatch, stage1_retrieval_score='pairwise_mlp',
                                 stage1_pairwise_feature='pair2')
    with torch.no_grad():
        scores = model.pairwise_scorer.score_bank_in_chunks(got['z_q'], got['z_mem'])
    manual = _manual_topk(scores, got['valid_mask'], got['top_idx'].size(-1))
    assert torch.equal(got['top_idx'], manual)


def test_D_asymmetric_selection_differs_from_cosine(monkeypatch):
    """The wiring has to change membership, or the arm is a representation
    ablation wearing a metric's name. Identity init would score exactly cosine,
    so the projections are given a real rotation first."""
    from models.RelationStage2 import Model

    torch.manual_seed(0)
    n_memory = 12
    cfg = _stage2_config(freeze_stage1_encoder=1, top_k=3,
                         stage1_retrieval_metric='asymmetric',
                         stage1_metric_output='cosine', stage1_metric_layer_norm=0)
    model = Model(cfg)
    model.relation_sources = [[c] for c in range(3)]
    model.eval()
    with torch.no_grad():
        q_w = torch.linalg.qr(torch.randn(cfg.d_model, cfg.d_model))[0]
        model.retrieval_metric.query_projection.weight.copy_(q_w)
        model.retrieval_metric.key_projection.weight.copy_(
            torch.linalg.qr(torch.randn(cfg.d_model, cfg.d_model))[0])
    batch = dict(
        batch_x=torch.randn(2, 16, 3), memory_y=torch.randn(n_memory, 8, 3),
        valid_mask=torch.ones(2, n_memory, dtype=torch.bool),
        memory_x_last=torch.randn(n_memory, 3),
        candidate_x=torch.randn(n_memory, 16, 3),
        key_bank=torch.randn(3, 3, n_memory, cfg.d_model),
    )
    got = _capture_selection(model, batch, monkeypatch)
    cosine_top = _manual_topk(torch.matmul(got['z_q'], got['z_mem'].T),
                              got['valid_mask'], got['top_idx'].size(-1))
    assert not torch.equal(got['top_idx'], cosine_top), (
        'asymmetric selection returned exactly the cosine Top-K, so the metric '
        'is not deciding membership'
    )


def test_frozen_arm_does_not_train_the_comparison():
    """"Stage-1 frozen" has to include the comparison; it is half the retriever."""
    from models.RelationStage2 import Model

    for over in ({'stage1_retrieval_metric': 'asymmetric'},
                 {'stage1_retrieval_score': 'pairwise_mlp',
                  'stage1_pairwise_feature': 'pair2'}):
        model = Model(_stage2_config(freeze_stage1_encoder=1, **over))
        comparison = model.retrieval_metric or model.pairwise_scorer
        assert comparison is not None
        assert all(not p.requires_grad for p in comparison.parameters())


def test_selection_diagnostic_reports_and_rejects_a_mismatch(capsys):
    from models.RelationStage2 import Model

    model = Model(_stage2_config(stage1_retrieval_metric='asymmetric'))
    model.report_retrieval_selection()
    out = capsys.readouterr().out
    assert 'configured_metric         = asymmetric' in out
    assert 'actual_selection_score_fn = asymmetric' in out

    # Simulate the bug the diagnostic exists to catch: the comparison is gone
    # from the selection path while the config still names it.
    model.retrieval_metric = None
    with pytest.raises(AssertionError, match='not decide Top-K membership'):
        model.report_retrieval_selection()
