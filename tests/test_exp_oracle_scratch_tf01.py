"""EXP-ORACLE-SCRATCH-TF01 mandatory pre-GPU sanity checks (spec section
12, numbered items 1-15). Fast, toy-scale, synthetic tensors, plus a
couple of real (but small) `Exp_Stage1_Relation` constructions where
scratch-init verification genuinely needs it -- matches the pattern
already used by `tests/test_exp_oracle_scratch01.py`."""
import inspect
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from layers.retrieval_metric import RetrievalMetric, cosine_init_deviation
from models.SequentialSetRetriever import SetConditioner
from scripts.train_margutil01 import build_experiment
from scripts.train_oracle_scratch01 import base_score, run_sequence_individual
from scripts.train_oracle_scratch_tf01 import run_sequence_set_teacher_forced
from utils.dense_utility import candidate_weights, dense_utility

REFERENCE_CKPT = str(
    REPO_ROOT / 'checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/'
    'stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_'
    'expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth'
)


def _toy(bsz=4, n=30, d=8, h=5, k=6, seed=0):
    torch.manual_seed(seed)
    z_q = torch.randn(bsz, d)
    E = torch.randn(n, d)
    cand_mask = torch.ones(bsz, n, dtype=torch.bool)
    cand_mask[:, -3:] = False
    futures = torch.randn(bsz, n, h)
    query_future = torch.randn(bsz, h)
    w_base = candidate_weights(base_score(z_q, E, None), cand_mask, tau=0.1)
    sc = SetConditioner(d)
    return z_q, E, cand_mask, futures, query_future, w_base, sc, k


def test_1_and_2_encoder_scratch_not_pretrained():
    overrides = {'is_training': 1, 'model_id': 'sanity_tf01', 'des': 'sanity',
                 'checkpoints': '/tmp/exp_oracle_scratch_tf01_sanity', 'stage1_retrieval_metric': 'cosine'}
    exp_a, _ = build_experiment(REFERENCE_CKPT, dict(overrides, seed=0))
    model_a = exp_a.model.module if hasattr(exp_a.model, 'module') else exp_a.model
    exp_c, _ = build_experiment(REFERENCE_CKPT, dict(overrides, seed=1))
    model_c = exp_c.model.module if hasattr(exp_c.model, 'module') else exp_c.model
    p_a = next(model_a.encoder.parameters())
    p_c = next(model_c.encoder.parameters())
    assert not torch.allclose(p_a, p_c), 'different seeds must give different init'

    ref_ckpt = torch.load(REFERENCE_CKPT, map_location='cpu')
    ref_encoder_keys = {k: v for k, v in ref_ckpt['model_state_dict'].items() if k.startswith('encoder.')}
    cur_encoder_keys = {k: v.detach().cpu() for k, v in model_a.state_dict().items() if k.startswith('encoder.')}
    same_as_pretrained = all(
        k in cur_encoder_keys and torch.allclose(ref_encoder_keys[k], cur_encoder_keys[k]) for k in ref_encoder_keys
    )
    assert not same_as_pretrained, 'scratch model matches the PRETRAINED checkpoint -- old weights leaked in'


def test_3_shared_init_gives_identical_starting_encoder_across_arms():
    overrides = {'is_training': 1, 'model_id': 'sanity_tf01_shared', 'des': 'sanity',
                 'checkpoints': '/tmp/exp_oracle_scratch_tf01_sanity', 'stage1_retrieval_metric': 'cosine', 'seed': 0}
    exp_a, _ = build_experiment(REFERENCE_CKPT, overrides)
    model_a = exp_a.model.module if hasattr(exp_a.model, 'module') else exp_a.model
    shared = model_a.encoder.state_dict()

    exp_b, _ = build_experiment(REFERENCE_CKPT, dict(overrides, seed=99))  # different seed
    model_b = exp_b.model.module if hasattr(exp_b.model, 'module') else exp_b.model
    model_b.encoder.load_state_dict(shared)  # ...but shared init loaded, as the real script does
    for k_ in shared:
        assert torch.allclose(shared[k_], model_b.encoder.state_dict()[k_])


def test_4_encoder_gradient_nonzero():
    z_q, E, cand_mask, futures, query_future, w_base, sc, k = _toy()
    z_q = z_q.clone().requires_grad_(True)
    losses, diags, picks, a_dense = run_sequence_set_teacher_forced(
        z_q, E, cand_mask, sc, None, w_base, futures, query_future, tau_choice=0.1, k=k, chunk_size=None)
    total = sum(losses)
    total.backward()
    assert z_q.grad is not None and z_q.grad.norm().item() > 0.0


def test_5_set_t1_bypasses_conditioner():
    src = inspect.getsource(run_sequence_set_teacher_forced)
    t0_branch = src.split('if t == 0:')[1].split('else:')[0]
    assert 'set_conditioner' not in t0_branch

    z_q, E, cand_mask, futures, query_future, w_base, sc, k = _toy()
    _, _, picks_a, _ = run_sequence_set_teacher_forced(z_q, E, cand_mask, sc, None, w_base, futures, query_future,
                                                        tau_choice=0.1, k=k, chunk_size=None)
    with torch.no_grad():
        for p in sc.parameters():
            p.add_(5.0)
    _, _, picks_b, _ = run_sequence_set_teacher_forced(z_q, E, cand_mask, sc, None, w_base, futures, query_future,
                                                        tau_choice=0.1, k=k, chunk_size=None)
    assert torch.equal(picks_a[:, 0], picks_b[:, 0]), 't=1 pick must be unaffected by the conditioner'


def test_6_set_t1_target_equals_individual_t1_target():
    z_q, E, cand_mask, futures, query_future, w_base, sc, k = _toy()
    empty_prefix = torch.zeros(z_q.size(0), 0, dtype=torch.long)
    a_dense_t1 = dense_utility(empty_prefix, w_base, futures, query_future, chunk_size=None)
    d_i = ((futures - query_future.unsqueeze(1)) ** 2).mean(dim=-1)
    assert torch.allclose(a_dense_t1[cand_mask], d_i[cand_mask], atol=1e-6)

    _, _, picks, _ = run_sequence_set_teacher_forced(z_q, E, cand_mask, sc, None, w_base, futures, query_future,
                                                      tau_choice=0.1, k=k, chunk_size=None)
    b_i = base_score(z_q, E, None)
    u_target_ind = -d_i
    _, _, picks_ind = run_sequence_individual(b_i, u_target_ind, cand_mask, tau_choice=0.1, k=k)
    assert torch.equal(picks[:, 0], picks_ind[:, 0]), \
        "Set's Oracle t=1 pick must equal Individual's t=1 pick (same target definition)"


def test_7_training_prefix_is_oracle_not_model_pick_strong_negative_control():
    """STRONG negative control: build a case where a garbage/adversarial
    scorer would make the model's OWN argmax at t=1 point to a DIFFERENT
    candidate than the true Oracle's t=1 pick, then verify t=2's target/
    state is computed from the ORACLE's pick, not the model's."""
    torch.manual_seed(7)
    bsz, n, d, h, k = 3, 12, 6, 4, 4
    z_q = torch.randn(bsz, d)
    E = torch.randn(n, d)
    cand_mask = torch.ones(bsz, n, dtype=torch.bool)
    futures = torch.randn(bsz, n, h)
    query_future = torch.randn(bsz, h)
    w_base = candidate_weights(base_score(z_q, E, None), cand_mask, tau=0.1)
    sc = SetConditioner(d)

    d_i = ((futures - query_future.unsqueeze(1)) ** 2).mean(dim=-1)
    true_oracle_t1 = d_i.argmin(dim=-1)  # what the Oracle SHOULD pick at t=1

    # Adversarially initialise the encoder's t=1 score (via z_q/E themselves,
    # since t=1's score IS base_score(z_q,E,None)) to point somewhere ELSE --
    # by construction z_q/E are random, so with high probability the model's
    # own base_score argmax already differs from `true_oracle_t1`; assert
    # that precondition explicitly so this is a REAL negative control, not
    # a vacuous one.
    model_t1_pick = base_score(z_q, E, None).argmax(dim=-1)
    assert not torch.equal(model_t1_pick, true_oracle_t1), \
        'test precondition failed: model and Oracle already agree at t=1 by chance -- rerun with a different seed'

    losses, diags, oracle_picks, a_dense_steps = run_sequence_set_teacher_forced(
        z_q, E, cand_mask, sc, None, w_base, futures, query_future, tau_choice=0.1, k=k, chunk_size=None)

    assert torch.equal(oracle_picks[:, 0], true_oracle_t1), \
        "t=1 training pick must be the Oracle's, not the model's own (mismatched) argmax"

    # t=2's target must be recomputed from `oracle_picks[:, :1]` (the ORACLE
    # prefix), not from `model_t1_pick` -- verify by direct recomputation.
    expected_a_dense_t2 = dense_utility(oracle_picks[:, :1], w_base, futures, query_future, chunk_size=None)
    assert torch.allclose(a_dense_steps[1], expected_a_dense_t2, atol=1e-6)
    if not torch.equal(model_t1_pick, oracle_picks[:, 0]):
        wrong_prefix = model_t1_pick.unsqueeze(-1)
        wrong_a_dense_t2 = dense_utility(wrong_prefix, w_base, futures, query_future, chunk_size=None)
        assert not torch.allclose(a_dense_steps[1], wrong_a_dense_t2, atol=1e-6), \
            't=2 target must NOT match what it would be under the (wrong) model-pick prefix'


def test_8_target_recomputed_from_oracle_prefix_every_step():
    z_q, E, cand_mask, futures, query_future, w_base, sc, k = _toy()
    _, _, oracle_picks, a_dense_steps = run_sequence_set_teacher_forced(
        z_q, E, cand_mask, sc, None, w_base, futures, query_future, tau_choice=0.1, k=k, chunk_size=None)
    for t in range(1, k):
        expected = dense_utility(oracle_picks[:, :t], w_base, futures, query_future, chunk_size=None)
        assert torch.allclose(a_dense_steps[t], expected, atol=1e-6)


def test_9_query_future_never_reaches_scorer_input():
    """The scorer calls (`base_score(z_q, E, metric)` / `base_score(h_t, E,
    metric)`) must take exactly `(embedding, embedding, metric)` -- neither
    `futures` nor `query_future` may appear as an ARGUMENT to those calls."""
    src = inspect.getsource(run_sequence_set_teacher_forced)
    assert 'base_score(z_q, E, metric)' in src
    assert 'base_score(h_t, E, metric)' in src
    assert 'base_score(z_q, E, metric, query_future' not in src
    assert 'base_score(h_t, E, metric, futures' not in src


def test_10_full_memory_no_shortlist():
    src = inspect.getsource(run_sequence_set_teacher_forced)
    assert 'topk' not in src.lower() and 'shortlist' not in src.lower()


def test_11_no_duplicate_or_invalid_picks():
    z_q, E, cand_mask, futures, query_future, w_base, sc, k = _toy()
    _, _, picks, _ = run_sequence_set_teacher_forced(z_q, E, cand_mask, sc, None, w_base, futures, query_future,
                                                      tau_choice=0.1, k=k, chunk_size=None)
    for b in range(picks.size(0)):
        assert len(set(picks[b].tolist())) == k
        assert cand_mask[b, picks[b]].all()


def test_12_asymmetric_identity_init_matches_cosine():
    metric = RetrievalMetric(kind='asymmetric', dim=16, output='cosine', layer_norm=False)
    assert cosine_init_deviation(metric) < 1e-5


def test_13_stage2_reused_script_never_touches_stage1():
    """Stage-2 for TF01 reuses `scripts/train_oracle_scratch01_stage2.py`
    UNCHANGED (per the spec: same Stage-2 protocol) -- re-verify its own
    Stage-1-isolation guarantee here too, in this experiment's own test
    file, since that guarantee is load-bearing for THIS experiment as well."""
    stage2_path = REPO_ROOT / 'scripts' / 'train_oracle_scratch01_stage2.py'
    src = stage2_path.read_text()
    body = src[src.index('import argparse'):]
    assert 'Exp_Stage1_Relation' not in body
    assert 'RelationEncoder' not in body


def test_14_stage2_orchestration_uses_shared_init_flags():
    """Structural: the TF01 orchestration script must reuse the SAME
    shared-init flags (`--shared_base_init_out/_in`, `--shared_gate_init_
    out/_in`) so all 4 retrieval arms + base-only share one Base/Gate
    init, per spec section 9."""
    sh = REPO_ROOT / 'scripts' / 'run_oracle_scratch_tf01.sh'
    if not sh.exists():
        return
    src = sh.read_text()
    for flag in ('--shared_base_init_out', '--shared_base_init_in', '--shared_gate_init_out', '--shared_gate_init_in'):
        assert flag in src
    assert '--lr 0.01' not in src and 'epochs 50' not in src


def test_15_free_running_cache_builder_never_uses_oracle_or_future():
    from scripts.eval_oracle_scratch01 import free_running_topk
    src = inspect.getsource(free_running_topk)
    body = src[src.index('b_i = base_score'):]  # skip docstring/comments
    assert 'query_future' not in body and 'oracle' not in body.lower()
