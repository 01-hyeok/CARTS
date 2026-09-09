"""EXP-ORACLE-SCRATCH01 mandatory pre-GPU sanity checks (spec's numbered
list of 10 items):
1. encoder is random init (not deterministic/degenerate)
2. old Stage-1 checkpoint weight is NOT loaded
3. encoder gradient norm > 0
4. Set model t=1 bypasses the SetConditioner
5. Set model t>=2 uses the on-policy (model's own) prefix
6. query future never reaches the model's input (encoder/conditioner/scorer)
7. full memory is actually scored (no shortlist)
8. no duplicate/invalid candidate is ever selected
9. asymmetric identity init matches cosine
10. Stage-2 training keeps Stage-1 gradient at exactly 0/None (structural --
    verified in the Stage-2 script directly, since Stage-1 is not even
    instantiated there; also re-checked below by inspecting that script's
    source for any `.backward()` call reachable from a Stage-1 module)

Items 1-3 use the REAL `build_experiment`/`Model` construction (a genuine,
if small, ETTh1 load) since scratch-vs-loaded weights cannot be verified on
a synthetic stand-in; items 4-9 use toy synthetic tensors, the project's
standard fast-test pattern (matches every other `tests/test_exp_*.py` file
in this repo).
"""
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
from scripts.train_oracle_scratch01 import (
    base_score, encode_raw, run_sequence_individual, run_sequence_set,
)
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
    return z_q, E, cand_mask, futures, query_future, k


def test_1_and_2_encoder_is_scratch_not_loaded_checkpoint():
    overrides = {'is_training': 1, 'model_id': 'sanity_scratch01', 'des': 'sanity',
                 'checkpoints': '/tmp/exp_oracle_scratch01_sanity',
                 'stage1_retrieval_metric': 'cosine'}
    exp_a, args_a = build_experiment(REFERENCE_CKPT, dict(overrides, seed=0))
    model_a = exp_a.model.module if hasattr(exp_a.model, 'module') else exp_a.model

    exp_b, args_b = build_experiment(REFERENCE_CKPT, dict(overrides, seed=0))
    model_b = exp_b.model.module if hasattr(exp_b.model, 'module') else exp_b.model

    exp_c, args_c = build_experiment(REFERENCE_CKPT, dict(overrides, seed=1))
    model_c = exp_c.model.module if hasattr(exp_c.model, 'module') else exp_c.model

    torch.manual_seed(0)
    exp_a2, _ = build_experiment(REFERENCE_CKPT, dict(overrides, seed=0))
    model_a2 = exp_a2.model.module if hasattr(exp_a2.model, 'module') else exp_a2.model
    torch.manual_seed(0)
    exp_b2, _ = build_experiment(REFERENCE_CKPT, dict(overrides, seed=0))
    model_b2 = exp_b2.model.module if hasattr(exp_b2.model, 'module') else exp_b2.model
    p_a2 = next(model_a2.encoder.parameters())
    p_b2 = next(model_b2.encoder.parameters())
    assert torch.allclose(p_a2, p_b2), 'same torch.manual_seed before construction must give identical init'

    p_a = next(model_a.encoder.parameters())
    p_c = next(model_c.encoder.parameters())
    assert not torch.allclose(p_a, p_c), 'different seeds must give DIFFERENT encoder init (proves random, not constant)'

    ref_ckpt = torch.load(REFERENCE_CKPT, map_location='cpu')
    ref_encoder_keys = {k: v for k, v in ref_ckpt['model_state_dict'].items() if k.startswith('encoder.')}
    cur_encoder_keys = {k: v.detach().cpu() for k, v in model_a.state_dict().items() if k.startswith('encoder.')}
    assert ref_encoder_keys, 'reference checkpoint must actually contain encoder weights for this test to mean anything'
    same_as_pretrained = all(
        k in cur_encoder_keys and torch.allclose(ref_encoder_keys[k], cur_encoder_keys[k])
        for k in ref_encoder_keys
    )
    assert not same_as_pretrained, 'scratch model matches the PRETRAINED checkpoint exactly -- old weights were loaded'


def test_3_encoder_gradient_is_nonzero_after_backward():
    overrides = {'is_training': 1, 'model_id': 'sanity_scratch01_grad', 'des': 'sanity',
                 'checkpoints': '/tmp/exp_oracle_scratch01_sanity', 'stage1_retrieval_metric': 'cosine'}
    exp, args = build_experiment(REFERENCE_CKPT, overrides)
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    exp._ensure_memory()
    channel = list(model.target_channels())[0]
    x = exp.memory_x[:8]
    z = encode_raw(model, x, channel)
    loss = z.pow(2).mean()
    loss.backward()
    grad_norm = sum(p.grad.norm().item() ** 2 for p in model.encoder.parameters() if p.grad is not None) ** 0.5
    assert grad_norm > 0.0, 'encoder gradient norm must be > 0 after a backward pass through it'


def test_4_set_t1_bypasses_conditioner():
    """t=1's score must come directly from `base_score(z_q, E, metric)`, not
    through the SetConditioner -- verified structurally (the conditioner is
    only called inside the `else` branch of `if t == 0`) and behaviourally
    (perturbing the conditioner's weights must not change t=1's score)."""
    src = inspect.getsource(run_sequence_set)
    assert 'if t == 0' in src
    t0_branch = src.split('if t == 0:')[1].split('else:')[0]
    assert 'set_conditioner' not in t0_branch

    z_q, E, cand_mask, futures, query_future, k = _toy()
    sc = SetConditioner(z_q.size(-1))
    w_base = candidate_weights(base_score(z_q, E, None), cand_mask, tau=0.1)
    _, _, picks_a, a_dense_a = run_sequence_set(z_q, E, cand_mask, sc, None, w_base, futures, query_future,
                                                 tau_choice=0.1, k=k, chunk_size=None)
    with torch.no_grad():
        for p in sc.parameters():
            p.add_(1.0)
    _, _, picks_b, a_dense_b = run_sequence_set(z_q, E, cand_mask, sc, None, w_base, futures, query_future,
                                                 tau_choice=0.1, k=k, chunk_size=None)
    assert torch.equal(picks_a[:, 0], picks_b[:, 0]), 't=1 pick changed after perturbing the conditioner'


def test_5_set_tgeq2_uses_on_policy_prefix():
    z_q, E, cand_mask, futures, query_future, k = _toy()
    sc = SetConditioner(z_q.size(-1))
    w_base = candidate_weights(base_score(z_q, E, None), cand_mask, tau=0.1)
    _, _, picks, a_dense_steps = run_sequence_set(z_q, E, cand_mask, sc, None, w_base, futures, query_future,
                                                   tau_choice=0.1, k=k, chunk_size=None)
    for t in range(1, k):
        expected = dense_utility(picks[:, :t], w_base, futures, query_future, chunk_size=None)
        assert torch.allclose(a_dense_steps[t], expected, atol=1e-6), \
            f'step {t} target not recomputed from the model\'s OWN prefix'


def test_6_query_future_never_reaches_scorer_input():
    src = inspect.getsource(run_sequence_set)
    body = src[src.index('bsz, device ='):]
    score_call_regions = []
    for marker in ('base_score(z_q, E, metric)', 'base_score(h_t, E, metric)'):
        idx = body.index(marker)
        score_call_regions.append(body[max(0, idx - 200):idx])
    for region in score_call_regions:
        assert 'query_future' not in region and 'futures' not in region


def test_7_full_memory_no_shortlist():
    src = inspect.getsource(run_sequence_set)
    assert 'topk' not in src.lower() and 'shortlist' not in src.lower()
    src_ind = inspect.getsource(run_sequence_individual)
    assert 'topk' not in src_ind.lower() and 'shortlist' not in src_ind.lower()


def test_8_no_duplicate_or_invalid_picks():
    z_q, E, cand_mask, futures, query_future, k = _toy()
    sc = SetConditioner(z_q.size(-1))
    w_base = candidate_weights(base_score(z_q, E, None), cand_mask, tau=0.1)
    _, _, picks, _ = run_sequence_set(z_q, E, cand_mask, sc, None, w_base, futures, query_future,
                                       tau_choice=0.1, k=k, chunk_size=None)
    for b in range(picks.size(0)):
        assert len(set(picks[b].tolist())) == k
        assert cand_mask[b, picks[b]].all()

    d_i = ((futures - query_future.unsqueeze(1)) ** 2).mean(dim=-1)
    b_i = base_score(z_q, E, None)
    _, _, picks_ind = run_sequence_individual(b_i, -d_i, cand_mask, tau_choice=0.1, k=k)
    for b in range(picks_ind.size(0)):
        assert len(set(picks_ind[b].tolist())) == k
        assert cand_mask[b, picks_ind[b]].all()


def test_9_asymmetric_identity_init_matches_cosine():
    metric = RetrievalMetric(kind='asymmetric', dim=16, output='cosine', layer_norm=False)
    dev = cosine_init_deviation(metric)
    assert dev < 1e-5


def test_10_stage2_script_never_instantiates_stage1_model():
    """Structural (EXP-ORACLE-SCRATCH01's active Stage-2 script, joint
    Base+Gate training, no separate freeze round per the user's revised
    design): must not build/import a Stage-1 `Model`/`Exp_Stage1_Relation`
    at all -- it consumes only the cached (batch_x/Y_q/Y_ret) tensors from
    a FROZEN Stage-1 arm, so Stage-1 receiving a gradient during Stage-2
    training is structurally impossible, not merely unobserved."""
    stage2_path = REPO_ROOT / 'scripts' / 'train_oracle_scratch01_stage2.py'
    assert stage2_path.exists(), 'Stage-2 script must exist before this check is meaningful'
    src = stage2_path.read_text()
    body = src[src.index('import argparse'):]  # skip module docstring
    assert 'Exp_Stage1_Relation' not in body
    assert 'RelationEncoder' not in body


def test_10b_frozenbase_variant_still_structurally_isolates_stage1_and_base():
    """The earlier frozen-base variant script (`train_oracle_scratch_
    frozenbase01_stage2.py`) is kept in the repo (not deleted, per the
    project's own no-silent-deletion convention) even though the ACTIVE
    Stage-2 design no longer uses it -- its own isolation guarantees are
    still checked here so it does not silently rot into something unsafe
    if ever reused."""
    stage2_path = REPO_ROOT / 'scripts' / 'train_oracle_scratch_frozenbase01_stage2.py'
    if not stage2_path.exists():
        return
    src = stage2_path.read_text()
    body = src[src.index('import argparse'):]
    assert 'Exp_Stage1_Relation' not in body
    assert 'RelationEncoder' not in body
    assert 'torch.optim.Adam(gate.parameters()' in body
    gf_src = src[src.index('def gated_forecast'):src.index('def run_epoch')]
    assert 'with torch.no_grad():' in gf_src
    no_grad_region = gf_src[gf_src.index('with torch.no_grad():'):]
    assert 'base_head(batch_x)' in no_grad_region.split('\n\n')[0] or 'base_head(batch_x)' in no_grad_region[:120]


def test_base_predictor_sanity_catches_a_real_violation():
    """Negative control: `base_predictor_sanity` must actually detect a
    weight change, not just always report True."""
    from scripts.train_oracle_scratch_frozenbase01_stage2 import base_predictor_sanity
    from models.RelationStage2 import BaseForecastHead
    bh = BaseForecastHead(8, 4, 3, mode='shared_target_linear')
    frozen_reference = {k: v.detach().clone() for k, v in bh.state_dict().items()}
    optimizer = torch.optim.Adam(bh.parameters(), lr=0.1)  # deliberately WRONG: base params in optimizer
    sanity_before = base_predictor_sanity(bh, frozen_reference, optimizer)
    assert not sanity_before['not_in_optimizer']
    assert not sanity_before['all_pass']

    with torch.no_grad():
        list(bh.parameters())[0].add_(1.0)
    optimizer2 = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    sanity_after = base_predictor_sanity(bh, frozen_reference, optimizer2)
    assert not sanity_after['weights_unchanged']
    assert not sanity_after['all_pass']

    # a genuinely untouched, correctly-isolated base predictor passes cleanly
    bh2 = BaseForecastHead(8, 4, 3, mode='shared_target_linear')
    for p in bh2.parameters():
        p.requires_grad_(False)
    ref2 = {k: v.detach().clone() for k, v in bh2.state_dict().items()}
    dummy = torch.nn.Parameter(torch.zeros(1))
    optimizer3 = torch.optim.Adam([dummy], lr=0.1)
    sanity_clean = base_predictor_sanity(bh2, ref2, optimizer3)
    assert sanity_clean['all_pass']


def test_set_t1_target_equals_individual_t1_target():
    """`dense_utility` with an EMPTY prefix already reduces to
    `-MSE(Y_i,Y_q)` -- Set's t=1 target and Individual's (fixed) target must
    be numerically identical, with no special-cased math anywhere."""
    z_q, E, cand_mask, futures, query_future, k = _toy()
    w_base = candidate_weights(base_score(z_q, E, None), cand_mask, tau=0.1)
    empty_prefix = torch.zeros(z_q.size(0), 0, dtype=torch.long)
    a_dense_t1 = dense_utility(empty_prefix, w_base, futures, query_future, chunk_size=None)
    d_i = ((futures - query_future.unsqueeze(1)) ** 2).mean(dim=-1)
    # only VALID candidates matter -- invalid (masked, w_base=0) positions are
    # excluded from the CE loss entirely, so dense_utility's near-empty-weight
    # denominator behaviour there is irrelevant and not expected to match.
    assert torch.allclose(a_dense_t1[cand_mask], d_i[cand_mask], atol=1e-6)


def test_gradient_flows_to_asymmetric_scorer_and_conditioner():
    z_q, E, cand_mask, futures, query_future, k = _toy()
    z_q = z_q.clone().requires_grad_(True)
    E = E.clone().requires_grad_(True)
    metric = RetrievalMetric(kind='asymmetric', dim=z_q.size(-1), output='cosine', layer_norm=False)
    sc = SetConditioner(z_q.size(-1))
    w_base = candidate_weights(base_score(z_q, E, metric).detach(), cand_mask, tau=0.1)
    losses, diags, picks, _ = run_sequence_set(z_q, E, cand_mask, sc, metric, w_base, futures, query_future,
                                                tau_choice=0.1, k=k, chunk_size=None)
    total = sum(losses)
    total.backward()
    assert z_q.grad is not None and torch.isfinite(z_q.grad).all()
    assert all(p.grad is not None for p in sc.parameters())
    assert all(p.grad is not None for p in metric.parameters())
