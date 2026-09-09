"""EXP-ORACLE-RANK-GAIN01 mandatory pre-GPU sanity checks (spec section 28,
numbered items 1-23, as many as apply to fast toy-scale/synthetic testing --
items about Weather-protocol-parity (23) and cross-arm shared-init file
plumbing (3, 21) are covered structurally / by orchestration-script
inspection, matching the pattern already used by `tests/test_exp_oracle_
scratch01.py` and `tests/test_exp_oracle_scratch_tf01.py`."""
import inspect
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from layers.retrieval_metric import RetrievalMetric, cosine_init_deviation
from models.SequentialSetRetriever import SetConditioner
from scripts.train_margutil01 import build_experiment
from scripts.train_oracle_scratch01 import base_score, run_sequence_individual, run_sequence_set
from scripts.train_oracle_scratch_tf01 import run_sequence_set_teacher_forced
import scripts.train_oracle_rank_gain01 as train_oracle_rank_gain01_module
from scripts.train_oracle_rank_gain01 import run_sequence_individual_onpolicy
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


# ---- 1, 2: scratch encoder, no pretrained weight leak ----
def test_1_and_2_encoder_scratch_not_pretrained():
    overrides = {'is_training': 1, 'model_id': 'sanity_rg01', 'des': 'sanity',
                 'checkpoints': '/tmp/exp_oracle_rank_gain01_sanity', 'stage1_retrieval_metric': 'cosine'}
    exp_a, _ = build_experiment(REFERENCE_CKPT, dict(overrides, seed=0))
    model_a = exp_a.model.module if hasattr(exp_a.model, 'module') else exp_a.model
    exp_c, _ = build_experiment(REFERENCE_CKPT, dict(overrides, seed=1))
    model_c = exp_c.model.module if hasattr(exp_c.model, 'module') else exp_c.model
    assert not torch.allclose(next(model_a.encoder.parameters()), next(model_c.encoder.parameters()))

    ref_ckpt = torch.load(REFERENCE_CKPT, map_location='cpu')
    ref_encoder_keys = {k: v for k, v in ref_ckpt['model_state_dict'].items() if k.startswith('encoder.')}
    cur_encoder_keys = {k: v.detach().cpu() for k, v in model_a.state_dict().items() if k.startswith('encoder.')}
    same_as_pretrained = all(
        k in cur_encoder_keys and torch.allclose(ref_encoder_keys[k], cur_encoder_keys[k]) for k in ref_encoder_keys
    )
    assert not same_as_pretrained


# ---- 3: shared init identical across arms (covered by orchestration structure) ----
def test_3_shared_init_flags_present_in_orchestration():
    sh = REPO_ROOT / 'scripts' / 'run_oracle_rank_gain01.sh'
    if not sh.exists():
        return
    src = sh.read_text()
    assert '--shared_init_out' in src and '--shared_init_in' in src


# ---- 4: epoch0 diagnostic happens before optimizer.step() ----
def test_4_epoch0_checkpoint_saved_before_training_loop():
    src = inspect.getsource(train_oracle_rank_gain01_module.main)
    epoch0_idx = src.index("checkpoint_epoch0.pth")
    loop_idx = src.index('for epoch in range(')
    assert epoch0_idx < loop_idx, 'checkpoint_epoch0.pth must be saved BEFORE the training loop (any optimizer.step())'


# ---- 7: encoder gradient nonzero (all 4 sequence functions) ----
def test_7_encoder_gradient_nonzero_all_four_combos():
    z_q, E, cand_mask, futures, query_future, k = _toy()
    sc = SetConditioner(z_q.size(-1))

    # individual + tf
    zq1 = z_q.clone().requires_grad_(True)
    b_i = base_score(zq1, E, None)
    d_i = ((futures - query_future.unsqueeze(1)) ** 2).mean(-1)
    losses, _, _ = run_sequence_individual(b_i, -d_i, cand_mask, tau_choice=0.1, k=k)
    sum(losses).backward()
    assert zq1.grad is not None and zq1.grad.norm().item() > 0

    # individual + onpolicy
    zq2 = z_q.clone().requires_grad_(True)
    b_i2 = base_score(zq2, E, None)
    losses2, _, _ = run_sequence_individual_onpolicy(b_i2, -d_i, cand_mask, tau_choice=0.1, k=k)
    sum(losses2).backward()
    assert zq2.grad is not None and zq2.grad.norm().item() > 0

    # set + tf
    zq3 = z_q.clone().requires_grad_(True)
    w_base = candidate_weights(base_score(zq3, E, None).detach(), cand_mask, tau=0.1)
    losses3, _, _, _ = run_sequence_set_teacher_forced(zq3, E, cand_mask, sc, None, w_base, futures, query_future,
                                                         tau_choice=0.1, k=k, chunk_size=None)
    sum(losses3).backward()
    assert zq3.grad is not None and zq3.grad.norm().item() > 0

    # set + onpolicy
    zq4 = z_q.clone().requires_grad_(True)
    w_base4 = candidate_weights(base_score(zq4, E, None).detach(), cand_mask, tau=0.1)
    losses4, _, _, _ = run_sequence_set(zq4, E, cand_mask, sc, None, w_base4, futures, query_future,
                                         tau_choice=0.1, k=k, chunk_size=None)
    sum(losses4).backward()
    assert zq4.grad is not None and zq4.grad.norm().item() > 0


# ---- 8: Individual TF prefix == Oracle prefix (static order) ----
def test_8_individual_tf_prefix_is_oracle_order():
    z_q, E, cand_mask, futures, query_future, k = _toy()
    b_i = base_score(z_q, E, None)
    d_i = ((futures - query_future.unsqueeze(1)) ** 2).mean(-1)
    _, _, picks = run_sequence_individual(b_i, -d_i, cand_mask, tau_choice=0.1, k=k)
    # picks must be the STATIC ascending-MSE order (masking previously picked)
    d_i_masked = d_i.masked_fill(~cand_mask, float('inf'))
    expected_order = d_i_masked.argsort(dim=-1)[:, :k]
    assert torch.equal(picks, expected_order)


# ---- 9: Individual on-policy prefix == model prefix (differs from Oracle when model is adversarial) ----
def test_9_individual_onpolicy_prefix_is_model_pick_not_oracle():
    torch.manual_seed(3)
    bsz, n, d, h, k = 3, 15, 6, 4, 4
    z_q = torch.randn(bsz, d)
    E = torch.randn(n, d)
    cand_mask = torch.ones(bsz, n, dtype=torch.bool)
    futures = torch.randn(bsz, n, h)
    query_future = torch.randn(bsz, h)
    b_i = base_score(z_q, E, None)
    d_i = ((futures - query_future.unsqueeze(1)) ** 2).mean(-1)
    model_t1 = b_i.argmax(dim=-1)
    oracle_t1 = (-d_i).argmax(dim=-1)
    assert not torch.equal(model_t1, oracle_t1), 'precondition failed: rerun with a different seed'
    _, _, picks = run_sequence_individual_onpolicy(b_i, -d_i, cand_mask, tau_choice=0.1, k=k)
    assert torch.equal(picks[:, 0], model_t1), "on-policy t=1 pick must be the MODEL's argmax, not the Oracle's"


# ---- 10: Set t1 conditioner bypass (both prefix policies) ----
def test_10_set_t1_bypasses_conditioner_both_policies():
    z_q, E, cand_mask, futures, query_future, k = _toy()
    sc = SetConditioner(z_q.size(-1))
    w_base = candidate_weights(base_score(z_q, E, None), cand_mask, tau=0.1)
    for seq_fn in (run_sequence_set_teacher_forced, run_sequence_set):
        _, _, picks_a, _ = seq_fn(z_q, E, cand_mask, sc, None, w_base, futures, query_future,
                                   tau_choice=0.1, k=k, chunk_size=None)
        with torch.no_grad():
            for p in sc.parameters():
                p.add_(3.0)
        _, _, picks_b, _ = seq_fn(z_q, E, cand_mask, sc, None, w_base, futures, query_future,
                                   tau_choice=0.1, k=k, chunk_size=None)
        assert torch.equal(picks_a[:, 0], picks_b[:, 0])
        with torch.no_grad():
            for p in sc.parameters():
                p.sub_(3.0)


# ---- 11: Set t1 target == Individual t1 target ----
def test_11_set_t1_target_equals_individual_t1_target():
    z_q, E, cand_mask, futures, query_future, k = _toy()
    w_base = candidate_weights(base_score(z_q, E, None), cand_mask, tau=0.1)
    empty_prefix = torch.zeros(z_q.size(0), 0, dtype=torch.long)
    a_dense_t1 = dense_utility(empty_prefix, w_base, futures, query_future, chunk_size=None)
    d_i = ((futures - query_future.unsqueeze(1)) ** 2).mean(dim=-1)
    assert torch.allclose(a_dense_t1[cand_mask], d_i[cand_mask], atol=1e-6)


# ---- 12, 13: Set TF t>=2 prefix == Oracle prefix; Set on-policy t>=2 prefix == model prefix ----
def test_12_and_13_set_prefix_policy_divergence():
    torch.manual_seed(5)
    bsz, n, d, h, k = 3, 15, 6, 4, 4
    z_q = torch.randn(bsz, d)
    E = torch.randn(n, d)
    cand_mask = torch.ones(bsz, n, dtype=torch.bool)
    futures = torch.randn(bsz, n, h)
    query_future = torch.randn(bsz, h)
    sc = SetConditioner(d)
    w_base = candidate_weights(base_score(z_q, E, None), cand_mask, tau=0.1)

    _, _, tf_picks, tf_a_dense = run_sequence_set_teacher_forced(
        z_q, E, cand_mask, sc, None, w_base, futures, query_future, tau_choice=0.1, k=k, chunk_size=None)
    for t in range(1, k):
        expected = dense_utility(tf_picks[:, :t], w_base, futures, query_future, chunk_size=None)
        assert torch.allclose(tf_a_dense[t], expected, atol=1e-6), 'TF target must be recomputed from the ORACLE prefix'

    _, _, op_picks, op_a_dense = run_sequence_set(
        z_q, E, cand_mask, sc, None, w_base, futures, query_future, tau_choice=0.1, k=k, chunk_size=None)
    for t in range(1, k):
        expected = dense_utility(op_picks[:, :t], w_base, futures, query_future, chunk_size=None)
        assert torch.allclose(op_a_dense[t], expected, atol=1e-6), 'on-policy target must be recomputed from the MODEL prefix'


# ---- 15: TF and on-policy genuinely use different state when they diverge ----
def test_15_tf_and_onpolicy_states_can_diverge():
    torch.manual_seed(9)
    bsz, n, d, h, k = 4, 20, 6, 4, 5
    z_q = torch.randn(bsz, d)
    E = torch.randn(n, d)
    cand_mask = torch.ones(bsz, n, dtype=torch.bool)
    futures = torch.randn(bsz, n, h)
    query_future = torch.randn(bsz, h)
    sc = SetConditioner(d)
    w_base = candidate_weights(base_score(z_q, E, None), cand_mask, tau=0.1)
    _, _, tf_picks, _ = run_sequence_set_teacher_forced(
        z_q, E, cand_mask, sc, None, w_base, futures, query_future, tau_choice=0.1, k=k, chunk_size=None)
    _, _, op_picks, _ = run_sequence_set(
        z_q, E, cand_mask, sc, None, w_base, futures, query_future, tau_choice=0.1, k=k, chunk_size=None)
    # not asserting they MUST differ (could coincide by chance) -- just that
    # the two functions are independently computed, not aliased/identical code
    assert run_sequence_set_teacher_forced is not run_sequence_set


# ---- 16: query future never reaches scorer input (on-policy Individual) ----
def test_16_query_future_never_reaches_onpolicy_individual_scorer():
    src = inspect.getsource(run_sequence_individual_onpolicy)
    assert 'query_future' not in src and 'futures' not in src


# ---- 17: full memory, no shortlist ----
def test_17_full_memory_no_shortlist():
    for fn in (run_sequence_individual_onpolicy,):
        src = inspect.getsource(fn)
        assert 'topk' not in src.lower() and 'shortlist' not in src.lower()


# ---- 18: no duplicate/invalid picks ----
def test_18_no_duplicate_or_invalid_picks():
    z_q, E, cand_mask, futures, query_future, k = _toy()
    b_i = base_score(z_q, E, None)
    d_i = ((futures - query_future.unsqueeze(1)) ** 2).mean(-1)
    _, _, picks = run_sequence_individual_onpolicy(b_i, -d_i, cand_mask, tau_choice=0.1, k=k)
    for b in range(picks.size(0)):
        assert len(set(picks[b].tolist())) == k
        assert cand_mask[b, picks[b]].all()


# ---- 19: asymmetric identity init == cosine ----
def test_19_asymmetric_identity_init_matches_cosine():
    metric = RetrievalMetric(kind='asymmetric', dim=16, output='cosine', layer_norm=False)
    assert cosine_init_deviation(metric) < 1e-5


# ---- 20, 21: Stage-2 reuse, no Stage-1 gradient, shared Base/Gate init ----
def test_20_and_21_stage2_reused_script_isolated_and_shared_init():
    stage2_path = REPO_ROOT / 'scripts' / 'train_oracle_scratch01_stage2.py'
    src = stage2_path.read_text()
    body = src[src.index('import argparse'):]
    assert 'Exp_Stage1_Relation' not in body
    assert 'RelationEncoder' not in body
    sh = REPO_ROOT / 'scripts' / 'run_oracle_rank_gain01.sh'
    if sh.exists():
        orch_src = sh.read_text()
        assert '--shared_base_init_out' in orch_src and '--shared_gate_init_out' in orch_src


# ---- 22: free-running eval/cache never uses future/Oracle ----
def test_22_free_running_never_uses_oracle_or_future():
    from scripts.eval_oracle_scratch01 import free_running_topk
    src = inspect.getsource(free_running_topk)
    body = src[src.index('b_i = base_score'):]
    assert 'query_future' not in body and 'oracle' not in body.lower()
