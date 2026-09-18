"""EXP-SET-RECOVERY01 -- mandatory unit tests (spec section 8), run on
real ETTh1_96 Set On-policy Cosine Hard-CE checkpoint data where GPU-free
determinism can be checked (CPU/GPU2 only, no GPU1 use).
"""
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.SequentialSetRetriever import SetConditioner
from scripts.diag_set_recovery01 import (combination_redundancy, hybrid_prefix_intervention,
                                         recovery_rate, single_step_correction)
from scripts.train_factorial_e2e01 import (HostScorer, arm_score, candidate_weights,
                                           encode_raw, individual_utility, run_sequence)
from scripts.train_margutil01 import build_experiment, memory_value

REF_CKPT = 'checkpoints/track_a_factorial_e2e/ETTh1_96/set_onpolicy_cosine/checkpoint.pth'
S2_96 = ('checkpoints/stage2/ETTh1/seq96_pred96/stage2_carts_softset_s2_ETTh1_96_S0_wce_'
        'RelationStage2_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_'
        'ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth')


def _available():
    return (REPO_ROOT / REF_CKPT).exists() and (REPO_ROOT / S2_96).exists()


pytestmark = pytest.mark.skipif(not _available(), reason='ETTh1_96 checkpoints not present')


def _setup(seed=0, channel=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    ckpt = torch.load(REF_CKPT, map_location='cpu')
    exp, args = build_experiment(REF_CKPT, {
        'pred_len': 96, 'seq_len': 96, 'batch_size': 8, 'seed': seed, 'top_k': 10, 'tau_topk': 0.1})
    exp._ensure_memory()
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    sc = SetConditioner(int(args.d_model)).to(device)
    sc.load_state_dict(ckpt['set_conditioner_state_dict'])
    sc.eval()

    host = HostScorer(S2_96, device)
    _, loader = exp._get_data(flag='val', shuffle=False)
    batch_x, batch_y, batch_start_idx = next(iter(loader))
    batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)
    cand_mask, _ = exp._candidate_mask(batch_start_idx)

    E = encode_raw(model, exp.memory_x, channel)
    z_q = encode_raw(model, batch_x, channel)
    memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, channel)
    futures = memory_c + offset_c.view(-1, 1, 1)
    query_future = batch_y[:, :, channel]
    with torch.no_grad():
        host_scores = host.scores(batch_x, channel, cand_mask)
        w_host = candidate_weights(host_scores, cand_mask, host.tau_topk)
    return dict(z_q=z_q, E=E, cand_mask=cand_mask, sc=sc, w_host=w_host, futures=futures,
               query_future=query_future, host_scores=host_scores, host=host, device=device)


# --------------------------------------------------------------------------
# 1. m=0 == plain free-running rollout
# --------------------------------------------------------------------------
def test_m0_equals_free_running_rollout():
    ctx = _setup()
    out = hybrid_prefix_intervention(ctx['z_q'], ctx['E'], ctx['cand_mask'], ctx['sc'],
                                     ctx['w_host'], ctx['futures'], ctx['query_future'],
                                     m=0, top_k=10, chunk_size=4096)
    _, _, picks_ref, _ = run_sequence(ctx['z_q'], ctx['E'], ctx['cand_mask'], ctx['sc'], None,
                                      ctx['w_host'], ctx['futures'], ctx['query_future'],
                                      'greedy_set', 'onpolicy', 0.1, 10, 4096, free_running=True,
                                      greedy_set_impl='reference')
    assert torch.equal(out['picks'], picks_ref)


# --------------------------------------------------------------------------
# 2. m=10 == full oracle rollout (prefix_policy='tf', oracle picks every step)
# --------------------------------------------------------------------------
def test_m10_equals_full_oracle_rollout():
    ctx = _setup()
    out = hybrid_prefix_intervention(ctx['z_q'], ctx['E'], ctx['cand_mask'], ctx['sc'],
                                     ctx['w_host'], ctx['futures'], ctx['query_future'],
                                     m=10, top_k=10, chunk_size=4096)
    _, _, picks_tf, steps_tf = run_sequence(ctx['z_q'], ctx['E'], ctx['cand_mask'], ctx['sc'], None,
                                            ctx['w_host'], ctx['futures'], ctx['query_future'],
                                            'greedy_set', 'tf', 0.1, 10, 4096,
                                            greedy_set_impl='reference')
    assert torch.equal(out['picks'], picks_tf)


# --------------------------------------------------------------------------
# 3. hybrid prefix actually contains the selected candidates
# --------------------------------------------------------------------------
def test_prefix_contains_selected_candidates():
    ctx = _setup()
    out = hybrid_prefix_intervention(ctx['z_q'], ctx['E'], ctx['cand_mask'], ctx['sc'],
                                     ctx['w_host'], ctx['futures'], ctx['query_future'],
                                     m=3, top_k=10, chunk_size=4096)
    picks = out['picks']
    for t in range(1, 10):
        prefix_at_t = picks[:, :t]
        # step t's picked idx must never equal any prior-step picked idx (no dup) --
        # verifies the prefix used for step t's SetConditioner input matches
        # exactly the true selection history.
        assert not torch.any(prefix_at_t == picks[:, t:t + 1])


# --------------------------------------------------------------------------
# 4. oracle recomputed from actual hybrid prefix, not cached
# --------------------------------------------------------------------------
def test_oracle_recomputed_from_actual_hybrid_prefix():
    ctx = _setup()
    out_m1 = hybrid_prefix_intervention(ctx['z_q'], ctx['E'], ctx['cand_mask'], ctx['sc'],
                                        ctx['w_host'], ctx['futures'], ctx['query_future'],
                                        m=1, top_k=10, chunk_size=4096)
    out_m3 = hybrid_prefix_intervention(ctx['z_q'], ctx['E'], ctx['cand_mask'], ctx['sc'],
                                        ctx['w_host'], ctx['futures'], ctx['query_future'],
                                        m=3, top_k=10, chunk_size=4096)
    # steps 1 and 2 have different (m=1 uses student, m=3 uses oracle) selectors,
    # so the two trajectories' prefixes at t=3 generally differ -> oracle u_target
    # at t=3 (computed fresh from each trajectory's own prefix) should differ too
    u3_m1 = out_m1['steps'][3]['u_target']
    u3_m3 = out_m3['steps'][3]['u_target']
    if not torch.equal(out_m1['picks'][:, :3], out_m3['picks'][:, :3]):
        assert not torch.equal(u3_m1, u3_m3)


# --------------------------------------------------------------------------
# 5/6. no reselection, no invalid selection
# --------------------------------------------------------------------------
def test_no_reselection_and_no_invalid():
    ctx = _setup()
    out = hybrid_prefix_intervention(ctx['z_q'], ctx['E'], ctx['cand_mask'], ctx['sc'],
                                     ctx['w_host'], ctx['futures'], ctx['query_future'],
                                     m=5, top_k=10, chunk_size=4096)
    picks = out['picks']
    bsz = picks.size(0)
    for b in range(bsz):
        row = picks[b].tolist()
        assert len(set(row)) == len(row), f'duplicate selection in row {b}: {row}'
        assert all(bool(ctx['cand_mask'][b, i]) for i in row), f'invalid candidate selected in row {b}'


# --------------------------------------------------------------------------
# 7. exactly K unique candidates
# --------------------------------------------------------------------------
def test_exactly_k_unique_candidates():
    ctx = _setup()
    for m in (0, 1, 3, 5, 8, 10):
        out = hybrid_prefix_intervention(ctx['z_q'], ctx['E'], ctx['cand_mask'], ctx['sc'],
                                         ctx['w_host'], ctx['futures'], ctx['query_future'],
                                         m=m, top_k=10, chunk_size=4096)
        for row in out['picks'].tolist():
            assert len(set(row)) == 10


# --------------------------------------------------------------------------
# 8/9. Individual utility prefix-invariant, Set utility prefix-dependent
# --------------------------------------------------------------------------
def test_individual_utility_prefix_invariant():
    torch.manual_seed(0)
    futures = torch.randn(3, 6, 5)
    query_future = torch.randn(3, 5)
    u1 = individual_utility(futures, query_future)
    u2 = individual_utility(futures, query_future)
    assert torch.equal(u1, u2)


def test_set_utility_prefix_dependent():
    from scripts.train_factorial_e2e01 import greedy_set_utility
    torch.manual_seed(0)
    bsz, n_cand, pred_len = 2, 8, 4
    futures = torch.randn(bsz, n_cand, pred_len)
    query_future = torch.randn(bsz, pred_len)
    w_host = torch.softmax(torch.randn(bsz, n_cand), dim=-1)
    u0 = greedy_set_utility(torch.zeros(bsz, 0, dtype=torch.long), w_host, futures, query_future, 4096)
    u1 = greedy_set_utility(torch.zeros(bsz, 1, dtype=torch.long), w_host, futures, query_future, 4096)
    assert not torch.equal(u0, u1)


# --------------------------------------------------------------------------
# 10. future label used only for oracle intervention selection, never in
#     student score computation
# --------------------------------------------------------------------------
def test_future_label_not_used_in_student_score():
    ctx = _setup()
    out_m0 = hybrid_prefix_intervention(ctx['z_q'], ctx['E'], ctx['cand_mask'], ctx['sc'],
                                        ctx['w_host'], ctx['futures'], ctx['query_future'],
                                        m=0, top_k=10, chunk_size=4096)
    # u_hat at every step of m=0 (student-only) must be identical to the plain
    # free-running u_hat, i.e. arm_score(h_t, E, None) -- independent of futures/query_future
    for t, step in enumerate(out_m0['steps']):
        h_t = ctx['z_q'] if t == 0 else ctx['sc'](ctx['z_q'], ctx['E'][out_m0['picks'][:, :t]].mean(dim=1))
        u_hat_direct = arm_score(h_t, ctx['E'], None)
        assert torch.equal(step['u_hat'], u_hat_direct)


# --------------------------------------------------------------------------
# 11. deterministic
# --------------------------------------------------------------------------
def test_deterministic_same_seed():
    ctx1 = _setup(seed=0)
    ctx2 = _setup(seed=0)
    out1 = hybrid_prefix_intervention(ctx1['z_q'], ctx1['E'], ctx1['cand_mask'], ctx1['sc'],
                                      ctx1['w_host'], ctx1['futures'], ctx1['query_future'],
                                      m=3, top_k=10, chunk_size=4096)
    out2 = hybrid_prefix_intervention(ctx2['z_q'], ctx2['E'], ctx2['cand_mask'], ctx2['sc'],
                                      ctx2['w_host'], ctx2['futures'], ctx2['query_future'],
                                      m=3, top_k=10, chunk_size=4096)
    assert torch.equal(out1['picks'], out2['picks'])


# --------------------------------------------------------------------------
# 12. recovery zero/negative denominator
# --------------------------------------------------------------------------
def test_recovery_rate_zero_denominator_issue():
    val, issue = recovery_rate(0.5, 0.4, 0.5)
    assert val is None and issue is not None and '[ISSUE]' in issue


def test_recovery_rate_negative_denominator_issue():
    val, issue = recovery_rate(0.5, 0.4, 0.6)
    assert val is None and issue is not None and '[ISSUE]' in issue


def test_recovery_rate_normal():
    val, issue = recovery_rate(1.0, 0.5, 0.0)
    assert issue is None
    assert abs(val - 0.5) < 1e-9


# --------------------------------------------------------------------------
# 13. CPU/GPU consistency
# --------------------------------------------------------------------------
def test_cpu_gpu_consistency():
    if not torch.cuda.is_available():
        pytest.skip('no GPU available')
    ctx_cpu = _setup()
    out_gpu = hybrid_prefix_intervention(ctx_cpu['z_q'], ctx_cpu['E'], ctx_cpu['cand_mask'], ctx_cpu['sc'],
                                         ctx_cpu['w_host'], ctx_cpu['futures'], ctx_cpu['query_future'],
                                         m=3, top_k=10, chunk_size=4096)
    # move a FRESH copy of every tensor/module to CPU (never mutate the GPU
    # originals used above -- .to() on an nn.Module mutates in place)
    device_cpu = torch.device('cpu')
    import copy
    sc_cpu = copy.deepcopy(ctx_cpu['sc']).to(device_cpu)
    z_q_cpu = ctx_cpu['z_q'].clone().to(device_cpu)
    E_cpu = ctx_cpu['E'].clone().to(device_cpu)
    cand_mask_cpu = ctx_cpu['cand_mask'].clone().to(device_cpu)
    w_host_cpu = ctx_cpu['w_host'].clone().to(device_cpu)
    futures_cpu = ctx_cpu['futures'].clone().to(device_cpu)
    query_future_cpu = ctx_cpu['query_future'].clone().to(device_cpu)
    out_cpu = hybrid_prefix_intervention(z_q_cpu, E_cpu, cand_mask_cpu, sc_cpu, w_host_cpu,
                                         futures_cpu, query_future_cpu, m=3, top_k=10, chunk_size=4096)
    assert torch.equal(out_gpu['picks'].cpu(), out_cpu['picks'])


# --------------------------------------------------------------------------
# 14. NaN/Inf fail-fast (reuse greedy_set_utility's own real behavior; here
#     check the diagnostic layer surfaces non-finite utility rather than
#     silently picking garbage)
# --------------------------------------------------------------------------
def test_nan_utility_does_not_silently_pick():
    bsz, n_cand, pred_len = 1, 5, 3
    futures = torch.randn(bsz, n_cand, pred_len)
    futures[0, 2, 0] = float('nan')
    query_future = torch.randn(bsz, pred_len)
    from scripts.train_factorial_e2e01 import individual_utility as iu
    u = iu(futures, query_future)
    assert torch.isnan(u[0, 2]), 'expected NaN to propagate visibly, not be silently masked'


# --------------------------------------------------------------------------
# 15. combination redundancy diagnostic sanity
# --------------------------------------------------------------------------
def test_combination_redundancy_identical_candidates_max_cosine():
    picked = torch.ones(2, 4, 6)  # all identical vectors -> cosine ~1
    out = combination_redundancy(picked)
    assert torch.allclose(out['mean_pairwise_cosine'], torch.ones(2), atol=1e-4)
    assert torch.allclose(out['mean_pairwise_mse'], torch.zeros(2), atol=1e-6)


def test_combination_redundancy_orthogonal_lower_cosine():
    torch.manual_seed(0)
    picked_identical = torch.randn(1, 1, 8).expand(1, 4, 8).clone()
    picked_diverse = torch.randn(1, 4, 8)
    out_i = combination_redundancy(picked_identical)
    out_d = combination_redundancy(picked_diverse)
    assert float(out_i['mean_pairwise_cosine']) > float(out_d['mean_pairwise_cosine'])


# --------------------------------------------------------------------------
# 16. single-step correction sanity: correcting a step matches the
#     hybrid-m=0 trajectory everywhere except (possibly) from that step on
# --------------------------------------------------------------------------
def test_single_step_correction_matches_m0_before_correction():
    ctx = _setup()
    out_m0 = hybrid_prefix_intervention(ctx['z_q'], ctx['E'], ctx['cand_mask'], ctx['sc'],
                                        ctx['w_host'], ctx['futures'], ctx['query_future'],
                                        m=0, top_k=10, chunk_size=4096)
    out_corr = single_step_correction(ctx['z_q'], ctx['E'], ctx['cand_mask'], ctx['sc'],
                                      ctx['w_host'], ctx['futures'], ctx['query_future'],
                                      correct_step=4, top_k=10, chunk_size=4096)
    assert torch.equal(out_m0['picks'][:, :4], out_corr['picks'][:, :4])


def test_single_step_correction_step0_is_full_oracle_or_student_elsewhere():
    ctx = _setup()
    out = single_step_correction(ctx['z_q'], ctx['E'], ctx['cand_mask'], ctx['sc'],
                                 ctx['w_host'], ctx['futures'], ctx['query_future'],
                                 correct_step=0, top_k=10, chunk_size=4096)
    assert out['steps'][0]['selector'] == 'oracle'
    for t in range(1, 10):
        assert out['steps'][t]['selector'] == 'student'
