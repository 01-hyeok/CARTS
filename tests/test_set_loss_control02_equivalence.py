"""TRACK-A-SET-LOSS-CONTROL02 -- mandatory A0-vs-Factorial-Set-OP exact-
equivalence gate (spec section 7). MUST pass before any real A0-A3
training is launched.

Compares two call sites on the SAME real small batches, SAME shared init,
SAME seed, for 20 real optimizer steps:

    Path F: scripts.train_factorial_e2e01.run_sequence(..., target=
            'greedy_set', prefix_policy='onpolicy', metric=None (cosine),
            choice_ce_impl='reference', greedy_set_impl='reference',
            loss_fn=None)   -- i.e. exactly what set_onpolicy_cosine uses.

    Path C: the SAME run_sequence call, but with
            loss_fn=make_loss_fn('A0_hard_choice', tau_choice) from
            scripts.train_set_loss_control02 -- i.e. exactly what
            CONTROL02's A0_hard_choice arm uses.

Both paths call the identical `run_sequence` function (not two
implementations) -- the only difference under test is whether the loss is
computed via the choice_ce_impl dispatch (Path F) or via the loss_fn hook
that, for A0 at every step (t==0 or arm=='A0_hard_choice' is always true
for A0), calls the exact same `oracle_choice_step_loss`. This test
verifies that this structural sharing is REAL, not just argued -- exact
equality of picks, logits, utility, loss, gradients, and post-step
parameters over 20 real optimizer steps.
"""
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.SequentialSetRetriever import SetConditioner
from scripts.train_factorial_e2e01 import (HostScorer, candidate_weights, encode_raw,
                                           run_sequence, state_sha)
from scripts.train_margutil01 import build_experiment, memory_value
from scripts.train_set_loss_control02 import make_loss_fn

REF_CKPT = ('checkpoints/track_a_factorial_e2e/ETTh1_96/set_onpolicy_cosine/checkpoint.pth')
S2_96 = ('checkpoints/stage2/ETTh1/seq96_pred96/stage2_carts_softset_s2_ETTh1_96_S0_wce_'
        'RelationStage2_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_'
        'ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth')


def _available():
    return (REPO_ROOT / REF_CKPT).exists() and (REPO_ROOT / S2_96).exists()


pytestmark = pytest.mark.skipif(not _available(), reason='ETTh1_96 reference/host checkpoints not present')


def _build_pair(seed, top_k=10, tau_choice=0.1):
    """Two independent (exp, model, set_conditioner) built from the same
    seed/config, with SHARED initial weights (copied tensor-for-tensor, not
    just same-seed -- eliminates any RNG-call-count drift between the two
    construction call sites as a confound)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp_f, args_f = build_experiment(REF_CKPT, {
        'pred_len': 96, 'seq_len': 96, 'batch_size': 8, 'seed': seed,
        'top_k': top_k, 'tau_topk': tau_choice})
    exp_f._ensure_memory()
    model_f = exp_f.model.module if hasattr(exp_f.model, 'module') else exp_f.model
    model_f.to(device)
    sc_f = SetConditioner(int(args_f.d_model)).to(device)

    exp_c, args_c = build_experiment(REF_CKPT, {
        'pred_len': 96, 'seq_len': 96, 'batch_size': 8, 'seed': seed,
        'top_k': top_k, 'tau_topk': tau_choice})
    exp_c._ensure_memory()
    model_c = exp_c.model.module if hasattr(exp_c.model, 'module') else exp_c.model
    model_c.to(device)
    sc_c = SetConditioner(int(args_c.d_model)).to(device)

    # force byte-identical init regardless of construction-order RNG drift
    model_c.load_state_dict(model_f.state_dict())
    sc_c.load_state_dict(sc_f.state_dict())
    assert state_sha(model_f.state_dict()) == state_sha(model_c.state_dict())
    assert state_sha(sc_f.state_dict()) == state_sha(sc_c.state_dict())

    return (exp_f, model_f, sc_f, args_f), (exp_c, model_c, sc_c, args_c), device


def test_a0_exact_equivalence_to_factorial_set_op_20_steps():
    (exp_f, model_f, sc_f, args_f), (exp_c, model_c, sc_c, args_c), device = _build_pair(seed=0)

    host = HostScorer(S2_96, device)
    tau_choice = 0.1
    tau_topk = host.tau_topk
    top_k = 10
    chunk_size = 4096
    channel = 0

    for p in model_f.parameters():
        p.requires_grad_(True)
    for p in model_c.parameters():
        p.requires_grad_(True)
    opt_f = torch.optim.Adam(list(model_f.parameters()) + list(sc_f.parameters()), lr=1e-3)
    opt_c = torch.optim.Adam(list(model_c.parameters()) + list(sc_c.parameters()), lr=1e-3)

    loss_fn_a0 = make_loss_fn('A0_hard_choice', tau_choice)

    _, loader = exp_f._get_data(flag='train', shuffle=False)
    batches = []
    for i, (batch_x, batch_y, batch_start_idx) in enumerate(loader):
        if i >= 20:
            break
        batches.append((batch_x, batch_y, batch_start_idx))
    assert len(batches) == 20, f'need >=20 batches, got {len(batches)}'

    # Both encoders have dropout (configs.dropout, active in .train() mode,
    # the mode real training uses). Two separate model instances draw
    # dropout masks from the SAME global torch RNG stream, so without
    # resetting the seed at each shared call site, mask draws for path F
    # and path C interleave differently and forward outputs diverge even
    # with byte-identical weights -- not a correctness bug, just the
    # standard "two independent RNG consumers" hazard. Reset to the same
    # per-step seed immediately before each matched call so both paths draw
    # the SAME dropout masks, isolating the comparison to the thing under
    # test (loss_fn dispatch), not incidental RNG-stream drift.
    max_abs_diff = {'loss': 0.0, 'u_hat': 0.0, 'u_target': 0.0}
    for step, (batch_x, batch_y, batch_start_idx) in enumerate(batches):
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp_f._candidate_mask(batch_start_idx)

        torch.manual_seed(10_000 + step)
        Ef = encode_raw(model_f, exp_f.memory_x, channel)
        zqf = encode_raw(model_f, batch_x, channel)
        torch.manual_seed(10_000 + step)
        Ec = encode_raw(model_c, exp_c.memory_x, channel)
        zqc = encode_raw(model_c, batch_x, channel)
        assert torch.equal(Ef, Ec), f'step {step}: memory embeddings diverged BEFORE any optimizer step'
        assert torch.equal(zqf, zqc), f'step {step}: query embeddings diverged'

        memory_c_f, offset_f = memory_value(args_f, batch_x, exp_f.memory_y, exp_f.memory_x_last, channel)
        futures_f = memory_c_f + offset_f.view(-1, 1, 1)
        memory_c_c, offset_c_ = memory_value(args_c, batch_x, exp_c.memory_y, exp_c.memory_x_last, channel)
        futures_c = memory_c_c + offset_c_.view(-1, 1, 1)
        assert torch.equal(futures_f, futures_c), f'step {step}: candidate futures diverged'
        query_future = batch_y[:, :, channel]

        with torch.no_grad():
            host_scores = host.scores(batch_x, channel, cand_mask)
            w_host = candidate_weights(host_scores, cand_mask, tau_topk)

        opt_f.zero_grad()
        losses_f, diags_f, picks_f, steps_f = run_sequence(
            zqf, Ef, cand_mask, sc_f, None, w_host, futures_f, query_future,
            'greedy_set', 'onpolicy', tau_choice, top_k, chunk_size,
            greedy_set_impl='reference', loss_fn=None)
        loss_f = sum(losses_f) / top_k
        loss_f.backward()

        opt_c.zero_grad()
        losses_c, diags_c, picks_c, steps_c = run_sequence(
            zqc, Ec, cand_mask, sc_c, None, w_host, futures_c, query_future,
            'greedy_set', 'onpolicy', tau_choice, top_k, chunk_size,
            greedy_set_impl='reference', loss_fn=loss_fn_a0)
        loss_c = sum(losses_c) / top_k
        loss_c.backward()

        assert torch.equal(picks_f, picks_c), (
            f'[ISSUE] A0 Hard CE is not implementation-equivalent to Factorial Set OP.\n'
            f'FIRST DIVERGENCE:\nTENSOR: picks\nSTEP: {step}\n'
            f'MAX_ABS_DIFF: n/a (index mismatch)\nLIKELY CAUSE: on-policy prefix advance diverged')
        for t, (df, dc) in enumerate(zip(steps_f, steps_c)):
            assert torch.equal(df['oracle_idx'], dc['oracle_idx']), f'step {step} t{t}: oracle_idx'
            assert torch.equal(df['model_idx'], dc['model_idx']), f'step {step} t{t}: model_idx'
            uh_diff = float((df['u_hat'] - dc['u_hat']).abs().max())
            ut_diff = float((df['u_target'] - dc['u_target']).abs().max())
            max_abs_diff['u_hat'] = max(max_abs_diff['u_hat'], uh_diff)
            max_abs_diff['u_target'] = max(max_abs_diff['u_target'], ut_diff)
            assert uh_diff <= 1e-7, (
                f'[ISSUE] A0 Hard CE is not implementation-equivalent to Factorial Set OP.\n'
                f'FIRST DIVERGENCE:\nTENSOR: u_hat\nSTEP: {step} t{t}\nMAX_ABS_DIFF: {uh_diff}\n'
                f'LIKELY CAUSE: encoder/SetConditioner state diverged')
            assert ut_diff <= 1e-7, (
                f'[ISSUE] A0 Hard CE is not implementation-equivalent to Factorial Set OP.\n'
                f'FIRST DIVERGENCE:\nTENSOR: u_target\nSTEP: {step} t{t}\nMAX_ABS_DIFF: {ut_diff}\n'
                f'LIKELY CAUSE: Set Oracle utility computation diverged')

        loss_diff = float((loss_f.detach() - loss_c.detach()).abs())
        max_abs_diff['loss'] = max(max_abs_diff['loss'], loss_diff)
        assert loss_diff <= 1e-7, (
            f'[ISSUE] A0 Hard CE is not implementation-equivalent to Factorial Set OP.\n'
            f'FIRST DIVERGENCE:\nTENSOR: loss\nSTEP: {step}\nMAX_ABS_DIFF: {loss_diff}\n'
            f'LIKELY CAUSE: loss_fn dispatch diverged from choice_ce_impl dispatch')

        for pf, pc in zip(model_f.parameters(), model_c.parameters()):
            if pf.grad is None and pc.grad is None:
                continue
            assert pf.grad is not None and pc.grad is not None, f'step {step}: gradient presence mismatch'
            g_diff = float((pf.grad - pc.grad).abs().max())
            assert g_diff <= 1e-7, (
                f'[ISSUE] A0 Hard CE is not implementation-equivalent to Factorial Set OP.\n'
                f'FIRST DIVERGENCE:\nTENSOR: encoder gradient\nSTEP: {step}\nMAX_ABS_DIFF: {g_diff}\n'
                f'LIKELY CAUSE: backward graph diverged')

        opt_f.step()
        opt_c.step()

        sha_f = state_sha(model_f.state_dict())
        sha_c = state_sha(model_c.state_dict())
        assert sha_f == sha_c, (
            f'[ISSUE] A0 Hard CE is not implementation-equivalent to Factorial Set OP.\n'
            f'FIRST DIVERGENCE:\nTENSOR: post-optimizer-step model parameters\nSTEP: {step}\n'
            f'MAX_ABS_DIFF: n/a (hash mismatch)\nLIKELY CAUSE: optimizer step diverged')

    print(f'[equivalence] 20/20 steps exact-equivalent. max_abs_diff={max_abs_diff}')
