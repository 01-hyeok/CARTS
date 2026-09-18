"""TRACK-A-SOLAR-VRAM-OPT01 P0.1 -- equivalence test for channel-wise
backward vs the original accumulate-then-backward-once pattern in
`scripts/train_factorial_e2e01.py::train_epoch`.

Verifies, on REAL ETTh1_96 data, real optimizer steps, identical init:
- total loss per batch
- gradient norm (encoder + SetConditioner) before optimizer.step()
- post-step parameter values (a representative sample + full-model hash)
match within tight tolerance between `channelwise_backward=False`
(reference) and `=True` (optimized), across several real training steps
in a row (not just step 1 -- per spec, divergence can compound).
"""
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.SequentialSetRetriever import SetConditioner
from scripts.train_factorial_e2e01 import HostScorer, state_sha, train_epoch
from scripts.train_margutil01 import build_experiment

REF_CKPT = 'checkpoints/track_a_factorial_e2e/ETTh1_96/set_onpolicy_cosine/checkpoint.pth'
S2_96 = ('checkpoints/stage2/ETTh1/seq96_pred96/stage2_carts_softset_s2_ETTh1_96_S0_wce_'
        'RelationStage2_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_'
        'ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth')


def _available():
    return (REPO_ROOT / REF_CKPT).exists() and (REPO_ROOT / S2_96).exists()


pytestmark = pytest.mark.skipif(not _available(), reason='ETTh1_96 checkpoints not present')


class _OneBatchRepeater:
    """Wraps a real DataLoader but only ever yields the SAME first N
    batches, in the SAME order, every time it's iterated -- lets both the
    reference and optimized runs see byte-identical batches across
    multiple simulated "epochs" without depending on shuffle determinism."""
    def __init__(self, loader, n_batches):
        self.batches = []
        for i, b in enumerate(loader):
            if i >= n_batches:
                break
            self.batches.append(b)

    def __iter__(self):
        return iter(self.batches)


def _build(seed=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    exp, args = build_experiment(REF_CKPT, {
        'pred_len': 96, 'seq_len': 96, 'batch_size': 8, 'seed': seed, 'top_k': 10, 'tau_topk': 0.1})
    exp._ensure_memory()
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.to(device)
    sc = SetConditioner(int(args.d_model)).to(device)
    for p in model.parameters():
        p.requires_grad_(True)
    return exp, args, model, sc, device


def test_channelwise_backward_matches_reference_multi_step():
    host = None
    exp_ref, args_ref, model_ref, sc_ref, device = _build(seed=0)
    exp_opt, args_opt, model_opt, sc_opt, _ = _build(seed=0)
    # force byte-identical init regardless of any construction-order RNG drift
    model_opt.load_state_dict(model_ref.state_dict())
    sc_opt.load_state_dict(sc_ref.state_dict())
    assert state_sha(model_ref.state_dict()) == state_sha(model_opt.state_dict())

    host_ref = HostScorer(S2_96, device)
    host_opt = HostScorer(S2_96, device)

    opt_ref = torch.optim.Adam(list(model_ref.parameters()) + list(sc_ref.parameters()), lr=1e-3)
    opt_opt = torch.optim.Adam(list(model_opt.parameters()) + list(sc_opt.parameters()), lr=1e-3)

    class Cli:
        pass
    cli_ref = Cli(); cli_ref.optimizer = opt_ref; cli_ref.target = 'greedy_set'
    cli_ref.prefix_policy = 'onpolicy'; cli_ref.tau_choice = 0.1; cli_ref.top_k = 10
    cli_ref.chunk_size = 4096; cli_ref.tau_topk = host_ref.tau_topk; cli_ref.limit_batches = 0
    cli_ref.resolved_oracle_compute = {'choice_ce_impl': 'reference', 'individual_impl': 'reference',
                                       'greedy_set_impl': 'reference'}

    cli_opt = Cli(); cli_opt.optimizer = opt_opt; cli_opt.target = 'greedy_set'
    cli_opt.prefix_policy = 'onpolicy'; cli_opt.tau_choice = 0.1; cli_opt.top_k = 10
    cli_opt.chunk_size = 4096; cli_opt.tau_topk = host_opt.tau_topk; cli_opt.limit_batches = 0
    cli_opt.resolved_oracle_compute = cli_ref.resolved_oracle_compute

    channels = list(range(int(args_ref.enc_in)))[:2]  # 2 channels is enough to exercise the
                                                       # per-channel accumulation logic; full 7
                                                       # would just be slower, not more informative
    _, loader_ref = exp_ref._get_data(flag='train', shuffle=False)
    _, loader_opt = exp_opt._get_data(flag='train', shuffle=False)
    loader_ref = _OneBatchRepeater(loader_ref, 3)
    loader_opt = _OneBatchRepeater(loader_opt, 3)

    for step in range(3):
        one_batch_ref = [loader_ref.batches[step]]
        one_batch_opt = [loader_opt.batches[step]]

        # Both models have dropout, drawn from the GLOBAL torch RNG stream at
        # forward time. Reseeding identically right before each matched
        # train_epoch call realigns the dropout masks between the ref/opt
        # runs (the SEQUENCE and COUNT of RNG-consuming forward calls is
        # identical between the two backward strategies -- only WHEN
        # .backward() is invoked differs, and backward does not itself draw
        # from this stream) -- same root cause/fix as this session's earlier
        # A0-vs-Factorial equivalence test (dropout RNG stream drift between
        # two independently-constructed model instances).
        torch.manual_seed(2000 + step)
        tr_ref = train_epoch(exp_ref, args_ref, host_ref, model_ref, sc_ref, None, cli_ref,
                             one_batch_ref, channels, device, channelwise_backward=False)
        torch.manual_seed(2000 + step)
        tr_opt = train_epoch(exp_opt, args_opt, host_opt, model_opt, sc_opt, None, cli_opt,
                             one_batch_opt, channels, device, channelwise_backward=True)

        assert abs(tr_ref['train_choice_ce'] - tr_opt['train_choice_ce']) < 1e-5, (
            f'step {step}: loss diverged {tr_ref["train_choice_ce"]} vs {tr_opt["train_choice_ce"]}')
        assert abs(tr_ref['encoder_grad_norm'] - tr_opt['encoder_grad_norm']) < 1e-4, (
            f'step {step}: encoder_grad_norm diverged {tr_ref["encoder_grad_norm"]} vs '
            f'{tr_opt["encoder_grad_norm"]}')
        assert abs(tr_ref['score_layer_grad_norm'] - tr_opt['score_layer_grad_norm']) < 1e-4, (
            f'step {step}: score_layer_grad_norm diverged')

        sha_ref = state_sha(model_ref.state_dict())
        sha_opt = state_sha(model_opt.state_dict())
        # exact SHA equality is too strict across two independently-summed floating point
        # paths (different addition order) -- check numerically instead, but ALSO expect
        # near-exact (not just "close"): parameters after Adam step should match tightly.
        for (n1, p1), (n2, p2) in zip(model_ref.named_parameters(), model_opt.named_parameters()):
            assert torch.allclose(p1, p2, atol=1e-5, rtol=1e-4), (
                f'step {step}: parameter {n1} diverged after optimizer.step(), '
                f'max_abs_diff={(p1-p2).abs().max().item()}')
