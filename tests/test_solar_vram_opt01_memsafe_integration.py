"""TRACK-A-SOLAR-VRAM-OPT01 P0.2 -- end-to-end equivalence test:
`train_epoch`/`eval_epoch` with `memsafe=True` vs `memsafe=False` (the
default/reference), real ETTh1_96 data, real optimizer steps.
"""
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.SequentialSetRetriever import SetConditioner
from scripts.train_factorial_e2e01 import HostScorer, eval_epoch, state_sha, train_epoch
from scripts.train_margutil01 import build_experiment

REF_CKPT = 'checkpoints/track_a_factorial_e2e/ETTh1_96/set_onpolicy_cosine/checkpoint.pth'
S2_96 = ('checkpoints/stage2/ETTh1/seq96_pred96/stage2_carts_softset_s2_ETTh1_96_S0_wce_'
        'RelationStage2_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_'
        'ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth')


def _available():
    return (REPO_ROOT / REF_CKPT).exists() and (REPO_ROOT / S2_96).exists()


pytestmark = pytest.mark.skipif(not _available(), reason='ETTh1_96 checkpoints not present')


class _OneBatchRepeater:
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


def test_memsafe_train_and_eval_match_reference():
    exp_ref, args_ref, model_ref, sc_ref, device = _build(seed=0)
    exp_opt, args_opt, model_opt, sc_opt, _ = _build(seed=0)
    model_opt.load_state_dict(model_ref.state_dict())
    sc_opt.load_state_dict(sc_ref.state_dict())
    assert state_sha(model_ref.state_dict()) == state_sha(model_opt.state_dict())

    host_ref = HostScorer(S2_96, device)
    host_opt = HostScorer(S2_96, device)
    opt_ref = torch.optim.Adam(list(model_ref.parameters()) + list(sc_ref.parameters()), lr=1e-3)
    opt_opt = torch.optim.Adam(list(model_opt.parameters()) + list(sc_opt.parameters()), lr=1e-3)

    class Cli:
        pass
    def make_cli(optimizer, tau_topk):
        c = Cli(); c.optimizer = optimizer; c.target = 'greedy_set'
        c.prefix_policy = 'onpolicy'; c.tau_choice = 0.1; c.top_k = 10
        c.chunk_size = 500; c.tau_topk = tau_topk; c.limit_batches = 0
        c.resolved_oracle_compute = {'choice_ce_impl': 'reference', 'individual_impl': 'reference',
                                     'greedy_set_impl': 'reference'}
        return c
    cli_ref = make_cli(opt_ref, host_ref.tau_topk)
    cli_opt = make_cli(opt_opt, host_opt.tau_topk)

    channels = list(range(int(args_ref.enc_in)))[:2]
    _, loader_ref = exp_ref._get_data(flag='train', shuffle=False)
    _, loader_opt = exp_opt._get_data(flag='train', shuffle=False)
    batches_ref = _OneBatchRepeater(loader_ref, 2).batches
    batches_opt = _OneBatchRepeater(loader_opt, 2).batches

    torch.manual_seed(3000)
    tr_ref = train_epoch(exp_ref, args_ref, host_ref, model_ref, sc_ref, None, cli_ref,
                         batches_ref, channels, device, channelwise_backward=False, memsafe=False)
    torch.manual_seed(3000)
    tr_opt = train_epoch(exp_opt, args_opt, host_opt, model_opt, sc_opt, None, cli_opt,
                         batches_opt, channels, device, channelwise_backward=False, memsafe=True)

    assert abs(tr_ref['train_choice_ce'] - tr_opt['train_choice_ce']) < 1e-5, (
        f"{tr_ref['train_choice_ce']} vs {tr_opt['train_choice_ce']}")
    assert abs(tr_ref['encoder_grad_norm'] - tr_opt['encoder_grad_norm']) < 1e-4
    for (n1, p1), (n2, p2) in zip(model_ref.named_parameters(), model_opt.named_parameters()):
        assert torch.allclose(p1, p2, atol=1e-5, rtol=1e-4), f'{n1} diverged'

    # eval equivalence (no gradient, just FR-Agg/metric numerics)
    _, val_loader_ref = exp_ref._get_data(flag='val', shuffle=False)
    _, val_loader_opt = exp_opt._get_data(flag='val', shuffle=False)
    val_batches_ref = _OneBatchRepeater(val_loader_ref, 1).batches
    val_batches_opt = _OneBatchRepeater(val_loader_opt, 1).batches

    va_ref, _ = eval_epoch(exp_ref, args_ref, host_ref, model_ref, sc_ref, None, cli_ref,
                           val_batches_ref, channels, device, memsafe=False)
    va_opt, _ = eval_epoch(exp_opt, args_opt, host_opt, model_opt, sc_opt, None, cli_opt,
                           val_batches_opt, channels, device, memsafe=True)
    assert abs(va_ref['free_running_aggregate_future_mse']
              - va_opt['free_running_aggregate_future_mse']) < 1e-5, (
        f"{va_ref['free_running_aggregate_future_mse']} vs {va_opt['free_running_aggregate_future_mse']}")
