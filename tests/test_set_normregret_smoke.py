"""TRACK-A-SET-NORMREGRET-CONTROL01 -- section 14 smoke test: A0 Hard CE,
A1 Raw-WCE, A2 Normalized Soft CE, A3 Normalized SRM, each 20 real
optimizer steps on the SAME small ETTh1_96 batches, SAME LR, recording
loss/grad-norm/param-norm/NaN-Inf/selected-candidate/Oracle-top1/chosen-
normalized-regret/Top-M student mass/teacher entropy/effective positive
count/teacher top-1 weight."""
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.SequentialSetRetriever import SetConditioner
from scripts.train_factorial_e2e01 import (HostScorer, candidate_weights, encode_raw,
                                           run_sequence)
from scripts.train_margutil01 import build_experiment, memory_value
from scripts.train_set_normregret_control01 import ARMS, make_loss_fn

REF_CKPT = 'checkpoints/track_a_factorial_e2e/ETTh1_96/set_onpolicy_cosine/checkpoint.pth'
S2_96 = ('checkpoints/stage2/ETTh1/seq96_pred96/stage2_carts_softset_s2_ETTh1_96_S0_wce_'
        'RelationStage2_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_'
        'ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth')


def _available():
    return (REPO_ROOT / REF_CKPT).exists() and (REPO_ROOT / S2_96).exists()


pytestmark = pytest.mark.skipif(not _available(), reason='ETTh1_96 checkpoints not present')


def _run_smoke(arm, n_steps=20, channel=0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    exp, args = build_experiment(REF_CKPT, {
        'pred_len': 96, 'seq_len': 96, 'batch_size': 8, 'seed': 0, 'top_k': 10, 'tau_topk': 0.1})
    exp._ensure_memory()
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.to(device)
    sc = SetConditioner(int(args.d_model)).to(device)
    for p in model.parameters():
        p.requires_grad_(True)
    opt = torch.optim.Adam(list(model.parameters()) + list(sc.parameters()), lr=1e-3)

    host = HostScorer(S2_96, device)
    loss_fn = make_loss_fn(arm)

    _, loader = exp._get_data(flag='train', shuffle=False)
    rows = []
    for step, (batch_x, batch_y, batch_start_idx) in enumerate(loader):
        if step >= n_steps:
            break
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)

        E = encode_raw(model, exp.memory_x, channel)
        z_q = encode_raw(model, batch_x, channel)
        memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, channel)
        futures = memory_c + offset_c.view(-1, 1, 1)
        query_future = batch_y[:, :, channel]
        with torch.no_grad():
            host_scores = host.scores(batch_x, channel, cand_mask)
            w_host = candidate_weights(host_scores, cand_mask, host.tau_topk)

        opt.zero_grad()
        losses, diags, picks, steps_rec = run_sequence(
            z_q, E, cand_mask, sc, None, w_host, futures, query_future,
            'greedy_set', 'onpolicy', 0.1, 10, 4096, greedy_set_impl='reference', loss_fn=loss_fn)
        loss = sum(losses) / 10
        loss.backward()

        params = list(model.parameters()) + list(sc.parameters())
        gnorm = sum(float(p.grad.detach().pow(2).sum()) for p in params if p.grad is not None) ** 0.5
        pnorm = sum(float(p.detach().pow(2).sum()) for p in params) ** 0.5
        has_nan_inf = not torch.isfinite(loss)
        opt.step()

        d0 = steps_rec[0]
        oracle_top1 = int(d0['oracle_idx'][0])
        diag0 = diags[0]
        rows.append({'step': step, 'loss': float(loss), 'grad_norm': gnorm, 'param_norm': pnorm,
                    'has_nan_inf': bool(has_nan_inf), 'selected_candidate_t0': int(picks[0, 0]),
                    'oracle_top1_t0': oracle_top1, 'diag0': diag0})
    return rows


@pytest.mark.parametrize('arm', ARMS)
def test_smoke_20_steps(arm):
    rows = _run_smoke(arm)
    assert len(rows) == 20
    assert not any(r['has_nan_inf'] for r in rows)
    assert all(torch.isfinite(torch.tensor(r['loss'])) for r in rows)
    print(f'[smoke {arm}]', rows[-1])
