#!/usr/bin/env python3
"""TRACK-A-SET-LOSS-CONTROL01 smoke test (spec section 8): same batch
subset, same initial model+SetConditioner state, ALL 7 ETTh1 channels, 20
optimizer steps each for A0/A1/A2/A3, ETTh1_96. No LR/clip/temperature
adjustment -- raw scale reported, not hidden."""
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.SequentialSetRetriever import SetConditioner
from scripts.train_factorial_e2e01 import HostScorer, candidate_weights, encode_raw, grad_norm, state_sha
from scripts.train_margutil01 import build_experiment, memory_value
from scripts.train_set_loss_control01 import ARMS, run_sequence


def main():
    S1 = 'checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth'
    S2 = 'checkpoints/stage2/ETTh1/seq96_pred96/stage2_carts_softset_s2_ETTh1_96_S0_wce_RelationStage2_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed = 1

    torch.manual_seed(seed)
    exp0, args0 = build_experiment(S1, {'pred_len': 96, 'seq_len': 96, 'batch_size': 32,
                                        'seed': seed, 'top_k': 10})
    exp0._ensure_memory()
    model0 = exp0.model.module if hasattr(exp0.model, 'module') else exp0.model
    model0.to(device)
    sc0 = SetConditioner(int(args0.d_model)).to(device)
    init_model = {k: v.detach().clone() for k, v in model0.state_dict().items()}
    init_sc = {k: v.detach().clone() for k, v in sc0.state_dict().items()}
    channels = list(range(int(args0.enc_in)))
    del exp0, args0, model0, sc0

    host = HostScorer(S2, device)
    tau_choice = 0.1
    tau_topk = float(dict(torch.load(S2, map_location='cpu')['args'])['tau_topk'])

    results = {}
    for arm in ARMS:
        torch.manual_seed(seed)
        exp, args = build_experiment(S1, {'pred_len': 96, 'seq_len': 96, 'batch_size': 32,
                                          'seed': seed, 'top_k': 10})
        exp._ensure_memory()
        model = exp.model.module if hasattr(exp.model, 'module') else exp.model
        model.to(device)
        model.load_state_dict(init_model)
        sc = SetConditioner(int(args.d_model)).to(device)
        sc.load_state_dict(init_sc)
        for p in model.parameters():
            p.requires_grad_(True)
        params = list(model.parameters()) + list(sc.parameters())
        optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=0.0)

        torch.manual_seed(seed)
        _, loader = exp._get_data(flag='train')
        it = iter(loader)
        step_log = []
        for step in range(20):
            batch_x, batch_y, batch_start_idx = next(it)
            batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)
            cand_mask, _ = exp._candidate_mask(batch_start_idx)
            optimizer.zero_grad()
            batch_loss = 0.0
            per_channel_has_grad = {}
            first_diag = None
            for c in channels:
                E = encode_raw(model, exp.memory_x, c)
                z_q = encode_raw(model, batch_x, c)
                memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
                futures = memory_c + offset_c.view(-1, 1, 1)
                query_future = batch_y[:, :, c]
                with torch.no_grad():
                    host_scores = host.scores(batch_x, c, cand_mask)
                    w_host = candidate_weights(host_scores, cand_mask, tau_topk)
                losses, diags, picks, steps_ = run_sequence(
                    z_q, E, cand_mask, sc, w_host, futures, query_future, arm, tau_choice, 10, 4096)
                ch_loss = sum(losses) / 10
                batch_loss = batch_loss + ch_loss
                if c == 0:
                    first_diag = diags[1] if len(diags) > 1 else diags[0]
            batch_loss = batch_loss / len(channels)
            batch_loss.backward()
            gn = grad_norm(params)
            optimizer.step()
            pnorm = float(sum(p.detach().float().pow(2).sum() for p in params) ** 0.5)
            has_nan = bool(torch.isnan(batch_loss)) or bool(torch.isinf(batch_loss))
            step_log.append({
                'step': step, 'loss': float(batch_loss.detach()), 'grad_norm': gn, 'param_norm': pnorm,
                'has_nan_inf': has_nan,
                'chosen_normalized_regret': first_diag.get('chosen_normalized_regret'),
                'support_size': first_diag.get('support_size'),
                'teacher_entropy': first_diag.get('teacher_entropy'),
                'topm_probability_mass': first_diag.get('topm_probability_mass'),
            })
        results[arm] = step_log
        print(f'[smoke_set_loss_control01] {arm}: step0 loss={step_log[0]["loss"]:.5f} '
             f'step19 loss={step_log[-1]["loss"]:.5f} step0 grad_norm={step_log[0]["grad_norm"]:.5f} '
             f'step19 grad_norm={step_log[-1]["grad_norm"]:.5f} '
             f'any_nan_inf={any(s["has_nan_inf"] for s in step_log)}')

    Path('results/TRACK-A-SET-LOSS-CONTROL01').mkdir(parents=True, exist_ok=True)
    Path('results/TRACK-A-SET-LOSS-CONTROL01/smoke_test.json').write_text(
        json.dumps({'channels': channels, 'results': results}, indent=2))
    print('[smoke_set_loss_control01] wrote results/TRACK-A-SET-LOSS-CONTROL01/smoke_test.json')


if __name__ == '__main__':
    main()
