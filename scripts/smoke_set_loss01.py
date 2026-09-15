#!/usr/bin/env python3
"""EXP-SET-LOSS01 smoke test (spec section 12): same tiny batch subset,
same initial model state, 20 optimizer steps each for hard_ce / srm /
setutility_softce, ETTh1_96, Set + on-policy + cosine. No learning-rate,
clipping, or temperature adjustment -- reports raw gradient/loss scale so
any anomaly is visible, not hidden."""
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_set_loss01 import (LOSS_TYPES, build_experiment, candidate_weights,
                                      compute_step_loss, encode_raw, greedy_set_utility,
                                      grad_norm, memory_value, run_sequence, state_sha)
from scripts.train_factorial_e2e01 import HostScorer
from models.SequentialSetRetriever import SetConditioner


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
    init_sha = state_sha(model0.state_dict())
    del exp0, args0, model0, sc0

    host = HostScorer(S2, device)
    results = {}
    for loss_type in LOSS_TYPES:
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
        tau_choice = float(dict(torch.load(S1, map_location='cpu')['args']).get('tau_topk', 0.1))
        tau_topk = float(dict(torch.load(S2, map_location='cpu')['args'])['tau_topk'])

        torch.manual_seed(seed)
        _, loader = exp._get_data(flag='train')
        it = iter(loader)
        channel = 0
        step_log = []
        for step in range(20):
            batch_x, batch_y, batch_start_idx = next(it)
            batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)
            cand_mask, _ = exp._candidate_mask(batch_start_idx)
            optimizer.zero_grad()
            E = encode_raw(model, exp.memory_x, channel)
            z_q = encode_raw(model, batch_x, channel)
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, channel)
            futures = memory_c + offset_c.view(-1, 1, 1)
            query_future = batch_y[:, :, channel]
            with torch.no_grad():
                host_scores = host.scores(batch_x, channel, cand_mask)
                w_host = candidate_weights(host_scores, cand_mask, tau_topk)
            losses, diags, picks, steps = run_sequence(
                z_q, E, cand_mask, sc, w_host, futures, query_future,
                'onpolicy', loss_type, tau_choice, 10, 4096)
            batch_loss = sum(losses) / 10
            batch_loss.backward()
            gn = grad_norm(params)
            optimizer.step()
            pnorm = float(sum(p.detach().float().pow(2).sum() for p in params) ** 0.5)
            has_nan = bool(torch.isnan(batch_loss)) or bool(torch.isinf(batch_loss))
            d0 = diags[0]
            step_log.append({
                'step': step, 'loss': float(batch_loss.detach()), 'grad_norm': gn, 'param_norm': pnorm,
                'has_nan_inf': has_nan,
                'chosen_normalized_regret': d0.get('chosen_normalized_regret'),
                'effective_positive_count': d0.get('effective_positive_count'),
                'teacher_entropy': d0.get('teacher_entropy'),
                'topm_probability_mass_mean': d0.get('topm_probability_mass_mean'),
            })
        results[loss_type] = step_log
        print(f'[smoke_set_loss01] {loss_type}: step0 loss={step_log[0]["loss"]:.5f} '
             f'step19 loss={step_log[-1]["loss"]:.5f} step0 grad_norm={step_log[0]["grad_norm"]:.5f} '
             f'step19 grad_norm={step_log[-1]["grad_norm"]:.5f} '
             f'any_nan_inf={any(s["has_nan_inf"] for s in step_log)}')

    Path('results/EXP-SET-LOSS01').mkdir(parents=True, exist_ok=True)
    Path('results/EXP-SET-LOSS01/smoke_test.json').write_text(
        json.dumps({'init_sha256': init_sha, 'results': results}, indent=2))
    print('[smoke_set_loss01] wrote results/EXP-SET-LOSS01/smoke_test.json')


if __name__ == '__main__':
    main()
