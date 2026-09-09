#!/usr/bin/env python3
"""Track B4 (EXP-CORRECTION-ENCODER01) -- Stage-2: for ONE encoder
(`--encoder {future,correction}`), retrieve Top-K by that encoder's OWN
score, fuse the historical CORRECTION aggregate (`C_ret = sum(alpha_i *
r_i)`, `r_i = Y_i - B_i`, exactly Track B2's own definition) with the
frozen Base Predictor's forecast. Two evaluation arms, mirroring B2/B3
exactly: Fixed (`RetrievalGate(fixed_lambda=1.0)`, no training) and Gate (a
FRESH `RetrievalGate`, trained per channel, train/val/test discipline
respected, never tuned against TEST).

Base Predictor and the encoder producing Top-K/alpha are BOTH frozen
throughout -- only the gate (Gate arm) is trained. `--encoder future` uses
the SAME production Stage-2 checkpoint as Track B1/B2/B3 (never retrained);
`--encoder correction` uses a `scripts/train_correction_encoder01.py`
checkpoint.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from layers.retrieval_gate import RetrievalGate
from scripts.eval_correction_encoder01 import future_score
from scripts.train_margutil01 import build_experiment
from scripts.train_oracle_scratch01 import base_score, encode_raw
from utils.retrieval_diagnostics import base_forecast, load_stage2, unwrap


@torch.no_grad()
def precompute_channel_features(exp, b0_model, c_model, encoder_kind, split, c, k, tau, chunk_size,
                                 device, r_i_full):
    _, loader = exp._get_data(flag=split, shuffle=False)
    B_q_rows, C_ret_rows, Y_q_rows = [], [], []
    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, counts = exp._candidate_mask(batch_start_idx)
        valid_query = counts.to(device) >= k
        if not bool(valid_query.any()):
            continue

        if encoder_kind == 'future':
            learned_ref = future_score(b0_model, batch_x, exp.memory_x, c)
        else:
            z_q = encode_raw(c_model, batch_x, c)
            E = encode_raw(c_model, exp.memory_x, c)
            learned_ref = base_score(z_q, E, None)

        neg_inf = torch.finfo(learned_ref.dtype).min / 4
        masked = learned_ref.masked_fill(~cand_mask, neg_inf)
        topk = masked.topk(k, dim=-1)
        topk_idx = topk.indices
        alpha = torch.softmax(topk.values / tau, dim=-1)

        r_i_c = r_i_full[:, :, c]
        C_ret = (alpha.unsqueeze(-1) * r_i_c[topk_idx]).sum(dim=1)

        B_q_full = base_forecast(b0_model, batch_x, chunk_size=chunk_size)
        B_q = B_q_full[:, :, c]
        Y_q = batch_y[:, :, c]

        vq = valid_query
        B_q_rows.append(B_q[vq].cpu())
        C_ret_rows.append(C_ret[vq].cpu())
        Y_q_rows.append(Y_q[vq].cpu())

    return {'B_q': torch.cat(B_q_rows, dim=0), 'C_ret': torch.cat(C_ret_rows, dim=0), 'Y_q': torch.cat(Y_q_rows, dim=0)}


def train_gate(feats_train, feats_val, pred_len, epochs, patience, lr, device):
    gate = RetrievalGate(pred_len, gate_mode='scalar', fusion_mode='residual', fixed_lambda=-1.0).to(device)
    optimizer = torch.optim.Adam(gate.parameters(), lr=lr)
    B_q_tr, C_ret_tr, Y_q_tr = (feats_train[k_].to(device) for k_ in ('B_q', 'C_ret', 'Y_q'))
    B_q_va, C_ret_va, Y_q_va = (feats_val[k_].to(device) for k_ in ('B_q', 'C_ret', 'Y_q'))

    best_val = float('inf')
    best_state = None
    patience_left = patience
    history = []
    for epoch in range(epochs):
        gate.train()
        optimizer.zero_grad()
        y_final, lam = gate(B_q_tr, C_ret_tr)
        loss = (y_final - Y_q_tr).pow(2).mean()
        loss.backward()
        optimizer.step()
        gate.eval()
        with torch.no_grad():
            y_final_va, lam_va = gate(B_q_va, C_ret_va)
            val_mse = float((y_final_va - Y_q_va).pow(2).mean())
        history.append({'epoch': epoch + 1, 'train_mse': float(loss.detach()), 'val_mse': val_mse,
                         'gamma_mean': float(lam_va.mean()), 'gamma_std': float(lam_va.std())})
        if val_mse < best_val:
            best_val = val_mse
            best_state = {kk: vv.clone() for kk, vv in gate.state_dict().items()}
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break
    gate.load_state_dict(best_state)
    gate.eval()
    return gate, best_val, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--stage2_checkpoint', required=True)
    ap.add_argument('--encoder', choices=['future', 'correction'], required=True)
    ap.add_argument('--correction_encoder_checkpoint', default=None)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--tau', type=float, default=0.1)
    ap.add_argument('--chunk_size', type=int, default=2048)
    ap.add_argument('--gate_epochs', type=int, default=200)
    ap.add_argument('--gate_patience', type=int, default=20)
    ap.add_argument('--gate_lr', type=float, default=0.01)
    ap.add_argument('--out_dir', required=True)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    overrides = {'is_training': 0, 'model_id': 'correction_encoder01_stage2', 'des': 'stage2',
                 'checkpoints': '/tmp/exp_correction_encoder01_stage2', 'stage1_retrieval_metric': 'cosine',
                 'pred_len': cli.pred_len, 'seq_len': cli.pred_len}
    exp, args = build_experiment(cli.reference_ckpt, overrides)
    exp._ensure_memory()

    b0_exp, b0_args = load_stage2(cli.stage2_checkpoint, device=device)
    b0_exp.model.to(device)
    b0_model = unwrap(b0_exp.model)
    b0_model.eval()
    for p in b0_model.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        B_i_full = base_forecast(b0_model, exp.memory_x, chunk_size=cli.chunk_size)
        r_i_full = exp.memory_y - B_i_full

    c_model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    if cli.encoder == 'correction':
        ck = torch.load(cli.correction_encoder_checkpoint, map_location='cpu')
        c_model.load_state_dict(ck['model_state_dict'], strict=True)
    c_model.to(device)
    c_model.eval()
    for p in c_model.parameters():
        p.requires_grad_(False)

    channels = list(c_model.target_channels())
    k = int(cli.top_k)
    pred_len = int(cli.pred_len)

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_channel = {}
    for c in channels:
        print(f'[correction_encoder01_stage2] encoder={cli.encoder} channel {c}: precomputing features...')
        feats = {}
        for split in ('train', 'val', 'test'):
            feats[split] = precompute_channel_features(exp, b0_model, c_model, cli.encoder, split, c, k,
                                                         cli.tau, cli.chunk_size, device, r_i_full)

        gate_fixed = RetrievalGate(pred_len, fixed_lambda=1.0)
        with torch.no_grad():
            y_final_c0, _ = gate_fixed(feats['test']['B_q'], feats['test']['C_ret'])
            c0_mse = float((y_final_c0 - feats['test']['Y_q']).pow(2).mean())

        gate_c1, c1_val_mse, history = train_gate(feats['train'], feats['val'], pred_len, cli.gate_epochs,
                                                    cli.gate_patience, cli.gate_lr, device)
        with torch.no_grad():
            y_final_c1, lam_c1 = gate_c1(feats['test']['B_q'].to(device), feats['test']['C_ret'].to(device))
            c1_mse = float((y_final_c1 - feats['test']['Y_q'].to(device)).pow(2).mean())

        b0_only_mse = float((feats['test']['B_q'] - feats['test']['Y_q']).pow(2).mean())
        per_channel[int(c)] = {
            'b0_mse': b0_only_mse, 'fixed_mse': c0_mse, 'gate_mse': c1_mse, 'gate_val_mse_at_selection': c1_val_mse,
            'gamma_mean_test': float(lam_c1.mean()), 'gamma_std_test': float(lam_c1.std()),
        }
        print(f'  channel {c}: B0={b0_only_mse:.5f} Fixed={c0_mse:.5f} Gate={c1_mse:.5f} gamma_mean={float(lam_c1.mean()):.4f}')

    agg = {
        'b0_mse_mean': sum(v['b0_mse'] for v in per_channel.values()) / len(per_channel),
        'fixed_mse_mean': sum(v['fixed_mse'] for v in per_channel.values()) / len(per_channel),
        'gate_mse_mean': sum(v['gate_mse'] for v in per_channel.values()) / len(per_channel),
        'encoder': cli.encoder, 'per_channel': per_channel,
    }
    with open(out_dir / 'summary.json', 'w') as fh:
        json.dump(agg, fh, indent=2, default=str)
    print(json.dumps({kk: vv for kk, vv in agg.items() if kk != 'per_channel'}, indent=2))
    print(f'[correction_encoder01_stage2] written to {out_dir}')


if __name__ == '__main__':
    main()
