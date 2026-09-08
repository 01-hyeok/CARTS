#!/usr/bin/env python3
"""EXP-CORRECTION-STAGE2-SEMANTICS01 (Track B2): given the SAME reference
retrieval (Top-K membership + alpha weights, from B0's own existing
production score -- no new selector trained), does Stage-2 do better
fusing the historical CORRECTION aggregate (`C_ret = sum(alpha_i * r_i)`,
`r_i = Y_i - B_i`) than the existing FUTURE aggregate (`Y_ret = sum(alpha_i
* Y_i)`)?

Arms (identical Top-K/alpha across all three -- only VALUE and FUSION
change):
  F0 -- existing production Stage-2, unforced (value=Y_i, current fusion).
        No new code: this is exactly `b0_exp.model(...)`'s own unforced
        forward pass, already computed with this exact retrieval.
  C0 -- value=r_i, fusion `Y_final = B_q + 1.0*C_ret` (FIXED gamma=1, no
        training at all -- `layers.retrieval_gate.RetrievalGate` with
        `fixed_lambda=1.0`, reused unmodified).
  C1 -- value=r_i, fusion `Y_final = B_q + gamma*C_ret`, gamma a NEW
        scalar gate (`RetrievalGate(fixed_lambda=-1.0)`, reused unmodified
        class, freshly initialised and trained per channel on TRAIN,
        selected on VAL, evaluated on TEST -- never tuned against TEST).

B0/base_forecast and the retrieval encoder/score stay frozen throughout;
no new selector is trained (per the spec's explicit critical control).
B_q/C_ret/Y_q are precomputed ONCE per split (they do not depend on the
gate), so gate training is a fast pass over cached tensors, not a full
retrieval re-run per epoch.
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
from utils.retrieval_diagnostics import base_forecast, load_stage2, unwrap


@torch.no_grad()
def precompute_channel_features(b0_exp, b0_model, split, c, k, tau, chunk_size, device,
                                 memory_x, memory_y, r_i_full, memory_x_last):
    """Returns dict of [N_valid, pred_len] tensors: B_q, C_ret, Y_ret (future
    aggregate, for reference/F0 cross-check only), Y_q -- one row per VALID
    query in this split, for this channel, using B0's OWN existing
    production retrieval score for Top-K/alpha (same selection for every
    arm)."""
    _, loader = b0_exp._get_data(flag=split, shuffle=False)
    B_q_rows, C_ret_rows, Y_ret_rows, Y_q_rows = [], [], [], []
    f0_se, f0_n = 0.0, 0.0

    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, counts = b0_exp._candidate_mask(batch_start_idx)
        valid_query = counts.to(device) >= k
        if not bool(valid_query.any()):
            continue

        z_q_ref = b0_model._branch_embedding(batch_x, c, c)
        z_mem_ref = b0_model._branch_memory(b0_exp.key_bank, c, 0, c, z_q_ref.dtype, device)
        cosine_ref = torch.matmul(z_q_ref, z_mem_ref.transpose(0, 1))
        score_fn = b0_model._retrieval_score_fn()
        learned_ref = score_fn(z_q_ref, z_mem_ref) if score_fn is not None else cosine_ref
        neg_inf = torch.finfo(learned_ref.dtype).min / 4
        masked = learned_ref.masked_fill(~cand_mask, neg_inf)
        topk = masked.topk(k, dim=-1)
        topk_idx = topk.indices  # [B, k] -- SAME Top-K for every arm
        alpha = torch.softmax(topk.values / tau, dim=-1)  # [B, k]

        r_i_c = r_i_full[:, :, c]  # [N, pred_len]
        y_i_c = memory_y[:, :, c]  # [N, pred_len]
        C_ret = (alpha.unsqueeze(-1) * r_i_c[topk_idx]).sum(dim=1)  # [B, pred_len]
        Y_ret = (alpha.unsqueeze(-1) * y_i_c[topk_idx]).sum(dim=1)  # [B, pred_len] (reference only)

        B_q_full = base_forecast(b0_model, batch_x, chunk_size=chunk_size)
        B_q = B_q_full[:, :, c]
        Y_q = batch_y[:, :, c]

        # ---- F0 cross-check: B0's own unforced production forward, using
        # THIS exact retrieval (no forcing needed -- it already is this) ----
        y_final, y_base, y_ret, beta, lam, debug = b0_exp.model(
            batch_x=batch_x, memory_y=b0_exp.memory_y, valid_mask=cand_mask,
            key_bank=b0_exp.key_bank, memory_x_last=b0_exp.memory_x_last,
            retrieval_cache=None, target_y=batch_y,
            teacher_key_bank=getattr(b0_exp, 'teacher_key_bank', None),
        )
        # NOTE: y_final/batch_y are the FULL multivariate Stage-2 output --
        # F0 must be sliced to THIS channel only (a per-channel MSE, to be
        # comparable with B0/C0/C1 below, all of which are per-channel by
        # construction). An earlier version of this script accumulated the
        # unsliced multivariate tensor here, which produced the SAME
        # (canonical, multi-channel-averaged) 0.37312-scale number for
        # every channel regardless of c -- caught before trusting results.
        f0_se += float(((y_final[:, :, c] - batch_y[:, :, c]).pow(2))[valid_query].sum())
        f0_n += float(batch_y[:, :, c][valid_query].numel())

        vq = valid_query
        B_q_rows.append(B_q[vq].cpu())
        C_ret_rows.append(C_ret[vq].cpu())
        Y_ret_rows.append(Y_ret[vq].cpu())
        Y_q_rows.append(Y_q[vq].cpu())

    return {
        'B_q': torch.cat(B_q_rows, dim=0), 'C_ret': torch.cat(C_ret_rows, dim=0),
        'Y_ret': torch.cat(Y_ret_rows, dim=0), 'Y_q': torch.cat(Y_q_rows, dim=0),
        'f0_mse': f0_se / max(f0_n, 1),
    }


def train_gate_c1(feats_train, feats_val, pred_len, epochs, patience, lr, device):
    gate = RetrievalGate(pred_len, gate_mode='scalar', fusion_mode='residual', fixed_lambda=-1.0).to(device)
    optimizer = torch.optim.Adam(gate.parameters(), lr=lr)
    B_q_tr = feats_train['B_q'].to(device)
    C_ret_tr = feats_train['C_ret'].to(device)
    Y_q_tr = feats_train['Y_q'].to(device)
    B_q_va = feats_val['B_q'].to(device)
    C_ret_va = feats_val['C_ret'].to(device)
    Y_q_va = feats_val['Y_q'].to(device)

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
            best_state = {k: v.clone() for k, v in gate.state_dict().items()}
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
    ap.add_argument('--stage2_checkpoint', required=True)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--tau', type=float, default=None)
    ap.add_argument('--chunk_size', type=int, default=2048)
    ap.add_argument('--gate_epochs', type=int, default=200)
    ap.add_argument('--gate_patience', type=int, default=20)
    ap.add_argument('--gate_lr', type=float, default=0.01)
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    b0_exp, b0_args = load_stage2(args.stage2_checkpoint, device=device)
    b0_exp.model.to(device)
    b0_exp._ensure_memory()
    b0_exp._build_key_bank(force=True)
    b0_model = unwrap(b0_exp.model)
    b0_model.eval()
    for p in b0_model.parameters():
        p.requires_grad_(False)

    tau = float(args.tau) if args.tau is not None else float(getattr(b0_args, 'tau_topk', 0.1))
    k = int(args.top_k)
    pred_len = int(b0_args.pred_len)

    memory_x = torch.from_numpy(b0_exp.memory_bank.memory_x).float().to(device)
    memory_y = b0_exp.memory_y.to(device)
    memory_x_last = b0_exp.memory_x_last.to(device)

    with torch.no_grad():
        B_i_full = base_forecast(b0_model, memory_x, chunk_size=args.chunk_size)
        assert torch.isfinite(B_i_full).all()
        r_i_full = memory_y - B_i_full
        assert torch.isfinite(r_i_full).all()

    channels = list(b0_model.target_channels())
    for c in channels:
        sources = b0_model.source_channels(c)
        if len(sources) != 1 or int(sources[0]) != int(c):
            raise ValueError(f'EXP-CORRECTION-STAGE2-SEMANTICS01 is self-only; channel {c} has sources {sources}')

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_channel = {}
    for c in channels:
        print(f'[correction_stage2_semantics01] === channel {c}: precomputing features ===')
        feats = {}
        for split in ('train', 'val', 'test'):
            feats[split] = precompute_channel_features(
                b0_exp, b0_model, split, c, k, tau, args.chunk_size, device,
                memory_x, memory_y, r_i_full, memory_x_last)
            print(f'  {split}: n={feats[split]["Y_q"].size(0)} f0_mse={feats[split]["f0_mse"]:.5f}')

        # ---- sanity 3: C_ret=0 -> Y_final == B_q exactly ----
        zero_ret = torch.zeros_like(feats['test']['C_ret'])
        gate_fixed = RetrievalGate(pred_len, fixed_lambda=1.0)
        y_final_zero, _ = gate_fixed(feats['test']['B_q'], zero_ret)
        assert torch.allclose(y_final_zero, feats['test']['B_q'], atol=1e-6)

        # ---- C0: fixed gamma=1, no training ----
        gate_c0 = RetrievalGate(pred_len, fixed_lambda=1.0).to(device)
        with torch.no_grad():
            y_final_c0, _ = gate_c0(feats['test']['B_q'].to(device), feats['test']['C_ret'].to(device))
            c0_mse = float((y_final_c0 - feats['test']['Y_q'].to(device)).pow(2).mean())

        # ---- C1: learned gamma, train/val/test split respected ----
        gate_c1, c1_val_mse, history = train_gate_c1(
            feats['train'], feats['val'], pred_len, args.gate_epochs, args.gate_patience, args.gate_lr, device)
        with torch.no_grad():
            y_final_c1, lam_c1 = gate_c1(feats['test']['B_q'].to(device), feats['test']['C_ret'].to(device))
            c1_mse = float((y_final_c1 - feats['test']['Y_q'].to(device)).pow(2).mean())

        b0_only_mse = float((feats['test']['B_q'] - feats['test']['Y_q']).pow(2).mean())
        f0_mse = feats['test']['f0_mse']

        per_channel[int(c)] = {
            'b0_mse': b0_only_mse, 'f0_mse': f0_mse, 'c0_mse': c0_mse, 'c1_mse': c1_mse,
            'c1_val_mse_at_selection': c1_val_mse,
            'c1_gamma_mean_test': float(lam_c1.mean()), 'c1_gamma_std_test': float(lam_c1.std()),
            'c1_gamma_near_zero_frac': float((lam_c1.abs() < 0.05).float().mean()),
            'gate_history': history,
        }
        print(f'  channel {c}: B0={b0_only_mse:.5f} F0={f0_mse:.5f} C0={c0_mse:.5f} C1={c1_mse:.5f} '
              f'gamma_mean={float(lam_c1.mean()):.4f}')
        with open(out_dir / f'channel{c}_gate_history.json', 'w') as fh:
            json.dump(history, fh, indent=2)

    agg = {
        'b0_mse_mean': sum(v['b0_mse'] for v in per_channel.values()) / len(per_channel),
        'f0_mse_mean': sum(v['f0_mse'] for v in per_channel.values()) / len(per_channel),
        'c0_mse_mean': sum(v['c0_mse'] for v in per_channel.values()) / len(per_channel),
        'c1_mse_mean': sum(v['c1_mse'] for v in per_channel.values()) / len(per_channel),
        'per_channel': per_channel,
    }
    with open(out_dir / 'summary.json', 'w') as fh:
        json.dump(agg, fh, indent=2, default=str)
    print(json.dumps({k_: v for k_, v in agg.items() if k_ != 'per_channel'}, indent=2))
    print(f'[correction_stage2_semantics01] written to {out_dir}')


if __name__ == '__main__':
    main()
