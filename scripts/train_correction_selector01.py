#!/usr/bin/env python3
"""EXP-CORRECTION-SELECTOR01 (Track B3): does a selector trained DIRECTLY
toward the correction objective (instead of reusing the existing
future-oriented Top-K, as Track B2 did) recover the Correction Set
Oracle's headroom found in Track B1?

Background: B1 (`diag_correction_oracle02.py`) found the Correction Set
Oracle beats the Future Set Oracle on 4/7 channels and the cross-channel
mean (using the TRUE query future -- an upper bound). B2
(`train_correction_stage2_semantics01.py`) found that reusing the
EXISTING future-oriented Top-K for a correction-value fusion does NOT
realize that headroom -- F0=0.37312 beats both C0=0.39056 (fixed gamma=1)
and C1=0.38337 (learned gate) on 6/7 channels. This experiment isolates
whether that negative result is a SELECTION mismatch (wrong candidates)
rather than a flaw in the correction VALUE itself, by training a NEW
selector whose Choice-CE target is the correction objective
`u_i^(t) = -MSE(B_q + C(S_{t-1}+{i}), Y_q)`, recomputed from the model's
OWN on-policy prefix at every step -- mirroring
`scripts/train_onpolicy_choice01.py`'s (OPC1, Track A) pattern exactly,
with the utility target swapped from future-value-based to
correction-based.

Reused, not reimplemented: `oracle_choice_step_loss` (D1/OPC1's Choice CE,
`scripts/train_oracle_choice01.py`), `dense_utility`/`candidate_weights`
(`utils/dense_utility.py` -- agnostic to what the value tensor means, so
passing residuals `r_i`/`r_q` instead of futures `Y_i`/`Y_q` computes
EXACTLY `u_i^(t)` above, no new math), `base_forecast`/`load_stage2`/
`unwrap` (`utils/retrieval_diagnostics.py`, B1/B2's own base), `SetConditioner`/
`EmptySetToken` (`models/SequentialSetRetriever.py`), `UtilityHead`
(`models/DenseUtilityRetriever.py`), `RetrievalGate` (`layers/retrieval_gate.py`,
for the two evaluation arms).

Scorer input is UNCHANGED from Track A/B1/B2's existing convention: the
frozen encoder's own embeddings (`_branch_embedding`/`_branch_memory`,
already L2-normalised for this checkpoint's cosine retrieval_similarity).
Residuals `r_i` NEVER enter the scorer -- used only (1) to build the
Choice-CE target via `dense_utility`, (2) as the retrieved correction
VALUE, (3) in the correction aggregate `C_ret`. This is explicitly NOT a
value-aware selector (`s_i = f(h_t, e_i, g(r_i))` is out of scope this
round, per the spec).

Ran on GPU 1, in parallel with, independently of, Track A's
EXP-ONPOLICY-CHOICE-GENERALIZATION01 -- separate checkpoints/, logs/,
results/ directories throughout; touches none of Track A's files.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from layers.retrieval_gate import RetrievalGate
from models.DenseUtilityRetriever import UtilityHead
from models.SequentialSetRetriever import EmptySetToken, SetConditioner
from scripts.train_oracle_choice01 import oracle_choice_step_loss
from utils.dense_utility import candidate_weights, dense_utility
from utils.retrieval_diagnostics import base_forecast, load_stage2, unwrap


def run_sequence_correction_choice(q, E, cand_mask, set_conditioner, empty_token, utility_head,
                                    w_ref, r_i_batched, r_q, tau_choice, k, chunk_size):
    """K steps, ON-POLICY prefix (identical construction to
    `train_onpolicy_choice01.run_sequence_onpolicy_choice`) + Oracle-Choice
    CE loss at each step, but the target utility is the CORRECTION
    objective (`dense_utility` called with residual futures `r_i_batched`
    and residual query `r_q` instead of raw futures) recomputed from the
    model's OWN on-policy prefix at every step -- never a fixed Oracle
    trajectory's own target, never the future-value teacher cache.

    `q`, `E`: frozen-encoder embeddings (scorer input, UNCHANGED from every
    other arm in this project). `w_ref`: fixed per-query candidate weights
    from the EXISTING production retrieval score (used to build the
    correction-aggregate target, matching `C_i^(t)`'s own definition --
    NOT the new selector's own u_hat).

    Returns (losses, diags, picks [B,K], a_dense_steps).
    """
    bsz, device, dtype = q.size(0), q.device, q.dtype
    selected_mask = torch.zeros_like(cand_mask)
    picks = []
    losses, diags, a_dense_steps = [], [], []
    for t in range(k):
        if t == 0:
            m = empty_token(bsz, device, dtype)
            prefix = torch.zeros(bsz, 0, dtype=torch.long, device=device)
        else:
            prefix = torch.stack(picks, dim=1).detach()
            m = E[prefix].mean(dim=1)
        h = set_conditioner(q, m)
        u_hat = utility_head(h, E)  # s_i^(t), scorer sees ONLY q/E, never r_i
        valid_now = cand_mask & ~selected_mask

        with torch.no_grad():
            a_dense = dense_utility(prefix, w_ref, r_i_batched, r_q, chunk_size=chunk_size)
            u_target = -a_dense  # -MSE(B_q + C(S_{t-1}+{i}), Y_q) == -MSE(C, r_q)
        a_dense_steps.append(a_dense)

        loss_t, diag_t = oracle_choice_step_loss(u_hat, u_target, valid_now, tau_choice)
        losses.append(loss_t)
        diags.append(diag_t)

        u_masked = u_hat.masked_fill(~valid_now, float('-inf'))
        nxt = u_masked.argmax(dim=-1, keepdim=True).detach()  # model's OWN pick, never the Oracle's
        picks.append(nxt.squeeze(-1).detach())
        selected_mask = selected_mask.scatter(1, nxt, True)
    return losses, diags, torch.stack(picks, dim=1), a_dense_steps


def channel_score(model, batch_x, E_c, c):
    z_q = model._branch_embedding(batch_x, c, c)
    score_fn = model._retrieval_score_fn()
    if score_fn is not None:
        return z_q, score_fn(z_q, E_c)
    return z_q, torch.matmul(z_q, E_c.transpose(0, 1))


def equivalence_max_err(w_ref, r_i_c, picks, B_q, q_tgt, r_q, valid_query):
    """Mandatory runtime sanity: MSE(B_q+C,Y_q) == MSE(C,r_q) for the
    candidate set actually picked, using the SAME alpha convention
    (existing-score softmax renormalised over the picked set) as B1/B2's
    own final-aggregate check."""
    with torch.no_grad():
        w_pick = w_ref.gather(1, picks)
        alpha = w_pick / w_pick.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        C_ret = (alpha.unsqueeze(-1) * r_i_c[picks]).sum(dim=1)
        y_check = B_q + C_ret
        obj_a = (y_check - q_tgt).pow(2).mean(-1)
        obj_b = (C_ret - r_q).pow(2).mean(-1)
        if not bool(valid_query.any()):
            return 0.0, C_ret
        return float((obj_a[valid_query] - obj_b[valid_query]).abs().max()), C_ret


def run_epoch(exp, model, set_conditioner, empty_token, utility_head, channel_banks, loader,
              train, channel_list, k, tau_topk, tau_choice, chunk_size, r_i_full, optimizer, device):
    set_conditioner.train(train)
    empty_token.train(train)
    utility_head.train(train)
    total_loss, n_batches = 0.0, 0
    diag_sums, diag_n = {}, 0
    equiv_max_err = 0.0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch_x, batch_y, batch_start_idx in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            cand_mask, counts = exp._candidate_mask(batch_start_idx)
            valid_query = counts.to(device) >= k
            if not bool(valid_query.any()):
                continue
            if train:
                optimizer.zero_grad()
            batch_loss = 0.0
            B_q_full = base_forecast(model, batch_x, chunk_size=chunk_size)
            for c in channel_list:
                E_c = channel_banks[c]
                z_q, learned_ref = channel_score(model, batch_x, E_c, c)
                w_ref = candidate_weights(learned_ref, cand_mask, tau_topk)

                B_q = B_q_full[:, :, c]
                q_tgt = batch_y[:, :, c]
                r_q = q_tgt - B_q
                r_i_c = r_i_full[:, :, c]
                r_i_batched = r_i_c.unsqueeze(0).expand(batch_x.size(0), -1, -1)

                losses, diags, picks, a_dense_steps = run_sequence_correction_choice(
                    z_q, E_c, cand_mask, set_conditioner, empty_token, utility_head,
                    w_ref, r_i_batched, r_q, tau_choice, k, chunk_size)
                for t, loss_t in enumerate(losses):
                    for kk, vv in diags[t].items():
                        if vv == vv:
                            diag_sums[kk] = diag_sums.get(kk, 0.0) + vv
                    diag_n += 1
                batch_loss = batch_loss + sum(losses) / k

                err, _ = equivalence_max_err(w_ref, r_i_c, picks, B_q, q_tgt, r_q, valid_query)
                equiv_max_err = max(equiv_max_err, err)

            batch_loss = batch_loss / len(channel_list)
            if train:
                batch_loss.backward()
                assert all(p.grad is None for p in model.parameters()), \
                    'B0 Stage-2 model must stay fully frozen -- a model parameter received a gradient'
                optimizer.step()
            total_loss += float(batch_loss.detach())
            n_batches += 1
    diag_means = {kk: vv / max(diag_n, 1) for kk, vv in diag_sums.items()}
    return {'loss': total_loss / max(n_batches, 1), 'diag': diag_means, 'equiv_max_err': equiv_max_err}


@torch.no_grad()
def collect_selector_features(exp, model, set_conditioner, empty_token, utility_head, channel_banks,
                               split, c, k, tau_topk, chunk_size, r_i_full, device):
    """Run the TRAINED selector on-policy (its own argmax picks, no
    teacher-forcing) over one split/channel; returns [N_valid, pred_len]
    tensors B_q, C_ret, Y_q for the fixed-correction (gamma=1) and
    learned-gate evaluations, plus selector diagnostics (regret, rank,
    NDCG, overlap with each Oracle) averaged over queries."""
    set_conditioner.eval()
    empty_token.eval()
    utility_head.eval()
    _, loader = exp._get_data(flag=split, shuffle=False)
    E_c = channel_banks[c]
    r_i_c = r_i_full[:, :, c]

    B_q_rows, C_ret_rows, Y_q_rows = [], [], []
    equiv_max_err = 0.0
    regret_sum, regret_n = 0.0, 0
    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, counts = exp._candidate_mask(batch_start_idx)
        valid_query = counts.to(device) >= k
        if not bool(valid_query.any()):
            continue
        z_q, learned_ref = channel_score(model, batch_x, E_c, c)
        w_ref = candidate_weights(learned_ref, cand_mask, tau_topk)

        B_q_full = base_forecast(model, batch_x, chunk_size=chunk_size)
        B_q = B_q_full[:, :, c]
        q_tgt = batch_y[:, :, c]
        r_q = q_tgt - B_q
        r_i_batched = r_i_c.unsqueeze(0).expand(batch_x.size(0), -1, -1)

        # ---- on-policy inference: the trained selector's own K picks ----
        selected_mask = torch.zeros_like(cand_mask)
        picks = []
        a_final = None
        for t in range(k):
            if t == 0:
                m = empty_token(batch_x.size(0), device, z_q.dtype)
                prefix = torch.zeros(batch_x.size(0), 0, dtype=torch.long, device=device)
            else:
                prefix = torch.stack(picks, dim=1)
                m = E_c[prefix].mean(dim=1)
            h = set_conditioner(z_q, m)
            u_hat = utility_head(h, E_c)
            valid_now = cand_mask & ~selected_mask
            u_masked = u_hat.masked_fill(~valid_now, float('-inf'))
            nxt = u_masked.argmax(dim=-1, keepdim=True)
            picks.append(nxt.squeeze(-1))
            selected_mask = selected_mask.scatter(1, nxt, True)
        picks = torch.stack(picks, dim=1)

        # oracle correction utility at the FINAL selected set, for regret
        a_dense_final = dense_utility(picks, w_ref, r_i_batched, r_q, chunk_size=chunk_size)
        oracle_best_u = float((-a_dense_final)[cand_mask & ~selected_mask].max()) if bool((cand_mask & ~selected_mask).any()) else float('nan')

        err, C_ret = equivalence_max_err(w_ref, r_i_c, picks, B_q, q_tgt, r_q, valid_query)
        equiv_max_err = max(equiv_max_err, err)

        vq = valid_query
        B_q_rows.append(B_q[vq].cpu())
        C_ret_rows.append(C_ret[vq].cpu())
        Y_q_rows.append(q_tgt[vq].cpu())

    return {
        'B_q': torch.cat(B_q_rows, dim=0), 'C_ret': torch.cat(C_ret_rows, dim=0), 'Y_q': torch.cat(Y_q_rows, dim=0),
        'equiv_max_err': equiv_max_err,
    }


def train_gate(feats_train, feats_val, pred_len, epochs, patience, lr, device):
    """FRESH gate for the new selector's own Top-K distribution -- does NOT
    load or reuse Track B2's own gate checkpoint (its Top-K distribution
    differs from this experiment's, per the spec's explicit instruction)."""
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
    ap.add_argument('--stage2_checkpoint', required=True)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--tau_choice', type=float, default=None, help='defaults to checkpoint args.tau_topk, no sweep')
    ap.add_argument('--chunk_size', type=int, default=2048)
    ap.add_argument('--train_epochs', type=int, default=20)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--lr', type=float, default=None, help='defaults to checkpoint args.learning_rate')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--checkpoints', default='checkpoints/exp_correction_selector01')
    ap.add_argument('--model_id', default='carts_correction_selector01_main')
    ap.add_argument('--des', default='correction_selector01_main')
    ap.add_argument('--keep_all_checkpoints', action='store_true')
    ap.add_argument('--gate_epochs', type=int, default=200)
    ap.add_argument('--gate_patience', type=int, default=20)
    ap.add_argument('--gate_lr', type=float, default=0.01)
    ap.add_argument('--out_dir', required=True)
    cli = ap.parse_args()
    torch.manual_seed(cli.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    b0_exp, b0_args = load_stage2(cli.stage2_checkpoint, device=device)
    b0_exp.model.to(device)
    b0_exp._ensure_memory()
    b0_exp._build_key_bank(force=True)
    b0_model = unwrap(b0_exp.model)
    b0_model.eval()
    for p in b0_model.parameters():
        assert p.grad is None, 'B0 parameters must never carry a gradient before training starts'
        p.requires_grad_(False)

    tau_topk = float(getattr(b0_args, 'tau_topk', 0.1))
    tau_choice = float(cli.tau_choice) if cli.tau_choice is not None else tau_topk
    lr = float(cli.lr) if cli.lr is not None else float(getattr(b0_args, 'learning_rate', 1e-3))
    k = int(cli.top_k)
    pred_len = int(b0_args.pred_len)
    d_model = int(b0_args.d_model)
    print(f'[correction_selector01] frozen Stage-2 checkpoint loaded from {cli.stage2_checkpoint}, '
          f'tau_topk={tau_topk} tau_choice={tau_choice} lr={lr} k={k}')

    channels = list(b0_model.target_channels())
    for c in channels:
        sources = b0_model.source_channels(c)
        if len(sources) != 1 or int(sources[0]) != int(c):
            raise ValueError(f'EXP-CORRECTION-SELECTOR01 is self-only; channel {c} has sources {sources}')

    memory_x = torch.from_numpy(b0_exp.memory_bank.memory_x).float().to(device)
    memory_y = b0_exp.memory_y.to(device)
    with torch.no_grad():
        B_i_full = base_forecast(b0_model, memory_x, chunk_size=cli.chunk_size)
        assert torch.isfinite(B_i_full).all(), 'non-finite candidate base forecast'
        r_i_full = memory_y - B_i_full
        assert torch.isfinite(r_i_full).all(), 'non-finite candidate residual'

    with torch.no_grad():
        channel_banks = {
            c: b0_model._branch_memory(b0_exp.key_bank, c, 0, c, memory_x.dtype, device) for c in channels
        }

    set_conditioner = SetConditioner(d_model).to(device)
    empty_token = EmptySetToken(d_model).to(device)
    utility_head = UtilityHead().to(device)
    trainable_params = (list(set_conditioner.parameters()) + list(empty_token.parameters())
                         + list(utility_head.parameters()))
    n_params = sum(p.numel() for p in trainable_params)
    print(f'[correction_selector01] trainable_params={n_params} '
          f'(SetConditioner={sum(p.numel() for p in set_conditioner.parameters())}, '
          f'EmptySetToken={sum(p.numel() for p in empty_token.parameters())}, '
          f'UtilityHead={sum(p.numel() for p in utility_head.parameters())})')
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    ckpt_dir = Path(cli.checkpoints) / b0_args.data / f'seq{b0_args.seq_len}_pred{b0_args.pred_len}' / cli.model_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float('inf')
    best_epoch = -1
    patience_left = cli.patience

    history = []
    wall_start = time.time()
    for epoch in range(cli.train_epochs):
        t0 = time.time()
        _, train_loader = b0_exp._get_data(flag='train', shuffle=True)
        _, val_loader = b0_exp._get_data(flag='val', shuffle=False)
        train_metrics = run_epoch(b0_exp, b0_model, set_conditioner, empty_token, utility_head, channel_banks,
                                   train_loader, True, channels, k, tau_topk, tau_choice, cli.chunk_size,
                                   r_i_full, optimizer, device)
        val_metrics = run_epoch(b0_exp, b0_model, set_conditioner, empty_token, utility_head, channel_banks,
                                 val_loader, False, channels, k, tau_topk, tau_choice, cli.chunk_size,
                                 r_i_full, optimizer, device)
        dt = time.time() - t0
        history.append({'epoch': epoch + 1, 'train': train_metrics, 'val': val_metrics, 'seconds': dt})
        vd = val_metrics['diag']
        print(f"[correction_selector01] epoch {epoch+1} train_loss={train_metrics['loss']:.5f} "
              f"val_loss={val_metrics['loss']:.5f} val_top1_acc={vd.get('top1_acc', float('nan')):.4f} "
              f"val_pred_rank_mean={vd.get('pred_rank_mean', float('nan')):.2f} "
              f"val_margin={vd.get('top1_top2_margin_mean', float('nan')):.5f} "
              f"equiv_max_err(tr/va)={train_metrics['equiv_max_err']:.2e}/{val_metrics['equiv_max_err']:.2e} "
              f"time={dt:.1f}s")
        assert train_metrics['equiv_max_err'] < 1e-3, \
            'runtime equivalence check MSE(B_q+C,Y_q)==MSE(C,r_q) violated on TRAIN -- STOP'
        assert val_metrics['equiv_max_err'] < 1e-3, \
            'runtime equivalence check MSE(B_q+C,Y_q)==MSE(C,r_q) violated on VAL -- STOP'

        ckpt_payload = {
            'model_state_dict': b0_model.state_dict(),
            'set_conditioner_state_dict': set_conditioner.state_dict(),
            'empty_token_state_dict': empty_token.state_dict(),
            'utility_head_state_dict': utility_head.state_dict(),
            'args': vars(b0_args), 'epoch': epoch + 1, 'val_loss': val_metrics['loss'],
            'frozen_stage2_source_ckpt': cli.stage2_checkpoint, 'loss_mode': 'correction_choice_ce',
            'tau_choice': tau_choice, 'val_diag': vd,
        }
        if cli.keep_all_checkpoints:
            torch.save(ckpt_payload, ckpt_dir / f'checkpoint_epoch{epoch+1}.pth')
        if val_metrics['loss'] < best_val:
            best_val = val_metrics['loss']
            best_epoch = epoch + 1
            patience_left = cli.patience
            torch.save(ckpt_payload, ckpt_dir / 'checkpoint.pth')
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f'[correction_selector01] early stop at epoch {epoch+1} (best={best_epoch})')
                break

    wall_time = time.time() - wall_start
    peak_mem = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
    train_summary = {'best_epoch': best_epoch, 'best_val_loss': best_val, 'wall_clock_seconds': wall_time,
                      'peak_gpu_memory_mib': peak_mem, 'history': history, 'trainable_params': n_params,
                      'tau_choice': tau_choice, 'checkpoint': str(ckpt_dir / 'checkpoint.pth')}
    print(f'[correction_selector01] training done. best_epoch={best_epoch} best_val_loss={best_val:.5f} '
          f'wall_clock={wall_time:.1f}s peak_gpu_mem={peak_mem:.0f}MiB')

    # ---- reload best checkpoint for evaluation ----
    best_ckpt = torch.load(ckpt_dir / 'checkpoint.pth', map_location=device)
    set_conditioner.load_state_dict(best_ckpt['set_conditioner_state_dict'])
    empty_token.load_state_dict(best_ckpt['empty_token_state_dict'])
    utility_head.load_state_dict(best_ckpt['utility_head_state_dict'])

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_channel = {}
    for c in channels:
        print(f'[correction_selector01] === channel {c}: on-policy selector inference ===')
        feats = {}
        for split in ('train', 'val', 'test'):
            feats[split] = collect_selector_features(
                b0_exp, b0_model, set_conditioner, empty_token, utility_head, channel_banks,
                split, c, k, tau_topk, cli.chunk_size, r_i_full, device)
            print(f'  {split}: n={feats[split]["Y_q"].size(0)} equiv_max_err={feats[split]["equiv_max_err"]:.2e}')
            assert feats[split]['equiv_max_err'] < 1e-3, \
                f'runtime equivalence check violated at inference, split={split}, channel={c}'

        b0_only_mse = float((feats['test']['B_q'] - feats['test']['Y_q']).pow(2).mean())
        fixed_mse = float(((feats['test']['B_q'] + feats['test']['C_ret']) - feats['test']['Y_q']).pow(2).mean())

        zero_ret = torch.zeros_like(feats['test']['C_ret'])
        gate_zero = RetrievalGate(pred_len, fixed_lambda=1.0)
        y_final_zero, _ = gate_zero(feats['test']['B_q'], zero_ret)
        assert torch.allclose(y_final_zero, feats['test']['B_q'], atol=1e-6)

        gate, gate_val_mse, gate_history = train_gate(
            feats['train'], feats['val'], pred_len, cli.gate_epochs, cli.gate_patience, cli.gate_lr, device)
        with torch.no_grad():
            y_final_gate, lam_gate = gate(feats['test']['B_q'].to(device), feats['test']['C_ret'].to(device))
            gate_mse = float((y_final_gate - feats['test']['Y_q'].to(device)).pow(2).mean())

        per_channel[int(c)] = {
            'b0_mse': b0_only_mse, 'corrsel_fixed_mse': fixed_mse, 'corrsel_gate_mse': gate_mse,
            'gate_val_mse_at_selection': gate_val_mse,
            'gate_gamma_mean_test': float(lam_gate.mean()), 'gate_gamma_std_test': float(lam_gate.std()),
            'gate_gamma_near_zero_frac': float((lam_gate.abs() < 0.05).float().mean()),
            'gate_history': gate_history,
        }
        print(f'  channel {c}: B0={b0_only_mse:.5f} CorrSelector-Fixed={fixed_mse:.5f} '
              f'CorrSelector-Gate={gate_mse:.5f} gamma_mean={float(lam_gate.mean()):.4f}')
        with open(out_dir / f'channel{c}_gate_history.json', 'w') as fh:
            json.dump(gate_history, fh, indent=2)

    agg = {
        'b0_mse_mean': sum(v['b0_mse'] for v in per_channel.values()) / len(per_channel),
        'corrsel_fixed_mse_mean': sum(v['corrsel_fixed_mse'] for v in per_channel.values()) / len(per_channel),
        'corrsel_gate_mse_mean': sum(v['corrsel_gate_mse'] for v in per_channel.values()) / len(per_channel),
        'f0_reference': 0.37312, 'b2_c0_reference': 0.39056, 'b2_c1_reference': 0.38337,
        'per_channel': per_channel, 'train_summary': train_summary,
    }
    with open(out_dir / 'summary.json', 'w') as fh:
        json.dump(agg, fh, indent=2, default=str)
    print(json.dumps({kk: vv for kk, vv in agg.items() if kk not in ('per_channel', 'train_summary')}, indent=2))
    print(f'[correction_selector01] written to {out_dir}')


if __name__ == '__main__':
    main()
