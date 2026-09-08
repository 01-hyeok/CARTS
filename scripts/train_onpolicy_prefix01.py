#!/usr/bin/env python3
"""EXP-ONPOLICY-PREFIX01: is R2's free-running failure caused (at least in
part) by the train-time Oracle-prefix / inference-time model-prefix
state-distribution mismatch, independent of the loss formula itself?

T1 is IDENTICAL to C0 (EXP-TOPTAIL-RANK01/R2, reused verbatim, not
retrained) except for ONE change: at every teacher-forced step, the SET
PREFIX used to build both the conditioning state `m` and the dense-utility
TARGET is the model's own on-policy pick history `S_hat_{t-1}`, not the
Set Oracle's `S*_{t-1}`. The LOSS FORMULA stays exactly R2 (SmoothL1 +
lambda_rank * pairwise, imported from `scripts/train_toptail_rank01.py`
unmodified) -- this experiment deliberately does NOT combine with
EXP-ORACLE-CHOICE01's Oracle-Choice CE loss, so exactly one mechanism
(prefix/state distribution) is varied at a time, per the approved spec.

Encoder stays FROZEN (same B0 checkpoint as C0) -- EXP-ENCODER-ANCHOR01's
trainable-encoder result must not be reused here.

Gradient semantics: candidate SELECTION (`argmax` over `u_hat`) is always
computed under `torch.no_grad()`/via a detached index tensor -- this is
NOT a differentiable-Top-K experiment. Only the R2 loss's own backward
(through `SetConditioner`/`UtilityHead`, from the SmoothL1/pairwise terms
evaluated at the on-policy state) carries gradient; the on-policy state
CONSTRUCTION itself (`dense_utility` on the model's own prefix, and the
`argmax` that produces the next pick) never receives gradient, matching
`utils.dense_utility.dense_utility`'s own `@torch.no_grad()` internal
convention and this project's established `run_sequence_dense` pattern
(EXP-MARGUTIL01) of never differentiating through prefix selection.
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

from models.DenseUtilityRetriever import UtilityHead
from models.SequentialSetRetriever import EmptySetToken, SetConditioner
from scripts.train_margutil01 import build_experiment, encode, memory_value
from scripts.train_toptail_rank01 import overlap_at_k, step_losses
from utils.dense_utility import candidate_weights, dense_utility


def run_sequence_onpolicy(q, E, cand_mask, set_conditioner, empty_token, utility_head,
                           futures, query_future, tau, k, chunk_size):
    """K steps, ON-POLICY: both the conditioning state `m` and the dense
    utility TARGET at every step are built from the MODEL's own prefix
    `S_hat_{t-1}` (never the oracle's). `dense_utility` (already
    `@torch.no_grad()` internally) is recomputed fresh at every step from
    the CURRENT on-policy prefix -- this is the one substantive difference
    from `train_margutil01.run_sequence_dense`'s teacher-forced branch,
    which uses the fixed oracle `teacher_idx` prefix throughout.

    Returns (u_hat_steps, u_target_steps, valid_steps, picks [B,K],
    a_dense_steps: list[B,N] raw A (not negated) for per-step regret).
    """
    bsz, device, dtype = q.size(0), q.device, q.dtype
    w = candidate_weights(torch.matmul(q, E.transpose(0, 1)), cand_mask, tau)
    selected_mask = torch.zeros_like(cand_mask)
    picks = []
    u_hat_steps, u_target_steps, valid_steps, a_dense_steps = [], [], [], []
    for t in range(k):
        if t == 0:
            m = empty_token(bsz, device, dtype)
            prefix = torch.zeros(bsz, 0, dtype=torch.long, device=device)
        else:
            prefix = torch.stack(picks, dim=1).detach()  # model's own on-policy prefix
            m = E[prefix].mean(dim=1)
        h = set_conditioner(q, m)
        u_hat = utility_head(h, E)
        valid_now = cand_mask & ~selected_mask
        u_hat_steps.append(u_hat)
        valid_steps.append(valid_now)

        with torch.no_grad():
            a_dense = dense_utility(prefix, w, futures, query_future, chunk_size=chunk_size)
        u_target_steps.append(-a_dense)
        a_dense_steps.append(a_dense)

        u_masked = u_hat.masked_fill(~valid_now, float('-inf'))
        nxt = u_masked.argmax(dim=-1, keepdim=True).detach()  # NEVER differentiated through
        picks.append(nxt.squeeze(-1).detach())
        selected_mask = selected_mask.scatter(1, nxt, True)
    return u_hat_steps, u_target_steps, valid_steps, torch.stack(picks, dim=1), a_dense_steps


def run_epoch_onpolicy(exp, args, model, set_conditioner, empty_token, utility_head, teacher, loader,
                        split, train, channel_list, k, tau, chunk_size, lambda_rank, optimizer, device):
    set_conditioner.train(train)
    empty_token.train(train)
    utility_head.train(train)
    total_loss, n_batches = 0.0, 0
    overlap_sum, overlap_n = 0.0, 0
    sc_grad_norm = 0.0
    diag_sums = {}
    diag_n = 0
    onpolicy_diag_sums = {}
    onpolicy_diag_n = 0

    def teacher_rows(batch_start_idx, channel, split_name):
        rows = [teacher['splits'][split_name]['start_to_row'][int(s)] for s in batch_start_idx.tolist()]
        return teacher['splits'][split_name]['teacher_idx'][channel][rows].to(device)

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch_x, batch_y, batch_start_idx in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            cand_mask, counts = exp._candidate_mask(batch_start_idx)
            if train:
                optimizer.zero_grad()
            batch_loss = 0.0
            for c in channel_list:
                E = encode(model, exp.memory_x, c)
                q = encode(model, batch_x, c)
                memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
                futures = memory_c + offset_c.view(-1, 1, 1)
                query_future = batch_y[:, :, c]

                teacher_idx = teacher_rows(batch_start_idx, c, split)
                query_valid = teacher_idx[:, 0] != -1

                u_hat_steps, u_target_steps, valid_steps, picks, a_dense_steps = run_sequence_onpolicy(
                    q, E, cand_mask, set_conditioner, empty_token, utility_head,
                    futures, query_future, tau, k, chunk_size)
                losses, diags = step_losses(u_hat_steps, u_target_steps, valid_steps,
                                             query_valid, 'hybrid', lambda_rank)
                batch_loss = batch_loss + sum(losses) / k
                for d in diags:
                    for kk, vv in d.items():
                        if vv == vv:
                            diag_sums[kk] = diag_sums.get(kk, 0.0) + vv
                diag_n += sum(1 for d in diags if d)

                with torch.no_grad():
                    vq = query_valid
                    if (not train) and bool(vq.any()):
                        # ---- mandatory state diagnostics: VAL split only (matches
                        # this project's convention of computing extra diagnostics
                        # only during validation, e.g. R2's own val-only picks pass;
                        # also avoids a python-loop-heavy cost on every train batch) ----
                        oracle_picks = teacher_idx.clamp_min(0)  # [B, K]
                        model_picks = picks  # [B, K]
                        first_div = torch.full((int(vq.sum()),), k, device=device)
                        overlap_by_t = []
                        divergence_by_t = []
                        for t in range(k):
                            ov = (model_picks[:, :t + 1] == oracle_picks[:, :t + 1]).float()
                            # overlap defined as SET intersection / (t+1), not positional match
                            set_overlap = torch.tensor([
                                len(set(model_picks[b, :t + 1].tolist()) & set(oracle_picks[b, :t + 1].tolist())) / (t + 1)
                                for b in range(model_picks.size(0)) if bool(vq[b])
                            ], device=device)
                            overlap_by_t.append(float(set_overlap.mean()) if set_overlap.numel() else float('nan'))
                            pos_match = (model_picks[vq, t] == oracle_picks[vq, t])
                            divergence_by_t.append(float((~pos_match).float().mean()) if pos_match.numel() else float('nan'))
                        # first divergence step (positional)
                        pos_eq = (model_picks[vq] == oracle_picks[vq])  # [Bv, K]
                        for row in range(pos_eq.size(0)):
                            neq = (~pos_eq[row]).nonzero(as_tuple=True)[0]
                            first_div[row] = int(neq[0]) if neq.numel() > 0 else k
                        per_step_regret = [float(
                            (a_dense_steps[t][vq].gather(1, model_picks[vq, t:t + 1]).squeeze(-1)
                             - a_dense_steps[t][vq].masked_fill(~valid_steps[t][vq], float('inf')).min(dim=-1).values
                             ).mean()
                        ) for t in range(k)]
                        final_agg = float(a_dense_steps[-1][vq].gather(
                            1, model_picks[vq, -1:]).squeeze(-1).mean())
                        onpolicy_diag_sums.setdefault('first_divergence_step_mean', 0.0)
                        onpolicy_diag_sums['first_divergence_step_mean'] += float(first_div.float().mean())
                        for t in range(k):
                            onpolicy_diag_sums.setdefault(f'prefix_overlap_t{t+1}', 0.0)
                            onpolicy_diag_sums[f'prefix_overlap_t{t+1}'] += overlap_by_t[t]
                            onpolicy_diag_sums.setdefault(f'divergence_rate_t{t+1}', 0.0)
                            onpolicy_diag_sums[f'divergence_rate_t{t+1}'] += divergence_by_t[t]
                            onpolicy_diag_sums.setdefault(f'per_step_regret_t{t+1}', 0.0)
                            onpolicy_diag_sums[f'per_step_regret_t{t+1}'] += per_step_regret[t]
                        onpolicy_diag_sums.setdefault('final_free_running_aggregate', 0.0)
                        onpolicy_diag_sums['final_free_running_aggregate'] += final_agg
                        onpolicy_diag_n += 1

                if not train:
                    overlap_sum += overlap_at_k(picks, teacher_idx, query_valid) * int(query_valid.sum())
                    overlap_n += int(query_valid.sum())

            batch_loss = batch_loss / len(channel_list)
            if train:
                batch_loss.backward()
                assert all(p.grad is None for p in model.encoder.parameters()), \
                    'encoder must stay frozen -- an encoder parameter received a gradient'
                sc_grads = [p.grad for p in set_conditioner.parameters() if p.grad is not None]
                sc_grad_norm = sum(g.norm().item() ** 2 for g in sc_grads) ** 0.5 if sc_grads else 0.0
                optimizer.step()
            total_loss += float(batch_loss.detach())
            n_batches += 1
    overlap = overlap_sum / max(overlap_n, 1)
    diag_means = {kk: vv / max(diag_n, 1) for kk, vv in diag_sums.items()}
    onpolicy_means = {kk: vv / max(onpolicy_diag_n, 1) for kk, vv in onpolicy_diag_sums.items()}
    return {'loss': total_loss / max(n_batches, 1), 'overlap_at_k': overlap,
            'set_conditioner_grad_norm': sc_grad_norm, 'diag': diag_means, 'onpolicy_diag': onpolicy_means}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_ckpt', required=True)
    ap.add_argument('--teacher_cache', required=True)
    ap.add_argument('--checkpoints', default='checkpoints/exp_onpolicy_prefix01')
    ap.add_argument('--model_id', default='carts_onpolicy_prefix01_main')
    ap.add_argument('--des', default='onpolicy_prefix01_main')
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--train_epochs', type=int, default=0)
    ap.add_argument('--patience', type=int, default=0)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lambda_rank', type=float, default=1.0)
    ap.add_argument('--keep_all_checkpoints', action='store_true')
    cli = ap.parse_args()

    overrides = {
        'is_training': 1, 'model_id': cli.model_id, 'des': cli.des,
        'checkpoints': cli.checkpoints, 'seed': cli.seed, 'top_k': cli.top_k,
        'stage1_residual_teacher': 0, 'stage1_query_base_conditioning': 0,
        'stage1_candidate_residual_conditioning': 0, 'stage1_retrieval_metric': 'cosine',
        'stage1_full_memory_gradient_mode': 'full_online',
    }
    if cli.train_epochs:
        overrides['train_epochs'] = cli.train_epochs
    if cli.patience:
        overrides['patience'] = cli.patience

    exp, args = build_experiment(cli.base_ckpt, overrides)
    torch.manual_seed(args.seed)
    device = exp.device
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    k = int(cli.top_k)
    tau = float(args.tau_topk)

    b0_ckpt = torch.load(cli.base_ckpt, map_location='cpu')
    model.load_state_dict(b0_ckpt['model_state_dict'], strict=True)
    for p in model.encoder.parameters():
        p.requires_grad = False
    model.encoder.eval()
    print(f'[onpolicy_prefix01] frozen encoder loaded from {cli.base_ckpt} (requires_grad=False)')

    exp._ensure_memory()
    channels = list(model.target_channels())
    for c in channels:
        sources = model.source_channels(c)
        if len(sources) != 1 or int(sources[0]) != int(c):
            raise ValueError(f'EXP-ONPOLICY-PREFIX01 is self-only; channel {c} has sources {sources}')
    teacher = torch.load(cli.teacher_cache, map_location='cpu')
    if teacher['meta']['top_k'] != k:
        raise ValueError(f"teacher cache top_k={teacher['meta']['top_k']} != --top_k={k}")

    d_model = int(args.d_model)
    set_conditioner = SetConditioner(d_model).to(device)
    empty_token = EmptySetToken(d_model).to(device)
    utility_head = UtilityHead().to(device)
    trainable_params = (list(set_conditioner.parameters()) + list(empty_token.parameters())
                         + list(utility_head.parameters()))
    n_params = sum(p.numel() for p in trainable_params)
    print(f'[onpolicy_prefix01] trainable_params={n_params} '
          f'(SetConditioner={sum(p.numel() for p in set_conditioner.parameters())}, '
          f'EmptySetToken={sum(p.numel() for p in empty_token.parameters())}, '
          f'UtilityHead={sum(p.numel() for p in utility_head.parameters())})')
    optimizer = torch.optim.Adam(trainable_params, lr=float(args.learning_rate))

    ckpt_dir = Path(cli.checkpoints) / 'onpolicy_prefix01' / args.data / f'seq{args.seq_len}_pred{args.pred_len}' / args.model_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = -1.0
    best_epoch = -1
    patience_left = int(args.patience)

    history = []
    wall_start = time.time()
    for epoch in range(int(args.train_epochs)):
        t0 = time.time()
        _, train_loader = exp._get_data(flag='train', shuffle=True)
        _, val_loader = exp._get_data(flag='val', shuffle=False)
        train_metrics = run_epoch_onpolicy(exp, args, model, set_conditioner, empty_token, utility_head, teacher,
                                            train_loader, 'train', True, channels, k, tau, cli.chunk_size,
                                            cli.lambda_rank, optimizer, device)
        val_metrics = run_epoch_onpolicy(exp, args, model, set_conditioner, empty_token, utility_head, teacher,
                                          val_loader, 'val', False, channels, k, tau, cli.chunk_size,
                                          cli.lambda_rank, optimizer, device)
        dt = time.time() - t0
        history.append({'epoch': epoch + 1, 'train': train_metrics, 'val': val_metrics, 'seconds': dt})
        od = val_metrics['onpolicy_diag']
        print(f"[onpolicy_prefix01] epoch {epoch+1} train_loss={train_metrics['loss']:.5f} "
              f"val_loss={val_metrics['loss']:.5f} val_overlap@{k}={val_metrics['overlap_at_k']:.4f} "
              f"sc_grad_norm={train_metrics['set_conditioner_grad_norm']:.5f} time={dt:.1f}s")
        print(f"[onpolicy_prefix01] epoch {epoch+1} onpolicy(val) "
              f"first_div_mean={od.get('first_divergence_step_mean', float('nan')):.3f} "
              f"prefix_overlap_t1={od.get('prefix_overlap_t1', float('nan')):.4f} "
              f"prefix_overlap_t10={od.get('prefix_overlap_t10', float('nan')):.4f} "
              f"regret_t1={od.get('per_step_regret_t1', float('nan')):.5f} "
              f"final_agg={od.get('final_free_running_aggregate', float('nan')):.5f}")
        if train_metrics['set_conditioner_grad_norm'] == 0.0:
            print('[onpolicy_prefix01][FAIL] SetConditioner gradient norm is exactly 0 -- STOP')
            break

        ckpt_payload = {
            'model_state_dict': model.state_dict(),
            'set_conditioner_state_dict': set_conditioner.state_dict(),
            'empty_token_state_dict': empty_token.state_dict(),
            'utility_head_state_dict': utility_head.state_dict(),
            'args': vars(args), 'epoch': epoch + 1, 'val_overlap_at_k': val_metrics['overlap_at_k'],
            'frozen_encoder_source_ckpt': cli.base_ckpt, 'loss_mode': 'hybrid_onpolicy', 'scorer_mode': 'cosine',
            'val_onpolicy_diag': od,
        }
        if cli.keep_all_checkpoints:
            torch.save(ckpt_payload, ckpt_dir / f'checkpoint_epoch{epoch+1}.pth')
        if val_metrics['overlap_at_k'] > best_val:
            best_val = val_metrics['overlap_at_k']
            best_epoch = epoch + 1
            patience_left = int(args.patience)
            torch.save(ckpt_payload, ckpt_dir / 'checkpoint.pth')
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f'[onpolicy_prefix01] early stop at epoch {epoch+1} (best={best_epoch})')
                break

    wall_time = time.time() - wall_start
    peak_mem = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
    summary = {'best_epoch': best_epoch, 'best_val_overlap_at_k': best_val,
               'wall_clock_seconds': wall_time, 'peak_gpu_memory_mib': peak_mem,
               'history': history, 'args': vars(args), 'checkpoint': str(ckpt_dir / 'checkpoint.pth'),
               'loss_mode': 'hybrid_onpolicy', 'scorer_mode': 'cosine', 'trainable_params': n_params}
    with open(ckpt_dir / 'summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f'[onpolicy_prefix01] done. best_epoch={best_epoch} best_val_overlap_at_k={best_val:.4f} '
          f'wall_clock={wall_time:.1f}s peak_gpu_mem={peak_mem:.0f}MiB')
    print(f'[onpolicy_prefix01] summary written to {ckpt_dir / "summary.json"}')


if __name__ == '__main__':
    main()
