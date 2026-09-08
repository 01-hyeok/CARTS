#!/usr/bin/env python3
"""EXP-ORACLE-CHOICE01: does replacing R2's SmoothL1+pairwise surrogate with
a direct Oracle-Choice Cross-Entropy loss (predict the Set Oracle's actual
next pick, not its whole utility landscape) improve free-running set
selection and Stage-2?

D1 is IDENTICAL to C0 (EXP-TOPTAIL-RANK01/R2, reused verbatim, not
retrained) except for ONE change: the loss. Frozen B0 encoder (same
checkpoint), cosine UtilityHead, SetConditioner, EmptySetToken,
oracle-prefix teacher forcing, full-memory candidate support, optimizer/LR/
epochs/patience, Stage-2 protocol -- all unchanged, imported from
`scripts/train_margutil01.py`/`scripts/train_toptail_rank01.py`, not
reimplemented.

Loss (masked full-memory softmax cross-entropy against the Set Oracle's
actual next choice):

    i_t* = argmax_{i in valid} u_i^(t)      (u_i^(t) = -A_weighted(S*_{t-1}+{i}),
                                               the SAME dense target R0/R1/R2 use)
    p_i^(t) = softmax(s_i^(t) / tau) over i in valid ONLY (invalid positions
              get exactly zero probability mass, never included in the
              softmax denominator)
    L_choice^(t) = -log p_{i_t*}^(t)
    L_choice = mean_t L_choice^(t)

No SmoothL1, no pairwise term -- ONLY this loss, per the spec's explicit
prohibition on mixing auxiliary losses in this experiment. `tau` defaults
to the base checkpoint's own `tau_topk` (this project's existing softmax-
temperature convention for candidate weighting) -- no temperature sweep.

Encoder stays FROZEN throughout (same as C0/R0/R1/R2) -- no memory-safe
streaming machinery is needed here (that complexity in
`train_encoder_unfreeze01.py`/`train_encoder_anchor01.py` exists ONLY
because those experiments backprop into the encoder; this one does not),
so this script reuses `run_sequence_dense`'s existing frozen-encoder
teacher-forced computation unmodified, exactly like R0/R1/R2 already do.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.DenseUtilityRetriever import UtilityHead
from models.SequentialSetRetriever import EmptySetToken, SetConditioner
from scripts.train_margutil01 import build_experiment, encode, memory_value, run_sequence_dense
from scripts.train_toptail_rank01 import overlap_at_k
from utils.dense_utility import dense_utility


def oracle_choice_step_loss(u_hat, u_target, valid_mask, tau):
    """Masked full-memory softmax CE against the Set Oracle's true next
    pick `i_t* = argmax_valid(u_target)`. Returns (loss, diagnostics dict).
    Invalid positions get -inf logits -> exactly 0 softmax probability mass
    (never enter the log-sum-exp denominator, verified by sanity check O1/O2).
    """
    neg_inf = torch.finfo(u_target.dtype).min / 4
    target_masked = u_target.masked_fill(~valid_mask, neg_inf)
    i_star = target_masked.argmax(dim=-1)  # [B], detached (argmax has no grad path)

    logits = u_hat.masked_fill(~valid_mask, neg_inf) / float(tau)
    row_has_valid = valid_mask.any(dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(1, i_star.unsqueeze(-1)).squeeze(-1)  # [B]
    nll = nll[row_has_valid]
    loss = nll.mean() if nll.numel() > 0 else u_hat.sum() * 0.0

    with torch.no_grad():
        pred_rank = (logits > logits.gather(1, i_star.unsqueeze(-1))).sum(dim=-1).float()  # 0 = top-1
        top1 = (u_hat.masked_fill(~valid_mask, neg_inf).argmax(dim=-1) == i_star).float()
        top5 = (pred_rank < 5).float()
        top10 = (pred_rank < 10).float()
        valid_count = valid_mask.sum(dim=-1).clamp_min(2)
        top2_target = target_masked.topk(2, dim=-1).values
        margin_abs = (top2_target[:, 0] - top2_target[:, 1])
        scale = target_masked.masked_fill(~valid_mask, float('nan'))
        std_per_row = torch.nanmean(
            torch.tensor([float(torch.std(scale[b][valid_mask[b]])) for b in range(u_hat.size(0))
                          if bool(valid_mask[b].sum() > 1)], device=u_hat.device)
        ) if valid_mask.any() else torch.tensor(float('nan'))
        rows = row_has_valid & (valid_count >= 2)

    diag = {
        'top1_acc': float(top1[rows].mean()) if bool(rows.any()) else float('nan'),
        'top5_acc': float(top5[rows].mean()) if bool(rows.any()) else float('nan'),
        'top10_acc': float(top10[rows].mean()) if bool(rows.any()) else float('nan'),
        'pred_rank_mean': float(pred_rank[rows].mean()) if bool(rows.any()) else float('nan'),
        'top1_top2_margin_mean': float(margin_abs[rows].mean()) if bool(rows.any()) else float('nan'),
    }
    return loss, diag


def run_epoch_choice(exp, model, set_conditioner, empty_token, utility_head, teacher, loader,
                      split, train, channel_list, k, tau, chunk_size, optimizer, device):
    set_conditioner.train(train)
    empty_token.train(train)
    utility_head.train(train)
    total_loss, n_batches = 0.0, 0
    overlap_sum, overlap_n = 0.0, 0
    sc_grad_norm = 0.0
    diag_sums = {}
    diag_n = 0

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
                memory_c, offset_c = memory_value(exp.args_ns, batch_x, exp.memory_y, exp.memory_x_last, c)
                futures = memory_c + offset_c.view(-1, 1, 1)
                query_future = batch_y[:, :, c]

                teacher_idx = teacher_rows(batch_start_idx, c, split)
                query_valid = teacher_idx[:, 0] != -1

                u_hat_steps, u_target_steps, valid_steps, _ = run_sequence_dense(
                    q, E, cand_mask, set_conditioner, empty_token, utility_head,
                    futures, query_future, tau, k, chunk_size, teacher_idx=teacher_idx)

                step_losses_list = []
                for u_hat, u_target, valid in zip(u_hat_steps, u_target_steps, valid_steps):
                    u_hat_v = u_hat[query_valid]
                    u_target_v = u_target[query_valid]
                    valid_v = valid[query_valid]
                    if u_hat_v.size(0) == 0:
                        step_losses_list.append(u_hat.sum() * 0.0)
                        continue
                    loss_t, diag_t = oracle_choice_step_loss(u_hat_v, u_target_v, valid_v, tau)
                    step_losses_list.append(loss_t)
                    for kk, vv in diag_t.items():
                        if vv == vv:
                            diag_sums[kk] = diag_sums.get(kk, 0.0) + vv
                    diag_n += 1
                batch_loss = batch_loss + sum(step_losses_list) / k

                if not train:
                    with torch.no_grad():
                        _, _, _, picks = run_sequence_dense(
                            q, E, cand_mask, set_conditioner, empty_token, utility_head,
                            futures, query_future, tau, k, chunk_size, teacher_idx=None)
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
    return {'loss': total_loss / max(n_batches, 1), 'overlap_at_k': overlap,
            'set_conditioner_grad_norm': sc_grad_norm, 'diag': diag_means}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_ckpt', required=True)
    ap.add_argument('--teacher_cache', required=True)
    ap.add_argument('--checkpoints', default='checkpoints/exp_oracle_choice01')
    ap.add_argument('--model_id', default='carts_oracle_choice01_main')
    ap.add_argument('--des', default='oracle_choice01_main')
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--train_epochs', type=int, default=0)
    ap.add_argument('--patience', type=int, default=0)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tau', type=float, default=None, help='defaults to base_ckpt args.tau_topk, no sweep')
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
    exp.args_ns = args
    torch.manual_seed(args.seed)
    device = exp.device
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    k = int(cli.top_k)
    tau = float(cli.tau) if cli.tau is not None else float(args.tau_topk)

    b0_ckpt = torch.load(cli.base_ckpt, map_location='cpu')
    model.load_state_dict(b0_ckpt['model_state_dict'], strict=True)
    for p in model.encoder.parameters():
        p.requires_grad = False
    model.encoder.eval()
    print(f'[oracle_choice01] frozen encoder loaded from {cli.base_ckpt}, tau={tau} '
          f'(base_ckpt tau_topk={args.tau_topk})')

    exp._ensure_memory()
    channels = list(model.target_channels())
    for c in channels:
        sources = model.source_channels(c)
        if len(sources) != 1 or int(sources[0]) != int(c):
            raise ValueError(f'EXP-ORACLE-CHOICE01 is self-only; channel {c} has sources {sources}')
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
    print(f'[oracle_choice01] trainable_params={n_params} '
          f'(SetConditioner={sum(p.numel() for p in set_conditioner.parameters())}, '
          f'EmptySetToken={sum(p.numel() for p in empty_token.parameters())}, '
          f'UtilityHead={sum(p.numel() for p in utility_head.parameters())})')
    optimizer = torch.optim.Adam(trainable_params, lr=float(args.learning_rate))

    ckpt_dir = Path(cli.checkpoints) / 'oracle_choice01' / args.data / f'seq{args.seq_len}_pred{args.pred_len}' / args.model_id
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
        train_metrics = run_epoch_choice(exp, model, set_conditioner, empty_token, utility_head, teacher,
                                          train_loader, 'train', True, channels, k, tau, cli.chunk_size,
                                          optimizer, device)
        val_metrics = run_epoch_choice(exp, model, set_conditioner, empty_token, utility_head, teacher,
                                        val_loader, 'val', False, channels, k, tau, cli.chunk_size,
                                        optimizer, device)
        dt = time.time() - t0
        history.append({'epoch': epoch + 1, 'train': train_metrics, 'val': val_metrics, 'seconds': dt})
        vd = val_metrics['diag']
        print(f"[oracle_choice01] epoch {epoch+1} train_loss={train_metrics['loss']:.5f} "
              f"val_loss={val_metrics['loss']:.5f} val_overlap@{k}={val_metrics['overlap_at_k']:.4f} "
              f"val_top1_acc={vd.get('top1_acc', float('nan')):.4f} "
              f"val_top5_acc={vd.get('top5_acc', float('nan')):.4f} "
              f"val_pred_rank_mean={vd.get('pred_rank_mean', float('nan')):.2f} "
              f"sc_grad_norm={train_metrics['set_conditioner_grad_norm']:.5f} time={dt:.1f}s")
        if train_metrics['set_conditioner_grad_norm'] == 0.0:
            print('[oracle_choice01][FAIL] SetConditioner gradient norm is exactly 0 -- STOP')
            break

        ckpt_payload = {
            'model_state_dict': model.state_dict(),
            'set_conditioner_state_dict': set_conditioner.state_dict(),
            'empty_token_state_dict': empty_token.state_dict(),
            'utility_head_state_dict': utility_head.state_dict(),
            'args': vars(args), 'epoch': epoch + 1, 'val_overlap_at_k': val_metrics['overlap_at_k'],
            'frozen_encoder_source_ckpt': cli.base_ckpt, 'loss_mode': 'oracle_choice_ce', 'scorer_mode': 'cosine',
            'tau': tau, 'val_top1_acc': vd.get('top1_acc'), 'val_diag': vd,
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
                print(f'[oracle_choice01] early stop at epoch {epoch+1} (best={best_epoch})')
                break

    wall_time = time.time() - wall_start
    peak_mem = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
    summary = {'best_epoch': best_epoch, 'best_val_overlap_at_k': best_val,
               'wall_clock_seconds': wall_time, 'peak_gpu_memory_mib': peak_mem,
               'history': history, 'args': vars(args), 'checkpoint': str(ckpt_dir / 'checkpoint.pth'),
               'loss_mode': 'oracle_choice_ce', 'scorer_mode': 'cosine', 'trainable_params': n_params, 'tau': tau}
    with open(ckpt_dir / 'summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f'[oracle_choice01] done. best_epoch={best_epoch} best_val_overlap_at_k={best_val:.4f} '
          f'wall_clock={wall_time:.1f}s peak_gpu_mem={peak_mem:.0f}MiB')
    print(f'[oracle_choice01] summary written to {ckpt_dir / "summary.json"}')


if __name__ == '__main__':
    main()
