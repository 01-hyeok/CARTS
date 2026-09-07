#!/usr/bin/env python3
"""EXP-MARGUTIL01: Full-Memory Set-Conditioned Dense Marginal Utility.

Successor to EXP-SEQFULL01/EXP-SEQDIAG01 (closed by D-0011): replaces the
one-hot next-candidate-ID cross-entropy target with a DENSE per-candidate
set-utility regression target, `u_i^(t) = -A_weighted(S*_{t-1} + {i})`, for
every valid remaining candidate at every teacher-forced step (oracle prefix
`S*_{t-1}`, exactly the cached EXP-SEQFULL01 teacher sequence -- no new
teacher precomputation needed for H96; H720 caches are built once via
`scripts/precompute_seqfull01_teacher.py --top_k 10` before this runs).

Encoder is ALWAYS frozen (loaded from --base_ckpt, requires_grad=False) --
EXP-SEQDIAG01 established that a trainable encoder collapses under this
class of full-memory sequential objective and that freezing helps but is not
sufficient; this experiment isolates the objective change (one-hot -> dense
utility) from the encoder-collapse confound entirely, per the user's
explicit instruction not to re-open the trainable-encoder comparison here.

Reused, not reimplemented: `Exp_Stage1_Relation` (data/memory-bank plumbing),
`RelationEncoder`/`Model._relation_tensor` (frozen embeddings), the cached
`select_greedy_weighted_set` oracle sequence, `models.SequentialSetRetriever`'s
`SetConditioner`/`EmptySetToken`, `utils.dense_utility`'s closed-form
incremental-weighted-mean utility (the same math `select_greedy_weighted_set`
already uses internally, exposed densely instead of only at its argmin).
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp.exp_stage1_relation import Exp_Stage1_Relation
from models.DenseUtilityRetriever import UtilityHead
from models.SequentialSetRetriever import EmptySetToken, SetConditioner
from utils.dense_utility import candidate_weights, dense_utility, normalize_utility


def build_experiment(base_ckpt, overrides):
    ckpt = torch.load(base_ckpt, map_location='cpu')
    if 'args' not in ckpt:
        raise ValueError(f'checkpoint has no saved args: {base_ckpt}')
    args_dict = dict(ckpt['args'])
    args_dict.update(overrides)
    args = SimpleNamespace(**args_dict)
    return Exp_Stage1_Relation(args), args


def encode(model, x, c):
    with torch.no_grad():
        z = model.encoder(model._relation_tensor(x, c, c))
        return F.normalize(z, dim=-1)


def memory_value(args, batch_x, memory_y, memory_x_last, target_channel):
    """Candidate future values in the same space Stage-2's own
    `RelationStage2._memory_value` computes them in (that method lives on
    the Stage-2 model class only; this Stage-1 training script needs the
    identical convention, so it is replicated here rather than imported from
    a class this script never instantiates)."""
    memory_value_c = memory_y[:, :, target_channel].to(batch_x.device)
    query_offset = batch_x[:, -1, target_channel].detach()
    if getattr(args, 'relation_value_space', None) == 'delta_last':
        memory_value_c = memory_value_c - memory_x_last[:, target_channel].to(batch_x.device).unsqueeze(-1)
    return memory_value_c, query_offset


def run_sequence_dense(q, E, cand_mask, set_conditioner, empty_token, utility_head,
                        futures, query_future, tau, k, chunk_size,
                        teacher_idx=None):
    """K steps. teacher-forced (teacher_idx given): set-state and the dense
    utility TARGET's prefix both come from the oracle sequence; nothing here
    ever differentiates through an argmax. Free-running (teacher_idx=None):
    set-state and picks come from the model's own u_hat argmax over the
    still-valid remaining candidates, structurally masked so duplicates/
    invalid picks are impossible.

    Returns (u_hat_per_step: list[B,N], u_target_per_step: list[B,N] or None,
    valid_mask_per_step: list[B,N], picks: [B,K]).
    """
    bsz, device, dtype = q.size(0), q.device, q.dtype
    w = candidate_weights(torch.matmul(q, E.transpose(0, 1)), cand_mask, tau)
    selected_mask = torch.zeros_like(cand_mask)
    picks = []
    u_hat_steps, u_target_steps, valid_steps = [], [], []
    for t in range(k):
        if t == 0:
            m = empty_token(bsz, device, dtype)
        elif teacher_idx is not None:
            m = E[teacher_idx[:, :t]].mean(dim=1)
        else:
            m = E[torch.stack(picks, dim=1)].mean(dim=1)
        h = set_conditioner(q, m)
        u_hat = utility_head(h, E)
        valid_now = cand_mask & ~selected_mask
        u_hat_steps.append(u_hat)
        valid_steps.append(valid_now)

        if teacher_idx is not None:
            prefix = teacher_idx[:, :t].clamp_min(0)
            a_dense = dense_utility(prefix, w, futures, query_future, chunk_size=chunk_size)
            u_target_steps.append(-a_dense)
            nxt = teacher_idx[:, t:t + 1].clamp_min(0)
        else:
            u_target_steps.append(None)
            u_masked = u_hat.masked_fill(~valid_now, float('-inf'))
            nxt = u_masked.argmax(dim=-1, keepdim=True)
        picks.append(nxt.squeeze(-1))
        selected_mask = selected_mask.scatter(1, nxt, True)
    return u_hat_steps, u_target_steps, valid_steps, torch.stack(picks, dim=1)


def step_losses(u_hat_steps, u_target_steps, valid_steps, query_valid):
    losses, spearmans = [], []
    for u_hat, u_target, valid in zip(u_hat_steps, u_target_steps, valid_steps):
        u_norm = normalize_utility(u_target, valid)
        diff = F.smooth_l1_loss(u_hat, u_norm, reduction='none')
        vf = valid.float()
        per_query = (diff * vf).sum(dim=-1) / vf.sum(dim=-1).clamp_min(1.0)
        if query_valid.any():
            losses.append(per_query[query_valid].mean())
        else:
            losses.append(u_hat.sum() * 0.0)
    return losses


def overlap_at_k(picks, teacher_idx, query_valid):
    k = picks.size(1)
    overlaps = []
    for b in range(picks.size(0)):
        if not query_valid[b]:
            continue
        overlaps.append(len(set(picks[b].tolist()) & set(teacher_idx[b].tolist())) / k)
    return sum(overlaps) / max(len(overlaps), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_ckpt', required=True,
                     help='B0 Stage-1 checkpoint: source of hyperparameters AND the '
                          'frozen encoder weights (always loaded, always frozen here).')
    ap.add_argument('--teacher_cache', required=True)
    ap.add_argument('--checkpoints', default='checkpoints/exp_margutil01')
    ap.add_argument('--model_id', default='carts_margutil01_main')
    ap.add_argument('--des', default='margutil01_main')
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--train_epochs', type=int, default=0, help='0 = use base_ckpt args value')
    ap.add_argument('--patience', type=int, default=0, help='0 = use base_ckpt args value')
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--seed', type=int, default=0)
    cli = ap.parse_args()

    overrides = {
        'is_training': 1,
        'model_id': cli.model_id,
        'des': cli.des,
        'checkpoints': cli.checkpoints,
        'seed': cli.seed,
        'top_k': cli.top_k,
        'stage1_residual_teacher': 0,
        'stage1_query_base_conditioning': 0,
        'stage1_candidate_residual_conditioning': 0,
        'stage1_retrieval_metric': 'cosine',
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
    print(f'[margutil01] frozen encoder loaded from {cli.base_ckpt} (requires_grad=False)')

    exp._ensure_memory()
    channels = list(model.target_channels())
    for c in channels:
        sources = model.source_channels(c)
        if len(sources) != 1 or int(sources[0]) != int(c):
            raise ValueError(f'EXP-MARGUTIL01 is self-only; channel {c} has sources {sources}')
    teacher = torch.load(cli.teacher_cache, map_location='cpu')
    if teacher['meta']['top_k'] != k:
        raise ValueError(f"teacher cache top_k={teacher['meta']['top_k']} != --top_k={k}")
    if int(teacher['meta']['pred_len']) != int(args.pred_len):
        raise ValueError(f"teacher cache pred_len={teacher['meta']['pred_len']} != args.pred_len={args.pred_len}")

    d_model = int(args.d_model)
    set_conditioner = SetConditioner(d_model).to(device)
    empty_token = EmptySetToken(d_model).to(device)
    utility_head = UtilityHead().to(device)
    trainable_params = (list(set_conditioner.parameters()) + list(empty_token.parameters())
                         + list(utility_head.parameters()))
    optimizer = torch.optim.Adam(trainable_params, lr=float(args.learning_rate))

    ckpt_dir = Path(cli.checkpoints) / 'margutil01' / args.data / f'seq{args.seq_len}_pred{args.pred_len}' / args.model_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = -1.0
    best_epoch = -1
    patience_left = int(args.patience)

    def teacher_rows(batch_start_idx, channel, split):
        rows = [teacher['splits'][split]['start_to_row'][int(s)] for s in batch_start_idx.tolist()]
        idx = teacher['splits'][split]['teacher_idx'][channel][rows].to(device)
        return idx

    def run_epoch(loader, split, train, channel_list):
        set_conditioner.train(train)
        empty_token.train(train)
        utility_head.train(train)
        model.encoder.eval()  # always frozen, regardless of the train flag above
        total_loss, n_batches = 0.0, 0
        overlap_sum, overlap_n = 0.0, 0
        sc_grad_norm = 0.0
        enc_grad_none = True
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

                    # Loss always uses the oracle-prefix teacher-forced pass
                    # (this experiment's question is "is the utility function
                    # itself learnable", not free-running error accumulation).
                    u_hat_steps, u_target_steps, valid_steps, _ = run_sequence_dense(
                        q, E, cand_mask, set_conditioner, empty_token, utility_head,
                        futures, query_future, tau, k, cli.chunk_size,
                        teacher_idx=teacher_idx)
                    losses = step_losses(u_hat_steps, u_target_steps, valid_steps, query_valid)
                    batch_loss = batch_loss + sum(losses) / k

                    if not train:
                        # Free-running pass, val only: this is what
                        # best-epoch selection and early stopping are keyed
                        # on -- teacher-forced picks are trivially always the
                        # oracle sequence itself and would make overlap_at_k
                        # measure nothing.
                        with torch.no_grad():
                            _, _, _, picks = run_sequence_dense(
                                q, E, cand_mask, set_conditioner, empty_token, utility_head,
                                futures, query_future, tau, k, cli.chunk_size,
                                teacher_idx=None)
                        overlap_sum += overlap_at_k(picks, teacher_idx, query_valid) * int(query_valid.sum())
                        overlap_n += int(query_valid.sum())
                batch_loss = batch_loss / len(channel_list)
                if train:
                    batch_loss.backward()
                    assert all(p.grad is None for p in model.encoder.parameters()), (
                        'encoder must stay frozen -- an encoder parameter received a gradient')
                    enc_grad_none = True
                    sc_grads = [p.grad for p in set_conditioner.parameters() if p.grad is not None]
                    sc_grad_norm = sum(g.norm().item() ** 2 for g in sc_grads) ** 0.5 if sc_grads else 0.0
                    optimizer.step()
                total_loss += float(batch_loss.detach())
                n_batches += 1
        overlap = overlap_sum / max(overlap_n, 1)
        return {
            'loss': total_loss / max(n_batches, 1),
            'overlap_at_k': overlap,
            'encoder_grad_none': enc_grad_none,
            'set_conditioner_grad_norm': sc_grad_norm,
        }

    history = []
    wall_start = time.time()
    for epoch in range(int(args.train_epochs)):
        t0 = time.time()
        _, train_loader = exp._get_data(flag='train', shuffle=True)
        _, val_loader = exp._get_data(flag='val', shuffle=False)
        train_metrics = run_epoch(train_loader, 'train', True, channels)
        val_metrics = run_epoch(val_loader, 'val', False, channels)
        dt = time.time() - t0
        row = {'epoch': epoch + 1, 'train': train_metrics, 'val': val_metrics, 'seconds': dt}
        history.append(row)
        print(f"[margutil01] epoch {epoch+1} train_loss={train_metrics['loss']:.5f} "
              f"val_loss={val_metrics['loss']:.5f} val_overlap@{k}={val_metrics['overlap_at_k']:.4f} "
              f"encoder_grad_none={train_metrics['encoder_grad_none']} "
              f"set_conditioner_grad_norm={train_metrics['set_conditioner_grad_norm']:.5f} "
              f"time={dt:.1f}s")
        if train_metrics['set_conditioner_grad_norm'] == 0.0:
            print('[margutil01][FAIL] SetConditioner gradient norm is exactly 0 -- STOP')
            break
        if val_metrics['overlap_at_k'] > best_val:
            best_val = val_metrics['overlap_at_k']
            best_epoch = epoch + 1
            patience_left = int(args.patience)
            torch.save({
                'model_state_dict': model.state_dict(),
                'set_conditioner_state_dict': set_conditioner.state_dict(),
                'empty_token_state_dict': empty_token.state_dict(),
                'utility_head_state_dict': utility_head.state_dict(),
                'args': vars(args),
                'epoch': epoch + 1,
                'val_overlap_at_k': best_val,
                'frozen_encoder_source_ckpt': cli.base_ckpt,
            }, ckpt_dir / 'checkpoint.pth')
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f'[margutil01] early stop at epoch {epoch+1} (best={best_epoch})')
                break

    wall_time = time.time() - wall_start
    peak_mem = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
    summary = {
        'best_epoch': best_epoch, 'best_val_overlap_at_k': best_val,
        'wall_clock_seconds': wall_time, 'peak_gpu_memory_mib': peak_mem,
        'history': history, 'args': vars(args), 'checkpoint': str(ckpt_dir / 'checkpoint.pth'),
    }
    with open(ckpt_dir / 'summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f'[margutil01] done. best_epoch={best_epoch} best_val_overlap_at_k={best_val:.4f} '
          f'wall_clock={wall_time:.1f}s peak_gpu_mem={peak_mem:.0f}MiB')
    print(f'[margutil01] summary written to {ckpt_dir / "summary.json"}')


if __name__ == '__main__':
    main()
