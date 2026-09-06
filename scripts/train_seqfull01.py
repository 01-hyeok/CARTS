#!/usr/bin/env python3
"""EXP-SEQFULL01: Full-Memory Set-Conditioned Sequential Retrieval.

Trains a small SetConditioner + the (freshly initialised, same architecture
as B0) RelationEncoder to imitate the offline Full-Memory Weighted Set
Oracle sequence (scripts/precompute_seqfull01_teacher.py) via teacher-forced,
step-wise full-memory cross-entropy. No shortlist anywhere: every step scores
every valid candidate; only already-selected and invalid candidates are
masked.

Reuses (not reimplemented):
  - Exp_Stage1_Relation for data/memory-bank plumbing (same class B0 uses)
  - RelationEncoder + Model._relation_tensor for embeddings (same as B0)
  - the same full_online idea (one live, non-detached candidate encoder
    forward per optimisation step, reused across all K steps)
  - utils.oracle_intervention.select_greedy_weighted_set for the small-N
    gate's own teacher (the full-memory teacher is precomputed separately,
    see scripts/precompute_seqfull01_teacher.py)

New code (this file + models/SequentialSetRetriever.py): the SetConditioner
module and the sequential teacher-forced training/inference loop itself,
since no existing Stage-1 loss mode is single-shot-vs-sequential compatible.
"""
import argparse
import copy
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
from models.SequentialSetRetriever import EmptySetToken, SetConditioner, step_logits
from utils.oracle_intervention import select_greedy_weighted_set


def build_experiment(base_ckpt, overrides):
    """Same hyperparameters as B0 (encoder, d_model, optimizer, lr, epochs,
    patience, preprocessing, candidate mask, split, seed) -- loaded from B0's
    own saved args rather than re-listing hundreds of CLI defaults by hand, so
    "same as B0 except what this experiment changes" is enforced by
    construction, not by trusting a hand-written flag list to stay in sync.
    Only hyperparameters are taken from the checkpoint; the model itself is
    freshly initialised (this experiment's encoder is trained from scratch by
    the new sequential objective, exactly like every other Stage-1 arm in
    this campaign -- B0's weights are never loaded here).
    """
    ckpt = torch.load(base_ckpt, map_location='cpu')
    if 'args' not in ckpt:
        raise ValueError(f'checkpoint has no saved args: {base_ckpt}')
    args_dict = dict(ckpt['args'])
    args_dict.update(overrides)
    args = SimpleNamespace(**args_dict)
    return Exp_Stage1_Relation(args), args


def encode(model, x, c):
    z = model.encoder(model._relation_tensor(x, c, c))
    return F.normalize(z, dim=-1)


def run_sequence(q, E, cand_mask, set_conditioner, empty_token, k,
                  teacher_idx=None):
    """K steps of full-memory scoring. teacher_forcing when teacher_idx is
    given (training): the set-state at every step is built from the ORACLE
    prefix, per the spec ("hard argmax를 통해 backprop하려고 하지 마"). Free-
    running (inference / eval) when teacher_idx is None: the set-state is
    built from the model's own picks so far.

    Returns (logits_per_step: list of [B, N], picks: [B, K] the sequence
    actually advanced the mask with -- the oracle's own sequence when
    teacher-forced, the model's own picks otherwise).
    """
    bsz, device, dtype = q.size(0), q.device, q.dtype
    selected_mask = torch.zeros_like(cand_mask)
    picks = []
    logits_per_step = []
    for t in range(k):
        if t == 0:
            m = empty_token(bsz, device, dtype)
        elif teacher_idx is not None:
            m = E[teacher_idx[:, :t]].mean(dim=1)
        else:
            m = E[torch.stack(picks, dim=1)].mean(dim=1)
        h = set_conditioner(q, m)
        logits = step_logits(h, E, selected_mask, cand_mask)
        logits_per_step.append(logits)
        nxt = (teacher_idx[:, t:t + 1].clamp_min(0) if teacher_idx is not None
               else logits.argmax(dim=-1, keepdim=True))
        picks.append(nxt.squeeze(-1))
        selected_mask = selected_mask.scatter(1, nxt, True)
    return logits_per_step, torch.stack(picks, dim=1)


def step_losses(logits_per_step, teacher_idx, query_valid):
    """CE at every step against the next oracle candidate, query-masked."""
    losses = []
    accs = []
    for t, logits in enumerate(logits_per_step):
        target = teacher_idx[:, t].clamp_min(0)
        ce = F.cross_entropy(logits, target, reduction='none')
        if query_valid.any():
            losses.append(ce[query_valid].mean())
            pred = logits.argmax(dim=-1)
            accs.append((pred[query_valid] == target[query_valid]).float().mean().item())
        else:
            losses.append(logits.sum() * 0.0)
            accs.append(float('nan'))
    return losses, accs


def overlap_at_k(picks, teacher_idx, query_valid):
    """Unordered final-set overlap / K, matching the spec's Set Recall@10."""
    k = picks.size(1)
    overlaps = []
    for b in range(picks.size(0)):
        if not query_valid[b]:
            continue
        overlaps.append(len(set(picks[b].tolist()) & set(teacher_idx[b].tolist())) / k)
    return sum(overlaps) / max(len(overlaps), 1)


def small_n_teacher(model, exp, args, device):
    """The small-N gate's own teacher: select_greedy_weighted_set (identical
    function the full-memory teacher uses) over the tiny 256-candidate
    universe, scored by a FROZEN snapshot of the same (randomly initialised)
    encoder architecture -- fixed before training starts, never updated, so
    it is offline/non-circular even though it started from the same init as
    the trainable encoder (there is no pretrained checkpoint restricted to
    this synthetic tiny candidate set to score with instead).
    """
    frozen_encoder = copy.deepcopy(model.encoder).to(device).eval()
    c = int(args.target_channel)
    with torch.no_grad():
        cand_x = exp.memory_x.to(device)
        z_mem = F.normalize(frozen_encoder(model._relation_tensor(cand_x, c, c)), dim=-1)
    return frozen_encoder, z_mem, c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_ckpt', required=True,
                     help='B0 Stage-1 checkpoint to copy hyperparameters from (weights not loaded)')
    ap.add_argument('--teacher_cache', default='cache/seqfull01_teacher/ETTh1_pred96.pt')
    ap.add_argument('--checkpoints', default='checkpoints/exp_seqfull01')
    ap.add_argument('--model_id', default='carts_seqfull01_main')
    ap.add_argument('--des', default='seqfull01_main')
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--train_epochs', type=int, default=0, help='0 = use base_ckpt args value')
    ap.add_argument('--patience', type=int, default=0, help='0 = use base_ckpt args value')
    ap.add_argument('--small_n', action='store_true')
    ap.add_argument('--small_n_queries', type=int, default=16)
    ap.add_argument('--small_n_candidates', type=int, default=256)
    ap.add_argument('--small_n_steps', type=int, default=400)
    ap.add_argument('--small_n_target_channel', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--frozen_encoder', action='store_true',
                     help='EXP-SEQDIAG01 control arm: load --base_ckpt weights into the '
                          'encoder and freeze it (requires_grad=False, excluded from the '
                          'optimizer); only SetConditioner + EmptySetToken train.')
    cli = ap.parse_args()

    overrides = {
        'is_training': 1,
        'model_id': cli.model_id,
        'des': cli.des,
        'checkpoints': cli.checkpoints,
        'seed': cli.seed,
        'top_k': cli.top_k,
        # This pilot uses no residual/query/candidate conditioning and plain
        # cosine (no asymmetric metric) -- B0's own architecture, not FRR01's.
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
    if cli.small_n:
        overrides.update({
            'target_mode': 'single',
            'target_channel': cli.small_n_target_channel,
            'stage1_overfit_queries': cli.small_n_queries,
            'stage1_overfit_candidates': cli.small_n_candidates,
            'stage1_overfit_steps': cli.small_n_steps,
            'stage1_overfit_self_only': 1,
            'stage1_overfit_oracle_per_query': 20,
            # _FixedBatchSampler(query_count, steps) already repeats the tiny
            # batch `steps` times in one pass through train_loader; wrapping
            # that in further outer "epochs" would silently multiply total
            # optimizer steps by whatever B0's train_epochs happens to be.
            'train_epochs': 1,
            'patience': 1,
        })

    exp, args = build_experiment(cli.base_ckpt, overrides)
    torch.manual_seed(args.seed)
    device = exp.device
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    k = int(cli.top_k)

    if cli.frozen_encoder:
        # EXP-SEQDIAG01 control: same encoder weights B0 was evaluated with,
        # never updated -- isolates whether set-conditioning alone (on top of
        # a representation that cannot collapse) is learnable.
        b0_ckpt = torch.load(cli.base_ckpt, map_location='cpu')
        model.load_state_dict(b0_ckpt['model_state_dict'], strict=True)
        for p in model.encoder.parameters():
            p.requires_grad = False
        model.encoder.eval()
        print(f'[seqfull01] frozen_encoder=1: loaded and froze encoder weights from {cli.base_ckpt}')

    exp._ensure_memory()
    if cli.small_n:
        train_data, _ = exp._get_data(flag='train', shuffle=False)
        train_loader, eval_loader = exp._configure_tiny_overfit(train_data)
        frozen_encoder, z_mem_frozen, c = small_n_teacher(model, exp, args, device)
        channels = [c]
    else:
        channels = list(model.target_channels())
        for c in channels:
            sources = model.source_channels(c)
            if len(sources) != 1 or int(sources[0]) != int(c):
                raise ValueError(
                    f'EXP-SEQFULL01 is self-only; channel {c} has sources {sources}')
        teacher = torch.load(cli.teacher_cache, map_location='cpu')
        if teacher['meta']['top_k'] != k:
            raise ValueError(
                f"teacher cache top_k={teacher['meta']['top_k']} != --top_k={k}")

    d_model = int(args.d_model)
    set_conditioner = SetConditioner(d_model).to(device)
    empty_token = EmptySetToken(d_model).to(device)
    trainable_params = list(set_conditioner.parameters()) + list(empty_token.parameters())
    if not cli.frozen_encoder:
        trainable_params = list(model.encoder.parameters()) + trainable_params
    optimizer = torch.optim.Adam(trainable_params, lr=float(args.learning_rate))

    ckpt_dir = Path(cli.checkpoints) / 'seqfull01' / args.data / f'seq{args.seq_len}_pred{args.pred_len}' / args.model_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = -1.0
    best_epoch = -1
    patience_left = int(args.patience)

    def teacher_rows(batch_start_idx, channel, split):
        rows = [teacher['splits'][split]['start_to_row'][int(s)] for s in batch_start_idx.tolist()]
        idx = teacher['splits'][split]['teacher_idx'][channel][rows].to(device)
        return idx

    def run_epoch(loader, split, train, channel_list, teacher_split_name=None):
        if train:
            model.train()
            set_conditioner.train()
        else:
            model.eval()
            set_conditioner.eval()
        if cli.frozen_encoder:
            # Frozen regardless of the train()/eval() call above -- no
            # dropout/BN drift, no gradient, ever.
            model.encoder.eval()
        total_loss, n_batches = 0.0, 0
        step_acc_sum = [0.0] * k
        step_acc_n = [0] * k
        overlap_sum, overlap_n = 0.0, 0
        candidate_grad_norm = None
        encoder_grad_norm = None
        set_conditioner_grad_norm = 0.0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch_x, batch_y, batch_start_idx in loader:
                batch_x = batch_x.float().to(device)
                cand_mask, counts = exp._candidate_mask(batch_start_idx)
                if train:
                    optimizer.zero_grad()
                batch_loss = 0.0
                for c in channel_list:
                    E = encode(model, exp.memory_x, c)
                    if train and c == channel_list[-1] and E.requires_grad:
                        # Frozen-encoder arm: E carries no gradient path at
                        # all (nothing upstream is trainable), so there is
                        # nothing to retain_grad on -- candidate_grad_norm
                        # stays None for that arm by construction, which is
                        # the expected/correct reading, not a bug.
                        E.retain_grad()
                    q = encode(model, batch_x, c)
                    if cli.small_n:
                        with torch.no_grad():
                            t_q = F.normalize(frozen_encoder(model._relation_tensor(batch_x, c, c)), dim=-1)
                        # candidate futures/query future for this tiny universe
                        cand_y = exp.memory_y[:, :, c]
                        cand_last = exp.memory_x_last[:, c]
                        q_fut = batch_y[:, :, c].to(device) - batch_x[:, -1, c].unsqueeze(-1)
                        k_fut = cand_y - cand_last.unsqueeze(-1)
                        scores = torch.matmul(t_q, z_mem_frozen.transpose(0, 1))
                        futures = k_fut.unsqueeze(0).expand(batch_x.size(0), -1, -1)
                        teacher_idx = select_greedy_weighted_set(
                            futures, q_fut, scores, cand_mask, k, float(args.tau_topk))
                        query_valid = cand_mask.sum(-1) >= k
                    else:
                        teacher_idx = teacher_rows(batch_start_idx, c, teacher_split_name)
                        query_valid = teacher_idx[:, 0] != -1

                    logits_per_step, picks = run_sequence(
                        q, E, cand_mask, set_conditioner, empty_token, k,
                        teacher_idx=teacher_idx if train else None)
                    if not train:
                        # validation/eval: also compute teacher-forced step
                        # accuracy on this split as a diagnostic, separate
                        # from the free-running overlap used for selection.
                        tf_logits, _ = run_sequence(
                            q, E, cand_mask, set_conditioner, empty_token, k,
                            teacher_idx=teacher_idx)
                        losses, accs = step_losses(tf_logits, teacher_idx, query_valid)
                    else:
                        losses, accs = step_losses(logits_per_step, teacher_idx, query_valid)
                    for t in range(k):
                        if accs[t] == accs[t]:  # not NaN
                            step_acc_sum[t] += accs[t]
                            step_acc_n[t] += 1
                    overlap_sum += overlap_at_k(picks, teacher_idx, query_valid) * int(query_valid.sum())
                    overlap_n += int(query_valid.sum())
                    batch_loss = batch_loss + sum(losses) / k
                batch_loss = batch_loss / len(channel_list)
                if train:
                    batch_loss.backward()
                    if E.grad is not None:
                        candidate_grad_norm = E.grad.norm().item()
                    if cli.frozen_encoder:
                        # Must be exactly None (never even a zero tensor):
                        # requires_grad=False means autograd never assigns
                        # .grad at all, which is the actual assertion this
                        # control arm needs, not merely a zero norm.
                        assert all(p.grad is None for p in model.encoder.parameters()), (
                            'frozen_encoder=1 but an encoder parameter received a gradient')
                        encoder_grad_norm = None
                    else:
                        encoder_grad_norm = sum(
                            p.grad.norm().item() ** 2 for p in model.encoder.parameters()
                            if p.grad is not None) ** 0.5
                    sc_grads = [p.grad for p in set_conditioner.parameters() if p.grad is not None]
                    set_conditioner_grad_norm = (
                        sum(g.norm().item() ** 2 for g in sc_grads) ** 0.5 if sc_grads else 0.0
                    )
                    optimizer.step()
                total_loss += float(batch_loss.detach())
                n_batches += 1
        step_acc = [step_acc_sum[t] / max(step_acc_n[t], 1) for t in range(k)]
        overlap = overlap_sum / max(overlap_n, 1)
        return {
            'loss': total_loss / max(n_batches, 1),
            'step_acc': step_acc,
            'overlap_at_k': overlap,
            'candidate_grad_norm': candidate_grad_norm,
            'encoder_grad_norm': encoder_grad_norm,
            'set_conditioner_grad_norm': set_conditioner_grad_norm,
        }

    history = []
    wall_start = time.time()
    for epoch in range(int(args.train_epochs)):
        t0 = time.time()
        if cli.small_n:
            train_metrics = run_epoch(train_loader, 'train', True, channels)
            val_metrics = run_epoch(eval_loader, 'val', False, channels)
        else:
            _, train_loader = exp._get_data(flag='train', shuffle=True)
            _, val_loader = exp._get_data(flag='val', shuffle=False)
            train_metrics = run_epoch(train_loader, 'train', True, channels, teacher_split_name='train')
            val_metrics = run_epoch(val_loader, 'val', False, channels, teacher_split_name='val')
        dt = time.time() - t0
        row = {'epoch': epoch + 1, 'train': train_metrics, 'val': val_metrics, 'seconds': dt}
        history.append(row)
        enc_grad_str = (
            'None(frozen)' if train_metrics['encoder_grad_norm'] is None
            else f"{train_metrics['encoder_grad_norm']:.5f}"
        )
        print(f"[seqfull01] epoch {epoch+1} train_loss={train_metrics['loss']:.5f} "
              f"val_loss={val_metrics['loss']:.5f} val_overlap@{k}={val_metrics['overlap_at_k']:.4f} "
              f"cand_grad_norm={train_metrics['candidate_grad_norm']} "
              f"encoder_grad_norm={enc_grad_str} "
              f"set_conditioner_grad_norm={train_metrics['set_conditioner_grad_norm']:.5f} "
              f"time={dt:.1f}s")
        print(f"[seqfull01] epoch {epoch+1} train_step_acc="
              f"{[round(a, 4) for a in train_metrics['step_acc']]}")
        print(f"[seqfull01] epoch {epoch+1} val_step_acc="
              f"{[round(a, 4) for a in val_metrics['step_acc']]}")
        if (not cli.frozen_encoder and train_metrics['candidate_grad_norm'] is not None
                and train_metrics['candidate_grad_norm'] == 0.0):
            print('[seqfull01][FAIL] candidate-side gradient norm is exactly 0 -- STOP')
            break
        if train_metrics['set_conditioner_grad_norm'] == 0.0:
            print('[seqfull01][FAIL] SetConditioner gradient norm is exactly 0 -- STOP')
            break
        if val_metrics['overlap_at_k'] > best_val:
            best_val = val_metrics['overlap_at_k']
            best_epoch = epoch + 1
            patience_left = int(args.patience)
            torch.save({
                'model_state_dict': model.state_dict(),
                'set_conditioner_state_dict': set_conditioner.state_dict(),
                'empty_token_state_dict': empty_token.state_dict(),
                'args': vars(args),
                'epoch': epoch + 1,
                'val_overlap_at_k': best_val,
                'frozen_encoder': cli.frozen_encoder,
                'frozen_encoder_source_ckpt': cli.base_ckpt if cli.frozen_encoder else None,
            }, ckpt_dir / 'checkpoint.pth')
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f'[seqfull01] early stop at epoch {epoch+1} (best={best_epoch})')
                break

    wall_time = time.time() - wall_start
    peak_mem = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
    summary = {
        'best_epoch': best_epoch, 'best_val_overlap_at_k': best_val,
        'wall_clock_seconds': wall_time, 'peak_gpu_memory_mib': peak_mem,
        'history': history, 'args': vars(args), 'checkpoint': str(ckpt_dir / 'checkpoint.pth'),
        'frozen_encoder': cli.frozen_encoder,
    }
    with open(ckpt_dir / 'summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f'[seqfull01] done. best_epoch={best_epoch} best_val_overlap_at_k={best_val:.4f} '
          f'wall_clock={wall_time:.1f}s peak_gpu_mem={peak_mem:.0f}MiB')
    print(f'[seqfull01] summary written to {ckpt_dir / "summary.json"}')


if __name__ == '__main__':
    main()
