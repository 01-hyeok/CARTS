#!/usr/bin/env python3
"""EXP-ORACLE-RANK-GAIN01: unified 8-arm Stage-1 trainer (Individual/Set
Oracle x Teacher-Forcing/On-policy prefix x Cosine/Asymmetric scorer),
scratch encoder, across ETTh1 and Weather, H96 and H720. Separate from,
and does not read/modify/overwrite, `EXP-ORACLE-SCRATCH01` or
`EXP-ORACLE-SCRATCH-TF01`'s own code/checkpoints/results -- this
experiment uses its OWN freshly-generated shared encoder init per
dataset/horizon (all 8 arms in a cell start from the SAME one), per the
user's explicit choice to retrain everything rather than mix inits across
experiments.

Sequence-construction functions, reused (imported, not reimplemented)
except where noted:
  - individual + teacher-forcing: `train_oracle_scratch01.run_sequence_
    individual` (already Oracle-forced by construction -- no change needed)
  - individual + on-policy: `run_sequence_individual_onpolicy` (NEW, this
    file -- the ONLY genuinely new sequence function this experiment adds;
    identical to the TF version except the mask/prefix trajectory advances
    along the MODEL's own argmax over `u_hat`, not the Oracle's `u_target`)
  - set + teacher-forcing: `train_oracle_scratch_tf01.run_sequence_set_
    teacher_forced` (imported unmodified)
  - set + on-policy: `train_oracle_scratch01.run_sequence_set` (imported
    unmodified)

Epoch 0: a full checkpoint (identical payload shape to every later epoch's)
is saved IMMEDIATELY after the model is built/shared-init-loaded, BEFORE
`optimizer.step()` is ever called for the first time -- proven by the
call-order in `main()` below (build -> save epoch0 -> THEN enter the
training loop). The eval script (`eval_oracle_rank_gain01.py`) evaluates
`checkpoint_epoch0.pth` with the EXACT SAME code path as `checkpoint.pth`
(the best epoch), so Epoch0-vs-Best is a same-protocol comparison.
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

from layers.retrieval_metric import RetrievalMetric, cosine_init_deviation, oracle_rank_statistics
from models.SequentialSetRetriever import SetConditioner
from scripts.train_margutil01 import build_experiment, memory_value
from scripts.train_oracle_choice01 import oracle_choice_step_loss
from scripts.train_oracle_scratch01 import (
    base_score, encoder_collapse_snapshot, encode_raw, run_sequence_individual,
    run_sequence_set, scorer_deviation_diagnostics,
)
from scripts.train_oracle_scratch_tf01 import run_sequence_set_teacher_forced
from utils.dense_utility import candidate_weights, dense_utility


def run_sequence_individual_onpolicy(u_hat, u_target, cand_mask, tau_choice, k):
    """Individual Oracle, ON-POLICY: `u_hat`/`u_target` are step-invariant
    (identical to the TF version), but the mask/prefix trajectory advances
    along the MODEL's OWN argmax over `u_hat` -- never the Oracle's
    `u_target` -- so a wrong early model pick genuinely changes which
    candidates remain valid at later steps, unlike the TF version where
    the trajectory is fixed regardless of what the model predicts."""
    selected_mask = torch.zeros_like(cand_mask)
    losses, diags, picks = [], [], []
    neg_inf = torch.finfo(u_hat.dtype).min / 4
    for _ in range(k):
        valid_now = cand_mask & ~selected_mask
        loss_t, diag_t = oracle_choice_step_loss(u_hat, u_target, valid_now, tau_choice)
        losses.append(loss_t)
        diags.append(diag_t)
        u_masked = u_hat.masked_fill(~valid_now, neg_inf)
        nxt = u_masked.argmax(dim=-1, keepdim=True).detach()  # model's OWN pick, on-policy
        picks.append(nxt.squeeze(-1).detach())
        selected_mask = selected_mask.scatter(1, nxt, True)
    return losses, diags, torch.stack(picks, dim=1)


def _rank_fraction_for_step(u_hat, u_target, valid_now):
    neg_inf = torch.finfo(u_target.dtype).min / 4
    target_masked = u_target.masked_fill(~valid_now, neg_inf)
    i_star = target_masked.argmax(dim=-1, keepdim=True)
    row_has_valid = valid_now.any(dim=-1)
    return oracle_rank_statistics(u_hat, i_star, valid_now, oracle_valid=row_has_valid)


def run_epoch(exp, args, model, set_conditioner, metric, target, prefix_policy, loader, split, train,
              channel_list, k, tau_topk, tau_choice, chunk_size, optimizer, device):
    is_set = target == 'set'
    model.train(train)
    if set_conditioner is not None:
        set_conditioner.train(train)
    if metric is not None:
        metric.train(train)

    total_loss, n_batches = 0.0, 0
    diag_sums, diag_n = {}, 0
    rank_frac_sums = {'t1': 0.0, 'tgeq2': 0.0}
    rank_frac_n = {'t1': 0.0, 'tgeq2': 0.0}

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
                E = encode_raw(model, exp.memory_x, c)
                z_q = encode_raw(model, batch_x, c)
                memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
                futures = memory_c + offset_c.view(-1, 1, 1)
                query_future = batch_y[:, :, c]

                if is_set:
                    with torch.no_grad():
                        b0 = base_score(z_q, E, metric)
                        w_base = candidate_weights(b0, cand_mask, tau_topk)
                    if prefix_policy == 'tf':
                        losses, diags, picks, a_dense_steps = run_sequence_set_teacher_forced(
                            z_q, E, cand_mask, set_conditioner, metric, w_base,
                            futures, query_future, tau_choice, k, chunk_size)
                    else:
                        losses, diags, picks, a_dense_steps = run_sequence_set(
                            z_q, E, cand_mask, set_conditioner, metric, w_base,
                            futures, query_future, tau_choice, k, chunk_size)
                else:
                    b_i = base_score(z_q, E, metric)
                    d_i = ((futures - query_future.unsqueeze(1)) ** 2).mean(dim=-1)
                    u_target = -d_i
                    if prefix_policy == 'tf':
                        losses, diags, picks = run_sequence_individual(b_i, u_target, cand_mask, tau_choice, k)
                    else:
                        losses, diags, picks = run_sequence_individual_onpolicy(b_i, u_target, cand_mask, tau_choice, k)

                for t, loss_t in enumerate(losses):
                    for kk, vv in diags[t].items():
                        if vv == vv:
                            diag_sums[kk] = diag_sums.get(kk, 0.0) + vv
                    diag_n += 1

                if not train:
                    selected_mask = torch.zeros_like(cand_mask)
                    for t in range(k):
                        valid_now = cand_mask & ~selected_mask
                        if is_set:
                            if t == 0:
                                s_hat_t = base_score(z_q, E, metric)
                            else:
                                prefix = picks[:, :t]
                                m = E[prefix].mean(dim=1)
                                h_t = set_conditioner(z_q, m)
                                s_hat_t = base_score(h_t, E, metric)
                            prefix_now = picks[:, :t]
                            a_dense = dense_utility(prefix_now, w_base, futures, query_future, chunk_size=chunk_size)
                            u_target_t = -a_dense
                        else:
                            s_hat_t = b_i
                            u_target_t = u_target
                        stats = _rank_fraction_for_step(s_hat_t, u_target_t, valid_now)
                        frac = float(stats['oracle_top10_rank_fraction'])
                        key = 't1' if t == 0 else 'tgeq2'
                        if frac == frac:
                            rank_frac_sums[key] += frac
                            rank_frac_n[key] += 1
                        nxt = picks[:, t:t + 1]
                        selected_mask = selected_mask.scatter(1, nxt, True)

                batch_loss = batch_loss + sum(losses) / k

            batch_loss = batch_loss / len(channel_list)
            if train:
                batch_loss.backward()
                optimizer.step()
            total_loss += float(batch_loss.detach())
            n_batches += 1

    diag_means = {kk: vv / max(diag_n, 1) for kk, vv in diag_sums.items()}
    rank_frac_means = {
        kk: (rank_frac_sums[kk] / rank_frac_n[kk] if rank_frac_n[kk] > 0 else float('nan'))
        for kk in rank_frac_sums
    }
    overall_rank_frac = (
        (rank_frac_sums['t1'] + rank_frac_sums['tgeq2']) / max(rank_frac_n['t1'] + rank_frac_n['tgeq2'], 1)
    )
    return {'loss': total_loss / max(n_batches, 1), 'diag': diag_means,
            'rank_fraction_t1': rank_frac_means['t1'], 'rank_fraction_tgeq2': rank_frac_means['tgeq2'],
            'rank_fraction_overall': overall_rank_frac}


def build_ckpt_payload(model, set_conditioner, metric, args, epoch, val_rank_fraction_overall,
                        target, prefix_policy, scorer, tau_choice, val_diag):
    return {
        'model_state_dict': model.state_dict(),
        'set_conditioner_state_dict': set_conditioner.state_dict() if set_conditioner is not None else None,
        'metric_state_dict': metric.state_dict() if metric is not None else None,
        'args': vars(args), 'epoch': epoch, 'val_rank_fraction_overall': val_rank_fraction_overall,
        'target': target, 'prefix_policy': prefix_policy, 'scorer': scorer, 'tau_choice': tau_choice,
        'val_diag': val_diag,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True, help='EXISTING checkpoint, ARGS ONLY -- weights never loaded')
    ap.add_argument('--target', choices=['individual', 'set'], required=True)
    ap.add_argument('--prefix_policy', choices=['tf', 'onpolicy'], required=True)
    ap.add_argument('--scorer', choices=['cosine', 'asymmetric'], required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--checkpoints', default='checkpoints/exp_oracle_rank_gain01')
    ap.add_argument('--model_id', required=True)
    ap.add_argument('--des', required=True)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--train_epochs', type=int, default=10)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tau_choice', type=float, default=None, help='defaults to reference args.tau_topk, no sweep')
    ap.add_argument('--shared_init_out', default=None)
    ap.add_argument('--shared_init_in', default=None)
    cli = ap.parse_args()

    overrides = {
        'is_training': 1, 'model_id': cli.model_id, 'des': cli.des,
        'checkpoints': cli.checkpoints, 'seed': cli.seed, 'top_k': cli.top_k,
        'pred_len': cli.pred_len, 'seq_len': cli.pred_len,
        'stage1_retrieval_metric': 'cosine',
        'stage1_residual_teacher': 0, 'stage1_query_base_conditioning': 0,
        'stage1_candidate_residual_conditioning': 0,
        'stage1_full_memory_gradient_mode': 'full_online',
        'train_epochs': cli.train_epochs, 'patience': cli.patience,
    }
    exp, args = build_experiment(cli.reference_ckpt, overrides)
    torch.manual_seed(args.seed)
    device = exp.device
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    k = int(cli.top_k)
    tau_topk = float(args.tau_topk)
    tau_choice = float(cli.tau_choice) if cli.tau_choice is not None else tau_topk

    ref_ckpt = torch.load(cli.reference_ckpt, map_location='cpu')
    ref_encoder_keys = {k_: v for k_, v in ref_ckpt.get('model_state_dict', {}).items() if k_.startswith('encoder.')}
    cur_encoder_keys = {k_: v.detach().cpu() for k_, v in model.state_dict().items() if k_.startswith('encoder.')}
    if ref_encoder_keys and cur_encoder_keys:
        same = all(
            k_ in cur_encoder_keys and torch.allclose(ref_encoder_keys[k_], cur_encoder_keys[k_])
            for k_ in ref_encoder_keys
        )
        assert not same, 'model.encoder matches the reference checkpoint EXACTLY -- scratch-init sanity failed'

    if cli.shared_init_in:
        shared_init = torch.load(cli.shared_init_in, map_location='cpu')
        model.encoder.load_state_dict(shared_init)
        print(f'[oracle_rank_gain01] loaded SHARED (not pretrained) encoder init from {cli.shared_init_in}')
    if cli.shared_init_out:
        Path(cli.shared_init_out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.encoder.state_dict(), cli.shared_init_out)
        print(f'[oracle_rank_gain01] saved this run\'s fresh encoder init to {cli.shared_init_out}')

    for p in model.encoder.parameters():
        assert p.requires_grad, 'encoder must be TRAINABLE (scratch)'

    exp._ensure_memory()
    channels = list(model.target_channels())
    for c in channels:
        sources = model.source_channels(c)
        if len(sources) != 1 or int(sources[0]) != int(c):
            raise ValueError(f'EXP-ORACLE-RANK-GAIN01 is self-only; channel {c} has sources {sources}')

    d_model = int(args.d_model)
    metric = None
    trainable_params = list(model.encoder.parameters())
    set_conditioner = None
    if cli.scorer == 'asymmetric':
        metric = RetrievalMetric(kind='asymmetric', dim=d_model, output='cosine', layer_norm=False).to(device)
        dev = cosine_init_deviation(metric)
        print(f'[oracle_rank_gain01] asymmetric scorer cosine_init_deviation={dev:.3e}')
        assert dev < 1e-5, 'asymmetric scorer does NOT match cosine at identity init'
        trainable_params += list(metric.parameters())
    if cli.target == 'set':
        set_conditioner = SetConditioner(d_model).to(device)
        trainable_params += list(set_conditioner.parameters())

    n_params = sum(p.numel() for p in trainable_params)
    print(f'[oracle_rank_gain01] target={cli.target} prefix={cli.prefix_policy} scorer={cli.scorer} '
          f'pred_len={cli.pred_len} trainable_params={n_params}')
    optimizer = torch.optim.Adam(trainable_params, lr=float(args.learning_rate))

    ckpt_dir = Path(cli.checkpoints) / args.data / f'seq{args.seq_len}_pred{args.pred_len}' / args.model_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- EPOCH 0: saved BEFORE optimizer.step() is ever called. ----
    epoch0_payload = build_ckpt_payload(model, set_conditioner, metric, args, 0, None,
                                         cli.target, cli.prefix_policy, cli.scorer, tau_choice, {})
    torch.save(epoch0_payload, ckpt_dir / 'checkpoint_epoch0.pth')
    print(f'[oracle_rank_gain01] EPOCH 0 checkpoint saved (pre-training, scratch/shared init) to '
          f'{ckpt_dir / "checkpoint_epoch0.pth"}')

    best_val = float('inf')
    best_epoch = -1
    patience_left = int(args.patience)
    history = []
    wall_start = time.time()

    for epoch in range(int(args.train_epochs)):
        t0 = time.time()
        _, train_loader = exp._get_data(flag='train', shuffle=True)
        _, val_loader = exp._get_data(flag='val', shuffle=False)
        train_metrics = run_epoch(exp, args, model, set_conditioner, metric, cli.target, cli.prefix_policy,
                                   train_loader, 'train', True, channels, k, tau_topk, tau_choice, cli.chunk_size,
                                   optimizer, device)
        val_metrics = run_epoch(exp, args, model, set_conditioner, metric, cli.target, cli.prefix_policy,
                                 val_loader, 'val', False, channels, k, tau_topk, tau_choice, cli.chunk_size,
                                 optimizer, device)
        dt = time.time() - t0
        history.append({'epoch': epoch + 1, 'train': train_metrics, 'val': val_metrics, 'seconds': dt})
        vd = val_metrics['diag']
        print(f"[oracle_rank_gain01] epoch {epoch+1} train_loss={train_metrics['loss']:.5f} "
              f"val_loss={val_metrics['loss']:.5f} val_rank_frac={val_metrics['rank_fraction_overall']:.5f} "
              f"val_rank_frac_t1={val_metrics['rank_fraction_t1']:.5f} "
              f"val_rank_frac_tgeq2={val_metrics['rank_fraction_tgeq2']:.5f} "
              f"val_top1_acc={vd.get('top1_acc', float('nan')):.4f} time={dt:.1f}s")

        ckpt_payload = build_ckpt_payload(model, set_conditioner, metric, args, epoch + 1,
                                           val_metrics['rank_fraction_overall'], cli.target, cli.prefix_policy,
                                           cli.scorer, tau_choice, vd)
        torch.save(ckpt_payload, ckpt_dir / f'checkpoint_epoch{epoch+1}.pth')
        if val_metrics['rank_fraction_overall'] < best_val:
            best_val = val_metrics['rank_fraction_overall']
            best_epoch = epoch + 1
            patience_left = int(args.patience)
            torch.save(ckpt_payload, ckpt_dir / 'checkpoint.pth')
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f'[oracle_rank_gain01] early stop at epoch {epoch+1} (best={best_epoch})')
                break

    wall_time = time.time() - wall_start
    peak_mem = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
    collapse = encoder_collapse_snapshot(model, exp.memory_x, channels)
    scorer_diag = scorer_deviation_diagnostics(metric)
    summary = {'best_epoch': best_epoch, 'best_val_rank_fraction': best_val, 'wall_clock_seconds': wall_time,
               'peak_gpu_memory_mib': peak_mem, 'history': history, 'trainable_params': n_params,
               'tau_choice': tau_choice, 'target': cli.target, 'prefix_policy': cli.prefix_policy,
               'scorer': cli.scorer, 'pred_len': cli.pred_len, 'checkpoint': str(ckpt_dir / 'checkpoint.pth'),
               'checkpoint_epoch0': str(ckpt_dir / 'checkpoint_epoch0.pth'),
               'encoder_collapse_final': collapse, 'scorer_diagnostics_final': scorer_diag}
    with open(ckpt_dir / 'summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f'[oracle_rank_gain01] done. best_epoch={best_epoch} best_val_rank_fraction={best_val:.5f} '
          f'wall_clock={wall_time:.1f}s peak_gpu_mem={peak_mem:.0f}MiB')
    print(f'[oracle_rank_gain01] summary written to {ckpt_dir / "summary.json"}')


if __name__ == '__main__':
    main()
