#!/usr/bin/env python3
"""EXP-ORACLE-SCRATCH-TF01: same Individual-vs-Set-Oracle scratch-encoder
comparison as `EXP-ORACLE-SCRATCH01`, but with Set trained under TEACHER
FORCING (the training/masking prefix is always the ORACLE's own greedy
Set-Oracle sequence, never the model's own argmax pick) instead of
on-policy -- removing the one structural asymmetry between the Individual
arm (already Oracle-forced by construction, since its per-step target is
step-invariant) and the original Set arm (on-policy).

Completely separate from `EXP-ORACLE-SCRATCH01`: new script, new
checkpoint/result/log paths, that experiment's code/checkpoints/results are
never read, modified, or overwritten here.

Reused, UNCHANGED, from `scripts/train_oracle_scratch01.py`: `encode_raw`,
`base_score`, `run_sequence_individual` (Individual's own sequence
function -- already Oracle-forced, so this experiment's Individual arm is
IDENTICAL code to the original; no new Individual logic needed at all),
`encoder_collapse_snapshot`, `scorer_deviation_diagnostics`. `utils.dense_
utility.dense_utility`/`candidate_weights` (unmodified). `scripts.train_
margutil01.build_experiment`/`memory_value` (unmodified).

NEW in this file: `run_sequence_set_teacher_forced`, the ONLY genuinely
new piece of math -- mirrors `train_oracle_scratch01.run_sequence_set`'s
structure exactly (same t=1 conditioner-bypass, same base-score aggregate
weighting, same Choice-CE loss), but the prefix used to build BOTH the
SetConditioner's input state AND the next step's Oracle-target recomputation
is the ORACLE's own greedy sequence -- computed by taking `argmax` of the
(no-grad) Oracle target itself, never the model's own `u_hat`/`s_hat`.
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

from layers.retrieval_metric import oracle_rank_statistics
from scripts.train_margutil01 import build_experiment, memory_value
from scripts.train_oracle_choice01 import oracle_choice_step_loss
from scripts.train_oracle_scratch01 import (
    base_score, encoder_collapse_snapshot, encode_raw, run_sequence_individual,
    scorer_deviation_diagnostics,
)
from utils.dense_utility import candidate_weights, dense_utility


def _rank_fraction_for_step(u_hat, u_target, valid_now):
    """Reused pattern from `train_oracle_scratch01.py`'s own helper of the
    same name (not imported directly since that module's version is a
    private helper, not part of its public reuse surface -- duplicating
    ~6 lines here is cheaper and clearer than reaching into another
    script's internals)."""
    neg_inf = torch.finfo(u_target.dtype).min / 4
    target_masked = u_target.masked_fill(~valid_now, neg_inf)
    i_star = target_masked.argmax(dim=-1, keepdim=True)
    row_has_valid = valid_now.any(dim=-1)
    return oracle_rank_statistics(u_hat, i_star, valid_now, oracle_valid=row_has_valid)


def run_sequence_set_teacher_forced(z_q, E, cand_mask, set_conditioner, metric, w_base,
                                     futures, query_future, tau_choice, k, chunk_size):
    """Set Oracle, TEACHER FORCED: the training prefix at every step is the
    ORACLE's own greedy sequence (`oracle_picks`), never the model's own
    argmax over its own score `s_hat`. t=1 still bypasses the conditioner
    (`h_1 = z_q`) -- the Oracle's own t=1 pick is identical to Individual's
    t=1 pick by construction (both are `argmin_i MSE(Y_i,Y_q)`, and
    `dense_utility` with an empty prefix reduces to exactly that, verified
    as a unit test). Returns the SAME shapes as the on-policy
    `train_oracle_scratch01.run_sequence_set` (losses, diags, oracle_picks,
    a_dense_steps) so the training-loop caller code stays structurally
    identical -- only the sequence-construction function differs."""
    bsz, device = z_q.size(0), z_q.device
    oracle_picks = []
    losses, diags, a_dense_steps = [], [], []
    neg_inf = torch.finfo(query_future.dtype).min / 4
    for t in range(k):
        if t == 0:
            s_hat = base_score(z_q, E, metric)  # conditioner BYPASSED, same as on-policy t=1
        else:
            oracle_prefix = torch.stack(oracle_picks, dim=1)  # ORACLE prefix, NOT model pick
            m = E[oracle_prefix].mean(dim=1)
            h_t = set_conditioner(z_q, m)
            s_hat = base_score(h_t, E, metric)

        oracle_prefix_now = (torch.stack(oracle_picks, dim=1) if t > 0
                              else torch.zeros(bsz, 0, dtype=torch.long, device=device))
        already_picked_mask = torch.zeros_like(cand_mask)
        if t > 0:
            already_picked_mask = already_picked_mask.scatter(1, oracle_prefix_now, True)
        valid_now = cand_mask & ~already_picked_mask

        with torch.no_grad():
            a_dense = dense_utility(oracle_prefix_now, w_base, futures, query_future, chunk_size=chunk_size)
            u_target = -a_dense
            # ORACLE's own next pick: argmax of the (no-grad) target, NEVER
            # the model's own s_hat -- this is what makes this teacher-forced.
            oracle_next = u_target.masked_fill(~valid_now, neg_inf).argmax(dim=-1, keepdim=True)
        a_dense_steps.append(a_dense)

        loss_t, diag_t = oracle_choice_step_loss(s_hat, u_target, valid_now, tau_choice)
        losses.append(loss_t)
        diags.append(diag_t)

        oracle_picks.append(oracle_next.squeeze(-1))
    return losses, diags, torch.stack(oracle_picks, dim=1), a_dense_steps


def run_epoch(exp, args, model, set_conditioner, metric, target, loader, split, train,
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
                    losses, diags, picks, a_dense_steps = run_sequence_set_teacher_forced(
                        z_q, E, cand_mask, set_conditioner, metric, w_base,
                        futures, query_future, tau_choice, k, chunk_size)
                else:
                    b_i = base_score(z_q, E, metric)
                    d_i = ((futures - query_future.unsqueeze(1)) ** 2).mean(dim=-1)
                    u_target = -d_i
                    losses, diags, picks = run_sequence_individual(b_i, u_target, cand_mask, tau_choice, k)

                for t, loss_t in enumerate(losses):
                    for kk, vv in diags[t].items():
                        if vv == vv:
                            diag_sums[kk] = diag_sums.get(kk, 0.0) + vv
                    diag_n += 1

                # per-step rank-fraction (t=1 vs t>=2), walking the SAME
                # (Oracle-forced, for Set; Oracle-forced by construction,
                # for Individual) prefix `picks` used for training -- matches
                # `train_oracle_scratch01.py`'s own eval-time diagnostic
                # pattern exactly, just always Oracle-forced here.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True, help='EXISTING checkpoint, ARGS ONLY -- weights never loaded')
    ap.add_argument('--target', choices=['individual', 'set'], required=True)
    ap.add_argument('--scorer', choices=['cosine', 'asymmetric'], required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--checkpoints', default='checkpoints/exp_oracle_scratch_tf01')
    ap.add_argument('--model_id', required=True)
    ap.add_argument('--des', required=True)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--train_epochs', type=int, default=10)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tau_choice', type=float, default=None, help='defaults to reference args.tau_topk, no sweep')
    ap.add_argument('--shared_init_out', default=None, help='save this run\'s fresh encoder init here')
    ap.add_argument('--shared_init_in', default=None, help='load a PREVIOUSLY saved encoder init from here')
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
        print(f'[oracle_scratch_tf01] loaded SHARED (not pretrained) encoder init from {cli.shared_init_in}')
    if cli.shared_init_out:
        Path(cli.shared_init_out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.encoder.state_dict(), cli.shared_init_out)
        print(f'[oracle_scratch_tf01] saved this run\'s fresh encoder init to {cli.shared_init_out}')

    for p in model.encoder.parameters():
        assert p.requires_grad, 'encoder must be TRAINABLE (scratch)'

    exp._ensure_memory()
    channels = list(model.target_channels())
    for c in channels:
        sources = model.source_channels(c)
        if len(sources) != 1 or int(sources[0]) != int(c):
            raise ValueError(f'EXP-ORACLE-SCRATCH-TF01 is self-only; channel {c} has sources {sources}')

    d_model = int(args.d_model)
    from layers.retrieval_metric import RetrievalMetric, cosine_init_deviation
    from models.SequentialSetRetriever import SetConditioner
    metric = None
    trainable_params = list(model.encoder.parameters())
    set_conditioner = None
    if cli.scorer == 'asymmetric':
        metric = RetrievalMetric(kind='asymmetric', dim=d_model, output='cosine', layer_norm=False).to(device)
        dev = cosine_init_deviation(metric)
        print(f'[oracle_scratch_tf01] asymmetric scorer cosine_init_deviation={dev:.3e}')
        assert dev < 1e-5, 'asymmetric scorer does NOT match cosine at identity init'
        trainable_params += list(metric.parameters())
    if cli.target == 'set':
        set_conditioner = SetConditioner(d_model).to(device)
        trainable_params += list(set_conditioner.parameters())

    n_params = sum(p.numel() for p in trainable_params)
    print(f'[oracle_scratch_tf01] target={cli.target} scorer={cli.scorer} pred_len={cli.pred_len} '
          f'trainable_params={n_params} (TEACHER FORCED)')
    optimizer = torch.optim.Adam(trainable_params, lr=float(args.learning_rate))

    ckpt_dir = Path(cli.checkpoints) / args.data / f'seq{args.seq_len}_pred{args.pred_len}' / args.model_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = float('inf')
    best_epoch = -1
    patience_left = int(args.patience)
    history = []
    wall_start = time.time()

    for epoch in range(int(args.train_epochs)):
        t0 = time.time()
        _, train_loader = exp._get_data(flag='train', shuffle=True)
        _, val_loader = exp._get_data(flag='val', shuffle=False)
        train_metrics = run_epoch(exp, args, model, set_conditioner, metric, cli.target, train_loader,
                                   'train', True, channels, k, tau_topk, tau_choice, cli.chunk_size, optimizer, device)
        val_metrics = run_epoch(exp, args, model, set_conditioner, metric, cli.target, val_loader,
                                 'val', False, channels, k, tau_topk, tau_choice, cli.chunk_size, optimizer, device)
        dt = time.time() - t0
        history.append({'epoch': epoch + 1, 'train': train_metrics, 'val': val_metrics, 'seconds': dt})
        vd = val_metrics['diag']
        print(f"[oracle_scratch_tf01] epoch {epoch+1} train_loss={train_metrics['loss']:.5f} "
              f"val_loss={val_metrics['loss']:.5f} val_rank_frac={val_metrics['rank_fraction_overall']:.5f} "
              f"val_rank_frac_t1={val_metrics['rank_fraction_t1']:.5f} "
              f"val_rank_frac_tgeq2={val_metrics['rank_fraction_tgeq2']:.5f} "
              f"val_top1_acc={vd.get('top1_acc', float('nan')):.4f} time={dt:.1f}s")

        ckpt_payload = {
            'model_state_dict': model.state_dict(),
            'set_conditioner_state_dict': set_conditioner.state_dict() if set_conditioner is not None else None,
            'metric_state_dict': metric.state_dict() if metric is not None else None,
            'args': vars(args), 'epoch': epoch + 1, 'val_rank_fraction_overall': val_metrics['rank_fraction_overall'],
            'target': cli.target, 'scorer': cli.scorer, 'tau_choice': tau_choice, 'val_diag': vd,
            'teacher_forced': True,
        }
        torch.save(ckpt_payload, ckpt_dir / f'checkpoint_epoch{epoch+1}.pth')
        if val_metrics['rank_fraction_overall'] < best_val:
            best_val = val_metrics['rank_fraction_overall']
            best_epoch = epoch + 1
            patience_left = int(args.patience)
            torch.save(ckpt_payload, ckpt_dir / 'checkpoint.pth')
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f'[oracle_scratch_tf01] early stop at epoch {epoch+1} (best={best_epoch})')
                break

    wall_time = time.time() - wall_start
    peak_mem = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
    collapse = encoder_collapse_snapshot(model, exp.memory_x, channels)
    scorer_diag = scorer_deviation_diagnostics(metric)
    summary = {'best_epoch': best_epoch, 'best_val_rank_fraction': best_val, 'wall_clock_seconds': wall_time,
               'peak_gpu_memory_mib': peak_mem, 'history': history, 'trainable_params': n_params,
               'tau_choice': tau_choice, 'target': cli.target, 'scorer': cli.scorer, 'pred_len': cli.pred_len,
               'checkpoint': str(ckpt_dir / 'checkpoint.pth'), 'encoder_collapse_final': collapse,
               'scorer_diagnostics_final': scorer_diag, 'teacher_forced': True}
    with open(ckpt_dir / 'summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f'[oracle_scratch_tf01] done. best_epoch={best_epoch} best_val_rank_fraction={best_val:.5f} '
          f'wall_clock={wall_time:.1f}s peak_gpu_mem={peak_mem:.0f}MiB')
    print(f'[oracle_scratch_tf01] summary written to {ckpt_dir / "summary.json"}')


if __name__ == '__main__':
    main()
