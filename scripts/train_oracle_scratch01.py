#!/usr/bin/env python3
"""EXP-ORACLE-SCRATCH01 -- Stage-1: Individual Oracle vs Set Oracle, Cosine vs
Asymmetric scorer, ALL trained from a SCRATCH (randomly initialised, never
pretrained/frozen/anchored) encoder. Completely separate from Track A's
`EXP-ONPOLICY-CHOICE01` (OPC1) -- that experiment's frozen-encoder checkpoints
and results are NEVER loaded or reused here in any form.

Four arms per horizon (`--target {individual,set}` x `--scorer {cosine,
asymmetric}`), sharing one random encoder initialisation per horizon (see
`--shared_init_out`/`--shared_init_in`) so the only difference between arms is
the training objective/scorer, never the starting point.

Design (mirrors the user's own spec, section by section):

- Encoder: `models.RelationStage1.Model`'s own `RelationEncoder`
  (`stage1_encoder_type=mlp`, same architecture/config as every other ETTh1
  Stage-1 run in this project), built via `Exp_Stage1_Relation(args)` --
  which never auto-loads any checkpoint weights (`_build_model` only calls
  `Model(args).float()`) -- so this IS scratch by construction, not merely by
  omission. `build_experiment` (from `scripts/train_margutil01.py`) is reused
  ONLY to reconstruct a full, valid `args` namespace from an EXISTING
  checkpoint's saved config (architecture/hyperparameters only); its WEIGHTS
  are never loaded onto anything trained here (verified by a sanity test).

- Scorer: cosine (`b_i = cos(z_q, z_i)`, no learnable parameters) or
  `layers.retrieval_metric.RetrievalMetric(kind='asymmetric', output='cosine',
  layer_norm=False)` (reused unmodified, identity-initialised `W_q`/`W_k`,
  trainable). No new scorer implementation.

- SetConditioner (`models.SequentialSetRetriever.SetConditioner`, reused
  unmodified) is used ONLY for the Set arm's t>=2 steps; t=1 always scores
  directly with the base encoder embedding (`h_1 = z_q`), matching the spec's
  explicit requirement that the FIRST pick is the encoder/scorer's own doing,
  never the conditioner's. `EmptySetToken` is deliberately NOT used anywhere
  in this experiment.

- Individual Oracle: sequential Choice CE (`oracle_choice_step_loss`, reused
  unmodified) against a FIXED per-candidate target `u_i = -MSE(Y_i, Y_q)`
  (identical every step); the step-to-step MASK advances along the ORACLE's
  own greedy order (not the model's argmax) -- this is a deterministic
  curriculum, not on-policy, exactly as specified.

- Set Oracle: sequential Choice CE against `u_i^(t) = -MSE(Aggregate(
  S_hat_{t-1} + {i}), Y_q)`, recomputed every step via `utils.dense_utility.
  dense_utility` (reused unmodified -- an EMPTY prefix already reduces
  `dense_utility`'s formula to exactly `-MSE(Y_i, Y_q)`, so t=1's target is
  automatically identical to the Individual Oracle's own t=1 target, with no
  special-cased math). The aggregate's weighting `alpha_i` always uses the
  FIXED base score `b_i` (`w_base`, computed once per batch, detached) --
  never the step-conditioned score -- matching the spec's explicit
  aggregation-weight rule. The mask/prefix advances along the MODEL's own
  on-policy argmax pick (`hat_i_t`), detached, no gradient through argmax.

Query future `Y_q` is used ONLY inside `dense_utility`'s no-grad target
construction; it is never an input to the encoder, SetConditioner, or scorer.
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

from layers.retrieval_metric import RetrievalMetric, cosine_init_deviation, oracle_rank_statistics
from models.RelationStage1 import relation_bank_collapse_metrics
from models.SequentialSetRetriever import SetConditioner
from scripts.train_margutil01 import build_experiment, memory_value
from scripts.train_oracle_choice01 import oracle_choice_step_loss
from utils.dense_utility import candidate_weights, dense_utility


def encode_raw(model, x, c):
    """RAW (unnormalised) encoder embedding -- normalisation happens only at
    the scoring boundary (`base_score`), never baked into the representation
    itself, so cosine and asymmetric arms see exactly the same encoder
    output."""
    return model.encoder(model._relation_tensor(x, c, c))


def base_score(z_q, z_k, metric):
    """b_i = cos(z_q, z_i) if `metric` is None, else `metric.score(z_q, z_k)`
    (asymmetric, `output='cosine'` -- already L2-normalises internally)."""
    if metric is None:
        return torch.matmul(F.normalize(z_q, dim=-1), F.normalize(z_k, dim=-1).transpose(0, 1))
    return metric.score(z_q, z_k)


def run_sequence_individual(u_hat, u_target, cand_mask, tau_choice, k):
    """Individual Oracle: `u_hat` (the model's base score) and `u_target`
    (fixed `-MSE(Y_i,Y_q)`) are BOTH step-invariant -- only the mask changes,
    advancing along the ORACLE's own greedy order (not the model's argmax),
    per the spec's explicit deterministic-curriculum design. No
    SetConditioner anywhere in this arm."""
    selected_mask = torch.zeros_like(cand_mask)
    losses, diags, picks = [], [], []
    neg_inf = torch.finfo(u_target.dtype).min / 4
    for _ in range(k):
        valid_now = cand_mask & ~selected_mask
        loss_t, diag_t = oracle_choice_step_loss(u_hat, u_target, valid_now, tau_choice)
        losses.append(loss_t)
        diags.append(diag_t)
        target_masked = u_target.masked_fill(~valid_now, neg_inf)
        nxt = target_masked.argmax(dim=-1, keepdim=True)  # Oracle's OWN next pick
        picks.append(nxt.squeeze(-1))
        selected_mask = selected_mask.scatter(1, nxt, True)
    return losses, diags, torch.stack(picks, dim=1)


def run_sequence_set(z_q, E, cand_mask, set_conditioner, metric, w_base,
                      futures, query_future, tau_choice, k, chunk_size):
    """Set Oracle: t=1 bypasses the conditioner entirely (`h_1 = z_q`); t>=2
    conditions on the mean of the model's OWN on-policy prefix's RAW (not
    L2-normalised beforehand) candidate embeddings. Target recomputed fresh
    every step via `dense_utility(prefix, w_base, futures, query_future)` --
    `w_base` is the FIXED base score's softmax weight (detached, computed
    once per batch from `metric`/cosine applied to `z_q`, never the
    step-conditioned score), matching Stage-2's own aggregation-weight
    convention. `torch.no_grad()` around the target construction blocks any
    gradient through `w_base`, satisfying the spec's detach requirement
    without a separate `.detach()` call."""
    bsz, device = z_q.size(0), z_q.device
    selected_mask = torch.zeros_like(cand_mask)
    picks = []
    losses, diags, a_dense_steps = [], [], []
    for t in range(k):
        if t == 0:
            s_hat = base_score(z_q, E, metric)  # conditioner BYPASSED
        else:
            prefix = torch.stack(picks, dim=1).detach()
            m = E[prefix].mean(dim=1)  # mean of RAW candidate embeddings
            h_t = set_conditioner(z_q, m)
            s_hat = base_score(h_t, E, metric)
        valid_now = cand_mask & ~selected_mask
        prefix_now = (torch.stack(picks, dim=1).detach() if t > 0
                      else torch.zeros(bsz, 0, dtype=torch.long, device=device))
        with torch.no_grad():
            a_dense = dense_utility(prefix_now, w_base, futures, query_future, chunk_size=chunk_size)
            u_target = -a_dense
        a_dense_steps.append(a_dense)
        loss_t, diag_t = oracle_choice_step_loss(s_hat, u_target, valid_now, tau_choice)
        losses.append(loss_t)
        diags.append(diag_t)
        s_masked = s_hat.masked_fill(~valid_now, float('-inf'))
        nxt = s_masked.argmax(dim=-1, keepdim=True).detach()  # model's OWN pick, on-policy
        picks.append(nxt.squeeze(-1).detach())
        selected_mask = selected_mask.scatter(1, nxt, True)
    return losses, diags, torch.stack(picks, dim=1), a_dense_steps


def _rank_fraction_for_step(u_hat, u_target, valid_now):
    """Oracle next-choice rank / valid-candidate-count for THIS step, via the
    existing `oracle_rank_statistics` (reused unmodified) -- `oracle_indices`
    is this step's single target index."""
    neg_inf = torch.finfo(u_target.dtype).min / 4
    target_masked = u_target.masked_fill(~valid_now, neg_inf)
    i_star = target_masked.argmax(dim=-1, keepdim=True)
    row_has_valid = valid_now.any(dim=-1)
    stats = oracle_rank_statistics(u_hat, i_star, valid_now, oracle_valid=row_has_valid)
    return stats


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
                    losses, diags, picks, a_dense_steps = run_sequence_set(
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

                # per-step rank-fraction, split t=1 vs t>=2 (Set only cares
                # about the split -- Individual's score never changes, but we
                # report the same split for symmetry).
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


def encoder_collapse_snapshot(model, memory_x, channels, sample_size=256):
    """Reuses `relation_bank_collapse_metrics` (existing, unmodified) on a
    [C,1,N,D] bank built from this arm's OWN (possibly still-training)
    encoder over the candidate memory."""
    with torch.no_grad():
        banks = [encode_raw(model, memory_x, c).unsqueeze(0).unsqueeze(0) for c in channels]
        key_bank = torch.cat(banks, dim=0)  # [C,1,N,D]
    return relation_bank_collapse_metrics(key_bank, sample_size=sample_size)


def scorer_deviation_diagnostics(metric):
    if metric is None:
        return {}
    with torch.no_grad():
        wq = metric.query_projection.weight
        wk = metric.key_projection.weight
        eye = torch.eye(wq.size(0), device=wq.device, dtype=wq.dtype)
        wq_dev = float((wq - eye).norm())
        wk_dev = float((wk - eye).norm())
        wq_cond = float(torch.linalg.cond(wq.float()))
        wk_cond = float(torch.linalg.cond(wk.float()))
    return {'wq_frobenius_dev_from_identity': wq_dev, 'wk_frobenius_dev_from_identity': wk_dev,
            'wq_condition_number': wq_cond, 'wk_condition_number': wk_cond,
            'cosine_init_deviation': cosine_init_deviation(metric)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True, help='EXISTING checkpoint, ARGS ONLY -- weights never loaded')
    ap.add_argument('--target', choices=['individual', 'set'], required=True)
    ap.add_argument('--scorer', choices=['cosine', 'asymmetric'], required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--checkpoints', default='checkpoints/exp_oracle_scratch01')
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
        'stage1_retrieval_metric': 'cosine',  # Model's OWN internal scorer stays unused/None -- we build our own
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

    # ---- SANITY: this model was built by Exp_Stage1_Relation._build_model(),
    # which only calls Model(args).float() -- no checkpoint weights loaded.
    # Verify against the reference checkpoint's OWN saved encoder weights, to
    # catch any accidental future refactor that starts auto-loading. ----
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
        print(f'[oracle_scratch01] loaded SHARED (not pretrained) encoder init from {cli.shared_init_in}')
    if cli.shared_init_out:
        Path(cli.shared_init_out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.encoder.state_dict(), cli.shared_init_out)
        print(f'[oracle_scratch01] saved this run\'s fresh encoder init to {cli.shared_init_out}')

    for p in model.encoder.parameters():
        assert p.requires_grad, 'encoder must be TRAINABLE (scratch) -- found a frozen parameter'

    exp._ensure_memory()
    channels = list(model.target_channels())
    for c in channels:
        sources = model.source_channels(c)
        if len(sources) != 1 or int(sources[0]) != int(c):
            raise ValueError(f'EXP-ORACLE-SCRATCH01 is self-only; channel {c} has sources {sources}')

    d_model = int(args.d_model)
    metric = None
    trainable_params = list(model.encoder.parameters())
    set_conditioner = None
    if cli.scorer == 'asymmetric':
        metric = RetrievalMetric(kind='asymmetric', dim=d_model, output='cosine', layer_norm=False).to(device)
        dev = cosine_init_deviation(metric)
        print(f'[oracle_scratch01] asymmetric scorer cosine_init_deviation={dev:.3e}')
        assert dev < 1e-5, 'asymmetric scorer does NOT match cosine at identity init'
        trainable_params += list(metric.parameters())
    if cli.target == 'set':
        set_conditioner = SetConditioner(d_model).to(device)
        trainable_params += list(set_conditioner.parameters())

    n_params = sum(p.numel() for p in trainable_params)
    print(f'[oracle_scratch01] target={cli.target} scorer={cli.scorer} pred_len={cli.pred_len} '
          f'trainable_params={n_params}')
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
        print(f"[oracle_scratch01] epoch {epoch+1} train_loss={train_metrics['loss']:.5f} "
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
                print(f'[oracle_scratch01] early stop at epoch {epoch+1} (best={best_epoch})')
                break

    wall_time = time.time() - wall_start
    peak_mem = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
    collapse = encoder_collapse_snapshot(model, exp.memory_x, channels)
    scorer_diag = scorer_deviation_diagnostics(metric)
    summary = {'best_epoch': best_epoch, 'best_val_rank_fraction': best_val, 'wall_clock_seconds': wall_time,
               'peak_gpu_memory_mib': peak_mem, 'history': history, 'trainable_params': n_params,
               'tau_choice': tau_choice, 'target': cli.target, 'scorer': cli.scorer, 'pred_len': cli.pred_len,
               'checkpoint': str(ckpt_dir / 'checkpoint.pth'), 'encoder_collapse_final': collapse,
               'scorer_diagnostics_final': scorer_diag}
    with open(ckpt_dir / 'summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f'[oracle_scratch01] done. best_epoch={best_epoch} best_val_rank_fraction={best_val:.5f} '
          f'wall_clock={wall_time:.1f}s peak_gpu_mem={peak_mem:.0f}MiB')
    print(f'[oracle_scratch01] summary written to {ckpt_dir / "summary.json"}')


if __name__ == '__main__':
    main()
