#!/usr/bin/env python3
"""TRACK-I-PCA-FUTURE-TEACHER01 -- Phase B: B0 (raw future-MSE teacher) vs
B1 (fixed PCA future-space teacher) student training.

This is a MINIMAL modification of `train_patch_retrieval_expert01.py`
(the exact script that produced the p120 baseline this whole session's
Track A/G/H work builds on): the ONLY change is how the teacher distance
`d` (fed into `normalized_teacher_prob` to build `p_t`) is computed.
Student (`arm_score`, plain cosine on `encode_raw`), loss (`kl_loss`,
KL(p_t||p_s)), optimizer/LR/batch size/epochs/patience, candidate
mask/support, checkpoint-selection metric (`model_top10_individual_mse`,
ALWAYS measured against raw future MSE, regardless of teacher_mode --
`eval_epoch` is imported UNCHANGED from the original script and never
touches PCA), and evaluation protocol are all byte-identical to that
script's own train_epoch/eval_epoch where not explicitly noted below.

--teacher_mode raw  : `d = -individual_utility_memsafe(...)` (unchanged
                      from the original script -- running this mode IS a
                      reproduction of the original p120 training run,
                      given the same reference_ckpt/patch_len/seeds).
--teacher_mode pca   : `d` computed via `pca_distance_memsafe` against a
                      PCA basis fit ONCE (before any training step) on
                      the train-only candidate bank
                      (`exp.memory_y`-derived, per channel) and frozen for
                      the entire run -- never refit, never touched by
                      the optimizer.

Paired-seed protocol (spec section 6): `--init_seed` controls model
weight initialization (`set_global_seeds`, called identically to the
original script, before `Exp_Stage1_Relation(args)` constructs the
model) -- running B0 and B1 with the SAME `--init_seed` gives them
byte-identical initial weights (verified via `encoder_init_sha256`,
logged every run). `--loader_seed` controls only training batch order
(`make_loader_generator`) -- also logged (`batch_order_sha256` per
epoch) so a B0/B1 pair's batch orders can be diffed.
"""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pca_future_teacher01 import fit_pca, pca_distance_memsafe
from scripts.rng_control01 import batch_order_sha256, make_loader_generator, set_global_seeds
from scripts.train_factorial_e2e01 import arm_score, encode_raw, individual_utility_memsafe
from scripts.train_horizon_retrieval_expert01 import kl_loss, normalized_teacher_prob, teacher_diagnostics
from scripts.train_margutil01 import build_experiment, memory_value
from scripts.train_patch_retrieval_expert01 import eval_epoch  # UNCHANGED, always raw-MSE evaluation

EPS = 1e-8


def _git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception as e:
        return f'UNKNOWN ({e})'


def compute_teacher_distance(teacher_mode, pca_bases, c, memory_c, offset_c, query_future, chunk_size):
    if teacher_mode == 'raw':
        return -individual_utility_memsafe(memory_c, offset_c, query_future, chunk_size)
    mean, comp = pca_bases[c]
    return pca_distance_memsafe(memory_c, offset_c, query_future, mean, comp, metric='l2', chunk_size=chunk_size)


def train_epoch(exp, args, model, cli, loader, channels, device, pca_bases,
                record_batch_order=False, limit_batches=0):
    """Identical to train_patch_retrieval_expert01.train_epoch except the
    single `d` computation line (teacher-mode dispatch)."""
    model.train(True)
    tot_loss, nb = 0.0, 0
    diag_sums, diag_n = {}, 0
    batch_starts = []
    memory_y, memory_x_last = exp.memory_y, exp.memory_x_last

    for bi, (batch_x, batch_y, batch_start_idx) in enumerate(loader):
        if limit_batches and bi >= limit_batches:
            break
        if record_batch_order:
            batch_starts.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx)
                                else torch.as_tensor(batch_start_idx))
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        cli.optimizer.zero_grad()
        batch_loss = 0.0

        for c in channels:
            z_q = encode_raw(model, batch_x, c)
            E = encode_raw(model, exp.memory_x, c)
            memory_c, offset_c = memory_value(args, batch_x, memory_y, memory_x_last, c)
            query_future = batch_y[:, :, c]

            s = arm_score(z_q, E, None)
            d = compute_teacher_distance(cli.teacher_mode, pca_bases, c, memory_c, offset_c,
                                         query_future, cli.chunk_size)
            p_t = normalized_teacher_prob(d, cand_mask, cli.tau_t)
            ch_loss = kl_loss(p_t, s, cand_mask, cli.tau_s)

            diag_sums['kl'] = diag_sums.get('kl', 0.0) + float(ch_loss.detach())
            with torch.no_grad():
                diag = teacher_diagnostics(p_t, cand_mask)
                for k, v in diag.items():
                    diag_sums[k] = diag_sums.get(k, 0.0) + v

            if getattr(cli, 'channelwise_backward', False):
                (ch_loss / len(channels)).backward()
                batch_loss = batch_loss + float(ch_loss.detach()) / len(channels)
            else:
                batch_loss = batch_loss + ch_loss
            diag_n += 1

        if getattr(cli, 'channelwise_backward', False):
            batch_loss_final = batch_loss
        else:
            batch_loss = batch_loss / len(channels)
            batch_loss.backward()
            batch_loss_final = float(batch_loss.detach())
        cli.optimizer.step()
        tot_loss += batch_loss_final
        nb += 1

    out = {'train_loss': tot_loss / max(nb, 1)}
    out.update({f'train_{k}': v / max(diag_n, 1) for k, v in diag_sums.items()})
    if record_batch_order:
        out['batch_order_sha256'] = batch_order_sha256(batch_starts)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--arm_name', required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--patch_len', type=int, default=120)
    ap.add_argument('--stride', type=int, default=None)
    ap.add_argument('--pred_len', type=int, default=720)
    ap.add_argument('--seq_len', type=int, default=720)
    ap.add_argument('--checkpoints', default='checkpoints/track_i_pca_future_teacher01')
    ap.add_argument('--out_dir', default='results/TRACK-I-PCA-FUTURE-TEACHER01')
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--train_epochs', type=int, default=10)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=0.0)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--init_seed', type=int, required=True, help='model weight init -- must match across a B0/B1 pair')
    ap.add_argument('--loader_seed', type=int, required=True)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--tau_t', type=float, default=0.02)
    ap.add_argument('--tau_s', type=float, default=0.1)
    ap.add_argument('--teacher_mode', required=True, choices=('raw', 'pca'))
    ap.add_argument('--pca_dim', type=int, default=None)
    ap.add_argument('--limit_batches', type=int, default=0, help='SMOKE ONLY')
    ap.add_argument('--channelwise_backward', dest='channelwise_backward', action='store_true', default=True)
    ap.add_argument('--no_channelwise_backward', dest='channelwise_backward', action='store_false')
    ap.add_argument('--smoke_test', action='store_true')
    cli = ap.parse_args()
    cli.stride = cli.stride or cli.patch_len

    if cli.teacher_mode == 'pca' and cli.pca_dim is None:
        raise SystemExit('[ISSUE][ABORT] --teacher_mode pca requires --pca_dim')

    set_global_seeds(cli.init_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cell_dir = Path(cli.out_dir) / cli.cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints) / cli.cell / cli.arm_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.seq_len, 'batch_size': cli.batch_size,
        'seed': cli.init_seed, 'top_k': cli.top_k, 'tau_topk': 0.1,
        'patch_len': cli.patch_len, 'stride': cli.stride,
        'relation_encoder_type': 'transformer', 'relation_self_fill': 'zero',
    })
    exp._ensure_memory()
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.to(device)
    channels = list(range(int(args.enc_in)))

    encoder_init_sha256 = hashlib.sha256(str(model.state_dict()).encode()).hexdigest()

    pca_bases = {}
    pca_actual_dims = {}
    if cli.teacher_mode == 'pca':
        dummy_batch_x = exp.memory_x[:1].to(device)
        for c in channels:
            memory_c, _ = memory_value(args, dummy_batch_x, exp.memory_y, exp.memory_x_last, c)
            mean, comp = fit_pca(memory_c, max_dim=cli.pca_dim)
            pca_bases[c] = (mean, comp)
            pca_actual_dims[c] = comp.shape[0]
        print(f'[track_i] PCA teacher: requested_dim={cli.pca_dim} actual_dims={pca_actual_dims}')

    for p in model.parameters():
        p.requires_grad_(True)
    param_count = sum(p.numel() for p in model.parameters())
    cli.optimizer = torch.optim.Adam(model.parameters(), lr=cli.learning_rate, weight_decay=cli.weight_decay)

    train_gen = make_loader_generator(cli.loader_seed)
    _, train_loader = exp._get_data(flag='train', shuffle=True, generator=train_gen)
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)

    code_commit = _git_commit()
    fingerprint = {
        'exp': 'TRACK-I-PCA-FUTURE-TEACHER01', 'cell': cli.cell, 'arm_name': cli.arm_name,
        'teacher_mode': cli.teacher_mode, 'pca_dim_requested': cli.pca_dim, 'pca_actual_dims': pca_actual_dims,
        'patch_len': cli.patch_len, 'stride': cli.stride, 'top_k': cli.top_k, 'tau_t': cli.tau_t, 'tau_s': cli.tau_s,
        'init_seed': cli.init_seed, 'loader_seed': cli.loader_seed, 'channels': channels,
        'encoder_init_sha256': encoder_init_sha256, 'learning_rate': cli.learning_rate,
        'batch_size': cli.batch_size, 'epochs': cli.train_epochs, 'patience': cli.patience,
        'checkpoint_criterion': 'min val model_top10_individual_mse (raw future MSE, always)',
        'code_commit': code_commit, 'param_count': param_count,
    }
    (cell_dir / f'config_fingerprint_{cli.arm_name}.json').write_text(json.dumps(fingerprint, indent=2))
    print(f'[track_i] {cli.cell}/{cli.arm_name} teacher_mode={cli.teacher_mode} pca_dim={cli.pca_dim} '
         f'encoder_init_sha={encoder_init_sha256[:16]} init_seed={cli.init_seed} loader_seed={cli.loader_seed}')

    if cli.smoke_test:
        tr = train_epoch(exp, args, model, cli, train_loader, channels, device, pca_bases,
                         limit_batches=cli.limit_batches or 2)
        va = eval_epoch(exp, args, model, cli, val_loader, channels, device, top_k=cli.top_k,
                        limit_batches=cli.limit_batches or 2)
        print(f'[track_i] {cli.arm_name} SMOKE PASS train_loss={tr["train_loss"]:.5f} '
             f'val_retMSE@10={va["model_top10_individual_mse"]:.6f}')
        return

    epoch_rows, best = [], {'val': float('inf'), 'epoch': -1}
    batch_order_hashes = {}
    t0 = time.time()
    for epoch in range(1, cli.train_epochs + 1):
        ep_t0 = time.time()
        tr = train_epoch(exp, args, model, cli, train_loader, channels, device, pca_bases,
                         record_batch_order=True, limit_batches=cli.limit_batches)
        batch_order_hashes[f'epoch{epoch}'] = tr.pop('batch_order_sha256')
        va = eval_epoch(exp, args, model, cli, val_loader, channels, device, top_k=cli.top_k,
                        limit_batches=cli.limit_batches)
        val_metric = va['model_top10_individual_mse']
        row = {'epoch': epoch, **tr, **{f'val_{k}': v for k, v in va.items()},
              'epoch_wall_seconds': time.time() - ep_t0}
        epoch_rows.append(row)
        payload = {'model_state_dict': model.state_dict(), 'args': vars(args),
                  'epoch': epoch, 'fingerprint': fingerprint, 'val_primary_mse': val_metric}
        torch.save(payload, ckpt_dir / f'checkpoint_epoch{epoch}.pth')
        if val_metric < best['val']:
            best = {'val': val_metric, 'epoch': epoch}
            torch.save(payload, ckpt_dir / 'checkpoint.pth')
        print(f'[track_i] {cli.arm_name} epoch={epoch} train_loss={tr["train_loss"]:.5f} '
             f'val_retMSE@10={val_metric:.6f} val_recall10={va["recall_at_10"]:.4f}')
        if epoch - best['epoch'] >= cli.patience:
            print(f'[track_i] {cli.arm_name} early stop at epoch {epoch} (best={best["epoch"]})')
            break

    bl = torch.load(ckpt_dir / 'checkpoint.pth', map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    te = eval_epoch(exp, args, model, cli, test_loader, channels, device, top_k=cli.top_k)

    summary = {
        'arm_name': cli.arm_name, 'teacher_mode': cli.teacher_mode, 'pca_dim': cli.pca_dim,
        'cell': cli.cell, 'init_seed': cli.init_seed, 'loader_seed': cli.loader_seed,
        'best_epoch': best['epoch'], 'best_val_retmse10': best['val'],
        'test_model_top10_individual_mse': te['model_top10_individual_mse'],
        'test_oracle_top10_individual_mse': te['oracle_top10_individual_mse'],
        'test_oracle_regret': te['oracle_regret'], 'test_recall_at_10': te['recall_at_10'],
        'test_ndcg_at_10': te['ndcg_at_10'], 'test_global_h720_mse': te['global_h720_mse'],
        'encoder_init_sha256': encoder_init_sha256, 'batch_order_hashes': batch_order_hashes,
        'code_commit': code_commit, 'wall_clock_seconds': time.time() - t0,
    }
    (cell_dir / f'summary_{cli.arm_name}.json').write_text(json.dumps(summary, indent=2))
    with open(cell_dir / f'epoch_curve_{cli.arm_name}.csv', 'w', newline='') as fh:
        fieldnames = sorted({k for r in epoch_rows for k in r})
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in epoch_rows:
            w.writerow(r)
    print(f'[track_i] done. {cli.arm_name} best_epoch={best["epoch"]} '
         f'test_retMSE@10={te["model_top10_individual_mse"]:.6f} test_recall10={te["recall_at_10"]:.4f} '
         f'test_ndcg10={te["ndcg_at_10"]:.4f}')


if __name__ == '__main__':
    main()
