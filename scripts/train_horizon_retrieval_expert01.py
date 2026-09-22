#!/usr/bin/env python3
"""TRACK-A-HORIZON-RETRIEVAL-EXPERT01 -- first LEARNED-retriever experiment
following TRACK-A-HORIZON-RETRIEVAL-HEADROOM01's Oracle diagnostic. Trains
a past-only, single-shot (non-sequential) bi-encoder retriever, in two
architecturally matched arms:

  Global: one score s_G(q,i) = cosine(z_q, E_i), one Top-10 used for the
          whole H=720 horizon.
  Block:  the SAME shared encoder, PLUS three small per-block correction
          heads. s_b(q,i) = cosine(z_q + W_b(z_q), E_i), W_b a bias-free
          Linear zero-initialized so s_b == s_G exactly at init (spec:
          "0에 가까운 값으로 초기화해 시작 시 Global 점수와 같도록"). Each
          block's Top-10 is selected independently by its own s_b.

Both arms select Top-10 by a SINGLE-SHOT score sort (`stable_topk_indices`)
-- no sequential SetConditioner state, so the two arms differ ONLY in the
presence of the (zero-init) block correction heads, never in selection
mechanism (spec: "구조 차이를 섞지 마라").

Candidate future reconstruction reuses `scripts.train_margutil01.memory_value`
and the aggregate-MSE reconstruction (`scripts.diag_horizon_retrieval_headroom01
._gather_mean`) unmodified -- the same value-space convention as every other
Track-A experiment.

Loss (Stage-1, spec's forward KL): per query/channel/block, teacher
`p_T,b = softmax(-normalized_d_b/tau_T)` over valid candidates (query future
used ONLY here, for supervision -- never touches `s_b`/`s_G` or candidate
selection, which are past-only). Student `p_S,b = softmax(s_b/tau_S)`.
`KL(p_T,b || p_S,b) = sum p_T,b * (log p_T,b - log p_S,b)`, teacher DETACHED.
Since the teacher-entropy term `sum p_T,b * log p_T,b` has zero gradient
w.r.t. student parameters, `d/d(theta) KL(p_T,b||p_S,b) == d/d(theta)
[-sum p_T,b * log p_S,b]` (standard soft cross-entropy) -- implemented as
the literal KL (both terms), gradient-equivalence verified by
`tests/test_horizon_retrieval_expert01.py`.

Normalization of `d_b` (spec requires this defined and documented, decided
on train/val only): per-row z-score over the query's own VALID candidates
only (`(d - mean_valid) / std_valid`), chosen because it is scale-invariant
per query/channel/block without requiring any dataset-wide statistic that
could leak across queries or splits.
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
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.diag_horizon_retrieval_headroom01 import BLOCKS, _gather_mean, _mse
from scripts.rng_control01 import batch_order_sha256, make_loader_generator, set_global_seeds
from scripts.train_factorial_e2e01 import arm_score, encode_raw, individual_utility_memsafe
from scripts.train_margutil01 import build_experiment, memory_value

ARMS = ('global', 'block')
BLOCK_NAMES = ('block1', 'block2', 'block3')
EPS = 1e-8


class BlockCorrectionHeads(nn.Module):
    """Three zero-initialized, bias-free linear corrections on the QUERY
    embedding -- `z_q_b = z_q + W_b(z_q)`, so `s_b(q,i) := cosine(z_q_b, E_i)`
    equals `s_G(q,i)` exactly at init (W_b == 0). Past-only (function of
    `z_q` alone) and bank-searchable in O(1) extra cost: `E` (the candidate
    embedding bank) is never re-encoded per block, only the query side gets
    an extra small linear map before the same cosine-against-E lookup."""

    def __init__(self, d_model):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(d_model, d_model, bias=False)
                                    for _ in BLOCK_NAMES])
        for h in self.heads:
            nn.init.zeros_(h.weight)

    def forward(self, z_q, block_idx):
        return z_q + self.heads[block_idx](z_q)


def normalized_teacher_prob(d, valid_mask, tau_t, eps=EPS):
    """z-score-normalize `d` (lower-is-better distance) over each row's own
    valid candidates, then `softmax(-normalized/tau_t)` -- masked entries
    get probability exactly 0.0 (softmax of -inf logits)."""
    n_valid = valid_mask.sum(-1, keepdim=True).clamp_min(1).float()
    d_masked = d.masked_fill(~valid_mask, 0.0)
    mean = d_masked.sum(-1, keepdim=True) / n_valid
    sq = ((d - mean) ** 2).masked_fill(~valid_mask, 0.0)
    std = (sq.sum(-1, keepdim=True) / n_valid).clamp_min(eps).sqrt()
    normalized = (d - mean) / std
    logits = (-normalized / tau_t).masked_fill(~valid_mask, float('-inf'))
    return torch.softmax(logits, dim=-1)


def kl_loss(p_t, s, valid_mask, tau_s, eps=EPS):
    """KL(p_t || p_s), p_t DETACHED (constant w.r.t. student params) --
    gradient-equivalent to soft cross-entropy `-sum p_t * log p_s`
    (verified by test_kl_gradient_equals_soft_ce_gradient)."""
    p_t = p_t.detach()
    logits_s = (s / tau_s).masked_fill(~valid_mask, float('-inf'))
    log_p_s = torch.log_softmax(logits_s, dim=-1)
    log_p_s_safe = log_p_s.masked_fill(~valid_mask, 0.0)
    log_p_t = torch.log(p_t.clamp_min(eps))
    term = (p_t * (log_p_t - log_p_s_safe)).masked_fill(~valid_mask, 0.0)
    return term.sum(-1).mean()


def teacher_diagnostics(p_t, valid_mask, eps=EPS):
    n_valid = valid_mask.sum(-1).clamp_min(1).float()
    p = p_t.masked_fill(~valid_mask, 0.0)
    entropy = -(p * torch.log(p.clamp_min(eps))).sum(-1)
    top1_mass = p.max(dim=-1).values
    effective_positives = 1.0 / (p.square().sum(-1).clamp_min(eps))
    return {'teacher_entropy': float(entropy.mean()),
           'teacher_top1_mass': float(top1_mass.mean()),
           'teacher_effective_positives': float(effective_positives.mean()),
           'n_valid_mean': float(n_valid.mean())}


def _file_sha256(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def _git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception as e:
        return f'UNKNOWN ({e})'


def train_epoch(exp, args, model, heads, cli, loader, channels, device,
                record_batch_order=False, limit_batches=0):
    model.train(True)
    if heads is not None:
        heads.train(True)
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

            s_g = arm_score(z_q, E, None)
            u_g = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
            d_g = -u_g
            p_t_g = normalized_teacher_prob(d_g, cand_mask, cli.tau_t)
            kl_g = kl_loss(p_t_g, s_g, cand_mask, cli.tau_s)

            diag_sums['kl_g'] = diag_sums.get('kl_g', 0.0) + float(kl_g.detach())
            if cli.arm == 'global':
                ch_loss = kl_g
                with torch.no_grad():
                    diag = teacher_diagnostics(p_t_g, cand_mask)
                    for k, v in diag.items():
                        diag_sums[f'global_{k}'] = diag_sums.get(f'global_{k}', 0.0) + v
            else:
                kl_blocks = {}
                for bidx, name in enumerate(BLOCK_NAMES):
                    lo, hi = BLOCKS[name]
                    z_q_b = heads(z_q, bidx)
                    s_b = arm_score(z_q_b, E, None)
                    u_b = individual_utility_memsafe(memory_c[:, lo:hi], offset_c,
                                                     query_future[:, lo:hi], cli.chunk_size)
                    d_b = -u_b
                    p_t_b = normalized_teacher_prob(d_b, cand_mask, cli.tau_t)
                    kl_blocks[name] = kl_loss(p_t_b, s_b, cand_mask, cli.tau_s)
                    diag_sums[f'kl_{name}'] = diag_sums.get(f'kl_{name}', 0.0) + float(kl_blocks[name].detach())
                    with torch.no_grad():
                        diag = teacher_diagnostics(p_t_b, cand_mask)
                        for k, v in diag.items():
                            diag_sums[f'{name}_{k}'] = diag_sums.get(f'{name}_{k}', 0.0) + v
                ch_loss = 0.5 * kl_g + sum(kl_blocks.values()) / 6.0
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


@torch.no_grad()
def eval_epoch(exp, args, model, heads, cli, loader, channels, device, top_k=10,
              limit_batches=0):
    """Hard, single-shot Top-K selection (NEVER touches query_future) then
    uniform-aggregate reconstruction -- the primary checkpoint-selection and
    reporting metric (full-H720 uniform MSE), computed identically for
    Global (one Top-10 for the whole horizon) and Block (concat of three
    per-block Top-10 aggregates)."""
    model.train(False)
    if heads is not None:
        heads.train(False)
    memory_y, memory_x_last = exp.memory_y, exp.memory_x_last
    sums = {}
    n = 0
    for bi, (batch_x, batch_y, batch_start_idx) in enumerate(loader):
        if limit_batches and bi >= limit_batches:
            break
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        bsz = batch_x.size(0)
        per_ch = {}
        for c in channels:
            z_q = encode_raw(model, batch_x, c)
            E = encode_raw(model, exp.memory_x, c)
            memory_c, offset_c = memory_value(args, batch_x, memory_y, memory_x_last, c)
            query_future = batch_y[:, :, c]
            neg_inf = float('-inf')

            s_g = arm_score(z_q, E, None).masked_fill(~cand_mask, neg_inf)
            s_g_picks = stable_topk_indices(s_g, top_k, largest=True)
            yhat_g = _gather_mean(memory_c, offset_c, s_g_picks, 0, 720)
            global_mse = _mse(yhat_g, query_future)
            per_ch.setdefault('global_h720_mse', []).append(global_mse.cpu())

            if cli.arm == 'block':
                block_preds = []
                for bidx, name in enumerate(BLOCK_NAMES):
                    lo, hi = BLOCKS[name]
                    z_q_b = heads(z_q, bidx)
                    s_b = arm_score(z_q_b, E, None).masked_fill(~cand_mask, neg_inf)
                    picks_b = stable_topk_indices(s_b, top_k, largest=True)
                    yb = _gather_mean(memory_c, offset_c, picks_b, lo, hi)
                    block_preds.append(yb)
                    bm = _mse(yb, query_future[:, lo:hi])
                    per_ch.setdefault(f'{name}_mse', []).append(bm.cpu())
                    ov = (s_g_picks.unsqueeze(-1) == picks_b.unsqueeze(-2)).any(-1).float().sum(-1) / top_k
                    per_ch.setdefault(f'overlap_global_{name}', []).append(ov.cpu())
                yhat_block = torch.cat(block_preds, dim=1)
                block_mse = _mse(yhat_block, query_future)
                per_ch.setdefault('block_h720_mse', []).append(block_mse.cpu())
                # reconstruction-consistency check: length-weighted block MSEs
                # must equal a length-weighted recombination of the same values
                recon = torch.cat([
                    ((block_preds[i] - query_future[:, lo:hi]) ** 2) for i, (lo, hi) in
                    enumerate(BLOCKS.values())], dim=1).mean(-1)
                per_ch.setdefault('block_h720_mse_reconcheck', []).append(recon.cpu())
        for key, vals in per_ch.items():
            sums[key] = sums.get(key, 0.0) + torch.cat(vals).sum().item()
        n += bsz

    n_channels = max(len(channels), 1)
    out = {key: sums[key] / max(n * n_channels, 1) for key in sums}
    out['n_queries_seen'] = n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--arm', choices=ARMS, required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--pred_len', type=int, default=720)
    ap.add_argument('--checkpoints', default='checkpoints/track_a_horizon_retrieval_expert01')
    ap.add_argument('--out_dir', default='results/TRACK-A-HORIZON-RETRIEVAL-EXPERT01')
    ap.add_argument('--shared_init_out', default=None)
    ap.add_argument('--shared_init_in', default=None)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--train_epochs', type=int, default=10)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=0.0)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--init_seed', type=int, default=0)
    ap.add_argument('--loader_seed', type=int, default=0)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--tau_t', type=float, default=0.1)
    ap.add_argument('--tau_s', type=float, default=0.1)
    ap.add_argument('--limit_batches', type=int, default=0, help='SMOKE ONLY')
    ap.add_argument('--channelwise_backward', action='store_true',
                    help='Per-channel immediate .backward() instead of accumulating every '
                         "channel's graph before one backward -- same VRAM lever as "
                         'TRACK-A-SOLAR-VRAM-OPT01 P0.1, for high-channel-count datasets '
                         '(Solar=137). Mathematically identical gradient (linearity), only '
                         'one channel resident at a time. optimizer.step() still called once.')
    cli = ap.parse_args()

    if cli.pred_len != 720:
        raise SystemExit('[ISSUE][ABORT] this experiment is defined only for pred_len=720')

    set_global_seeds(cli.init_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cell_dir = Path(cli.out_dir) / cli.cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints) / cli.cell / cli.arm
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.pred_len, 'batch_size': cli.batch_size,
        'seed': cli.init_seed, 'top_k': cli.top_k, 'tau_topk': 0.1,
    })
    exp._ensure_memory()
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.to(device)
    channels = list(range(int(args.enc_in)))
    d_model = int(args.d_model)

    heads = BlockCorrectionHeads(d_model).to(device) if cli.arm == 'block' else None

    code_commit = _git_commit()
    if cli.shared_init_in:
        blob = torch.load(cli.shared_init_in, map_location='cpu')
        model.load_state_dict(blob['model_state_dict'])
        encoder_init_sha256 = blob['encoder_init_sha256']
    else:
        encoder_init_sha256 = hashlib.sha256(str(model.state_dict()).encode()).hexdigest()
        if cli.shared_init_out:
            Path(cli.shared_init_out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                        'encoder_init_sha256': encoder_init_sha256,
                        'seed': cli.init_seed, 'cell': cli.cell, 'code_commit': code_commit},
                       cli.shared_init_out)

    for p in model.parameters():
        p.requires_grad_(True)
    params = list(model.parameters())
    if heads is not None:
        params += list(heads.parameters())
    cli.optimizer = torch.optim.Adam(params, lr=cli.learning_rate, weight_decay=cli.weight_decay)

    train_gen = make_loader_generator(cli.loader_seed)
    _, train_loader = exp._get_data(flag='train', shuffle=True, generator=train_gen)
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)

    fingerprint = {
        'exp': 'TRACK-A-HORIZON-RETRIEVAL-EXPERT01', 'cell': cli.cell, 'arm': cli.arm,
        'blocks': BLOCKS, 'top_k': cli.top_k, 'tau_t': cli.tau_t, 'tau_s': cli.tau_s,
        'loss': 'block' == cli.arm and '0.5*KL_G + (KL_B1+KL_B2+KL_B3)/6' or 'KL_G',
        'init_seed': cli.init_seed, 'loader_seed': cli.loader_seed, 'channels': channels,
        'encoder_init_sha256': encoder_init_sha256, 'learning_rate': cli.learning_rate,
        'batch_size': cli.batch_size, 'epochs': cli.train_epochs, 'patience': cli.patience,
        'checkpoint_criterion': 'min val full-H720 uniform-aggregate hard Top-10 MSE',
        'code_commit': code_commit,
    }
    (cell_dir / f'config_fingerprint_{cli.arm}.json').write_text(json.dumps(fingerprint, indent=2))
    print(f'[horizon_retrieval_expert01] {cli.cell}/{cli.arm} encoder_init_sha='
         f'{encoder_init_sha256[:16]} init_seed={cli.init_seed} loader_seed={cli.loader_seed}')

    epoch_rows, best = [], {'val': float('inf'), 'epoch': -1}
    batch_order_hashes = {}
    t0 = time.time()
    for epoch in range(1, cli.train_epochs + 1):
        tr = train_epoch(exp, args, model, heads, cli, train_loader, channels, device,
                         record_batch_order=True, limit_batches=cli.limit_batches)
        batch_order_hashes[f'epoch{epoch}'] = tr.pop('batch_order_sha256')
        va = eval_epoch(exp, args, model, heads, cli, val_loader, channels, device,
                        top_k=cli.top_k, limit_batches=cli.limit_batches)
        val_metric = va['block_h720_mse'] if cli.arm == 'block' else va['global_h720_mse']
        row = {'epoch': epoch, **tr, **{f'val_{k}': v for k, v in va.items()}}
        epoch_rows.append(row)
        payload = {'model_state_dict': model.state_dict(),
                   'heads_state_dict': heads.state_dict() if heads is not None else None,
                   'args': vars(args), 'epoch': epoch, 'fingerprint': fingerprint,
                   'val_primary_mse': val_metric}
        torch.save(payload, ckpt_dir / f'checkpoint_epoch{epoch}.pth')
        if val_metric < best['val']:
            best = {'val': val_metric, 'epoch': epoch}
            torch.save(payload, ckpt_dir / 'checkpoint.pth')
        print(f"[horizon_retrieval_expert01] {cli.cell}/{cli.arm} epoch {epoch} "
             f"train_loss={tr['train_loss']:.5f} val_primary_mse={val_metric:.6f} "
             f"batch_order_sha256={batch_order_hashes[f'epoch{epoch}'][:16]}")
        if epoch - best['epoch'] >= cli.patience:
            print(f'[horizon_retrieval_expert01] early stop at epoch {epoch} (best={best["epoch"]})')
            break

    bl = torch.load(ckpt_dir / 'checkpoint.pth', map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    if heads is not None:
        heads.load_state_dict(bl['heads_state_dict'])
    te = eval_epoch(exp, args, model, heads, cli, test_loader, channels, device, top_k=cli.top_k)
    tr_full = eval_epoch(exp, args, model, heads, cli, train_loader, channels, device, top_k=cli.top_k,
                         limit_batches=cli.limit_batches)

    with open(cell_dir / f'epoch_metrics_{cli.arm}.csv', 'w', newline='') as fh:
        keys = sorted({k for r in epoch_rows for k in r})
        w = csv.DictWriter(fh, fieldnames=['epoch'] + [k for k in keys if k != 'epoch'])
        w.writeheader()
        for r in epoch_rows:
            w.writerow(r)
    (cell_dir / f'batch_order_hashes_{cli.arm}.json').write_text(json.dumps(batch_order_hashes, indent=2))

    summary = {
        'exp': 'TRACK-A-HORIZON-RETRIEVAL-EXPERT01', 'cell': cli.cell, 'arm': cli.arm,
        'best_epoch': best['epoch'], 'best_val_primary_mse': best['val'],
        'test_metrics': te, 'train_metrics_subset': tr_full,
        'encoder_init_sha256': encoder_init_sha256, 'batch_order_hashes': batch_order_hashes,
        'wall_clock_seconds': time.time() - t0, 'checkpoint': str(ckpt_dir / 'checkpoint.pth'),
        'fingerprint': fingerprint,
    }
    (cell_dir / f'retrieval_metrics_{cli.arm}.json').write_text(json.dumps(summary, indent=2))
    (cell_dir / f'DONE_{cli.arm}.marker').write_text(json.dumps({'done': True, 'best_epoch': best['epoch']}))
    primary = te.get('block_h720_mse' if cli.arm == 'block' else 'global_h720_mse')
    print(f"[horizon_retrieval_expert01] done. {cli.cell}/{cli.arm} best_epoch={best['epoch']} "
         f"test_primary_mse={primary:.6f}")


if __name__ == '__main__':
    main()
