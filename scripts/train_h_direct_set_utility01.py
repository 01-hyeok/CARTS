#!/usr/bin/env python3
"""TRACK-H-DIRECT-SET-UTILITY01.

Question: does candidate-wise retrieval quality (individual retMSE@10)
diverge from set-wise forecasting utility (Top-10 uniform-aggregate MSE),
and if so, can a simple differentiable auxiliary loss
    L = L_ind + lambda_agg * L_agg
(NOT a Greedy-Set-Oracle imitation, no SetConditioner/HostScorer, no
teacher re-derivation) close that gap without hurting individual
retrieval quality?

Four arms, all starting from a dataset's own p120 validation-best
checkpoint (ETTh1_720 or Weather_720):
  H0 Frozen-Individual:            frozen trunk + fresh asym Wq/Wk, L_ind only
                                   (== TRACK-G's A1).
  H1 Frozen-Individual+Aggregate:  frozen trunk + fresh asym Wq/Wk,
                                   L_ind + lambda_agg * L_agg.
  H2 Joint-Individual:             trunk trainable + fresh asym Wq/Wk,
                                   L_ind only (== TRACK-G's A5).
  H3 Joint-Individual+Aggregate:   trunk trainable + fresh asym Wq/Wk,
                                   L_ind + lambda_agg * L_agg.
`--encoder_lr` (default 1e-3, matching TRACK-G) lets H2/H3 be re-run at
1e-4 as the "low-LR control" required by the spec, without a separate
arm name -- the output filename encodes `encoder_lr`.

L_agg: full-memory soft-attention over the SAME masked candidate set the
individual objective sees, temperature `tau_agg` (default = tau_s =
0.1), weights over student score `s`, reconstructed candidate futures via
`memory_value()` (byte-identical convention to every other Track-A/G
script -- not reimplemented). Gradient explicitly kept OFF the candidate
futures and the query future (both are raw data tensors, detached
defensively); gradient flows only through `s` (student score) into the
metric head and, for H2/H3, the trainable encoder. No Greedy Set Oracle,
no new teacher -- `L_ind`'s teacher (`normalized_teacher_prob` +
`individual_utility_memsafe`) is reused unmodified from
`train_horizon_retrieval_expert01`/`train_factorial_e2e01`.

Reused UNMODIFIED: `individual_utility_memsafe`/`arm_score`/`encode_raw`
(`train_factorial_e2e01`), `normalized_teacher_prob`/`kl_loss`
(`train_horizon_retrieval_expert01`), `memory_value`/`build_experiment`
(`train_margutil01`), `stable_topk_indices` (`RelationStage1`),
`recall_at_k`/`ndcg_at_k` (`train_patch_retrieval_expert01`),
`AsymMetric`/`batch_order_sha`/`state_sha` (`train_g_decoupled_metric_adaptation01`
/ `train_f_late_interaction_control01`).
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.train_f_late_interaction_control01 import batch_order_sha, state_sha
from scripts.train_factorial_e2e01 import arm_score, encode_raw, individual_utility_memsafe
from scripts.train_g_decoupled_metric_adaptation01 import AsymMetric
from scripts.train_horizon_retrieval_expert01 import kl_loss, normalized_teacher_prob
from scripts.train_margutil01 import build_experiment, memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k

ARMS = ('H0_frozen_individual', 'H1_frozen_individual_aggregate',
       'H2_joint_individual', 'H3_joint_individual_aggregate')
FROZEN_ARMS = ('H0_frozen_individual', 'H1_frozen_individual_aggregate')
AGGREGATE_ARMS = ('H1_frozen_individual_aggregate', 'H3_joint_individual_aggregate')


def aggregate_prediction(memory_c, offset_c, scores, cand_mask, tau, chunk_size=None):
    """Full-memory soft-attention aggregate prediction. Returns
    (y_soft [B,H], weights [B,N]). Candidate futures / offsets are detached
    before use -- gradient into `y_soft` flows only through `weights`
    (i.e. only through `scores`)."""
    bsz = offset_c.size(0)
    n, h = memory_c.shape
    chunk_size = chunk_size or n
    logits = (scores / tau).masked_fill(~cand_mask, float('-inf'))
    w = torch.softmax(logits, dim=-1)
    memory_c_d = memory_c.detach()
    offset_c_d = offset_c.detach()
    y_soft = offset_c.new_zeros(bsz, h)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        y_c = memory_c_d[start:end].unsqueeze(0) + offset_c_d.view(-1, 1, 1)
        w_c = w[:, start:end].unsqueeze(-1)
        y_soft = y_soft + (w_c * y_c).sum(dim=1)
    return y_soft, w


def effective_number(w):
    """N_eff = 1 / sum(w_i^2), per row."""
    return 1.0 / (w ** 2).sum(-1).clamp_min(1e-12)


def uniform_and_weighted_aggregate(memory_c, offset_c, picks, scores_sel, query_future, tau_s):
    """Byte-compatible with TRACK-G's uniform metric; FIXES TRACK-G's
    weighted-aggregate bug (that script used un-tempered
    `softmax(scores_sel)` -- this one applies `tau_s`, per this round's
    explicit instruction not to repeat that mistake)."""
    y_sel = memory_c[picks] + offset_c.view(-1, 1, 1)
    uniform_agg = y_sel.mean(dim=1)
    uniform_mse = ((uniform_agg - query_future) ** 2).mean(-1)
    w = torch.softmax(scores_sel / tau_s, dim=-1)
    weighted_agg = (y_sel * w.unsqueeze(-1)).sum(dim=1)
    weighted_mse = ((weighted_agg - query_future) ** 2).mean(-1)
    return uniform_mse, weighted_mse, w


def pairwise_mse(y):
    """y: [B, K, H]. Returns [B] mean over all i<j pairs of MSE(y_i, y_j),
    vectorized (no python double loop over K)."""
    bsz, k, h = y.shape
    diff = y.unsqueeze(2) - y.unsqueeze(1)  # [B,K,K,H]
    sq = (diff ** 2).mean(-1)  # [B,K,K]
    iu = torch.triu_indices(k, k, offset=1)
    pair_vals = sq[:, iu[0], iu[1]]  # [B, K*(K-1)/2]
    return pair_vals.mean(-1)


def entropy_of(w):
    return -(w * torch.log(w.clamp_min(1e-12))).sum(-1)


def get_score_inputs(model, x, c):
    return encode_raw(model, x, c)


@torch.no_grad()
def cache_frozen_reps(model, exp, channels, device):
    return {c: get_score_inputs(model, exp.memory_x, c) for c in channels}


def train_epoch(exp, args, model, head, cli, loader, channels, device, candidate_reps, arm,
                record_batch_order=False):
    encoder_trainable = arm not in FROZEN_ARMS
    use_agg = arm in AGGREGATE_ARMS
    model.train(encoder_trainable)
    head.train(True)
    memory_y, memory_x_last = exp.memory_y, exp.memory_x_last
    tot_loss, tot_ind, tot_agg, nb = 0.0, 0.0, 0.0, 0
    starts = []
    for batch_x, batch_y, batch_start_idx in loader:
        if record_batch_order:
            starts.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx)
                          else torch.as_tensor(batch_start_idx))
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        cli.optimizer.zero_grad()
        batch_loss, batch_ind, batch_agg = 0.0, 0.0, 0.0
        for c in channels:
            rep_q = get_score_inputs(model, batch_x, c)
            rep_i = candidate_reps[c] if not encoder_trainable else get_score_inputs(model, exp.memory_x, c)
            zq, zk = head.project_q(rep_q), head.project_k(rep_i)
            s = zq @ zk.T

            memory_c, offset_c = memory_value(args, batch_x, memory_y, memory_x_last, c)
            query_future = batch_y[:, :, c].detach()
            with torch.no_grad():
                u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
                d = -u
                p_t = normalized_teacher_prob(d.masked_fill(~cand_mask, float('inf')), cand_mask, cli.tau_t)
                assert not p_t.requires_grad
            l_ind = kl_loss(p_t, s, cand_mask, cli.tau_s)
            l = l_ind
            l_agg_val = 0.0
            if use_agg:
                y_soft, _w = aggregate_prediction(memory_c, offset_c, s, cand_mask, cli.tau_agg, cli.chunk_size)
                l_agg = ((y_soft - query_future) ** 2).mean(-1).mean()
                l = l_ind + cli.lambda_agg * l_agg
                l_agg_val = float(l_agg.detach())
            (l / len(channels)).backward()
            batch_loss += float(l.detach()) / len(channels)
            batch_ind += float(l_ind.detach()) / len(channels)
            batch_agg += l_agg_val / len(channels)
        params_to_check = list(model.parameters()) if not encoder_trainable else []
        if params_to_check:
            assert all(p.grad is None for p in params_to_check), f'[ISSUE] frozen trunk received gradient in {arm}'
        cli.optimizer.step()
        tot_loss += batch_loss
        tot_ind += batch_ind
        tot_agg += batch_agg
        nb += 1
    out = {'train_loss': tot_loss / max(nb, 1), 'train_kl': tot_ind / max(nb, 1),
          'train_agg_loss': tot_agg / max(nb, 1)}
    if record_batch_order:
        out['batch_order_sha256'] = batch_order_sha(starts)
    return out


@torch.no_grad()
def eval_epoch(exp, args, model, head, cli, loader, channels, device, candidate_reps, top_k, arm,
              collect_per_query=False, encoder_trainable=False):
    model.eval()
    head.eval()
    memory_y, memory_x_last = exp.memory_y, exp.memory_x_last
    sums, n = {}, 0
    per_query_rows = []
    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        bsz = batch_x.size(0)
        per_ch = {}
        for c in channels:
            rep_q = get_score_inputs(model, batch_x, c)
            rep_i = candidate_reps[c] if not encoder_trainable else get_score_inputs(model, exp.memory_x, c)
            zq, zk = head.project_q(rep_q), head.project_k(rep_i)
            s = (zq @ zk.T).masked_fill(~cand_mask, float('-inf'))

            memory_c, offset_c = memory_value(args, batch_x, memory_y, memory_x_last, c)
            query_future = batch_y[:, :, c]
            u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
            d = -u
            model_idx = stable_topk_indices(s, top_k, largest=True)
            oracle_idx = stable_topk_indices(d.masked_fill(~cand_mask, float('inf')), top_k, largest=False)
            model_ind_mse = d.gather(1, model_idx).mean(-1)
            oracle_ind_mse = d.gather(1, oracle_idx).mean(-1)
            scores_sel = s.gather(1, model_idx)
            uniform_mse, weighted_mse, w_sel = uniform_and_weighted_aggregate(
                memory_c, offset_c, model_idx, scores_sel, query_future, cli.tau_s)
            complementarity_gain = model_ind_mse - uniform_mse

            y_sel = memory_c[model_idx] + offset_c.view(-1, 1, 1)  # [B,K,H]
            div_y = pairwise_mse(y_sel)
            e_sel = y_sel - query_future.unsqueeze(1)
            div_e = pairwise_mse(e_sel)
            n_eff = effective_number(w_sel)
            ent = entropy_of(w_sel)
            max_w = w_sel.max(-1).values

            per_ch.setdefault('model_top10_individual_mse', []).append(model_ind_mse.cpu())
            per_ch.setdefault('oracle_top10_individual_mse', []).append(oracle_ind_mse.cpu())
            per_ch.setdefault('oracle_regret', []).append((model_ind_mse - oracle_ind_mse).cpu())
            per_ch.setdefault('recall_at_10', []).append(recall_at_k(model_idx, oracle_idx, top_k).cpu())
            binary_hit = (model_idx.unsqueeze(-1) == oracle_idx.unsqueeze(-2)).any(-1).float()
            disc = 1.0 / torch.log2(torch.arange(2, top_k + 2, device=device).float())
            b_dcg = (binary_hit * disc.unsqueeze(0)).sum(-1)
            b_idcg = disc.sum()
            per_ch.setdefault('binary_oracle_ndcg_at_10', []).append((b_dcg / b_idcg).cpu())
            per_ch.setdefault('uniform_aggregate_mse10', []).append(uniform_mse.cpu())
            per_ch.setdefault('weighted_aggregate_mse10', []).append(weighted_mse.cpu())
            per_ch.setdefault('set_complementarity_gain', []).append(complementarity_gain.cpu())
            per_ch.setdefault('pairwise_future_mse', []).append(div_y.cpu())
            per_ch.setdefault('pairwise_error_mse', []).append(div_e.cpu())
            per_ch.setdefault('topk_score_entropy', []).append(ent.cpu())
            per_ch.setdefault('topk_effective_number', []).append(n_eff.cpu())
            per_ch.setdefault('topk_max_weight', []).append(max_w.cpu())

            n_valid = cand_mask.sum(-1).float()
            ranks_full = (-s).argsort(dim=-1).argsort(dim=-1).float() + 1.0
            oracle_ranks = ranks_full.gather(1, oracle_idx)
            per_ch.setdefault('oracle_mean_rank', []).append(oracle_ranks.mean(-1).cpu())
            per_ch.setdefault('oracle_median_rank', []).append(oracle_ranks.median(-1).values.cpu())
            per_ch.setdefault('rank_fraction_mean', []).append(
                ((oracle_ranks - 1) / (n_valid.unsqueeze(-1) - 1).clamp_min(1)).mean(-1).cpu())

            if collect_per_query:
                for b in range(bsz):
                    per_query_rows.append({
                        'query_start_idx': int(batch_start_idx[b]), 'channel': c,
                        'individual_retmse10': float(model_ind_mse[b]),
                        'uniform_aggregate_mse10': float(uniform_mse[b]),
                        'weighted_aggregate_mse10': float(weighted_mse[b]),
                        'set_complementarity_gain': float(complementarity_gain[b]),
                        'oracle_regret': float((model_ind_mse - oracle_ind_mse)[b]),
                        'pairwise_future_mse': float(div_y[b]), 'pairwise_error_mse': float(div_e[b]),
                        'topk_score_entropy': float(ent[b]), 'topk_effective_number': float(n_eff[b]),
                        'topk_max_weight': float(max_w[b]),
                    })
        for k, vals in per_ch.items():
            sums[k] = sums.get(k, 0.0) + torch.cat(vals).sum().item()
        n += bsz
    n_ch = max(len(channels), 1)
    out = {k: v / max(n * n_ch, 1) for k, v in sums.items()}
    out['n_queries_seen'] = n
    if collect_per_query:
        return out, per_query_rows
    return out


PER_QUERY_FIELDS = ['query_start_idx', 'channel', 'individual_retmse10', 'uniform_aggregate_mse10',
                    'weighted_aggregate_mse10', 'set_complementarity_gain', 'oracle_regret',
                    'pairwise_future_mse', 'pairwise_error_mse', 'topk_score_entropy',
                    'topk_effective_number', 'topk_max_weight']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True, choices=('ETTh1_720', 'Weather_720'))
    ap.add_argument('--arm', required=True, choices=ARMS)
    ap.add_argument('--checkpoint', required=True, help='p120 checkpoint (weights + args)')
    ap.add_argument('--pred_len', type=int, default=720)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--tau_t', type=float, default=0.02)
    ap.add_argument('--tau_s', type=float, default=0.1)
    ap.add_argument('--tau_agg', type=float, default=0.1)
    ap.add_argument('--lambda_agg', type=float, default=0.0)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--train_epochs', type=int, default=10)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--learning_rate', type=float, default=1e-3, help='metric head LR')
    ap.add_argument('--encoder_lr', type=float, default=1e-3, help='trunk LR (H2/H3 only)')
    ap.add_argument('--loader_seed', type=int, required=True)
    ap.add_argument('--out_dir', default='results/TRACK-H-DIRECT-SET-UTILITY01')
    ap.add_argument('--checkpoints', default='checkpoints/track_h_direct_set_utility01')
    ap.add_argument('--smoke_test', action='store_true')
    ap.add_argument('--limit_batches', type=int, default=0)
    cli = ap.parse_args()

    if cli.arm in AGGREGATE_ARMS and cli.lambda_agg <= 0.0:
        raise SystemExit(f'[ISSUE][ABORT] arm {cli.arm} requires --lambda_agg > 0')
    if cli.arm not in AGGREGATE_ARMS and cli.lambda_agg != 0.0:
        raise SystemExit(f'[ISSUE][ABORT] arm {cli.arm} is individual-only; --lambda_agg must be 0')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(cli.checkpoint, map_location='cpu')
    ckpt_hash = hashlib.sha256(Path(cli.checkpoint).read_bytes()).hexdigest()
    torch.manual_seed(0)
    exp, args = build_experiment(cli.checkpoint, {
        'pred_len': cli.pred_len, 'seq_len': cli.pred_len, 'batch_size': cli.batch_size, 'seed': 0,
        'patch_len': 120, 'stride': 120,
        'relation_encoder_type': 'transformer', 'relation_self_fill': 'zero',
    })
    exp._ensure_memory()
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    channels = list(range(int(args.enc_in)))
    d_model = int(args.d_model)
    encoder_init_sha = state_sha(model.state_dict())

    frozen = cli.arm in FROZEN_ARMS
    for p in model.parameters():
        p.requires_grad_(not frozen)
    if frozen:
        model.eval()

    head = AsymMetric(d_model).to(device)
    head_init_sha = state_sha(head.state_dict())

    if frozen:
        param_groups = [{'params': head.parameters(), 'lr': cli.learning_rate}]
    else:
        param_groups = [{'params': head.parameters(), 'lr': cli.learning_rate},
                        {'params': model.parameters(), 'lr': cli.encoder_lr}]
    cli.optimizer = torch.optim.Adam(param_groups)

    candidate_reps = cache_frozen_reps(model, exp, channels, device) if frozen else None

    torch.manual_seed(cli.loader_seed)
    _, train_loader = exp._get_data(flag='train', shuffle=True)
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)

    if cli.smoke_test:
        tr = train_epoch(exp, args, model, head, cli, train_loader, channels, device, candidate_reps, cli.arm)
        va = eval_epoch(exp, args, model, head, cli, val_loader, channels, device, candidate_reps,
                        cli.top_k, cli.arm, encoder_trainable=not frozen)
        if frozen:
            assert state_sha(model.state_dict()) == encoder_init_sha, '[ISSUE] frozen trunk drifted'
        print(f'[track_h] {cli.cell} {cli.arm} lr_enc={cli.encoder_lr} lambda={cli.lambda_agg} '
             f'loader_seed={cli.loader_seed} SMOKE PASS train_loss={tr["train_loss"]:.5f} '
             f'val_retMSE@10={va["model_top10_individual_mse"]:.6f} '
             f'val_uniform_agg={va["uniform_aggregate_mse10"]:.6f}')
        return

    tag = f'{cli.arm}_lr{cli.encoder_lr:g}_lam{cli.lambda_agg:g}'
    ckpt_dir = Path(cli.checkpoints) / cli.cell / tag / f'seed{cli.loader_seed}'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best = {'val': float('inf'), 'epoch': -1}
    best_agg = {'val': float('inf'), 'epoch': -1}
    batch_order_hashes = {}
    epoch_rows = []
    t0 = time.time()
    for epoch in range(1, cli.train_epochs + 1):
        tr = train_epoch(exp, args, model, head, cli, train_loader, channels, device, candidate_reps, cli.arm,
                         record_batch_order=True)
        batch_order_hashes[f'epoch{epoch}'] = tr.pop('batch_order_sha256')
        va = eval_epoch(exp, args, model, head, cli, val_loader, channels, device, candidate_reps,
                        cli.top_k, cli.arm, encoder_trainable=not frozen)
        val_metric = va['model_top10_individual_mse']
        val_agg = va['uniform_aggregate_mse10']
        epoch_rows.append({'epoch': epoch, 'arm': cli.arm, 'loader_seed': cli.loader_seed,
                          'train_loss': tr['train_loss'], 'train_kl': tr['train_kl'],
                          'train_agg_loss': tr['train_agg_loss'],
                          'val_retmse10': val_metric, 'val_uniform_agg_mse10': val_agg,
                          'val_complementarity_gain': val_metric - val_agg,
                          'val_recall10': va['recall_at_10'], 'val_oracle_mean_rank': va['oracle_mean_rank'],
                          'val_topk_effective_number': va['topk_effective_number']})
        if val_metric < best['val']:
            best = {'val': val_metric, 'epoch': epoch}
            payload = {'model_state_dict': model.state_dict() if not frozen else None,
                      'head_state_dict': head.state_dict(), 'epoch': epoch, 'val_metric': val_metric}
            torch.save(payload, ckpt_dir / 'checkpoint.pth')
        if val_agg < best_agg['val']:
            best_agg = {'val': val_agg, 'epoch': epoch}
        print(f'[track_h] {cli.cell} {tag} seed={cli.loader_seed} epoch={epoch} '
             f'train_loss={tr["train_loss"]:.5f} val_retMSE@10={val_metric:.6f} val_uniform_agg={val_agg:.6f}')
        if epoch - best['epoch'] >= cli.patience:
            print(f'[track_h] {cli.cell} {tag} seed={cli.loader_seed} early stop at epoch {epoch} '
                 f'(best={best["epoch"]})')
            break

    if frozen:
        assert state_sha(model.state_dict()) == encoder_init_sha, '[ISSUE][ABORT] frozen trunk drifted'
        encoder_param_displacement = 0.0
    else:
        with torch.no_grad():
            disp = sum(float((p - p0.to(p.device)).norm()) for p, p0 in
                      zip(model.state_dict().values(), ckpt['model_state_dict'].values()))
        encoder_param_displacement = disp

    bl = torch.load(ckpt_dir / 'checkpoint.pth', map_location=device)
    if not frozen and bl['model_state_dict'] is not None:
        model.load_state_dict(bl['model_state_dict'])
    head.load_state_dict(bl['head_state_dict'])
    te, per_query = eval_epoch(exp, args, model, head, cli, test_loader, channels, device, candidate_reps,
                               cli.top_k, cli.arm, collect_per_query=True, encoder_trainable=not frozen)

    out_dir = Path(cli.out_dir) / cli.cell
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        'cell': cli.cell, 'arm': cli.arm, 'encoder_lr': cli.encoder_lr, 'lambda_agg': cli.lambda_agg,
        'tau_agg': cli.tau_agg, 'loader_seed': cli.loader_seed, 'frozen': frozen,
        'best_epoch': best['epoch'], 'best_val_retmse10': best['val'],
        'best_aggregate_epoch': best_agg['epoch'], 'best_aggregate_val_uniform_agg_mse10': best_agg['val'],
        'test_retmse10': te['model_top10_individual_mse'],
        'test_uniform_aggregate_mse10': te['uniform_aggregate_mse10'],
        'test_weighted_aggregate_mse10': te['weighted_aggregate_mse10'],
        'test_set_complementarity_gain': te['set_complementarity_gain'],
        'test_recall10': te['recall_at_10'], 'test_binary_oracle_ndcg10': te['binary_oracle_ndcg_at_10'],
        'test_oracle_regret': te['oracle_regret'], 'test_oracle_mean_rank': te['oracle_mean_rank'],
        'test_oracle_median_rank': te['oracle_median_rank'], 'test_rank_fraction_mean': te['rank_fraction_mean'],
        'test_pairwise_future_mse': te['pairwise_future_mse'], 'test_pairwise_error_mse': te['pairwise_error_mse'],
        'test_topk_score_entropy': te['topk_score_entropy'], 'test_topk_effective_number': te['topk_effective_number'],
        'test_topk_max_weight': te['topk_max_weight'],
        'encoder_init_sha256': encoder_init_sha, 'head_init_sha256': head_init_sha,
        'encoder_param_displacement': encoder_param_displacement,
        'checkpoint_hash': ckpt_hash, 'batch_order_hashes': batch_order_hashes,
        'wall_clock_seconds': time.time() - t0,
    }
    (out_dir / f'{tag}_seed{cli.loader_seed}_metrics.json').write_text(json.dumps(summary, indent=2))
    with open(out_dir / f'per_query_{tag}_seed{cli.loader_seed}.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=PER_QUERY_FIELDS)
        w.writeheader()
        for r in per_query:
            w.writerow(r)
    with open(out_dir / f'epoch_curve_{tag}_seed{cli.loader_seed}.csv', 'w', newline='') as fh:
        fieldnames = ['epoch', 'arm', 'loader_seed', 'train_loss', 'train_kl', 'train_agg_loss', 'val_retmse10',
                     'val_uniform_agg_mse10', 'val_complementarity_gain', 'val_recall10', 'val_oracle_mean_rank',
                     'val_topk_effective_number']
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in epoch_rows:
            w.writerow(r)
    print(f'[track_h] done. {cli.cell} {tag} seed={cli.loader_seed} best_epoch={best["epoch"]} '
         f'(agg-best epoch={best_agg["epoch"]}) test_retMSE@10={te["model_top10_individual_mse"]:.6f} '
         f'test_uniform_agg={te["uniform_aggregate_mse10"]:.6f} '
         f'test_complementarity_gain={te["set_complementarity_gain"]:.6f} recall10={te["recall_at_10"]:.4f}')


if __name__ == '__main__':
    main()
