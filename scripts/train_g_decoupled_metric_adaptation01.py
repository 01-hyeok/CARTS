#!/usr/bin/env python3
"""TRACK-G-DECOUPLED-METRIC-ADAPTATION01.

CORRECTION to TRACK-F-LATE-INTERACTION-CONTROL01's interpretation: p120's
own training (`train_patch_retrieval_expert01.py`) put `model.parameters()`
-- the WHOLE encoder, norm and proj included -- into the Adam optimizer.
The "never-retrained norm->proj bottleneck" framing in that report was
WRONG. The correct question this script answers: does a SECOND, decoupled
adaptation stage (frozen already-trained trunk + a fresh metric head) beat
(a) the frozen trunk alone, (b) continuing to train the whole trunk with
no metric head, and (c) training the trunk and a fresh metric head jointly?

Six arms, all starting from the SAME p120 validation-best checkpoint:
  A0 Frozen-Final-Cosine:      frozen trunk, existing forward() output, cosine.
  A1 Frozen-Final+FreshAsym:   frozen trunk, existing forward() output,
                                + fresh identity-init Wq/Wk on top.
  A2 Frozen-RawCLS+FreshAsym:  frozen trunk, raw pre-norm/proj CLS,
                                + fresh Wq/Wk (reproduces CONTROL01's B0).
  A3 Frozen-PatchMean+FreshAsym: frozen trunk, patch tokens, projected +
                                mean-pooled (reproduces CONTROL01's B1).
  A4 Continue-Encoder-NoHead:  trunk trainable (continues from p120 init),
                                no metric head, cosine on existing forward().
  A5 Joint-Encoder+FreshAsym:  trunk trainable + fresh identity-init Wq/Wk,
                                trained together.

Reuses UNMODIFIED: `individual_utility_memsafe`/`arm_score`/`encode_raw`
(`train_factorial_e2e01`), `normalized_teacher_prob`/`kl_loss`
(`train_horizon_retrieval_expert01`), `memory_value`/`build_experiment`
(`train_margutil01`), `stable_topk_indices` (`RelationStage1`),
`recall_at_k`/`ndcg_at_k` (`train_patch_retrieval_expert01`),
`encode_tokens` (`diag_f_late_interaction_feasibility01`),
`RawClsHead`/`PooledTokenHead`/`get_raw_input`/`state_sha`/`batch_order_sha`
(`train_f_late_interaction_control01`, for A2/A3's exact reproduction of
CONTROL01's B0/B1).
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.diag_f_late_interaction_feasibility01 import encode_tokens
from scripts.train_f_late_interaction_control01 import (RawClsHead, batch_order_sha, get_raw_input, state_sha)
from scripts.train_factorial_e2e01 import arm_score, encode_raw, individual_utility_memsafe
from scripts.train_horizon_retrieval_expert01 import kl_loss, normalized_teacher_prob
from scripts.train_margutil01 import build_experiment, memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k

ARMS = ('A0_frozen_final_cosine', 'A1_frozen_final_freshasym', 'A2_frozen_rawcls_freshasym',
       'A3_frozen_patchmean_freshasym', 'A4_continue_encoder_nohead', 'A5_joint_encoder_freshasym')
FROZEN_ARMS = ('A0_frozen_final_cosine', 'A1_frozen_final_freshasym', 'A2_frozen_rawcls_freshasym',
              'A3_frozen_patchmean_freshasym')
HAS_METRIC_HEAD = ('A1_frozen_final_freshasym', 'A2_frozen_rawcls_freshasym',
                   'A3_frozen_patchmean_freshasym', 'A5_joint_encoder_freshasym')


class AsymMetric(nn.Module):
    """Fresh identity-init Wq/Wk applied to whatever representation the
    caller hands it (final embedding for A1/A5, raw CLS for A2). Identical
    math to `train_f_late_interaction_control01.RawClsHead`, kept as a
    separate small class here since A1/A5 apply it to a DIFFERENT input
    (final normalized embedding, not raw CLS)."""

    def __init__(self, d_model):
        super().__init__()
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.w_q.weight)
        nn.init.eye_(self.w_k.weight)

    def project_q(self, z):
        return F.normalize(self.w_q(z), dim=-1)

    def project_k(self, z):
        return F.normalize(self.w_k(z), dim=-1)


class PatchMeanMetric(nn.Module):
    """A3: identical to CONTROL01's PooledTokenHead (token-wise Wq/Wk,
    normalize, mean-pool, re-normalize) -- redefined here (not imported)
    only because CONTROL01's class is named for its own arm; math is
    byte-for-byte the same, verified by the reproduction check in Phase 0."""

    def __init__(self, d_model):
        super().__init__()
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.w_q.weight)
        nn.init.eye_(self.w_k.weight)

    def project_q(self, tok):
        t = F.normalize(self.w_q(tok), dim=-1)
        return F.normalize(t.mean(dim=1), dim=-1)

    def project_k(self, tok):
        t = F.normalize(self.w_k(tok), dim=-1)
        return F.normalize(t.mean(dim=1), dim=-1)


def build_metric_head(arm, d_model):
    if arm in ('A1_frozen_final_freshasym', 'A5_joint_encoder_freshasym'):
        return AsymMetric(d_model)
    if arm == 'A2_frozen_rawcls_freshasym':
        return RawClsHead(d_model)
    if arm == 'A3_frozen_patchmean_freshasym':
        return PatchMeanMetric(d_model)
    return None  # A0, A4


def get_score_inputs(arm, model, x, c):
    """Returns the tensor(s) the arm's score function needs, from a live
    (gradient-tracked when encoder is trainable) forward pass."""
    if arm in ('A0_frozen_final_cosine', 'A1_frozen_final_freshasym', 'A4_continue_encoder_nohead',
              'A5_joint_encoder_freshasym'):
        return encode_raw(model, x, c)  # existing final embedding, forward()'s own path
    if arm == 'A2_frozen_rawcls_freshasym':
        return get_raw_input('B0_raw_cls', model, x, c)
    if arm == 'A3_frozen_patchmean_freshasym':
        relation_x = model._relation_tensor(x, c, c)
        _, tok = model.encoder(relation_x, return_tokens=True)
        return tok
    raise ValueError(arm)


def score_fn(arm, head, rep_q, rep_i):
    if arm in ('A0_frozen_final_cosine', 'A4_continue_encoder_nohead'):
        return arm_score(rep_q, rep_i, None)  # rep already normalized by forward()
    zq = head.project_q(rep_q)
    zk = head.project_k(rep_i)
    return zq @ zk.T


@torch.no_grad()
def cache_frozen_reps(arm, model, exp, channels, device):
    return {c: get_score_inputs(arm, model, exp.memory_x, c) for c in channels}


def uniform_and_weighted_aggregate(memory_c, offset_c, picks, scores_sel, query_future):
    y_sel = memory_c[picks] + offset_c.view(-1, 1, 1)
    uniform_agg = y_sel.mean(dim=1)
    uniform_mse = ((uniform_agg - query_future) ** 2).mean(-1)
    w = torch.softmax(scores_sel, dim=-1)
    weighted_agg = (y_sel * w.unsqueeze(-1)).sum(dim=1)
    weighted_mse = ((weighted_agg - query_future) ** 2).mean(-1)
    return uniform_mse, weighted_mse


def train_epoch(exp, args, model, head, cli, loader, channels, device, candidate_reps, tau_t, arm,
                record_batch_order=False):
    model.train(arm in ('A4_continue_encoder_nohead', 'A5_joint_encoder_freshasym'))
    if head is not None:
        head.train(True)
    memory_y, memory_x_last = exp.memory_y, exp.memory_x_last
    tot_loss, nb = 0.0, 0
    starts = []
    encoder_trainable = arm in ('A4_continue_encoder_nohead', 'A5_joint_encoder_freshasym')
    for batch_x, batch_y, batch_start_idx in loader:
        if record_batch_order:
            starts.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx)
                          else torch.as_tensor(batch_start_idx))
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        cli.optimizer.zero_grad()
        batch_loss = 0.0
        for c in channels:
            rep_q = get_score_inputs(arm, model, batch_x, c)
            rep_i = candidate_reps[c] if not encoder_trainable else get_score_inputs(arm, model, exp.memory_x, c)
            s = score_fn(arm, head, rep_q, rep_i)

            memory_c, offset_c = memory_value(args, batch_x, memory_y, memory_x_last, c)
            query_future = batch_y[:, :, c]
            with torch.no_grad():
                u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
                d = -u
                p_t = normalized_teacher_prob(d.masked_fill(~cand_mask, float('inf')), cand_mask, cli.tau_t)
                assert not p_t.requires_grad
            l = kl_loss(p_t, s, cand_mask, cli.tau_s)
            if cli.channelwise_backward:
                (l / len(channels)).backward()
            else:
                batch_loss = batch_loss + l / len(channels)
                continue
            batch_loss += float(l.detach()) / len(channels)
        if not cli.channelwise_backward:
            batch_loss.backward()
            batch_loss = float(batch_loss.detach())
        params_to_check = list(model.parameters()) if not encoder_trainable else []
        if params_to_check:
            trunk_untouched = all(p.grad is None for p in params_to_check)
            assert trunk_untouched, f'[ISSUE] frozen trunk received gradient in arm {arm}'
        cli.optimizer.step()
        tot_loss += batch_loss
        nb += 1
    out = {'train_loss': tot_loss / max(nb, 1)}
    if record_batch_order:
        out['batch_order_sha256'] = batch_order_sha(starts)
    return out


@torch.no_grad()
def eval_epoch(exp, args, model, head, cli, loader, channels, device, candidate_reps, top_k, arm,
              collect_per_query=False, encoder_trainable=False):
    model.eval()
    if head is not None:
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
            rep_q = get_score_inputs(arm, model, batch_x, c)
            rep_i = candidate_reps[c] if not encoder_trainable else get_score_inputs(arm, model, exp.memory_x, c)
            s = score_fn(arm, head, rep_q, rep_i).masked_fill(~cand_mask, float('-inf'))
            memory_c, offset_c = memory_value(args, batch_x, memory_y, memory_x_last, c)
            query_future = batch_y[:, :, c]
            u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
            d = -u
            model_idx = stable_topk_indices(s, top_k, largest=True)
            oracle_idx = stable_topk_indices(d.masked_fill(~cand_mask, float('inf')), top_k, largest=False)
            model_ind_mse = d.gather(1, model_idx).mean(-1)
            oracle_ind_mse = d.gather(1, oracle_idx).mean(-1)
            scores_sel = s.gather(1, model_idx)
            uniform_mse, weighted_mse = uniform_and_weighted_aggregate(memory_c, offset_c, model_idx,
                                                                       scores_sel, query_future)

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

            n_valid = cand_mask.sum(-1).float()
            ranks_full = (-s).argsort(dim=-1).argsort(dim=-1).float() + 1.0
            oracle_ranks = ranks_full.gather(1, oracle_idx)
            per_ch.setdefault('oracle_mean_rank', []).append(oracle_ranks.mean(-1).cpu())
            per_ch.setdefault('oracle_median_rank', []).append(oracle_ranks.median(-1).values.cpu())
            per_ch.setdefault('rank_fraction_mean', []).append(
                ((oracle_ranks - 1) / (n_valid.unsqueeze(-1) - 1).clamp_min(1)).mean(-1).cpu())

            if collect_per_query:
                for b in range(bsz):
                    per_query_rows.append({'query_start_idx': int(batch_start_idx[b]), 'channel': c,
                                           'individual_retmse10': float(model_ind_mse[b]),
                                           'uniform_aggregate_mse10': float(uniform_mse[b]),
                                           'weighted_aggregate_mse10': float(weighted_mse[b]),
                                           'oracle_regret': float((model_ind_mse - oracle_ind_mse)[b])})
        for k, vals in per_ch.items():
            sums[k] = sums.get(k, 0.0) + torch.cat(vals).sum().item()
        n += bsz
    n_ch = max(len(channels), 1)
    out = {k: v / max(n * n_ch, 1) for k, v in sums.items()}
    out['n_queries_seen'] = n
    if collect_per_query:
        return out, per_query_rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', default='ETTh1_720')
    ap.add_argument('--arm', required=True, choices=ARMS)
    ap.add_argument('--arm_checkpoint', required=True)
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--pred_len', type=int, default=720)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--tau_t', type=float, default=0.02)
    ap.add_argument('--tau_s', type=float, default=0.1)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--train_epochs', type=int, default=10)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--loader_seed', type=int, required=True)
    ap.add_argument('--channelwise_backward', action='store_true', default=True)
    ap.add_argument('--out_dir', default='results/TRACK-G-DECOUPLED-METRIC-ADAPTATION01/ETTh1_720')
    ap.add_argument('--checkpoints', default='checkpoints/track_g_decoupled_metric_adaptation01')
    ap.add_argument('--smoke_test', action='store_true')
    ap.add_argument('--limit_batches', type=int, default=0)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(cli.arm_checkpoint, map_location='cpu')
    ckpt_hash = hashlib.sha256(Path(cli.arm_checkpoint).read_bytes()).hexdigest()
    torch.manual_seed(0)
    exp, args = build_experiment(cli.reference_ckpt, {
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

    head = build_metric_head(cli.arm, d_model)
    if head is not None:
        head.to(device)
        head_init_sha = state_sha(head.state_dict())
    else:
        head_init_sha = None

    params = list(model.parameters()) if not frozen else []
    if head is not None:
        params += list(head.parameters())

    candidate_reps = cache_frozen_reps(cli.arm, model, exp, channels, device) if frozen else None

    torch.manual_seed(cli.loader_seed)
    _, train_loader = exp._get_data(flag='train', shuffle=True)
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)

    if cli.arm == 'A0_frozen_final_cosine':
        # No trainable parameters at all -- pure baseline reproduction,
        # zero training, evaluated once on every split.
        va = eval_epoch(exp, args, model, head, cli, val_loader, channels, device, candidate_reps,
                        cli.top_k, cli.arm, encoder_trainable=False)
        te, per_query = eval_epoch(exp, args, model, head, cli, test_loader, channels, device, candidate_reps,
                                   cli.top_k, cli.arm, collect_per_query=True, encoder_trainable=False)
        assert state_sha(model.state_dict()) == encoder_init_sha, '[ISSUE][ABORT] A0 trunk drifted (should be impossible, no optimizer)'
        out_dir = Path(cli.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {'arm': cli.arm, 'loader_seed': cli.loader_seed, 'frozen': True, 'best_epoch': 0,
                  'best_val_retmse10': va['model_top10_individual_mse'],
                  'test_retmse10': te['model_top10_individual_mse'],
                  'test_uniform_aggregate_mse10': te['uniform_aggregate_mse10'],
                  'test_weighted_aggregate_mse10': te['weighted_aggregate_mse10'],
                  'test_recall10': te['recall_at_10'], 'test_binary_oracle_ndcg10': te['binary_oracle_ndcg_at_10'],
                  'test_oracle_regret': te['oracle_regret'], 'test_oracle_mean_rank': te['oracle_mean_rank'],
                  'test_oracle_median_rank': te['oracle_median_rank'],
                  'test_rank_fraction_mean': te['rank_fraction_mean'],
                  'encoder_init_sha256': encoder_init_sha, 'head_init_sha256': None,
                  'encoder_param_displacement': 0.0, 'checkpoint_hash': ckpt_hash, 'batch_order_hashes': {}}
        (out_dir / f'{cli.arm}_seed{cli.loader_seed}_metrics.json').write_text(json.dumps(summary, indent=2))
        with open(out_dir / f'per_query_{cli.arm}_seed{cli.loader_seed}.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=['query_start_idx', 'channel', 'individual_retmse10',
                                              'uniform_aggregate_mse10', 'weighted_aggregate_mse10', 'oracle_regret'])
            w.writeheader()
            for r in per_query:
                w.writerow(r)
        print(f'[track_g] done. {cli.arm} seed={cli.loader_seed} (no training) '
             f'val_retMSE@10={va["model_top10_individual_mse"]:.6f} '
             f'test_retMSE@10={te["model_top10_individual_mse"]:.6f}')
        return

    cli.optimizer = torch.optim.Adam(params, lr=cli.learning_rate)

    if cli.smoke_test:
        tr = train_epoch(exp, args, model, head, cli, train_loader, channels, device, candidate_reps,
                         cli.tau_t, cli.arm)
        va = eval_epoch(exp, args, model, head, cli, val_loader, channels, device, candidate_reps,
                        cli.top_k, cli.arm, encoder_trainable=not frozen)
        if frozen:
            assert state_sha(model.state_dict()) == encoder_init_sha, '[ISSUE] frozen trunk drifted'
        print(f'[track_g] {cli.arm} loader_seed={cli.loader_seed} SMOKE PASS train_loss={tr["train_loss"]:.5f} '
             f'val_retMSE@10={va["model_top10_individual_mse"]:.6f}')
        return

    ckpt_dir = Path(cli.checkpoints) / cli.cell / cli.arm / f'seed{cli.loader_seed}'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best = {'val': float('inf'), 'epoch': -1}
    batch_order_hashes = {}
    epoch_rows = []
    t0 = time.time()
    for epoch in range(1, cli.train_epochs + 1):
        tr = train_epoch(exp, args, model, head, cli, train_loader, channels, device, candidate_reps,
                         cli.tau_t, cli.arm, record_batch_order=True)
        batch_order_hashes[f'epoch{epoch}'] = tr.pop('batch_order_sha256')
        va = eval_epoch(exp, args, model, head, cli, val_loader, channels, device, candidate_reps,
                        cli.top_k, cli.arm, encoder_trainable=not frozen)
        val_metric = va['model_top10_individual_mse']
        epoch_rows.append({'epoch': epoch, 'arm': cli.arm, 'loader_seed': cli.loader_seed,
                          'train_loss': tr['train_loss'], 'val_retmse10': val_metric,
                          'val_uniform_agg_mse10': va['uniform_aggregate_mse10'],
                          'val_recall10': va['recall_at_10'], 'val_oracle_mean_rank': va['oracle_mean_rank']})
        if val_metric < best['val']:
            best = {'val': val_metric, 'epoch': epoch}
            payload = {'model_state_dict': model.state_dict() if not frozen else None,
                      'head_state_dict': head.state_dict() if head is not None else None,
                      'epoch': epoch, 'val_metric': val_metric}
            torch.save(payload, ckpt_dir / 'checkpoint.pth')
        print(f'[track_g] {cli.arm} seed={cli.loader_seed} epoch={epoch} train_loss={tr["train_loss"]:.5f} '
             f'val_retMSE@10={val_metric:.6f} val_uniform_agg={va["uniform_aggregate_mse10"]:.6f}')
        if epoch - best['epoch'] >= cli.patience:
            print(f'[track_g] {cli.arm} seed={cli.loader_seed} early stop at epoch {epoch} (best={best["epoch"]})')
            break

    if frozen:
        assert state_sha(model.state_dict()) == encoder_init_sha, '[ISSUE][ABORT] frozen trunk drifted'
        encoder_param_displacement = 0.0
    else:
        with torch.no_grad():
            disp = sum(float((p - p0).norm()) for p, p0 in
                      zip(model.state_dict().values(), torch.load(cli.arm_checkpoint, map_location=device)['model_state_dict'].values()))
        encoder_param_displacement = disp

    bl = torch.load(ckpt_dir / 'checkpoint.pth', map_location=device)
    if not frozen and bl['model_state_dict'] is not None:
        model.load_state_dict(bl['model_state_dict'])
    if head is not None and bl['head_state_dict'] is not None:
        head.load_state_dict(bl['head_state_dict'])
    te, per_query = eval_epoch(exp, args, model, head, cli, test_loader, channels, device, candidate_reps,
                               cli.top_k, cli.arm, collect_per_query=True, encoder_trainable=not frozen)

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        'arm': cli.arm, 'loader_seed': cli.loader_seed, 'frozen': frozen, 'best_epoch': best['epoch'],
        'best_val_retmse10': best['val'], 'test_retmse10': te['model_top10_individual_mse'],
        'test_uniform_aggregate_mse10': te['uniform_aggregate_mse10'],
        'test_weighted_aggregate_mse10': te['weighted_aggregate_mse10'],
        'test_recall10': te['recall_at_10'], 'test_binary_oracle_ndcg10': te['binary_oracle_ndcg_at_10'],
        'test_oracle_regret': te['oracle_regret'], 'test_oracle_mean_rank': te['oracle_mean_rank'],
        'test_oracle_median_rank': te['oracle_median_rank'], 'test_rank_fraction_mean': te['rank_fraction_mean'],
        'encoder_init_sha256': encoder_init_sha, 'head_init_sha256': head_init_sha,
        'encoder_param_displacement': encoder_param_displacement,
        'checkpoint_hash': ckpt_hash, 'batch_order_hashes': batch_order_hashes,
        'wall_clock_seconds': time.time() - t0,
    }
    (out_dir / f'{cli.arm}_seed{cli.loader_seed}_metrics.json').write_text(json.dumps(summary, indent=2))
    with open(out_dir / f'per_query_{cli.arm}_seed{cli.loader_seed}.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['query_start_idx', 'channel', 'individual_retmse10',
                                          'uniform_aggregate_mse10', 'weighted_aggregate_mse10', 'oracle_regret'])
        w.writeheader()
        for r in per_query:
            w.writerow(r)
    with open(out_dir / f'epoch_curve_{cli.arm}_seed{cli.loader_seed}.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['epoch', 'arm', 'loader_seed', 'train_loss', 'val_retmse10',
                                          'val_uniform_agg_mse10', 'val_recall10', 'val_oracle_mean_rank'])
        w.writeheader()
        for r in epoch_rows:
            w.writerow(r)
    print(f'[track_g] done. {cli.arm} seed={cli.loader_seed} best_epoch={best["epoch"]} '
         f'test_retMSE@10={te["model_top10_individual_mse"]:.6f} '
         f'test_uniform_agg={te["uniform_aggregate_mse10"]:.6f} recall10={te["recall_at_10"]:.4f}')


if __name__ == '__main__':
    main()
