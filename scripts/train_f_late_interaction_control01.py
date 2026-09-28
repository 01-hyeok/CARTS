#!/usr/bin/env python3
"""TRACK-F-LATE-INTERACTION-CONTROL01 -- B0/B1/B2 parameter/training-matched
control comparison.

Three arms, all on the SAME frozen p120 trunk, SAME teacher, SAME loss,
SAME optimizer/epoch/patience/checkpoint-criterion, SAME `loader_seed`
(a batch-order replication seed, NOT an independent model seed -- W_q/W_k
are identity-initialized, deterministic, seed-independent, exactly like
`LateInteractionHead` already was in F4/FEASIBILITY01):

  B0 Raw-CLS Learned Control: bypasses the existing norm->proj path,
    applies fresh W_q/W_k directly to the raw CLS token.
  B1 Pooled-Token Learned Control (the PRIMARY control for F4): applies
    the SAME shape W_q/W_k to the patch-token bank, mean-pools the
    projected+normalized tokens, re-normalizes. Identical frozen tokens,
    identical projections, identical training budget as B2 -- the only
    difference from B2 is mean-pooling vs local-LSE.
  B2 LateInteraction-F4: reuses `LateInteractionHead` from
    `train_f_late_interaction_probe01.py` UNMODIFIED (same w=2, same
    tau_a/lambda softplus parameterization, same identity init).

Reuses `encode_tokens` (`diag_f_late_interaction_feasibility01`),
`individual_utility_memsafe`/`normalized_teacher_prob`/`kl_loss`/
`recall_at_k`/`ndcg_at_k`/`stable_topk_indices`/`memory_value`/
`build_experiment` unmodified -- only the student score function differs
between arms.
"""
import argparse
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
from scripts.train_f_late_interaction_probe01 import LateInteractionHead
from scripts.train_factorial_e2e01 import individual_utility_memsafe
from scripts.train_horizon_retrieval_expert01 import kl_loss, normalized_teacher_prob
from scripts.train_margutil01 import build_experiment, memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k

ARMS = ('B0_raw_cls', 'B1_pooled_token', 'B2_late_interaction')


def state_sha(state_dict):
    h = hashlib.sha256()
    for key in sorted(state_dict):
        h.update(key.encode())
        h.update(state_dict[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def batch_order_sha(starts_list):
    h = hashlib.sha256()
    for s in starts_list:
        arr = s.numpy() if torch.is_tensor(s) else s
        h.update(arr.tobytes())
    return h.hexdigest()


class RawClsHead(nn.Module):
    """B0: identity-init W_q/W_k directly on the raw CLS token (bypasses
    RelationEncoder's own norm->proj)."""

    def __init__(self, d_model):
        super().__init__()
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.w_q.weight)
        nn.init.eye_(self.w_k.weight)

    def project_q(self, h_cls):
        return F.normalize(self.w_q(h_cls), dim=-1)

    def project_k(self, h_cls):
        return F.normalize(self.w_k(h_cls), dim=-1)

    def score(self, zq, zk):
        return zq @ zk.T


class PooledTokenHead(nn.Module):
    """B1: SAME token-level projection shape as B2, but mean-pooled
    instead of local-LSE -- the primary control for whether late
    interaction (not just extra W_q/W_k capacity) is doing the work."""

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

    def score(self, zq, zk):
        return zq @ zk.T


def build_head(arm, d_model, w):
    if arm == 'B0_raw_cls':
        return RawClsHead(d_model)
    if arm == 'B1_pooled_token':
        return PooledTokenHead(d_model)
    if arm == 'B2_late_interaction':
        return LateInteractionHead(d_model, w)
    raise ValueError(arm)


def get_raw_input(arm, model, x, c):
    """B0 needs the raw CLS vector (pre norm->proj); B1/B2 need the patch
    token bank. `RelationEncoder.forward(..., return_pre_normalized=True,
    return_tokens=True)` isn't wired for the CLS-raw case, so B0 reads the
    encoder's own patch_embed+transformer path directly via a one-line
    reuse of the same call `encode_tokens` already makes, then takes the
    CLS-equivalent by re-deriving it from the untouched pooled output path
    is unnecessary complexity -- simplest correct option: for B0 we reuse
    the tokens' own un-pooled CLS by calling the encoder internals once."""
    relation_x = model._relation_tensor(x, c, c)
    if arm == 'B0_raw_cls':
        enc = model.encoder
        tokens = enc.patch_embed(relation_x)
        cls = enc.cls_token.expand(tokens.size(0), -1, -1)
        out = enc.encoder(torch.cat([cls, tokens], dim=1))
        return out[:, 0]  # raw CLS, no norm/proj
    _, tok = model.encoder(relation_x, return_tokens=True)
    return tok


@torch.no_grad()
def cache_candidate_raw(arm, model, exp, channels, device):
    return {c: get_raw_input(arm, model, exp.memory_x, c) for c in channels}


def train_epoch(exp, args, model, head, cli, loader, channels, device, candidate_raw, tau_t,
                arm, record_batch_order=False):
    head.train(True)
    memory_y, memory_x_last = exp.memory_y, exp.memory_x_last
    tot_loss, nb = 0.0, 0
    starts = []
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
            raw_q = get_raw_input(arm, model, batch_x, c)
            raw_i = candidate_raw[c]
            zq = head.project_q(raw_q)
            zk = head.project_k(raw_i)
            s = head.score(zq, zk)

            memory_c, offset_c = memory_value(args, batch_x, memory_y, memory_x_last, c)
            query_future = batch_y[:, :, c]
            with torch.no_grad():
                u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
                d = -u
                p_t = normalized_teacher_prob(d.masked_fill(~cand_mask, float('inf')), cand_mask, cli.tau_t)
                assert not p_t.requires_grad
            l = kl_loss(p_t, s, cand_mask, cli.tau_s)
            (l / len(channels)).backward()
            batch_loss += float(l.detach()) / len(channels)
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
        trunk_untouched = all(p.grad is None for p in model.parameters())
        assert has_grad and trunk_untouched
        cli.optimizer.step()
        tot_loss += batch_loss
        nb += 1
    out = {'train_loss': tot_loss / max(nb, 1)}
    if record_batch_order:
        out['batch_order_sha256'] = batch_order_sha(starts)
    return out


@torch.no_grad()
def eval_epoch(exp, args, model, head, cli, loader, channels, device, candidate_raw, top_k, arm,
              collect_per_query=False):
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
        per_ch_full_agg = torch.zeros(bsz, len(channels), 720, device=device) if collect_per_query else None
        for ci, c in enumerate(channels):
            raw_q = get_raw_input(arm, model, batch_x, c)
            raw_i = candidate_raw[c]
            zq = head.project_q(raw_q)
            zk = head.project_k(raw_i)
            s = head.score(zq, zk).masked_fill(~cand_mask, float('-inf'))
            memory_c, offset_c = memory_value(args, batch_x, memory_y, memory_x_last, c)
            query_future = batch_y[:, :, c]
            u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
            d = -u
            model_idx = stable_topk_indices(s, top_k, largest=True)
            oracle_idx = stable_topk_indices(d.masked_fill(~cand_mask, float('inf')), top_k, largest=False)
            model_ind_mse = d.gather(1, model_idx).mean(-1)
            oracle_ind_mse = d.gather(1, oracle_idx).mean(-1)
            y_sel = memory_c[model_idx] + offset_c.view(-1, 1, 1)
            agg = y_sel.mean(dim=1)
            hard_agg_mse = ((agg - query_future) ** 2).mean(-1)

            per_ch.setdefault('model_top10_individual_mse', []).append(model_ind_mse.cpu())
            per_ch.setdefault('oracle_regret', []).append((model_ind_mse - oracle_ind_mse).cpu())
            per_ch.setdefault('recall_at_10', []).append(recall_at_k(model_idx, oracle_idx, top_k).cpu())
            per_ch.setdefault('ndcg_at_10', []).append(ndcg_at_k(model_idx, d, cand_mask, top_k).cpu())
            per_ch.setdefault('hard_aggregate_mse10', []).append(hard_agg_mse.cpu())
            if collect_per_query:
                for b in range(bsz):
                    per_query_rows.append({'query_start_idx': int(batch_start_idx[b]), 'channel': c,
                                           'individual_retmse10': float(model_ind_mse[b]),
                                           'hard_aggregate_mse10': float(hard_agg_mse[b]),
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
    ap.add_argument('--w', type=int, default=2)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--train_epochs', type=int, default=10)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--loader_seed', type=int, required=True)
    ap.add_argument('--out_dir', default='results/TRACK-F-LATE-INTERACTION-CONTROL01/ETTh1_720')
    ap.add_argument('--checkpoints', default='checkpoints/track_f_late_interaction_control01')
    ap.add_argument('--smoke_test', action='store_true')
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(cli.arm_checkpoint, map_location='cpu')
    torch.manual_seed(0)  # trunk/data construction determinism; NOT the loader-order seed
    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.pred_len, 'batch_size': cli.batch_size, 'seed': 0,
        'patch_len': 120, 'stride': 120,
        'relation_encoder_type': 'transformer', 'relation_self_fill': 'zero',
    })
    exp._ensure_memory()
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    channels = list(range(int(args.enc_in)))
    d_model = int(args.d_model)
    frozen_sha = state_sha(model.state_dict())

    head = build_head(cli.arm, d_model, cli.w).to(device)
    head_init_sha = state_sha(head.state_dict())
    cli.optimizer = torch.optim.Adam(head.parameters(), lr=cli.learning_rate)

    candidate_raw = cache_candidate_raw(cli.arm, model, exp, channels, device)
    assert state_sha(model.state_dict()) == frozen_sha

    torch.manual_seed(cli.loader_seed)
    _, train_loader = exp._get_data(flag='train', shuffle=True)
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)

    if cli.smoke_test:
        tr = train_epoch(exp, args, model, head, cli, train_loader, channels, device, candidate_raw,
                         cli.tau_t, cli.arm)
        va = eval_epoch(exp, args, model, head, cli, val_loader, channels, device, candidate_raw,
                        cli.top_k, cli.arm)
        assert state_sha(model.state_dict()) == frozen_sha, '[ISSUE] trunk drifted'
        print(f'[control01] {cli.arm} loader_seed={cli.loader_seed} SMOKE PASS '
             f'train_loss={tr["train_loss"]:.5f} val_retMSE@10={va["model_top10_individual_mse"]:.6f}')
        return

    ckpt_dir = Path(cli.checkpoints) / cli.cell / cli.arm / f'seed{cli.loader_seed}'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best = {'val': float('inf'), 'epoch': -1}
    batch_order_hashes = {}
    epoch_rows = []
    t0 = time.time()
    for epoch in range(1, cli.train_epochs + 1):
        tr = train_epoch(exp, args, model, head, cli, train_loader, channels, device, candidate_raw,
                         cli.tau_t, cli.arm, record_batch_order=True)
        batch_order_hashes[f'epoch{epoch}'] = tr.pop('batch_order_sha256')
        va = eval_epoch(exp, args, model, head, cli, val_loader, channels, device, candidate_raw,
                        cli.top_k, cli.arm)
        val_metric = va['model_top10_individual_mse']
        epoch_rows.append({'epoch': epoch, 'arm': cli.arm, 'loader_seed': cli.loader_seed,
                          'train_loss': tr['train_loss'], 'val_retmse10': val_metric})
        if val_metric < best['val']:
            best = {'val': val_metric, 'epoch': epoch}
            torch.save({'head_state_dict': head.state_dict(), 'epoch': epoch, 'val_metric': val_metric},
                      ckpt_dir / 'checkpoint.pth')
        print(f'[control01] {cli.arm} seed={cli.loader_seed} epoch={epoch} train_loss={tr["train_loss"]:.5f} '
             f'val_retMSE@10={val_metric:.6f}')
        if epoch - best['epoch'] >= cli.patience:
            print(f'[control01] {cli.arm} seed={cli.loader_seed} early stop at epoch {epoch} (best={best["epoch"]})')
            break

    frozen_sha_after = state_sha(model.state_dict())
    assert frozen_sha_after == frozen_sha, '[ISSUE][ABORT] trunk drifted during training'

    bl = torch.load(ckpt_dir / 'checkpoint.pth', map_location=device)
    head.load_state_dict(bl['head_state_dict'])
    te, per_query = eval_epoch(exp, args, model, head, cli, test_loader, channels, device, candidate_raw,
                               cli.top_k, cli.arm, collect_per_query=True)
    head_final_sha = state_sha(head.state_dict())

    trainable = sum(p.numel() for p in head.parameters())
    final_tau_a = float(head.tau_a()) if hasattr(head, 'tau_a') else None
    final_lambda = float(head.lam()) if hasattr(head, 'lam') else None

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        'arm': cli.arm, 'loader_seed': cli.loader_seed, 'best_epoch': best['epoch'],
        'best_val_retmse10': best['val'], 'test_retmse10': te['model_top10_individual_mse'],
        'test_hard_aggregate_mse10': te['hard_aggregate_mse10'], 'test_recall10': te['recall_at_10'],
        'test_ndcg10': te['ndcg_at_10'], 'oracle_regret': te['oracle_regret'],
        'initial_head_hash': head_init_sha, 'final_head_hash': head_final_sha,
        'frozen_trunk_hash_before': frozen_sha, 'frozen_trunk_hash_after': frozen_sha_after,
        'batch_order_hashes': batch_order_hashes, 'trainable_param_count': trainable,
        'effective_trainable_param_count': trainable,
        'final_tau_a': final_tau_a, 'final_lambda': final_lambda,
        'wall_clock_seconds': time.time() - t0,
    }
    (out_dir / f'{cli.arm}_seed{cli.loader_seed}_metrics.json').write_text(json.dumps(summary, indent=2))
    with open(out_dir / f'per_query_{cli.arm}_seed{cli.loader_seed}.csv', 'w') as fh:
        import csv
        w = csv.DictWriter(fh, fieldnames=['query_start_idx', 'channel', 'individual_retmse10',
                                          'hard_aggregate_mse10', 'oracle_regret'])
        w.writeheader()
        for r in per_query:
            w.writerow(r)
    print(f'[control01] done. {cli.arm} seed={cli.loader_seed} best_epoch={best["epoch"]} '
         f'test_retMSE@10={te["model_top10_individual_mse"]:.6f} '
         f'test_hard_agg_mse10={te["hard_aggregate_mse10"]:.6f} params={trainable}')


if __name__ == '__main__':
    main()
