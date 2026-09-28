#!/usr/bin/env python3
"""TRACK-F-LATE-INTERACTION-FEASIBILITY01 -- Phase F1 (per the spec's own
numbering, the frozen-trunk head-only probe; this repo's Phase-F0 script
is the no-training diagnostics): F4-Frozen-Trunk Learned Late Interaction.

The p120 Transformer trunk (`model.encoder`, `models.RelationStage1.
RelationEncoder`) is loaded from the existing checkpoint and completely
frozen (`requires_grad_(False)` on every parameter, `.eval()`). Only a
small learnable head is trained: `W_q`, `W_k` (per-token linear
projections, D->D), a positive-constrained score temperature
(`tau_a = softplus(raw_tau_a)`), and a positive-constrained position
penalty (`lambda_pos = softplus(raw_lambda)`). The window `w` is fixed
from F2's val-only grid search (not learned, not retuned here). Candidate
RAW tokens are cached once (trunk frozen -> tokens never change); `W_k`
is applied on the fly to the cached raw tokens every step (cheap linear
layer, not a Transformer re-encode) -- so this is fast for the same
reason `train_c_horizon_frozen03.py`'s frozen-trunk probes were fast.

Teacher/loss: byte-for-byte the same `normalized_teacher_prob`/`kl_loss`
(`scripts.train_horizon_retrieval_expert01`) and `individual_utility_
memsafe` (`scripts.train_factorial_e2e01`) used by the p120 baseline
itself -- the ONLY thing this script changes is the student score
function (pooled cosine -> frozen-trunk local-LSE token scoring).
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
from scripts.train_factorial_e2e01 import individual_utility_memsafe
from scripts.train_horizon_retrieval_expert01 import kl_loss, normalized_teacher_prob
from scripts.train_margutil01 import build_experiment, memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k


def state_sha(state_dict):
    h = hashlib.sha256()
    for key in sorted(state_dict):
        h.update(key.encode())
        h.update(state_dict[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class LateInteractionHead(nn.Module):
    def __init__(self, d_model, w):
        super().__init__()
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.w_q.weight)
        nn.init.eye_(self.w_k.weight)
        self.raw_tau_a = nn.Parameter(torch.tensor(-1.5))   # softplus(-1.5) ~ 0.20
        self.raw_lambda = nn.Parameter(torch.tensor(-2.0))  # softplus(-2.0) ~ 0.13
        self.w = w

    def tau_a(self):
        return F.softplus(self.raw_tau_a) + 1e-3

    def lam(self):
        return F.softplus(self.raw_lambda)

    def project_q(self, tok):
        return F.normalize(self.w_q(tok), dim=-1)

    def project_k(self, tok):
        return F.normalize(self.w_k(tok), dim=-1)

    def score(self, tok_q_proj, tok_i_proj):
        P = tok_q_proj.size(1)
        sim = torch.einsum('bpd,njd->bnpj', tok_q_proj, tok_i_proj)
        p_idx = torch.arange(P, device=sim.device).view(P, 1)
        j_idx = torch.arange(P, device=sim.device).view(1, P)
        dist = (p_idx - j_idx).abs()
        mask = dist <= self.w
        logits = sim / self.tau_a() - self.lam() * dist.float()
        logits = logits.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        return torch.logsumexp(logits, dim=-1).mean(dim=-1)


@torch.no_grad()
def cache_candidate_tokens(model, exp, channels, device):
    return {c: encode_tokens(model, exp.memory_x, c)[1] for c in channels}


def train_epoch(exp, args, model, head, cli, loader, channels, device, candidate_tokens_raw, tau_t):
    head.train(True)
    memory_y, memory_x_last = exp.memory_y, exp.memory_x_last
    tot_loss, nb = 0.0, 0
    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        cli.optimizer.zero_grad()
        batch_loss = 0.0
        for c in channels:
            _, tok_q_raw = encode_tokens(model, batch_x, c)
            tok_i_raw = candidate_tokens_raw[c]
            tok_q_proj = head.project_q(tok_q_raw)
            tok_i_proj = head.project_k(tok_i_raw)
            s = head.score(tok_q_proj, tok_i_proj)

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
        assert has_grad, '[ISSUE] head has no gradient'
        assert trunk_untouched, '[ISSUE] frozen trunk received gradient'
        cli.optimizer.step()
        tot_loss += batch_loss
        nb += 1
    return {'train_loss': tot_loss / max(nb, 1)}


@torch.no_grad()
def eval_epoch(exp, args, model, head, cli, loader, channels, device, candidate_tokens_raw, top_k):
    head.eval()
    memory_y, memory_x_last = exp.memory_y, exp.memory_x_last
    sums, n = {}, 0
    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        bsz = batch_x.size(0)
        per_ch = {}
        for c in channels:
            _, tok_q_raw = encode_tokens(model, batch_x, c)
            tok_i_raw = candidate_tokens_raw[c]
            s = head.score(head.project_q(tok_q_raw), head.project_k(tok_i_raw)).masked_fill(
                ~cand_mask, float('-inf'))
            memory_c, offset_c = memory_value(args, batch_x, memory_y, memory_x_last, c)
            query_future = batch_y[:, :, c]
            u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
            d = -u
            model_idx = stable_topk_indices(s, top_k, largest=True)
            oracle_idx = stable_topk_indices(d.masked_fill(~cand_mask, float('inf')), top_k, largest=False)
            model_ind_mse = d.gather(1, model_idx).mean(-1)
            oracle_ind_mse = d.gather(1, oracle_idx).mean(-1)
            per_ch.setdefault('model_top10_individual_mse', []).append(model_ind_mse.cpu())
            per_ch.setdefault('oracle_regret', []).append((model_ind_mse - oracle_ind_mse).cpu())
            per_ch.setdefault('recall_at_10', []).append(recall_at_k(model_idx, oracle_idx, top_k).cpu())
            per_ch.setdefault('ndcg_at_10', []).append(ndcg_at_k(model_idx, d, cand_mask, top_k).cpu())
        for k, vals in per_ch.items():
            sums[k] = sums.get(k, 0.0) + torch.cat(vals).sum().item()
        n += bsz
    n_ch = max(len(channels), 1)
    out = {k: v / max(n * n_ch, 1) for k, v in sums.items()}
    out['n_queries_seen'] = n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', default='ETTh1_720')
    ap.add_argument('--arm_checkpoint', required=True)
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--pred_len', type=int, default=720)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--tau_t', type=float, default=0.02)
    ap.add_argument('--tau_s', type=float, default=0.1)
    ap.add_argument('--w', type=int, required=True, help='fixed local window, from F2 val grid')
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--train_epochs', type=int, default=10)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--loader_seed', type=int, default=0)
    ap.add_argument('--out_dir', default='results/TRACK-F-LATE-INTERACTION-FEASIBILITY01/ETTh1_720')
    ap.add_argument('--checkpoints', default='checkpoints/track_f_late_interaction_feasibility01')
    ap.add_argument('--smoke_test', action='store_true')
    ap.add_argument('--limit_batches', type=int, default=0)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cli.seed)
    ckpt = torch.load(cli.arm_checkpoint, map_location='cpu')
    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.pred_len, 'batch_size': cli.batch_size, 'seed': cli.seed,
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

    torch.manual_seed(cli.seed)
    head = LateInteractionHead(d_model, cli.w).to(device)
    head_init_sha = state_sha(head.state_dict())
    cli.optimizer = torch.optim.Adam(head.parameters(), lr=cli.learning_rate)

    candidate_tokens_raw = cache_candidate_tokens(model, exp, channels, device)
    assert state_sha(model.state_dict()) == frozen_sha

    torch.manual_seed(cli.loader_seed)
    _, train_loader = exp._get_data(flag='train', shuffle=True)
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)

    if cli.smoke_test:
        tr = train_epoch(exp, args, model, head, cli, train_loader, channels, device,
                         candidate_tokens_raw, cli.tau_t)
        va = eval_epoch(exp, args, model, head, cli, val_loader, channels, device,
                        candidate_tokens_raw, cli.top_k)
        assert state_sha(model.state_dict()) == frozen_sha, '[ISSUE] trunk drifted during smoke'
        print(f'[train_f_probe] seed={cli.seed} SMOKE PASS train_loss={tr["train_loss"]:.5f} '
             f'val_retMSE@10={va["model_top10_individual_mse"]:.6f}')
        return

    ckpt_dir = Path(cli.checkpoints) / cli.cell / f'seed{cli.seed}'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best = {'val': float('inf'), 'epoch': -1}
    rows = []
    t0 = time.time()
    for epoch in range(1, cli.train_epochs + 1):
        tr = train_epoch(exp, args, model, head, cli, train_loader, channels, device,
                         candidate_tokens_raw, cli.tau_t)
        va = eval_epoch(exp, args, model, head, cli, val_loader, channels, device,
                        candidate_tokens_raw, cli.top_k)
        val_metric = va['model_top10_individual_mse']
        rows.append({'epoch': epoch, **tr, **{f'val_{k}': v for k, v in va.items()}})
        if val_metric < best['val']:
            best = {'val': val_metric, 'epoch': epoch}
            torch.save({'head_state_dict': head.state_dict(), 'epoch': epoch, 'val_metric': val_metric,
                       'w': cli.w, 'head_init_sha256': head_init_sha, 'frozen_trunk_sha256': frozen_sha,
                       'arm_checkpoint': cli.arm_checkpoint}, ckpt_dir / 'checkpoint.pth')
        print(f'[train_f_probe] seed={cli.seed} epoch={epoch} train_loss={tr["train_loss"]:.5f} '
             f'val_retMSE@10={val_metric:.6f} tau_a={float(head.tau_a()):.4f} lambda={float(head.lam()):.4f}')
        if epoch - best['epoch'] >= cli.patience:
            print(f'[train_f_probe] seed={cli.seed} early stop at epoch {epoch} (best={best["epoch"]})')
            break

    assert state_sha(model.state_dict()) == frozen_sha, '[ISSUE][ABORT] trunk drifted during training'
    bl = torch.load(ckpt_dir / 'checkpoint.pth', map_location=device)
    head.load_state_dict(bl['head_state_dict'])
    te = eval_epoch(exp, args, model, head, cli, test_loader, channels, device, candidate_tokens_raw, cli.top_k)

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {'seed': cli.seed, 'w': cli.w, 'best_epoch': best['epoch'], 'best_val': best['val'],
              'test_metrics': te, 'head_init_sha256': head_init_sha, 'frozen_trunk_sha256': frozen_sha,
              'wall_clock_seconds': time.time() - t0}
    (out_dir / f'probe_seed{cli.seed}_metrics.json').write_text(json.dumps(summary, indent=2))
    print(f'[train_f_probe] done. seed={cli.seed} best_epoch={best["epoch"]} best_val={best["val"]:.6f} '
         f'test_retMSE@10={te["model_top10_individual_mse"]:.6f}')


if __name__ == '__main__':
    main()
