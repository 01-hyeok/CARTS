#!/usr/bin/env python3
"""TRACK-F-LATE-INTERACTION-FEASIBILITY01 -- Phase F0: no-training
diagnostics. Reproduces the pooled p120 baseline (F0-Pooled) and evaluates
three training-free patch-token scoring variants (F1-Aligned,
F2-Local-LSE, F3-Unordered-MaxSim) on the SAME frozen p120 checkpoint,
SAME candidate mask, SAME teacher, SAME evaluation function used
everywhere else in Track A this session
(`stable_topk_indices`/`individual_utility_memsafe`/`recall_at_k`/
`ndcg_at_k`, all reused unmodified). No new distance math, no
retraining -- this script only changes which score function selects the
Top-K.
"""
import argparse
import itertools
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
from scripts.train_factorial_e2e01 import arm_score, individual_utility_memsafe
from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob
from scripts.train_margutil01 import build_experiment, memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k

ARM_NAMES = ('F0_pooled', 'F1_aligned', 'F3_unordered_maxsim')  # F2 grid added separately


def encode_tokens(model, x, c):
    """encode_raw's sibling: same `_relation_tensor` input, but asks the
    RelationEncoder for the patch-token bank too."""
    relation_x = model._relation_tensor(x, c, c)
    z, tokens = model.encoder(relation_x, return_tokens=True)
    return z, tokens


def score_f0(z_q, z_i, tok_q, tok_i):
    return arm_score(z_q, z_i, None)


def score_f1_aligned(z_q, z_i, tok_q, tok_i):
    """mean_p cos(h_q_p, h_i_p); tokens already L2-normalized by
    RelationEncoder, so a dot product IS the cosine."""
    return torch.einsum('bpd,npd->bnp', tok_q, tok_i).mean(-1)


def score_f3_maxsim(z_q, z_i, tok_q, tok_i):
    """mean_p max_j cos(h_q_p, h_i_j) -- position-unaware ablation."""
    sim = torch.einsum('bpd,njd->bnpj', tok_q, tok_i)  # [B,N,P,P]
    return sim.max(dim=-1).values.mean(dim=-1)


def score_f2_local_lse(z_q, z_i, tok_q, tok_i, w, lam, tau_a):
    """mean_p LSE_{j: |p-j|<=w} [cos(h_q_p,h_i_j)/tau_a - lam*|p-j|]."""
    P = tok_q.size(1)
    sim = torch.einsum('bpd,njd->bnpj', tok_q, tok_i)  # [B,N,P,P]
    p_idx = torch.arange(P, device=sim.device).view(P, 1)
    j_idx = torch.arange(P, device=sim.device).view(1, P)
    dist = (p_idx - j_idx).abs()
    mask = dist <= w
    logits = sim / tau_a - lam * dist.float()
    logits = logits.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    lse = torch.logsumexp(logits, dim=-1)  # [B,N,P]
    return lse.mean(dim=-1)


@torch.no_grad()
def evaluate_arm(score_fn, exp, args, model, loader, channels, device, tau_t, top_k, chunk_size,
                 limit_batches=0):
    memory_y, memory_x_last = exp.memory_y, exp.memory_x_last
    sums = {}
    n = 0
    t_score = 0.0
    for bi, (batch_x, batch_y, batch_start_idx) in enumerate(loader):
        if limit_batches and bi >= limit_batches:
            break
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        bsz = batch_x.size(0)
        per_ch = {}
        for c in channels:
            z_q, tok_q = encode_tokens(model, batch_x, c)
            z_i, tok_i = encode_tokens(model, exp.memory_x, c)
            memory_c, offset_c = memory_value(args, batch_x, memory_y, memory_x_last, c)
            query_future = batch_y[:, :, c]

            u = individual_utility_memsafe(memory_c, offset_c, query_future, chunk_size)
            d = -u
            p_t = normalized_teacher_prob(d, cand_mask, tau_t)  # unused here, teacher independent of scorer

            t0 = time.time()
            s = score_fn(z_q, z_i, tok_q, tok_i).masked_fill(~cand_mask, float('-inf'))
            torch.cuda.synchronize() if device.type == 'cuda' else None
            t_score += time.time() - t0

            model_idx = stable_topk_indices(s, top_k, largest=True)
            oracle_idx = stable_topk_indices(d.masked_fill(~cand_mask, float('inf')), top_k, largest=False)
            model_ind_mse = d.gather(1, model_idx).mean(-1)
            oracle_ind_mse = d.gather(1, oracle_idx).mean(-1)

            per_ch.setdefault('model_top10_individual_mse', []).append(model_ind_mse.cpu())
            per_ch.setdefault('oracle_top10_individual_mse', []).append(oracle_ind_mse.cpu())
            per_ch.setdefault('oracle_regret', []).append((model_ind_mse - oracle_ind_mse).cpu())
            per_ch.setdefault('recall_at_10', []).append(recall_at_k(model_idx, oracle_idx, top_k).cpu())
            per_ch.setdefault('ndcg_at_10', []).append(ndcg_at_k(model_idx, d, cand_mask, top_k).cpu())
            top1_idx = model_idx[:, :1]
            per_ch.setdefault('top1_individual_mse', []).append(d.gather(1, top1_idx).mean(-1).cpu())
        for key, vals in per_ch.items():
            sums[key] = sums.get(key, 0.0) + torch.cat(vals).sum().item()
        n += bsz
    n_channels = max(len(channels), 1)
    out = {key: sums[key] / max(n * n_channels, 1) for key in sums}
    out['n_queries_seen'] = n
    out['scoring_time_seconds'] = t_score
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', default='ETTh1_720')
    ap.add_argument('--arm_checkpoint', required=True)
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--pred_len', type=int, default=720)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--tau_t', type=float, default=0.02)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--split', required=True, choices=('train', 'val', 'test'))
    ap.add_argument('--out_dir', default='results/TRACK-F-LATE-INTERACTION-FEASIBILITY01/ETTh1_720')
    ap.add_argument('--limit_batches', type=int, default=0)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(cli.arm_checkpoint, map_location='cpu')
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

    _, loader = exp._get_data(flag=cli.split, shuffle=False)

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for name, fn in [('F0_pooled', score_f0), ('F1_aligned', score_f1_aligned),
                     ('F3_unordered_maxsim', score_f3_maxsim)]:
        t0 = time.time()
        _, loader_i = exp._get_data(flag=cli.split, shuffle=False)
        r = evaluate_arm(fn, exp, args, model, loader_i, channels, device,
                         cli.tau_t, cli.top_k, cli.chunk_size, limit_batches=cli.limit_batches)
        r['wall_clock_seconds'] = time.time() - t0
        results[name] = r
        print(f"[diag_f_la] {cli.split}/{name}: retMSE@10={r['model_top10_individual_mse']:.6f} "
             f"oracle_regret={r['oracle_regret']:.6f} recall@10={r['recall_at_10']:.4f} "
             f"ndcg@10={r['ndcg_at_10']:.4f} scoring_time={r['scoring_time_seconds']:.2f}s")

    for w, lam, tau_a in itertools.product((0, 1, 2), (0.0, 0.1, 0.5), (0.05, 0.1)):
        name = f'F2_local_lse_w{w}_lam{str(lam).replace(".", "")}_tau{str(tau_a).replace(".", "")}'
        fn = lambda z_q, z_i, tq, ti, _w=w, _l=lam, _t=tau_a: score_f2_local_lse(z_q, z_i, tq, ti, _w, _l, _t)
        _, loader_i = exp._get_data(flag=cli.split, shuffle=False)
        r = evaluate_arm(fn, exp, args, model, loader_i, channels,
                         device, cli.tau_t, cli.top_k, cli.chunk_size, limit_batches=cli.limit_batches)
        results[name] = r
        print(f"[diag_f_la] {cli.split}/{name}: retMSE@10={r['model_top10_individual_mse']:.6f}")

    (out_dir / f'posthoc_metrics_{cli.split}.json').write_text(json.dumps(results, indent=2))
    print(f'[diag_f_la] wrote {out_dir / f"posthoc_metrics_{cli.split}.json"}')


if __name__ == '__main__':
    main()
