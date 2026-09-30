#!/usr/bin/env python3
"""TRACK-U-ASYMMETRY-CAPACITY-DECOMPOSITION01.

Decomposes TRACK-T2's single largest finding (Original KL -> Query-only
Projection, H720 Stage2 MSE 0.529134 -> 0.506830) into two competing
explanations: added learnable metric CAPACITY (a D x D matrix, however
applied) vs. broken QUERY/CANDIDATE SYMMETRY (comparing query and
candidate in different transformed spaces). Four arms, all single
-score-vector (NO multi-slot), all sharing encoder/teacher/KL objective/
candidate memory/mask/optimizer/batch order/checkpoint criterion:

  U0 (0 params):  s(q,k) = cos(z_q, z_k)                    -- TRUE Original KL
  U1 (D^2 params): s(q,k) = cos(W z_q, W z_k)                -- SHARED symmetric
  U2 (D^2 params): s(q,k) = cos(W_q z_q, z_k)                -- query-only asymmetric
  U3 (D^2 params): s(q,k) = cos(z_q, W_k z_k)                -- key-only asymmetric

U1/U2/U3's projection matrix is ALWAYS initialized identically:
`W = I + eps`, `eps ~ N(0, std^2)`, generated via a FIXED
`torch.Generator().manual_seed(PROJECTION_INIT_SEED)` call inside
`SingleProjection.__init__` -- calling this constructor in three
separate arm processes yields bit-identical initial `W` tensors (no
shared file needed), verified by unit test 5 and by the recorded
checkpoint-fingerprint init-projection hash. No bias term anywhere
(PART 5: "bias=False로 통일").

Reused UNMODIFIED: `build_model`/`state_hash`
(`train_j_shared_encoder_drift01`), `encode_raw`/`individual_utility_memsafe`
(`train_factorial_e2e01`), `normalized_teacher_prob`/`kl_loss`
(`train_horizon_retrieval_expert01`), `memory_value`
(`train_margutil01`), `stable_topk_indices` (`RelationStage1`),
`recall_at_k`/`ndcg_at_k` (`train_patch_retrieval_expert01`),
`make_loader_generator`/`batch_order_sha256` (`rng_control01`).
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.rng_control01 import batch_order_sha256, make_loader_generator
from scripts.train_factorial_e2e01 import encode_raw, individual_utility_memsafe
from scripts.train_horizon_retrieval_expert01 import kl_loss, normalized_teacher_prob
from scripts.train_j_shared_encoder_drift01 import build_model, state_hash
from scripts.train_margutil01 import memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k

EPS = 1e-8
TOP_K = 10
ARMS = ('U0', 'U1', 'U2', 'U3')
PROJECTION_INIT_SEED = 1000
PROJECTION_INIT_STD = 1e-3


class SingleProjection(nn.Module):
    """ONE D x D matrix, no bias, `W = I + eps`, `eps ~ N(0, std^2)`,
    deterministically generated via a fixed seed -- identical every time
    this constructor runs, in any process, guaranteeing U1/U2/U3 start
    from bit-identical initial weights without needing a shared file."""

    def __init__(self, d_model, std=PROJECTION_INIT_STD, seed=PROJECTION_INIT_SEED):
        super().__init__()
        w = torch.eye(d_model)
        g = torch.Generator().manual_seed(seed)
        w = w + torch.randn(d_model, d_model, generator=g) * std
        self.W = nn.Parameter(w)

    def forward(self, z):
        return F.normalize(torch.matmul(z, self.W.t()), dim=-1)


def projection_weight_sha(projection):
    if projection is None:
        return None
    h = hashlib.sha256()
    h.update(projection.W.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def compute_scores(model, projection, batch_x, memory_x, c, arm):
    """PART 3/7: query AND candidate encoder gradients both ON, no
    `.detach()` anywhere. `arm` selects which side(s) the shared
    `projection` (a single `SingleProjection`, or `None` for U0) is
    applied to -- U1 applies it to BOTH sides (symmetric), U2 to query
    only, U3 to candidate only."""
    z_q = encode_raw(model, batch_x, c)
    z_k = encode_raw(model, memory_x, c)
    if arm == 'U0':
        q, k = F.normalize(z_q, dim=-1), F.normalize(z_k, dim=-1)
    elif arm == 'U1':
        q, k = projection(z_q), projection(z_k)
    elif arm == 'U2':
        q, k = projection(z_q), F.normalize(z_k, dim=-1)
    elif arm == 'U3':
        q, k = F.normalize(z_q, dim=-1), projection(z_k)
    else:
        raise ValueError(arm)
    return torch.matmul(q, k.t())


def spearman_batch(score, d_raw, cand_mask):
    rhos = []
    score_np = score.detach().cpu().numpy()
    rel_np = (-d_raw).detach().cpu().numpy()
    mask_np = cand_mask.detach().cpu().numpy()
    for b in range(score_np.shape[0]):
        m = mask_np[b]
        if m.sum() < 2:
            continue
        rho, _ = spearmanr(score_np[b, m], rel_np[b, m])
        if rho == rho:
            rhos.append(rho)
    return float(sum(rhos) / len(rhos)) if rhos else float('nan')


def hard_eval_decomposition(scores, cand_mask, memory_c, offset_c, query_future, d_raw, oracle_idx, top_k=TOP_K):
    """Ordinary (single-score-vector) Top-K -- NO multi-slot round-robin
    anywhere in this track (PART 10: "single score vector에서 ordinary
    Top-10을 선택")."""
    s_masked = scores.masked_fill(~cand_mask, float('-inf'))
    model_idx = stable_topk_indices(s_masked, top_k, largest=True)
    model_ind_mse = d_raw.gather(1, model_idx).mean(-1)
    oracle_ind_mse = d_raw.gather(1, oracle_idx).mean(-1)
    recall10 = recall_at_k(model_idx, oracle_idx, top_k)
    ndcg10 = ndcg_at_k(model_idx, d_raw, cand_mask, top_k)
    y_sel = memory_c[model_idx] + offset_c.view(-1, 1, 1)
    e = y_sel - query_future.unsqueeze(1)
    individual_mse_i = (e ** 2).mean(-1)
    D_ = individual_mse_i.mean(-1) / top_k
    agg_pred = y_sel.mean(dim=1)
    agg_mse = ((agg_pred - query_future) ** 2).mean(-1)
    C_ = agg_mse - D_
    return dict(model_idx=model_idx, model_ind_mse=model_ind_mse, oracle_ind_mse=oracle_ind_mse,
               recall10=recall10, ndcg10=ndcg10, agg_mse=agg_mse, D=D_, C=C_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--arm', required=True, choices=ARMS)
    ap.add_argument('--cell', default='ETTh1_720')
    ap.add_argument('--pred_len', type=int, default=720)
    ap.add_argument('--seq_len', type=int, default=720)
    ap.add_argument('--patch_len', type=int, default=16)
    ap.add_argument('--top_k', type=int, default=TOP_K)
    ap.add_argument('--tau_t', type=float, default=0.1)
    ap.add_argument('--tau_s', type=float, default=0.1)
    ap.add_argument('--projection_std', type=float, default=PROJECTION_INIT_STD)
    ap.add_argument('--projection_seed', type=int, default=PROJECTION_INIT_SEED)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--train_epochs', type=int, default=10)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--init_seed', type=int, default=0)
    ap.add_argument('--loader_seed', type=int, default=0)
    ap.add_argument('--out_dir', default='results/TRACK-U-ASYMMETRY-CAPACITY-DECOMPOSITION01')
    ap.add_argument('--checkpoints', default='checkpoints/track_u_asymmetry_capacity_decomposition01')
    ap.add_argument('--limit_batches', type=int, default=0, help='SMOKE ONLY')
    ap.add_argument('--smoke_test', action='store_true')
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    exp, args, model = build_model(cli, device)
    channels = list(range(int(args.enc_in)))
    init_hash = state_hash(model)
    d_model = int(args.d_model)

    projection = None if cli.arm == 'U0' else SingleProjection(d_model, cli.projection_std, cli.projection_seed).to(device)
    proj_init_sha = projection_weight_sha(projection)

    for p in model.parameters():
        p.requires_grad_(True)
    params = list(model.parameters()) + (list(projection.parameters()) if projection is not None else [])
    n_proj_params = sum(p.numel() for p in projection.parameters()) if projection is not None else 0
    optimizer = torch.optim.Adam(params, lr=cli.learning_rate)

    train_gen = make_loader_generator(cli.loader_seed)
    _, train_loader = exp._get_data(flag='train', shuffle=True, generator=train_gen)
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)

    config = {'cell': cli.cell, 'arm': cli.arm, 'tau_t': cli.tau_t, 'tau_s': cli.tau_s, 'top_k': cli.top_k,
              'batch_size': cli.batch_size, 'learning_rate': cli.learning_rate, 'train_epochs': cli.train_epochs,
              'patience': cli.patience, 'init_seed': cli.init_seed, 'loader_seed': cli.loader_seed,
              'n_candidates': int(exp.memory_x.size(0)), 'n_projection_params': n_proj_params,
              'projection_init_seed': cli.projection_seed, 'projection_init_std': cli.projection_std,
              'checkpoint_criterion': 'min val retmse10 (ordinary Top-10)'}
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2))
    print(f'[track_u] arm={cli.arm} cell={cli.cell} N={config["n_candidates"]} init_hash={init_hash[:16]} '
         f'n_proj_params={n_proj_params} proj_init_sha={(proj_init_sha or "N/A")[:16]}')

    def eval_channel(batch_x, batch_y, batch_start_idx, c, want_diag=False):
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        scores = compute_scores(model, projection, batch_x, exp.memory_x, c, cli.arm)
        memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
        query_future = batch_y[:, :, c]
        u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
        d_raw = -u
        oracle_idx = stable_topk_indices(d_raw.masked_fill(~cand_mask, float('inf')), cli.top_k, largest=False)
        res = hard_eval_decomposition(scores, cand_mask, memory_c, offset_c, query_future, d_raw, oracle_idx,
                                      cli.top_k)
        spearman = spearman_batch(scores, d_raw, cand_mask) if want_diag else None
        return res, spearman

    if cli.smoke_test:
        model.train()
        for bi, (batch_x, batch_y, batch_start_idx) in enumerate(train_loader):
            if bi >= 2:
                break
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            cand_mask, _ = exp._candidate_mask(batch_start_idx)
            optimizer.zero_grad()
            for c in channels:
                scores = compute_scores(model, projection, batch_x, exp.memory_x, c, cli.arm)
                memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
                query_future = batch_y[:, :, c]
                with torch.no_grad():
                    u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
                    p_t = normalized_teacher_prob(-u, cand_mask, cli.tau_t)
                l = kl_loss(p_t, scores, cand_mask, cli.tau_s)
                (l / len(channels)).backward()
            assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters()), \
                '[ISSUE] encoder grad is zero -- full-gradient violated'
            if projection is not None:
                assert projection.W.grad is not None and projection.W.grad.abs().sum() > 0, \
                    '[ISSUE] projection grad is zero'
            optimizer.step()
        vram = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
        print(f'[track_u] {cli.arm} SMOKE PASS max_vram_mb={vram:.1f}')
        return

    epoch_val_rows, step_rows, batch_order_hashes = [], [], []
    best = {'val_retmse10': float('inf'), 'epoch': -1}
    t0 = time.time()
    global_step = 0
    for epoch in range(1, cli.train_epochs + 1):
        model.train()
        starts = []
        for bi, (batch_x, batch_y, batch_start_idx) in enumerate(train_loader):
            if cli.limit_batches and bi >= cli.limit_batches:
                break
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            starts.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx) else torch.as_tensor(batch_start_idx))
            cand_mask, _ = exp._candidate_mask(batch_start_idx)
            optimizer.zero_grad()
            batch_loss = 0.0
            for c in channels:
                scores = compute_scores(model, projection, batch_x, exp.memory_x, c, cli.arm)
                memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
                query_future = batch_y[:, :, c]
                with torch.no_grad():
                    u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
                    p_t = normalized_teacher_prob(-u, cand_mask, cli.tau_t)
                l = kl_loss(p_t, scores, cand_mask, cli.tau_s)
                (l / len(channels)).backward()
                batch_loss += float(l.detach()) / len(channels)
            optimizer.step()
            global_step += 1
            step_rows.append({'global_step': global_step, 'epoch': epoch, 'batch_in_epoch': bi,
                              'train_loss': batch_loss})
        batch_order_hashes.append({'epoch': epoch, 'batch_order_sha256': batch_order_sha256(starts)})

        model.eval()
        val_sums, n = {}, 0
        with torch.no_grad():
            for batch_x, batch_y, batch_start_idx in val_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                bsz = batch_x.size(0)
                per_ch = {}
                for c in channels:
                    res, _ = eval_channel(batch_x, batch_y, batch_start_idx, c, want_diag=False)
                    per_ch.setdefault('retmse10', []).append(res['model_ind_mse'].cpu())
                    per_ch.setdefault('agg_mse10', []).append(res['agg_mse'].cpu())
                    per_ch.setdefault('recall10', []).append(res['recall10'].cpu())
                    per_ch.setdefault('ndcg10', []).append(res['ndcg10'].cpu())
                for k_, vals in per_ch.items():
                    val_sums[k_] = val_sums.get(k_, 0.0) + torch.cat(vals).sum().item()
                n += bsz
        val_metrics = {k_: v / max(n * len(channels), 1) for k_, v in val_sums.items()}
        val_retmse10 = val_metrics['retmse10']
        val_metrics['epoch'] = epoch
        epoch_val_rows.append(val_metrics)

        payload = {'model_state_dict': model.state_dict(), 'epoch': epoch, 'val_retmse10': val_retmse10,
                  'config': config}
        if projection is not None:
            payload['projection_state_dict'] = projection.state_dict()
        torch.save(payload, ckpt_dir / f'checkpoint_epoch{epoch}.pth')
        if val_retmse10 < best['val_retmse10']:
            best = {'val_retmse10': val_retmse10, 'epoch': epoch}
            torch.save(payload, ckpt_dir / 'checkpoint.pth')
        print(f'[track_u] {cli.arm} epoch={epoch} val_retmse10={val_retmse10:.6f} val_agg={val_metrics["agg_mse10"]:.6f} '
             f'(best={best["epoch"]}:{best["val_retmse10"]:.6f})')
        if epoch - best['epoch'] >= cli.patience:
            print(f'[track_u] {cli.arm} early stop at epoch {epoch} (best={best["epoch"]})')
            break

    bl = torch.load(ckpt_dir / 'checkpoint.pth', map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    if projection is not None:
        projection.load_state_dict(bl['projection_state_dict'])
    model.eval()

    test_sums, n = {}, 0
    spearman_accum = []
    with torch.no_grad():
        for batch_x, batch_y, batch_start_idx in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            bsz = batch_x.size(0)
            per_ch = {}
            for c in channels:
                res, spearman = eval_channel(batch_x, batch_y, batch_start_idx, c, want_diag=True)
                per_ch.setdefault('retmse10', []).append(res['model_ind_mse'].cpu())
                per_ch.setdefault('agg_mse10', []).append(res['agg_mse'].cpu())
                per_ch.setdefault('recall10', []).append(res['recall10'].cpu())
                per_ch.setdefault('ndcg10', []).append(res['ndcg10'].cpu())
                per_ch.setdefault('D', []).append(res['D'].cpu())
                per_ch.setdefault('C', []).append(res['C'].cpu())
                spearman_accum.append(spearman)
            for k_, vals in per_ch.items():
                test_sums[k_] = test_sums.get(k_, 0.0) + torch.cat(vals).sum().item()
            n += bsz
    test_metrics = {k_: v / max(n * len(channels), 1) for k_, v in test_sums.items()}
    test_metrics['spearman'] = float(sum(spearman_accum) / len(spearman_accum))
    test_metrics['n_queries_seen'] = n
    test_metrics['best_epoch'] = best['epoch']

    with open(out_dir / 'train_metrics.csv', 'w', newline='') as fh:
        fieldnames = sorted({k for r in step_rows for k in r})
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in step_rows:
            w.writerow(r)
    with open(out_dir / 'val_metrics.csv', 'w', newline='') as fh:
        fieldnames = sorted({k for r in epoch_val_rows for k in r})
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in epoch_val_rows:
            w.writerow(r)
    with open(out_dir / 'batch_order_hashes.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['epoch', 'batch_order_sha256'])
        w.writeheader()
        for r in batch_order_hashes:
            w.writerow(r)
    (out_dir / 'final_test_metrics.json').write_text(json.dumps(test_metrics, indent=2, default=str))

    final_proj_sha = projection_weight_sha(projection) if projection is not None else None
    diag = {}
    if projection is not None:
        W = projection.W.detach().cpu()
        I = torch.eye(W.shape[0])
        diff = W - I
        sv = torch.linalg.svdvals(W)
        diag = {'frobenius_norm_W_minus_I': float(diff.norm()), 'singular_values': sv.tolist(),
               'condition_number': float((sv.max() / sv.min().clamp_min(EPS))),
               'effective_rank_entropy': float(torch.exp(-(sv / sv.sum().clamp_min(EPS) *
                                                           (sv / sv.sum().clamp_min(EPS)).clamp_min(EPS).log()).sum()))}
    (out_dir / 'projection_diagnostics.json').write_text(json.dumps(diag, indent=2))
    (out_dir / 'checkpoint_fingerprints.json').write_text(json.dumps(
        {'init_hash': init_hash, 'best_epoch': best['epoch'], 'final_model_hash': state_hash(model),
         'n_projection_params': n_proj_params, 'projection_init_sha256': proj_init_sha,
         'projection_final_sha256': final_proj_sha, 'batch_order_hashes': batch_order_hashes}, indent=2))
    vram = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
    print(f'[track_u] done. {cli.arm} best_epoch={best["epoch"]} test_retMSE@10={test_metrics["retmse10"]:.6f} '
         f'test_agg_mse10={test_metrics["agg_mse10"]:.6f} max_vram_mb={vram:.1f} wall_seconds={time.time()-t0:.1f}')


if __name__ == '__main__':
    main()
