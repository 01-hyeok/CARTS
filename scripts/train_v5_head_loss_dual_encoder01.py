#!/usr/bin/env python3
"""TRACK-V5-HEAD-LOSS-DUAL-ENCODER01 -- Shared(S)/Separate(D) Encoder x
{L0 Original KL, L3a Full-P100 Soft Aggregate, L3b Top-K-aligned
Aggregate} training script. Six arms total: S0/S1/S2 (shared encoder,
`model_k is model_q`) and D0/D1/D2 (separate encoders, `model_k =
deepcopy(model_q)` at init, both trained with full gradient, NO
detach/frozen-key -- reusing the exact E0/E1 construction pattern
already smoke-tested in `scripts/train_p100_dual_encoder01.py`, which
this script imports `build_model`/`compute_scores_dual` from rather
than re-deriving).

    L0  = KL(p_T || p_mix)
    L3a = L0 + lambda_A * MSE(R_soft100, Y_q),   R_soft100 = sum_i p_mix(i) V_i
    L3b = L0 + lambda_A * MSE(R_softTopK, Y_q),  R_softTopK via
          `utils.differentiable_topk01.soft_topk_mask` on `log(p_mix)`
          (a monotonic transform of p_mix, inducing the IDENTICAL
          ordering/Top-10 set as p_mix itself) -- a genuinely
          boundary-aware relaxation, NOT the repo's existing
          hard-then-renormalize `soft_topk_weights` pattern (see that
          module's own docstring and `utils/differentiable_topk01.py`'s
          docstring for the documented distinction).

Evaluation (ALL arms, identical): Mean-Mixture Top-10 inference
(`mean_mixture_topk_selection`), validation-selected checkpoint (min
val Mean-Mixture RetMSE@10 -- PRIMARY; min val Hard AggMSE / min val
Soft AggMSE saved as secondary diagnostics, never used to pick the
primary checkpoint).
"""
import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.rng_control01 import batch_order_sha256, make_loader_generator, set_global_seeds
from scripts.train_factorial_e2e01 import encode_raw, individual_utility_memsafe
from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob
from scripts.train_j_shared_encoder_drift01 import build_model, geometry_metrics, state_hash
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads, kl_loss_from_prob
from scripts.train_margutil01 import memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k
from scripts.train_retriever_pool01 import CandidatePoolCache
from utils.candidate_pool import gather_candidate_histories, gather_candidate_values, \
    encode_pooled_candidates, local_to_global, pooled_future_mse
from utils.differentiable_topk01 import soft_topk_forecast, soft_topk_mask
from utils.mean_mixture_selection import mean_mixture_topk_selection

EPS = 1e-8
TOP_K = 10
N_SLOTS = 5
ARMS = ('S0', 'S1', 'S2', 'D0', 'D1', 'D2')
LOSS_OF_ARM = {'S0': 'L0', 'S1': 'L3a', 'S2': 'L3b', 'D0': 'L0', 'D1': 'L3a', 'D2': 'L3b'}
SHARED_OF_ARM = {'S0': True, 'S1': True, 'S2': True, 'D0': False, 'D1': False, 'D2': False}


def compute_scores_dual(model_q, model_k, slot_heads, batch_x, exp, channel, pool_cache, batch_start_idx, device):
    """Shared/Separate-agnostic scorer -- reduces exactly to the
    canonical `compute_scores`'s coarse_topk branch when `model_k is
    model_q` (byte-identical computation, same call order)."""
    z_q = encode_raw(model_q, batch_x, channel)
    pool_idx_global = pool_cache.lookup(batch_start_idx, channel, device)
    pooled_x = gather_candidate_histories(exp.memory_x, pool_idx_global)
    z_k_raw = encode_pooled_candidates(lambda x, c: encode_raw(model_k, x, c), pooled_x, channel)
    z_k = F.normalize(z_k_raw, dim=-1)
    q = slot_heads(z_q)
    scores = torch.einsum('bsd,bmd->bsm', q, z_k)
    pool_valid_mask = torch.ones(pool_idx_global.shape, dtype=torch.bool, device=device)
    return scores, pool_valid_mask, pool_idx_global


def aggregate_loss(loss_name, p_mix, pooled_memory_c, offset_c, query_future, valid_mask, cli):
    """Returns (L_agg scalar, R used for diagnostics [B,H])."""
    if loss_name == 'L0':
        return torch.zeros((), device=p_mix.device), None
    y_pool = pooled_memory_c + offset_c.view(-1, 1, 1)
    if loss_name == 'L3a':
        r_soft = torch.einsum('bm,bmh->bh', p_mix, y_pool)
        return ((r_soft - query_future) ** 2).mean(), r_soft
    if loss_name == 'L3b':
        log_p_mix = torch.log(p_mix.clamp_min(EPS))
        a = soft_topk_mask(log_p_mix, valid_mask, k=cli.top_k, tau=cli.tau_topk)
        r_soft = soft_topk_forecast(a, y_pool)
        return ((r_soft - query_future) ** 2).mean(), r_soft
    raise ValueError(f'[ISSUE][ABORT] unknown loss {loss_name}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--arm', required=True, choices=ARMS)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--patch_len', type=int, default=16)
    ap.add_argument('--top_k', type=int, default=TOP_K)
    ap.add_argument('--tau_t', type=float, default=0.1)
    ap.add_argument('--tau_s', type=float, default=0.1)
    ap.add_argument('--tau_topk', type=float, default=0.05, help='L3b relaxation temperature')
    ap.add_argument('--lambda_A', type=float, default=1.0)
    ap.add_argument('--slot_std', type=float, default=1e-3)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--train_epochs', type=int, default=10)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--disable_early_stopping', action='store_true')
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--init_seed', type=int, default=0)
    ap.add_argument('--loader_seed', type=int, default=0)
    ap.add_argument('--candidate_mask', default='raft')
    ap.add_argument('--candidate_pool_size', type=int, default=100)
    ap.add_argument('--candidate_pool_cache', required=True)
    ap.add_argument('--n_probe_rank', type=int, default=256)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--checkpoints', required=True)
    ap.add_argument('--limit_batches', type=int, default=0)
    ap.add_argument('--smoke_test', action='store_true')
    ap.add_argument('--gradient_probe', action='store_true',
                    help='PHASE 0 diagnostic: measure KL/Agg grad norms+cosine+ratio on a few '
                         'batches, write gradient_probe.json, then exit (no training).')
    ap.add_argument('--n_probe_batches', type=int, default=4)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loss_name = LOSS_OF_ARM[cli.arm]
    shared = SHARED_OF_ARM[cli.arm]
    out_dir = Path(cli.out_dir) / cli.cell / cli.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints) / cli.cell / cli.arm
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    exp, args, model_q = build_model(cli, device)
    channels = list(range(int(args.enc_in)))
    d_model = int(args.d_model)
    init_hash_q = state_hash(model_q)
    if shared:
        model_k = model_q
    else:
        model_k = copy.deepcopy(model_q)
        assert state_hash(model_k) == init_hash_q, '[ISSUE][ABORT] Separate-encoder init mismatch'
        for p in model_k.parameters():
            p.requires_grad_(True)
    init_hash_k = state_hash(model_k)
    slot_heads = SlotHeads(d_model, n_slots=N_SLOTS, std=cli.slot_std).to(device)

    expected_meta = {'seq_len': cli.seq_len, 'pred_len': cli.pred_len, 'candidate_mask': cli.candidate_mask,
                     'candidate_pool_size': cli.candidate_pool_size, 'candidate_pool_metric': 'delta_last_cosine'}
    cache_dir = Path(cli.candidate_pool_cache)
    pool_caches = {}
    for split in ('train', 'val', 'test'):
        cache_path = cache_dir / f'{split}.pt'
        if not cache_path.exists():
            raise SystemExit(f'[ISSUE][ABORT] missing candidate pool cache file: {cache_path}')
        pool_caches[split] = CandidatePoolCache(cache_path, expected_meta)

    for p in model_q.parameters():
        p.requires_grad_(True)
    params = list(model_q.parameters())
    if not shared:
        params += list(model_k.parameters())
    params += list(slot_heads.parameters())
    optimizer = torch.optim.Adam(params, lr=cli.learning_rate)

    train_gen = make_loader_generator(cli.loader_seed)
    _, train_loader = exp._get_data(flag='train', shuffle=True, generator=train_gen)
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)

    config = {
        'exp': 'TRACK-V5-HEAD-LOSS-DUAL-ENCODER01', 'cell': cli.cell, 'arm': cli.arm, 'loss': loss_name,
        'shared_encoder': shared, 'lambda_A': cli.lambda_A, 'tau_topk': cli.tau_topk,
        'candidate_pool_mode': 'coarse_topk', 'candidate_pool_size': cli.candidate_pool_size,
        'candidate_pool_metric': 'delta_last_cosine', 'full_candidate_count': int(exp.memory_x.size(0)),
        'tau_t': cli.tau_t, 'tau_s': cli.tau_s, 'top_k': cli.top_k, 'batch_size': cli.batch_size,
        'learning_rate': cli.learning_rate, 'train_epochs': cli.train_epochs, 'patience': cli.patience,
        'init_seed': cli.init_seed, 'loader_seed': cli.loader_seed, 'init_hash_q': init_hash_q,
        'init_hash_k': init_hash_k,
        'checkpoint_policy': 'primary=min_val_retmse10 (Mean-Mixture); secondary=min_val_hard_aggmse / '
                             'min_val_soft_aggmse (diagnostic only, never fed to Stage2)',
    }
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2, default=str))
    print(f'[head_loss] arm={cli.arm} loss={loss_name} shared={shared} cell={cli.cell} '
         f'lambda_A={cli.lambda_A} init_hash_q={init_hash_q[:16]} init_hash_k={init_hash_k[:16]}')

    def eval_channel(batch_x, batch_y, batch_start_idx, c, cand_mask, split, want_full_oracle=False):
        scores, valid_mask, pool_idx_global = compute_scores_dual(
            model_q, model_k, slot_heads, batch_x, exp, c, pool_caches[split], batch_start_idx, device)
        memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
        query_future = batch_y[:, :, c]
        pooled_memory_c = gather_candidate_values(memory_c, pool_idx_global)
        d_pool = pooled_future_mse(pooled_memory_c, offset_c, query_future)

        mean_idx, p_heads, p_mix = mean_mixture_topk_selection(scores, valid_mask, cli.tau_s, cli.top_k)
        p_t = normalized_teacher_prob(d_pool, valid_mask, cli.tau_t)
        p_bar = p_heads.mean(dim=1)
        kl = kl_loss_from_prob(p_t, p_bar, valid_mask)
        l_agg, r_agg_used = aggregate_loss(loss_name, p_mix, pooled_memory_c, offset_c, query_future, valid_mask,
                                           cli)
        total_loss = kl + cli.lambda_A * l_agg

        y_sel_hard = pooled_memory_c.gather(1, mean_idx.unsqueeze(-1).expand(-1, -1, pooled_memory_c.size(-1))) \
            + offset_c.view(-1, 1, 1)
        ind_mse_hard = d_pool.gather(1, mean_idx)
        agg_pred_hard = y_sel_hard.mean(dim=1)
        hard_aggmse = ((agg_pred_hard - query_future) ** 2).mean(-1)
        D_ = ind_mse_hard.mean(-1) / cli.top_k
        C_ = hard_aggmse - D_
        restricted_oracle_local = stable_topk_indices(d_pool, cli.top_k, largest=False)
        recall10 = recall_at_k(mean_idx, restricted_oracle_local, cli.top_k)
        ndcg10 = ndcg_at_k(mean_idx, d_pool, valid_mask, cli.top_k)
        r_soft100 = torch.einsum('bm,bmh->bh', p_mix, pooled_memory_c + offset_c.view(-1, 1, 1))
        soft_aggmse = ((r_soft100 - query_future) ** 2).mean(-1)

        res = dict(kl=kl, l_agg=l_agg, total_loss=total_loss, model_ind_mse=ind_mse_hard.mean(-1),
                  agg_mse=hard_aggmse, D=D_, C=C_, recall10=recall10, ndcg10=ndcg10, soft_aggmse=soft_aggmse,
                  model_idx_global=local_to_global(pool_idx_global, mean_idx), pool_idx_global=pool_idx_global)
        if want_full_oracle:
            with torch.no_grad():
                u_full = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
                d_full = -u_full
                full_oracle_idx = stable_topk_indices(d_full.masked_fill(~cand_mask, float('inf')),
                                                      cli.top_k, largest=False)
                recall10_full = recall_at_k(res['model_idx_global'], full_oracle_idx, cli.top_k)
            res['recall10_full'] = recall10_full
        return res

    if cli.gradient_probe:
        model_q.train()
        if not shared:
            model_k.train()
        slot_heads.train()
        kl_norms, agg_norms, cosines, ratios = [], [], [], []
        for bi, (batch_x, batch_y, batch_start_idx) in enumerate(train_loader):
            if bi >= cli.n_probe_batches:
                break
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            cand_mask, _ = exp._candidate_mask(batch_start_idx)
            for c in channels[:1]:  # one channel per batch is enough for a gradient-scale probe
                res = eval_channel(batch_x, batch_y, batch_start_idx, c, cand_mask, 'train')
                optimizer.zero_grad()
                res['kl'].backward(retain_graph=(loss_name != 'L0'))
                g_kl = torch.cat([p.grad.detach().flatten() for p in params if p.grad is not None]).clone()
                if loss_name != 'L0':
                    optimizer.zero_grad()
                    res['l_agg'].backward()
                    g_agg = torch.cat([p.grad.detach().flatten() for p in params if p.grad is not None]).clone()
                    kl_norms.append(float(g_kl.norm())); agg_norms.append(float(g_agg.norm()))
                    cosines.append(float(F.cosine_similarity(g_kl.unsqueeze(0), g_agg.unsqueeze(0))))
                    ratios.append(float(g_agg.norm() / g_kl.norm().clamp_min(EPS)))
                else:
                    kl_norms.append(float(g_kl.norm()))
        probe = {'cell': cli.cell, 'arm': cli.arm, 'loss': loss_name,
                'mean_kl_grad_norm': sum(kl_norms) / max(len(kl_norms), 1)}
        if loss_name != 'L0':
            probe.update({'mean_agg_grad_norm': sum(agg_norms) / len(agg_norms),
                         'mean_grad_cosine_kl_vs_agg': sum(cosines) / len(cosines),
                         'mean_agg_over_kl_ratio': sum(ratios) / len(ratios)})
        (out_dir / 'gradient_probe.json').write_text(json.dumps(probe, indent=2, default=str))
        print(f'[head_loss] {cli.arm} gradient_probe DONE: {probe}')
        return

    if cli.smoke_test:
        model_q.train()
        if not shared:
            model_k.train()
        slot_heads.train()
        for bi, (batch_x, batch_y, batch_start_idx) in enumerate(train_loader):
            if bi >= 2:
                break
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            cand_mask, _ = exp._candidate_mask(batch_start_idx)
            optimizer.zero_grad()
            for c in channels:
                res = eval_channel(batch_x, batch_y, batch_start_idx, c, cand_mask, 'train')
                (res['total_loss'] / len(channels)).backward()
            assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model_q.encoder.parameters())
            if not shared:
                assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model_k.encoder.parameters())
            optimizer.step()
        vram = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
        print(f'[head_loss] {cli.arm} SMOKE PASS max_vram_mb={vram:.1f}')
        return

    epoch_val_rows, step_rows, batch_order_hashes = [], [], []
    best_retmse = {'val': float('inf'), 'epoch': -1}
    best_hard_agg = {'val': float('inf'), 'epoch': -1}
    best_soft_agg = {'val': float('inf'), 'epoch': -1}
    t0 = time.time()
    global_step = 0
    for epoch in range(1, cli.train_epochs + 1):
        model_q.train()
        if not shared:
            model_k.train()
        slot_heads.train()
        starts = []
        for bi, (batch_x, batch_y, batch_start_idx) in enumerate(train_loader):
            if cli.limit_batches and bi >= cli.limit_batches:
                break
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            starts.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx)
                         else torch.as_tensor(batch_start_idx))
            cand_mask, _ = exp._candidate_mask(batch_start_idx)
            optimizer.zero_grad()
            batch_loss = 0.0
            for c in channels:
                res = eval_channel(batch_x, batch_y, batch_start_idx, c, cand_mask, 'train')
                l = res['total_loss']
                (l / len(channels)).backward()
                batch_loss += float(l.detach()) / len(channels)
            optimizer.step()
            global_step += 1
            step_rows.append({'global_step': global_step, 'epoch': epoch, 'batch_in_epoch': bi,
                              'train_loss_total': batch_loss})
        batch_order_hashes.append({'epoch': epoch, 'batch_order_sha256': batch_order_sha256(starts)})

        model_q.eval()
        if not shared:
            model_k.eval()
        slot_heads.eval()
        val_sums, n = {}, 0
        with torch.no_grad():
            for batch_x, batch_y, batch_start_idx in val_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                cand_mask, _ = exp._candidate_mask(batch_start_idx)
                bsz = batch_x.size(0)
                per_ch = {}
                for c in channels:
                    res = eval_channel(batch_x, batch_y, batch_start_idx, c, cand_mask, 'val', want_full_oracle=True)
                    for key_ in ('kl', 'model_ind_mse', 'agg_mse', 'soft_aggmse', 'recall10', 'recall10_full',
                               'ndcg10'):
                        v = res[key_]
                        per_ch.setdefault(key_, []).append(v.reshape(-1).cpu())
                for k_, vals in per_ch.items():
                    val_sums[k_] = val_sums.get(k_, 0.0) + torch.cat(vals).sum().item()
                n += bsz
        n_eff = max(n * len(channels), 1)
        val_metrics = {k_: v / n_eff for k_, v in val_sums.items()}
        val_metrics['epoch'] = epoch
        epoch_val_rows.append(val_metrics)

        payload = {'model_q_state_dict': model_q.state_dict(),
                  'model_k_state_dict': model_k.state_dict() if not shared else None,
                  'slot_heads_state_dict': slot_heads.state_dict(), 'shared_encoder': shared,
                  'epoch': epoch, 'val_retmse10': val_metrics['model_ind_mse'],
                  'val_hard_aggmse': val_metrics['agg_mse'], 'val_soft_aggmse': val_metrics['soft_aggmse'],
                  'config': config}
        torch.save(payload, ckpt_dir / f'checkpoint_epoch{epoch}.pth')
        if val_metrics['model_ind_mse'] < best_retmse['val']:
            best_retmse = {'val': val_metrics['model_ind_mse'], 'epoch': epoch}
            torch.save(payload, ckpt_dir / 'checkpoint_best_retmse.pth')
        if val_metrics['agg_mse'] < best_hard_agg['val']:
            best_hard_agg = {'val': val_metrics['agg_mse'], 'epoch': epoch}
            torch.save(payload, ckpt_dir / 'checkpoint_best_hard_agg.pth')
        if val_metrics['soft_aggmse'] < best_soft_agg['val']:
            best_soft_agg = {'val': val_metrics['soft_aggmse'], 'epoch': epoch}
            torch.save(payload, ckpt_dir / 'checkpoint_best_soft_agg.pth')
        print(f'[head_loss] {cli.arm} epoch={epoch} val_retmse10={val_metrics["model_ind_mse"]:.6f} '
             f'val_hard_agg={val_metrics["agg_mse"]:.6f} val_soft_agg={val_metrics["soft_aggmse"]:.6f} '
             f'(best_retmse_epoch={best_retmse["epoch"]})')
        if not cli.disable_early_stopping and epoch - best_retmse['epoch'] >= cli.patience:
            print(f'[head_loss] {cli.arm} early stop at epoch {epoch}')
            break

    def run_test(ckpt_path, tag):
        bl = torch.load(ckpt_path, map_location=device)
        model_q.load_state_dict(bl['model_q_state_dict'])
        if not shared:
            model_k.load_state_dict(bl['model_k_state_dict'])
        slot_heads.load_state_dict(bl['slot_heads_state_dict'])
        model_q.eval()
        if not shared:
            model_k.eval()
        slot_heads.eval()
        test_sums, n = {}, 0
        with torch.no_grad():
            for batch_x, batch_y, batch_start_idx in test_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                cand_mask, _ = exp._candidate_mask(batch_start_idx)
                bsz = batch_x.size(0)
                per_ch = {}
                for c in channels:
                    res = eval_channel(batch_x, batch_y, batch_start_idx, c, cand_mask, 'test', want_full_oracle=True)
                    for key_ in ('kl', 'model_ind_mse', 'agg_mse', 'soft_aggmse', 'recall10', 'recall10_full',
                               'ndcg10', 'D', 'C'):
                        v = res[key_]
                        per_ch.setdefault(key_, []).append(v.reshape(-1).cpu())
                for k_, vals in per_ch.items():
                    test_sums[k_] = test_sums.get(k_, 0.0) + torch.cat(vals).sum().item()
                n += bsz
        n_eff = max(n * len(channels), 1)
        test_metrics = {k_: v / n_eff for k_, v in test_sums.items()}
        test_metrics['n_queries_seen'] = n
        test_metrics['best_epoch'] = int(bl['epoch'])
        (out_dir / f'final_test_metrics_{tag}.json').write_text(json.dumps(test_metrics, indent=2, default=str))
        print(f'[head_loss] {cli.arm} [{tag}] best_epoch={test_metrics["best_epoch"]} '
             f'test_retMSE@10={test_metrics["model_ind_mse"]:.6f} test_hard_agg={test_metrics["agg_mse"]:.6f} '
             f'test_soft_agg={test_metrics["soft_aggmse"]:.6f}')
        return test_metrics

    run_test(ckpt_dir / 'checkpoint_best_retmse.pth', 'best_retmse')
    run_test(ckpt_dir / 'checkpoint_best_hard_agg.pth', 'best_hard_agg')
    run_test(ckpt_dir / 'checkpoint_best_soft_agg.pth', 'best_soft_agg')

    with open(out_dir / 'val_metrics.csv', 'w', newline='') as fh:
        fieldnames = sorted({k for r in epoch_val_rows for k in r})
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in epoch_val_rows:
            w.writerow(r)
    (out_dir / 'checkpoint_fingerprints.json').write_text(json.dumps(
        {'init_hash_q': init_hash_q, 'init_hash_k': init_hash_k, 'best_retmse_epoch': best_retmse['epoch'],
         'best_hard_agg_epoch': best_hard_agg['epoch'], 'best_soft_agg_epoch': best_soft_agg['epoch'],
         'batch_order_hashes': batch_order_hashes}, indent=2, default=str))
    vram = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
    print(f'[head_loss] done. {cli.arm} best_retmse_epoch={best_retmse["epoch"]} max_vram_mb={vram:.1f} '
         f'wall_seconds={time.time()-t0:.1f}')


if __name__ == '__main__':
    main()
