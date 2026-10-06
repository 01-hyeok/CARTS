#!/usr/bin/env python3
"""Unified Original-KL / Multi-Query retriever trainer with a
switchable candidate universe (spec: candidate-pool refactor).

    --candidate_pool_mode full         exact legacy full-memory behavior
    --candidate_pool_mode coarse_topk  learned encoder only ever sees M
                                       candidates per query/channel,
                                       pre-filtered by a cheap, future
                                       -blind, parameter-free delta-last
                                       -cosine score (utils/candidate_pool.py)

    --num_query_views 0   TRUE Original KL, zero projection parameters
    --num_query_views 1/2/5   SlotHeads(n_slots=num_query_views), exactly
                              TRACK-V's own V1/V2/V5 convention (bit
                              -identical init across view counts, reused
                              unmodified from train_k_multislot_predictive_retrieval01)

Reused UNMODIFIED (never reimplemented): `build_model`/`state_hash`
(`train_j_shared_encoder_drift01`), `encode_raw`/`arm_score`/
`individual_utility_memsafe` (`train_factorial_e2e01`),
`normalized_teacher_prob` (`train_horizon_retrieval_expert01`),
`memory_value` (`train_margutil01`), `stable_topk_indices`
(`RelationStage1`), `recall_at_k`/`ndcg_at_k` (`train_patch_retrieval_expert01`),
`make_loader_generator`/`batch_order_sha256`/`set_global_seeds`
(`rng_control01`), `SlotHeads`/`kl_loss_from_prob`
(`train_k_multislot_predictive_retrieval01`), `round_robin_topk_selection`/
`hard_eval_decomposition`/`spearman_batch`/`slot_mechanism_diagnostics`
(`train_t_pure_multislot01`), and every `utils/candidate_pool.py`
primitive for the coarse-pool path.

Historical scripts (`train_j_shared_encoder_drift01.py`,
`train_t_pure_multislot01.py`, `build_t_multislot_cache01.py`,
`build_t2_true_original_kl_cache01.py`) are NOT modified or used as a
base for this file beyond importing their already-reusable functions --
this is a clean, new entry point.

Checkpoint/early-stop criterion is ALWAYS min validation KL (spec
section 16) -- the training objective itself, never a TYPE C retrieval
diagnostic.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.rng_control01 import batch_order_sha256, make_loader_generator, set_global_seeds
from scripts.train_factorial_e2e01 import arm_score, encode_raw, individual_utility_memsafe
from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob
from scripts.train_j_shared_encoder_drift01 import build_model, state_hash
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads, kl_loss_from_prob
from scripts.train_margutil01 import memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k
from scripts.train_t_pure_multislot01 import (
    hard_eval_decomposition, round_robin_topk_selection, slot_mechanism_diagnostics,
    spearman_batch,
)
from utils.candidate_pool import (
    CandidatePoolConfig, build_coarse_pool, compute_coarse_delta_last_cosine_scores,
    encode_pooled_candidates, gather_candidate_histories, gather_candidate_values,
    local_to_global, pooled_future_mse,
)

EPS = 1e-8
TOP_K = 10
VALID_NUM_QUERY_VIEWS = (0, 1, 2, 5)


class CandidatePoolCache:
    """Loads one split's precomputed pool (`precompute_candidate_pool01.py`
    output) and looks up the right rows for an arbitrary batch via its
    `query_start_idx`. Aborts fast (never silently reuses) on any
    config-fingerprint mismatch."""

    def __init__(self, path, expected_meta):
        payload = torch.load(path, map_location='cpu')
        self.query_start_idx = payload['query_start_idx']
        self.pool_idx_global = payload['pool_idx_global'].long()
        self.meta = payload['meta']
        for key in ('seq_len', 'pred_len', 'candidate_mask', 'candidate_pool_size',
                   'candidate_pool_metric'):
            if key in expected_meta and self.meta.get(key) != expected_meta[key]:
                raise ValueError(
                    f'[ISSUE][ABORT] candidate pool cache fingerprint mismatch at {path}: '
                    f'{key}={self.meta.get(key)!r} != expected {expected_meta[key]!r}')
        self._lut = {int(s): i for i, s in enumerate(self.query_start_idx.tolist())}

    def lookup(self, batch_start_idx, channel, device):
        rows = [self._lut[int(s)] for s in batch_start_idx.tolist()]
        return self.pool_idx_global[rows, channel, :].to(device)


def state_hash_module(module):
    h = hashlib.sha256()
    for k in sorted(module.state_dict()):
        h.update(k.encode())
        h.update(module.state_dict()[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def compute_scores(pool_cfg, model, slot_heads, batch_x, exp, channel, cand_mask, pool_cache,
                   batch_start_idx, device):
    """Returns (scores [B,S,K], valid_mask [B,K], pool_idx_global or None).
    K=N in full mode, K=M in coarse_topk mode. `pool_idx_global` is None
    in full mode (there is no local<->global distinction to make)."""
    z_q = encode_raw(model, batch_x, channel)
    if pool_cfg.mode == 'full':
        k_raw = encode_raw(model, exp.memory_x, channel)
        k_full = F.normalize(k_raw, dim=-1)
        if slot_heads is None:
            scores = arm_score(z_q, k_raw, None).unsqueeze(1)
        else:
            q = slot_heads(z_q)
            scores = torch.einsum('bsd,nd->bsn', q, k_full)
        return scores, cand_mask, None

    # coarse_topk
    pool_idx_global = pool_cache.lookup(batch_start_idx, channel, device)  # [B, M]
    pooled_x = gather_candidate_histories(exp.memory_x, pool_idx_global)    # [B, M, L, C]
    z_k = encode_pooled_candidates(lambda x, c: encode_raw(model, x, c), pooled_x, channel)
    z_k = F.normalize(z_k, dim=-1)
    if slot_heads is None:
        scores = torch.einsum('bd,bmd->bm', F.normalize(z_q, dim=-1), z_k).unsqueeze(1)
    else:
        q = slot_heads(z_q)
        scores = torch.einsum('bsd,bmd->bsm', q, z_k)
    pool_valid_mask = torch.ones(pool_idx_global.shape, dtype=torch.bool, device=device)
    return scores, pool_valid_mask, pool_idx_global


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--num_query_views', type=int, required=True, choices=VALID_NUM_QUERY_VIEWS)
    ap.add_argument('--cell', default='ETTh1_720')
    ap.add_argument('--pred_len', type=int, default=720)
    ap.add_argument('--seq_len', type=int, default=720)
    ap.add_argument('--patch_len', type=int, default=16)
    ap.add_argument('--top_k', type=int, default=TOP_K)
    ap.add_argument('--tau_t', type=float, default=0.1)
    ap.add_argument('--tau_s', type=float, default=0.1)
    ap.add_argument('--slot_std', type=float, default=1e-3)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--train_epochs', type=int, default=10)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--init_seed', type=int, default=0)
    ap.add_argument('--loader_seed', type=int, default=0)
    ap.add_argument('--candidate_mask', default='raft')
    ap.add_argument('--candidate_pool_mode', default='full', choices=('full', 'coarse_topk'))
    ap.add_argument('--candidate_pool_size', type=int, default=100)
    ap.add_argument('--candidate_pool_metric', default='delta_last_cosine', choices=('delta_last_cosine',))
    ap.add_argument('--candidate_pool_cache', default=None,
                    help='directory with train.pt/val.pt/test.pt from precompute_candidate_pool01.py '
                         '-- REQUIRED if --candidate_pool_mode coarse_topk')
    ap.add_argument('--out_dir', default='results/CANDIDATE-POOL01')
    ap.add_argument('--checkpoints', default='checkpoints/candidate_pool01')
    ap.add_argument('--limit_batches', type=int, default=0, help='SMOKE/DEBUG ONLY')
    ap.add_argument('--smoke_test', action='store_true')
    cli = ap.parse_args()

    if cli.candidate_pool_size <= cli.top_k:
        raise ValueError(f'[ISSUE][ABORT] --candidate_pool_size ({cli.candidate_pool_size}) must be '
                         f'> --top_k ({cli.top_k})')

    pool_cfg = CandidatePoolConfig(mode=cli.candidate_pool_mode, size=cli.candidate_pool_size,
                                   metric=cli.candidate_pool_metric)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    arm = f'V{cli.num_query_views}'
    out_dir = Path(cli.out_dir) / cli.cell / arm
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints) / cli.cell / arm
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    exp, args, model = build_model(cli, device)
    channels = list(range(int(args.enc_in)))
    init_hash = state_hash(model)
    d_model = int(args.d_model)

    slot_heads = None
    if cli.num_query_views >= 1:
        slot_heads = SlotHeads(d_model, n_slots=cli.num_query_views, std=cli.slot_std).to(device)

    pool_caches = {}
    if pool_cfg.mode == 'coarse_topk':
        if not cli.candidate_pool_cache:
            raise SystemExit(
                '[ISSUE][ABORT] --candidate_pool_mode coarse_topk requires --candidate_pool_cache '
                '(a directory produced by scripts/precompute_candidate_pool01.py). Refusing to '
                'silently start an expensive live precompute inside the trainer -- run that script '
                'first.')
        expected_meta = {'seq_len': cli.seq_len, 'pred_len': cli.pred_len,
                         'candidate_mask': cli.candidate_mask,
                         'candidate_pool_size': cli.candidate_pool_size,
                         'candidate_pool_metric': cli.candidate_pool_metric}
        cache_dir = Path(cli.candidate_pool_cache)
        for split in ('train', 'val', 'test'):
            cache_path = cache_dir / f'{split}.pt'
            if not cache_path.exists():
                raise SystemExit(f'[ISSUE][ABORT] missing candidate pool cache file: {cache_path}')
            pool_caches[split] = CandidatePoolCache(cache_path, expected_meta)

    for p in model.parameters():
        p.requires_grad_(True)
    params = list(model.parameters()) + (list(slot_heads.parameters()) if slot_heads else [])
    optimizer = torch.optim.Adam(params, lr=cli.learning_rate)

    train_gen = make_loader_generator(cli.loader_seed)
    _, train_loader = exp._get_data(flag='train', shuffle=True, generator=train_gen)
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)

    config = {
        'exp': 'CANDIDATE-POOL01', 'cell': cli.cell, 'arm': arm, 'num_query_views': cli.num_query_views,
        'candidate_pool_mode': pool_cfg.mode, 'candidate_pool_size': pool_cfg.size,
        'candidate_pool_metric': pool_cfg.metric,
        'full_candidate_count': int(exp.memory_x.size(0)),
        'learned_candidates_encoded_per_query': (int(exp.memory_x.size(0)) if pool_cfg.mode == 'full'
                                                 else pool_cfg.size),
        'tau_t': cli.tau_t, 'tau_s': cli.tau_s, 'top_k': cli.top_k, 'batch_size': cli.batch_size,
        'learning_rate': cli.learning_rate, 'train_epochs': cli.train_epochs, 'patience': cli.patience,
        'init_seed': cli.init_seed, 'loader_seed': cli.loader_seed,
        'checkpoint_criterion': 'min_val_kl',
        'init_hash': init_hash,
    }
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2, default=str))
    print(f'[retriever_pool] arm={arm} cell={cli.cell} pool_mode={pool_cfg.mode} '
         f'pool_size={pool_cfg.size if pool_cfg.mode == "coarse_topk" else "N/A"} '
         f'N={config["full_candidate_count"]} init_hash={init_hash[:16]}')
    print('[MODEL-SELECTION AUDIT]')
    print('Training objective:          KL(p_T || mean_m softmax(s_m/tau_s))')
    print('Validation selection metric:  val_kl (SAME as training objective)')
    print('Early-stopping metric:        val_kl (SAME as selection metric)')
    print('Are they identical?          YES')
    print('Decision:                     APPROVED (TYPE A/B match; no TYPE C diagnostic metric used for selection)')

    def eval_channel(batch_x, batch_y, batch_start_idx, c, cand_mask, split, want_diag=False):
        scores, valid_mask, pool_idx_global = compute_scores(
            pool_cfg, model, slot_heads, batch_x, exp, c, cand_mask,
            pool_caches.get(split), batch_start_idx, device)
        memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
        query_future = batch_y[:, :, c]

        if pool_cfg.mode == 'full':
            u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
            d_raw = -u
            oracle_idx = stable_topk_indices(d_raw.masked_fill(~valid_mask, float('inf')), cli.top_k, largest=False)
            res = hard_eval_decomposition(scores, valid_mask, memory_c, offset_c, query_future, d_raw,
                                          oracle_idx, cli.top_k)
        else:
            pooled_memory_c = gather_candidate_values(memory_c, pool_idx_global)  # [B,M,H]
            d_pool = pooled_future_mse(pooled_memory_c, offset_c, query_future)
            restricted_oracle_local = stable_topk_indices(d_pool, cli.top_k, largest=False)
            model_idx_local = round_robin_topk_selection(scores, valid_mask, k=cli.top_k)
            model_idx_global = local_to_global(pool_idx_global, model_idx_local)
            restricted_oracle_global = local_to_global(pool_idx_global, restricted_oracle_local)
            y_sel = pooled_memory_c.gather(
                1, model_idx_local.unsqueeze(-1).expand(-1, -1, pooled_memory_c.size(-1))
            ) + offset_c.view(-1, 1, 1)  # BUGFIX: offset must be added here too (hard_eval_decomposition's
            # own `y_sel = memory_c[model_idx] + offset_c.view(-1,1,1)` convention) -- d_pool already bakes
            # it in internally via pooled_future_mse, but agg_pred/agg_mse/C below are built from y_sel
            # directly, so omitting it here silently corrupted every coarse_topk Agg/C value (caught by
            # comparing this trainer's test agg_mse against build_retrieval_cache_pool01.py's own
            # independently-computed agg_mse on the same checkpoint -- they must match and didn't before this fix).
            individual_mse_i = d_pool.gather(1, model_idx_local)
            oracle_ind_mse = d_pool.gather(1, restricted_oracle_local)
            D_ = individual_mse_i.mean(-1) / cli.top_k
            agg_pred = y_sel.mean(dim=1)
            agg_mse = ((agg_pred - query_future) ** 2).mean(-1)
            C_ = agg_mse - D_
            recall10 = recall_at_k(model_idx_local, restricted_oracle_local, cli.top_k)
            ndcg10 = ndcg_at_k(model_idx_local, d_pool, valid_mask, cli.top_k)
            res = dict(model_idx=model_idx_local, model_ind_mse=individual_mse_i.mean(-1),
                      oracle_ind_mse=oracle_ind_mse.mean(-1), recall10=recall10, ndcg10=ndcg10,
                      agg_mse=agg_mse, D=D_, C=C_)
            if want_diag:
                res['_model_idx_global'] = model_idx_global
                res['_restricted_oracle_global'] = restricted_oracle_global
                res['_pool_idx_global'] = pool_idx_global

        teacher_d = d_pool if pool_cfg.mode == 'coarse_topk' else d_raw
        with torch.no_grad():
            # teacher has no learnable params either way; kl_loss_from_prob
            # also .detach()'s p_t itself, so this no_grad is for efficiency
            # only -- it must NOT be allowed to also swallow the STUDENT
            # (scores/p_m/p_bar) side, or training silently gets zero gradient.
            p_t = normalized_teacher_prob(-teacher_d, valid_mask, cli.tau_t)
        s_masked = scores.masked_fill(~valid_mask.unsqueeze(1), float('-inf'))
        p_m = torch.softmax(s_masked / cli.tau_s, dim=-1)  # [B, S, K] -- per-slot, NOT averaged
        p_bar = p_m.mean(dim=1)
        res['kl'] = kl_loss_from_prob(p_t, p_bar, valid_mask)

        diag = None
        if want_diag:
            spearman = spearman_batch(scores.mean(dim=1), teacher_d, valid_mask)
            mech = slot_mechanism_diagnostics(scores, valid_mask, p_m, cli.tau_s)
            diag = {'spearman': spearman, **mech}
        return res, diag

    if cli.smoke_test:
        model.train()
        if slot_heads is not None:
            slot_heads.train()
        for bi, (batch_x, batch_y, batch_start_idx) in enumerate(train_loader):
            if bi >= 2:
                break
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            cand_mask, _ = exp._candidate_mask(batch_start_idx)
            optimizer.zero_grad()
            for c in channels:
                res, _ = eval_channel(batch_x, batch_y, batch_start_idx, c, cand_mask, 'train', want_diag=False)
                l = res['kl']
                (l / len(channels)).backward()
            assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters()), \
                '[ISSUE] candidate/query encoder grad is zero'
            optimizer.step()
        vram = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
        print(f'[retriever_pool] {arm} SMOKE PASS max_vram_mb={vram:.1f}')
        return

    epoch_val_rows, step_rows, batch_order_hashes = [], [], []
    best = {'val_kl': float('inf'), 'epoch': -1}
    t0 = time.time()
    global_step = 0
    for epoch in range(1, cli.train_epochs + 1):
        model.train()
        if slot_heads is not None:
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
                res, _ = eval_channel(batch_x, batch_y, batch_start_idx, c, cand_mask, 'train', want_diag=False)
                l = res['kl']
                (l / len(channels)).backward()
                batch_loss += float(l.detach()) / len(channels)
            optimizer.step()
            global_step += 1
            step_rows.append({'global_step': global_step, 'epoch': epoch, 'batch_in_epoch': bi,
                              'train_loss_kl': batch_loss})
        batch_order_hashes.append({'epoch': epoch, 'batch_order_sha256': batch_order_sha256(starts)})

        model.eval()
        if slot_heads is not None:
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
                    res, _ = eval_channel(batch_x, batch_y, batch_start_idx, c, cand_mask, 'val', want_diag=False)
                    for key_ in ('kl', 'model_ind_mse', 'agg_mse', 'recall10', 'ndcg10'):
                        v = res[key_]
                        per_ch.setdefault(key_, []).append(v.reshape(-1).cpu() if torch.is_tensor(v)
                                                           else torch.tensor([float(v)]))
                for k_, vals in per_ch.items():
                    val_sums[k_] = val_sums.get(k_, 0.0) + torch.cat(vals).sum().item()
                n += bsz
        n_eff = max(n * len(channels), 1)
        val_metrics = {k_: v / n_eff for k_, v in val_sums.items()}
        val_kl = val_metrics['kl']
        val_metrics['epoch'] = epoch
        epoch_val_rows.append(val_metrics)

        payload = {'model_state_dict': model.state_dict(),
                  'slot_heads_state_dict': slot_heads.state_dict() if slot_heads else None,
                  'epoch': epoch, 'val_kl': val_kl, 'config': config}
        torch.save(payload, ckpt_dir / f'checkpoint_epoch{epoch}.pth')
        if val_kl < best['val_kl']:
            best = {'val_kl': val_kl, 'epoch': epoch}
            torch.save(payload, ckpt_dir / 'checkpoint.pth')
        print(f'[retriever_pool] {arm} epoch={epoch} val_kl={val_kl:.6f} '
             f'val_retmse10={val_metrics["model_ind_mse"]:.6f} (best={best["epoch"]}:{best["val_kl"]:.6f})')
        if epoch - best['epoch'] >= cli.patience:
            print(f'[retriever_pool] {arm} early stop at epoch {epoch} (best={best["epoch"]})')
            break

    bl = torch.load(ckpt_dir / 'checkpoint.pth', map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    if slot_heads is not None:
        slot_heads.load_state_dict(bl['slot_heads_state_dict'])
    model.eval()
    if slot_heads is not None:
        slot_heads.eval()

    test_sums, n = {}, 0
    diag_accum = {}
    full_oracle_overlap_sum, full_oracle_overlap_n = 0.0, 0
    full_oracle_retmse_sum, full_oracle_retmse_n = 0.0, 0
    with torch.no_grad():
        for batch_x, batch_y, batch_start_idx in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            cand_mask, _ = exp._candidate_mask(batch_start_idx)
            bsz = batch_x.size(0)
            per_ch = {}
            for c in channels:
                res, diag = eval_channel(batch_x, batch_y, batch_start_idx, c, cand_mask, 'test', want_diag=True)
                for key_ in ('kl', 'model_ind_mse', 'agg_mse', 'recall10', 'ndcg10', 'D', 'C'):
                    v = res[key_]
                    per_ch.setdefault(key_, []).append(v.reshape(-1).cpu() if torch.is_tensor(v)
                                                       else torch.tensor([float(v)]))
                if pool_cfg.mode == 'coarse_topk':
                    per_ch.setdefault('oracle_ind_mse', []).append(res['oracle_ind_mse'].reshape(-1).cpu())
                for k_, v in diag.items():
                    diag_accum.setdefault(k_, []).append(v)
                if pool_cfg.mode == 'coarse_topk':
                    memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
                    query_future = batch_y[:, :, c]
                    u_full = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
                    d_full = -u_full
                    full_oracle_idx = stable_topk_indices(d_full.masked_fill(~cand_mask, float('inf')),
                                                          cli.top_k, largest=False)
                    pool_idx_global = res['_pool_idx_global']
                    overlap = (full_oracle_idx.unsqueeze(-1) == pool_idx_global.unsqueeze(1)).any(-1).float().sum(-1)
                    full_oracle_overlap_sum += float((overlap / cli.top_k).sum())
                    full_oracle_overlap_n += bsz
                    full_oracle_retmse_sum += float(d_full.gather(1, full_oracle_idx).mean(-1).sum())
                    full_oracle_retmse_n += bsz
            for k_, vals in per_ch.items():
                test_sums[k_] = test_sums.get(k_, 0.0) + torch.cat(vals).sum().item()
            n += bsz
    n_eff = max(n * len(channels), 1)
    test_metrics = {k_: v / n_eff for k_, v in test_sums.items()}
    test_metrics['n_queries_seen'] = n
    test_metrics['best_epoch'] = best['epoch']
    if pool_cfg.mode == 'coarse_topk':
        restricted_oracle_retmse = test_metrics.get('oracle_ind_mse', float('nan'))
        full_oracle_retmse = full_oracle_retmse_sum / max(full_oracle_retmse_n, 1)
        test_metrics['full_oracle_coverage_at_pool'] = full_oracle_overlap_sum / max(full_oracle_overlap_n, 1)
        test_metrics['restricted_oracle_retMSE'] = restricted_oracle_retmse
        test_metrics['full_oracle_retMSE'] = full_oracle_retmse
        test_metrics['restricted_oracle_gap'] = restricted_oracle_retmse - full_oracle_retmse
    slot_mech_metrics = {k_: (sum(v) / len(v) if v else float('nan'))
                         for k_, v in diag_accum.items() if not k_.startswith('_')}

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
    (out_dir / 'slot_mechanism_metrics.json').write_text(json.dumps(slot_mech_metrics, indent=2, default=str))
    (out_dir / 'checkpoint_fingerprints.json').write_text(json.dumps(
        {'init_hash': init_hash, 'best_epoch': best['epoch'], 'final_model_hash': state_hash(model),
         'batch_order_hashes': batch_order_hashes}, indent=2))
    vram = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
    print(f'[retriever_pool] done. {arm} best_epoch={best["epoch"]} test_KL={test_metrics["kl"]:.6f} '
         f'test_retMSE@10={test_metrics["model_ind_mse"]:.6f} max_vram_mb={vram:.1f} '
         f'wall_seconds={time.time()-t0:.1f}')


if __name__ == '__main__':
    main()
