#!/usr/bin/env python3
"""Shared-Top-100 TRACK-V re-run (V0/V1/V2/V5), per the user's explicit
spec: all retrieval arms share ONE cached, future-blind, encoder-free
Top-100 candidate pool per setting (never recomputed per-arm, never
re-derived from a learned encoder), and EVERY arm saves TWO
checkpoints from the SAME single training trajectory:

    checkpoint_best_retmse.pth   PRIMARY -- min val retMSE@10 (restricted
                                 to the P100 pool). Stage-2 uses ONLY this
                                 one. Chosen here, by explicit user
                                 instruction, specifically for direct
                                 comparability with the EXISTING Full
                                 -memory TRACK-V results (which were
                                 themselves retMSE-selected) -- NOT a
                                 reversion of the project's general
                                 checkpoint-selection governance policy
                                 (that policy's own 4-pronged justification
                                 is satisfied here: (A) literature/
                                 precedent -- matches this project's own
                                 established TRACK-V convention exactly;
                                 (B) task-alignment -- the explicit goal
                                 IS apples-to-apples comparability with
                                 the prior retMSE-selected results, not an
                                 independent claim about objective
                                 -optimality; (C) fairness -- applied
                                 identically to every arm; (D) non
                                 -circularity -- retMSE and KL are
                                 different quantities). See AUDIT.md.
    checkpoint_best_kl.pth       SECONDARY -- min val KL, diagnostic only,
                                 NEVER fed into Stage-2 here.

Reused UNMODIFIED from `scripts/train_retriever_pool01.py` (today's
candidate-pool refactor, itself built on `build_model`/`encode_raw`/
`SlotHeads`/etc. -- see that file's own docstring for the full reuse
chain): `compute_scores`, `CandidatePoolCache`. Reused from
`train_j_shared_encoder_drift01.py`: `build_model`, `state_hash`,
`geometry_metrics`/`effective_rank_entropy` (for the effective-rank
diagnostic, computed on a fixed deterministic candidate-embedding
subset exactly as that script already does it).
"""
import argparse
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
from scripts.rng_control01 import batch_order_sha256, make_loader_generator
from scripts.train_factorial_e2e01 import encode_raw, individual_utility_memsafe
from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob
from scripts.train_j_shared_encoder_drift01 import build_model, geometry_metrics, state_hash
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads, kl_loss_from_prob
from scripts.train_margutil01 import memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k
from scripts.train_retriever_pool01 import CandidatePoolCache, compute_scores
from scripts.train_t_pure_multislot01 import (
    hard_eval_decomposition, round_robin_topk_selection, slot_mechanism_diagnostics,
    spearman_batch,
)
from utils.candidate_pool import CandidatePoolConfig, gather_candidate_values, local_to_global, pooled_future_mse

EPS = 1e-8
TOP_K = 10
VALID_NUM_QUERY_VIEWS = (0, 1, 2, 5)


def _entropy_and_effective(p, valid_mask):
    """p: [B, N_pool] probability distribution over the (all-valid-by
    -construction, per `build_coarse_pool`) pool. `valid_mask` unused --
    kept in the signature for call-site symmetry/readability."""
    p_safe = p.clamp_min(EPS)
    ent = -(p_safe * p_safe.log()).sum(-1)
    return float(ent.mean()), float(ent.mean().exp())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--num_query_views', type=int, required=True, choices=VALID_NUM_QUERY_VIEWS)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--patch_len', type=int, default=16)
    ap.add_argument('--top_k', type=int, default=TOP_K)
    ap.add_argument('--tau_t', type=float, default=0.1)
    ap.add_argument('--tau_s', type=float, default=0.1)
    ap.add_argument('--slot_std', type=float, default=1e-3)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--train_epochs', type=int, default=10)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--disable_early_stopping', action='store_true',
                    help='Run all --train_epochs epochs regardless of val_retmse10 plateau; '
                         'every epoch checkpoint is still saved either way (standing fixed-epoch policy).')
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--init_seed', type=int, default=0)
    ap.add_argument('--loader_seed', type=int, default=0)
    ap.add_argument('--candidate_mask', default='raft')
    ap.add_argument('--candidate_pool_size', type=int, default=100)
    ap.add_argument('--candidate_pool_cache', required=True,
                    help='directory with train.pt/val.pt/test.pt from precompute_candidate_pool01.py')
    ap.add_argument('--n_probe_rank', type=int, default=256,
                    help='fixed deterministic candidate subset size for the effective-rank diagnostic')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--checkpoints', required=True)
    ap.add_argument('--limit_batches', type=int, default=0, help='SMOKE/DEBUG ONLY')
    ap.add_argument('--smoke_test', action='store_true')
    cli = ap.parse_args()

    if cli.candidate_pool_size <= cli.top_k:
        raise ValueError(f'[ISSUE][ABORT] --candidate_pool_size ({cli.candidate_pool_size}) must be '
                         f'> --top_k ({cli.top_k})')

    pool_cfg = CandidatePoolConfig(mode='coarse_topk', size=cli.candidate_pool_size)
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

    expected_meta = {'seq_len': cli.seq_len, 'pred_len': cli.pred_len, 'candidate_mask': cli.candidate_mask,
                     'candidate_pool_size': cli.candidate_pool_size, 'candidate_pool_metric': 'delta_last_cosine'}
    cache_dir = Path(cli.candidate_pool_cache)
    pool_caches = {}
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
        'exp': 'TRACK-V-SHARED-TOP100-01', 'cell': cli.cell, 'arm': arm,
        'num_query_views': cli.num_query_views, 'candidate_pool_mode': 'coarse_topk',
        'candidate_pool_size': cli.candidate_pool_size, 'candidate_pool_metric': 'delta_last_cosine',
        'full_candidate_count': int(exp.memory_x.size(0)),
        'tau_t': cli.tau_t, 'tau_s': cli.tau_s, 'top_k': cli.top_k, 'batch_size': cli.batch_size,
        'learning_rate': cli.learning_rate, 'train_epochs': cli.train_epochs, 'patience': cli.patience,
        'init_seed': cli.init_seed, 'loader_seed': cli.loader_seed, 'init_hash': init_hash,
        'checkpoint_policy': 'DUAL -- primary=min_val_retmse10 (Stage-2 uses ONLY this one, for direct '
                             'comparability with existing Full-memory TRACK-V results), '
                             'secondary=min_val_kl (diagnostic only, never fed to Stage-2)',
    }
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2, default=str))
    print(f'[v_p100] arm={arm} cell={cli.cell} N={config["full_candidate_count"]} pool_size={cli.candidate_pool_size} '
         f'init_hash={init_hash[:16]}')
    print('[MODEL-SELECTION AUDIT]')
    print('Training objective:           KL(p_T || mean_m softmax(s_m/tau_s)), support=P100')
    print('Validation PRIMARY selection:  val_retmse10 (TYPE C -- EXPLICIT, documented override: see docstring)')
    print('Validation SECONDARY (diag):  val_kl (SAME as training objective; saved, never used for Stage-2)')
    print('Are primary/training objective identical?  NO (deliberate, justified -- see docstring/AUDIT.md)')
    print('Decision:                      APPROVED-WITH-JUSTIFICATION (explicit user instruction for '
         'comparability with prior Full-TRACK-V retMSE-selected results; 4-pronged check satisfied)')

    def eval_channel(batch_x, batch_y, batch_start_idx, c, cand_mask, split, want_full_oracle=False):
        scores, valid_mask, pool_idx_global = compute_scores(
            pool_cfg, model, slot_heads, batch_x, exp, c, cand_mask, pool_caches[split], batch_start_idx, device)
        memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
        query_future = batch_y[:, :, c]

        pooled_memory_c = gather_candidate_values(memory_c, pool_idx_global)
        d_pool = pooled_future_mse(pooled_memory_c, offset_c, query_future)
        restricted_oracle_local = stable_topk_indices(d_pool, cli.top_k, largest=False)
        model_idx_local = round_robin_topk_selection(scores, valid_mask, k=cli.top_k)
        y_sel = pooled_memory_c.gather(
            1, model_idx_local.unsqueeze(-1).expand(-1, -1, pooled_memory_c.size(-1))
        ) + offset_c.view(-1, 1, 1)
        individual_mse_i = d_pool.gather(1, model_idx_local)
        oracle_ind_mse = d_pool.gather(1, restricted_oracle_local)
        D_ = individual_mse_i.mean(-1) / cli.top_k
        agg_pred = y_sel.mean(dim=1)
        agg_mse = ((agg_pred - query_future) ** 2).mean(-1)
        C_ = agg_mse - D_
        recall10_p100 = recall_at_k(model_idx_local, restricted_oracle_local, cli.top_k)
        ndcg10 = ndcg_at_k(model_idx_local, d_pool, valid_mask, cli.top_k)

        teacher_ent, teacher_eff = _entropy_and_effective(
            normalized_teacher_prob(d_pool, valid_mask, cli.tau_t), valid_mask)
        s_masked = scores.masked_fill(~valid_mask.unsqueeze(1), float('-inf'))
        p_m = torch.softmax(s_masked / cli.tau_s, dim=-1)
        p_bar = p_m.mean(dim=1)
        student_ent, student_eff = _entropy_and_effective(p_bar, valid_mask)
        p_t = normalized_teacher_prob(d_pool, valid_mask, cli.tau_t)
        kl = kl_loss_from_prob(p_t, p_bar, valid_mask)

        res = dict(model_idx_global=local_to_global(pool_idx_global, model_idx_local),
                  model_ind_mse=individual_mse_i.mean(-1), oracle_ind_mse=oracle_ind_mse.mean(-1),
                  recall10_p100=recall10_p100, ndcg10=ndcg10, agg_mse=agg_mse, D=D_, C=C_, kl=kl,
                  teacher_entropy=teacher_ent, teacher_effective_candidates=teacher_eff,
                  student_entropy=student_ent, student_effective_candidates=student_eff,
                  pool_idx_global=pool_idx_global)

        if want_full_oracle:
            with torch.no_grad():
                u_full = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
                d_full = -u_full
                full_oracle_idx = stable_topk_indices(d_full.masked_fill(~cand_mask, float('inf')),
                                                      cli.top_k, largest=False)
                recall10_full = recall_at_k(res['model_idx_global'], full_oracle_idx, cli.top_k)
                overlap = (full_oracle_idx.unsqueeze(-1) == pool_idx_global.unsqueeze(1)).any(-1).float().sum(-1)
                coverage = overlap / cli.top_k
            res['recall10_full'] = recall10_full
            res['coverage_at_pool'] = coverage
        return res

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
                res = eval_channel(batch_x, batch_y, batch_start_idx, c, cand_mask, 'train')
                (res['kl'] / len(channels)).backward()
            assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
            optimizer.step()
        vram = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
        print(f'[v_p100] {arm} SMOKE PASS max_vram_mb={vram:.1f}')
        return

    epoch_val_rows, step_rows, batch_order_hashes = [], [], []
    best_retmse = {'val': float('inf'), 'epoch': -1}
    best_kl = {'val': float('inf'), 'epoch': -1}
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
                res = eval_channel(batch_x, batch_y, batch_start_idx, c, cand_mask, 'train')
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
                    res = eval_channel(batch_x, batch_y, batch_start_idx, c, cand_mask, 'val',
                                       want_full_oracle=True)
                    for key_ in ('kl', 'model_ind_mse', 'agg_mse', 'recall10_p100', 'recall10_full',
                               'ndcg10', 'coverage_at_pool', 'teacher_entropy', 'teacher_effective_candidates',
                               'student_entropy', 'student_effective_candidates'):
                        v = res[key_]
                        per_ch.setdefault(key_, []).append(v.reshape(-1).cpu() if torch.is_tensor(v)
                                                           else torch.tensor([float(v)]))
                for k_, vals in per_ch.items():
                    val_sums[k_] = val_sums.get(k_, 0.0) + torch.cat(vals).sum().item()
                n += bsz
        n_eff = max(n * len(channels), 1)
        val_metrics = {k_: v / n_eff for k_, v in val_sums.items()}
        val_kl = val_metrics['kl']
        val_retmse10 = val_metrics['model_ind_mse']
        val_metrics['epoch'] = epoch
        epoch_val_rows.append(val_metrics)

        payload = {'model_state_dict': model.state_dict(),
                  'slot_heads_state_dict': slot_heads.state_dict() if slot_heads else None,
                  'epoch': epoch, 'val_kl': val_kl, 'val_retmse10': val_retmse10, 'config': config}
        torch.save(payload, ckpt_dir / f'checkpoint_epoch{epoch}.pth')
        if val_retmse10 < best_retmse['val']:
            best_retmse = {'val': val_retmse10, 'epoch': epoch}
            torch.save(payload, ckpt_dir / 'checkpoint_best_retmse.pth')
        if val_kl < best_kl['val']:
            best_kl = {'val': val_kl, 'epoch': epoch}
            torch.save(payload, ckpt_dir / 'checkpoint_best_kl.pth')
        print(f'[v_p100] {arm} epoch={epoch} val_kl={val_kl:.6f} val_retmse10={val_retmse10:.6f} '
             f'val_recall10_full={val_metrics["recall10_full"]:.4f} '
             f'coverage@pool={val_metrics["coverage_at_pool"]:.4f} '
             f'(best_retmse={best_retmse["epoch"]}:{best_retmse["val"]:.6f} '
             f'best_kl={best_kl["epoch"]}:{best_kl["val"]:.6f})')
        if not cli.disable_early_stopping and epoch - best_retmse['epoch'] >= cli.patience:
            print(f'[v_p100] {arm} early stop at epoch {epoch} (best_retmse={best_retmse["epoch"]})')
            break

    # ---- test evaluation on BOTH checkpoints ----
    def run_test(ckpt_path, tag):
        bl = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(bl['model_state_dict'])
        if slot_heads is not None:
            slot_heads.load_state_dict(bl['slot_heads_state_dict'])
        model.eval()
        if slot_heads is not None:
            slot_heads.eval()
        test_sums, n = {}, 0
        diag_accum = {}
        probe_embeds = []
        with torch.no_grad():
            for batch_x, batch_y, batch_start_idx in test_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                cand_mask, _ = exp._candidate_mask(batch_start_idx)
                bsz = batch_x.size(0)
                per_ch = {}
                for c in channels:
                    res = eval_channel(batch_x, batch_y, batch_start_idx, c, cand_mask, 'test',
                                       want_full_oracle=True)
                    for key_ in ('kl', 'model_ind_mse', 'agg_mse', 'recall10_p100', 'recall10_full',
                               'ndcg10', 'coverage_at_pool', 'D', 'C', 'oracle_ind_mse'):
                        v = res[key_]
                        per_ch.setdefault(key_, []).append(v.reshape(-1).cpu() if torch.is_tensor(v)
                                                           else torch.tensor([float(v)]))
                    if c == channels[0] and len(probe_embeds) < cli.n_probe_rank:
                        probe_embeds.append(encode_raw(model, exp.memory_x[:min(cli.n_probe_rank, exp.memory_x.size(0))], c))
                for k_, vals in per_ch.items():
                    test_sums[k_] = test_sums.get(k_, 0.0) + torch.cat(vals).sum().item()
                n += bsz
        n_eff = max(n * len(channels), 1)
        test_metrics = {k_: v / n_eff for k_, v in test_sums.items()}
        test_metrics['n_queries_seen'] = n
        test_metrics['best_epoch'] = int(bl['epoch'])
        if probe_embeds:
            with torch.no_grad():
                geom = geometry_metrics(probe_embeds[0])
            test_metrics.update(geom)
        (out_dir / f'final_test_metrics_{tag}.json').write_text(json.dumps(test_metrics, indent=2, default=str))
        print(f'[v_p100] {arm} [{tag}] best_epoch={test_metrics["best_epoch"]} test_KL={test_metrics["kl"]:.6f} '
             f'test_retMSE@10={test_metrics["model_ind_mse"]:.6f} test_agg_mse10={test_metrics["agg_mse"]:.6f} '
             f'coverage@pool={test_metrics["coverage_at_pool"]:.4f} recall10_full={test_metrics["recall10_full"]:.4f}')
        return test_metrics

    test_metrics_retmse = run_test(ckpt_dir / 'checkpoint_best_retmse.pth', 'best_retmse')
    test_metrics_kl = run_test(ckpt_dir / 'checkpoint_best_kl.pth', 'best_kl')

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
    (out_dir / 'checkpoint_fingerprints.json').write_text(json.dumps(
        {'init_hash': init_hash, 'best_retmse_epoch': best_retmse['epoch'], 'best_kl_epoch': best_kl['epoch'],
         'batch_order_hashes': batch_order_hashes}, indent=2))
    vram = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
    print(f'[v_p100] done. {arm} best_retmse_epoch={best_retmse["epoch"]} best_kl_epoch={best_kl["epoch"]} '
         f'max_vram_mb={vram:.1f} wall_seconds={time.time()-t0:.1f}')


if __name__ == '__main__':
    main()
