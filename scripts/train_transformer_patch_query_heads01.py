#!/usr/bin/env python3
"""TRACK-TRANSFORMER-PATCH-QUERY-HEADS01.

Separates three effects that TRACK-TEMPORAL-QUERY-HEADS01's masked-MLP
construction could not (it never had real patch tokens -- V5's
`relation_encoder_type='mlp'` has no patch structure at all):

    MLP encoder -> real patch-based Transformer encoder   (backbone effect)
    Transformer pooled-only query -> patch-aware query     (patch-token-utilization effect)
    1 patch-aware query -> 5 patch-aware queries            (multi-head-complementarity effect)

Four arms, one shared trainer, `--arm {V1,V5,PH1,PH5}`:

    V1   Transformer backbone, pooled z_q, 1 static linear head (SlotHeads M=1)
    V5   Transformer backbone, pooled z_q, 5 static linear heads (SlotHeads M=5), Mean-Mixture
    PH1  Transformer backbone, SAME SlotHeads(M=1) g_1 attends over the Transformer's OWN
         contextual patch tokens (`model.encoder(..., return_tokens=True)`) -- reuses
         `models.TemporalQueryHeads.temporal_attention_heads` UNMODIFIED (parameter-free,
         no 1/sqrt(D) -- that scaling was already diagnosed wrong for L2-unit vectors
         in TRACK-TEMPORAL-QUERY-HEADS01 and is not reintroduced here)
    PH5  Same as PH1 but SlotHeads(M=5), Mean-Mixture over the 5 resulting patch-aware queries

Candidate branch is IDENTICAL across all 4 arms and never uses patch
tokens (pooled-only, `encode_raw` -- see module docstring section 8 of
the spec: mixing patch-aware query construction with any candidate
-side patch interaction would make the two effects inseparable).

Backbone: `models.RelationStage1.RelationEncoder`,
`relation_encoder_type='transformer'`, `relation_self_fill='zero'`
(transformer branch does not support `self_fill='linear'`),
`relation_pooling='cls'` (so `return_tokens=True` excludes the CLS
slot from the token bank, per that method's own docstring). Patch
config fixed, no sweep: `patch_len=stride` chosen per horizon so
`num_patches=6` for both H96 (16) and H720 (120) -- asserted at
startup, never varied.

Reused UNMODIFIED: `build_experiment` (`train_margutil01`), `encode_raw`/
`individual_utility_memsafe` (`train_factorial_e2e01`), `state_hash`/
`geometry_metrics` (`train_j_shared_encoder_drift01`), `SlotHeads`/
`kl_loss_from_prob` (`train_k_multislot_predictive_retrieval01`),
`temporal_attention_heads`/`attention_entropy`/
`normalized_attention_entropy`/`pairwise_head_attention_cosine`/
`query_attention_diversity` (`models.TemporalQueryHeads` -- the
parameter-free attention mechanism itself, byte-for-byte the same code
TRACK-TEMPORAL-QUERY-HEADS01 used), `normalized_teacher_prob`
(`train_horizon_retrieval_expert01`), `memory_value` (`train_margutil01`),
`recall_at_k`/`ndcg_at_k` (`train_patch_retrieval_expert01`),
`round_robin_topk_selection`/`slot_mechanism_diagnostics`
(`train_t_pure_multislot01`), `mean_mixture_topk_selection`
(`utils.mean_mixture_selection`), `CandidatePoolCache`
(`train_retriever_pool01`), `gather_candidate_*`/`pooled_future_mse`/
`local_to_global`/`encode_pooled_candidates`/`CandidatePoolConfig`
(`utils.candidate_pool`), `stable_topk_indices` (`RelationStage1`),
`batch_order_sha256`/`make_loader_generator`/`set_global_seeds`
(`rng_control01`), `union_oracle_recall_curve` (`train_temporal_query_heads01`).
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from models.TemporalQueryHeads import (
    attention_entropy, normalized_attention_entropy, pairwise_head_attention_cosine,
    query_attention_diversity, temporal_attention_heads,
)
from scripts.rng_control01 import batch_order_sha256, make_loader_generator, set_global_seeds
from scripts.train_factorial_e2e01 import encode_raw, individual_utility_memsafe
from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob
from scripts.train_j_shared_encoder_drift01 import geometry_metrics, state_hash
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads, kl_loss_from_prob
from scripts.train_margutil01 import build_experiment, memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k
from scripts.train_retriever_pool01 import CandidatePoolCache
from scripts.train_t_pure_multislot01 import round_robin_topk_selection, slot_mechanism_diagnostics
from scripts.train_temporal_query_heads01 import union_oracle_recall_curve
from utils.candidate_pool import (
    CandidatePoolConfig, encode_pooled_candidates, gather_candidate_histories,
    gather_candidate_values, local_to_global, pooled_future_mse,
)
from utils.mean_mixture_selection import mean_mixture_topk_selection

EPS = 1e-8
TOP_K = 10
ARMS = ('V1', 'V5', 'PH1', 'PH5')
ARM_N_SLOTS = {'V1': 1, 'V5': 5, 'PH1': 1, 'PH5': 5}
PATCH_AWARE_ARMS = ('PH1', 'PH5')


def build_transformer_model(cli, device):
    """Sibling of `train_j_shared_encoder_drift01.build_model`, swapping
    the MLP relation encoder for the real patch-based Transformer branch.
    `d_model`/`n_heads`/`e_layers`/`d_ff` are passed explicitly (matching
    the spec's fixed architecture) rather than relying on whatever the
    reference checkpoint's own saved args happen to carry."""
    set_global_seeds(cli.init_seed)
    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.seq_len, 'batch_size': cli.batch_size,
        'seed': cli.init_seed, 'top_k': cli.top_k,
        'relation_encoder_type': 'transformer', 'relation_self_fill': 'zero',
        'relation_pooling': 'cls',
        'relation_input_space': 'delta_last', 'relation_teacher_space': 'delta_last',
        'relation_value_space': 'delta_last', 'candidate_mask': 'raft',
        'patch_len': cli.patch_len, 'stride': cli.stride,
        'd_model': 128, 'n_heads': 4, 'e_layers': 2, 'd_ff': 256, 'dropout': 0.1,
    })
    exp._ensure_memory()
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.to(device)
    return exp, args, model


def encode_raw_with_tokens(model, x, c):
    """Query-side only (spec section 8: candidates NEVER use patch
    tokens). `model.encoder(relation_x, return_tokens=True)` returns
    `(pooled_normalized [B,D], token_bank [B,P,D])` for
    `encoder_type='transformer'` -- see that method's own docstring for
    why CLS is excluded from the token bank when `relation_pooling='cls'`."""
    relation_x = model._relation_tensor(x, c, c)
    z, tokens = model.encoder(relation_x, return_tokens=True)
    return z, tokens


def compute_scores_arm(arm, model, slot_heads, batch_x, exp, channel, pool_cache, batch_start_idx, device):
    """Candidate branch (pooled-only `encode_raw`) is byte-for-byte
    identical across all 4 arms. Query branch differs only in whether
    `g_m = slot_heads(z_q)` is used directly (V1/V5) or as an attention
    query over the Transformer's own contextual patch tokens (PH1/PH5)."""
    attn = None
    if arm in PATCH_AWARE_ARMS:
        z_q, h_patches = encode_raw_with_tokens(model, batch_x, channel)
        g = slot_heads(z_q)
        q, attn = temporal_attention_heads(g, h_patches)
    else:
        z_q = encode_raw(model, batch_x, channel)
        q = slot_heads(z_q)

    pool_idx_global = pool_cache.lookup(batch_start_idx, channel, device)
    pooled_x = gather_candidate_histories(exp.memory_x, pool_idx_global)
    z_k = encode_pooled_candidates(lambda x, c: encode_raw(model, x, c), pooled_x, channel)
    z_k = F.normalize(z_k, dim=-1)
    scores = torch.einsum('bsd,bmd->bsm', q, z_k)
    pool_valid_mask = torch.ones(pool_idx_global.shape, dtype=torch.bool, device=device)
    return scores, pool_valid_mask, pool_idx_global, attn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--arm', required=True, choices=ARMS)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--patch_len', type=int, required=True)
    ap.add_argument('--stride', type=int, default=None, help='defaults to patch_len (no stride sweep)')
    ap.add_argument('--top_k', type=int, default=TOP_K)
    ap.add_argument('--tau_t', type=float, default=0.1)
    ap.add_argument('--tau_s', type=float, default=0.1)
    ap.add_argument('--slot_std', type=float, default=1e-3)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--train_epochs', type=int, default=10)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--init_seed', type=int, default=0)
    ap.add_argument('--loader_seed', type=int, default=0)
    ap.add_argument('--candidate_mask', default='raft')
    ap.add_argument('--candidate_pool_size', type=int, default=100)
    ap.add_argument('--candidate_pool_cache', required=True)
    ap.add_argument('--n_probe_rank', type=int, default=256)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--checkpoints', required=True)
    ap.add_argument('--expected_num_patches', type=int, default=6)
    ap.add_argument('--limit_batches', type=int, default=0, help='SMOKE/DEBUG ONLY')
    ap.add_argument('--smoke_test', action='store_true')
    cli = ap.parse_args()
    if cli.stride is None:
        cli.stride = cli.patch_len
    if cli.candidate_pool_size <= cli.top_k:
        raise ValueError(f'[ISSUE][ABORT] --candidate_pool_size ({cli.candidate_pool_size}) must be '
                         f'> --top_k ({cli.top_k})')

    n_slots = ARM_N_SLOTS[cli.arm]
    is_patch_aware = cli.arm in PATCH_AWARE_ARMS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(cli.out_dir) / cli.cell / cli.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints) / cli.cell / cli.arm
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    exp, args, model = build_transformer_model(cli, device)
    channels = list(range(int(args.enc_in)))
    init_hash = state_hash(model)
    d_model = int(args.d_model)

    # S1: num_patches actually produced by the patch embedding must match the fixed spec (6).
    actual_num_patches = model.encoder.patch_embed.num_patches
    assert actual_num_patches == cli.expected_num_patches, (
        f'[ISSUE][ABORT] num_patches={actual_num_patches}, expected {cli.expected_num_patches} '
        f'(seq_len={cli.seq_len}, patch_len={cli.patch_len}, stride={cli.stride})')

    slot_heads = SlotHeads(d_model, n_slots=n_slots, std=cli.slot_std).to(device)
    slot_heads_param_count = sum(p.numel() for p in slot_heads.parameters())
    # PH-arms add ZERO new trainable parameters beyond the paired V-arm's own
    # SlotHeads (attention over patch tokens is parameter-free) -- assert, not just report.
    assert slot_heads_param_count == sum(
        p.numel() for p in SlotHeads(d_model, n_slots, cli.slot_std).parameters())
    encoder_param_count = sum(p.numel() for p in model.parameters())
    total_trainable = encoder_param_count + slot_heads_param_count

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
    params = list(model.parameters()) + list(slot_heads.parameters())
    optimizer = torch.optim.Adam(params, lr=cli.learning_rate)

    train_gen = make_loader_generator(cli.loader_seed)
    _, train_loader = exp._get_data(flag='train', shuffle=True, generator=train_gen)
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)

    config = {
        'exp': 'TRACK-TRANSFORMER-PATCH-QUERY-HEADS01', 'cell': cli.cell, 'arm': cli.arm,
        'n_slots': n_slots, 'is_patch_aware': is_patch_aware, 'patch_len': cli.patch_len,
        'stride': cli.stride, 'num_patches': actual_num_patches,
        'candidate_pool_mode': 'coarse_topk', 'candidate_pool_size': cli.candidate_pool_size,
        'candidate_pool_metric': 'delta_last_cosine', 'full_candidate_count': int(exp.memory_x.size(0)),
        'tau_t': cli.tau_t, 'tau_s': cli.tau_s, 'top_k': cli.top_k, 'batch_size': cli.batch_size,
        'learning_rate': cli.learning_rate, 'train_epochs': cli.train_epochs,
        'init_seed': cli.init_seed, 'loader_seed': cli.loader_seed, 'init_hash': init_hash,
        'slot_heads_param_count': slot_heads_param_count, 'encoder_param_count': encoder_param_count,
        'total_trainable_param_count': total_trainable,
        'checkpoint_policy': 'FIXED 10 epochs, no early stopping, every epoch saved; '
                             'post-hoc selection by validation RetMSE@10 (n_slots=1) or '
                             'Mean-Mixture RetMSE@10 (n_slots=5), test split NEVER used',
    }
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2, default=str))
    print(f'[tpqh] arm={cli.arm} cell={cli.cell} N={config["full_candidate_count"]} '
         f'num_patches={actual_num_patches} init_hash={init_hash[:16]} '
         f'slot_heads_params={slot_heads_param_count} encoder_params={encoder_param_count} '
         f'total_trainable={total_trainable}')

    def eval_channel(batch_x, batch_y, batch_start_idx, c, split, want_full_oracle=False, want_attn_diag=False):
        scores, valid_mask, pool_idx_global, attn = compute_scores_arm(
            cli.arm, model, slot_heads, batch_x, exp, c, pool_caches[split], batch_start_idx, device)
        memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
        query_future = batch_y[:, :, c]

        pooled_memory_c = gather_candidate_values(memory_c, pool_idx_global)
        d_pool = pooled_future_mse(pooled_memory_c, offset_c, query_future)
        restricted_oracle_local = stable_topk_indices(d_pool, cli.top_k, largest=False)

        if n_slots == 1:
            model_idx_local = round_robin_topk_selection(scores, valid_mask, k=cli.top_k)
        else:
            model_idx_local, _, _ = mean_mixture_topk_selection(scores, valid_mask, cli.tau_s, k=cli.top_k)

        y_sel = pooled_memory_c.gather(
            1, model_idx_local.unsqueeze(-1).expand(-1, -1, pooled_memory_c.size(-1))
        ) + offset_c.view(-1, 1, 1)
        individual_mse_i = d_pool.gather(1, model_idx_local)
        oracle_ind_mse = d_pool.gather(1, restricted_oracle_local)
        D_ = individual_mse_i.mean(-1) / cli.top_k
        agg_pred = y_sel.mean(dim=1)
        agg_mse = ((agg_pred - query_future) ** 2).mean(-1)
        C_ = agg_mse - D_
        recall10 = recall_at_k(model_idx_local, restricted_oracle_local, cli.top_k)
        ndcg10 = ndcg_at_k(model_idx_local, d_pool, valid_mask, cli.top_k)

        s_masked = scores.masked_fill(~valid_mask.unsqueeze(1), float('-inf'))
        p_m = torch.softmax(s_masked / cli.tau_s, dim=-1)
        p_bar = p_m.mean(dim=1)
        p_t = normalized_teacher_prob(d_pool, valid_mask, cli.tau_t)
        kl = kl_loss_from_prob(p_t, p_bar, valid_mask)

        res = dict(model_idx_global=local_to_global(pool_idx_global, model_idx_local),
                  model_ind_mse=individual_mse_i.mean(-1), oracle_ind_mse=oracle_ind_mse.mean(-1),
                  recall10=recall10, ndcg10=ndcg10, agg_mse=agg_mse, D=D_, C=C_, kl=kl,
                  pool_idx_global=pool_idx_global)

        if want_full_oracle:
            with torch.no_grad():
                cand_mask, _ = exp._candidate_mask(batch_start_idx)
                u_full = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
                d_full = -u_full
                full_oracle_idx = stable_topk_indices(d_full.masked_fill(~cand_mask, float('inf')),
                                                      cli.top_k, largest=False)
                recall10_full = recall_at_k(res['model_idx_global'], full_oracle_idx, cli.top_k)
            res['recall10_full'] = recall10_full

        if want_attn_diag and n_slots > 1:
            with torch.no_grad():
                mech = slot_mechanism_diagnostics(scores, valid_mask, p_m, cli.tau_s)
                curve, union_size = union_oracle_recall_curve(scores, valid_mask, restricted_oracle_local,
                                                               cli.top_k)
                res['slot_top1_overlap'] = mech['slot_top1_overlap']
                res['slot_top10_overlap'] = mech['slot_top10_overlap']
                res['mean_pairwise_slot_cosine'] = mech['mean_pairwise_slot_cosine']
                res['union_oracle_recall_curve'] = curve
                res['union_size'] = union_size
                if is_patch_aware and attn is not None:
                    res['attn_entropy_normalized_mean'] = float(normalized_attention_entropy(attn).mean())
                    res['attn_max_prob_mean'] = float(attn.max(dim=-1).values.mean())
                    res['attn_pairwise_head_cosine'] = pairwise_head_attention_cosine(attn)
                    res['attn_query_diversity'] = query_attention_diversity(attn)
        return res

    if cli.smoke_test:
        model.train()
        slot_heads.train()
        for bi, (batch_x, batch_y, batch_start_idx) in enumerate(train_loader):
            if bi >= 3:
                break
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            optimizer.zero_grad()
            for c in channels:
                res = eval_channel(batch_x, batch_y, batch_start_idx, c, 'train')
                (res['kl'] / len(channels)).backward()
            assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters()), \
                '[ISSUE] no gradient reached model.encoder'
            assert slot_heads.W.grad is not None and slot_heads.W.grad.abs().sum() > 0, \
                '[ISSUE] no gradient reached SlotHeads.W'
            optimizer.step()
        vram = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
        print(f'[tpqh] {cli.arm} SMOKE PASS max_vram_mb={vram:.1f} num_patches={actual_num_patches}')
        return

    epoch_val_rows, step_rows, batch_order_hashes = [], [], []
    t0 = time.time()
    global_step = 0
    for epoch in range(1, cli.train_epochs + 1):
        model.train()
        slot_heads.train()
        starts = []
        for bi, (batch_x, batch_y, batch_start_idx) in enumerate(train_loader):
            if cli.limit_batches and bi >= cli.limit_batches:
                break
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            starts.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx)
                         else torch.as_tensor(batch_start_idx))
            optimizer.zero_grad()
            batch_loss = 0.0
            for c in channels:
                res = eval_channel(batch_x, batch_y, batch_start_idx, c, 'train')
                l = res['kl']
                (l / len(channels)).backward()
                batch_loss += float(l.detach()) / len(channels)
            optimizer.step()
            global_step += 1
            step_rows.append({'global_step': global_step, 'epoch': epoch, 'batch_in_epoch': bi,
                              'train_loss_kl': batch_loss})
        batch_order_hashes.append({'epoch': epoch, 'batch_order_sha256': batch_order_sha256(starts)})

        model.eval()
        slot_heads.eval()
        val_sums, n = {}, 0
        with torch.no_grad():
            for batch_x, batch_y, batch_start_idx in val_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                bsz = batch_x.size(0)
                per_ch = {}
                for c in channels:
                    res = eval_channel(batch_x, batch_y, batch_start_idx, c, 'val', want_full_oracle=True)
                    for key_ in ('kl', 'model_ind_mse', 'agg_mse', 'recall10', 'recall10_full', 'ndcg10'):
                        v = res[key_]
                        per_ch.setdefault(key_, []).append(v.reshape(-1).cpu() if torch.is_tensor(v)
                                                           else torch.tensor([float(v)]))
                for k_, vals in per_ch.items():
                    val_sums[k_] = val_sums.get(k_, 0.0) + torch.cat(vals).sum().item()
                n += bsz
        n_eff = max(n * len(channels), 1)
        val_metrics = {k_: v / n_eff for k_, v in val_sums.items()}
        val_metrics['epoch'] = epoch
        epoch_val_rows.append(val_metrics)

        payload = {'model_state_dict': model.state_dict(), 'slot_heads_state_dict': slot_heads.state_dict(),
                  'epoch': epoch, 'val_retmse10': val_metrics['model_ind_mse'], 'config': config}
        torch.save(payload, ckpt_dir / f'checkpoint_epoch{epoch}.pth')
        print(f'[tpqh] {cli.arm} epoch={epoch} val_kl={val_metrics["kl"]:.6f} '
             f'val_retmse10={val_metrics["model_ind_mse"]:.6f} val_agg_mse10={val_metrics["agg_mse"]:.6f} '
             f'val_recall10_full={val_metrics["recall10_full"]:.4f}')

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
        {'init_hash': init_hash, 'slot_heads_param_count': slot_heads_param_count,
         'batch_order_hashes': batch_order_hashes}, indent=2))
    vram = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
    print(f'[tpqh] done. {cli.arm} max_vram_mb={vram:.1f} wall_seconds={time.time()-t0:.1f}')


if __name__ == '__main__':
    main()
