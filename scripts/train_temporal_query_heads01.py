#!/usr/bin/env python3
"""TRACK-TEMPORAL-QUERY-HEADS01 -- TQH-5-Masked.

Architecture-only ablation against V5 (`scripts/train_v_sharedtop100_01.py`,
`arm=V5`): the ONLY difference is how the 5 query heads are generated.
V5: `q_m = normalize(W_m z_q)`, a static linear projection of the pooled
query embedding. TQH-5-Masked: `q_m` is instead produced by having `g_m
= normalize(W_m z_q)` (V5's OWN SlotHeads, unchanged, same W/init/std)
attend over a per-query "temporal token bank" built by re-running the
SAME shared MLP relation-encoder on P time-masked copies of the exact
same relation input, baseline-subtracted
(`models/TemporalQueryHeads.compute_temporal_token_bank`). See that
module's docstring and `research/EXPERIMENT_LOG.md` for why this
masked-MLP-token approach was used instead of the existing
TRACK-F-LATE-INTERACTION-FEASIBILITY01 `return_tokens=True` path (that
path only supports `encoder_type='transformer'`; V5 uses `mlp`).

Everything else -- loss, teacher, candidate pool (P100, SHARED cache
file with V5, same ids), Top-K, checkpoint-selection policy, Stage-2 --
is byte-for-byte identical to V5. No new trainable parameters: the
only parameters anywhere in the head-generation mechanism are V5's own
`SlotHeads.W`; this is asserted and reported at startup.

Reused UNMODIFIED: `build_model`/`state_hash`/`geometry_metrics`
(`train_j_shared_encoder_drift01`), `SlotHeads`/`kl_loss_from_prob`
(`train_k_multislot_predictive_retrieval01`), `encode_raw`/
`individual_utility_memsafe` (`train_factorial_e2e01`),
`normalized_teacher_prob` (`train_horizon_retrieval_expert01`),
`memory_value` (`train_margutil01`), `recall_at_k`/`ndcg_at_k`
(`train_patch_retrieval_expert01`), `round_robin_topk_selection`/
`slot_mechanism_diagnostics`/`spearman_batch` (`train_t_pure_multislot01`),
`CandidatePoolCache` (`train_retriever_pool01`), `gather_candidate_*`/
`pooled_future_mse`/`local_to_global`/`encode_pooled_candidates`/
`CandidatePoolConfig` (`utils.candidate_pool`), `stable_topk_indices`
(`RelationStage1`), `batch_order_sha256`/`make_loader_generator`
(`rng_control01`).
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
from models.TemporalQueryHeads import (
    attention_entropy, build_patch_time_masks, compute_temporal_token_bank,
    normalized_attention_entropy, pairwise_head_attention_cosine, query_attention_diversity,
    temporal_attention_heads,
)
from scripts.rng_control01 import batch_order_sha256, make_loader_generator
from scripts.train_factorial_e2e01 import encode_raw, individual_utility_memsafe
from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob
from scripts.train_j_shared_encoder_drift01 import build_model, geometry_metrics, state_hash
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads, kl_loss_from_prob
from scripts.train_margutil01 import memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k
from scripts.train_retriever_pool01 import CandidatePoolCache
from scripts.train_t_pure_multislot01 import (
    round_robin_topk_selection, slot_mechanism_diagnostics, spearman_batch,
)
from utils.candidate_pool import (
    CandidatePoolConfig, gather_candidate_histories, gather_candidate_values, local_to_global,
    pooled_future_mse,
)

EPS = 1e-8
TOP_K = 10
VALID_NUM_QUERY_VIEWS = (1, 5)


def _entropy_and_effective(p, valid_mask):
    p_safe = p.clamp_min(EPS)
    ent = -(p_safe * p_safe.log()).sum(-1)
    return float(ent.mean()), float(ent.mean().exp())


def encode_pooled_candidates(encode_fn, pooled_x, channel):
    bsz, pool_m, seq_len, n_channels = pooled_x.shape
    flat = pooled_x.reshape(bsz * pool_m, seq_len, n_channels)
    z_flat = encode_fn(flat, channel)
    d_model = z_flat.size(-1)
    return z_flat.reshape(bsz, pool_m, d_model)


def compute_scores_tqh(model, slot_heads, batch_x, exp, channel, pool_cache, batch_start_idx, device,
                       mask_cache, patch_len):
    """Candidate branch byte-for-byte identical to V5's `coarse_topk`
    `compute_scores` (`scripts/train_retriever_pool01.py`); only the
    query-side `q` construction differs."""
    relation_x = model._relation_tensor(batch_x, channel, channel)  # [B, F, L]
    z_q = model.encoder(relation_x)  # [B, D], already L2-normalized (cosine mode, same as encode_raw)
    seq_len = relation_x.size(-1)
    if seq_len not in mask_cache:
        mask_cache[seq_len] = build_patch_time_masks(seq_len, patch_len, device=device)
    masks = mask_cache[seq_len]
    t_p, z_p_raw, z_0 = compute_temporal_token_bank(model.encoder, relation_x, masks)
    g = slot_heads(z_q)  # [B, S, D] -- V5's OWN SlotHeads, unchanged
    q, attn = temporal_attention_heads(g, t_p)  # [B, S, D], [B, S, P]
    with torch.no_grad():
        attn_logit_std = float(torch.einsum('bsd,bpd->bsp', g, t_p).std())

    pool_idx_global = pool_cache.lookup(batch_start_idx, channel, device)  # [B, M]
    pooled_x = gather_candidate_histories(exp.memory_x, pool_idx_global)
    z_k = encode_pooled_candidates(lambda x, c: encode_raw(model, x, c), pooled_x, channel)
    z_k = F.normalize(z_k, dim=-1)
    scores = torch.einsum('bsd,bmd->bsm', q, z_k)
    pool_valid_mask = torch.ones(pool_idx_global.shape, dtype=torch.bool, device=device)
    diag = {'attn': attn, 't_p': t_p, 'z_p_raw': z_p_raw, 'z_0': z_0, 'attn_logit_std': attn_logit_std}
    return scores, pool_valid_mask, pool_idx_global, diag


# 2026-10-22: these attention/mechanism diagnostics are NEW metrics this
# track introduces -- each is already a scalar MEAN over one (channel,
# batch)'s `bsz` queries (e.g. `attention_entropy(attn).mean()`), unlike
# the genuinely per-query tensors (`model_ind_mse`, `agg_mse`, ...) this
# script otherwise reuses from V5. Appending a batch-level scalar as a
# SINGLE value (`torch.tensor([float(v)])`, V5's own convention for its
# OWN scalar diagnostics `kl`/`teacher_entropy`/`student_entropy`) and
# then dividing the running sum by `n_eff = n_queries * n_channels` at
# the end systematically suppresses it by roughly 1/bsz (confirmed: val_kl
# -- which uses exactly this V5-inherited convention -- reads 0.069 on an
# ETTh1_96 pilot epoch where 1/32-undoing it gives a far more plausible
# ~2.2; this is a PRE-EXISTING scaling artifact in V5/K/M-family `kl`/
# `teacher_entropy`/`student_entropy` reporting going back through this
# project's history, reported to the user as a separate finding rather
# than silently patched here -- fixing it would break byte-for-byte
# comparability with V5's own already-reported numbers and checkpoint
# selection, which this track must not touch). For the diagnostics THIS
# track owns, the correct fix is applied: weight each batch-level scalar
# by `bsz` before accumulating.
_BATCH_SCALAR_DIAG_KEYS = {
    'attn_entropy_mean', 'attn_entropy_normalized_mean', 'attn_max_prob_mean', 'attn_logit_std',
    'attn_pairwise_head_cosine', 'attn_query_diversity', 'union_size',
    'slot_top1_overlap', 'slot_top10_overlap', 'mean_pairwise_slot_cosine',
}


def _append_metric(per_ch, key_, v, bsz):
    if torch.is_tensor(v):
        per_ch.setdefault(key_, []).append(v.reshape(-1).cpu())
        return
    fv = float(v)
    if fv != fv:  # NaN (e.g. attn_query_diversity on a bsz==1 trailing batch) -- skip, don't poison the sum
        return
    if key_ in _BATCH_SCALAR_DIAG_KEYS:
        per_ch.setdefault(key_, []).append(torch.full((bsz,), fv))
    else:
        per_ch.setdefault(key_, []).append(torch.tensor([fv]))  # V5's own convention, unchanged


def union_oracle_recall_curve(scores, valid_mask, oracle_idx_local, top_k=TOP_K):
    """Deterministic head order 0..S-1 (not averaged over all 5! orderings
    -- documented simplification, spec section 10 permits either).
    Returns (curve: list[float] len S, union_size: float) -- curve[i] is
    the union-oracle-recall using heads 0..i inclusive."""
    bsz, s, m = scores.shape
    device = scores.device
    in_union = torch.zeros(bsz, m, dtype=torch.bool, device=device)
    curve = []
    for h in range(s):
        s_h = scores[:, h, :].masked_fill(~valid_mask, float('-inf'))
        idx_h = stable_topk_indices(s_h, top_k, largest=True)
        in_union.scatter_(1, idx_h, True)
        oracle_in_union = in_union.gather(1, oracle_idx_local)
        curve.append(float(oracle_in_union.float().mean()))
    union_size = float(in_union.float().sum(dim=-1).mean())
    return curve, union_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--num_query_views', type=int, required=True, choices=VALID_NUM_QUERY_VIEWS,
                    help='1 -> TQH-1 (single temporal-attention head, isolates query-construction effect '
                         'from multi-head diversity), 5 -> TQH-5 (full proposed architecture).')
    ap.add_argument('--cell', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--patch_len', type=int, default=16, help='TQH patch size along the time axis '
                    '(unlike V5, where this CLI arg is accepted but UNUSED by the mlp encoder).')
    ap.add_argument('--top_k', type=int, default=TOP_K)
    ap.add_argument('--tau_t', type=float, default=0.1)
    ap.add_argument('--tau_s', type=float, default=0.1)
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
    ap.add_argument('--candidate_pool_cache', required=True,
                    help='SAME directory V5 used for this (cell,seq_len,pred_len) -- shared P100 ids')
    ap.add_argument('--n_probe_rank', type=int, default=256)
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
    n_slots = cli.num_query_views
    arm_name = f'TQH{n_slots}Masked'
    out_dir = Path(cli.out_dir) / cli.cell / arm_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints) / cli.cell / arm_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    exp, args, model = build_model(cli, device)
    channels = list(range(int(args.enc_in)))
    init_hash = state_hash(model)
    d_model = int(args.d_model)

    slot_heads = SlotHeads(d_model, n_slots=n_slots, std=cli.slot_std).to(device)
    slot_heads_param_count = sum(p.numel() for p in slot_heads.parameters())
    # TQH-{n_slots}-Masked adds ZERO new trainable parameters beyond V5's own
    # SlotHeads (attention is a parameter-free function of g/t_p) -- this IS
    # the whole param-count-fairness claim, asserted here rather than just
    # reported.
    assert slot_heads_param_count == sum(p.numel() for p in SlotHeads(d_model, n_slots, cli.slot_std).parameters())
    tqh_total_trainable = sum(p.numel() for p in model.parameters()) + slot_heads_param_count

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

    mask_cache = {}

    config = {
        'exp': 'TRACK-TEMPORAL-QUERY-HEADS01', 'cell': cli.cell, 'arm': arm_name,
        'num_query_views': n_slots, 'patch_len': cli.patch_len,
        'candidate_pool_mode': 'coarse_topk', 'candidate_pool_size': cli.candidate_pool_size,
        'candidate_pool_metric': 'delta_last_cosine', 'full_candidate_count': int(exp.memory_x.size(0)),
        'tau_t': cli.tau_t, 'tau_s': cli.tau_s, 'top_k': cli.top_k, 'batch_size': cli.batch_size,
        'learning_rate': cli.learning_rate, 'train_epochs': cli.train_epochs, 'patience': cli.patience,
        'init_seed': cli.init_seed, 'loader_seed': cli.loader_seed, 'init_hash': init_hash,
        'slot_heads_param_count': slot_heads_param_count, 'total_trainable_param_count': tqh_total_trainable,
        'head_generation': 'temporal_attention_over_masked_mlp_tokens (ZERO new trainable params vs V5)',
        'checkpoint_policy': 'DUAL -- primary=min_val_retmse10 (Stage-2 uses ONLY this one, same policy as V5), '
                             'secondary=min_val_kl (diagnostic only, never fed to Stage-2)',
    }
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2, default=str))
    print(f'[tqh5] arm={arm_name} cell={cli.cell} N={config["full_candidate_count"]} '
         f'pool_size={cli.candidate_pool_size} init_hash={init_hash[:16]} '
         f'slot_heads_params={slot_heads_param_count} total_trainable={tqh_total_trainable}')

    def eval_channel(batch_x, batch_y, batch_start_idx, c, cand_mask, split, want_full_oracle=False,
                     want_attn_diag=False):
        scores, valid_mask, pool_idx_global, diag = compute_scores_tqh(
            model, slot_heads, batch_x, exp, c, pool_caches[split], batch_start_idx, device,
            mask_cache, cli.patch_len)
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

        if want_attn_diag:
            with torch.no_grad():
                attn = diag['attn']  # [B, S, P]
                mech = slot_mechanism_diagnostics(scores, valid_mask, p_m, cli.tau_s)
                curve, union_size = union_oracle_recall_curve(scores, valid_mask, restricted_oracle_local,
                                                               cli.top_k)
                res['attn_entropy_mean'] = float(attention_entropy(attn).mean())
                res['attn_entropy_normalized_mean'] = float(normalized_attention_entropy(attn).mean())
                res['attn_max_prob_mean'] = float(attn.max(dim=-1).values.mean())
                res['attn_logit_std'] = diag['attn_logit_std']
                res['attn_pairwise_head_cosine'] = pairwise_head_attention_cosine(attn)
                res['attn_query_diversity'] = query_attention_diversity(attn)
                res['union_oracle_recall_curve'] = curve
                res['union_size'] = union_size
                res['slot_top1_overlap'] = mech['slot_top1_overlap']
                res['slot_top10_overlap'] = mech['slot_top10_overlap']
                res['mean_pairwise_slot_cosine'] = mech['mean_pairwise_slot_cosine']
        return res

    if cli.smoke_test:
        model.train()
        slot_heads.train()
        for bi, (batch_x, batch_y, batch_start_idx) in enumerate(train_loader):
            if bi >= 3:
                break
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            cand_mask, _ = exp._candidate_mask(batch_start_idx)
            optimizer.zero_grad()
            for c in channels:
                res = eval_channel(batch_x, batch_y, batch_start_idx, c, cand_mask, 'train',
                                   want_attn_diag=True)
                (res['kl'] / len(channels)).backward()
            assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters()), \
                '[ISSUE] no gradient reached model.encoder'
            assert slot_heads.W.grad is not None and slot_heads.W.grad.abs().sum() > 0, \
                '[ISSUE] no gradient reached SlotHeads.W'
            optimizer.step()
        print(f'[tqh5] attn_logit_std(sample)={res["attn_logit_std"]:.4f} '
             f'attn_entropy_norm(sample)={res["attn_entropy_normalized_mean"]:.4f} '
             f'attn_max_prob(sample)={res["attn_max_prob_mean"]:.4f} '
             f'pairwise_head_cosine(sample)={res["attn_pairwise_head_cosine"]:.4f} '
             f'query_diversity(sample)={res["attn_query_diversity"]:.4f}')
        vram = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
        print(f'[tqh5] {arm_name} SMOKE PASS max_vram_mb={vram:.1f}')
        return

    epoch_val_rows, step_rows, batch_order_hashes = [], [], []
    best_retmse = {'val': float('inf'), 'epoch': -1}
    best_kl = {'val': float('inf'), 'epoch': -1}
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
                                       want_full_oracle=True, want_attn_diag=True)
                    for key_ in ('kl', 'model_ind_mse', 'agg_mse', 'recall10_p100', 'recall10_full',
                               'ndcg10', 'coverage_at_pool', 'teacher_entropy', 'teacher_effective_candidates',
                               'student_entropy', 'student_effective_candidates', 'attn_entropy_mean',
                               'attn_entropy_normalized_mean', 'attn_max_prob_mean', 'attn_logit_std',
                               'attn_pairwise_head_cosine', 'attn_query_diversity', 'union_size',
                               'slot_top1_overlap', 'slot_top10_overlap', 'mean_pairwise_slot_cosine'):
                        _append_metric(per_ch, key_, res[key_], bsz)
                for k_, vals in per_ch.items():
                    val_sums[k_] = val_sums.get(k_, 0.0) + torch.cat(vals).sum().item()
                n += bsz
        n_eff = max(n * len(channels), 1)
        val_metrics = {k_: v / n_eff for k_, v in val_sums.items()}
        val_kl = val_metrics['kl']
        val_retmse10 = val_metrics['model_ind_mse']
        val_metrics['epoch'] = epoch
        epoch_val_rows.append(val_metrics)

        payload = {'model_state_dict': model.state_dict(), 'slot_heads_state_dict': slot_heads.state_dict(),
                  'epoch': epoch, 'val_kl': val_kl, 'val_retmse10': val_retmse10, 'config': config}
        torch.save(payload, ckpt_dir / f'checkpoint_epoch{epoch}.pth')
        if val_retmse10 < best_retmse['val']:
            best_retmse = {'val': val_retmse10, 'epoch': epoch}
            torch.save(payload, ckpt_dir / 'checkpoint_best_retmse.pth')
        if val_kl < best_kl['val']:
            best_kl = {'val': val_kl, 'epoch': epoch}
            torch.save(payload, ckpt_dir / 'checkpoint_best_kl.pth')
        print(f'[tqh5] {arm_name} epoch={epoch} val_kl={val_kl:.6f} val_retmse10={val_retmse10:.6f} '
             f'val_agg_mse10={val_metrics["agg_mse"]:.6f} '
             f'val_recall10_full={val_metrics["recall10_full"]:.4f} '
             f'coverage@pool={val_metrics["coverage_at_pool"]:.4f} '
             f'attn_logit_std={val_metrics["attn_logit_std"]:.4f} '
             f'attn_entropy_norm={val_metrics["attn_entropy_normalized_mean"]:.4f} '
             f'attn_max_prob={val_metrics["attn_max_prob_mean"]:.4f} '
             f'pairwise_head_cos={val_metrics.get("attn_pairwise_head_cosine", float("nan")):.4f} '
             f'query_div={val_metrics["attn_query_diversity"]:.4f} '
             f'slot_top10_overlap={val_metrics.get("slot_top10_overlap", float("nan")):.4f} '
             f'union_size={val_metrics["union_size"]:.2f} '
             f'(best_retmse={best_retmse["epoch"]}:{best_retmse["val"]:.6f} '
             f'best_kl={best_kl["epoch"]}:{best_kl["val"]:.6f})')
        if not cli.disable_early_stopping and epoch - best_retmse['epoch'] >= cli.patience:
            print(f'[tqh5] {arm_name} early stop at epoch {epoch} (best_retmse={best_retmse["epoch"]})')
            break

    def run_test(ckpt_path, tag):
        bl = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(bl['model_state_dict'])
        slot_heads.load_state_dict(bl['slot_heads_state_dict'])
        model.eval()
        slot_heads.eval()
        test_sums, n = {}, 0
        probe_embeds = []
        union_curves = []
        with torch.no_grad():
            for batch_x, batch_y, batch_start_idx in test_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                cand_mask, _ = exp._candidate_mask(batch_start_idx)
                bsz = batch_x.size(0)
                per_ch = {}
                for c in channels:
                    res = eval_channel(batch_x, batch_y, batch_start_idx, c, cand_mask, 'test',
                                       want_full_oracle=True, want_attn_diag=True)
                    for key_ in ('kl', 'model_ind_mse', 'agg_mse', 'recall10_p100', 'recall10_full',
                               'ndcg10', 'coverage_at_pool', 'D', 'C', 'oracle_ind_mse', 'attn_entropy_mean',
                               'attn_entropy_normalized_mean', 'attn_max_prob_mean', 'attn_logit_std',
                               'attn_pairwise_head_cosine', 'attn_query_diversity', 'union_size',
                               'slot_top1_overlap', 'slot_top10_overlap', 'mean_pairwise_slot_cosine'):
                        _append_metric(per_ch, key_, res[key_], bsz)
                    union_curves.append(res['union_oracle_recall_curve'])
                    if c == channels[0] and len(probe_embeds) < cli.n_probe_rank:
                        probe_embeds.append(encode_raw(model, exp.memory_x[:min(cli.n_probe_rank, exp.memory_x.size(0))], c))
                for k_, vals in per_ch.items():
                    test_sums[k_] = test_sums.get(k_, 0.0) + torch.cat(vals).sum().item()
                n += bsz
        n_eff = max(n * len(channels), 1)
        test_metrics = {k_: v / n_eff for k_, v in test_sums.items()}
        test_metrics['n_queries_seen'] = n
        test_metrics['best_epoch'] = int(bl['epoch'])
        test_metrics['union_oracle_recall_by_n_heads'] = torch.tensor(union_curves).mean(dim=0).tolist()
        if probe_embeds:
            with torch.no_grad():
                geom = geometry_metrics(probe_embeds[0])
            test_metrics.update(geom)
        (out_dir / f'final_test_metrics_{tag}.json').write_text(json.dumps(test_metrics, indent=2, default=str))
        print(f'[tqh5] {arm_name} [{tag}] best_epoch={test_metrics["best_epoch"]} test_KL={test_metrics["kl"]:.6f} '
             f'test_retMSE@10={test_metrics["model_ind_mse"]:.6f} test_agg_mse10={test_metrics["agg_mse"]:.6f} '
             f'coverage@pool={test_metrics["coverage_at_pool"]:.4f} recall10_full={test_metrics["recall10_full"]:.4f} '
             f'union_oracle_recall_5heads={test_metrics["union_oracle_recall_by_n_heads"][-1]:.4f}')
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
        {'init_hash': init_hash, 'slot_heads_param_count': slot_heads_param_count,
         'best_retmse_epoch': best_retmse['epoch'], 'best_kl_epoch': best_kl['epoch'],
         'batch_order_hashes': batch_order_hashes}, indent=2))
    vram = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
    print(f'[tqh5] done. {arm_name} best_retmse_epoch={best_retmse["epoch"]} best_kl_epoch={best_kl["epoch"]} '
         f'max_vram_mb={vram:.1f} wall_seconds={time.time()-t0:.1f}')


if __name__ == '__main__':
    main()
