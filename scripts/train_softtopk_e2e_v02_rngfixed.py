#!/usr/bin/env python3
"""TRACK-P100-QA-CAUSAL-VALIDATION01 -- Phase A1/B fix of
`train_softtopk_e2e_v01.py` for the V-family (V0/V1/V2/V5).

ONLY intended change vs v01: the QA arm's one-time feature-normalizer
fitting pass previously iterated the SAME `train_loader` object used
for real training (`for bi, (...) in enumerate(train_loader): if bi >=
50: break`). Each `enumerate(train_loader)` call triggers a fresh
`RandomSampler.__iter__()`, which draws ONE new permutation from the
loader's `torch.Generator` -- so the fitting pass silently advanced
`train_gen`'s state before the real training loop's first
`enumerate(train_loader)` call. NOQA never runs this fitting pass, so
NOQA's epoch-1 batch order is the generator's FIRST draw while QA's is
its SECOND draw -- a confirmed, systematic one-draw offset between the
two arms' batch order for the entire run (verified against real
artifacts: `ETTh1_96/V5`, `ETTh1_192/V5`, `Weather_720/Joint`,
`Solar_96/V0` NOQA vs QA epoch-1 `batch_order_sha256` all differ in the
original track).

Fix: the fitting pass now uses a SEPARATE DataLoader built from its OWN
throwaway generator (`fit_loader_seed`, clearly distinct from
`loader_seed`), so `train_gen`/`train_loader`'s state is completely
untouched before the real training loop begins, for BOTH arms. NOQA
and QA now draw IDENTICAL batch orders every epoch (asserted by
`scripts/verify_rngfixed_pairs01.py` after both arms of a pair finish).

Also newly recorded (not previously saved, per the causal-validation
audit): `slot_heads_init_hash` (SlotHeads' own state hash, independent
of the shared encoder's `init_hash`), so a paired NOQA/QA run's common
retriever-side initialization can be verified byte-for-byte, not just
assumed from both arms calling `set_global_seeds(cli.init_seed)`
identically.

Everything else (model/loss/fusion/checkpoint-selection logic,
channel-wise backward, TEST never constructed) is UNCHANGED from v01 --
see that file's docstring for the shared rationale, not repeated here.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from models.RelationStage2 import BaseForecastHead
from scripts.rng_control01 import batch_order_sha256, make_loader_generator, set_global_seeds
from scripts.train_factorial_e2e01 import state_sha
from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob
from scripts.train_j_shared_encoder_drift01 import build_model, state_hash
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads, kl_loss_from_prob
from scripts.train_margutil01 import memory_value
from scripts.train_retriever_pool01 import CandidatePoolCache, compute_scores
from utils.candidate_pool import CandidatePoolConfig, gather_candidate_values, pooled_future_mse
from utils.e2e_fusion_training01 import (
    GlobalLambda, combined_loss, fuse_noqa, fuse_qa, soft_topk_forecast,
)
from utils.quality_aware_gate01 import (
    QA_FEATURE_NAMES, QualityGate, TrainFeatureNormalizer, compute_qa_features,
)

TOP_K = 10
VALID_NUM_QUERY_VIEWS = (0, 1, 2, 5)
GAMMA_FIXED = 1.0
# Arbitrary, large, fixed offset so the fitting-pass generator's seed
# space never collides with any real `--loader_seed` value used in this
# track (0, 1, 2) -- the exact value is irrelevant (the fitting pass's
# own permutation is discarded, only its features matter), what matters
# is that it is a DIFFERENT generator object from `train_gen`.
FIT_LOADER_SEED_OFFSET = 10_000_003


def _load_base(base_checkpoint, seq_len, pred_len, channels, device):
    base = BaseForecastHead(seq_len=seq_len, pred_len=pred_len, channels=channels,
                            mode='per_channel_linear').to(device)
    sd = torch.load(base_checkpoint, map_location=device)
    state = sd['model_state_dict'] if 'model_state_dict' in sd else sd
    base.load_state_dict(state)
    base.eval()
    for p in base.parameters():
        p.requires_grad_(False)
    return base


def _channel_step(model, slot_heads, pool_cfg, batch_x, batch_y, exp, c, cand_mask,
                  pool_cache, batch_start_idx, device, args, cli, base_forecast_full,
                  gate_or_lambda, normalizer, arm):
    scores, valid_mask, pool_idx_global = compute_scores(
        pool_cfg, model, slot_heads, batch_x, exp, c, cand_mask, pool_cache, batch_start_idx, device)
    memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
    query_future = batch_y[:, :, c]

    pooled_memory_c = gather_candidate_values(memory_c, pool_idx_global)
    d_pool = pooled_future_mse(pooled_memory_c, offset_c, query_future)

    s_masked = scores.masked_fill(~valid_mask.unsqueeze(1), float('-inf'))
    p_heads = torch.softmax(s_masked / cli.tau_s, dim=-1)
    p_mix = p_heads.mean(dim=1)  # [B, M] -- the SAME p_full every V-arm's Soft Top-K uses

    picks = stable_topk_indices(p_mix.masked_fill(~valid_mask, float('-inf')), cli.top_k, largest=True)
    y_sel = pooled_memory_c.gather(
        1, picks.unsqueeze(-1).expand(-1, -1, pooled_memory_c.size(-1))) + offset_c.view(-1, 1, 1)
    r_c, weights = soft_topk_forecast(p_mix, picks, y_sel)

    p_t = normalized_teacher_prob(d_pool, valid_mask, cli.tau_t)
    retrieval_loss_c = kl_loss_from_prob(p_t, p_mix, valid_mask)

    base_c = base_forecast_full[:, :, c]
    if arm == 'NOQA':
        y_hat_c, lam_c = fuse_noqa(base_c, r_c, gate_or_lambda)
        feats_c = None
    else:
        y_hat_c, lam_c, feats_c = fuse_qa(
            base_c, r_c, y_sel, weights, p_mix, picks, valid_mask, gate_or_lambda, normalizer)

    total_c, l_forecast_c = combined_loss(y_hat_c, query_future, retrieval_loss_c, gamma=cli.gamma)
    agg_mse_c = float(((r_c - query_future) ** 2).mean().detach())
    return total_c, l_forecast_c, lam_c.detach(), retrieval_loss_c.detach(), agg_mse_c, feats_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--num_query_views', type=int, required=True, choices=VALID_NUM_QUERY_VIEWS)
    ap.add_argument('--arm', required=True, choices=['NOQA', 'QA'])
    ap.add_argument('--cell', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--base_checkpoint', required=True)
    ap.add_argument('--candidate_pool_cache', required=True)
    ap.add_argument('--patch_len', type=int, default=16)
    ap.add_argument('--top_k', type=int, default=TOP_K)
    ap.add_argument('--tau_t', type=float, default=0.1)
    ap.add_argument('--tau_s', type=float, default=0.1)
    ap.add_argument('--gamma', type=float, default=GAMMA_FIXED)
    ap.add_argument('--slot_std', type=float, default=1e-3)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--train_epochs', type=int, default=10)
    ap.add_argument('--init_seed', type=int, default=0)
    ap.add_argument('--loader_seed', type=int, default=0)
    ap.add_argument('--candidate_pool_size', type=int, default=100)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--checkpoints', required=True)
    ap.add_argument('--limit_batches', type=int, default=0, help='SMOKE/DEBUG ONLY')
    cli = ap.parse_args()

    if abs(cli.gamma - GAMMA_FIXED) > 1e-9:
        raise SystemExit(f'[ISSUE][ABORT] gamma is fixed at {GAMMA_FIXED} for this track (no sweep); got {cli.gamma}')

    pool_cfg = CandidatePoolConfig(mode='coarse_topk', size=cli.candidate_pool_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    arm_name = f'V{cli.num_query_views}'
    out_dir = Path(cli.out_dir) / cli.cell / arm_name / cli.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints) / cli.cell / arm_name / cli.arm
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    exp, args, model = build_model(cli, device)
    channels = list(range(int(args.enc_in)))
    init_hash = state_hash(model)
    d_model = int(args.d_model)

    slot_heads = None
    slot_heads_init_hash = None
    if cli.num_query_views >= 1:
        set_global_seeds(cli.init_seed)
        slot_heads = SlotHeads(d_model, n_slots=cli.num_query_views, std=cli.slot_std).to(device)
        slot_heads_init_hash = state_sha(slot_heads.state_dict())

    base = _load_base(cli.base_checkpoint, cli.seq_len, cli.pred_len, len(channels), device)

    set_global_seeds(cli.init_seed)
    if cli.arm == 'NOQA':
        gate_or_lambda = GlobalLambda().to(device)
        normalizer = None
    else:
        gate_or_lambda = QualityGate().to(device)
        normalizer = TrainFeatureNormalizer()
    fusion_init_hash = state_sha(gate_or_lambda.state_dict())

    expected_meta = {'seq_len': cli.seq_len, 'pred_len': cli.pred_len, 'candidate_mask': 'raft',
                     'candidate_pool_size': cli.candidate_pool_size, 'candidate_pool_metric': 'delta_last_cosine'}
    cache_dir = Path(cli.candidate_pool_cache)
    pool_caches = {}
    for split in ('train', 'val'):
        cache_path = cache_dir / f'{split}.pt'
        if not cache_path.exists():
            raise SystemExit(f'[ISSUE][ABORT] missing candidate pool cache file: {cache_path}')
        pool_caches[split] = CandidatePoolCache(cache_path, expected_meta)

    for p in model.parameters():
        p.requires_grad_(True)
    params = list(model.parameters()) + (list(slot_heads.parameters()) if slot_heads else []) \
        + list(gate_or_lambda.parameters())
    optimizer = torch.optim.Adam(params, lr=cli.learning_rate)

    # ---- train_gen/train_loader created ONCE, consumed ONLY by the real
    # training loop below. NOT touched by the QA fitting pass (fix for
    # the batch-order confound -- see module docstring). ----
    train_gen = make_loader_generator(cli.loader_seed)
    _, train_loader = exp._get_data(flag='train', shuffle=True, generator=train_gen)
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    # TEST loader is intentionally never constructed in this script.

    # ---- Fit the QA feature TRAIN normalizer once, before any training step,
    # on one fixed pass over a SEPARATE loader/generator (never `train_gen`),
    # with gate_or_lambda's initial (fairness-init) state. Never on VAL. ----
    if cli.arm == 'QA':
        fit_gen = make_loader_generator(cli.loader_seed + FIT_LOADER_SEED_OFFSET)
        _, fit_loader = exp._get_data(flag='train', shuffle=True, generator=fit_gen)
        feat_rows = []
        with torch.no_grad():
            for bi, (batch_x, batch_y, batch_start_idx) in enumerate(fit_loader):
                if bi >= 50:
                    break
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                cand_mask, _ = exp._candidate_mask(batch_start_idx)
                base_forecast_full = base(batch_x) + batch_x[:, -1:, :].detach()
                for c in channels:
                    scores, valid_mask, pool_idx_global = compute_scores(
                        pool_cfg, model, slot_heads, batch_x, exp, c, cand_mask,
                        pool_caches['train'], batch_start_idx, device)
                    memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
                    query_future = batch_y[:, :, c]
                    pooled_memory_c = gather_candidate_values(memory_c, pool_idx_global)
                    s_masked = scores.masked_fill(~valid_mask.unsqueeze(1), float('-inf'))
                    p_heads = torch.softmax(s_masked / cli.tau_s, dim=-1)
                    p_mix = p_heads.mean(dim=1)
                    picks = stable_topk_indices(p_mix.masked_fill(~valid_mask, float('-inf')), cli.top_k, largest=True)
                    y_sel = pooled_memory_c.gather(
                        1, picks.unsqueeze(-1).expand(-1, -1, pooled_memory_c.size(-1))) + offset_c.view(-1, 1, 1)
                    r_c, weights = soft_topk_forecast(p_mix, picks, y_sel)
                    base_c = base_forecast_full[:, :, c]
                    feats_c = compute_qa_features(base_c, r_c, y_sel, weights, p_mix, picks, valid_mask)
                    feat_rows.append(feats_c.cpu())
        all_feats = torch.cat(feat_rows, dim=0)
        normalizer.fit(all_feats).to(device)
        del fit_gen, fit_loader  # explicit: never reused, never touches train_gen's state

    config = {
        'exp': 'TRACK-P100-QA-CAUSAL-VALIDATION01', 'protocol': 'rng_corrected',
        'cell': cli.cell, 'arm_v': arm_name, 'arm_qa': cli.arm,
        'num_query_views': cli.num_query_views, 'gamma': cli.gamma, 'tau_t': cli.tau_t, 'tau_s': cli.tau_s,
        'top_k': cli.top_k, 'init_seed': cli.init_seed, 'loader_seed': cli.loader_seed,
        'init_hash': init_hash, 'slot_heads_init_hash': slot_heads_init_hash, 'fusion_init_hash': fusion_init_hash,
        'checkpoint_criterion': 'min VALIDATION final-forecast MSE (Y_hat vs Y), never TEST',
        'qa_feature_names': None if cli.arm == 'NOQA' else list(QA_FEATURE_NAMES),
        'rng_fix': 'QA feature-fitting pass uses a separate loader/generator, never train_gen',
    }
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2, default=str))
    print(f'[softtopk_e2e_v02] arm={arm_name}/{cli.arm} cell={cli.cell} init_hash={init_hash[:16]} '
         f'slot_heads_init_hash={str(slot_heads_init_hash)[:16]} fusion_init_hash={fusion_init_hash[:16]}')

    best = {'val_mse': float('inf'), 'epoch': -1}
    epoch_rows = []
    for epoch in range(1, cli.train_epochs + 1):
        model.train()
        if slot_heads is not None:
            slot_heads.train()
        gate_or_lambda.train()
        starts = []
        train_forecast_sum, train_n = 0.0, 0
        for bi, (batch_x, batch_y, batch_start_idx) in enumerate(train_loader):
            if cli.limit_batches and bi >= cli.limit_batches:
                break
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            starts.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx)
                         else torch.as_tensor(batch_start_idx))
            cand_mask, _ = exp._candidate_mask(batch_start_idx)
            with torch.no_grad():
                base_forecast_full = base(batch_x) + batch_x[:, -1:, :].detach()
            optimizer.zero_grad()
            for c in channels:
                total_c, l_forecast_c, lam_c, ret_c, agg_mse_c, _ = _channel_step(
                    model, slot_heads, pool_cfg, batch_x, batch_y, exp, c, cand_mask,
                    pool_caches['train'], batch_start_idx, device, args, cli, base_forecast_full,
                    gate_or_lambda, normalizer, cli.arm)
                (total_c / len(channels)).backward()
                train_forecast_sum += float(l_forecast_c.detach()) * batch_x.size(0)
                train_n += batch_x.size(0)
            optimizer.step()
        batch_order_hashes_epoch = batch_order_sha256(starts)

        model.eval()
        if slot_heads is not None:
            slot_heads.eval()
        gate_or_lambda.eval()
        val_forecast_sum, val_n, val_lambda_vals = 0.0, 0, []
        with torch.no_grad():
            for batch_x, batch_y, batch_start_idx in val_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                cand_mask, _ = exp._candidate_mask(batch_start_idx)
                base_forecast_full = base(batch_x) + batch_x[:, -1:, :].detach()
                for c in channels:
                    total_c, l_forecast_c, lam_c, ret_c, agg_mse_c, _ = _channel_step(
                        model, slot_heads, pool_cfg, batch_x, batch_y, exp, c, cand_mask,
                        pool_caches['val'], batch_start_idx, device, args, cli, base_forecast_full,
                        gate_or_lambda, normalizer, cli.arm)
                    val_forecast_sum += float(l_forecast_c.detach()) * batch_x.size(0)
                    val_n += batch_x.size(0)
                    val_lambda_vals.append(lam_c.cpu())
        val_mse = val_forecast_sum / max(val_n, 1)
        train_mse = train_forecast_sum / max(train_n, 1)
        lam_all = torch.cat(val_lambda_vals) if val_lambda_vals else torch.zeros(1)
        row = {'epoch': epoch, 'train_forecast_mse': train_mse, 'val_forecast_mse': val_mse,
              'val_lambda_mean': float(lam_all.mean()), 'val_lambda_std': float(lam_all.std()),
              'batch_order_sha256': batch_order_hashes_epoch}
        epoch_rows.append(row)
        print(f'[softtopk_e2e_v02] {arm_name}/{cli.arm} cell={cli.cell} epoch={epoch} '
             f'train_forecast_mse={train_mse:.6f} val_forecast_mse={val_mse:.6f} '
             f'val_lambda_mean={row["val_lambda_mean"]:.4f}')

        ckpt_path = ckpt_dir / f'checkpoint_epoch{epoch}.pth'
        torch.save({'model_state': model.state_dict(),
                   'slot_heads_state': slot_heads.state_dict() if slot_heads else None,
                   'fusion_state': gate_or_lambda.state_dict(),
                   'normalizer_state': normalizer.state_dict() if normalizer else None,
                   'epoch': epoch, 'val_forecast_mse': val_mse}, ckpt_path)
        if val_mse < best['val_mse']:
            best = {'val_mse': val_mse, 'epoch': epoch, 'checkpoint_path': str(ckpt_path)}

    (out_dir / 'epoch_metrics.json').write_text(json.dumps(epoch_rows, indent=2))
    (out_dir / 'checkpoint_selection.json').write_text(json.dumps(best, indent=2))
    print(f'[softtopk_e2e_v02] done. {arm_name}/{cli.arm} cell={cli.cell} best_epoch={best["epoch"]} '
         f'val_forecast_mse={best["val_mse"]:.6f}')


if __name__ == '__main__':
    main()
