#!/usr/bin/env python3
"""TRACK-P100-SOFTTOPK-QUALITY-AWARE01 -- end-to-end SoftTopK trainer for
Joint (Encoder + SlotHeads + Router, ALL trainable from scratch), NOQA
or QA arm.

Per spec section 12: Joint is NOT built as a 2-stage Frozen structure
(no pretrained-Hard initialization, unlike the earlier
TRACK-JOINT-AGGREGATE-ROUTED-MQ01 Joint trainer this reuses pieces
from) -- Encoder, SlotHeads (5, same architecture as Hard's), and the
AggregateRouter are all FRESH-initialized and trained jointly from the
very first step. The retrieval loss is purely
`L_route = KL(p_T || p_R)`, `p_R = sum_h pi_h(q,c) * p_h(i|q,c)` (the
Router-mixed distribution) -- no auxiliary router loss / alpha term
(that was a different, EARLIER track's hyperparameter; this spec never
asks for it). `p_R` is also exactly the `p_full` the Soft Top-K
forecast branch uses -- Top10(p_R) -> Soft Top-K weighted mean.

Reused UNMODIFIED: `compute_scores_pool_channel_first_grad`,
`encode_raw_channel_first`, `AggregateRouter` (fixed architecture, no
sweep), `kl_loss_from_prob`, `normalized_teacher_prob`,
`memory_value`/`gather_candidate_values`/`pooled_future_mse`,
`CandidatePoolCache`, `BaseForecastHead`, `build_model`/`state_hash`,
`SlotHeads`. NEW, already unit-tested: `soft_topk_forecast`/
`fuse_noqa`/`fuse_qa`/`combined_loss`, `compute_qa_features`/
`QualityGate`/`TrainFeatureNormalizer`.

Same channel-wise-backward memory pattern, VALIDATION
final-forecast-MSE checkpoint selection, TEST-never-constructed rule as
`train_softtopk_e2e_v01.py`/`train_softtopk_e2e_hard01.py`.
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
from scripts.train_retriever_pool01 import CandidatePoolCache
from utils.aggregate_router01 import AggregateRouter
from utils.candidate_pool import gather_candidate_values, pooled_future_mse
from utils.e2e_fusion_training01 import GlobalLambda, combined_loss, fuse_noqa, fuse_qa, soft_topk_forecast
from utils.full_candidate_bank import compute_scores_pool_channel_first_grad, encode_raw_channel_first
from utils.quality_aware_gate01 import QA_FEATURE_NAMES, QualityGate, TrainFeatureNormalizer, compute_qa_features

TOP_K = 10
NUM_SLOTS = 5
GAMMA_FIXED = 1.0


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


def _channel_step(model, slot_heads, router, batch_x, batch_y, exp, c, batch_start_idx, device, args, cli,
                  pool_cache, base_forecast_full, gate_or_lambda, normalizer, arm):
    scores, pool_valid_mask, pool_idx_global = compute_scores_pool_channel_first_grad(
        model, slot_heads, batch_x, exp, c, pool_cache, batch_start_idx, device)  # [B,5,M] grad ON
    memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
    query_future = batch_y[:, :, c]
    pooled_memory_c = gather_candidate_values(memory_c, pool_idx_global)
    d_pool = pooled_future_mse(pooled_memory_c, offset_c, query_future)
    with torch.no_grad():
        p_t = normalized_teacher_prob(d_pool, pool_valid_mask, cli.tau_t)

    p_h = torch.softmax(scores / cli.tau_s, dim=-1)  # [B,5,M] grad ON
    z_qc = encode_raw_channel_first(model, batch_x, c)  # grad ON
    router_logits = router(z_qc)
    pi = torch.softmax(router_logits, dim=-1)  # [B,5]
    p_r = (pi.unsqueeze(-1) * p_h).sum(dim=1)  # [B,M] grad ON through model, slot_heads, AND router

    retrieval_loss_c = kl_loss_from_prob(p_t, p_r, pool_valid_mask)

    picks = stable_topk_indices(p_r.masked_fill(~pool_valid_mask, float('-inf')), cli.top_k, largest=True)
    y_sel = pooled_memory_c.gather(
        1, picks.unsqueeze(-1).expand(-1, -1, pooled_memory_c.size(-1))) + offset_c.view(-1, 1, 1)
    r_c, weights = soft_topk_forecast(p_r, picks, y_sel)

    base_c = base_forecast_full[:, :, c]
    if arm == 'NOQA':
        y_hat_c, lam_c = fuse_noqa(base_c, r_c, gate_or_lambda)
        feats_c = None
    else:
        y_hat_c, lam_c, feats_c = fuse_qa(
            base_c, r_c, y_sel, weights, p_r, picks, pool_valid_mask, gate_or_lambda, normalizer)

    total_c, l_forecast_c = combined_loss(y_hat_c, query_future, retrieval_loss_c, gamma=cli.gamma)
    return total_c, l_forecast_c, lam_c.detach(), feats_c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
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
        raise SystemExit(f'[ISSUE][ABORT] gamma is fixed at {GAMMA_FIXED}; got {cli.gamma}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(cli.out_dir) / cli.cell / 'Joint' / cli.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints) / cli.cell / 'Joint' / cli.arm
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    exp, args, model = build_model(cli, device)
    channels = list(range(int(args.enc_in)))
    init_hash = state_hash(model)
    d_model = int(args.d_model)

    set_global_seeds(cli.init_seed)
    slot_heads = SlotHeads(d_model, n_slots=NUM_SLOTS, std=cli.slot_std).to(device)
    router = AggregateRouter(d_model).to(device)
    assert not any('gate' in n.lower() for n, _ in router.named_parameters())
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
    pool_caches = {s: CandidatePoolCache(cache_dir / f'{s}.pt', expected_meta) for s in ('train', 'val')}

    for p in model.parameters():
        p.requires_grad_(True)
    params = (list(model.parameters()) + list(slot_heads.parameters()) + list(router.parameters())
             + list(gate_or_lambda.parameters()))
    optimizer = torch.optim.Adam(params, lr=cli.learning_rate)

    train_gen = make_loader_generator(cli.loader_seed)
    _, train_loader = exp._get_data(flag='train', shuffle=True, generator=train_gen)
    _, val_loader = exp._get_data(flag='val', shuffle=False)

    if cli.arm == 'QA':
        feat_rows = []
        with torch.no_grad():
            for bi, (batch_x, batch_y, batch_start_idx) in enumerate(train_loader):
                if bi >= 50:
                    break
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                base_forecast_full = base(batch_x) + batch_x[:, -1:, :].detach()
                for c in channels:
                    scores, pool_valid_mask, pool_idx_global = compute_scores_pool_channel_first_grad(
                        model, slot_heads, batch_x, exp, c, pool_caches['train'], batch_start_idx, device)
                    memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
                    query_future = batch_y[:, :, c]
                    pooled_memory_c = gather_candidate_values(memory_c, pool_idx_global)
                    p_h = torch.softmax(scores / cli.tau_s, dim=-1)
                    z_qc = encode_raw_channel_first(model, batch_x, c)
                    pi = torch.softmax(router(z_qc), dim=-1)
                    p_r = (pi.unsqueeze(-1) * p_h).sum(dim=1)
                    picks = stable_topk_indices(p_r.masked_fill(~pool_valid_mask, float('-inf')), cli.top_k, largest=True)
                    y_sel = pooled_memory_c.gather(
                        1, picks.unsqueeze(-1).expand(-1, -1, pooled_memory_c.size(-1))) + offset_c.view(-1, 1, 1)
                    r_c, weights = soft_topk_forecast(p_r, picks, y_sel)
                    base_c = base_forecast_full[:, :, c]
                    feats_c = compute_qa_features(base_c, r_c, y_sel, weights, p_r, picks, pool_valid_mask)
                    feat_rows.append(feats_c.cpu())
        normalizer.fit(torch.cat(feat_rows, dim=0))

    config = {
        'exp': 'TRACK-P100-SOFTTOPK-QUALITY-AWARE01', 'cell': cli.cell, 'arm': 'Joint', 'qa_arm': cli.arm,
        'num_slots': NUM_SLOTS, 'gamma': cli.gamma, 'tau_t': cli.tau_t, 'tau_s': cli.tau_s, 'top_k': cli.top_k,
        'init_seed': cli.init_seed, 'loader_seed': cli.loader_seed, 'init_hash': init_hash,
        'fusion_init_hash': fusion_init_hash, 'fresh_init': True,
        'checkpoint_criterion': 'min VALIDATION final-forecast MSE (Y_hat vs Y), never TEST',
        'retrieval_loss': 'L_route = KL(p_T || p_R), p_R = router-mixed distribution (no aux alpha term)',
        'qa_feature_names': None if cli.arm == 'NOQA' else list(QA_FEATURE_NAMES),
    }
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2, default=str))
    print(f'[softtopk_e2e_joint] cell={cli.cell} arm={cli.arm} init_hash={init_hash[:16]} '
         f'fusion_init_hash={fusion_init_hash[:16]}')

    best = {'val_mse': float('inf'), 'epoch': -1}
    epoch_rows = []
    for epoch in range(1, cli.train_epochs + 1):
        model.train(); slot_heads.train(); router.train(); gate_or_lambda.train()
        starts = []
        train_forecast_sum, train_n = 0.0, 0
        for bi, (batch_x, batch_y, batch_start_idx) in enumerate(train_loader):
            if cli.limit_batches and bi >= cli.limit_batches:
                break
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            starts.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx)
                         else torch.as_tensor(batch_start_idx))
            with torch.no_grad():
                base_forecast_full = base(batch_x) + batch_x[:, -1:, :].detach()
            optimizer.zero_grad()
            for c in channels:
                total_c, l_forecast_c, lam_c, _ = _channel_step(
                    model, slot_heads, router, batch_x, batch_y, exp, c, batch_start_idx, device, args, cli,
                    pool_caches['train'], base_forecast_full, gate_or_lambda, normalizer, cli.arm)
                (total_c / len(channels)).backward()
                train_forecast_sum += float(l_forecast_c.detach()) * batch_x.size(0)
                train_n += batch_x.size(0)
            optimizer.step()
        batch_order_hashes_epoch = batch_order_sha256(starts)

        model.eval(); slot_heads.eval(); router.eval(); gate_or_lambda.eval()
        val_forecast_sum, val_n, val_lambda_vals = 0.0, 0, []
        with torch.no_grad():
            for batch_x, batch_y, batch_start_idx in val_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                base_forecast_full = base(batch_x) + batch_x[:, -1:, :].detach()
                for c in channels:
                    total_c, l_forecast_c, lam_c, _ = _channel_step(
                        model, slot_heads, router, batch_x, batch_y, exp, c, batch_start_idx, device, args, cli,
                        pool_caches['val'], base_forecast_full, gate_or_lambda, normalizer, cli.arm)
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
        print(f'[softtopk_e2e_joint] cell={cli.cell} arm={cli.arm} epoch={epoch} '
             f'train_forecast_mse={train_mse:.6f} val_forecast_mse={val_mse:.6f} '
             f'val_lambda_mean={row["val_lambda_mean"]:.4f}')

        ckpt_path = ckpt_dir / f'checkpoint_epoch{epoch}.pth'
        torch.save({'model_state': model.state_dict(), 'slot_heads_state': slot_heads.state_dict(),
                   'router_state': router.state_dict(), 'fusion_state': gate_or_lambda.state_dict(),
                   'normalizer_state': normalizer.state_dict() if normalizer else None,
                   'epoch': epoch, 'val_forecast_mse': val_mse}, ckpt_path)
        if val_mse < best['val_mse']:
            best = {'val_mse': val_mse, 'epoch': epoch, 'checkpoint_path': str(ckpt_path)}

    (out_dir / 'epoch_metrics.json').write_text(json.dumps(epoch_rows, indent=2))
    (out_dir / 'checkpoint_selection.json').write_text(json.dumps(best, indent=2))
    print(f'[softtopk_e2e_joint] done. cell={cli.cell} arm={cli.arm} best_epoch={best["epoch"]} '
         f'val_forecast_mse={best["val_mse"]:.6f}')


if __name__ == '__main__':
    main()
