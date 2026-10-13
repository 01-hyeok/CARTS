#!/usr/bin/env python3
"""TRACK-P100-SOFTTOPK-QUALITY-AWARE01 -- THE ONLY script in this track
that reads TEST. Refuses to run unless `--locked_config` points to an
existing `LOCKED_CONFIG.json` (spec section 22/23) -- this is a hard
gate, not a convention: see `test_refuses_without_locked_config` below
and in `tests/test_evaluate_locked_test01.py`.

Evaluates ONE (cell, method, arm) combination's ALREADY-SELECTED
checkpoint (per `checkpoint_selection.json`, chosen on VALIDATION only,
long before this script ever runs) on the TEST split, and reports every
metric spec section 24 requires: forecast MSE/MAE, retrieval
RetMSE@10/Recall@10/NDCG@10/SoftTopK Aggregate MSE, fusion lambda
stats + Base/Retrieval/Final-better fractions, and (QA arm only) the
QA-vs-NOQA delta (computed by the caller, which runs this script once
per arm and diffs the two `test_metrics.json` outputs -- this script
itself only ever evaluates ONE arm per invocation, so it cannot itself
produce a quantity that depends on having seen both).
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
from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob
from scripts.train_j_shared_encoder_drift01 import build_model
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_margutil01 import memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k
from scripts.train_retriever_pool01 import CandidatePoolCache, compute_scores
from utils.aggregate_router01 import AggregateRouter
from utils.candidate_pool import CandidatePoolConfig, gather_candidate_values, pooled_future_mse
from utils.e2e_fusion_training01 import GlobalLambda, fuse_noqa, fuse_qa, soft_topk_forecast
from utils.full_candidate_bank import compute_scores_pool_channel_first_grad, encode_raw_channel_first
from utils.quality_aware_gate01 import QualityGate, TrainFeatureNormalizer, compute_qa_features

TOP_K = 10


def _require_locked_config(path):
    p = Path(path)
    if not p.exists():
        raise SystemExit(
            f'[ISSUE][ABORT] --locked_config {path} does not exist. This script refuses to run '
            f'without an existing LOCKED_CONFIG.json (spec section 22/23) -- TEST may only be read '
            f'after the method/architecture/hyperparameters are frozen.')
    cfg = json.loads(p.read_text())
    for required in ('git_commit', 'datasets', 'horizons', 'method_list', 'test_accessed'):
        if required not in cfg:
            raise SystemExit(f'[ISSUE][ABORT] LOCKED_CONFIG.json missing required field: {required}')
    return cfg


def _channel_p_full(method, model, slot_heads, router, pool_cfg, batch_x, exp, c, cand_mask,
                    pool_cache, batch_start_idx, device, cli):
    if method == 'Joint':
        scores, valid_mask, pool_idx_global = compute_scores_pool_channel_first_grad(
            model, slot_heads, batch_x, exp, c, pool_cache, batch_start_idx, device)
        p_h = torch.softmax(scores / cli.tau_s, dim=-1)
        z_qc = encode_raw_channel_first(model, batch_x, c)
        pi = torch.softmax(router(z_qc), dim=-1)
        p_full = (pi.unsqueeze(-1) * p_h).sum(dim=1)
    elif method == 'Hard':
        scores, valid_mask, pool_idx_global = compute_scores_pool_channel_first_grad(
            model, slot_heads, batch_x, exp, c, pool_cache, batch_start_idx, device)
        s_masked = scores.masked_fill(~valid_mask.unsqueeze(1), float('-inf'))
        p_heads = torch.softmax(s_masked / cli.tau_s, dim=-1)
        p_full = p_heads.mean(dim=1)
    else:
        scores, valid_mask, pool_idx_global = compute_scores(
            pool_cfg, model, slot_heads, batch_x, exp, c, cand_mask, pool_cache, batch_start_idx, device)
        s_masked = scores.masked_fill(~valid_mask.unsqueeze(1), float('-inf'))
        p_heads = torch.softmax(s_masked / cli.tau_s, dim=-1)
        p_full = p_heads.mean(dim=1)
    return p_full, valid_mask, pool_idx_global


@torch.no_grad()
def evaluate(method, model, slot_heads, router, base, gate_or_lambda, normalizer, pool_cfg, exp, args,
            cli, channels, test_loader, pool_cache, device, arm):
    se_f, ae_f, n_f = 0.0, 0.0, 0
    retmse_sum, recall_sum, ndcg_sum, aggmse_sum, n_ch = 0.0, 0.0, 0.0, 0.0, 0
    lam_vals, base_better, retr_better, final_better = [], 0, 0, 0
    for batch_x, batch_y, batch_start_idx in test_loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        base_forecast_full = base(batch_x) + batch_x[:, -1:, :].detach()
        for c in channels:
            p_full, valid_mask, pool_idx_global = _channel_p_full(
                method, model, slot_heads, router, pool_cfg, batch_x, exp, c, cand_mask,
                pool_cache, batch_start_idx, device, cli)
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            pooled_memory_c = gather_candidate_values(memory_c, pool_idx_global)
            d_pool = pooled_future_mse(pooled_memory_c, offset_c, query_future)

            picks = stable_topk_indices(p_full.masked_fill(~valid_mask, float('-inf')), cli.top_k, largest=True)
            y_sel = pooled_memory_c.gather(
                1, picks.unsqueeze(-1).expand(-1, -1, pooled_memory_c.size(-1))) + offset_c.view(-1, 1, 1)
            r_c, weights = soft_topk_forecast(p_full, picks, y_sel)

            base_c = base_forecast_full[:, :, c]
            if arm == 'NOQA':
                y_hat_c, lam_c = fuse_noqa(base_c, r_c, gate_or_lambda)
            else:
                y_hat_c, lam_c, _ = fuse_qa(
                    base_c, r_c, y_sel, weights, p_full, picks, valid_mask, gate_or_lambda, normalizer)

            se_f += float(((y_hat_c - query_future) ** 2).sum())
            ae_f += float((y_hat_c - query_future).abs().sum())
            n_f += query_future.numel()

            oracle_idx = stable_topk_indices(d_pool, cli.top_k, largest=False)
            retmse_sum += float(d_pool.gather(1, picks).mean(dim=-1).sum())
            recall_sum += float(recall_at_k(picks, oracle_idx, cli.top_k).sum())
            ndcg_sum += float(ndcg_at_k(picks, d_pool, valid_mask, cli.top_k).sum())
            aggmse_sum += float(((r_c - query_future) ** 2).mean(dim=-1).sum())
            n_ch += batch_x.size(0)

            base_mse_c = ((base_c - query_future) ** 2).mean(dim=-1)
            retr_mse_c = ((r_c - query_future) ** 2).mean(dim=-1)
            final_mse_c = ((y_hat_c - query_future) ** 2).mean(dim=-1)
            base_better += int((base_mse_c < retr_mse_c).sum())
            retr_better += int((retr_mse_c < base_mse_c).sum())
            final_better += int((final_mse_c < base_mse_c).sum())
            lam_vals.append(lam_c.cpu())

    lam_all = torch.cat(lam_vals)
    sorted_lam, _ = torch.sort(lam_all)

    def pct(q):
        idx = min(int(q * (len(sorted_lam) - 1)), len(sorted_lam) - 1)
        return float(sorted_lam[idx])

    return {
        'test_mse': se_f / max(n_f, 1), 'test_mae': ae_f / max(n_f, 1),
        'retmse10': retmse_sum / max(n_ch, 1), 'recall10': recall_sum / max(n_ch, 1),
        'ndcg10': ndcg_sum / max(n_ch, 1), 'soft_topk_agg_mse': aggmse_sum / max(n_ch, 1),
        'lambda_mean': float(lam_all.mean()), 'lambda_std': float(lam_all.std()),
        'lambda_p10': pct(0.10), 'lambda_p50': pct(0.50), 'lambda_p90': pct(0.90),
        'base_better_fraction': base_better / max(n_ch, 1),
        'retrieval_better_fraction': retr_better / max(n_ch, 1),
        'final_better_than_base_fraction': final_better / max(n_ch, 1),
        'n_query_channel': n_ch,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--locked_config', required=True)
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--method', required=True, choices=['V0', 'V1', 'V2', 'V5', 'Hard', 'Joint'])
    ap.add_argument('--arm', required=True, choices=['NOQA', 'QA'])
    ap.add_argument('--cell', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--base_checkpoint', required=True)
    ap.add_argument('--candidate_pool_cache', required=True)
    ap.add_argument('--checkpoint_path', required=True, help='selected (val-chosen) checkpoint for this cell/method/arm')
    ap.add_argument('--patch_len', type=int, default=16)
    ap.add_argument('--top_k', type=int, default=TOP_K)
    ap.add_argument('--tau_s', type=float, default=0.1)
    ap.add_argument('--slot_std', type=float, default=1e-3)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--init_seed', type=int, default=0)
    ap.add_argument('--candidate_pool_size', type=int, default=100)
    ap.add_argument('--out_dir', required=True)
    cli = ap.parse_args()

    _require_locked_config(cli.locked_config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(cli.out_dir) / cli.cell / cli.method / cli.arm
    out_dir.mkdir(parents=True, exist_ok=True)

    cli.num_query_views = {'V0': 0, 'V1': 1, 'V2': 2, 'V5': 5}.get(cli.method, 5)
    exp, args, model = build_model(cli, device)
    channels = list(range(int(args.enc_in)))
    d_model = int(args.d_model)
    pool_cfg = CandidatePoolConfig(mode='coarse_topk', size=cli.candidate_pool_size)

    ckpt = torch.load(cli.checkpoint_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    slot_heads, router = None, None
    if cli.method == 'Joint':
        slot_heads = SlotHeads(d_model, n_slots=5, std=cli.slot_std).to(device)
        slot_heads.load_state_dict(ckpt['slot_heads_state'])
        router = AggregateRouter(d_model).to(device)
        router.load_state_dict(ckpt['router_state'])
    elif cli.method == 'Hard' or cli.num_query_views >= 1:
        n_slots = 5 if cli.method == 'Hard' else cli.num_query_views
        slot_heads = SlotHeads(d_model, n_slots=n_slots, std=cli.slot_std).to(device)
        slot_heads.load_state_dict(ckpt['slot_heads_state'])
    model.eval()
    if slot_heads is not None:
        slot_heads.eval()
    if router is not None:
        router.eval()

    base = BaseForecastHead(seq_len=cli.seq_len, pred_len=cli.pred_len, channels=len(channels),
                            mode='per_channel_linear').to(device)
    base_sd = torch.load(cli.base_checkpoint, map_location=device)
    base.load_state_dict(base_sd['model_state_dict'] if 'model_state_dict' in base_sd else base_sd)
    base.eval()

    if cli.arm == 'NOQA':
        gate_or_lambda = GlobalLambda().to(device)
        gate_or_lambda.load_state_dict(ckpt['fusion_state'])
        normalizer = None
    else:
        gate_or_lambda = QualityGate().to(device)
        gate_or_lambda.load_state_dict(ckpt['fusion_state'])
        normalizer = TrainFeatureNormalizer()
        normalizer.load_state_dict(ckpt['normalizer_state'])
    gate_or_lambda.eval()

    expected_meta = {'seq_len': cli.seq_len, 'pred_len': cli.pred_len, 'candidate_mask': 'raft',
                     'candidate_pool_size': cli.candidate_pool_size, 'candidate_pool_metric': 'delta_last_cosine'}
    pool_cache = CandidatePoolCache(Path(cli.candidate_pool_cache) / 'test.pt', expected_meta)
    _, test_loader = exp._get_data(flag='test', shuffle=False)  # the ONLY place this flag appears in this track

    metrics = evaluate(cli.method, model, slot_heads, router, base, gate_or_lambda, normalizer, pool_cfg,
                       exp, args, cli, channels, test_loader, pool_cache, device, cli.arm)
    metrics.update({'cell': cli.cell, 'method': cli.method, 'arm': cli.arm,
                   'checkpoint_path': cli.checkpoint_path})
    (out_dir / 'test_metrics.json').write_text(json.dumps(metrics, indent=2))
    print(f'[evaluate_locked_test] cell={cli.cell} method={cli.method} arm={cli.arm} '
         f'test_mse={metrics["test_mse"]:.6f} test_mae={metrics["test_mae"]:.6f}')


if __name__ == '__main__':
    main()
