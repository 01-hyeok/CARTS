#!/usr/bin/env python3
"""TRACK-P100-QA-CAUSAL-VALIDATION01 -- Phase C raw-tensor extraction.

For one (cell, method) RNG-corrected pair (the "clean" checkpoints,
free of the A1 batch-order confound -- Phase C uses these, not the
Original/confounded ones), runs ONE forward pass over VALIDATION (never
TEST) and saves, per query-channel row: Base forecast `B`, Soft Top-K
retrieval forecast `R`, true future `Y`, Soft Top-K weights, the QA
gate's own learned lambda and its 7 raw (unnormalized) features, and
the channel id (needed because the Val-Select/Val-Holdout split from
`utils.validation_split01` is defined on WINDOW index, not
query-channel row, so channel has to be carried through separately).

Uses the QA checkpoint's retriever (model+slot_heads+router) for BOTH
the NOQA-side and QA-side extraction where possible -- but per Phase
A1/B, NOQA and QA are two SEPARATELY TRAINED retrievers (same init,
different gradient paths), so this script extracts once per arm,
producing two aligned-by-query-channel-row tensor sets:
`noqa_raw.pt` (B, R_noqa, Y, weights_noqa) and
`qa_raw.pt` (B, R_qa, Y, weights_qa, qa_features, qa_lambda) --
`B` and `Y` are identical between the two files (same frozen Base
checkpoint, same data), kept duplicated for simplicity/self-containment
rather than factored out.
"""
import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from models.RelationStage2 import BaseForecastHead
from scripts.train_j_shared_encoder_drift01 import build_model
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_margutil01 import memory_value
from scripts.train_retriever_pool01 import CandidatePoolCache, compute_scores
from utils.aggregate_router01 import AggregateRouter
from utils.candidate_pool import CandidatePoolConfig, gather_candidate_values, pooled_future_mse
from utils.e2e_fusion_training01 import soft_topk_forecast
from utils.full_candidate_bank import compute_scores_pool_channel_first_grad, encode_raw_channel_first
from utils.quality_aware_gate01 import QualityGate, TrainFeatureNormalizer, compute_qa_features

TOP_K = 10
METHOD_TO_NQV = {'V0': 0, 'V1': 1, 'V2': 2, 'V5': 5}


@torch.no_grad()
def extract_one_arm(method, cli, arm, ckpt_path, base, pool_cache, device):
    cli.num_query_views = METHOD_TO_NQV.get(method, 5)
    exp, args, model = build_model(cli, device)
    channels = list(range(int(args.enc_in)))
    d_model = int(args.d_model)
    pool_cfg = CandidatePoolConfig(mode='coarse_topk', size=cli.candidate_pool_size)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    slot_heads, router = None, None
    if method == 'Joint':
        slot_heads = SlotHeads(d_model, n_slots=5, std=1e-3).to(device)
        slot_heads.load_state_dict(ckpt['slot_heads_state'])
        router = AggregateRouter(d_model).to(device)
        router.load_state_dict(ckpt['router_state'])
    elif method == 'Hard' or cli.num_query_views >= 1:
        n_slots = 5 if method == 'Hard' else cli.num_query_views
        slot_heads = SlotHeads(d_model, n_slots=n_slots, std=1e-3).to(device)
        slot_heads.load_state_dict(ckpt['slot_heads_state'])
    model.eval()
    if slot_heads is not None:
        slot_heads.eval()
    if router is not None:
        router.eval()

    gate, normalizer = None, None
    if arm == 'QA':
        gate = QualityGate().to(device)
        gate.load_state_dict(ckpt['fusion_state'])
        gate.eval()
        normalizer = TrainFeatureNormalizer()
        normalizer.load_state_dict(ckpt['normalizer_state'])
        normalizer.to(device)

    _, val_loader = exp._get_data(flag='val', shuffle=False)
    rows_B, rows_R, rows_Y, rows_W, rows_feat, rows_lam, rows_c, rows_start = [], [], [], [], [], [], [], []

    for batch_x, batch_y, batch_start_idx in val_loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx) if method not in ('Hard', 'Joint') else (None, None)
        base_forecast_full = base(batch_x) + batch_x[:, -1:, :].detach()
        for c in channels:
            if method == 'Joint':
                scores, valid_mask, pool_idx_global = compute_scores_pool_channel_first_grad(
                    model, slot_heads, batch_x, exp, c, pool_cache, batch_start_idx, device)
                p_h = torch.softmax(scores / 0.1, dim=-1)
                z_qc = encode_raw_channel_first(model, batch_x, c)
                pi = torch.softmax(router(z_qc), dim=-1)
                p_full = (pi.unsqueeze(-1) * p_h).sum(dim=1)
            elif method == 'Hard':
                scores, valid_mask, pool_idx_global = compute_scores_pool_channel_first_grad(
                    model, slot_heads, batch_x, exp, c, pool_cache, batch_start_idx, device)
                s_masked = scores.masked_fill(~valid_mask.unsqueeze(1), float('-inf'))
                p_heads = torch.softmax(s_masked / 0.1, dim=-1)
                p_full = p_heads.mean(dim=1)
            else:
                scores, valid_mask, pool_idx_global = compute_scores(
                    pool_cfg, model, slot_heads, batch_x, exp, c, cand_mask, pool_cache, batch_start_idx, device)
                s_masked = scores.masked_fill(~valid_mask.unsqueeze(1), float('-inf'))
                p_heads = torch.softmax(s_masked / 0.1, dim=-1)
                p_full = p_heads.mean(dim=1)

            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            pooled_memory_c = gather_candidate_values(memory_c, pool_idx_global)
            picks = stable_topk_indices(p_full.masked_fill(~valid_mask, float('-inf')), cli.top_k, largest=True)
            y_sel = pooled_memory_c.gather(
                1, picks.unsqueeze(-1).expand(-1, -1, pooled_memory_c.size(-1))) + offset_c.view(-1, 1, 1)
            r_c, weights = soft_topk_forecast(p_full, picks, y_sel)
            base_c = base_forecast_full[:, :, c]

            # compute_qa_features is retriever-agnostic (only needs B/R/y_sel/
            # weights/p_full/picks/valid_mask) -- saved for BOTH arms so Phase
            # C2 can later fit a NEW QualityGate on the NOQA-retriever's own
            # features too (isolating fusion-architecture effects from
            # retriever-training effects, per spec C1+C2). Only the LEARNED
            # lambda (this checkpoint's actual trained gate output) is QA-arm
            # -specific.
            feats_c = compute_qa_features(base_c, r_c, y_sel, weights, p_full, picks, valid_mask)
            rows_feat.append(feats_c.cpu())
            if arm == 'QA':
                lam_c = torch.sigmoid(gate.net(normalizer.transform(feats_c)).squeeze(-1))
                rows_lam.append(lam_c.cpu())

            rows_B.append(base_c.cpu()); rows_R.append(r_c.cpu()); rows_Y.append(query_future.cpu())
            rows_W.append(weights.cpu())
            rows_c.append(torch.full((base_c.size(0),), c, dtype=torch.long))
            rows_start.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx)
                              else torch.as_tensor(batch_start_idx))

    out = {
        'B': torch.cat(rows_B), 'R': torch.cat(rows_R), 'Y': torch.cat(rows_Y), 'weights': torch.cat(rows_W),
        'channel': torch.cat(rows_c), 'window_start_idx': torch.cat(rows_start),
        'qa_features_raw': torch.cat(rows_feat),  # retriever-agnostic, saved for BOTH arms
    }
    if arm == 'QA':
        out['qa_lambda'] = torch.cat(rows_lam)  # THIS checkpoint's own trained gate's output
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--method', required=True, choices=['V0', 'V1', 'V2', 'V5', 'Hard', 'Joint'])
    ap.add_argument('--cell', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--base_checkpoint', required=True)
    ap.add_argument('--candidate_pool_cache', required=True)
    ap.add_argument('--noqa_checkpoint', required=True)
    ap.add_argument('--qa_checkpoint', required=True)
    ap.add_argument('--candidate_pool_size', type=int, default=100)
    ap.add_argument('--top_k', type=int, default=TOP_K)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--patch_len', type=int, default=16)
    ap.add_argument('--init_seed', type=int, default=0)
    ap.add_argument('--loader_seed', type=int, default=0)
    ap.add_argument('--out_dir', required=True)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(cli.out_dir) / cli.cell / cli.method
    out_dir.mkdir(parents=True, exist_ok=True)

    base_sd = torch.load(cli.base_checkpoint, map_location=device)
    base_state = base_sd['model_state_dict'] if 'model_state_dict' in base_sd else base_sd

    expected_meta = {'seq_len': cli.seq_len, 'pred_len': cli.pred_len, 'candidate_mask': 'raft',
                     'candidate_pool_size': cli.candidate_pool_size, 'candidate_pool_metric': 'delta_last_cosine'}
    val_pool_cache = CandidatePoolCache(Path(cli.candidate_pool_cache) / 'val.pt', expected_meta)

    # build once to get channel count, then build BaseForecastHead
    exp_probe, args_probe, _ = build_model(cli, device)
    channels = list(range(int(args_probe.enc_in)))
    base = BaseForecastHead(seq_len=cli.seq_len, pred_len=cli.pred_len, channels=len(channels),
                            mode='per_channel_linear').to(device)
    base.load_state_dict(base_state)
    base.eval()

    noqa_raw = extract_one_arm(cli.method, cli, 'NOQA', cli.noqa_checkpoint, base, val_pool_cache, device)
    qa_raw = extract_one_arm(cli.method, cli, 'QA', cli.qa_checkpoint, base, val_pool_cache, device)

    torch.save(noqa_raw, out_dir / 'noqa_raw.pt')
    torch.save(qa_raw, out_dir / 'qa_raw.pt')
    print(f'[extract_fusion_diagnostics_raw] {cli.cell}/{cli.method}: '
         f'noqa_rows={noqa_raw["B"].shape[0]} qa_rows={qa_raw["B"].shape[0]} channels={len(channels)}')


if __name__ == '__main__':
    main()
