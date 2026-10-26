#!/usr/bin/env python3
"""TRACK-P100-COARSE-TO-FINE-RETRIEVAL01 -- Stage2 unification cache
builder. Builds a `relation_outputs`/`query_start_idx` cache (the SAME
schema `scripts/train_r_stage2_lambda01.py::load_tensors` already
reads, reused UNMODIFIED for Stage2 itself) for each of the 5 arms
this track produced, all restricted to the SAME frozen Shared-P100
candidate identity:

  - V5            canonical sign-fixed V5, Mean-Mixture over P100 (beta=1 in Phase A's own terms)
  - PhaseA_fusion  Phase A's val-selected fixed beta, log-domain coarse+V5 fusion
  - PhaseB_adaptive  Phase B's adaptive-beta MLP -- RETRAINED here (its per-epoch
                    state was deliberately never checkpointed to disk, only the
                    aggregate JSON result; same architecture/training exactly as
                    `scripts/train_p100_adaptive_fusion01.py`, imported not duplicated)
  - PhaseC_residual  Phase C's trained Residual-V5 (frozen checkpoint, read-only)
  - PhaseD_setaware  Phase C's Residual-V5 AS the primary scorer, reranked within
                    its own Top-20 by Phase D1's trained set-aware reranker
                    (frozen checkpoint, read-only; greedy no-grad rollout at
                    inference, same `on_policy_rollout` function used in training)

Every arm's `relation_outputs[:, :, c]` is the SAME quantity --
mean of the 10 selected candidates' absolute future values -- so
`train_r_stage2_lambda01.py`'s `Y_final = B + lambda*(R-B)` Stage-2
gate is reused byte-for-byte identically downstream, never modified.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.diag_p100_coarse_v5_fusion01 import BETAS, EPS, TAU_C
from scripts.train_factorial_e2e01 import encode_raw
from scripts.train_j_shared_encoder_drift01 import build_model
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_margutil01 import memory_value
from scripts.train_p100_adaptive_fusion01 import (
    AdaptiveBetaMLP, N_BETA, N_FEATURES, collect_features_and_targets, extract_features, train_one_model,
)
from scripts.train_p100_set_reranker01 import (
    SetAwareReranker, candidate_features, on_policy_rollout, set_summary_features,
)
from scripts.train_retriever_pool01 import CandidatePoolCache
from utils.candidate_pool import compute_coarse_delta_last_cosine_scores

TOP_K = 10
TOP_M = 20
N_SLOTS = 5


@torch.no_grad()
def build_v5_like(exp, args, model, slot_heads, pool_cache, split, device, tau_s, fusion_beta=None, gamma=None):
    channels = list(range(int(args.enc_in)))
    _, loader = exp._get_data(flag=split, shuffle=False)
    # pass 1: deterministic batch order (shuffle=False) is identical across channel
    # re-iterations of the SAME loader object only if we rebuild/reuse it per channel
    # without randomness -- true here. Collect per-channel R lists, concat at the end.
    per_channel_R = {c: [] for c in channels}
    starts_ref = None
    for c in channels:
        k_full = F.normalize(encode_raw(model, exp.memory_x, c), dim=-1)
        starts_this = []
        for batch_x, batch_y, batch_start_idx in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            pool_idx_global = pool_cache.lookup(batch_start_idx, c, device)

            z_q = encode_raw(model, batch_x, c)
            q = slot_heads(z_q)
            k_pool = k_full[pool_idx_global]
            r = torch.einsum('bsd,bmd->bsm', q, k_pool)

            if gamma is not None:
                coarse_full = compute_coarse_delta_last_cosine_scores(batch_x, exp.memory_x, c)
                c_i = coarse_full.gather(1, pool_idx_global)
                s = c_i.unsqueeze(1) + gamma * r
                p_mix = F.softmax(s / tau_s, dim=-1).mean(dim=1)
            elif fusion_beta is not None:
                coarse_full = compute_coarse_delta_last_cosine_scores(batch_x, exp.memory_x, c)
                c_i = coarse_full.gather(1, pool_idx_global)
                p_c = F.softmax(c_i / TAU_C, dim=-1)
                p_v5 = F.softmax(r / tau_s, dim=-1).mean(dim=1)
                raw = (1 - fusion_beta) * torch.log(p_c + EPS) + fusion_beta * torch.log(p_v5 + EPS)
                p_mix = F.softmax(raw, dim=-1)
            else:
                p_mix = F.softmax(r / tau_s, dim=-1).mean(dim=1)

            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            pooled_future = memory_c[pool_idx_global] + offset_c.view(-1, 1, 1)
            picks = stable_topk_indices(p_mix, TOP_K, largest=True)
            y_sel = pooled_future.gather(1, picks.unsqueeze(-1).expand(-1, -1, pooled_future.size(-1)))
            per_channel_R[c].append(y_sel.mean(dim=1).cpu())
            starts_this.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx)
                               else torch.as_tensor(batch_start_idx))
        if starts_ref is None:
            starts_ref = torch.cat(starts_this)
        else:
            assert torch.equal(starts_ref, torch.cat(starts_this)), \
                '[ISSUE][ABORT] batch order differs across channels -- loader not deterministic'

    R_by_channel = [torch.cat(per_channel_R[c], dim=0) for c in channels]  # each [N,H]
    relation_outputs = torch.stack(R_by_channel, dim=-1)  # [N,H,C]
    return starts_ref, relation_outputs


@torch.no_grad()
def build_setaware(exp, args, model, slot_heads, gamma, reranker, pool_cache, split, device, tau_s):
    channels = list(range(int(args.enc_in)))
    _, loader = exp._get_data(flag=split, shuffle=False)
    per_channel_R = {c: [] for c in channels}
    starts_ref = None
    for c in channels:
        k_full = F.normalize(encode_raw(model, exp.memory_x, c), dim=-1)
        starts_this = []
        for batch_x, batch_y, batch_start_idx in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            pool_idx_global = pool_cache.lookup(batch_start_idx, c, device)

            coarse_full = compute_coarse_delta_last_cosine_scores(batch_x, exp.memory_x, c)
            c_i_full = coarse_full.gather(1, pool_idx_global)
            z_q = encode_raw(model, batch_x, c)
            q = slot_heads(z_q)
            k_pool = k_full[pool_idx_global]
            r_full = torch.einsum('bsd,bmd->bsm', q, k_pool)
            s_final_full = c_i_full.unsqueeze(1) + gamma * r_full
            p_mix_full = F.softmax(s_final_full / tau_s, dim=-1).mean(dim=1)
            r_mix_full = r_full.mean(dim=1)

            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            pooled_future_full = memory_c[pool_idx_global] + offset_c.view(-1, 1, 1)
            cand_hist = exp.memory_x[pool_idx_global][:, :, :, c]
            cand_temporal_summary_full = (cand_hist - cand_hist[:, :, -1:]).norm(dim=-1)

            top20_local = stable_topk_indices(p_mix_full, TOP_M, largest=True)
            gather20 = lambda t: t.gather(1, top20_local)  # noqa: E731
            primary_score = gather20(p_mix_full)
            coarse_score = gather20(c_i_full)
            residual_mix_score = gather20(r_mix_full)
            temporal_summary = gather20(cand_temporal_summary_full)
            top20_future = pooled_future_full.gather(1, top20_local.unsqueeze(-1).expand(-1, -1, pooled_future_full.size(-1)))
            z_cand = k_pool.gather(1, top20_local.unsqueeze(-1).expand(-1, -1, k_pool.size(-1)))

            _, picks20 = on_policy_rollout(reranker, z_q, z_cand, primary_score, coarse_score, residual_mix_score,
                                          temporal_summary, top20_future, query_future, TOP_K, 1.0, 0.5,
                                          target_kind='aggregate')
            y_sel = top20_future.gather(1, picks20.unsqueeze(-1).expand(-1, -1, top20_future.size(-1)))
            per_channel_R[c].append(y_sel.mean(dim=1).cpu())
            starts_this.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx)
                               else torch.as_tensor(batch_start_idx))
        if starts_ref is None:
            starts_ref = torch.cat(starts_this)
    R_by_channel = [torch.cat(per_channel_R[c], dim=0) for c in channels]
    relation_outputs = torch.stack(R_by_channel, dim=-1)
    return starts_ref, relation_outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--tau_s', type=float, default=0.1)
    ap.add_argument('--candidate_mask', default='raft')
    ap.add_argument('--candidate_pool_size', type=int, default=100)
    ap.add_argument('--candidate_pool_cache', required=True)
    ap.add_argument('--v5_checkpoint', required=True)
    ap.add_argument('--residual_checkpoint', required=True)
    ap.add_argument('--reranker_checkpoint', default=None,
                    help='required only for --arm PhaseD_setaware')
    ap.add_argument('--fusion_beta', type=float, default=None,
                    help='required only for --arm PhaseA_fusion (Phase A val-selected beta)')
    ap.add_argument('--arm', required=True,
                    choices=('V5', 'PhaseA_fusion', 'PhaseB_adaptive', 'PhaseC_residual', 'PhaseD_setaware'))
    ap.add_argument('--adaptive_epochs', type=int, default=30)
    ap.add_argument('--out_dir', required=True)
    cli = ap.parse_args()
    cli.init_seed = 0
    cli.patch_len = 16
    cli.top_k = TOP_K

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp, args, model = build_model(cli, device)
    exp.memory_x = exp.memory_x.to(device)
    model.eval()
    slot_heads = SlotHeads(int(args.d_model), n_slots=N_SLOTS).to(device)
    bl_v5 = torch.load(cli.v5_checkpoint, map_location=device)

    expected_meta = {'seq_len': cli.seq_len, 'pred_len': cli.pred_len, 'candidate_mask': cli.candidate_mask,
                     'candidate_pool_size': cli.candidate_pool_size, 'candidate_pool_metric': 'delta_last_cosine'}
    pool_caches = {s: CandidatePoolCache(Path(cli.candidate_pool_cache) / f'{s}.pt', expected_meta)
                  for s in ('train', 'val', 'test')}

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cli.arm in ('V5', 'PhaseA_fusion'):
        model.load_state_dict(bl_v5['model_state_dict'])
        slot_heads.load_state_dict(bl_v5['slot_heads_state_dict'])
        slot_heads.eval()
        fusion_beta = cli.fusion_beta if cli.arm == 'PhaseA_fusion' else None
        for split in ('train', 'val', 'test'):
            starts, rel = build_v5_like(exp, args, model, slot_heads, pool_caches[split], split, device,
                                                cli.tau_s, fusion_beta=fusion_beta)
            torch.save({'query_start_idx': starts, 'relation_outputs': rel}, out_dir / f'{split}.pt')
        print(f'[build_cache] {cli.arm} done for {cli.cell}')
        return

    if cli.arm == 'PhaseC_residual':
        bl_res = torch.load(cli.residual_checkpoint, map_location=device)
        model.load_state_dict(bl_res['model_state_dict'])
        slot_heads.load_state_dict(bl_res['slot_heads_state_dict'])
        slot_heads.eval()
        gamma = float(bl_res['gamma'])
        for split in ('train', 'val', 'test'):
            starts, rel = build_v5_like(exp, args, model, slot_heads, pool_caches[split], split, device,
                                                cli.tau_s, gamma=gamma)
            torch.save({'query_start_idx': starts, 'relation_outputs': rel}, out_dir / f'{split}.pt')
        print(f'[build_cache] PhaseC_residual done for {cli.cell} (gamma={gamma:.6f})')
        return

    if cli.arm == 'PhaseB_adaptive':
        model.load_state_dict(bl_v5['model_state_dict'])
        slot_heads.load_state_dict(bl_v5['slot_heads_state_dict'])
        slot_heads.eval()
        print(f'[build_cache] PhaseB_adaptive: retraining the adaptive-beta MLP for {cli.cell} '
             f'(never checkpointed in Phase B -- same architecture/procedure, reused not reimplemented)')
        train_data = collect_features_and_targets(cli, device, 'train', exp, args, model, slot_heads, pool_caches['train'])
        val_data = collect_features_and_targets(cli, device, 'val', exp, args, model, slot_heads, pool_caches['val'])
        tau_beta = float(train_data['agg_mse_by_beta'].std())
        beta_model, _ = train_one_model(lambda: AdaptiveBetaMLP(N_FEATURES, N_BETA), train_data, val_data, tau_beta,
                                        device, shuffle_features=False, epochs=cli.adaptive_epochs)
        beta_model.eval()
        beta_t = torch.tensor(BETAS, device=device)

        def build_split_adaptive(split):
            channels = list(range(int(args.enc_in)))
            _, loader = exp._get_data(flag=split, shuffle=False)
            per_channel_R = {c: [] for c in channels}
            starts_ref = None
            for c in channels:
                k_full = F.normalize(encode_raw(model, exp.memory_x, c), dim=-1)
                starts_this = []
                for batch_x, batch_y, batch_start_idx in loader:
                    batch_x = batch_x.float().to(device)
                    batch_y = batch_y.float().to(device)
                    pool_idx_global = pool_caches[split].lookup(batch_start_idx, c, device)
                    coarse_full = compute_coarse_delta_last_cosine_scores(batch_x, exp.memory_x, c)
                    c_i = coarse_full.gather(1, pool_idx_global)
                    p_c = F.softmax(c_i / TAU_C, dim=-1)
                    z_q = encode_raw(model, batch_x, c)
                    q = slot_heads(z_q)
                    k_pool = k_full[pool_idx_global]
                    scores = torch.einsum('bsd,bmd->bsm', q, k_pool)
                    p_heads = F.softmax(scores / cli.tau_s, dim=-1)
                    p_v5 = p_heads.mean(dim=1)
                    feats = extract_features(p_c, p_v5, p_heads, z_q, c, len(channels))
                    p_pred = beta_model(feats)
                    beta_expected = (p_pred * beta_t.unsqueeze(0)).sum(dim=-1)
                    raw = (1 - beta_expected.unsqueeze(-1)) * torch.log(p_c + EPS) + \
                        beta_expected.unsqueeze(-1) * torch.log(p_v5 + EPS)
                    p_fused = F.softmax(raw, dim=-1)
                    memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
                    pooled_future = memory_c[pool_idx_global] + offset_c.view(-1, 1, 1)
                    picks = stable_topk_indices(p_fused, TOP_K, largest=True)
                    y_sel = pooled_future.gather(1, picks.unsqueeze(-1).expand(-1, -1, pooled_future.size(-1)))
                    per_channel_R[c].append(y_sel.mean(dim=1).cpu())
                    starts_this.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx)
                                       else torch.as_tensor(batch_start_idx))
                if starts_ref is None:
                    starts_ref = torch.cat(starts_this)
            R_by_channel = [torch.cat(per_channel_R[c], dim=0) for c in channels]
            return starts_ref, torch.stack(R_by_channel, dim=-1)

        for split in ('train', 'val', 'test'):
            starts, rel = build_split_adaptive(split)
            torch.save({'query_start_idx': starts, 'relation_outputs': rel}, out_dir / f'{split}.pt')
        print(f'[build_cache] PhaseB_adaptive done for {cli.cell}')
        return

    if cli.arm == 'PhaseD_setaware':
        bl_res = torch.load(cli.residual_checkpoint, map_location=device)
        model.load_state_dict(bl_res['model_state_dict'])
        slot_heads.load_state_dict(bl_res['slot_heads_state_dict'])
        slot_heads.eval()
        gamma = float(bl_res['gamma'])
        reranker = SetAwareReranker(int(args.d_model)).to(device)
        bl_rr = torch.load(cli.reranker_checkpoint, map_location=device)
        reranker.load_state_dict(bl_rr)
        reranker.eval()
        for split in ('train', 'val', 'test'):
            starts, rel = build_setaware(exp, args, model, slot_heads, gamma, reranker, pool_caches[split],
                                         split, device, cli.tau_s)
            torch.save({'query_start_idx': starts, 'relation_outputs': rel}, out_dir / f'{split}.pt')
        print(f'[build_cache] PhaseD_setaware done for {cli.cell}')
        return


if __name__ == '__main__':
    main()
