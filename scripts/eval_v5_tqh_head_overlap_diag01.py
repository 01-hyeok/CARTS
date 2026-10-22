#!/usr/bin/env python3
"""TRACK-V5-TQH-HEAD-OVERLAP-DIAG01 -- direct, same-metric head
-redundancy comparison between canonical sign-fixed V5 and corrected
TQH-5 on ETTh1_96/P100/seed0/test. Evaluation-only: no optimizer, no
backward, no checkpoint write, no retraining. GPU1, read-only.

Reuses, never reimplements:
- `scripts.train_t_pure_multislot01.slot_mechanism_diagnostics` for
  `slot_top1_overlap` / `slot_top10_overlap` / `mean_pairwise_slot_cosine`
  (the EXACT function TQH's own trainer already calls for these numbers).
- `scripts.train_retriever_pool01.compute_scores` for V5's own score
  path (coarse_topk/P100 branch, same code V5 was trained/selected with).
- `scripts.train_temporal_query_heads01.compute_scores_tqh` for TQH-5's
  own score path.
- `scripts.train_temporal_query_heads01.union_oracle_recall_curve` for
  the 1-5 head Union Oracle Recall curve (same deterministic head-order
  convention already used in the TQH report).
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
from scripts.train_j_shared_encoder_drift01 import build_model
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_margutil01 import memory_value
from scripts.train_retriever_pool01 import CandidatePoolCache, compute_scores
from scripts.train_t_pure_multislot01 import slot_mechanism_diagnostics
from scripts.train_temporal_query_heads01 import compute_scores_tqh, union_oracle_recall_curve
from utils.candidate_pool import CandidatePoolConfig, gather_candidate_values, pooled_future_mse

TOP_K = 10
N_SLOTS = 5


def _pair_overlap_distribution(scores, cand_mask, pair_k):
    """Fine-grained distribution (one value per query x channel-call x
    pair) of Top-`pair_k` overlap between every head pair -- same
    per-pair definition `slot_mechanism_diagnostics` averages into a
    single scalar, kept un-averaged here for Table 2's distribution
    stats."""
    bsz, s, n = scores.shape
    tops = []
    for m in range(s):
        s_m = scores[:, m, :].masked_fill(~cand_mask, float('-inf'))
        idx_m = stable_topk_indices(s_m, pair_k, largest=True)
        tops.append(idx_m)
    vals = []
    for i in range(s):
        for j in range(i + 1, s):
            match = (tops[i].unsqueeze(-1) == tops[j].unsqueeze(-2)).any(-1).float().sum(-1)
            vals.append(match / pair_k)
    return torch.stack(vals, dim=-1)  # [B, C(s,2)]


def _union_size(scores, cand_mask, top_k=TOP_K):
    bsz, s, n = scores.shape
    in_union = torch.zeros(bsz, n, dtype=torch.bool, device=scores.device)
    for m in range(s):
        s_m = scores[:, m, :].masked_fill(~cand_mask, float('-inf'))
        idx_m = stable_topk_indices(s_m, top_k, largest=True)
        in_union.scatter_(1, idx_m, True)
    return in_union.float().sum(dim=-1)  # [B]


@torch.no_grad()
def evaluate_model(tag, score_fn, exp, args, device, pool_cache, oracle_fn, mask_cache=None, patch_len=16):
    channels = list(range(int(args.enc_in)))
    _, loader = exp._get_data(flag='test', shuffle=False)
    pairwise_top10_vals, pairwise_top1_vals = [], []
    query_mean_overlap, union_sizes = [], []
    union_curves = []
    slot_top1_list, slot_top10_list, pairwise_cos_list = [], [], []
    pool_ids_sample = []

    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        bsz = batch_x.size(0)
        for c in channels:
            scores, valid_mask, pool_idx_global = score_fn(batch_x, exp, c, pool_cache, batch_start_idx, device,
                                                            mask_cache, patch_len)
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            pooled_memory_c = gather_candidate_values(memory_c, pool_idx_global)
            d_pool = pooled_future_mse(pooled_memory_c, offset_c, query_future)
            oracle_idx = stable_topk_indices(d_pool, TOP_K, largest=False)

            s_masked = scores.masked_fill(~valid_mask.unsqueeze(1), float('-inf'))
            p_m = torch.softmax(s_masked / 0.1, dim=-1)
            mech = slot_mechanism_diagnostics(scores, valid_mask, p_m, 0.1)
            slot_top1_list.append(mech['slot_top1_overlap'])
            slot_top10_list.append(mech['slot_top10_overlap'])
            pairwise_cos_list.append(mech['mean_pairwise_slot_cosine'])

            pw10 = _pair_overlap_distribution(scores, valid_mask, TOP_K)  # [B, 10]
            pw1 = _pair_overlap_distribution(scores, valid_mask, 1)
            pairwise_top10_vals.append(pw10.reshape(-1).cpu())
            pairwise_top1_vals.append(pw1.reshape(-1).cpu())
            query_mean_overlap.append(pw10.mean(dim=-1).cpu())

            union_sizes.append(_union_size(scores, valid_mask).cpu())
            curve, _ = union_oracle_recall_curve(scores, valid_mask, oracle_idx, TOP_K)
            union_curves.append(curve)

            pool_ids_sample.append((int(batch_start_idx[0]), c, pool_idx_global[0].cpu().clone()))

    pairwise_top10_vals = torch.cat(pairwise_top10_vals)
    pairwise_top1_vals = torch.cat(pairwise_top1_vals)
    query_mean_overlap = torch.cat(query_mean_overlap)
    union_sizes = torch.cat(union_sizes)
    union_curve_mean = torch.tensor(union_curves).mean(dim=0).tolist()

    def _dist(t):
        return {'mean': float(t.mean()), 'std': float(t.std()), 'p25': float(t.quantile(0.25)),
               'median': float(t.median()), 'p75': float(t.quantile(0.75)), 'min': float(t.min()),
               'max': float(t.max())}

    return {
        'tag': tag,
        'slot_top1_overlap': sum(slot_top1_list) / len(slot_top1_list),
        'slot_top10_overlap': sum(slot_top10_list) / len(slot_top10_list),
        'mean_pairwise_slot_cosine': sum(pairwise_cos_list) / len(pairwise_cos_list),
        'union_size_mean': float(union_sizes.mean()),
        'union_oracle_recall_by_n_heads': union_curve_mean,
        'pairwise_top10_overlap_distribution': _dist(pairwise_top10_vals),
        'pairwise_top1_overlap_distribution': _dist(pairwise_top1_vals),
        'query_level_mean_overlap_distribution': _dist(query_mean_overlap),
        'pool_ids_sample': pool_ids_sample[:3],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--candidate_pool_size', type=int, default=100)
    ap.add_argument('--candidate_pool_cache', required=True)
    ap.add_argument('--candidate_mask', default='raft')
    ap.add_argument('--patch_len', type=int, default=16)
    ap.add_argument('--v5_checkpoint', required=True)
    ap.add_argument('--tqh5_checkpoint', required=True)
    ap.add_argument('--out_dir', required=True)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cli.init_seed = cli.seed
    cli.top_k = TOP_K
    cli.batch_size = 32

    expected_meta = {'seq_len': cli.seq_len, 'pred_len': cli.pred_len, 'candidate_mask': cli.candidate_mask,
                     'candidate_pool_size': cli.candidate_pool_size, 'candidate_pool_metric': 'delta_last_cosine'}
    pool_cache = CandidatePoolCache(Path(cli.candidate_pool_cache) / 'test.pt', expected_meta)
    pool_cfg = CandidatePoolConfig(mode='coarse_topk', size=cli.candidate_pool_size)

    # ---- V5 ----
    exp_v5, args_v5, model_v5 = build_model(cli, device)
    exp_v5.memory_x = exp_v5.memory_x.to(device)
    model_v5.eval()
    slot_heads_v5 = SlotHeads(int(args_v5.d_model), n_slots=N_SLOTS).to(device)
    bl_v5 = torch.load(cli.v5_checkpoint, map_location=device)
    model_v5.load_state_dict(bl_v5['model_state_dict'])
    slot_heads_v5.load_state_dict(bl_v5['slot_heads_state_dict'])
    slot_heads_v5.eval()

    def v5_score_fn(batch_x, exp, c, pcache, start_idx, dev, _mc, _pl):
        cand_mask, _ = exp._candidate_mask(start_idx)
        scores, valid_mask, pool_idx_global = compute_scores(
            pool_cfg, model_v5, slot_heads_v5, batch_x, exp, c, cand_mask, pcache, start_idx, dev)
        return scores, valid_mask, pool_idx_global

    print(f'[audit] V5 checkpoint = {cli.v5_checkpoint}')
    res_v5 = evaluate_model('canonical_V5', v5_score_fn, exp_v5, args_v5, device, pool_cache, None)
    print(f'[v5_tqh_overlap_diag] V5 slot_top10_overlap={res_v5["slot_top10_overlap"]:.6f} '
         f'slot_top1_overlap={res_v5["slot_top1_overlap"]:.6f} union_size={res_v5["union_size_mean"]:.4f}')

    # ---- TQH-5 ----
    exp_tqh, args_tqh, model_tqh = build_model(cli, device)
    exp_tqh.memory_x = exp_tqh.memory_x.to(device)
    model_tqh.eval()
    slot_heads_tqh = SlotHeads(int(args_tqh.d_model), n_slots=N_SLOTS).to(device)
    bl_tqh = torch.load(cli.tqh5_checkpoint, map_location=device)
    model_tqh.load_state_dict(bl_tqh['model_state_dict'])
    slot_heads_tqh.load_state_dict(bl_tqh['slot_heads_state_dict'])
    slot_heads_tqh.eval()
    mask_cache = {}

    def tqh_score_fn(batch_x, exp, c, pcache, start_idx, dev, mc, pl):
        scores, valid_mask, pool_idx_global, _ = compute_scores_tqh(
            model_tqh, slot_heads_tqh, batch_x, exp, c, pcache, start_idx, dev, mc, pl)
        return scores, valid_mask, pool_idx_global

    print(f'[audit] TQH-5 checkpoint = {cli.tqh5_checkpoint}')
    res_tqh = evaluate_model('TQH5_corrected', tqh_score_fn, exp_tqh, args_tqh, device, pool_cache, None,
                             mask_cache, cli.patch_len)
    print(f'[v5_tqh_overlap_diag] TQH-5 slot_top10_overlap={res_tqh["slot_top10_overlap"]:.6f} '
         f'slot_top1_overlap={res_tqh["slot_top1_overlap"]:.6f} union_size={res_tqh["union_size_mean"]:.4f}')

    # S7 sanity: TQH-5 slot_top10_overlap should reproduce the original report's ~0.9769433165611625
    s7_ref = 0.9769433165611625
    s7_diff = abs(res_tqh['slot_top10_overlap'] - s7_ref)
    print(f'[S7 sanity] TQH-5 slot_top10_overlap diff vs original report = {s7_diff:.2e}')

    # S3: candidate pool id identity -- both models read the SAME CandidatePoolCache instance,
    # pool_idx_global depends only on (batch_start_idx, channel), never on model weights.
    s3_pass = all(torch.equal(a[2], b[2]) for a, b in zip(res_v5['pool_ids_sample'], res_tqh['pool_ids_sample']))
    print(f'[S3 sanity] candidate pool ids identical (V5 vs TQH, sampled) = {s3_pass}')

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in (res_v5, res_tqh):
        r.pop('pool_ids_sample', None)
    (out_dir / 'overlap_diag.json').write_text(json.dumps({
        'v5': res_v5, 'tqh5': res_tqh,
        'sanity': {'s7_diff_vs_original_report': s7_diff, 's3_pool_ids_identical': s3_pass},
    }, indent=2))
    print(f'[v5_tqh_overlap_diag] wrote {out_dir / "overlap_diag.json"}')


if __name__ == '__main__':
    main()
