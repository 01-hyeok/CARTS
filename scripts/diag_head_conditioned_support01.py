#!/usr/bin/env python3
"""TRACK-HEAD-CONDITIONED-CANDIDATE-SUPPORT01 -- PHASE A: does
Shared-P100 suppress V5's own multi-head retrieval diversity? No
Stage-1 training -- the SAME canonical sign-fixed P100-trained V5
checkpoint (`TRACK-V-SHARED-TOP100-SIGNFIX01`) is scored against four
different CANDIDATE SUPPORTS, all exactly 100 candidates (except Full):

  S0 Shared-P100   the existing production pool cache (baseline, never regenerated)
  S1 MeanMix-P100  Top-100 by the SAME checkpoint's own Mean-Mixture p_mix
                   over the FULL valid candidate bank (model-aligned shortlist,
                   no per-head conditioning)
  S2 Head-P100     union of each head's own Top-20 (over the full bank),
                   deduplicated, topped up to exactly 100 by p_mix rank
  S3 Full          the full valid candidate bank, no truncation (upper reference)

Candidate validity mask / future-blindness / Mean-Mixture final
selection are IDENTICAL across all 4 supports -- only which
candidates are visible changes. Reuses UNMODIFIED: `encode_raw`
(`train_factorial_e2e01`), `SlotHeads` (`train_k_multislot_predictive_retrieval01`),
`memory_value` (`train_margutil01`), `recall_at_k`/`ndcg_at_k`
(`train_patch_retrieval_expert01`), `mean_mixture_topk_selection`
(`utils.mean_mixture_selection`), `CandidatePoolCache`
(`train_retriever_pool01`), `stable_topk_indices` (`RelationStage1`),
`build_model` (`train_j_shared_encoder_drift01` -- canonical MLP
backbone, same as V5's own training).
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
from scripts.train_factorial_e2e01 import encode_raw
from scripts.train_j_shared_encoder_drift01 import build_model
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_margutil01 import memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k
from scripts.train_retriever_pool01 import CandidatePoolCache
from utils.mean_mixture_selection import mean_mixture_topk_selection

TOP_K = 10
HEAD_TOPK = 20
BUDGET = 100
N_SLOTS = 5
SUPPORTS = ('shared_p100', 'meanmix_p100', 'head_p100', 'full')


def _dist(t):
    t = torch.as_tensor(t, dtype=torch.float32)
    return {'mean': float(t.mean()), 'std': float(t.std()), 'median': float(t.median()),
           'p25': float(t.quantile(0.25)), 'p75': float(t.quantile(0.75)),
           'min': float(t.min()), 'max': float(t.max())}


def build_head_p100_support(scores_full, cand_mask, p_mix):
    """scores_full: [B,5,N] (already masked to -inf on invalid). p_mix: [B,N]
    (already masked to 0 on invalid). Returns (support_idx [B,100] long,
    stats dict of per-batch-row construction diagnostics)."""
    bsz, n_slots, n = scores_full.shape
    head_top20 = [stable_topk_indices(scores_full[:, m, :], HEAD_TOPK, largest=True) for m in range(n_slots)]
    support_rows = []
    raw_union_sizes, unique_counts, topup_counts = [], [[] for _ in range(n_slots)], []
    p_mix_rank = p_mix.argsort(dim=-1, descending=True)  # [B,N], global idx sorted by p_mix desc

    for b in range(bsz):
        per_head_sets = [set(head_top20[m][b].tolist()) for m in range(n_slots)]
        union_set = set()
        for s in per_head_sets:
            union_set |= s
        raw_union_sizes.append(len(union_set))
        # unique contribution: candidates only this head brought (not in any other head's top20)
        for m in range(n_slots):
            others = set()
            for mm in range(n_slots):
                if mm != m:
                    others |= per_head_sets[mm]
            unique_counts[m].append(len(per_head_sets[m] - others))

        support = list(union_set)
        n_topup = 0
        if len(support) < BUDGET:
            for idx in p_mix_rank[b].tolist():
                if idx in union_set:
                    continue
                support.append(idx)
                union_set.add(idx)
                n_topup += 1
                if len(support) >= BUDGET:
                    break
        elif len(support) > BUDGET:
            # extremely unlikely (5*20=100 max), but keep budget exact: drop lowest-p_mix extras
            support = sorted(support, key=lambda i: -float(p_mix[b, i]))[:BUDGET]
        topup_counts.append(n_topup)
        support_rows.append(torch.tensor(sorted(support), dtype=torch.long))

    support_idx = torch.stack(support_rows, dim=0).to(scores_full.device)
    # duplicate_count = total raw picks (5*20) minus the union size actually achieved
    stats = {
        'raw_union_size': raw_union_sizes,
        'duplicate_count': [n_slots * HEAD_TOPK - u for u in raw_union_sizes],
        'per_head_unique_count': unique_counts,
        'topup_count': topup_counts,
    }
    return support_idx, stats


@torch.no_grad()
def run_cell(cli, device):
    exp, args, model = build_model(cli, device)
    exp.memory_x = exp.memory_x.to(device)
    model.eval()

    slot_heads = SlotHeads(int(args.d_model), n_slots=N_SLOTS).to(device)
    bl = torch.load(cli.v5_checkpoint, map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    slot_heads.load_state_dict(bl['slot_heads_state_dict'])
    slot_heads.eval()

    expected_meta = {'seq_len': cli.seq_len, 'pred_len': cli.pred_len, 'candidate_mask': cli.candidate_mask,
                     'candidate_pool_size': cli.candidate_pool_size, 'candidate_pool_metric': 'delta_last_cosine'}
    shared_pool_cache = CandidatePoolCache(Path(cli.candidate_pool_cache) / 'val.pt', expected_meta)

    channels = list(range(int(args.enc_in)))
    _, loader = exp._get_data(flag='val', shuffle=False)

    agg = {s: {'ret': [], 'agg': [], 'recall': [], 'ndcg': [], 'D': [], 'C': [],
              'oracle_ret': [], 'oracle_agg': [],
              'top1_overlap': [], 'top10_overlap': [], 'union_size': [], 'union_curves': [],
              'coverage_head_full': [[] for _ in range(N_SLOTS)], 'coverage_mix_full': []}
          for s in SUPPORTS}
    hc_stats_all = {'raw_union_size': [], 'duplicate_count': [], 'topup_count': [],
                    'per_head_unique_count': [[] for _ in range(N_SLOTS)]}

    for c in channels:
        # k_full depends only on (model, channel) -- never on the query batch --
        # so it is computed ONCE per channel here, not re-encoded on every batch
        # (that would be an N-candidate MLP forward pass repeated per batch for
        # nothing, since the candidate bank itself never changes across batches).
        k_full = F.normalize(encode_raw(model, exp.memory_x, c), dim=-1)  # [N,D]
        for batch_x, batch_y, batch_start_idx in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            bsz = batch_x.size(0)
            cand_mask, _ = exp._candidate_mask(batch_start_idx)  # [B,N]
            z_q = encode_raw(model, batch_x, c)
            q = slot_heads(z_q)  # [B,5,D]
            scores_full_raw = torch.einsum('bsd,nd->bsn', q, k_full)  # [B,5,N]
            scores_full = scores_full_raw.masked_fill(~cand_mask.unsqueeze(1), float('-inf'))

            p_heads_full = torch.softmax(scores_full / cli.tau_s, dim=-1)
            p_mix_full = p_heads_full.mean(dim=1)  # [B,N]

            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]

            # ---- full-memory reference Top-10s (per head + mix), for coverage metrics ----
            f_m = [stable_topk_indices(scores_full[:, m, :], TOP_K, largest=True) for m in range(N_SLOTS)]
            f_mix = stable_topk_indices(p_mix_full, TOP_K, largest=True)

            # ---- build 4 supports (as GLOBAL candidate indices) ----
            support_idx = {}
            support_idx['shared_p100'] = shared_pool_cache.lookup(batch_start_idx, c, device)
            support_idx['meanmix_p100'] = stable_topk_indices(p_mix_full, BUDGET, largest=True)
            hc_idx, hc_stats = build_head_p100_support(scores_full, cand_mask, p_mix_full)
            support_idx['head_p100'] = hc_idx
            # S3 full: represented by cand_mask itself (variable count), handled separately below

            hc_stats_all['raw_union_size'].extend(hc_stats['raw_union_size'])
            hc_stats_all['duplicate_count'].extend(hc_stats['duplicate_count'])
            hc_stats_all['topup_count'].extend(hc_stats['topup_count'])
            for m in range(N_SLOTS):
                hc_stats_all['per_head_unique_count'][m].extend(hc_stats['per_head_unique_count'][m])

            for sname in ('shared_p100', 'meanmix_p100', 'head_p100'):
                sidx = support_idx[sname]  # [B,100] global idx
                # restrict scores to this support
                s_scores = scores_full_raw.gather(2, sidx.unsqueeze(1).expand(-1, N_SLOTS, -1))  # [B,5,100]
                s_valid = torch.ones_like(sidx, dtype=torch.bool)
                s_pooled_future = memory_c[sidx] + offset_c.view(-1, 1, 1)  # [B,100,H]
                d_pool = ((s_pooled_future - query_future.unsqueeze(1)) ** 2).mean(dim=-1)  # [B,100]
                oracle_idx_local = stable_topk_indices(d_pool, TOP_K, largest=False)

                picks_local, p_heads_s, p_mix_s = mean_mixture_topk_selection(s_scores, s_valid, cli.tau_s, k=TOP_K)
                ind_mse = d_pool.gather(1, picks_local)
                y_sel = s_pooled_future.gather(1, picks_local.unsqueeze(-1).expand(-1, -1, s_pooled_future.size(-1)))
                r_c = y_sel.mean(dim=1)
                agg_mse = ((r_c - query_future) ** 2).mean(dim=-1)
                D_ = ind_mse.mean(dim=-1) / TOP_K
                C_ = agg_mse - D_
                recall = recall_at_k(picks_local, oracle_idx_local, TOP_K)
                ndcg = ndcg_at_k(picks_local, d_pool, s_valid, TOP_K)

                oracle_y = s_pooled_future.gather(
                    1, oracle_idx_local.unsqueeze(-1).expand(-1, -1, s_pooled_future.size(-1)))
                oracle_agg_pred = oracle_y.mean(dim=1)
                oracle_agg_mse = ((oracle_agg_pred - query_future) ** 2).mean(dim=-1)
                oracle_ret = d_pool.gather(1, oracle_idx_local).mean(dim=-1)

                a = agg[sname]
                a['ret'].append(ind_mse.mean(dim=-1).cpu()); a['agg'].append(agg_mse.cpu())
                a['recall'].append(recall.cpu()); a['ndcg'].append(ndcg.cpu())
                a['D'].append(D_.cpu()); a['C'].append(C_.cpu())
                a['oracle_ret'].append(oracle_ret.cpu()); a['oracle_agg'].append(oracle_agg_mse.cpu())

                # coverage: full-memory F_m / F_mix inside this support's global idx set
                sidx_set_cpu = sidx.cpu()
                for m in range(N_SLOTS):
                    fm_cpu = f_m[m].cpu()
                    cov = torch.tensor([len(set(fm_cpu[b].tolist()) & set(sidx_set_cpu[b].tolist())) / TOP_K
                                        for b in range(bsz)])
                    a['coverage_head_full'][m].append(cov)
                fmix_cpu = f_mix.cpu()
                cov_mix = torch.tensor([len(set(fmix_cpu[b].tolist()) & set(sidx_set_cpu[b].tolist())) / TOP_K
                                        for b in range(bsz)])
                a['coverage_mix_full'].append(cov_mix)

                # head complementarity WITHIN this support
                head_top10_local = [stable_topk_indices(s_scores[:, m, :], TOP_K, largest=True) for m in range(N_SLOTS)]
                head_top1_local = [stable_topk_indices(s_scores[:, m, :], 1, largest=True) for m in range(N_SLOTS)]
                pair_overlaps10, pair_overlaps1 = [], []
                for m in range(N_SLOTS):
                    for mm in range(m + 1, N_SLOTS):
                        inter10 = (head_top10_local[m].unsqueeze(-1) == head_top10_local[mm].unsqueeze(-2)).any(-1).float().sum(-1)
                        pair_overlaps10.append((inter10 / TOP_K))
                        inter1 = (head_top1_local[m] == head_top1_local[mm]).float().squeeze(-1)
                        pair_overlaps1.append(inter1)
                a['top10_overlap'].append(torch.stack(pair_overlaps10, dim=-1).mean(dim=-1).cpu())
                a['top1_overlap'].append(torch.stack(pair_overlaps1, dim=-1).mean(dim=-1).cpu())

                union_mask = torch.zeros(bsz, s_scores.size(-1), dtype=torch.bool, device=device)
                curve_steps = []
                for m in range(N_SLOTS):
                    union_mask.scatter_(1, head_top10_local[m], True)
                    oracle_in_union = union_mask.gather(1, oracle_idx_local)
                    curve_steps.append(oracle_in_union.float().mean(dim=-1))
                a['union_size'].append(union_mask.float().sum(dim=-1).cpu())
                a['union_curves'].append(torch.stack(curve_steps, dim=0).cpu())  # [5, B] -- B varies per batch

            # ---- S3 Full (variable-count, uses cand_mask directly) ----
            d_pool_full = ((memory_c.unsqueeze(0) + offset_c.view(-1, 1, 1) - query_future.unsqueeze(1)) ** 2
                          ).mean(dim=-1)  # [B,N]
            d_pool_full = d_pool_full.masked_fill(~cand_mask, float('inf'))
            oracle_idx_full = stable_topk_indices(d_pool_full, TOP_K, largest=False)
            picks_full, p_heads_f, p_mix_f = mean_mixture_topk_selection(scores_full, cand_mask, cli.tau_s, k=TOP_K)
            ind_mse = d_pool_full.gather(1, picks_full)
            y_sel = memory_c[picks_full] + offset_c.view(-1, 1, 1)
            r_c = y_sel.mean(dim=1)
            agg_mse = ((r_c - query_future) ** 2).mean(dim=-1)
            D_ = ind_mse.mean(dim=-1) / TOP_K
            C_ = agg_mse - D_
            recall = recall_at_k(picks_full, oracle_idx_full, TOP_K)
            ndcg = ndcg_at_k(picks_full, d_pool_full, cand_mask, TOP_K)
            oracle_y = memory_c[oracle_idx_full] + offset_c.view(-1, 1, 1)
            oracle_agg_mse = ((oracle_y.mean(dim=1) - query_future) ** 2).mean(dim=-1)
            oracle_ret = d_pool_full.gather(1, oracle_idx_full).mean(dim=-1)

            a = agg['full']
            a['ret'].append(ind_mse.mean(dim=-1).cpu()); a['agg'].append(agg_mse.cpu())
            a['recall'].append(recall.cpu()); a['ndcg'].append(ndcg.cpu())
            a['D'].append(D_.cpu()); a['C'].append(C_.cpu())
            a['oracle_ret'].append(oracle_ret.cpu()); a['oracle_agg'].append(oracle_agg_mse.cpu())
            for m in range(N_SLOTS):
                a['coverage_head_full'][m].append(torch.ones(bsz))  # trivially 1.0 (same scores, no truncation)
            a['coverage_mix_full'].append(torch.ones(bsz))
            head_top10_f = [stable_topk_indices(scores_full[:, m, :].masked_fill(~cand_mask, float('-inf')),
                                                TOP_K, largest=True) for m in range(N_SLOTS)]
            head_top1_f = [stable_topk_indices(scores_full[:, m, :].masked_fill(~cand_mask, float('-inf')),
                                               1, largest=True) for m in range(N_SLOTS)]
            pair_overlaps10, pair_overlaps1 = [], []
            for m in range(N_SLOTS):
                for mm in range(m + 1, N_SLOTS):
                    inter10 = (head_top10_f[m].unsqueeze(-1) == head_top10_f[mm].unsqueeze(-2)).any(-1).float().sum(-1)
                    pair_overlaps10.append(inter10 / TOP_K)
                    inter1 = (head_top1_f[m] == head_top1_f[mm]).float().squeeze(-1)
                    pair_overlaps1.append(inter1)
            a['top10_overlap'].append(torch.stack(pair_overlaps10, dim=-1).mean(dim=-1).cpu())
            a['top1_overlap'].append(torch.stack(pair_overlaps1, dim=-1).mean(dim=-1).cpu())
            union_mask = torch.zeros(bsz, scores_full.size(-1), dtype=torch.bool, device=device)
            curve_steps = []
            for m in range(N_SLOTS):
                union_mask.scatter_(1, head_top10_f[m], True)
                oracle_in_union = union_mask.gather(1, oracle_idx_full)
                curve_steps.append(oracle_in_union.float().mean(dim=-1))
            a['union_size'].append(union_mask.float().sum(dim=-1).cpu())
            a['union_curves'].append(torch.stack(curve_steps, dim=0).cpu())  # [5, B] -- B varies per batch

    out = {'cell': cli.cell}
    for sname in SUPPORTS:
        a = agg[sname]
        cov_head_mean = [float(torch.cat(a['coverage_head_full'][m]).mean()) for m in range(N_SLOTS)]
        out[sname] = {
            'retmse10': float(torch.cat(a['ret']).mean()), 'agg_mse10': float(torch.cat(a['agg']).mean()),
            'recall10': float(torch.cat(a['recall']).mean()), 'ndcg10': float(torch.cat(a['ndcg']).mean()),
            'D': float(torch.cat(a['D']).mean()), 'C': float(torch.cat(a['C']).mean()),
            'oracle_retmse10': float(torch.cat(a['oracle_ret']).mean()),
            'oracle_agg_mse10': float(torch.cat(a['oracle_agg']).mean()),
            'coverage_head_full_mean': cov_head_mean,
            'coverage_head_full_mean_avg': sum(cov_head_mean) / N_SLOTS,
            'coverage_head_full_min': min(cov_head_mean),
            'coverage_mix_full_mean': float(torch.cat(a['coverage_mix_full']).mean()),
            'pairwise_top1_overlap': float(torch.cat(a['top1_overlap']).mean()),
            'pairwise_top10_overlap': float(torch.cat(a['top10_overlap']).mean()),
            'union_size_mean': float(torch.cat(a['union_size']).mean()),
            'union_oracle_recall_by_n_heads': torch.cat(a['union_curves'], dim=1).mean(dim=1).tolist(),
        }
    out['head_p100_construction'] = {
        'raw_union_size': _dist(hc_stats_all['raw_union_size']),
        'duplicate_count': _dist(hc_stats_all['duplicate_count']),
        'topup_count': _dist(hc_stats_all['topup_count']),
        'per_head_unique_count_mean': [float(torch.tensor(hc_stats_all['per_head_unique_count'][m], dtype=torch.float32).mean())
                                       for m in range(N_SLOTS)],
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--top_k', type=int, default=TOP_K)
    ap.add_argument('--tau_s', type=float, default=0.1)
    ap.add_argument('--candidate_mask', default='raft')
    ap.add_argument('--candidate_pool_size', type=int, default=100)
    ap.add_argument('--candidate_pool_cache', required=True)
    ap.add_argument('--v5_checkpoint', required=True)
    ap.add_argument('--out_dir', required=True)
    cli = ap.parse_args()
    cli.init_seed = 0
    cli.patch_len = 16

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[audit] cell={cli.cell} V5 checkpoint = {cli.v5_checkpoint}')
    out = run_cell(cli, device)

    # S5/S6/S7 sanity asserts
    assert out['head_p100']['coverage_head_full_min'] >= 0.999, (
        f'[ISSUE][ABORT] Head-P100 does not contain each head\'s own full-memory Top10: '
        f'{out["head_p100"]["coverage_head_full_mean"]}')
    assert out['meanmix_p100']['coverage_mix_full_mean'] >= 0.999, (
        f'[ISSUE][ABORT] MeanMix-P100 does not contain the full-memory Mean-Mixture Top10: '
        f'{out["meanmix_p100"]["coverage_mix_full_mean"]}')
    print('[sanity] S6 (Head-P100 superset of each head\'s full Top10): PASS')
    print('[sanity] S7 (MeanMix-P100 superset of full Mean-Mixture Top10): PASS')

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'{cli.cell}_phaseA_support.json').write_text(json.dumps(out, indent=2))
    for s in SUPPORTS:
        r = out[s]
        print(f'[diag_head_conditioned] {cli.cell}/{s}: agg_mse10={r["agg_mse10"]:.6f} retmse10={r["retmse10"]:.6f} '
             f'top10_overlap={r["pairwise_top10_overlap"]:.4f} union_recall5={r["union_oracle_recall_by_n_heads"][-1]:.4f} '
             f'cov_head={r["coverage_head_full_mean_avg"]:.4f} cov_mix={r["coverage_mix_full_mean"]:.4f}')


if __name__ == '__main__':
    main()
