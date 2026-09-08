#!/usr/bin/env python3
"""EXP-TEACHER-FORCING-DIAG01: before running EXP-ONPOLICY-PREFIX01 (T1),
directly check whether the EXISTING C0/R2 checkpoint (reused verbatim, NO
new training) can reproduce the Set Oracle's actual next choice when GIVEN
the correct oracle prefix, vs. how it behaves free-running (its own prefix).

Two evaluation modes on the SAME frozen checkpoint, same query subsample:

  A. Oracle-prefix: at every step t, the conditioning state and the "next
     choice" target both come from the oracle's own prefix S*_{t-1} (same
     `dense_utility` target this project's loss functions already use).
     Measures: can the model tell WHICH candidate the oracle would pick
     NEXT, given it is standing exactly where the oracle stands?

  B. Free-running: the model's own prefix S_hat_{t-1}, built from its own
     argmax picks (identical mechanism to every other free-running
     evaluation in this project -- `run_arm`/`run_sequence_dense` with
     teacher_idx=None). Measures actual deployed behavior.

This is a pure diagnostic: no gradient, no checkpoint is written, no model
is trained. `research/RESEARCH_CONTEXT.md`-documented finding this
diagnostic explicitly guards against over-interpreting: EXP-FIRSTANCHOR-DIAG
found NDCG@10 as high as 0.9628 on ETTh1 H96 coexisting with oracle-choice
rank median 489/top-10 containment 2.5% -- so this script ALWAYS reports
NDCG/Spearman (top-region quality) and oracle-rank/containment (exact
greedy-choice reproduction) as separate, not-mutually-implying metrics.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_firstanchor_diag import (_ndcg_at_k, a_weighted_prefix,
    encode, hard_agg, load_trained_selector)
from utils.dense_utility import candidate_weights, dense_utility
from utils.retrieval_diagnostics import load_stage2


def _spearman_top1pct(u_hat_valid, u_target_valid):
    n = u_hat_valid.numel()
    top1pct_n = max(int(0.01 * n), 2)
    true_order = torch.argsort(u_target_valid, descending=True)[:top1pct_n]
    uh_top = u_hat_valid[true_order]
    ut_top = u_target_valid[true_order]
    ra = uh_top.argsort().argsort().float() - (top1pct_n - 1) / 2
    rb = ut_top.argsort().argsort().float() - (top1pct_n - 1) / 2
    denom = (ra.norm() * rb.norm()).clamp_min(1e-12)
    return float((ra * rb).sum() / denom)


def _spearman_global(u_hat_valid, u_target_valid):
    ra = u_hat_valid.argsort().argsort().float()
    rb = u_target_valid.argsort().argsort().float()
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = (ra.norm() * rb.norm()).clamp_min(1e-12)
    return float((ra * rb).sum() / denom)


@torch.no_grad()
def evaluate(s2_ckpt, seq_ckpt, teacher_cache_path, k, chunk_size, rank_eval_queries, device, split='test'):
    b0_exp, b0_args = load_stage2(s2_ckpt)
    b0_exp._ensure_memory()
    b0_exp._build_key_bank()
    b0_model = b0_exp.model.module if hasattr(b0_exp.model, 'module') else b0_exp.model
    memory_x = torch.from_numpy(b0_exp.memory_bank.memory_x).float().to(device)
    memory_y = b0_exp.memory_y.to(device)
    memory_x_last = b0_exp.memory_x_last.to(device)

    seq_model, set_cond, empty_token, utility_head, seq_args = load_trained_selector(seq_ckpt, device)
    tau = float(seq_args.tau_topk)

    teacher = torch.load(teacher_cache_path, map_location='cpu')
    _, loader = b0_exp._get_data(flag=split, shuffle=False)
    channels = list(b0_model.target_channels())
    c = channels[0]  # single channel, matching the project's own per-channel diagnostic convention

    step_metrics_tf = {t: {'rank': [], 'top1': [], 'top5': [], 'top10': [], 'top50': [],
                            'ndcg10': [], 'ndcg50': [], 'spearman_top1pct': [], 'spearman_global': [],
                            'regret': []} for t in range(k)}
    step_metrics_on = {t: {'regret': [], 'divergence': [], 'overlap': []} for t in range(k)}
    first_divergence = []

    rows_seen = 0
    forced_oracle = {}
    forced_free = {}
    stage2_row_valid = None
    stage2_batches = []

    for batch_x, batch_y, batch_start_idx in loader:
        # NOTE: Stage-2 Final MSE (the primary metric) is accumulated over
        # the FULL test/train split, matching every other Stage-2 eval in
        # this project (eval_margutil01_stage2.py/eval_firstanchor_diag.py
        # never subsample for the MSE itself) -- only the detailed per-step
        # rank/NDCG/regret/divergence diagnostics below are subsampled to
        # `rank_eval_queries`, via `idx_take` being empty once the quota is
        # reached. Do NOT `break` the loader early or Stage-2 MSE would be
        # computed on a small, non-representative subset (a bug caught
        # during this script's first real-data run: the original version
        # broke early and produced a Stage-2 MSE ~0.30-0.35 that could not
        # be compared to any other experiment's full-test-set number).
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, counts = b0_exp._candidate_mask(batch_start_idx)
        valid_query = counts.to(device) >= k
        rows = [teacher['splits'][split]['start_to_row'][int(s)] for s in batch_start_idx.tolist()]

        E = encode(seq_model, memory_x, c)
        q = encode(seq_model, batch_x, c)
        teacher_idx_c = teacher['splits'][split]['teacher_idx'][int(c)][rows].to(device)
        vq = valid_query & (teacher_idx_c[:, 0] != -1)
        if not bool(vq.any()):
            continue
        take = max(0, min(int(vq.sum()), rank_eval_queries - rows_seen))
        idx_take = vq.nonzero(as_tuple=True)[0][:take]

        memory_c, offset_c = b0_model._memory_value(batch_x, memory_y, memory_x_last, c)
        all_tgt = memory_c + offset_c.view(-1, 1, 1)
        q_tgt = batch_y[:, :, c]
        w = candidate_weights(torch.matmul(q, E.transpose(0, 1)), cand_mask, tau)

        z_q_ref = b0_model._branch_embedding(batch_x, c, c)
        z_mem_ref = b0_model._branch_memory(b0_exp.key_bank, c, 0, c, z_q_ref.dtype, device)
        cosine_ref = torch.matmul(z_q_ref, z_mem_ref.transpose(0, 1))
        score_fn = b0_model._retrieval_score_fn()
        learned_ref = score_fn(z_q_ref, z_mem_ref) if score_fn is not None else cosine_ref

        # ============ A. Oracle-prefix evaluation ============
        selected_mask = torch.zeros_like(cand_mask)
        for t in range(k):
            prefix = teacher_idx_c[:, :t].clamp_min(0)
            if t == 0:
                m = empty_token(q.size(0), q.device, q.dtype)
            else:
                m = E[prefix].mean(dim=1)
            h = set_cond(q, m)
            u_hat = utility_head(h, E)
            valid_now = cand_mask & ~selected_mask
            a_dense = dense_utility(prefix, w, all_tgt, q_tgt, chunk_size=chunk_size)
            u_target = -a_dense

            for b in idx_take.tolist():
                valid_b = valid_now[b]
                if int(valid_b.sum()) < 5:
                    continue
                uh = u_hat[b][valid_b]
                ut = u_target[b][valid_b]
                ad = a_dense[b][valid_b]
                order = torch.argsort(uh, descending=True)
                i_star_local = int(ut.argmax())
                rank_of_star = int((order == i_star_local).nonzero())
                model_pick_local = int(order[0])
                step_metrics_tf[t]['rank'].append(rank_of_star)
                step_metrics_tf[t]['top1'].append(1.0 if rank_of_star == 0 else 0.0)
                step_metrics_tf[t]['top5'].append(1.0 if rank_of_star < 5 else 0.0)
                step_metrics_tf[t]['top10'].append(1.0 if rank_of_star < 10 else 0.0)
                step_metrics_tf[t]['top50'].append(1.0 if rank_of_star < 50 else 0.0)
                rel = ut - ut.min() + 1e-6
                rel_pred_order = rel[order]
                step_metrics_tf[t]['ndcg10'].append(_ndcg_at_k(rel_pred_order, 10))
                step_metrics_tf[t]['ndcg50'].append(_ndcg_at_k(rel_pred_order, 50))
                step_metrics_tf[t]['spearman_top1pct'].append(_spearman_top1pct(uh, ut))
                step_metrics_tf[t]['spearman_global'].append(_spearman_global(uh, ut))
                step_metrics_tf[t]['regret'].append(float(ad[model_pick_local] - ad[i_star_local]))

            nxt = teacher_idx_c[:, t:t + 1].clamp_min(0)
            selected_mask = selected_mask.scatter(1, nxt, True)
        forced_oracle[(int(c), 0)] = teacher_idx_c.clamp_min(0)

        # ============ B. Free-running evaluation ============
        selected_mask = torch.zeros_like(cand_mask)
        picks_list = []
        for t in range(k):
            if t == 0:
                m = empty_token(q.size(0), q.device, q.dtype)
                prefix = torch.zeros(q.size(0), 0, dtype=torch.long, device=device)
            else:
                prefix = torch.stack(picks_list, dim=1)
                m = E[prefix].mean(dim=1)
            h = set_cond(q, m)
            u_hat = utility_head(h, E)
            valid_now = cand_mask & ~selected_mask
            a_dense = dense_utility(prefix, w, all_tgt, q_tgt, chunk_size=chunk_size)
            u_masked = u_hat.masked_fill(~valid_now, float('-inf'))
            nxt = u_masked.argmax(dim=-1)

            for b in idx_take.tolist():
                valid_b = valid_now[b]
                if int(valid_b.sum()) < 1:
                    continue
                ad = a_dense[b][valid_b]
                best_local = int(ad.argmin())
                model_pick_global = int(nxt[b])
                model_pick_local = int((valid_b.nonzero(as_tuple=True)[0] == model_pick_global).nonzero())
                step_metrics_on[t]['regret'].append(float(ad[model_pick_local] - ad[best_local]))
                oracle_pick_t = int(teacher_idx_c[b, t].clamp_min(0))
                step_metrics_on[t]['divergence'].append(0.0 if model_pick_global == oracle_pick_t else 1.0)
                model_so_far = set(int(p[b]) for p in picks_list) | {model_pick_global}
                oracle_so_far = set(teacher_idx_c[b, :t + 1].clamp_min(0).tolist())
                step_metrics_on[t]['overlap'].append(len(model_so_far & oracle_so_far) / (t + 1))

            picks_list.append(nxt)
            selected_mask = selected_mask.scatter(1, nxt.unsqueeze(-1), True)
        picks_free = torch.stack(picks_list, dim=1)
        forced_free[(int(c), 0)] = picks_free

        for b in idx_take.tolist():
            model_row = picks_free[b].tolist()
            oracle_row = teacher_idx_c[b].clamp_min(0).tolist()
            div = k
            for t in range(k):
                if model_row[t] != oracle_row[t]:
                    div = t
                    break
            first_divergence.append(div)

        final_a_oracle = a_weighted_prefix(teacher_idx_c.clamp_min(0), learned_ref, all_tgt, q_tgt, tau)
        final_a_free = a_weighted_prefix(picks_free, learned_ref, all_tgt, q_tgt, tau)
        stage2_batches.append({
            'batch_x': batch_x, 'batch_y': batch_y, 'cand_mask': cand_mask, 'valid_query': valid_query,
            'forced_oracle': dict(forced_oracle), 'forced_free': dict(forced_free),
            'final_a_oracle': final_a_oracle[idx_take].tolist(), 'final_a_free': final_a_free[idx_take].tolist(),
        })
        forced_oracle = {}
        forced_free = {}
        rows_seen += take

    # ============ Stage-2 Final MSE (oracle-forced whole-sequence vs free-running whole-sequence) ============
    stage2_res = {'oracle_prefix': {'se': 0.0, 'n': 0.0}, 'free_running': {'se': 0.0, 'n': 0.0}}
    for batch in stage2_batches:
        for label, table in (('oracle_prefix', batch['forced_oracle']), ('free_running', batch['forced_free'])):
            b0_model.set_forced_selection(table)
            y_final, y_base, y_ret, beta, lam, debug = b0_exp.model(
                batch_x=batch['batch_x'], memory_y=b0_exp.memory_y, valid_mask=batch['cand_mask'],
                key_bank=b0_exp.key_bank, memory_x_last=b0_exp.memory_x_last,
                retrieval_cache=None, target_y=batch['batch_y'],
                teacher_key_bank=getattr(b0_exp, 'teacher_key_bank', None),
            )
            b0_model.set_forced_selection(None)
            yf, by = y_final[batch['valid_query']], batch['batch_y'][batch['valid_query']]
            stage2_res[label]['se'] += float((yf - by).pow(2).sum())
            stage2_res[label]['n'] += float(yf.numel())

    def summarize(step_dict, keys):
        out = {}
        for t in range(k):
            for kk in keys:
                vals = step_dict[t][kk]
                out[f't{t+1}_{kk}_mean'] = sum(vals) / max(len(vals), 1) if vals else float('nan')
                if kk in ('rank', 'regret'):
                    sv = sorted(vals)
                    out[f't{t+1}_{kk}_median'] = sv[len(sv) // 2] if sv else float('nan')
        return out

    tf_summary = summarize(step_metrics_tf, ['rank', 'top1', 'top5', 'top10', 'top50',
                                              'ndcg10', 'ndcg50', 'spearman_top1pct', 'spearman_global', 'regret'])
    on_summary = summarize(step_metrics_on, ['regret', 'divergence', 'overlap'])

    def overall_mean(step_dict, key):
        allv = [v for t in range(k) for v in step_dict[t][key]]
        return sum(allv) / max(len(allv), 1) if allv else float('nan')

    all_final_a_oracle = [v for batch in stage2_batches for v in batch['final_a_oracle']]
    all_final_a_free = [v for batch in stage2_batches for v in batch['final_a_free']]

    result = {
        'oracle_prefix_summary': {
            'rank_mean': overall_mean(step_metrics_tf, 'rank'),
            'top1_acc': overall_mean(step_metrics_tf, 'top1'),
            'top5_containment': overall_mean(step_metrics_tf, 'top5'),
            'top10_containment': overall_mean(step_metrics_tf, 'top10'),
            'top50_containment': overall_mean(step_metrics_tf, 'top50'),
            'ndcg10_mean': overall_mean(step_metrics_tf, 'ndcg10'),
            'ndcg50_mean': overall_mean(step_metrics_tf, 'ndcg50'),
            'spearman_top1pct_mean': overall_mean(step_metrics_tf, 'spearman_top1pct'),
            'spearman_global_mean': overall_mean(step_metrics_tf, 'spearman_global'),
            'mean_step_regret': overall_mean(step_metrics_tf, 'regret'),
            'final_A_weighted_mean': sum(all_final_a_oracle) / max(len(all_final_a_oracle), 1),
            'stage2_final_mse': stage2_res['oracle_prefix']['se'] / max(stage2_res['oracle_prefix']['n'], 1),
            'per_step': tf_summary,
        },
        'free_running_summary': {
            'mean_step_regret': overall_mean(step_metrics_on, 'regret'),
            'mean_divergence_rate': overall_mean(step_metrics_on, 'divergence'),
            'mean_prefix_overlap': overall_mean(step_metrics_on, 'overlap'),
            'first_divergence_step_mean': sum(first_divergence) / max(len(first_divergence), 1) if first_divergence else float('nan'),
            'final_A_weighted_mean': sum(all_final_a_free) / max(len(all_final_a_free), 1),
            'stage2_final_mse': stage2_res['free_running']['se'] / max(stage2_res['free_running']['n'], 1),
            'per_step': on_summary,
        },
        'n_queries': rows_seen,
        'split': split,
        'channel': int(c),
    }
    return result, step_metrics_tf, step_metrics_on, first_divergence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage2_checkpoint', required=True)
    ap.add_argument('--sequential_checkpoint', required=True)
    ap.add_argument('--teacher_cache', required=True)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--rank_eval_queries', type=int, default=200)
    ap.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    ap.add_argument('--out_dir', required=True)
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    result, step_tf, step_on, first_div = evaluate(
        args.stage2_checkpoint, args.sequential_checkpoint, args.teacher_cache,
        args.top_k, args.chunk_size, args.rank_eval_queries, device, args.split)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f'oracle_prefix_summary_{args.split}.json', 'w') as fh:
        json.dump(result['oracle_prefix_summary'], fh, indent=2)
    with open(out_dir / f'free_running_summary_{args.split}.json', 'w') as fh:
        json.dump(result['free_running_summary'], fh, indent=2)
    with open(out_dir / f'full_result_{args.split}.json', 'w') as fh:
        json.dump(result, fh, indent=2)

    with open(out_dir / f'step_metrics_{args.split}.csv', 'w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['mode', 'step', 'metric', 'value'])
        for t in range(args.top_k):
            for kk, vals in step_tf[t].items():
                v = sum(vals) / max(len(vals), 1) if vals else float('nan')
                writer.writerow(['oracle_prefix', t + 1, kk, v])
            for kk, vals in step_on[t].items():
                v = sum(vals) / max(len(vals), 1) if vals else float('nan')
                writer.writerow(['free_running', t + 1, kk, v])

    print(json.dumps({'oracle_prefix_summary': result['oracle_prefix_summary'],
                       'free_running_summary': result['free_running_summary']}, indent=2, default=str))
    print(f'[teacher_forcing_diag01] written to {out_dir}')


if __name__ == '__main__':
    main()
