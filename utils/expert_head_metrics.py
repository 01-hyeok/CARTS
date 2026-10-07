"""TRACK-EXPERT-V5-FULL01 -- shared per-head diagnostic/loss primitives.

Used by BOTH the new Expert-V5 trainer and the standalone-head
evaluator applied to the EXISTING canonical Current-V5 checkpoint, so
the two are compared through byte-identical measurement code. Full
candidate support only (N candidates, never pruned); nothing here
subsets/prefilters/caches candidates -- nothing here touches the
encoder call pattern at all, it only consumes an already-computed
`scores: [B, S, N]` tensor exactly as `compute_scores_full_grad`
produces it.
"""
import json
from pathlib import Path

import torch

from models.RelationStage1 import stable_topk_indices
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k

EPS = 1e-8


def per_head_standalone_topk(scores, cand_mask, k=10):
    """scores: [B,S,N]. Each head ranks ONLY its own score row -- no
    round-robin, no forced cross-head uniqueness. Returns [B,S,k]."""
    bsz, s, n = scores.shape
    out = torch.zeros(bsz, s, k, dtype=torch.long, device=scores.device)
    for h in range(s):
        masked = scores[:, h, :].masked_fill(~cand_mask, float('-inf'))
        out[:, h, :] = stable_topk_indices(masked, k, largest=True)
    return out


def per_head_future_utility(head_topk_idx, d_raw):
    """U[b,h] = mean future-MSE (d_raw, lower=better) over head h's own
    standalone Top-K. Caller is expected to wrap this in `torch.no_grad()`
    per spec section 4.2 (expert-assignment inputs are never
    gradient-sources)."""
    bsz, s, k = head_topk_idx.shape
    out = torch.zeros(bsz, s, device=d_raw.device)
    for h in range(s):
        out[:, h] = d_raw.gather(1, head_topk_idx[:, h, :]).mean(dim=-1)
    return out


def per_head_decomposition(head_topk_idx, cand_mask, memory_c, offset_c, query_future, d_raw, oracle_idx,
                           top_k=10):
    """Per-head standalone retrieval-quality decomposition, same D/C
    convention as `train_t_pure_multislot01.hard_eval_decomposition`
    (D = mean_individual_mse / top_k; C = agg_mse - D) applied
    independently to each head's own Top-K. Returns a dict of [B,S]
    tensors: retmse10, agg_mse, D, C, recall10, ndcg10."""
    bsz, s, k = head_topk_idx.shape
    out = {name: torch.zeros(bsz, s, device=d_raw.device)
          for name in ('retmse10', 'agg_mse', 'D', 'C', 'recall10', 'ndcg10')}
    for h in range(s):
        idx_h = head_topk_idx[:, h, :]
        ind_mse_i = d_raw.gather(1, idx_h)
        retmse_h = ind_mse_i.mean(dim=-1)
        y_sel = memory_c[idx_h] + offset_c.view(-1, 1, 1)
        agg_pred = y_sel.mean(dim=1)
        agg_mse_h = ((agg_pred - query_future) ** 2).mean(dim=-1)
        D_h = ind_mse_i.mean(dim=-1) / top_k
        C_h = agg_mse_h - D_h
        out['retmse10'][:, h] = retmse_h
        out['agg_mse'][:, h] = agg_mse_h
        out['D'][:, h] = D_h
        out['C'][:, h] = C_h
        out['recall10'][:, h] = recall_at_k(idx_h, oracle_idx, top_k)
        out['ndcg10'][:, h] = ndcg_at_k(idx_h, d_raw, cand_mask, top_k)
    return out


def responsibility_from_utility(U, tau_e=1.0, eps=EPS):
    """Spec section 5: z-score U across heads per query, then
    softmax(-z/tau_e). ALWAYS detached -- this is an assignment signal,
    never a gradient path (spec section 4.2 / unit test 10)."""
    mu = U.mean(dim=1, keepdim=True)
    sigma = U.std(dim=1, keepdim=True, unbiased=False)
    z = (U - mu) / (sigma + eps)
    r = torch.softmax(-z / tau_e, dim=1)
    return r.detach()


def kl_per_head(p_t, scores, cand_mask, tau_s, eps=EPS):
    """Spec section 6: KL(p_T || p_h) kept PER (query, head) -- never
    reduced to a scalar here (that reduction only happens in
    `expert_weighted_loss`, after the responsibility weighting).
    Returns (kl: [B,S], p_h: [B,S,N]) -- p_h returned for diagnostics
    (entropy, spread) so callers never need to recompute softmax."""
    s_masked = scores.masked_fill(~cand_mask.unsqueeze(1), float('-inf'))
    log_p_h = torch.log_softmax(s_masked / tau_s, dim=-1)
    p_h = log_p_h.exp()
    p_t_d = p_t.detach()
    log_p_t = torch.log(p_t_d.clamp_min(eps))
    term = (p_t_d.unsqueeze(1) * (log_p_t.unsqueeze(1) - log_p_h))
    term = term.masked_fill(~cand_mask.unsqueeze(1), 0.0)
    kl = term.sum(dim=-1)  # [B,S]
    return kl, p_h


def expert_weighted_loss(responsibility, kl_vals):
    """Spec section 7: L_expert = mean_q [ sum_h r_h(q) * KL_h(q) ].
    The ONLY loss term -- no auxiliary terms are added by this function
    or anywhere else in this track."""
    return (responsibility * kl_vals).sum(dim=1).mean()


def hard_expert_loss(kl_vals, winner_idx):
    """TRACK-HARD-EXPERT-V5-FULL01 spec section 4: L_hard(q) =
    KL(p_T(q) || p_h*(q)(q)), batch mean, where h*(q) = argmin_h U_h(q).
    `winner_idx` MUST already be computed under `torch.no_grad()`
    (e.g. via `winner_margin_stats(U)[0]` or `U.argmin(dim=1)`) -- this
    function only gathers the winning column of `kl_vals` and does not
    itself detach anything, so a non-detached `winner_idx` would be a
    caller bug, not something this function can guard against (indices
    carry no gradient regardless, but the CALLER is responsible for
    making sure `U`/`winner_idx` never backprop through the assignment
    itself, per spec section 4.2's detach requirement). This is the
    ONLY loss term in TRACK-HARD-EXPERT-V5-FULL01 -- no auxiliary term
    is added here or anywhere else in that track."""
    winner_kl = kl_vals.gather(dim=1, index=winner_idx.unsqueeze(1)).squeeze(1)
    return winner_kl.mean()


def head_pairwise_overlap(head_topk_idx):
    """head_topk_idx: [B,S,k]. Returns ({(h,j): overlap[B]} for h<j,
    union_size[B]) -- Top-K overlap fraction and per-query union size
    across all S standalone head picks."""
    bsz, s, k = head_topk_idx.shape
    overlaps = {}
    for h in range(s):
        for j in range(h + 1, s):
            match = (head_topk_idx[:, h, :].unsqueeze(-1) ==
                    head_topk_idx[:, j, :].unsqueeze(-2)).any(-1).float().sum(-1)
            overlaps[(h, j)] = match / k
    union_sizes = torch.zeros(bsz, device=head_topk_idx.device)
    for b in range(bsz):
        u = set()
        for h in range(s):
            u |= set(head_topk_idx[b, h].tolist())
        union_sizes[b] = len(u)
    return overlaps, union_sizes


def winner_margin_stats(U):
    """U: [B,S] (lower=better). Returns (winner_idx[B], U_best[B],
    U_second[B], margin[B] = U_second - U_best)."""
    sorted_U, sorted_idx = U.sort(dim=1)
    return sorted_idx[:, 0], sorted_U[:, 0], sorted_U[:, 1], sorted_U[:, 1] - sorted_U[:, 0]


def full_head_diagnostics(scores, cand_mask, memory_c, offset_c, query_future, d_raw, p_t, tau_s, top_k,
                          tau_e=1.0):
    """Everything needed for the per-head report, computed ONCE per
    (batch, channel) from an already-computed `scores: [B,S,N]` tensor
    -- no extra encoder calls, usable identically whether `scores` came
    from the Expert-V5 loss path or a plain Current-V5 checkpoint."""
    from scripts.train_t_pure_multislot01 import spearman_batch
    with torch.no_grad():
        head_topk_idx = per_head_standalone_topk(scores, cand_mask, k=top_k)
        U = per_head_future_utility(head_topk_idx, d_raw)
        responsibility = responsibility_from_utility(U, tau_e=tau_e)
        oracle_idx = stable_topk_indices(d_raw.masked_fill(~cand_mask, float('inf')), top_k, largest=False)
        decomp = per_head_decomposition(head_topk_idx, cand_mask, memory_c, offset_c, query_future, d_raw,
                                        oracle_idx, top_k)
        winner, u_best, u_second, margin = winner_margin_stats(U)
        overlaps, union_size = head_pairwise_overlap(head_topk_idx)
    kl_vals, p_h = kl_per_head(p_t, scores, cand_mask, tau_s)
    with torch.no_grad():
        entropy, spread = per_head_entropy_and_spread(p_h, cand_mask)
        spearman_h = [spearman_batch(scores[:, h, :], d_raw, cand_mask) for h in range(scores.size(1))]
    return dict(U=U, responsibility=responsibility, decomp=decomp, winner=winner, margin=margin,
               overlaps=overlaps, union_size=union_size, kl=kl_vals.detach(), entropy=entropy, spread=spread,
               spearman_h=spearman_h, oracle_idx=oracle_idx)


def evaluate_and_save_head_report(model, slot_heads, exp, args, cli, channels, device, train_loader, val_loader,
                                  test_loader, out_dir, best_epoch, num_slots=5, tau_e=1.0):
    """Shared post-training evaluation: the FULL per-head report (spec
    TRACK-EXPERT-V5-FULL01 sections 9-16), applied identically whether
    `model`/`slot_heads` came from the new Expert-V5 trainer or an
    EXISTING canonical Current-V5 checkpoint -- the whole point of
    sharing this function is that the two are compared through
    byte-identical measurement code (spec section 11). Writes every
    `per_head_*` / `winner_fraction` / `utility_margin` /
    `head_overlap_matrix` / `head_union_metrics` / `oracle_vs_fixed`
    file into `out_dir` and returns the primary (round-robin)
    `final_test_metrics` dict.

    Requires `cli` to carry: top_k, tau_t, tau_s, chunk_size."""
    from scripts.train_factorial_e2e01 import individual_utility_memsafe
    from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob
    from scripts.train_margutil01 import memory_value
    from scripts.train_t_pure_multislot01 import compute_scores_full_grad, hard_eval_decomposition

    model.eval()

    def full_pass(loader):
        rr_sums, n = {}, 0
        head_decomp_sums = {name: torch.zeros(num_slots) for name in
                            ('retmse10', 'agg_mse', 'D', 'C', 'recall10', 'ndcg10')}
        head_kl_sum = torch.zeros(num_slots)
        head_U_sum = torch.zeros(num_slots)
        head_resp_sum = torch.zeros(num_slots)
        head_entropy_sum = torch.zeros(num_slots)
        head_spread_sum = torch.zeros(num_slots)
        head_spearman_sum = torch.zeros(num_slots)
        win_count = torch.zeros(num_slots)
        overlap_sum = {}
        union_sum = 0.0
        margin_vals = []
        with torch.no_grad():
            for batch_x, batch_y, batch_start_idx in loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                cand_mask, _ = exp._candidate_mask(batch_start_idx)
                bsz = batch_x.size(0)
                per_ch_rr = {}
                for c in channels:
                    scores = compute_scores_full_grad(model, slot_heads, batch_x, exp.memory_x, c)
                    memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
                    query_future = batch_y[:, :, c]
                    u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
                    d_raw = -u
                    p_t = normalized_teacher_prob(d_raw, cand_mask, cli.tau_t)
                    oracle_idx = stable_topk_indices(d_raw.masked_fill(~cand_mask, float('inf')), cli.top_k,
                                                      largest=False)
                    res_rr = hard_eval_decomposition(scores, cand_mask, memory_c, offset_c, query_future, d_raw,
                                                     oracle_idx, cli.top_k)
                    per_ch_rr.setdefault('retmse10', []).append(res_rr['model_ind_mse'].cpu())
                    per_ch_rr.setdefault('agg_mse10', []).append(res_rr['agg_mse'].cpu())
                    per_ch_rr.setdefault('recall10', []).append(res_rr['recall10'].cpu())
                    per_ch_rr.setdefault('ndcg10', []).append(res_rr['ndcg10'].cpu())

                    diag = full_head_diagnostics(scores, cand_mask, memory_c, offset_c, query_future, d_raw, p_t,
                                                 cli.tau_s, cli.top_k, tau_e=tau_e)
                    for name in head_decomp_sums:
                        head_decomp_sums[name] += diag['decomp'][name].sum(dim=0).cpu()
                    head_kl_sum += diag['kl'].sum(dim=0).cpu()
                    head_U_sum += diag['U'].sum(dim=0).cpu()
                    head_resp_sum += diag['responsibility'].sum(dim=0).cpu()
                    head_entropy_sum += diag['entropy'].sum(dim=0).cpu()
                    head_spread_sum += diag['spread'].sum(dim=0).cpu()
                    head_spearman_sum += torch.tensor(diag['spearman_h'])
                    win_count += torch.nn.functional.one_hot(diag['winner'], num_slots).float().sum(dim=0).cpu()
                    for pair, v in diag['overlaps'].items():
                        overlap_sum[pair] = overlap_sum.get(pair, 0.0) + float(v.sum())
                    union_sum += float(diag['union_size'].sum())
                    margin_vals.append(diag['margin'].cpu())
                for k_, vals in per_ch_rr.items():
                    rr_sums[k_] = rr_sums.get(k_, 0.0) + torch.cat(vals).sum().item()
                n += bsz
        denom = max(n * len(channels), 1)
        return {
            'rr': {k_: v / denom for k_, v in rr_sums.items()},
            'head_decomp': {name: (t / denom).tolist() for name, t in head_decomp_sums.items()},
            'head_kl': (head_kl_sum / denom).tolist(),
            'head_U': (head_U_sum / denom).tolist(),
            'head_responsibility': (head_resp_sum / denom).tolist(),
            'head_entropy': (head_entropy_sum / denom).tolist(),
            'head_spread': (head_spread_sum / denom).tolist(),
            'head_spearman': (head_spearman_sum / denom).tolist(),
            'win_count': win_count.tolist(),
            'win_fraction': (win_count / denom).tolist(),
            'overlap_matrix_mean': {f'H{p[0]+1}-H{p[1]+1}': v / denom for p, v in overlap_sum.items()},
            'union_size_mean': union_sum / denom,
            'margin': torch.cat(margin_vals),
            'n_queries_seen': n,
        }

    train_full = full_pass(train_loader)
    val_full = full_pass(val_loader)
    test_full = full_pass(test_loader)
    final_test_metrics = {**test_full['rr'], 'n_queries_seen': test_full['n_queries_seen'], 'best_epoch': best_epoch}

    def margin_stats(margin):
        q = torch.quantile(margin, torch.tensor([0.5, 0.9]))
        return {'mean': float(margin.mean()), 'median': float(q[0]), 'p90': float(q[1]),
               'near_tie_fraction': float((margin.abs() < 1e-4).float().mean())}

    val_head_retmse = torch.tensor(val_full['head_decomp']['retmse10'])
    h_fixed = int(val_head_retmse.argmin())
    fixed_retmse_test = test_full['head_decomp']['retmse10'][h_fixed]

    def oracle_per_query_mean(loader):
        total, n = 0.0, 0
        with torch.no_grad():
            for batch_x, batch_y, batch_start_idx in loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                cand_mask, _ = exp._candidate_mask(batch_start_idx)
                bsz = batch_x.size(0)
                for c in channels:
                    scores = compute_scores_full_grad(model, slot_heads, batch_x, exp.memory_x, c)
                    memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
                    query_future = batch_y[:, :, c]
                    u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
                    d_raw = -u
                    head_topk_idx = per_head_standalone_topk(scores, cand_mask, k=cli.top_k)
                    U = per_head_future_utility(head_topk_idx, d_raw)
                    total += U.min(dim=1).values.sum().item()
                n += bsz * len(channels)
        return total / max(n, 1)

    oracle_test = oracle_per_query_mean(test_loader)
    oracle_vs_fixed = {
        'fixed_head': f'H{h_fixed+1}', 'fixed_retmse10_test': fixed_retmse_test,
        'oracle_retmse10_test_per_query_mean': oracle_test,
        'oracle_gain_pct': (fixed_retmse_test - oracle_test) / max(fixed_retmse_test, 1e-9) * 100.0,
        'note': 'oracle is argmin_h U_h(q) applied per query, not the column-wise min of the mean table.',
    }

    out_dir = Path(out_dir) if not hasattr(out_dir, 'mkdir') else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'final_test_metrics.json').write_text(json.dumps(final_test_metrics, indent=2, default=str))
    for split_name, full in (('train', train_full), ('val', val_full), ('test', test_full)):
        (out_dir / f'per_head_{split_name}_metrics.json').write_text(json.dumps(
            {k_: v for k_, v in full.items() if k_ != 'margin'}, indent=2, default=str))
    (out_dir / 'per_head_kl.json').write_text(json.dumps({'test': test_full['head_kl']}, indent=2))
    (out_dir / 'per_head_future_utility.json').write_text(json.dumps({'test': test_full['head_U']}, indent=2))
    (out_dir / 'responsibility_metrics.json').write_text(json.dumps(
        {'test_mean_responsibility': test_full['head_responsibility']}, indent=2))
    (out_dir / 'winner_fraction.json').write_text(json.dumps(
        {'train': train_full['win_fraction'], 'val': val_full['win_fraction'], 'test': test_full['win_fraction']},
        indent=2))
    (out_dir / 'utility_margin.json').write_text(json.dumps(margin_stats(test_full['margin']), indent=2))
    (out_dir / 'head_overlap_matrix.json').write_text(json.dumps(test_full['overlap_matrix_mean'], indent=2))
    (out_dir / 'head_union_metrics.json').write_text(json.dumps(
        {'test_union_size_mean': test_full['union_size_mean']}, indent=2))
    (out_dir / 'oracle_vs_fixed.json').write_text(json.dumps(oracle_vs_fixed, indent=2))
    return final_test_metrics


def per_head_entropy_and_spread(p_h, cand_mask, eps=EPS):
    """p_h: [B,S,N] softmax probabilities. Returns (entropy[B,S],
    score_spread[B,S]) where score_spread is the std of the
    probability mass over the query's own valid candidates (an
    operational "how concentrated is this head's distribution"
    measure -- documented here since the exact historical definition
    of 'score spread' is not pinned down elsewhere in the repo for
    this exact diagnostic)."""
    bsz, s, n = p_h.shape
    mask = cand_mask.unsqueeze(1).expand(-1, s, -1)
    p_valid = p_h.clamp_min(eps)
    ent = -(p_valid * p_valid.log()).sum(dim=-1)
    n_valid = mask.float().sum(dim=-1).clamp_min(1.0)
    mean_valid = (p_h * mask).sum(dim=-1) / n_valid
    var_valid = (((p_h - mean_valid.unsqueeze(-1)) ** 2) * mask).sum(dim=-1) / n_valid
    spread = var_valid.clamp_min(0).sqrt()
    return ent, spread
