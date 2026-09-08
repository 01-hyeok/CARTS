#!/usr/bin/env python3
"""EXP-TOPTAIL-RANK01: does a top-tail-focused pairwise ranking loss fix
EXP-MARGUTIL01's pointwise-SmoothL1 continuation failure?

Controlled experiment: EVERYTHING is identical to EXP-MARGUTIL01
(scripts/train_margutil01.py) -- frozen B0 encoder, `SetConditioner`,
`EmptySetToken`, `UtilityHead` (`a*cosine+b`), oracle-prefix teacher
forcing, full-memory candidate support, dense utility target
`u_i^(t)=-A_weighted(S*_{t-1}+{i})` (same `utils.dense_utility` call) --
EXCEPT the training loss. `run_sequence_dense`, `encode`, `memory_value`,
`candidate_weights`, `dense_utility` are imported from
`train_margutil01.py`/`utils.dense_utility` verbatim, not reimplemented.

Loss modes:
  smoothl1  -- EXP-MARGUTIL01's own baseline (R0), included here only for
               completeness/param-count parity check, not meant to be
               retrained (R0's own checkpoint is reused as-is).
  pairwise  -- R1. logistic pairwise ranking loss between TRUE top-1%
               positives and PREDICTED-high-but-not-positive hard
               negatives, at every teacher-forced step.
  hybrid    -- R2 (optional secondary arm). smoothl1 + lambda_rank*pairwise.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp.exp_stage1_relation import Exp_Stage1_Relation
from models.DenseUtilityRetriever import AsymmetricUtilityHead, StrongResidualPairScorer, UtilityHead
from models.SequentialSetRetriever import EmptySetToken, SetConditioner
from scripts.train_margutil01 import build_experiment, encode, memory_value, run_sequence_dense
from utils.dense_utility import candidate_weights, dense_utility, normalize_utility
from layers.retrieval_metric import cosine_init_deviation


def pairwise_step_loss(u_hat, u_target, valid_mask, top_pct=0.01, min_pos=10,
                        predicted_top_h_min=50, num_pos_samples=16,
                        num_hard_neg_samples=32):
    """Logistic pairwise ranking loss between TRUE top-`top_pct` positives
    and PREDICTED-high hard negatives that are not themselves positives.

    Fully batched/vectorised (no per-row Python loop, no `.tolist()`
    GPU->CPU sync): every row in a step shares the same candidate count N
    (the full memory bank), only WHICH positions are valid differs, so
    `topk` over the whole batch replaces a per-row loop without changing
    the definition. `n_pos`/`predicted_top_h` use `valid_mask`'s per-row
    count where it matters (the topk calls below already respect per-row
    validity via masking to -inf) but the SIZE of the topk slice is a
    single scalar shared across the batch, derived from `top_pct` and the
    batch's median valid count -- a documented simplification (this
    project's candidate pools have only a handful of invalid entries per
    query out of thousands, so the per-row vs. shared-scalar `n_pos`
    difference is negligible; see REPORT.md).

    Rather than building an explicit per-row hard-negative index list
    (impossible to batch cleanly since the count varies per row), this
    computes the full `num_pos_samples x predicted_top_h` pairwise-loss
    matrix per row and MASKS OUT pairs whose "negative" slot is actually a
    true positive (mathematically equivalent to excluding it from the
    negative pool -- masked pairs contribute exactly 0 and are excluded
    from the mean's denominator), then caps the contributing negatives per
    row at `num_hard_neg_samples` by only keeping the highest-predicted
    `num_hard_neg_samples` positions within `predicted_top_h` that pass the
    mask (also vectorised, via a second masked-topk over the pair mask).

    Rows with fewer than 2 valid candidates, or with zero surviving hard
    negatives, contribute exactly 0 to the loss and are naturally excluded
    via the mask-weighted average (not via a Python-level skip).
    """
    bsz, n = u_hat.shape
    neg_inf = torch.finfo(u_target.dtype).min / 4
    valid_count = valid_mask.sum(dim=-1)  # [B]
    median_valid = int(valid_count.float().median().clamp_min(2))

    n_pos = min(max(min_pos, int(median_valid * top_pct + 0.999999)), median_valid, n)
    predicted_top_h = min(max(predicted_top_h_min, n_pos), median_valid, n)

    target_masked = u_target.masked_fill(~valid_mask, neg_inf)
    pos_idx = torch.topk(target_masked, n_pos, dim=-1).indices  # [B, n_pos]
    positive_mask = torch.zeros_like(valid_mask).scatter(1, pos_idx, True)  # [B, N]

    pred_masked = u_hat.masked_fill(~valid_mask, neg_inf)
    pred_top_idx = torch.topk(pred_masked, predicted_top_h, dim=-1).indices  # [B, H]
    pred_top_is_positive = positive_mask.gather(1, pred_top_idx)  # [B, H]
    pred_top_scores = u_hat.gather(1, pred_top_idx)  # [B, H]
    # Rank the predicted-top-H slots by score so the highest-scoring
    # non-positive ones are kept first when capping at num_hard_neg_samples.
    hard_neg_priority = pred_top_scores.masked_fill(pred_top_is_positive, neg_inf)
    cap = min(num_hard_neg_samples, predicted_top_h)
    kept_local = torch.topk(hard_neg_priority, cap, dim=-1)
    neg_scores = kept_local.values  # [B, cap] -- -inf where no valid hard negative remains
    neg_valid = neg_scores > (neg_inf / 2)

    pos_sample_idx = pos_idx[:, :min(num_pos_samples, n_pos)]  # [B, P]
    pos_scores = u_hat.gather(1, pos_sample_idx)  # [B, P]

    diff = pos_scores.unsqueeze(-1) - neg_scores.unsqueeze(1)  # [B, P, cap]
    pair_valid = neg_valid.unsqueeze(1).expand_as(diff) & (valid_count.unsqueeze(-1).unsqueeze(-1) >= 2)
    loss_terms = F.softplus(-diff)
    weight = pair_valid.float()
    row_has_pairs = weight.sum(dim=(1, 2)) > 0

    stats = None
    with torch.no_grad():
        wsum = weight.sum(dim=(1, 2)).clamp_min(1.0)
        pos_mean_per_row = (pos_scores.unsqueeze(-1) * pair_valid.float()).sum(dim=(1, 2)) / wsum
        neg_mean_per_row = (neg_scores.unsqueeze(1) * pair_valid.float()).sum(dim=(1, 2)) / wsum
        rows = row_has_pairs
        if bool(rows.any()):
            pred_masked_valid = u_hat.masked_fill(~valid_mask, float('nan'))
            score_std = torch.nanmean(
                torch.tensor([float(torch.std(pred_masked_valid[b][valid_mask[b]]))
                              for b in range(bsz) if bool(valid_mask[b].any())], device=u_hat.device)
            ) if valid_mask.any() else torch.tensor(float('nan'))
            stats = {
                'pos_score_mean': float(pos_mean_per_row[rows].mean()),
                'neg_score_mean': float(neg_mean_per_row[rows].mean()),
                'margin': float((pos_mean_per_row - neg_mean_per_row)[rows].mean()),
                'score_std': float(score_std),
            }
        else:
            stats = {'pos_score_mean': float('nan'), 'neg_score_mean': float('nan'),
                      'margin': float('nan'), 'score_std': float('nan')}

    if not bool(row_has_pairs.any()):
        return u_hat.sum() * 0.0, stats
    per_row = (loss_terms * weight).sum(dim=(1, 2)) / wsum
    return per_row[row_has_pairs].mean(), stats


def step_losses(u_hat_steps, u_target_steps, valid_steps, query_valid, loss_mode, lambda_rank):
    """Returns (losses, diagnostics) -- `diagnostics` is a list (one dict per
    step) with the separate, un-weighted-sum loss components and score
    geometry, purely for logging; the returned `losses` (used for backward)
    are unaffected by whether diagnostics are collected."""
    losses = []
    diags = []
    for u_hat, u_target, valid in zip(u_hat_steps, u_target_steps, valid_steps):
        u_hat_v = u_hat[query_valid]
        u_target_v = u_target[query_valid]
        valid_v = valid[query_valid]
        if u_hat_v.size(0) == 0:
            losses.append(u_hat.sum() * 0.0)
            diags.append({})
            continue
        parts = []
        step_diag = {}
        if loss_mode in ('smoothl1', 'hybrid'):
            u_norm = normalize_utility(u_target_v, valid_v)
            diff = F.smooth_l1_loss(u_hat_v, u_norm, reduction='none')
            vf = valid_v.float()
            per_query = (diff * vf).sum(dim=-1) / vf.sum(dim=-1).clamp_min(1.0)
            smoothl1_val = per_query.mean()
            parts.append(smoothl1_val)
            step_diag['smoothl1_loss'] = float(smoothl1_val.detach())
        if loss_mode in ('pairwise', 'hybrid'):
            pair_loss, pair_stats = pairwise_step_loss(u_hat_v, u_target_v, valid_v)
            parts.append(lambda_rank * pair_loss if loss_mode == 'hybrid' else pair_loss)
            step_diag['pairwise_loss'] = float(pair_loss.detach())
            step_diag.update(pair_stats)
        losses.append(sum(parts))
        diags.append(step_diag)
    return losses, diags


def mine_pairs(u_target, u_hat_nograd, valid_mask, top_pct=0.01, min_pos=10,
               predicted_top_h_min=50, num_pos_samples=16, num_hard_neg_samples=32):
    """Global full-memory positive/hard-negative index mining, factored out
    of `pairwise_step_loss` so the memory-safe streaming path
    (`strong_pair_step`) can mine indices from a no-grad full-memory
    predicted-score pass, then build a small per-row gathered forward graph
    from just those indices -- never holding a graph over the full N.
    Returns (pos_idx [B,P], neg_idx [B,cap], neg_valid [B,cap])."""
    bsz, n = u_hat_nograd.shape
    neg_inf = torch.finfo(u_target.dtype).min / 4
    valid_count = valid_mask.sum(dim=-1)
    median_valid = int(valid_count.float().median().clamp_min(2))

    n_pos = min(max(min_pos, int(median_valid * top_pct + 0.999999)), median_valid, n)
    predicted_top_h = min(max(predicted_top_h_min, n_pos), median_valid, n)

    target_masked = u_target.masked_fill(~valid_mask, neg_inf)
    pos_idx = torch.topk(target_masked, n_pos, dim=-1).indices
    positive_mask = torch.zeros_like(valid_mask).scatter(1, pos_idx, True)

    pred_masked = u_hat_nograd.masked_fill(~valid_mask, neg_inf)
    pred_top_idx = torch.topk(pred_masked, predicted_top_h, dim=-1).indices
    pred_top_is_positive = positive_mask.gather(1, pred_top_idx)
    pred_top_scores = u_hat_nograd.gather(1, pred_top_idx)
    hard_neg_priority = pred_top_scores.masked_fill(pred_top_is_positive, neg_inf)
    cap = min(num_hard_neg_samples, predicted_top_h)
    kept_local = torch.topk(hard_neg_priority, cap, dim=-1)
    neg_idx = pred_top_idx.gather(1, kept_local.indices)
    neg_valid = kept_local.values > (neg_inf / 2)
    pos_idx = pos_idx[:, :min(num_pos_samples, n_pos)]
    return pos_idx, neg_idx, neg_valid


def strong_pair_step(q, h_state_inputs, E, cand_mask, set_conditioner, utility_head,
                      u_target, valid_now, tau, lambda_rank, grad_scale, chunk_size, train):
    """Memory-safe streaming computation of ONE teacher-forced step's hybrid
    loss for `scorer_mode=strong_pair`, replacing `step_losses`'s
    hold-the-whole-graph approach for this (expensive) scorer only --
    `cosine`/`asymmetric` are untouched and still use the original path.

    Never holds an autograd graph over the full candidate bank: (1) mines
    global positive/hard-negative indices from a NO-GRAD full-memory
    predicted-score pass (chunked internally by the scorer itself, but no
    graph retained since no_grad); (2) the pairwise loss is a small forward
    on just the gathered positive/hard-negative candidates (WITH grad,
    backward immediately, `retain_graph=False`); (3) the dense SmoothL1
    target is matched chunk-by-chunk, each chunk getting its OWN fresh
    `h_t = SetConditioner(...)` forward (never sharing/retaining one h_t's
    graph across chunks) and its own immediate backward. `optimizer.step()`
    is never called here -- only `.backward()`, so gradients accumulate
    exactly as they would from a single full-batch backward (linearity of
    differentiation), and the caller calls `optimizer.step()` once after
    every step/channel/chunk in the batch has contributed.

    `grad_scale` folds in the 1/K and 1/num_channels normalisation the
    original `run_epoch` applied before its single backward() call, so the
    accumulated gradient here is mathematically identical to that call.

    `h_state_inputs = (q, m_fn)`: `m_fn` is a ZERO-ARG CALLABLE returning a
    FRESH set-state summary tensor on every call, not a precomputed tensor.
    This matters specifically at t=1 (empty selected set), where `m` comes
    from `EmptySetToken` -- a TRAINABLE module. A precomputed `m` tensor
    reused across multiple `set_conditioner(q, m)` forward+immediate-backward
    calls would silently share that upstream (EmptySetToken) portion of the
    graph across calls, and the second `.backward()` on it would raise
    ("Trying to backward through the graph a second time") -- caught by a
    real-data GPU run before the full experiment started; `m_fn()` gives
    every chunk/pair-forward its own independent graph all the way back
    through EmptySetToken, exactly like `set_conditioner`'s own fresh
    per-chunk forward already did. For t>1, `m` has no trainable upstream
    (mean of frozen candidate embeddings at fixed oracle-prefix indices), so
    `m_fn` recomputing it is a cheap no-op recomputation, not a correctness
    requirement -- but using the same callable for both cases keeps this
    function's contract uniform and correct regardless of t.

    Returns a diagnostics dict (python floats only, no tensors retained).
    """
    q_state, m_fn = h_state_inputs
    bsz, n = u_target.shape

    with torch.no_grad():
        h_nograd = set_conditioner(q_state, m_fn())
        u_hat_nograd = utility_head(h_nograd, E)
        u_norm = normalize_utility(u_target, valid_now)
        pos_idx, neg_idx, neg_valid = mine_pairs(u_target, u_hat_nograd, valid_now)
        valid_count_row = valid_now.sum(dim=-1).clamp_min(1).float()
        pos_scores_ng = u_hat_nograd.gather(1, pos_idx)
        neg_scores_ng = u_hat_nograd.gather(1, neg_idx)
        margin_val = float((pos_scores_ng.mean(dim=-1) - neg_scores_ng.mean(dim=-1)).mean())
        valid_vals = u_hat_nograd[valid_now]
        score_std_val = float(valid_vals.std()) if valid_vals.numel() > 1 else float('nan')

    # ---- (2) pairwise: small per-row gathered forward, WITH grad ----
    pos_e = E[pos_idx]  # [B, P, D]
    neg_e = E[neg_idx]  # [B, cap, D]
    h_pair = set_conditioner(q_state, m_fn())
    pos_scores = utility_head.forward_batched(h_pair, pos_e)
    neg_scores = utility_head.forward_batched(h_pair, neg_e)
    diff = pos_scores.unsqueeze(-1) - neg_scores.unsqueeze(1)  # [B, P, cap]
    pair_valid = neg_valid.unsqueeze(1).expand_as(diff) & (valid_count_row.unsqueeze(-1).unsqueeze(-1) >= 2)
    loss_terms = F.softplus(-diff)
    weight = pair_valid.float()
    wsum = weight.sum(dim=(1, 2))
    row_has_pairs = wsum > 0
    pairwise_loss_val = float('nan')
    if bool(row_has_pairs.any()):
        per_row = (loss_terms * weight).sum(dim=(1, 2)) / wsum.clamp_min(1.0)
        pair_loss = per_row[row_has_pairs].mean()
        pairwise_loss_val = float(pair_loss.detach())
        if train:
            (lambda_rank * grad_scale * pair_loss).backward()

    # ---- (3) SmoothL1: chunked, fresh h_t per chunk, immediate backward ----
    smoothl1_total = 0.0
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        e_chunk = E[start:end]
        h_chunk = set_conditioner(q_state, m_fn())
        u_hat_chunk = utility_head(h_chunk, e_chunk)
        target_chunk = u_norm[:, start:end]
        valid_chunk = valid_now[:, start:end]
        diff_chunk = F.smooth_l1_loss(u_hat_chunk, target_chunk, reduction='none')
        weight_chunk = valid_chunk.float()
        chunk_partial = (diff_chunk * weight_chunk).sum(dim=-1)  # [B]
        chunk_loss = (chunk_partial / valid_count_row).mean()
        smoothl1_total += float(chunk_loss.detach())
        if train:
            (grad_scale * chunk_loss).backward()

    return {
        'pairwise_loss': pairwise_loss_val, 'smoothl1_loss': smoothl1_total,
        'pos_score_mean': float(pos_scores_ng.mean()), 'neg_score_mean': float(neg_scores_ng.mean()),
        'margin': margin_val, 'score_std': score_std_val,
    }


def overlap_at_k(picks, teacher_idx, query_valid):
    k = picks.size(1)
    overlaps = []
    for b in range(picks.size(0)):
        if not query_valid[b]:
            continue
        overlaps.append(len(set(picks[b].tolist()) & set(teacher_idx[b].tolist())) / k)
    return sum(overlaps) / max(len(overlaps), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base_ckpt', required=True)
    ap.add_argument('--teacher_cache', required=True)
    ap.add_argument('--checkpoints', default='checkpoints/exp_toptail_rank01')
    ap.add_argument('--model_id', default='carts_toptail_rank01_main')
    ap.add_argument('--des', default='toptail_rank01_main')
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--train_epochs', type=int, default=0)
    ap.add_argument('--patience', type=int, default=0)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--loss_mode', required=True, choices=['smoothl1', 'pairwise', 'hybrid'])
    ap.add_argument('--lambda_rank', type=float, default=1.0)
    ap.add_argument('--scorer_mode', default='cosine', choices=['cosine', 'asymmetric', 'strong_pair'],
                     help='EXP-ASYM-SCORER01: cosine (default, UtilityHead, EXP-TOPTAIL-RANK01 '
                          'unchanged), asymmetric (AsymmetricUtilityHead), or strong_pair '
                          '(EXP-STRONG-SCORER-DIAG01: StrongResidualPairScorer, a zero-init '
                          'nonlinear residual ON TOP OF the base cosine score). All non-cosine '
                          'modes are identity/zero-init-equivalent to cosine at step 0 -- verified '
                          'before training starts.')
    ap.add_argument('--scorer_chunk_size', type=int, default=1024,
                     help='strong_pair only: candidate-dimension chunk size for the pairwise MLP '
                          '(GPU-memory optimisation only -- every valid candidate is still scored).')
    cli = ap.parse_args()

    overrides = {
        'is_training': 1, 'model_id': cli.model_id, 'des': cli.des,
        'checkpoints': cli.checkpoints, 'seed': cli.seed, 'top_k': cli.top_k,
        'stage1_residual_teacher': 0, 'stage1_query_base_conditioning': 0,
        'stage1_candidate_residual_conditioning': 0, 'stage1_retrieval_metric': 'cosine',
        'stage1_full_memory_gradient_mode': 'full_online',
    }
    if cli.train_epochs:
        overrides['train_epochs'] = cli.train_epochs
    if cli.patience:
        overrides['patience'] = cli.patience

    exp, args = build_experiment(cli.base_ckpt, overrides)
    torch.manual_seed(args.seed)
    device = exp.device
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    k = int(cli.top_k)
    tau = float(args.tau_topk)

    b0_ckpt = torch.load(cli.base_ckpt, map_location='cpu')
    model.load_state_dict(b0_ckpt['model_state_dict'], strict=True)
    for p in model.encoder.parameters():
        p.requires_grad = False
    model.encoder.eval()
    print(f'[toptail_rank01] frozen encoder loaded from {cli.base_ckpt}, loss_mode={cli.loss_mode}')

    exp._ensure_memory()
    channels = list(model.target_channels())
    teacher = torch.load(cli.teacher_cache, map_location='cpu')
    if teacher['meta']['top_k'] != k:
        raise ValueError(f"teacher cache top_k={teacher['meta']['top_k']} != --top_k={k}")

    d_model = int(args.d_model)
    set_conditioner = SetConditioner(d_model).to(device)
    empty_token = EmptySetToken(d_model).to(device)
    if cli.scorer_mode == 'asymmetric':
        utility_head = AsymmetricUtilityHead(d_model).to(device)
        deviation = cosine_init_deviation(utility_head.metric, samples=256,
                                           generator=torch.Generator(device=device).manual_seed(args.seed))
        print(f'[toptail_rank01] scorer_mode=asymmetric identity-init check: '
              f'max_abs_score_deviation={deviation:.3e} (must be < 1e-6)')
        if deviation >= 1e-6:
            raise RuntimeError(
                f'asymmetric scorer is not identity-initialised to cosine (deviation={deviation:.3e} '
                f'>= 1e-6) -- refusing to start an expensive training run on a broken controlled variable')
    elif cli.scorer_mode == 'strong_pair':
        utility_head = StrongResidualPairScorer(d_model, chunk_size=cli.scorer_chunk_size).to(device)
        gen = torch.Generator(device=device).manual_seed(args.seed)
        h_probe = torch.nn.functional.normalize(torch.randn(32, d_model, device=device, generator=gen), dim=-1)
        e_probe = torch.nn.functional.normalize(torch.randn(256, d_model, device=device, generator=gen), dim=-1)
        with torch.no_grad():
            base_probe = utility_head.scale * torch.matmul(h_probe, e_probe.transpose(0, 1)) + utility_head.bias
            full_probe = utility_head(h_probe, e_probe)
            deviation = float((full_probe - base_probe).abs().max())
        print(f'[toptail_rank01] scorer_mode=strong_pair zero-init check: '
              f'max_abs_score_deviation={deviation:.3e} (must be < 1e-6)')
        if deviation >= 1e-6:
            raise RuntimeError(
                f'strong_pair residual is not zero-initialised (deviation={deviation:.3e} >= 1e-6) '
                f'-- refusing to start an expensive training run on a broken controlled variable')
    else:
        utility_head = UtilityHead().to(device)
    trainable_params = (list(set_conditioner.parameters()) + list(empty_token.parameters())
                         + list(utility_head.parameters()))
    n_params = sum(p.numel() for p in trainable_params)
    print(f'[toptail_rank01] scorer_mode={cli.scorer_mode} trainable_params={n_params} '
          f'(SetConditioner={sum(p.numel() for p in set_conditioner.parameters())}, '
          f'EmptySetToken={sum(p.numel() for p in empty_token.parameters())}, '
          f'UtilityHead/ScorerHead={sum(p.numel() for p in utility_head.parameters())}) '
          f'-- SetConditioner/EmptySetToken must equal EXP-TOPTAIL-RANK01 R2\'s own params exactly; '
          f'only the scorer head differs by design (W_q/W_k added when asymmetric)')
    optimizer = torch.optim.Adam(trainable_params, lr=float(args.learning_rate))

    ckpt_dir = Path(cli.checkpoints) / 'toptail_rank01' / args.data / f'seq{args.seq_len}_pred{args.pred_len}' / args.model_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = -1.0
    best_epoch = -1
    patience_left = int(args.patience)

    def teacher_rows(batch_start_idx, channel, split):
        rows = [teacher['splits'][split]['start_to_row'][int(s)] for s in batch_start_idx.tolist()]
        return teacher['splits'][split]['teacher_idx'][channel][rows].to(device)

    def run_epoch(loader, split, train, channel_list):
        set_conditioner.train(train)
        empty_token.train(train)
        utility_head.train(train)
        model.encoder.eval()
        total_loss, n_batches = 0.0, 0
        overlap_sum, overlap_n = 0.0, 0
        sc_grad_norm = 0.0
        diag_sums = {}
        diag_n = 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch_x, batch_y, batch_start_idx in loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                cand_mask, counts = exp._candidate_mask(batch_start_idx)
                if train:
                    optimizer.zero_grad()
                batch_loss = 0.0
                for c in channel_list:
                    E = encode(model, exp.memory_x, c)
                    q = encode(model, batch_x, c)
                    memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
                    futures = memory_c + offset_c.view(-1, 1, 1)
                    query_future = batch_y[:, :, c]

                    teacher_idx = teacher_rows(batch_start_idx, c, split)
                    query_valid = teacher_idx[:, 0] != -1

                    if cli.scorer_mode == 'strong_pair' and train:
                        # Memory-safe streaming path (see strong_pair_step's
                        # own docstring): never holds an autograd graph over
                        # the full candidate bank across K steps/chunks.
                        # backward() is called incrementally inside
                        # strong_pair_step; batch_loss stays a float for
                        # logging only, optimizer.step() still happens once
                        # below, after every step/channel/chunk in this
                        # batch has contributed its gradient.
                        w_fixed = candidate_weights(torch.matmul(q, E.transpose(0, 1)), cand_mask, tau)
                        selected_mask = torch.zeros_like(cand_mask)
                        step_loss_sum = 0.0
                        for t in range(k):
                            with torch.no_grad():
                                prefix = teacher_idx[:, :t].clamp_min(0)
                                a_dense = dense_utility(prefix, w_fixed, futures, query_future, chunk_size=cli.chunk_size)
                                u_target = -a_dense
                                valid_now = cand_mask & ~selected_mask
                            u_target_v = u_target[query_valid]
                            valid_v = valid_now[query_valid]
                            q_v = q[query_valid]
                            bsz_v = q_v.size(0)

                            def m_fn(t=t, bsz_v=bsz_v):
                                # Fresh graph every call -- see strong_pair_step's
                                # docstring: a precomputed m tensor reused across
                                # multiple backward() calls double-frees the
                                # EmptySetToken portion of the graph at t=0.
                                if t == 0:
                                    return empty_token(bsz_v, device, q.dtype)
                                return E[teacher_idx[query_valid][:, :t]].mean(dim=1)

                            step_diag = {}
                            if bsz_v > 0:
                                step_diag = strong_pair_step(
                                    q_v, (q_v, m_fn), E, cand_mask[query_valid], set_conditioner,
                                    utility_head, u_target_v, valid_v, tau, cli.lambda_rank,
                                    grad_scale=1.0 / k / len(channel_list),
                                    chunk_size=cli.scorer_chunk_size, train=True)
                                step_loss_sum += step_diag.get('smoothl1_loss', 0.0) + (
                                    cli.lambda_rank * step_diag.get('pairwise_loss', 0.0)
                                    if step_diag.get('pairwise_loss', float('nan')) == step_diag.get('pairwise_loss', float('nan'))
                                    else 0.0)
                            for kk, vv in step_diag.items():
                                if vv == vv:
                                    diag_sums[kk] = diag_sums.get(kk, 0.0) + vv
                            if step_diag:
                                diag_n += 1
                            nxt = teacher_idx[:, t:t + 1].clamp_min(0)
                            selected_mask = selected_mask.scatter(1, nxt, True)
                        batch_loss = batch_loss + step_loss_sum / k
                    else:
                        u_hat_steps, u_target_steps, valid_steps, _ = run_sequence_dense(
                            q, E, cand_mask, set_conditioner, empty_token, utility_head,
                            futures, query_future, tau, k, cli.chunk_size,
                            teacher_idx=teacher_idx)
                        losses, diags = step_losses(u_hat_steps, u_target_steps, valid_steps,
                                                     query_valid, cli.loss_mode, cli.lambda_rank)
                        batch_loss = batch_loss + sum(losses) / k
                        for d in diags:
                            for kk, vv in d.items():
                                if vv == vv:  # skip NaN
                                    diag_sums[kk] = diag_sums.get(kk, 0.0) + vv
                        diag_n += sum(1 for d in diags if d)

                    if not train:
                        with torch.no_grad():
                            _, _, _, picks = run_sequence_dense(
                                q, E, cand_mask, set_conditioner, empty_token, utility_head,
                                futures, query_future, tau, k, cli.chunk_size, teacher_idx=None)
                        overlap_sum += overlap_at_k(picks, teacher_idx, query_valid) * int(query_valid.sum())
                        overlap_n += int(query_valid.sum())
                streaming = (cli.scorer_mode == 'strong_pair' and train)
                if streaming:
                    # strong_pair_step already called .backward() incrementally
                    # per step/chunk with grad_scale=1/k/len(channel_list)
                    # baked in -- batch_loss is a plain float here, nothing
                    # left to backward.
                    batch_loss_value = batch_loss / len(channel_list)
                    if train:
                        assert all(p.grad is None for p in model.encoder.parameters()), (
                            'encoder must stay frozen -- an encoder parameter received a gradient')
                        sc_grads = [p.grad for p in set_conditioner.parameters() if p.grad is not None]
                        sc_grad_norm = sum(g.norm().item() ** 2 for g in sc_grads) ** 0.5 if sc_grads else 0.0
                        optimizer.step()
                else:
                    batch_loss = batch_loss / len(channel_list)
                    if train:
                        batch_loss.backward()
                        assert all(p.grad is None for p in model.encoder.parameters()), (
                            'encoder must stay frozen -- an encoder parameter received a gradient')
                        sc_grads = [p.grad for p in set_conditioner.parameters() if p.grad is not None]
                        sc_grad_norm = sum(g.norm().item() ** 2 for g in sc_grads) ** 0.5 if sc_grads else 0.0
                        optimizer.step()
                    batch_loss_value = float(batch_loss.detach())
                total_loss += batch_loss_value
                n_batches += 1
        overlap = overlap_sum / max(overlap_n, 1)
        diag_means = {kk: vv / max(diag_n, 1) for kk, vv in diag_sums.items()}
        return {'loss': total_loss / max(n_batches, 1), 'overlap_at_k': overlap,
                'set_conditioner_grad_norm': sc_grad_norm, 'diag': diag_means}

    history = []
    wall_start = time.time()
    for epoch in range(int(args.train_epochs)):
        t0 = time.time()
        _, train_loader = exp._get_data(flag='train', shuffle=True)
        _, val_loader = exp._get_data(flag='val', shuffle=False)
        train_metrics = run_epoch(train_loader, 'train', True, channels)
        val_metrics = run_epoch(val_loader, 'val', False, channels)
        dt = time.time() - t0
        history.append({'epoch': epoch + 1, 'train': train_metrics, 'val': val_metrics, 'seconds': dt})
        print(f"[toptail_rank01] epoch {epoch+1} train_loss={train_metrics['loss']:.5f} "
              f"val_loss={val_metrics['loss']:.5f} val_overlap@{k}={val_metrics['overlap_at_k']:.4f} "
              f"set_conditioner_grad_norm={train_metrics['set_conditioner_grad_norm']:.5f} time={dt:.1f}s")
        vd = val_metrics['diag']
        a_val = float(utility_head.scale.detach())
        b_val = float(utility_head.bias.detach())
        print(f"[toptail_rank01] epoch {epoch+1} diag(val) "
              f"pairwise_loss={vd.get('pairwise_loss', float('nan')):.5f} "
              f"smoothl1_loss={vd.get('smoothl1_loss', float('nan')):.5f} "
              f"pos_score_mean={vd.get('pos_score_mean', float('nan')):.5f} "
              f"neg_score_mean={vd.get('neg_score_mean', float('nan')):.5f} "
              f"margin={vd.get('margin', float('nan')):.5f} "
              f"score_std={vd.get('score_std', float('nan')):.5f} "
              f"utility_head_a={a_val:.5f} utility_head_b={b_val:.5f}")
        if cli.scorer_mode == 'asymmetric':
            with torch.no_grad():
                eye = torch.eye(d_model, device=device)
                wq = utility_head.metric.query_projection.weight
                wk = utility_head.metric.key_projection.weight
                wq_dev = float((wq - eye).norm())
                wk_dev = float((wk - eye).norm())
                wq_norm = float(wq.norm())
                wk_norm = float(wk.norm())
                try:
                    wq_cond = float(torch.linalg.cond(wq.double()))
                    wk_cond = float(torch.linalg.cond(wk.double()))
                except Exception:
                    wq_cond = wk_cond = float('nan')
            print(f"[toptail_rank01] epoch {epoch+1} scorer(asym) "
                  f"||Wq-I||_F={wq_dev:.5f} ||Wk-I||_F={wk_dev:.5f} "
                  f"||Wq||_F={wq_norm:.5f} ||Wk||_F={wk_norm:.5f} "
                  f"cond(Wq)={wq_cond:.3f} cond(Wk)={wk_cond:.3f}")
        if train_metrics['set_conditioner_grad_norm'] == 0.0:
            print('[toptail_rank01][FAIL] SetConditioner gradient norm is exactly 0 -- STOP')
            break
        if val_metrics['overlap_at_k'] > best_val:
            best_val = val_metrics['overlap_at_k']
            best_epoch = epoch + 1
            patience_left = int(args.patience)
            torch.save({
                'model_state_dict': model.state_dict(),
                'set_conditioner_state_dict': set_conditioner.state_dict(),
                'empty_token_state_dict': empty_token.state_dict(),
                'utility_head_state_dict': utility_head.state_dict(),
                'args': vars(args), 'epoch': epoch + 1, 'val_overlap_at_k': best_val,
                'frozen_encoder_source_ckpt': cli.base_ckpt, 'loss_mode': cli.loss_mode,
                'scorer_mode': cli.scorer_mode,
            }, ckpt_dir / 'checkpoint.pth')
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f'[toptail_rank01] early stop at epoch {epoch+1} (best={best_epoch})')
                break

    wall_time = time.time() - wall_start
    peak_mem = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
    summary = {'best_epoch': best_epoch, 'best_val_overlap_at_k': best_val,
               'wall_clock_seconds': wall_time, 'peak_gpu_memory_mib': peak_mem,
               'history': history, 'args': vars(args), 'checkpoint': str(ckpt_dir / 'checkpoint.pth'),
               'loss_mode': cli.loss_mode, 'scorer_mode': cli.scorer_mode, 'trainable_params': n_params}
    with open(ckpt_dir / 'summary.json', 'w') as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f'[toptail_rank01] done. best_epoch={best_epoch} best_val_overlap_at_k={best_val:.4f} '
          f'wall_clock={wall_time:.1f}s peak_gpu_mem={peak_mem:.0f}MiB')
    print(f'[toptail_rank01] summary written to {ckpt_dir / "summary.json"}')


if __name__ == '__main__':
    main()
