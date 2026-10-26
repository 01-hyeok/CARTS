#!/usr/bin/env python3
"""TRACK-P100-COARSE-TO-FINE-RETRIEVAL01 -- PHASE D1-D3: learned
set-aware reranker within the primary scorer's own Top-20 (of
Shared-P100), trained on-policy against a dense marginal-aggregate
-utility soft target. Primary scorer = Phase C's Residual-V5 (frozen,
read-only here -- see `scripts/diag_p100_set_oracle01.py`'s docstring
for the validation-only selection audit). D0's own diagnostic
(`results/TRACK-P100-COARSE-TO-FINE-RETRIEVAL01/PHASE_D/*_set_oracle.json`)
already confirmed large oracle headroom (13.8-54.5% across cells) --
this script tests whether a SMALL, deployable model can capture any
of it.

Repo audit (spec section 23, done before writing this): the exact
failure modes of `EXP-SEQFULL01` (full-memory teacher-forced one-hot
CE against an offline oracle sequence -- val overlap never rose above
chance) and `EXP-MARGUTIL01`/`EXP-TOPTAIL-RANK01` (dense/pairwise
marginal-utility regression, but teacher-forced prefix -- utility
Spearman was learnable (0.38-0.66) yet free-running Top-K selection
was 3-19x worse than the unforced baseline, a train/inference prefix
mismatch) are NOT repeated here: (1) candidate support is the primary
scorer's own fixed Top-20 (never the full memory bank), (2) the
target is the SAME dense marginal-aggregate-utility quantity
EXP-MARGUTIL01 used, but wrapped in a SOFT listwise KL loss (not
one-hot CE, not pairwise ranking), and (3) prefix construction is
ON-POLICY during training (the model's own no-grad greedy pick at
step t becomes the actual prefix for step t+1's target/loss), exactly
to avoid EXP-MARGUTIL01's documented teacher-forced/free-running
mismatch.

Deployable (no-future) candidate features at each remaining-candidate
i, step t: primary-scorer score+rank, coarse score+rank, mean-head
raw residual score+rank ("V5 mixture score" in spec section 28 --
here Residual-V5's underlying per-head score mean-pooled over heads,
distinct from the primary (c_i+gamma*r) score), candidate embedding
z_i, query embedding z_q, cosine(z_q,z_i), and a past-only temporal
summary (the candidate's own delta-last window norm -- the same raw
ingredient the coarse score itself is built from, nothing new).
Current-set-summary features: mean selected z_i, mean selected
temporal summary, mean/min/max selected primary-scorer score, t/K.
No future feature anywhere.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.train_factorial_e2e01 import encode_raw
from scripts.train_j_shared_encoder_drift01 import build_model
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_margutil01 import memory_value
from scripts.train_retriever_pool01 import CandidatePoolCache
from utils.candidate_pool import compute_coarse_delta_last_cosine_scores

TOP_K = 10
TOP_M = 20
N_SLOTS = 5
TAU_U = 1.0   # teacher softmax temperature over marginal utility -- see main() for TRAIN-only calibration
TAU_S_SET = 0.5


def _rank(score):
    """score: [B,M]. Returns rank in [0,M) ascending with largest score -> rank 0."""
    order = score.argsort(dim=-1, descending=True)
    rank = torch.zeros_like(order)
    rank.scatter_(1, order, torch.arange(score.size(-1), device=score.device).unsqueeze(0).expand_as(order))
    return rank.float()


def candidate_features(primary_score, coarse_score, residual_mix_score, z_q, z_cand, temporal_summary):
    """All [B,20] or [B,20,D] / [B,D]. Returns [B,20,F_cand]."""
    bsz, m = primary_score.shape
    z_q_exp = z_q.unsqueeze(1).expand(-1, m, -1)
    cos_qi = F.cosine_similarity(z_q_exp, z_cand, dim=-1)  # [B,20]
    feats = torch.cat([
        primary_score.unsqueeze(-1), _rank(primary_score).unsqueeze(-1) / m,
        coarse_score.unsqueeze(-1), _rank(coarse_score).unsqueeze(-1) / m,
        residual_mix_score.unsqueeze(-1), _rank(residual_mix_score).unsqueeze(-1) / m,
        cos_qi.unsqueeze(-1), temporal_summary.unsqueeze(-1),
        z_cand, z_q_exp,
    ], dim=-1)  # [B,20, 8 + 2*D]
    return feats


def set_summary_features(z_cand, temporal_summary, primary_score, selected_mask, t, k):
    """selected_mask: [B,20] bool. Returns [B, F_set]."""
    bsz, m, d = z_cand.shape
    n_sel = selected_mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean_z = (z_cand * selected_mask.unsqueeze(-1)).sum(dim=1) / n_sel  # [B,D]
    mean_temporal = (temporal_summary * selected_mask.float()).sum(dim=-1, keepdim=True) / n_sel
    sel_scores = primary_score.masked_fill(~selected_mask, 0.0)
    mean_score = sel_scores.sum(dim=-1, keepdim=True) / n_sel
    min_score = primary_score.masked_fill(~selected_mask, float('inf')).min(dim=-1, keepdim=True).values
    min_score = torch.where(torch.isinf(min_score), torch.zeros_like(min_score), min_score)
    max_score = primary_score.masked_fill(~selected_mask, float('-inf')).max(dim=-1, keepdim=True).values
    max_score = torch.where(torch.isinf(max_score), torch.zeros_like(max_score), max_score)
    frac = torch.full((bsz, 1), float(t) / k, device=z_cand.device)
    return torch.cat([mean_z, mean_temporal, mean_score, min_score, max_score, frac], dim=-1)


class SetAwareReranker(nn.Module):
    """Tiny: candidate features + set-summary features -> scalar utility score."""

    def __init__(self, d_model, hidden=64):
        super().__init__()
        f_cand = 8 + 2 * d_model
        f_set = d_model + 5
        self.net = nn.Sequential(
            nn.Linear(f_cand + f_set, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, cand_feats, set_feats):
        bsz, m, _ = cand_feats.shape
        set_exp = set_feats.unsqueeze(1).expand(-1, m, -1)
        x = torch.cat([cand_feats, set_exp], dim=-1)
        return self.net(x).squeeze(-1)  # [B,M]


def on_policy_rollout(model, z_q, z_cand, primary_score, coarse_score, residual_mix_score, temporal_summary,
                      top20_future, query_future, k, tau_u, tau_s_set, target_kind='aggregate',
                      shuffle_set_feats=False):
    """Returns (loss_sum, picks [B,k]). `target_kind`: 'aggregate' (D1/D2's complement
    -- marginal aggregate improvement) or 'individual' (D2 control -- plain individual
    future-MSE, the EXP-MARGUTIL01-style ablation of the aggregate target).
    `shuffle_set_feats` (D3 control, spec section 31: "current selected-set summary를
    query 간 shuffle"): after computing the CURRENT-SET-SUMMARY features each step,
    permute them across the batch dimension before feeding the model -- candidate
    -level features, the environment (futures, true picks, loss target) all stay
    correctly paired to their own query; only the set-state input the model SEES is
    mismatched. If D1 (trained on correctly-paired set-state) collapses to roughly
    the same loss/AggMSE under this shuffle, the set-state input was not doing
    anything query-specific."""
    bsz, m, h = top20_future.shape
    device = z_cand.device
    selected_mask = torch.zeros(bsz, m, dtype=torch.bool, device=device)
    running_sum = torch.zeros(bsz, h, device=device)
    loss_sum = 0.0
    picks = []
    individual_mse = ((top20_future - query_future.unsqueeze(1)) ** 2).mean(dim=-1)  # [B,20], future-blind at deploy, TRAIN-only target use

    # candidate-level features depend only on the (batch, channel)'s own fixed Top-20 --
    # never on the set built so far -- so they are computed ONCE here, not re-built on
    # every one of the k steps (that would be 10x the concat/MLP cost for identical output).
    cand_feats = candidate_features(primary_score, coarse_score, residual_mix_score, z_q, z_cand, temporal_summary)

    for t in range(1, k + 1):
        set_feats = set_summary_features(z_cand, temporal_summary, primary_score, selected_mask, t - 1, k)
        if shuffle_set_feats:
            perm = torch.randperm(bsz, device=device)
            set_feats = set_feats[perm]
        scores = model(cand_feats, set_feats)
        scores_masked = scores.masked_fill(selected_mask, float('-inf'))

        if target_kind == 'aggregate':
            # u_i^(t) = AggMSE(S_{t-1}) - AggMSE(S_{t-1} U {i}); higher = better.
            # t=1 has an empty prefix (no well-defined AggMSE(S_0)), so u is simply -AggMSE({i}).
            cand_mean = (running_sum.unsqueeze(1) + top20_future) / t
            new_agg_mse = ((cand_mean - query_future.unsqueeze(1)) ** 2).mean(dim=-1)
            if t > 1:
                cur_mean = running_sum / (t - 1)
                cur_agg_mse = ((cur_mean.unsqueeze(1) - query_future.unsqueeze(1)) ** 2).mean(dim=-1)
                u = cur_agg_mse - new_agg_mse
            else:
                u = -new_agg_mse
        else:  # 'individual' control (D2): plain per-candidate future MSE, no set interaction
            u = -individual_mse

        u = u.masked_fill(selected_mask, float('-inf'))
        p_t = F.softmax(u / tau_u, dim=-1).detach()
        log_p_s = F.log_softmax(scores_masked / tau_s_set, dim=-1)
        # NOT F.kl_div(log_p_s, p_t) here: once `selected_mask` has any True entries
        # (t>1), BOTH p_t and log_p_s are exactly 0 / -inf at those positions, and
        # `target * (log(target) - input)` hits the classic 0*(-inf)=NaN trap. Masking
        # the per-element term to a literal 0.0 (not relying on p_t=0 to cancel it)
        # avoids this -- every selected position's gradient contribution is exactly
        # zero either way, since the model must never be pushed to re-pick it.
        term = (p_t * (p_t.clamp_min(1e-12).log() - log_p_s)).masked_fill(selected_mask, 0.0)
        step_loss = term.sum(dim=-1).mean()
        loss_sum = loss_sum + step_loss

        with torch.no_grad():
            best_j = scores_masked.argmax(dim=-1)
        picks.append(best_j)
        selected_mask = selected_mask.scatter(1, best_j.unsqueeze(1), True)
        running_sum = running_sum + top20_future.gather(1, best_j.view(-1, 1, 1).expand(-1, 1, h)).squeeze(1)

    return loss_sum / k, torch.stack(picks, dim=1)


@torch.no_grad()
def compute_cell_tensors(cli, device, split, exp, args, model, slot_heads, gamma, pool_cache):
    """One streaming generator over (channel,batch): yields everything
    `on_policy_rollout` needs for one batch, restricted to the primary
    scorer's own fixed Top-20."""
    channels = list(range(int(args.enc_in)))
    _, loader = exp._get_data(flag=split, shuffle=False)
    for c in channels:
        k_full = F.normalize(encode_raw(model, exp.memory_x, c), dim=-1)
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
            p_heads_full = F.softmax(s_final_full / cli.tau_s, dim=-1)
            p_mix_full = p_heads_full.mean(dim=1)
            r_mix_full = r_full.mean(dim=1)

            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            pooled_future_full = memory_c[pool_idx_global] + offset_c.view(-1, 1, 1)
            # candidate's OWN past-window norm (query-independent), a genuinely candidate-side summary:
            cand_hist = exp.memory_x[pool_idx_global][:, :, :, c]  # [B,100,L]
            cand_temporal_summary_full = (cand_hist - cand_hist[:, :, -1:]).norm(dim=-1)  # [B,100]

            top20_local = stable_topk_indices(p_mix_full, TOP_M, largest=True)
            gather20 = lambda t: t.gather(1, top20_local)  # noqa: E731
            primary_score = gather20(p_mix_full)
            coarse_score = gather20(c_i_full)
            residual_mix_score = gather20(r_mix_full)
            temporal_summary = gather20(cand_temporal_summary_full)
            top20_future = pooled_future_full.gather(1, top20_local.unsqueeze(-1).expand(-1, -1, pooled_future_full.size(-1)))
            # k_pool (= k_full gathered at the P100 pool indices) already holds this
            # channel's candidate embeddings -- gather the Top-20 subset directly instead
            # of re-running `encode_raw` on their raw histories a second time.
            z_cand = k_pool.gather(1, top20_local.unsqueeze(-1).expand(-1, -1, k_pool.size(-1)))

            yield dict(z_q=z_q, z_cand=z_cand, primary_score=primary_score, coarse_score=coarse_score,
                      residual_mix_score=residual_mix_score, temporal_summary=temporal_summary,
                      top20_future=top20_future, query_future=query_future)


@torch.no_grad()
def eval_arm(model_fn, cli, device, split, exp, args, model, slot_heads, gamma, pool_cache):
    aggs = []
    for b in compute_cell_tensors(cli, device, split, exp, args, model, slot_heads, gamma, pool_cache):
        picks = model_fn(b)  # [B,10] local indices into the 20
        y_sel = b['top20_future'].gather(1, picks.unsqueeze(-1).expand(-1, -1, b['top20_future'].size(-1)))
        agg = ((y_sel.mean(dim=1) - b['query_future']) ** 2).mean(dim=-1)
        aggs.append(agg.cpu())
    return float(torch.cat(aggs).mean())


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
    ap.add_argument('--residual_checkpoint', required=True)
    ap.add_argument('--train_epochs', type=int, default=10)
    ap.add_argument('--learning_rate', type=float, default=1e-3)
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
    bl = torch.load(cli.residual_checkpoint, map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    slot_heads.load_state_dict(bl['slot_heads_state_dict'])
    gamma = float(bl['gamma'])
    slot_heads.eval()
    print(f'[audit] cell={cli.cell} residual checkpoint epoch={bl["epoch"]} gamma={gamma:.6f}')

    expected_meta = {'seq_len': cli.seq_len, 'pred_len': cli.pred_len, 'candidate_mask': cli.candidate_mask,
                     'candidate_pool_size': cli.candidate_pool_size, 'candidate_pool_metric': 'delta_last_cosine'}
    pool_caches = {s: CandidatePoolCache(Path(cli.candidate_pool_cache) / f'{s}.pt', expected_meta)
                  for s in ('train', 'val', 'test')}

    d_model = int(args.d_model)

    def train_arm(target_kind, seed):
        torch.manual_seed(seed)
        reranker = SetAwareReranker(d_model).to(device)
        opt = torch.optim.Adam(reranker.parameters(), lr=cli.learning_rate)
        best_val, best_state = float('inf'), None
        for epoch in range(1, cli.train_epochs + 1):
            reranker.train()
            for b in compute_cell_tensors(cli, device, 'train', exp, args, model, slot_heads, gamma, pool_caches['train']):
                loss, _ = on_policy_rollout(reranker, b['z_q'], b['z_cand'], b['primary_score'], b['coarse_score'],
                                           b['residual_mix_score'], b['temporal_summary'], b['top20_future'],
                                           b['query_future'], TOP_K, TAU_U, TAU_S_SET, target_kind=target_kind)
                opt.zero_grad(); loss.backward(); opt.step()
            reranker.eval()

            def _model_fn(b):
                _, picks = on_policy_rollout(reranker, b['z_q'], b['z_cand'], b['primary_score'], b['coarse_score'],
                                            b['residual_mix_score'], b['temporal_summary'], b['top20_future'],
                                            b['query_future'], TOP_K, TAU_U, TAU_S_SET, target_kind=target_kind)
                return picks
            val_agg = eval_arm(_model_fn, cli, device, 'val', exp, args, model, slot_heads, gamma, pool_caches['val'])
            print(f'[set_reranker] target={target_kind} seed={seed} epoch={epoch} val_agg_mse10={val_agg:.6f}')
            if val_agg < best_val:
                best_val = val_agg
                best_state = {k: v.clone() for k, v in reranker.state_dict().items()}
        reranker.load_state_dict(best_state)
        return reranker, best_val

    results = {}

    # D0: primary scorer Top10 (no set selector) -- trivial, already computed in diag_p100_set_oracle01
    def _d0_fn(b):
        return stable_topk_indices(b['primary_score'], TOP_K, largest=True)
    results['D0_primary_top10'] = {'test_agg_mse10': eval_arm(
        _d0_fn, cli, device, 'test', exp, args, model, slot_heads, gamma, pool_caches['test'])}

    # D1: learned set-aware selector (aggregate target, on-policy)
    model_d1, val_d1 = train_arm('aggregate', seed=0)

    def _d1_fn(b):
        _, picks = on_policy_rollout(model_d1, b['z_q'], b['z_cand'], b['primary_score'], b['coarse_score'],
                                     b['residual_mix_score'], b['temporal_summary'], b['top20_future'],
                                     b['query_future'], TOP_K, TAU_U, TAU_S_SET, target_kind='aggregate')
        return picks
    results['D1_learned_set_aware'] = {'val_agg_mse10': val_d1, 'test_agg_mse10': eval_arm(
        _d1_fn, cli, device, 'test', exp, args, model, slot_heads, gamma, pool_caches['test'])}

    # D2: same architecture, individual (non-aggregate) target control
    model_d2, val_d2 = train_arm('individual', seed=0)

    def _d2_fn(b):
        _, picks = on_policy_rollout(model_d2, b['z_q'], b['z_cand'], b['primary_score'], b['coarse_score'],
                                     b['residual_mix_score'], b['temporal_summary'], b['top20_future'],
                                     b['query_future'], TOP_K, TAU_U, TAU_S_SET, target_kind='individual')
        return picks
    results['D2_individual_target_control'] = {'val_agg_mse10': val_d2, 'test_agg_mse10': eval_arm(
        _d2_fn, cli, device, 'test', exp, args, model, slot_heads, gamma, pool_caches['test'])}

    # D3: shuffled set-state control -- D1's trained model, but the current-SET-SUMMARY
    # features (not candidate features, not the environment) are permuted across the
    # batch at every step (see `on_policy_rollout`'s `shuffle_set_feats`).
    def _d3_fn(b):
        _, picks = on_policy_rollout(model_d1, b['z_q'], b['z_cand'], b['primary_score'], b['coarse_score'],
                                     b['residual_mix_score'], b['temporal_summary'], b['top20_future'],
                                     b['query_future'], TOP_K, TAU_U, TAU_S_SET, target_kind='aggregate',
                                     shuffle_set_feats=True)
        return picks
    results['D3_shuffled_set_state_control'] = {'test_agg_mse10': eval_arm(
        _d3_fn, cli, device, 'test', exp, args, model, slot_heads, gamma, pool_caches['test'])}

    out = {'cell': cli.cell, 'controls': results}
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'{cli.cell}_phaseD_reranker.json').write_text(json.dumps(out, indent=2))
    print(f'[set_reranker] {cli.cell}: {json.dumps(results, indent=2)}')


if __name__ == '__main__':
    main()
