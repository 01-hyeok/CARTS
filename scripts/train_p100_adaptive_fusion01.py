#!/usr/bin/env python3
"""TRACK-P100-COARSE-TO-FINE-RETRIEVAL01 -- PHASE B step 2: query
-adaptive coarse/V5 trust fusion. A tiny 2-layer MLP predicts a
5-way distribution over the SAME discrete beta candidates Phase A
already swept ({0,.25,.5,.75,1}), from past-only (future-blind)
confidence features computed over the SAME frozen Shared-P100 support
and the SAME frozen canonical V5 checkpoint -- no new encoder, no new
Stage-1 training, no architecture change to V5 itself. The predicted
distribution's EXPECTED beta (sum_j p_j*beta_j, a continuous value)
is plugged into the identical log-domain fusion formula Phase A uses
(`scripts.diag_p100_coarse_v5_fusion01`'s (1-beta)*log(p_c)+beta*log(p_v5)).

Spec section 14 note on an apparent inconsistency, resolved here and
recorded rather than silently picked: section 14 first writes
"beta(q,c)=sigmoid(f(features))" (a scalar), then immediately below
defines a discrete oracle-beta SOFT TARGET distribution over the 5
candidates, a cross-entropy/KL loss against it, and "inference beta =
sum_j p_j*beta_j" (an expectation over a distribution). These two
descriptions are only consistent with each other if the model head is
a 5-way softmax (not a scalar sigmoid) trained against the soft
target by KL/CE, with the scalar "beta(q,c)" being the resulting
EXPECTED value, not the model's raw output. That is the interpretation
implemented here.

Mandatory controls (section 15): B0 best global fixed beta from Phase
A (no model); B1 this adaptive model; B2 a single trainable global
scalar beta (same training data, same loss family, zero input
features); B3 adaptive model with features shuffled across queries
(breaks the query<->feature pairing, keeps architecture/training
identical) -- B1 must beat B0/B2/B3 or there is no adaptive
contribution.

tau_beta (soft-target temperature): one-time TRAIN-only calibration,
not swept per cell -- set to the per-query std of AggMSE(beta) across
the 5 beta candidates, pooled over TRAIN.

MEMORY DISCIPLINE (corrected after a design review before any GPU run
was launched at scale -- recorded here, not silently fixed): an
earlier version of this script cached `p_c`/`p_v5`/`pooled_future`
/`query_future` for every (query, channel) row of every split,
including TRAIN. `pooled_future` alone is `[rows, 100, H]` -- for
Weather H720 (21 channels x ~35k-55k queries per split) this is
50-200+ GB, which does not fit in RAM. Those tensors are never
materialized for more than ONE BATCH at a time: `collect_features_and_targets`
caches only the tiny per-row `feats [N,F]` and `agg_mse_by_beta
[N,5]` (a few tens of MB even for Weather), used both as the KL
training target AND as the (cheap) per-epoch validation loss for
model selection; the actual deployable AggMSE number for a trained
model is computed by `evaluate_beta_model_streaming`, which re-walks
the data batch-by-batch (recomputing p_c/p_v5/pooled_future fresh per
batch, exactly as `diag_p100_coarse_v5_fusion01` does) and only
accumulates a running mean -- called once per model, per split, after
training is finished, never per-epoch.
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
from scripts.diag_p100_coarse_v5_fusion01 import BETAS, EPS, TAU_C, TOP_K, N_SLOTS, spearman_batched
from scripts.train_factorial_e2e01 import encode_raw
from scripts.train_j_shared_encoder_drift01 import build_model
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_margutil01 import memory_value
from scripts.train_retriever_pool01 import CandidatePoolCache
from utils.candidate_pool import compute_coarse_delta_last_cosine_scores

N_BETA = len(BETAS)
FEATURE_NAMES = [
    'v5_entropy', 'v5_top1_prob', 'v5_top1_top2_margin', 'v5_top1_top10_margin',
    'v5_head_entropy_mean', 'v5_pairwise_head_top10_overlap', 'v5_head_disagreement',
    'coarse_entropy', 'coarse_top1_top2_margin', 'coarse_top1_top10_margin',
    'disagree_top10_overlap', 'disagree_spearman', 'disagree_top1_match',
    'disagree_js_divergence', 'disagree_rank_disp_mean', 'disagree_rank_disp_p90',
    'zq_norm', 'channel_id_norm',
]
N_FEATURES = len(FEATURE_NAMES)


def _entropy(p, eps=1e-12):
    return -(p * (p + eps).log()).sum(dim=-1)


def _margin(p, k_lo, k_hi):
    vals = p.topk(k_hi, dim=-1, largest=True).values  # [B,k_hi] descending
    return vals[:, k_lo - 1] - vals[:, k_hi - 1]


def _js_divergence(p, q, eps=1e-12):
    m = 0.5 * (p + q)
    kl_pm = (p * ((p + eps).log() - (m + eps).log())).sum(dim=-1)
    kl_qm = (q * ((q + eps).log() - (m + eps).log())).sum(dim=-1)
    return 0.5 * (kl_pm + kl_qm)


def extract_features(p_c, p_v5, p_heads, z_q, channel, n_channels):
    """All inputs are [B,...]; p_c/p_v5: [B,100], p_heads: [B,5,100],
    z_q: [B,D] (PRE-normalization encoder output -- its norm is a
    legitimate deployable past-only statistic already present on the
    production forward path, nothing new is computed to get it)."""
    bsz = p_c.size(0)
    rank_c = p_c.argsort(dim=-1).argsort(dim=-1).float()
    rank_v5 = p_v5.argsort(dim=-1).argsort(dim=-1).float()
    rank_disp = (rank_c - rank_v5).abs()

    head_top10 = [stable_topk_indices(p_heads[:, m, :], TOP_K, largest=True) for m in range(N_SLOTS)]
    pair_overlaps = []
    for m in range(N_SLOTS):
        for mm in range(m + 1, N_SLOTS):
            inter = (head_top10[m].unsqueeze(-1) == head_top10[mm].unsqueeze(-2)).any(-1).float().sum(-1)
            pair_overlaps.append(inter / TOP_K)
    head_top10_overlap = torch.stack(pair_overlaps, dim=-1).mean(dim=-1)  # [B]
    head_entropy_mean = torch.stack([_entropy(p_heads[:, m, :]) for m in range(N_SLOTS)], dim=-1).mean(dim=-1)

    top10_c = stable_topk_indices(p_c, TOP_K, largest=True)
    top10_v5 = stable_topk_indices(p_v5, TOP_K, largest=True)
    overlap10 = (top10_c.unsqueeze(-1) == top10_v5.unsqueeze(-2)).any(-1).float().sum(-1) / TOP_K
    top1_match = (top10_c[:, 0] == top10_v5[:, 0]).float()

    channel_id_norm = torch.full((bsz,), float(channel) / max(n_channels - 1, 1), device=p_c.device)

    feats = torch.stack([
        _entropy(p_v5), p_v5.max(dim=-1).values, _margin(p_v5, 1, 2), _margin(p_v5, 1, TOP_K),
        head_entropy_mean, head_top10_overlap, 1.0 - head_top10_overlap,
        _entropy(p_c), _margin(p_c, 1, 2), _margin(p_c, 1, TOP_K),
        overlap10, spearman_batched(p_c, p_v5), top1_match,
        _js_divergence(p_c, p_v5), rank_disp.mean(dim=-1), rank_disp.quantile(0.9, dim=-1),
        z_q.norm(dim=-1), channel_id_norm,
    ], dim=-1)  # [B, N_FEATURES]
    return feats


class AdaptiveBetaMLP(nn.Module):
    def __init__(self, n_features, n_beta, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(),
                                 nn.Linear(hidden, n_beta))
        self.feature_mean = nn.Parameter(torch.zeros(n_features), requires_grad=False)
        self.feature_std = nn.Parameter(torch.ones(n_features), requires_grad=False)

    def forward(self, feats):
        x = (feats - self.feature_mean) / self.feature_std
        logits = self.net(x)
        return F.softmax(logits, dim=-1)  # [B, n_beta]


class StaticScalarBeta(nn.Module):
    """B2 control: one trainable global scalar beta, no features at all."""

    def __init__(self, n_beta):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_beta))

    def forward(self, feats):
        bsz = feats.size(0)
        return F.softmax(self.logits, dim=-1).unsqueeze(0).expand(bsz, -1)


def _batch_iter(cli, device, split, exp, args, model, slot_heads, pool_cache):
    """Shared per-(channel,batch) generator yielding (p_c, p_v5, p_heads,
    z_q, c, channels, pooled_future, query_future) -- the ONE place both
    `collect_features_and_targets` and `evaluate_beta_model_streaming`
    compute these, so they can never silently drift apart."""
    channels = list(range(int(args.enc_in)))
    _, loader = exp._get_data(flag=split, shuffle=False)
    for c in channels:
        k_full = F.normalize(encode_raw(model, exp.memory_x, c), dim=-1)
        for batch_x, batch_y, batch_start_idx in loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            pool_idx_global = pool_cache.lookup(batch_start_idx, c, device)

            coarse_full = compute_coarse_delta_last_cosine_scores(batch_x, exp.memory_x, c)
            c_i = coarse_full.gather(1, pool_idx_global)
            p_c = F.softmax(c_i / TAU_C, dim=-1)

            z_q = encode_raw(model, batch_x, c)
            q = slot_heads(z_q)
            k_pool = k_full[pool_idx_global]
            scores = torch.einsum('bsd,bmd->bsm', q, k_pool)
            p_heads = F.softmax(scores / cli.tau_s, dim=-1)
            p_v5 = p_heads.mean(dim=1)

            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            pooled_future = memory_c[pool_idx_global] + offset_c.view(-1, 1, 1)

            yield p_c, p_v5, p_heads, z_q, c, len(channels), pooled_future, query_future


@torch.no_grad()
def collect_features_and_targets(cli, device, split, exp, args, model, slot_heads, pool_cache):
    """Returns ONLY the small, cacheable-for-a-full-split tensors:
    feats [N,F], agg_mse_by_beta [N,5] (future-informed -- fine for
    TRAIN; for VAL/TEST this is used only as a cheap proxy loss for
    model selection, never as the reported deployable number, which
    comes from `evaluate_beta_model_streaming` instead)."""
    all_feats, all_agg = [], []
    for p_c, p_v5, p_heads, z_q, c, n_ch, pooled_future, query_future in _batch_iter(
            cli, device, split, exp, args, model, slot_heads, pool_cache):
        feats = extract_features(p_c, p_v5, p_heads, z_q, c, n_ch)
        agg_this = []
        for beta in BETAS:
            raw = (1 - beta) * torch.log(p_c + EPS) + beta * torch.log(p_v5 + EPS)
            p_fused = F.softmax(raw, dim=-1)
            picks = stable_topk_indices(p_fused, TOP_K, largest=True)
            y_sel = pooled_future.gather(1, picks.unsqueeze(-1).expand(-1, -1, pooled_future.size(-1)))
            agg_mse = ((y_sel.mean(dim=1) - query_future) ** 2).mean(dim=-1)
            agg_this.append(agg_mse)
        all_feats.append(feats.cpu()); all_agg.append(torch.stack(agg_this, dim=-1).cpu())
    return {'feats': torch.cat(all_feats), 'agg_mse_by_beta': torch.cat(all_agg)}


@torch.no_grad()
def evaluate_beta_model_streaming(cli, device, split, exp, args, model, slot_heads, pool_cache, beta_model):
    """Re-walks `split` batch-by-batch (never caching more than one
    batch's `pooled_future`/`query_future` at a time), runs the ALREADY
    -TRAINED `beta_model` on each batch's features to get a continuous
    expected beta, fuses, and returns the mean AggMSE@10 plus beta
    summary stats -- the one place a model's REPORTED number comes
    from."""
    beta_t = torch.tensor(BETAS, device=device)
    total, count = 0.0, 0
    beta_vals = []
    for p_c, p_v5, p_heads, z_q, c, n_ch, pooled_future, query_future in _batch_iter(
            cli, device, split, exp, args, model, slot_heads, pool_cache):
        feats = extract_features(p_c, p_v5, p_heads, z_q, c, n_ch)
        p_pred = beta_model(feats)
        beta_expected = (p_pred * beta_t.unsqueeze(0)).sum(dim=-1)
        raw = (1 - beta_expected.unsqueeze(-1)) * torch.log(p_c + EPS) + \
            beta_expected.unsqueeze(-1) * torch.log(p_v5 + EPS)
        p_fused = F.softmax(raw, dim=-1)
        picks = stable_topk_indices(p_fused, TOP_K, largest=True)
        y_sel = pooled_future.gather(1, picks.unsqueeze(-1).expand(-1, -1, pooled_future.size(-1)))
        agg_mse = ((y_sel.mean(dim=1) - query_future) ** 2).mean(dim=-1)
        total += float(agg_mse.sum()); count += agg_mse.size(0)
        beta_vals.append(beta_expected.cpu())
    beta_cat = torch.cat(beta_vals)
    return {'agg_mse10': total / count, 'beta_mean': float(beta_cat.mean()), 'beta_std': float(beta_cat.std())}


def train_one_model(model_ctor, train_data, val_data, tau_beta, device, shuffle_features=False,
                    epochs=30, lr=1e-3, batch_size=256, seed=0):
    """Model selection uses the CHEAP cached-feature KL/CE validation
    loss (same quantity being optimized), not a full streaming AggMSE
    pass -- that full pass is reserved for the one-time post-training
    report via `evaluate_beta_model_streaming`."""
    torch.manual_seed(seed)
    model = model_ctor().to(device)
    if hasattr(model, 'feature_mean'):
        mean = train_data['feats'].mean(dim=0)
        std = train_data['feats'].std(dim=0).clamp_min(1e-6)
        model.feature_mean.data.copy_(mean.to(device))
        model.feature_std.data.copy_(std.to(device))

    target_train = F.softmax(-train_data['agg_mse_by_beta'].to(device) / tau_beta, dim=-1)
    feats_train = train_data['feats'].to(device)
    if shuffle_features:
        perm = torch.randperm(feats_train.size(0), generator=torch.Generator().manual_seed(seed))
        feats_train = feats_train[perm]
    n = feats_train.size(0)

    feats_val = val_data['feats'].to(device)
    target_val = F.softmax(-val_data['agg_mse_by_beta'].to(device) / tau_beta, dim=-1)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, best_state = float('inf'), None
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            p_pred = model(feats_train[idx]).clamp_min(1e-12)
            loss = F.kl_div(p_pred.log(), target_train[idx], reduction='batchmean')
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            p_val = model(feats_val).clamp_min(1e-12)
            val_loss = float(F.kl_div(p_val.log(), target_val, reduction='batchmean'))
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_val


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
    ap.add_argument('--epochs', type=int, default=30)
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
    bl = torch.load(cli.v5_checkpoint, map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    slot_heads.load_state_dict(bl['slot_heads_state_dict'])
    slot_heads.eval()

    expected_meta = {'seq_len': cli.seq_len, 'pred_len': cli.pred_len, 'candidate_mask': cli.candidate_mask,
                     'candidate_pool_size': cli.candidate_pool_size, 'candidate_pool_metric': 'delta_last_cosine'}
    pool_caches = {s: CandidatePoolCache(Path(cli.candidate_pool_cache) / f'{s}.pt', expected_meta)
                  for s in ('train', 'val', 'test')}

    print(f'[audit] cell={cli.cell} collecting TRAIN/VAL/TEST features+beta-grid targets (small cache only) ...')
    train_data = collect_features_and_targets(cli, device, 'train', exp, args, model, slot_heads, pool_caches['train'])
    val_data = collect_features_and_targets(cli, device, 'val', exp, args, model, slot_heads, pool_caches['val'])
    print(f'[audit] {cli.cell}: train_n={train_data["feats"].size(0)} val_n={val_data["feats"].size(0)}')

    tau_beta = float(train_data['agg_mse_by_beta'].std())
    print(f'[audit] {cli.cell}: tau_beta (TRAIN-only std of AggMSE(beta)) = {tau_beta:.6f}')

    results = {}

    # B0: best global fixed beta from Phase A (selected on VAL, from the cached beta-grid)
    val_mean_by_beta = val_data['agg_mse_by_beta'].mean(dim=0)
    best_j = int(val_mean_by_beta.argmin())
    b0_beta = BETAS[best_j]

    def _const_beta_model(beta_val):
        class _Const(nn.Module):
            def forward(self, feats):
                p = torch.zeros(feats.size(0), N_BETA, device=feats.device)
                j = BETAS.index(beta_val)
                p[:, j] = 1.0
                return p
        return _Const().to(device)

    b0_model = _const_beta_model(b0_beta)
    b0_test = evaluate_beta_model_streaming(cli, device, 'test', exp, args, model, slot_heads, pool_caches['test'], b0_model)
    results['B0_best_global_fixed'] = {'beta': b0_beta, 'val_agg_mse10': float(val_mean_by_beta[best_j]), **b0_test}

    # B1: adaptive MLP
    model_b1, val_loss_b1 = train_one_model(
        lambda: AdaptiveBetaMLP(N_FEATURES, N_BETA), train_data, val_data, tau_beta, device,
        shuffle_features=False, epochs=cli.epochs)
    test_b1 = evaluate_beta_model_streaming(cli, device, 'test', exp, args, model, slot_heads, pool_caches['test'], model_b1)
    results['B1_adaptive'] = {'val_kl_loss': val_loss_b1, **test_b1}

    # B2: static trainable scalar (no features)
    model_b2, val_loss_b2 = train_one_model(
        lambda: StaticScalarBeta(N_BETA), train_data, val_data, tau_beta, device,
        shuffle_features=False, epochs=cli.epochs)
    test_b2 = evaluate_beta_model_streaming(cli, device, 'test', exp, args, model, slot_heads, pool_caches['test'], model_b2)
    results['B2_static_scalar'] = {'val_kl_loss': val_loss_b2, **test_b2}

    # B3: adaptive MLP, shuffled feature<->query pairing in TRAIN only
    model_b3, val_loss_b3 = train_one_model(
        lambda: AdaptiveBetaMLP(N_FEATURES, N_BETA), train_data, val_data, tau_beta, device,
        shuffle_features=True, epochs=cli.epochs)
    test_b3 = evaluate_beta_model_streaming(cli, device, 'test', exp, args, model, slot_heads, pool_caches['test'], model_b3)
    results['B3_shuffled_control'] = {'val_kl_loss': val_loss_b3, **test_b3}

    out = {'cell': cli.cell, 'tau_beta': tau_beta, 'feature_names': FEATURE_NAMES, 'controls': results}
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'{cli.cell}_phaseB_adaptive.json').write_text(json.dumps(out, indent=2))
    print(f'[train_adaptive_fusion] {cli.cell}: {json.dumps(results, indent=2)}')


if __name__ == '__main__':
    main()
