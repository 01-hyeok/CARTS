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
the 5 beta candidates, pooled over TRAIN (a standard self-normalizing
choice: scales the softmax so the teacher target is neither
near-one-hot nor near-uniform relative to the data's own noise
scale), computed once per cell at the start of `main()` and then
fixed for that cell's own training (every other hyperparameter is
identical and global across cells; tau_beta is the one quantity that
is inherently scale-dependent on AggMSE's own units, which differ by
cell/horizon, so calibrating it per-cell from TRAIN-only statistics is
not equivalent to tuning on held-out data).
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


@torch.no_grad()
def collect_rows(cli, device, split, exp, args, model, slot_heads, pool_cache):
    """Returns a dict of stacked CPU tensors: feats [Q,F], agg_mse_by_beta [Q,n_beta]
    (future-informed, TRAIN-only usage allowed; used for oracle target / eval),
    p_c [Q,100], p_v5 [Q,100], d_pool [Q,100], pooled_future [Q,100,H], query_future [Q,H]."""
    channels = list(range(int(args.enc_in)))
    _, loader = exp._get_data(flag=split, shuffle=False)
    all_feats, all_agg, all_pc, all_pv5, all_dpool, all_pf, all_qf = [], [], [], [], [], [], []

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
            d_pool = ((pooled_future - query_future.unsqueeze(1)) ** 2).mean(dim=-1)

            feats = extract_features(p_c, p_v5, p_heads, z_q, c, len(channels))

            agg_this = []
            for beta in BETAS:
                raw = (1 - beta) * torch.log(p_c + EPS) + beta * torch.log(p_v5 + EPS)
                p_fused = F.softmax(raw, dim=-1)
                picks = stable_topk_indices(p_fused, TOP_K, largest=True)
                y_sel = pooled_future.gather(1, picks.unsqueeze(-1).expand(-1, -1, pooled_future.size(-1)))
                agg_mse = ((y_sel.mean(dim=1) - query_future) ** 2).mean(dim=-1)
                agg_this.append(agg_mse)
            agg_stack = torch.stack(agg_this, dim=-1)  # [B, n_beta]

            all_feats.append(feats.cpu()); all_agg.append(agg_stack.cpu())
            all_pc.append(p_c.cpu()); all_pv5.append(p_v5.cpu()); all_dpool.append(d_pool.cpu())
            all_pf.append(pooled_future.cpu()); all_qf.append(query_future.cpu())

    return {'feats': torch.cat(all_feats), 'agg_mse_by_beta': torch.cat(all_agg),
           'p_c': torch.cat(all_pc), 'p_v5': torch.cat(all_pv5), 'd_pool': torch.cat(all_dpool),
           'pooled_future': torch.cat(all_pf), 'query_future': torch.cat(all_qf)}


def fused_agg_mse_from_expected_beta(beta_expected, p_c, p_v5, pooled_future, query_future):
    """beta_expected: [B] continuous in [0,1]. Returns AggMSE@10 [B]."""
    beta = beta_expected.unsqueeze(-1)  # [B,1]
    raw = (1 - beta) * torch.log(p_c + EPS) + beta * torch.log(p_v5 + EPS)
    p_fused = F.softmax(raw, dim=-1)
    picks = stable_topk_indices(p_fused, TOP_K, largest=True)
    y_sel = pooled_future.gather(1, picks.unsqueeze(-1).expand(-1, -1, pooled_future.size(-1)))
    return ((y_sel.mean(dim=1) - query_future) ** 2).mean(dim=-1)


def train_one_model(model_ctor, train_data, val_data, tau_beta, device, shuffle_features=False,
                    epochs=30, lr=1e-3, batch_size=256, seed=0):
    torch.manual_seed(seed)
    model = model_ctor().to(device)
    if hasattr(model, 'feature_mean'):
        mean = train_data['feats'].mean(dim=0)
        std = train_data['feats'].std(dim=0).clamp_min(1e-6)
        model.feature_mean.data.copy_(mean.to(device))
        model.feature_std.data.copy_(std.to(device))

    beta_t = torch.tensor(BETAS, device=device)
    target = F.softmax(-train_data['agg_mse_by_beta'].to(device) / tau_beta, dim=-1)  # [N, n_beta]
    feats = train_data['feats'].to(device)
    if shuffle_features:
        perm = torch.randperm(feats.size(0), generator=torch.Generator().manual_seed(seed))
        feats = feats[perm]
    n = feats.size(0)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float('inf')
    best_state = None
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            p_pred = model(feats[idx]).clamp_min(1e-12)
            loss = F.kl_div(p_pred.log(), target[idx], reduction='batchmean')
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            p_val = model(val_data['feats'].to(device))
            beta_expected = (p_val * beta_t.unsqueeze(0)).sum(dim=-1)
            val_agg = fused_agg_mse_from_expected_beta(
                beta_expected, val_data['p_c'].to(device), val_data['p_v5'].to(device),
                val_data['pooled_future'].to(device), val_data['query_future'].to(device))
            val_mean = float(val_agg.mean())
        if val_mean < best_val:
            best_val = val_mean
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

    print(f'[audit] cell={cli.cell} collecting TRAIN/VAL/TEST rows ...')
    train_data = collect_rows(cli, device, 'train', exp, args, model, slot_heads, pool_caches['train'])
    val_data = collect_rows(cli, device, 'val', exp, args, model, slot_heads, pool_caches['val'])
    test_data = collect_rows(cli, device, 'test', exp, args, model, slot_heads, pool_caches['test'])
    print(f'[audit] {cli.cell}: train_n={train_data["feats"].size(0)} val_n={val_data["feats"].size(0)} '
         f'test_n={test_data["feats"].size(0)}')

    tau_beta = float(train_data['agg_mse_by_beta'].std())
    print(f'[audit] {cli.cell}: tau_beta (TRAIN-only std of AggMSE(beta)) = {tau_beta:.6f}')

    results = {}

    # B0: best global fixed beta from Phase A (selected on VAL)
    val_mean_by_beta = val_data['agg_mse_by_beta'].mean(dim=0)
    best_j = int(val_mean_by_beta.argmin())
    b0_beta = BETAS[best_j]
    b0_test_agg = float(test_data['agg_mse_by_beta'][:, best_j].mean())
    results['B0_best_global_fixed'] = {'beta': b0_beta, 'val_agg_mse10': float(val_mean_by_beta[best_j]),
                                       'test_agg_mse10': b0_test_agg}

    # B1: adaptive MLP
    model_b1, val_b1 = train_one_model(
        lambda: AdaptiveBetaMLP(N_FEATURES, N_BETA), train_data, val_data, tau_beta, device,
        shuffle_features=False, epochs=cli.epochs)
    beta_t = torch.tensor(BETAS, device=device)
    with torch.no_grad():
        p_test = model_b1(test_data['feats'].to(device))
        beta_expected_test = (p_test * beta_t.unsqueeze(0)).sum(dim=-1)
        test_agg_b1 = fused_agg_mse_from_expected_beta(
            beta_expected_test, test_data['p_c'].to(device), test_data['p_v5'].to(device),
            test_data['pooled_future'].to(device), test_data['query_future'].to(device))
    results['B1_adaptive'] = {'val_agg_mse10': val_b1, 'test_agg_mse10': float(test_agg_b1.mean()),
                              'beta_expected_mean_test': float(beta_expected_test.mean()),
                              'beta_expected_std_test': float(beta_expected_test.std())}

    # B2: static trainable scalar (no features)
    model_b2, val_b2 = train_one_model(
        lambda: StaticScalarBeta(N_BETA), train_data, val_data, tau_beta, device,
        shuffle_features=False, epochs=cli.epochs)
    with torch.no_grad():
        p_test2 = model_b2(test_data['feats'].to(device))
        beta_expected_test2 = (p_test2 * beta_t.unsqueeze(0)).sum(dim=-1)
        test_agg_b2 = fused_agg_mse_from_expected_beta(
            beta_expected_test2, test_data['p_c'].to(device), test_data['p_v5'].to(device),
            test_data['pooled_future'].to(device), test_data['query_future'].to(device))
    results['B2_static_scalar'] = {'val_agg_mse10': val_b2, 'test_agg_mse10': float(test_agg_b2.mean()),
                                   'beta_learned': float(beta_expected_test2.mean())}

    # B3: adaptive MLP, shuffled feature<->query pairing in TRAIN only
    model_b3, val_b3 = train_one_model(
        lambda: AdaptiveBetaMLP(N_FEATURES, N_BETA), train_data, val_data, tau_beta, device,
        shuffle_features=True, epochs=cli.epochs)
    with torch.no_grad():
        p_test3 = model_b3(test_data['feats'].to(device))
        beta_expected_test3 = (p_test3 * beta_t.unsqueeze(0)).sum(dim=-1)
        test_agg_b3 = fused_agg_mse_from_expected_beta(
            beta_expected_test3, test_data['p_c'].to(device), test_data['p_v5'].to(device),
            test_data['pooled_future'].to(device), test_data['query_future'].to(device))
    results['B3_shuffled_control'] = {'val_agg_mse10': val_b3, 'test_agg_mse10': float(test_agg_b3.mean())}

    out = {'cell': cli.cell, 'tau_beta': tau_beta, 'feature_names': FEATURE_NAMES, 'controls': results}
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'{cli.cell}_phaseB_adaptive.json').write_text(json.dumps(out, indent=2))
    print(f'[train_adaptive_fusion] {cli.cell}: {json.dumps(results, indent=2)}')


if __name__ == '__main__':
    main()
