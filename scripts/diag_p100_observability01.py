#!/usr/bin/env python3
"""TRACK-P100-RETRIEVAL-UTILIZATION-AUDIT01 -- PHASE B: past-only
observability diagnostic. Tests whether a query/channel where
retrieval is useful can be identified from PAST-ONLY information
(never the true future Y, never any statistic derived from Y). No new
forecasting model, no new Stage1 encoder/router/Transformer -- only
the three tiny probes spec section 16 explicitly allows (B0 constant,
B1 logistic regression, B2 a 2-layer MLP with hidden=32).

Scope decision (recorded, not silent): Phase A found V5 and Phase D
Set-aware produce nearly identical Oracle Lambda behavior on every
cell (`PHASE_A_ORACLE_LAMBDA_REPORT.md` Tables 1-4 -- every number
within a few percent between the two arms). Phase B therefore
analyzes ONLY the canonical V5 arm, to avoid duplicating a diagnostic
that Phase A already showed would not materially differ for Phase D.

Features (spec section 13): the 18 V5-confidence/coarse-confidence/
disagreement features are `scripts.train_p100_adaptive_fusion01.extract_features`
UNMODIFIED (reused, not reimplemented -- the exact function
TRACK-P100-COARSE-TO-FINE-RETRIEVAL01's own Phase B already used for
an analogous purpose). New features added here: query past mean/std/
recent-slope/first-diff-variance/lag-1-ACF (from `batch_x` only, never
`batch_y`), Base-forecast mean/std/slope, Base-vs-Retrieval
disagreement (`norm(R-B)`), and candidate-side dispersion (past
-window dispersion + embedding dispersion among the actually-retrieved
Top-10, plus retrieval score variance over the full P100 pool).

Target (spec section 14): `t_q = 1[MSE(R_q,Y_q) < MSE(B_q,Y_q)]`,
computed by recomputing R identically to the EXISTING cache (verified
to match `diag_p100_oracle_lambda01.py`'s own cached numbers) rather
than re-reading the cache file, so the SAME forward pass that yields
the features also yields the matching target and the Top-10 set
needed for candidate-dispersion features -- TRAIN-only label; val/
test are evaluation-only (spec section 14/17's leakage rule).
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
from models.RelationStage2 import BaseForecastHead
from scripts.diag_p100_coarse_v5_fusion01 import TAU_C
from scripts.rng_control01 import set_global_seeds
from scripts.train_factorial_e2e01 import encode_raw
from scripts.train_j_shared_encoder_drift01 import build_model
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_margutil01 import memory_value
from scripts.train_p100_adaptive_fusion01 import FEATURE_NAMES as V5_FEATURE_NAMES
from scripts.train_p100_adaptive_fusion01 import extract_features as extract_v5_features
from scripts.train_retriever_pool01 import CandidatePoolCache
from utils.candidate_pool import compute_coarse_delta_last_cosine_scores

TOP_K = 10
N_SLOTS = 5
EPS = 1e-8

NEW_FEATURE_NAMES = [
    'q_past_mean', 'q_past_std', 'q_recent_slope', 'q_first_diff_var', 'q_acf_lag1',
    'base_mean', 'base_std', 'base_slope', 'base_vs_retrieval_disagreement_norm',
    'retrieved_top10_past_dispersion', 'retrieved_top10_embed_dispersion', 'p_mix_score_variance',
]
ALL_FEATURE_NAMES = V5_FEATURE_NAMES + NEW_FEATURE_NAMES
N_FEATURES = len(ALL_FEATURE_NAMES)


def _acf_lag1(x):
    xm = x - x.mean(dim=-1, keepdim=True)
    num = (xm[:, :-1] * xm[:, 1:]).sum(dim=-1)
    den = (xm ** 2).sum(dim=-1).clamp_min(EPS)
    return num / den


def _slope(x):
    L = x.size(-1)
    t = torch.arange(L, device=x.device, dtype=x.dtype) - (L - 1) / 2.0
    t_var = (t * t).sum().clamp_min(EPS)
    xm = x - x.mean(dim=-1, keepdim=True)
    return (xm * t).sum(dim=-1) / t_var


def extract_new_features(batch_x, c, base_pred_c, r_c, b_c, top10_future_c, top10_embed, p_mix_full):
    """batch_x: [B,L,C_total]. base_pred_c/r_c/b_c: [B,H] (absolute future
    space, same convention as the Stage2 cache). top10_future_c: [B,10,H]
    (the actually-retrieved 10 candidates' own future values, for
    dispersion -- NOT the query future, future-blind at deploy since it
    is the CANDIDATES' stored past-matched future, known at retrieval
    time exactly like R itself is). top10_embed: [B,10,D]. p_mix_full:
    [B,100]."""
    q = batch_x[:, :, c]
    feats = {
        'q_past_mean': q.mean(dim=-1), 'q_past_std': q.std(dim=-1),
        'q_recent_slope': _slope(q), 'q_first_diff_var': (q[:, 1:] - q[:, :-1]).var(dim=-1),
        'q_acf_lag1': _acf_lag1(q),
        'base_mean': base_pred_c.mean(dim=-1), 'base_std': base_pred_c.std(dim=-1),
        'base_slope': _slope(base_pred_c),
        'base_vs_retrieval_disagreement_norm': (r_c - b_c).norm(dim=-1),
        'retrieved_top10_past_dispersion': top10_future_c.std(dim=1).mean(dim=-1),
        'retrieved_top10_embed_dispersion': top10_embed.std(dim=1).mean(dim=-1),
        'p_mix_score_variance': p_mix_full.var(dim=-1),
    }
    return torch.stack([feats[k] for k in NEW_FEATURE_NAMES], dim=-1)  # [B, len(NEW_FEATURE_NAMES)]


@torch.no_grad()
def collect_cell(cli, device):
    set_global_seeds(cli.seed)
    exp, args, model = build_model(cli, device)
    exp.memory_x = exp.memory_x.to(device)
    model.eval()
    slot_heads = SlotHeads(int(args.d_model), n_slots=N_SLOTS).to(device)
    bl = torch.load(cli.v5_checkpoint, map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    slot_heads.load_state_dict(bl['slot_heads_state_dict'])
    slot_heads.eval()

    channels = int(args.enc_in)
    base = BaseForecastHead(seq_len=cli.seq_len, pred_len=cli.pred_len, channels=channels,
                            mode='per_channel_linear').to(device)
    bbl = torch.load(cli.base_checkpoint, map_location=device)
    base.load_state_dict(bbl['model_state_dict'])
    base.eval()

    expected_meta = {'seq_len': cli.seq_len, 'pred_len': cli.pred_len, 'candidate_mask': cli.candidate_mask,
                     'candidate_pool_size': cli.candidate_pool_size, 'candidate_pool_metric': 'delta_last_cosine'}
    pool_caches = {s: CandidatePoolCache(Path(cli.candidate_pool_cache) / f'{s}.pt', expected_meta)
                  for s in ('train', 'val', 'test')}

    out = {}
    for split in ('train', 'val', 'test'):
        _, loader = exp._get_data(flag=split, shuffle=False)
        all_feats, all_target, all_gain = [], [], []
        for c in range(channels):
            k_full = F.normalize(encode_raw(model, exp.memory_x, c), dim=-1)
            for batch_x, batch_y, batch_start_idx in loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                pool_idx_global = pool_caches[split].lookup(batch_start_idx, c, device)

                coarse_full = compute_coarse_delta_last_cosine_scores(batch_x, exp.memory_x, c)
                c_i = coarse_full.gather(1, pool_idx_global)
                p_c = F.softmax(c_i / TAU_C, dim=-1)

                z_q = encode_raw(model, batch_x, c)
                q_proj = slot_heads(z_q)
                k_pool = k_full[pool_idx_global]
                scores = torch.einsum('bsd,bmd->bsm', q_proj, k_pool)
                p_heads = F.softmax(scores / cli.tau_s, dim=-1)
                p_v5 = p_heads.mean(dim=1)

                v5_feats = extract_v5_features(p_c, p_v5, p_heads, z_q, c, channels)

                memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
                query_future = batch_y[:, :, c]
                pooled_future = memory_c[pool_idx_global] + offset_c.view(-1, 1, 1)

                picks = stable_topk_indices(p_v5, TOP_K, largest=True)
                top10_future_c = pooled_future.gather(1, picks.unsqueeze(-1).expand(-1, -1, pooled_future.size(-1)))
                r_c = top10_future_c.mean(dim=1)
                top10_embed = k_pool.gather(1, picks.unsqueeze(-1).expand(-1, -1, k_pool.size(-1)))

                offset = batch_x[:, -1:, :].detach()
                b_c = (base(batch_x) + offset)[:, :, c]

                new_feats = extract_new_features(batch_x, c, b_c, r_c, b_c, top10_future_c, top10_embed, p_v5)
                feats = torch.cat([v5_feats, new_feats], dim=-1)

                mse_b = ((b_c - query_future) ** 2).mean(dim=-1)
                mse_r = ((r_c - query_future) ** 2).mean(dim=-1)
                target = (mse_r < mse_b).float()
                gain = mse_b - mse_r

                all_feats.append(feats.cpu()); all_target.append(target.cpu()); all_gain.append(gain.cpu())

        out[split] = {'feats': torch.cat(all_feats), 'target': torch.cat(all_target), 'gain': torch.cat(all_gain)}
        print(f'[observability] {cli.cell} {split}: n={out[split]["feats"].size(0)} '
             f'positive_rate={float(out[split]["target"].mean()):.4f}')
    return out


def spearman_1d(a, b):
    ra = a.argsort().argsort().float()
    rb = b.argsort().argsort().float()
    ra = ra - ra.mean(); rb = rb - rb.mean()
    denom = (ra.norm() * rb.norm()).clamp_min(EPS)
    return float((ra * rb).sum() / denom)


def feature_observability(train_data):
    """Spec section 15: Spearman(feature, retrieval gain) on TRAIN only."""
    feats, gain = train_data['feats'], train_data['gain']
    return {name: spearman_1d(feats[:, i], gain) for i, name in enumerate(ALL_FEATURE_NAMES)}


class B2MLP(nn.Module):
    def __init__(self, n_features, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.feature_mean = nn.Parameter(torch.zeros(n_features), requires_grad=False)
        self.feature_std = nn.Parameter(torch.ones(n_features), requires_grad=False)

    def forward(self, feats):
        x = (feats - self.feature_mean) / self.feature_std
        return self.net(x).squeeze(-1)


class B1Logistic(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
        self.feature_mean = nn.Parameter(torch.zeros(n_features), requires_grad=False)
        self.feature_std = nn.Parameter(torch.ones(n_features), requires_grad=False)

    def forward(self, feats):
        x = (feats - self.feature_mean) / self.feature_std
        return self.linear(x).squeeze(-1)


def _ece(probs, targets, n_bins=10):
    bins = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = probs.size(0)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        if mask.sum() == 0:
            continue
        acc = targets[mask].mean()
        conf = probs[mask].mean()
        ece += (mask.float().mean()) * float((acc - conf).abs())
    return float(ece)


def _auroc(probs, targets):
    pos = probs[targets == 1]
    neg = probs[targets == 0]
    if pos.numel() == 0 or neg.numel() == 0:
        return float('nan')
    diff = pos.unsqueeze(1) - neg.unsqueeze(0)
    return float((diff > 0).float().mean() + 0.5 * (diff == 0).float().mean())


def _aupr(probs, targets):
    order = probs.argsort(descending=True)
    t_sorted = targets[order]
    tp = torch.cumsum(t_sorted, dim=0)
    fp = torch.cumsum(1 - t_sorted, dim=0)
    precision = tp / (tp + fp).clamp_min(EPS)
    recall = tp / t_sorted.sum().clamp_min(EPS)
    recall_prev = torch.cat([torch.zeros(1), recall[:-1]])
    return float((precision * (recall - recall_prev)).sum())


def _balanced_accuracy(probs, targets, thresh=0.5):
    pred = (probs >= thresh).float()
    pos_mask, neg_mask = targets == 1, targets == 0
    tpr = pred[pos_mask].mean() if pos_mask.any() else torch.tensor(float('nan'))
    tnr = (1 - pred[neg_mask]).mean() if neg_mask.any() else torch.tensor(float('nan'))
    return float((tpr + tnr) / 2)


def evaluate_probe(probs, targets, gains):
    return {
        'auroc': _auroc(probs, targets), 'aupr': _aupr(probs, targets),
        'balanced_accuracy': _balanced_accuracy(probs, targets),
        'brier': float(((probs - targets) ** 2).mean()),
        'ece': _ece(probs, targets),
        'spearman_prob_vs_gain': spearman_1d(probs, gains),
    }


def train_probe(model_ctor, train_data, val_data, device, epochs=50, lr=1e-2, batch_size=512, seed=0):
    torch.manual_seed(seed)
    model = model_ctor().to(device)
    if hasattr(model, 'feature_mean'):
        mean = train_data['feats'].mean(dim=0).to(device)
        std = train_data['feats'].std(dim=0).clamp_min(1e-6).to(device)
        model.feature_mean.data.copy_(mean)
        model.feature_std.data.copy_(std)
    feats = train_data['feats'].to(device)
    targets = train_data['target'].to(device)
    n = feats.size(0)
    feats_val = val_data['feats'].to(device)
    targets_val = val_data['target'].to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss, best_state = float('inf'), None
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            logits = model(feats[idx])
            loss = F.binary_cross_entropy_with_logits(logits, targets[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(feats_val)
            val_loss = float(F.binary_cross_entropy_with_logits(val_logits, targets_val))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


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
    ap.add_argument('--base_checkpoint', required=True)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out_dir', required=True)
    cli = ap.parse_args()
    cli.init_seed = cli.seed
    cli.patch_len = 16
    cli.top_k = TOP_K

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = collect_cell(cli, device)

    obs = feature_observability(data['train'])
    print(f'[observability] {cli.cell} feature-gain Spearman (train): '
         f'{ {k: round(v,4) for k,v in sorted(obs.items(), key=lambda kv: -abs(kv[1]))[:8]} }')

    results = {}

    # B0: constant predictor (train positive rate)
    const_p = float(data['train']['target'].mean())
    probs_test_b0 = torch.full_like(data['test']['target'], const_p)
    results['B0_constant'] = evaluate_probe(probs_test_b0, data['test']['target'], data['test']['gain'])
    results['B0_constant']['constant_value'] = const_p

    # B1: logistic regression
    model_b1 = train_probe(lambda: B1Logistic(N_FEATURES), data['train'], data['val'], device)
    with torch.no_grad():
        probs_test_b1 = torch.sigmoid(model_b1(data['test']['feats'].to(device))).cpu()
    results['B1_logistic'] = evaluate_probe(probs_test_b1, data['test']['target'], data['test']['gain'])

    # B2: tiny MLP
    model_b2 = train_probe(lambda: B2MLP(N_FEATURES), data['train'], data['val'], device)
    with torch.no_grad():
        probs_test_b2 = torch.sigmoid(model_b2(data['test']['feats'].to(device))).cpu()
    results['B2_mlp'] = evaluate_probe(probs_test_b2, data['test']['target'], data['test']['gain'])

    # Deployable binary gate (spec section 18): Yhat = R if p(useful|x)>=t else B.
    # Threshold selected on VAL ONLY by maximizing the gate's own implied mean
    # gain over "always Base" (routing to R adds `gain_q` for every row routed,
    # adds 0 for every row kept at B) -- never touches test until evaluation.
    with torch.no_grad():
        probs_val_b2 = torch.sigmoid(model_b2(data['val']['feats'].to(device))).cpu()
    best_t, best_val_gain = 0.5, float('-inf')
    for t in torch.linspace(0.05, 0.95, 19):
        pred_r = (probs_val_b2 >= t).float()
        implied_gain = float((pred_r * data['val']['gain']).mean())
        if implied_gain > best_val_gain:
            best_val_gain, best_t = implied_gain, float(t)
    gate_pred_test = (probs_test_b2 >= best_t).float()
    gate_implied_gain_test = float((gate_pred_test * data['test']['gain']).mean())
    # for comparison: the CURRENT global-lambda arm's own implied gain over
    # always-Base is, by definition, current_lambda * mean(gain) (since the
    # global-lambda fusion routes EVERY row by the same fixed weight, not 0/1)
    results['deployable_binary_gate'] = {
        'threshold_selected_on_val': best_t,
        'mean_gain_vs_always_base_test': gate_implied_gain_test,
        'fraction_routed_to_retrieval_test': float(gate_pred_test.mean()),
        'val_selected_implied_gain': best_val_gain,
    }

    out = {'cell': cli.cell, 'feature_names': ALL_FEATURE_NAMES, 'feature_gain_spearman_train': obs,
          'train_positive_rate': const_p, 'val_positive_rate': float(data['val']['target'].mean()),
          'test_positive_rate': float(data['test']['target'].mean()), 'probes': results}
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f'{cli.cell}_observability.json').write_text(json.dumps(out, indent=2))
    print(f'[observability] {cli.cell} DONE: {json.dumps(results, indent=2)}')


if __name__ == '__main__':
    main()
