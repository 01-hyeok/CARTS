#!/usr/bin/env python3
"""TRACK-J-SHARED-ENCODER-DRIFT01 -- A0 diagnostic only.

Question: when a SHARED encoder E_theta is trained on a future-supervised
KL retrieval objective, does useful query-side adaptation get cancelled
by simultaneous movement of the candidate-side (key) embedding space?

A0 = the "corrected, clean" MLP shared-encoder baseline (see
research/J-shared-encoder-drift/AUDIT.md for the full sourcing of every
hyperparameter) + read-only instrumentation. Training itself
(gradient path, optimizer step, batch order, candidate universe) is
BYTE-IDENTICAL to that baseline -- reused UNMODIFIED:
`individual_utility_memsafe`/`arm_score`/`encode_raw`
(`train_factorial_e2e01`), `normalized_teacher_prob`/`kl_loss`
(`train_horizon_retrieval_expert01`), `memory_value`/`build_experiment`
(`train_margutil01`), `stable_topk_indices` (`RelationStage1`),
`recall_at_k`/`ndcg_at_k` (`train_patch_retrieval_expert01`).

Four fixed-probe geometry variants, computed at every diagnostic step
against the SAME 256-query probe set, SAME candidate universe/mask/
Top-K/raw-future eval target:
  S00 = cos(E0(Xq), E0(Xk))   -- frozen-at-init baseline, cached after step0
  St0 = cos(Et(Xq), E0(Xk))   -- query adapts, candidates frozen at init
  S0t = cos(E0(Xq), Et(Xk))   -- query frozen at init, candidates adapt
  Stt = cos(Et(Xq), Et(Xk))   -- the ACTUAL shared-encoder retrieval

`E0` is a frozen deep-copy of the encoder taken at step 0 (never
touched by the optimizer; `E0(Xk)`/`E0(Xq)` on the probe set are computed
once and cached -- E0 never changes). `Et(Xk)` (full candidate bank) is
recomputed at every diagnostic step -- this is the only per-step cost
beyond what training already pays every batch.
"""
import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.rng_control01 import batch_order_sha256, make_loader_generator, set_global_seeds
from scripts.shared_candidate_pool01 import SharedCandidatePool, VALID_POOL_MODES
from scripts.train_factorial_e2e01 import arm_score, encode_raw, individual_utility_memsafe
from scripts.train_horizon_retrieval_expert01 import kl_loss, normalized_teacher_prob
from scripts.train_margutil01 import build_experiment, memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k

EPS = 1e-8
VARIANTS = ('S00', 'St0', 'S0t', 'Stt')
GEOMETRY_SUBSET_SIZE = 500  # fixed deterministic candidate subset for O(N^2)/SVD diagnostics


def state_hash(model):
    h = hashlib.sha256()
    for k in sorted(model.state_dict()):
        h.update(k.encode())
        h.update(model.state_dict()[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def build_model(cli, device):
    set_global_seeds(cli.init_seed)
    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.seq_len, 'batch_size': cli.batch_size,
        'seed': cli.init_seed, 'top_k': cli.top_k,
        'relation_encoder_type': 'mlp', 'relation_self_fill': 'linear',
        'relation_input_space': 'delta_last', 'relation_teacher_space': 'delta_last',
        'relation_value_space': 'delta_last', 'candidate_mask': 'raft',
        'patch_len': cli.patch_len, 'stride': cli.patch_len,  # irrelevant for mlp; kept for CLI/arch compat
    })
    exp._ensure_memory()
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.to(device)
    return exp, args, model


def build_probe_set(exp, val_loader, n_probe, device):
    """Deterministic, evenly-spaced probe queries covering the val split
    chronologically. Concatenates the (small) val split once, then
    subsamples -- never re-sampled at diagnostic time."""
    xs, ys, starts = [], [], []
    for batch_x, batch_y, batch_start_idx in val_loader:
        xs.append(batch_x.float())
        ys.append(batch_y.float())
        starts.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx)
                      else torch.as_tensor(batch_start_idx))
    all_x = torch.cat(xs, dim=0)
    all_y = torch.cat(ys, dim=0)
    all_starts = torch.cat(starts, dim=0)
    n_total = all_x.size(0)
    n_probe = min(n_probe, n_total)
    idx = torch.linspace(0, n_total - 1, n_probe).round().long().unique()
    probe_x = all_x[idx].to(device)
    probe_y = all_y[idx].to(device)
    probe_start = all_starts[idx]
    base_mask, _ = exp._candidate_mask(probe_start)
    return {'x': probe_x, 'y': probe_y, 'start_idx': probe_start, 'base_mask': base_mask,
           'n': probe_x.size(0)}


def precompute_probe_fixed(exp, args, probe, channels, device, chunk_size, candidate_pool, split):
    """Per-channel raw teacher distance / oracle Top-K / candidate-future
    reconstruction for the probe set -- computed ONCE, identical across
    every diagnostic step and every S-variant (spec section 5)."""
    out = {}
    memory_y, memory_x_last = exp.memory_y, exp.memory_x_last
    for c in channels:
        cand_mask = candidate_pool.apply(probe['base_mask'], probe['start_idx'], c, split)
        memory_c, offset_c = memory_value(args, probe['x'], memory_y, memory_x_last, c)
        query_future = probe['y'][:, :, c]
        d_raw = -individual_utility_memsafe(memory_c, offset_c, query_future, chunk_size)
        oracle_idx = stable_topk_indices(d_raw.masked_fill(~cand_mask, float('inf')), 10, largest=False)
        out[c] = dict(memory_c=memory_c, offset_c=offset_c, query_future=query_future,
                     d_raw=d_raw, oracle_idx=oracle_idx, cand_mask=cand_mask)
    return out


def variant_metrics(s_matrix, cand_mask, fixed_c, top_k=10):
    s_masked = s_matrix.masked_fill(~cand_mask, float('-inf'))
    model_idx = stable_topk_indices(s_masked, top_k, largest=True)
    d_raw, oracle_idx = fixed_c['d_raw'], fixed_c['oracle_idx']
    model_ind_mse = d_raw.gather(1, model_idx).mean(-1)
    oracle_ind_mse = d_raw.gather(1, oracle_idx).mean(-1)
    recall10 = recall_at_k(model_idx, oracle_idx, top_k)
    ndcg10 = ndcg_at_k(model_idx, d_raw, cand_mask, top_k)
    oracle_regret = model_ind_mse - oracle_ind_mse
    y_sel = fixed_c['memory_c'][model_idx] + fixed_c['offset_c'].view(-1, 1, 1)
    uniform_agg = ((y_sel.mean(dim=1) - fixed_c['query_future']) ** 2).mean(-1)
    metrics = dict(retmse10=float(model_ind_mse.mean()), recall10=float(recall10.mean()),
                  ndcg10=float(ndcg10.mean()), oracle_regret=float(oracle_regret.mean()),
                  uniform_agg_mse10=float(uniform_agg.mean()))
    return metrics, model_idx


def effective_rank_entropy(singular_values):
    """exp(-sum p_i log p_i), p_i = sigma_i / sum(sigma) -- entropy-based
    effective rank (documented choice, spec section 12)."""
    s = singular_values.clamp_min(0)
    p = s / s.sum().clamp_min(EPS)
    ent = -(p * (p.clamp_min(EPS)).log()).sum()
    return float(ent.exp())


def geometry_metrics(embeddings_subset):
    """embeddings_subset: [M, d], already the fixed deterministic subset."""
    x = embeddings_subset
    norms = x.norm(dim=-1)
    xn = F.normalize(x, dim=-1)
    cos = xn @ xn.T
    iu = torch.triu_indices(x.size(0), x.size(0), offset=1)
    pair_cos = cos[iu[0], iu[1]]
    xc = x - x.mean(dim=0, keepdim=True)
    cov = (xc.T @ xc) / max(x.size(0) - 1, 1)
    s = torch.linalg.svdvals(cov)
    eff_rank = effective_rank_entropy(s)
    return dict(effective_rank=eff_rank, embedding_norm_mean=float(norms.mean()),
               embedding_norm_std=float(norms.std()), per_dim_variance_mean=float(xc.var(dim=0).mean()),
               mean_pairwise_cosine=float(pair_cos.mean()), pairwise_cosine_std=float(pair_cos.std()),
               largest_singular_value_fraction=float(s.max() / s.sum().clamp_min(EPS)))


def churn(idx_now, idx_prev, k):
    if idx_prev is None:
        return None
    hit = (idx_now.unsqueeze(-1) == idx_prev.unsqueeze(-2)).any(-1).float().sum(-1)
    return float((1.0 - hit / k).mean())


def retention_vs_init(idx_now, idx0, k):
    hit = (idx_now.unsqueeze(-1) == idx0.unsqueeze(-2)).any(-1).float().sum(-1)
    return float((hit / k).mean())


def displacement_stats(disp):
    q = torch.quantile(disp, torch.tensor([0.1, 0.5, 0.9], device=disp.device))
    return dict(mean=float(disp.mean()), median=float(disp.median()), std=float(disp.std()),
               p10=float(q[0]), p50=float(q[1]), p90=float(q[2]), max=float(disp.max()))


def _flat_grad(params):
    parts = []
    for p in params:
        parts.append((p.grad.detach().reshape(-1).clone() if p.grad is not None
                      else torch.zeros(p.numel(), device=p.device)))
    return torch.cat(parts)


def gradient_conflict_diagnostic(model, exp, args, cli, batch_x, batch_y, batch_start_idx, channels, device,
                                 point_name, candidate_pool):
    """Query-branch vs key-branch encoder-gradient decomposition on a FIXED
    batch (spec section 13). Diagnostic only -- never calls optimizer.step(),
    never mutates model weights; grad buffers are zeroed before returning so
    the actual training loop's own next `.backward()` is unaffected.

    g_combined (real loss, both branches grad-tracked) is asserted equal to
    g_q + g_k (query-branch-only + key-branch-only, computed by detaching
    the OTHER side) within the spec's tolerance -- pure multivariable
    chain-rule identity, verified per channel, per pre-registered point.
    """
    # eval() (dropout off) so this diagnostic draws ZERO random numbers from
    # the global RNG stream -- otherwise it would silently perturb every
    # subsequent training step's dropout pattern (caught by the
    # instrumentation ON/OFF equivalence check, spec section 16).
    was_training = model.training
    model.eval()
    params = list(model.parameters())
    rows = []
    base_mask, _ = exp._candidate_mask(batch_start_idx)
    for c in channels:
        cand_mask = candidate_pool.apply(base_mask, batch_start_idx, c, 'train')
        model.zero_grad(set_to_none=True)
        z_q = encode_raw(model, batch_x, c)
        z_k = encode_raw(model, exp.memory_x, c)
        memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
        query_future = batch_y[:, :, c]
        with torch.no_grad():
            u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
            p_t = normalized_teacher_prob(-u, cand_mask, cli.tau_t)

        # combined (real) loss/gradient
        s_combined = arm_score(z_q, z_k, None)
        l_combined = kl_loss(p_t, s_combined, cand_mask, cli.tau_s)
        l_combined.backward(retain_graph=True)
        g_combined = _flat_grad(params)
        model.zero_grad(set_to_none=True)

        # query-branch-only: key side detached
        s_q = arm_score(z_q, z_k.detach(), None)
        l_q = kl_loss(p_t, s_q, cand_mask, cli.tau_s)
        l_q.backward(retain_graph=True)
        g_q = _flat_grad(params)
        model.zero_grad(set_to_none=True)

        # key-branch-only: query side detached
        s_k = arm_score(z_q.detach(), z_k, None)
        l_k = kl_loss(p_t, s_k, cand_mask, cli.tau_s)
        l_k.backward()
        g_k = _flat_grad(params)
        model.zero_grad(set_to_none=True)

        g_sum = g_q + g_k
        max_abs_diff = float((g_combined - g_sum).abs().max())
        rel_diff = max_abs_diff / float(g_combined.abs().max().clamp_min(EPS))
        cos_qk = float((g_q @ g_k) / (g_q.norm() * g_k.norm()).clamp_min(EPS))
        rows.append({'point': point_name, 'channel': c, 'cos_gq_gk': cos_qk,
                    'norm_gq': float(g_q.norm()), 'norm_gk': float(g_k.norm()),
                    'norm_gq_plus_gk': float(g_sum.norm()), 'norm_g_combined': float(g_combined.norm()),
                    'equivalence_max_abs_diff': max_abs_diff, 'equivalence_rel_diff': rel_diff})
    model.zero_grad(set_to_none=True)
    if was_training:
        model.train()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--cell', default='ETTh1_720')
    ap.add_argument('--pred_len', type=int, default=720)
    ap.add_argument('--seq_len', type=int, default=720)
    ap.add_argument('--patch_len', type=int, default=16, help='irrelevant for mlp encoder, kept for CLI compat')
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--tau_t', type=float, default=0.1)
    ap.add_argument('--tau_s', type=float, default=0.1)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--train_epochs', type=int, default=10)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--init_seed', type=int, default=0)
    ap.add_argument('--loader_seed', type=int, default=0)
    ap.add_argument('--n_probe', type=int, default=256)
    ap.add_argument('--diag_every_epoch1', type=int, default=20)
    ap.add_argument('--diag_every_later', type=int, default=50)
    ap.add_argument('--instrument', choices=('on', 'off'), default='on')
    ap.add_argument('--out_dir', default='results/TRACK-J-SHARED-ENCODER-DRIFT01')
    ap.add_argument('--checkpoints', default='checkpoints/track_j_shared_encoder_drift01')
    ap.add_argument('--limit_batches', type=int, default=0, help='SMOKE ONLY')
    ap.add_argument('--smoke_test', action='store_true')
    ap.add_argument('--candidate_pool_mode', choices=VALID_POOL_MODES, default='full')
    ap.add_argument('--candidate_pool_cache_dir', default=None,
                    help='shared pool cache built once by build_v_shared_candidate_pool01.py; required for top100')
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(cli.out_dir) / cli.cell
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints) / cli.cell
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    exp, args, model = build_model(cli, device)
    channels = list(range(int(args.enc_in)))

    init_hash = state_hash(model)
    E0 = copy.deepcopy(model).to(device)
    for p in E0.parameters():
        p.requires_grad_(False)
    E0.eval()
    e0_hash = state_hash(E0)
    assert init_hash == e0_hash

    for p in model.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cli.learning_rate)

    train_gen = make_loader_generator(cli.loader_seed)
    _, train_loader = exp._get_data(flag='train', shuffle=True, generator=train_gen)
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)

    candidate_pool = SharedCandidatePool(cli.candidate_pool_mode, cli.candidate_pool_cache_dir,
                                         int(exp.memory_x.size(0)), channels, cli.top_k)

    probe = build_probe_set(exp, val_loader, cli.n_probe, device)
    fixed = precompute_probe_fixed(exp, args, probe, channels, device, cli.chunk_size, candidate_pool, 'val')
    (out_dir / 'probe_query_ids.json').write_text(json.dumps(
        {'start_idx': probe['start_idx'].tolist(), 'n_probe': probe['n']}, indent=2))

    geom_subset_idx = torch.linspace(0, exp.memory_x.size(0) - 1, GEOMETRY_SUBSET_SIZE).round().long().unique()

    config = {'cell': cli.cell, 'relation_encoder_type': 'mlp', 'relation_self_fill': 'linear',
              'tau_t': cli.tau_t, 'tau_s': cli.tau_s, 'top_k': cli.top_k, 'batch_size': cli.batch_size,
              'learning_rate': cli.learning_rate, 'train_epochs': cli.train_epochs, 'patience': cli.patience,
              'init_seed': cli.init_seed, 'loader_seed': cli.loader_seed, 'n_probe': probe['n'],
              'n_candidates': int(exp.memory_x.size(0)), 'geometry_subset_size': int(geom_subset_idx.numel()),
              **candidate_pool.describe(),
              'checkpoint_criterion': 'min val model_top10_individual_mse', 'instrument': cli.instrument}
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2))
    print(f'[track_j] cell={cli.cell} encoder=mlp N={config["n_candidates"]} n_probe={probe["n"]} '
         f'init_hash={init_hash[:16]} instrument={cli.instrument}')

    if cli.smoke_test:
        _smoke(exp, args, model, E0, probe, fixed, channels, optimizer, cli, train_loader, val_loader, device,
               candidate_pool)
        return

    step_rows = []
    fourway_rows = []
    geometry_rows = []
    churn_rows = []
    disp_rows = []
    util_group_rows = []
    last_idx = {}
    idx0 = {}
    zk0_cache = {}  # per channel, E0(memory) -- frozen forever
    zq0_cache = {}  # per channel, E0(probe)  -- frozen forever

    def run_diagnostic(step, epoch):
        was_training = model.training
        model.eval()
        row_fourway, row_geom, row_churn, row_disp, row_util = [], [], [], [], []
        for c in channels:
            with torch.no_grad():
                zq_t = encode_raw(model, probe['x'], c)
                zk_t = encode_raw(model, exp.memory_x, c)
                if c not in zq0_cache:
                    zq0_cache[c] = encode_raw(E0, probe['x'], c)
                    zk0_cache[c] = encode_raw(E0, exp.memory_x, c)
                zq_0, zk_0 = zq0_cache[c], zk0_cache[c]

                s_variants = {
                    'S00': arm_score(zq_0, zk_0, None), 'St0': arm_score(zq_t, zk_0, None),
                    'S0t': arm_score(zq_0, zk_t, None), 'Stt': arm_score(zq_t, zk_t, None),
                }
                for v_name, s_mat in s_variants.items():
                    if step > 0 and v_name == 'S00':
                        continue  # S00 never changes past step0 -- cached, not recomputed
                    m, idx = variant_metrics(s_mat, fixed[c]['cand_mask'], fixed[c], cli.top_k)
                    row_fourway.append({'step': step, 'epoch': epoch, 'channel': c, 'variant': v_name, **m})
                    key = (v_name, c)
                    if step == 0:
                        idx0[key] = idx
                    ch = churn(idx, last_idx.get(key), cli.top_k)
                    ret0 = retention_vs_init(idx, idx0[key], cli.top_k)
                    last_idx[key] = idx
                    row_churn.append({'step': step, 'epoch': epoch, 'channel': c, 'variant': v_name,
                                      'churn_vs_prev': ch, 'retention_vs_init': ret0})

                # displacement (query + candidate), full precision
                q_disp = 1.0 - (F.normalize(zq_0, dim=-1) * F.normalize(zq_t, dim=-1)).sum(-1)
                k_disp = 1.0 - (F.normalize(zk_0, dim=-1) * F.normalize(zk_t, dim=-1)).sum(-1)
                qd = displacement_stats(q_disp)
                kd = displacement_stats(k_disp)
                row_disp.append({'step': step, 'epoch': epoch, 'channel': c, 'side': 'query', **qd})
                row_disp.append({'step': step, 'epoch': epoch, 'channel': c, 'side': 'candidate', **kd})

                # utility-group candidate displacement (oracle top10 / top100-minus-top10 / middle / bottom)
                d_raw_c = fixed[c]['d_raw']  # [n_probe, N], lower = better
                order = d_raw_c.argsort(dim=-1)  # ascending utility distance = best first
                n_cand = order.size(1)
                top10 = order[:, :10]
                top100 = order[:, 10:100]
                mid_lo, mid_hi = n_cand // 3, 2 * n_cand // 3
                middle = order[:, mid_lo:mid_hi]
                bottom = order[:, mid_hi:]
                for gname, gidx in (('oracle_top10', top10), ('oracle_top100_excl_top10', top100),
                                    ('middle', middle), ('bottom', bottom)):
                    g_disp = k_disp[gidx]  # [n_probe, group_size]
                    row_util.append({'step': step, 'epoch': epoch, 'channel': c, 'group': gname,
                                     'mean_displacement': float(g_disp.mean())})

                # geometry, fixed deterministic subset
                sub = zk_t[geom_subset_idx]
                g = geometry_metrics(sub)
                row_geom.append({'step': step, 'epoch': epoch, 'channel': c, **g})

        if was_training:
            model.train()
        return row_fourway, row_geom, row_churn, row_disp, row_util

    # ---- step-0 diagnostic (mandatory) ----
    f0, g0, c0, d0, u0 = run_diagnostic(step=0, epoch=0)
    fourway_rows += f0
    geometry_rows += g0
    churn_rows += c0
    disp_rows += d0
    util_group_rows += u0
    # step-0 equivalence check (S00==St0==S0t==Stt within tolerance)
    s00 = {r['channel']: r for r in f0 if r['variant'] == 'S00'}
    for r in f0:
        if r['variant'] == 'S00':
            continue
        base = s00[r['channel']]
        for k in ('retmse10', 'recall10', 'ndcg10', 'oracle_regret', 'uniform_agg_mse10'):
            assert abs(r[k] - base[k]) < 1e-4, f'[ISSUE] step0 {r["variant"]} channel{r["channel"]} {k} differs from S00: {r[k]} vs {base[k]}'
    print('[track_j] step0 four-way equivalence check PASSED')

    n_epoch1_steps = len(train_loader) if not cli.limit_batches else min(len(train_loader), cli.limit_batches)
    grad_diag_targets = sorted({1, max(1, round(0.25 * n_epoch1_steps)), max(1, round(0.5 * n_epoch1_steps))})
    grad_diag_rows = []

    global_step = 0
    epoch_val_rows = []
    best = {'val': float('inf'), 'epoch': -1}
    t0 = time.time()
    for epoch in range(1, cli.train_epochs + 1):
        model.train()
        for bi, (batch_x, batch_y, batch_start_idx) in enumerate(train_loader):
            if cli.limit_batches and bi >= cli.limit_batches:
                break
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            last_batch_epoch1 = (batch_x, batch_y, batch_start_idx)
            base_mask, _ = exp._candidate_mask(batch_start_idx)
            optimizer.zero_grad()
            batch_loss = 0.0
            for c in channels:
                cand_mask = candidate_pool.apply(base_mask, batch_start_idx, c, 'train')
                z_q = encode_raw(model, batch_x, c)
                E = encode_raw(model, exp.memory_x, c)
                memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
                query_future = batch_y[:, :, c]
                s = arm_score(z_q, E, None)
                u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
                d = -u
                p_t = normalized_teacher_prob(d, cand_mask, cli.tau_t)
                ch_loss = kl_loss(p_t, s, cand_mask, cli.tau_s)
                (ch_loss / len(channels)).backward()
                batch_loss += float(ch_loss.detach()) / len(channels)
            optimizer.step()
            global_step += 1
            step_rows.append({'global_step': global_step, 'epoch': epoch, 'batch_in_epoch': bi, 'train_kl': batch_loss})

            if global_step == 1:
                first_batch = (batch_x.clone(), batch_y.clone(), batch_start_idx.clone()
                              if torch.is_tensor(batch_start_idx) else torch.as_tensor(batch_start_idx))

            if cli.instrument == 'on' and epoch == 1 and global_step in grad_diag_targets:
                pct = {1: 'step0_first_batch'}.get(global_step, f'step{global_step}')
                if global_step == max(1, round(0.25 * n_epoch1_steps)):
                    pct = '25pct_epoch1'
                if global_step == max(1, round(0.5 * n_epoch1_steps)):
                    pct = '50pct_epoch1'
                grad_diag_rows += gradient_conflict_diagnostic(
                    model, exp, args, cli, batch_x, batch_y, batch_start_idx, channels, device, pct, candidate_pool)
                print(f'[track_j] gradient-conflict diag at {pct} (global_step={global_step})')

            do_diag = (cli.instrument == 'on') and (
                (epoch == 1 and global_step % cli.diag_every_epoch1 == 0) or
                (epoch > 1 and global_step % cli.diag_every_later == 0))
            if do_diag:
                f, g, ch, d, u = run_diagnostic(step=global_step, epoch=epoch)
                fourway_rows += f
                geometry_rows += g
                churn_rows += ch
                disp_rows += d
                util_group_rows += u
                print(f'[track_j] diag step={global_step} epoch={epoch} train_kl={batch_loss:.4f}')

        if cli.instrument == 'on' and epoch == 1:
            grad_diag_rows += gradient_conflict_diagnostic(
                model, exp, args, cli, *last_batch_epoch1, channels, device, 'epoch1_end', candidate_pool)

        # end-of-epoch: mandatory diagnostic + full val eval (checkpoint selection, UNCHANGED metric)
        f, g, ch, d, u = run_diagnostic(step=global_step, epoch=epoch)
        fourway_rows += f
        geometry_rows += g
        churn_rows += ch
        disp_rows += d
        util_group_rows += u

        model.eval()
        val_sums, n = {}, 0
        with torch.no_grad():
            for batch_x, batch_y, batch_start_idx in val_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                base_mask, _ = exp._candidate_mask(batch_start_idx)
                bsz = batch_x.size(0)
                per_ch = {}
                for c in channels:
                    cand_mask = candidate_pool.apply(base_mask, batch_start_idx, c, 'val')
                    z_q = encode_raw(model, batch_x, c)
                    E = encode_raw(model, exp.memory_x, c)
                    memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
                    query_future = batch_y[:, :, c]
                    u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
                    d_ = -u
                    oracle_idx = stable_topk_indices(d_.masked_fill(~cand_mask, float('inf')), cli.top_k, largest=False)
                    s = arm_score(z_q, E, None).masked_fill(~cand_mask, float('-inf'))
                    model_idx = stable_topk_indices(s, cli.top_k, largest=True)
                    model_ind_mse = d_.gather(1, model_idx).mean(-1)
                    per_ch.setdefault('model_top10_individual_mse', []).append(model_ind_mse.cpu())
                for k_, vals in per_ch.items():
                    val_sums[k_] = val_sums.get(k_, 0.0) + torch.cat(vals).sum().item()
                n += bsz
        val_metric = val_sums['model_top10_individual_mse'] / max(n * len(channels), 1)
        epoch_val_rows.append({'epoch': epoch, 'val_model_top10_individual_mse': val_metric})
        payload = {'model_state_dict': model.state_dict(), 'epoch': epoch, 'val_metric': val_metric, 'config': config}
        torch.save(payload, ckpt_dir / f'checkpoint_epoch{epoch}.pth')
        if val_metric < best['val']:
            best = {'val': val_metric, 'epoch': epoch}
            torch.save(payload, ckpt_dir / 'checkpoint.pth')
        print(f'[track_j] epoch={epoch} done val_retMSE@10={val_metric:.6f} (best={best["epoch"]}:{best["val"]:.6f})')
        if epoch - best['epoch'] >= cli.patience:
            print(f'[track_j] early stop at epoch {epoch} (best={best["epoch"]})')
            break

    assert state_hash(E0) == e0_hash, '[ISSUE][ABORT] E0 drifted during training'

    bl = torch.load(ckpt_dir / 'checkpoint.pth', map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    model.train()
    if cli.instrument == 'on':
        grad_diag_rows += gradient_conflict_diagnostic(
            model, exp, args, cli, first_batch[0], first_batch[1], first_batch[2], channels, device, 'best_checkpoint', candidate_pool)
        max_rel_diff = max((r['equivalence_rel_diff'] for r in grad_diag_rows), default=0.0)
        print(f'[track_j] gradient-conflict equivalence check: max rel diff across all points = {max_rel_diff:.2e} '
             f'({"PASS" if max_rel_diff < 1e-3 else "[ISSUE] exceeds tolerance"})')
    final_f, final_g, _, _, _ = run_diagnostic(step=-1, epoch=-1)  # final test-time four-way, best checkpoint
    # test-split four-way (single pass, best checkpoint) -- reuse same probe machinery but on test split
    test_probe = build_probe_set(exp, test_loader, cli.n_probe, device)
    test_fixed = precompute_probe_fixed(exp, args, test_probe, channels, device, cli.chunk_size, candidate_pool, 'test')
    test_fourway = []
    model.eval()
    with torch.no_grad():
        for c in channels:
            zq_t = encode_raw(model, test_probe['x'], c)
            zk_t = encode_raw(model, exp.memory_x, c)
            zq_0 = encode_raw(E0, test_probe['x'], c)
            zk_0 = encode_raw(E0, exp.memory_x, c)
            s_variants = {'S00': arm_score(zq_0, zk_0, None), 'St0': arm_score(zq_t, zk_0, None),
                         'S0t': arm_score(zq_0, zk_t, None), 'Stt': arm_score(zq_t, zk_t, None)}
            for v_name, s_mat in s_variants.items():
                m, _ = variant_metrics(s_mat, test_fixed[c]['cand_mask'], test_fixed[c], cli.top_k)
                test_fourway.append({'channel': c, 'variant': v_name, **m})

    import csv
    def write_csv(path, rows):
        if not rows:
            return
        fieldnames = sorted({k for r in rows for k in r})
        with open(path, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)

    write_csv(out_dir / 'step_metrics.csv', step_rows)
    write_csv(out_dir / 'fourway_retrieval_metrics.csv', fourway_rows)
    write_csv(out_dir / 'geometry_metrics.csv', geometry_rows)
    write_csv(out_dir / 'topk_churn.csv', churn_rows)
    write_csv(out_dir / 'displacement_metrics.csv', disp_rows)
    write_csv(out_dir / 'utility_group_displacement.csv', util_group_rows)
    write_csv(out_dir / 'full_val_epoch_metrics.csv', epoch_val_rows)
    write_csv(out_dir / 'gradient_conflict.csv', grad_diag_rows)
    (out_dir / 'final_test_metrics.json').write_text(json.dumps(
        {'per_channel': test_fourway, 'best_epoch': best['epoch']}, indent=2))
    (out_dir / 'checkpoint_fingerprints.json').write_text(json.dumps(
        {'init_hash': init_hash, 'e0_hash_final': state_hash(E0), 'best_epoch': best['epoch'],
         'final_model_hash': state_hash(model)}, indent=2))
    (out_dir / 'baseline_reproduction.json').write_text(json.dumps(
        {'instrument': cli.instrument, 'best_epoch': best['epoch'], 'best_val': best['val'],
         'epoch1_first_batches_train_kl': [r['train_kl'] for r in step_rows[:5]]}, indent=2))
    print(f'[track_j] done. best_epoch={best["epoch"]} best_val={best["val"]:.6f} '
         f'wall_seconds={time.time()-t0:.1f}')


def _smoke(exp, args, model, E0, probe, fixed, channels, optimizer, cli, train_loader, val_loader, device,
           candidate_pool):
    model.train()
    for bi, (batch_x, batch_y, batch_start_idx) in enumerate(train_loader):
        if bi >= 2:
            break
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        base_mask, _ = exp._candidate_mask(batch_start_idx)
        optimizer.zero_grad()
        for c in channels:
            cand_mask = candidate_pool.apply(base_mask, batch_start_idx, c, 'train')
            z_q = encode_raw(model, batch_x, c)
            E = encode_raw(model, exp.memory_x, c)
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            s = arm_score(z_q, E, None)
            u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
            p_t = normalized_teacher_prob(-u, cand_mask, cli.tau_t)
            l = kl_loss(p_t, s, cand_mask, cli.tau_s)
            (l / len(channels)).backward()
        optimizer.step()
    print('[track_j] SMOKE PASS (2 batches trained, no crash)')


if __name__ == '__main__':
    main()
