#!/usr/bin/env python3
"""TRACK-P100-ASYMMETRIC-DUAL-ENCODER-PRR01 -- PHASE 1: Shared vs
Separate Query/Key Encoder, within the fixed canonical Shared-P100
candidate support (`candidate_pool_mode='coarse_topk'`, size=100,
metric='delta_last_cosine' -- the EXACT same pool caches used by every
other arm this session; never recomputed, never full-mode, per the
spec's explicit prohibition).

Four arms, unified through ONE `compute_scores_dual` function so the
training loop/loss/checkpoint-selection/most diagnostics are written
once:

  E0  Canonical Shared Encoder.  model_k IS model_q (same object).
      This is an exact re-implementation of the V5 arm of
      `train_v_sharedtop100_01.py` (num_query_views=5), just routed
      through `compute_scores_dual` instead of `compute_scores` --
      confirmed equivalent by construction (model_k=model_q,
      key_proj=None reduces `compute_scores_dual` to `compute_scores`'s
      coarse_topk branch exactly).
  E1  Separate Query/Key Encoder.  model_k = deepcopy(model_q) taken
      immediately after `build_model` returns -- bit-identical initial
      weights by construction (the SAME pattern already used and
      verified in `train_j2_key_update_decomposition01.py`), then BOTH
      encoders receive full gradient every step (no `.detach()`/
      `no_grad` on the key branch -- this is what makes E1 differ from
      TRACK-J2's J2 arm, whose key encoder was frozen at init).
  E2  Shared Encoder + Key Projection.  model_k IS model_q; an
      additional `KeyProjection` (Linear(d_model,d_model), identity
      weight + zero bias at init) is applied to the raw key embedding
      before L2-normalization. At step 0 this is a no-op by
      construction (identity map) -- asserted numerically
      (max abs diff vs. key_proj=None < 1e-6) before training starts.
  E3  Capacity-Matched Shared Encoder.  model_k IS model_q, but built
      with a WIDENED `d_ff` (closed-form-solved by
      `solve_capacity_matched_d_ff`) so the encoder's own parameter
      count lands within +/-5% of E1's total (TWO full encoders);
      d_model (= embedding dim) is held fixed at 128 per spec. The
      achieved percentage is computed and recorded, never assumed.

Reused UNMODIFIED (imports, no local redefinition) from the canonical
V5 script's own reuse chain: `build_model`/`state_hash`/
`geometry_metrics`/`effective_rank_entropy`
(`train_j_shared_encoder_drift01.py`), `encode_raw`/
`individual_utility_memsafe` (`train_factorial_e2e01.py`),
`normalized_teacher_prob` (`train_horizon_retrieval_expert01.py`),
`SlotHeads`/`kl_loss_from_prob` (`train_k_multislot_predictive_retrieval01.py`),
`memory_value` (`train_margutil01.py`), `ndcg_at_k`/`recall_at_k`
(`train_patch_retrieval_expert01.py`), `round_robin_topk_selection`
(`train_t_pure_multislot01.py`), `CandidatePoolCache`
(`train_retriever_pool01.py`), `gather_candidate_histories`/
`encode_pooled_candidates`/`gather_candidate_values`/`local_to_global`/
`pooled_future_mse`/`CandidatePoolConfig` (`utils/candidate_pool.py`),
`batch_order_sha256`/`make_loader_generator`/`set_global_seeds`
(`scripts/rng_control01.py`), `stable_topk_indices`
(`models/RelationStage1.py`).

The ONLY new code in this file is: `KeyProjection`, `compute_scores_dual`
(a drop-in generalization of `compute_scores`'s coarse_topk branch to
two possibly-distinct models), `solve_capacity_matched_d_ff` (closed
-form param-count solver for E3), and the CLI/training loop that
dispatches on `--arm`.
"""
import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.rng_control01 import batch_order_sha256, make_loader_generator, set_global_seeds
from scripts.train_factorial_e2e01 import encode_raw, individual_utility_memsafe
from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob
from scripts.train_j_shared_encoder_drift01 import build_model, geometry_metrics, state_hash
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads, kl_loss_from_prob
from scripts.train_margutil01 import build_experiment, memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k
from scripts.train_retriever_pool01 import CandidatePoolCache
from scripts.train_t_pure_multislot01 import round_robin_topk_selection
from utils.candidate_pool import gather_candidate_histories, gather_candidate_values, \
    encode_pooled_candidates, local_to_global, pooled_future_mse

EPS = 1e-8
TOP_K = 10
N_SLOTS = 5
ARMS = ('E0', 'E1', 'E2', 'E3')


def _entropy_and_effective(p, valid_mask):
    p_safe = p.clamp_min(EPS)
    ent = -(p_safe * p_safe.log()).sum(-1)
    return float(ent.mean()), float(ent.mean().exp())


def build_model_with_dff(cli, device, d_ff_override):
    """Same as `build_model` (`train_j_shared_encoder_drift01.py`),
    with ONE addition: an explicit `d_ff` override for E3's capacity
    -matched architecture -- that function does not expose this knob,
    so it is re-specified here rather than modifying the shared
    helper. Every other override is copy-identical to `build_model`."""
    set_global_seeds(cli.init_seed)
    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.seq_len, 'batch_size': cli.batch_size,
        'seed': cli.init_seed, 'top_k': cli.top_k,
        'relation_encoder_type': 'mlp', 'relation_self_fill': 'linear',
        'relation_input_space': 'delta_last', 'relation_teacher_space': 'delta_last',
        'relation_value_space': 'delta_last', 'candidate_mask': 'raft',
        'patch_len': cli.patch_len, 'stride': cli.patch_len,
        'd_ff': d_ff_override,
    })
    exp._ensure_memory()
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.to(device)
    return exp, args, model


def solve_capacity_matched_d_ff(d_model, seq_len, target_total_params):
    """Closed-form `d_ff` for a single MLP RelationEncoder
    (`models/RelationStage1.py`, `encoder_type='mlp'`,
    `relation_self_fill='linear'`) so its total parameter count hits
    `target_total_params` exactly (both `encoder.0` and `encoder.3`
    are linear in `d_ff`, so this is an exact linear solve, not a
    search):

      encoder = Linear(seq_len, d_ff) -> GELU -> Dropout -> Linear(d_ff, d_model)
      norm    = LayerNorm(d_model)                      [d_ff-independent]
      proj    = Linear(d_model,d_model) -> GELU -> Linear(d_model,d_model)  [d_ff-independent]

    params(d_ff) = [seq_len*d_ff + d_ff]              encoder.0 (w+b)
                  + [d_model*d_ff + d_model]            encoder.3 (w+b)
                  + [2*d_model]                         norm (w+b)
                  + [2*(d_model*d_model + d_model)]     proj (2 linear layers)
                  = d_ff*(seq_len + 1 + d_model) + FIXED
    """
    fixed = d_model + 2 * d_model + 2 * (d_model * d_model + d_model)
    coeff = seq_len + 1 + d_model
    d_ff = max(1, round((target_total_params - fixed) / coeff))
    achieved = coeff * d_ff + fixed
    return d_ff, achieved


def encoder_param_count(model):
    """Total parameter count of the trainable RelationEncoder submodule
    (`model.encoder`: the `encoder`+`norm`+`proj` MLP stack) -- the part
    E1 duplicates and E3 widens. Everything else (SlotHeads, etc.) is
    counted separately and identically across all 4 arms."""
    return sum(p.numel() for p in model.encoder.parameters())


class KeyProjection(nn.Module):
    """E2's key-side asymmetric projection. Identity-initialized (zero
    bias) so E2's forward is numerically identical to E0's at step 0
    (asserted in `main()` before training starts: max abs diff < 1e-6)."""

    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(d_model))
            self.linear.bias.zero_()

    def forward(self, x):
        return self.linear(x)


def compute_scores_dual(model_q, model_k, key_proj, slot_heads, batch_x, exp, channel, pool_cache,
                        batch_start_idx, device):
    """Drop-in generalization of `train_retriever_pool01.compute_scores`'s
    `coarse_topk` branch to two POSSIBLY distinct models. When
    `model_k is model_q` and `key_proj is None`, this is byte-for-byte
    the same computation as the canonical `compute_scores` (verified:
    same `encode_raw`/`gather_candidate_histories`/
    `encode_pooled_candidates`/einsum calls, same order).
    Returns (scores [B,S,M], valid_mask [B,M], pool_idx_global, z_q [B,d], z_k_raw [B,M,d])
    -- `z_q`/`z_k_raw` are returned (pre-normalize for z_k) so callers
    can compute encoder-specific diagnostics (grad norms, effective
    rank, cross-encoder cosine) without a second forward pass."""
    z_q = encode_raw(model_q, batch_x, channel)
    pool_idx_global = pool_cache.lookup(batch_start_idx, channel, device)  # [B, M]
    pooled_x = gather_candidate_histories(exp.memory_x, pool_idx_global)    # [B, M, L, C]
    z_k_raw = encode_pooled_candidates(lambda x, c: encode_raw(model_k, x, c), pooled_x, channel)
    z_k_proj = key_proj(z_k_raw) if key_proj is not None else z_k_raw
    z_k = F.normalize(z_k_proj, dim=-1)
    if slot_heads is None:
        scores = torch.einsum('bd,bmd->bm', F.normalize(z_q, dim=-1), z_k).unsqueeze(1)
    else:
        q = slot_heads(z_q)
        scores = torch.einsum('bsd,bmd->bsm', q, z_k)
    pool_valid_mask = torch.ones(pool_idx_global.shape, dtype=torch.bool, device=device)
    return scores, pool_valid_mask, pool_idx_global, z_q, z_k_raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--arm', required=True, choices=ARMS)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--patch_len', type=int, default=16)
    ap.add_argument('--top_k', type=int, default=TOP_K)
    ap.add_argument('--tau_t', type=float, default=0.1)
    ap.add_argument('--tau_s', type=float, default=0.1)
    ap.add_argument('--slot_std', type=float, default=1e-3)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--train_epochs', type=int, default=10)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--disable_early_stopping', action='store_true')
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--init_seed', type=int, default=0)
    ap.add_argument('--loader_seed', type=int, default=0)
    ap.add_argument('--candidate_mask', default='raft')
    ap.add_argument('--candidate_pool_size', type=int, default=100)
    ap.add_argument('--candidate_pool_cache', required=True,
                    help='directory with train.pt/val.pt/test.pt -- the SAME Shared-P100 cache '
                         'used by every other arm/track this session')
    ap.add_argument('--n_probe_rank', type=int, default=256)
    ap.add_argument('--capacity_tolerance_pct', type=float, default=5.0,
                    help='E3 must land within this %% of E1 total params (spec requirement)')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--checkpoints', required=True)
    ap.add_argument('--limit_batches', type=int, default=0, help='SMOKE/DEBUG ONLY')
    ap.add_argument('--smoke_test', action='store_true')
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(cli.out_dir) / cli.cell / cli.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints) / cli.cell / cli.arm
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- build model_q / model_k / key_proj per arm ----
    exp, args, model_q = build_model(cli, device)
    channels = list(range(int(args.enc_in)))
    d_model = int(args.d_model)
    init_hash_q = state_hash(model_q)
    e0_encoder_params = encoder_param_count(model_q)  # == E2/E3's base-encoder count at d_ff=args.d_ff

    model_k, key_proj, capacity_info = model_q, None, None
    if cli.arm == 'E1':
        model_k = copy.deepcopy(model_q)
        # bit-identical init is BY CONSTRUCTION (deepcopy), verified anyway:
        assert state_hash(model_k) == init_hash_q, '[ISSUE][ABORT] E1 key encoder init does not match query encoder init'
        for p in model_k.parameters():
            p.requires_grad_(True)
    elif cli.arm == 'E2':
        key_proj = KeyProjection(d_model).to(device)
    elif cli.arm == 'E3':
        e1_target = 2 * e0_encoder_params
        d_ff_e3, achieved = solve_capacity_matched_d_ff(d_model, cli.seq_len, e1_target)
        exp, args, model_q = build_model_with_dff(cli, device, d_ff_e3)
        model_k = model_q
        channels = list(range(int(args.enc_in)))
        init_hash_q = state_hash(model_q)
        e3_encoder_params = encoder_param_count(model_q)
        pct_off = 100.0 * (e3_encoder_params - e1_target) / e1_target
        capacity_info = {'d_ff_e0': int(getattr(args, 'd_ff', None) if False else None) or None,
                         'd_ff_e3_solved': d_ff_e3, 'e0_encoder_params': e0_encoder_params,
                         'e1_target_params (2x E0)': e1_target, 'e3_encoder_params_achieved': e3_encoder_params,
                         'pct_off_target': pct_off}
        if abs(pct_off) > cli.capacity_tolerance_pct:
            raise SystemExit(f'[ISSUE][ABORT] E3 capacity match {pct_off:.2f}% exceeds '
                             f'+/-{cli.capacity_tolerance_pct}% tolerance (solved d_ff={d_ff_e3})')
        print(f'[dual_encoder] E3 capacity match: e0_encoder_params={e0_encoder_params} '
             f'e1_target={e1_target} d_ff_e3={d_ff_e3} achieved={e3_encoder_params} '
             f'pct_off={pct_off:.3f}%')

    init_hash_k = state_hash(model_k) if model_k is not model_q else init_hash_q

    slot_heads = SlotHeads(d_model, n_slots=N_SLOTS, std=cli.slot_std).to(device)

    # ---- E2 step-0 numerical equivalence check (spec-required) ----
    if cli.arm == 'E2':
        _, _val_loader_probe = exp._get_data(flag='val', shuffle=False)
        probe_batch = next(iter(_val_loader_probe))
        pbx, _pby, pbs = probe_batch
        pbx = pbx.float().to(device)
        with torch.no_grad():
            s_e2, _, _, _, _ = compute_scores_dual(model_q, model_k, key_proj, slot_heads, pbx, exp, 0,
                                                    CandidatePoolCache(Path(cli.candidate_pool_cache) / 'val.pt',
                                                                      {'seq_len': cli.seq_len, 'pred_len': cli.pred_len,
                                                                       'candidate_mask': cli.candidate_mask,
                                                                       'candidate_pool_size': cli.candidate_pool_size,
                                                                       'candidate_pool_metric': 'delta_last_cosine'}),
                                                    pbs, device)
            s_e0, _, _, _, _ = compute_scores_dual(model_q, model_k, None, slot_heads, pbx, exp, 0,
                                                    CandidatePoolCache(Path(cli.candidate_pool_cache) / 'val.pt',
                                                                      {'seq_len': cli.seq_len, 'pred_len': cli.pred_len,
                                                                       'candidate_mask': cli.candidate_mask,
                                                                       'candidate_pool_size': cli.candidate_pool_size,
                                                                       'candidate_pool_metric': 'delta_last_cosine'}),
                                                    pbs, device)
        max_abs_diff = float((s_e2 - s_e0).abs().max())
        print(f'[dual_encoder] E2 init-equivalence check: max_abs_diff(E2,E0)={max_abs_diff:.3e}')
        assert max_abs_diff < 1e-6, f'[ISSUE][ABORT] E2 init != E0 init, max_abs_diff={max_abs_diff:.3e}'

    expected_meta = {'seq_len': cli.seq_len, 'pred_len': cli.pred_len, 'candidate_mask': cli.candidate_mask,
                     'candidate_pool_size': cli.candidate_pool_size, 'candidate_pool_metric': 'delta_last_cosine'}
    cache_dir = Path(cli.candidate_pool_cache)
    pool_caches = {}
    for split in ('train', 'val', 'test'):
        cache_path = cache_dir / f'{split}.pt'
        if not cache_path.exists():
            raise SystemExit(f'[ISSUE][ABORT] missing candidate pool cache file: {cache_path}')
        pool_caches[split] = CandidatePoolCache(cache_path, expected_meta)

    for p in model_q.parameters():
        p.requires_grad_(True)
    params = list(model_q.parameters())
    if model_k is not model_q:
        params += list(model_k.parameters())
    if key_proj is not None:
        params += list(key_proj.parameters())
    params += list(slot_heads.parameters())
    optimizer = torch.optim.Adam(params, lr=cli.learning_rate)

    train_gen = make_loader_generator(cli.loader_seed)
    _, train_loader = exp._get_data(flag='train', shuffle=True, generator=train_gen)
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)

    config = {
        'exp': 'TRACK-P100-ASYMMETRIC-DUAL-ENCODER-PRR01-PHASE1', 'cell': cli.cell, 'arm': cli.arm,
        'candidate_pool_mode': 'coarse_topk', 'candidate_pool_size': cli.candidate_pool_size,
        'candidate_pool_metric': 'delta_last_cosine', 'full_candidate_count': int(exp.memory_x.size(0)),
        'tau_t': cli.tau_t, 'tau_s': cli.tau_s, 'top_k': cli.top_k, 'batch_size': cli.batch_size,
        'learning_rate': cli.learning_rate, 'train_epochs': cli.train_epochs, 'patience': cli.patience,
        'init_seed': cli.init_seed, 'loader_seed': cli.loader_seed, 'init_hash_q': init_hash_q,
        'init_hash_k': init_hash_k, 'shares_encoder': model_k is model_q, 'capacity_info': capacity_info,
        'n_slots': N_SLOTS, 'slot_std': cli.slot_std,
        'checkpoint_policy': 'primary=min_val_retmse10 (Stage-2 uses ONLY this one, matching V5/every '
                             'other arm this session), secondary=min_val_kl (diagnostic only)',
    }
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2, default=str))
    print(f'[dual_encoder] arm={cli.arm} cell={cli.cell} shares_encoder={model_k is model_q} '
         f'init_hash_q={init_hash_q[:16]} init_hash_k={init_hash_k[:16]}')

    def eval_channel(batch_x, batch_y, batch_start_idx, c, cand_mask, split, want_full_oracle=False,
                     want_encoder_diag=False):
        scores, valid_mask, pool_idx_global, z_q, z_k_raw = compute_scores_dual(
            model_q, model_k, key_proj, slot_heads, batch_x, exp, c, pool_caches[split], batch_start_idx, device)
        memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
        query_future = batch_y[:, :, c]

        pooled_memory_c = gather_candidate_values(memory_c, pool_idx_global)
        d_pool = pooled_future_mse(pooled_memory_c, offset_c, query_future)
        restricted_oracle_local = stable_topk_indices(d_pool, cli.top_k, largest=False)
        model_idx_local = round_robin_topk_selection(scores, valid_mask, k=cli.top_k)
        y_sel = pooled_memory_c.gather(
            1, model_idx_local.unsqueeze(-1).expand(-1, -1, pooled_memory_c.size(-1))
        ) + offset_c.view(-1, 1, 1)
        individual_mse_i = d_pool.gather(1, model_idx_local)
        oracle_ind_mse = d_pool.gather(1, restricted_oracle_local)
        D_ = individual_mse_i.mean(-1) / cli.top_k
        agg_pred = y_sel.mean(dim=1)
        agg_mse = ((agg_pred - query_future) ** 2).mean(-1)
        C_ = agg_mse - D_
        recall10_p100 = recall_at_k(model_idx_local, restricted_oracle_local, cli.top_k)
        ndcg10 = ndcg_at_k(model_idx_local, d_pool, valid_mask, cli.top_k)

        teacher_ent, teacher_eff = _entropy_and_effective(
            normalized_teacher_prob(d_pool, valid_mask, cli.tau_t), valid_mask)
        s_masked = scores.masked_fill(~valid_mask.unsqueeze(1), float('-inf'))
        p_m = torch.softmax(s_masked / cli.tau_s, dim=-1)
        p_bar = p_m.mean(dim=1)
        student_ent, student_eff = _entropy_and_effective(p_bar, valid_mask)
        p_t = normalized_teacher_prob(d_pool, valid_mask, cli.tau_t)
        kl = kl_loss_from_prob(p_t, p_bar, valid_mask)

        res = dict(model_idx_global=local_to_global(pool_idx_global, model_idx_local),
                  model_ind_mse=individual_mse_i.mean(-1), oracle_ind_mse=oracle_ind_mse.mean(-1),
                  recall10_p100=recall10_p100, ndcg10=ndcg10, agg_mse=agg_mse, D=D_, C=C_, kl=kl,
                  teacher_entropy=teacher_ent, teacher_effective_candidates=teacher_eff,
                  student_entropy=student_ent, student_effective_candidates=student_eff,
                  pool_idx_global=pool_idx_global)

        if want_full_oracle:
            with torch.no_grad():
                u_full = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
                d_full = -u_full
                full_oracle_idx = stable_topk_indices(d_full.masked_fill(~cand_mask, float('inf')),
                                                      cli.top_k, largest=False)
                recall10_full = recall_at_k(res['model_idx_global'], full_oracle_idx, cli.top_k)
                overlap = (full_oracle_idx.unsqueeze(-1) == pool_idx_global.unsqueeze(1)).any(-1).float().sum(-1)
                coverage = overlap / cli.top_k
            res['recall10_full'] = recall10_full
            res['coverage_at_pool'] = coverage
        if want_encoder_diag:
            res['z_q_detached'] = z_q.detach()
            res['z_k_raw_detached'] = z_k_raw.detach()
        return res

    if cli.smoke_test:
        model_q.train()
        if model_k is not model_q:
            model_k.train()
        slot_heads.train()
        for bi, (batch_x, batch_y, batch_start_idx) in enumerate(train_loader):
            if bi >= 2:
                break
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            cand_mask, _ = exp._candidate_mask(batch_start_idx)
            optimizer.zero_grad()
            for c in channels:
                res = eval_channel(batch_x, batch_y, batch_start_idx, c, cand_mask, 'train')
                (res['kl'] / len(channels)).backward()
            assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model_q.encoder.parameters()), \
                '[ISSUE][ABORT] query encoder has zero/missing gradient'
            if model_k is not model_q:
                assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model_k.encoder.parameters()), \
                    '[ISSUE][ABORT] key encoder has zero/missing gradient (E1 must train both branches)'
                q_grad = torch.cat([p.grad.detach().flatten() for p in model_q.encoder.parameters()])
                k_grad = torch.cat([p.grad.detach().flatten() for p in model_k.encoder.parameters()])
                assert not torch.allclose(q_grad, k_grad), \
                    '[ISSUE][ABORT] E1 query/key encoder gradients are identical -- key branch may be aliased to query'
            optimizer.step()
        if model_k is not model_q:
            q_vec = torch.cat([p.detach().flatten() for p in model_q.encoder.parameters()])
            k_vec = torch.cat([p.detach().flatten() for p in model_k.encoder.parameters()])
            assert not torch.allclose(q_vec, k_vec), \
                '[ISSUE][ABORT] E1 query/key encoder WEIGHTS still identical after an optimizer step -- not training independently'
        vram = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
        print(f'[dual_encoder] {cli.arm} SMOKE PASS max_vram_mb={vram:.1f}')
        return

    epoch_val_rows, step_rows, batch_order_hashes = [], [], []
    best_retmse = {'val': float('inf'), 'epoch': -1}
    best_kl = {'val': float('inf'), 'epoch': -1}
    t0 = time.time()
    global_step = 0
    for epoch in range(1, cli.train_epochs + 1):
        model_q.train()
        if model_k is not model_q:
            model_k.train()
        slot_heads.train()
        starts = []
        for bi, (batch_x, batch_y, batch_start_idx) in enumerate(train_loader):
            if cli.limit_batches and bi >= cli.limit_batches:
                break
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            starts.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx)
                         else torch.as_tensor(batch_start_idx))
            cand_mask, _ = exp._candidate_mask(batch_start_idx)
            optimizer.zero_grad()
            batch_loss = 0.0
            q_grad_norm_sum, k_grad_norm_sum = 0.0, 0.0
            for c in channels:
                res = eval_channel(batch_x, batch_y, batch_start_idx, c, cand_mask, 'train')
                l = res['kl']
                (l / len(channels)).backward()
                batch_loss += float(l.detach()) / len(channels)
            q_grad_norm_sum = float(torch.cat([p.grad.flatten() for p in model_q.encoder.parameters()
                                              if p.grad is not None]).norm())
            if model_k is not model_q:
                k_grad_norm_sum = float(torch.cat([p.grad.flatten() for p in model_k.encoder.parameters()
                                                  if p.grad is not None]).norm())
            optimizer.step()
            global_step += 1
            step_rows.append({'global_step': global_step, 'epoch': epoch, 'batch_in_epoch': bi,
                              'train_loss_kl': batch_loss, 'q_encoder_grad_norm': q_grad_norm_sum,
                              'k_encoder_grad_norm': k_grad_norm_sum})
        batch_order_hashes.append({'epoch': epoch, 'batch_order_sha256': batch_order_sha256(starts)})

        model_q.eval()
        if model_k is not model_q:
            model_k.eval()
        slot_heads.eval()
        val_sums, n = {}, 0
        with torch.no_grad():
            for batch_x, batch_y, batch_start_idx in val_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                cand_mask, _ = exp._candidate_mask(batch_start_idx)
                bsz = batch_x.size(0)
                per_ch = {}
                for c in channels:
                    res = eval_channel(batch_x, batch_y, batch_start_idx, c, cand_mask, 'val',
                                       want_full_oracle=True)
                    for key_ in ('kl', 'model_ind_mse', 'agg_mse', 'recall10_p100', 'recall10_full',
                               'ndcg10', 'coverage_at_pool', 'teacher_entropy', 'teacher_effective_candidates',
                               'student_entropy', 'student_effective_candidates'):
                        v = res[key_]
                        per_ch.setdefault(key_, []).append(v.reshape(-1).cpu() if torch.is_tensor(v)
                                                           else torch.tensor([float(v)]))
                for k_, vals in per_ch.items():
                    val_sums[k_] = val_sums.get(k_, 0.0) + torch.cat(vals).sum().item()
                n += bsz
        n_eff = max(n * len(channels), 1)
        val_metrics = {k_: v / n_eff for k_, v in val_sums.items()}
        val_kl = val_metrics['kl']
        val_retmse10 = val_metrics['model_ind_mse']
        val_metrics['epoch'] = epoch
        epoch_val_rows.append(val_metrics)

        payload = {'model_q_state_dict': model_q.state_dict(),
                  'model_k_state_dict': model_k.state_dict() if model_k is not model_q else None,
                  'key_proj_state_dict': key_proj.state_dict() if key_proj is not None else None,
                  'slot_heads_state_dict': slot_heads.state_dict(),
                  'shares_encoder': model_k is model_q,
                  'epoch': epoch, 'val_kl': val_kl, 'val_retmse10': val_retmse10, 'config': config}
        torch.save(payload, ckpt_dir / f'checkpoint_epoch{epoch}.pth')
        if val_retmse10 < best_retmse['val']:
            best_retmse = {'val': val_retmse10, 'epoch': epoch}
            torch.save(payload, ckpt_dir / 'checkpoint_best_retmse.pth')
        if val_kl < best_kl['val']:
            best_kl = {'val': val_kl, 'epoch': epoch}
            torch.save(payload, ckpt_dir / 'checkpoint_best_kl.pth')
        print(f'[dual_encoder] {cli.arm} epoch={epoch} val_kl={val_kl:.6f} val_retmse10={val_retmse10:.6f} '
             f'val_recall10_full={val_metrics["recall10_full"]:.4f} '
             f'coverage@pool={val_metrics["coverage_at_pool"]:.4f} '
             f'(best_retmse={best_retmse["epoch"]}:{best_retmse["val"]:.6f} '
             f'best_kl={best_kl["epoch"]}:{best_kl["val"]:.6f})')
        if not cli.disable_early_stopping and epoch - best_retmse['epoch'] >= cli.patience:
            print(f'[dual_encoder] {cli.arm} early stop at epoch {epoch} (best_retmse={best_retmse["epoch"]})')
            break

    def run_test(ckpt_path, tag):
        bl = torch.load(ckpt_path, map_location=device)
        model_q.load_state_dict(bl['model_q_state_dict'])
        if model_k is not model_q and bl['model_k_state_dict'] is not None:
            model_k.load_state_dict(bl['model_k_state_dict'])
        if key_proj is not None and bl['key_proj_state_dict'] is not None:
            key_proj.load_state_dict(bl['key_proj_state_dict'])
        slot_heads.load_state_dict(bl['slot_heads_state_dict'])
        model_q.eval()
        if model_k is not model_q:
            model_k.eval()
        slot_heads.eval()
        test_sums, n = {}, 0
        probe_embeds_q, probe_embeds_k = [], []
        with torch.no_grad():
            for batch_x, batch_y, batch_start_idx in test_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                cand_mask, _ = exp._candidate_mask(batch_start_idx)
                bsz = batch_x.size(0)
                per_ch = {}
                for c in channels:
                    res = eval_channel(batch_x, batch_y, batch_start_idx, c, cand_mask, 'test',
                                       want_full_oracle=True)
                    for key_ in ('kl', 'model_ind_mse', 'agg_mse', 'recall10_p100', 'recall10_full',
                               'ndcg10', 'coverage_at_pool', 'D', 'C', 'oracle_ind_mse'):
                        v = res[key_]
                        per_ch.setdefault(key_, []).append(v.reshape(-1).cpu() if torch.is_tensor(v)
                                                           else torch.tensor([float(v)]))
                    if c == channels[0] and len(probe_embeds_q) < cli.n_probe_rank:
                        n_take = min(cli.n_probe_rank, exp.memory_x.size(0))
                        probe_embeds_q.append(encode_raw(model_q, exp.memory_x[:n_take], c))
                        probe_embeds_k.append(encode_raw(model_k, exp.memory_x[:n_take], c))
                for k_, vals in per_ch.items():
                    test_sums[k_] = test_sums.get(k_, 0.0) + torch.cat(vals).sum().item()
                n += bsz
        n_eff = max(n * len(channels), 1)
        test_metrics = {k_: v / n_eff for k_, v in test_sums.items()}
        test_metrics['n_queries_seen'] = n
        test_metrics['best_epoch'] = int(bl['epoch'])
        if probe_embeds_q:
            with torch.no_grad():
                geom_q = geometry_metrics(probe_embeds_q[0])
                geom_k = geometry_metrics(probe_embeds_k[0])
                cross_cos = F.cosine_similarity(probe_embeds_q[0], probe_embeds_k[0], dim=-1)
                drift_l2 = float(torch.cat([(pq - pk).flatten() for pq, pk in
                                            zip(model_q.encoder.parameters(), model_k.encoder.parameters())]).norm()) \
                    if model_k is not model_q else 0.0
            test_metrics['query_geometry'] = geom_q
            test_metrics['key_geometry'] = geom_k
            test_metrics['query_key_same_input_cosine_mean'] = float(cross_cos.mean())
            test_metrics['query_key_same_input_cosine_std'] = float(cross_cos.std())
            test_metrics['query_key_param_l2_drift'] = drift_l2
        (out_dir / f'final_test_metrics_{tag}.json').write_text(json.dumps(test_metrics, indent=2, default=str))
        print(f'[dual_encoder] {cli.arm} [{tag}] best_epoch={test_metrics["best_epoch"]} test_KL={test_metrics["kl"]:.6f} '
             f'test_retMSE@10={test_metrics["model_ind_mse"]:.6f} test_agg_mse10={test_metrics["agg_mse"]:.6f} '
             f'coverage@pool={test_metrics["coverage_at_pool"]:.4f} recall10_full={test_metrics["recall10_full"]:.4f}')
        return test_metrics

    test_metrics_retmse = run_test(ckpt_dir / 'checkpoint_best_retmse.pth', 'best_retmse')
    test_metrics_kl = run_test(ckpt_dir / 'checkpoint_best_kl.pth', 'best_kl')

    with open(out_dir / 'train_metrics.csv', 'w', newline='') as fh:
        fieldnames = sorted({k for r in step_rows for k in r})
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in step_rows:
            w.writerow(r)
    with open(out_dir / 'val_metrics.csv', 'w', newline='') as fh:
        fieldnames = sorted({k for r in epoch_val_rows for k in r})
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in epoch_val_rows:
            w.writerow(r)
    with open(out_dir / 'batch_order_hashes.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['epoch', 'batch_order_sha256'])
        w.writeheader()
        for r in batch_order_hashes:
            w.writerow(r)
    (out_dir / 'checkpoint_fingerprints.json').write_text(json.dumps(
        {'init_hash_q': init_hash_q, 'init_hash_k': init_hash_k, 'best_retmse_epoch': best_retmse['epoch'],
         'best_kl_epoch': best_kl['epoch'], 'batch_order_hashes': batch_order_hashes,
         'capacity_info': capacity_info}, indent=2, default=str))
    vram = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
    print(f'[dual_encoder] done. {cli.arm} best_retmse_epoch={best_retmse["epoch"]} best_kl_epoch={best_kl["epoch"]} '
         f'max_vram_mb={vram:.1f} wall_seconds={time.time()-t0:.1f}')


if __name__ == '__main__':
    main()
