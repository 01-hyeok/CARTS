#!/usr/bin/env python3
"""TRACK-J2-KEY-UPDATE-DECOMPOSITION01.

Direct follow-up to TRACK-J-SHARED-ENCODER-DRIFT01 (commit d3e0aa4).
TRACK-J's St0/S0t are POST-HOC cross-time evaluations (Et(Xq) vs the
frozen E0(Xk) snapshot) -- never actual training interventions. This
track runs two REAL training interventions that isolate the causal
contribution of the key/candidate branch, holding dataset, horizon, MLP
architecture, raw future-MSE teacher, teacher normalization, cosine
scoring, KL loss, tau, batch size, LR, optimizer, candidate mask/
universe, input/value space, Top-K, checkpoint metric, seed, and batch
order IDENTICAL to TRACK-J's own A0 (audited again here from
`results/TRACK-J-SHARED-ENCODER-DRIFT01/ETTh1_720/config.json` and
`exact_commands.txt`, not from the report prose).

J0 (existing A0, NOT re-run): both branches grad-tracked, single shared
   encoder. `train_kl`, probe trajectory, checkpoint = TRACK-J's own
   artifacts, reused as-is.
J1 StopGrad-Key: single shared encoder (same as J0), candidates
   RE-ENCODED EVERY STEP from the current encoder (identical forward
   call to J0), then `.detach()`-ed before scoring -- so candidate
   embeddings keep moving with the encoder, but contribute ZERO
   gradient. Isolates candidate-side GRADIENT's causal effect (J0 vs J1).
J2 True-Frozen-Key: TWO encoders, E_q (trainable) and E_k (frozen at
   init, `E_k = deepcopy(E_q^(0))`, `requires_grad_(False)`, `.eval()`
   always). Candidate bank K_0 = E_k(memory) computed ONCE, reused for
   every step -- never recomputed. Isolates the "does the candidate
   INDEX itself need to move" question (J1 vs J2), separate from the
   gradient question.

Reused UNMODIFIED from `train_j_shared_encoder_drift01.py`: `build_model`,
`build_probe_set`, `precompute_probe_fixed`, `variant_metrics`,
`geometry_metrics`, `effective_rank_entropy`, `churn`,
`retention_vs_init`, `displacement_stats`, `state_hash`. Reused
UNMODIFIED from the underlying library: `encode_raw`/`arm_score`
(`train_factorial_e2e01`), `normalized_teacher_prob`/`kl_loss`
(`train_horizon_retrieval_expert01`), `memory_value`
(`train_margutil01`), `stable_topk_indices` (`RelationStage1`),
`recall_at_k`/`ndcg_at_k` (`train_patch_retrieval_expert01`),
`make_loader_generator`/`batch_order_sha256` (`rng_control01`).
"""
import argparse
import copy
import csv
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
from scripts.rng_control01 import batch_order_sha256, make_loader_generator
from scripts.train_factorial_e2e01 import arm_score, encode_raw, individual_utility_memsafe
from scripts.train_horizon_retrieval_expert01 import kl_loss, normalized_teacher_prob
from scripts.train_j_shared_encoder_drift01 import (
    build_model, build_probe_set, churn, displacement_stats, geometry_metrics,
    precompute_probe_fixed, retention_vs_init, state_hash, variant_metrics,
)
from scripts.train_margutil01 import memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k

EPS = 1e-8
EXPECTED_J0_INIT_HASH = 'b37fa4031f538e4b5f5c522ae22e7d03613e4ee66283d2637656c7c4f872372e'


def score_fn(arm, model, E_k_or_none, K0, x, c, exp, device):
    """Returns (z_q, z_k_for_score, z_k_for_display). For J1: z_k is
    encode_raw(model, memory) FRESH every call, then the caller detaches
    it for the loss only (this function itself returns the un-detached
    tensor so callers can also inspect pre-detach values if needed).
    For J2: z_k is the cached K0 bank, never recomputed."""
    z_q = encode_raw(model, x, c)
    if arm == 'J1_stopgrad_key':
        z_k = encode_raw(model, exp.memory_x, c)
    else:  # J2
        z_k = K0[c]
    return z_q, z_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--arm', required=True, choices=('J1_stopgrad_key', 'J2_true_frozen_key'))
    ap.add_argument('--cell', default='ETTh1_720')
    ap.add_argument('--pred_len', type=int, default=720)
    ap.add_argument('--seq_len', type=int, default=720)
    ap.add_argument('--patch_len', type=int, default=16)
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
    ap.add_argument('--out_dir', default='results/TRACK-J2-KEY-UPDATE-DECOMPOSITION01')
    ap.add_argument('--checkpoints', default='checkpoints/track_j2_key_update_decomposition01')
    ap.add_argument('--limit_batches', type=int, default=0, help='SMOKE ONLY')
    ap.add_argument('--smoke_test', action='store_true')
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    arm_dir_name = cli.arm
    out_dir = Path(cli.out_dir) / cli.cell / arm_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints) / cli.cell / cli.arm
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    exp, args, model = build_model(cli, device)
    channels = list(range(int(args.enc_in)))

    init_hash = state_hash(model)
    assert init_hash == EXPECTED_J0_INIT_HASH, (
        f'[ISSUE][ABORT] init hash mismatch: {init_hash} != J0 {EXPECTED_J0_INIT_HASH}')

    E_k = None
    K0 = None
    if cli.arm == 'J2_true_frozen_key':
        E_k = copy.deepcopy(model).to(device)
        ek_init_hash = state_hash(E_k)
        assert ek_init_hash == EXPECTED_J0_INIT_HASH, '[ISSUE][ABORT] E_k init hash mismatch'
        for p in E_k.parameters():
            p.requires_grad_(False)
        E_k.eval()
        with torch.no_grad():
            K0 = {c: encode_raw(E_k, exp.memory_x, c).clone() for c in channels}

    for p in model.parameters():
        p.requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cli.learning_rate)

    train_gen = make_loader_generator(cli.loader_seed)
    _, train_loader = exp._get_data(flag='train', shuffle=True, generator=train_gen)
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)

    probe = build_probe_set(exp, val_loader, cli.n_probe, device)
    fixed = precompute_probe_fixed(exp, args, probe, channels, device, cli.chunk_size)
    probe_ids_path = Path(cli.out_dir) / cli.cell / 'probe_query_ids.json'
    probe_ids_path.parent.mkdir(parents=True, exist_ok=True)
    if not probe_ids_path.exists():
        probe_ids_path.write_text(json.dumps({'start_idx': probe['start_idx'].tolist(), 'n_probe': probe['n']}, indent=2))

    geom_subset_idx = torch.linspace(0, exp.memory_x.size(0) - 1, 500).round().long().unique()

    config = {'cell': cli.cell, 'arm': cli.arm, 'relation_encoder_type': 'mlp', 'relation_self_fill': 'linear',
              'tau_t': cli.tau_t, 'tau_s': cli.tau_s, 'top_k': cli.top_k, 'batch_size': cli.batch_size,
              'learning_rate': cli.learning_rate, 'train_epochs': cli.train_epochs, 'patience': cli.patience,
              'init_seed': cli.init_seed, 'loader_seed': cli.loader_seed, 'n_probe': probe['n'],
              'n_candidates': int(exp.memory_x.size(0)), 'checkpoint_criterion': 'min val model_top10_individual_mse'}
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2))
    print(f'[track_j2] arm={cli.arm} N={config["n_candidates"]} n_probe={probe["n"]} init_hash={init_hash[:16]}')

    if cli.smoke_test:
        _smoke(exp, args, model, E_k, K0, probe, channels, optimizer, cli, train_loader, device)
        return

    def compute_loss_and_step(batch_x, batch_y, batch_start_idx, backward=True):
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        if backward:
            optimizer.zero_grad()
        batch_loss = 0.0
        for c in channels:
            z_q, z_k_raw = score_fn(cli.arm, model, E_k, K0, batch_x, c, exp, device)
            z_k = z_k_raw.detach() if cli.arm == 'J1_stopgrad_key' else z_k_raw
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            s = arm_score(z_q, z_k, None)
            with torch.no_grad():
                u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
                p_t = normalized_teacher_prob(-u, cand_mask, cli.tau_t)
            l = kl_loss(p_t, s, cand_mask, cli.tau_s)
            if backward:
                (l / len(channels)).backward()
            batch_loss += float(l.detach()) / len(channels)
        if backward:
            optimizer.step()
        return batch_loss

    def run_diagnostic(step, epoch):
        was_training = model.training
        model.eval()
        rows_fw, rows_geom, rows_churn, rows_disp = [], [], [], []
        for c in channels:
            with torch.no_grad():
                z_q = encode_raw(model, probe['x'], c)
                if cli.arm == 'J1_stopgrad_key':
                    z_k = encode_raw(model, exp.memory_x, c)
                else:
                    z_k = K0[c]
                s = arm_score(z_q, z_k, None)
                m, idx = variant_metrics(s, probe['cand_mask'], fixed[c], cli.top_k)
                rows_fw.append({'step': step, 'epoch': epoch, 'channel': c, **m})

                ch = churn(idx, last_idx.get(c), cli.top_k)
                ret0 = retention_vs_init(idx, idx0[c], cli.top_k) if c in idx0 else 1.0
                if step == 0:
                    idx0[c] = idx
                last_idx[c] = idx
                rows_churn.append({'step': step, 'epoch': epoch, 'channel': c, 'churn_vs_prev': ch,
                                   'retention_vs_init': ret0})

                if c not in zq0_cache:
                    zq0_cache[c] = encode_raw(E0, probe['x'], c)
                    zk0_cache[c] = encode_raw(E0, exp.memory_x, c)
                q_disp = 1.0 - (F.normalize(zq0_cache[c], dim=-1) * F.normalize(z_q, dim=-1)).sum(-1)
                if cli.arm == 'J2_true_frozen_key':
                    k_disp = torch.zeros(exp.memory_x.size(0), device=device)
                else:
                    k_disp = 1.0 - (F.normalize(zk0_cache[c], dim=-1) * F.normalize(z_k, dim=-1)).sum(-1)
                rows_disp.append({'step': step, 'epoch': epoch, 'channel': c, 'side': 'query',
                                  **displacement_stats(q_disp)})
                rows_disp.append({'step': step, 'epoch': epoch, 'channel': c, 'side': 'candidate',
                                  **displacement_stats(k_disp)})
                if cli.arm == 'J2_true_frozen_key':
                    assert float(k_disp.max()) == 0.0, '[ISSUE][ABORT] J2 candidate embeddings moved'

                sub = z_k[geom_subset_idx] if z_k.size(0) == exp.memory_x.size(0) else z_k[
                    geom_subset_idx[geom_subset_idx < z_k.size(0)]]
                g = geometry_metrics(sub)
                rows_geom.append({'step': step, 'epoch': epoch, 'channel': c, **g})
        if was_training:
            model.train()
        return rows_fw, rows_geom, rows_churn, rows_disp

    E0 = copy.deepcopy(model).to(device)
    for p in E0.parameters():
        p.requires_grad_(False)
    E0.eval()
    e0_hash = state_hash(E0)

    last_idx, idx0, zq0_cache, zk0_cache = {}, {}, {}, {}
    fw_rows, geom_rows, churn_rows, disp_rows, step_rows = [], [], [], [], []

    f0, g0, c0, d0 = run_diagnostic(step=0, epoch=0)
    fw_rows += f0
    geom_rows += g0
    churn_rows += c0
    disp_rows += d0
    print(f'[track_j2] {cli.arm} step0 diagnostic recorded')

    n_epoch1_steps = len(train_loader) if not cli.limit_batches else min(len(train_loader), cli.limit_batches)
    global_step = 0
    epoch_val_rows = []
    best = {'val': float('inf'), 'epoch': -1}
    batch_order_hashes = {}
    t0 = time.time()
    for epoch in range(1, cli.train_epochs + 1):
        model.train()
        if cli.arm == 'J2_true_frozen_key':
            E_k.eval()
        starts = []
        for bi, (batch_x, batch_y, batch_start_idx) in enumerate(train_loader):
            if cli.limit_batches and bi >= cli.limit_batches:
                break
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            starts.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx) else torch.as_tensor(batch_start_idx))
            batch_loss = compute_loss_and_step(batch_x, batch_y, batch_start_idx, backward=True)

            if cli.arm == 'J2_true_frozen_key':
                for c in channels:
                    assert torch.equal(K0[c], encode_raw(E_k, exp.memory_x, c)), \
                        '[ISSUE][ABORT] J2 key bank changed after optimizer step'

            global_step += 1
            step_rows.append({'global_step': global_step, 'epoch': epoch, 'batch_in_epoch': bi, 'train_kl': batch_loss})

            do_diag = (epoch == 1 and global_step % cli.diag_every_epoch1 == 0) or \
                     (epoch > 1 and global_step % cli.diag_every_later == 0)
            if do_diag:
                f, g, ch, d = run_diagnostic(step=global_step, epoch=epoch)
                fw_rows += f
                geom_rows += g
                churn_rows += ch
                disp_rows += d
                print(f'[track_j2] {cli.arm} diag step={global_step} epoch={epoch} train_kl={batch_loss:.4f}')
        batch_order_hashes[f'epoch{epoch}'] = batch_order_sha256(starts)

        f, g, ch, d = run_diagnostic(step=global_step, epoch=epoch)
        fw_rows += f
        geom_rows += g
        churn_rows += ch
        disp_rows += d

        model.eval()
        val_sums, n = {}, 0
        with torch.no_grad():
            for batch_x, batch_y, batch_start_idx in val_loader:
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                cand_mask, _ = exp._candidate_mask(batch_start_idx)
                bsz = batch_x.size(0)
                per_ch = {}
                for c in channels:
                    z_q, z_k = score_fn(cli.arm, model, E_k, K0, batch_x, c, exp, device)
                    memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
                    query_future = batch_y[:, :, c]
                    u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
                    d_ = -u
                    s = arm_score(z_q, z_k, None).masked_fill(~cand_mask, float('-inf'))
                    model_idx = stable_topk_indices(s, cli.top_k, largest=True)
                    model_ind_mse = d_.gather(1, model_idx).mean(-1)
                    oracle_idx = stable_topk_indices(d_.masked_fill(~cand_mask, float('inf')), cli.top_k, largest=False)
                    recall10 = recall_at_k(model_idx, oracle_idx, cli.top_k)
                    ndcg10 = ndcg_at_k(model_idx, d_, cand_mask, cli.top_k)
                    oracle_ind_mse = d_.gather(1, oracle_idx).mean(-1)
                    y_sel = memory_c[model_idx] + offset_c.view(-1, 1, 1)
                    uniform_agg = ((y_sel.mean(dim=1) - query_future) ** 2).mean(-1)
                    per_ch.setdefault('model_top10_individual_mse', []).append(model_ind_mse.cpu())
                    per_ch.setdefault('recall_at_10', []).append(recall10.cpu())
                    per_ch.setdefault('ndcg_at_10', []).append(ndcg10.cpu())
                    per_ch.setdefault('oracle_regret', []).append((model_ind_mse - oracle_ind_mse).cpu())
                    per_ch.setdefault('uniform_agg_mse10', []).append(uniform_agg.cpu())
                for k_, vals in per_ch.items():
                    val_sums[k_] = val_sums.get(k_, 0.0) + torch.cat(vals).sum().item()
                n += bsz
        val_metrics = {k_: v / max(n * len(channels), 1) for k_, v in val_sums.items()}
        val_metric = val_metrics['model_top10_individual_mse']
        epoch_val_rows.append({'epoch': epoch, **val_metrics})
        payload = {'model_state_dict': model.state_dict(), 'epoch': epoch, 'val_metric': val_metric, 'config': config}
        torch.save(payload, ckpt_dir / f'checkpoint_epoch{epoch}.pth')
        if val_metric < best['val']:
            best = {'val': val_metric, 'epoch': epoch}
            torch.save(payload, ckpt_dir / 'checkpoint.pth')
        print(f'[track_j2] {cli.arm} epoch={epoch} done val_retMSE@10={val_metric:.6f} '
             f'(best={best["epoch"]}:{best["val"]:.6f})')
        if epoch - best['epoch'] >= cli.patience:
            print(f'[track_j2] {cli.arm} early stop at epoch {epoch} (best={best["epoch"]})')
            break

    assert state_hash(E0) == e0_hash, '[ISSUE][ABORT] E0 drifted'
    if cli.arm == 'J2_true_frozen_key':
        ek_final_hash = state_hash(E_k)
        assert ek_final_hash == EXPECTED_J0_INIT_HASH, '[ISSUE][ABORT] E_k drifted during training'

    bl = torch.load(ckpt_dir / 'checkpoint.pth', map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    model.eval()

    test_probe = build_probe_set(exp, test_loader, cli.n_probe, device)
    test_fixed = precompute_probe_fixed(exp, args, test_probe, channels, device, cli.chunk_size)
    test_fw = []
    with torch.no_grad():
        for c in channels:
            z_q = encode_raw(model, test_probe['x'], c)
            z_k = encode_raw(model, exp.memory_x, c) if cli.arm == 'J1_stopgrad_key' else K0[c]
            s = arm_score(z_q, z_k, None)
            m, _ = variant_metrics(s, test_probe['cand_mask'], test_fixed[c], cli.top_k)
            test_fw.append({'channel': c, **m})

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
    write_csv(out_dir / 'val_metrics.csv', epoch_val_rows)
    write_csv(out_dir / 'train_metrics.csv', step_rows)
    write_csv(out_dir / 'geometry_metrics.csv', geom_rows)
    write_csv(out_dir / 'topk_churn.csv', churn_rows)
    write_csv(out_dir / 'displacement_metrics.csv', disp_rows)
    write_csv(out_dir / 'probe_fourway_metrics.csv', fw_rows)
    (out_dir / 'final_test_metrics.json').write_text(json.dumps(
        {'per_channel': test_fw, 'best_epoch': best['epoch']}, indent=2))
    (out_dir / 'checkpoint_fingerprint.json').write_text(json.dumps(
        {'init_hash': init_hash, 'e0_hash_final': state_hash(E0), 'best_epoch': best['epoch'],
         'final_model_hash': state_hash(model), 'batch_order_hashes': batch_order_hashes}, indent=2))
    if cli.arm == 'J2_true_frozen_key':
        (out_dir / 'key_cache_fingerprint.json').write_text(json.dumps(
            {'ek_init_hash': ek_init_hash, 'ek_final_hash': ek_final_hash,
             'k0_norms_mean': {int(c): float(K0[c].norm(dim=-1).mean()) for c in channels}}, indent=2))
    import numpy as np
    print(f'[track_j2] done. {cli.arm} best_epoch={best["epoch"]} '
         f'test_retMSE@10={np.mean([r["retmse10"] for r in test_fw]):.6f} '
         f'wall_seconds={time.time()-t0:.1f}')


def _smoke(exp, args, model, E_k, K0, probe, channels, optimizer, cli, train_loader, device):
    model.train()
    for bi, (batch_x, batch_y, batch_start_idx) in enumerate(train_loader):
        if bi >= 2:
            break
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        optimizer.zero_grad()
        for c in channels:
            z_q, z_k_raw = score_fn(cli.arm, model, E_k, K0, batch_x, c, exp, device)
            z_k = z_k_raw.detach() if cli.arm == 'J1_stopgrad_key' else z_k_raw
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            s = arm_score(z_q, z_k, None)
            with torch.no_grad():
                u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
                p_t = normalized_teacher_prob(-u, cand_mask, cli.tau_t)
            l = kl_loss(p_t, s, cand_mask, cli.tau_s)
            (l / len(channels)).backward()
        optimizer.step()
    print(f'[track_j2] {cli.arm} SMOKE PASS')


if __name__ == '__main__':
    main()
