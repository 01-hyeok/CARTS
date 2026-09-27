#!/usr/bin/env python3
"""TRACK-C-HORIZON-RETRIEVAL-CLEAN03 -- frozen-trunk adapter-only isolation
of Clean02's H-Symmetric failure.

Loads Clean02's G-Transformer validation-best checkpoint, freezes it
completely (never `.train()`, `requires_grad_(False)` on every parameter,
never touched by the optimizer), and trains ONLY the (small) HorizonAdapter
against a candidate teacher temperature (arm a1=0.10 / a2=0.02 / a3=0.01,
one process per arm). A0 is the frozen trunk's own Global retrieval,
re-evaluated here (not copied from Clean02's report) to confirm it
reproduces val=1.810685.

Reuses, UNMODIFIED: `HorizonAdapter`/`BLOCKS`/`block_distance` from
`scripts.train_c_horizon_clean02`; `kl_loss`/`normalized_teacher_prob`/
`BLOCK_NAMES` from `scripts.train_horizon_retrieval_expert01`;
`arm_score`/`encode_raw` from `scripts.train_factorial_e2e01`;
`build_experiment` from `scripts.train_margutil01`; `stable_topk_indices`
from `models.RelationStage1`. No new distance/aggregation math.

Candidate-side value reconstruction: `models.RelationStage1.RelationEncoder`'s
sibling helper `memory_value()` (`scripts.train_margutil01`) shows that the
delta-space candidate future (`memory_c`) is a per-channel BANK-LEVEL
constant (`memory_y[...,c]` minus `memory_x_last[...,c]`, both exp-level
fixed tensors) -- it never depends on the query at all. Only the per-query
`offset_c = batch_x[:, -1, c]` (the query's own last observed value, added
back once at absolute-reconstruction time) is query-specific. So caching
`x_last[:, c]` per query (a scalar) alongside the frozen embeddings is
sufficient; the full `batch_x` is never needed again after the one-time
embedding cache pass.
"""
import argparse
import csv
import hashlib
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.train_c_horizon_clean02 import BLOCKS, HorizonAdapter, block_distance
from scripts.train_factorial_e2e01 import arm_score, encode_raw
from scripts.train_horizon_retrieval_expert01 import BLOCK_NAMES, kl_loss, normalized_teacher_prob
from scripts.train_margutil01 import build_experiment

TAU_BY_ARM = {'a1_tau010': 0.10, 'a2_tau002': 0.02, 'a3_tau001': 0.01}


def state_sha(state_dict):
    h = hashlib.sha256()
    for key in sorted(state_dict):
        h.update(key.encode())
        h.update(state_dict[key].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def build_query_cache(model, exp, split, channels, device):
    """One live-encode pass (shuffle=False) per split -- valid for the
    whole run since the trunk is frozen. Stores per-channel embeddings,
    the query's own last-observed value per channel (`x_last`), the future
    `y`, and a start_idx -> row lookup for O(1) gather during shuffled
    training batches."""
    _, loader = exp._get_data(flag=split, shuffle=False)
    all_starts, ys, x_lasts, emb = [], [], [], {c: [] for c in channels}
    for bx, by, bstart in loader:
        bx = bx.float().to(device)
        all_starts.append(bstart.clone() if torch.is_tensor(bstart) else torch.as_tensor(bstart))
        ys.append(by)
        x_lasts.append(bx[:, -1, :].detach().cpu())
        for c in channels:
            emb[c].append(encode_raw(model, bx, c).detach().cpu())
    starts = torch.cat(all_starts)
    start_to_row = {int(s): i for i, s in enumerate(starts.tolist())}
    return {'emb': {c: torch.cat(v) for c, v in emb.items()}, 'y': torch.cat(ys),
           'x_last': torch.cat(x_lasts), 'starts': starts, 'start_to_row': start_to_row}


def rows_for_batch(cache, batch_start_idx):
    starts_np = batch_start_idx.numpy() if torch.is_tensor(batch_start_idx) else np.asarray(batch_start_idx)
    return torch.tensor([cache['start_to_row'][int(s)] for s in starts_np], dtype=torch.long)


def candidate_value(memory_y, memory_x_last, c):
    """memory_c: bank-level constant delta future for channel c (never
    depends on the query) -- exactly `train_margutil01.memory_value`'s
    delta-space math, with the query-independent half factored out."""
    return memory_y[:, :, c] - memory_x_last[:, c].unsqueeze(-1)


@torch.no_grad()
def eval_split(cache, candidate_emb, memory_c_by_channel, channels, adapter, top_k,
               batch_size, exp, device, limit_rows=0):
    starts = cache['starts']
    n_rows = starts.size(0) if not limit_rows else min(limit_rows, starts.size(0))
    sums = {}
    for s in range(0, n_rows, batch_size):
        e = min(s + batch_size, n_rows)
        bstart = starts[s:e]
        by = cache['y'][s:e].float().to(device)
        x_last = cache['x_last'][s:e].to(device)
        cand_mask, _ = exp._candidate_mask(bstart)
        for c in channels:
            h_q = cache['emb'][c][s:e].to(device)
            h_i = candidate_emb[c]
            z_q_g = F.normalize(h_q, dim=-1)
            z_i_g = F.normalize(h_i, dim=-1)
            memory_c = memory_c_by_channel[c]
            offset_c = x_last[:, c]
            query_future = by[:, :, c]
            neg_inf = float('-inf')
            s_g = arm_score(z_q_g, z_i_g, None).masked_fill(~cand_mask, neg_inf)
            picks_g = stable_topk_indices(s_g, top_k, largest=True)
            y_sel_g = memory_c[picks_g] + offset_c.view(-1, 1, 1)
            sums.setdefault('global_h720_mse', []).append(
                ((y_sel_g.mean(dim=1) - query_future) ** 2).mean(dim=-1).cpu())
            if adapter is not None:
                concat = torch.zeros_like(query_future)
                block_picks = {}
                for bidx, bname in enumerate(BLOCK_NAMES):
                    lo, hi = BLOCKS[bname]
                    z_q_b = adapter(h_q, bidx)
                    z_i_b = adapter(h_i, bidx)
                    s_b = arm_score(z_q_b, z_i_b, None).masked_fill(~cand_mask, neg_inf)
                    picks_b = stable_topk_indices(s_b, top_k, largest=True)
                    block_picks[bname] = picks_b
                    y_sel_b = memory_c[picks_b][:, :, lo:hi] + offset_c.view(-1, 1, 1)
                    agg_b = y_sel_b.mean(dim=1)
                    sums.setdefault(f'{bname}_mse', []).append(
                        ((agg_b - query_future[:, lo:hi]) ** 2).mean(dim=-1).cpu())
                    concat[:, lo:hi] = agg_b
                sums.setdefault('block_h720_mse', []).append(
                    ((concat - query_future) ** 2).mean(dim=-1).cpu())
                for bname in BLOCK_NAMES:
                    inter = (block_picks[bname].unsqueeze(-1) == picks_g.unsqueeze(-2)).any(-1).float().sum(-1)
                    union = float(top_k) * 2 - inter
                    sums.setdefault(f'global_vs_{bname}_jaccard', []).append(
                        (inter / union.clamp_min(1e-9)).cpu())
                for a, b in itertools.combinations(BLOCK_NAMES, 2):
                    inter = (block_picks[a].unsqueeze(-1) == block_picks[b].unsqueeze(-2)).any(-1).float().sum(-1)
                    union = float(top_k) * 2 - inter
                    sums.setdefault(f'{a}_vs_{b}_jaccard', []).append((inter / union.clamp_min(1e-9)).cpu())
    return {k: float(torch.cat(v).mean()) for k, v in sums.items()}


@torch.no_grad()
def per_query_val_mse(cache, candidate_emb, memory_c_by_channel, channels, adapter, top_k,
                      batch_size, exp, device, use_block):
    """Per (start_idx) row -> mean-over-channel MSE (global if adapter is
    None, else blockwise-concat) -- for the paired bootstrap."""
    starts = cache['starts']
    n_rows = starts.size(0)
    rows_out = []
    for s in range(0, n_rows, batch_size):
        e = min(s + batch_size, n_rows)
        bstart = starts[s:e]
        by = cache['y'][s:e].float().to(device)
        x_last = cache['x_last'][s:e].to(device)
        cand_mask, _ = exp._candidate_mask(bstart)
        bsz = e - s
        mse_ch = torch.zeros(bsz, len(channels))
        for ci, c in enumerate(channels):
            h_q = cache['emb'][c][s:e].to(device)
            h_i = candidate_emb[c]
            z_q_g = F.normalize(h_q, dim=-1)
            z_i_g = F.normalize(h_i, dim=-1)
            memory_c = memory_c_by_channel[c]
            offset_c = x_last[:, c]
            query_future = by[:, :, c]
            neg_inf = float('-inf')
            s_g = arm_score(z_q_g, z_i_g, None).masked_fill(~cand_mask, neg_inf)
            if not use_block:
                picks_g = stable_topk_indices(s_g, top_k, largest=True)
                y_sel = memory_c[picks_g] + offset_c.view(-1, 1, 1)
                mse_ch[:, ci] = ((y_sel.mean(dim=1) - query_future) ** 2).mean(dim=-1).cpu()
            else:
                concat = torch.zeros_like(query_future)
                for bidx, bname in enumerate(BLOCK_NAMES):
                    lo, hi = BLOCKS[bname]
                    z_q_b = adapter(h_q, bidx)
                    z_i_b = adapter(h_i, bidx)
                    s_b = arm_score(z_q_b, z_i_b, None).masked_fill(~cand_mask, neg_inf)
                    picks_b = stable_topk_indices(s_b, top_k, largest=True)
                    y_sel_b = memory_c[picks_b][:, :, lo:hi] + offset_c.view(-1, 1, 1)
                    concat[:, lo:hi] = y_sel_b.mean(dim=1)
                mse_ch[:, ci] = ((concat - query_future) ** 2).mean(dim=-1).cpu()
        rows_out.append(mse_ch.mean(dim=-1))
    per_row = torch.cat(rows_out)
    return {int(s): float(m) for s, m in zip(starts.tolist(), per_row.tolist())}


def cluster_bootstrap_paired(a0_by_start, arm_by_start, n_reps=2000, seed=0):
    starts = sorted(a0_by_start.keys())
    delta = np.array([a0_by_start[s] - arm_by_start[s] for s in starts])  # positive = arm better
    rng = np.random.RandomState(seed)
    n = len(starts)
    boot = np.empty(n_reps)
    for i in range(n_reps):
        idx = rng.randint(0, n, size=n)
        boot[i] = delta[idx].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {'mean_delta': float(delta.mean()), 'median_delta': float(np.median(delta)),
           'ci_low': float(lo), 'ci_high': float(hi), 'ci_excludes_zero_positive': bool(lo > 0),
           'frac_improved': float((delta > 0).mean()), 'frac_improved_1pct': float((delta > 0.01 * np.abs(a0_arr := np.array([a0_by_start[s] for s in starts]))).mean()),
           'frac_improved_5pct': float((delta > 0.05 * a0_arr).mean()),
           'frac_improved_10pct': float((delta > 0.10 * a0_arr).mean()),
           'n_base_windows': n, 'n_reps': n_reps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True)
    ap.add_argument('--g_best_checkpoint', required=True)
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--arm', required=True, choices=('a0', *TAU_BY_ARM.keys()))
    ap.add_argument('--pred_len', type=int, default=720)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--chunk_size', type=int, default=2048)
    ap.add_argument('--tau_s', type=float, default=0.10)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--train_epochs', type=int, default=20)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--init_seed', type=int, default=0)
    ap.add_argument('--out_dir', default='results/TRACK-C-HORIZON-RETRIEVAL-CLEAN03')
    ap.add_argument('--checkpoints', default='checkpoints/track_c_horizon_retrieval_clean03')
    ap.add_argument('--smoke_test', action='store_true')
    ap.add_argument('--limit_batches', type=int, default=0)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cell_dir = Path(cli.out_dir) / cli.cell
    cell_dir.mkdir(parents=True, exist_ok=True)

    g_ckpt = torch.load(cli.g_best_checkpoint, map_location='cpu')
    fp = g_ckpt['fingerprint']
    problems = []
    if fp.get('arm') != 'g_transformer':
        problems.append(f"arm={fp.get('arm')} != g_transformer")
    if fp.get('relation_encoder_type') != 'transformer':
        problems.append('relation_encoder_type != transformer')
    if int(fp.get('patch_len', -1)) != 16 or int(fp.get('stride', -1)) != 16:
        problems.append('patch_len/stride != 16')
    if int(fp.get('d_model', -1)) != 128:
        problems.append('d_model != 128')
    if problems:
        raise SystemExit(f'[ISSUE][ABORT] G-best checkpoint fingerprint mismatch: {problems}')

    torch.manual_seed(cli.init_seed)
    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.pred_len, 'batch_size': cli.batch_size,
        'seed': cli.init_seed, 'top_k': cli.top_k, 'tau_topk': 0.1,
        'patch_len': 16, 'stride': 16,
        'relation_encoder_type': 'transformer', 'relation_self_fill': 'zero',
    })
    exp._ensure_memory()
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.load_state_dict(g_ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    channels = list(range(int(args.enc_in)))
    d_model = int(args.d_model)

    frozen_sha_before = state_sha(model.state_dict())
    g_ckpt_hash = hashlib.sha256(Path(cli.g_best_checkpoint).read_bytes()).hexdigest()
    print(f'[frozen03] {cli.cell}/{cli.arm} loaded G-best (epoch={g_ckpt["epoch"]}, '
         f'val_primary_mse={g_ckpt["val_primary_mse"]:.6f}) frozen_sha_before={frozen_sha_before[:16]}')

    t0 = time.time()
    candidate_emb = {c: encode_raw(model, exp.memory_x, c).detach() for c in channels}
    train_cache = build_query_cache(model, exp, 'train', channels, device)
    val_cache = build_query_cache(model, exp, 'val', channels, device)
    test_cache = build_query_cache(model, exp, 'test', channels, device)
    print(f'[frozen03] embedding cache built in {time.time()-t0:.1f}s')

    frozen_sha_after_cache = state_sha(model.state_dict())
    assert frozen_sha_after_cache == frozen_sha_before, '[ISSUE][ABORT] trunk changed during cache build'

    memory_c_by_channel = {c: candidate_value(exp.memory_y, exp.memory_x_last, c).to(device) for c in channels}

    # ---- cache-vs-live verification ----
    _, val_loader_live = exp._get_data(flag='val', shuffle=False)
    bx0, by0, bst0 = next(iter(val_loader_live))
    bx0 = bx0.float().to(device)
    c0 = channels[0]
    live_emb = encode_raw(model, bx0, c0).detach().cpu()
    rows0 = rows_for_batch(val_cache, bst0)
    cached_emb = val_cache['emb'][c0][rows0]
    emb_max_err = float((live_emb - cached_emb).abs().max())
    live_score = arm_score(F.normalize(live_emb.to(device), dim=-1), F.normalize(candidate_emb[c0], dim=-1), None).cpu()
    cached_score = arm_score(F.normalize(cached_emb.to(device), dim=-1), F.normalize(candidate_emb[c0], dim=-1), None).cpu()
    score_max_err = float((live_score - cached_score).abs().max())
    print(f'[frozen03] cache-vs-live: embedding_max_abs_error={emb_max_err:.2e} score_max_abs_error={score_max_err:.2e}')
    assert emb_max_err <= 1e-6 and score_max_err <= 1e-6, '[ISSUE][ABORT] cache/live mismatch'

    if cli.arm == 'a0':
        va = eval_split(val_cache, candidate_emb, memory_c_by_channel, channels, None, cli.top_k,
                        cli.batch_size, exp, device)
        te = eval_split(test_cache, candidate_emb, memory_c_by_channel, channels, None, cli.top_k,
                        cli.batch_size, exp, device)
        result = {'cell': cli.cell, 'arm': 'a0', 'val_global_h720_mse': va['global_h720_mse'],
                 'test_global_h720_mse': te['global_h720_mse'],
                 'reproduces_clean02_val': abs(va['global_h720_mse'] - g_ckpt['val_primary_mse']) < 1e-4,
                 'clean02_val_primary_mse': g_ckpt['val_primary_mse'],
                 'frozen_trunk_sha256_before': frozen_sha_before,
                 'frozen_trunk_sha256_after': frozen_sha_after_cache,
                 'g_best_checkpoint_hash': g_ckpt_hash}
        (cell_dir / 'frozen_global_baseline.json').write_text(json.dumps(result, indent=2))
        print(f'[frozen03] A0: val={va["global_h720_mse"]:.6f} test={te["global_h720_mse"]:.6f} '
             f'reproduces_clean02={result["reproduces_clean02_val"]}')
        return

    tau_t_block = TAU_BY_ARM[cli.arm]
    adapter = HorizonAdapter(d_model).to(device)
    adapter_init_sha = state_sha(adapter.state_dict())
    for p in model.parameters():
        assert not p.requires_grad
    optimizer = torch.optim.Adam(adapter.parameters(), lr=cli.learning_rate)

    ckpt_dir = Path(cli.checkpoints) / cli.cell / cli.arm
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if cli.smoke_test:
        # zero-init equality (before any step)
        with torch.no_grad():
            h_q = train_cache['emb'][channels[0]][:4].to(device)
            h_i = candidate_emb[channels[0]][:50]
            s_g = arm_score(F.normalize(h_q, dim=-1), F.normalize(h_i, dim=-1), None)
            zero_ok = True
            for bidx in range(3):
                s_b = arm_score(adapter(h_q, bidx), adapter(h_i, bidx), None)
                if not torch.allclose(s_b, s_g, atol=1e-4):
                    zero_ok = False
        print(f'[frozen03] smoke zero-init check: {"PASS" if zero_ok else "FAIL"}')
        assert zero_ok

    n_rows = train_cache['starts'].size(0)
    perm_gen = torch.Generator().manual_seed(cli.init_seed)
    epoch_rows, best = [], {'val': float('inf'), 'epoch': -1}
    limit_batches = cli.limit_batches or (3 if cli.smoke_test else 0)
    for epoch in range(1, (cli.train_epochs if not cli.smoke_test else 1) + 1):
        adapter.train(True)
        perm = torch.randperm(n_rows, generator=perm_gen)
        tot_loss, nb = 0.0, 0
        for bi, s in enumerate(range(0, n_rows, cli.batch_size)):
            if limit_batches and bi >= limit_batches:
                break
            rows = perm[s:s + cli.batch_size]
            bstart = train_cache['starts'][rows]
            by = train_cache['y'][rows].float().to(device)
            x_last = train_cache['x_last'][rows].to(device)
            cand_mask, _ = exp._candidate_mask(bstart)
            optimizer.zero_grad()
            batch_loss = 0.0
            for c in channels:
                h_q = train_cache['emb'][c][rows].to(device)
                h_i = candidate_emb[c]
                memory_c = memory_c_by_channel[c]
                offset_c = x_last[:, c]
                query_future = by[:, :, c]
                block_losses = []
                for bidx, bname in enumerate(BLOCK_NAMES):
                    lo, hi = BLOCKS[bname]
                    z_q_b = adapter(h_q, bidx)
                    z_i_b = adapter(h_i, bidx)
                    s_b = arm_score(z_q_b, z_i_b, None)
                    with torch.no_grad():
                        d_b = block_distance(memory_c, offset_c, query_future, lo, hi, cli.chunk_size)
                        d_b = d_b.masked_fill(~cand_mask, float('inf'))
                        p_t_b = normalized_teacher_prob(d_b, cand_mask, tau_t_block)
                        assert not p_t_b.requires_grad
                    l_b = kl_loss(p_t_b, s_b, cand_mask, cli.tau_s)
                    block_losses.append(l_b)
                ch_loss = sum(block_losses) / 3.0
                (ch_loss / len(channels)).backward()
                batch_loss += float(ch_loss.detach()) / len(channels)
            has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.parameters())
            trunk_untouched = all(p.grad is None for p in model.parameters())
            if cli.smoke_test:
                assert has_grad, '[smoke] adapter has no gradient'
                assert trunk_untouched, '[smoke] trunk received gradient'
            optimizer.step()
            tot_loss += batch_loss
            nb += 1

        va = eval_split(val_cache, candidate_emb, memory_c_by_channel, channels, adapter, cli.top_k,
                        cli.batch_size, exp, device, limit_rows=(64 if cli.smoke_test else 0))
        val_metric = va['block_h720_mse']
        row = {'epoch': epoch, 'train_loss': tot_loss / max(nb, 1), **{f'val_{k}': v for k, v in va.items()}}
        epoch_rows.append(row)
        payload = {'adapter_state_dict': adapter.state_dict(), 'g_best_checkpoint': cli.g_best_checkpoint,
                  'g_best_checkpoint_hash': g_ckpt_hash, 'frozen_trunk_sha256': frozen_sha_before,
                  'adapter_init_sha256': adapter_init_sha, 'tau_t': tau_t_block, 'epoch': epoch,
                  'val_block_h720_mse': val_metric}
        if val_metric < best['val']:
            best = {'val': val_metric, 'epoch': epoch}
            torch.save(payload, ckpt_dir / 'checkpoint.pth')
        print(f'[frozen03] {cli.cell}/{cli.arm} epoch {epoch} train_loss={row["train_loss"]:.5f} '
             f'val_block_h720_mse={val_metric:.6f} val_global_h720_mse={va["global_h720_mse"]:.6f}')
        if cli.smoke_test:
            break
        if epoch - best['epoch'] >= cli.patience:
            print(f'[frozen03] early stop at epoch {epoch} (best={best["epoch"]})')
            break

    frozen_sha_final = state_sha(model.state_dict())
    assert frozen_sha_final == frozen_sha_before, '[ISSUE][ABORT] trunk drifted during training'

    if cli.smoke_test:
        report = {'cell': cli.cell, 'arm': cli.arm, 'zero_init_check': zero_ok,
                  'frozen_trunk_unchanged': frozen_sha_final == frozen_sha_before,
                  'embedding_cache_max_abs_error': emb_max_err, 'score_cache_max_abs_error': score_max_err,
                  'smoke_epoch_row': epoch_rows[-1]}
        smoke_dir = Path(cli.out_dir) / 'smoke' / cli.cell
        smoke_dir.mkdir(parents=True, exist_ok=True)
        (smoke_dir / f'smoke_report_{cli.arm}.json').write_text(json.dumps(report, indent=2, default=str))
        print(f'[frozen03] {cli.cell}/{cli.arm} SMOKE TEST: PASS')
        return

    with open(cell_dir / f'epoch_metrics_{cli.arm}.csv', 'w', newline='') as fh:
        keys = sorted({k for r in epoch_rows for k in r})
        w = csv.DictWriter(fh, fieldnames=['epoch'] + [k for k in keys if k != 'epoch'])
        w.writeheader()
        for r in epoch_rows:
            w.writerow(r)

    bl = torch.load(ckpt_dir / 'checkpoint.pth', map_location=device)
    adapter.load_state_dict(bl['adapter_state_dict'])
    va_best = eval_split(val_cache, candidate_emb, memory_c_by_channel, channels, adapter, cli.top_k,
                         cli.batch_size, exp, device)

    a0_val = per_query_val_mse(val_cache, candidate_emb, memory_c_by_channel, channels, None, cli.top_k,
                               cli.batch_size, exp, device, use_block=False)
    arm_val = per_query_val_mse(val_cache, candidate_emb, memory_c_by_channel, channels, adapter, cli.top_k,
                                cli.batch_size, exp, device, use_block=True)
    boot = cluster_bootstrap_paired(a0_val, arm_val, n_reps=2000, seed=0)

    summary = {
        'exp': 'TRACK-C-HORIZON-RETRIEVAL-CLEAN03', 'cell': cli.cell, 'arm': cli.arm,
        'tau_t': tau_t_block, 'best_epoch': best['epoch'], 'best_val_block_h720_mse': best['val'],
        'val_metrics_at_best': va_best, 'paired_bootstrap_vs_a0': boot,
        'adapter_init_sha256': adapter_init_sha, 'frozen_trunk_sha256': frozen_sha_before,
        'g_best_checkpoint_hash': g_ckpt_hash, 'wall_clock_seconds': time.time() - t0,
    }
    (cell_dir / f'retrieval_metrics_{cli.arm}.json').write_text(json.dumps(summary, indent=2))
    print(f'[frozen03] done. {cli.cell}/{cli.arm} best_epoch={best["epoch"]} '
         f'val_block_h720_mse={best["val"]:.6f} val_global_h720_mse={va_best["global_h720_mse"]:.6f} '
         f'bootstrap_mean_delta={boot["mean_delta"]:.6f} ci=[{boot["ci_low"]:.6f},{boot["ci_high"]:.6f}]')


if __name__ == '__main__':
    main()
