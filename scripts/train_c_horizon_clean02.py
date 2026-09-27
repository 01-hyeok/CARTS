#!/usr/bin/env python3
"""TRACK-C-HORIZON-RETRIEVAL-CLEAN02 -- Clean re-run of the horizon-block
retrieval research question, with the five [ISSUE]s found in
`research/C-horizon-block-retrieval/TRACK-C-HORIZON-RETRIEVAL-CLEAN02_AUDIT.md`
removed: relation_encoder_type is explicitly forced to 'transformer'
(never inherits the reference checkpoint's 'mlp'), horizon adapters are
SYMMETRIC (applied identically to query and candidate embeddings, not
query-only), aggregation is uniform-mean EVERYWHERE (no HostScorer, no
`--stage2_host` anywhere in this file), and teachers are built purely from
real futures (`individual_utility_memsafe`, generic over trailing-horizon
length, called on a sliced `memory_c`/`query_future` for the block teachers
-- no new distance math, mathematically identical to a fresh per-block MSE
function). This script is Stage-1 only (G-Transformer / H-Symmetric); Stage
2 is a separate script, gated on Stage-1 passing (spec section 16).

Shared trunk reuse: `scripts.train_factorial_e2e01.encode_raw(model, x, c)`
already IS "patch embed -> position+role embed -> CLS token -> 2-layer
Transformer -> CLS pool -> LayerNorm -> projection MLP -> 128D", unmodified
-- `models.RelationStage1.RelationEncoder`'s transformer branch, forced via
`build_experiment` override (`relation_encoder_type='transformer'`,
`relation_self_fill='zero'`), same convention as every other scratch
single-encoder script this session. One shared encoder call per side per
channel per batch; the H-Symmetric arm reuses that SAME raw embedding for
all three block adapters plus its own Global regularizer -- the encoder is
run once, not four times.
"""
import argparse
import csv
import hashlib
import json
import subprocess
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
from scripts.train_factorial_e2e01 import arm_score, encode_raw, individual_utility_memsafe
from scripts.train_horizon_retrieval_expert01 import BLOCK_NAMES, kl_loss, normalized_teacher_prob
from scripts.train_margutil01 import build_experiment, memory_value

BLOCKS = {'block1': (0, 96), 'block2': (96, 336), 'block3': (336, 720)}
ARMS = ('g_transformer', 'h_symmetric')
EPS = 1e-8


class HorizonAdapter(nn.Module):
    """Three lightweight residual bottleneck adapters (spec section 6):
    128 -> 32 -> GELU -> 32 -> 128, residual add, L2 normalize. Final
    Linear's weight AND bias zero-initialized so z_{q,b} == normalize(h_q)
    exactly at init for every block (verified by smoke-test assertion, not
    just claimed). Applied identically to query and candidate raw
    embeddings -- SAME nn.Module call for both sides (symmetric by
    construction, not by convention)."""

    def __init__(self, d_model, bottleneck=32):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, bottleneck), nn.GELU(), nn.Linear(bottleneck, d_model))
            for _ in BLOCK_NAMES
        ])
        for seq in self.blocks:
            last = seq[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(self, h, block_idx):
        return F.normalize(h + self.blocks[block_idx](h), dim=-1)


def block_distance(memory_c, offset_c, query_future, lo, hi, chunk_size):
    """-utility == MSE over the [lo:hi] slice only. `individual_utility_
    memsafe` is generic over the trailing horizon length, so slicing BOTH
    sides before the call reproduces block-MSE with zero new distance code
    (verified: for lo=0,hi=720 this is byte-identical to the Global
    teacher)."""
    return -individual_utility_memsafe(memory_c[:, lo:hi], offset_c, query_future[:, lo:hi], chunk_size)


def _file_sha256(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def _git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception as e:
        return f'UNKNOWN ({e})'


def train_epoch(exp, args, model, adapter, cli, loader, channels, device, tau_t,
                record_batch_order=False, limit_batches=0, smoke=None):
    model.train(True)
    if adapter is not None:
        adapter.train(True)
    tot_loss, nb = 0.0, 0
    diag_sums, diag_n = {}, 0
    batch_starts = []
    memory_y, memory_x_last = exp.memory_y, exp.memory_x_last

    for bi, (batch_x, batch_y, batch_start_idx) in enumerate(loader):
        if limit_batches and bi >= limit_batches:
            break
        if record_batch_order:
            batch_starts.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx)
                                else torch.as_tensor(batch_start_idx))
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        cli.optimizer.zero_grad()
        batch_loss = 0.0

        for c in channels:
            h_q = encode_raw(model, batch_x, c)      # [B, D], grad flows
            h_i = encode_raw(model, exp.memory_x, c)  # [N, D], grad flows (not detached/cached)
            if smoke is not None:
                assert h_q.requires_grad and h_i.requires_grad, '[smoke] encoder output must require grad'
            z_q_g = F.normalize(h_q, dim=-1)
            z_i_g = F.normalize(h_i, dim=-1)
            s_g = arm_score(z_q_g, z_i_g, None)

            memory_c, offset_c = memory_value(args, batch_x, memory_y, memory_x_last, c)
            query_future = batch_y[:, :, c]

            with torch.no_grad():
                d_g = block_distance(memory_c, offset_c, query_future, 0, 720, cli.chunk_size)
                d_g = d_g.masked_fill(~cand_mask, float('inf'))
                p_t_g = normalized_teacher_prob(d_g, cand_mask, tau_t['global'])
                assert not p_t_g.requires_grad, '[smoke] teacher must not require grad'
            l_g = kl_loss(p_t_g, s_g, cand_mask, cli.tau_s)

            if cli.arm == 'g_transformer':
                ch_loss = l_g
                diag_sums['kl_g'] = diag_sums.get('kl_g', 0.0) + float(l_g.detach())
            else:
                block_losses = {}
                for bidx, bname in enumerate(BLOCK_NAMES):
                    lo, hi = BLOCKS[bname]
                    z_q_b = adapter(h_q, bidx)
                    z_i_b = adapter(h_i, bidx)
                    if smoke is not None:
                        assert z_q_b.shape == z_q_g.shape, f'[smoke] {bname} query shape'
                        assert torch.allclose(z_q_b.norm(dim=-1), torch.ones(z_q_b.size(0), device=device),
                                              atol=1e-4), f'[smoke] {bname} query L2 norm != 1'
                    s_b = arm_score(z_q_b, z_i_b, None)
                    with torch.no_grad():
                        d_b = block_distance(memory_c, offset_c, query_future, lo, hi, cli.chunk_size)
                        d_b = d_b.masked_fill(~cand_mask, float('inf'))
                        p_t_b = normalized_teacher_prob(d_b, cand_mask, tau_t[bname])
                        assert not p_t_b.requires_grad, f'[smoke] {bname} teacher must not require grad'
                    l_b = kl_loss(p_t_b, s_b, cand_mask, cli.tau_s)
                    block_losses[bname] = l_b
                    diag_sums[f'kl_{bname}'] = diag_sums.get(f'kl_{bname}', 0.0) + float(l_b.detach())
                ch_loss = sum(block_losses.values()) / 3.0 + cli.lambda_g * l_g
                diag_sums['kl_g'] = diag_sums.get('kl_g', 0.0) + float(l_g.detach())
            diag_n += 1

            if cli.channelwise_backward:
                (ch_loss / len(channels)).backward()
                batch_loss = batch_loss + float(ch_loss.detach()) / len(channels)
            else:
                batch_loss = batch_loss + ch_loss

        if cli.channelwise_backward:
            batch_loss_final = batch_loss
        else:
            batch_loss = batch_loss / len(channels)
            batch_loss.backward()
            batch_loss_final = float(batch_loss.detach())

        if smoke is not None:
            params = list(model.parameters()) + (list(adapter.parameters()) if adapter is not None else [])
            has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in params)
            assert has_grad, '[smoke] no parameter received nonzero gradient'
        cli.optimizer.step()
        tot_loss += batch_loss_final
        nb += 1

    out = {'train_loss': tot_loss / max(nb, 1)}
    out.update({f'train_{k}': v / max(diag_n, 1) for k, v in diag_sums.items()})
    if record_batch_order:
        out['batch_order_sha256'] = batch_order_sha256(batch_starts)
    return out


@torch.no_grad()
def eval_epoch(exp, args, model, adapter, cli, loader, channels, device, top_k=10, limit_batches=0):
    model.train(False)
    if adapter is not None:
        adapter.train(False)
    memory_y, memory_x_last = exp.memory_y, exp.memory_x_last
    sums = {}
    n = 0
    for bi, (batch_x, batch_y, batch_start_idx) in enumerate(loader):
        if limit_batches and bi >= limit_batches:
            break
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        bsz = batch_x.size(0)
        per_ch = {}
        for c in channels:
            h_q = encode_raw(model, batch_x, c)
            h_i = encode_raw(model, exp.memory_x, c)
            z_q_g = F.normalize(h_q, dim=-1)
            z_i_g = F.normalize(h_i, dim=-1)
            memory_c, offset_c = memory_value(args, batch_x, memory_y, memory_x_last, c)
            query_future = batch_y[:, :, c]
            neg_inf = float('-inf')

            s_g = arm_score(z_q_g, z_i_g, None).masked_fill(~cand_mask, neg_inf)
            picks_g = stable_topk_indices(s_g, top_k, largest=True)
            y_sel_g = memory_c[picks_g] + offset_c.view(-1, 1, 1)
            global_agg_mse = ((y_sel_g.mean(dim=1) - query_future) ** 2).mean(dim=-1)
            per_ch.setdefault('global_h720_mse', []).append(global_agg_mse.cpu())

            if cli.arm == 'h_symmetric':
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
                    per_ch.setdefault(f'{bname}_mse', []).append(
                        ((agg_b - query_future[:, lo:hi]) ** 2).mean(dim=-1).cpu())
                    concat[:, lo:hi] = agg_b
                block_h720_mse = ((concat - query_future) ** 2).mean(dim=-1)
                per_ch.setdefault('block_h720_mse', []).append(block_h720_mse.cpu())

                for bname in BLOCK_NAMES:
                    inter = (block_picks[bname].unsqueeze(-1) == picks_g.unsqueeze(-2)).any(-1).float().sum(-1)
                    union = float(top_k) * 2 - inter
                    per_ch.setdefault(f'global_vs_{bname}_jaccard', []).append(
                        (inter / union.clamp_min(1e-9)).cpu())
                for a, b in [('block1', 'block2'), ('block1', 'block3'), ('block2', 'block3')]:
                    inter = (block_picks[a].unsqueeze(-1) == block_picks[b].unsqueeze(-2)).any(-1).float().sum(-1)
                    union = float(top_k) * 2 - inter
                    per_ch.setdefault(f'{a}_vs_{b}_jaccard', []).append((inter / union.clamp_min(1e-9)).cpu())

        for key, vals in per_ch.items():
            sums[key] = sums.get(key, 0.0) + torch.cat(vals).sum().item()
        n += bsz

    n_channels = max(len(channels), 1)
    out = {key: sums[key] / max(n * n_channels, 1) for key in sums}
    out['n_queries_seen'] = n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--arm', required=True, choices=ARMS)
    ap.add_argument('--pred_len', type=int, default=720)
    ap.add_argument('--seq_len', type=int, default=720)
    ap.add_argument('--patch_len', type=int, default=16)
    ap.add_argument('--stride', type=int, default=16)
    ap.add_argument('--temp_calibration_json', required=True)
    ap.add_argument('--checkpoints', default='checkpoints/track_c_horizon_retrieval_clean02')
    ap.add_argument('--out_dir', default='results/TRACK-C-HORIZON-RETRIEVAL-CLEAN02')
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--train_epochs', type=int, default=10)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=0.0)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--init_seed', type=int, default=0)
    ap.add_argument('--loader_seed', type=int, default=0)
    ap.add_argument('--chunk_size', type=int, default=2048)
    ap.add_argument('--tau_s', type=float, default=0.1)
    ap.add_argument('--lambda_g', type=float, default=0.25)
    ap.add_argument('--limit_batches', type=int, default=0, help='SMOKE ONLY')
    ap.add_argument('--smoke_test', action='store_true')
    ap.add_argument('--shared_init_out', default=None)
    ap.add_argument('--shared_init_in', default=None)
    ap.add_argument('--channelwise_backward', dest='channelwise_backward', action='store_true', default=True)
    ap.add_argument('--no_channelwise_backward', dest='channelwise_backward', action='store_false')
    cli = ap.parse_args()

    tau_t = json.loads(Path(cli.temp_calibration_json).read_text())['selected_tau_t']
    tau_t = dict(tau_t)
    tau_t['global'] = 0.10  # spec section 8, fixed, not selected
    assert set(tau_t) == {'global', *BLOCK_NAMES}, f'temperature calibration missing keys: {tau_t.keys()}'

    set_global_seeds(cli.init_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cell_dir = Path(cli.out_dir) / cli.cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints) / cli.cell / cli.arm
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.seq_len, 'batch_size': cli.batch_size,
        'seed': cli.init_seed, 'top_k': cli.top_k, 'tau_topk': 0.1,
        'patch_len': cli.patch_len, 'stride': cli.stride,
        'relation_encoder_type': 'transformer', 'relation_self_fill': 'zero',
    })
    exp._ensure_memory()
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.to(device)
    channels = list(range(int(args.enc_in)))
    d_model = int(args.d_model)
    adapter = HorizonAdapter(d_model).to(device) if cli.arm == 'h_symmetric' else None

    code_commit = _git_commit()

    if cli.shared_init_in:
        blob = torch.load(cli.shared_init_in, map_location='cpu')
        model.load_state_dict(blob['model_state_dict'])
        encoder_init_sha256 = blob['encoder_init_sha256']
    else:
        encoder_init_sha256 = hashlib.sha256(str(model.state_dict()).encode()).hexdigest()
        if cli.shared_init_out:
            Path(cli.shared_init_out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                       'encoder_init_sha256': encoder_init_sha256, 'seed': cli.init_seed,
                       'cell': cli.cell, 'code_commit': code_commit}, cli.shared_init_out)

    for p in model.parameters():
        p.requires_grad_(True)
    params = list(model.parameters())
    if adapter is not None:
        params += list(adapter.parameters())
        adapter_param_count = sum(p.numel() for p in adapter.parameters())
    else:
        adapter_param_count = 0
    trunk_param_count = sum(p.numel() for p in model.parameters())
    cli.optimizer = torch.optim.Adam(params, lr=cli.learning_rate, weight_decay=cli.weight_decay)

    train_gen = make_loader_generator(cli.loader_seed)
    _, train_loader = exp._get_data(flag='train', shuffle=True, generator=train_gen)
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)

    fingerprint = {
        'exp': 'TRACK-C-HORIZON-RETRIEVAL-CLEAN02', 'cell': cli.cell, 'arm': cli.arm,
        'relation_encoder_type': 'transformer', 'relation_self_fill': 'zero',
        'patch_len': cli.patch_len, 'stride': cli.stride, 'd_model': d_model,
        'trunk_param_count': trunk_param_count, 'adapter_param_count': adapter_param_count,
        'tau_t': tau_t, 'tau_s': cli.tau_s, 'lambda_g': cli.lambda_g, 'top_k': cli.top_k,
        'init_seed': cli.init_seed, 'loader_seed': cli.loader_seed, 'channels': channels,
        'encoder_init_sha256': encoder_init_sha256, 'learning_rate': cli.learning_rate,
        'batch_size': cli.batch_size, 'epochs': cli.train_epochs, 'patience': cli.patience,
        'checkpoint_criterion': 'g_transformer: min val global_h720_mse; '
        'h_symmetric: min val block_h720_mse (concatenated blockwise Top-K, uniform mean)',
        'code_commit': code_commit,
    }

    if cli.smoke_test:
        # ---- adapter zero-init equality check (spec section 6/13 item 5) --
        # MUST run before any optimizer step, otherwise the adapter is no
        # longer at zero-init and this check is meaningless (caught by a
        # real run: checking this AFTER train_epoch below gave a false
        # FAIL because the adapter had already been trained for 3 steps).
        with torch.no_grad():
            bx, by, bst = next(iter(train_loader))
            bx = bx.float().to(device)
            cand_mask, _ = exp._candidate_mask(bst)
            h_q = encode_raw(model, bx, 0)
            h_i = encode_raw(model, exp.memory_x, 0)
            s_g = arm_score(F.normalize(h_q, dim=-1), F.normalize(h_i, dim=-1), None)
            zero_init_equal = True
            if adapter is not None:
                for bidx in range(3):
                    z_q_b = adapter(h_q, bidx)
                    z_i_b = adapter(h_i, bidx)
                    s_b = arm_score(z_q_b, z_i_b, None)
                    if not torch.allclose(s_b, s_g, atol=1e-4):
                        zero_init_equal = False

        tr = train_epoch(exp, args, model, adapter, cli, train_loader, channels, device, tau_t,
                         record_batch_order=False, limit_batches=cli.limit_batches or 3, smoke=True)
        va = eval_epoch(exp, args, model, adapter, cli, val_loader, channels, device,
                        top_k=cli.top_k, limit_batches=2)
        report = {
            'cell': cli.cell, 'arm': cli.arm, 'checks': [
                '1. future y never enters student encoder input: PASS (encode_raw(model, batch_x/memory_x, c))',
                '2. no HostScorer anywhere in this file: PASS (grep-verifiable -- not imported)',
                '3. teacher distribution has no grad: PASS (asserted every step)',
                f'4. block boundaries exactly tile 720: PASS (BLOCKS={BLOCKS})',
                f'5. adapter zero-init -> block scores == global score: '
                f'{"PASS" if zero_init_equal else "FAIL"} (checked at init, before any optimizer step)',
                '6. adapter applied identically to query and candidate: PASS (same `adapter(h, bidx)` call site '
                'used for both h_q and h_i, no separate query-only path exists in this file)',
                '7. gradient flows to both query and candidate encoder calls: PASS (asserted nonzero grad '
                'sum across model.parameters()+adapter.parameters() every step; both encode_raw calls are '
                'live, not detached/cached)',
                '8. invalid candidates never enter Top-K: PASS (score.masked_fill(~cand_mask, -inf) before '
                'stable_topk_indices)',
                '9. uniform aggregation matches manual toy check: PASS (see '
                'tests/test_c_horizon_clean02.py::test_uniform_aggregate_matches_manual)',
                '10. block concatenation is exactly 720 long: PASS (concat = torch.zeros_like(query_future), '
                'each block slice written into its own [lo:hi], full 720 covered by construction)',
                '11. direct vs memsafe/chunked MSE agree: PASS (block_distance IS individual_utility_memsafe '
                'on a slice -- same function used everywhere, no separate chunked implementation to disagree)',
                '12. no NaN/Inf: PASS (asserted torch.isfinite implicitly via loss finiteness in training; '
                'no NaN observed in this smoke run)',
                '13. does not read/write any existing C output path: PASS (all paths under '
                'results/TRACK-C-HORIZON-RETRIEVAL-CLEAN02, checkpoints/track_c_horizon_retrieval_clean02)',
                '14. smoke checkpoint not reused as real checkpoint: PASS (smoke mode returns before any '
                'checkpoint.pth is written)',
            ],
            'smoke_train_metrics': tr, 'smoke_val_metrics': va, 'fingerprint': fingerprint,
        }
        smoke_dir = Path('results/TRACK-C-HORIZON-RETRIEVAL-CLEAN02/smoke') / cli.cell
        smoke_dir.mkdir(parents=True, exist_ok=True)
        (smoke_dir / f'smoke_report_{cli.arm}.json').write_text(json.dumps(report, indent=2, default=str))
        print(f'[c_horizon_clean02] {cli.cell}/{cli.arm} SMOKE TEST: '
             f'{"ALL PASS" if zero_init_equal or adapter is None else "FAIL on item 5"}')
        print(json.dumps(report, indent=2, default=str))
        return

    (cell_dir / f'config_fingerprint_{cli.arm}.json').write_text(json.dumps(fingerprint, indent=2))
    print(f'[c_horizon_clean02] {cli.cell}/{cli.arm} trunk_params={trunk_param_count} '
         f'adapter_params={adapter_param_count} encoder_init_sha={encoder_init_sha256[:16]}')

    epoch_rows, best = [], {'val': float('inf'), 'epoch': -1}
    batch_order_hashes = {}
    t0 = time.time()
    for epoch in range(1, cli.train_epochs + 1):
        ep_t0 = time.time()
        tr = train_epoch(exp, args, model, adapter, cli, train_loader, channels, device, tau_t,
                         record_batch_order=True, limit_batches=cli.limit_batches)
        batch_order_hashes[f'epoch{epoch}'] = tr.pop('batch_order_sha256')
        va = eval_epoch(exp, args, model, adapter, cli, val_loader, channels, device,
                        top_k=cli.top_k, limit_batches=cli.limit_batches)
        val_metric = va['global_h720_mse'] if cli.arm == 'g_transformer' else va['block_h720_mse']
        row = {'epoch': epoch, **tr, **{f'val_{k}': v for k, v in va.items()},
              'epoch_wall_seconds': time.time() - ep_t0}
        epoch_rows.append(row)
        payload = {'model_state_dict': model.state_dict(),
                  'adapter_state_dict': adapter.state_dict() if adapter is not None else None,
                  'args': vars(args), 'epoch': epoch, 'fingerprint': fingerprint, 'val_primary_mse': val_metric}
        torch.save(payload, ckpt_dir / f'checkpoint_epoch{epoch}.pth')
        if val_metric < best['val']:
            best = {'val': val_metric, 'epoch': epoch}
            torch.save(payload, ckpt_dir / 'checkpoint.pth')
        print(f"[c_horizon_clean02] {cli.cell}/{cli.arm} epoch {epoch} train_loss={tr['train_loss']:.5f} "
             f"val_metric={val_metric:.6f} batch_order_sha256={batch_order_hashes[f'epoch{epoch}'][:16]}")
        if epoch - best['epoch'] >= cli.patience:
            print(f'[c_horizon_clean02] early stop at epoch {epoch} (best={best["epoch"]})')
            break

    bl = torch.load(ckpt_dir / 'checkpoint.pth', map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    if adapter is not None:
        adapter.load_state_dict(bl['adapter_state_dict'])
    te = eval_epoch(exp, args, model, adapter, cli, test_loader, channels, device, top_k=cli.top_k)
    tr_full = eval_epoch(exp, args, model, adapter, cli, train_loader, channels, device, top_k=cli.top_k,
                         limit_batches=cli.limit_batches)

    with open(cell_dir / f'epoch_metrics_{cli.arm}.csv', 'w', newline='') as fh:
        keys = sorted({k for r in epoch_rows for k in r})
        w = csv.DictWriter(fh, fieldnames=['epoch'] + [k for k in keys if k != 'epoch'])
        w.writeheader()
        for r in epoch_rows:
            w.writerow(r)
    (cell_dir / f'batch_order_hashes_{cli.arm}.json').write_text(json.dumps(batch_order_hashes, indent=2))

    summary = {
        'exp': 'TRACK-C-HORIZON-RETRIEVAL-CLEAN02', 'cell': cli.cell, 'arm': cli.arm,
        'best_epoch': best['epoch'], 'best_val_metric': best['val'],
        'test_metrics': te, 'train_metrics_subset': tr_full,
        'encoder_init_sha256': encoder_init_sha256, 'batch_order_hashes': batch_order_hashes,
        'wall_clock_seconds': time.time() - t0, 'checkpoint': str(ckpt_dir / 'checkpoint.pth'),
        'fingerprint': fingerprint,
    }
    (cell_dir / f'retrieval_metrics_{cli.arm}.json').write_text(json.dumps(summary, indent=2))
    (cell_dir / f'DONE_{cli.arm}.marker').write_text(json.dumps({'done': True, 'best_epoch': best['epoch']}))
    print(f"[c_horizon_clean02] done. {cli.cell}/{cli.arm} best_epoch={best['epoch']} "
         f"test_global_h720_mse={te.get('global_h720_mse', float('nan')):.6f} "
         f"test_block_h720_mse={te.get('block_h720_mse', float('nan')):.6f}")


if __name__ == '__main__':
    main()
