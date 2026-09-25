#!/usr/bin/env python3
"""Experiment E, Stage E2 -- Minimal Multi-Subspace Retrieval Encoder.

Single scratch `RelationEncoder` (relation_encoder_type='transformer', same
[ISSUE][CORRECTED] override as every other scratch single-encoder script
this session -- the S0_wce reference checkpoints' 'mlp' branch ignores
patch_len entirely). One forward pass per side (`encode_raw`, unmodified
from `train_factorial_e2e01`) produces a single D-dim raw embedding, which
is SPLIT (not separately projected -- spec section 8.1: "expert마다 별도의
Transformer encoder를 두는 것이 아니다... 최종 embedding만 네 subspace로
분할한다") into four contiguous chunks (shared/local/trend/seasonal, as
equal as D allows, remainder assigned to the leading subspaces and
recorded in the fingerprint), each independently L2-normalized. Component
teachers are the exact, unmodified E1 `component_distance_memsafe`/
`transform` functions (`diag_experiment_e_teacher01`) -- no host encoder,
no learned teacher. Per-component soft listwise KL loss reuses
`normalized_teacher_prob`/`kl_loss` (`train_horizon_retrieval_expert01`)
per component, combined as
    L_E2 = L_shared + (lambda_component/3)*(L_local+L_trend+L_seasonal)
(spec section 8.3, lambda_component=1.0 default). Each component gets its
OWN pre-calibrated temperature (`calibrate_experiment_e2_temperature.py`)
since the four transforms have different natural distance scales.

Per spec section 3, this script is only ever invoked for a dataset that
passed E1 (Weather_720, per the E1/E1.5 diagnostics this session -- ETTh1
FAILed on every split and is excluded).
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
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.diag_experiment_e_teacher01 import COMPONENTS, component_distance_memsafe
from scripts.rng_control01 import batch_order_sha256, make_loader_generator, set_global_seeds
from scripts.train_factorial_e2e01 import encode_raw
from scripts.train_horizon_retrieval_expert01 import kl_loss, normalized_teacher_prob, teacher_diagnostics
from scripts.train_margutil01 import build_experiment, memory_value

EPS = 1e-8


def subspace_sizes(d_model, n=4):
    """As-equal-as-possible split, remainder assigned to the leading
    subspaces (shared first) -- deterministic, recorded in fingerprint."""
    base = d_model // n
    rem = d_model % n
    sizes = [base + (1 if i < rem else 0) for i in range(n)]
    return sizes  # [shared, local, trend, seasonal]


def split_normalize(z, sizes):
    """z: [B, D] raw encoder output -> dict of 4 independently-L2-normalized
    slices, contiguous, in COMPONENTS order (shared,local,trend,seasonal)."""
    out = {}
    start = 0
    for comp, size in zip(COMPONENTS, sizes):
        out[comp] = F.normalize(z[:, start:start + size], dim=-1)
        start += size
    return out


def cosine_score(z_q_e, z_k_e):
    """Both sides already L2-normalized -> dot product IS cosine."""
    return torch.matmul(z_q_e, z_k_e.transpose(0, 1))


def recall_at_k(model_idx, oracle_idx, k):
    hit = (model_idx.unsqueeze(-1) == oracle_idx.unsqueeze(-2)).any(-1)
    return hit.float().sum(-1) / k


def ndcg_at_k(model_idx, d, valid_mask, k):
    tgt = (-d).masked_fill(~valid_mask, float('-inf'))
    rel = tgt - tgt.masked_fill(~valid_mask, float('inf')).min(dim=-1, keepdim=True).values
    rel = rel.masked_fill(~valid_mask, 0.0)
    gains = rel.gather(1, model_idx)
    disc = 1.0 / torch.log2(torch.arange(2, k + 2, device=d.device).float()).unsqueeze(0)
    dcg = (gains * disc).sum(-1)
    ideal = rel.topk(k, dim=-1).values
    idcg = (ideal * disc).sum(-1).clamp_min(1e-12)
    return dcg / idcg


def _git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception as e:
        return f'UNKNOWN ({e})'


def train_epoch(exp, args, model, cli, loader, channels, device, period, sizes, tau_t,
                record_batch_order=False, limit_batches=0, smoke=None):
    model.train(True)
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
            z_q_raw = encode_raw(model, batch_x, c)          # [B, D], grad flows
            z_k_raw = encode_raw(model, exp.memory_x, c)      # [N, D], grad flows (not detached)
            if smoke is not None:
                assert z_q_raw.requires_grad and z_k_raw.requires_grad, \
                    '[smoke] encoder output must require grad'
            zq = split_normalize(z_q_raw, sizes)
            zk = split_normalize(z_k_raw, sizes)
            if smoke is not None:
                for comp, size in zip(COMPONENTS, sizes):
                    assert zq[comp].shape == (batch_x.size(0), size), f'[smoke] {comp} query shape'
                    assert zk[comp].shape == (exp.memory_x.size(0), size), f'[smoke] {comp} cand shape'
                    assert torch.allclose(zq[comp].norm(dim=-1), torch.ones(batch_x.size(0), device=device),
                                          atol=1e-4), f'[smoke] {comp} query L2 norm != 1'
                    assert torch.allclose(zk[comp].norm(dim=-1), torch.ones(exp.memory_x.size(0), device=device),
                                          atol=1e-4), f'[smoke] {comp} candidate L2 norm != 1'

            memory_c, offset_c = memory_value(args, batch_x, memory_y, memory_x_last, c)
            query_future = batch_y[:, :, c]

            comp_losses = {}
            for comp in COMPONENTS:
                s_e = cosine_score(zq[comp], zk[comp])
                if smoke is not None:
                    assert s_e.shape == (batch_x.size(0), exp.memory_x.size(0)), f'[smoke] {comp} score shape'
                with torch.no_grad():
                    d_e = component_distance_memsafe(memory_c, offset_c, query_future, comp, period,
                                                      cli.chunk_size).masked_fill(~cand_mask, float('inf'))
                    p_t_e = normalized_teacher_prob(d_e, cand_mask, tau_t[comp])
                    assert not p_t_e.requires_grad, f'[smoke] {comp} teacher must not require grad'
                ch_loss_e = kl_loss(p_t_e, s_e, cand_mask, cli.tau_s)
                assert torch.isfinite(ch_loss_e).all(), f'[smoke] {comp} loss not finite'
                comp_losses[comp] = ch_loss_e
                diag_sums[f'kl_{comp}'] = diag_sums.get(f'kl_{comp}', 0.0) + float(ch_loss_e.detach())
                if comp == 'shared':
                    with torch.no_grad():
                        diag = teacher_diagnostics(p_t_e, cand_mask)
                        for k, v in diag.items():
                            diag_sums[k] = diag_sums.get(k, 0.0) + v

            ch_total = comp_losses['shared'] + (cli.lambda_component / 3.0) * (
                comp_losses['local'] + comp_losses['trend'] + comp_losses['seasonal'])
            diag_n += 1

            if getattr(cli, 'channelwise_backward', False):
                (ch_total / len(channels)).backward()
                batch_loss = batch_loss + float(ch_total.detach()) / len(channels)
            else:
                batch_loss = batch_loss + ch_total

        if getattr(cli, 'channelwise_backward', False):
            batch_loss_final = batch_loss
        else:
            batch_loss = batch_loss / len(channels)
            batch_loss.backward()
            batch_loss_final = float(batch_loss.detach())

        if smoke is not None:
            has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
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
def eval_epoch(exp, args, model, cli, loader, channels, device, period, sizes, tau_t, top_k=10, limit_batches=0):
    """Per-subspace (E-Shared/E-Local/E-Trend/E-Seasonal) hard Top-K, each
    scored against ITS OWN component teacher distance (component_retmse10,
    recall@10, ndcg@10) and also against the overall/shared distance
    (overall_agg_mse -- uniform-aggregate absolute-future MSE, for the
    arms-comparison table). Checkpoint/primary metric ([ISSUE] spec section
    12 does not literally say which subspace's Top-10 -- E-Shared is used
    here as the direct analogue of the existing single-encoder production
    metric): val_shared_overall_agg_mse."""
    model.train(False)
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
            z_q_raw = encode_raw(model, batch_x, c)
            z_k_raw = encode_raw(model, exp.memory_x, c)
            zq = split_normalize(z_q_raw, sizes)
            zk = split_normalize(z_k_raw, sizes)
            memory_c, offset_c = memory_value(args, batch_x, memory_y, memory_x_last, c)
            query_future = batch_y[:, :, c]

            d_by_comp = {}
            for comp in COMPONENTS:
                d_by_comp[comp] = component_distance_memsafe(memory_c, offset_c, query_future, comp, period,
                                                              cli.chunk_size).masked_fill(~cand_mask, float('inf'))

            for comp in COMPONENTS:
                s_e = cosine_score(zq[comp], zk[comp]).masked_fill(~cand_mask, float('-inf'))
                model_idx = stable_topk_indices(s_e, top_k, largest=True)
                d_e = d_by_comp[comp]
                oracle_idx = stable_topk_indices(d_e, top_k, largest=False)

                per_ch.setdefault(f'{comp}_retmse10', []).append(d_e.gather(1, model_idx).mean(-1).cpu())
                per_ch.setdefault(f'{comp}_oracle_retmse10', []).append(d_e.gather(1, oracle_idx).mean(-1).cpu())
                per_ch.setdefault(f'{comp}_recall_at_10', []).append(
                    recall_at_k(model_idx, oracle_idx, top_k).cpu())
                per_ch.setdefault(f'{comp}_ndcg_at_10', []).append(
                    ndcg_at_k(model_idx, d_e, cand_mask, top_k).cpu())

                y_sel = memory_c[model_idx] + offset_c.view(-1, 1, 1)
                agg = y_sel.mean(dim=1)
                overall_mse = ((agg - query_future) ** 2).mean(dim=-1)
                per_ch.setdefault(f'{comp}_overall_agg_mse', []).append(overall_mse.cpu())

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
    ap.add_argument('--period_json', required=True)
    ap.add_argument('--temp_calibration_json', required=True)
    ap.add_argument('--pred_len', type=int, default=720)
    ap.add_argument('--seq_len', type=int, default=720)
    ap.add_argument('--patch_len', type=int, default=16)
    ap.add_argument('--stride', type=int, default=None)
    ap.add_argument('--checkpoints', default='checkpoints/experiment_e_e2_multisubspace01')
    ap.add_argument('--out_dir', default='results/EXPERIMENT-E-E2')
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--train_epochs', type=int, default=10)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=0.0)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--init_seed', type=int, default=0)
    ap.add_argument('--loader_seed', type=int, default=0)
    ap.add_argument('--chunk_size', type=int, default=2048)
    ap.add_argument('--tau_s', type=float, default=0.1)
    ap.add_argument('--lambda_component', type=float, default=1.0)
    ap.add_argument('--limit_batches', type=int, default=0, help='SMOKE ONLY')
    ap.add_argument('--smoke_test', action='store_true')
    ap.add_argument('--channelwise_backward', dest='channelwise_backward', action='store_true', default=True)
    ap.add_argument('--no_channelwise_backward', dest='channelwise_backward', action='store_false')
    cli = ap.parse_args()
    cli.stride = cli.stride or cli.patch_len

    period = int(json.loads(Path(cli.period_json).read_text())['median_period'])
    tau_t = json.loads(Path(cli.temp_calibration_json).read_text())['selected_tau_t']
    assert set(tau_t) == set(COMPONENTS), f'temperature calibration missing components: {tau_t.keys()}'

    set_global_seeds(cli.init_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cell_dir = Path(cli.out_dir) / cli.cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints) / cli.cell
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
    sizes = subspace_sizes(d_model)
    print(f'[e2_multisubspace] {cli.cell}: d_model={d_model} subspace_sizes(shared,local,trend,seasonal)={sizes} '
         f'period={period} tau_t={tau_t}')

    code_commit = _git_commit()
    encoder_init_sha256 = hashlib.sha256(str(model.state_dict()).encode()).hexdigest()

    for p in model.parameters():
        p.requires_grad_(True)
    param_count = sum(p.numel() for p in model.parameters())
    cli.optimizer = torch.optim.Adam(model.parameters(), lr=cli.learning_rate, weight_decay=cli.weight_decay)

    train_gen = make_loader_generator(cli.loader_seed)
    _, train_loader = exp._get_data(flag='train', shuffle=True, generator=train_gen)
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)

    fingerprint = {
        'exp': 'EXPERIMENT-E-E2-MULTISUBSPACE01', 'cell': cli.cell,
        'd_model': d_model, 'subspace_sizes_shared_local_trend_seasonal': sizes,
        'period': period, 'tau_t_per_component': tau_t, 'tau_s': cli.tau_s,
        'lambda_component': cli.lambda_component,
        'relation_encoder_type_corrected_from_host': 'mlp -> transformer',
        'relation_self_fill': 'zero', 'patch_len': cli.patch_len, 'stride': cli.stride,
        'param_count': param_count, 'top_k': cli.top_k, 'init_seed': cli.init_seed,
        'loader_seed': cli.loader_seed, 'channels': channels,
        'encoder_init_sha256': encoder_init_sha256, 'learning_rate': cli.learning_rate,
        'batch_size': cli.batch_size, 'epochs': cli.train_epochs, 'patience': cli.patience,
        'checkpoint_criterion': '[ISSUE] spec ambiguous on which subspace Top-10 to checkpoint on -- '
        'using E-Shared subspace val overall_agg_mse as the direct analogue of the existing '
        'single-encoder production metric',
        'code_commit': code_commit,
    }

    if cli.smoke_test:
        peak_before = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
        tr = train_epoch(exp, args, model, cli, train_loader, channels, device, period, sizes, tau_t,
                         record_batch_order=False, limit_batches=cli.limit_batches or 3, smoke=True)
        va = eval_epoch(exp, args, model, cli, val_loader, channels, device, period, sizes, tau_t,
                        top_k=cli.top_k, limit_batches=2)
        peak_after = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
        report = {
            'cell': cli.cell, 'checks': [
                '1. encoder input never includes future y: PASS (encode_raw(model, batch_x/memory_x, c) only)',
                '2/3. teacher detached, no grad to teacher: PASS (component_distance_memsafe is @torch.no_grad(); '
                'asserted p_t_e.requires_grad is False every step)',
                '4. four subspace shapes: PASS (asserted zq/zk shapes == subspace_sizes every step)',
                '5. per-subspace L2 norm == 1: PASS (asserted every step, atol=1e-4)',
                '6. component score shapes [B,N]: PASS (asserted every step)',
                '7. mask -> no invalid candidate selected: PASS (stable_topk_indices operates on '
                'score.masked_fill(~cand_mask, -inf); invalid candidates score -inf, cannot enter Top-K '
                'given n_valid >> top_k)',
                '8. component loss finite: PASS (asserted torch.isfinite every step)',
                '9. gradient exists on encoder parameters after backward: PASS (asserted nonzero grad '
                'sum across model.parameters() every step)',
                '10. router freeze/unfreeze: N/A (no router in E2)',
                '11. candidate re-encoding policy matches production: PASS (encode_raw(model, exp.memory_x, c) '
                'recomputed every batch, unmodified from train_factorial_e2e01/train_patch_retrieval_expert01)',
                f'12. peak VRAM (this smoke run, {cli.cell}, subset): {peak_after / 1e9:.3f} GB '
                f'(delta from pre-smoke: {(peak_after - peak_before) / 1e9:.3f} GB) -- full-candidate '
                'run will scale with candidate-bank size per channel, same channelwise_backward pattern '
                'as train_patch_retrieval_expert01/train_horizon_retrieval_expert01',
            ],
            'smoke_train_metrics': tr, 'smoke_val_metrics': va, 'fingerprint': fingerprint,
        }
        (cell_dir / 'SMOKE_TEST_REPORT.md').write_text(
            '# Experiment E2 Smoke Test Report\n\n' + '\n\n'.join(
                [f'## {k}\n```json\n{json.dumps(v, indent=2)}\n```' if isinstance(v, (dict, list)) else f'## {k}\n{v}'
                 for k, v in report.items()]))
        print(f'[e2_multisubspace] {cli.cell} SMOKE TEST PASSED. Report: {cell_dir / "SMOKE_TEST_REPORT.md"}')
        print(json.dumps({'smoke_train': tr, 'smoke_val': va}, indent=2, default=str))
        return

    (cell_dir / 'config_fingerprint.json').write_text(json.dumps(fingerprint, indent=2))
    print(f'[e2_multisubspace] {cli.cell} params={param_count} encoder_init_sha={encoder_init_sha256[:16]} '
         f'init_seed={cli.init_seed} loader_seed={cli.loader_seed}')

    epoch_rows, best = [], {'val': float('inf'), 'epoch': -1}
    batch_order_hashes = {}
    t0 = time.time()
    for epoch in range(1, cli.train_epochs + 1):
        ep_t0 = time.time()
        tr = train_epoch(exp, args, model, cli, train_loader, channels, device, period, sizes, tau_t,
                         record_batch_order=True, limit_batches=cli.limit_batches)
        batch_order_hashes[f'epoch{epoch}'] = tr.pop('batch_order_sha256')
        va = eval_epoch(exp, args, model, cli, val_loader, channels, device, period, sizes, tau_t,
                        top_k=cli.top_k, limit_batches=cli.limit_batches)
        val_metric = va['shared_overall_agg_mse']
        row = {'epoch': epoch, **tr, **{f'val_{k}': v for k, v in va.items()},
              'epoch_wall_seconds': time.time() - ep_t0}
        epoch_rows.append(row)
        payload = {'model_state_dict': model.state_dict(), 'args': vars(args),
                  'epoch': epoch, 'fingerprint': fingerprint, 'val_primary_mse': val_metric}
        torch.save(payload, ckpt_dir / f'checkpoint_epoch{epoch}.pth')
        if val_metric < best['val']:
            best = {'val': val_metric, 'epoch': epoch}
            torch.save(payload, ckpt_dir / 'checkpoint.pth')
        print(f"[e2_multisubspace] {cli.cell} epoch {epoch} train_loss={tr['train_loss']:.5f} "
             f"val_shared_overall_agg_mse={val_metric:.6f} "
             f"val_local_overall_agg_mse={va['local_overall_agg_mse']:.6f} "
             f"val_trend_overall_agg_mse={va['trend_overall_agg_mse']:.6f} "
             f"val_seasonal_overall_agg_mse={va['seasonal_overall_agg_mse']:.6f} "
             f"batch_order_sha256={batch_order_hashes[f'epoch{epoch}'][:16]}")
        if epoch - best['epoch'] >= cli.patience:
            print(f'[e2_multisubspace] early stop at epoch {epoch} (best={best["epoch"]})')
            break

    bl = torch.load(ckpt_dir / 'checkpoint.pth', map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    te = eval_epoch(exp, args, model, cli, test_loader, channels, device, period, sizes, tau_t, top_k=cli.top_k)
    tr_full = eval_epoch(exp, args, model, cli, train_loader, channels, device, period, sizes, tau_t,
                         top_k=cli.top_k, limit_batches=cli.limit_batches)

    with open(cell_dir / 'epoch_metrics.csv', 'w', newline='') as fh:
        keys = sorted({k for r in epoch_rows for k in r})
        w = csv.DictWriter(fh, fieldnames=['epoch'] + [k for k in keys if k != 'epoch'])
        w.writeheader()
        for r in epoch_rows:
            w.writerow(r)
    (cell_dir / 'batch_order_hashes.json').write_text(json.dumps(batch_order_hashes, indent=2))

    summary = {
        'exp': 'EXPERIMENT-E-E2-MULTISUBSPACE01', 'cell': cli.cell,
        'best_epoch': best['epoch'], 'best_val_shared_overall_agg_mse': best['val'],
        'test_metrics': te, 'train_metrics_subset': tr_full,
        'encoder_init_sha256': encoder_init_sha256, 'batch_order_hashes': batch_order_hashes,
        'wall_clock_seconds': time.time() - t0, 'checkpoint': str(ckpt_dir / 'checkpoint.pth'),
        'fingerprint': fingerprint,
    }
    (cell_dir / 'retrieval_metrics.json').write_text(json.dumps(summary, indent=2))
    (cell_dir / 'DONE.marker').write_text(json.dumps({'done': True, 'best_epoch': best['epoch']}))
    print(f"[e2_multisubspace] done. {cli.cell} best_epoch={best['epoch']} "
         f"test_shared_overall_agg_mse={te['shared_overall_agg_mse']:.6f} "
         f"test_local_overall_agg_mse={te['local_overall_agg_mse']:.6f} "
         f"test_trend_overall_agg_mse={te['trend_overall_agg_mse']:.6f} "
         f"test_seasonal_overall_agg_mse={te['seasonal_overall_agg_mse']:.6f}")


if __name__ == '__main__':
    main()
