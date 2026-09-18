#!/usr/bin/env python3
"""TRACK-A-SET-NORMREGRET-CONTROL01 -- arm-specific Stage-2 retraining.

Builds a FRESH Exp_Stage2_Relation (same architecture/config as the
existing S0_wce host, read from that checkpoint's own `args` -- read-only
reuse per spec section 1) but does NOT load the host's trained
gate/forecast/fusion weights: those are freshly initialized (shared seed
across arms of the same cell, via a shared-init checkpoint, matching this
session's established convention). The Stage-1 retrieval branch is never
computed online here at all -- every batch's `relation_outputs` /
`relation_query_embs` are looked up from the per-arm cache built by
`scripts/build_choicece_retrieval_cache01.py` (the arm's own frozen
encoder+SetConditioner+RetrievalMetric free-running selection, aggregated
with the FIXED S0_wce host score) and injected via
`RelationStage2.forward_from_retrieval_values`, the model's own existing
hook for exactly this purpose.

Reuses, unmodified: `Exp_Stage2_Relation._loss` (the real Stage-2 training
loss, with whatever aux terms the host's own args configure) and
`Exp_Stage2_Relation._select_optimizer` (which already filters to
`requires_grad` parameters only, so freezing the retrieval-irrelevant
submodules is sufficient to keep them out of the optimizer without any
further filtering logic here).
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp.exp_stage2_relation import Exp_Stage2_Relation
from scripts.train_factorial_e2e01 import state_sha

FREEZE_SUBMODULES = ('stage1_encoder', 'shared_cross_projection', 'retrieval_metric',
                     'pairwise_scorer', 'query_cond_proj', 'candidate_cond_proj')


def _file_sha256(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def build_fresh_stage2(stage2_host, seed):
    """Fresh Exp_Stage2_Relation, same architecture/config as the host,
    NOT loading the host's own trained weights."""
    host_ck = torch.load(stage2_host, map_location='cpu')
    from types import SimpleNamespace
    args = SimpleNamespace(**host_ck['args'])
    args.num_workers = 0
    torch.manual_seed(seed)
    exp = Exp_Stage2_Relation(args)
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    return exp, args, model, host_ck


def freeze_retrieval_submodules(model):
    frozen_shas = {}
    for name in FREEZE_SUBMODULES:
        sub = getattr(model, name, None)
        if sub is None:
            continue
        for p in sub.parameters():
            p.requires_grad_(False)
        sub.eval()
        frozen_shas[name] = state_sha(sub.state_dict())
    return frozen_shas


REQUIRED_CACHE_SCHEMA = 'corrected_delta_v1'


def load_cache_as_lookup(cache_path):
    cache = torch.load(cache_path, map_location='cpu')
    if cache.get('cache_schema_version') != REQUIRED_CACHE_SCHEMA or cache.get('value_space') != 'delta':
        raise SystemExit(
            f'[ISSUE][ABORT] {cache_path}: cache_schema_version={cache.get("cache_schema_version")!r} '
            f'value_space={cache.get("value_space")!r} -- expected '
            f'cache_schema_version={REQUIRED_CACHE_SCHEMA!r} value_space=\'delta\'. This cache is either '
            'a legacy (pre-correction, absolute-space) cache or otherwise unrecognized -- refusing to '
            'silently train on it.')
    starts = cache['batch_start_idx'].tolist()
    start_to_row = {int(s): i for i, s in enumerate(starts)}
    return cache, start_to_row


def restore_absolute(delta_relation_outputs, query_offset):
    """VALIDATION-ONLY restoration to absolute scale -- same formula as
    `scripts/build_choicece_retrieval_cache01.py::restore_absolute`, NEVER
    used to feed Stage-2's forward pass (only for the raw_retrieval_aggregate_mse
    diagnostic and the Stage-1 FR-Agg cross-check)."""
    return delta_relation_outputs + query_offset.unsqueeze(-1)


def gate_distribution(gate_cat):
    g = gate_cat.float()
    q = torch.quantile(g, torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90]))
    return {
        'gate_mean': float(g.mean()), 'gate_std': float(g.std()), 'gate_median': float(q[2]),
        'gate_p10': float(q[0]), 'gate_p25': float(q[1]), 'gate_p75': float(q[3]), 'gate_p90': float(q[4]),
        'gate_min': float(g.min()), 'gate_max': float(g.max()),
        'gate_lt_0.02': float((g < 0.02).float().mean()),
        'gate_gt_0.10': float((g > 0.10).float().mean()),
        'gate_gt_0.50': float((g > 0.50).float().mean()),
        'gate_gt_0.98': float((g > 0.98).float().mean()),
    }


def lookup_batch(cache, start_to_row, batch_start_idx, device):
    rows = [start_to_row[int(s)] for s in batch_start_idx.tolist()]
    idx = torch.tensor(rows, dtype=torch.long)
    return {
        'relation_outputs': cache['relation_outputs'][idx].to(device),
        'relation_query_embs': cache['relation_query_embs'][idx].to(device),
    }


def train_epoch(exp, model, optimizer, loader, cache, start_to_row, device):
    model.train(True)
    for name in FREEZE_SUBMODULES:
        sub = getattr(model, name, None)
        if sub is not None:
            sub.eval()
    tot_loss, n = 0.0, 0
    for batch_x, batch_y, batch_start_idx in loader:
        batch_x, batch_y, batch_start_idx = exp._move_batch(batch_x, batch_y, batch_start_idx)
        cand_mask, counts = exp._candidate_mask(batch_start_idx)
        valid_query = counts.to(batch_x.device) > 0
        if bool((~valid_query).all()):
            continue
        rcache = lookup_batch(cache, start_to_row, batch_start_idx, batch_x.device)
        optimizer.zero_grad()
        y_final, y_base, y_ret, beta, lam, debug = model.forward_from_retrieval_values(
            rcache['relation_outputs'], batch_x=batch_x, retrieval_cache=rcache,
            memory_y=exp.memory_y, valid_mask=cand_mask, key_bank=exp.key_bank,
            memory_x_last=exp.memory_x_last, target_y=batch_y)
        loss = exp._loss(y_final, y_base, y_ret, batch_y, debug, valid_query)
        loss.backward()
        optimizer.step()
        tot_loss += float(loss.detach())
        n += 1
    return tot_loss / max(n, 1)


@torch.no_grad()
def eval_epoch(exp, model, loader, cache, start_to_row, device):
    """final/base/projected-retrieval-branch MSE come from a real forward
    pass (`model.forward_from_retrieval_values`); `raw_retrieval_aggregate_mse`
    is a SEPARATE diagnostic computed directly from the cache (delta
    restored to absolute via `restore_absolute`), never through the mixer
    -- it measures the retrieved aggregate BEFORE any Stage-2 projection,
    matching what Stage-1's own free_running_aggregate_future_mse measures.
    Counterfactuals (spec section 8): lambda=0 is exactly `base_mse` (the
    same base_head forward every batch already computes); lambda=1 is
    exactly `projected_retrieval_branch_mse` (the same y_ret every batch
    already computes) -- both reused directly rather than re-run, since
    they are literally the same quantities under a linear/gated fusion."""
    model.train(False)
    se_final, ae_final, se_base, ae_base, se_proj, ae_proj, se_raw, ae_raw, cnt = (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    gate_vals = []
    for batch_x, batch_y, batch_start_idx in loader:
        batch_x, batch_y, batch_start_idx = exp._move_batch(batch_x, batch_y, batch_start_idx)
        cand_mask, counts = exp._candidate_mask(batch_start_idx)
        valid_query = counts.to(batch_x.device) > 0
        if bool((~valid_query).all()):
            continue
        rcache = lookup_batch(cache, start_to_row, batch_start_idx, batch_x.device)
        y_final, y_base, y_ret, beta, lam, debug = model.forward_from_retrieval_values(
            rcache['relation_outputs'], batch_x=batch_x, retrieval_cache=rcache,
            memory_y=exp.memory_y, valid_mask=cand_mask, key_bank=exp.key_bank,
            memory_x_last=exp.memory_x_last, target_y=batch_y)

        idx = [start_to_row[int(s)] for s in batch_start_idx.tolist()]
        cache_offset = cache['query_offset'][idx].to(device)  # [B, channels]
        raw_abs = restore_absolute(rcache['relation_outputs'][:, :, 0, :], cache_offset)  # [B,C,H]
        raw_abs = raw_abs.permute(0, 2, 1)  # -> [B, H, C], matches batch_y layout

        yf, yb, yp, yr, yt = (y_final[valid_query], y_base[valid_query], y_ret[valid_query],
                              raw_abs[valid_query], batch_y[valid_query])
        se_final += float(((yf - yt) ** 2).sum()); ae_final += float((yf - yt).abs().sum())
        se_base += float(((yb - yt) ** 2).sum()); ae_base += float((yb - yt).abs().sum())
        se_proj += float(((yp - yt) ** 2).sum()); ae_proj += float((yp - yt).abs().sum())
        se_raw += float(((yr - yt) ** 2).sum()); ae_raw += float((yr - yt).abs().sum())
        cnt += yt.numel()
        if lam is not None:
            gate_vals.append(lam[valid_query].detach().cpu().flatten())
    gate_cat = torch.cat(gate_vals) if gate_vals else torch.zeros(1)
    out = {
        'final_mse': se_final / max(cnt, 1), 'final_mae': ae_final / max(cnt, 1),
        'base_mse': se_base / max(cnt, 1), 'base_mae': ae_base / max(cnt, 1),
        'projected_retrieval_branch_mse': se_proj / max(cnt, 1),
        'projected_retrieval_branch_mae': ae_proj / max(cnt, 1),
        'raw_retrieval_aggregate_mse': se_raw / max(cnt, 1),
        'raw_retrieval_aggregate_mae': ae_raw / max(cnt, 1),
        'counterfactual_lambda0_mse': se_base / max(cnt, 1),   # == base_mse, reused not re-run
        'counterfactual_lambda1_mse': se_proj / max(cnt, 1),   # == projected_retrieval_branch_mse
    }
    out.update(gate_distribution(gate_cat))
    return out


def compute_cache_restored_fragg(exp, loader, cache, start_to_row, device):
    """FR-Agg computed PURELY from the cache (delta restored to absolute)
    against real batch_y -- no model forward at all. This is the exact
    same formula Stage-1's own free_running_aggregate_future_mse uses;
    used as the pre-training cross-check gate (spec section 7)."""
    se, cnt = 0.0, 0
    for batch_x, batch_y, batch_start_idx in loader:
        batch_x, batch_y, batch_start_idx = exp._move_batch(batch_x, batch_y, batch_start_idx)
        cand_mask, counts = exp._candidate_mask(batch_start_idx)
        valid_query = counts.to(device) > 0
        if bool((~valid_query).all()):
            continue
        idx = [start_to_row[int(s)] for s in batch_start_idx.tolist()]
        cache_offset = cache['query_offset'][idx].to(device)
        delta = cache['relation_outputs'][idx].to(device)[:, :, 0, :]
        raw_abs = restore_absolute(delta, cache_offset).permute(0, 2, 1)
        yt = batch_y[valid_query]
        yr = raw_abs[valid_query]
        se += float(((yr - yt) ** 2).sum())
        cnt += yt.numel()
    return se / max(cnt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True)
    ap.add_argument('--arm_name', required=True)
    ap.add_argument('--stage2_host', required=True)
    ap.add_argument('--cache_dir', required=True)
    ap.add_argument('--stage1_retrieval_metrics_json', default=None,
                    help='results/track_a_factorial_e2e/<cell>/retrieval_metrics_<arm>.json '
                         '-- used ONLY for the pre-training FR-Agg cross-check, spec section 7')
    ap.add_argument('--fragg_tolerance', type=float, default=0.02,
                    help='max allowed |cache-restored FR-Agg - Stage-1 reported FR-Agg| '
                         'before aborting as [ISSUE]')
    ap.add_argument('--checkpoints', default='checkpoints/track_a_set_normregret_control01_stage2')
    ap.add_argument('--out_dir', default='results/TRACK-A-SET-NORMREGRET-CONTROL01/stage2')
    ap.add_argument('--shared_init_out', default=None)
    ap.add_argument('--shared_init_in', default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--train_epochs', type=int, default=10)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--limit_batches', type=int, default=0)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp, args, model, host_ck = build_fresh_stage2(cli.stage2_host, cli.seed)
    model.to(device)
    exp._ensure_memory()
    exp._build_key_bank(force=True)

    frozen_shas_before = freeze_retrieval_submodules(model)

    if cli.shared_init_in:
        blob = torch.load(cli.shared_init_in, map_location='cpu')
        model.load_state_dict(blob['model_state_dict'])
        got = state_sha(model.state_dict())
        if got != blob['sha256']:
            raise SystemExit(f'[ABORT] shared Stage-2 init SHA mismatch: {got} != {blob["sha256"]}')
        init_sha = got
    else:
        init_sha = state_sha(model.state_dict())
        if cli.shared_init_out:
            Path(cli.shared_init_out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({'model_state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                       'sha256': init_sha}, cli.shared_init_out)

    trainable_names = sorted(n for n, p in model.named_parameters() if p.requires_grad)
    optimizer = exp._select_optimizer()

    cache_dir = Path(cli.cache_dir)
    train_cache, train_lut = load_cache_as_lookup(cache_dir / 'train.pt')
    val_cache, val_lut = load_cache_as_lookup(cache_dir / 'val.pt')
    test_cache, test_lut = load_cache_as_lookup(cache_dir / 'test.pt')

    _, train_loader = exp._get_data(flag='train', shuffle=True)
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)

    cell_dir = Path(cli.out_dir) / cli.cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints) / cli.cell / cli.arm_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- Stage-1 FR-Agg cross-check (spec section 7) -- MUST pass before
    # any Stage-2 training starts ----
    cache_restored_fragg = compute_cache_restored_fragg(exp, val_loader, val_cache, val_lut, device)
    stage1_fragg = None
    if cli.stage1_retrieval_metrics_json:
        rm = json.loads(Path(cli.stage1_retrieval_metrics_json).read_text())
        stage1_fragg = rm.get('best_val_free_running_aggregate_future_mse')
        if stage1_fragg is not None:
            diff = abs(cache_restored_fragg - float(stage1_fragg))
            print(f'[stage2_retrain01] FR-Agg cross-check: stage1={stage1_fragg:.6f} '
                 f'cache_restored={cache_restored_fragg:.6f} diff={diff:.6f} '
                 f'tolerance={cli.fragg_tolerance}')
            if diff > cli.fragg_tolerance:
                raise SystemExit(
                    f'[ISSUE][ABORT] {cli.cell}/{cli.arm_name}: cache-restored FR-Agg '
                    f'({cache_restored_fragg:.6f}) does not match Stage-1 reported FR-Agg '
                    f'({stage1_fragg:.6f}), diff={diff:.6f} > tolerance={cli.fragg_tolerance}. '
                    'Refusing to start Stage-2 training on a cache that may still be wrong.')
        else:
            print('[stage2_retrain01] WARNING: stage1 retrieval_metrics.json has no '
                 'best_val_free_running_aggregate_future_mse -- cross-check skipped.')
    else:
        print(f'[stage2_retrain01] cache-restored val FR-Agg (no Stage-1 reference supplied to '
             f'cross-check against) = {cache_restored_fragg:.6f}')

    stage1_ckpt_path = train_cache.get('stage1_checkpoint_path')
    stage1_ckpt_hash = train_cache.get('stage1_checkpoint_hash')
    cache_config_hash = train_cache.get('config_hash')

    arm_dir = cell_dir / cli.arm_name
    arm_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = {
        'exp': 'TRACK-A-SET-NORMREGRET-CONTROL01', 'cell': cli.cell, 'arm': cli.arm_name,
        'seed': cli.seed, 'stage2_init_sha256': init_sha,
        'frozen_submodule_sha256_before_training': frozen_shas_before,
        'trainable_parameter_names': trainable_names,
        'trainable_parameter_count': sum(p.numel() for n, p in model.named_parameters() if p.requires_grad),
        'stage2_host': cli.stage2_host, 'stage2_host_sha256': _file_sha256(cli.stage2_host),
        'cache_dir': str(cache_dir),
        'stage1_checkpoint_path': stage1_ckpt_path, 'stage1_checkpoint_hash': stage1_ckpt_hash,
        'cache_config_hash': cache_config_hash,
        'cache_restored_val_fragg': cache_restored_fragg, 'stage1_reported_fragg': stage1_fragg,
        'fragg_tolerance': cli.fragg_tolerance,
        'pid': __import__('os').getpid(), 'start_time': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    (cell_dir / f'config_fingerprint_{cli.arm_name}.json').write_text(json.dumps(fingerprint, indent=2))
    (arm_dir / 'config.json').write_text(json.dumps(fingerprint, indent=2))
    (arm_dir / 'config_hash.txt').write_text(str(cache_config_hash))
    (arm_dir / 'stage1_checkpoint_manifest.json').write_text(json.dumps(
        {'stage1_checkpoint_path': stage1_ckpt_path, 'stage1_checkpoint_hash': stage1_ckpt_hash,
         'stage2_host': cli.stage2_host, 'stage2_host_sha256': fingerprint['stage2_host_sha256']}, indent=2))
    (arm_dir / 'trainable_parameters.json').write_text(json.dumps(
        {'trainable_parameter_names': trainable_names,
         'trainable_parameter_count': fingerprint['trainable_parameter_count'],
         'frozen_submodules': sorted(frozen_shas_before.keys())}, indent=2))
    cache_manifest_src = cache_dir / 'cache_manifest.json'
    if cache_manifest_src.exists():
        (arm_dir / 'cache_manifest.json').write_text(cache_manifest_src.read_text())
    print(f'[stage2_retrain01] {cli.cell}/{cli.arm_name} init_sha={init_sha[:16]} '
         f'trainable_params={fingerprint["trainable_parameter_count"]}')

    training_log_lines = []
    t0 = time.time()
    try:
        best = {'val': float('inf'), 'epoch': -1}
        epoch_rows = []
        for epoch in range(1, cli.train_epochs + 1):
            tr_loss = train_epoch(exp, model, optimizer, train_loader, train_cache, train_lut, device)
            va = eval_epoch(exp, model, val_loader, val_cache, val_lut, device)
            row = {'epoch': epoch, 'train_loss': tr_loss, **{f'val_{k}': v for k, v in va.items()}}
            epoch_rows.append(row)
            payload = {'model_state_dict': model.state_dict(), 'args': vars(args),
                      'epoch': epoch, 'fingerprint': fingerprint, 'val_final_mse': va['final_mse']}
            torch.save(payload, ckpt_dir / f'checkpoint_epoch{epoch}.pth')
            if va['final_mse'] < best['val']:
                best = {'val': va['final_mse'], 'epoch': epoch}
                torch.save(payload, ckpt_dir / 'checkpoint.pth')
            line = (f"[stage2_retrain01] {cli.arm_name} epoch {epoch} train_loss={tr_loss:.5f} "
                   f"val_final_mse={va['final_mse']:.6f} gate_mean={va['gate_mean']:.3f}")
            print(line)
            training_log_lines.append(line)
            if epoch - best['epoch'] >= cli.patience:
                stop_line = f'[stage2_retrain01] early stop at epoch {epoch} (best={best["epoch"]})'
                print(stop_line)
                training_log_lines.append(stop_line)
                break

        frozen_shas_after = freeze_retrieval_submodules(model)  # re-check post-training
        mismatched = {k: v for k, v in frozen_shas_after.items() if frozen_shas_before.get(k) != v}
        if mismatched:
            raise SystemExit(f'[ISSUE][ABORT] frozen submodule(s) changed during training: {mismatched}')

        bl = torch.load(ckpt_dir / 'checkpoint.pth', map_location=device)
        model.load_state_dict(bl['model_state_dict'])
        te_best = eval_epoch(exp, model, test_loader, test_cache, test_lut, device)

        last_epoch_ckpt = ckpt_dir / f"checkpoint_epoch{epoch_rows[-1]['epoch']}.pth"
        fl = torch.load(last_epoch_ckpt, map_location=device)
        model.load_state_dict(fl['model_state_dict'])
        te_final = eval_epoch(exp, model, test_loader, test_cache, test_lut, device)

        with open(cell_dir / f'epoch_metrics_{cli.arm_name}.csv', 'w', newline='') as fh:
            keys = sorted({k for r in epoch_rows for k in r})
            w = csv.DictWriter(fh, fieldnames=['epoch'] + [k for k in keys if k != 'epoch'])
            w.writeheader()
            for r in epoch_rows:
                w.writerow(r)

        summary = {'exp': 'TRACK-A-SET-NORMREGRET-CONTROL01', 'cell': cli.cell, 'arm': cli.arm_name,
                  'best_epoch': best['epoch'], 'best_val_final_mse': best['val'],
                  'test_best': te_best, 'test_final': te_final,
                  'frozen_submodule_sha256_matched': True,
                  'wall_clock_seconds': time.time() - t0, 'fingerprint': fingerprint}
        (cell_dir / f'stage2_metrics_{cli.arm_name}.json').write_text(json.dumps(summary, indent=2))
        (cell_dir / f'DONE_{cli.arm_name}.marker').write_text(json.dumps({'done': True, 'best_epoch': best['epoch']}))

        torch.save(bl, arm_dir / 'best_checkpoint.pt')
        torch.save(fl, arm_dir / 'final_checkpoint.pt')
        (arm_dir / 'training_log.txt').write_text('\n'.join(training_log_lines) + '\n')
        (arm_dir / 'metrics_best.json').write_text(json.dumps(
            {'epoch': best['epoch'], 'val': {k: v for k, v in epoch_rows[best['epoch'] - 1].items()},
             'test': te_best}, indent=2))
        (arm_dir / 'metrics_final.json').write_text(json.dumps(
            {'epoch': epoch_rows[-1]['epoch'], 'val': epoch_rows[-1], 'test': te_final}, indent=2))
        (arm_dir / 'lambda_distribution.json').write_text(json.dumps(
            {k: v for k, v in te_best.items() if k.startswith('gate_')}, indent=2))
        (arm_dir / 'counterfactual_metrics.json').write_text(json.dumps(
            {'counterfactual_lambda0_mse': te_best['counterfactual_lambda0_mse'],
             'counterfactual_lambda1_mse': te_best['counterfactual_lambda1_mse'],
             'trained_final_mse': te_best['final_mse'],
             'note': 'lambda0 == base_mse (retrieval branch off, reused not re-run); '
                     'lambda1 == projected_retrieval_branch_mse (base branch off, reused not re-run)'},
            indent=2))
        (arm_dir / 'DONE').write_text(json.dumps({'done': True, 'best_epoch': best['epoch'],
                                                   'end_time': time.strftime('%Y-%m-%dT%H:%M:%S')}))
        print(f"[stage2_retrain01] done. {cli.cell}/{cli.arm_name} best_epoch={best['epoch']} "
             f"test_best_final_mse={te_best['final_mse']:.6f}")
    except BaseException as e:
        import traceback
        (arm_dir / 'FAILED.json').write_text(json.dumps(
            {'cause': str(e), 'traceback': traceback.format_exc(),
             'end_time': time.strftime('%Y-%m-%dT%H:%M:%S')}, indent=2))
        raise


if __name__ == '__main__':
    main()
