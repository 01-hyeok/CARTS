#!/usr/bin/env python3
"""TRACK-W-CHECKPOINT-CRITERION-CORRECTION01 -- CASE A re-selection for
B Host-Free (`TRACK-A-SETORACLE-HOSTFREE01`). For a (cell, arm) whose
FULL intended epoch trajectory already exists on disk (no early-stop
truncation under the OLD criterion), this does NOT retrain anything:
it (1) reads the already-logged per-epoch `val_tf_tf_choice_ce` from
the OLD run's `epoch_metrics_{arm}.csv`, (2) picks the argmin epoch,
(3) reloads that EXACT epoch's saved checkpoint and independently
RECOMPUTES `val_tf_choice_ce` via the same `teacher_forced_diagnostics`
used at training time (cross-check against the logged CSV value), (4)
copies that checkpoint into TRACK-W's own tree as the corrected
`checkpoint.pth`, and (5) writes a corrected `retrieval_metrics_{arm}
.json` + `DONE_{arm}.marker` in TRACK-W's own results tree, in the
exact schema `build_setoracle_hostfree01_retrieval_cache.py` (reused
UNMODIFIED downstream) expects.

No training math touched anywhere -- pure re-selection + verification.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.SequentialSetRetriever import SetConditioner
from scripts.train_factorial_e2e01 import eval_epoch, state_sha
from scripts.train_margutil01 import build_experiment
from scripts.train_setoracle_hostfree01 import (ARMS, UniformHost,
                                                 _materialize_loader,
                                                 teacher_forced_diagnostics)

OLD_ROOT_RESULTS = REPO_ROOT / 'results/TRACK-A-SETORACLE-HOSTFREE01'
OLD_ROOT_CKPT = REPO_ROOT / 'checkpoints/track_a_setoracle_hostfree01'


def _file_sha256(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True)
    ap.add_argument('--arm_name', required=True, choices=list(ARMS))
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--out_results_root', default='results/TRACK-W-CHECKPOINT-CRITERION-CORRECTION01/B_hostfree_stage1')
    ap.add_argument('--out_ckpt_root', default='checkpoints/track_w_checkpoint_criterion_correction01/B_hostfree_stage1')
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    old_cell_dir = OLD_ROOT_RESULTS / cli.cell
    old_ckpt_dir = OLD_ROOT_CKPT / cli.cell / cli.arm_name
    csv_path = old_cell_dir / f'epoch_metrics_{cli.arm_name}.csv'
    rows = list(csv.DictReader(open(csv_path)))
    n_epochs_found = len(rows)
    print(f'[reselect_w] {cli.cell}/{cli.arm_name}: {n_epochs_found} epochs found in {csv_path}')

    # pick argmin epoch by the already-logged val_tf_tf_choice_ce
    scored = [(int(r['epoch']), float(r['val_tf_tf_choice_ce'])) for r in rows]
    new_epoch, new_val_ce_logged = min(scored, key=lambda t: t[1])
    old_val_fr = {int(r['epoch']): float(r['val_free_running_aggregate_future_mse']) for r in rows}
    old_epoch, old_val_fr_value = min(old_val_fr.items(), key=lambda t: t[1])
    print(f'[reselect_w] {cli.cell}/{cli.arm_name}: OLD best_epoch={old_epoch} '
         f'(val_fr_agg={old_val_fr_value:.6f}) -> NEW best_epoch={new_epoch} '
         f'(val_tf_choice_ce={new_val_ce_logged:.6f}, logged)')

    ckpt_path = old_ckpt_dir / f'checkpoint_epoch{new_epoch}.pth'
    assert ckpt_path.exists(), f'[ISSUE][ABORT] {ckpt_path} missing -- CASE A assumption violated'
    ckpt = torch.load(ckpt_path, map_location=device)
    assert int(ckpt['epoch']) == new_epoch

    # independent re-verification: reload + recompute val_tf_choice_ce fresh
    args_dict = ckpt['args']
    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.pred_len,
        'batch_size': cli.batch_size, 'seed': int(args_dict.get('seed', 0)), 'top_k': cli.top_k,
        'tau_topk': 0.1,
    })
    exp._ensure_memory()
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(device)
    d_model = int(args.d_model)
    sc = SetConditioner(d_model).to(device)
    sc.load_state_dict(ckpt['set_conditioner_state_dict'])
    sc.eval()
    channels = list(range(int(args.enc_in)))
    host = UniformHost(cli.top_k, tau_topk=1.0)

    class Cli:
        pass
    diag_cli = Cli()
    diag_cli.target = ARMS[cli.arm_name]
    diag_cli.prefix_policy = 'tf'
    diag_cli.tau_choice = 0.1
    diag_cli.tau_topk = 1.0
    diag_cli.top_k = cli.top_k
    diag_cli.chunk_size = cli.chunk_size
    diag_cli.resolved_oracle_compute = None

    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)
    val_x, val_y, val_start = _materialize_loader(val_loader)
    test_x, test_y, test_start = _materialize_loader(test_loader)

    val_tf, _ = teacher_forced_diagnostics(exp, args, host, model, sc, None, diag_cli,
                                           val_x, val_y, val_start, channels, device,
                                           batch_size=cli.batch_size)
    recomputed_val_ce = val_tf['tf_choice_ce']
    match = abs(recomputed_val_ce - new_val_ce_logged) < 1e-4
    print(f'[reselect_w] {cli.cell}/{cli.arm_name}: recomputed val_tf_choice_ce={recomputed_val_ce:.6f} '
         f'vs logged={new_val_ce_logged:.6f} match={match}')
    assert match, '[ISSUE][ABORT] recomputed val_tf_choice_ce does not match logged CSV value'

    te_fr, _ = eval_epoch(exp, args, host, model, sc, None, diag_cli, test_loader, channels, device)
    te_tf, _ = teacher_forced_diagnostics(exp, args, host, model, sc, None, diag_cli,
                                          test_x, test_y, test_start, channels, device,
                                          batch_size=cli.batch_size)

    out_cell_dir = Path(cli.out_results_root) / cli.cell
    out_cell_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt_dir = Path(cli.out_ckpt_root) / cli.cell / cli.arm_name
    out_ckpt_dir.mkdir(parents=True, exist_ok=True)

    corrected_fp = dict(ckpt['fingerprint'])
    corrected_fp['checkpoint_criterion'] = 'min val_tf_choice_ce (teacher-forced Choice CE, CORRECTED -- TRACK-W)'
    corrected_fp['correction_note'] = ('OLD criterion was min val_free_running_aggregate_future_mse; '
                                       'CORRECTED to match training objective (hard_choice_ce, teacher-forced).')
    ckpt['fingerprint'] = corrected_fp
    ckpt_out_path = out_ckpt_dir / 'checkpoint.pth'
    torch.save(ckpt, ckpt_out_path)

    summary = {
        'exp': 'TRACK-W-CHECKPOINT-CRITERION-CORRECTION01', 'cell': cli.cell, 'arm': cli.arm_name,
        'target': ARMS[cli.arm_name], 'best_epoch': new_epoch,
        'best_val_tf_choice_ce': recomputed_val_ce,
        'old_best_epoch': old_epoch, 'old_best_val_free_running_aggregate_future_mse': old_val_fr_value,
        'case': 'A (re-selection, no retraining -- full epoch trajectory already existed)',
        'test_free_running_aggregate_future_mse': te_fr['free_running_aggregate_future_mse'],
        'test_free_running_internal': te_fr, 'test_tf': te_tf,
        'encoder_init_sha256': ckpt['fingerprint']['encoder_init_sha256'],
        'set_conditioner_init_sha256': ckpt['fingerprint']['set_conditioner_init_sha256'],
        'checkpoint_sha256': _file_sha256(ckpt_out_path),
        'source_old_checkpoint': str(ckpt_path), 'source_old_checkpoint_sha256': _file_sha256(ckpt_path),
        'fingerprint': corrected_fp,
    }
    (out_cell_dir / f'retrieval_metrics_{cli.arm_name}.json').write_text(json.dumps(summary, indent=2, default=str))
    (out_cell_dir / f'DONE_{cli.arm_name}.marker').write_text(json.dumps({'done': True, 'best_epoch': new_epoch}))
    print(f'[reselect_w] done. {cli.cell}/{cli.arm_name} CASE A corrected: '
         f'old_epoch={old_epoch} -> new_epoch={new_epoch}, '
         f'test_fr_agg={te_fr["free_running_aggregate_future_mse"]:.6f} (old test metric in old report), '
         f'test_tf_acc={te_tf["oracle_action_acc"]:.4f}')


if __name__ == '__main__':
    main()
