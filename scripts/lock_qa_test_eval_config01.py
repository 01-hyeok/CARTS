#!/usr/bin/env python3
"""TRACK-P100-QA-TEST-EVALUATION01 -- LOCKED_CONFIG.json builder.

Append-only, idempotent. Each invocation scans for VALIDATION-selected
combos (both protocols) not yet present in the existing
`LOCKED_CONFIG.json` and adds them -- freezing checkpoint path,
checkpoint SHA256, base checkpoint SHA256, candidate pool hash,
selected epoch, and the recorded VALIDATION MSE for each, at the
moment they are added. Already-locked entries are NEVER re-hashed or
modified on a later run, even if the underlying files somehow changed
-- this is what makes "locked" mean something.

Two protocols, kept in separate registries (per the user's explicit
instruction: never average Original and RNG-corrected together):

  original_e2e      -- results/TRACK-P100-SOFTTOPK-QUALITY-AWARE01
                        (has the confirmed A1 batch-order confound in
                        every QA run -- locked and later evaluated
                        anyway, per explicit user approval, but every
                        downstream artifact must carry the
                        ORIGINAL_RNG_CONFOUNDED label).
  rng_corrected_e2e -- results/TRACK-P100-QA-CAUSAL-VALIDATION01/rng_corrected
                        ONLY locked for a (cell, method) pair if
                        `verify_rngfixed_pairs01.verify_pair` passes
                        for BOTH arms -- a pair that fails verification
                        is recorded under `rng_corrected_issues`
                        instead and TEST is never opened for it.

This script never trains, never selects a checkpoint by any Test
metric, and never constructs a `flag='test'` DataLoader itself (it
only hashes files).
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.qa_test_eval_paths01 import (
    METHOD_TO_NQV, all_cells, base_checkpoint, candidate_pool_cache, cell_name, reference_ckpt,
)
from scripts.verify_rngfixed_pairs01 import verify_pair

METHODS = ('V0', 'V1', 'V2', 'V5', 'Hard', 'Joint')
ARMS = ('NOQA', 'QA')

ORIGINAL_ROOT = REPO_ROOT / 'results' / 'TRACK-P100-SOFTTOPK-QUALITY-AWARE01'
RNG_ROOT = REPO_ROOT / 'results' / 'TRACK-P100-QA-CAUSAL-VALIDATION01' / 'rng_corrected'
RNG_REPRESENTATIVE_CELLS_METHODS = [
    ('ETTh1_192', 'V5'), ('ETTh1_336', 'Hard'), ('Weather_720', 'V5'),
    ('Weather_336', 'Joint'), ('Solar_96', 'V5'),
]


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, check=True,
                             capture_output=True, text=True).stdout.strip()
    except Exception:
        return 'UNKNOWN'


def eval_code_sha256():
    return sha256_of_file(REPO_ROOT / 'scripts' / 'evaluate_locked_test01.py')


def _load_or_init(out_path):
    if out_path.exists():
        return json.loads(out_path.read_text())
    return {
        'experiment': 'TRACK-P100-QA-TEST-EVALUATION01',
        'protocol_label': 'EXPLORATORY_TEST_EVALUATION',
        'note': 'Approved by explicit user decision to open TEST before the original track\'s '
                'GO/STOP decision and before the causal-validation track\'s Phase C/D/E concluded. '
                'TEST is therefore NOT an untouched final holdout for this dataset going forward -- '
                'any future model development on these datasets must not claim an independent final '
                'evaluation. All results here are EXPLORATORY, not confirmatory.',
        'git_commit': git_commit(),
        'locked_timestamp_utc_first': __import__('datetime').datetime.utcnow().isoformat() + 'Z',
        'datasets': ['ETTh1', 'Weather', 'Solar'],
        'horizons': [96, 192, 336, 720],
        'method_list': list(METHODS),
        'arms': list(ARMS),
        'test_accessed': True,
        'checkpoint_selection_rule': 'min VALIDATION final-forecast MSE (Y_hat vs Y), never TEST, fixed before any TEST read',
        'eval_code_sha256': eval_code_sha256(),
        'val_parity_tolerance_abs': 1e-3,
        'val_parity_tolerance_rel': 0.005,
        'combos': {},
        'rng_corrected_issues': {},
    }


def _lock_original_combo(cell, method, arm):
    sel_path = ORIGINAL_ROOT / cell / method / arm / 'checkpoint_selection.json'
    if not sel_path.exists():
        return None
    sel = json.loads(sel_path.read_text())
    ds, h = cell.rsplit('_', 1)
    h = int(h)
    ckpt_path = sel['checkpoint_path']
    base_ckpt = base_checkpoint(ds, h)
    pool_dir = candidate_pool_cache(ds, h)
    pool_test_file = Path(pool_dir) / 'test.pt'
    return {
        'protocol': 'original_e2e',
        'cell': cell, 'method': method, 'arm': arm,
        'dataset': ds, 'horizon': h,
        'num_query_views': METHOD_TO_NQV.get(method, 5),
        'checkpoint_path': ckpt_path,
        'checkpoint_sha256': sha256_of_file(ckpt_path),
        'base_checkpoint_path': base_ckpt,
        'base_checkpoint_sha256': sha256_of_file(base_ckpt),
        'candidate_pool_cache': pool_dir,
        'candidate_pool_test_hash': sha256_of_file(pool_test_file) if pool_test_file.exists() else None,
        'selected_epoch': sel['epoch'],
        'val_mse_recorded': sel['val_mse'],
        'rng_status': 'ORIGINAL_RNG_CONFOUNDED' if arm == 'QA' else 'NOQA_UNAFFECTED_BY_RNG_CONFOUND',
    }


def _lock_rng_combo(cell, method, arm):
    sel_path = RNG_ROOT / cell / method / arm / 'checkpoint_selection.json'
    if not sel_path.exists():
        return None
    sel = json.loads(sel_path.read_text())
    ds, h = cell.rsplit('_', 1)
    h = int(h)
    ckpt_path = sel['checkpoint_path']
    base_ckpt = base_checkpoint(ds, h)
    pool_dir = candidate_pool_cache(ds, h)
    pool_test_file = Path(pool_dir) / 'test.pt'
    return {
        'protocol': 'rng_corrected_e2e',
        'cell': cell, 'method': method, 'arm': arm,
        'dataset': ds, 'horizon': h,
        'num_query_views': METHOD_TO_NQV.get(method, 5),
        'checkpoint_path': ckpt_path,
        'checkpoint_sha256': sha256_of_file(ckpt_path),
        'base_checkpoint_path': base_ckpt,
        'base_checkpoint_sha256': sha256_of_file(base_ckpt),
        'candidate_pool_cache': pool_dir,
        'candidate_pool_test_hash': sha256_of_file(pool_test_file) if pool_test_file.exists() else None,
        'selected_epoch': sel['epoch'],
        'val_mse_recorded': sel['val_mse'],
        'rng_status': 'RNG_PARITY_VERIFIED',
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_json', required=True)
    cli = ap.parse_args()

    out_path = Path(cli.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = _load_or_init(out_path)

    added = []
    for ds, h in all_cells():
        cell = cell_name(ds, h)
        for method in METHODS:
            for arm in ARMS:
                key = f'original_e2e/{cell}/{method}/{arm}'
                if key in cfg['combos']:
                    continue
                combo = _lock_original_combo(cell, method, arm)
                if combo is not None:
                    cfg['combos'][key] = combo
                    added.append(key)

    for cell, method in RNG_REPRESENTATIVE_CELLS_METHODS:
        noqa_key = f'rng_corrected_e2e/{cell}/{method}/NOQA'
        qa_key = f'rng_corrected_e2e/{cell}/{method}/QA'
        if noqa_key in cfg['combos'] or qa_key in cfg['combos']:
            continue  # already locked (both arms locked together, atomically)
        issue_key = f'{cell}/{method}'
        if issue_key in cfg['rng_corrected_issues']:
            continue  # already recorded as a failed-verification issue, don't re-check
        noqa_sel = RNG_ROOT / cell / method / 'NOQA' / 'checkpoint_selection.json'
        qa_sel = RNG_ROOT / cell / method / 'QA' / 'checkpoint_selection.json'
        if not (noqa_sel.exists() and qa_sel.exists()):
            continue  # not trained yet -- wait, per explicit instruction, don't lock partial pairs
        ok, report = verify_pair(RNG_ROOT, cell, method)
        if not ok:
            cfg['rng_corrected_issues'][issue_key] = {
                'status': 'ISSUE_RNG_VERIFICATION_FAILED', 'report': report,
                'note': 'TEST was NOT opened for this pair -- RNG parity verification failed.',
            }
            added.append(f'ISSUE:{issue_key}')
            continue
        noqa_combo = _lock_rng_combo(cell, method, 'NOQA')
        qa_combo = _lock_rng_combo(cell, method, 'QA')
        cfg['combos'][noqa_key] = noqa_combo
        cfg['combos'][qa_key] = qa_combo
        added.append(noqa_key)
        added.append(qa_key)

    out_path.write_text(json.dumps(cfg, indent=2))
    lock_sha256 = sha256_of_file(out_path)
    print(f'[lock_qa_test_eval_config] wrote {out_path} (sha256={lock_sha256})')
    print(f'[lock_qa_test_eval_config] newly added this run: {len(added)}')
    for a in added:
        print(f'  + {a}')
    print(f'[lock_qa_test_eval_config] total combos locked: {len(cfg["combos"])}, '
         f'rng issues recorded: {len(cfg["rng_corrected_issues"])}')


if __name__ == '__main__':
    main()
