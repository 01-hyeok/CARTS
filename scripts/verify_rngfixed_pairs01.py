#!/usr/bin/env python3
"""TRACK-P100-QA-CAUSAL-VALIDATION01 -- Phase A1/B verification.

For one (cell, method) pair of RNG-corrected runs (produced by
`train_softtopk_e2e_{v,hard,joint}02_rngfixed.py`), asserts:

  1. `init_hash` (shared encoder) is identical between NOQA and QA.
  2. `slot_heads_init_hash` is identical between NOQA and QA (when the
     method has slot heads -- V0 has none).
  3. `router_init_hash` is identical between NOQA and QA (Joint only).
  4. `batch_order_sha256` is identical between NOQA and QA at EVERY
     epoch (not just epoch 1) -- the actual fix this track exists to
     verify.

Exits non-zero with `[ISSUE]` on any mismatch, so this can gate whether
a pair's result is trusted. Reads ONLY `config.json`/`epoch_metrics.json`
-- never touches TEST, never touches `checkpoint_selection.json`'s
VALIDATION MSE values (this script checks RNG hygiene only, not model
quality).
"""
import argparse
import json
import sys
from pathlib import Path


def _load(results_root, cell, method, arm):
    d = Path(results_root) / cell / method / arm
    config = json.loads((d / 'config.json').read_text())
    epochs = json.loads((d / 'epoch_metrics.json').read_text())
    return config, epochs


def verify_pair(results_root, cell, method):
    """Returns (ok: bool, report: dict)."""
    noqa_cfg, noqa_epochs = _load(results_root, cell, method, 'NOQA')
    qa_cfg, qa_epochs = _load(results_root, cell, method, 'QA')

    problems = []

    if noqa_cfg['init_hash'] != qa_cfg['init_hash']:
        problems.append(f"init_hash mismatch: NOQA={noqa_cfg['init_hash']} QA={qa_cfg['init_hash']}")

    noqa_sh = noqa_cfg.get('slot_heads_init_hash')
    qa_sh = qa_cfg.get('slot_heads_init_hash')
    if noqa_sh != qa_sh:
        problems.append(f"slot_heads_init_hash mismatch: NOQA={noqa_sh} QA={qa_sh}")

    noqa_rt = noqa_cfg.get('router_init_hash')
    qa_rt = qa_cfg.get('router_init_hash')
    if noqa_rt != qa_rt:
        problems.append(f"router_init_hash mismatch: NOQA={noqa_rt} QA={qa_rt}")

    if len(noqa_epochs) != len(qa_epochs):
        problems.append(f'epoch count mismatch: NOQA={len(noqa_epochs)} QA={len(qa_epochs)}')
    else:
        for i, (ne, qe) in enumerate(zip(noqa_epochs, qa_epochs), start=1):
            if ne['batch_order_sha256'] != qe['batch_order_sha256']:
                problems.append(
                    f'epoch {i}: batch_order_sha256 mismatch '
                    f"NOQA={ne['batch_order_sha256'][:16]} QA={qe['batch_order_sha256'][:16]}")

    report = {
        'cell': cell, 'method': method, 'ok': len(problems) == 0, 'problems': problems,
        'init_hash_match': noqa_cfg['init_hash'] == qa_cfg['init_hash'],
        'slot_heads_init_hash_match': noqa_sh == qa_sh,
        'router_init_hash_match': noqa_rt == qa_rt,
        'epochs_checked': min(len(noqa_epochs), len(qa_epochs)),
        'all_epoch_batch_orders_match': all(
            ne['batch_order_sha256'] == qe['batch_order_sha256']
            for ne, qe in zip(noqa_epochs, qa_epochs)
        ) if len(noqa_epochs) == len(qa_epochs) else False,
    }
    return len(problems) == 0, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_root', required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--method', required=True)
    ap.add_argument('--out_json', default=None)
    cli = ap.parse_args()

    ok, report = verify_pair(cli.results_root, cli.cell, cli.method)
    print(json.dumps(report, indent=2))
    if cli.out_json:
        Path(cli.out_json).write_text(json.dumps(report, indent=2))
    if not ok:
        print(f'[ISSUE] {cli.cell}/{cli.method}: RNG parity check FAILED -- {report["problems"]}',
             file=sys.stderr)
        sys.exit(1)
    print(f'[OK] {cli.cell}/{cli.method}: NOQA/QA init hashes and all {report["epochs_checked"]} '
         f'epochs\' batch_order_sha256 match exactly.')


if __name__ == '__main__':
    main()
