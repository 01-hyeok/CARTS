#!/usr/bin/env python3
"""TRACK-A-SET-LOSS-CONTROL02 -- cross-arm batch-order hash gate (spec
section 5): "네 arm의 epoch별 batch-order hash가 완전히 동일하지 않으면
즉시 [ISSUE]로 중단하라." Reads each arm's
`batch_order_hashes_<arm>.json` (written by scripts/train_set_loss_control02.py)
and asserts all four arms' per-epoch hashes are identical. Exits 1 with an
[ISSUE] message on any mismatch or missing file; prints [OK] and exits 0
otherwise.
"""
import json
import sys
from pathlib import Path

ARMS = ('A0_hard_choice', 'A1_adaptive_multipos', 'A2_srm', 'A3_setutility_softce')


def main():
    if len(sys.argv) != 2:
        print('usage: gate_set_loss_control02_batch_order.py <results/TRACK-A-SET-LOSS-CONTROL02/CELL>')
        sys.exit(2)
    cell_dir = Path(sys.argv[1])
    per_arm = {}
    for arm in ARMS:
        p = cell_dir / f'batch_order_hashes_{arm}.json'
        if not p.exists():
            print(f'[ISSUE][ABORT] missing {p} -- arm not yet complete or trainer not run')
            sys.exit(1)
        per_arm[arm] = json.loads(p.read_text())

    epochs = set()
    for h in per_arm.values():
        epochs |= set(h.keys())
    epochs = sorted(epochs, key=lambda s: int(s.replace('epoch', '')))

    mismatches = []
    for ep in epochs:
        vals = {arm: per_arm[arm].get(ep) for arm in ARMS}
        present = {k: v for k, v in vals.items() if v is not None}
        if len(set(present.values())) > 1:
            mismatches.append((ep, vals))

    if mismatches:
        print('[ISSUE][ABORT] batch-order hashes diverge across arms:')
        for ep, vals in mismatches:
            print(f'  {ep}:')
            for arm, h in vals.items():
                print(f'    {arm}: {h}')
        sys.exit(1)

    print(f'[OK] batch-order hashes identical across all {len(ARMS)} arms for {len(epochs)} epochs.')
    for ep in epochs:
        print(f'  {ep}: {next(iter(per_arm.values()))[ep]}')
    sys.exit(0)


if __name__ == '__main__':
    main()
