#!/usr/bin/env python3
"""TRACK-V5-TOPK-CORRECTION-MECHANISM01 Phase G -- does Differentiable
Top-K v2 change the head-redundancy mechanism Phase C found for v1
(L3b wins via individual-head accuracy/specialization, NOT reduced
head-error-cosine/complementarity)? Reuses Phase C's own `analyze_cell`
function (same metrics, same zero-norm rule, same within-set-10
candidate cosine) unmodified, pointed at the new S2-v2/D2-v2
checkpoints across all 4 primary cells, and prints a direct v1-vs-v2
comparison.
"""
import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_v5_topk_head_complementarity01 import CELLS, analyze_cell

V2_CKPT = ('checkpoints/track_v5_topk_correction_mechanism01/{cell}/{label}/{cell}/{arm}/'
          'checkpoint_best_retmse.pth')
V1_CKPT = 'checkpoints/track_v5_head_loss_dual_encoder01/{cell}/{arm}/checkpoint_best_retmse.pth'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cells', nargs='+', default=list(CELLS.keys()))
    ap.add_argument('--out_dir', default='results/TRACK-V5-TOPK-CORRECTION-MECHANISM01/phaseG')
    cli = ap.parse_args()

    rows = []
    for cell in cli.cells:
        cfg = CELLS[cell]
        # v1 baselines (S0/D0, the L0 arms) + v1 L3b arms (S2/D2), reused from Phase C
        for arm in ('S0', 'D0', 'S2', 'D2'):
            ckpt = Path(V1_CKPT.format(cell=cell, arm=arm))
            if not ckpt.exists():
                print(f'[phaseG][SKIP] missing v1 checkpoint {ckpt}')
                continue
            label = f'{arm}-v1' if arm in ('S2', 'D2') else arm
            row = analyze_cell(cfg, label, ckpt, cfg['pool'], rows)
        # v2 L3b arms
        for arm, label in (('S2', 'S2-v2'), ('D2', 'D2-v2')):
            ckpt = Path(V2_CKPT.format(cell=cell, label=label, arm=arm))
            if not ckpt.exists():
                print(f'[phaseG][SKIP] missing v2 checkpoint {ckpt}')
                continue
            analyze_cell(cfg, label, ckpt, cfg['pool'], rows)

    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'v1_v2_head_mechanism_results.json'
    json.dump(rows, open(out_path, 'w'), indent=2)
    print(f'[phaseG] wrote {out_path}')


if __name__ == '__main__':
    main()
