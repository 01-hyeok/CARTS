"""TRACK-P100-QA-TEST-EVALUATION01 -- shared path-resolution helper.

Parses the EXACT `REF_*`/`BASE_*`/`pool_for()` variable assignments out
of `scripts/run_softtopk_quality_aware_grid01.sh` (simple `VAR="value"`
bash lines, read as text -- never executed) so this track's own
checkpoint/pool path lookups are guaranteed identical to what the
original training grid actually used, instead of a hand-transcribed
(and therefore error-prone) copy.
"""
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GRID_SCRIPT = REPO_ROOT / 'scripts' / 'run_softtopk_quality_aware_grid01.sh'

METHOD_TO_NQV = {'V0': 0, 'V1': 1, 'V2': 2, 'V5': 5}
DATASETS = ('ETTh1', 'Weather', 'Solar')
HORIZONS = (96, 192, 336, 720)


def _parse_bash_vars(script_path):
    text = Path(script_path).read_text()
    out = {}
    for m in re.finditer(r'^([A-Z0-9_]+)="([^"]*)"', text, flags=re.MULTILINE):
        out[m.group(1)] = m.group(2)
    return out


_VARS = _parse_bash_vars(GRID_SCRIPT)


def reference_ckpt(dataset, horizon):
    return _VARS[f'REF_{dataset.upper()}_{horizon}']


def base_checkpoint(dataset, horizon):
    return _VARS[f'BASE_{dataset.upper()}_{horizon}']


def candidate_pool_cache(dataset, horizon):
    if dataset == 'Solar':
        return f'results/TRACK-V-MULTIQUERY-GENERALIZATION01/Solar/H{horizon}/seed0/pool_top100/shared_candidate_pool'
    return f'results/TRACK-V-MULTIQUERY-GENERALIZATION01/{dataset}/H{horizon}/seed0/pool_top100/shared_candidate_pool'


def all_cells():
    return [(ds, h) for ds in DATASETS for h in HORIZONS]


def cell_name(dataset, horizon):
    return f'{dataset}_{horizon}'
