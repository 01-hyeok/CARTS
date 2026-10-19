import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'compute_qa_go_decision02_strict01.py'
METHODS = ('V0', 'V1', 'V2', 'V5', 'Hard', 'Joint')
CELLS = [f'{ds}_{h}' for ds in ('ETTh1', 'Weather', 'Solar') for h in (96, 192, 336, 720)]


def _write_combo(root, cell, method, arm, val_mse):
    d = root / cell / method / arm
    d.mkdir(parents=True, exist_ok=True)
    (d / 'checkpoint_selection.json').write_text(json.dumps({'val_mse': val_mse, 'epoch': 5}))


def _run(root, out_json, oracle_json=None, allow_incomplete=False):
    cmd = [sys.executable, str(SCRIPT), '--results_root', str(root), '--out_json', str(out_json)]
    if oracle_json:
        cmd += ['--oracle_headroom_json', str(oracle_json)]
    if allow_incomplete:
        cmd += ['--allow_incomplete_preview']
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(Path(out_json).read_text())


def test_partial_grid_without_flag_is_incomplete_not_go(tmp_path):
    """Regression test for the exact bug this script fixes: the ORIGINAL
    script would return QA_GO=True on a 62/72 partial grid with no
    oracle data. This script must refuse."""
    root = tmp_path / 'results'
    for cell in CELLS:
        for method in METHODS:
            if cell == 'Solar_720':
                continue  # leave this cell entirely missing
            _write_combo(root, cell, method, 'NOQA', 0.5)
            _write_combo(root, cell, method, 'QA', 0.4)  # QA uniformly better
    out = _run(root, tmp_path / 'decision.json')
    assert out['status'] == 'INCOMPLETE'
    assert out['QA_GO'] is None
    assert out['combos_available'] == 66  # 72 - 6 (Solar_720's 6 methods)


def test_complete_grid_but_no_oracle_is_incomplete(tmp_path):
    root = tmp_path / 'results'
    for cell in CELLS:
        for method in METHODS:
            _write_combo(root, cell, method, 'NOQA', 0.5)
            _write_combo(root, cell, method, 'QA', 0.4)
    out = _run(root, tmp_path / 'decision.json')
    assert out['status'] == 'INCOMPLETE'
    assert out['QA_GO'] is None
    assert out['combos_available'] == 72
    assert out['oracle_headroom_computed'] is False


def test_complete_grid_with_oracle_and_all_criteria_true_is_go(tmp_path):
    root = tmp_path / 'results'
    for cell in CELLS:
        for method in METHODS:
            _write_combo(root, cell, method, 'NOQA', 0.5)
            _write_combo(root, cell, method, 'QA', 0.4)  # QA wins everywhere
    oracle_json = tmp_path / 'oracle.json'
    oracle_json.write_text(json.dumps({'positive_headroom_exists': True}))
    out = _run(root, tmp_path / 'decision.json', oracle_json=oracle_json)
    assert out['status'] == 'COMPLETE'
    assert out['QA_GO'] is True
    assert out['criteria']['6_positive_dynamic_oracle_headroom'] is True


def test_complete_grid_with_false_oracle_is_stop(tmp_path):
    root = tmp_path / 'results'
    for cell in CELLS:
        for method in METHODS:
            _write_combo(root, cell, method, 'NOQA', 0.5)
            _write_combo(root, cell, method, 'QA', 0.4)
    oracle_json = tmp_path / 'oracle.json'
    oracle_json.write_text(json.dumps({'positive_headroom_exists': False}))
    out = _run(root, tmp_path / 'decision.json', oracle_json=oracle_json)
    assert out['status'] == 'COMPLETE'
    assert out['QA_GO'] is False


def test_allow_incomplete_preview_still_returns_null_go(tmp_path):
    """Even with the preview flag, QA_GO must stay null -- the preview
    flag unlocks seeing the numbers, never a verdict."""
    root = tmp_path / 'results'
    for cell in CELLS:
        for method in METHODS:
            if cell == 'Solar_720':
                continue
            _write_combo(root, cell, method, 'NOQA', 0.5)
            _write_combo(root, cell, method, 'QA', 0.4)
    out = _run(root, tmp_path / 'decision.json', allow_incomplete=True)
    assert out['status'] == 'INCOMPLETE_PREVIEW'
    assert out['QA_GO'] is None
    assert 'overall_avg_delta' in out  # preview numbers ARE present
