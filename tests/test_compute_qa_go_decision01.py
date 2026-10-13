import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
METHODS = ('V0', 'V1', 'V2', 'V5', 'Hard', 'Joint')
CELLS = ['ETTh1_96', 'ETTh1_192', 'ETTh1_336', 'ETTh1_720',
        'Weather_96', 'Weather_192', 'Weather_336', 'Weather_720',
        'Solar_96', 'Solar_192', 'Solar_336', 'Solar_720']


def _write_fixture(root, qa_delta_fn):
    """qa_delta_fn(cell, method) -> (noqa_mse, qa_mse)."""
    for cell in CELLS:
        for method in METHODS:
            noqa_mse, qa_mse = qa_delta_fn(cell, method)
            for arm, mse in (('NOQA', noqa_mse), ('QA', qa_mse)):
                d = Path(root) / cell / method / arm
                d.mkdir(parents=True, exist_ok=True)
                (d / 'checkpoint_selection.json').write_text(json.dumps({'val_mse': mse, 'epoch': 5}))


def _run(root, out_json):
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / 'scripts' / 'compute_qa_go_decision01.py'),
         '--results_root', str(root), '--out_json', str(out_json)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
    )
    return result


def test_go_true_when_qa_uniformly_better(tmp_path):
    _write_fixture(tmp_path, lambda cell, method: (0.50, 0.45))  # QA always 10% better
    out = tmp_path / 'decision.json'
    result = _run(tmp_path, out)
    assert result.returncode == 0, result.stderr
    decision = json.loads(out.read_text())
    assert decision['QA_GO'] is True
    assert decision['cells_with_qa_win'] == 12


def test_go_false_when_qa_uniformly_worse(tmp_path):
    _write_fixture(tmp_path, lambda cell, method: (0.45, 0.50))  # QA always worse
    out = tmp_path / 'decision.json'
    result = _run(tmp_path, out)
    assert result.returncode == 0, result.stderr
    decision = json.loads(out.read_text())
    assert decision['QA_GO'] is False
    assert decision['cells_with_qa_win'] == 0


def test_go_false_when_fewer_than_7_cells_win_even_if_overall_avg_wins(tmp_path):
    # 6 cells big QA win, 6 cells small QA loss -> overall avg could still
    # favor QA, but criterion 2 (>=7/12 cells) must independently fail.
    big_win_cells = set(CELLS[:6])

    def fn(cell, method):
        if cell in big_win_cells:
            return (1.0, 0.1)  # huge QA win
        return (0.10, 0.11)  # small QA loss
    _write_fixture(tmp_path, fn)
    out = tmp_path / 'decision.json'
    result = _run(tmp_path, out)
    decision = json.loads(out.read_text())
    assert decision['cells_with_qa_win'] == 6
    assert decision['criteria']['2_cells_qa_wins_ge_7of12'] is False
    assert decision['QA_GO'] is False


def test_go_false_when_one_dataset_average_fails_even_if_others_pass(tmp_path):
    def fn(cell, method):
        if cell.startswith('Solar'):
            return (0.10, 0.20)  # Solar: QA uniformly worse
        return (0.50, 0.40)  # ETTh1/Weather: QA uniformly better
    _write_fixture(tmp_path, fn)
    out = tmp_path / 'decision.json'
    result = _run(tmp_path, out)
    decision = json.loads(out.read_text())
    assert decision['dataset_avg_delta']['Solar'] > 0
    assert decision['criteria']['3_5_all_dataset_avgs_qa_le_noqa'] is False
    assert decision['QA_GO'] is False


def test_missing_combos_reported_but_decision_still_computed(tmp_path):
    _write_fixture(tmp_path, lambda cell, method: (0.5, 0.4))
    # Delete one combo entirely.
    import shutil
    shutil.rmtree(Path(tmp_path) / 'ETTh1_96' / 'Joint')
    out = tmp_path / 'decision.json'
    result = _run(tmp_path, out)
    assert result.returncode == 0
    decision = json.loads(out.read_text())
    assert any('ETTh1_96/Joint' in m for m in decision['missing_combos'])


def test_refuses_when_zero_combos_available(tmp_path):
    out = tmp_path / 'decision.json'
    result = _run(tmp_path, out)
    assert result.returncode != 0
