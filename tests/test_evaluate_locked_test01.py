import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_refuses_to_run_without_locked_config(tmp_path):
    missing = tmp_path / 'LOCKED_CONFIG.json'
    assert not missing.exists()
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / 'scripts' / 'evaluate_locked_test01.py'),
         '--locked_config', str(missing),
         '--reference_ckpt', 'does-not-matter.pth', '--method', 'V5', '--arm', 'NOQA',
         '--cell', 'ETTh1_96', '--pred_len', '96', '--seq_len', '96',
         '--base_checkpoint', 'does-not-matter.pth', '--candidate_pool_cache', str(tmp_path),
         '--checkpoint_path', 'does-not-matter.pth', '--out_dir', str(tmp_path)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert 'LOCKED_CONFIG' in (result.stdout + result.stderr)


def test_refuses_with_malformed_locked_config_missing_fields(tmp_path):
    cfg = tmp_path / 'LOCKED_CONFIG.json'
    cfg.write_text('{"git_commit": "abc"}')  # missing datasets/horizons/method_list/test_accessed
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / 'scripts' / 'evaluate_locked_test01.py'),
         '--locked_config', str(cfg),
         '--reference_ckpt', 'does-not-matter.pth', '--method', 'V5', '--arm', 'NOQA',
         '--cell', 'ETTh1_96', '--pred_len', '96', '--seq_len', '96',
         '--base_checkpoint', 'does-not-matter.pth', '--candidate_pool_cache', str(tmp_path),
         '--checkpoint_path', 'does-not-matter.pth', '--out_dir', str(tmp_path)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert 'missing required field' in (result.stdout + result.stderr)
