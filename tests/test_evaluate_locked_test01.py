import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / 'scripts' / 'evaluate_locked_test01.py'

COMMON_ARGS = ['--protocol', 'original_e2e', '--reference_ckpt', 'does-not-matter.pth',
              '--method', 'V5', '--arm', 'NOQA', '--cell', 'ETTh1_96',
              '--pred_len', '96', '--seq_len', '96']


def _sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def test_refuses_to_run_without_locked_config(tmp_path):
    missing = tmp_path / 'LOCKED_CONFIG.json'
    assert not missing.exists()
    result = subprocess.run(
        [sys.executable, str(SCRIPT), '--locked_config', str(missing), *COMMON_ARGS,
         '--base_checkpoint', 'does-not-matter.pth', '--candidate_pool_cache', str(tmp_path),
         '--checkpoint_path', 'does-not-matter.pth', '--out_dir', str(tmp_path)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert 'LOCKED_CONFIG' in (result.stdout + result.stderr)


def test_refuses_with_malformed_locked_config_missing_fields(tmp_path):
    cfg = tmp_path / 'LOCKED_CONFIG.json'
    cfg.write_text('{"git_commit": "abc"}')  # missing datasets/horizons/method_list/test_accessed/combos
    result = subprocess.run(
        [sys.executable, str(SCRIPT), '--locked_config', str(cfg), *COMMON_ARGS,
         '--base_checkpoint', 'does-not-matter.pth', '--candidate_pool_cache', str(tmp_path),
         '--checkpoint_path', 'does-not-matter.pth', '--out_dir', str(tmp_path)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert 'missing required field' in (result.stdout + result.stderr)


def _minimal_valid_cfg():
    return {'git_commit': 'abc', 'datasets': ['ETTh1'], 'horizons': [96], 'method_list': ['V5'],
            'test_accessed': True, 'combos': {}}


def test_refuses_when_combo_key_not_locked(tmp_path):
    cfg_path = tmp_path / 'LOCKED_CONFIG.json'
    cfg_path.write_text(json.dumps(_minimal_valid_cfg()))  # combos is empty -- nothing locked
    ckpt = tmp_path / 'ckpt.pth'
    ckpt.write_bytes(b'fake-checkpoint-bytes')
    base_ckpt = tmp_path / 'base.pth'
    base_ckpt.write_bytes(b'fake-base-bytes')
    result = subprocess.run(
        [sys.executable, str(SCRIPT), '--locked_config', str(cfg_path), *COMMON_ARGS,
         '--base_checkpoint', str(base_ckpt), '--candidate_pool_cache', str(tmp_path),
         '--checkpoint_path', str(ckpt), '--out_dir', str(tmp_path)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert 'never frozen' in (result.stdout + result.stderr)


def test_refuses_when_checkpoint_hash_does_not_match_locked_hash(tmp_path):
    ckpt = tmp_path / 'ckpt.pth'
    ckpt.write_bytes(b'fake-checkpoint-bytes-v1')
    base_ckpt = tmp_path / 'base.pth'
    base_ckpt.write_bytes(b'fake-base-bytes')

    cfg = _minimal_valid_cfg()
    cfg['combos']['original_e2e/ETTh1_96/V5/NOQA'] = {
        'checkpoint_sha256': 'WRONG_HASH_NOT_MATCHING_ACTUAL_FILE',
        'base_checkpoint_sha256': _sha256_bytes(base_ckpt.read_bytes()),
        'horizon': 96, 'num_query_views': 5,
    }
    cfg_path = tmp_path / 'LOCKED_CONFIG.json'
    cfg_path.write_text(json.dumps(cfg))

    result = subprocess.run(
        [sys.executable, str(SCRIPT), '--locked_config', str(cfg_path), *COMMON_ARGS,
         '--base_checkpoint', str(base_ckpt), '--candidate_pool_cache', str(tmp_path),
         '--checkpoint_path', str(ckpt), '--out_dir', str(tmp_path)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert 'checkpoint SHA256 mismatch' in (result.stdout + result.stderr)


def test_refuses_when_base_checkpoint_hash_does_not_match(tmp_path):
    ckpt = tmp_path / 'ckpt.pth'
    ckpt.write_bytes(b'fake-checkpoint-bytes-v1')
    base_ckpt = tmp_path / 'base.pth'
    base_ckpt.write_bytes(b'fake-base-bytes-v1')

    cfg = _minimal_valid_cfg()
    cfg['combos']['original_e2e/ETTh1_96/V5/NOQA'] = {
        'checkpoint_sha256': _sha256_bytes(ckpt.read_bytes()),
        'base_checkpoint_sha256': 'WRONG_BASE_HASH',
        'horizon': 96, 'num_query_views': 5,
    }
    cfg_path = tmp_path / 'LOCKED_CONFIG.json'
    cfg_path.write_text(json.dumps(cfg))

    result = subprocess.run(
        [sys.executable, str(SCRIPT), '--locked_config', str(cfg_path), *COMMON_ARGS,
         '--base_checkpoint', str(base_ckpt), '--candidate_pool_cache', str(tmp_path),
         '--checkpoint_path', str(ckpt), '--out_dir', str(tmp_path)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert 'base checkpoint SHA256 mismatch' in (result.stdout + result.stderr)


def test_refuses_when_num_query_views_mismatch(tmp_path):
    ckpt = tmp_path / 'ckpt.pth'
    ckpt.write_bytes(b'fake-checkpoint-bytes-v1')
    base_ckpt = tmp_path / 'base.pth'
    base_ckpt.write_bytes(b'fake-base-bytes')

    cfg = _minimal_valid_cfg()
    cfg['combos']['original_e2e/ETTh1_96/V5/NOQA'] = {
        'checkpoint_sha256': _sha256_bytes(ckpt.read_bytes()),
        'base_checkpoint_sha256': _sha256_bytes(base_ckpt.read_bytes()),
        'horizon': 96, 'num_query_views': 999,  # deliberately wrong vs V5's real 5
    }
    cfg_path = tmp_path / 'LOCKED_CONFIG.json'
    cfg_path.write_text(json.dumps(cfg))

    result = subprocess.run(
        [sys.executable, str(SCRIPT), '--locked_config', str(cfg_path), *COMMON_ARGS,
         '--base_checkpoint', str(base_ckpt), '--candidate_pool_cache', str(tmp_path),
         '--checkpoint_path', str(ckpt), '--out_dir', str(tmp_path)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0
    assert 'num_query_views mismatch' in (result.stdout + result.stderr)
