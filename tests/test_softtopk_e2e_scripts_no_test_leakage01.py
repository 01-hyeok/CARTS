"""Spec section 15-1/28-9: training and validation-selection scripts for
this track must never construct a TEST loader. Static source check
(not just a runtime behavior check) so this fails loudly if a future
edit adds one."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

TRAIN_SCRIPTS = (
    'scripts/train_softtopk_e2e_v01.py',
    'scripts/train_softtopk_e2e_hard01.py',
    'scripts/train_softtopk_e2e_joint01.py',
)

FORBIDDEN_PATTERNS = ("flag='test'", 'flag="test"')


def test_no_e2e_trainer_constructs_a_test_loader():
    for rel in TRAIN_SCRIPTS:
        src = (REPO_ROOT / rel).read_text()
        for pat in FORBIDDEN_PATTERNS:
            assert pat not in src, f'{rel} must never construct a TEST loader (found {pat!r})'


def test_evaluate_locked_test_script_exists_and_is_the_only_test_reader():
    path = REPO_ROOT / 'scripts' / 'evaluate_locked_test01.py'
    assert path.exists(), 'scripts/evaluate_locked_test01.py must exist -- it is the ONLY place TEST is read'
    src = path.read_text()
    assert 'LOCKED_CONFIG' in src, 'evaluate_locked_test01.py must require a LOCKED_CONFIG.json to run'
