"""TRACK-V-R100-EFFICIENCY01 -- unit/integration tests for
`utils/full_candidate_bank.py` (spec section 18, tests 1-4).

Numerical-equivalence tests against the REAL reference checkpoint
(ETTh1_96) skip cleanly if that checkpoint is not present in this
environment (same pattern as `tests/test_crh_v5_integration01.py`)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_factorial_e2e01 import encode_raw
from scripts.train_j_shared_encoder_drift01 import build_model
from utils.full_candidate_bank import FullCandidateBank, encode_raw_channel_first

REF = (REPO_ROOT / 'checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/'
      'stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_'
      'dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth')


def _skip_if_missing():
    if not REF.exists():
        pytest.skip('ETTh1_96 reference checkpoint not available in this environment')


def _setup():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cli = SimpleNamespace(reference_ckpt=str(REF), pred_len=96, seq_len=96, batch_size=32,
                          init_seed=0, top_k=10, patch_len=16)
    exp, args, model = build_model(cli, device)
    model.eval()
    return device, exp, args, model


def test_channel_first_matches_legacy_encode_raw_on_query_batch():
    _skip_if_missing()
    device, exp, args, model = _setup()
    _, loader = exp._get_data(flag='val', shuffle=False)
    batch_x, _, _ = next(iter(loader))
    batch_x = batch_x.float().to(device)
    for c in range(int(args.enc_in)):
        with torch.no_grad():
            legacy = encode_raw(model, batch_x, c)
            fast = encode_raw_channel_first(model, batch_x, c)
        max_abs_diff = float((legacy - fast).abs().max())
        assert max_abs_diff < 1e-6, f'channel {c}: max_abs_diff={max_abs_diff}'


def test_channel_first_matches_legacy_encode_raw_on_candidates():
    _skip_if_missing()
    device, exp, args, model = _setup()
    # a modest slice of the candidate bank is enough to prove the identity
    memory_slice = exp.memory_x[:256].to(device)
    for c in range(int(args.enc_in)):
        with torch.no_grad():
            legacy = encode_raw(model, memory_slice, c)
            fast = encode_raw_channel_first(model, memory_slice, c)
        max_abs_diff = float((legacy - fast).abs().max())
        assert max_abs_diff < 1e-6, f'channel {c}: max_abs_diff={max_abs_diff}'


def test_bank_requires_grad_false_after_refresh():
    _skip_if_missing()
    device, exp, args, model = _setup()
    channels = list(range(int(args.enc_in)))
    bank = FullCandidateBank(channels)
    bank.refresh(model, exp.memory_x)
    for c in channels:
        assert bank.get(c).requires_grad is False


def test_bank_full_N_maintained_not_pruned():
    _skip_if_missing()
    device, exp, args, model = _setup()
    channels = list(range(int(args.enc_in)))
    bank = FullCandidateBank(channels)
    bank.refresh(model, exp.memory_x)
    n = exp.memory_x.size(0)
    for c in channels:
        assert bank.get(c).size(0) == n, 'candidate support must stay full N -- never pruned/subset'


def test_bank_refresh_resets_age_and_counts():
    _skip_if_missing()
    device, exp, args, model = _setup()
    channels = list(range(int(args.enc_in)))
    bank = FullCandidateBank(channels)
    bank.refresh(model, exp.memory_x)
    assert bank.age == 0
    assert bank.n_refreshes == 1
    assert bank.n_candidate_encoder_calls == len(channels)
    for _ in range(50):
        bank.tick()
    assert bank.age == 50
    bank.refresh(model, exp.memory_x)
    assert bank.age == 0
    assert bank.n_refreshes == 2
    assert bank.n_candidate_encoder_calls == 2 * len(channels)


def test_bank_cached_matches_fresh_at_age_zero():
    """age=0 bank embedding == fresh encoder embedding (spec test 2)."""
    _skip_if_missing()
    device, exp, args, model = _setup()
    channels = list(range(int(args.enc_in)))
    bank = FullCandidateBank(channels)
    bank.refresh(model, exp.memory_x)
    for c in channels:
        with torch.no_grad():
            fresh = encode_raw_channel_first(model, exp.memory_x, c)
        max_abs_diff = float((bank.get(c) - fresh).abs().max())
        assert max_abs_diff < 1e-6, f'channel {c}: cached-vs-fresh age0 diff={max_abs_diff}'
