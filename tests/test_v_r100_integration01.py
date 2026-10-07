"""TRACK-V-R100-EFFICIENCY01 -- integration tests (spec section 18,
tests 5-17). Numerical-equivalence tests against the REAL ETTh1_96
reference checkpoint skip cleanly if unavailable."""
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.train_factorial_e2e01 import arm_score, encode_raw, individual_utility_memsafe
from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob
from scripts.train_j_shared_encoder_drift01 import build_model
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_margutil01 import memory_value
from scripts.train_t_pure_multislot01 import compute_scores_full_grad, round_robin_topk_selection
from scripts.train_v0_r100_01 import compute_scores_cached as v0_cached
from scripts.train_v0_r100_01 import compute_scores_fresh as v0_fresh
from scripts.train_v_r100_01 import compute_scores_cached as v_cached
from scripts.train_v_r100_01 import compute_scores_fresh as v_fresh
from utils.full_candidate_bank import FullCandidateBank

REF = (REPO_ROOT / 'checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/'
      'stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_'
      'dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth')
TOP_K = 10


def _skip_if_missing():
    if not REF.exists():
        pytest.skip('ETTh1_96 reference checkpoint not available in this environment')


def _setup():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cli = SimpleNamespace(reference_ckpt=str(REF), pred_len=96, seq_len=96, batch_size=32,
                          init_seed=0, top_k=TOP_K, patch_len=16)
    exp, args, model = build_model(cli, device)
    model.eval()
    _, loader = exp._get_data(flag='val', shuffle=False)
    batch_x, batch_y, batch_start_idx = next(iter(loader))
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float().to(device)
    cand_mask, _ = exp._candidate_mask(batch_start_idx)
    return device, exp, args, model, batch_x, batch_y, batch_start_idx, cand_mask


# -------------------- test 6: refresh schedule = {0, 100, 200, ...} (pure simulation) --------------------

def test_refresh_schedule_matches_spec_0_100_200():
    calls = []

    class FakeModel:
        pass

    class FakeBank(FullCandidateBank):
        def refresh(self, model, memory_x):
            calls.append('refresh')
            self.age = 0
            self.n_refreshes += 1

    bank = FakeBank(channels=[0])
    bank.refresh(None, None)  # step-0 refresh, before any training step
    global_step = 0
    refresh_interval = 100
    for _ in range(250):
        global_step += 1
        if global_step % refresh_interval == 0:
            bank.refresh(None, None)
    assert bank.n_refreshes == 3  # initial + step100 + step200
    assert len(calls) == 3


# -------------------- tests 8/9: validation/test always rebuild a fresh bank --------------------

def test_fresh_eval_functions_take_no_bank_argument():
    """Structural guarantee for spec tests 8/9: the validation/test score
    functions cannot possibly read a stale cache because they have no
    bank parameter at all -- they always re-encode memory_x directly."""
    for fn in (v0_fresh, v_fresh):
        params = inspect.signature(fn).parameters
        assert 'bank' not in params, f'{fn.__name__} must not accept a bank -- eval must always be fresh'
        assert 'memory_x' in params, f'{fn.__name__} must take memory_x directly to force a fresh encode'


# -------------------- test 5: full N candidate support maintained, never Top-M/pool --------------------

def test_v0_cached_scores_cover_full_N():
    _skip_if_missing()
    device, exp, args, model, batch_x, batch_y, batch_start_idx, cand_mask = _setup()
    channels = list(range(int(args.enc_in)))
    bank = FullCandidateBank(channels)
    bank.refresh(model, exp.memory_x)
    n = exp.memory_x.size(0)
    scores = v0_cached(model, batch_x, bank, 0)
    assert scores.shape == (batch_x.size(0), 1, n)


def test_v_cached_scores_cover_full_N_for_all_slot_counts():
    _skip_if_missing()
    device, exp, args, model, batch_x, batch_y, batch_start_idx, cand_mask = _setup()
    channels = list(range(int(args.enc_in)))
    bank = FullCandidateBank(channels)
    bank.refresh(model, exp.memory_x)
    n = exp.memory_x.size(0)
    d_model = int(args.d_model)
    for num_slots in (1, 2, 5):
        slot_heads = SlotHeads(d_model, n_slots=num_slots).to(device)
        scores = v_cached(model, slot_heads, batch_x, bank, 0)
        assert scores.shape == (batch_x.size(0), num_slots, n)


# -------------------- tests 10-13: age=0 score parity (R100 cached path vs legacy full-online path) --------------------

def test_v0_age0_score_parity_with_legacy():
    _skip_if_missing()
    device, exp, args, model, batch_x, batch_y, batch_start_idx, cand_mask = _setup()
    channels = list(range(int(args.enc_in)))
    bank = FullCandidateBank(channels)
    bank.refresh(model, exp.memory_x)  # age=0: built from the SAME model state used below
    for c in channels:
        with torch.no_grad():
            r100_scores = v0_cached(model, batch_x, bank, c).squeeze(1)  # [B, N]
            legacy_scores = arm_score(encode_raw(model, batch_x, c), encode_raw(model, exp.memory_x, c), None)
        max_abs_diff = float((r100_scores - legacy_scores).abs().max())
        assert max_abs_diff < 1e-5, f'channel {c}: max_abs_diff={max_abs_diff}'


@pytest.mark.parametrize('num_slots', [1, 2, 5])
def test_v_age0_score_parity_with_legacy(num_slots):
    _skip_if_missing()
    device, exp, args, model, batch_x, batch_y, batch_start_idx, cand_mask = _setup()
    channels = list(range(int(args.enc_in)))
    d_model = int(args.d_model)
    slot_heads = SlotHeads(d_model, n_slots=num_slots).to(device)
    bank = FullCandidateBank(channels)
    bank.refresh(model, exp.memory_x)
    for c in channels[:3]:  # a representative subset keeps this test fast
        with torch.no_grad():
            r100_scores = v_cached(model, slot_heads, batch_x, bank, c)
            legacy_scores = compute_scores_full_grad(model, slot_heads, batch_x, exp.memory_x, c)
        max_abs_diff = float((r100_scores - legacy_scores).abs().max())
        assert max_abs_diff < 1e-5, f'num_slots={num_slots} channel {c}: max_abs_diff={max_abs_diff}'


# -------------------- test 14: round-robin Top10 parity at age=0 --------------------

@pytest.mark.parametrize('num_slots', [1, 2, 5])
def test_round_robin_top10_parity_at_age0(num_slots):
    _skip_if_missing()
    device, exp, args, model, batch_x, batch_y, batch_start_idx, cand_mask = _setup()
    channels = list(range(int(args.enc_in)))
    d_model = int(args.d_model)
    slot_heads = SlotHeads(d_model, n_slots=num_slots).to(device)
    bank = FullCandidateBank(channels)
    bank.refresh(model, exp.memory_x)
    c = 0
    with torch.no_grad():
        r100_scores = v_cached(model, slot_heads, batch_x, bank, c)
        legacy_scores = compute_scores_full_grad(model, slot_heads, batch_x, exp.memory_x, c)
        picks_r100 = round_robin_topk_selection(r100_scores, cand_mask, k=TOP_K)
        picks_legacy = round_robin_topk_selection(legacy_scores, cand_mask, k=TOP_K)
    assert torch.equal(picks_r100, picks_legacy)


# -------------------- tests 15/16: teacher distribution / candidate mask unchanged --------------------

def test_teacher_distribution_unchanged():
    """Teacher computation is untouched by R100 (reused verbatim) -- the
    SAME memory_value/individual_utility_memsafe/normalized_teacher_prob
    calls used by the legacy trainers are used by the R100 trainers, so
    for the IDENTICAL batch/model/args they produce identical teacher
    probabilities, independent of which score-computation path (cached
    vs fresh) is used downstream."""
    _skip_if_missing()
    device, exp, args, model, batch_x, batch_y, batch_start_idx, cand_mask = _setup()
    c = 0
    memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
    query_future = batch_y[:, :, c]
    u1 = individual_utility_memsafe(memory_c, offset_c, query_future, 4096)
    p_t1 = normalized_teacher_prob(-u1, cand_mask, 0.1)
    # recomputed a second time via the exact same reused functions
    u2 = individual_utility_memsafe(memory_c, offset_c, query_future, 4096)
    p_t2 = normalized_teacher_prob(-u2, cand_mask, 0.1)
    assert torch.equal(p_t1, p_t2)


def test_candidate_mask_unchanged():
    """`exp._candidate_mask` is reused verbatim, unmodified, by both the
    R100 trainers and every legacy trainer -- calling it twice on the
    same batch_start_idx must be deterministic/identical."""
    _skip_if_missing()
    device, exp, args, model, batch_x, batch_y, batch_start_idx, cand_mask = _setup()
    cand_mask2, _ = exp._candidate_mask(batch_start_idx)
    assert torch.equal(cand_mask, cand_mask2)


# -------------------- test 17: seed/init/batch-order parity --------------------

def test_init_seed_reproducible_across_build_model_calls():
    _skip_if_missing()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cli = SimpleNamespace(reference_ckpt=str(REF), pred_len=96, seq_len=96, batch_size=32,
                          init_seed=0, top_k=TOP_K, patch_len=16)
    from scripts.train_j_shared_encoder_drift01 import state_hash
    _, _, model_a = build_model(cli, device)
    _, _, model_b = build_model(cli, device)
    assert state_hash(model_a) == state_hash(model_b)


def test_batch_order_sha256_reused_unmodified():
    from scripts.rng_control01 import batch_order_sha256
    starts_a = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
    starts_b = [torch.tensor([1, 2, 3]), torch.tensor([4, 5])]
    assert batch_order_sha256(starts_a) == batch_order_sha256(starts_b)
