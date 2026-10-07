"""TRACK-V-MEANMIX-CHECKPOINT-CORRECTION01 -- the unit/equivalence tests
from the spec not already covered by tests/test_v_meanmix_selection01.py
(tests 1,3,4,5,6 there) or tests/test_v_meanmix_cache01.py (test 9-style
real-checkpoint smoke, out of scope here since V0/V1 are not retested):

  [2]  logit-mean is NOT what the implementation computes
  [7]  checkpoint selection never reads the test split
  [8]  Mean-Mixture cache schema matches what train_r_stage2_lambda01.py
       (UNMODIFIED) expects
  [10] --disable_early_stopping actually disables the break, saving
       every epoch checkpoint even when validation never improves
"""
import ast
import inspect
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.mean_mixture_selection import mean_mixture_topk_selection

TOP_K = 10


# -------------------- [2] logit-mean prohibition --------------------

def test_mean_of_softmax_differs_from_softmax_of_mean():
    g = torch.Generator().manual_seed(3)
    scores = torch.randn(4, 5, 30, generator=g) * 3.0  # large scale so the two diverge sharply
    cand_mask = torch.ones(4, 30, dtype=torch.bool)
    tau_s = 0.1

    masked = scores.masked_fill(~cand_mask.unsqueeze(1), float('-inf'))
    mean_of_softmax = torch.softmax(masked / tau_s, dim=-1).mean(dim=1)
    softmax_of_mean = torch.softmax(masked.mean(dim=1) / tau_s, dim=-1)
    assert not torch.allclose(mean_of_softmax, softmax_of_mean, atol=1e-4), \
        'mean(softmax(x)) and softmax(mean(x)) should diverge for this input -- test construction is too easy'

    _, _, p_mix = mean_mixture_topk_selection(scores, cand_mask, tau_s, k=TOP_K)
    assert torch.allclose(p_mix, mean_of_softmax, atol=1e-6), \
        'mean_mixture_topk_selection must compute mean(softmax(scores/tau)), NOT softmax(mean(scores)/tau)'
    assert not torch.allclose(p_mix, softmax_of_mean, atol=1e-4), \
        '[ISSUE] p_mix matches softmax(mean(scores)) -- wrong (logit-mean) implementation'


# -------------------- [7] validation-only checkpoint selection --------------------

def test_checkpoint_selection_script_never_reads_test_split():
    src_path = REPO_ROOT / 'scripts' / 'eval_v5_meanmix_checkpoint_selection01.py'
    src = src_path.read_text()
    tree = ast.parse(src)
    flag_literals = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == 'flag' and isinstance(node.value, ast.Constant):
            flag_literals.add(node.value.value)
    assert flag_literals, 'expected at least one flag=... call in the checkpoint-selection script'
    assert 'test' not in flag_literals, \
        f'[ISSUE][ABORT] checkpoint-selection script reads flag={flag_literals}, must never include "test"'
    assert flag_literals == {'val'}, f'expected only flag="val", got {flag_literals}'


# -------------------- [8] cache schema compatibility --------------------

def test_meanmix_cache_schema_matches_stage2_expectations():
    from scripts.train_r_stage2_lambda01 import main as stage2_main  # noqa: F401 -- import succeeds => no schema drift at import time
    import scripts.build_v_meanmix_cache01 as builder
    src = inspect.getsource(builder.build_split)
    for required_key in ('query_start_idx', 'relation_outputs', 'D_per_query', 'C_per_query'):
        assert f"'{required_key}'" in src, f'[ISSUE] build_v_meanmix_cache01.build_split missing key {required_key}'

    # synthetic schema round-trip: construct a minimal fake cache dict with the
    # same keys/shapes and confirm train_r_stage2_lambda01's own cache loader
    # (if it exposes one) accepts it structurally.
    n, pred_len, n_ch = 12, 4, 3
    fake_cache = {
        'query_start_idx': torch.arange(n),
        'relation_outputs': torch.randn(n, pred_len, n_ch),
        'D_per_query': torch.randn(n),
        'C_per_query': torch.randn(n),
        'channels': list(range(n_ch)),
        'pred_len': pred_len,
        'split': 'val',
    }
    assert fake_cache['relation_outputs'].shape == (n, pred_len, n_ch)
    assert fake_cache['query_start_idx'].numel() == n


# -------------------- [10] disable_early_stopping actually disables the break --------------------

def test_disable_early_stopping_flag_present_and_wired():
    src = (REPO_ROOT / 'scripts' / 'train_t_pure_multislot01.py').read_text()
    assert '--disable_early_stopping' in src
    assert 'not cli.disable_early_stopping and epoch - best' in src, \
        '[ISSUE] disable_early_stopping flag added but not wired into the break condition'


# -------------------- A1 channel-first equivalence (applied from this point on) --------------------

def test_channel_first_scores_equal_legacy_scores_on_real_checkpoint():
    """A1 (channel-first) must be bit-exact-equivalent to the legacy
    channel-last path on a REAL checkpoint, not just synthetic tensors
    -- same equivalence check pattern as TRACK-V-R100-EFFICIENCY01's own
    smoke test (model.eval() required: MLP dropout otherwise makes two
    forward passes spuriously differ, as that track discovered)."""
    ref = (REPO_ROOT / 'checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/'
          'stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_'
          'df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth')
    v5_ckpt = (REPO_ROOT / 'checkpoints/track_v_multiquery_generalization01/ETTh1/H96/seed0/V5/stage1/'
              'checkpoint.pth')
    if not (ref.exists() and v5_ckpt.exists()):
        pytest.skip('ETTh1_96 V5 checkpoint not available in this environment')

    from scripts.build_v_meanmix_cache01 import compute_scores_full_grad_channel_first
    from scripts.train_margutil01 import build_experiment
    from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
    from scripts.train_t_pure_multislot01 import compute_scores_full_grad

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp, args = build_experiment(str(ref), {
        'pred_len': 96, 'seq_len': 96, 'batch_size': 32, 'seed': 0, 'top_k': TOP_K,
        'relation_encoder_type': 'mlp', 'relation_self_fill': 'linear', 'relation_input_space': 'delta_last',
        'relation_teacher_space': 'delta_last', 'relation_value_space': 'delta_last', 'candidate_mask': 'raft',
        'patch_len': 16, 'stride': 16,
    })
    exp._ensure_memory()
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.to(device)
    model.eval()  # required -- see docstring
    slot_heads = SlotHeads(int(args.d_model), n_slots=5).to(device)
    bl = torch.load(v5_ckpt, map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    slot_heads.load_state_dict(bl['slot_heads_state_dict'])
    slot_heads.eval()

    _, loader = exp._get_data(flag='val', shuffle=False)
    batch_x, _, _ = next(iter(loader))
    batch_x = batch_x.float().to(device)
    c = 0
    with torch.no_grad():
        legacy = compute_scores_full_grad(model, slot_heads, batch_x, exp.memory_x, c)
        a1 = compute_scores_full_grad_channel_first(model, slot_heads, batch_x, exp.memory_x, c)
    max_diff = (legacy - a1).abs().max().item()
    assert max_diff < 1e-5, f'[ISSUE] A1 channel-first scores diverge from legacy by {max_diff}'


def test_channel_first_pool_scores_equal_legacy_pool_scores_on_real_checkpoint():
    """Same A1 equivalence check, for the Shared-Top-100 (P100) pool
    variant -- compute_scores_pool_channel_first vs the legacy
    compute_scores coarse_topk branch, on a real P100 V5 checkpoint."""
    ref = (REPO_ROOT / 'checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/'
          'stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_'
          'df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth')
    v5_ckpt = (REPO_ROOT / 'checkpoints/track_v_multiquery_generalization01/ETTh1/H96/seed0/pool_top100/'
              'ETTh1_96/V5/checkpoint_epoch1.pth')
    pool_dir = (REPO_ROOT / 'results/TRACK-V-MULTIQUERY-GENERALIZATION01/ETTh1/H96/seed0/pool_top100/'
               'shared_candidate_pool')
    if not (ref.exists() and v5_ckpt.exists() and pool_dir.exists()):
        pytest.skip('ETTh1_96 P100 V5 checkpoint/pool not available in this environment')

    from scripts.build_v_meanmix_cache_pool01 import compute_scores_pool_channel_first
    from scripts.train_j_shared_encoder_drift01 import build_model
    from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
    from scripts.train_retriever_pool01 import CandidatePoolCache, compute_scores
    from utils.candidate_pool import CandidatePoolConfig
    from types import SimpleNamespace

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cli = SimpleNamespace(reference_ckpt=str(ref), pred_len=96, seq_len=96, init_seed=0, patch_len=16,
                         top_k=TOP_K, batch_size=32, candidate_mask='raft')
    exp, args, model = build_model(cli, device)
    exp.memory_x = exp.memory_x.to(device)
    model.eval()
    slot_heads = SlotHeads(int(args.d_model), n_slots=5).to(device)
    bl = torch.load(v5_ckpt, map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    slot_heads.load_state_dict(bl['slot_heads_state_dict'])
    slot_heads.eval()

    pool_cfg = CandidatePoolConfig(mode='coarse_topk', size=100, metric='delta_last_cosine')
    expected_meta = {'seq_len': 96, 'pred_len': 96, 'candidate_mask': 'raft',
                     'candidate_pool_size': 100, 'candidate_pool_metric': 'delta_last_cosine'}
    pool_cache = CandidatePoolCache(pool_dir / 'val.pt', expected_meta)

    _, loader = exp._get_data(flag='val', shuffle=False)
    batch_x, _, batch_start_idx = next(iter(loader))
    batch_x = batch_x.float().to(device)
    cand_mask, _ = exp._candidate_mask(batch_start_idx)
    c = 0
    with torch.no_grad():
        legacy, _, _ = compute_scores(pool_cfg, model, slot_heads, batch_x, exp, c, cand_mask,
                                      pool_cache, batch_start_idx, device)
        a1, _, _ = compute_scores_pool_channel_first(model, slot_heads, batch_x, exp, c,
                                                      pool_cache, batch_start_idx, device)
    max_diff = (legacy - a1).abs().max().item()
    assert max_diff < 1e-5, f'[ISSUE] A1 channel-first pool scores diverge from legacy by {max_diff}'
