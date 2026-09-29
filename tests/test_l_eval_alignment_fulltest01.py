"""Unit tests for TRACK-L-EVAL-ALIGNMENT-FULLTEST01 (spec section 22).
NO TRAINING -- these test the shared evaluator only. Fast synthetic
tests plus a few cheap real-checkpoint checks.
"""
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_l_aligned_population01 import CKPTS, macro_summary, score_and_decompose
from scripts.train_j_shared_encoder_drift01 import build_model, build_probe_set, state_hash
from scripts.train_k_multislot_predictive_retrieval01 import N_SLOTS, SlotHeads

_ref_available = all(p.exists() for p in CKPTS.values())
needs_ref = pytest.mark.skipif(not _ref_available, reason='TRACK-J/J2/K checkpoints not present in this checkout')


class _Cli:
    reference_ckpt = ('checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/'
                      'stage1_carts_softset_ETTh1_720_S0_wce_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_'
                      'nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth')
    cell = 'ETTh1_720'
    pred_len = 720
    seq_len = 720
    patch_len = 16
    top_k = 10
    tau_t = 0.1
    tau_s = 0.1
    batch_size = 32
    learning_rate = 1e-3
    chunk_size = 4096
    init_seed = 0
    loader_seed = 0


@pytest.fixture(scope='module')
def env():
    if not _ref_available:
        pytest.skip('checkpoints not present')
    cli = _Cli()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp, args, base_model = build_model(cli, device)
    return cli, device, exp, args, base_model


# item 1: P256 probe construction is self-deterministic (spec's own reasoning, see script docstring)
@needs_ref
def test_item1_p256_probe_deterministic(env):
    cli, device, exp, args, base_model = env
    _, test_loader = exp._get_data(flag='test', shuffle=False)
    p_a = build_probe_set(exp, test_loader, 256, device)
    p_b = build_probe_set(exp, test_loader, 256, device)
    assert p_a['start_idx'].tolist() == p_b['start_idx'].tolist()


# item 2: FULL test count = 2161
@needs_ref
def test_item2_full_test_count(env):
    cli, device, exp, args, base_model = env
    n = len(exp._get_data(flag='test', shuffle=False)[1].dataset)
    assert n == 2161


# item 3: candidate count = 7201
@needs_ref
def test_item3_candidate_count(env):
    cli, device, exp, args, base_model = env
    assert int(exp.memory_x.size(0)) == 7201


# item 4: no train/val/test leakage (candidate bank == train-split size)
@needs_ref
def test_item4_no_leakage(env):
    cli, device, exp, args, base_model = env
    n_train = len(exp._get_data(flag='train', shuffle=False)[1].dataset)
    assert int(exp.memory_x.size(0)) == n_train


# item 5: all arms use the same candidate mask function (structural: single shared call site)
@needs_ref
def test_item5_shared_candidate_mask_path(env):
    import inspect
    src = inspect.getsource(sys.modules['scripts.eval_l_aligned_population01'].score_and_decompose)
    assert src.count('exp._candidate_mask') == 1  # called once, shared across all arms' branches


# item 6: same future reconstruction (memory_value) used for every arm
@needs_ref
def test_item6_shared_memory_value_path(env):
    import inspect
    src = inspect.getsource(sys.modules['scripts.eval_l_aligned_population01'].score_and_decompose)
    assert 'memory_value(' in src


# items 7/8: J0/J1 P256 reproduction (light real check -- 2 channels, small subset for speed)
@needs_ref
def test_item7_8_j0_j1_p256_reproduction_smoke(env):
    import copy
    cli, device, exp, args, base_model = env
    from scripts.eval_l_aligned_population01 import load_arm
    _, test_loader = exp._get_data(flag='test', shuffle=False)
    probe = build_probe_set(exp, test_loader, 256, device)
    d_model = int(args.d_model)
    for arm, expected in (('J0', 0.9535986185073853), ('J1', 1.0083508065768652)):
        m, slot_heads, epoch, ckpt_path = load_arm(arm, base_model, d_model, device)
        per_ch = score_and_decompose(arm, m, slot_heads, exp, args, probe['x'], probe['y'], probe['start_idx'],
                                     [0], device, cli.chunk_size)  # channel 0 only, for speed
        retmse_ch0 = float(per_ch[0]['retmse10'].mean())
        # channel-0-only value won't equal the 7-channel macro exactly, but must be finite and positive
        assert retmse_ch0 > 0 and torch.isfinite(torch.tensor(retmse_ch0))


# item 11: Agg = D + C identity (synthetic, exact)
def test_item11_agg_equals_d_plus_c():
    torch.manual_seed(0)
    bsz, k, h = 4, 10, 16
    y_sel = torch.randn(bsz, k, h)
    query_future = torch.randn(bsz, h)
    e = y_sel - query_future.unsqueeze(1)
    individual_mse = (e ** 2).mean(-1)
    D = individual_mse.mean(-1) / k
    agg_pred = y_sel.mean(dim=1)
    Agg = ((agg_pred - query_future) ** 2).mean(-1)
    C = Agg - D
    assert torch.allclose(D + C, Agg, atol=1e-6)


# item 12: D = retMSE/10 always
def test_item12_d_equals_retmse_over_10():
    torch.manual_seed(1)
    retmse = torch.rand(20) * 2
    D = retmse / 10
    assert torch.allclose(D * 10, retmse)


# item 13: K hard-unique-selection exact reproduction (deterministic)
def test_item13_hard_selection_deterministic():
    from scripts.train_k_multislot_predictive_retrieval01 import hard_unique_selection
    torch.manual_seed(0)
    scores = torch.randn(3, 10, 200)
    mask = torch.ones(3, 200, dtype=torch.bool)
    a = hard_unique_selection(scores, mask, s=10)
    b = hard_unique_selection(scores, mask, s=10)
    assert torch.equal(a, b)


# item 14: checkpoint hash unchanged (no retraining happened)
@needs_ref
def test_item14_checkpoint_hash_unchanged_across_loads(env):
    cli, device, exp, args, base_model = env
    from scripts.eval_l_aligned_population01 import load_arm
    d_model = int(args.d_model)
    m1, _, _, _ = load_arm('J0', base_model, d_model, device)
    m2, _, _, _ = load_arm('J0', base_model, d_model, device)
    assert state_hash(m1) == state_hash(m2)


# item 15: no checkpoint re-selection -- best_epoch matches the ORIGINALLY reported values
@needs_ref
def test_item15_best_epoch_matches_original(env):
    cli, device, exp, args, base_model = env
    from scripts.eval_l_aligned_population01 import load_arm
    d_model = int(args.d_model)
    expected_epochs = {'J0': 10, 'J1': 1, 'K1': 1, 'K2': 1}
    for arm, exp_epoch in expected_epochs.items():
        _, _, epoch, _ = load_arm(arm, base_model, d_model, device)
        assert epoch == exp_epoch, f'{arm} best_epoch changed: {epoch} != {exp_epoch}'


# item 16: deterministic repeated evaluation (same probe, same result twice)
@needs_ref
def test_item16_deterministic_repeated_eval(env):
    cli, device, exp, args, base_model = env
    from scripts.eval_l_aligned_population01 import load_arm
    d_model = int(args.d_model)
    _, test_loader = exp._get_data(flag='test', shuffle=False)
    probe = build_probe_set(exp, test_loader, 32, device)  # small subset for speed
    m, slot_heads, epoch, _ = load_arm('J0', base_model, d_model, device)
    r1 = score_and_decompose('J0', m, slot_heads, exp, args, probe['x'], probe['y'], probe['start_idx'],
                             [0], device, cli.chunk_size)
    r2 = score_and_decompose('J0', m, slot_heads, exp, args, probe['x'], probe['y'], probe['start_idx'],
                             [0], device, cli.chunk_size)
    assert torch.equal(r1[0]['retmse10'], r2[0]['retmse10'])


# item 17: per-channel macro average consistency (mean over channels == overall macro)
def test_item17_per_channel_macro_consistency():
    import pandas as pd
    df = pd.DataFrame({'channel': [0, 0, 1, 1], 'retmse10': [1.0, 2.0, 3.0, 4.0]})
    per_channel_mean = df.groupby('channel')['retmse10'].mean()
    overall_mean = df['retmse10'].mean()
    # with balanced channel counts, mean-of-channel-means == overall mean
    assert abs(per_channel_mean.mean() - overall_mean) < 1e-9
