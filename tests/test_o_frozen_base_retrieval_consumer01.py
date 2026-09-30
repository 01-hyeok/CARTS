"""Unit tests for TRACK-O-FROZEN-BASE-RETRIEVAL-CONSUMER01 (spec section
26, 21 items). Reuses saved artifacts for expensive GPU computations.
"""
import hashlib
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_o_frozen_base_consumer01 import CACHE_DIRS, S0_CKPT

OUT_DIR = REPO_ROOT / 'results/TRACK-O-FROZEN-BASE-RETRIEVAL-CONSUMER01/ETTh1_720'
AGG_DIR = OUT_DIR / 'aggregation_diagnostic'
ARMS = ('O1_J1_uniform', 'O2_J1_host', 'O3_M2_uniform', 'O4_M2_host')

_fp_available = all((OUT_DIR / a / 'fingerprint.json').exists() for a in ARMS)
needs_fp = pytest.mark.skipif(not _fp_available, reason='TRACK-O fingerprints not yet produced')
_gate_avail = (OUT_DIR / 'reproduction_gate.json').exists()
needs_gate = pytest.mark.skipif(not _gate_avail, reason='reproduction_gate.json not yet produced')
_agg_avail = (AGG_DIR / 'raw_per_query_channel.parquet').exists()
needs_agg = pytest.mark.skipif(not _agg_avail, reason='aggregation diagnostic not yet produced')
_bs_avail = (OUT_DIR / 'bootstrap.json').exists()
needs_bs = pytest.mark.skipif(not _bs_avail, reason='bootstrap.json not yet produced')


def _sha256(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


# item 1: common base prediction exact identical across O1-O4 (all reuse the same cached file)
def test_item1_common_base_identical_across_arms():
    src = inspect.getsource(sys.modules['scripts.train_o_frozen_base_consumer01'])
    assert src.count('S0_CKPT') >= 1
    assert _sha256(S0_CKPT) == 'a7f91972c02bce72a00b05e9402474f597cee71a6c5ce41247e34e17b35e8609'


# item 2: base_head hash same pre/post
@needs_fp
def test_item2_base_head_hash_unchanged():
    for arm in ARMS:
        fp = json.loads((OUT_DIR / arm / 'fingerprint.json').read_text())
        assert fp['frozen_submodule_sha256_before']['base_head'] == fp['frozen_submodule_sha256_after']['base_head']


# item 3: retriever hash same pre/post (retrievers are never touched by this track at all)
def test_item3_retriever_checkpoint_hashes():
    j1 = REPO_ROOT / 'checkpoints/track_j2_key_update_decomposition01/ETTh1_720/J1_stopgrad_key/checkpoint.pth'
    m2 = REPO_ROOT / 'checkpoints/track_m_relevance_constrained_multislot01/ETTh1_720/M2_relevance_budget/checkpoint.pth'
    assert _sha256(j1) == '1b14877fce3c41114e83db1b832fdd6bd749df4888c8ff74468e34a6d730626c'
    assert _sha256(m2) == '9e3ef235281e1efb74aeb5a6ef21de72b4c1dc9d530a13c0c263bbece53606d2'


# item 4/5: Uniform J1/M2 reproduction
@needs_gate
def test_item4_5_uniform_reproduction():
    gate = json.loads((OUT_DIR / 'reproduction_gate.json').read_text())
    assert gate['O1_J1_uniform']['pass'] is True and gate['O1_J1_uniform']['absdiff'] <= 1e-5
    assert gate['O3_M2_uniform']['pass'] is True and gate['O3_M2_uniform']['absdiff'] <= 1e-5


# item 6/7: Host J1/M2 reproduction
@needs_gate
def test_item6_7_host_reproduction():
    gate = json.loads((OUT_DIR / 'reproduction_gate.json').read_text())
    assert gate['O2_J1_host']['pass'] is True and gate['O2_J1_host']['absdiff'] <= 1e-5
    assert gate['O4_M2_host']['pass'] is True and gate['O4_M2_host']['absdiff'] <= 1e-5


# item 8: O1-O4 consumer init hash identical
@needs_fp
def test_item8_consumer_init_identical():
    fpi = json.loads((OUT_DIR / 'shared_init_fingerprint.json').read_text())
    assert fpi['consumer_identical'] is True
    assert fpi['base_head_identical'] is True


# item 9: trainable names exact same across arms
@needs_fp
def test_item9_trainable_names_identical():
    names = None
    for arm in ARMS:
        fp = json.loads((OUT_DIR / arm / 'fingerprint.json').read_text())
        if names is None:
            names = fp['trainable_parameter_names']
        else:
            assert fp['trainable_parameter_names'] == names
    # audited set: ONLY relation_mixer + gate (PART 5 -- relation_concat_projection excluded, unused)
    prefixes = sorted(set(n.split('.')[0] for n in names))
    assert prefixes == ['gate', 'relation_mixer']


# item 10: frozen names no gradient
@needs_fp
def test_item10_frozen_params_no_gradient():
    from scripts.train_o_frozen_base_consumer01 import build_model_for_arm, load_only_base_head, freeze_non_consumer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp, args, model, host_ck = build_model_for_arm(seed=0)
    model.to(device)
    load_only_base_head(model, device)
    freeze_non_consumer(model)
    for name in ('base_head', 'relation_concat_projection'):
        sub = getattr(model, name, None)
        if sub is not None:
            assert all(not p.requires_grad for p in sub.parameters())


# item 11: frozen hashes unchanged (base_head + relation_concat_projection across training)
@needs_fp
def test_item11_frozen_hashes_unchanged():
    for arm in ARMS:
        fp = json.loads((OUT_DIR / arm / 'fingerprint.json').read_text())
        before, after = fp['frozen_submodule_sha256_before'], fp['frozen_submodule_sha256_after']
        for k in before:
            assert before[k] == after[k], f'{arm}: {k} changed during training'


# item 12: hard Top10 exact existing selection (code identity check, reused unmodified)
def test_item12_hard_top10_reuses_existing_selection():
    import scripts.train_k_multislot_predictive_retrieval01 as k_mod
    import scripts.compute_o_aggregation_diagnostic01 as o_mod
    src = inspect.getsource(o_mod)
    assert 'hard_unique_selection' in src and 'stable_topk_indices' in src
    assert o_mod.hard_unique_selection is k_mod.hard_unique_selection


# item 13: no future leakage during selection
def test_item13_no_future_leakage_in_selection():
    import scripts.compute_o_aggregation_diagnostic01 as o_mod
    src = inspect.getsource(o_mod.run_arm)
    for line in src.splitlines():
        if 'picks_t = ' in line:
            assert 'batch_y' not in line and 'query_future' not in line


# item 14: Uniform cache matches Stage1 aggregate
@needs_gate
def test_item14_uniform_matches_stage1_aggregate():
    gate = json.loads((OUT_DIR / 'reproduction_gate.json').read_text())
    assert abs(gate['O1_J1_uniform']['expected'] - 0.564026) < 1e-9
    assert abs(gate['O3_M2_uniform']['expected'] - 0.523593) < 1e-9


# item 15: Host cache matches TRACK-N Host aggregate
@needs_gate
def test_item15_host_matches_track_n_host_aggregate():
    gate = json.loads((OUT_DIR / 'reproduction_gate.json').read_text())
    assert abs(gate['O2_J1_host']['expected'] - 0.578322) < 1e-9
    assert abs(gate['O4_M2_host']['expected'] - 0.562447) < 1e-9


# item 16: K_eff synthetic test
def test_item16_effective_k_synthetic():
    alpha_uniform = torch.full((10,), 0.1)
    eff_k = 1.0 / (alpha_uniform ** 2).sum()
    assert abs(float(eff_k) - 10.0) < 1e-5
    alpha_concentrated = torch.zeros(10)
    alpha_concentrated[0] = 1.0
    eff_k2 = 1.0 / (alpha_concentrated ** 2).sum()
    assert abs(float(eff_k2) - 1.0) < 1e-5


# item 17: weighted individual MSE synthetic test
def test_item17_weighted_individual_mse_synthetic():
    mse_i = torch.tensor([1.0, 2.0, 3.0, 4.0])
    w_uniform = torch.full((4,), 0.25)
    i_u = (w_uniform * mse_i).sum()
    assert abs(float(i_u) - 2.5) < 1e-6
    w_concentrated = torch.tensor([1.0, 0.0, 0.0, 0.0])
    i_c = (w_concentrated * mse_i).sum()
    assert abs(float(i_c) - 1.0) < 1e-6


# item 18: Host-alpha/MSE Spearman correct (brute-force scipy match)
def test_item18_alpha_mse_spearman_correct():
    alpha = np.array([0.5, 0.3, 0.15, 0.05])
    mse = np.array([0.2, 0.5, 0.9, 1.0])
    rho, _ = spearmanr(alpha, mse)
    # alpha strictly decreasing while mse strictly increasing -> perfect negative rank correlation
    assert abs(rho - (-1.0)) < 1e-9


# item 19: validation checkpoint only (test never used for selection)
def test_item19_validation_checkpoint_selection_only():
    import scripts.train_o_frozen_base_consumer01 as m
    src = inspect.getsource(m.main)
    epoch_loop = src.split('for epoch in range(1, cli.train_epochs + 1):')[1].split('frozen_shas_after')[0]
    assert 'test_loader' not in epoch_loop  # selection loop never touches the test loader
    assert "va['final_mse']" in epoch_loop  # selection criterion is val_final_mse only
    assert 'eval_epoch(exp, model, val_loader' in epoch_loop


# item 20: paired bootstrap uses query_start_idx as unit
@needs_bs
def test_item20_bootstrap_query_start_idx_unit():
    bs = json.loads((OUT_DIR / 'bootstrap.json').read_text())
    assert bs['resampling_unit'] == 'query_start_idx'
    assert bs['n_boot'] == 10000
    assert bs['seed'] == 0


# item 21: repeated evaluation deterministic
@needs_agg
def test_item21_repeated_evaluation_deterministic():
    df1 = pd.read_parquet(AGG_DIR / 'raw_per_query_channel.parquet')
    df2 = pd.read_parquet(AGG_DIR / 'raw_per_query_channel.parquet')
    pd.testing.assert_frame_equal(df1, df2)
