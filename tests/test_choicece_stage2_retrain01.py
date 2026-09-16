"""TRACK-A-CHOICECE-STAGE2-RETRAIN01 -- unit tests.

Scope note (reported plainly in the experiment's final report too): this
covers the highest-risk correctness claims -- frozen/trainable module
separation, cache lookup correctness/order-invariance, and the forced-
retrieval forward pass producing finite loss/gradient with the frozen
submodules' hashes unchanged -- using the REAL `Exp_Stage2_Relation`/
`RelationStage2` classes built from the real ETTh1_96 S0_wce host config
(CPU-only, no GPU1 use). It is NOT the full ~40-item battery the spec
enumerates; the remaining items (online-vs-cache Top-K bit-identity,
duplicate/invalid counting, per-channel cache coverage, NaN-query-future
invariance, checkpoint-selection-uses-validation-only, etc.) are exercised
implicitly by `build_choicece_retrieval_cache01.py`'s own runtime
assertions and print diagnostics, but are not yet re-verified by a
dedicated pytest here -- see the experiment report's NOT EXECUTED list.
"""
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_choicece_stage2_retrain01 import (FREEZE_SUBMODULES, build_fresh_stage2,
                                                      freeze_retrieval_submodules,
                                                      load_cache_as_lookup, lookup_batch)
from scripts.train_factorial_e2e01 import state_sha

S2_96 = ('checkpoints/stage2/ETTh1/seq96_pred96/stage2_carts_softset_s2_ETTh1_96_S0_wce_'
        'RelationStage2_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_'
        'ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth')


def _host_available():
    return (REPO_ROOT / S2_96).exists()


pytestmark = pytest.mark.skipif(not _host_available(), reason='ETTh1_96 S0_wce host checkpoint not present')


# --------------------------------------------------------------------------
# Frozen / trainable separation (real model, CPU only)
# --------------------------------------------------------------------------
def test_build_fresh_stage2_does_not_load_host_trained_weights():
    """The fresh model's base_head weights must NOT equal the host
    checkpoint's own trained base_head weights (spec: WCE gate/head
    parameters must never be loaded into the new model)."""
    exp, args, model, host_ck = build_fresh_stage2(S2_96, seed=1)
    host_state = host_ck['model_state_dict']
    fresh_bh_keys = [k for k in model.state_dict() if k.startswith('base_head.')]
    assert fresh_bh_keys, 'expected at least one base_head parameter'
    any_differs = False
    for k in fresh_bh_keys:
        if k in host_state and not torch.equal(model.state_dict()[k].float().cpu(), host_state[k].float().cpu()):
            any_differs = True
            break
    assert any_differs, 'fresh base_head matches host base_head bit-for-bit -- host weights may have leaked in'


def test_freeze_retrieval_submodules_sets_requires_grad_false():
    exp, args, model, host_ck = build_fresh_stage2(S2_96, seed=1)
    freeze_retrieval_submodules(model)
    for name in FREEZE_SUBMODULES:
        sub = getattr(model, name, None)
        if sub is None:
            continue
        assert all(not p.requires_grad for p in sub.parameters()), name


def test_frozen_submodule_sha_reproducible_and_matches_reference_hash_fn():
    exp, args, model, host_ck = build_fresh_stage2(S2_96, seed=1)
    shas = freeze_retrieval_submodules(model)
    for name, sha in shas.items():
        sub = getattr(model, name)
        assert sha == state_sha(sub.state_dict())


def test_trainable_parameters_exclude_frozen_submodules():
    exp, args, model, host_ck = build_fresh_stage2(S2_96, seed=1)
    freeze_retrieval_submodules(model)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    for name in FREEZE_SUBMODULES:
        assert not any(n.startswith(name + '.') for n in trainable), name


def test_optimizer_from_exp_excludes_frozen_params():
    exp, args, model, host_ck = build_fresh_stage2(S2_96, seed=1)
    freeze_retrieval_submodules(model)
    optimizer = exp._select_optimizer()
    opt_params = {id(p) for group in optimizer.param_groups for p in group['params']}
    for name in FREEZE_SUBMODULES:
        sub = getattr(model, name, None)
        if sub is None:
            continue
        for p in sub.parameters():
            assert id(p) not in opt_params, f'{name} parameter leaked into optimizer'


def test_two_arms_same_seed_produce_identical_fresh_init():
    """Two independent build_fresh_stage2 calls with the SAME seed must give
    bit-identical initial weights (the shared-init contract the trainer's
    --shared_init_in/--shared_init_out flags rely on)."""
    exp1, args1, model1, _ = build_fresh_stage2(S2_96, seed=1)
    exp2, args2, model2, _ = build_fresh_stage2(S2_96, seed=1)
    assert state_sha(model1.state_dict()) == state_sha(model2.state_dict())


# --------------------------------------------------------------------------
# Cache lookup correctness (synthetic cache, order-invariant)
# --------------------------------------------------------------------------
def _make_synthetic_cache(tmp_path, n=6, channels=7, pred_len=96, d_model=128):
    cache = {
        'batch_start_idx': torch.tensor([100, 5, 300, 1, 200, 50]),
        'relation_outputs': torch.randn(n, channels, 1, pred_len),
        'relation_query_embs': torch.randn(n, channels, 1, d_model),
    }
    path = tmp_path / 'test.pt'
    torch.save(cache, path)
    return path, cache


def test_cache_lookup_matches_by_start_idx_not_by_position(tmp_path):
    path, cache = _make_synthetic_cache(tmp_path)
    loaded, lut = load_cache_as_lookup(path)
    batch_start_idx = torch.tensor([300, 1, 50])  # different order than build order
    out = lookup_batch(loaded, lut, batch_start_idx, torch.device('cpu'))
    expected_rows = [2, 3, 5]  # positions of 300, 1, 50 in the original cache
    assert torch.equal(out['relation_outputs'], cache['relation_outputs'][expected_rows])
    assert torch.equal(out['relation_query_embs'], cache['relation_query_embs'][expected_rows])


def test_cache_lookup_shuffled_batch_order_gives_same_per_query_values(tmp_path):
    path, cache = _make_synthetic_cache(tmp_path)
    loaded, lut = load_cache_as_lookup(path)
    a = lookup_batch(loaded, lut, torch.tensor([5, 100]), torch.device('cpu'))
    b = lookup_batch(loaded, lut, torch.tensor([100, 5]), torch.device('cpu'))
    assert torch.equal(a['relation_outputs'][0], b['relation_outputs'][1])
    assert torch.equal(a['relation_outputs'][1], b['relation_outputs'][0])


# --------------------------------------------------------------------------
# forward_from_retrieval_values: finite loss/gradient, frozen hash unchanged
# --------------------------------------------------------------------------
def test_forward_from_retrieval_values_finite_loss_and_selective_gradient():
    exp, args, model, host_ck = build_fresh_stage2(S2_96, seed=1)
    exp._ensure_memory()
    exp._build_key_bank(force=True)
    shas_before = freeze_retrieval_submodules(model)

    _, loader = exp._get_data(flag='train', shuffle=False)
    batch_x, batch_y, batch_start_idx = next(iter(loader))
    batch_x, batch_y, batch_start_idx = exp._move_batch(batch_x, batch_y, batch_start_idx)
    cand_mask, counts = exp._candidate_mask(batch_start_idx)
    valid_query = counts > 0

    bsz = batch_x.size(0)
    channels = model.channels
    slots = model.num_source_slots()
    relation_outputs = torch.randn(bsz, channels, slots, model.pred_len)
    relation_query_embs = torch.randn(bsz, channels, slots, int(args.d_model))
    rcache = {'relation_outputs': relation_outputs, 'relation_query_embs': relation_query_embs}

    optimizer = exp._select_optimizer()
    optimizer.zero_grad()
    y_final, y_base, y_ret, beta, lam, debug = model.forward_from_retrieval_values(
        relation_outputs, batch_x=batch_x, retrieval_cache=rcache, memory_y=exp.memory_y,
        valid_mask=cand_mask, key_bank=exp.key_bank, memory_x_last=exp.memory_x_last, target_y=batch_y)
    loss = exp._loss(y_final, y_base, y_ret, batch_y, debug, valid_query)
    assert torch.isfinite(loss)
    loss.backward()

    for name in FREEZE_SUBMODULES:
        sub = getattr(model, name, None)
        if sub is None:
            continue
        for p in sub.parameters():
            assert p.grad is None or torch.equal(p.grad, torch.zeros_like(p.grad)), name

    trainable = [p for n, p in model.named_parameters() if p.requires_grad]
    assert any(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
              for p in trainable), 'no trainable parameter received a nonzero finite gradient'

    optimizer.step()
    shas_after = freeze_retrieval_submodules(model)
    assert shas_before == shas_after
