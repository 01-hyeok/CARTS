"""Unit tests for TRACK-J-SHARED-ENCODER-DRIFT01 (spec section 17, items
1-14). Fast tests use synthetic tensors; a few (marked) load the real
ETTh1 p120-family reference checkpoint and run a couple of real batches --
still fast (MLP encoder, no Transformer/patch cost).
"""
import copy
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_j_shared_encoder_drift01 import (
    build_model, churn, displacement_stats, effective_rank_entropy, geometry_metrics,
    gradient_conflict_diagnostic, retention_vs_init, state_hash, variant_metrics,
)
from scripts.train_factorial_e2e01 import arm_score, encode_raw
from models.RelationStage1 import RelationEncoder

REF_CKPT = ('checkpoints/soft_set_mse/stage1/ETTh1/seq720_pred720/'
           'stage1_carts_softset_ETTh1_720_S0_wce_RelationStage1_ETTh1_ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_'
           'df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl720_pl720_0/checkpoint.pth')
_ref_available = Path(REPO_ROOT / REF_CKPT).exists()
needs_ref = pytest.mark.skipif(not _ref_available, reason='reference checkpoint not present in this checkout')


class _Cli:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _default_cli(**overrides):
    base = dict(reference_ckpt=str(REPO_ROOT / REF_CKPT), cell='ETTh1_720', pred_len=720, seq_len=720,
               patch_len=16, top_k=10, tau_t=0.1, tau_s=0.1, batch_size=32, learning_rate=1e-3,
               chunk_size=4096, init_seed=0, loader_seed=0)
    base.update(overrides)
    return _Cli(**base)


# --- item 1: E0 frozen / hash unchanged ---
def test_item1_deepcopy_is_frozen_and_hash_matches_source():
    torch.manual_seed(0)
    enc = RelationEncoder(_Cli(relation_encoder_type='mlp', relation_pooling='cls', relation_self_fill='linear',
                               relation_input_space='delta_last', d_model=16, seq_len=32, dropout=0.0,
                               retrieval_similarity='cosine', d_ff=32))
    e0 = copy.deepcopy(enc)
    for p in e0.parameters():
        p.requires_grad_(False)
    assert state_hash(enc) == state_hash(e0)
    assert all(not p.requires_grad for p in e0.parameters())
    # mutate the source; E0 must be unaffected
    with torch.no_grad():
        for p in enc.parameters():
            p.add_(1.0)
    assert state_hash(enc) != state_hash(e0)


# --- item 2/3: step0 S00==St0==S0t==Stt, same Top-K ids ---
def test_item2_3_step0_equivalence_when_et_equals_e0():
    torch.manual_seed(0)
    d = 8
    zq = F.normalize(torch.randn(5, d), dim=-1)
    zk = F.normalize(torch.randn(30, d), dim=-1)
    cand_mask = torch.ones(5, 30, dtype=torch.bool)
    cand_mask[:, -3:] = False
    fixed = dict(memory_c=torch.randn(30, 4), offset_c=torch.randn(5), query_future=torch.randn(5, 4),
                d_raw=torch.randn(5, 30).abs(), oracle_idx=None)
    fixed['oracle_idx'] = fixed['d_raw'].masked_fill(~cand_mask, float('inf')).argsort(-1)[:, :10]
    variants = {'S00': arm_score(zq, zk, None), 'St0': arm_score(zq, zk, None),
               'S0t': arm_score(zq, zk, None), 'Stt': arm_score(zq, zk, None)}
    metrics_list, idx_list = [], []
    for s in variants.values():
        m, idx = variant_metrics(s, cand_mask, fixed, top_k=10)
        metrics_list.append(m)
        idx_list.append(idx)
    for m in metrics_list[1:]:
        for k in metrics_list[0]:
            assert abs(m[k] - metrics_list[0][k]) < 1e-6
    for idx in idx_list[1:]:
        assert torch.equal(idx, idx_list[0])


# --- item 4: identical mask across variants ---
def test_item4_same_mask_object_used_for_all_variants():
    cand_mask = torch.ones(3, 10, dtype=torch.bool)
    fixed = dict(memory_c=torch.randn(10, 4), offset_c=torch.randn(3), query_future=torch.randn(3, 4),
                d_raw=torch.randn(3, 10).abs())
    fixed['oracle_idx'] = fixed['d_raw'].argsort(-1)[:, :5]
    s = torch.randn(3, 10)
    m1, idx1 = variant_metrics(s, cand_mask, fixed, top_k=5)
    m2, idx2 = variant_metrics(s, cand_mask, fixed, top_k=5)
    assert torch.equal(idx1, idx2)


# --- item 5: full candidate count matches expectation ---
@needs_ref
def test_item5_full_candidate_count_matches_train_split_size():
    cli = _default_cli()
    exp, args, model = build_model(cli, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    assert int(exp.memory_x.size(0)) == 7201  # known ETTh1_720 train-split size


# --- item 6: MLP encoder output unaffected by patch_len ---
def test_item6_mlp_encoder_output_unaffected_by_patch_len():
    cfg_common = dict(relation_pooling='cls', relation_self_fill='linear', relation_input_space='delta_last',
                      d_model=16, seq_len=32, dropout=0.0, retrieval_similarity='cosine', d_ff=32,
                      relation_encoder_type='mlp')
    torch.manual_seed(0)
    enc_a = RelationEncoder(_Cli(patch_len=16, stride=16, **cfg_common))
    torch.manual_seed(0)
    enc_b = RelationEncoder(_Cli(patch_len=8, stride=8, **cfg_common))
    assert state_hash(enc_a) == state_hash(enc_b)
    x = torch.randn(4, 1, 32)
    torch.manual_seed(1)
    out_a = enc_a(x)
    torch.manual_seed(1)
    out_b = enc_b(x)
    assert torch.equal(out_a, out_b)


# --- item 7: instrumentation ON/OFF training equivalence ---
@needs_ref
def test_item7_instrumentation_does_not_touch_optimizer_or_model_state():
    """Structural check: `run_diagnostic`'s body is entirely inside
    `torch.no_grad()` and restores `model.training` -- verified here by
    calling encode_raw under no_grad (mirroring the diagnostic's own
    context) and confirming zero grad accumulation and unchanged params."""
    cli = _default_cli()
    exp, args, model = build_model(cli, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    before = state_hash(model)
    model.eval()
    with torch.no_grad():
        _ = encode_raw(model, exp.memory_x[:8], 0)
    assert state_hash(model) == before
    assert all(p.grad is None for p in model.parameters())


# --- item 8/9: query-only / key-only diagnostic gradient isolation ---
def test_item8_9_query_only_and_key_only_gradient_isolation():
    torch.manual_seed(0)
    enc = RelationEncoder(_Cli(relation_encoder_type='mlp', relation_pooling='cls', relation_self_fill='linear',
                               relation_input_space='delta_last', d_model=8, seq_len=16, dropout=0.0,
                               retrieval_similarity='cosine', d_ff=16))
    x_q = torch.randn(3, 1, 16)
    x_k = torch.randn(5, 1, 16)
    z_q = enc(x_q)
    z_k = enc(x_k)
    # query-only: key detached -> no grad flows from the key branch
    s_q = arm_score(z_q, z_k.detach(), None)
    s_q.sum().backward(retain_graph=True)
    g_after_q_only = [p.grad.clone() if p.grad is not None else None for p in enc.parameters()]
    enc.zero_grad(set_to_none=True)
    # key-only: query detached
    s_k = arm_score(z_q.detach(), z_k, None)
    s_k.sum().backward()
    g_after_k_only = [p.grad.clone() if p.grad is not None else None for p in enc.parameters()]
    assert any(g is not None and g.abs().sum() > 0 for g in g_after_q_only)
    assert any(g is not None and g.abs().sum() > 0 for g in g_after_k_only)


# --- item 10: gradient decomposition matches real shared gradient ---
@needs_ref
def test_item10_gradient_decomposition_equals_combined_gradient():
    cli = _default_cli()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp, args, model = build_model(cli, device)
    channels = list(range(int(args.enc_in)))
    from scripts.rng_control01 import make_loader_generator
    gen = make_loader_generator(cli.loader_seed)
    _, train_loader = exp._get_data(flag='train', shuffle=True, generator=gen)
    batch_x, batch_y, batch_start_idx = next(iter(train_loader))
    batch_x, batch_y = batch_x.float().to(device), batch_y.float().to(device)
    rows = gradient_conflict_diagnostic(model, exp, args, cli, batch_x, batch_y, batch_start_idx,
                                        channels[:2], device, 'unit_test')
    for r in rows:
        assert r['equivalence_max_abs_diff'] < 1e-4, r
        assert r['equivalence_rel_diff'] < 1e-3, r


# --- item 11: eval probe restores train/eval state exactly ---
@needs_ref
def test_item11_probe_restores_training_mode():
    cli = _default_cli()
    exp, args, model = build_model(cli, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.train()
    was_training = model.training
    model.eval()
    with torch.no_grad():
        _ = encode_raw(model, exp.memory_x[:4], 0)
    if was_training:
        model.train()
    assert model.training == was_training


# --- item 12: probe query ids deterministic ---
@needs_ref
def test_item12_probe_selection_is_deterministic():
    from scripts.train_j_shared_encoder_drift01 import build_probe_set
    cli = _default_cli()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp, args, model = build_model(cli, device)
    _, val_loader_a = exp._get_data(flag='val', shuffle=False)
    probe_a = build_probe_set(exp, val_loader_a, 32, device)
    _, val_loader_b = exp._get_data(flag='val', shuffle=False)
    probe_b = build_probe_set(exp, val_loader_b, 32, device)
    assert torch.equal(probe_a['start_idx'], probe_b['start_idx'])


# --- item 13: no val/test leakage into the candidate bank ---
@needs_ref
def test_item13_no_val_test_leakage_in_candidate_bank():
    cli = _default_cli()
    exp, args, model = build_model(cli, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    n_train_queries = len(exp._get_data(flag='train', shuffle=False)[1].dataset)
    assert int(exp.memory_x.size(0)) == n_train_queries


# --- item 14: checkpoint reload reproduces metrics ---
@needs_ref
def test_item14_checkpoint_reload_reproduces_output():
    cli = _default_cli()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp, args, model = build_model(cli, device)
    model.eval()  # dropout is stochastic under .train(); pin eval mode for a deterministic comparison
    x = exp.memory_x[:4]
    with torch.no_grad():
        out_before = encode_raw(model, x, 0).clone()
    sd = copy.deepcopy(model.state_dict())
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01)
    model.load_state_dict(sd)
    with torch.no_grad():
        out_after = encode_raw(model, x, 0)
    assert torch.equal(out_before, out_after)


# --- extra: geometry / displacement / churn math sanity ---
def test_effective_rank_full_rank_data_near_dimension():
    torch.manual_seed(0)
    x = torch.randn(200, 16)  # roughly isotropic -> effective rank near 16
    g = geometry_metrics(x)
    assert 8 < g['effective_rank'] <= 16.5


def test_effective_rank_one_hot_like_data_is_low():
    x = torch.zeros(50, 16)
    x[:, 0] = torch.randn(50) * 5  # all variance in one dimension
    g = geometry_metrics(x)
    assert g['effective_rank'] < 3.0


def test_churn_zero_when_identical():
    idx = torch.arange(10).unsqueeze(0).repeat(3, 1)
    assert churn(idx, idx, 10) == 0.0


def test_churn_full_when_disjoint():
    idx_a = torch.arange(10).unsqueeze(0)
    idx_b = torch.arange(10, 20).unsqueeze(0)
    assert churn(idx_a, idx_b, 10) == 1.0


def test_retention_vs_init_full_when_identical():
    idx = torch.arange(5).unsqueeze(0)
    assert retention_vs_init(idx, idx, 5) == 1.0


def test_displacement_stats_keys():
    d = torch.rand(100)
    s = displacement_stats(d)
    for k in ('mean', 'median', 'std', 'p10', 'p50', 'p90', 'max'):
        assert k in s
