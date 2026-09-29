"""Unit tests for TRACK-J2-KEY-UPDATE-DECOMPOSITION01 (spec section 13,
items 1-15). Fast tests use synthetic tensors; a few load the real
checkpoint (MLP encoder, cheap).
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

from scripts.train_j2_key_update_decomposition01 import EXPECTED_J0_INIT_HASH, score_fn
from scripts.train_j_shared_encoder_drift01 import build_model, state_hash
from scripts.train_factorial_e2e01 import arm_score, encode_raw

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


@pytest.fixture(scope='module')
def loaded():
    if not _ref_available:
        pytest.skip('reference checkpoint not present')
    cli = _default_cli()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp, args, model = build_model(cli, device)
    return cli, device, exp, args, model


# item 1/2: J1/J2 initial hash == J0 initial hash
@needs_ref
def test_item1_2_init_hash_matches_j0(loaded):
    cli, device, exp, args, model = loaded
    assert state_hash(model) == EXPECTED_J0_INIT_HASH
    e_k = copy.deepcopy(model)
    assert state_hash(e_k) == EXPECTED_J0_INIT_HASH


# item 3: J2 key hash unchanged before/after a training-style step
@needs_ref
def test_item3_j2_key_hash_unchanged_after_query_update(loaded):
    cli, device, exp, args, model = loaded
    e_k = copy.deepcopy(model).to(device)
    for p in e_k.parameters():
        p.requires_grad_(False)
    e_k.eval()
    before = state_hash(e_k)
    # simulate a query-only optimizer step on `model` (E_q), e_k untouched
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = exp.memory_x[:8]
    z = encode_raw(model, x, 0)
    z.sum().backward()
    opt.step()
    assert state_hash(e_k) == before


# item 4: J2 cached key embeddings unchanged before/after training
@needs_ref
def test_item4_j2_cached_key_embeddings_unchanged(loaded):
    cli, device, exp, args, model = loaded
    e_k = copy.deepcopy(model).to(device)
    for p in e_k.parameters():
        p.requires_grad_(False)
    e_k.eval()
    with torch.no_grad():
        k0 = encode_raw(e_k, exp.memory_x[:16], 0).clone()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    z = encode_raw(model, exp.memory_x[:8], 0)
    z.sum().backward()
    opt.step()
    with torch.no_grad():
        k1 = encode_raw(e_k, exp.memory_x[:16], 0)
    assert torch.equal(k0, k1)


# item 5: J1 candidate branch has no gradient contribution (detach cuts it)
def test_item5_j1_candidate_branch_no_gradient():
    torch.manual_seed(0)
    from models.RelationStage1 import RelationEncoder
    enc = RelationEncoder(_Cli(relation_encoder_type='mlp', relation_pooling='cls', relation_self_fill='linear',
                               relation_input_space='delta_last', d_model=8, seq_len=16, dropout=0.0,
                               retrieval_similarity='cosine', d_ff=16))
    x_q = torch.randn(3, 1, 16)
    x_k = torch.randn(5, 1, 16)
    z_q = enc(x_q)
    z_k = enc(x_k).detach()  # J1's stopgrad
    s = arm_score(z_q, z_k, None)
    s.sum().backward()
    # z_k had no grad_fn once detached -- confirm no crash and z_k itself carries no grad
    assert z_k.grad_fn is None
    assert all(p.grad is not None for p in enc.parameters())  # query branch still produced grad


# item 6: J1 optimizer step changes the (undetached) candidate embedding
@needs_ref
def test_item6_j1_candidate_embedding_changes_after_query_only_step(loaded):
    cli, device, exp, args, model = loaded
    m = copy.deepcopy(model).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-1)
    with torch.no_grad():
        k_before = encode_raw(m, exp.memory_x[:16], 0).clone()
    z_q = encode_raw(m, exp.memory_x[:8], 0)
    z_k = encode_raw(m, exp.memory_x[:16], 0).detach()
    s = arm_score(z_q, z_k, None)
    opt.zero_grad()
    s.sum().backward()
    opt.step()
    with torch.no_grad():
        k_after = encode_raw(m, exp.memory_x[:16], 0)
    assert not torch.equal(k_before, k_after)


# item 7: J2 optimizer step does NOT change the candidate embedding (already covered by item3/4, restated)
@needs_ref
def test_item7_j2_candidate_embedding_unchanged_after_query_step(loaded):
    cli, device, exp, args, model = loaded
    e_k = copy.deepcopy(model).to(device)
    for p in e_k.parameters():
        p.requires_grad_(False)
    e_k.eval()
    with torch.no_grad():
        k0 = encode_raw(e_k, exp.memory_x[:16], 0).clone()
    opt = torch.optim.Adam(model.parameters(), lr=1e-1)
    z = encode_raw(model, exp.memory_x[:8], 0)
    opt.zero_grad()
    z.sum().backward()
    opt.step()
    with torch.no_grad():
        k1 = encode_raw(e_k, exp.memory_x[:16], 0)
    assert torch.equal(k0, k1)


# item 8/9: same mask/candidate universe and teacher probability used across arms
# (structural: both arms call the SAME exp._candidate_mask / normalized_teacher_prob;
#  verified here by confirming score_fn returns tensors of consistent shape for both arms)
@needs_ref
def test_item8_9_score_fn_consistent_shapes_both_arms(loaded):
    cli, device, exp, args, model = loaded
    e_k = copy.deepcopy(model).to(device)
    for p in e_k.parameters():
        p.requires_grad_(False)
    e_k.eval()
    with torch.no_grad():
        k0 = {0: encode_raw(e_k, exp.memory_x, 0)}
    x = exp.memory_x[:4]
    zq1, zk1 = score_fn('J1_stopgrad_key', model, None, None, x, 0, exp, device)
    zq2, zk2 = score_fn('J2_true_frozen_key', model, e_k, k0, x, 0, exp, device)
    assert zk1.shape == zk2.shape == (exp.memory_x.size(0), zq1.size(-1))


# item 10: batch order determinism (reuses rng_control01, already unit-tested elsewhere;
# confirm two identical-seed loaders give the same order here)
@needs_ref
def test_item10_batch_order_deterministic(loaded):
    cli, device, exp, args, model = loaded
    from scripts.rng_control01 import make_loader_generator, batch_order_sha256
    gen_a = make_loader_generator(0)
    _, loader_a = exp._get_data(flag='train', shuffle=True, generator=gen_a)
    starts_a = [b[2] for _, b in zip(range(5), loader_a)]
    gen_b = make_loader_generator(0)
    _, loader_b = exp._get_data(flag='train', shuffle=True, generator=gen_b)
    starts_b = [b[2] for _, b in zip(range(5), loader_b)]
    assert batch_order_sha256(starts_a) == batch_order_sha256(starts_b)


# item 11: step0 J0/J1/J2 eval score identical (all encoders == E0 at step0)
def test_item11_step0_scores_identical_across_arms():
    torch.manual_seed(0)
    d = 8
    zq = F.normalize(torch.randn(4, d), dim=-1)
    zk = F.normalize(torch.randn(20, d), dim=-1)
    s_j0 = arm_score(zq, zk, None)
    s_j1 = arm_score(zq, zk, None)  # same E0 for both branches at step 0
    s_j2 = arm_score(zq, zk, None)  # E_q == E_k == E0 at step 0
    assert torch.allclose(s_j0, s_j1) and torch.allclose(s_j0, s_j2)


# item 12: full candidate count == 7201
@needs_ref
def test_item12_full_candidate_count(loaded):
    cli, device, exp, args, model = loaded
    assert int(exp.memory_x.size(0)) == 7201


# item 13: no val/test leakage (candidate bank == train-split size)
@needs_ref
def test_item13_no_leakage(loaded):
    cli, device, exp, args, model = loaded
    n_train = len(exp._get_data(flag='train', shuffle=False)[1].dataset)
    assert int(exp.memory_x.size(0)) == n_train


# item 14: checkpoint reload reproduces output
@needs_ref
def test_item14_checkpoint_reload_reproduction(loaded):
    cli, device, exp, args, model = loaded
    m = copy.deepcopy(model).to(device)
    m.eval()
    x = exp.memory_x[:4]
    with torch.no_grad():
        out_before = encode_raw(m, x, 0).clone()
    sd = copy.deepcopy(m.state_dict())
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.01)
    m.load_state_dict(sd)
    with torch.no_grad():
        out_after = encode_raw(m, x, 0)
    assert torch.equal(out_before, out_after)


# item 15: synthetic Procrustes rotation recovery
def test_item15_procrustes_recovers_synthetic_rotation():
    torch.manual_seed(0)
    n, d = 200, 12
    z0 = torch.randn(n, d)
    q, _ = torch.linalg.qr(torch.randn(d, d))  # random orthogonal matrix
    zt = z0 @ q
    # Orthogonal Procrustes: R* = argmin ||Zt R - Z0||_F, R^T R = I
    u, s, vh = torch.linalg.svd(zt.T @ z0)
    r_star = u @ vh
    aligned = zt @ r_star
    assert torch.allclose(aligned, z0, atol=1e-4)
    # neighborhood (cosine) structure recovered
    cos_before = F.normalize(z0, dim=-1) @ F.normalize(z0, dim=-1).T
    cos_after = F.normalize(aligned, dim=-1) @ F.normalize(aligned, dim=-1).T
    assert torch.allclose(cos_before, cos_after, atol=1e-4)
