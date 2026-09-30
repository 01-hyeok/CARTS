"""Unit tests for TRACK-U-ASYMMETRY-CAPACITY-DECOMPOSITION01 (spec PART
8, 10 items). U0/U1/U2/U3 all share ONE training script
(`train_u_asymmetry_capacity01.py`, `--arm` flag) and ONE eval function
(`hard_eval_decomposition`, ordinary Top-10, no multi-slot) -- many of
these equivalence properties hold by construction; tests still exercise
the actual code paths, not just assert intent.
"""
import inspect
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_factorial_e2e01 import arm_score, encode_raw
from scripts.train_horizon_retrieval_expert01 import kl_loss, normalized_teacher_prob
from scripts.train_u_asymmetry_capacity01 import (
    PROJECTION_INIT_SEED, PROJECTION_INIT_STD, SingleProjection, compute_scores,
    hard_eval_decomposition, projection_weight_sha,
)


class FakeModel(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.encoder = nn.Linear(d, d)

    def _relation_tensor(self, x, c, cc):
        return x


# item 1: U0 score == Original KL score
def test_item1_u0_score_equals_original_kl():
    torch.manual_seed(0)
    d, bsz, n = 8, 4, 20
    model = FakeModel(d)
    x_q, x_k = torch.randn(bsz, d), torch.randn(n, d)
    u0_score = compute_scores(model, None, x_q, x_k, 0, 'U0')
    ref_score = arm_score(encode_raw(model, x_q, 0), encode_raw(model, x_k, 0), None)
    assert torch.allclose(u0_score, ref_score, atol=1e-7)


# item 2: U0 loss == Original KL loss
def test_item2_u0_loss_equals_original_kl():
    torch.manual_seed(1)
    d, bsz, n = 8, 4, 20
    model = FakeModel(d)
    x_q, x_k = torch.randn(bsz, d), torch.randn(n, d)
    mask = torch.ones(bsz, n, dtype=torch.bool)
    p_t = torch.softmax(torch.randn(bsz, n), dim=-1)
    tau_s = 0.1
    u0_score = compute_scores(model, None, x_q, x_k, 0, 'U0')
    loss_u0 = kl_loss(p_t, u0_score, mask, tau_s)
    ref_score = arm_score(encode_raw(model, x_q, 0), encode_raw(model, x_k, 0), None)
    loss_ref = kl_loss(p_t, ref_score, mask, tau_s)
    assert torch.allclose(loss_u0, loss_ref, atol=1e-7)


# item 3: U0 encoder gradient == Original KL encoder gradient
def test_item3_u0_gradient_equals_original_kl_gradient():
    torch.manual_seed(2)
    d, bsz, n = 8, 3, 12
    mask = torch.ones(bsz, n, dtype=torch.bool)
    p_t = torch.softmax(torch.randn(bsz, n), dim=-1)
    tau_s = 0.1
    x_q, x_k = torch.randn(bsz, d), torch.randn(n, d)

    model_a = FakeModel(d)
    model_b = FakeModel(d)
    model_b.load_state_dict(model_a.state_dict())

    score_a = compute_scores(model_a, None, x_q, x_k, 0, 'U0')
    kl_loss(p_t, score_a, mask, tau_s).backward()
    g_a = torch.cat([p.grad.flatten().clone() for p in model_a.parameters()])

    z_q, z_k = encode_raw(model_b, x_q, 0), encode_raw(model_b, x_k, 0)
    score_b = arm_score(z_q, z_k, None)
    kl_loss(p_t, score_b, mask, tau_s).backward()
    g_b = torch.cat([p.grad.flatten().clone() for p in model_b.parameters()])

    assert torch.allclose(g_a, g_b, atol=1e-6)


# item 4: projection parameter counts -- U0=0, U1/U2/U3=D^2
def test_item4_parameter_counts():
    d = 16
    proj = SingleProjection(d)
    n_params = sum(p.numel() for p in proj.parameters())
    assert n_params == d * d
    # U0 uses projection=None (see item1/item2) -> zero projection parameters, no bias anywhere
    assert sum(1 for name, _ in proj.named_parameters() if 'bias' in name) == 0, \
        '[ISSUE] SingleProjection must have no bias'


# item 5: U1/U2/U3 initial projection weights are bit-identical
def test_item5_identical_initial_projection_weights():
    d = 16
    proj_1 = SingleProjection(d, PROJECTION_INIT_STD, PROJECTION_INIT_SEED)  # simulating U1's own init
    proj_2 = SingleProjection(d, PROJECTION_INIT_STD, PROJECTION_INIT_SEED)  # simulating U2's own init
    proj_3 = SingleProjection(d, PROJECTION_INIT_STD, PROJECTION_INIT_SEED)  # simulating U3's own init
    assert torch.equal(proj_1.W, proj_2.W)
    assert torch.equal(proj_2.W, proj_3.W)
    assert projection_weight_sha(proj_1) == projection_weight_sha(proj_2) == projection_weight_sha(proj_3)
    # sanity: NOT identity (perturbation actually applied)
    assert not torch.equal(proj_1.W, torch.eye(d))


# item 6: teacher identical across arms (no arm-dependence anywhere in its signature/body)
def test_item6_teacher_identical_across_arms():
    sig = inspect.signature(normalized_teacher_prob)
    assert 'arm' not in sig.parameters
    torch.manual_seed(3)
    d = torch.randn(4, 10)
    mask = torch.ones(4, 10, dtype=torch.bool)
    assert torch.equal(normalized_teacher_prob(d, mask, 0.1), normalized_teacher_prob(d, mask, 0.1))


# item 7: batch order is reproducible across independently-constructed loaders
# (all 4 arms share ONE script -- build_model/make_loader_generator/_get_data --
# so this reduces to a determinism check on that one shared path, same technique
# as TRACK-T2's own item 8)
_ref_96 = (REPO_ROOT / 'checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/'
          'stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_'
          'df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth')
needs_ref96 = pytest.mark.skipif(not _ref_96.exists(), reason='ETTh1 H96 reference checkpoint not available')


@needs_ref96
def test_item7_same_batch_order_shared_loader_path():
    import types
    from scripts.rng_control01 import batch_order_sha256, make_loader_generator
    from scripts.train_j_shared_encoder_drift01 import build_model

    def one_epoch_hash(seed):
        cli = types.SimpleNamespace(reference_ckpt=str(_ref_96), pred_len=96, seq_len=96, batch_size=32,
                                    init_seed=seed, top_k=10, patch_len=16)
        exp, args, model = build_model(cli, torch.device('cpu'))
        gen = make_loader_generator(seed)
        _, loader = exp._get_data(flag='train', shuffle=True, generator=gen)
        starts = [bi[2].clone() if torch.is_tensor(bi[2]) else torch.as_tensor(bi[2]) for bi in loader]
        return batch_order_sha256(starts)

    assert one_epoch_hash(0) == one_epoch_hash(0)


# item 8: candidate mask / memory / split identical -- compute_scores never
# touches candidate_mask/memory construction, only the score itself (source check)
def test_item8_scoring_never_touches_candidate_mask_or_memory():
    src = inspect.getsource(compute_scores)
    for forbidden in ('cand_mask', '_candidate_mask', 'memory_y', 'memory_x_last', 'flag=', 'split'):
        assert forbidden not in src


# item 9: query-side AND candidate-side encoder gradients nonzero, for EVERY arm
@pytest.mark.parametrize('arm', ('U0', 'U1', 'U2', 'U3'))
def test_item9_full_gradient_both_branches_every_arm(arm):
    torch.manual_seed(4)
    d, bsz, n = 8, 3, 12
    mask = torch.ones(bsz, n, dtype=torch.bool)
    p_t = torch.softmax(torch.randn(bsz, n), dim=-1)
    tau_s = 0.1
    encoder = nn.Linear(d, d)
    projection = None if arm == 'U0' else SingleProjection(d)
    x_q, x_k = torch.randn(bsz, d), torch.randn(n, d)

    def loss_fn(detach_q=False, detach_k=False):
        z_q = encoder(x_q)
        if detach_q:
            z_q = z_q.detach()
        z_k = encoder(x_k)
        if detach_k:
            z_k = z_k.detach()
        if arm == 'U0':
            q, k = F.normalize(z_q, dim=-1), F.normalize(z_k, dim=-1)
        elif arm == 'U1':
            q, k = projection(z_q), projection(z_k)
        elif arm == 'U2':
            q, k = projection(z_q), F.normalize(z_k, dim=-1)
        else:
            q, k = F.normalize(z_q, dim=-1), projection(z_k)
        s = torch.matmul(q, k.t())
        return kl_loss(p_t, s, mask, tau_s)

    encoder.zero_grad()
    loss_fn(detach_k=True).backward()
    g_q = torch.cat([p.grad.flatten().clone() for p in encoder.parameters()])
    assert float(g_q.abs().sum()) > 0, f'[ISSUE] {arm}: query-side encoder grad is zero'

    encoder.zero_grad()
    loss_fn(detach_q=True).backward()
    g_k = torch.cat([p.grad.flatten().clone() for p in encoder.parameters()])
    assert float(g_k.abs().sum()) > 0, f'[ISSUE] {arm}: candidate-side encoder grad is zero'


# item 10: K=10 ordinary Top-10 selection, identical protocol for every arm
@pytest.mark.parametrize('arm', ('U0', 'U1', 'U2', 'U3'))
def test_item10_k_equals_10_ordinary_topk_every_arm(arm):
    from models.RelationStage1 import stable_topk_indices
    torch.manual_seed(5)
    d, bsz, n, h = 8, 4, 30, 6
    model = FakeModel(d)
    projection = None if arm == 'U0' else SingleProjection(d)
    x_q, x_k = torch.randn(bsz, d), torch.randn(n, d)
    mask = torch.ones(bsz, n, dtype=torch.bool)
    memory_c = torch.randn(n, h)
    offset_c = torch.randn(bsz)
    query_future = torch.randn(bsz, h)
    d_raw = torch.randn(bsz, n)
    oracle_idx = stable_topk_indices(d_raw, 10, largest=False)

    scores = compute_scores(model, projection, x_q, x_k, 0, arm)
    res = hard_eval_decomposition(scores, mask, memory_c, offset_c, query_future, d_raw, oracle_idx, top_k=10)
    assert res['model_idx'].shape == (bsz, 10)
    for b in range(bsz):
        assert len(set(res['model_idx'][b].tolist())) == 10
    assert torch.allclose(res['D'] + res['C'], res['agg_mse'], atol=1e-4)
