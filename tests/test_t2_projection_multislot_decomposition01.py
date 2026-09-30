"""Unit tests for TRACK-T2-PROJECTION-MULTISLOT-DECOMPOSITION01 (spec
PART 6, 10 items). T0 = TRUE Original KL (zero SlotHeads parameters,
`train_j_shared_encoder_drift01.py` run completely unmodified). T1-T10
are read-only reuses of the existing, already-validated
TRACK-T-PURE-MULTISLOT-VALIDATION01 artifacts -- not retrained here.
"""
import inspect
import re
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_factorial_e2e01 import arm_score
from scripts.train_horizon_retrieval_expert01 import kl_loss
from scripts.train_k_multislot_predictive_retrieval01 import kl_loss_from_prob
from scripts.train_t_pure_multislot01 import round_robin_topk_selection
from scripts.build_t2_true_original_kl_cache01 import compute_scores_true_original_kl

T2_CACHE_SRC = (REPO_ROOT / 'scripts/build_t2_true_original_kl_cache01.py').read_text()
TRAIN_J_SRC = (REPO_ROOT / 'scripts/train_j_shared_encoder_drift01.py').read_text()
TRAIN_T_SRC = (REPO_ROOT / 'scripts/train_t_pure_multislot01.py').read_text()
RUN_T2_SH_SRC = (REPO_ROOT / 'scripts/run_t2_one_setting01.sh').read_text()


class FakeModel(nn.Module):
    """Minimal stand-in for the real RelationStage1 model: `.encoder` is
    a plain Linear, `._relation_tensor(x, c, c)` is identity -- enough
    to exercise `encode_raw`'s call pattern without needing a full
    dataset/experiment."""
    def __init__(self, d):
        super().__init__()
        self.encoder = nn.Linear(d, d)

    def _relation_tensor(self, x, c, cc):
        return x


# item 1: T0 has zero SlotHeads parameters
def test_item1_t0_has_no_slot_heads():
    # no SlotHeads() construction, no import of the class, no slot_heads_state_dict
    # written/read anywhere in T0's cache-building code (docstrings may mention the
    # word in prose explaining what T0 deliberately lacks -- that's fine)
    assert 'SlotHeads(' not in T2_CACHE_SRC
    import_lines = [l for l in T2_CACHE_SRC.splitlines() if l.strip().startswith(('import ', 'from '))]
    assert not any('SlotHeads' in l for l in import_lines), '[ISSUE] SlotHeads imported into T0 cache builder'
    assert 'SlotHeads(' not in RUN_T2_SH_SRC
    # positive: the cache-builder explicitly ASSERTS the loaded T0 checkpoint has
    # no slot_heads_state_dict key (a runtime guard, not just an absence of code)
    assert "'slot_heads_state_dict' not in bl" in T2_CACHE_SRC


# item 2: score equivalence -- compute_scores_true_original_kl == arm_score(encode_raw(...), encode_raw(...), None)
def test_item2_score_equivalence():
    torch.manual_seed(0)
    d, bsz, n = 8, 4, 20
    model = FakeModel(d)
    x_q = torch.randn(bsz, d)
    x_k = torch.randn(n, d)
    from scripts.train_factorial_e2e01 import encode_raw
    ref_score = arm_score(encode_raw(model, x_q, 0), encode_raw(model, x_k, 0), None)
    t0_score = compute_scores_true_original_kl(model, x_q, x_k, 0)
    assert t0_score.shape == (bsz, 1, n)
    assert torch.allclose(t0_score.squeeze(1), ref_score, atol=1e-7)


# item 3: loss equivalence -- T0's loss path matches Original-KL kl_loss exactly
def test_item3_loss_equivalence():
    torch.manual_seed(1)
    d, bsz, n = 8, 4, 20
    model = FakeModel(d)
    x_q = torch.randn(bsz, d)
    x_k = torch.randn(n, d)
    mask = torch.ones(bsz, n, dtype=torch.bool)
    tau_s = 0.1
    p_t = torch.softmax(torch.randn(bsz, n), dim=-1)

    t0_score = compute_scores_true_original_kl(model, x_q, x_k, 0).squeeze(1)  # [B, N]
    loss_t0 = kl_loss(p_t, t0_score, mask, tau_s)

    from scripts.train_factorial_e2e01 import encode_raw
    ref_score = arm_score(encode_raw(model, x_q, 0), encode_raw(model, x_k, 0), None)
    loss_ref = kl_loss(p_t, ref_score, mask, tau_s)
    assert torch.allclose(loss_t0, loss_ref, atol=1e-7)


# item 4: gradient equivalence -- T0's compute path vs Original KL's own training-loop pattern
def test_item4_gradient_equivalence():
    torch.manual_seed(2)
    d, bsz, n = 8, 3, 12
    mask = torch.ones(bsz, n, dtype=torch.bool)
    tau_s = 0.1
    p_t = torch.softmax(torch.randn(bsz, n), dim=-1)
    x_q = torch.randn(bsz, d)
    x_k = torch.randn(n, d)

    model_a = FakeModel(d)
    model_b = FakeModel(d)
    model_b.load_state_dict(model_a.state_dict())

    # path A: T2's own T0 wrapper
    from scripts.train_factorial_e2e01 import encode_raw
    score_a = compute_scores_true_original_kl(model_a, x_q, x_k, 0).squeeze(1)
    loss_a = kl_loss(p_t, score_a, mask, tau_s)
    loss_a.backward()
    g_a = torch.cat([p.grad.flatten().clone() for p in model_a.parameters()])

    # path B: train_j_shared_encoder_drift01.py's own training-loop pattern
    # (s = arm_score(z_q, E, None); ch_loss = kl_loss(p_t, s, cand_mask, tau_s))
    z_q = encode_raw(model_b, x_q, 0)
    E = encode_raw(model_b, x_k, 0)
    s = arm_score(z_q, E, None)
    ch_loss = kl_loss(p_t, s, mask, tau_s)
    ch_loss.backward()
    g_b = torch.cat([p.grad.flatten().clone() for p in model_b.parameters()])

    assert torch.allclose(g_a, g_b, atol=1e-6), '[ISSUE] T0 gradient differs from Original-KL reference gradient'


# item 5: optimizer parameters -- T0's trainer (train_j) only ever optimizes model.parameters()
def test_item5_optimizer_parameters():
    m = re.search(r'optimizer = torch\.optim\.Adam\((.*?), lr=', TRAIN_J_SRC)
    assert m, 'optimizer construction not found in train_j_shared_encoder_drift01.py'
    assert m.group(1).strip() == 'model.parameters()', \
        f'[ISSUE] T0 optimizer includes more than model.parameters(): {m.group(1)}'
    # T1-T10's trainer DOES include slot_heads params (sanity check on the reused script)
    m2 = re.search(r'params = (.*?)\n', TRAIN_T_SRC)
    assert m2 and 'slot_heads.parameters()' in m2.group(1)


# item 6: full gradient (query AND candidate branches nonzero) for the T0 path
def test_item6_full_gradient_both_branches():
    torch.manual_seed(3)
    d, bsz, n = 8, 3, 12
    mask = torch.ones(bsz, n, dtype=torch.bool)
    tau_s = 0.1
    p_t = torch.softmax(torch.randn(bsz, n), dim=-1)
    encoder = nn.Linear(d, d)
    x_q = torch.randn(bsz, d)
    x_k = torch.randn(n, d)

    def loss_fn(detach_q=False, detach_k=False):
        z_q = encoder(x_q)
        if detach_q:
            z_q = z_q.detach()
        z_k = encoder(x_k)
        if detach_k:
            z_k = z_k.detach()
        s = arm_score(z_q, z_k, None)
        return kl_loss(p_t, s, mask, tau_s)

    encoder.zero_grad()
    loss_fn(detach_k=True).backward()
    g_q = torch.cat([p.grad.flatten().clone() for p in encoder.parameters()])
    assert float(g_q.abs().sum()) > 0

    encoder.zero_grad()
    loss_fn(detach_q=True).backward()
    g_k = torch.cat([p.grad.flatten().clone() for p in encoder.parameters()])
    assert float(g_k.abs().sum()) > 0


# item 7: teacher is S-independent (no slot-count parameter anywhere)
def test_item7_teacher_s_independent():
    from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob
    sig = inspect.signature(normalized_teacher_prob)
    assert 'num_slots' not in sig.parameters and 's' not in sig.parameters
    torch.manual_seed(4)
    d = torch.randn(4, 10)
    mask = torch.ones(4, 10, dtype=torch.bool)
    assert torch.equal(normalized_teacher_prob(d, mask, 0.1), normalized_teacher_prob(d, mask, 0.1))


# item 8: same batch order across T0 and T1. `train_j_shared_encoder_drift01.py`
# (T0) never actually persists a batch-order-hash file to disk (it's computed as a
# local variable, `batch_order_epoch1`, but not saved anywhere) -- so this is an
# empirical LIVE check instead: T0's trainer (`train_j_shared_encoder_drift01
# .build_model`) and T1-T10's trainer (`train_t_pure_multislot01.py`) both import
# and call that exact same `build_model` function, then both call
# `make_loader_generator(cli.loader_seed)` -> `exp._get_data(flag='train',
# shuffle=True, generator=train_gen)` with identical arguments -- literally the
# same code path, so this reduces to a determinism check on that shared path.
_ref_96 = (REPO_ROOT / 'checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/'
          'stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_'
          'df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth')
needs_ref96 = pytest.mark.skipif(not _ref_96.exists(), reason='ETTh1 H96 reference checkpoint not available')


@needs_ref96
def test_item8_same_batch_order_shared_loader_path():
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

    h_a = one_epoch_hash(0)  # simulating T0's own loader construction
    h_b = one_epoch_hash(0)  # simulating T1-T10's own loader construction (same shared code path)
    assert h_a == h_b, '[ISSUE] batch order not reproducible across independently-constructed loaders'


# item 9: K=10 for T0's exact score shape convention ([B, 1, N])
def test_item9_k_equals_10_for_t0_shape():
    torch.manual_seed(5)
    bsz, n = 3, 30
    scores = torch.randn(bsz, 1, n)  # T0's own [B, 1, N] convention
    mask = torch.ones(bsz, n, dtype=torch.bool)
    idx = round_robin_topk_selection(scores, mask, k=10)
    assert idx.shape == (bsz, 10)
    for b in range(bsz):
        assert len(set(idx[b].tolist())) == 10


# item 10: T0 selected Top10 == ordinary Top10 (exact index-set equality)
def test_item10_t0_top10_equals_ordinary_top10():
    from models.RelationStage1 import stable_topk_indices
    torch.manual_seed(6)
    d, bsz, n = 8, 4, 30
    model = FakeModel(d)
    x_q = torch.randn(bsz, d)
    x_k = torch.randn(n, d)
    mask = torch.ones(bsz, n, dtype=torch.bool)
    scores = compute_scores_true_original_kl(model, x_q, x_k, 0)  # [B, 1, N]
    rr_idx = round_robin_topk_selection(scores, mask, k=10)
    ordinary_idx = stable_topk_indices(scores.squeeze(1), 10, largest=True)
    for b in range(bsz):
        assert set(rr_idx[b].tolist()) == set(ordinary_idx[b].tolist())
