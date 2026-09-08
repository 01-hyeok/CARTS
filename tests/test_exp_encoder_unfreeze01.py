"""EXP-ENCODER-UNFREEZE01 mandatory pre-GPU sanity checks (spec section 12):
A. encoder gradient test (query-side AND candidate-side, not detached)
B. frozen baseline regression (requires_grad=False -> exactly no grad)
C. online-vs-stale check (candidate embeddings change immediately when the
   encoder's parameters change -- no cached/stale bank)
D. chunk equivalence (unchunked vs. chunked streaming: scores/losses/every
   gradient -- encoder, SetConditioner, UtilityHead -- match within
   tolerance; multi-chunk execution also exercises `q_fn`/`m_fn`'s
   fresh-recompute-per-call contract, so a double-backward regression in
   either would surface here as a RuntimeError, not a silent tolerance
   miss)
"""
import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.DenseUtilityRetriever import UtilityHead
from models.SequentialSetRetriever import EmptySetToken, SetConditioner
from scripts.train_encoder_unfreeze01 import encode_raw, encoder_unfreeze_step


class ToyEncoder(nn.Module):
    """Stand-in for `RelationEncoder`: small trainable MLP over a flattened
    raw candidate/query window. Sufficient to test `encoder_unfreeze_step`'s
    memory-safety/gradient-correctness contract without needing the full
    3000-line Stage-1 model -- exactly the same "toy" testing convention
    `tests/test_exp_strong_scorer_streaming.py` used for the scorer head."""

    def __init__(self, seq_len, n_ch, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len * n_ch, 2 * d_model), nn.GELU(), nn.Linear(2 * d_model, d_model))

    def forward(self, x):
        return self.net(x.reshape(x.size(0), -1))


class ToyModel:
    def __init__(self, encoder):
        self.encoder = encoder

    def _relation_tensor(self, x, target_channel, source_channel):
        return x  # ToyEncoder consumes the raw window directly


def _toy_setup(seed=0, n=23, seq_len=8, n_ch=3, d_model=16, bsz=5, k_prefix=3):
    torch.manual_seed(seed)
    encoder = ToyEncoder(seq_len, n_ch, d_model)
    model = ToyModel(encoder)
    set_conditioner = SetConditioner(d_model)
    empty_token = EmptySetToken(d_model)
    utility_head = UtilityHead()
    memory_x = torch.randn(n, seq_len, n_ch)
    batch_x_v = torch.randn(bsz, seq_len, n_ch)
    teacher_idx_v = torch.randint(0, n, (bsz, k_prefix))
    u_target = torch.randn(bsz, n)
    valid_now = torch.ones(bsz, n, dtype=torch.bool)
    valid_now[:, -2:] = False
    with torch.no_grad():
        E_nograd = encode_raw(model, memory_x, 0, grad=False)
    return model, encoder, set_conditioner, empty_token, utility_head, memory_x, batch_x_v, teacher_idx_v, u_target, valid_now, E_nograd


def test_A_encoder_gradient_flows_query_and_candidate_side():
    (model, encoder, sc, et, uh, memory_x, batch_x_v, teacher_idx_v, u_target,
     valid_now, E_nograd) = _toy_setup()
    diag = encoder_unfreeze_step(model, 0, memory_x, batch_x_v, sc, et, uh,
                                  teacher_idx_v, E_nograd, u_target, valid_now,
                                  lambda_rank=1.0, grad_scale=1.0, cand_chunk_size=7, t=1, train=True)
    assert diag['smoothl1_loss'] == diag['smoothl1_loss']  # not NaN
    enc_grads = [p.grad for p in encoder.parameters()]
    assert all(g is not None for g in enc_grads), 'encoder must receive a gradient (query+candidate side)'
    total_norm = sum(g.norm().item() ** 2 for g in enc_grads) ** 0.5
    assert total_norm > 0.0, 'encoder gradient norm is exactly 0 -- candidate branch may be silently detached'


def test_B_frozen_encoder_regression_receives_no_grad():
    (model, encoder, sc, et, uh, memory_x, batch_x_v, teacher_idx_v, u_target,
     valid_now, E_nograd) = _toy_setup()
    for p in encoder.parameters():
        p.requires_grad = False
    encoder_unfreeze_step(model, 0, memory_x, batch_x_v, sc, et, uh,
                           teacher_idx_v, E_nograd, u_target, valid_now,
                           lambda_rank=1.0, grad_scale=1.0, cand_chunk_size=7, t=1, train=True)
    assert all(p.grad is None for p in encoder.parameters()), 'frozen encoder must never accumulate a gradient'


def test_C_no_stale_candidate_bank_embeddings_track_current_encoder():
    model, encoder, *_rest = _toy_setup()
    memory_x = _rest[4]
    with torch.no_grad():
        e_before = encode_raw(model, memory_x, 0, grad=False).clone()
        for p in encoder.parameters():
            p.add_(0.1 * torch.randn_like(p))
        e_after = encode_raw(model, memory_x, 0, grad=False)
    assert not torch.allclose(e_before, e_after), (
        'candidate embeddings did not change after perturbing encoder parameters -- '
        'suggests a cached/stale candidate bank instead of live re-encoding')


def test_D_chunk_equivalence_scores_losses_and_all_gradients_match():
    torch.manual_seed(0)
    encoder_a = ToyEncoder(8, 3, 16)
    model_a = ToyModel(encoder_a)
    sc_a = SetConditioner(16)
    et_a = EmptySetToken(16)
    uh_a = UtilityHead()

    encoder_b = copy.deepcopy(encoder_a)
    model_b = ToyModel(encoder_b)
    sc_b = copy.deepcopy(sc_a)
    et_b = copy.deepcopy(et_a)
    uh_b = copy.deepcopy(uh_a)

    torch.manual_seed(1)
    n, seq_len, n_ch, bsz, k_prefix = 23, 8, 3, 5, 3
    memory_x = torch.randn(n, seq_len, n_ch)
    batch_x_v = torch.randn(bsz, seq_len, n_ch)
    teacher_idx_v = torch.randint(0, n, (bsz, k_prefix))
    u_target = torch.randn(bsz, n)
    valid_now = torch.ones(bsz, n, dtype=torch.bool)
    valid_now[:, -2:] = False
    with torch.no_grad():
        E_nograd_a = encode_raw(model_a, memory_x, 0, grad=False)
        E_nograd_b = encode_raw(model_b, memory_x, 0, grad=False)
    assert torch.allclose(E_nograd_a, E_nograd_b)

    diag_a = encoder_unfreeze_step(model_a, 0, memory_x, batch_x_v, sc_a, et_a, uh_a,
                                    teacher_idx_v, E_nograd_a, u_target, valid_now,
                                    lambda_rank=1.0, grad_scale=1.0, cand_chunk_size=1000, t=1, train=True)
    diag_b = encoder_unfreeze_step(model_b, 0, memory_x, batch_x_v, sc_b, et_b, uh_b,
                                    teacher_idx_v, E_nograd_b, u_target, valid_now,
                                    lambda_rank=1.0, grad_scale=1.0, cand_chunk_size=5, t=1, train=True)

    assert abs(diag_a['smoothl1_loss'] - diag_b['smoothl1_loss']) < 1e-4
    if diag_a['pairwise_loss'] == diag_a['pairwise_loss']:
        assert abs(diag_a['pairwise_loss'] - diag_b['pairwise_loss']) < 1e-4

    for (na, pa), (nb, pb) in zip(encoder_a.named_parameters(), encoder_b.named_parameters()):
        assert na == nb
        ga = pa.grad if pa.grad is not None else torch.zeros_like(pa)
        gb = pb.grad if pb.grad is not None else torch.zeros_like(pb)
        assert torch.allclose(ga, gb, atol=1e-3, rtol=1e-2), f'encoder.{na} gradient mismatch (chunked vs unchunked)'
    for (na, pa), (nb, pb) in zip(sc_a.named_parameters(), sc_b.named_parameters()):
        ga = pa.grad if pa.grad is not None else torch.zeros_like(pa)
        gb = pb.grad if pb.grad is not None else torch.zeros_like(pb)
        assert torch.allclose(ga, gb, atol=1e-3, rtol=1e-2), f'SetConditioner.{na} gradient mismatch'
    for (na, pa), (nb, pb) in zip(uh_a.named_parameters(), uh_b.named_parameters()):
        ga = pa.grad if pa.grad is not None else torch.zeros_like(pa)
        gb = pb.grad if pb.grad is not None else torch.zeros_like(pb)
        assert torch.allclose(ga, gb, atol=1e-3, rtol=1e-2), f'UtilityHead.{na} gradient mismatch'
