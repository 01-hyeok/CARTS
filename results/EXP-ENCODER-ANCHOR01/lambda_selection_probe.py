#!/usr/bin/env python3
"""Small fixed-batch diagnostic (NOT full training, NOT a lambda sweep) to
choose EXP-ENCODER-ANCHOR01's single fixed lambda_anchor: compare R2
gradient norm vs. (unweighted, lambda=1) anchor gradient norm on the
EXISTING EXP-ENCODER-UNFREEZE01 epoch-1 checkpoint's already-drifted
encoder (a more representative "mid-collapse" state than the pristine B0
init, where anchor grad would trivially be ~0). Uses ONE real training
batch, ONE channel, ONE K-step (t=1), never touches test data."""
import copy
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path('/data/pjh_workspace/CARTS')
sys.path.insert(0, str(REPO_ROOT))

from scripts.train_encoder_unfreeze01 import (build_experiment, encode_raw,
    encoder_unfreeze_step, _encode_candidates_nograd)
from scripts.train_encoder_anchor01 import anchor_step, encode_with
from models.DenseUtilityRetriever import UtilityHead
from models.SequentialSetRetriever import EmptySetToken, SetConditioner
from scripts.train_margutil01 import memory_value
from utils.dense_utility import candidate_weights, dense_utility

BASE = 'checkpoints/soft_set_mse/stage1/ETTh1/seq96_pred96/stage1_carts_softset_ETTh1_96_S0_wce_RelationStage1_ETTh1_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_ETTh1_sl96_pl96_0/checkpoint.pth'
E1_EPOCH1 = 'checkpoints/exp_encoder_unfreeze01/encoder_unfreeze01/ETTh1/seq96_pred96/carts_e1_ETTh1_H96/checkpoint_epoch1.pth'
TEACHER = 'cache/seqfull01_teacher/ETTh1_pred96.pt'

overrides = {'is_training': 1, 'model_id': 'lambda_probe', 'des': 'lambda_probe',
             'checkpoints': '/tmp/lambda_probe_ckpt', 'seed': 0, 'top_k': 10,
             'stage1_residual_teacher': 0, 'stage1_query_base_conditioning': 0,
             'stage1_candidate_residual_conditioning': 0, 'stage1_retrieval_metric': 'cosine',
             'stage1_full_memory_gradient_mode': 'full_online'}
exp, args = build_experiment(BASE, overrides)
device = exp.device
model = exp.model.module if hasattr(exp.model, 'module') else exp.model

# Reference encoder = the ORIGINAL B0 (pristine) weights.
b0 = torch.load(BASE, map_location='cpu')
model.load_state_dict(b0['model_state_dict'], strict=True)
encoder_ref = copy.deepcopy(model.encoder).to(device)
encoder_ref.eval()
for p in encoder_ref.parameters():
    p.requires_grad = False

# Trainable encoder = E1's OWN epoch-1 checkpoint (already-drifted state) --
# a more representative point on the collapse trajectory than pristine B0,
# where the anchor gradient would trivially still be ~0.
e1_ckpt = torch.load(E1_EPOCH1, map_location='cpu')
model.load_state_dict(e1_ckpt['model_state_dict'], strict=True)
for p in model.encoder.parameters():
    p.requires_grad = True
exp._ensure_memory()
channels = list(model.target_channels())
c = channels[0]
teacher = torch.load(TEACHER, map_location='cpu')
k = 10
tau = float(args.tau_topk)
d_model = int(args.d_model)

set_conditioner = SetConditioner(d_model).to(device)
set_conditioner.load_state_dict(e1_ckpt['set_conditioner_state_dict'])
empty_token = EmptySetToken(d_model).to(device)
empty_token.load_state_dict(e1_ckpt['empty_token_state_dict'])
utility_head = UtilityHead().to(device)
utility_head.load_state_dict(e1_ckpt['utility_head_state_dict'])

_, train_loader = exp._get_data(flag='train', shuffle=True)
batch_x, batch_y, batch_start_idx = next(iter(train_loader))
batch_x = batch_x.float().to(device)
batch_y = batch_y.float().to(device)
cand_mask, counts = exp._candidate_mask(batch_start_idx)

rows = [teacher['splits']['train']['start_to_row'][int(s)] for s in batch_start_idx.tolist()]
teacher_idx = teacher['splits']['train']['teacher_idx'][c][rows].to(device)
query_valid = teacher_idx[:, 0] != -1
batch_x_v = batch_x[query_valid]
teacher_idx_v = teacher_idx[query_valid]
cand_mask_v = cand_mask[query_valid]

cand_chunk_size = 2048
E_nograd = _encode_candidates_nograd(model, exp.memory_x, c, cand_chunk_size)
q_nograd = encode_raw(model, batch_x, c, grad=False)
memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
futures = memory_c + offset_c.view(-1, 1, 1)
query_future = batch_y[:, :, c]
w_fixed = candidate_weights(torch.matmul(q_nograd, E_nograd.transpose(0, 1)), cand_mask, tau)
w_fixed_v = w_fixed[query_valid]
futures_v = futures[query_valid]
query_future_v = query_future[query_valid]

for p in model.encoder.parameters():
    if p.grad is not None:
        p.grad = None

t = 1
selected_mask_v = torch.zeros_like(cand_mask_v)
nxt0 = teacher_idx_v[:, 0:1].clamp_min(0)
selected_mask_v = selected_mask_v.scatter(1, nxt0, True)
with torch.no_grad():
    prefix = teacher_idx_v[:, :t].clamp_min(0)
    a_dense = dense_utility(prefix, w_fixed_v, futures_v, query_future_v, chunk_size=4096)
    u_target = -a_dense
    valid_now = cand_mask_v & ~selected_mask_v

encoder_unfreeze_step(model, c, exp.memory_x, batch_x_v, set_conditioner, empty_token,
                       utility_head, teacher_idx_v, E_nograd, u_target, valid_now,
                       lambda_rank=1.0, grad_scale=1.0, cand_chunk_size=cand_chunk_size, t=t, train=True)
r2_grad_norm = sum(p.grad.norm().item() ** 2 for p in model.encoder.parameters() if p.grad is not None) ** 0.5
print(f'R2-only encoder grad norm (1 step, 1 channel, batch={batch_x_v.size(0)}): {r2_grad_norm:.6f}')

for p in model.encoder.parameters():
    if p.grad is not None:
        p.grad = None
adiag = anchor_step(model, encoder_ref, c, exp.memory_x, batch_x_v, lambda_anchor=1.0,
                     grad_scale=1.0, cand_chunk_size=cand_chunk_size, train=True)
anchor_grad_norm_lambda1 = sum(p.grad.norm().item() ** 2 for p in model.encoder.parameters() if p.grad is not None) ** 0.5
print(f'Anchor(lambda=1)-only encoder grad norm: {anchor_grad_norm_lambda1:.6f}')
print(f'anchor_loss={adiag["anchor_loss"]:.6f} (query={adiag["anchor_loss_query"]:.6f}, '
      f'candidates={adiag["anchor_loss_candidates"]:.6f})')
print(f'ratio r2/anchor(lambda=1) = {r2_grad_norm/max(anchor_grad_norm_lambda1,1e-12):.4f}')
print(f'=> to make anchor grad norm ~= R2 grad norm, lambda_anchor ~= {r2_grad_norm/max(anchor_grad_norm_lambda1,1e-12):.4f}')
