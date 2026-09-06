#!/usr/bin/env python3
"""Post-hoc effective-rank diagnostic for EXP-SEQDIAG01 Arm C (Weather
Trainable) trained encoder, mirroring the ad-hoc computation used for
EXP-SEQFULL01's ETTh1 trained encoder (results/EXP-SEQFULL01/notes.md).
Channel 0, full candidate bank, embedding_geometry() from utils/rank_losses.py.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path('/data/pjh_workspace/CARTS')
sys.path.insert(0, str(REPO_ROOT))

from exp.exp_stage1_relation import Exp_Stage1_Relation
from utils.rank_losses import embedding_geometry

CKPT = 'checkpoints/exp_seqdiag01/seqfull01/custom/seq96_pred96/carts_seqdiag01_weather_trainable/checkpoint.pth'
WEATHER_S1_CKPT = ('checkpoints/soft_set_mse/stage1/custom/seq96_pred96/'
    'stage1_carts_softset_Weather_96_S0_wce_RelationStage1_custom_ftM_sl96_ll0_pl96_dm128_nh4_el2_dl1_df256'
    '_expand2_dc4_fc1_ebtimeF_dtTrue_softset_S0_wce_Weather_sl96_pl96_0/checkpoint.pth')

import argparse
from types import SimpleNamespace

base = torch.load(WEATHER_S1_CKPT, map_location='cpu')
args_dict = dict(base['args'])
args = SimpleNamespace(**args_dict)
exp = Exp_Stage1_Relation(args)
model = exp.model

ckpt = torch.load(CKPT, map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
model.eval()

device = exp.device
model.to(device)

# force needs_history=True so exp.memory_x (full multi-channel tensor) gets built
args.stage1_full_memory_gradient_mode = 'full'
exp._ensure_memory()

with torch.no_grad():
    z = model.encoder(model._relation_tensor(exp.memory_x, 0, 0))
    z = F.normalize(z, dim=-1)

geom = embedding_geometry(z)
print(f"n_candidates={z.size(0)} d_model={z.size(-1)}")
for k, v in geom.items():
    print(f"{k} = {float(v):.6f}")
