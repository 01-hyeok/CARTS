#!/usr/bin/env python3
"""TRACK-N-FORECAST-CONDITIONAL-UTILITY01 PART 4 -- common frozen base
predictions B_q.

NO TRAINING. Loads TRACK-M's S0_base Stage2 checkpoint (the base-only,
no-retrieval arm), freezes it completely, and computes `B_q =
base_head(batch_x)` directly -- `BaseForecastHead.forward` is
independent of the retrieval branch (see `models/RelationStage2.py`
lines 33-56), so this is exactly the same B_q every downstream retriever
(J1/K2/M2) will condition on. This removes the base-head-co-training
confound TRACK-M's own Stage2 had (there, S0/S1/S2/S3 each trained their
OWN base head jointly with their own retrieval branch).
"""
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_factorial_e2e01 import state_sha
from scripts.train_setlossctrl_stage2_retrain02 import build_fresh_stage2

S2_720 = ('checkpoints/stage2/ETTh1/seq720_pred720/stage2_carts_softset_s2_ETTh1_720_S0_wce_RelationStage2_ETTh1_'
         'ftM_sl720_ll0_pl720_dm128_nh4_el2_dl1_df256_expand2_dc4_fc1_ebtimeF_dtTrue_softset_s2_S0_wce_ETTh1_'
         'sl720_pl720_0/checkpoint.pth')
S0_CKPT = REPO_ROOT / 'checkpoints/track_m_relevance_constrained_multislot01/stage2/ETTh1_720/S0_base/checkpoint.pth'
OUT_DIR = REPO_ROOT / 'results/TRACK-N-FORECAST-CONDITIONAL-UTILITY01/ETTh1_720/common_base_predictions'


@torch.no_grad()
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp, args, model, host_ck = build_fresh_stage2(S2_720, seed=0)
    model.to(device)
    exp._ensure_memory()

    bl = torch.load(S0_CKPT, map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    base_sha = state_sha(model.base_head.state_dict())

    audit = {'s0_checkpoint': str(S0_CKPT.relative_to(REPO_ROOT)), 's0_checkpoint_sha256': None,
             'base_head_state_sha': base_sha, 's0_best_epoch': bl.get('epoch')}
    import hashlib
    h = hashlib.sha256()
    h.update(S0_CKPT.read_bytes())
    audit['s0_checkpoint_sha256'] = h.hexdigest()

    for split in ('train', 'val', 'test'):
        _, loader = exp._get_data(flag=split, shuffle=False)
        starts, preds = [], []
        for batch_x, batch_y, batch_start_idx in loader:
            batch_x = batch_x.float().to(device)
            output_offset = batch_x[:, -1:, :].detach()  # matches Model.forward()'s own `+ output_offset` restore
            b_q = model.base_head(batch_x) + output_offset  # [B, pred_len, channels], independent of retrieval branch
            starts.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx)
                         else torch.as_tensor(batch_start_idx))
            preds.append(b_q.cpu())
        out = {'batch_start_idx': torch.cat(starts), 'base_predictions': torch.cat(preds),
              'split': split, 'source_checkpoint': str(S0_CKPT.relative_to(REPO_ROOT))}
        torch.save(out, OUT_DIR / f'base_predictions_{split}.pt')
        print(f'[precompute_n_base] {split}: n={out["batch_start_idx"].numel()} '
             f'shape={tuple(out["base_predictions"].shape)} -> base_predictions_{split}.pt')

    (OUT_DIR / 'audit.json').write_text(json.dumps(audit, indent=2))
    print(f'[precompute_n_base] wrote {OUT_DIR / "audit.json"}')


if __name__ == '__main__':
    main()
