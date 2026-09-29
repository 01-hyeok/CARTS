#!/usr/bin/env python3
"""TRACK-N-FORECAST-CONDITIONAL-UTILITY01 PART 18 -- Uniform-aggregation
retrieval cache for the conditional Gate-Only experiment (G0=J1, G1=K2,
G2=M2). Sibling of `scripts/build_m_stage2_retrieval_cache01.py`,
differing ONLY in the aggregation weighting: `alpha_i = 1/K` (Uniform,
matching Stage1's own AggMSE definition and PART 6's `R^U`) instead of
the HostScorer-softmax weighting used for TRACK-M's own Stage2 cache.
Same `cache_schema_version='corrected_delta_v1'` schema, so
`train_setlossctrl_stage2_retrain02.py`-style consumption (used here by
`train_n_gate_only01.py`) needs no changes.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.train_factorial_e2e01 import arm_score, encode_raw
from scripts.train_j_shared_encoder_drift01 import build_model, state_hash
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads, compute_scores, hard_unique_selection
from scripts.train_margutil01 import memory_value

ARMS = ('G0_J1', 'G1_K2', 'G2_M2')
EXPECTED_INIT_HASH = 'b37fa4031f538e4b5f5c522ae22e7d03613e4ee66283d2637656c7c4f872372e'
CKPTS = {
    'G0_J1': REPO_ROOT / 'checkpoints/track_j2_key_update_decomposition01/ETTh1_720/J1_stopgrad_key/checkpoint.pth',
    'G1_K2': REPO_ROOT / 'checkpoints/track_k_multislot_predictive_retrieval01/ETTh1_720/K2_multislot_aggregate/checkpoint.pth',
    'G2_M2': REPO_ROOT / 'checkpoints/track_m_relevance_constrained_multislot01/ETTh1_720/M2_relevance_budget/checkpoint.pth',
}
IS_MULTISLOT = {'G0_J1': False, 'G1_K2': True, 'G2_M2': True}
TOP_K = 10


def _file_sha256(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


@torch.no_grad()
def build_cache_for_split(arm, base_model, d_model, exp, args, pred_len, split, device):
    channels = list(range(int(args.enc_in)))
    slot_heads = None
    bl = torch.load(CKPTS[arm], map_location=device)
    base_model.load_state_dict(bl['model_state_dict'])
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)
    if IS_MULTISLOT[arm]:
        slot_heads = SlotHeads(d_model).to(device)
        slot_heads.load_state_dict(bl['slot_heads_state_dict'])
        slot_heads.eval()

    _, loader = exp._get_data(flag=split, shuffle=False)
    all_start_idx, all_relation_outputs, all_query_embs, all_query_offset = [], [], [], []

    for batch_x, _batch_y_unused, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        cand_mask, counts = exp._candidate_mask(batch_start_idx)
        bsz = batch_x.size(0)

        rel_out_ch = torch.zeros(bsz, len(channels), 1, pred_len)
        query_emb_ch = torch.zeros(bsz, len(channels), 1, d_model)
        query_offset_ch = torch.zeros(bsz, len(channels))

        for c in channels:
            offset_c = batch_x[:, -1, c].detach()
            query_offset_ch[:, c] = offset_c.cpu()
            z_q = encode_raw(base_model, batch_x, c)
            query_emb_ch[:, c, 0, :] = z_q.detach().cpu()
            memory_c, offset_c2 = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            memory_c_b = memory_c.unsqueeze(0).expand(bsz, -1, -1)

            if IS_MULTISLOT[arm]:
                scores = compute_scores(base_model, slot_heads, batch_x, exp.memory_x, c)
                picks_t = hard_unique_selection(scores, cand_mask, s=TOP_K)
            else:
                z_k = encode_raw(base_model, exp.memory_x, c)
                s = arm_score(z_q, z_k, None).masked_fill(~cand_mask, float('-inf'))
                picks_t = stable_topk_indices(s, TOP_K, largest=True)

            tgt_delta = memory_c_b.gather(1, picks_t.unsqueeze(-1).expand(-1, -1, memory_c_b.size(-1)))
            delta_ret = tgt_delta.mean(dim=1)  # UNIFORM aggregation: alpha_i = 1/K

            rel_out_ch[:, c, 0, :] = delta_ret.cpu()

        all_start_idx.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx)
                             else torch.as_tensor(batch_start_idx))
        all_relation_outputs.append(rel_out_ch)
        all_query_embs.append(query_emb_ch)
        all_query_offset.append(query_offset_ch)

    return {
        'batch_start_idx': torch.cat(all_start_idx),
        'relation_outputs': torch.cat(all_relation_outputs),
        'relation_query_embs': torch.cat(all_query_embs),
        'query_offset': torch.cat(all_query_offset),
        'channels': channels, 'top_k': TOP_K, 'pred_len': pred_len,
        'arm_name': arm, 'split': split,
        'value_space': 'delta', 'offset_included': False,
        'cache_schema_version': 'corrected_delta_v1',
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=ARMS)
    ap.add_argument('--cell', default='ETTh1_720')
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--pred_len', type=int, default=720)
    ap.add_argument('--seq_len', type=int, default=720)
    ap.add_argument('--patch_len', type=int, default=16)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--tau_t', type=float, default=0.1)
    ap.add_argument('--tau_s', type=float, default=0.1)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--init_seed', type=int, default=0)
    ap.add_argument('--loader_seed', type=int, default=0)
    ap.add_argument('--out_dir', default='results/TRACK-N-FORECAST-CONDITIONAL-UTILITY01/gate_only')
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp, args, base_model = build_model(cli, device)
    assert state_hash(base_model) == EXPECTED_INIT_HASH
    d_model = int(args.d_model)

    out_dir = Path(cli.out_dir) / 'cache' / cli.cell / cli.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    stage1_ckpt_hash = _file_sha256(CKPTS[cli.arm])
    for split in ('train', 'val', 'test'):
        cache = build_cache_for_split(cli.arm, base_model, d_model, exp, args, cli.pred_len, split, device)
        cache.update({'stage1_checkpoint_path': str(CKPTS[cli.arm]), 'stage1_checkpoint_hash': stage1_ckpt_hash,
                     'config_hash': hashlib.sha256(json.dumps(
                         {'arm': cli.arm, 'cell': cli.cell, 'top_k': cli.top_k, 'aggregation': 'uniform'},
                         sort_keys=True).encode()).hexdigest()})
        out_path = out_dir / f'{split}.pt'
        torch.save(cache, out_path)
        print(f'[build_n_gate_cache] {cli.cell}/{cli.arm}/{split}: n={cache["batch_start_idx"].numel()} -> {out_path}')

    manifest = {'value_space': 'delta', 'offset_included': False, 'cache_schema_version': 'corrected_delta_v1',
               'aggregation': 'uniform', 'stage1_checkpoint_path': str(CKPTS[cli.arm]),
               'stage1_checkpoint_hash': stage1_ckpt_hash, 'cell': cli.cell, 'arm_name': cli.arm}
    (out_dir / 'cache_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'[build_n_gate_cache] wrote {out_dir / "cache_manifest.json"}')


if __name__ == '__main__':
    main()
