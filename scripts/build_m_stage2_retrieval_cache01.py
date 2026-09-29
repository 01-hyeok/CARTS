#!/usr/bin/env python3
"""TRACK-M-RELEVANCE-CONSTRAINED-MULTISLOT01 Stage2 -- retrieval cache
builder for S0 (base/no-retrieval), S1 (J1), S2 (K2), S3 (M*, here M2).

Sibling of `scripts/build_setlossctrl_retrieval_cache02.py` (identical
method: same reuse of `encode_raw`/`arm_score`/`HostScorer`/`memory_value`
from `scripts/train_factorial_e2e01.py`/`scripts/train_margutil01.py`,
same fixed-stage2-host alpha-weighting, same delta-space cache
correctness convention, same `cache_schema_version='corrected_delta_v1'`
schema so `scripts/train_setlossctrl_stage2_retrain02.py` can train on it
UNMODIFIED). Differs only in WHICH candidates are retrieved per arm
(PART 17: "only the retrieval source differs between arms" -- alpha
weighting is the SAME fixed HostScorer for every arm):

  S0 (base/no-retrieval): relation_outputs/relation_query_embs are all
    ZERO -- an uninformative retrieval branch, so the jointly-trained
    Stage2 model has no real signal to gate onto. This is its own
    dedicated arm (PART 15), not a reuse of another arm's
    counterfactual_lambda0_mse (which reflects a base head partly shaped
    by that OTHER arm's own real retrieval gradient).
  S1 (J1): Top-10 via J1's own single-score-vector selection
    (`arm_score` + `stable_topk_indices`, future-blind).
  S2 (K2): Top-10 via K2's own multi-slot hard-unique selection
    (`compute_scores` + `hard_unique_selection`, future-blind).
  S3 (M*=M2): identical mechanism to S2, M2's own checkpoint.

Retrievers are FULLY FROZEN here (PART 16): loaded `.eval()`,
`requires_grad_(False)`, used only to produce this one-time cache; no
retriever gradient exists anywhere in Stage2 (the online retriever is
never even instantiated by the Stage2 trainer, which only reads this
cache). Only the pre-existing retrieved-futures format is fed forward
(PART 17): no slot embeddings, teacher probabilities, future labels, or
oracle information enter the cache.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.train_factorial_e2e01 import HostScorer, arm_score, encode_raw
from scripts.train_j_shared_encoder_drift01 import build_model, state_hash
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads, compute_scores, hard_unique_selection
from scripts.train_margutil01 import memory_value

ARMS = ('S0_base', 'S1_J1', 'S2_K2', 'S3_Mstar')
EXPECTED_INIT_HASH = 'b37fa4031f538e4b5f5c522ae22e7d03613e4ee66283d2637656c7c4f872372e'

CKPTS = {
    'S1_J1': REPO_ROOT / 'checkpoints/track_j2_key_update_decomposition01/ETTh1_720/J1_stopgrad_key/checkpoint.pth',
    'S2_K2': REPO_ROOT / 'checkpoints/track_k_multislot_predictive_retrieval01/ETTh1_720/K2_multislot_aggregate/checkpoint.pth',
    'S3_Mstar': REPO_ROOT / 'checkpoints/track_m_relevance_constrained_multislot01/ETTh1_720/M2_relevance_budget/checkpoint.pth',
}
IS_MULTISLOT = {'S1_J1': False, 'S2_K2': True, 'S3_Mstar': True}


def _file_sha256(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


@torch.no_grad()
def build_cache_for_split(arm, base_model, d_model, exp, args, stage2_host_scorer, pred_len, split, top_k, device):
    channels = list(range(int(args.enc_in)))
    slot_heads = None
    if arm != 'S0_base':
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
    all_topk_idx, all_alpha = [], []

    for batch_x, _batch_y_unused, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        cand_mask, counts = exp._candidate_mask(batch_start_idx)
        bsz = batch_x.size(0)

        rel_out_ch = torch.zeros(bsz, len(channels), 1, pred_len)
        query_emb_ch = torch.zeros(bsz, len(channels), 1, d_model)
        query_offset_ch = torch.zeros(bsz, len(channels))
        topk_idx_ch = torch.zeros(bsz, len(channels), top_k, dtype=torch.long)
        alpha_ch = torch.zeros(bsz, len(channels), top_k)

        for c in channels:
            offset_c = batch_x[:, -1, c].detach()
            query_offset_ch[:, c] = offset_c.cpu()
            if arm == 'S0_base':
                continue  # relation_outputs/query_embs stay zero -- no retrieval signal at all

            z_q = encode_raw(base_model, batch_x, c)
            query_emb_ch[:, c, 0, :] = z_q.detach().cpu()
            memory_c, offset_c2 = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            memory_c_b = memory_c.unsqueeze(0).expand(bsz, -1, -1)
            host_scores = stage2_host_scorer.scores(batch_x, c, cand_mask)

            if IS_MULTISLOT[arm]:
                scores = compute_scores(base_model, slot_heads, batch_x, exp.memory_x, c)  # [B,S,N]
                picks_t = hard_unique_selection(scores, cand_mask, s=top_k)  # [B, top_k], future-blind
            else:
                z_k = encode_raw(base_model, exp.memory_x, c)
                s = arm_score(z_q, z_k, None).masked_fill(~cand_mask, float('-inf'))
                picks_t = stable_topk_indices(s, top_k, largest=True)  # future-blind

            sc_sel = host_scores.gather(1, picks_t)
            alpha = torch.softmax(sc_sel / float(stage2_host_scorer.tau_topk), dim=-1)
            tgt_delta = memory_c_b.gather(1, picks_t.unsqueeze(-1).expand(-1, -1, memory_c_b.size(-1)))
            delta_ret = (alpha.unsqueeze(-1) * tgt_delta).sum(1)  # [B, pred_len], DELTA space

            rel_out_ch[:, c, 0, :] = delta_ret.cpu()
            topk_idx_ch[:, c, :] = picks_t.cpu()
            alpha_ch[:, c, :] = alpha.cpu()

        all_start_idx.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx)
                             else torch.as_tensor(batch_start_idx))
        all_relation_outputs.append(rel_out_ch)
        all_query_embs.append(query_emb_ch)
        all_query_offset.append(query_offset_ch)
        all_topk_idx.append(topk_idx_ch)
        all_alpha.append(alpha_ch)

    return {
        'batch_start_idx': torch.cat(all_start_idx),
        'relation_outputs': torch.cat(all_relation_outputs),  # DELTA space, never offset-added
        'relation_query_embs': torch.cat(all_query_embs),
        'query_offset': torch.cat(all_query_offset),
        'topk_idx': torch.cat(all_topk_idx),
        'alpha': torch.cat(all_alpha),
        'channels': channels, 'top_k': top_k, 'pred_len': pred_len,
        'arm_name': arm, 'split': split,
        'value_space': 'delta', 'offset_included': False,
        'cache_schema_version': 'corrected_delta_v1',
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=ARMS)
    ap.add_argument('--cell', default='ETTh1_720')
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--stage2_host', required=True)
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
    ap.add_argument('--out_dir', default='results/TRACK-M-RELEVANCE-CONSTRAINED-MULTISLOT01/stage2')
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp, args, base_model = build_model(cli, device)
    assert state_hash(base_model) == EXPECTED_INIT_HASH
    d_model = int(args.d_model)
    host = HostScorer(cli.stage2_host, device)

    out_dir = Path(cli.out_dir) / 'cache' / cli.cell / cli.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    stage1_ckpt_hash = _file_sha256(CKPTS[cli.arm]) if cli.arm != 'S0_base' else None
    for split in ('train', 'val', 'test'):
        cache = build_cache_for_split(cli.arm, base_model, d_model, exp, args, host, cli.pred_len, split,
                                      cli.top_k, device)
        cache.update({'stage1_checkpoint_path': str(CKPTS.get(cli.arm, '')), 'stage1_checkpoint_hash': stage1_ckpt_hash,
                     'config_hash': hashlib.sha256(json.dumps(
                         {'arm': cli.arm, 'cell': cli.cell, 'top_k': cli.top_k}, sort_keys=True).encode()).hexdigest()})
        out_path = out_dir / f'{split}.pt'
        torch.save(cache, out_path)
        print(f'[build_m_cache] {cli.cell}/{cli.arm}/{split}: n_queries={cache["batch_start_idx"].numel()} -> {out_path}')

    manifest = {'value_space': 'delta', 'offset_included': False, 'cache_schema_version': 'corrected_delta_v1',
               'stage1_checkpoint_path': str(CKPTS.get(cli.arm, '')), 'stage1_checkpoint_hash': stage1_ckpt_hash,
               'cell': cli.cell, 'arm_name': cli.arm}
    (out_dir / 'cache_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'[build_m_cache] wrote {out_dir / "cache_manifest.json"}')


if __name__ == '__main__':
    main()
