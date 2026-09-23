#!/usr/bin/env python3
"""TRACK-A-HORIZON-RETRIEVAL-EXPERT01 -- Stage-2 delta-space cache builder.

Sibling of `scripts/build_tf_oracle_learnability01_retrieval_cache.py`
(same reuse: `encode_raw`/`arm_score`/`HostScorer` from
`scripts.train_factorial_e2e01`, same delta-space-only caching -- the
query's own last-observed value (`query_offset`) is NEVER added into the
cached `relation_outputs`, only carried alongside for validation-time
restoration by the reused Stage-2 trainer). Differs because
TRACK-A-HORIZON-RETRIEVAL-EXPERT01's retrievers are SINGLE-SHOT (no
sequential SetConditioner state) and, for the block arm, retrieve a
DIFFERENT Top-10 set per future block:

  Global arm: one Top-10 set (by `s_G`), host-alpha-weighted delta applied
    uniformly across the full [0:720] horizon -- structurally identical to
    every other Track-A Stage-2 cache (one retrieval, one 720-length delta).

  Block arm: three INDEPENDENT Top-10 sets (one per `s_b`, via the trained
    zero-init `BlockCorrectionHeads`), each host-alpha-weighted over ITS OWN
    picks (renormalized softmax on just that block's 10 host scores, not a
    720-wide one), and each written into ONLY its own [lo:hi] slice of a
    single 720-length delta vector -- block1 into [0:96], block2 into
    [96:336], block3 into [336:720]. The three slices exactly tile [0:720]
    (no gap, no overlap), so the result is one ordinary 720-length delta per
    channel -- the reused Stage-2 trainer (`train_setlossctrl_stage2_
    retrain02.py`) never needs to know a block split happened; it only ever
    reads `relation_outputs[:, :, 0, :]` as ONE cached delta signal, exactly
    as for every other arm this repo has cached. `query_offset` is added
    back exactly ONCE at restoration time (the trainer's own
    `restore_absolute`), never per block -- so there is no double-adding of
    the offset across the three slices.

Since TRACK-A-HORIZON-RETRIEVAL-EXPERT01 has no SetConditioner (single-shot
scoring only), the cache's `relation_query_embs` stores the plain Global
`z_q` (shared encoder's own query embedding) for BOTH arms -- this is a
contextual feature fed to Stage-2's own fusion gate
(`models/RelationStage2.py`), not used to re-score candidates, so using the
same shared-encoder embedding for both arms keeps Stage-2's gate input
comparable across arms (spec requirement: common base, fair comparison).

Self-consistency gate (this script's own, since the reused Stage-2
trainer's built-in FR-Agg cross-check looks for a key
[`best_val_free_running_aggregate_future_mse`] this experiment's
`retrieval_metrics_*.json` does not have -- block/global here use hard
Top-K + host-alpha weighting, not a free-running rollout, so that key does
not apply): after building the cache, restores absolute predictions
(`delta + query_offset`) and recomputes their MSE against the same split's
query_future, comparing against a SEPARATELY, freshly recomputed reference
(same math, done again independently in this script) -- must match to
float precision or the script aborts. This catches indexing/slicing bugs
(the exact class of bug this session already hit twice in adjacent scripts:
the Oracle-contrast `n_rows` bug and the orchestrator's dropped
`--channelwise_backward` flag) before any Stage-2 training starts on a
possibly-wrong cache.
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
from scripts.diag_horizon_retrieval_headroom01 import BLOCKS, _mse
from scripts.train_factorial_e2e01 import HostScorer, arm_score, encode_raw
from scripts.train_horizon_retrieval_expert01 import BLOCK_NAMES, BlockCorrectionHeads
from scripts.train_margutil01 import build_experiment, memory_value

ARMS = ('global', 'block')


def _file_sha256(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def validate_arm_checkpoint(arm_ckpt, retrieval_metrics_json, cell, arm_name, top_k):
    ckpt = torch.load(arm_ckpt, map_location='cpu')
    fp = ckpt.get('fingerprint', {})
    rm = json.loads(Path(retrieval_metrics_json).read_text())
    problems = []
    if fp.get('arm') != arm_name:
        problems.append(f"arm={fp.get('arm')!r} != {arm_name!r}")
    if int(fp.get('top_k', -1)) != int(top_k):
        problems.append(f"top_k={fp.get('top_k')} != {top_k}")
    if int(ckpt.get('epoch', -1)) != int(rm.get('best_epoch', -2)):
        problems.append(f"checkpoint epoch={ckpt.get('epoch')} != "
                        f"retrieval_metrics best_epoch={rm.get('best_epoch')}")
    for k, v in ckpt['model_state_dict'].items():
        if torch.is_tensor(v) and (torch.isnan(v).any() or torch.isinf(v).any()):
            problems.append(f'NaN/Inf in model_state_dict[{k}]')
            break
    if problems:
        raise SystemExit(f'[ISSUE][ABORT] checkpoint validation failed for {cell}/{arm_name}: '
                         + '; '.join(problems))
    return ckpt, fp


@torch.no_grad()
def build_cache_for_split(arm_ckpt_path, arm_name, stage2_host, reference_ckpt, pred_len, split,
                          top_k, chunk_size, device):
    ckpt = torch.load(arm_ckpt_path, map_location='cpu')
    args_dict = ckpt['args']
    exp, args = build_experiment(reference_ckpt, {
        'pred_len': pred_len, 'seq_len': pred_len, 'batch_size': 32,
        'seed': 0, 'top_k': top_k,
    })
    exp._ensure_memory()
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    d_model = int(args.d_model)
    heads = None
    if arm_name == 'block':
        heads = BlockCorrectionHeads(d_model).to(device)
        heads.load_state_dict(ckpt['heads_state_dict'])
        heads.eval()
        for p in heads.parameters():
            p.requires_grad_(False)

    host = HostScorer(stage2_host, device)
    channels = list(range(int(args.enc_in)))
    _, loader = exp._get_data(flag=split, shuffle=False)

    all_start_idx, all_relation_outputs, all_query_embs = [], [], []
    all_query_offset, all_valid_count, all_invalid = [], [], []
    verify_pred_sum, verify_pred_n = 0.0, 0

    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, counts = exp._candidate_mask(batch_start_idx)
        bsz = batch_x.size(0)

        rel_out_ch = torch.zeros(bsz, len(channels), 1, pred_len)
        query_emb_ch = torch.zeros(bsz, len(channels), 1, d_model)
        query_offset_ch = torch.zeros(bsz, len(channels))

        for c in channels:
            E = encode_raw(model, exp.memory_x, c)
            z_q = encode_raw(model, batch_x, c)
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            host_scores = host.scores(batch_x, c, cand_mask)
            query_offset_ch[:, c] = offset_c.detach().cpu()
            query_emb_ch[:, c, 0, :] = z_q.detach().cpu()

            if arm_name == 'global':
                s_g = arm_score(z_q, E, None).masked_fill(~cand_mask, float('-inf'))
                picks = stable_topk_indices(s_g, top_k, largest=True)
                sc_sel = host_scores.gather(1, picks)
                alpha = torch.softmax(sc_sel / float(host.tau_topk), dim=-1)
                tgt = memory_c[picks]  # [B, K, 720]
                delta = (alpha.unsqueeze(-1) * tgt).sum(1)  # [B, 720]
                rel_out_ch[:, c, 0, :] = delta.cpu()
            else:
                for bidx, name in enumerate(BLOCK_NAMES):
                    lo, hi = BLOCKS[name]
                    z_q_b = heads(z_q, bidx)
                    s_b = arm_score(z_q_b, E, None).masked_fill(~cand_mask, float('-inf'))
                    picks_b = stable_topk_indices(s_b, top_k, largest=True)
                    sc_sel_b = host_scores.gather(1, picks_b)
                    alpha_b = torch.softmax(sc_sel_b / float(host.tau_topk), dim=-1)
                    tgt_b = memory_c[picks_b][:, :, lo:hi]  # [B, K, hi-lo]
                    delta_b = (alpha_b.unsqueeze(-1) * tgt_b).sum(1)  # [B, hi-lo]
                    rel_out_ch[:, c, 0, lo:hi] = delta_b.cpu()

        invalid = (counts.cpu().unsqueeze(-1).expand(-1, len(channels)) < top_k).float()

        all_start_idx.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx)
                             else torch.as_tensor(batch_start_idx))
        all_relation_outputs.append(rel_out_ch)
        all_query_embs.append(query_emb_ch)
        all_valid_count.append(counts.cpu())
        all_invalid.append(invalid)
        all_query_offset.append(query_offset_ch)

        # self-consistency: restore absolute, MSE against this batch's own
        # query_future, accumulate for a fresh, independent recomputation
        # below (not reused from anything computed above).
        abs_pred = rel_out_ch[:, :, 0, :].to(device) + query_offset_ch.to(device).unsqueeze(-1)
        qf = batch_y.permute(0, 2, 1)  # [B, C, 720]
        se = ((abs_pred - qf) ** 2).mean(-1).sum()
        verify_pred_sum += float(se)
        verify_pred_n += bsz * len(channels)

    cache = {
        'batch_start_idx': torch.cat(all_start_idx),
        'relation_outputs': torch.cat(all_relation_outputs),
        'relation_query_embs': torch.cat(all_query_embs),
        'query_offset': torch.cat(all_query_offset),
        'valid_count': torch.cat(all_valid_count),
        'invalid_row': torch.cat(all_invalid),
        'channels': channels, 'top_k': top_k, 'pred_len': pred_len,
        'arm_name': arm_name, 'split': split,
        'value_space': 'delta', 'offset_included': False,
        # MUST match REQUIRED_CACHE_SCHEMA in train_setlossctrl_stage2_retrain02.py
        # ('corrected_delta_v1') -- that trainer hard-refuses any other value.
        # This cache follows the identical corrected convention (offset never
        # added into relation_outputs), so reusing the same tag is correct,
        # not a mislabeling.
        'cache_schema_version': 'corrected_delta_v1',
    }
    cache_mse = verify_pred_sum / max(verify_pred_n, 1)
    return cache, cache_mse


def _independent_reverify(cache, split_mse):
    """Second, independently-computed pass over the SAME cache tensors
    (not re-deriving from the model at all -- deliberately a pure-cache
    sanity check that catches a bug introduced between building rel_out_ch
    and stacking `all_relation_outputs`, distinct from the model-level
    check above)."""
    abs_pred = cache['relation_outputs'][:, :, 0, :] + cache['query_offset'].unsqueeze(-1)
    return abs_pred, float(split_mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True)
    ap.add_argument('--arm_name', required=True, choices=ARMS)
    ap.add_argument('--stage1_out_dir', default='results/TRACK-A-HORIZON-RETRIEVAL-EXPERT01')
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--stage2_host', required=True)
    ap.add_argument('--pred_len', type=int, default=720)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--out_dir', default='results/TRACK-A-HORIZON-RETRIEVAL-EXPERT01-STAGE2')
    ap.add_argument('--checkpoints_root', default='checkpoints/track_a_horizon_retrieval_expert01')
    cli = ap.parse_args()

    if cli.pred_len != 720:
        raise SystemExit('[ISSUE][ABORT] this experiment is defined only for pred_len=720')

    rm_json = Path(cli.stage1_out_dir) / cli.cell / f'retrieval_metrics_{cli.arm_name}.json'
    if not rm_json.exists():
        raise SystemExit(f'[ISSUE][ABORT] {rm_json} not found -- Stage-1 arm not complete yet.')
    marker = Path(cli.stage1_out_dir) / cli.cell / f'DONE_{cli.arm_name}.marker'
    if not marker.exists():
        raise SystemExit(f'[ISSUE][ABORT] {marker} not found -- Stage-1 arm has no completion marker.')

    arm_ckpt = f'{cli.checkpoints_root}/{cli.cell}/{cli.arm_name}/checkpoint.pth'
    validate_arm_checkpoint(arm_ckpt, rm_json, cli.cell, cli.arm_name, cli.top_k)
    print(f'[build_cache_horizon_retrieval_expert01] {cli.cell}/{cli.arm_name}: '
         f'checkpoint validated -- {arm_ckpt}')

    stage1_hash = _file_sha256(arm_ckpt)
    config_hash = hashlib.sha256(json.dumps({
        'cell': cli.cell, 'arm_name': cli.arm_name, 'pred_len': cli.pred_len,
        'top_k': cli.top_k, 'stage1_hash': stage1_hash,
    }, sort_keys=True).encode()).hexdigest()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(cli.out_dir) / 'cache' / cli.cell / cli.arm_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ('train', 'val', 'test'):
        cache, cache_mse = build_cache_for_split(
            arm_ckpt, cli.arm_name, cli.stage2_host, cli.reference_ckpt, cli.pred_len, split,
            cli.top_k, cli.chunk_size, device)
        cache['stage1_checkpoint_path'] = arm_ckpt
        cache['stage1_checkpoint_hash'] = stage1_hash
        cache['config_hash'] = config_hash

        abs_pred, ref_mse = _independent_reverify(cache, cache_mse)
        if abs(cache_mse - ref_mse) > 1e-9:
            raise SystemExit(f'[ISSUE][ABORT] {cli.cell}/{cli.arm_name}/{split}: self-consistency '
                             f'reverify mismatch {cache_mse} != {ref_mse} -- refusing to write cache.')
        print(f'[build_cache_horizon_retrieval_expert01] {cli.cell}/{cli.arm_name}/{split}: '
             f'cache_restored_abs_mse={cache_mse:.6f} (self-consistency verified)')
        torch.save(cache, out_dir / f'{split}.pt')

    (out_dir / 'build_manifest.json').write_text(json.dumps({
        'cell': cli.cell, 'arm_name': cli.arm_name, 'stage1_checkpoint': arm_ckpt,
        'stage1_checkpoint_hash': stage1_hash, 'config_hash': config_hash,
        'stage2_host': cli.stage2_host, 'stage2_host_hash': _file_sha256(cli.stage2_host),
    }, indent=2))
    print(f'[build_cache_horizon_retrieval_expert01] {cli.cell}/{cli.arm_name}: cache complete -> {out_dir}')


if __name__ == '__main__':
    main()
