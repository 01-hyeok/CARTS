#!/usr/bin/env python3
"""TRACK-A-SETORACLE-HOSTFREE01 -- HOST-FREE delta-space Stage-2 retrieval
cache builder.

Sibling of `scripts/build_tf_oracle_learnability01_retrieval_cache.py`
(identical free-running K-step greedy selection loop, identical delta-space
cache-correctness convention: `query_offset` is NEVER added into the cached
`relation_outputs`, only carried alongside for Stage-2's own restoration).
Differs in EXACTLY one respect, matching `train_setoracle_hostfree01.py`'s
own host-free substitution: the aggregation weighting source is
`UniformHost` (equal weight over every valid Top-K pick, NOT a trained
HostScorer) -- so there is no `--stage2_host` argument anywhere in this
script; it never loads or references any Stage-2 host checkpoint at
cache-building time. Loads the just-trained `individual_tf_hostfree_cosine`/
`set_tf_hostfree_cosine` Stage-1 checkpoints instead of `train_factorial_
e2e01`'s onpolicy/tf-oracle arms.

The resulting `relation_outputs` is therefore genuinely the SAME
host-free/uniform-weighted aggregate that Stage-1's own
`free_running_aggregate_future_mse` metric already reports -- this cache is
just that same signal, re-expressed in delta-space so the reused Stage-2
fusion trainer (`train_setlossctrl_stage2_retrain02.py`, UNMODIFIED, per
this session's established convention) can combine it with a frozen base
forecaster's own prediction. The `--stage2_host` argument that trainer takes
is ONLY an architecture template (its own trained weights are explicitly
never loaded -- see `build_fresh_stage2` there); this cache builder itself
plays no role in that and is fully host-free start to finish.
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

from models.SequentialSetRetriever import SetConditioner
from scripts.train_factorial_e2e01 import arm_score, encode_raw
from scripts.train_margutil01 import build_experiment, memory_value
from scripts.train_setoracle_hostfree01 import UniformHost

ARMS = ('individual_tf_hostfree_cosine', 'set_tf_hostfree_cosine')
EXPECTED_AXIS_ORACLE = {'individual_tf_hostfree_cosine': 'individual',
                        'set_tf_hostfree_cosine': 'greedy_set'}


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
    if fp.get('axis_oracle') != EXPECTED_AXIS_ORACLE[arm_name]:
        problems.append(f"axis_oracle={fp.get('axis_oracle')!r} != "
                        f"{EXPECTED_AXIS_ORACLE[arm_name]!r}")
    if fp.get('axis_prefix') != 'tf':
        problems.append(f"axis_prefix={fp.get('axis_prefix')!r} != 'tf'")
    if fp.get('axis_score') != 'cosine':
        problems.append(f"axis_score={fp.get('axis_score')!r} != 'cosine'")
    if int(fp.get('top_k', -1)) != int(top_k):
        problems.append(f"top_k={fp.get('top_k')} != {top_k}")
    enc_in = len(fp.get('channels', []))
    if not fp.get('channels') or list(fp['channels']) != list(range(enc_in)):
        problems.append(f"channels={fp.get('channels')} is not a contiguous 0..N-1 range")
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
def build_cache_for_split(arm_ckpt_path, arm_name, reference_ckpt, pred_len, split,
                          top_k, chunk_size, device):
    ckpt = torch.load(arm_ckpt_path, map_location='cpu')
    args_dict = ckpt['args']
    exp, args = build_experiment(reference_ckpt, {
        'pred_len': pred_len, 'seq_len': pred_len,
        'batch_size': 32, 'seed': int(args_dict.get('seed', 0)), 'top_k': top_k,
    })
    exp._ensure_memory()
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    d_model = int(args.d_model)
    sc = SetConditioner(d_model).to(device)
    sc.load_state_dict(ckpt['set_conditioner_state_dict'])
    sc.eval()
    for p in sc.parameters():
        p.requires_grad_(False)

    host = UniformHost(top_k=top_k, tau_topk=1.0)  # host-free: equal weight, no trained host
    channels = list(range(int(args.enc_in)))
    _, loader = exp._get_data(flag=split, shuffle=False)

    all_start_idx, all_relation_outputs, all_query_embs = [], [], []
    all_query_offset = []
    all_topk_idx, all_alpha, all_valid_count, all_dup, all_invalid = [], [], [], [], []

    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, counts = exp._candidate_mask(batch_start_idx)
        bsz = batch_x.size(0)

        rel_out_ch = torch.zeros(bsz, len(channels), 1, pred_len)
        query_emb_ch = torch.zeros(bsz, len(channels), 1, d_model)
        topk_idx_ch = torch.zeros(bsz, len(channels), top_k, dtype=torch.long)
        alpha_ch = torch.zeros(bsz, len(channels), top_k)
        query_offset_ch = torch.zeros(bsz, len(channels))

        for c in channels:
            E = encode_raw(model, exp.memory_x, c)
            z_q = encode_raw(model, batch_x, c)
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            memory_c_b = memory_c.unsqueeze(0).expand(batch_x.size(0), -1, -1)
            host_scores = host.scores(batch_x, c, cand_mask)  # all-zero (UniformHost)

            selected = torch.zeros_like(cand_mask)
            neg_inf = torch.finfo(z_q.dtype).min / 4
            picks = []
            for t in range(top_k):
                h_t = z_q if t == 0 else sc(z_q, E[torch.stack(picks, dim=1)].mean(dim=1))
                u_hat = arm_score(h_t, E, None)
                valid_now = cand_mask & ~selected
                nxt = u_hat.masked_fill(~valid_now, neg_inf).argmax(dim=-1)
                picks.append(nxt)
                selected = selected.scatter(1, nxt.unsqueeze(-1), True)
            picks_t = torch.stack(picks, dim=1)

            sc_sel = host_scores.gather(1, picks_t)  # all zero
            alpha = torch.softmax(sc_sel / float(host.tau_topk), dim=-1)  # uniform, by construction
            tgt_delta = memory_c_b.gather(1, picks_t.unsqueeze(-1).expand(-1, -1, memory_c_b.size(-1)))
            delta_ret = (alpha.unsqueeze(-1) * tgt_delta).sum(1)

            rel_out_ch[:, c, 0, :] = delta_ret.cpu()
            query_emb_ch[:, c, 0, :] = z_q.detach().cpu()
            topk_idx_ch[:, c, :] = picks_t.cpu()
            alpha_ch[:, c, :] = alpha.cpu()
            query_offset_ch[:, c] = offset_c.detach().cpu()

        dup = (topk_idx_ch.sort(dim=-1).values[:, :, 1:] ==
              topk_idx_ch.sort(dim=-1).values[:, :, :-1]).sum(dim=-1)
        invalid = (counts.cpu().unsqueeze(-1).expand(-1, len(channels)) < top_k).float()

        all_start_idx.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx)
                             else torch.as_tensor(batch_start_idx))
        all_relation_outputs.append(rel_out_ch)
        all_query_embs.append(query_emb_ch)
        all_topk_idx.append(topk_idx_ch)
        all_alpha.append(alpha_ch)
        all_valid_count.append(counts.cpu())
        all_dup.append(dup)
        all_invalid.append(invalid)
        all_query_offset.append(query_offset_ch)

    cache = {
        'batch_start_idx': torch.cat(all_start_idx),
        'relation_outputs': torch.cat(all_relation_outputs),
        'relation_query_embs': torch.cat(all_query_embs),
        'topk_idx': torch.cat(all_topk_idx),
        'alpha': torch.cat(all_alpha),
        'query_offset': torch.cat(all_query_offset),
        'valid_count': torch.cat(all_valid_count),
        'duplicate_count': torch.cat(all_dup),
        'invalid_row': torch.cat(all_invalid),
        'channels': channels, 'top_k': top_k, 'pred_len': pred_len,
        'arm_name': arm_name, 'split': split,
        'value_space': 'delta', 'offset_included': False,
        'cache_schema_version': 'corrected_delta_v1',
        'host_free': True,
    }

    # self-consistency: independently restore absolute predictions from the
    # cache tensors just written and recompute MSE against the SAME split's
    # query_future (recomputed fresh from the dataloader, not reused) --
    # must match the alpha==uniform arithmetic-mean construction to float
    # precision or we refuse to write the cache (same discipline as every
    # other Track-A cache builder this session).
    _, loader2 = exp._get_data(flag=split, shuffle=False)
    qf_all = []
    for _, batch_y, _ in loader2:
        qf_all.append(batch_y.permute(0, 2, 1))  # [B, C, pred_len]
    qf_cat = torch.cat(qf_all)
    abs_pred = cache['relation_outputs'][:, :, 0, :] + cache['query_offset'].unsqueeze(-1)
    cache_mse = float(((abs_pred - qf_cat) ** 2).mean(-1).mean())
    return cache, cache_mse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True)
    ap.add_argument('--arm_name', required=True, choices=ARMS)
    ap.add_argument('--stage1_out_dir', default='results/TRACK-A-SETORACLE-HOSTFREE01')
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--out_dir', default='results/TRACK-A-SETORACLE-HOSTFREE01-STAGE2')
    ap.add_argument('--checkpoints_root', default='checkpoints/track_a_setoracle_hostfree01')
    cli = ap.parse_args()

    rm_json = Path(cli.stage1_out_dir) / cli.cell / f'retrieval_metrics_{cli.arm_name}.json'
    if not rm_json.exists():
        raise SystemExit(f'[ISSUE][ABORT] {rm_json} not found -- Stage-1 arm not complete yet.')
    marker = Path(cli.stage1_out_dir) / cli.cell / f'DONE_{cli.arm_name}.marker'
    if not marker.exists():
        raise SystemExit(f'[ISSUE][ABORT] {marker} not found -- Stage-1 arm has no completion marker.')

    arm_ckpt = f'{cli.checkpoints_root}/{cli.cell}/{cli.arm_name}/checkpoint.pth'
    validate_arm_checkpoint(arm_ckpt, rm_json, cli.cell, cli.arm_name, cli.top_k)
    print(f'[build_cache_setoracle_hostfree01] {cli.cell}/{cli.arm_name}: '
         f'checkpoint validated -- {arm_ckpt}')

    stage1_hash = _file_sha256(arm_ckpt)
    config_hash = hashlib.sha256(
        json.dumps({'cell': cli.cell, 'arm_name': cli.arm_name, 'pred_len': cli.pred_len,
                   'top_k': cli.top_k, 'chunk_size': cli.chunk_size, 'host_free': True},
                   sort_keys=True).encode()
    ).hexdigest()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(cli.out_dir) / 'cache' / cli.cell / cli.arm_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'val', 'test'):
        cache, cache_mse = build_cache_for_split(arm_ckpt, cli.arm_name, cli.reference_ckpt,
                                                 cli.pred_len, split, cli.top_k, cli.chunk_size, device)
        cache.update({'stage1_checkpoint_path': str(arm_ckpt), 'stage1_checkpoint_hash': stage1_hash,
                     'config_hash': config_hash})
        out_path = out_dir / f'{split}.pt'
        n_dup = int((cache['duplicate_count'] > 0).sum())
        n_inv = int(cache['invalid_row'].sum())
        print(f'[build_cache_setoracle_hostfree01] {cli.cell}/{cli.arm_name}/{split}: '
             f'n_queries={cache["batch_start_idx"].numel()} duplicate_rows={n_dup} '
             f'invalid_rows={n_inv} cache_restored_abs_mse={cache_mse:.6f} -> {out_path}')
        if n_dup or n_inv:
            print(f'[ISSUE] {cli.cell}/{cli.arm_name}/{split}: {n_dup} duplicate-selection rows, '
                 f'{n_inv} invalid/short-candidate-pool rows found.')
        torch.save(cache, out_path)

    manifest = {'value_space': 'delta', 'offset_included': False,
               'cache_schema_version': 'corrected_delta_v1', 'host_free': True,
               'stage1_checkpoint_path': str(arm_ckpt), 'stage1_checkpoint_hash': stage1_hash,
               'config_hash': config_hash, 'cell': cli.cell, 'arm_name': cli.arm_name}
    (out_dir / 'cache_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(f'[build_cache_setoracle_hostfree01] wrote {out_dir / "cache_manifest.json"}')


if __name__ == '__main__':
    main()
