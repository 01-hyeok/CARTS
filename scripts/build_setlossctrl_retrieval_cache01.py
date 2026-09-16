#!/usr/bin/env python3
"""EXP-SET-LOSS-STAGE2-RETRAIN01 -- retrieval cache builder for
TRACK-A-SET-LOSS-CONTROL01 checkpoints (Hard CE / MultiPos / SRM / SoftCE,
Greedy Set Oracle, On-policy, Cosine, all 7 ETTh1 channels).

Sibling of `scripts/build_choicece_retrieval_cache01.py` (same method,
same reuse of `encode_raw`/`arm_score`/`HostScorer`/`candidate_weights`
from `scripts/train_factorial_e2e01.py`, same free-running-only,
never-Oracle-prefixed sequential selection, same fixed-S0_wce-host alpha).
Differs only in: (1) reads TRACK-A-SET-LOSS-CONTROL01's own checkpoint
format (`model_state_dict` + `set_conditioner_state_dict`, no
`metric_state_dict` -- that experiment's scope is cosine-only, so there is
no asymmetric RetrievalMetric to load), (2) validates against that
experiment's own fingerprint fields (`loss_name` instead of Factorial's
`axis_oracle`/`axis_prefix`/`axis_score` triple, though the latter three
are also present and checked since SET-LOSS-CONTROL01's fingerprint
includes them too, fixed at greedy_set/onpolicy/cosine for every arm).
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
from scripts.train_factorial_e2e01 import HostScorer, arm_score, candidate_weights, encode_raw
from scripts.train_margutil01 import build_experiment, memory_value

ARMS = ('A0_hard_choice', 'A1_adaptive_multipos', 'A2_srm', 'A3_setutility_softce')


def validate_arm_checkpoint(arm_ckpt, retrieval_metrics_json, cell, arm_name, top_k):
    ckpt = torch.load(arm_ckpt, map_location='cpu')
    fp = ckpt.get('fingerprint', {})
    rm = json.loads(Path(retrieval_metrics_json).read_text())

    problems = []
    if fp.get('loss_name') != arm_name:
        problems.append(f"loss_name={fp.get('loss_name')!r} != {arm_name!r}")
    if fp.get('axis_oracle') != 'greedy_set':
        problems.append(f"axis_oracle={fp.get('axis_oracle')!r} != 'greedy_set'")
    if fp.get('axis_prefix') != 'onpolicy':
        problems.append(f"axis_prefix={fp.get('axis_prefix')!r} != 'onpolicy'")
    if fp.get('axis_score') != 'cosine':
        problems.append(f"axis_score={fp.get('axis_score')!r} != 'cosine'")
    if int(fp.get('top_k', -1)) != int(top_k):
        problems.append(f"top_k={fp.get('top_k')} != {top_k}")
    if list(fp.get('channels', [])) != list(range(7)):
        problems.append(f"channels={fp.get('channels')} != [0..6]")
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
        'pred_len': pred_len, 'seq_len': pred_len,
        'batch_size': 32, 'seed': int(args_dict.get('seed', 1)), 'top_k': top_k,
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

    host = HostScorer(stage2_host, device)
    channels = list(range(int(args.enc_in)))
    _, loader = exp._get_data(flag=split, shuffle=False)

    all_start_idx, all_relation_outputs, all_query_embs = [], [], []
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

        for c in channels:
            E = encode_raw(model, exp.memory_x, c)
            z_q = encode_raw(model, batch_x, c)
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            futures = memory_c + offset_c.view(-1, 1, 1)
            host_scores = host.scores(batch_x, c, cand_mask)

            selected = torch.zeros_like(cand_mask)
            neg_inf = torch.finfo(z_q.dtype).min / 4
            picks = []
            for t in range(top_k):
                h_t = z_q if t == 0 else sc(z_q, E[torch.stack(picks, dim=1)].mean(dim=1))
                u_hat = arm_score(h_t, E, None)  # cosine only, no metric
                valid_now = cand_mask & ~selected
                nxt = u_hat.masked_fill(~valid_now, neg_inf).argmax(dim=-1)
                picks.append(nxt)
                selected = selected.scatter(1, nxt.unsqueeze(-1), True)
            picks_t = torch.stack(picks, dim=1)  # FREE-RUNNING only, on-policy by construction

            sc_sel = host_scores.gather(1, picks_t)
            alpha = torch.softmax(sc_sel / float(host.tau_topk), dim=-1)
            tgt = futures.gather(1, picks_t.unsqueeze(-1).expand(-1, -1, futures.size(-1)))
            y_ret = (alpha.unsqueeze(-1) * tgt).sum(1)

            rel_out_ch[:, c, 0, :] = y_ret.cpu()
            query_emb_ch[:, c, 0, :] = z_q.detach().cpu()
            topk_idx_ch[:, c, :] = picks_t.cpu()
            alpha_ch[:, c, :] = alpha.cpu()

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

    return {
        'batch_start_idx': torch.cat(all_start_idx),
        'relation_outputs': torch.cat(all_relation_outputs),
        'relation_query_embs': torch.cat(all_query_embs),
        'topk_idx': torch.cat(all_topk_idx),
        'alpha': torch.cat(all_alpha),
        'valid_count': torch.cat(all_valid_count),
        'duplicate_count': torch.cat(all_dup),
        'invalid_row': torch.cat(all_invalid),
        'channels': channels, 'top_k': top_k, 'pred_len': pred_len,
        'arm_name': arm_name, 'split': split,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True)
    ap.add_argument('--arm_name', required=True, choices=ARMS)
    ap.add_argument('--stage1_out_dir', default='results/TRACK-A-SET-LOSS-CONTROL01')
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--stage2_host', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--out_dir', default='results/EXP-SET-LOSS-STAGE2-RETRAIN01')
    cli = ap.parse_args()

    rm_json = Path(cli.stage1_out_dir) / cli.cell / f'retrieval_metrics_{cli.arm_name}.json'
    if not rm_json.exists():
        raise SystemExit(f'[ISSUE][ABORT] {rm_json} not found -- Stage-1 arm not complete yet.')
    marker = Path(cli.stage1_out_dir) / cli.cell / f'DONE_{cli.arm_name}.marker'
    if not marker.exists():
        raise SystemExit(f'[ISSUE][ABORT] {marker} not found -- Stage-1 arm has no completion marker.')

    rm = json.loads(rm_json.read_text())
    arm_ckpt = f'checkpoints/track_a_set_loss_control01/{cli.cell}/{cli.arm_name}/checkpoint.pth'
    validate_arm_checkpoint(arm_ckpt, rm_json, cli.cell, cli.arm_name, cli.top_k)
    print(f'[build_cache_setlossctrl01] {cli.cell}/{cli.arm_name}: checkpoint validated -- {arm_ckpt}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(cli.out_dir) / 'cache' / cli.cell / cli.arm_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'val', 'test'):
        cache = build_cache_for_split(arm_ckpt, cli.arm_name, cli.stage2_host, cli.reference_ckpt,
                                      cli.pred_len, split, cli.top_k, cli.chunk_size, device)
        out_path = out_dir / f'{split}.pt'
        torch.save(cache, out_path)
        n_dup = int((cache['duplicate_count'] > 0).sum())
        n_inv = int(cache['invalid_row'].sum())
        print(f'[build_cache_setlossctrl01] {cli.cell}/{cli.arm_name}/{split}: '
             f'n_queries={cache["batch_start_idx"].numel()} duplicate_rows={n_dup} '
             f'invalid_rows={n_inv} -> {out_path}')
        if n_dup or n_inv:
            print(f'[ISSUE] {cli.cell}/{cli.arm_name}/{split}: {n_dup} duplicate-selection rows, '
                 f'{n_inv} invalid/short-candidate-pool rows found.')


if __name__ == '__main__':
    main()
