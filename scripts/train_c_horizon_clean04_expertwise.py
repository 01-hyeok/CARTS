#!/usr/bin/env python3
"""TRACK-C-HORIZON-RETRIEVAL-CLEAN04 -- expert-wise independent adapter
training with step-level checkpointing.

Reuses, UNMODIFIED, from `scripts.train_c_horizon_frozen03`:
`build_query_cache`, `candidate_value`, `state_sha`, `rows_for_batch`.
Reuses `block_distance`/`BLOCKS` from `scripts.train_c_horizon_clean02` and
`kl_loss`/`normalized_teacher_prob`/`BLOCK_NAMES` from
`scripts.train_horizon_retrieval_expert01`. One process trains ONE block's
adapter at ONE teacher temperature (9 separate invocations cover the full
grid: 3 blocks x 3 taus) -- each with its own optimizer, its own
early-stop-free step-level checkpoint selection (min validation MSE for
THAT block only, never a combined H720 metric), so no expert's checkpoint
choice is contaminated by another expert's overfitting.
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.train_c_horizon_clean02 import BLOCKS, block_distance
from scripts.train_c_horizon_frozen03 import build_query_cache, candidate_value, state_sha
from scripts.train_factorial_e2e01 import arm_score, encode_raw
from scripts.train_horizon_retrieval_expert01 import kl_loss, normalized_teacher_prob
from scripts.train_margutil01 import build_experiment

CHECKPOINT_STEPS = (0, 25, 50, 75, 100, 150, 225, 337, 450, 562, 675)


class BlockAdapter(nn.Module):
    """Single symmetric zero-init residual bottleneck adapter for ONE
    block (spec section 9) -- identical math to
    `train_c_horizon_clean02.HorizonAdapter.blocks[i]`, just not bundled
    with the other two blocks since each is trained fully independently
    here."""

    def __init__(self, d_model, bottleneck=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, bottleneck), nn.GELU(), nn.Linear(bottleneck, d_model))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h):
        return F.normalize(h + self.net(h), dim=-1)


@torch.no_grad()
def eval_block_mse(cache, candidate_emb, memory_c_by_channel, channels, adapter, bname, top_k,
                   batch_size, exp, device, limit_rows=0):
    lo, hi = BLOCKS[bname]
    starts = cache['starts']
    n_rows = starts.size(0) if not limit_rows else min(limit_rows, starts.size(0))
    mses = []
    for s in range(0, n_rows, batch_size):
        e = min(s + batch_size, n_rows)
        bstart = starts[s:e]
        by = cache['y'][s:e].float().to(device)
        x_last = cache['x_last'][s:e].to(device)
        cand_mask, _ = exp._candidate_mask(bstart)
        bsz = e - s
        mse_ch = torch.zeros(bsz, len(channels))
        for ci, c in enumerate(channels):
            h_q = cache['emb'][c][s:e].to(device)
            h_i = candidate_emb[c]
            z_q = adapter(h_q) if adapter is not None else F.normalize(h_q, dim=-1)
            z_i = adapter(h_i) if adapter is not None else F.normalize(h_i, dim=-1)
            s_b = arm_score(z_q, z_i, None).masked_fill(~cand_mask, float('-inf'))
            picks = stable_topk_indices(s_b, top_k, largest=True)
            memory_c = memory_c_by_channel[c]
            offset_c = x_last[:, c]
            y_sel = memory_c[picks][:, :, lo:hi] + offset_c.view(-1, 1, 1)
            query_future = by[:, :, c][:, lo:hi]
            mse_ch[:, ci] = ((y_sel.mean(dim=1) - query_future) ** 2).mean(dim=-1).cpu()
        mses.append(mse_ch.mean(dim=-1))
    return float(torch.cat(mses).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell', required=True)
    ap.add_argument('--g_best_checkpoint', required=True)
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--block', required=True, choices=('block1', 'block2', 'block3'))
    ap.add_argument('--tau_t', type=float, required=True)
    ap.add_argument('--pred_len', type=int, default=720)
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--chunk_size', type=int, default=2048)
    ap.add_argument('--tau_s', type=float, default=0.10)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--max_steps', type=int, default=675)
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--init_seed', type=int, default=0)
    ap.add_argument('--out_dir', default='results/TRACK-C-HORIZON-RETRIEVAL-CLEAN04')
    ap.add_argument('--checkpoints', default='checkpoints/track_c_horizon_retrieval_clean04')
    ap.add_argument('--smoke_test', action='store_true')
    ap.add_argument('--checkpoint_steps', default=None,
                    help='comma-separated int list; overrides the ETTh1-tuned CHECKPOINT_STEPS default '
                         '(used for CLEAN05 dataset-size-normalized epoch-fraction schedules)')
    cli = ap.parse_args()
    custom_checkpoint_steps = (tuple(int(x) for x in cli.checkpoint_steps.split(','))
                               if cli.checkpoint_steps else None)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tau_tag = f"tau{str(cli.tau_t).replace('0.', '0').replace('.', '')}"
    arm_name = f'{cli.block}_{tau_tag}'
    cell_dir = Path(cli.out_dir) / cli.cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints) / cli.cell / arm_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    g_ckpt = torch.load(cli.g_best_checkpoint, map_location='cpu')
    fp = g_ckpt['fingerprint']
    assert fp.get('arm') == 'g_transformer'
    assert fp.get('relation_encoder_type') == 'transformer'
    assert int(fp.get('patch_len', -1)) == 16 and int(fp.get('stride', -1)) == 16
    assert int(fp.get('d_model', -1)) == 128

    torch.manual_seed(cli.init_seed)
    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.pred_len, 'batch_size': cli.batch_size,
        'seed': cli.init_seed, 'top_k': cli.top_k, 'tau_topk': 0.1,
        'patch_len': 16, 'stride': 16,
        'relation_encoder_type': 'transformer', 'relation_self_fill': 'zero',
    })
    exp._ensure_memory()
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.load_state_dict(g_ckpt['model_state_dict'])
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    channels = list(range(int(args.enc_in)))
    d_model = int(args.d_model)

    frozen_sha_before = state_sha(model.state_dict())
    g_ckpt_hash = hashlib.sha256(Path(cli.g_best_checkpoint).read_bytes()).hexdigest()

    candidate_emb = {c: encode_raw(model, exp.memory_x, c).detach() for c in channels}
    train_cache = build_query_cache(model, exp, 'train', channels, device)
    val_cache = build_query_cache(model, exp, 'val', channels, device)
    memory_c_by_channel = {c: candidate_value(exp.memory_y, exp.memory_x_last, c).to(device) for c in channels}
    assert state_sha(model.state_dict()) == frozen_sha_before

    lo, hi = BLOCKS[cli.block]
    adapter = BlockAdapter(d_model).to(device)
    adapter_init_sha = state_sha(adapter.state_dict())
    optimizer = torch.optim.Adam(adapter.parameters(), lr=cli.learning_rate)

    if cli.smoke_test:
        with torch.no_grad():
            h_q = train_cache['emb'][channels[0]][:4].to(device)
            h_i = candidate_emb[channels[0]][:50]
            s_g = arm_score(F.normalize(h_q, dim=-1), F.normalize(h_i, dim=-1), None)
            s_a = arm_score(adapter(h_q), adapter(h_i), None)
            zero_ok = torch.allclose(s_a, s_g, atol=1e-4)
        assert zero_ok, 'zero-init adapter score must equal global score'
        print(f'[clean04_expertwise] {arm_name} smoke zero-init check: PASS')

    n_rows = train_cache['starts'].size(0)
    perm_gen = torch.Generator().manual_seed(cli.init_seed)
    perm = torch.randperm(n_rows, generator=perm_gen)
    step_rows = []
    best = {'val': float('inf'), 'step': -1}
    step = 0
    max_steps = 6 if cli.smoke_test else cli.max_steps
    checkpoint_steps = (0, 3, 6) if cli.smoke_test else (custom_checkpoint_steps or CHECKPOINT_STEPS)
    t0 = time.time()

    def do_eval_and_maybe_checkpoint(step):
        adapter.eval()
        val_mse = eval_block_mse(val_cache, candidate_emb, memory_c_by_channel, channels, adapter,
                                 cli.block, cli.top_k, cli.batch_size, exp, device,
                                 limit_rows=(64 if cli.smoke_test else 0))
        step_rows.append({'step': step, 'val_block_mse': val_mse})
        if val_mse < best['val']:
            best.update({'val': val_mse, 'step': step})
            torch.save({'adapter_state_dict': adapter.state_dict(), 'block': cli.block, 'tau_t': cli.tau_t,
                       'step': step, 'val_block_mse': val_mse, 'adapter_init_sha256': adapter_init_sha,
                       'frozen_trunk_sha256': frozen_sha_before, 'g_best_checkpoint_hash': g_ckpt_hash,
                       'g_best_checkpoint': cli.g_best_checkpoint}, ckpt_dir / 'checkpoint.pth')
        print(f'[clean04_expertwise] {arm_name} step={step} val_block_mse={val_mse:.6f}'
             f'{" *best*" if step == best["step"] else ""}')
        adapter.train(True)

    if 0 in checkpoint_steps:
        do_eval_and_maybe_checkpoint(0)

    epoch_perm_idx = 0
    while step < max_steps:
        if epoch_perm_idx >= n_rows:
            perm = torch.randperm(n_rows, generator=perm_gen)
            epoch_perm_idx = 0
        rows = perm[epoch_perm_idx:epoch_perm_idx + cli.batch_size]
        epoch_perm_idx += cli.batch_size
        if rows.numel() == 0:
            continue
        bstart = train_cache['starts'][rows]
        by = train_cache['y'][rows].float().to(device)
        x_last = train_cache['x_last'][rows].to(device)
        cand_mask, _ = exp._candidate_mask(bstart)
        optimizer.zero_grad()
        batch_loss = 0.0
        for c in channels:
            h_q = train_cache['emb'][c][rows].to(device)
            h_i = candidate_emb[c]
            memory_c = memory_c_by_channel[c]
            offset_c = x_last[:, c]
            query_future = by[:, :, c]
            z_q = adapter(h_q)
            z_i = adapter(h_i)
            s_b = arm_score(z_q, z_i, None)
            with torch.no_grad():
                d_b = block_distance(memory_c, offset_c, query_future, lo, hi, cli.chunk_size)
                d_b = d_b.masked_fill(~cand_mask, float('inf'))
                p_t_b = normalized_teacher_prob(d_b, cand_mask, cli.tau_t)
            l_b = kl_loss(p_t_b, s_b, cand_mask, cli.tau_s)
            (l_b / len(channels)).backward()
            batch_loss += float(l_b.detach()) / len(channels)
        if cli.smoke_test:
            has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.parameters())
            trunk_untouched = all(p.grad is None for p in model.parameters())
            assert has_grad and trunk_untouched
        optimizer.step()
        step += 1
        if step in checkpoint_steps:
            do_eval_and_maybe_checkpoint(step)

    assert state_sha(model.state_dict()) == frozen_sha_before, '[ISSUE][ABORT] trunk drifted'

    with open(cell_dir / f'expert_step_metrics_{arm_name}.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['step', 'val_block_mse'])
        w.writeheader()
        for r in step_rows:
            w.writerow(r)

    summary = {'cell': cli.cell, 'block': cli.block, 'tau_t': cli.tau_t, 'arm_name': arm_name,
              'best_step': best['step'], 'best_val_block_mse': best['val'],
              'adapter_init_sha256': adapter_init_sha, 'frozen_trunk_sha256': frozen_sha_before,
              'g_best_checkpoint_hash': g_ckpt_hash, 'wall_clock_seconds': time.time() - t0}
    if cli.smoke_test:
        smoke_dir = Path(cli.out_dir) / 'smoke' / cli.cell
        smoke_dir.mkdir(parents=True, exist_ok=True)
        (smoke_dir / f'smoke_report_{arm_name}.json').write_text(json.dumps(summary, indent=2))
        print(f'[clean04_expertwise] {arm_name} SMOKE TEST: PASS')
    else:
        (cell_dir / f'expert_summary_{arm_name}.json').write_text(json.dumps(summary, indent=2))
        print(f'[clean04_expertwise] done. {arm_name} best_step={best["step"]} best_val={best["val"]:.6f}')


if __name__ == '__main__':
    main()
