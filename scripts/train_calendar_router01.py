#!/usr/bin/env python3
"""TRACK-V-CALENDAR-ROUTER01 -- Calendar Router training. V5 (encoder +
5 SlotHeads) is loaded from a FROZEN best-retMSE checkpoint and NEVER
updated here (only `router.parameters()` are ever in the optimizer).
Router: `models.CalendarRouter.CalendarRouter`, tiny per-channel
Linear(6,5), zero-init. Loss: KL(soft head-teacher || router), spec
section 11 -- no auxiliary loss, no entropy/diversity/load-balancing
term, no forecasting/Stage-2 loss mixed in.

Dual checkpoint, same convention as `train_v_sharedtop100_01.py`'s V5:
`router_checkpoint_best_retmse.pth` (PRIMARY -- min validation
HARD-routed Top-1-head retMSE@10, used for Stage-2) and
`router_checkpoint_best_kl.pth` (SECONDARY -- min validation router KL,
diagnostic only).
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.CalendarRouter import CalendarRouter
from models.RelationStage1 import stable_topk_indices
from scripts.train_j_shared_encoder_drift01 import build_model
from scripts.train_k_multislot_predictive_retrieval01 import SlotHeads
from scripts.train_margutil01 import memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k
from scripts.train_retriever_pool01 import CandidatePoolCache, compute_scores
from utils.calendar_features import (
    forecast_start_calendar_features, load_date_column, shuffle_calendar_features,
)
from utils.candidate_pool import CandidatePoolConfig, gather_candidate_values, pooled_future_mse

TOP_K = 10
N_HEADS = 5
EPS = 1e-8


def load_head_oracle(path):
    return torch.load(path, map_location='cpu')


@torch.no_grad()
def hard_route_eval(exp, args, model, slot_heads, router, pool_cfg, pool_cache, calendar_feat_by_start,
                    loader, channels, device, chunk_size=4096):
    """Returns per-(query,channel)-averaged: retmse10 (hard-routed Top-10
    within the router-selected head), agg_mse, recall10, ndcg10, D, C,
    router_kl (against this split's own head-oracle soft teacher, IF
    provided via calendar_feat_by_start's companion teacher -- handled
    by caller), router_entropy, selected_head histogram."""
    sums, n = {}, 0
    head_hist = torch.zeros(N_HEADS)
    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        bsz = batch_x.size(0)
        cal = calendar_feat_by_start(batch_start_idx).to(device)
        per_ch = {}
        for c in channels:
            scores, valid_mask, pool_idx_global = compute_scores(
                pool_cfg, model, slot_heads, batch_x, exp, c, cand_mask, pool_cache, batch_start_idx, device)
            memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
            query_future = batch_y[:, :, c]
            pooled_memory_c = gather_candidate_values(memory_c, pool_idx_global)
            d_pool = pooled_future_mse(pooled_memory_c, offset_c, query_future)

            p_r = router(cal, c)  # [B, 5]
            head_sel = p_r.argmax(dim=-1)  # [B]
            for h in range(N_HEADS):
                head_hist[h] += int((head_sel == h).sum())

            # gather the selected head's own score row per-query, then top-10 within it
            scores_sel = scores.gather(1, head_sel.view(-1, 1, 1).expand(-1, 1, scores.size(-1))).squeeze(1)
            picks_local = stable_topk_indices(scores_sel, TOP_K, largest=True)
            oracle_local = stable_topk_indices(d_pool, TOP_K, largest=False)
            ind_mse_i = d_pool.gather(1, picks_local)
            y_sel = pooled_memory_c.gather(
                1, picks_local.unsqueeze(-1).expand(-1, -1, pooled_memory_c.size(-1))) + offset_c.view(-1, 1, 1)
            D_ = ind_mse_i.mean(-1) / TOP_K
            agg_pred = y_sel.mean(dim=1)
            agg_mse = ((agg_pred - query_future) ** 2).mean(-1)
            C_ = agg_mse - D_
            recall10 = recall_at_k(picks_local, oracle_local, TOP_K)
            ndcg10 = ndcg_at_k(picks_local, d_pool, torch.ones_like(d_pool, dtype=torch.bool), TOP_K)
            p_r_safe = p_r.clamp_min(EPS)
            router_entropy = -(p_r_safe * p_r_safe.log()).sum(-1)

            for key_, v in (('retmse10', ind_mse_i.mean(-1)), ('agg_mse', agg_mse), ('D', D_), ('C', C_),
                           ('recall10', recall10), ('ndcg10', ndcg10), ('router_entropy', router_entropy)):
                per_ch.setdefault(key_, []).append(v.reshape(-1).cpu())
        for k_, vals in per_ch.items():
            sums[k_] = sums.get(k_, 0.0) + torch.cat(vals).sum().item()
        n += bsz
    n_eff = max(n * len(channels), 1)
    out = {k_: v / n_eff for k_, v in sums.items()}
    out['selected_head_fraction'] = (head_hist / head_hist.sum().clamp_min(1)).tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--candidate_mask', default='raft')
    ap.add_argument('--candidate_pool_size', type=int, default=100)
    ap.add_argument('--candidate_pool_cache', required=True)
    ap.add_argument('--v5_checkpoint', required=True)
    ap.add_argument('--head_oracle_dir', required=True)
    ap.add_argument('--root_path', required=True)
    ap.add_argument('--data_path', required=True)
    ap.add_argument('--tau_h', type=float, default=1.0)
    ap.add_argument('--learning_rate', type=float, default=1e-2)
    ap.add_argument('--train_epochs', type=int, default=30)
    ap.add_argument('--patience', type=int, default=8)
    ap.add_argument('--shuffle_calendar', action='store_true', help='CRH-Shuffled control (spec 15.E)')
    ap.add_argument('--shuffle_seed', type=int, default=0)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--checkpoints', required=True)
    ap.add_argument('--smoke_test', action='store_true')
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pool_cfg = CandidatePoolConfig(mode='coarse_topk', size=cli.candidate_pool_size)
    cli.init_seed = 0
    cli.patch_len = 16
    cli.top_k = TOP_K
    cli.batch_size = 32
    exp, args, model = build_model(cli, device)
    channels = list(range(int(args.enc_in)))
    d_model = int(args.d_model)
    slot_heads = SlotHeads(d_model, n_slots=N_HEADS).to(device)

    bl = torch.load(cli.v5_checkpoint, map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    slot_heads.load_state_dict(bl['slot_heads_state_dict'])
    model.eval()
    slot_heads.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in slot_heads.parameters():
        p.requires_grad_(False)

    router = CalendarRouter(n_channels=len(channels), n_heads=N_HEADS).to(device)
    optimizer = torch.optim.Adam(router.parameters(), lr=cli.learning_rate)

    expected_meta = {'seq_len': cli.seq_len, 'pred_len': cli.pred_len, 'candidate_mask': cli.candidate_mask,
                     'candidate_pool_size': cli.candidate_pool_size, 'candidate_pool_metric': 'delta_last_cosine'}
    pool_cache_dir = Path(cli.candidate_pool_cache)
    pool_caches = {s: CandidatePoolCache(pool_cache_dir / f'{s}.pt', expected_meta) for s in ('train', 'val', 'test')}

    head_oracle = {s: load_head_oracle(Path(cli.head_oracle_dir) / f'head_oracle_{s}.pt')
                  for s in ('train', 'val', 'test')}

    date_column = load_date_column(cli.root_path, cli.data_path)
    calendar_cache = {}
    perm_by_split = {}
    for split in ('train', 'val', 'test'):
        starts = head_oracle[split]['query_start_idx']
        feat = forecast_start_calendar_features(date_column, starts, cli.seq_len)
        if cli.shuffle_calendar:
            feat, perm = shuffle_calendar_features(feat, seed=cli.shuffle_seed)
            perm_by_split[split] = perm
        calendar_cache[split] = {'starts': starts, 'feat': feat.to(device)}

    def calendar_feat_lookup(split):
        starts = calendar_cache[split]['starts']
        lut = {int(s): i for i, s in enumerate(starts.tolist())}
        feat = calendar_cache[split]['feat']

        def _lookup(batch_start_idx):
            rows = [lut[int(s)] for s in batch_start_idx.tolist()]
            return feat[rows]
        return _lookup

    loaders = {}
    for split in ('train', 'val', 'test'):
        _, loaders[split] = exp._get_data(flag=split, shuffle=False)

    def router_kl_loss(split_name, batch_start_idx, loader_calendar_fn):
        starts = head_oracle[split_name]['query_start_idx']
        lut = {int(s): i for i, s in enumerate(starts.tolist())}
        rows = [lut[int(s)] for s in batch_start_idx.tolist()]
        soft_teacher = head_oracle[split_name]['soft_teacher'][rows].to(device)  # [B, C, 5]
        cal = loader_calendar_fn(batch_start_idx)
        total = 0.0
        for c in channels:
            p_r = router(cal, c)
            teacher_c = soft_teacher[:, c, :].clamp_min(EPS)
            log_p_r = p_r.clamp_min(EPS).log()
            kl = (teacher_c * (teacher_c.log() - log_p_r)).sum(-1).mean()
            total = total + kl
        return total / len(channels)

    train_cal_fn = calendar_feat_lookup('train')
    val_cal_fn = calendar_feat_lookup('val')
    test_cal_fn = calendar_feat_lookup('test')

    config = {
        'exp': 'TRACK-V-CALENDAR-ROUTER01', 'cell': cli.cell, 'tau_h': cli.tau_h,
        'shuffle_calendar': cli.shuffle_calendar, 'shuffle_seed': cli.shuffle_seed if cli.shuffle_calendar else None,
        'router_param_count': router.param_count(), 'n_channels': len(channels), 'n_heads': N_HEADS,
        'checkpoint_policy': 'DUAL -- primary=min_val_hard_retmse10 (Stage-2 uses ONLY this), '
                             'secondary=min_val_router_kl (diagnostic only)',
    }
    out_dir = Path(cli.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2, default=str))
    print(f'[router] cell={cli.cell} shuffle={cli.shuffle_calendar} param_count={config["router_param_count"]}')
    print('[MODEL-SELECTION AUDIT]')
    print('Training objective:           KL(head_soft_teacher || router)')
    print('Validation PRIMARY selection:  val hard-routed retmse10 (TYPE C -- explicit, documented: Stage-2 '
         'comparability is the point of this experiment, see docstring)')
    print('Validation SECONDARY (diag):  val router_kl (SAME as training objective; never used for Stage-2)')
    print('Decision:                      APPROVED-WITH-JUSTIFICATION')

    if cli.smoke_test:
        for bi, (batch_x, batch_y, batch_start_idx) in enumerate(loaders['train']):
            if bi >= 2:
                break
            optimizer.zero_grad()
            loss = router_kl_loss('train', batch_start_idx, train_cal_fn)
            loss.backward()
            assert router.linear[0].weight.grad is not None and router.linear[0].weight.grad.abs().sum() > 0
            assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.parameters()), \
                '[ISSUE] encoder received gradient -- V5 must stay frozen'
            assert all(p.grad is None or p.grad.abs().sum() == 0 for p in slot_heads.parameters()), \
                '[ISSUE] slot_heads received gradient -- V5 must stay frozen'
            optimizer.step()
        print('[router] SMOKE PASS')
        return

    best_retmse = {'val': float('inf'), 'epoch': -1}
    best_kl = {'val': float('inf'), 'epoch': -1}
    epoch_rows = []
    t0 = time.time()
    for epoch in range(1, cli.train_epochs + 1):
        router.train()
        train_losses = []
        for batch_x, batch_y, batch_start_idx in loaders['train']:
            optimizer.zero_grad()
            loss = router_kl_loss('train', batch_start_idx, train_cal_fn)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))

        router.eval()
        val_eval = hard_route_eval(exp, args, model, slot_heads, router, pool_cfg, pool_caches['val'],
                                   val_cal_fn, loaders['val'], channels, device)
        val_kl_sum, n_val = 0.0, 0
        with torch.no_grad():
            for batch_x, batch_y, batch_start_idx in loaders['val']:
                val_kl_sum += float(router_kl_loss('val', batch_start_idx, val_cal_fn)) * batch_x.size(0)
                n_val += batch_x.size(0)
        val_kl = val_kl_sum / max(n_val, 1)
        val_retmse = val_eval['retmse10']
        row = {'epoch': epoch, 'train_loss': float(np.mean(train_losses)), 'val_kl': val_kl,
              'val_retmse10': val_retmse, 'val_agg_mse10': val_eval['agg_mse']}
        epoch_rows.append(row)

        payload = {'router_state_dict': router.state_dict(), 'epoch': epoch, 'val_kl': val_kl,
                  'val_retmse10': val_retmse, 'config': config}
        if val_retmse < best_retmse['val']:
            best_retmse = {'val': val_retmse, 'epoch': epoch}
            torch.save(payload, ckpt_dir / 'router_checkpoint_best_retmse.pth')
        if val_kl < best_kl['val']:
            best_kl = {'val': val_kl, 'epoch': epoch}
            torch.save(payload, ckpt_dir / 'router_checkpoint_best_kl.pth')
        print(f'[router] epoch={epoch} val_kl={val_kl:.6f} val_retmse10={val_retmse:.6f} '
             f'(best_retmse={best_retmse["epoch"]}:{best_retmse["val"]:.6f} '
             f'best_kl={best_kl["epoch"]}:{best_kl["val"]:.6f})')
        if epoch - best_retmse['epoch'] >= cli.patience:
            print(f'[router] early stop at epoch {epoch} (best_retmse={best_retmse["epoch"]})')
            break

    with open(out_dir / 'router_train_metrics.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=sorted({k for r in epoch_rows for k in r}))
        w.writeheader()
        for r in epoch_rows:
            w.writerow(r)

    def run_test(ckpt_path, tag):
        bl_r = torch.load(ckpt_path, map_location=device)
        router.load_state_dict(bl_r['router_state_dict'])
        router.eval()
        test_eval = hard_route_eval(exp, args, model, slot_heads, router, pool_cfg, pool_caches['test'],
                                    test_cal_fn, loaders['test'], channels, device)
        test_kl_sum, n_test = 0.0, 0
        with torch.no_grad():
            for batch_x, batch_y, batch_start_idx in loaders['test']:
                test_kl_sum += float(router_kl_loss('test', batch_start_idx, test_cal_fn)) * batch_x.size(0)
                n_test += batch_x.size(0)
        test_eval['router_kl'] = test_kl_sum / max(n_test, 1)
        test_eval['best_epoch'] = int(bl_r['epoch'])
        (out_dir / f'final_test_metrics_{tag}.json').write_text(json.dumps(test_eval, indent=2, default=str))
        print(f'[router] [{tag}] best_epoch={test_eval["best_epoch"]} test_retmse10={test_eval["retmse10"]:.6f} '
             f'test_router_kl={test_eval["router_kl"]:.6f} selected_head_frac={test_eval["selected_head_fraction"]}')
        return test_eval

    run_test(ckpt_dir / 'router_checkpoint_best_retmse.pth', 'best_retmse')
    run_test(ckpt_dir / 'router_checkpoint_best_kl.pth', 'best_kl')
    print(f'[router] done. cell={cli.cell} best_retmse_epoch={best_retmse["epoch"]} '
         f'best_kl_epoch={best_kl["epoch"]} wall_seconds={time.time()-t0:.1f}')


if __name__ == '__main__':
    main()
