#!/usr/bin/env python3
"""TRACK-W-TIMESTAMP-FUSION01 -- Stage1 trainer, Phase 1 (C0/C1/C2) and
Phase 2 (T-V0/T-V1/T-V2/T-V5) in one script, mirroring
`train_t_pure_multislot01.py`'s own num_slots-generic design.

    --mode zero_proj   (C0/C1/C2, and T-V0 which is identical to C1):
        no SlotHeads at all -- z_q used directly against candidates,
        exactly `build_t2_true_original_kl_cache01.compute_scores_true_original_kl`'s
        own convention (zero projection parameters), reshaped to [B,1,N]
        so every downstream function (round_robin_topk_selection,
        hard_eval_decomposition, kl_loss_from_prob, ...) stays identical
        to the --mode slots path.
    --mode slots       (T-V1/T-V2/T-V5): `SlotHeads(n_slots=--num_slots)`
        applied to z_q, candidates unprojected -- exactly
        `train_t_pure_multislot01.compute_scores_full_grad`'s own
        convention, with `encode_raw` swapped for `encode_raw_timestamp`.

    --time_mode no_time       (C0): P_t's input is always an all-zero
        tensor (AUDIT.md PART 2.3) -- architecture/param count/init hash
        identical to real_time/shuffled_time.
    --time_mode real_time     (C1, and every T-V* arm): real, aligned
        timestamp windows throughout train/val/test.
    --time_mode shuffled_time (C2): train-split query/candidate
        timestamp correspondence broken by one fixed seeded permutation
        over the train window index space (AUDIT.md PART 2.6);
        val/test evaluation uses REAL correspondence, same as C1, so the
        C1-vs-C2 TEST comparison isolates "did training against broken
        correspondence hurt real-world retrieval."

Checkpoint/early-stop criterion: min validation KL -- per spec section 8,
NOT the historical V0-V5 convention (min val retmse10), which this
script's own val loop also still logs for comparison. See AUDIT.md
PART 3 for the historical-comparison-validity caveat this creates.

Reused UNMODIFIED: `round_robin_topk_selection`, `hard_eval_decomposition`,
`spearman_batch`, `slot_mechanism_diagnostics`, `naive_round_robin_picks`
(`train_t_pure_multislot01`), `SlotHeads`, `kl_loss_from_prob`,
`slot_overlap_penalty` (`train_k_multislot_predictive_retrieval01`),
`normalized_teacher_prob` (`train_horizon_retrieval_expert01`),
`memory_value`, `individual_utility_memsafe` (`train_margutil01`/
`train_factorial_e2e01`), `recall_at_k`/`ndcg_at_k`
(`train_patch_retrieval_expert01`), `stable_topk_indices`
(`RelationStage1`), `make_loader_generator`/`batch_order_sha256`/
`set_global_seeds` (`rng_control01`), `effective_rank_entropy`
(`train_j_shared_encoder_drift01`).
"""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from models.TimestampRelationEncoder import TimestampFusionEncoder, zero_time_features
from scripts.rng_control01 import batch_order_sha256, make_loader_generator
from scripts.train_factorial_e2e01 import individual_utility_memsafe
from scripts.train_horizon_retrieval_expert01 import normalized_teacher_prob
from scripts.train_j_shared_encoder_drift01 import effective_rank_entropy
from scripts.train_k_multislot_predictive_retrieval01 import (
    SlotHeads, kl_loss_from_prob, slot_overlap_penalty,
)
from scripts.train_margutil01 import memory_value
from scripts.train_patch_retrieval_expert01 import ndcg_at_k, recall_at_k
from scripts.train_t_pure_multislot01 import (
    hard_eval_decomposition, naive_round_robin_picks, round_robin_topk_selection,
    slot_mechanism_diagnostics, spearman_batch,
)
from scripts.train_w_timestamp_common01 import (
    build_timestamp_experiment, encode_raw_timestamp, make_shuffle_permutation,
    relation_encoder_param_count, shuffled_memory_x_mark,
)

EPS = 1e-8
TOP_K = 10
VALID_NUM_SLOTS = (1, 2, 5)
TIME_MODES = ('no_time', 'real_time', 'shuffled_time')


def state_hash_module(module):
    h = hashlib.sha256()
    for k in sorted(module.state_dict()):
        h.update(k.encode())
        h.update(module.state_dict()[k].detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def get_time_input(real_mark, time_mode, time_feat_dim, x_for_shape):
    """Resolve the actual `c` tensor fed into the encoder for a given
    arm/time_mode. `real_mark` is whatever the loader/memory bank
    actually carries (possibly already shuffled, for TRAIN+C2 -- see
    `resolve_marks_for_split`); `no_time` always overrides to zero
    regardless of what was loaded, so C0 never depends on correct
    shuffling/alignment plumbing at all."""
    if time_mode == 'no_time':
        return zero_time_features(x_for_shape, time_feat_dim)
    return real_mark


def compute_scores_zero_proj(time_encoder, batch_x, batch_x_mark, memory_x, memory_x_mark, c):
    z_q = encode_raw_timestamp(time_encoder, batch_x, batch_x_mark, c)      # [B,D]
    k_full = F.normalize(encode_raw_timestamp(time_encoder, memory_x, memory_x_mark, c), dim=-1)  # [N,D]
    scores = torch.matmul(F.normalize(z_q, dim=-1), k_full.transpose(0, 1))  # [B,N]
    return scores.unsqueeze(1)  # [B,1,N] -- S=1, zero projection params


def compute_scores_slots(time_encoder, slot_heads, batch_x, batch_x_mark, memory_x, memory_x_mark, c):
    z_q = encode_raw_timestamp(time_encoder, batch_x, batch_x_mark, c)      # [B,D]
    q = slot_heads(z_q)  # [B,S,D], L2-normalized inside SlotHeads
    k_full = F.normalize(encode_raw_timestamp(time_encoder, memory_x, memory_x_mark, c), dim=-1)  # [N,D]
    scores = torch.einsum('bsd,nd->bsn', q, k_full)  # [B,S,N]
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--mode', required=True, choices=('zero_proj', 'slots'))
    ap.add_argument('--num_slots', type=int, default=1, choices=VALID_NUM_SLOTS,
                    help='only used when --mode slots')
    ap.add_argument('--time_mode', required=True, choices=TIME_MODES)
    ap.add_argument('--arm_name', required=True, help='C0/C1/C2/T-V0/T-V1/T-V2/T-V5, for logging only')
    ap.add_argument('--cell', required=True)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--seq_len', type=int, required=True)
    ap.add_argument('--patch_len', type=int, default=16)
    ap.add_argument('--top_k', type=int, default=TOP_K)
    ap.add_argument('--tau_t', type=float, default=0.1)
    ap.add_argument('--tau_s', type=float, default=0.1)
    ap.add_argument('--slot_std', type=float, default=1e-3)
    ap.add_argument('--time_proj_dim', type=int, default=32)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--train_epochs', type=int, default=10)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--init_seed', type=int, default=0)
    ap.add_argument('--loader_seed', type=int, default=0)
    ap.add_argument('--shuffle_seed', type=int, default=0)
    ap.add_argument('--out_dir', default='results/TRACK-W-TIMESTAMP-FUSION01')
    ap.add_argument('--checkpoints', default='checkpoints/track_w_timestamp_fusion01')
    ap.add_argument('--limit_batches', type=int, default=0, help='SMOKE ONLY')
    ap.add_argument('--smoke_test', action='store_true')
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(cli.out_dir) / cli.cell / cli.arm_name
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints) / cli.cell / cli.arm_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    exp, args, time_feat_dim = build_timestamp_experiment(cli, device)
    channels = list(range(int(args.enc_in)))
    d_model = int(args.d_model)
    d_ff = int(args.d_ff)

    time_encoder = TimestampFusionEncoder(
        seq_len=cli.seq_len, d_model=d_model, d_ff=d_ff, time_feat_dim=time_feat_dim,
        time_proj_dim=cli.time_proj_dim, dropout=float(args.dropout),
        retrieval_similarity=getattr(args, 'retrieval_similarity', 'cosine'),
    ).to(device)
    init_hash = state_hash_module(time_encoder)
    new_param_count = time_encoder.param_count()
    old_param_count = relation_encoder_param_count(args)

    slot_heads = None
    if cli.mode == 'slots':
        slot_heads = SlotHeads(d_model, n_slots=cli.num_slots, std=cli.slot_std).to(device)

    params = list(time_encoder.parameters()) + (list(slot_heads.parameters()) if slot_heads else [])
    optimizer = torch.optim.Adam(params, lr=cli.learning_rate)

    train_gen = make_loader_generator(cli.loader_seed)
    train_dataset, train_loader = exp._get_data(flag='train', shuffle=True, generator=train_gen,
                                               include_time_mark=True)
    _, val_loader = exp._get_data(flag='val', shuffle=False, include_time_mark=True)
    _, test_loader = exp._get_data(flag='test', shuffle=False, include_time_mark=True)

    perm = None
    shuffled_train_loader = train_loader
    if cli.time_mode == 'shuffled_time':
        from scripts.train_w_timestamp_common01 import ShuffledMarkQueryWrapper
        from torch.utils.data import DataLoader
        perm = make_shuffle_permutation(len(train_dataset), seed=cli.shuffle_seed)
        wrapped = ShuffledMarkQueryWrapper(train_dataset, perm)
        shuffled_train_loader = DataLoader(wrapped, batch_size=cli.batch_size, shuffle=True,
                                          num_workers=int(args.num_workers), drop_last=False,
                                          generator=train_gen)

    memory_x_mark_real = exp.memory_x_mark
    memory_x_mark_shuffled = (shuffled_memory_x_mark(memory_x_mark_real, perm)
                             if cli.time_mode == 'shuffled_time' else None)

    config = {
        'exp': 'TRACK-W-TIMESTAMP-FUSION01', 'cell': cli.cell, 'arm': cli.arm_name,
        'mode': cli.mode, 'num_slots': cli.num_slots if cli.mode == 'slots' else 0,
        'time_mode': cli.time_mode, 'time_proj_dim': cli.time_proj_dim,
        'time_feat_dim': time_feat_dim, 'cadence_info': exp.cadence_info,
        'old_relation_encoder_param_count': old_param_count,
        'timestamp_fusion_encoder_param_count': new_param_count,
        'slot_heads_param_count': sum(p.numel() for p in slot_heads.parameters()) if slot_heads else 0,
        'tau_t': cli.tau_t, 'tau_s': cli.tau_s, 'top_k': cli.top_k, 'batch_size': cli.batch_size,
        'learning_rate': cli.learning_rate, 'train_epochs': cli.train_epochs, 'patience': cli.patience,
        'init_seed': cli.init_seed, 'loader_seed': cli.loader_seed, 'shuffle_seed': cli.shuffle_seed,
        'n_candidates': int(exp.memory_x.size(0)),
        'checkpoint_criterion': 'min validation KL (training objective) -- spec section 8, '
                                'NOT the historical V0-V5 val-retmse10 convention',
        'init_hash': init_hash,
    }
    (out_dir / 'config.json').write_text(json.dumps(config, indent=2, default=str))
    print(f'[track_w_ts] arm={cli.arm_name} cell={cli.cell} mode={cli.mode} time_mode={cli.time_mode} '
         f'N={config["n_candidates"]} K_time={time_feat_dim} init_hash={init_hash[:16]} '
         f'old_params={old_param_count} new_params={new_param_count}')
    print('[MODEL-SELECTION AUDIT]')
    print('Training objective:          KL(p_T || mean_m softmax(s_m/tau_s))')
    print('Validation selection metric:  val_KL (SAME as training objective)')
    print('Early-stopping metric:        val_KL (SAME as selection metric)')
    print('Are they identical?          YES')
    print('Decision:                     APPROVED (TYPE A/B match; no TYPE C diagnostic metric used for selection)')

    def resolve_marks_for_batch(batch_x_mark, split):
        """split in {'train','eval'}. Returns (query_mark, memory_mark)
        to actually feed into the encoder for THIS split, given
        --time_mode."""
        if cli.time_mode == 'shuffled_time' and split == 'train':
            return batch_x_mark, memory_x_mark_shuffled
        return batch_x_mark, memory_x_mark_real

    def compute_scores(batch_x, batch_x_mark, c, split):
        query_mark_real, mem_mark = resolve_marks_for_batch(batch_x_mark, split)
        query_mark = get_time_input(query_mark_real, cli.time_mode, time_feat_dim, batch_x[:, :, c])
        mem_mark_use = get_time_input(mem_mark, cli.time_mode, time_feat_dim, exp.memory_x[:, :, c])
        if cli.mode == 'zero_proj':
            return compute_scores_zero_proj(time_encoder, batch_x, query_mark, exp.memory_x, mem_mark_use, c)
        return compute_scores_slots(time_encoder, slot_heads, batch_x, query_mark, exp.memory_x, mem_mark_use, c)

    def eval_channel(batch_x, batch_y, batch_start_idx, c, cand_mask, batch_x_mark, split, want_diag=False):
        scores = compute_scores(batch_x, batch_x_mark, c, split)
        memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
        query_future = batch_y[:, :, c]
        u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
        d_raw = -u
        oracle_idx = stable_topk_indices(d_raw.masked_fill(~cand_mask, float('inf')), cli.top_k, largest=False)
        res = hard_eval_decomposition(scores, cand_mask, memory_c, offset_c, query_future, d_raw, oracle_idx,
                                      cli.top_k)
        with torch.no_grad():
            p_t = normalized_teacher_prob(-u, cand_mask, cli.tau_t)
            s_masked = scores.masked_fill(~cand_mask.unsqueeze(1), float('-inf'))
            p_m = torch.softmax(s_masked / cli.tau_s, dim=-1)
            p_bar = p_m.mean(dim=1)
            kl = kl_loss_from_prob(p_t, p_bar, cand_mask)
            res['kl'] = kl
            picks = round_robin_topk_selection(scores, cand_mask, k=cli.top_k)
            uniq_counts = torch.tensor([torch.unique(picks[b]).numel() for b in range(picks.size(0))],
                                       dtype=torch.float32)
            dup_rate = 1.0 - (uniq_counts.mean() / cli.top_k)
            invalid_rate = (~cand_mask.gather(1, picks)).float().mean()
            res['duplicate_rate'] = dup_rate
            res['invalid_rate'] = invalid_rate
            teacher_ent = -(p_t.clamp_min(EPS) * p_t.clamp_min(EPS).log()).sum(-1).mean()
            student_ent = -(p_bar.clamp_min(EPS) * p_bar.clamp_min(EPS).log()).sum(-1).mean()
            spread_max = scores.masked_fill(~cand_mask.unsqueeze(1), float('-inf')).amax(dim=-1)
            spread_min = scores.masked_fill(~cand_mask.unsqueeze(1), float('inf')).amin(dim=-1)
            score_spread = (spread_max - spread_min).mean()
            res['teacher_entropy'] = teacher_ent
            res['student_entropy'] = student_ent
            res['score_spread'] = score_spread
        diag = None
        if want_diag:
            spearman = spearman_batch(scores.mean(dim=1), d_raw, cand_mask)
            mech = slot_mechanism_diagnostics(scores, cand_mask, p_m, cli.tau_s)
            diag = {'spearman': spearman, **mech}
        return res, diag

    if cli.smoke_test:
        time_encoder.train()
        if slot_heads is not None:
            slot_heads.train()
        for bi, batch in enumerate(train_loader):
            if bi >= 2:
                break
            batch_x, batch_y, batch_start_idx, batch_x_mark = batch
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            cand_mask, _ = exp._candidate_mask(batch_start_idx)
            optimizer.zero_grad()
            for c in channels:
                scores = compute_scores(batch_x, batch_x_mark, c, 'train')
                s_masked = scores.masked_fill(~cand_mask.unsqueeze(1), float('-inf'))
                p_m = torch.softmax(s_masked / cli.tau_s, dim=-1)
                p_bar = p_m.mean(dim=1)
                memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
                query_future = batch_y[:, :, c]
                with torch.no_grad():
                    u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
                    p_t = normalized_teacher_prob(-u, cand_mask, cli.tau_t)
                l = kl_loss_from_prob(p_t, p_bar, cand_mask)
                (l / len(channels)).backward()
            assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in time_encoder.value_proj.parameters()), \
                '[ISSUE] value_proj grad is zero'
            if cli.time_mode != 'no_time':
                assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in time_encoder.time_proj.parameters()), \
                    '[ISSUE] time_proj grad is zero under a real/shuffled time_mode'
            optimizer.step()
        vram = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
        print(f'[track_w_ts] {cli.arm_name} SMOKE PASS max_vram_mb={vram:.1f}')
        return

    epoch_val_rows, step_rows, batch_order_hashes = [], [], []
    best = {'val_kl': float('inf'), 'epoch': -1}
    t0 = time.time()
    global_step = 0
    active_train_loader = shuffled_train_loader if cli.time_mode == 'shuffled_time' else train_loader
    for epoch in range(1, cli.train_epochs + 1):
        time_encoder.train()
        if slot_heads is not None:
            slot_heads.train()
        starts = []
        for bi, batch in enumerate(active_train_loader):
            if cli.limit_batches and bi >= cli.limit_batches:
                break
            if cli.time_mode == 'shuffled_time':
                batch_x, batch_y, batch_start_idx, batch_x_mark = batch
            else:
                batch_x, batch_y, batch_start_idx, batch_x_mark = batch
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            starts.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx) else torch.as_tensor(batch_start_idx))
            cand_mask, _ = exp._candidate_mask(batch_start_idx)
            optimizer.zero_grad()
            batch_loss = 0.0
            for c in channels:
                scores = compute_scores(batch_x, batch_x_mark, c, 'train')
                s_masked = scores.masked_fill(~cand_mask.unsqueeze(1), float('-inf'))
                p_m = torch.softmax(s_masked / cli.tau_s, dim=-1)
                p_bar = p_m.mean(dim=1)
                memory_c, offset_c = memory_value(args, batch_x, exp.memory_y, exp.memory_x_last, c)
                query_future = batch_y[:, :, c]
                with torch.no_grad():
                    u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
                    p_t = normalized_teacher_prob(-u, cand_mask, cli.tau_t)
                l = kl_loss_from_prob(p_t, p_bar, cand_mask)
                (l / len(channels)).backward()
                batch_loss += float(l.detach()) / len(channels)
            optimizer.step()
            global_step += 1
            step_rows.append({'global_step': global_step, 'epoch': epoch, 'batch_in_epoch': bi,
                              'train_loss_kl': batch_loss})
        batch_order_hashes.append({'epoch': epoch, 'batch_order_sha256': batch_order_sha256(starts)})

        time_encoder.eval()
        if slot_heads is not None:
            slot_heads.eval()
        val_sums, n = {}, 0
        with torch.no_grad():
            for batch in val_loader:
                batch_x, batch_y, batch_start_idx, batch_x_mark = batch
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                cand_mask, _ = exp._candidate_mask(batch_start_idx)
                bsz = batch_x.size(0)
                per_ch = {}
                for c in channels:
                    res, _ = eval_channel(batch_x, batch_y, batch_start_idx, c, cand_mask, batch_x_mark,
                                          'eval', want_diag=False)
                    for key_ in ('kl', 'model_ind_mse', 'agg_mse', 'recall10', 'ndcg10',
                               'duplicate_rate', 'invalid_rate', 'teacher_entropy', 'student_entropy',
                               'score_spread'):
                        per_ch.setdefault(key_, []).append(res[key_].reshape(-1).cpu() if torch.is_tensor(res[key_]) else torch.tensor([float(res[key_])]))
                for k_, vals in per_ch.items():
                    val_sums[k_] = val_sums.get(k_, 0.0) + torch.cat(vals).sum().item()
                n += bsz
        n_eff = max(n * len(channels), 1)
        val_metrics = {k_: v / n_eff for k_, v in val_sums.items()}
        val_kl = val_metrics['kl']
        val_metrics['epoch'] = epoch
        epoch_val_rows.append(val_metrics)

        payload = {'time_encoder_state_dict': time_encoder.state_dict(),
                  'slot_heads_state_dict': slot_heads.state_dict() if slot_heads else None,
                  'epoch': epoch, 'val_kl': val_kl, 'config': config}
        torch.save(payload, ckpt_dir / f'checkpoint_epoch{epoch}.pth')
        if val_kl < best['val_kl']:
            best = {'val_kl': val_kl, 'epoch': epoch}
            torch.save(payload, ckpt_dir / 'checkpoint.pth')
        print(f'[track_w_ts] {cli.arm_name} epoch={epoch} val_kl={val_kl:.6f} '
             f'val_retmse10={val_metrics["model_ind_mse"]:.6f} val_agg={val_metrics["agg_mse"]:.6f} '
             f'(best={best["epoch"]}:{best["val_kl"]:.6f})')
        if epoch - best['epoch'] >= cli.patience:
            print(f'[track_w_ts] {cli.arm_name} early stop at epoch {epoch} (best={best["epoch"]})')
            break

    bl = torch.load(ckpt_dir / 'checkpoint.pth', map_location=device)
    time_encoder.load_state_dict(bl['time_encoder_state_dict'])
    if slot_heads is not None:
        slot_heads.load_state_dict(bl['slot_heads_state_dict'])
    time_encoder.eval()
    if slot_heads is not None:
        slot_heads.eval()

    test_sums, n = {}, 0
    diag_accum = {}
    per_view_retmse = {}
    with torch.no_grad():
        for batch in test_loader:
            batch_x, batch_y, batch_start_idx, batch_x_mark = batch
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            cand_mask, _ = exp._candidate_mask(batch_start_idx)
            bsz = batch_x.size(0)
            per_ch = {}
            for c in channels:
                res, diag = eval_channel(batch_x, batch_y, batch_start_idx, c, cand_mask, batch_x_mark,
                                         'eval', want_diag=True)
                for key_ in ('kl', 'model_ind_mse', 'agg_mse', 'recall10', 'ndcg10', 'D', 'C',
                           'duplicate_rate', 'invalid_rate', 'teacher_entropy', 'student_entropy',
                           'score_spread'):
                    v = res[key_]
                    per_ch.setdefault(key_, []).append(v.reshape(-1).cpu() if torch.is_tensor(v) else torch.tensor([float(v)]))
                for k_, v in diag.items():
                    diag_accum.setdefault(k_, []).append(v)
            for k_, vals in per_ch.items():
                test_sums[k_] = test_sums.get(k_, 0.0) + torch.cat(vals).sum().item()
            n += bsz
    n_eff = max(n * len(channels), 1)
    test_metrics = {k_: v / n_eff for k_, v in test_sums.items()}
    test_metrics['n_queries_seen'] = n
    test_metrics['best_epoch'] = best['epoch']
    slot_mech_metrics = {k_: (sum(v) / len(v) if v else float('nan')) for k_, v in diag_accum.items()}

    # timestamp branch diagnostics (spec section 9)
    with torch.no_grad():
        p_x_norm = time_encoder.value_proj.weight.norm().item()
        p_t_norm = time_encoder.time_proj.weight.norm().item()
    # one extra backward pass purely to read gradient norms post-hoc (does
    # not affect the already-saved/selected checkpoint's weights)
    time_encoder.zero_grad()
    sample_batch = next(iter(test_loader))
    sx, sy, sstart, smark = sample_batch
    sx = sx.float().to(device); sy = sy.float().to(device); smark = smark.float().to(device)
    cand_mask, _ = exp._candidate_mask(sstart)
    c0 = channels[0]
    scores = compute_scores(sx, smark, c0, 'eval')
    s_masked = scores.masked_fill(~cand_mask.unsqueeze(1), float('-inf'))
    p_bar = torch.softmax(s_masked / cli.tau_s, dim=-1).mean(dim=1)
    memory_c, offset_c = memory_value(args, sx, exp.memory_y, exp.memory_x_last, c0)
    query_future = sy[:, :, c0]
    u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
    p_t_teacher = normalized_teacher_prob(-u, cand_mask, cli.tau_t)
    kl_sample = kl_loss_from_prob(p_t_teacher, p_bar, cand_mask)
    kl_sample.backward()
    value_proj_grad_norm = time_encoder.value_proj.weight.grad.norm().item() if time_encoder.value_proj.weight.grad is not None else 0.0
    time_proj_grad_norm = time_encoder.time_proj.weight.grad.norm().item() if time_encoder.time_proj.weight.grad is not None else 0.0
    with torch.no_grad():
        delta = sx[:, :, c0] - sx[:, -1:, c0].detach()
        query_mark_for_diag = get_time_input(smark, cli.time_mode, time_feat_dim, delta)
        value_out_norm = time_encoder.value_proj(delta.unsqueeze(-1)).norm(dim=-1).mean().item()
        time_out_norm = time_encoder.time_proj(query_mark_for_diag).norm(dim=-1).mean().item()
    timestamp_diag = {
        'value_proj_weight_norm': p_x_norm, 'time_proj_weight_norm': p_t_norm,
        'value_proj_grad_norm_sample': value_proj_grad_norm,
        'time_proj_grad_norm_sample': time_proj_grad_norm,
        'value_branch_output_norm_mean': value_out_norm, 'time_branch_output_norm_mean': time_out_norm,
        'time_over_value_output_ratio': (time_out_norm / value_out_norm) if value_out_norm > 0 else float('nan'),
    }
    optimizer.zero_grad()

    with open(out_dir / 'train_metrics.csv', 'w', newline='') as fh:
        fieldnames = sorted({k for r in step_rows for k in r})
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in step_rows:
            w.writerow(r)
    with open(out_dir / 'val_metrics.csv', 'w', newline='') as fh:
        fieldnames = sorted({k for r in epoch_val_rows for k in r})
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in epoch_val_rows:
            w.writerow(r)
    with open(out_dir / 'batch_order_hashes.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['epoch', 'batch_order_sha256'])
        w.writeheader()
        for r in batch_order_hashes:
            w.writerow(r)
    (out_dir / 'final_test_metrics.json').write_text(json.dumps(test_metrics, indent=2, default=str))
    (out_dir / 'slot_mechanism_metrics.json').write_text(json.dumps(slot_mech_metrics, indent=2, default=str))
    (out_dir / 'timestamp_diagnostics.json').write_text(json.dumps(timestamp_diag, indent=2, default=str))
    (out_dir / 'checkpoint_fingerprints.json').write_text(json.dumps(
        {'init_hash': init_hash, 'best_epoch': best['epoch'], 'final_model_hash': state_hash_module(time_encoder),
         'batch_order_hashes': batch_order_hashes}, indent=2))
    vram = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == 'cuda' else 0.0
    print(f'[track_w_ts] done. {cli.arm_name} best_epoch={best["epoch"]} best_val_kl={best["val_kl"]:.6f} '
         f'test_KL={test_metrics["kl"]:.6f} test_retMSE@10={test_metrics["model_ind_mse"]:.6f} '
         f'test_agg_mse10={test_metrics["agg_mse"]:.6f} max_vram_mb={vram:.1f} wall_seconds={time.time()-t0:.1f}')


if __name__ == '__main__':
    main()
