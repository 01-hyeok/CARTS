#!/usr/bin/env python3
"""TRACK-A-PATCH-RETRIEVAL-EXPERT01 -- Phase A: single past-input patch size.

Separate experiment from TRACK-A-HORIZON-RETRIEVAL-EXPERT01. That experiment
varies which FUTURE block (0:96 / 96:336 / 336:720) is retrieved for; this
experiment varies the PAST input's patch granularity (patch_len in
{16-native, 24, 48, 120}) used by the scratch retrieval encoder, with a
single full-horizon (0:720) relevance target throughout. No future-block
teacher is used here. Does not read or write any
TRACK-A-HORIZON-RETRIEVAL-EXPERT01 file/checkpoint.

Architecture reuse: `models.RelationStage1.RelationEncoder` already IS
"patch embedding -> Transformer encoder -> pooling -> LayerNorm -> MLP
projection" with `patch_len`/`stride` as plain config fields
(`layers/relation_patch_embed.py`). No new model class is introduced --
`scripts.train_margutil01.build_experiment` overrides `patch_len`/`stride`
(non-overlap: stride==patch_len) on top of a reference checkpoint's OTHER
args (d_model, n_heads, e_layers, enc_in, relation_value_space=delta_last,
...), producing a freshly-initialized (never weight-loaded) scratch encoder
at the requested patch granularity. Depth/width/optimizer/loss/candidate set
are therefore identical across patch-size arms by construction -- the only
architectural difference is `RelationPatchEmbedding`'s patch/stride/
num_patches/position-embedding size.

Candidate future reconstruction: `scripts.train_margutil01.memory_value`
(delta_last), unmodified -- same convention as every other Track-A
experiment. Candidate validity: `exp._candidate_mask` (time-ordered, no
leakage), unmodified. Full-memory search only -- no shortlist, no reranker:
`arm_score`/`encode_raw`/`individual_utility_memsafe` from
`scripts.train_factorial_e2e01`, unmodified.

Loss: forward KL, single full-horizon (0:720) individual-candidate-MSE
teacher (no future-block split). `normalized_teacher_prob`/`kl_loss` reused
byte-for-byte from `scripts.train_horizon_retrieval_expert01` -- those two
functions never reference blocks; they take a generic per-row distance `d`
and a validity mask. Gradient-equivalence to soft cross-entropy is therefore
already covered by that module's own test
(`test_kl_gradient_equals_soft_ce_gradient`); not re-derived here.

Checkpoint-selection / primary metric (spec section 3, distinct from
TRACK-A-HORIZON-RETRIEVAL-EXPERT01's uniform-aggregate MSE): mean INDIVIDUAL
future MSE of the model's own hard Top-10 picks, and its regret against the
full-memory individual-Oracle Top-10's mean individual MSE on the same
queries. Both are future-blind at selection time (query_future is used only
to *score* the already-selected Top-10 and the Oracle's own Top-10, never to
pick them) and computed on val for checkpointing, on train/val/test for
reporting. Secondary: Oracle Recall@10, NDCG@10 (vectorized, no per-row
Python loop), uniform-aggregate H720 MSE (for downstream Stage-2 relevance).

Candidate-side gradient: `E = encode_raw(model, exp.memory_x, c)` is
recomputed (not cached/detached) every batch, exactly as in
`train_horizon_retrieval_expert01.py` -- so encoder gradients from the
candidate side are never silently dropped. Verified by
`tests/test_patch_retrieval_expert01.py::test_candidate_side_gradient_flows`.
"""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import stable_topk_indices
from scripts.diag_horizon_retrieval_headroom01 import _gather_mean, _mse
from scripts.rng_control01 import batch_order_sha256, make_loader_generator, set_global_seeds
from scripts.train_factorial_e2e01 import arm_score, encode_raw, individual_utility_memsafe
from scripts.train_horizon_retrieval_expert01 import kl_loss, normalized_teacher_prob, teacher_diagnostics
from scripts.train_margutil01 import build_experiment, memory_value

EPS = 1e-8


def recall_at_k(model_idx, oracle_idx, k):
    """Fraction of the Oracle's own Top-k set contained in the model's
    Top-k set (vectorized, per row)."""
    hit = (model_idx.unsqueeze(-1) == oracle_idx.unsqueeze(-2)).any(-1)
    return hit.float().sum(-1) / k


def ndcg_at_k(model_idx, d, valid_mask, k):
    """NDCG@k with relevance = shifted (rel>=0) negative distance, ranked by
    the model's own Top-k order (already score-sorted by `stable_topk_indices`
    so no re-sort needed here). Fully vectorized (no Python per-row loop)."""
    tgt = (-d).masked_fill(~valid_mask, float('-inf'))
    rel = tgt - tgt.masked_fill(~valid_mask, float('inf')).min(dim=-1, keepdim=True).values
    rel = rel.masked_fill(~valid_mask, 0.0)
    gains = rel.gather(1, model_idx)
    disc = 1.0 / torch.log2(torch.arange(2, k + 2, device=d.device).float()).unsqueeze(0)
    dcg = (gains * disc).sum(-1)
    ideal = rel.topk(k, dim=-1).values
    idcg = (ideal * disc).sum(-1).clamp_min(1e-12)
    return dcg / idcg


def _file_sha256(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def _git_commit():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception as e:
        return f'UNKNOWN ({e})'


def train_epoch(exp, args, model, cli, loader, channels, device,
                record_batch_order=False, limit_batches=0):
    model.train(True)
    tot_loss, nb = 0.0, 0
    diag_sums, diag_n = {}, 0
    batch_starts = []
    memory_y, memory_x_last = exp.memory_y, exp.memory_x_last

    for bi, (batch_x, batch_y, batch_start_idx) in enumerate(loader):
        if limit_batches and bi >= limit_batches:
            break
        if record_batch_order:
            batch_starts.append(batch_start_idx.clone() if torch.is_tensor(batch_start_idx)
                                else torch.as_tensor(batch_start_idx))
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        cli.optimizer.zero_grad()
        batch_loss = 0.0

        for c in channels:
            z_q = encode_raw(model, batch_x, c)
            E = encode_raw(model, exp.memory_x, c)  # NOT detached/cached -- candidate-side gradient flows
            memory_c, offset_c = memory_value(args, batch_x, memory_y, memory_x_last, c)
            query_future = batch_y[:, :, c]

            s = arm_score(z_q, E, None)
            u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
            d = -u
            p_t = normalized_teacher_prob(d, cand_mask, cli.tau_t)
            ch_loss = kl_loss(p_t, s, cand_mask, cli.tau_s)

            diag_sums['kl'] = diag_sums.get('kl', 0.0) + float(ch_loss.detach())
            with torch.no_grad():
                diag = teacher_diagnostics(p_t, cand_mask)
                for k, v in diag.items():
                    diag_sums[k] = diag_sums.get(k, 0.0) + v

            if getattr(cli, 'channelwise_backward', False):
                (ch_loss / len(channels)).backward()
                batch_loss = batch_loss + float(ch_loss.detach()) / len(channels)
            else:
                batch_loss = batch_loss + ch_loss
            diag_n += 1

        if getattr(cli, 'channelwise_backward', False):
            batch_loss_final = batch_loss
        else:
            batch_loss = batch_loss / len(channels)
            batch_loss.backward()
            batch_loss_final = float(batch_loss.detach())
        cli.optimizer.step()
        tot_loss += batch_loss_final
        nb += 1

    out = {'train_loss': tot_loss / max(nb, 1)}
    out.update({f'train_{k}': v / max(diag_n, 1) for k, v in diag_sums.items()})
    if record_batch_order:
        out['batch_order_sha256'] = batch_order_sha256(batch_starts)
    return out


@torch.no_grad()
def eval_epoch(exp, args, model, cli, loader, channels, device, top_k=10, limit_batches=0):
    """Future-blind hard Top-K selection (query_future never used to pick),
    then scored against the same query_future. Primary/checkpoint metric:
    `model_top10_individual_mse` and `oracle_regret` (== model - oracle,
    oracle recomputed identically every call so this is well-defined per
    split). Secondary: recall@10, ndcg@10, uniform-aggregate H720 MSE."""
    model.train(False)
    memory_y, memory_x_last = exp.memory_y, exp.memory_x_last
    sums = {}
    n = 0
    for bi, (batch_x, batch_y, batch_start_idx) in enumerate(loader):
        if limit_batches and bi >= limit_batches:
            break
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, _ = exp._candidate_mask(batch_start_idx)
        bsz = batch_x.size(0)
        per_ch = {}
        for c in channels:
            z_q = encode_raw(model, batch_x, c)
            E = encode_raw(model, exp.memory_x, c)
            memory_c, offset_c = memory_value(args, batch_x, memory_y, memory_x_last, c)
            query_future = batch_y[:, :, c]
            neg_inf = float('-inf')

            u = individual_utility_memsafe(memory_c, offset_c, query_future, cli.chunk_size)
            d = -u
            oracle_idx = stable_topk_indices(d.masked_fill(~cand_mask, float('inf')), top_k, largest=False)
            oracle_ind_mse = d.gather(1, oracle_idx).mean(-1)

            s = arm_score(z_q, E, None).masked_fill(~cand_mask, neg_inf)
            model_idx = stable_topk_indices(s, top_k, largest=True)
            model_ind_mse = d.gather(1, model_idx).mean(-1)

            per_ch.setdefault('model_top10_individual_mse', []).append(model_ind_mse.cpu())
            per_ch.setdefault('oracle_top10_individual_mse', []).append(oracle_ind_mse.cpu())
            per_ch.setdefault('oracle_regret', []).append((model_ind_mse - oracle_ind_mse).cpu())
            per_ch.setdefault('recall_at_10', []).append(recall_at_k(model_idx, oracle_idx, top_k).cpu())
            per_ch.setdefault('ndcg_at_10', []).append(ndcg_at_k(model_idx, d, cand_mask, top_k).cpu())

            yhat = _gather_mean(memory_c, offset_c, model_idx, 0, 720)
            per_ch.setdefault('global_h720_mse', []).append(_mse(yhat, query_future).cpu())
        for key, vals in per_ch.items():
            sums[key] = sums.get(key, 0.0) + torch.cat(vals).sum().item()
        n += bsz

    n_channels = max(len(channels), 1)
    out = {key: sums[key] / max(n * n_channels, 1) for key in sums}
    out['n_queries_seen'] = n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--arm_name', required=True,
                    help="e.g. 'native_p16', 'p24', 'p48', 'p120' -- free label, "
                         "output path only, does not affect training.")
    ap.add_argument('--cell', required=True)
    ap.add_argument('--patch_len', type=int, required=True)
    ap.add_argument('--stride', type=int, default=None,
                    help='defaults to patch_len (non-overlap)')
    ap.add_argument('--pred_len', type=int, default=720)
    ap.add_argument('--seq_len', type=int, default=720)
    ap.add_argument('--checkpoints', default='checkpoints/track_a_patch_retrieval_expert01')
    ap.add_argument('--out_dir', default='results/TRACK-A-PATCH-RETRIEVAL-EXPERT01')
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--train_epochs', type=int, default=10)
    ap.add_argument('--patience', type=int, default=5)
    ap.add_argument('--learning_rate', type=float, default=1e-3)
    ap.add_argument('--weight_decay', type=float, default=0.0)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--init_seed', type=int, default=0)
    ap.add_argument('--loader_seed', type=int, default=0)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--tau_t', type=float, required=True,
                    help='dataset-calibrated teacher temperature (section 2) -- required, no default')
    ap.add_argument('--tau_s', type=float, default=0.1)
    ap.add_argument('--limit_batches', type=int, default=0, help='SMOKE ONLY')
    ap.add_argument('--channelwise_backward', action='store_true')
    cli = ap.parse_args()
    cli.stride = cli.stride or cli.patch_len

    if cli.seq_len % cli.patch_len != 0:
        raise SystemExit(f'[ISSUE][ABORT] seq_len={cli.seq_len} must be evenly divisible by '
                         f'patch_len={cli.patch_len} (non-overlap patches, spec section 3)')
    if cli.pred_len != 720 or cli.seq_len != 720:
        raise SystemExit('[ISSUE][ABORT] this experiment is defined only for seq_len=pred_len=720')

    set_global_seeds(cli.init_seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cell_dir = Path(cli.out_dir) / cli.cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(cli.checkpoints) / cli.cell / cli.arm_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # [ISSUE][CORRECTED] The S0_wce reference checkpoints this whole Track-A
    # session reuses actually have relation_encoder_type='mlp' (a flat MLP
    # over the raw 720-length input; relation_self_fill='linear'), NOT the
    # patch-Transformer branch -- discovered via a smoke test here (patch_len
    # 16 vs 24 produced byte-identical loss/metrics because the override had
    # no effect at all under 'mlp'). This experiment is specifically about
    # patch granularity, so it explicitly forces relation_encoder_type=
    # 'transformer' (which DOES read patch_len/stride via
    # RelationPatchEmbedding) and relation_self_fill='zero' (transformer
    # forbids 'linear'; 'zero' is a fixed, identical-across-arms constant,
    # not a free variable). Verified: with this override, patch_len in
    # {16,24,48,120} now produces distinct RelationPatchEmbedding.num_patches
    # (45/30/15/6) and distinct parameter counts, confirmed by direct
    # inspection before any real run was launched.
    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.seq_len, 'batch_size': cli.batch_size,
        'seed': cli.init_seed, 'top_k': cli.top_k, 'tau_topk': 0.1,
        'patch_len': cli.patch_len, 'stride': cli.stride,
        'relation_encoder_type': 'transformer', 'relation_self_fill': 'zero',
    })
    exp._ensure_memory()
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.to(device)
    channels = list(range(int(args.enc_in)))
    num_patches = (cli.seq_len - cli.patch_len) // cli.stride + 1

    code_commit = _git_commit()
    encoder_init_sha256 = hashlib.sha256(str(model.state_dict()).encode()).hexdigest()

    for p in model.parameters():
        p.requires_grad_(True)
    param_count = sum(p.numel() for p in model.parameters())
    cli.optimizer = torch.optim.Adam(model.parameters(), lr=cli.learning_rate,
                                     weight_decay=cli.weight_decay)

    train_gen = make_loader_generator(cli.loader_seed)
    _, train_loader = exp._get_data(flag='train', shuffle=True, generator=train_gen)
    _, val_loader = exp._get_data(flag='val', shuffle=False)
    _, test_loader = exp._get_data(flag='test', shuffle=False)

    fingerprint = {
        'exp': 'TRACK-A-PATCH-RETRIEVAL-EXPERT01', 'cell': cli.cell, 'arm_name': cli.arm_name,
        'relation_encoder_type_corrected_from_host': 'mlp -> transformer (host default does not '
        'read patch_len at all; see script docstring)',
        'relation_self_fill': 'zero', 'patch_len': cli.patch_len, 'stride': cli.stride,
        'num_patches': num_patches,
        'param_count': param_count, 'top_k': cli.top_k, 'tau_t': cli.tau_t, 'tau_s': cli.tau_s,
        'loss': 'KL(p_T||p_S), full-horizon [0:720] individual-candidate-MSE teacher, no block split',
        'init_seed': cli.init_seed, 'loader_seed': cli.loader_seed, 'channels': channels,
        'encoder_init_sha256': encoder_init_sha256, 'learning_rate': cli.learning_rate,
        'batch_size': cli.batch_size, 'epochs': cli.train_epochs, 'patience': cli.patience,
        'checkpoint_criterion': 'min val model_top10_individual_mse (== min val oracle_regret)',
        'code_commit': code_commit,
    }
    (cell_dir / f'config_fingerprint_{cli.arm_name}.json').write_text(json.dumps(fingerprint, indent=2))
    print(f'[patch_retrieval_expert01] {cli.cell}/{cli.arm_name} patch_len={cli.patch_len} '
         f'stride={cli.stride} num_patches={num_patches} params={param_count} '
         f'encoder_init_sha={encoder_init_sha256[:16]} init_seed={cli.init_seed} '
         f'loader_seed={cli.loader_seed}')

    epoch_rows, best = [], {'val': float('inf'), 'epoch': -1}
    batch_order_hashes = {}
    t0 = time.time()
    for epoch in range(1, cli.train_epochs + 1):
        ep_t0 = time.time()
        tr = train_epoch(exp, args, model, cli, train_loader, channels, device,
                         record_batch_order=True, limit_batches=cli.limit_batches)
        batch_order_hashes[f'epoch{epoch}'] = tr.pop('batch_order_sha256')
        va = eval_epoch(exp, args, model, cli, val_loader, channels, device,
                        top_k=cli.top_k, limit_batches=cli.limit_batches)
        val_metric = va['model_top10_individual_mse']
        row = {'epoch': epoch, **tr, **{f'val_{k}': v for k, v in va.items()},
              'epoch_wall_seconds': time.time() - ep_t0}
        epoch_rows.append(row)
        payload = {'model_state_dict': model.state_dict(), 'args': vars(args),
                  'epoch': epoch, 'fingerprint': fingerprint, 'val_primary_mse': val_metric}
        torch.save(payload, ckpt_dir / f'checkpoint_epoch{epoch}.pth')
        if val_metric < best['val']:
            best = {'val': val_metric, 'epoch': epoch}
            torch.save(payload, ckpt_dir / 'checkpoint.pth')
        print(f"[patch_retrieval_expert01] {cli.cell}/{cli.arm_name} epoch {epoch} "
             f"train_loss={tr['train_loss']:.5f} val_model_top10_ind_mse={val_metric:.6f} "
             f"val_oracle_regret={va['oracle_regret']:.6f} val_recall@10={va['recall_at_10']:.4f} "
             f"val_ndcg@10={va['ndcg_at_10']:.4f} "
             f"batch_order_sha256={batch_order_hashes[f'epoch{epoch}'][:16]}")
        if epoch - best['epoch'] >= cli.patience:
            print(f'[patch_retrieval_expert01] early stop at epoch {epoch} (best={best["epoch"]})')
            break

    bl = torch.load(ckpt_dir / 'checkpoint.pth', map_location=device)
    model.load_state_dict(bl['model_state_dict'])
    te = eval_epoch(exp, args, model, cli, test_loader, channels, device, top_k=cli.top_k)
    tr_full = eval_epoch(exp, args, model, cli, train_loader, channels, device, top_k=cli.top_k,
                         limit_batches=cli.limit_batches)

    with open(cell_dir / f'epoch_metrics_{cli.arm_name}.csv', 'w', newline='') as fh:
        keys = sorted({k for r in epoch_rows for k in r})
        w = csv.DictWriter(fh, fieldnames=['epoch'] + [k for k in keys if k != 'epoch'])
        w.writeheader()
        for r in epoch_rows:
            w.writerow(r)
    (cell_dir / f'batch_order_hashes_{cli.arm_name}.json').write_text(json.dumps(batch_order_hashes, indent=2))

    summary = {
        'exp': 'TRACK-A-PATCH-RETRIEVAL-EXPERT01', 'cell': cli.cell, 'arm_name': cli.arm_name,
        'best_epoch': best['epoch'], 'best_val_model_top10_individual_mse': best['val'],
        'test_metrics': te, 'train_metrics_subset': tr_full,
        'encoder_init_sha256': encoder_init_sha256, 'batch_order_hashes': batch_order_hashes,
        'wall_clock_seconds': time.time() - t0, 'checkpoint': str(ckpt_dir / 'checkpoint.pth'),
        'fingerprint': fingerprint,
    }
    (cell_dir / f'retrieval_metrics_{cli.arm_name}.json').write_text(json.dumps(summary, indent=2))
    (cell_dir / f'DONE_{cli.arm_name}.marker').write_text(json.dumps({'done': True, 'best_epoch': best['epoch']}))
    print(f"[patch_retrieval_expert01] done. {cli.cell}/{cli.arm_name} best_epoch={best['epoch']} "
         f"test_model_top10_individual_mse={te['model_top10_individual_mse']:.6f} "
         f"test_oracle_regret={te['oracle_regret']:.6f}")


if __name__ == '__main__':
    main()
