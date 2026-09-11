#!/usr/bin/env python3
"""TRACK-A-FACTORIAL-E2E01 -- Stage-2 forecasting evaluation.

For one arm: take its BEST Stage-1 checkpoint, run FREE-RUNNING selection on
the test split (no oracle/future information anywhere on the selection path),
inject the resulting Top-K into the UNCHANGED Stage-2 host via
`RelationStage2.set_forced_selection`, and read two tensors out of the SAME
forward pass:

    y_final  -> "Retrieval-Augmented Frozen-Host Stage-2"
    y_base   -> "Frozen Host Retrieval-Ablated Branch"  (DIAGNOSTIC ONLY)

TERMINOLOGY, and why it matters
-------------------------------
`y_base` is the host's own branch taken before retrieval fusion. It is NOT a
forecaster and must never be called "Base Forecaster Only", "baseline" or
"lower bound": the host was trained as `y_final = y_base + lambda*y_ret` with
lambda ~ 0.51, so that branch only works with the retrieval term added. On
ETTh1 H96 it measures 0.6467 while the host's own `y_final` is 0.3731. It is
recorded here purely as a diagnostic -- "what does the host do when retrieval
is removed from the same forward" -- and its bit-identity across arms is still
asserted, because that is what proves the comparison is controlled.

THE forecasting baseline of this experiment is the
"Independent Base-only Forecaster" trained by
`scripts/train_factorial_e2e01_base_only.py` with retrieval, fusion and gate
entirely absent. The primary metric is

    delta_mse_vs_independent_base = retrieval_augmented_mse
                                    - independent_base_only_mse

This script also records "Frozen Host Original Retrieval" -- the host's own
UNFORCED `y_final` on the same batches -- as a third diagnostic reference.

Scope limit (spec S5): a result here answers "can this new retrieval, injected
into the EXISTING frozen S0_wce host, already beat an independent Base-only
forecaster?". It is not the final forecasting performance of a Stage-2 trained
for this retrieval, in either direction.

Fairness is not assumed: `--assert_base_against` compares this run's
`y_base` tensor bit-for-bit against a reference arm's stored copy and aborts
on any difference.

Host verification: all four cells' hosts use `relation_top_n=1` (self-only),
so forcing slot `(c, 0)` controls 100% of the retrieval path. The script
asserts this rather than trusting it.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from exp.exp_stage1_relation import Exp_Stage1_Relation
from layers.retrieval_metric import RetrievalMetric
from models.SequentialSetRetriever import SetConditioner
from scripts.train_factorial_e2e01 import (arm_score, encode_raw,
                                           free_running_aggregate_future_mse, run_sequence)
from scripts.train_margutil01 import memory_value
from utils.retrieval_diagnostics import load_stage2


def load_arm(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    args = SimpleNamespace(**ckpt['args'])
    exp = Exp_Stage1_Relation(args)
    model = exp.model.module if hasattr(exp.model, 'module') else exp.model
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval().to(device)
    d_model = int(args.d_model)
    sc = SetConditioner(d_model).to(device)
    sc.load_state_dict(ckpt['set_conditioner_state_dict'])
    sc.eval()
    metric = None
    if ckpt.get('metric_state_dict') is not None:
        metric = RetrievalMetric(kind='asymmetric', dim=d_model, output='cosine',
                                 layer_norm=False).to(device)
        metric.load_state_dict(ckpt['metric_state_dict'])
        metric.eval()
    return exp, args, model, sc, metric, ckpt


@torch.no_grad()
def evaluate(arm_ckpt, stage2_host, top_k, chunk_size, device, base_ref=None,
             limit_batches=0, independent_base=None):
    s2_exp, s2_args = load_stage2(stage2_host)
    s2_exp._ensure_memory()
    s2_exp._build_key_bank()
    s2_model = s2_exp.model.module if hasattr(s2_exp.model, 'module') else s2_exp.model
    s2_model.eval()

    if int(getattr(s2_args, 'relation_top_n', 1)) != 1:
        raise SystemExit('[ABORT] host relation_top_n != 1: forcing slot (c,0) would '
                         'leave other source slots retrieving on their own, which is a '
                         'confound. Refusing to evaluate.')

    a_exp, a_args, model, set_cond, metric, ckpt = load_arm(arm_ckpt, device)
    a_exp._ensure_memory()
    fp = ckpt.get('fingerprint', {})
    target = fp.get('axis_oracle')
    prefix_policy = fp.get('axis_prefix')

    if int(a_args.pred_len) != int(s2_args.pred_len) or a_args.data != s2_args.data:
        raise SystemExit('[ABORT] arm checkpoint and Stage-2 host disagree on dataset/pred_len')
    if a_exp.memory_x.shape != torch.from_numpy(s2_exp.memory_bank.memory_x).shape:
        raise SystemExit(f'[ABORT] candidate pool differs: Stage-1 {tuple(a_exp.memory_x.shape)} '
                         f'vs Stage-2 {tuple(s2_exp.memory_bank.memory_x.shape)}')

    tau_topk = float(s2_args.tau_topk)
    score_fn = s2_model._retrieval_score_fn()
    memory_y = s2_exp.memory_y.to(device)
    memory_x_last = s2_exp.memory_x_last.to(device)
    channels = list(s2_model.target_channels())
    _, loader = s2_exp._get_data(flag='test', shuffle=False)

    tot = {'ret_se': 0.0, 'ret_ae': 0.0, 'base_se': 0.0, 'base_ae': 0.0,
           'host_se': 0.0, 'host_ae': 0.0, 'n': 0.0}
    fr_sum, fr_n = 0.0, 0
    dup, invalid, rows = 0, 0, 0
    base_chunks = []

    for bi, (batch_x, batch_y, batch_start_idx) in enumerate(loader):
        if limit_batches and bi >= limit_batches:
            break   # SMOKE ONLY
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, counts = s2_exp._candidate_mask(batch_start_idx)
        valid_query = counts.to(device) >= top_k

        forced = {}
        for c in channels:
            E = encode_raw(model, a_exp.memory_x, c)
            z_q = encode_raw(model, batch_x, c)
            memory_c, offset_c = memory_value(a_args, batch_x, memory_y, memory_x_last, c)
            futures = memory_c + offset_c.view(-1, 1, 1)
            q_future = batch_y[:, :, c]

            # --- FREE-RUNNING selection: no oracle, no future, no host score ---
            _, _, picks, _ = run_sequence(
                z_q, E, cand_mask, set_cond, metric, None,
                futures, q_future, target, prefix_policy,
                tau_topk, top_k, chunk_size, free_running=True)
            forced[(int(c), 0)] = picks

            z_hq = s2_model._branch_embedding(batch_x, c, c)
            z_hm = s2_model._branch_memory(s2_exp.key_bank, c, 0, c, z_hq.dtype, device)
            host_scores = (score_fn(z_hq, z_hm) if score_fn is not None
                           else torch.matmul(z_hq, z_hm.transpose(0, 1)))
            host_scores = host_scores.masked_fill(~cand_mask,
                                                  torch.finfo(host_scores.dtype).min / 4)
            fr = free_running_aggregate_future_mse(picks, host_scores, futures,
                                                   q_future, tau_topk)
            fr_sum += float(fr[valid_query].sum())
            fr_n += int(valid_query.sum())

            for b in range(picks.size(0)):
                if not valid_query[b]:
                    continue
                p = picks[b].tolist()
                rows += 1
                if len(set(p)) != len(p):
                    dup += 1
                if not bool(cand_mask[b, p].all()):
                    invalid += 1

        s2_model.set_forced_selection(forced)
        y_final, y_base, y_ret, beta, lam, debug = s2_exp.model(
            batch_x=batch_x, memory_y=s2_exp.memory_y, valid_mask=cand_mask,
            key_bank=s2_exp.key_bank, memory_x_last=s2_exp.memory_x_last,
            retrieval_cache=None, target_y=batch_y,
            teacher_key_bank=getattr(s2_exp, 'teacher_key_bank', None),
        )
        s2_model.set_forced_selection(None)

        yf, yb, by = y_final[valid_query], y_base[valid_query], batch_y[valid_query]
        tot['ret_se'] += float((yf - by).pow(2).sum())
        tot['ret_ae'] += float((yf - by).abs().sum())
        tot['base_se'] += float((yb - by).pow(2).sum())
        tot['base_ae'] += float((yb - by).abs().sum())
        tot['n'] += float(yf.numel())
        base_chunks.append(y_base.detach().cpu())

        # --- Frozen Host Original Retrieval: the host's OWN unforced Top-K,
        # same batches, same metric. Diagnostic reference only. ---
        hf, _, _, _, _, _ = s2_exp.model(
            batch_x=batch_x, memory_y=s2_exp.memory_y, valid_mask=cand_mask,
            key_bank=s2_exp.key_bank, memory_x_last=s2_exp.memory_x_last,
            retrieval_cache=None, target_y=batch_y,
            teacher_key_bank=getattr(s2_exp, 'teacher_key_bank', None),
        )
        hv = hf[valid_query]
        tot['host_se'] += float((hv - by).pow(2).sum())
        tot['host_ae'] += float((hv - by).abs().sum())

    base_all = torch.cat(base_chunks, dim=0)
    base_equal = None
    if base_ref is not None and Path(base_ref).exists():
        ref = torch.load(base_ref, map_location='cpu')
        base_equal = bool(torch.equal(ref, base_all))
        if not base_equal:
            raise SystemExit('[ABORT] y_base differs from the reference arm '
                             f'(max_abs_diff={float((ref - base_all).abs().max()):.3e}). '
                             'The Base-Forecaster-Only comparison would not be controlled.')


    ret_mse = tot['ret_se'] / max(tot['n'], 1)
    ret_mae = tot['ret_ae'] / max(tot['n'], 1)
    host_ablated_mse = tot['base_se'] / max(tot['n'], 1)
    host_ablated_mae = tot['base_ae'] / max(tot['n'], 1)
    host_orig_mse = tot['host_se'] / max(tot['n'], 1)
    host_orig_mae = tot['host_ae'] / max(tot['n'], 1)

    out = {
        'arm_checkpoint': str(arm_ckpt),
        'stage2_host': str(stage2_host),
        'oracle': target, 'prefix': prefix_policy,
        'score': 'asymmetric' if metric is not None else 'cosine',
        'best_epoch': int(ckpt.get('epoch', -1)),

        # ---- PRIMARY: retrieval-augmented vs the independent baseline ----
        'retrieval_augmented_frozen_host_mse': ret_mse,
        'retrieval_augmented_frozen_host_mae': ret_mae,
        'independent_base_only_mse': independent_base.get('mse') if independent_base else None,
        'independent_base_only_mae': independent_base.get('mae') if independent_base else None,
        'delta_mse_vs_independent_base': (
            ret_mse - independent_base['mse'] if independent_base else None),
        'delta_mae_vs_independent_base': (
            ret_mae - independent_base['mae'] if independent_base else None),
        'relative_improvement_vs_independent_base_pct': (
            (independent_base['mse'] - ret_mse) / max(independent_base['mse'], 1e-12) * 100.0
            if independent_base else None),

        # ---- DIAGNOSTIC ONLY -- not baselines, never to be called one ----
        'frozen_host_retrieval_ablated_branch_mse': host_ablated_mse,
        'frozen_host_retrieval_ablated_branch_mae': host_ablated_mae,
        'diag_delta_mse_vs_host_ablated_branch': ret_mse - host_ablated_mse,
        'frozen_host_original_retrieval_mse': host_orig_mse,
        'frozen_host_original_retrieval_mae': host_orig_mae,
        'diag_delta_mse_vs_host_original_retrieval': ret_mse - host_orig_mse,

        'test_free_running_aggregate_future_mse': fr_sum / max(fr_n, 1),
        'duplicate_rate': dup / max(rows, 1),
        'invalid_rate': invalid / max(rows, 1),
        'n_rows_per_channel': rows // max(len(channels), 1),
        'n_channels': len(channels),
        'relation_top_n': int(getattr(s2_args, 'relation_top_n', 1)),
        'fusion_mode': getattr(s2_args, 'fusion_mode', None),
        'gate_mode': getattr(s2_args, 'gate_mode', None),
        'y_base_identical_to_reference': base_equal,
        'independent_base_only_source': independent_base.get('source') if independent_base else None,
        'scope_note': ('Retrieval-Augmented FROZEN-HOST Stage-2. Diagnostic: shows whether '
                       'this retrieval, injected into the existing S0_wce host, already beats '
                       'an Independent Base-only Forecaster. NOT the performance of a Stage-2 '
                       'retrained for this retrieval.'),
    }
    return out, base_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm_checkpoint', required=True)
    ap.add_argument('--stage2_host', required=True)
    ap.add_argument('--cell', required=True)
    ap.add_argument('--arm_name', required=True)
    ap.add_argument('--out_dir', default='results/track_a_factorial_e2e')
    ap.add_argument('--top_k', type=int, default=10)
    ap.add_argument('--chunk_size', type=int, default=4096)
    ap.add_argument('--base_ref', default=None,
                    help='path to a stored reference y_base tensor; asserted bit-identical')
    ap.add_argument('--save_base_ref', default=None)
    ap.add_argument('--limit_batches', type=int, default=0,
                    help='SMOKE ONLY: cap test batches. 0 = full test split.')
    ap.add_argument('--independent_base_json', default=None,
                    help='independent_base_only.json for this cell -- supplies THE baseline')
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    independent_base = None
    if cli.independent_base_json:
        b = json.loads(Path(cli.independent_base_json).read_text())
        independent_base = {'mse': b['independent_base_only_mse'],
                            'mae': b['independent_base_only_mae'],
                            'source': cli.independent_base_json}
    elif not cli.limit_batches:
        raise SystemExit('[ABORT] --independent_base_json is required for a real run: '
                         'the Independent Base-only Forecaster is THE baseline.')
    out, base_all = evaluate(cli.arm_checkpoint, cli.stage2_host, cli.top_k,
                             cli.chunk_size, device, base_ref=cli.base_ref,
                             limit_batches=cli.limit_batches,
                             independent_base=independent_base)
    cell_dir = Path(cli.out_dir) / cli.cell
    cell_dir.mkdir(parents=True, exist_ok=True)
    out['cell'] = cli.cell
    out['arm'] = cli.arm_name
    out['limit_batches'] = cli.limit_batches
    out['is_smoke_run'] = bool(cli.limit_batches)
    (cell_dir / f'stage2_metrics_{cli.arm_name}.json').write_text(json.dumps(out, indent=2))
    if cli.save_base_ref:
        Path(cli.save_base_ref).parent.mkdir(parents=True, exist_ok=True)
        torch.save(base_all, cli.save_base_ref)
    for k, v in out.items():
        print(f'[factorial_e2e01-stage2] {k} = {v}')


if __name__ == '__main__':
    main()
