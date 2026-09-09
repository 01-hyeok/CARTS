#!/usr/bin/env python3
"""Track B4 (EXP-CORRECTION-ENCODER01) -- retrieval-quality evaluation,
diagnostic only (no training). For ONE encoder (`--encoder {future,
correction}`) and one split, computes the Correction-Oracle-relative
metrics the spec requires: Recall@{1,5,10,20,50} (does the Correction
Oracle's single best candidate fall within the encoder's own Top-K, for
varying K -- derived from ONE rank computation per query via
`oracle_rank_statistics`, reused unmodified), NDCG@{10,50} (`_ndcg_at_k`,
reused from `scripts/eval_firstanchor_diag.py`), and Best@{10,20,50,100}
MSE (`min_i MSE(B_q+r_i,Y_q)` restricted to the encoder's own Top-K, vs.
the full-memory Correction Oracle floor).

`--encoder future` uses the EXISTING, frozen production Stage-2
checkpoint's own encoder/score (`_branch_embedding`, `_retrieval_score_fn`)
-- never retrained here. `--encoder correction` loads a checkpoint trained
by `scripts/train_correction_encoder01.py`. Both are scored on EXACTLY the
same candidate/query tensors (`exp.memory_x`/`batch_x` from the SAME
`Exp_Stage1_Relation` built from `--reference_ckpt`'s args) so the two
arms never differ because of a data-alignment mismatch between two
different `Exp_*` objects -- `_branch_embedding` is a stateless function of
its input tensor, so calling it on the scratch exp's own `memory_x`/
`batch_x` is exactly as valid as calling it on Stage-2's own.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval_firstanchor_diag import _ndcg_at_k
from scripts.train_margutil01 import build_experiment
from scripts.train_oracle_scratch01 import base_score, encode_raw
from utils.retrieval_diagnostics import base_forecast, load_stage2, unwrap


def future_score(b0_model, z_q_input, z_k_input, c):
    z_q = b0_model._branch_embedding(z_q_input, c, c)
    z_k = b0_model._branch_embedding(z_k_input, c, c)
    score_fn = b0_model._retrieval_score_fn()
    if score_fn is not None:
        return score_fn(z_q, z_k)
    import torch.nn.functional as F
    return torch.matmul(F.normalize(z_q, dim=-1), F.normalize(z_k, dim=-1).transpose(0, 1))


@torch.no_grad()
def evaluate(exp, b0_model, c_model, encoder_kind, split, channels, chunk_size, r_i_full, device, k_values):
    _, loader = exp._get_data(flag=split, shuffle=False)
    agg = {kk: {'recall': 0.0, 'best_mse_se': 0.0} for kk in k_values}
    ndcg_sum = {10: 0.0, 50: 0.0}
    n_rows = 0
    oracle_full_se = 0.0
    n_channels = len(channels)

    for batch_x, batch_y, batch_start_idx in loader:
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        cand_mask, counts = exp._candidate_mask(batch_start_idx)
        valid_query = counts.to(device) >= max(k_values)
        if not bool(valid_query.any()):
            continue
        B_q_full = base_forecast(b0_model, batch_x, chunk_size=chunk_size)
        for c in channels:
            if encoder_kind == 'future':
                s_hat = future_score(b0_model, batch_x, exp.memory_x, c)
            else:
                z_q = encode_raw(c_model, batch_x, c)
                E = encode_raw(c_model, exp.memory_x, c)
                s_hat = base_score(z_q, E, None)

            B_q = B_q_full[:, :, c]
            query_future = batch_y[:, :, c]
            r_q = query_future - B_q
            r_i_c = r_i_full[:, :, c]
            d_i = (r_i_c.unsqueeze(0) - r_q.unsqueeze(1)).pow(2).mean(dim=-1)  # [B, N]
            u_i = -d_i

            neg_inf = torch.finfo(s_hat.dtype).min / 4
            u_masked = u_i.masked_fill(~cand_mask, neg_inf)
            i_star = u_masked.argmax(dim=-1, keepdim=True)

            s_masked = s_hat.masked_fill(~cand_mask, neg_inf)
            order = s_masked.argsort(dim=-1, descending=True)
            rank_positions = torch.empty_like(order)
            positions = torch.arange(1, order.size(-1) + 1, device=device).expand_as(order)
            rank_positions.scatter_(1, order, positions)
            i_star_rank = rank_positions.gather(1, i_star).squeeze(-1).float()

            d_i_sorted_by_model = d_i.gather(1, order)
            oracle_best_valid = d_i.masked_fill(~cand_mask, float('inf')).min(dim=-1).values

            vq = valid_query
            oracle_full_se += float(oracle_best_valid[vq].sum())

            for kk in k_values:
                recall_hit = (i_star_rank <= kk).float()
                agg[kk]['recall'] += float(recall_hit[vq].sum())
                best_k = d_i_sorted_by_model[:, :kk].min(dim=-1).values
                agg[kk]['best_mse_se'] += float(best_k[vq].sum())

            for kk in (10, 50):
                rel = u_i.masked_fill(~cand_mask, float('nan'))
                rel_valid_min = torch.nan_to_num(rel, nan=float('inf')).min(dim=-1, keepdim=True).values
                rel_shifted = (u_i - rel_valid_min + 1e-6).masked_fill(~cand_mask, 0.0)
                rel_pred_order = rel_shifted.gather(1, order)
                for b in range(batch_x.size(0)):
                    if not bool(vq[b]):
                        continue
                    ndcg_sum[kk] += _ndcg_at_k(rel_pred_order[b], kk)

            n_rows += int(vq.sum())

    n_rows_total = max(n_rows, 1) / n_channels if n_channels else max(n_rows, 1)
    out = {'n_rows': n_rows, 'oracle_full_mse': oracle_full_se / max(n_rows, 1)}
    for kk in k_values:
        out[f'recall@{kk}'] = agg[kk]['recall'] / max(n_rows, 1)
        out[f'best@{kk}_mse'] = agg[kk]['best_mse_se'] / max(n_rows, 1)
    for kk in (10, 50):
        out[f'ndcg@{kk}'] = ndcg_sum[kk] / max(n_rows, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference_ckpt', required=True)
    ap.add_argument('--stage2_checkpoint', required=True)
    ap.add_argument('--encoder', choices=['future', 'correction'], required=True)
    ap.add_argument('--correction_encoder_checkpoint', default=None)
    ap.add_argument('--pred_len', type=int, required=True)
    ap.add_argument('--chunk_size', type=int, default=2048)
    ap.add_argument('--out', required=True)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    overrides = {'is_training': 0, 'model_id': 'eval_correction_encoder01', 'des': 'eval',
                 'checkpoints': '/tmp/exp_correction_encoder01_eval', 'stage1_retrieval_metric': 'cosine',
                 'pred_len': cli.pred_len, 'seq_len': cli.pred_len}
    exp, args = build_experiment(cli.reference_ckpt, overrides)
    exp._ensure_memory()
    channels = list(exp.model.target_channels() if not hasattr(exp.model, 'module') else exp.model.module.target_channels())

    b0_exp, b0_args = load_stage2(cli.stage2_checkpoint, device=device)
    b0_exp.model.to(device)
    b0_model = unwrap(b0_exp.model)
    b0_model.eval()
    for p in b0_model.parameters():
        p.requires_grad_(False)

    with torch.no_grad():
        B_i_full = base_forecast(b0_model, exp.memory_x, chunk_size=cli.chunk_size)
        r_i_full = exp.memory_y - B_i_full

    c_model = None
    if cli.encoder == 'correction':
        assert cli.correction_encoder_checkpoint, '--correction_encoder_checkpoint required for --encoder correction'
        ck = torch.load(cli.correction_encoder_checkpoint, map_location='cpu')
        c_model = exp.model.module if hasattr(exp.model, 'module') else exp.model
        c_model.load_state_dict(ck['model_state_dict'], strict=True)
        c_model.to(device)
        c_model.eval()
        for p in c_model.parameters():
            p.requires_grad_(False)
    else:
        c_model = exp.model.module if hasattr(exp.model, 'module') else exp.model

    k_values = [1, 5, 10, 20, 50]
    report = {}
    for split in ('train', 'val', 'test'):
        print(f'[eval_correction_encoder01] encoder={cli.encoder} split={split}...')
        res = evaluate(exp, b0_model, c_model, cli.encoder, split, channels, cli.chunk_size, r_i_full, device, k_values)
        report[split] = res
        print(f'  recall@10={res["recall@10"]:.4f} recall@50={res["recall@50"]:.4f} '
              f'ndcg@10={res["ndcg@10"]:.4f} best@10_mse={res["best@10_mse"]:.5f} '
              f'oracle_full_mse={res["oracle_full_mse"]:.5f}')

    Path(cli.out).parent.mkdir(parents=True, exist_ok=True)
    with open(cli.out, 'w') as fh:
        json.dump({'encoder': cli.encoder, **report}, fh, indent=2, default=str)
    print(f'[eval_correction_encoder01] written to {cli.out}')


if __name__ == '__main__':
    main()
