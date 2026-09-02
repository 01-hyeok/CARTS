#!/usr/bin/env python3
"""STEP 4 -- can candidate quality be defined as forecast utility directly?

Future similarity and residual similarity are both proxies. The quantity that
actually matters is whether applying a candidate's correction makes the forecast
better:

    U(q,k) = MSE(Y_q, Yhat_q^base) - MSE(Y_q, Yhat_q^base + alpha * R_k)
           = 2*alpha*<R_q, R_k>/T - alpha^2*||R_k||^2/T

The closed form means the whole [queries x candidates] utility matrix is one
matmul, so the Oracle over it is cheap to compute exactly.

This is a diagnostic, not a model. U depends on the query future, so no
deployable retriever can rank by it. What the numbers decide is whether utility
is worth *learning to predict* -- which needs three things at once: a utility
Oracle that beats the other Oracles, a usable fraction of positive-utility
candidates, and some correlation between past-side signal and utility.
"""

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.retrieval_diagnostics import (  # noqa: E402
    COVERAGE_DEPTHS, append_row, coverage_at_m, gap_recovery, load_stage2,
    mse_mae, rank_correlations, unwrap,
)
from scripts.analyze_residual_oracle import _pair_mse, oracle_predictions, prepare  # noqa: E402

COLUMNS = [
    'dataset', 'pred_len', 'top_k', 'alpha',
    'base_mse', 'future_oracle_mse', 'residual_oracle_mse', 'utility_oracle_mse',
    'positive_utility_fraction', 'retrieved_positive_utility_fraction',
    'past_utility_pearson', 'past_utility_spearman',
    'encoder_utility_pearson', 'encoder_utility_spearman',
    *[f'raw_utility_coverage_at_{m}' for m in COVERAGE_DEPTHS],
    'utility_gap_recovery_raw', 'utility_gap_recovery_encoder',
    'verdict', 'checkpoint',
]


@torch.no_grad()
def analyse(checkpoint_path, top_k=10, alpha=1.0, max_batches=0, corr_queries=128):
    experiment, args = load_stage2(checkpoint_path)
    model = unwrap(experiment.model)
    device = experiment.device
    # The encoder-vs-utility correlation needs the Stage-1 key bank.
    experiment._ensure_memory()
    experiment._build_key_bank(force=True)
    test = prepare(experiment, 'test', max_batches)
    future_pred, res_uniform, _ = oracle_predictions(experiment, test, top_k)

    base_mse, _ = mse_mae(test['query_base'], test['query_y'])
    future_mse_, _ = mse_mae(future_pred, test['query_y'])
    residual_mse_, _ = mse_mae(test['query_base'] + alpha * res_uniform, test['query_y'])

    channels = test['query_y'].size(-1)
    n_query = test['query_y'].size(0)
    utility_pred = torch.zeros_like(test['query_base'])
    pos_frac, ret_pos_frac = [], []
    cov_acc, past_p, past_s, enc_p, enc_s = [], [], [], [], []
    gap_raw, gap_enc = [], []
    corr_done = 0

    memory_past = test['memory_x']
    for start in range(0, n_query, 256):
        stop = min(start + 256, n_query)
        cand_mask, _ = experiment._candidate_mask(test['query_start'][start:stop])
        cand_mask = cand_mask.to(device)
        want_corr = corr_done < corr_queries

        for c in range(channels):
            q_res = test['query_residual'][start:stop, :, c]
            k_res = test['memory_residual'][:, :, c]
            horizon = float(k_res.size(-1))
            # U(q,k) in closed form: one matmul over the whole candidate bank.
            cross = torch.matmul(q_res, k_res.transpose(0, 1)) / horizon
            k_energy = k_res.square().mean(-1).unsqueeze(0)
            utility = 2.0 * alpha * cross - (alpha ** 2) * k_energy
            utility = utility.masked_fill(~cand_mask, float('-inf'))

            idx = utility.topk(min(top_k, k_res.size(0)), dim=-1, largest=True).indices
            utility_pred[start:stop, :, c] = k_res[idx].mean(dim=1)

            finite = cand_mask
            pos_frac.append(((utility > 0) & finite).sum(-1).float() / finite.sum(-1).clamp_min(1))

            # Raw past L2 ranking, the encoder-free retriever from the previous step.
            qp = test['query_x'][start:stop, :, c] - test['query_x'][start:stop, -1:, c]
            kp = memory_past[:, :, c] - memory_past[:, -1:, c]
            raw_scores = (-_pair_mse(qp, kp)).masked_fill(~cand_mask, float('-inf'))
            raw_topk = raw_scores.topk(min(top_k, kp.size(0)), dim=-1).indices
            ret_pos_frac.append(
                (utility.gather(1, raw_topk) > 0).float().mean(-1)
            )
            cov_acc.append(coverage_at_m(raw_scores, idx, cand_mask))

            # Utility captured by a retriever vs a random pick vs the Oracle.
            oracle_u = utility.gather(1, idx).mean(-1)
            raw_u = utility.gather(1, raw_topk).mean(-1)
            random_u = utility.masked_fill(~cand_mask, 0.0).sum(-1) / cand_mask.sum(-1).clamp_min(1)
            gap_raw.append(gap_recovery(raw_u, random_u, oracle_u, higher_is_better=True).mean())

            enc_scores = _encoder_scores(experiment, model, test, start, stop, c, cand_mask)
            if enc_scores is not None:
                enc_topk = enc_scores.topk(min(top_k, kp.size(0)), dim=-1).indices
                enc_u = utility.gather(1, enc_topk).mean(-1)
                gap_enc.append(gap_recovery(enc_u, random_u, oracle_u, higher_is_better=True).mean())

            if want_corr:
                p, s = rank_correlations(raw_scores, utility, cand_mask)
                past_p.append(p); past_s.append(s)
                if enc_scores is not None:
                    p, s = rank_correlations(enc_scores, utility, cand_mask)
                    enc_p.append(p); enc_s.append(s)
        if want_corr:
            corr_done += stop - start

    utility_mse, _ = mse_mae(test['query_base'] + alpha * utility_pred, test['query_y'])
    mean = lambda xs: float(torch.stack([x.mean() if torch.is_tensor(x) else torch.tensor(x) for x in xs]).mean()) if xs else float('nan')
    row = {
        'dataset': args.data, 'pred_len': int(args.pred_len),
        'top_k': top_k, 'alpha': alpha,
        'base_mse': base_mse, 'future_oracle_mse': future_mse_,
        'residual_oracle_mse': residual_mse_, 'utility_oracle_mse': utility_mse,
        'positive_utility_fraction': mean(pos_frac),
        'retrieved_positive_utility_fraction': mean(ret_pos_frac),
        'past_utility_pearson': mean(past_p), 'past_utility_spearman': mean(past_s),
        'encoder_utility_pearson': mean(enc_p), 'encoder_utility_spearman': mean(enc_s),
        'utility_gap_recovery_raw': mean(gap_raw),
        'utility_gap_recovery_encoder': mean(gap_enc),
        'checkpoint': checkpoint_path,
    }
    for depth in COVERAGE_DEPTHS:
        key = f'coverage_at_{depth}'
        row[f'raw_utility_coverage_at_{depth}'] = mean([c[key] for c in cov_acc])
    row['verdict'] = classify(row)
    return row


def _encoder_scores(experiment, model, test, start, stop, channel, cand_mask):
    """Current Stage-1 retrieval scores, when the checkpoint has an encoder."""
    if experiment.key_bank is None:
        return None
    sources = model.source_channels(channel)
    if channel not in sources:
        return None
    slot = sources.index(channel)
    z_q = model._branch_embedding(test['query_x'][start:stop], channel, channel)
    z_k = experiment.key_bank[channel, slot].to(z_q.device, z_q.dtype)
    return torch.matmul(z_q, z_k.transpose(0, 1)).masked_fill(~cand_mask, float('-inf'))


def classify(row):
    utility, residual = row['utility_oracle_mse'], row['residual_oracle_mse']
    future, base = row['future_oracle_mse'], row['base_mse']
    signal = max(
        abs(row['past_utility_spearman'] or 0.0),
        abs(row['encoder_utility_spearman'] or 0.0),
    )
    if utility >= base:
        return 'RETRIEVAL_SIGNAL_TOO_WEAK'
    best_oracle = min(utility, residual, future)
    if utility <= best_oracle and signal >= 0.10 and row['raw_utility_coverage_at_500'] >= 0.20:
        return 'UTILITY_TARGET_PROMISING'
    if residual < future and residual <= utility:
        return 'RESIDUAL_TARGET_PREFERRED'
    if future <= min(utility, residual):
        return 'FUTURE_TARGET_STILL_PREFERRED'
    if signal < 0.05:
        return 'RETRIEVAL_SIGNAL_TOO_WEAK'
    return 'RESIDUAL_TARGET_PREFERRED'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--max_batches', type=int, default=0)
    parser.add_argument('--csv', default='')
    args = parser.parse_args()

    row = analyse(args.checkpoint, args.top_k, args.alpha, args.max_batches)
    print(f"=== {row['dataset']}/{row['pred_len']} forecast utility ===")
    for key in COLUMNS:
        if key == 'checkpoint':
            continue
        value = row[key]
        print(f'  {key}: {value:.6f}' if isinstance(value, float) else f'  {key}: {value}')
    if args.csv:
        append_row(args.csv, row, COLUMNS)
        print(f'appended to {args.csv}')


if __name__ == '__main__':
    main()
