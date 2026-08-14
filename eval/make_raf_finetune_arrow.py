"""Build Advanced-RAF fine-tuning data from the CARTS train split.

RAF's `generate_fine_tune_data.py` writes, for every training window, one series

    [ retrieved context | retrieved future | query context | query future ]

where the query future is normalised with the *query context* mean/std, so the
model learns to continue an augmented context. This script reproduces that using
our temporal train split instead of RAF's series-level random split, so nothing
from validation or test can leak into the fine-tuning data.

The output arrow file is consumed by the RAF repository's own trainer
(`chronos_training/train.py`), which keeps the Chronos training objective intact.
"""

import argparse
import os
import sys

import numpy as np
import torch
from datasets.arrow_writer import ArrowWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.run_raf_baseline import (  # noqa: E402
    build_args,
    collect_windows,
    data_provider,
    embed_series,
    normalize_own,
    raf_distance,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='ETTh1', choices=['ETTh1', 'ETTm1', 'custom'])
    p.add_argument('--root_path', default='../Dataset/Time-Series-Library_dataset/ETT-small/')
    p.add_argument('--data_path', default='ETTh1.csv')
    p.add_argument('--freq', default='h')
    p.add_argument('--seq_len', type=int, default=96)
    p.add_argument('--pred_len', type=int, default=96)
    p.add_argument('--top_n', type=int, default=1)
    p.add_argument('--embed_batch_size', type=int, default=256)
    p.add_argument('--chronos_model_id', default='amazon/chronos-t5-base')
    p.add_argument('--chronos_dtype', default='bfloat16')
    p.add_argument('--augment', dest='augment', action='store_true', default=True)
    p.add_argument('--no-augment', dest='augment', action='store_false',
                   help='Baseline fine-tuning data with no retrieval')
    p.add_argument('--out', required=True, help='Output .arrow path')
    cli = p.parse_args()

    if cli.seq_len != cli.pred_len:
        raise ValueError('this harness uses the seq_len == pred_len protocol')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = build_args(cli)
    train_set, _ = data_provider(args, flag='train')
    train_x, train_y = collect_windows(train_set)
    channels = train_x.shape[-1]
    print(f'[raf-arrow] train windows={train_x.shape[0]} channels={channels}')

    from chronos import ChronosPipeline
    pipeline = ChronosPipeline.from_pretrained(
        cli.chronos_model_id,
        device_map=str(device),
        torch_dtype=getattr(torch, cli.chronos_dtype),
    )

    series = []
    for c in range(channels):
        ctx_all = train_x[:, :, c]
        fut_all = train_y[:, :, c]
        seg_all = np.concatenate([ctx_all, fut_all], axis=1)

        if cli.augment:
            key_emb = embed_series(pipeline, ctx_all, cli.embed_batch_size, device)
            scores = raf_distance(key_emb, key_emb)
            # A window must not retrieve itself, otherwise the target future is
            # copied straight into the input.
            scores.fill_diagonal_(float('inf'))
            top_idx = torch.topk(
                scores, k=cli.top_n, dim=1, largest=False
            ).indices.cpu().numpy()
            del key_emb, scores
            torch.cuda.empty_cache()

        for i in range(ctx_all.shape[0]):
            context, mean, std = normalize_own(
                torch.as_tensor(ctx_all[i], dtype=torch.float32)
            )
            if cli.augment:
                avg_segment = torch.as_tensor(
                    seg_all[top_idx[i]], dtype=torch.float32
                ).mean(dim=0)
                avg_segment, _, _ = normalize_own(avg_segment)
                avg_segment = avg_segment + (context[0] - avg_segment[-1])
                context = torch.cat([avg_segment, context])
            # The label uses the query context statistics, matching RAF.
            label = (torch.as_tensor(fut_all[i], dtype=torch.float32) - mean) / std
            series.append(torch.cat([context, label]).numpy().astype(np.float32))

        print(f'[raf-arrow] channel {c + 1}/{channels} done')

    os.makedirs(os.path.dirname(cli.out) or '.', exist_ok=True)
    start = np.datetime64('2000-01-01 00:00', 's')
    ArrowWriter(compression='lz4').write_to_file(
        [{'start': start, 'target': ts} for ts in series],
        path=cli.out,
    )
    total_len = len(series[0])
    print(f'[raf-arrow] wrote {len(series)} series of length {total_len} to {cli.out}')
    print(f'[raf-arrow] chronos_training config should use '
          f'context_length={total_len - cli.pred_len} prediction_length={cli.pred_len}')


if __name__ == '__main__':
    main()
