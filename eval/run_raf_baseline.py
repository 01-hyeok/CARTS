"""RAF (arXiv:2411.08249) evaluated on the CARTS retrieval-ablation harness.

RAF's own code loads HuggingFace series, splits them randomly by series, and
reports MASE/WQL. None of that is comparable with the ablation table, so this
script keeps RAF's *method* and swaps in our evaluation setup:

  from RAF   Chronos encoder embeddings, per-token L2 retrieval, top-n averaging,
             offset alignment, in-context concatenation, Chronos forecasting
  from CARTS TSLib loaders and temporal split, train-fitted StandardScaler,
             identical test windows, MSE/MAE in the normalised space

RAF is univariate: every channel is retrieved and forecast independently, and no
cross-channel information is used. That is faithful to `to_gluonts_univariate`
in the original repository.

Only pred_len 96 is supported. RAF concatenates
[retrieved context (L) | retrieved future (H) | query context (L)], so the model
input is 2L+H. With seq_len == pred_len that is 3L, and Chronos T5 truncates to
its 512-token context: 96 -> 288 fits, 192 -> 576 already loses part of the
retrieved context, and 336 -> 1008 drops it entirely.
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_provider.data_factory import data_provider  # noqa: E402


def build_args(cli):
    """Minimal namespace accepted by CARTS' data_provider."""
    class A:
        pass

    a = A()
    a.task_name = 'long_term_forecast'
    a.data = cli.data
    a.root_path = cli.root_path
    a.data_path = cli.data_path
    a.features = 'M'
    a.target = 'OT'
    a.freq = cli.freq
    a.seq_len = cli.seq_len
    a.label_len = 0
    a.pred_len = cli.pred_len
    a.embed = 'timeF'
    a.batch_size = 32
    a.num_workers = 0
    a.seasonal_patterns = None
    a.augmentation_ratio = 0
    return a


def collect_windows(dataset):
    """Stack a TSLib dataset into [N, L, C] history and [N, H, C] future.

    CARTS datasets yield (index, seq_x, seq_y, seq_x_mark, seq_y_mark).
    """
    xs, ys = [], []
    for i in range(len(dataset)):
        item = dataset[i]
        seq_x, seq_y = item[1], item[2]
        xs.append(np.asarray(seq_x, dtype=np.float32))
        ys.append(np.asarray(seq_y, dtype=np.float32))
    return np.stack(xs), np.stack(ys)


@torch.no_grad()
def embed_series(pipeline, series, batch_size, device):
    """Chronos encoder embeddings for [N, L] -> [N, T, D] (kept in fp16)."""
    out = []
    for start in range(0, series.shape[0], batch_size):
        chunk = torch.as_tensor(series[start:start + batch_size], dtype=torch.float32)
        emb, _ = pipeline.embed(chunk)
        out.append(emb.to(torch.float16).to(device))
    return torch.cat(out, dim=0)


def raf_distance(query_emb, key_emb, query_chunk=64):
    """RAF's score: per-token L2 distance summed over token positions.

    Reproduces
        torch.norm(target[:, None] - other[None], dim=3, p=2).sum(dim=2)
    without materialising the [B, N, T, D] difference tensor.
    """
    n_query, n_token, _ = query_emb.shape
    n_key = key_emb.shape[0]
    scores = torch.empty(n_query, n_key, dtype=torch.float32, device=query_emb.device)
    key_sq = (key_emb.float() ** 2).sum(dim=-1)                      # [N, T]
    for start in range(0, n_query, query_chunk):
        q = query_emb[start:start + query_chunk].float()             # [b, T, D]
        q_sq = (q ** 2).sum(dim=-1)                                  # [b, T]
        acc = torch.zeros(q.shape[0], n_key, device=q.device)
        for t in range(n_token):
            # [b, D] x [D, N] -> [b, N] inner products for this token position
            dot = q[:, t] @ key_emb[:, t].float().transpose(0, 1)
            sq = (q_sq[:, t].unsqueeze(1) + key_sq[:, t].unsqueeze(0) - 2.0 * dot)
            acc += sq.clamp_min(0.0).sqrt()
        scores[start:start + query_chunk] = acc
    return scores


def normalize_own(x):
    """RAF normalises each series by its own mean/std before Chronos sees it."""
    mean = x.mean()
    std = torch.sqrt(((x - mean) ** 2).mean()) + 1e-7
    return (x - mean) / std, mean, std


def build_augmented_context(query_ctx, retrieved_segments):
    """RAF augmentation for one query: average top-n, align, concatenate."""
    avg_segment = torch.as_tensor(retrieved_segments, dtype=torch.float32).mean(dim=0)
    avg_segment, _, _ = normalize_own(avg_segment)
    context, ctx_mean, ctx_std = normalize_own(
        torch.as_tensor(query_ctx, dtype=torch.float32)
    )
    # Align the end of the retrieved segment with the start of the query context.
    avg_segment = avg_segment + (context[0] - avg_segment[-1])
    return torch.cat([avg_segment, context]), ctx_mean, ctx_std


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', default='ETTh1', choices=['ETTh1', 'ETTm1', 'custom'])
    p.add_argument('--root_path', default='../Dataset/Time-Series-Library_dataset/ETT-small/')
    p.add_argument('--data_path', default='ETTh1.csv')
    p.add_argument('--freq', default='h')
    p.add_argument('--seq_len', type=int, default=96)
    p.add_argument('--pred_len', type=int, default=96)
    p.add_argument('--top_n', type=int, default=1, help="RAF's default is 1")
    p.add_argument('--num_samples', type=int, default=20)
    p.add_argument('--predict_batch_size', type=int, default=30)
    p.add_argument('--embed_batch_size', type=int, default=256)
    p.add_argument('--chronos_model_id', default='amazon/chronos-t5-base')
    p.add_argument('--chronos_dtype', default='bfloat16')
    p.add_argument('--augment', dest='augment', action='store_true', default=True)
    p.add_argument('--no-augment', dest='augment', action='store_false',
                   help='Plain zero-shot Chronos, the RAF paper baseline')
    p.add_argument('--output_csv', default='')
    cli = p.parse_args()

    if cli.seq_len != cli.pred_len:
        raise ValueError('this harness uses the seq_len == pred_len protocol')
    if cli.augment and 2 * cli.seq_len + cli.pred_len > 512:
        raise ValueError(
            f'augmented input is {2 * cli.seq_len + cli.pred_len} > 512 Chronos context; '
            'RAF is only faithfully reproducible at pred_len 96 under seq==pred'
        )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = build_args(cli)

    train_set, _ = data_provider(args, flag='train')
    test_set, _ = data_provider(args, flag='test')
    print(f'[raf] train windows={len(train_set)} test windows={len(test_set)}')

    train_x, train_y = collect_windows(train_set)
    test_x, test_y = collect_windows(test_set)
    channels = train_x.shape[-1]

    from chronos import ChronosPipeline
    pipeline = ChronosPipeline.from_pretrained(
        cli.chronos_model_id,
        device_map=str(device),
        torch_dtype=getattr(torch, cli.chronos_dtype),
    )

    preds = np.zeros_like(test_y)
    started = time.time()

    for c in range(channels):
        query_ctx = test_x[:, :, c]
        contexts = []
        scales = []

        if cli.augment:
            # Candidates are train windows of the same channel: their history is
            # matched against the query, their history+future is what gets pasted in.
            cand_ctx = train_x[:, :, c]
            cand_seg = np.concatenate([train_x[:, :, c], train_y[:, :, c]], axis=1)

            key_emb = embed_series(pipeline, cand_ctx, cli.embed_batch_size, device)
            query_emb = embed_series(pipeline, query_ctx, cli.embed_batch_size, device)
            scores = raf_distance(query_emb, key_emb)
            top_idx = torch.topk(scores, k=cli.top_n, dim=1, largest=False).indices.cpu().numpy()
            del key_emb, query_emb, scores
            torch.cuda.empty_cache()

            for i in range(query_ctx.shape[0]):
                ctx, mean, std = build_augmented_context(
                    query_ctx[i], cand_seg[top_idx[i]]
                )
                contexts.append(ctx)
                scales.append((mean, std))
        else:
            for i in range(query_ctx.shape[0]):
                ctx, mean, std = normalize_own(
                    torch.as_tensor(query_ctx[i], dtype=torch.float32)
                )
                contexts.append(ctx)
                scales.append((mean, std))

        for start in range(0, len(contexts), cli.predict_batch_size):
            batch = contexts[start:start + cli.predict_batch_size]
            forecast = pipeline.predict(
                batch,
                prediction_length=cli.pred_len,
                num_samples=cli.num_samples,
                limit_prediction_length=False,
            )
            # Chronos reports samples; the point forecast is the median, matching
            # the 0.5 quantile RAF scores with.
            point = torch.median(forecast, dim=1).values
            for j in range(point.shape[0]):
                mean, std = scales[start + j]
                preds[start + j, :, c] = (
                    torch.nan_to_num(point[j] * std + mean, nan=0.0).cpu().numpy()
                )

        print(f'[raf] channel {c + 1}/{channels} done ({time.time() - started:.0f}s)')

    mse = float(np.mean((preds - test_y) ** 2))
    mae = float(np.mean(np.abs(preds - test_y)))
    mode = 'RAF' if cli.augment else 'Chronos-zeroshot'
    print(f'dataset: {cli.data}')
    print(f'mode: {mode}')
    print(f'seq_len: {cli.seq_len}')
    print(f'pred_len: {cli.pred_len}')
    print(f'final_mse: {mse:.6f}')
    print(f'final_mae: {mae:.6f}')

    if cli.output_csv:
        os.makedirs(os.path.dirname(cli.output_csv) or '.', exist_ok=True)
        exists = os.path.exists(cli.output_csv)
        with open(cli.output_csv, 'a') as handle:
            if not exists:
                handle.write('dataset,mode,seq_len,pred_len,top_n,mse,mae\n')
            handle.write(
                f'{cli.data},{mode},{cli.seq_len},{cli.pred_len},{cli.top_n},{mse:.6f},{mae:.6f}\n'
            )
        print(f'[raf] appended {cli.output_csv}')


if __name__ == '__main__':
    main()
