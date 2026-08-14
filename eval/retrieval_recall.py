"""Retrieval-only Recall@K study.

Measures one thing: how well a (representation, encoder, similarity) triple
retrieves candidates whose real future is close to the query's. No Stage-2, no
forecasting model, no NDCG/Spearman/regret.

The oracle is fixed for every configuration: the K candidates with the smallest
raw-future MSE. It is built from the futures alone and never sees the
representation, the encoder or the similarity, so the 192 configurations stay
comparable by construction.

Fairness comes from sharing one code path rather than from matching flags: the
query set, candidate pool, valid mask and overlap exclusion all come from
RelationMemorySampler, exactly as Stage-1 builds them.

Recall@K compares retrieved top-K against oracle top-K of the same size K:
    recall@k = |retrieved_topk ∩ oracle_topk| / k     averaged over queries.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_provider.data_factory import data_provider
from utils.relation_memory import RelationMemorySampler, Stage1WindowDataset
from utils.retrieval_mlp import (
    TEACHERS,
    RelationMLP,
    checkpoint_path,
    train_encoders,
)
from utils.retrieval_representations import (
    DEFAULT_PERIOD,
    REPRESENTATIONS,
    fit_ar1_on_differences,
    representation_length,
    transform,
)
from utils.retrieval_scoring import (
    SIMILARITIES,
    fuse_decomposition_scores,
    raw_future_mse,
    recall_at_k,
    similarity_scores,
)

KS = (1, 5, 10)


def build_args(cli):
    """Minimal args object the shared data_provider expects."""
    ns = argparse.Namespace(
        data=cli.data,
        root_path=cli.root_path,
        data_path=cli.data_path,
        features='M',
        target='OT',
        freq=cli.freq,
        embed='timeF',
        seq_len=cli.seq_len,
        label_len=0,
        pred_len=cli.pred_len,
        batch_size=cli.batch_size,
        num_workers=0,
        seasonal_patterns=None,
        # 'raw' makes Stage1WindowDataset hand back inverse-transformed futures,
        # which is the literal reading of a raw-future-MSE oracle. Per channel
        # this is an affine rescale, so the oracle top-K is unchanged either way.
        teacher_mse_space='raw',
        task_name='stage1_relation',
        # data_loader reads this on the train split; augmentation must stay off
        # or the candidate pool would differ between configurations.
        augmentation_ratio=0,
    )
    return ns


def load_split(args, flag):
    dataset, _ = data_provider(args, flag=flag, shuffle=False)
    return dataset


def channel_names_of(dataset, channels):
    names = getattr(dataset, 'channel_names', None)
    if names is not None and len(names) == channels:
        return list(names)
    return [f'ch{i}' for i in range(channels)]


@torch.no_grad()
def encode_parts(parts, encoder, device):
    """identity returns the representation unchanged.

    For encoder='mlp' each part gets its own encoder (trend and seasonal do not
    share weights), which is why this takes a list of encoders.
    """
    if encoder is None:
        return list(parts)
    return [enc(part.to(device)) for enc, part in zip(encoder, parts)]


@torch.no_grad()
def evaluate(cli):
    device = torch.device(cli.device if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)

    args = build_args(cli)
    # task_name='stage1_relation' makes data_provider hand back a
    # Stage1WindowDataset already, so it must not be wrapped a second time.
    train_ds = load_split(args, 'train')
    test_ds = load_split(args, 'test')
    if not isinstance(train_ds, Stage1WindowDataset):
        train_ds = Stage1WindowDataset(train_ds)
        test_ds = Stage1WindowDataset(test_ds)
    train_raw = train_ds.base_dataset

    sampler = RelationMemorySampler(
        train_ds,
        seq_len=cli.seq_len,
        pred_len=cli.pred_len,
        mask_mode=cli.candidate_mask,
    )
    memory_x = torch.from_numpy(sampler.memory_x).float()      # [N, L, C]
    memory_y = torch.from_numpy(sampler.memory_y).float()      # [N, H, C]
    channels = memory_x.size(-1)
    names = channel_names_of(train_raw, channels)

    # Queries: every test window, in order. Same set for all 192 configs.
    q_starts = test_ds.get_all_valid_starts()
    query_x = []
    query_y = []
    for row in range(len(test_ds)):
        seq_x, future, _ = test_ds[row]
        query_x.append(np.asarray(seq_x, dtype=np.float32))
        query_y.append(np.asarray(future, dtype=np.float32))
    query_x = torch.from_numpy(np.stack(query_x))              # [B, L, C]
    query_y = torch.from_numpy(np.stack(query_y))              # [B, H, C]

    # Two scales, deliberately kept apart:
    #   memory_y_normalized -> what the MLP trains on, matching the pipeline
    #   memory_y            -> raw scale, and only the oracle reads it
    # query_y already arrives inverse-transformed because teacher_mse_space is
    # 'raw', so the oracle compares raw futures against raw futures.
    memory_y_normalized = memory_y.clone()
    if getattr(train_raw, 'scale', False) and args.teacher_mse_space == 'raw':
        flat = memory_y.reshape(-1, channels).numpy()
        memory_y = torch.from_numpy(
            train_raw.inverse_transform(flat).astype(np.float32)
        ).reshape(memory_y.shape)

    # Seasonal lag for sarima_residual, 1 for the plain arima_residual.
    period = cli.period if cli.period > 0 else DEFAULT_PERIOD.get(cli.freq, 24)
    ar_lag = period if cli.representation == 'sarima_residual' else 1

    ar_params = None
    if cli.representation in ('arima_residual', 'sarima_residual'):
        # Global per channel, fitted on the training series only, on the same
        # difference the representation uses.
        series = np.asarray(train_ds.data_x, dtype=np.float64)
        ar_params = [
            fit_ar1_on_differences(series[:, c], lag=ar_lag)
            for c in range(channels)
        ]

    # One encoder set per channel, trained (or loaded) before scoring starts.
    channel_encoders = None
    if cli.encoder == 'mlp':
        channel_encoders = get_mlp_encoders(
            cli, memory_x, memory_y_normalized, sampler, channels,
            ar_params, device,
        )

    n_candidates = memory_x.size(0)
    per_channel = {k: [] for k in KS}
    per_channel_named = {}

    for c in range(channels):
        mem_parts = transform(
            memory_x[:, :, c],
            cli.representation,
            ar_params=None if ar_params is None else ar_params[c],
            moving_avg=cli.moving_avg,
            period=period,
        )
        mem_parts = [p.to(device) for p in mem_parts]
        encoders = None if channel_encoders is None else channel_encoders[c]
        mem_parts = encode_parts(mem_parts, encoders, device)
        mem_future = memory_y[:, :, c].to(device)

        chan_recall = {k: [] for k in KS}
        for start in range(0, query_x.size(0), cli.query_chunk):
            stop = min(start + cli.query_chunk, query_x.size(0))
            qx = query_x[start:stop, :, c].to(device)
            qy = query_y[start:stop, :, c].to(device)

            q_parts = transform(
                qx,
                cli.representation,
                ar_params=None if ar_params is None else ar_params[c],
                moving_avg=cli.moving_avg,
                period=period,
            )
            q_parts = encode_parts(list(q_parts), encoders, device)

            mask_np, _ = sampler.valid_mask_batch(q_starts[start:stop])
            valid = mask_np.bool().to(device)

            part_scores = [
                similarity_scores(qp, mp, cli.similarity)
                for qp, mp in zip(q_parts, mem_parts)
            ]
            scores = (
                part_scores[0]
                if len(part_scores) == 1
                else fuse_decomposition_scores(part_scores, valid)
            )
            oracle = raw_future_mse(qy, mem_future)
            got = recall_at_k(scores, oracle, valid, ks=KS)
            for k in KS:
                chan_recall[k].append(got[k].cpu())

        merged = {k: torch.cat(chan_recall[k]) for k in KS}
        for k in KS:
            per_channel[k].append(merged[k].mean().item())
        per_channel_named[names[c]] = {k: merged[k].mean().item() for k in KS}

    result = {
        'dataset': cli.data,
        'seq_len': cli.seq_len,
        'pred_len': cli.pred_len,
        'representation': cli.representation,
        'encoder': cli.encoder,
        'similarity': cli.similarity,
        'n_queries': int(query_x.size(0)),
        'n_candidates': int(n_candidates),
        'seed': cli.seed,
    }
    if cli.representation == 'sarima_residual':
        result['period'] = period
    # Only written when it is not the default, so a run keeps appending to a CSV
    # opened before the teacher axis existed without shifting its columns. A row
    # with no teacher column is an ema_future row.
    if cli.teacher != 'ema_future':
        result['teacher'] = cli.teacher
    for k in KS:
        result[f'recall_at_{k}'] = float(np.mean(per_channel[k]))
    for name, vals in per_channel_named.items():
        for k in KS:
            result[f'recall_at_{k}_{name}'] = vals[k]
    return result


def get_mlp_encoders(cli, memory_x, memory_y_normalized, sampler, channels,
                     ar_params, device):
    """Train (or reuse) one encoder set per channel for this configuration.

    Checkpoints are keyed by dataset/length/representation/similarity/seed, so a
    re-run reuses the encoders instead of retraining them. The encoder is fitted
    per similarity because the KL scores candidates with the same similarity the
    retrieval uses -- training and retrieval stay under one metric.
    """
    root = cli.mlp_ckpt or os.path.join('.', 'checkpoints', 'retrieval_recall')
    os.makedirs(root, exist_ok=True)
    path = checkpoint_path(
        root, cli.data, cli.seq_len, cli.representation, cli.similarity, cli.seed,
        teacher=cli.teacher,
    )

    if os.path.exists(path) and not cli.retrain:
        state = torch.load(path, map_location=device)
        out = []
        for c in range(channels):
            encs = []
            for sd in state[c]:
                enc = RelationMLP(sd['input_dim'], cli.d_model, cli.d_ff).to(device)
                enc.load_state_dict(sd['state'])
                enc.eval()
                encs.append(enc)
            out.append(encs)
        print(f'[recall] loaded MLP encoders from {path}', flush=True)
        return out

    prefix = (
        f'[train] {cli.data} L={cli.seq_len} '
        f'{cli.representation}/{cli.similarity} teacher={cli.teacher}'
    )
    out = []
    for c in range(channels):
        # evaluate() is wrapped in torch.no_grad(); training has to opt back in.
        with torch.enable_grad():
            encs = train_encoders(
                memory_x=memory_x,
                memory_y=memory_y_normalized,
                sampler=sampler,
                channel=c,
                representation=cli.representation,
                similarity=cli.similarity,
                ar_params=None if ar_params is None else ar_params[c],
                moving_avg=cli.moving_avg,
                device=device,
                seq_len=cli.seq_len,
                epochs=cli.train_epochs,
                lr=cli.learning_rate,
                batch_size=cli.batch_size,
                d_model=cli.d_model,
                d_ff=cli.d_ff,
                teacher=cli.teacher,
                log_every=cli.log_every,
                log_prefix=prefix,
            )
        out.append(encs)

    torch.save(
        [
            [
                {
                    'input_dim': enc.net[0].in_features,
                    'state': {k: v.cpu() for k, v in enc.state_dict().items()},
                }
                for enc in encs
            ]
            for encs in out
        ],
        path,
    )
    print(f'[recall] saved MLP encoders to {path}', flush=True)
    return out


def append_csv(path, row):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    exists = os.path.exists(path)
    # Per-channel columns vary by dataset, so the union is written per file and
    # the header is fixed on first write. An appended row whose keys disagree
    # with that header would be written in the wrong column order, so it is
    # refused instead: a loud failure on one configuration beats a CSV that
    # silently stops meaning what its header says.
    fieldnames = list(row.keys())
    if exists:
        with open(path, newline='', encoding='utf-8') as handle:
            header = next(csv.reader(handle), None)
        if header and set(header) != set(fieldnames):
            raise ValueError(
                f'row columns do not match the header of {path}: '
                f'missing={sorted(set(header) - set(fieldnames))} '
                f'extra={sorted(set(fieldnames) - set(header))}'
            )
        if header:
            fieldnames = header
    with open(path, 'a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--root_path', required=True)
    p.add_argument('--data_path', required=True)
    p.add_argument('--freq', default='h')
    p.add_argument('--seq_len', type=int, required=True)
    p.add_argument('--pred_len', type=int, required=True)
    p.add_argument('--representation', required=True, choices=REPRESENTATIONS)
    p.add_argument('--encoder', default='identity', choices=('identity', 'mlp'))
    p.add_argument('--teacher', default='ema_future', choices=TEACHERS,
                   help=('KL target the mlp encoder trains against; ignored by '
                         'encoder=identity, which trains nothing'))
    p.add_argument('--similarity', required=True, choices=SIMILARITIES)
    p.add_argument('--moving_avg', type=int, default=25)
    p.add_argument('--period', type=int, default=0,
                   help='seasonal lag for sarima_residual; 0 picks it from --freq')
    p.add_argument('--candidate_mask', default='raft')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--query_chunk', type=int, default=256)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--mlp_ckpt', default='')
    p.add_argument('--retrain', action='store_true',
                   help='retrain MLP encoders even if a checkpoint exists')
    p.add_argument('--train_epochs', type=int, default=10)
    p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument('--d_model', type=int, default=128)
    p.add_argument('--d_ff', type=int, default=256)
    p.add_argument('--log_every', type=int, default=100)
    p.add_argument('--output', default='./metrics/retrieval_recall/results.csv')
    cli = p.parse_args()
    if cli.encoder == 'identity' and cli.teacher != 'ema_future':
        p.error('--teacher only applies to --encoder mlp; identity trains nothing')

    started = time.time()
    row = evaluate(cli)
    row['seconds'] = round(time.time() - started, 2)
    append_csv(cli.output, row)
    print(
        f"[recall] {row['dataset']} L={row['seq_len']} "
        f"{row['representation']}/{row['encoder']}/{row['similarity']}"
        f"{'' if cli.teacher == 'ema_future' else '/' + cli.teacher} "
        f"R@1={row['recall_at_1']:.4f} R@5={row['recall_at_5']:.4f} "
        f"R@10={row['recall_at_10']:.4f} "
        f"(queries={row['n_queries']} candidates={row['n_candidates']} "
        f"{row['seconds']}s)"
    )


if __name__ == '__main__':
    main()
