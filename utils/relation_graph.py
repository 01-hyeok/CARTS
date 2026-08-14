import csv
import json
import os

import numpy as np


def relation_graph_enabled(args):
    mode = getattr(args, 'source_mode', 'auto')
    if mode == 'all':
        return False
    if mode in ('auto', 'topk_corr'):
        return True
    raise ValueError(f'Unsupported source_mode: {mode}')


def default_relation_graph_path(args):
    dataset_name = os.path.splitext(os.path.basename(args.data_path))[0]
    top_n = int(getattr(args, 'relation_top_n', 3))
    return os.path.join(
        getattr(args, 'metrics_csv_dir', './metrics'),
        'relation_graphs',
        dataset_name,
        f'pearson_self_top{top_n}.json',
    )


def _channel_names(train_dataset, channels):
    names = getattr(train_dataset, 'channel_names', None)
    if names is None and hasattr(train_dataset, 'base_dataset'):
        names = getattr(train_dataset.base_dataset, 'channel_names', None)
    if names is None:
        return [f'ch{i}' for i in range(channels)]
    names = [str(name) for name in names]
    if len(names) != channels:
        raise ValueError(
            f'channel name count mismatch: names={len(names)} channels={channels}'
        )
    return names


def _validate_graph(graph, args):
    channels = int(args.enc_in)
    top_n = min(int(getattr(args, 'relation_top_n', 3)), channels)
    if top_n <= 0:
        raise ValueError('relation_top_n must include at least the self source')
    if int(graph.get('version', -1)) != 2:
        raise ValueError(
            'relation graph uses the old non-self Top-N format; rebuild it with '
            'relation_top_n interpreted as the total source count including self'
        )
    if int(graph.get('channels', -1)) != channels:
        raise ValueError(
            f'relation graph channel mismatch: graph={graph.get("channels")} current={channels}'
        )
    if int(graph.get('top_n', -1)) != top_n:
        raise ValueError(
            f'relation graph top_n mismatch: graph={graph.get("top_n")} current={top_n}'
        )
    if os.path.basename(str(graph.get('data_path', ''))) != os.path.basename(str(args.data_path)):
        raise ValueError(
            f'relation graph dataset mismatch: graph={graph.get("data_path")} '
            f'current={args.data_path}'
        )
    sources = graph.get('sources')
    correlations = graph.get('correlations')
    if len(sources or []) != channels or len(correlations or []) != channels:
        raise ValueError('relation graph must contain one source/correlation row per target channel')
    expected_width = top_n
    for target, (source_row, corr_row) in enumerate(zip(sources, correlations)):
        if len(source_row) != expected_width or len(corr_row) != expected_width:
            raise ValueError(
                f'invalid relation graph row width for target={target}: '
                f'sources={len(source_row)} correlations={len(corr_row)} expected={expected_width}'
            )
        if int(source_row[0]) != target:
            raise ValueError(f'relation graph target={target} must keep self relation in slot 0')


def _write_graph_csv(graph, json_path):
    csv_path = os.path.splitext(json_path)[0] + '.csv'
    csv_dir = os.path.dirname(csv_path)
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)
    names = graph['channel_names']
    with open(csv_path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            'target_index', 'target_channel', 'source_rank',
            'source_index', 'source_channel', 'is_self',
            'pearson', 'abs_pearson',
        ])
        writer.writeheader()
        for target, (sources, correlations) in enumerate(
            zip(graph['sources'], graph['correlations'])
        ):
            for rank, (source, corr) in enumerate(zip(sources, correlations)):
                writer.writerow({
                    'target_index': target,
                    'target_channel': names[target],
                    'source_rank': rank,
                    'source_index': source,
                    'source_channel': names[source],
                    'is_self': int(source == target),
                    'pearson': float(corr),
                    'abs_pearson': abs(float(corr)),
                })
    return csv_path


def load_or_build_relation_graph(train_dataset, args, require_existing=False):
    if not relation_graph_enabled(args):
        return None

    channels = int(args.enc_in)
    top_n = min(int(getattr(args, 'relation_top_n', 3)), channels)
    if top_n <= 0:
        raise ValueError('relation_top_n must include at least the self source')

    graph_path = getattr(args, 'relation_graph_path', '') or default_relation_graph_path(args)
    args.relation_graph_path = graph_path
    if os.path.exists(graph_path):
        with open(graph_path, 'r') as handle:
            graph = json.load(handle)
        _validate_graph(graph, args)
        csv_path = _write_graph_csv(graph, graph_path)
        print(f'[relation_graph] loaded {graph_path}')
        print(f'[relation_graph] metrics {csv_path}')
        return graph

    if require_existing:
        raise FileNotFoundError(
            f'Stage2 requires the Stage1 relation graph, but it does not exist: {graph_path}'
        )

    values = np.asarray(train_dataset.data_x, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != channels:
        raise ValueError(
            f'expected train data [T, C] with C={channels}, got {values.shape}'
        )
    with np.errstate(invalid='ignore', divide='ignore'):
        corr = np.corrcoef(values, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)

    sources = []
    correlations = []
    for target in range(channels):
        abs_row = np.abs(corr[target]).copy()
        abs_row[target] = -np.inf
        ranked = np.argsort(-abs_row, kind='stable')[:max(top_n - 1, 0)].tolist()
        source_row = [target] + [int(source) for source in ranked]
        sources.append(source_row)
        correlations.append([float(corr[target, source]) for source in source_row])

    graph = {
        'version': 2,
        'method': 'absolute_pearson',
        'data_path': str(args.data_path),
        'channels': channels,
        'top_n': top_n,
        'threshold': int(getattr(args, 'relation_graph_threshold', 21)),
        'channel_names': _channel_names(train_dataset, channels),
        'sources': sources,
        'correlations': correlations,
    }
    _validate_graph(graph, args)
    graph_dir = os.path.dirname(graph_path)
    if graph_dir:
        os.makedirs(graph_dir, exist_ok=True)
    tmp_path = graph_path + '.tmp'
    with open(tmp_path, 'w') as handle:
        json.dump(graph, handle, indent=2)
    os.replace(tmp_path, graph_path)
    csv_path = _write_graph_csv(graph, graph_path)
    print(f'[relation_graph] built {graph_path}')
    print(f'[relation_graph] metrics {csv_path}')
    return graph
