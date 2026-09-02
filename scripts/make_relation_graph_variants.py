"""Build source-selection control graphs from an existing Pearson relation graph.

Two controls answer whether concatenating a correlated source channel helps:

  self   sources = [target]                 -> is a source channel useful at all?
  random sources = [target, k random peers] -> is the *correlated* source better
                                               than an arbitrary one?

The self variant has top_n=1, so runs using it must pass --relation_top_n 1.
The random variant keeps the original top_n so it is directly comparable with
the correlation-selected baseline.
"""

import argparse
import json
import os
import random


def load_graph(path):
    with open(path, 'r') as handle:
        return json.load(handle)


def self_only(graph):
    channels = int(graph['channels'])
    out = dict(graph)
    out['top_n'] = 1
    out['method'] = 'self_only_control'
    out['sources'] = [[target] for target in range(channels)]
    out['correlations'] = [[1.0] for _ in range(channels)]
    return out


def random_sources(graph, seed):
    channels = int(graph['channels'])
    top_n = int(graph['top_n'])
    rng = random.Random(seed)
    base_corr = graph['correlations']
    sources, correlations = [], []
    for target in range(channels):
        peers = [c for c in range(channels) if c != target]
        picked = rng.sample(peers, k=min(top_n - 1, len(peers)))
        sources.append([target] + picked)
        # Correlations are logged for analysis only; keep the self entry exact
        # and fill peers with the value the base graph recorded for them when
        # available so downstream CSVs stay interpretable.
        row = [1.0]
        for peer in picked:
            known = None
            for src, corr in zip(graph['sources'][target], base_corr[target]):
                if int(src) == peer:
                    known = float(corr)
                    break
            row.append(known if known is not None else 0.0)
        correlations.append(row)
    out = dict(graph)
    out['method'] = f'random_source_control_seed{seed}'
    out['sources'] = sources
    out['correlations'] = correlations
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base_graph', required=True,
                        help='Existing pearson_self_top{N}.json to derive from')
    parser.add_argument('--out_dir', required=True,
                        help='Directory the variant graphs are written to')
    parser.add_argument('--random_seeds', type=int, nargs='+', default=[0],
                        help='Seeds used for the random-source control')
    args = parser.parse_args()

    graph = load_graph(args.base_graph)
    os.makedirs(args.out_dir, exist_ok=True)

    written = []

    path = os.path.join(args.out_dir, 'self_only_top1.json')
    with open(path, 'w') as handle:
        json.dump(self_only(graph), handle, indent=2)
    written.append(path)

    for seed in args.random_seeds:
        variant = random_sources(graph, seed)
        path = os.path.join(
            args.out_dir, f'random_source_top{variant["top_n"]}_seed{seed}.json'
        )
        with open(path, 'w') as handle:
            json.dump(variant, handle, indent=2)
        written.append(path)

    for path in written:
        print(f'[graph_variant] wrote {path}')


if __name__ == '__main__':
    main()
