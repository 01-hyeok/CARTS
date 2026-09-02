#!/usr/bin/env python3
"""Build the shared candidate pool and its three teacher scores, once per setting.

Everything downstream -- the alignment table, the teacher-distribution study and
the Stage-1 ablation training -- reads this cache, so all arms provably rank the
same candidate ids and the expensive utility measurement is paid for once.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.retrieval_diagnostics import load_stage2  # noqa: E402
from utils.utility_teacher import build_teacher_cache, save_cache  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='reference Stage-2 checkpoint')
    parser.add_argument('--out_dir', default='./cache/utility_teacher')
    parser.add_argument('--splits', default='train,val,test')
    parser.add_argument('--pool_m', type=int, default=100)
    parser.add_argument('--candidate_chunk', type=int, default=25)
    parser.add_argument('--max_queries', type=int, default=0)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    experiment, saved = load_stage2(args.checkpoint)
    experiment.reference_checkpoint = args.checkpoint
    out_dir = Path(args.out_dir) / f'{saved.data}_pred{saved.pred_len}_m{args.pool_m}'

    for split in args.splits.split(','):
        split = split.strip()
        target = out_dir / f'{split}.pt'
        if target.exists() and not args.force:
            print(f'[skip] {target} already exists')
            continue
        cache = build_teacher_cache(
            experiment, split, pool_m=args.pool_m,
            candidate_chunk=args.candidate_chunk, max_queries=args.max_queries,
        )
        save_cache(cache, target)
        utility = cache['utility']
        print(f'[done] {target} queries={cache["meta"]["queries"]} '
              f'pool={cache["meta"]["pool_m"]} '
              f'positive_rate={float((utility > 0).float().mean()):.4f} '
              f'best_utility={float(utility.max(-1).values.mean()):+.5f}')


if __name__ == '__main__':
    main()
