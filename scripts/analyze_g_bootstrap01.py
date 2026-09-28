"""
TRACK-G-DECOUPLED-METRIC-ADAPTATION01 -- bootstrap analysis.

Reads per-query CSVs already produced by train_g_decoupled_metric_adaptation01.py
(test split; columns: query_start_idx, channel, individual_retmse10, ...).

Computes, for each requested pair (arm_a, arm_b), per seed:
  - cluster bootstrap (resample query_start_idx, all 7 channels move together),
    10000 reps, on individual_retmse10 mean(A) - mean(B)
  - moving-block bootstrap at block lengths {96, 240, 720} on the same statistic,
    blocks defined over the sorted unique query_start_idx sequence (contiguous
    query starts), sampled with replacement to reconstruct a resample of the
    same total number of query_start_idx values, all 7 channels of a given
    query_start_idx always move together.

Does not touch any existing result file; writes to a new file only:
  results/TRACK-G-DECOUPLED-METRIC-ADAPTATION01/ETTh1_720/bootstrap_summary.json
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

RESULT_DIR = Path("results/TRACK-G-DECOUPLED-METRIC-ADAPTATION01/ETTh1_720")
SEEDS = (0, 1, 2)
N_REPS = 10000
BLOCK_LENGTHS = (96, 240, 720)
RNG_SEED_BASE = 20260928

PAIRS = [
    ("A0_frozen_final_cosine", "A1_frozen_final_freshasym"),
    ("A1_frozen_final_freshasym", "A2_frozen_rawcls_freshasym"),
    ("A1_frozen_final_freshasym", "A3_frozen_patchmean_freshasym"),
    ("A1_frozen_final_freshasym", "A4_continue_encoder_nohead"),
    ("A1_frozen_final_freshasym", "A5_joint_encoder_freshasym"),
]


METRIC_COLUMNS = ("individual_retmse10", "uniform_aggregate_mse10")


def load_query_means(arm, seed, column="individual_retmse10"):
    """Return a Series indexed by query_start_idx: mean of `column` over the 7 channels."""
    f = RESULT_DIR / f"per_query_{arm}_seed{seed}.csv"
    df = pd.read_csv(f)
    g = df.groupby("query_start_idx")[column].mean().sort_index()
    return g


def cluster_bootstrap(vals_a, vals_b, rng, n_reps=N_REPS):
    """vals_a, vals_b: 1D arrays aligned by query_start_idx (same index order)."""
    n = len(vals_a)
    diffs = np.empty(n_reps)
    for i in range(n_reps):
        idx = rng.integers(0, n, size=n)
        diffs[i] = vals_a[idx].mean() - vals_b[idx].mean()
    return diffs


def moving_block_bootstrap(vals_a, vals_b, block_len, rng, n_reps=N_REPS):
    """Resample contiguous blocks (in query_start_idx order) with replacement
    until reaching >= n queries, then truncate to n."""
    n = len(vals_a)
    n_blocks_needed = int(np.ceil(n / block_len))
    max_start = n - block_len
    if max_start < 0:
        # series shorter than block: fall back to whole-series block
        block_len = n
        max_start = 0
        n_blocks_needed = 1
    diffs = np.empty(n_reps)
    for i in range(n_reps):
        starts = rng.integers(0, max_start + 1, size=n_blocks_needed)
        idx = np.concatenate([np.arange(s, s + block_len) for s in starts])[:n]
        diffs[i] = vals_a[idx].mean() - vals_b[idx].mean()
    return diffs


def ci_from_diffs(diffs):
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {
        "mean_diff": float(diffs.mean()),
        "ci_2.5": float(lo),
        "ci_97.5": float(hi),
        "ci_excludes_zero": bool(lo > 0 or hi < 0),
        "favors": ("A" if diffs.mean() < 0 else "B"),  # diff = A - B ; A better means A has lower retMSE -> diff<0
    }


def main():
    summary = {}
    for column in METRIC_COLUMNS:
        summary[column] = {}
        for arm_a, arm_b in PAIRS:
            pair_key = f"{arm_a}__vs__{arm_b}"
            summary[column][pair_key] = {"cluster_bootstrap": {}, "moving_block_bootstrap": {}}
            for seed in SEEDS:
                ga = load_query_means(arm_a, seed, column)
                gb = load_query_means(arm_b, seed, column)
                common_idx = ga.index.intersection(gb.index)
                assert len(common_idx) == len(ga) == len(gb), (
                    f"query_start_idx mismatch for {arm_a} vs {arm_b} seed {seed}: "
                    f"{len(ga)} vs {len(gb)} vs {len(common_idx)} common"
                )
                vals_a = ga.loc[common_idx].to_numpy()
                vals_b = gb.loc[common_idx].to_numpy()

                rng = np.random.default_rng(RNG_SEED_BASE + seed)
                cdiffs = cluster_bootstrap(vals_a, vals_b, rng)
                summary[column][pair_key]["cluster_bootstrap"][f"seed{seed}"] = ci_from_diffs(cdiffs)

                mb_result = {}
                for bl in BLOCK_LENGTHS:
                    rng_mb = np.random.default_rng(RNG_SEED_BASE + seed + 1000 * bl)
                    mdiffs = moving_block_bootstrap(vals_a, vals_b, bl, rng_mb)
                    mb_result[f"block{bl}"] = ci_from_diffs(mdiffs)
                summary[column][pair_key]["moving_block_bootstrap"][f"seed{seed}"] = mb_result

                c = summary[column][pair_key]["cluster_bootstrap"][f"seed{seed}"]
                print(
                    f"[{column}][{pair_key}] seed{seed}: n_queries={len(common_idx)} "
                    f"cluster mean_diff={c['mean_diff']:.5f} "
                    f"CI=[{c['ci_2.5']:.5f}, {c['ci_97.5']:.5f}] excludes_zero={c['ci_excludes_zero']}"
                )

    out_path = RESULT_DIR / "bootstrap_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
