#!/usr/bin/env python3
"""TRACK-W-TIMESTAMP-FUSION01: shared library.

Reused UNMODIFIED from elsewhere in this repository (imported, not
reimplemented): `build_experiment` (`train_margutil01`), `set_global_seeds`
(`rng_control01`), `RelationEncoder` (`models.RelationStage1`, used here
ONLY to measure the historical architecture's parameter count for the
mandatory capacity-accounting log -- never used for any forward pass).

New in this module (TRACK-W-TIMESTAMP-FUSION01-specific):
`detect_cadence` (spec section 3 -- measure, never assume, the real
per-dataset timestamp interval), `build_timestamp_experiment` (wires up
an `Exp_Stage1_Relation` with the CORRECT per-dataset `freq` override and
builds `memory_x_mark` aligned to `memory_x` by construction -- see
`research/W-timestamp-fusion/AUDIT.md` PART 2.4), `encode_raw_timestamp`
(the one-line swap-in replacement for `train_factorial_e2e01.encode_raw`
that this track's whole design otherwise reuses verbatim everywhere
else), and the C2 (SHUFFLED-TIME) deterministic window-correspondence
permutation (AUDIT.md PART 2.6).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.RelationStage1 import RelationEncoder
from scripts.rng_control01 import set_global_seeds
from scripts.train_margutil01 import build_experiment

# Per-dataset cadence: measured once (see detect_cadence) and asserted
# against this expectation every run -- never assumed from run.py's
# freq='h' global default. See AUDIT.md PART 1.A.6.
EXPECTED_CADENCE = {
    'ETTh1': pd.Timedelta(hours=1),
    'custom': pd.Timedelta(minutes=10),  # Weather, data='custom'
}
FREQ_OVERRIDE = {
    'ETTh1': 'h',
    'custom': '10min',
}


def detect_cadence(root_path, data_path):
    """Measure the median timestamp interval directly from the CSV --
    spec section 3's explicit instruction never to assume freq='h'."""
    csv_path = Path(root_path) / data_path
    df = pd.read_csv(csv_path)
    date_col = df.columns[0]
    dt = pd.to_datetime(df[date_col])
    deltas = dt.diff().dropna()
    median_interval = deltas.median()
    return median_interval, len(df)


def build_timestamp_experiment(cli, device):
    """Mirrors `train_j_shared_encoder_drift01.build_model`'s override
    dict exactly, PLUS the per-dataset `freq` correction, PLUS building
    `exp.memory_x_mark` (new attribute, additive -- no other code reads
    it) aligned to `exp.memory_x` by construction (AUDIT.md PART 2.4)."""
    set_global_seeds(cli.init_seed)
    base_ckpt_args = torch.load(cli.reference_ckpt, map_location='cpu')['args']
    data_name = base_ckpt_args['data']
    root_path = base_ckpt_args['root_path']
    data_path = base_ckpt_args['data_path']

    median_interval, n_rows = detect_cadence(root_path, data_path)
    expected = EXPECTED_CADENCE[data_name]
    assert median_interval == expected, (
        f'[ISSUE][ABORT] measured cadence for {data_path} is {median_interval}, '
        f'expected {expected} -- refusing to proceed with a wrong freq')
    freq = FREQ_OVERRIDE[data_name]

    exp, args = build_experiment(cli.reference_ckpt, {
        'pred_len': cli.pred_len, 'seq_len': cli.seq_len, 'batch_size': cli.batch_size,
        'seed': cli.init_seed, 'top_k': cli.top_k,
        'relation_encoder_type': 'mlp', 'relation_self_fill': 'linear',
        'relation_input_space': 'delta_last', 'relation_teacher_space': 'delta_last',
        'relation_value_space': 'delta_last', 'candidate_mask': 'raft',
        'patch_len': cli.patch_len, 'stride': cli.patch_len,
        'freq': freq,
    })
    exp._include_time_mark_for_memory = True
    exp._ensure_memory()
    train_dataset = exp.train_data_for_memory
    data_stamp = np.asarray(train_dataset.data_stamp, dtype=np.float32)
    time_feat_dim = data_stamp.shape[-1]
    seq_len = int(args.seq_len)
    num_windows = len(train_dataset)
    # IDENTICAL operation RelationMemorySampler.__init__ applies to data_x
    # for memory_x, applied to data_stamp instead -- guarantees
    # memory_x_mark[i] corresponds to exactly the same window as
    # memory_x[i], by construction (verified by a dedicated unit test).
    memory_x_mark = np.lib.stride_tricks.sliding_window_view(
        data_stamp, seq_len, axis=0
    ).transpose(0, 2, 1)[:num_windows]
    exp.memory_x_mark = torch.from_numpy(memory_x_mark).float().to(device)
    exp.cadence_info = {
        'dataset': data_name, 'data_path': data_path, 'measured_median_interval': str(median_interval),
        'expected_interval': str(expected), 'freq_used': freq, 'time_feat_dim': time_feat_dim,
        'n_rows_in_csv': n_rows,
    }
    return exp, args, time_feat_dim


def encode_raw_timestamp(time_encoder, x, x_mark, c):
    """Drop-in analog of `train_factorial_e2e01.encode_raw(model, x, c)`,
    for the timestamp-fusion encoder: extracts channel c's own delta-last
    scalar sequence (byte-identical transform to `_transform_relation_feature`
    with feature='delta_last'), then runs it + the aligned timestamp
    window through `time_encoder`."""
    x_c = x[:, :, c]
    delta = x_c - x_c[:, -1:].detach()
    return time_encoder(delta, x_mark)


def relation_encoder_param_count(args):
    """Measures (not hand-derives) the HISTORICAL RelationEncoder's
    parameter count for the mandatory capacity-accounting log --
    instantiated once, used only for .numel() counting, never for any
    forward pass."""
    old_encoder = RelationEncoder(args)
    return sum(p.numel() for p in old_encoder.parameters())


class ShuffledMarkQueryWrapper(Dataset):
    """C2 (SHUFFLED-TIME), query side. Wraps an already-built TRAIN
    `Stage1WindowDataset(include_time_mark=True)` instance. Returns the
    SAME (seq_x, future, start_idx) as the base dataset, but with
    seq_x_mark replaced by the mark belonging to window `perm[index]`
    instead of `index`'s own -- a deterministic, seeded permutation over
    the train split's own window-index space (AUDIT.md PART 2.6). The
    SAME `perm` array must also be used to reorder `memory_x_mark`
    (candidate side) via `shuffled_memory_x_mark`, so a query at train
    index i and the candidate at train index i are shuffled to the same
    alternate window consistently."""

    def __init__(self, base_dataset_with_mark, perm):
        self.base = base_dataset_with_mark
        self.perm = perm
        assert len(perm) == len(base_dataset_with_mark)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        seq_x, future, start_idx, _ = self.base[index]
        _, _, _, shuffled_mark = self.base[int(self.perm[index])]
        return seq_x, future, start_idx, shuffled_mark


def make_shuffle_permutation(num_train_windows, seed=0):
    return np.random.default_rng(seed).permutation(num_train_windows)


def shuffled_memory_x_mark(memory_x_mark_real, perm):
    """Candidate-side counterpart to `ShuffledMarkQueryWrapper` -- same
    `perm`, applied as a fancy-index reorder along the candidate axis."""
    perm_t = torch.as_tensor(perm, dtype=torch.long, device=memory_x_mark_real.device)
    return memory_x_mark_real[perm_t]
