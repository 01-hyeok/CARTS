"""Calendar-Router feature extraction for TRACK-V-CALENDAR-ROUTER01.

Reads the ACTUAL source CSV's own `date` column directly (never derives
timestamps from `args.freq`/`time_features`, and never feeds raw
timestamp integers, absolute year, or a learned date embedding into
anything) and produces a fixed 6-D cyclic feature:

    [sin(tod), cos(tod), sin(dow), cos(dow), sin(doy), cos(doy)]

tod = (hour*60+minute)/1440, dow = dayofweek/7, doy = dayofyear/365.2425.

Global row index convention: `query_start_idx` (as already used
throughout this project's `Stage1WindowDataset.starts` /
`RelationMemorySampler` -- 0-indexed into the FULL, un-split CSV, since
`border1` is already added back in) + `seq_len` = the index of the
FIRST forecast-horizon row, i.e. the "forecast start." This file reads
that row's `date` value directly from the raw CSV, never from a
re-split/re-indexed dataframe.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

N_CALENDAR_FEATURES = 6


def load_date_column(root_path, data_path):
    """Returns the FULL csv's `date` column as a pandas DatetimeIndex,
    0-indexed exactly as `query_start_idx`/`memory` indices already are
    throughout this project (no border slicing here -- that happens
    only inside `Dataset_ETT_hour`/`Dataset_Custom`, never in this
    calendar-feature path)."""
    csv_path = Path(root_path) / data_path
    df = pd.read_csv(csv_path)
    date_col = df.columns[0]
    return pd.to_datetime(df[date_col])


def cyclic_features_from_datetimeindex(dates):
    """dates: pandas Series/DatetimeIndex of length M. Returns a
    [M, 6] float32 numpy array -- the fixed cyclic feature, never any
    other representation of calendar time."""
    dt = pd.DatetimeIndex(dates)
    minute_of_day = dt.hour.values * 60 + dt.minute.values
    tod = minute_of_day / 1440.0
    dow = dt.dayofweek.values / 7.0
    doy = (dt.dayofyear.values - 1) / 365.2425
    two_pi = 2 * math.pi
    feats = np.stack([
        np.sin(two_pi * tod), np.cos(two_pi * tod),
        np.sin(two_pi * dow), np.cos(two_pi * dow),
        np.sin(two_pi * doy), np.cos(two_pi * doy),
    ], axis=-1).astype(np.float32)
    return feats


def forecast_start_calendar_features(date_column, query_start_idx, seq_len):
    """`date_column`: full-CSV DatetimeIndex/Series from `load_date_column`.
    `query_start_idx`: 1-D LongTensor or array of GLOBAL start indices
    (the project's own convention). `seq_len`: int. Returns a
    [len(query_start_idx), 6] float32 tensor -- the forecast-start
    calendar feature for every query, computed with
    `forecast_start_idx = query_start_idx + seq_len` directly against
    the raw CSV's own date rows."""
    if torch.is_tensor(query_start_idx):
        idx_np = query_start_idx.detach().cpu().numpy()
    else:
        idx_np = np.asarray(query_start_idx)
    forecast_start_idx = idx_np + int(seq_len)
    n_rows = len(date_column)
    if forecast_start_idx.max() >= n_rows or forecast_start_idx.min() < 0:
        raise ValueError(
            f'[ISSUE][ABORT] forecast_start_idx out of range: '
            f'min={forecast_start_idx.min()} max={forecast_start_idx.max()} n_rows={n_rows}')
    dates = pd.DatetimeIndex(date_column).to_numpy()[forecast_start_idx]
    feats = cyclic_features_from_datetimeindex(pd.DatetimeIndex(dates))
    return torch.from_numpy(feats)


def shuffle_calendar_features(features, seed=0):
    """Deterministic row-permutation of an already-computed [Q, 6]
    calendar-feature tensor -- the CRH-Shuffled control (spec section
    15.E): breaks query<->calendar correspondence while keeping the
    SAME marginal distribution of calendar feature values, exactly
    mirroring this project's existing shuffled-timestamp convention
    (TRACK-W-TIMESTAMP-FUSION01's C2 arm)."""
    n = features.size(0)
    perm = np.random.default_rng(seed).permutation(n)
    return features[torch.as_tensor(perm, dtype=torch.long)], perm
