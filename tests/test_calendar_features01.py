"""Unit tests for utils/calendar_features.py -- TRACK-V-CALENDAR-ROUTER01
spec section 20, Test 9 (forecast-start timestamp mapping) plus basic
correctness of the 6-D cyclic feature itself."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.calendar_features import (
    N_CALENDAR_FEATURES, cyclic_features_from_datetimeindex,
    forecast_start_calendar_features, load_date_column, shuffle_calendar_features,
)

ETTH1_ROOT = '../Dataset/Time-Series-Library_dataset/ETT-small/'
ETTH1_PATH = 'ETTh1.csv'
WEATHER_ROOT = '../Dataset/Time-Series-Library_dataset/weather/'
WEATHER_PATH = 'weather.csv'


def _skip_if_missing(root, path):
    if not (Path(REPO_ROOT) / root / path).exists():
        pytest.skip(f'dataset file not found: {root}{path}')


def test_cyclic_feature_hand_computable_midnight_sunday_jan1():
    dates = pd.DatetimeIndex(['2021-01-03 00:00:00'])  # a Sunday, Jan 3 (doy=3)
    feats = cyclic_features_from_datetimeindex(dates)
    assert feats.shape == (1, N_CALENDAR_FEATURES)
    # tod=0 -> sin=0, cos=1
    assert math.isclose(feats[0, 0], 0.0, abs_tol=1e-6)
    assert math.isclose(feats[0, 1], 1.0, abs_tol=1e-6)
    # dow: Sunday -> pandas dayofweek=6 -> dow=6/7
    expected_dow = 6 / 7.0
    assert math.isclose(feats[0, 2], math.sin(2 * math.pi * expected_dow), abs_tol=1e-5)
    assert math.isclose(feats[0, 3], math.cos(2 * math.pi * expected_dow), abs_tol=1e-5)


def test_cyclic_feature_bounded():
    dates = pd.date_range('2020-01-01', periods=500, freq='37min')
    feats = cyclic_features_from_datetimeindex(dates)
    assert feats.shape == (500, 6)
    assert np.all(feats >= -1.0 - 1e-6) and np.all(feats <= 1.0 + 1e-6)


def test_forecast_start_mapping_boundary_etth1():
    _skip_if_missing(ETTH1_ROOT, ETTH1_PATH)
    dates = load_date_column(str(Path(REPO_ROOT) / ETTH1_ROOT), ETTH1_PATH)
    seq_len = 96
    # first possible query (start=0), last possible query near the end
    query_idx = torch.tensor([0, len(dates) - seq_len - 1])
    feats = forecast_start_calendar_features(dates, query_idx, seq_len)
    assert feats.shape == (2, 6)
    # directly re-derive the SAME forecast row and compare feature-for-feature
    expected_row0 = dates.iloc[0 + seq_len]
    expected_feat0 = cyclic_features_from_datetimeindex(pd.DatetimeIndex([expected_row0]))[0]
    assert np.allclose(feats[0].numpy(), expected_feat0, atol=1e-6)


def test_forecast_start_mapping_out_of_range_raises():
    dates = pd.date_range('2020-01-01', periods=10, freq='h')
    with pytest.raises(ValueError):
        forecast_start_calendar_features(dates, torch.tensor([8]), seq_len=5)  # 8+5=13 >= 10


def test_shuffle_is_pure_reorder():
    torch.manual_seed(0)
    feats = torch.randn(20, 6)
    shuffled, perm = shuffle_calendar_features(feats, seed=0)
    assert shuffled.shape == feats.shape
    for i in range(20):
        assert torch.allclose(shuffled[i], feats[perm[i]])
    assert not np.array_equal(perm, np.arange(20))
