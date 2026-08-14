import pytest

from utils.stage1_metrics import MetricAverager


def test_metric_averager_weights_batches_by_sample_count():
    average = MetricAverager()
    average.update({'final_mse': 1.0}, weight=32)
    average.update({'final_mse': 3.0}, weight=8)

    assert average.average()['final_mse'] == pytest.approx(1.4)


def test_metric_averager_tracks_each_metric_denominator_independently():
    average = MetricAverager()
    average.update({'final_mse': 1.0}, weight=4)
    average.update({'skipped_batches': 1.0}, weight=2)

    metrics = average.average()
    assert metrics['final_mse'] == pytest.approx(1.0)
    assert metrics['skipped_batches'] == pytest.approx(1.0)
