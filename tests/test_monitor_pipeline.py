import numpy as np

from src.monitoring.monitor_pipeline import check_endpoint_metrics, check_feature_drift


def make_entries(predictions, probability=0.9):
    return [{"prediction": p, "probability": probability} for p in predictions]


def test_check_endpoint_metrics_counts_predictions_by_class():
    entries = make_entries([1, 1, 0, 1, 0])

    metrics = check_endpoint_metrics(entries)

    assert metrics["invocation_count"] == 5
    assert metrics["disease_predictions"] == 3
    assert metrics["no_disease_predictions"] == 2
    assert metrics["avg_confidence"] == 0.9


def test_check_endpoint_metrics_handles_no_predictions():
    metrics = check_endpoint_metrics([])

    assert metrics["invocation_count"] == 0
    assert metrics["avg_confidence"] == 0.0


def test_check_feature_drift_reports_not_enough_samples():
    train_mean = np.zeros(3)
    train_std = np.ones(3)
    entries = [{"features": [0, 0, 0]} for _ in range(3)]

    drift_detected, drift_score = check_feature_drift(entries, train_mean, train_std, min_samples=10)

    assert drift_detected is False
    assert drift_score == 0.0


def test_check_feature_drift_is_not_flagged_when_features_match_training():
    train_mean = np.array([0.0, 0.0, 0.0])
    train_std = np.array([1.0, 1.0, 1.0])
    entries = [{"features": [0, 0, 0]} for _ in range(10)]

    drift_detected, _ = check_feature_drift(entries, train_mean, train_std, threshold=0.5, min_samples=10)

    assert drift_detected is False


def test_check_feature_drift_is_flagged_when_features_shift_far_from_training():
    train_mean = np.array([0.0, 0.0, 0.0])
    train_std = np.array([1.0, 1.0, 1.0])
    entries = [{"features": [5, 5, 5]} for _ in range(10)]

    drift_detected, drift_score = check_feature_drift(entries, train_mean, train_std, threshold=0.5, min_samples=10)

    assert drift_detected is True
    assert drift_score > 0.5
