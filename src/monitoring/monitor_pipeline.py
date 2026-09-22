"""
This script monitors the pipeline.

It reads the local server's own prediction log,
monitoring/predictions_log.jsonl, written by src/serving/serve.py, and
compares the mean of each incoming feature against the mean and standard
deviation recorded at training time in data/processed/metadata_*.json. It
flags drift when those values have moved too far apart.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.paths import DATA_PROCESSED_DIR, MONITORING_DIR, ensure_dirs, latest_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PREDICTIONS_LOG = os.path.join(MONITORING_DIR, "predictions_log.jsonl")


def load_prediction_log(log_path=PREDICTIONS_LOG):
    if not os.path.exists(log_path):
        return []
    entries = []
    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_training_feature_stats(processed_dir=DATA_PROCESSED_DIR):
    """Read the feature mean and standard deviation recorded by data_preprocess.py."""
    metadata_path = latest_file(processed_dir, ".json")
    if metadata_path is None:
        raise FileNotFoundError("No processed data metadata found. Run data_preprocess.py first.")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return np.array(metadata["feature_mean"]), np.array(metadata["feature_std"])


def check_endpoint_metrics(entries):
    """Compute basic invocation stats from the prediction log entries."""
    return {
        "invocation_count": len(entries),
        "disease_predictions": sum(1 for e in entries if e["prediction"] == 1),
        "no_disease_predictions": sum(1 for e in entries if e["prediction"] == 0),
        "avg_confidence": float(np.mean([e["probability"] for e in entries])) if entries else 0.0,
    }


def check_feature_drift(entries, train_mean, train_std, threshold=0.5, min_samples=10):
    """Flag drift when recent request features have drifted, on average, more than
    `threshold` standard deviations from the training distribution."""
    if len(entries) < min_samples:
        logger.info(f"Not enough predictions yet ({len(entries)}/{min_samples}) to assess drift")
        return False, 0.0

    recent = np.array([e["features"] for e in entries[-min_samples:]])
    recent_mean = recent.mean(axis=0)

    # This measures the mean absolute shift across all features, in units of the training standard deviation.
    z_shift = np.abs((recent_mean - train_mean) / np.where(train_std == 0, 1, train_std))
    drift_score = float(z_shift.mean())

    drift_detected = drift_score > threshold
    if drift_detected:
        logger.warning(f"Feature drift detected: score={drift_score:.3f} (threshold={threshold})")
    else:
        logger.info(f"No significant drift: score={drift_score:.3f} (threshold={threshold})")

    return drift_detected, drift_score


def trigger_retraining(reason):
    logger.info("Triggering model retraining...")
    os.makedirs(MONITORING_DIR, exist_ok=True)

    retrain_metadata = {
        "trigger_time": datetime.now(timezone.utc).isoformat(),
        "trigger_reason": reason,
        "status": "pending",
        "next_step": "python -m src.pipeline --skip-download --skip-preprocess",
    }
    path = os.path.join(
        MONITORING_DIR, f"retrain_trigger_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(path, "w") as f:
        json.dump(retrain_metadata, f, indent=2)

    logger.info(f"Retraining trigger written to {path}")
    return retrain_metadata


def monitor_pipeline(drift_threshold=0.5):
    logger.info("=" * 70)
    logger.info("PIPELINE MONITORING")
    logger.info("=" * 70)

    ensure_dirs()
    entries = load_prediction_log()
    metrics = check_endpoint_metrics(entries)

    logger.info("\nServer metrics (all-time, from local prediction log):")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    drift_detected, drift_score = False, 0.0
    try:
        train_mean, train_std = load_training_feature_stats()
        drift_detected, drift_score = check_feature_drift(entries, train_mean, train_std, drift_threshold)
    except FileNotFoundError as e:
        logger.warning(str(e))

    monitoring_report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "drift_detected": drift_detected,
        "drift_score": drift_score,
        "retraining_triggered": False,
    }

    if drift_detected:
        monitoring_report["retrain_info"] = trigger_retraining("feature_drift_detected")
        monitoring_report["retraining_triggered"] = True

    report_path = os.path.join(MONITORING_DIR, f"report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json")
    with open(report_path, "w") as f:
        json.dump(monitoring_report, f, indent=2)

    logger.info("=" * 70)
    logger.info(f"MONITORING COMPLETE - report saved to {report_path}")
    logger.info("=" * 70)

    return monitoring_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--drift-threshold", type=float, default=0.5)
    args = parser.parse_args()

    try:
        monitor_pipeline(args.drift_threshold)
    except Exception as e:
        logger.error(f"Monitoring failed: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
