"""
This script deploys a trained model.

Deploying locally means promoting a trained model into
models/registry/latest/, the folder that the local serving process in
src/serving/serve.py always loads its model from. A quick self-test
prediction confirms the promoted model actually loads and runs.
"""

import argparse
import json
import logging
import os
import re
import shutil
import sys
from datetime import datetime

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.paths import (
    DATA_PROCESSED_DIR,
    MODELS_REGISTRY_LATEST_DIR,
    MODELS_SCALERS_DIR,
    ensure_dirs,
    latest_file,
    latest_run_dir,
)
from src.inference.inference import load_model, predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# data_preprocess.py names its .npz, scaler, and metadata files with a
# shared "processed_data_<timestamp>.npz" / "scaler_<timestamp>.pkl" /
# "metadata_<timestamp>.json" pattern. This lets us recover which scaler
# and metadata belong to a given .npz from the timestamp alone.
_NPZ_TIMESTAMP_RE = re.compile(r"processed_data_(\d{8}_\d{6})\.npz$")


def _matching_scaler_and_metadata(model_dir):
    """Find the scaler and metadata that were actually fitted on the data this
    model was trained on, instead of assuming "most recently modified" is
    the right one.

    model_train.py records the resolved .npz path it trained on as
    `data_path` in model.pth's checkpoint. We use that to look up the
    scaler/metadata sharing its timestamp. Older checkpoints saved before
    this field existed fall back to "most recently modified", with a
    warning, since there is no provenance to recover.
    """
    model_path = os.path.join(model_dir, "model.pth")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    data_path = checkpoint.get("data_path") or checkpoint.get("hyperparameters", {}).get("data_path")

    if not data_path:
        logger.warning(
            "This model's checkpoint has no recorded data_path (it was likely trained "
            "before this was tracked). Falling back to the most recently modified "
            "scaler, which may not match the data this model was actually trained on."
        )
        return latest_file(MODELS_SCALERS_DIR, ".pkl"), latest_file(DATA_PROCESSED_DIR, ".json")

    match = _NPZ_TIMESTAMP_RE.search(os.path.basename(data_path))
    if not match:
        logger.warning(
            f"Could not parse a timestamp out of recorded data_path '{data_path}'. "
            "Falling back to the most recently modified scaler."
        )
        return latest_file(MODELS_SCALERS_DIR, ".pkl"), latest_file(DATA_PROCESSED_DIR, ".json")

    timestamp = match.group(1)
    scaler_path = os.path.join(MODELS_SCALERS_DIR, f"scaler_{timestamp}.pkl")
    metadata_path = os.path.join(DATA_PROCESSED_DIR, f"metadata_{timestamp}.json")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Model {model_dir} was trained on {data_path} (timestamp {timestamp}), but its "
            f"matching scaler {scaler_path} no longer exists on disk. Re-run preprocessing "
            "and training together, or deploy a different run."
        )

    logger.info(f"  Matched scaler to this model's training run via timestamp {timestamp}: {scaler_path}")
    return scaler_path, metadata_path if os.path.exists(metadata_path) else None


def promote_model(model_dir=None, registry_dir=MODELS_REGISTRY_LATEST_DIR):
    """Copy a trained model's artifacts into the local model registry to deploy it."""
    logger.info("=" * 70)
    logger.info("PREPARING MODEL FOR LOCAL DEPLOYMENT")
    logger.info("=" * 70)

    if model_dir is None:
        model_dir = latest_run_dir()
        if model_dir is None:
            raise FileNotFoundError("No training runs found under models/output/. Run model_train.py first.")
    logger.info(f"Promoting model from: {model_dir}")

    os.makedirs(registry_dir, exist_ok=True)

    for filename in ["model.pth", "metrics.json", "training_history.json"]:
        src_path = os.path.join(model_dir, filename)
        if os.path.exists(src_path):
            shutil.copy(src_path, os.path.join(registry_dir, filename))

    scaler_path, metadata_path = _matching_scaler_and_metadata(model_dir)
    if scaler_path:
        shutil.copy(scaler_path, os.path.join(registry_dir, "scaler.pkl"))
    if metadata_path:
        shutil.copy(metadata_path, os.path.join(registry_dir, "training_metadata.json"))

    logger.info(f"Model artifacts copied to: {registry_dir}")
    return registry_dir


def self_test(registry_dir=MODELS_REGISTRY_LATEST_DIR):
    """Load the promoted model and run one dummy prediction to confirm it works."""
    logger.info("Running self-test prediction...")

    model, scaler = load_model(registry_dir)
    dummy_features = [63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1, 9.135, 3.698, 2.381, 0, 1, 0]

    import numpy as np
    result = predict(np.array(dummy_features, dtype="float32"), model, scaler)

    logger.info(f"Self-test prediction: {result}")
    return result


def deploy(model_dir=None, registry_dir=MODELS_REGISTRY_LATEST_DIR, output_file="deployment_info.json"):
    model_dir = promote_model(model_dir, registry_dir)
    self_test(registry_dir)

    deployment_info = {
        "model_dir": model_dir,
        "registry_dir": registry_dir,
        "status": "deployed",
        "timestamp": datetime.now().isoformat(),
    }

    with open(os.path.join(registry_dir, output_file), "w") as f:
        json.dump(deployment_info, f, indent=2)

    logger.info("=" * 70)
    logger.info("DEPLOYMENT COMPLETE")
    logger.info(f"  Model is ready to serve from: {registry_dir}")
    logger.info("  Start it with: python -m src.serving.serve")
    logger.info("=" * 70)

    return deployment_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=None, help="Training run to deploy (default: latest)")
    parser.add_argument("--registry-dir", default=MODELS_REGISTRY_LATEST_DIR)
    parser.add_argument("--output-file", default="deployment_info.json")
    args = parser.parse_args()

    ensure_dirs()

    try:
        deploy(args.model_dir, args.registry_dir, args.output_file)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
