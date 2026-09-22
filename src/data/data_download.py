"""
This script downloads the Heart Disease dataset.

It writes the CSV and its metadata to ./data/raw/ on local disk.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from io import StringIO
from urllib.request import urlopen

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.features import RAW_COLUMNS
from src.common.paths import DATA_RAW_DIR, ensure_dirs

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"


def download_dataset():
    """Download the UCI Heart Disease dataset."""
    logger.info("Starting dataset download...")
    logger.info(f"Downloading from {DATASET_URL}")

    data = urlopen(DATASET_URL).read().decode("utf-8")
    df = pd.read_csv(StringIO(data), names=RAW_COLUMNS, na_values="?")

    logger.info(f"Dataset downloaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def save_locally(df, project_name, data_dir=DATA_RAW_DIR):
    """Save the raw dataset and its metadata to the local data/raw/ folder."""
    os.makedirs(data_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_filename = f"{project_name}_{timestamp}.csv"
    csv_path = os.path.join(data_dir, csv_filename)
    df.to_csv(csv_path, index=False)
    logger.info(f"Data saved to {csv_path}")

    metadata = {
        "timestamp": timestamp,
        "rows": len(df),
        "columns": len(df.columns),
        "path": csv_path,
    }
    metadata_path = os.path.join(data_dir, f"metadata_{timestamp}.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return csv_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="heart-disease", help="Project name (used as file prefix)")
    parser.add_argument("--data-dir", default=DATA_RAW_DIR, help="Local folder to save raw data into")
    args = parser.parse_args()

    ensure_dirs()

    df = download_dataset()
    path = save_locally(df, args.project, args.data_dir)

    logger.info(f"SUCCESS - Data at {path}")
    return path


if __name__ == "__main__":
    main()
