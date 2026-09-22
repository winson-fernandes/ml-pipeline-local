"""
This script preprocesses the Heart Disease dataset.

It reads the latest raw CSV from ./data/raw/, cleans it, engineers
features, splits and normalizes the data, and then writes the processed
arrays to ./data/processed/ and the fitted scaler to ./models/scalers/.
"""

import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERICAL_FEATURES,
    engineer_features_df,
)
from src.common.paths import DATA_PROCESSED_DIR, DATA_RAW_DIR, MODELS_SCALERS_DIR, ensure_dirs, latest_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_latest_raw_data(data_dir=DATA_RAW_DIR):
    """Load the most recently downloaded raw CSV from the local data/raw/ folder."""
    logger.info(f"Loading data from {data_dir}")

    csv_path = latest_file(data_dir, ".csv")
    if csv_path is None:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}. Run data_download.py first."
        )

    df = pd.read_csv(csv_path)
    logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    logger.info(f"Columns: {list(df.columns)}")

    return df, csv_path


def handle_missing_values(df):
    """Handle missing values in the dataset."""
    logger.info("Handling missing values...")

    missing_before = df.isnull().sum().sum()
    logger.info(f"  Missing values before: {missing_before}")

    for col in NUMERICAL_FEATURES:
        if col in df.columns and df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info(f"  Filled {col} with median: {median_val}")

    for col in CATEGORICAL_FEATURES:
        if col in df.columns and df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.info(f"  Filled {col} with mode: {mode_val}")

    logger.info(f"  Missing values after: {df.isnull().sum().sum()}")
    return df


def engineer_features(df):
    """Create the target label and the engineered feature columns."""
    logger.info("Engineering features...")

    if "target" in df.columns:
        df["target_binary"] = (df["target"] > 0).astype(int)
        logger.info(f"  No disease (0): {(df['target_binary'] == 0).sum()}")
        logger.info(f"  Disease (1): {(df['target_binary'] == 1).sum()}")

    df = engineer_features_df(df)
    logger.info(f"  Feature engineering complete (total columns: {len(df.columns)})")

    return df


def split_and_normalize(df):
    """Split data into train/val/test and normalize with a StandardScaler."""
    logger.info("Splitting and normalizing data...")

    X = df[FEATURE_COLUMNS].values
    y = df["target_binary"].values

    logger.info(f"  Feature matrix: {X.shape}")
    logger.info(f"  Target vector: {y.shape}")

    # This split allocates 70 percent of the data to training, 15 percent to validation, and 15 percent to testing.
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
    )

    logger.info(f"  Train set: {X_train.shape[0]} samples")
    logger.info(f"  Val set:   {X_val.shape[0]} samples")
    logger.info(f"  Test set:  {X_test.shape[0]} samples")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    logger.info("  Normalization complete")
    logger.info(f"  Mean: {X_train_scaled.mean():.6f} (should be ~0)")
    logger.info(f"  Std:  {X_train_scaled.std():.6f} (should be ~1)")

    return {
        "X_train": X_train_scaled, "X_val": X_val_scaled, "X_test": X_test_scaled,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "feature_names": FEATURE_COLUMNS, "scaler": scaler,
    }


def save_outputs(data, processed_dir=DATA_PROCESSED_DIR, scalers_dir=MODELS_SCALERS_DIR):
    """Save the processed arrays, scaler, and metadata to local disk."""
    logger.info("Saving processed data locally...")

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(scalers_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    npz_path = os.path.join(processed_dir, f"processed_data_{timestamp}.npz")
    np.savez(
        npz_path,
        X_train=data["X_train"], X_val=data["X_val"], X_test=data["X_test"],
        y_train=data["y_train"], y_val=data["y_val"], y_test=data["y_test"],
    )
    logger.info(f"  Saved arrays to {npz_path}")

    scaler_path = os.path.join(scalers_dir, f"scaler_{timestamp}.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(data["scaler"], f)
    logger.info(f"  Saved scaler to {scaler_path}")

    # The monitoring stage reads these per-feature training statistics later to detect drift.
    X_train_raw_mean = data["scaler"].mean_.tolist()
    X_train_raw_std = data["scaler"].scale_.tolist()

    metadata = {
        "timestamp": timestamp,
        "train_samples": int(len(data["y_train"])),
        "val_samples": int(len(data["y_val"])),
        "test_samples": int(len(data["y_test"])),
        "num_features": int(data["X_train"].shape[1]),
        "feature_names": data["feature_names"],
        "train_disease_count": int(data["y_train"].sum()),
        "val_disease_count": int(data["y_val"].sum()),
        "test_disease_count": int(data["y_test"].sum()),
        "feature_mean": X_train_raw_mean,
        "feature_std": X_train_raw_std,
        "npz_path": npz_path,
        "scaler_path": scaler_path,
    }
    metadata_path = os.path.join(processed_dir, f"metadata_{timestamp}.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  Saved metadata to {metadata_path}")

    return timestamp, metadata


def main():
    parser = argparse.ArgumentParser(description="Preprocess heart disease data")
    parser.add_argument("--raw-dir", default=DATA_RAW_DIR, help="Folder containing raw data")
    parser.add_argument("--processed-dir", default=DATA_PROCESSED_DIR, help="Folder to write processed data")
    parser.add_argument("--scalers-dir", default=MODELS_SCALERS_DIR, help="Folder to write the fitted scaler")
    args = parser.parse_args()

    ensure_dirs()

    logger.info("=" * 70)
    logger.info("HEART DISEASE DATA PREPROCESSING")
    logger.info("=" * 70)

    try:
        df, source_file = load_latest_raw_data(args.raw_dir)
        df = handle_missing_values(df)
        df = engineer_features(df)
        processed_data = split_and_normalize(df)
        timestamp, metadata = save_outputs(processed_data, args.processed_dir, args.scalers_dir)

        logger.info("=" * 70)
        logger.info("PREPROCESSING COMPLETE")
        logger.info(f"  Source: {source_file}")
        logger.info(f"  Timestamp: {timestamp}")
        logger.info(f"  Train: {metadata['train_samples']} samples")
        logger.info(f"  Val: {metadata['val_samples']} samples")
        logger.info(f"  Test: {metadata['test_samples']} samples")
        logger.info(f"  Features: {metadata['num_features']}")
        logger.info("=" * 70)

        return 0

    except Exception as e:
        logger.error(f"PREPROCESSING FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
