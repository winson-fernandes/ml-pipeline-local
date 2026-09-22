"""
This module orchestrates the pipeline end to end.

It runs every stage, from downloading data through deploying the trained
model, as a sequence of local Python function calls on your machine.

Usage:
    python -m src.pipeline
    python -m src.pipeline --epochs 50 --skip-download
    python -m src.pipeline --no-deploy
"""

import argparse
import logging
import os
import sys
from datetime import datetime

from src.common.paths import MODELS_OUTPUT_DIR, ensure_dirs
from src.data import data_download, data_preprocess
from src.deployment import model_deploy
from src.evaluation import model_evaluate
from src.model import model_train

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline(args):
    logger.info("#" * 70)
    logger.info("# HEART DISEASE ML PIPELINE (local machine)")
    logger.info("#" * 70)

    ensure_dirs()

    if not args.skip_download:
        logger.info("\n>>> STEP 1/5: Download data")
        df = data_download.download_dataset()
        data_download.save_locally(df, args.project)
    else:
        logger.info("\n>>> STEP 1/5: Download data (skipped)")

    if not args.skip_preprocess:
        logger.info("\n>>> STEP 2/5: Preprocess data")
        df, source_file = data_preprocess.load_latest_raw_data()
        df = data_preprocess.handle_missing_values(df)
        df = data_preprocess.engineer_features(df)
        processed = data_preprocess.split_and_normalize(df)
        data_preprocess.save_outputs(processed)
    else:
        logger.info("\n>>> STEP 2/5: Preprocess data (skipped)")

    logger.info("\n>>> STEP 3/5: Train model")
    train_args = argparse.Namespace(
        data_path=None,
        model_dir=os.path.join(MODELS_OUTPUT_DIR, "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")),
        epochs=args.epochs,
        batch_size=32,
        learning_rate=0.001,
        weight_decay=1e-5,
        dropout_rate=0.3,
        early_stopping_patience=15,
    )
    model_train.train(train_args)

    logger.info("\n>>> STEP 4/5: Evaluate model")
    model_evaluate.evaluate_model(model_dir=train_args.model_dir)

    if args.deploy:
        logger.info("\n>>> STEP 5/5: Deploy model (promote to local registry)")
        model_deploy.deploy(model_dir=train_args.model_dir)
        logger.info("\nStart the local model server with: python -m src.serving.serve")
    else:
        logger.info("\n>>> STEP 5/5: Deploy model (skipped)")

    logger.info("\n" + "#" * 70)
    logger.info("# PIPELINE COMPLETE")
    logger.info("#" * 70)


def main():
    parser = argparse.ArgumentParser(description="Run the full local ML pipeline")
    parser.add_argument("--project", default="heart-disease")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--no-deploy", dest="deploy", action="store_false")
    parser.set_defaults(deploy=True)
    args = parser.parse_args()

    try:
        run_pipeline(args)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
