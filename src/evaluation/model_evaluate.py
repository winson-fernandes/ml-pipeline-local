"""
This script evaluates the trained model for heart disease prediction.

It loads the model checkpoint from ./models/output/<run>/ and the test
split from ./data/processed/, computes the test-set metrics, and saves a
report and plots to ./evaluation/.
"""

import argparse
import json
import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.model_def import HeartDiseaseNN
from src.common.paths import DATA_PROCESSED_DIR, EVALUATION_DIR, ensure_dirs, latest_file, latest_run_dir

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model(model_dir=None, data_path=None, output_dir=EVALUATION_DIR):
    logger.info("=" * 70)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 70)

    if model_dir is None:
        model_dir = latest_run_dir()
        if model_dir is None:
            raise FileNotFoundError("No training runs found under models/output/. Run model_train.py first.")
    logger.info(f"Model dir: {model_dir}")

    if data_path is None:
        data_path = latest_file(DATA_PROCESSED_DIR, ".npz")
        if data_path is None:
            raise FileNotFoundError("No processed .npz files found. Run data_preprocess.py first.")

    data = np.load(data_path)
    X_test, y_test = data["X_test"], data["y_test"]
    logger.info(f"Test set: {X_test.shape}")

    device = torch.device("cpu")
    model = HeartDiseaseNN(input_size=X_test.shape[1]).to(device)
    checkpoint = torch.load(os.path.join(model_dir, "model.pth"), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        outputs = model(X_test_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        _, predicted = torch.max(outputs, 1)
        y_pred = predicted.cpu().numpy()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    auc = roc_auc_score(y_test, probs[:, 1])

    cm = confusion_matrix(y_test, y_pred)

    results = {
        "model_dir": model_dir,
        "test_accuracy": float(accuracy),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1": float(f1),
        "test_auc": float(auc),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["No Disease", "Disease"], output_dict=True
        ),
    }

    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test AUC: {auc:.4f}")
    logger.info(f"Test F1: {f1:.4f}")

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Disease", "Disease"], yticklabels=["No Disease", "Disease"])
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, probs[:, 1])
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, "roc_curve.png"), dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("=" * 70)
    logger.info("EVALUATION COMPLETE")
    logger.info(f"  Report + plots saved to: {output_dir}")
    logger.info("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=None, help="Training run folder (default: latest under models/output/)")
    parser.add_argument("--data-path", default=None, help="Processed .npz file (default: latest)")
    parser.add_argument("--output-dir", default=EVALUATION_DIR)
    args = parser.parse_args()

    ensure_dirs()

    try:
        evaluate_model(args.model_dir, args.data_path, args.output_dir)
        sys.exit(0)
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
