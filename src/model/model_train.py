"""
This script trains the PyTorch model for heart disease prediction.

It runs the training loop on your own CPU or GPU, reading processed data
from ./data/processed/ and writing the trained model to
./models/output/<run_name>/ on local disk.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.model_def import HeartDiseaseNN
from src.common.paths import DATA_PROCESSED_DIR, MODELS_OUTPUT_DIR, ensure_dirs, latest_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return running_loss / len(train_loader), accuracy_score(all_labels, all_preds)


def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    epoch_loss = running_loss / len(val_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)

    all_probs = np.array(all_probs)
    try:
        epoch_auc = roc_auc_score(all_labels, all_probs[:, 1]) if len(np.unique(all_labels)) > 1 else 0.0
        epoch_f1 = f1_score(all_labels, all_preds, average="weighted")
    except Exception:
        epoch_auc, epoch_f1 = 0.0, 0.0

    return epoch_loss, epoch_acc, epoch_auc, epoch_f1


def load_processed_data(data_path=None, processed_dir=DATA_PROCESSED_DIR):
    """Load the processed .npz produced by data_preprocess.py."""
    if data_path is None:
        data_path = latest_file(processed_dir, ".npz")
        if data_path is None:
            raise FileNotFoundError(
                f"No processed .npz files found in {processed_dir}. Run data_preprocess.py first."
            )

    logger.info(f"Loading data from {data_path}")
    data = np.load(data_path)

    X_train, X_val = data["X_train"], data["X_val"]
    y_train, y_val = data["y_train"], data["y_val"]

    logger.info(f"  Train: {X_train.shape}")
    logger.info(f"  Val:   {X_val.shape}")

    return X_train, X_val, y_train, y_val, data_path


def train(args):
    logger.info("=" * 70)
    logger.info("HEART DISEASE MODEL TRAINING")
    logger.info("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")

    X_train, X_val, y_train, y_val, data_path = load_processed_data(args.data_path)
    # Record the exact .npz actually used (it may have been auto-resolved to
    # "latest" above) so the deploy stage can promote the scaler that was
    # fitted on this same preprocessing run, instead of guessing from mtime.
    args.data_path = data_path

    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.LongTensor(y_val))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    logger.info("=" * 70)

    model = HeartDiseaseNN(input_size=X_train.shape[1], dropout_rate=args.dropout_rate).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model parameters: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    os.makedirs(args.model_dir, exist_ok=True)

    logger.info("\nStarting training...")
    logger.info("-" * 70)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_auc": [], "val_f1": []}

    start_time = time.time()

    for epoch in range(args.epochs):
        epoch_start = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc, val_f1 = validate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["val_auc"].append(float(val_auc))
        history["val_f1"].append(float(val_f1))

        epoch_time = time.time() - epoch_start

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch [{epoch + 1:3d}/{args.epochs}] ({epoch_time:.1f}s) | "
                f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, AUC: {val_auc:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_auc": val_auc,
                "val_f1": val_f1,
                "data_path": data_path,
                "hyperparameters": vars(args),
            }
            model_path = os.path.join(args.model_dir, "model.pth")
            torch.save(checkpoint, model_path)
            logger.info(f"  Best model saved (val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f})")
        else:
            patience_counter += 1

        if patience_counter >= args.early_stopping_patience:
            logger.info(f"\nEarly stopping triggered at epoch {epoch + 1}")
            break

    total_time = time.time() - start_time

    with open(os.path.join(args.model_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    final_metrics = {
        "best_val_loss": float(best_val_loss),
        "best_val_acc": float(best_val_acc),
        "final_val_loss": float(history["val_loss"][-1]),
        "final_val_acc": float(history["val_acc"][-1]),
        "final_val_auc": float(history["val_auc"][-1]),
        "total_epochs": len(history["train_loss"]),
        "training_time_seconds": float(total_time),
        "training_time_minutes": float(total_time / 60),
    }
    with open(os.path.join(args.model_dir, "metrics.json"), "w") as f:
        json.dump(final_metrics, f, indent=2)

    logger.info("-" * 70)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Total time: {total_time / 60:.2f} minutes")
    logger.info(f"  Total epochs: {len(history['train_loss'])}")
    logger.info(f"  Best val loss: {best_val_loss:.4f}")
    logger.info(f"  Best val acc: {best_val_acc:.4f}")
    logger.info(f"  Model saved to: {args.model_dir}")
    logger.info("=" * 70)

    return model, history, final_metrics


def main():
    parser = argparse.ArgumentParser()

    default_model_dir = os.path.join(MODELS_OUTPUT_DIR, "run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))

    parser.add_argument("--data-path", type=str, default=None, help="Path to a processed .npz file, default latest")
    parser.add_argument("--model-dir", type=str, default=default_model_dir, help="Where to save model.pth and metrics")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--dropout-rate", type=float, default=0.3)
    parser.add_argument("--early-stopping-patience", type=int, default=15)

    args = parser.parse_args()

    ensure_dirs()

    try:
        train(args)
        logger.info("\nTraining job completed successfully!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\nTraining failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
