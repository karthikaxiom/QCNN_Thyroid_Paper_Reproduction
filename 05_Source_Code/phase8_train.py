

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)

# Project paths

SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIR.parent
MODEL_DIR = PROJECT_ROOT / "06_Models"
RESULT_DIR = PROJECT_ROOT / "07_Results" / "phase8_tirads"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# Allow importing phase8_qucnet_model.py
sys.path.insert(0, str(SOURCE_DIR))

from phase8_qucnet_model import build_tirads_qucnet


# Experiment configuration

RANDOM_SEED = 42
EPOCHS = 100
BATCH_SIZE = 16
VALIDATION_BATCH_SIZE = 16

BEST_MODEL_PATH = MODEL_DIR / "best_tirads_qucnet.keras"
FINAL_MODEL_PATH = MODEL_DIR / "final_tirads_qucnet.keras"

HISTORY_PATH = RESULT_DIR / "training_history.json"
CSV_LOG_PATH = RESULT_DIR / "training_log.csv"
CONFIG_PATH = RESULT_DIR / "training_config.json"


# Dataset loading

def load_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate the Phase 8 TI-RADS quantum datasets."""

    x_train_path = MODEL_DIR / "X_tirads_train_quantum.npy"
    y_train_path = MODEL_DIR / "y_tirads_train_smote.npy"
    x_test_path = MODEL_DIR / "X_tirads_test_quantum.npy"
    y_test_path = MODEL_DIR / "y_tirads_test.npy"

    required_files = [
        x_train_path,
        y_train_path,
        x_test_path,
        y_test_path,
    ]

    missing_files = [str(path) for path in required_files if not path.exists()]

    if missing_files:
        missing_text = "\n".join(missing_files)
        raise FileNotFoundError(
            "The following required dataset files were not found:\n"
            f"{missing_text}"
        )

    print("\nLoading TI-RADS quantum datasets...")

    x_train = np.load(x_train_path)
    y_train = np.load(y_train_path)
    x_test = np.load(x_test_path)
    y_test = np.load(y_test_path)

    validate_dataset(x_train, y_train, x_test, y_test)

    return x_train, y_train, x_test, y_test


def validate_dataset(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Validate shapes, values and labels before training."""

    expected_feature_shape = (64, 64, 4)

    if x_train.ndim != 4:
        raise ValueError(
            f"X_train must be four-dimensional, received {x_train.shape}"
        )

    if x_test.ndim != 4:
        raise ValueError(
            f"X_test must be four-dimensional, received {x_test.shape}"
        )

    if tuple(x_train.shape[1:]) != expected_feature_shape:
        raise ValueError(
            "Unexpected X_train feature shape: "
            f"{x_train.shape[1:]}, expected {expected_feature_shape}"
        )

    if tuple(x_test.shape[1:]) != expected_feature_shape:
        raise ValueError(
            "Unexpected X_test feature shape: "
            f"{x_test.shape[1:]}, expected {expected_feature_shape}"
        )

    if len(x_train) != len(y_train):
        raise ValueError(
            f"Training sample mismatch: X={len(x_train)}, y={len(y_train)}"
        )

    if len(x_test) != len(y_test):
        raise ValueError(
            f"Testing sample mismatch: X={len(x_test)}, y={len(y_test)}"
        )

    if np.isnan(x_train).any() or np.isnan(x_test).any():
        raise ValueError("NaN values found in quantum feature datasets.")

    if np.isinf(x_train).any() or np.isinf(x_test).any():
        raise ValueError("Infinite values found in quantum feature datasets.")

    train_labels = np.unique(y_train)
    test_labels = np.unique(y_test)

    valid_labels = np.arange(6)

    if not np.all(np.isin(train_labels, valid_labels)):
        raise ValueError(
            f"Unexpected training labels detected: {train_labels}"
        )

    if not np.all(np.isin(test_labels, valid_labels)):
        raise ValueError(
            f"Unexpected testing labels detected: {test_labels}"
        )


def print_dataset_information(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    """Print dataset shapes and class distributions."""

    print("\nDataset Information")
    print("-" * 70)
    print(f"Training features : {x_train.shape}")
    print(f"Training labels   : {y_train.shape}")
    print(f"Testing features  : {x_test.shape}")
    print(f"Testing labels    : {y_test.shape}")

    print("\nTraining Class Distribution")
    print("-" * 70)

    train_counts = np.bincount(y_train.astype(int), minlength=6)

    for class_id, count in enumerate(train_counts):
        print(f"Class {class_id}: {count}")

    print("\nTesting Class Distribution")
    print("-" * 70)

    test_counts = np.bincount(y_test.astype(int), minlength=6)

    for class_id, count in enumerate(test_counts):
        print(f"Class {class_id}: {count}")


def convert_datatypes(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert arrays to TensorFlow-compatible datatypes."""

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)

    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    return x_train, y_train, x_test, y_test


# Training callbacks

def create_callbacks() -> list[tf.keras.callbacks.Callback]:
    """Create callbacks for robust and recoverable training."""

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=12,
        mode="min",
        restore_best_weights=True,
        verbose=1,
    )

    reduce_learning_rate = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        mode="min",
        verbose=1,
    )

    model_checkpoint = ModelCheckpoint(
        filepath=str(BEST_MODEL_PATH),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        mode="min",
        verbose=1,
    )

    csv_logger = CSVLogger(
        filename=str(CSV_LOG_PATH),
        separator=",",
        append=False,
    )

    return [
        early_stopping,
        reduce_learning_rate,
        model_checkpoint,
        csv_logger,
    ]


# Saving experiment information

def save_training_history(
    history: tf.keras.callbacks.History,
) -> None:
    """Save model training history as JSON."""

    serializable_history = {
        key: [float(value) for value in values]
        for key, values in history.history.items()
    }

    with HISTORY_PATH.open("w", encoding="utf-8") as file:
        json.dump(serializable_history, file, indent=4)


def save_training_configuration() -> None:
    """Save the fixed experiment configuration."""

    configuration = {
        "phase": "Phase 8 - TI-RADS QuCNet",
        "random_seed": RANDOM_SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "input_shape": [64, 64, 4],
        "number_of_classes": 6,
        "optimizer": "Adam",
        "learning_rate": 0.0001,
        "loss": "sparse_categorical_crossentropy",
        "early_stopping_patience": 12,
        "reduce_lr_patience": 5,
        "dropout_rate": 0.30,
        "class_mapping": {
            "0": "TI-RADS 2",
            "1": "TI-RADS 3",
            "2": "TI-RADS 4a",
            "3": "TI-RADS 4b",
            "4": "TI-RADS 4c",
            "5": "TI-RADS 5",
        },
    }

    with CONFIG_PATH.open("w", encoding="utf-8") as file:
        json.dump(configuration, file, indent=4)


# Main training procedure

def main() -> None:
    """Run Phase 8 TI-RADS QuCNet training."""

    print("=" * 70)
    print("PHASE 8 : TI-RADS QUCNET TRAINING")
    print("=" * 70)

    np.random.seed(RANDOM_SEED)
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    x_train, y_train, x_test, y_test = load_dataset()

    print_dataset_information(
        x_train,
        y_train,
        x_test,
        y_test,
    )

    x_train, y_train, x_test, y_test = convert_datatypes(
        x_train,
        y_train,
        x_test,
        y_test,
    )

    print("\nBuilding TI-RADS QuCNet model...")

    model = build_tirads_qucnet()

    model.summary()

    save_training_configuration()

    callbacks = create_callbacks()

    print("\nPROTOCOL NOTE: x_test and y_test are used for validation-based model selection in this historical Phase 8 experiment.")
    print("Phase 8 results must therefore be reported as validation-selected holdout performance.")
    print("\nStarting training...")
    print("-" * 70)

    history = model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_batch_size=VALIDATION_BATCH_SIZE,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    save_training_history(history)

    model.save(FINAL_MODEL_PATH)

    best_epoch = int(np.argmin(history.history["val_loss"])) + 1
    best_val_loss = float(np.min(history.history["val_loss"]))

    best_val_accuracy = float(
        history.history["val_accuracy"][best_epoch - 1]
    )

    print("\n" + "=" * 70)
    print("PHASE 8 TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best epoch               : {best_epoch}")
    print(f"Best validation loss     : {best_val_loss:.6f}")
    print(f"Validation accuracy      : {best_val_accuracy * 100:.2f}%")
    print(f"Best model saved at      : {BEST_MODEL_PATH}")
    print(f"Final model saved at     : {FINAL_MODEL_PATH}")
    print(f"Training history saved at: {HISTORY_PATH}")
    print(f"Training log saved at    : {CSV_LOG_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()