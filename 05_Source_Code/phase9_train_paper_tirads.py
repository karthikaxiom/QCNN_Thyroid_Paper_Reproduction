from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import (
    CSVLogger,
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TerminateOnNaN,
)


# Project paths

SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIR.parent

MODEL_DIR = PROJECT_ROOT / "06_Models"

RESULT_DIR = (
    PROJECT_ROOT
    / "07_Results"
    / "phase9_paper_tirads_class_weighted"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SOURCE_DIR))

from phase9_paper_tirads_model import build_paper_tirads_qucnet


# Experiment configuration

RANDOM_SEED = 42

EPOCHS = 120
BATCH_SIZE = 16
VALIDATION_SIZE = 0.20

X_DATA_PATH = (
    MODEL_DIR / "X_tirads_train_quantum_original.npy"
)

Y_DATA_PATH = (
    MODEL_DIR / "y_tirads_train_original.npy"
)

BEST_MODEL_PATH = (
    MODEL_DIR / "best_phase9_paper_tirads.keras"
)

FINAL_MODEL_PATH = (
    MODEL_DIR / "final_phase9_paper_tirads.keras"
)

HISTORY_PATH = (
    RESULT_DIR / "training_history.json"
)

CSV_LOG_PATH = (
    RESULT_DIR / "training_log.csv"
)

CONFIG_PATH = (
    RESULT_DIR / "training_config.json"
)

CLASS_WEIGHTS_PATH = (
    RESULT_DIR / "class_weights.json"
)

TRAIN_INDICES_PATH = (
    RESULT_DIR / "train_indices.npy"
)

VALIDATION_INDICES_PATH = (
    RESULT_DIR / "validation_indices.npy"
)


CLASS_NAMES = [
    "TI-RADS 2",
    "TI-RADS 3",
    "TI-RADS 4a",
    "TI-RADS 4b",
    "TI-RADS 4c",
    "TI-RADS 5",
]


# Dataset loading

def require_files(paths: list[Path]) -> None:
    """Check that all required files exist."""

    missing_files = [
        str(path)
        for path in paths
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Required files were not found:\n"
            + "\n".join(missing_files)
        )


def load_original_quantum_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Load and validate the original-only quantum dataset."""

    require_files(
        [
            X_DATA_PATH,
            Y_DATA_PATH,
        ]
    )

    print("\nLoading original-only TI-RADS quantum dataset...")

    x_data = np.load(X_DATA_PATH)
    y_data = np.load(Y_DATA_PATH).astype(np.int32).ravel()

    validate_dataset(
        x_data,
        y_data,
    )

    x_data = x_data.astype(np.float32)

    return x_data, y_data


def validate_dataset(
    x_data: np.ndarray,
    y_data: np.ndarray,
) -> None:
    """Validate quantum feature maps and labels."""

    expected_shape = (64, 64, 4)

    if x_data.ndim != 4:
        raise ValueError(
            "Quantum feature dataset must be four-dimensional. "
            f"Received: {x_data.shape}"
        )

    if tuple(x_data.shape[1:]) != expected_shape:
        raise ValueError(
            f"Expected feature shape {expected_shape}, "
            f"received {x_data.shape[1:]}"
        )

    if len(x_data) != len(y_data):
        raise ValueError(
            "Feature-label count mismatch: "
            f"{len(x_data)} features and {len(y_data)} labels."
        )

    if np.isnan(x_data).any():
        raise ValueError(
            "NaN values were detected in the quantum features."
        )

    if np.isinf(x_data).any():
        raise ValueError(
            "Infinite values were detected in the quantum features."
        )

    expected_classes = np.arange(6)
    detected_classes = np.unique(y_data)

    if not np.array_equal(
        detected_classes,
        expected_classes,
    ):
        raise ValueError(
            "Expected labels 0 through 5. "
            f"Detected labels: {detected_classes}"
        )


def print_dataset_information(
    x_data: np.ndarray,
    y_data: np.ndarray,
) -> None:
    """Display dataset shape and distribution."""

    print("\nOriginal Quantum Dataset")
    print("-" * 76)

    print(f"Feature shape       : {x_data.shape}")
    print(f"Label shape         : {y_data.shape}")
    print(f"Feature data type   : {x_data.dtype}")
    print(f"Minimum value       : {np.min(x_data):.6f}")
    print(f"Maximum value       : {np.max(x_data):.6f}")
    print(f"Mean value          : {np.mean(x_data):.6f}")
    print(f"Standard deviation  : {np.std(x_data):.6f}")

    counts = np.bincount(
        y_data,
        minlength=6,
    )

    print("\nOriginal Class Distribution")
    print("-" * 76)

    for class_id, count in enumerate(counts):
        print(
            f"Class {class_id} "
            f"({CLASS_NAMES[class_id]}): {count}"
        )


# Train-validation split

def create_train_validation_split(
    x_data: np.ndarray,
    y_data: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Create a reproducible stratified split from original samples only.
    """

    all_indices = np.arange(len(x_data))

    train_indices, validation_indices = train_test_split(
        all_indices,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_data,
        shuffle=True,
    )

    np.save(
        TRAIN_INDICES_PATH,
        train_indices,
    )

    np.save(
        VALIDATION_INDICES_PATH,
        validation_indices,
    )

    x_train = x_data[train_indices]
    y_train = y_data[train_indices]

    x_validation = x_data[validation_indices]
    y_validation = y_data[validation_indices]

    return (
        x_train,
        y_train,
        x_validation,
        y_validation,
    )


def print_split_information(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
) -> None:
    """Print split sizes and class counts."""

    print("\nOriginal-Only Train-Validation Split")
    print("-" * 76)

    print(f"Training features   : {x_train.shape}")
    print(f"Training labels     : {y_train.shape}")
    print(f"Validation features : {x_validation.shape}")
    print(f"Validation labels   : {y_validation.shape}")

    training_counts = np.bincount(
        y_train,
        minlength=6,
    )

    validation_counts = np.bincount(
        y_validation,
        minlength=6,
    )

    print("\nTraining Class Distribution")
    print("-" * 76)

    for class_id, count in enumerate(training_counts):
        print(
            f"Class {class_id} "
            f"({CLASS_NAMES[class_id]}): {count}"
        )

    print("\nValidation Class Distribution")
    print("-" * 76)

    for class_id, count in enumerate(validation_counts):
        print(
            f"Class {class_id} "
            f"({CLASS_NAMES[class_id]}): {count}"
        )


# Class weights

def calculate_class_weights(
    y_train: np.ndarray,
) -> dict[int, float]:
    """
    Calculate balanced class weights using training labels only.
    """

    classes = np.arange(6)

    weight_values = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )

    class_weights = {
        int(class_id): float(weight)
        for class_id, weight in zip(
            classes,
            weight_values,
        )
    }

    print("\nTraining Class Weights")
    print("-" * 76)

    for class_id, weight in class_weights.items():
        print(
            f"Class {class_id} "
            f"({CLASS_NAMES[class_id]}): {weight:.6f}"
        )

    with CLASS_WEIGHTS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                str(class_id): weight
                for class_id, weight
                in class_weights.items()
            },
            file,
            indent=4,
        )

    return class_weights


# ---------------------------------------------------------------------
# Training callbacks
# ---------------------------------------------------------------------

def create_callbacks() -> list[tf.keras.callbacks.Callback]:
    """Create training control callbacks."""

    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=15,
        min_delta=1e-4,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_learning_rate = ReduceLROnPlateau(
        monitor="val_loss",
        mode="min",
        factor=0.5,
        patience=6,
        min_delta=1e-4,
        min_lr=1e-7,
        verbose=1,
    )

    model_checkpoint = ModelCheckpoint(
        filepath=str(BEST_MODEL_PATH),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )

    csv_logger = CSVLogger(
        filename=str(CSV_LOG_PATH),
        separator=",",
        append=False,
    )

    terminate_on_nan = TerminateOnNaN()

    return [
        early_stopping,
        reduce_learning_rate,
        model_checkpoint,
        csv_logger,
        terminate_on_nan,
    ]


# ---------------------------------------------------------------------
# Experiment recording
# ---------------------------------------------------------------------

def save_training_configuration(
    training_samples: int,
    validation_samples: int,
) -> None:
    """Save all fixed experiment settings."""

    configuration = {
        "phase": (
            "Phase 9 - Corrected Paper-Structured "
            "TI-RADS QuCNet"
        ),
        "experiment_type": (
            "Paper-structured class-weighted reproduction"
        ),
        "random_seed": RANDOM_SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "validation_fraction": VALIDATION_SIZE,
        "training_samples": training_samples,
        "validation_samples": validation_samples,
        "input_shape": [64, 64, 4],
        "number_of_classes": 6,
        "dataset": (
            "Original-only TI-RADS quantum feature maps"
        ),
        "imbalance_method": "Balanced class weights",
        "smote_used": False,
        "test_set_used_during_training": False,
        "optimizer": "Adam",
        "initial_learning_rate": 0.0001,
        "loss": "sparse_categorical_crossentropy",
        "early_stopping_patience": 15,
        "reduce_lr_patience": 6,
        "model_parameters": 1062582,
        "architecture": [
            "Conv2D 16",
            "MaxPooling2D",
            "Conv2D 32",
            "MaxPooling2D",
            "Flatten",
            "Dense 128",
            "Dense 64",
            "Softmax 6",
        ],
        "class_mapping": {
            "0": "TI-RADS 2",
            "1": "TI-RADS 3",
            "2": "TI-RADS 4a",
            "3": "TI-RADS 4b",
            "4": "TI-RADS 4c",
            "5": "TI-RADS 5",
        },
        "scientific_note": (
            "Layer sequence follows the paper description. "
            "Filter counts, dense neurons and learning rate "
            "are documented implementation decisions."
        ),
    }

    with CONFIG_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            configuration,
            file,
            indent=4,
        )


def save_training_history(
    history: tf.keras.callbacks.History,
) -> None:
    """Save training curves as JSON."""

    serializable_history = {
        key: [
            float(value)
            for value in values
        ]
        for key, values in history.history.items()
    }

    with HISTORY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            serializable_history,
            file,
            indent=4,
        )


# ---------------------------------------------------------------------
# Main training procedure
# ---------------------------------------------------------------------

def main() -> None:
    """Train the corrected paper-structured TI-RADS model."""

    print("=" * 76)
    print(
        "PHASE 9 : TRAIN CORRECTED PAPER-STRUCTURED "
        "TI-RADS QUCNET"
    )
    print("=" * 76)

    np.random.seed(RANDOM_SEED)
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    tf.keras.backend.clear_session()

    x_data, y_data = load_original_quantum_dataset()

    print_dataset_information(
        x_data,
        y_data,
    )

    (
        x_train,
        y_train,
        x_validation,
        y_validation,
    ) = create_train_validation_split(
        x_data,
        y_data,
    )

    print_split_information(
        x_train,
        y_train,
        x_validation,
        y_validation,
    )

    class_weights = calculate_class_weights(
        y_train,
    )

    print("\nBuilding corrected paper-structured model...")

    model = build_paper_tirads_qucnet()

    model.summary()

    save_training_configuration(
        training_samples=len(x_train),
        validation_samples=len(x_validation),
    )

    callbacks = create_callbacks()

    print("\nStarting corrected paper-structured training...")
    print("-" * 76)

    history = model.fit(
        x=x_train,
        y=y_train,
        validation_data=(
            x_validation,
            y_validation,
        ),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    save_training_history(history)

    model.save(FINAL_MODEL_PATH)

    best_epoch = (
        int(np.argmin(history.history["val_loss"])) + 1
    )

    best_validation_loss = float(
        np.min(history.history["val_loss"])
    )

    best_validation_accuracy = float(
        history.history["val_accuracy"][best_epoch - 1]
    )

    final_training_accuracy = float(
        history.history["accuracy"][-1]
    )

    print("\n" + "=" * 76)
    print(
        "CORRECTED PAPER-STRUCTURED TRAINING COMPLETE"
    )
    print("=" * 76)

    print(f"Best epoch                 : {best_epoch}")
    print(
        "Best validation loss       : "
        f"{best_validation_loss:.6f}"
    )
    print(
        "Best validation accuracy   : "
        f"{best_validation_accuracy * 100:.2f}%"
    )
    print(
        "Final training accuracy    : "
        f"{final_training_accuracy * 100:.2f}%"
    )

    print(f"Best model saved at        : {BEST_MODEL_PATH}")
    print(f"Final model saved at       : {FINAL_MODEL_PATH}")
    print(f"Training history saved at  : {HISTORY_PATH}")
    print(f"Training log saved at      : {CSV_LOG_PATH}")
    print(f"Class weights saved at     : {CLASS_WEIGHTS_PATH}")
    print(f"Train indices saved at     : {TRAIN_INDICES_PATH}")
    print(
        "Validation indices saved at: "
        f"{VALIDATION_INDICES_PATH}"
    )

    print("\nImportant:")
    print(
        "Only the 279 original quantum samples were used."
    )
    print(
        "SMOTE samples were not used in this experiment."
    )
    print(
        "The untouched 70-image test set was not used "
        "during training or model selection."
    )

    print("=" * 76)


if __name__ == "__main__":
    main()