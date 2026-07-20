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
    TerminateOnNaN,
)


# Project paths

SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIR.parent

MODEL_DIR = PROJECT_ROOT / "06_Models"

PHASE9_RESULT_DIR = (
    PROJECT_ROOT
    / "07_Results"
    / "phase9_paper_tirads_class_weighted"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "07_Results"
    / "phase10_classical_tirads_class_weighted"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SOURCE_DIR))

from phase10_classical_tirads_model import (
    build_classical_tirads_model,
)


# Configuration

RANDOM_SEED = 42

EPOCHS = 120
BATCH_SIZE = 16

X_DATA_PATH = (
    MODEL_DIR / "X_tirads_train.npy"
)

Y_DATA_PATH = (
    MODEL_DIR / "y_tirads_train.npy"
)

TRAIN_INDICES_PATH = (
    PHASE9_RESULT_DIR / "train_indices.npy"
)

VALIDATION_INDICES_PATH = (
    PHASE9_RESULT_DIR / "validation_indices.npy"
)

PHASE9_CLASS_WEIGHTS_PATH = (
    PHASE9_RESULT_DIR / "class_weights.json"
)

BEST_MODEL_PATH = (
    MODEL_DIR / "best_phase10_classical_tirads.keras"
)

FINAL_MODEL_PATH = (
    MODEL_DIR / "final_phase10_classical_tirads.keras"
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

CLASS_WEIGHTS_COPY_PATH = (
    RESULT_DIR / "class_weights.json"
)

TRAIN_INDICES_COPY_PATH = (
    RESULT_DIR / "train_indices.npy"
)

VALIDATION_INDICES_COPY_PATH = (
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


# File and dataset validation

def require_files(paths: list[Path]) -> None:
    """Confirm that all files required by this experiment exist."""

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


def load_original_image_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Load the original 279-image TI-RADS training dataset."""

    require_files(
        [
            X_DATA_PATH,
            Y_DATA_PATH,
            TRAIN_INDICES_PATH,
            VALIDATION_INDICES_PATH,
            PHASE9_CLASS_WEIGHTS_PATH,
        ]
    )

    print("\nLoading original grayscale TI-RADS dataset...")

    x_data = np.load(X_DATA_PATH)
    y_data = np.load(Y_DATA_PATH).astype(np.int32).ravel()

    validate_dataset(
        x_data,
        y_data,
    )

    x_data = x_data.astype(np.float32)

    # Safety normalization only when values appear to be 0-255.
    normalization_applied = False

    if np.max(x_data) > 1.5:
        x_data = x_data / 255.0
        normalization_applied = True

    print(
        "Automatic division by 255 applied: "
        f"{normalization_applied}"
    )

    return x_data, y_data


def validate_dataset(
    x_data: np.ndarray,
    y_data: np.ndarray,
) -> None:
    """Validate image dimensions, labels and numerical integrity."""

    expected_shape = (128, 128, 1)

    if x_data.ndim != 4:
        raise ValueError(
            "Expected four-dimensional image data. "
            f"Received shape: {x_data.shape}"
        )

    if tuple(x_data.shape[1:]) != expected_shape:
        raise ValueError(
            f"Expected image shape {expected_shape}, "
            f"received {x_data.shape[1:]}"
        )

    if len(x_data) != len(y_data):
        raise ValueError(
            "Image-label count mismatch: "
            f"{len(x_data)} images and {len(y_data)} labels."
        )

    if len(x_data) != 279:
        print(
            "Warning: the expected original training count was 279, "
            f"but {len(x_data)} samples were found."
        )

    if np.isnan(x_data).any():
        raise ValueError(
            "NaN values were found in the grayscale dataset."
        )

    if np.isinf(x_data).any():
        raise ValueError(
            "Infinite values were found in the grayscale dataset."
        )

    detected_classes = np.unique(y_data)
    expected_classes = np.arange(6)

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
    """Print image statistics and class distribution."""

    print("\nOriginal Grayscale Dataset")
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

    print("\nComplete Class Distribution")
    print("-" * 76)

    for class_id, count in enumerate(counts):
        print(
            f"Class {class_id} "
            f"({CLASS_NAMES[class_id]}): {count}"
        )


# Reuse Phase 9 split

def load_phase9_split(
    x_data: np.ndarray,
    y_data: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Load the exact train-validation indices used by Phase 9.
    """

    train_indices = np.load(
        TRAIN_INDICES_PATH
    ).astype(np.int64)

    validation_indices = np.load(
        VALIDATION_INDICES_PATH
    ).astype(np.int64)

    combined_indices = np.concatenate(
        [
            train_indices,
            validation_indices,
        ]
    )

    if len(np.unique(combined_indices)) != len(x_data):
        raise ValueError(
            "Phase 9 train-validation indices do not form a "
            "complete non-overlapping split of the current dataset."
        )

    if np.min(combined_indices) < 0:
        raise ValueError(
            "Negative sample index detected."
        )

    if np.max(combined_indices) >= len(x_data):
        raise ValueError(
            "A Phase 9 index exceeds the current dataset size."
        )

    x_train = x_data[train_indices]
    y_train = y_data[train_indices]

    x_validation = x_data[validation_indices]
    y_validation = y_data[validation_indices]

    np.save(
        TRAIN_INDICES_COPY_PATH,
        train_indices,
    )

    np.save(
        VALIDATION_INDICES_COPY_PATH,
        validation_indices,
    )

    return (
        x_train,
        y_train,
        x_validation,
        y_validation,
        train_indices,
        validation_indices,
    )


def print_split_information(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
) -> None:
    """Display the reused Phase 9 split."""

    print("\nReused Phase 9 Train-Validation Split")
    print("-" * 76)

    print(f"Training images     : {x_train.shape}")
    print(f"Training labels     : {y_train.shape}")
    print(f"Validation images   : {x_validation.shape}")
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

def load_phase9_class_weights() -> dict[int, float]:
    """Load exactly the same class weights used in Phase 9."""

    with PHASE9_CLASS_WEIGHTS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        stored_weights = json.load(file)

    class_weights = {
        int(class_id): float(weight)
        for class_id, weight in stored_weights.items()
    }

    expected_classes = set(range(6))

    if set(class_weights.keys()) != expected_classes:
        raise ValueError(
            "Phase 9 class-weight file does not contain "
            "exactly classes 0 through 5."
        )

    with CLASS_WEIGHTS_COPY_PATH.open(
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

    print("\nReused Phase 9 Class Weights")
    print("-" * 76)

    for class_id, weight in class_weights.items():
        print(
            f"Class {class_id} "
            f"({CLASS_NAMES[class_id]}): {weight:.6f}"
        )

    return class_weights


# Callbacks

def create_callbacks() -> list[tf.keras.callbacks.Callback]:
    """Create callbacks matching the Phase 9 experiment."""

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


# Experiment recording

def save_training_configuration(
    training_samples: int,
    validation_samples: int,
    normalization_applied: bool,
) -> None:
    """Save the complete controlled-experiment configuration."""

    configuration = {
        "phase": "Phase 10 - Classical TI-RADS Ablation",
        "experiment_type": (
            "Controlled raw-image versus quantum-feature comparison"
        ),
        "random_seed": RANDOM_SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "training_samples": training_samples,
        "validation_samples": validation_samples,
        "input_shape": [128, 128, 1],
        "internal_resize": [64, 64],
        "number_of_classes": 6,
        "optimizer": "Adam",
        "initial_learning_rate": 0.0001,
        "loss": "sparse_categorical_crossentropy",
        "imbalance_method": "Same class weights as Phase 9",
        "smote_used": False,
        "augmentation_used": False,
        "test_set_used_during_training": False,
        "phase9_indices_reused": True,
        "phase9_class_weights_reused": True,
        "automatic_division_by_255": normalization_applied,
        "model_parameters": 1062150,
        "architecture": [
            "Resize 128x128 to 64x64",
            "Conv2D 16",
            "MaxPooling2D",
            "Conv2D 32",
            "MaxPooling2D",
            "Flatten",
            "Dense 128",
            "Dense 64",
            "Softmax 6",
        ],
        "controlled_difference": {
            "phase9_input": "64x64x4 quantum feature maps",
            "phase10_input": "128x128x1 grayscale images",
            "common_training_split": True,
            "common_validation_split": True,
            "common_class_weights": True,
            "common_random_seed": True,
            "common_optimizer": True,
            "common_learning_rate": True,
            "common_batch_size": True,
        },
        "class_mapping": {
            "0": "TI-RADS 2",
            "1": "TI-RADS 3",
            "2": "TI-RADS 4a",
            "3": "TI-RADS 4b",
            "4": "TI-RADS 4c",
            "5": "TI-RADS 5",
        },
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
    """Save all training history values as JSON."""

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


# Main training procedure

def main() -> None:
    """Train the Phase 10 controlled classical baseline."""

    print("=" * 76)
    print("PHASE 10 : TRAIN CLASSICAL TI-RADS ABLATION MODEL")
    print("=" * 76)

    np.random.seed(RANDOM_SEED)
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    tf.keras.backend.clear_session()

    x_data, y_data = load_original_image_dataset()

    normalization_applied = bool(
        np.max(np.load(X_DATA_PATH)) > 1.5
    )

    print_dataset_information(
        x_data,
        y_data,
    )

    (
        x_train,
        y_train,
        x_validation,
        y_validation,
        _,
        _,
    ) = load_phase9_split(
        x_data,
        y_data,
    )

    print_split_information(
        x_train,
        y_train,
        x_validation,
        y_validation,
    )

    class_weights = load_phase9_class_weights()

    print("\nBuilding classical TI-RADS model...")

    model = build_classical_tirads_model()

    model.summary()

    save_training_configuration(
        training_samples=len(x_train),
        validation_samples=len(x_validation),
        normalization_applied=normalization_applied,
    )

    callbacks = create_callbacks()

    print("\nStarting controlled classical training...")
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

    # EarlyStopping restores the weights from the best validation-loss epoch.
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

    last_recorded_training_accuracy = float(
        history.history["accuracy"][-1]
    )

    print("\n" + "=" * 76)
    print("PHASE 10 CLASSICAL TRAINING COMPLETE")
    print("=" * 76)

    print(f"Best epoch                      : {best_epoch}")

    print(
        "Best validation loss            : "
        f"{best_validation_loss:.6f}"
    )

    print(
        "Best validation accuracy        : "
        f"{best_validation_accuracy * 100:.2f}%"
    )

    print(
        "Last recorded training accuracy : "
        f"{last_recorded_training_accuracy * 100:.2f}%"
    )

    print(f"Best model saved at             : {BEST_MODEL_PATH}")
    print(f"Restored final model saved at   : {FINAL_MODEL_PATH}")
    print(f"Training history saved at       : {HISTORY_PATH}")
    print(f"Training log saved at           : {CSV_LOG_PATH}")
    print(f"Copied class weights saved at   : {CLASS_WEIGHTS_COPY_PATH}")
    print(f"Copied train indices saved at   : {TRAIN_INDICES_COPY_PATH}")
    print(
        "Copied validation indices at    : "
        f"{VALIDATION_INDICES_COPY_PATH}"
    )

    print("\nControlled comparison safeguards:")
    print("Same original 279 samples              : Yes")
    print("Same Phase 9 training indices          : Yes")
    print("Same Phase 9 validation indices        : Yes")
    print("Same Phase 9 class weights             : Yes")
    print("Same optimizer and learning rate       : Yes")
    print("Same batch size and callback settings  : Yes")
    print("Untouched 70-image test set used       : No")

    print("=" * 76)


if __name__ == "__main__":
    main()