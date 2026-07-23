
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

# Reduce unnecessary TensorFlow console output.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers


# =============================================================================
# Project paths
# =============================================================================

SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIR.parent

PHASE13_DATA_DIR = (
    PROJECT_ROOT
    / "03_Dataset_Processed"
    / "phase13_roi_dataset"
)

RESULT_DIR = (
    PROJECT_ROOT
    / "07_Results"
    / "phase14_roi_categorical_baseline"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "06_Models"
    / "phase14_roi_categorical_baseline"
)

RESULT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Input files
# =============================================================================

X_TRAIN_PATH = PHASE13_DATA_DIR / "X_roi_train.npy"
Y_TRAIN_PATH = PHASE13_DATA_DIR / "y_roi_train.npy"
CASE_TRAIN_PATH = PHASE13_DATA_DIR / "case_ids_roi_train.npy"
ROI_TRAIN_PATH = PHASE13_DATA_DIR / "roi_ids_train.npy"

X_VALIDATION_PATH = PHASE13_DATA_DIR / "X_roi_validation.npy"
Y_VALIDATION_PATH = PHASE13_DATA_DIR / "y_roi_validation.npy"
CASE_VALIDATION_PATH = PHASE13_DATA_DIR / "case_ids_roi_validation.npy"
ROI_VALIDATION_PATH = PHASE13_DATA_DIR / "roi_ids_validation.npy"

X_TEST_PATH = PHASE13_DATA_DIR / "X_roi_test.npy"
Y_TEST_PATH = PHASE13_DATA_DIR / "y_roi_test.npy"
CASE_TEST_PATH = PHASE13_DATA_DIR / "case_ids_roi_test.npy"
ROI_TEST_PATH = PHASE13_DATA_DIR / "roi_ids_test.npy"


# =============================================================================
# Output files
# =============================================================================

BEST_MODEL_PATH = MODEL_DIR / "best_roi_categorical_baseline.keras"
FINAL_MODEL_PATH = MODEL_DIR / "final_roi_categorical_baseline.keras"

TRAINING_HISTORY_CSV = RESULT_DIR / "training_history.csv"
TRAINING_CURVES_PATH = RESULT_DIR / "training_curves.png"

ROI_PREDICTIONS_PATH = RESULT_DIR / "test_roi_predictions.csv"
CASE_PREDICTIONS_PATH = RESULT_DIR / "test_case_predictions.csv"

ROI_CONFUSION_MATRIX_CSV = RESULT_DIR / "roi_confusion_matrix.csv"
CASE_CONFUSION_MATRIX_CSV = RESULT_DIR / "case_confusion_matrix.csv"

ROI_CONFUSION_MATRIX_PNG = RESULT_DIR / "roi_confusion_matrix.png"
CASE_CONFUSION_MATRIX_PNG = RESULT_DIR / "case_confusion_matrix.png"

ROI_CLASSIFICATION_REPORT_CSV = (
    RESULT_DIR / "roi_classification_report.csv"
)

CASE_CLASSIFICATION_REPORT_CSV = (
    RESULT_DIR / "case_classification_report.csv"
)

METRICS_JSON_PATH = RESULT_DIR / "evaluation_metrics.json"
SUMMARY_PATH = RESULT_DIR / "phase14_summary.txt"
CONFIG_PATH = RESULT_DIR / "training_configuration.json"


# =============================================================================
# Configuration
# =============================================================================

RANDOM_SEED = 42

IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
CHANNELS = 1
NUM_CLASSES = 6

BATCH_SIZE = 16
MAX_EPOCHS = 100
INITIAL_LEARNING_RATE = 0.001

EARLY_STOPPING_PATIENCE = 18
REDUCE_LR_PATIENCE = 7
REDUCE_LR_FACTOR = 0.5
MINIMUM_LEARNING_RATE = 1e-6

TIRADS_LABELS = [
    "2",
    "3",
    "4a",
    "4b",
    "4c",
    "5",
]

CLASS_NAMES = [
    "TI-RADS 2",
    "TI-RADS 3",
    "TI-RADS 4a",
    "TI-RADS 4b",
    "TI-RADS 4c",
    "TI-RADS 5",
]


# =============================================================================
# Reproducibility
# =============================================================================

def set_reproducibility(seed: int) -> None:
    """
    Set available random seeds for reproducible execution.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    try:
        tf.config.experimental.enable_op_determinism()
        print("TensorFlow deterministic operations : ENABLED")
    except Exception:
        print("TensorFlow deterministic operations : NOT AVAILABLE")


# =============================================================================
# Data loading and validation
# =============================================================================

def require_file(path: Path) -> None:
    """
    Raise a clear error if a required file is absent.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Required Phase 13 file was not found:\n{path}"
        )


def validate_input_files() -> None:
    """
    Confirm that all frozen Phase 13 arrays exist.
    """

    required_paths = [
        X_TRAIN_PATH,
        Y_TRAIN_PATH,
        CASE_TRAIN_PATH,
        ROI_TRAIN_PATH,
        X_VALIDATION_PATH,
        Y_VALIDATION_PATH,
        CASE_VALIDATION_PATH,
        ROI_VALIDATION_PATH,
        X_TEST_PATH,
        Y_TEST_PATH,
        CASE_TEST_PATH,
        ROI_TEST_PATH,
    ]

    for path in required_paths:
        require_file(path)


def load_string_array(path: Path) -> np.ndarray:
    """
    Load case or ROI identifiers safely as strings.
    """

    return np.load(
        path,
        allow_pickle=False,
    ).astype(str)


def load_dataset() -> dict[str, dict[str, np.ndarray]]:
    """
    Load the complete frozen Phase 13 dataset.
    """

    dataset = {
        "train": {
            "X": np.load(X_TRAIN_PATH),
            "y": np.load(Y_TRAIN_PATH),
            "case_ids": load_string_array(CASE_TRAIN_PATH),
            "roi_ids": load_string_array(ROI_TRAIN_PATH),
        },
        "validation": {
            "X": np.load(X_VALIDATION_PATH),
            "y": np.load(Y_VALIDATION_PATH),
            "case_ids": load_string_array(CASE_VALIDATION_PATH),
            "roi_ids": load_string_array(ROI_VALIDATION_PATH),
        },
        "test": {
            "X": np.load(X_TEST_PATH),
            "y": np.load(Y_TEST_PATH),
            "case_ids": load_string_array(CASE_TEST_PATH),
            "roi_ids": load_string_array(ROI_TEST_PATH),
        },
    }

    return dataset


def validate_split(
    split_name: str,
    split_data: dict[str, np.ndarray],
) -> None:
    """
    Validate shapes, labels, identifiers and pixel ranges.
    """

    X = split_data["X"]
    y = split_data["y"]
    case_ids = split_data["case_ids"]
    roi_ids = split_data["roi_ids"]

    expected_shape = (
        len(X),
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        CHANNELS,
    )

    if X.shape != expected_shape:
        raise ValueError(
            f"{split_name}: unexpected image shape {X.shape}. "
            f"Expected {expected_shape}."
        )

    if len(X) != len(y):
        raise ValueError(
            f"{split_name}: image and label counts differ."
        )

    if len(X) != len(case_ids):
        raise ValueError(
            f"{split_name}: image and case-ID counts differ."
        )

    if len(X) != len(roi_ids):
        raise ValueError(
            f"{split_name}: image and ROI-ID counts differ."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            f"{split_name}: non-finite image values found."
        )

    if float(X.min()) < 0.0 or float(X.max()) > 1.0:
        raise ValueError(
            f"{split_name}: images are not normalized to [0, 1]."
        )

    unique_labels = set(
        np.unique(y).astype(int).tolist()
    )

    valid_labels = set(range(NUM_CLASSES))

    if not unique_labels.issubset(valid_labels):
        raise ValueError(
            f"{split_name}: unsupported labels found: "
            f"{sorted(unique_labels)}"
        )

    if len(set(roi_ids.tolist())) != len(roi_ids):
        raise ValueError(
            f"{split_name}: duplicated ROI IDs found."
        )


def validate_dataset(
    dataset: dict[str, dict[str, np.ndarray]],
) -> None:
    """
    Validate all splits and confirm zero case and ROI leakage.
    """

    for split_name, split_data in dataset.items():
        validate_split(split_name, split_data)

    train_cases = set(dataset["train"]["case_ids"].tolist())
    validation_cases = set(dataset["validation"]["case_ids"].tolist())
    test_cases = set(dataset["test"]["case_ids"].tolist())

    train_rois = set(dataset["train"]["roi_ids"].tolist())
    validation_rois = set(dataset["validation"]["roi_ids"].tolist())
    test_rois = set(dataset["test"]["roi_ids"].tolist())

    case_overlaps = {
        "train_validation": len(train_cases & validation_cases),
        "train_test": len(train_cases & test_cases),
        "validation_test": len(validation_cases & test_cases),
    }

    roi_overlaps = {
        "train_validation": len(train_rois & validation_rois),
        "train_test": len(train_rois & test_rois),
        "validation_test": len(validation_rois & test_rois),
    }

    if any(case_overlaps.values()):
        raise RuntimeError(
            f"Case leakage detected: {case_overlaps}"
        )

    if any(roi_overlaps.values()):
        raise RuntimeError(
            f"ROI leakage detected: {roi_overlaps}"
        )


# =============================================================================
# Class weighting
# =============================================================================

def calculate_class_weights(
    y_train: np.ndarray,
) -> dict[int, float]:
    """
    Calculate balanced class weights using training labels only.
    """

    classes = np.arange(NUM_CLASSES)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train.astype(int),
    )

    return {
        int(class_index): float(weight)
        for class_index, weight in zip(classes, weights)
    }


# =============================================================================
# Model
# =============================================================================

def build_model() -> keras.Model:
    """
    Build a compact CPU-friendly categorical CNN.
    """

    augmentation = keras.Sequential(
        [
            layers.RandomFlip(
                mode="horizontal",
                seed=RANDOM_SEED,
            ),
            layers.RandomRotation(
                factor=0.05,
                fill_mode="reflect",
                seed=RANDOM_SEED,
            ),
            layers.RandomZoom(
                height_factor=(-0.08, 0.08),
                width_factor=(-0.08, 0.08),
                fill_mode="reflect",
                seed=RANDOM_SEED,
            ),
            layers.RandomTranslation(
                height_factor=0.05,
                width_factor=0.05,
                fill_mode="reflect",
                seed=RANDOM_SEED,
            ),
        ],
        name="training_augmentation",
    )

    inputs = keras.Input(
        shape=(
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            CHANNELS,
        ),
        name="roi_image",
    )

    x = augmentation(inputs)

    x = layers.Conv2D(
        16,
        kernel_size=3,
        padding="same",
        use_bias=False,
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(
        16,
        kernel_size=3,
        padding="same",
        use_bias=False,
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(0.15)(x)

    x = layers.Conv2D(
        32,
        kernel_size=3,
        padding="same",
        use_bias=False,
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(
        32,
        kernel_size=3,
        padding="same",
        use_bias=False,
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(0.20)(x)

    x = layers.Conv2D(
        64,
        kernel_size=3,
        padding="same",
        use_bias=False,
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(
        64,
        kernel_size=3,
        padding="same",
        use_bias=False,
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(
        96,
        kernel_size=3,
        padding="same",
        use_bias=False,
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(
        64,
        activation="relu",
    )(x)

    x = layers.Dropout(0.35)(x)

    outputs = layers.Dense(
        NUM_CLASSES,
        activation="softmax",
        name="tirads_probabilities",
    )(x)

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="phase14_roi_categorical_cnn",
    )

    optimizer = keras.optimizers.Adam(
        learning_rate=INITIAL_LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
            )
        ],
    )

    return model


# =============================================================================
# Metrics
# =============================================================================

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Calculate the required multiclass metrics.
    """

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "cohen_kappa": float(
            cohen_kappa_score(y_true, y_pred)
        ),
        "quadratic_weighted_kappa": float(
            cohen_kappa_score(
                y_true,
                y_pred,
                weights="quadratic",
            )
        ),
        "mean_absolute_class_error": float(
            np.mean(
                np.abs(
                    y_true.astype(float)
                    - y_pred.astype(float)
                )
            )
        ),
    }


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    """
    Save per-class precision, recall, F1 and support.
    """

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(output_path, index=True)


# =============================================================================
# Case-level aggregation
# =============================================================================

def aggregate_case_predictions(
    case_ids: np.ndarray,
    roi_ids: np.ndarray,
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """
    Aggregate multiple ROI predictions belonging to the same case.

    Probability vectors are averaged within each case. The case prediction is
    the class with the highest mean probability.
    """

    rows = []

    unique_cases = sorted(
        np.unique(case_ids).tolist()
    )

    for case_id in unique_cases:
        indices = np.where(case_ids == case_id)[0]

        case_true_labels = np.unique(
            y_true[indices]
        )

        if len(case_true_labels) != 1:
            raise RuntimeError(
                f"Case {case_id} contains inconsistent labels: "
                f"{case_true_labels.tolist()}"
            )

        mean_probabilities = probabilities[
            indices
        ].mean(axis=0)

        predicted_class = int(
            np.argmax(mean_probabilities)
        )

        true_class = int(
            case_true_labels[0]
        )

        row = {
            "case_id": str(case_id),
            "roi_count": int(len(indices)),
            "roi_ids": "|".join(
                roi_ids[indices].astype(str).tolist()
            ),
            "true_class": true_class,
            "true_tirads": TIRADS_LABELS[true_class],
            "predicted_class": predicted_class,
            "predicted_tirads": TIRADS_LABELS[predicted_class],
            "correct": bool(true_class == predicted_class),
            "absolute_class_error": int(
                abs(true_class - predicted_class)
            ),
        }

        for class_index, label in enumerate(TIRADS_LABELS):
            row[f"probability_tirads_{label}"] = float(
                mean_probabilities[class_index]
            )

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Plotting
# =============================================================================

def save_training_curves(
    history: keras.callbacks.History,
) -> None:
    """
    Save loss and accuracy curves in one image.
    """

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(
        TRAINING_HISTORY_CSV,
        index_label="epoch_index",
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(13, 5),
    )

    epoch_numbers = np.arange(
        1,
        len(history_df) + 1,
    )

    axes[0].plot(
        epoch_numbers,
        history_df["loss"],
        label="Training loss",
    )

    axes[0].plot(
        epoch_numbers,
        history_df["val_loss"],
        label="Validation loss",
    )

    axes[0].set_title("Categorical cross-entropy loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(
        epoch_numbers,
        history_df["accuracy"],
        label="Training accuracy",
    )

    axes[1].plot(
        epoch_numbers,
        history_df["val_accuracy"],
        label="Validation accuracy",
    )

    axes[1].set_title("Classification accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(
        TRAINING_CURVES_PATH,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_confusion_matrix(
    matrix: np.ndarray,
    csv_path: Path,
    png_path: Path,
    title: str,
) -> None:
    """
    Save confusion matrix as CSV and PNG.
    """

    matrix_df = pd.DataFrame(
        matrix,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    )

    matrix_df.to_csv(csv_path)

    figure, axis = plt.subplots(
        figsize=(8, 7)
    )

    image = axis.imshow(
        matrix,
        interpolation="nearest",
        cmap="Blues",
    )

    figure.colorbar(image, ax=axis)

    axis.set(
        xticks=np.arange(NUM_CLASSES),
        yticks=np.arange(NUM_CLASSES),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ylabel="True class",
        xlabel="Predicted class",
        title=title,
    )

    plt.setp(
        axis.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    threshold = (
        matrix.max() / 2.0
        if matrix.size
        else 0
    )

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                str(matrix[row_index, column_index]),
                ha="center",
                va="center",
                color=(
                    "white"
                    if matrix[row_index, column_index] > threshold
                    else "black"
                ),
            )

    figure.tight_layout()

    figure.savefig(
        png_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(figure)


# =============================================================================
# Output reports
# =============================================================================

def save_configuration(
    class_weights: dict[int, float],
    model: keras.Model,
) -> None:
    """
    Save the frozen training configuration.
    """

    configuration = {
        "phase": 14,
        "task": "six-class ROI categorical TI-RADS classification",
        "random_seed": RANDOM_SEED,
        "input_shape": [
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            CHANNELS,
        ],
        "number_of_classes": NUM_CLASSES,
        "class_mapping": {
            str(index): label
            for index, label in enumerate(TIRADS_LABELS)
        },
        "batch_size": BATCH_SIZE,
        "maximum_epochs": MAX_EPOCHS,
        "initial_learning_rate": INITIAL_LEARNING_RATE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "reduce_lr_patience": REDUCE_LR_PATIENCE,
        "reduce_lr_factor": REDUCE_LR_FACTOR,
        "minimum_learning_rate": MINIMUM_LEARNING_RATE,
        "class_weights": {
            str(key): value
            for key, value in class_weights.items()
        },
        "augmentation": {
            "horizontal_flip": True,
            "rotation_factor": 0.05,
            "zoom_factor": 0.08,
            "translation_factor": 0.05,
            "validation_augmentation": False,
            "test_augmentation": False,
        },
        "smote_used": False,
        "model_parameters": int(model.count_params()),
        "tensorflow_version": tf.__version__,
        "numpy_version": np.__version__,
    }

    CONFIG_PATH.write_text(
        json.dumps(
            configuration,
            indent=4,
        )
        + "\n",
        encoding="utf-8",
    )


def save_summary(
    history: keras.callbacks.History,
    roi_metrics: dict[str, float],
    case_metrics: dict[str, float],
    class_weights: dict[int, float],
    roi_count: int,
    case_count: int,
) -> None:
    """
    Save a readable final summary.
    """

    history_df = pd.DataFrame(history.history)

    best_epoch_index = int(
        history_df["val_loss"].idxmin()
    )

    best_epoch = best_epoch_index + 1

    lines = [
        "PHASE 14 - ROI CATEGORICAL CNN BASELINE",
        "=" * 78,
        "",
        "Experiment",
        "-" * 78,
        "Task                         : Six-class TI-RADS classification",
        "Input                        : Frozen Phase 13 ROI arrays",
        f"Input shape                  : {IMAGE_HEIGHT} x {IMAGE_WIDTH} x {CHANNELS}",
        "Augmentation                 : Training only",
        "Class balancing              : Training class weights",
        "SMOTE                        : Not used",
        "Validation set               : Used for checkpoint selection",
        "Test set                     : Evaluated after training",
        "",
        "Training",
        "-" * 78,
        f"Maximum epochs               : {MAX_EPOCHS}",
        f"Completed epochs             : {len(history_df)}",
        f"Best validation-loss epoch   : {best_epoch}",
        f"Best validation loss         : {history_df.loc[best_epoch_index, 'val_loss']:.6f}",
        f"Best validation accuracy     : {history_df.loc[best_epoch_index, 'val_accuracy']:.6f}",
        "",
        "Training class weights",
        "-" * 78,
    ]

    for class_index, label in enumerate(TIRADS_LABELS):
        lines.append(
            f"TI-RADS {label:<2}                  : "
            f"{class_weights[class_index]:.6f}"
        )

    lines.extend(
        [
            "",
            f"ROI-level test evaluation ({roi_count} ROIs)",
            "-" * 78,
        ]
    )

    for metric_name, metric_value in roi_metrics.items():
        lines.append(
            f"{metric_name:<30}: {metric_value:.6f}"
        )

    lines.extend(
        [
            "",
            f"Case-level test evaluation ({case_count} cases)",
            "-" * 78,
        ]
    )

    for metric_name, metric_value in case_metrics.items():
        lines.append(
            f"{metric_name:<30}: {metric_value:.6f}"
        )

    lines.extend(
        [
            "",
            "Interpretation rules",
            "-" * 78,
            (
                "ROI-level metrics treat each valid annotation region as one "
                "prediction unit."
            ),
            (
                "Case-level metrics average all ROI probability vectors from "
                "the same patient case."
            ),
            (
                "Macro F1 and balanced accuracy are important because the "
                "TI-RADS classes are imbalanced."
            ),
            (
                "Quadratic weighted kappa and mean absolute class error are "
                "included because TI-RADS categories are ordered."
            ),
            "",
            "Generated model",
            "-" * 78,
            str(BEST_MODEL_PATH),
            "",
            "Result directory",
            "-" * 78,
            str(RESULT_DIR),
        ]
    )

    SUMMARY_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# =============================================================================
# Main execution
# =============================================================================

def main() -> None:
    """
    Execute Phase 14.
    """

    print("=" * 78)
    print("PHASE 14 : TRAIN ROI CATEGORICAL CNN BASELINE")
    print("=" * 78)

    set_reproducibility(RANDOM_SEED)

    print("\nTensorFlow environment")
    print("-" * 78)
    print(f"TensorFlow version : {tf.__version__}")
    print(f"Available GPUs     : {len(tf.config.list_physical_devices('GPU'))}")
    print("Execution device   : CPU is expected on this laptop")

    print("\nValidating Phase 13 files...")
    validate_input_files()

    print("Loading frozen Phase 13 arrays...")
    dataset = load_dataset()

    print("Validating shapes and leakage...")
    validate_dataset(dataset)

    for split_name in [
        "train",
        "validation",
        "test",
    ]:
        split_data = dataset[split_name]

        print(
            f"{split_name.capitalize():10s} "
            f"images={len(split_data['X']):3d}, "
            f"cases={len(np.unique(split_data['case_ids'])):3d}, "
            f"shape={split_data['X'].shape}"
        )

    X_train = dataset["train"]["X"].astype(np.float32)
    y_train = dataset["train"]["y"].astype(np.int64)

    X_validation = dataset["validation"]["X"].astype(np.float32)
    y_validation = dataset["validation"]["y"].astype(np.int64)

    X_test = dataset["test"]["X"].astype(np.float32)
    y_test = dataset["test"]["y"].astype(np.int64)

    test_case_ids = dataset["test"]["case_ids"]
    test_roi_ids = dataset["test"]["roi_ids"]

    print("\nCalculating training class weights...")

    class_weights = calculate_class_weights(y_train)

    for class_index, label in enumerate(TIRADS_LABELS):
        class_count = int(
            np.sum(y_train == class_index)
        )

        print(
            f"TI-RADS {label:<2} : "
            f"count={class_count:3d}, "
            f"weight={class_weights[class_index]:.6f}"
        )

    print("\nBuilding categorical CNN...")

    model = build_model()
    model.summary()

    save_configuration(
        class_weights=class_weights,
        model=model,
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=BEST_MODEL_PATH,
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=REDUCE_LR_FACTOR,
            patience=REDUCE_LR_PATIENCE,
            min_lr=MINIMUM_LEARNING_RATE,
            verbose=1,
        ),
        keras.callbacks.TerminateOnNaN(),
    ]

    print("\nStarting model training...")
    print("-" * 78)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(
            X_validation,
            y_validation,
        ),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        shuffle=True,
        verbose=2,
    )

    model.save(FINAL_MODEL_PATH)

    save_training_curves(history)

    print("\nLoading best validation checkpoint...")

    best_model = keras.models.load_model(
        BEST_MODEL_PATH
    )

    print("\nEvaluating untouched test set...")

    test_loss, test_accuracy = best_model.evaluate(
        X_test,
        y_test,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    test_probabilities = best_model.predict(
        X_test,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    test_predictions = np.argmax(
        test_probabilities,
        axis=1,
    ).astype(int)

    roi_metrics = calculate_metrics(
        y_true=y_test,
        y_pred=test_predictions,
    )

    roi_metrics["categorical_crossentropy_loss"] = float(
        test_loss
    )

    roi_metrics["keras_accuracy"] = float(
        test_accuracy
    )

    roi_prediction_df = pd.DataFrame(
        {
            "roi_id": test_roi_ids.astype(str),
            "case_id": test_case_ids.astype(str),
            "true_class": y_test.astype(int),
            "true_tirads": [
                TIRADS_LABELS[index]
                for index in y_test.astype(int)
            ],
            "predicted_class": test_predictions,
            "predicted_tirads": [
                TIRADS_LABELS[index]
                for index in test_predictions
            ],
            "correct": (
                y_test == test_predictions
            ),
            "absolute_class_error": np.abs(
                y_test - test_predictions
            ).astype(int),
        }
    )

    for class_index, label in enumerate(TIRADS_LABELS):
        roi_prediction_df[
            f"probability_tirads_{label}"
        ] = test_probabilities[:, class_index]

    roi_prediction_df.to_csv(
        ROI_PREDICTIONS_PATH,
        index=False,
    )

    case_prediction_df = aggregate_case_predictions(
        case_ids=test_case_ids,
        roi_ids=test_roi_ids,
        y_true=y_test,
        probabilities=test_probabilities,
    )

    case_prediction_df.to_csv(
        CASE_PREDICTIONS_PATH,
        index=False,
    )

    case_y_true = (
        case_prediction_df["true_class"]
        .to_numpy(dtype=int)
    )

    case_y_pred = (
        case_prediction_df["predicted_class"]
        .to_numpy(dtype=int)
    )

    case_metrics = calculate_metrics(
        y_true=case_y_true,
        y_pred=case_y_pred,
    )

    roi_matrix = confusion_matrix(
        y_test,
        test_predictions,
        labels=list(range(NUM_CLASSES)),
    )

    case_matrix = confusion_matrix(
        case_y_true,
        case_y_pred,
        labels=list(range(NUM_CLASSES)),
    )

    save_confusion_matrix(
        matrix=roi_matrix,
        csv_path=ROI_CONFUSION_MATRIX_CSV,
        png_path=ROI_CONFUSION_MATRIX_PNG,
        title="Phase 14 ROI-Level Confusion Matrix",
    )

    save_confusion_matrix(
        matrix=case_matrix,
        csv_path=CASE_CONFUSION_MATRIX_CSV,
        png_path=CASE_CONFUSION_MATRIX_PNG,
        title="Phase 14 Case-Level Confusion Matrix",
    )

    save_classification_report(
        y_true=y_test,
        y_pred=test_predictions,
        output_path=ROI_CLASSIFICATION_REPORT_CSV,
    )

    save_classification_report(
        y_true=case_y_true,
        y_pred=case_y_pred,
        output_path=CASE_CLASSIFICATION_REPORT_CSV,
    )

    metrics_payload: dict[str, Any] = {
        "roi_level": roi_metrics,
        "case_level": case_metrics,
        "test_roi_count": int(len(y_test)),
        "test_case_count": int(len(case_prediction_df)),
        "class_mapping": {
            str(index): label
            for index, label in enumerate(TIRADS_LABELS)
        },
    }

    METRICS_JSON_PATH.write_text(
        json.dumps(
            metrics_payload,
            indent=4,
        )
        + "\n",
        encoding="utf-8",
    )

    save_summary(
        history=history,
        roi_metrics=roi_metrics,
        case_metrics=case_metrics,
        class_weights=class_weights,
        roi_count=len(y_test),
        case_count=len(case_prediction_df),
    )

    print("\n" + "=" * 78)
    print("PHASE 14 CATEGORICAL BASELINE COMPLETE")
    print("=" * 78)

    print("\nROI-level test metrics")
    print("-" * 78)

    for metric_name, metric_value in roi_metrics.items():
        print(
            f"{metric_name:<32}: {metric_value:.6f}"
        )

    print("\nCase-level test metrics")
    print("-" * 78)

    for metric_name, metric_value in case_metrics.items():
        print(
            f"{metric_name:<32}: {metric_value:.6f}"
        )

    print("\nTraining summary")
    print("-" * 78)
    print(f"Completed epochs : {len(history.history['loss'])}")
    print(f"Best model       : {BEST_MODEL_PATH}")

    print("\nGenerated results")
    print("-" * 78)
    print(RESULT_DIR)

    print("\nImportant")
    print("-" * 78)
    print("The frozen Phase 12 split was not modified.")
    print("The frozen Phase 13 arrays were not modified.")
    print("Augmentation was applied only during training.")
    print("Class weights were derived only from training labels.")
    print("No SMOTE was used.")
    print("=" * 78)


if __name__ == "__main__":
    main()