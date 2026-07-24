from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIR.parent

MODEL_DIR = PROJECT_ROOT / "06_Models"
RESULT_DIR = PROJECT_ROOT / "07_Results" / "phase8_tirads"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "best_tirads_qucnet.keras"

X_TEST_PATH = MODEL_DIR / "X_tirads_test_quantum.npy"
Y_TEST_PATH = MODEL_DIR / "y_tirads_test.npy"

HISTORY_PATH = RESULT_DIR / "training_history.json"

METRICS_TEXT_PATH = RESULT_DIR / "metrics.txt"
METRICS_JSON_PATH = RESULT_DIR / "metrics.json"
REPORT_PATH = RESULT_DIR / "classification_report.txt"
PREDICTIONS_PATH = RESULT_DIR / "predictions.csv"

CONFUSION_MATRIX_PATH = RESULT_DIR / "confusion_matrix.png"
NORMALIZED_CONFUSION_MATRIX_PATH = (
    RESULT_DIR / "normalized_confusion_matrix.png"
)

ACCURACY_PLOT_PATH = RESULT_DIR / "accuracy_plot.png"
LOSS_PLOT_PATH = RESULT_DIR / "loss_plot.png"


CLASS_NAMES = [
    "TI-RADS 2",
    "TI-RADS 3",
    "TI-RADS 4a",
    "TI-RADS 4b",
    "TI-RADS 4c",
    "TI-RADS 5",
]


def validate_required_files() -> None:
    """Ensure that all required model and dataset files exist."""

    required_files = [
        BEST_MODEL_PATH,
        X_TEST_PATH,
        Y_TEST_PATH,
        HISTORY_PATH,
    ]

    missing_files = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Required files were not found:\n"
            + "\n".join(missing_files)
        )


def load_test_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Load and validate the validation-selected TI-RADS holdout dataset."""

    x_test = np.load(X_TEST_PATH).astype(np.float32)
    y_test = np.load(Y_TEST_PATH).astype(np.int32)

    if x_test.shape != (70, 64, 64, 4):
        print(
            "Warning: Expected test shape approximately "
            f"(70, 64, 64, 4), received {x_test.shape}"
        )

    if len(x_test) != len(y_test):
        raise ValueError(
            f"Test feature-label mismatch: "
            f"{len(x_test)} features and {len(y_test)} labels."
        )

    if np.isnan(x_test).any():
        raise ValueError("NaN values found in test features.")

    if np.isinf(x_test).any():
        raise ValueError("Infinite values found in test features.")

    return x_test, y_test


def save_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    """Save individual predictions and probabilities to CSV."""

    probability_headers = [
        f"probability_{name.replace(' ', '_').replace('-', '_')}"
        for name in CLASS_NAMES
    ]

    with PREDICTIONS_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "sample_index",
                "true_class_id",
                "true_class_name",
                "predicted_class_id",
                "predicted_class_name",
                "confidence",
                "correct",
                *probability_headers,
            ]
        )

        for index, (true_label, predicted_label) in enumerate(
            zip(y_true, y_pred)
        ):
            writer.writerow(
                [
                    index,
                    int(true_label),
                    CLASS_NAMES[int(true_label)],
                    int(predicted_label),
                    CLASS_NAMES[int(predicted_label)],
                    float(probabilities[index, predicted_label]),
                    bool(true_label == predicted_label),
                    *[
                        float(value)
                        for value in probabilities[index]
                    ],
                ]
            )


def plot_confusion_matrix(
    matrix: np.ndarray,
    output_path: Path,
    title: str,
    normalized: bool = False,
) -> None:
    """Create and save a confusion-matrix plot."""

    display_matrix = matrix.astype(np.float64)

    if normalized:
        row_totals = display_matrix.sum(axis=1, keepdims=True)

        display_matrix = np.divide(
            display_matrix,
            row_totals,
            out=np.zeros_like(display_matrix),
            where=row_totals != 0,
        )

    figure, axis = plt.subplots(figsize=(9, 7))

    image = axis.imshow(display_matrix)

    figure.colorbar(image, ax=axis)

    axis.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )

    plt.setp(
        axis.get_xticklabels(),
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )

    threshold = (
        display_matrix.max() / 2.0
        if display_matrix.size
        else 0
    )

    for row in range(display_matrix.shape[0]):
        for column in range(display_matrix.shape[1]):
            value = display_matrix[row, column]

            label = (
                f"{value:.2f}"
                if normalized
                else str(int(value))
            )

            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def plot_training_history() -> None:
    """Create accuracy and loss plots from saved history."""

    with HISTORY_PATH.open("r", encoding="utf-8") as file:
        history = json.load(file)

    epochs = range(1, len(history["loss"]) + 1)

    figure = plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["accuracy"],
        label="Training accuracy",
    )

    plt.plot(
        epochs,
        history["val_accuracy"],
        label="Validation accuracy",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("TI-RADS QuCNet Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    figure.savefig(
        ACCURACY_PLOT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    figure = plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        history["loss"],
        label="Training loss",
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation loss",
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("TI-RADS QuCNet Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    figure.savefig(
        LOSS_PLOT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def main() -> None:
    """Evaluate the Phase 8 TI-RADS QuCNet baseline."""

    print("=" * 70)
    print("PHASE 8 : TI-RADS QUCNET EVALUATION")
    print("=" * 70)

    validate_required_files()

    print("\nLoading validation-selected holdout dataset...")

    x_test, y_test = load_test_dataset()

    print(f"Test features: {x_test.shape}")
    print(f"Test labels  : {y_test.shape}")

    print("\nLoading best TI-RADS QuCNet model...")

    model = tf.keras.models.load_model(BEST_MODEL_PATH)

    print("\nGenerating predictions...")

    probabilities = model.predict(
        x_test,
        batch_size=16,
        verbose=1,
    )

    y_pred = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_test, y_pred)

    macro_precision = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    weighted_precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    weighted_recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_test,
        y_pred,
        labels=np.arange(6),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=np.arange(6),
    )

    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "test_samples": int(len(y_test)),
        "number_of_classes": 6,
        "model_path": str(BEST_MODEL_PATH),
    }

    metrics_text = (
        "PHASE 8: TI-RADS QUCNET EVALUATION\n"
        + "=" * 50
        + "\n\n"
        + f"Test samples       : {len(y_test)}\n"
        + f"Accuracy           : {accuracy * 100:.2f}%\n"
        + f"Macro precision    : {macro_precision * 100:.2f}%\n"
        + f"Macro recall       : {macro_recall * 100:.2f}%\n"
        + f"Macro F1-score     : {macro_f1 * 100:.2f}%\n"
        + f"Weighted precision : {weighted_precision * 100:.2f}%\n"
        + f"Weighted recall    : {weighted_recall * 100:.2f}%\n"
        + f"Weighted F1-score  : {weighted_f1 * 100:.2f}%\n"
    )

    with METRICS_TEXT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(metrics_text)

    with METRICS_JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metrics, file, indent=4)

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(report)

    save_predictions(
        y_true=y_test,
        y_pred=y_pred,
        probabilities=probabilities,
    )

    plot_confusion_matrix(
        matrix=matrix,
        output_path=CONFUSION_MATRIX_PATH,
        title="TI-RADS QuCNet Confusion Matrix",
        normalized=False,
    )

    plot_confusion_matrix(
        matrix=matrix,
        output_path=NORMALIZED_CONFUSION_MATRIX_PATH,
        title="Normalized TI-RADS QuCNet Confusion Matrix",
        normalized=True,
    )

    plot_training_history()

    print("\n" + metrics_text)

    print("\nClassification Report")
    print("-" * 70)
    print(report)

    print("Confusion Matrix")
    print("-" * 70)
    print(matrix)

    print("\nGenerated files")
    print("-" * 70)
    print(METRICS_TEXT_PATH)
    print(METRICS_JSON_PATH)
    print(REPORT_PATH)
    print(PREDICTIONS_PATH)
    print(CONFUSION_MATRIX_PATH)
    print(NORMALIZED_CONFUSION_MATRIX_PATH)
    print(ACCURACY_PLOT_PATH)
    print(LOSS_PLOT_PATH)

    print("\n" + "=" * 70)
    print("PHASE 8 EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()