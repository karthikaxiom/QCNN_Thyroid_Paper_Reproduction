
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
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)




SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIR.parent

MODEL_DIR = PROJECT_ROOT / "06_Models"

RESULT_DIR = (
    PROJECT_ROOT
    / "07_Results"
    / "phase10_classical_tirads_class_weighted"
)

COMPARISON_DIR = (
    PROJECT_ROOT
    / "07_Results"
    / "final_reproduction_comparison"
)

RESULT_DIR.mkdir(parents=True, exist_ok=True)
COMPARISON_DIR.mkdir(parents=True, exist_ok=True)


BEST_MODEL_PATH = (
    MODEL_DIR / "best_phase10_classical_tirads.keras"
)

X_TEST_PATH = (
    MODEL_DIR / "X_tirads_test.npy"
)

Y_TEST_PATH = (
    MODEL_DIR / "y_tirads_test.npy"
)

HISTORY_PATH = (
    RESULT_DIR / "training_history.json"
)

PHASE9_METRICS_PATH = (
    PROJECT_ROOT
    / "07_Results"
    / "phase9_paper_tirads_class_weighted"
    / "test_metrics.json"
)


# ---------------------------------------------------------------------
# Phase 10 output files
# ---------------------------------------------------------------------

METRICS_TEXT_PATH = RESULT_DIR / "test_metrics.txt"
METRICS_JSON_PATH = RESULT_DIR / "test_metrics.json"

CLASSIFICATION_REPORT_PATH = (
    RESULT_DIR / "test_classification_report.txt"
)

PREDICTIONS_PATH = RESULT_DIR / "test_predictions.csv"

CONFUSION_MATRIX_PATH = (
    RESULT_DIR / "test_confusion_matrix.png"
)

NORMALIZED_CONFUSION_MATRIX_PATH = (
    RESULT_DIR / "test_confusion_matrix_normalized.png"
)

ACCURACY_PLOT_PATH = (
    RESULT_DIR / "training_accuracy_plot.png"
)

LOSS_PLOT_PATH = (
    RESULT_DIR / "training_loss_plot.png"
)


# ---------------------------------------------------------------------
# Final comparison output files
# ---------------------------------------------------------------------

COMPARISON_CSV_PATH = (
    COMPARISON_DIR / "quantum_vs_classical_comparison.csv"
)

COMPARISON_JSON_PATH = (
    COMPARISON_DIR / "quantum_vs_classical_comparison.json"
)

COMPARISON_TEXT_PATH = (
    COMPARISON_DIR / "final_reproduction_conclusion.txt"
)


CLASS_NAMES = [
    "TI-RADS 2",
    "TI-RADS 3",
    "TI-RADS 4a",
    "TI-RADS 4b",
    "TI-RADS 4c",
    "TI-RADS 5",
]


# ---------------------------------------------------------------------
# Validation and loading
# ---------------------------------------------------------------------

def require_files(paths: list[Path]) -> None:
    """Ensure that all required files exist."""

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


def load_test_dataset() -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Load and normalize the untouched grayscale TI-RADS test dataset.
    """

    x_test = np.load(X_TEST_PATH).astype(np.float32)
    y_test = np.load(Y_TEST_PATH).astype(np.int32).ravel()

    if x_test.ndim != 4:
        raise ValueError(
            "Expected four-dimensional test features. "
            f"Received: {x_test.shape}"
        )

    if tuple(x_test.shape[1:]) != (128, 128, 1):
        raise ValueError(
            "Expected test image shape (128, 128, 1). "
            f"Received: {x_test.shape[1:]}"
        )

    if len(x_test) != len(y_test):
        raise ValueError(
            "Test image-label mismatch: "
            f"{len(x_test)} images and {len(y_test)} labels."
        )

    if np.isnan(x_test).any():
        raise ValueError("NaN values found in test images.")

    if np.isinf(x_test).any():
        raise ValueError("Infinite values found in test images.")

    normalization_applied = False

    if np.max(x_test) > 1.5:
        x_test = x_test / 255.0
        normalization_applied = True

    unexpected_labels = np.setdiff1d(
        np.unique(y_test),
        np.arange(6),
    )

    if len(unexpected_labels) > 0:
        raise ValueError(
            f"Unexpected test labels: {unexpected_labels}"
        )

    return x_test, y_test, normalization_applied


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_confusion_matrix(
    matrix: np.ndarray,
    output_path: Path,
    title: str,
    normalized: bool,
) -> None:
    """Create raw or normalized confusion matrix."""

    display_matrix = matrix.astype(np.float64)

    if normalized:
        row_totals = display_matrix.sum(
            axis=1,
            keepdims=True,
        )

        display_matrix = np.divide(
            display_matrix,
            row_totals,
            out=np.zeros_like(display_matrix),
            where=row_totals != 0,
        )

    figure, axis = plt.subplots(figsize=(9, 7))

    image = axis.imshow(display_matrix)
    figure.colorbar(image, ax=axis)

    axis.set_xticks(np.arange(6))
    axis.set_yticks(np.arange(6))

    axis.set_xticklabels(
        CLASS_NAMES,
        rotation=45,
        ha="right",
    )

    axis.set_yticklabels(CLASS_NAMES)

    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(title)

    threshold = (
        display_matrix.max() / 2.0
        if display_matrix.size
        else 0.0
    )

    for row in range(display_matrix.shape[0]):
        for column in range(display_matrix.shape[1]):
            value = display_matrix[row, column]

            text = (
                f"{value:.2f}"
                if normalized
                else str(int(value))
            )

            axis.text(
                column,
                row,
                text,
                ha="center",
                va="center",
                color=(
                    "white"
                    if value > threshold
                    else "black"
                ),
            )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def plot_training_history() -> None:
    """Generate accuracy and loss plots."""

    with HISTORY_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        history = json.load(file)

    epochs = np.arange(
        1,
        len(history["loss"]) + 1,
    )

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
    plt.title("Classical TI-RADS Training Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
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
    plt.title("Classical TI-RADS Training Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    figure.savefig(
        LOSS_PLOT_PATH,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ---------------------------------------------------------------------
# Saving predictions
# ---------------------------------------------------------------------

def save_predictions(
    y_true: np.ndarray,
    y_predicted: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    """Save sample-level predictions."""

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
                "probability_tirads_2",
                "probability_tirads_3",
                "probability_tirads_4a",
                "probability_tirads_4b",
                "probability_tirads_4c",
                "probability_tirads_5",
            ]
        )

        for index in range(len(y_true)):
            true_class = int(y_true[index])
            predicted_class = int(y_predicted[index])

            writer.writerow(
                [
                    index,
                    true_class,
                    CLASS_NAMES[true_class],
                    predicted_class,
                    CLASS_NAMES[predicted_class],
                    float(
                        probabilities[
                            index,
                            predicted_class,
                        ]
                    ),
                    bool(true_class == predicted_class),
                    *[
                        float(value)
                        for value in probabilities[index]
                    ],
                ]
            )


# ---------------------------------------------------------------------
# Final comparison
# ---------------------------------------------------------------------

def create_final_comparison(
    classical_metrics: dict,
) -> None:
    """
    Compare the Phase 9 quantum model with the Phase 10 classical model.
    """

    quantum_metrics = {}

    if PHASE9_METRICS_PATH.exists():
        with PHASE9_METRICS_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            quantum_metrics = json.load(file)

    quantum_accuracy = float(
        quantum_metrics.get("accuracy", 0.0)
    )

    quantum_balanced_accuracy = float(
        quantum_metrics.get(
            "balanced_accuracy",
            0.0,
        )
    )

    quantum_macro_f1 = float(
        quantum_metrics.get("macro_f1", 0.0)
    )

    quantum_weighted_f1 = float(
        quantum_metrics.get("weighted_f1", 0.0)
    )

    classical_accuracy = float(
        classical_metrics["accuracy"]
    )

    classical_balanced_accuracy = float(
        classical_metrics["balanced_accuracy"]
    )

    classical_macro_f1 = float(
        classical_metrics["macro_f1"]
    )

    classical_weighted_f1 = float(
        classical_metrics["weighted_f1"]
    )

    comparison = {
        "quantum_model": {
            "experiment": (
                "Phase 9 paper-structured "
                "quantum TI-RADS model"
            ),
            "accuracy": quantum_accuracy,
            "balanced_accuracy": (
                quantum_balanced_accuracy
            ),
            "macro_f1": quantum_macro_f1,
            "weighted_f1": quantum_weighted_f1,
        },
        "classical_model": {
            "experiment": (
                "Phase 10 controlled "
                "classical TI-RADS model"
            ),
            "accuracy": classical_accuracy,
            "balanced_accuracy": (
                classical_balanced_accuracy
            ),
            "macro_f1": classical_macro_f1,
            "weighted_f1": classical_weighted_f1,
        },
        "difference_classical_minus_quantum": {
            "accuracy": (
                classical_accuracy
                - quantum_accuracy
            ),
            "balanced_accuracy": (
                classical_balanced_accuracy
                - quantum_balanced_accuracy
            ),
            "macro_f1": (
                classical_macro_f1
                - quantum_macro_f1
            ),
            "weighted_f1": (
                classical_weighted_f1
                - quantum_weighted_f1
            ),
        },
    }

    with COMPARISON_JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            comparison,
            file,
            indent=4,
        )

    with COMPARISON_CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "experiment",
                "accuracy",
                "balanced_accuracy",
                "macro_f1",
                "weighted_f1",
            ]
        )

        writer.writerow(
            [
                "Phase 9 Quantum TI-RADS",
                quantum_accuracy,
                quantum_balanced_accuracy,
                quantum_macro_f1,
                quantum_weighted_f1,
            ]
        )

        writer.writerow(
            [
                "Phase 10 Classical TI-RADS",
                classical_accuracy,
                classical_balanced_accuracy,
                classical_macro_f1,
                classical_weighted_f1,
            ]
        )

    accuracy_difference = (
        classical_accuracy - quantum_accuracy
    )

    if accuracy_difference > 0:
        comparison_statement = (
            "The classical model achieved higher test accuracy "
            "than the quantum-feature model."
        )
    elif accuracy_difference < 0:
        comparison_statement = (
            "The quantum-feature model achieved higher test accuracy "
            "than the classical model."
        )
    else:
        comparison_statement = (
            "Both models achieved identical test accuracy."
        )

    conclusion = f"""
FINAL QUCNET REPRODUCTION CONCLUSION
====================================

Phase 9 Quantum TI-RADS Model
-----------------------------
Accuracy          : {quantum_accuracy * 100:.2f}%
Balanced accuracy : {quantum_balanced_accuracy * 100:.2f}%
Macro F1-score    : {quantum_macro_f1 * 100:.2f}%
Weighted F1-score : {quantum_weighted_f1 * 100:.2f}%

Phase 10 Classical TI-RADS Model
--------------------------------
Accuracy          : {classical_accuracy * 100:.2f}%
Balanced accuracy : {classical_balanced_accuracy * 100:.2f}%
Macro F1-score    : {classical_macro_f1 * 100:.2f}%
Weighted F1-score : {classical_weighted_f1 * 100:.2f}%

Accuracy Difference
-------------------
Classical minus quantum:
{accuracy_difference * 100:.2f} percentage points

Interpretation
--------------
{comparison_statement}

The available QuCNet methodology and processing pipeline were
implemented successfully. However, the numerical performance reported
in the original paper could not be reproduced using the publicly
available DDTI data and the implementation details disclosed in the
paper.

The main reproducibility limitations include:

1. Missing original pathology labels for binary classification.
2. Incomplete architecture and quantum-circuit specifications.
3. Small number of publicly labelled TI-RADS samples.
4. Severe similarity between adjacent TI-RADS categories.
5. Weak generalization in both quantum and classical experiments.

The reproduction stage is therefore considered complete. Further work
must be presented separately as model improvement rather than as part
of the original paper reproduction.
""".strip()

    COMPARISON_TEXT_PATH.write_text(
        conclusion + "\n",
        encoding="utf-8",
    )

    print("\nFinal Quantum-versus-Classical Comparison")
    print("-" * 76)
    print(conclusion)


# ---------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------

def main() -> None:
    """Evaluate the final classical TI-RADS model."""

    print("=" * 76)
    print("PHASE 10 : FINAL CLASSICAL TI-RADS EVALUATION")
    print("=" * 76)

    require_files(
        [
            BEST_MODEL_PATH,
            X_TEST_PATH,
            Y_TEST_PATH,
            HISTORY_PATH,
        ]
    )

    print("\nLoading untouched test dataset...")

    (
        x_test,
        y_test,
        normalization_applied,
    ) = load_test_dataset()

    print(f"Test features             : {x_test.shape}")
    print(f"Test labels               : {y_test.shape}")
    print(
        "Division by 255 applied  : "
        f"{normalization_applied}"
    )

    true_distribution = np.bincount(
        y_test,
        minlength=6,
    )

    print("\nTest Class Distribution")
    print("-" * 76)

    for class_id, count in enumerate(true_distribution):
        print(
            f"Class {class_id} "
            f"({CLASS_NAMES[class_id]}): {count}"
        )

    majority_class = int(
        np.argmax(true_distribution)
    )

    majority_accuracy = float(
        true_distribution[majority_class]
        / len(y_test)
    )

    print("\nLoading best Phase 10 model...")

    model = tf.keras.models.load_model(
        BEST_MODEL_PATH
    )

    test_loss, keras_accuracy = model.evaluate(
        x_test,
        y_test,
        batch_size=16,
        verbose=1,
    )

    print("\nGenerating test predictions...")

    probabilities = model.predict(
        x_test,
        batch_size=16,
        verbose=1,
    )

    y_predicted = np.argmax(
        probabilities,
        axis=1,
    ).astype(np.int32)

    predicted_distribution = np.bincount(
        y_predicted,
        minlength=6,
    )

    accuracy = accuracy_score(
        y_test,
        y_predicted,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_test,
        y_predicted,
    )

    macro_precision = precision_score(
        y_test,
        y_predicted,
        labels=np.arange(6),
        average="macro",
        zero_division=0,
    )

    macro_recall = recall_score(
        y_test,
        y_predicted,
        labels=np.arange(6),
        average="macro",
        zero_division=0,
    )

    macro_f1 = f1_score(
        y_test,
        y_predicted,
        labels=np.arange(6),
        average="macro",
        zero_division=0,
    )

    weighted_precision = precision_score(
        y_test,
        y_predicted,
        labels=np.arange(6),
        average="weighted",
        zero_division=0,
    )

    weighted_recall = recall_score(
        y_test,
        y_predicted,
        labels=np.arange(6),
        average="weighted",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        y_predicted,
        labels=np.arange(6),
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        y_test,
        y_predicted,
        labels=np.arange(6),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        y_predicted,
        labels=np.arange(6),
    )

    predicted_classes_used = int(
        np.count_nonzero(predicted_distribution)
    )

    single_class_collapse = (
        predicted_classes_used == 1
    )

    metrics = {
        "test_samples": int(len(y_test)),
        "test_loss": float(test_loss),
        "keras_accuracy": float(keras_accuracy),
        "accuracy": float(accuracy),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "macro_precision": float(
            macro_precision
        ),
        "macro_recall": float(
            macro_recall
        ),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(
            weighted_precision
        ),
        "weighted_recall": float(
            weighted_recall
        ),
        "weighted_f1": float(
            weighted_f1
        ),
        "majority_class_id": majority_class,
        "majority_class_name": (
            CLASS_NAMES[majority_class]
        ),
        "majority_baseline_accuracy": (
            majority_accuracy
        ),
        "predicted_classes_used": (
            predicted_classes_used
        ),
        "single_class_collapse": bool(
            single_class_collapse
        ),
        "normalization_applied": bool(
            normalization_applied
        ),
        "true_class_distribution": (
            true_distribution.tolist()
        ),
        "predicted_class_distribution": (
            predicted_distribution.tolist()
        ),
        "model_path": str(BEST_MODEL_PATH),
    }

    with METRICS_JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    metrics_text = f"""
PHASE 10 CLASSICAL TI-RADS TEST EVALUATION
==========================================

Test samples               : {len(y_test)}
Test loss                  : {test_loss:.6f}
Accuracy                   : {accuracy * 100:.2f}%
Balanced accuracy          : {balanced_accuracy * 100:.2f}%

Macro precision            : {macro_precision * 100:.2f}%
Macro recall               : {macro_recall * 100:.2f}%
Macro F1-score             : {macro_f1 * 100:.2f}%

Weighted precision         : {weighted_precision * 100:.2f}%
Weighted recall            : {weighted_recall * 100:.2f}%
Weighted F1-score          : {weighted_f1 * 100:.2f}%

Majority class             : {CLASS_NAMES[majority_class]}
Majority baseline accuracy : {majority_accuracy * 100:.2f}%

Predicted classes used     : {predicted_classes_used}/6
Single-class collapse      : {single_class_collapse}
""".strip()

    METRICS_TEXT_PATH.write_text(
        metrics_text + "\n",
        encoding="utf-8",
    )

    CLASSIFICATION_REPORT_PATH.write_text(
        report + "\n",
        encoding="utf-8",
    )

    save_predictions(
        y_true=y_test,
        y_predicted=y_predicted,
        probabilities=probabilities,
    )

    plot_confusion_matrix(
        matrix=matrix,
        output_path=CONFUSION_MATRIX_PATH,
        title=(
            "Classical TI-RADS "
            "Confusion Matrix"
        ),
        normalized=False,
    )

    plot_confusion_matrix(
        matrix=matrix,
        output_path=(
            NORMALIZED_CONFUSION_MATRIX_PATH
        ),
        title=(
            "Normalized Classical TI-RADS "
            "Confusion Matrix"
        ),
        normalized=True,
    )

    plot_training_history()

    print("\n" + metrics_text)

    print("\nPredicted Class Distribution")
    print("-" * 76)

    for class_id, count in enumerate(
        predicted_distribution
    ):
        print(
            f"Class {class_id} "
            f"({CLASS_NAMES[class_id]}): {count}"
        )

    print("\nClassification Report")
    print("-" * 76)
    print(report)

    print("Confusion Matrix")
    print("-" * 76)
    print(matrix)

    create_final_comparison(metrics)

    print("\nGenerated Phase 10 Files")
    print("-" * 76)
    print(METRICS_TEXT_PATH)
    print(METRICS_JSON_PATH)
    print(CLASSIFICATION_REPORT_PATH)
    print(PREDICTIONS_PATH)
    print(CONFUSION_MATRIX_PATH)
    print(NORMALIZED_CONFUSION_MATRIX_PATH)
    print(ACCURACY_PLOT_PATH)
    print(LOSS_PLOT_PATH)

    print("\nGenerated Final Comparison Files")
    print("-" * 76)
    print(COMPARISON_CSV_PATH)
    print(COMPARISON_JSON_PATH)
    print(COMPARISON_TEXT_PATH)

    print("\n" + "=" * 76)
    print("REPRODUCTION EXPERIMENTS COMPLETE")
    print("=" * 76)


if __name__ == "__main__":
    main()