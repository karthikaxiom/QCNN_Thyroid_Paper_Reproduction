from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from tensorflow import keras


SOURCE_DIR = Path(__file__).resolve().parent
ROOT = SOURCE_DIR.parent

MODEL_PATH = ROOT / "06_Models" / "best_qucnet.keras"
X_TEST_PATH = ROOT / "06_Models" / "X_test_quantum.npy"
Y_TEST_PATH = ROOT / "06_Models" / "y_test.npy"

OUT = ROOT / "07_Results" / "phase16_binary_qucnet_evaluation"
OUT.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["Benign", "Malignant"]


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def main() -> None:
    print("=" * 78)
    print("PHASE 16 : BINARY QUCNET FINAL EVALUATION")
    print("=" * 78)

    for path in [MODEL_PATH, X_TEST_PATH, Y_TEST_PATH]:
        require(path)

    print("[1/6] Loading model and untouched binary test set...")
    model = keras.models.load_model(MODEL_PATH)
    X_test = np.load(X_TEST_PATH).astype(np.float32)
    y_true = np.load(Y_TEST_PATH).astype(int)

    if X_test.shape[0] != y_true.shape[0]:
        raise ValueError("X_test and y_test counts do not match.")

    print(f"Test samples : {len(y_true)}")
    print(f"Input shape  : {X_test.shape}")

    print("[2/6] Generating test probabilities...")
    probabilities = model.predict(X_test, verbose=0)

    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError(
            f"Expected two-class softmax output, got {probabilities.shape}"
        )

    malignant_probability = probabilities[:, 1]
    y_pred = np.argmax(probabilities, axis=1).astype(int)

    print("[3/6] Calculating verified binary metrics...")
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, malignant_probability)
    average_precision = average_precision_score(
        y_true,
        malignant_probability,
    )

    metrics = {
        "task": "binary benign versus malignant classification",
        "test_samples": int(len(y_true)),
        "benign_test_samples": int(np.sum(y_true == 0)),
        "malignant_test_samples": int(np.sum(y_true == 1)),
        "accuracy": float(accuracy),
        "precision_positive_malignant": float(precision),
        "sensitivity_recall_malignant": float(sensitivity),
        "specificity_benign": float(specificity),
        "negative_predictive_value": float(npv),
        "f1_score_malignant": float(f1),
        "matthews_correlation_coefficient": float(mcc),
        "cohen_kappa": float(kappa),
        "roc_auc": float(roc_auc),
        "average_precision": float(average_precision),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }

    (OUT / "binary_evaluation_metrics.json").write_text(
        json.dumps(metrics, indent=4) + "\n",
        encoding="utf-8",
    )

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        OUT / "binary_classification_report.csv"
    )

    predictions = pd.DataFrame({
        "sample_index": np.arange(len(y_true)),
        "true_class": y_true,
        "true_label": [CLASS_NAMES[index] for index in y_true],
        "predicted_class": y_pred,
        "predicted_label": [CLASS_NAMES[index] for index in y_pred],
        "probability_benign": probabilities[:, 0],
        "probability_malignant": probabilities[:, 1],
        "confidence": probabilities.max(axis=1),
        "correct": y_true == y_pred,
        "error_type": np.where(
            (y_true == 0) & (y_pred == 1),
            "False Positive",
            np.where(
                (y_true == 1) & (y_pred == 0),
                "False Negative",
                "Correct",
            ),
        ),
    })
    predictions.to_csv(
        OUT / "binary_test_predictions.csv",
        index=False,
    )

    print("[4/6] Creating ROC and precision-recall curves...")
    fpr, tpr, _ = roc_curve(y_true, malignant_probability)

    plt.figure(figsize=(7.5, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"QuCNet (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Binary QuCNet ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.25)
    save_plot(OUT / "binary_roc_curve.png")

    pr_precision, pr_recall, _ = precision_recall_curve(
        y_true,
        malignant_probability,
    )
    prevalence = float(np.mean(y_true))

    plt.figure(figsize=(7.5, 6))
    plt.plot(
        pr_recall,
        pr_precision,
        linewidth=2,
        label=f"QuCNet (AP = {average_precision:.3f})",
    )
    plt.axhline(
        prevalence,
        linestyle="--",
        label=f"Malignant prevalence = {prevalence:.3f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Binary QuCNet Precision–Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.25)
    save_plot(OUT / "binary_precision_recall_curve.png")

    print("[5/6] Creating confusion-matrix figures...")
    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual Benign", "Actual Malignant"],
        columns=["Predicted Benign", "Predicted Malignant"],
    )
    matrix_df.to_csv(OUT / "binary_confusion_matrix.csv")

    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(matrix, cmap="Blues")
    plt.colorbar()
    plt.xticks([0, 1], CLASS_NAMES)
    plt.yticks([0, 1], CLASS_NAMES)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Binary QuCNet Confusion Matrix")

    threshold = matrix.max() / 2
    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
                fontsize=13,
            )
    save_plot(OUT / "binary_confusion_matrix.png")

    normalized = matrix.astype(float)
    normalized = normalized / normalized.sum(axis=1, keepdims=True)

    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar()
    plt.xticks([0, 1], CLASS_NAMES)
    plt.yticks([0, 1], CLASS_NAMES)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("Normalized Binary QuCNet Confusion Matrix")

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                f"{normalized[i, j]:.2%}",
                ha="center",
                va="center",
                color="white" if normalized[i, j] > 0.5 else "black",
                fontsize=12,
            )
    save_plot(OUT / "binary_confusion_matrix_normalized.png")

    print("[6/6] Writing concise scientific summary...")
    lines = [
        "PHASE 16 - BINARY QUCNET FINAL EVALUATION",
        "=" * 78,
        "",
        f"Test samples                         : {len(y_true)}",
        f"Benign / Malignant                  : {np.sum(y_true == 0)} / {np.sum(y_true == 1)}",
        f"Accuracy                            : {accuracy:.4f} ({accuracy * 100:.2f}%)",
        f"Precision (Malignant)               : {precision:.4f} ({precision * 100:.2f}%)",
        f"Sensitivity / Recall (Malignant)    : {sensitivity:.4f} ({sensitivity * 100:.2f}%)",
        f"Specificity (Benign)                : {specificity:.4f} ({specificity * 100:.2f}%)",
        f"Negative predictive value           : {npv:.4f} ({npv * 100:.2f}%)",
        f"F1 score (Malignant)                : {f1:.4f} ({f1 * 100:.2f}%)",
        f"Matthews correlation coefficient    : {mcc:.4f}",
        f"Cohen kappa                         : {kappa:.4f}",
        f"ROC-AUC                             : {roc_auc:.4f}",
        f"Average precision                   : {average_precision:.4f}",
        "",
        "Confusion matrix",
        "-" * 78,
        f"True negatives                      : {tn}",
        f"False positives                     : {fp}",
        f"False negatives                     : {fn}",
        f"True positives                      : {tp}",
        "",
        "Interpretation",
        "-" * 78,
        (
            "The model prioritizes malignant-case detection, producing high "
            "sensitivity but low specificity. The ROC-AUC and average precision "
            "must be reported together with the threshold-dependent confusion "
            "matrix metrics."
        ),
    ]

    (OUT / "phase16_binary_summary.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\nPhase 16 completed successfully.")
    print(f"Outputs saved in: {OUT}")


if __name__ == "__main__":
    main()