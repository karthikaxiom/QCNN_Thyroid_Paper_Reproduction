from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SOURCE_DIR = Path(__file__).resolve().parent
ROOT = SOURCE_DIR.parent

X_TEST_PATH = ROOT / "06_Models" / "X_test.npy"
PREDICTIONS_PATH = (
    ROOT
    / "07_Results"
    / "phase16_binary_qucnet_evaluation"
    / "binary_test_predictions.csv"
)

OUT = ROOT / "07_Results" / "phase18_binary_error_analysis"
OUT.mkdir(parents=True, exist_ok=True)


def require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def prepare_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)

    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[..., 0]

    if image.ndim != 2:
        raise ValueError(f"Expected a 2D grayscale image, got {image.shape}")

    return image


def save_gallery(
    images: np.ndarray,
    rows_df: pd.DataFrame,
    output_path: Path,
    title: str,
    max_items: int = 8,
) -> None:
    selected = rows_df.head(max_items).copy()

    if selected.empty:
        return

    columns = 4
    rows = int(np.ceil(len(selected) / columns))

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(14, 3.5 * rows),
    )

    axes = np.atleast_1d(axes).ravel()

    for axis, (_, row) in zip(axes, selected.iterrows()):
        sample_index = int(row["sample_index"])
        image = prepare_image(images[sample_index])

        axis.imshow(image, cmap="gray")
        axis.axis("off")

        axis.set_title(
            (
                f"Sample {sample_index}\n"
                f"True: {row['true_label']} | "
                f"Pred: {row['predicted_label']}\n"
                f"P(high-suspicion proxy): "
                f"{float(row['probability_malignant']):.3f}"
            ),
            fontsize=9,
        )

    for axis in axes[len(selected):]:
        axis.axis("off")

    fig.suptitle(title, fontsize=15)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_combined_gallery(
    images: np.ndarray,
    false_positives: pd.DataFrame,
    false_negatives: pd.DataFrame,
) -> None:
    fp = false_positives.head(6)
    fn = false_negatives.head(6)

    combined = pd.concat(
        [
            fp.assign(group="False Positive"),
            fn.assign(group="False Negative"),
        ],
        ignore_index=True,
    )

    if combined.empty:
        return

    columns = 4
    rows = int(np.ceil(len(combined) / columns))

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(14, 3.5 * rows),
    )

    axes = np.atleast_1d(axes).ravel()

    for axis, (_, row) in zip(axes, combined.iterrows()):
        sample_index = int(row["sample_index"])
        image = prepare_image(images[sample_index])

        axis.imshow(image, cmap="gray")
        axis.axis("off")

        axis.set_title(
            (
                f"{row['group']}\n"
                f"Sample {sample_index}\n"
                f"True: {row['true_label']} | "
                f"Pred: {row['predicted_label']}\n"
                f"P(high-suspicion proxy): "
                f"{float(row['probability_malignant']):.3f}"
            ),
            fontsize=8.5,
        )

    for axis in axes[len(combined):]:
        axis.axis("off")

    fig.suptitle(
        "Binary QuCNet Error Analysis",
        fontsize=15,
    )

    plt.tight_layout()
    plt.savefig(
        OUT / "binary_error_analysis_gallery.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_error_summary(
    false_positives: pd.DataFrame,
    false_negatives: pd.DataFrame,
) -> None:
    rows = [
        {
            "error_type": "False Positive",
            "count": int(len(false_positives)),
            "meaning": "TI-RADS-derived low-suspicion proxy image predicted as high-suspicion proxy",
            "clinical_effect": (
                "No direct clinical consequence can be inferred because these are TI-RADS-derived proxy labels, not pathology-confirmed diagnoses."
            ),
        },
        {
            "error_type": "False Negative",
            "count": int(len(false_negatives)),
            "meaning": "TI-RADS-derived high-suspicion proxy image predicted as low-suspicion proxy",
            "clinical_effect": (
                "No direct clinical consequence can be inferred because these are TI-RADS-derived proxy labels, not pathology-confirmed diagnoses."
            ),
        },
    ]

    pd.DataFrame(rows).to_csv(
        OUT / "binary_error_summary.csv",
        index=False,
    )


def main() -> None:
    print("=" * 78)
    print("PHASE 18 : BINARY ERROR ANALYSIS")
    print("=" * 78)

    require(X_TEST_PATH)
    require(PREDICTIONS_PATH)

    print("[1/4] Loading original test images and predictions...")
    images = np.load(X_TEST_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)

    if len(images) != len(predictions):
        raise ValueError(
            "The number of test images and prediction rows does not match."
        )

    print("[2/4] Separating false positives and false negatives...")
    false_positives = predictions[
        predictions["error_type"] == "False Positive"
    ].copy()

    false_negatives = predictions[
        predictions["error_type"] == "False Negative"
    ].copy()

    false_positives = false_positives.sort_values(
        "probability_malignant",
        ascending=False,
    )

    false_negatives = false_negatives.sort_values(
        "probability_benign",
        ascending=False,
    )

    false_positives.to_csv(
        OUT / "false_positive_cases.csv",
        index=False,
    )

    false_negatives.to_csv(
        OUT / "false_negative_cases.csv",
        index=False,
    )

    print("[3/4] Creating error-analysis galleries...")
    save_gallery(
        images=images,
        rows_df=false_positives,
        output_path=OUT / "false_positive_gallery.png",
        title="False Positives: Low-Suspicion Proxy Predicted as High-Suspicion Proxy",
    )

    save_gallery(
        images=images,
        rows_df=false_negatives,
        output_path=OUT / "false_negative_gallery.png",
        title="False Negatives: High-Suspicion Proxy Predicted as Low-Suspicion Proxy",
    )

    save_combined_gallery(
        images=images,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )

    print("[4/4] Writing error summary...")
    save_error_summary(
        false_positives=false_positives,
        false_negatives=false_negatives,
    )

    print("\nPhase 18 completed successfully.")
    print(f"False positives : {len(false_positives)}")
    print(f"False negatives : {len(false_negatives)}")
    print(f"Outputs saved in: {OUT}")


if __name__ == "__main__":
    main()