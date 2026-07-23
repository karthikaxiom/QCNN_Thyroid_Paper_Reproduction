from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =====================================================================
# Project paths
# =====================================================================

SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIR.parent

PHASE12_DIR = (
    PROJECT_ROOT
    / "07_Results"
    / "phase12_case_level_split"
)

TRAIN_MANIFEST_PATH = (
    PHASE12_DIR
    / "train_roi_manifest.csv"
)

VALIDATION_MANIFEST_PATH = (
    PHASE12_DIR
    / "validation_roi_manifest.csv"
)

TEST_MANIFEST_PATH = (
    PHASE12_DIR
    / "test_roi_manifest.csv"
)


# Processed dataset output directory
PROCESSED_DIR = (
    PROJECT_ROOT
    / "03_Dataset_Processed"
    / "phase13_roi_dataset"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# Results and audit output directory
RESULT_DIR = (
    PROJECT_ROOT
    / "07_Results"
    / "phase13_roi_dataset"
)

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =====================================================================
# Output files
# =====================================================================

TRAIN_X_PATH = PROCESSED_DIR / "X_roi_train.npy"
TRAIN_Y_PATH = PROCESSED_DIR / "y_roi_train.npy"
TRAIN_CASE_IDS_PATH = PROCESSED_DIR / "case_ids_roi_train.npy"
TRAIN_ROI_IDS_PATH = PROCESSED_DIR / "roi_ids_train.npy"
TRAIN_METADATA_PATH = PROCESSED_DIR / "train_roi_metadata.csv"

VALIDATION_X_PATH = PROCESSED_DIR / "X_roi_validation.npy"
VALIDATION_Y_PATH = PROCESSED_DIR / "y_roi_validation.npy"
VALIDATION_CASE_IDS_PATH = (
    PROCESSED_DIR
    / "case_ids_roi_validation.npy"
)
VALIDATION_ROI_IDS_PATH = (
    PROCESSED_DIR
    / "roi_ids_validation.npy"
)
VALIDATION_METADATA_PATH = (
    PROCESSED_DIR
    / "validation_roi_metadata.csv"
)

TEST_X_PATH = PROCESSED_DIR / "X_roi_test.npy"
TEST_Y_PATH = PROCESSED_DIR / "y_roi_test.npy"
TEST_CASE_IDS_PATH = PROCESSED_DIR / "case_ids_roi_test.npy"
TEST_ROI_IDS_PATH = PROCESSED_DIR / "roi_ids_test.npy"
TEST_METADATA_PATH = PROCESSED_DIR / "test_roi_metadata.csv"

DATASET_SUMMARY_PATH = (
    RESULT_DIR
    / "roi_dataset_summary.txt"
)

DATASET_DISTRIBUTION_PATH = (
    RESULT_DIR
    / "roi_dataset_distribution.json"
)

DATASET_AUDIT_PATH = (
    RESULT_DIR
    / "roi_dataset_audit.csv"
)

FAILED_RECORDS_PATH = (
    RESULT_DIR
    / "failed_roi_records.csv"
)

TRAIN_PREVIEW_PATH = (
    RESULT_DIR
    / "train_roi_preview.png"
)

VALIDATION_PREVIEW_PATH = (
    RESULT_DIR
    / "validation_roi_preview.png"
)

TEST_PREVIEW_PATH = (
    RESULT_DIR
    / "test_roi_preview.png"
)


# =====================================================================
# Configuration
# =====================================================================

IMAGE_SIZE = 64

# Margin added around each lesion bounding box.
# 0.10 means 10% of the bounding-box width and height.
ROI_MARGIN_FRACTION = 0.10

EXPECTED_SPLITS = [
    "train",
    "validation",
    "test",
]

EXPECTED_TIRADS_ORDER = [
    "2",
    "3",
    "4a",
    "4b",
    "4c",
    "5",
]

TIRADS_TO_CLASS = {
    "2": 0,
    "3": 1,
    "4a": 2,
    "4b": 3,
    "4c": 4,
    "5": 5,
}

PREVIEW_IMAGES_PER_SPLIT = 24
RANDOM_SEED = 42


# =====================================================================
# Utility functions
# =====================================================================

def normalize_case_id(value: Any) -> str:
    """
    Convert case IDs to stable string values.
    """

    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    return text


def normalize_tirads(value: Any) -> str:
    """
    Normalize TI-RADS labels.
    """

    if pd.isna(value):
        return ""

    return str(value).strip().lower()


def require_file(path: Path) -> None:
    """
    Raise a clear error when a required file is missing.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Required file was not found:\n{path}"
        )


def validate_required_files() -> None:
    """
    Validate all Phase 12 manifest files.
    """

    require_file(TRAIN_MANIFEST_PATH)
    require_file(VALIDATION_MANIFEST_PATH)
    require_file(TEST_MANIFEST_PATH)


def load_split_manifest(
    path: Path,
    expected_split: str,
) -> pd.DataFrame:
    """
    Load and validate one frozen Phase 12 ROI manifest.
    """

    dataframe = pd.read_csv(path)

    required_columns = {
        "roi_id",
        "case_id",
        "image_name",
        "image_path",
        "image_width",
        "image_height",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "bbox_width",
        "bbox_height",
        "tirads",
        "tirads_class",
        "roi_valid",
        "split",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{path.name} is missing required columns:\n"
            + ", ".join(sorted(missing_columns))
        )

    dataframe["case_id"] = (
        dataframe["case_id"]
        .map(normalize_case_id)
    )

    dataframe["roi_id"] = (
        dataframe["roi_id"]
        .astype(str)
        .str.strip()
    )

    dataframe["tirads"] = (
        dataframe["tirads"]
        .map(normalize_tirads)
    )

    dataframe["split"] = (
        dataframe["split"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    invalid_split_rows = dataframe[
        dataframe["split"] != expected_split
    ]

    if not invalid_split_rows.empty:
        raise ValueError(
            f"{path.name} contains rows not assigned to "
            f"the expected split '{expected_split}'."
        )

    invalid_tirads_rows = dataframe[
        ~dataframe["tirads"].isin(
            EXPECTED_TIRADS_ORDER
        )
    ]

    if not invalid_tirads_rows.empty:
        raise ValueError(
            f"{path.name} contains unsupported TI-RADS labels."
        )

    dataframe["tirads_class"] = (
        dataframe["tirads"]
        .map(TIRADS_TO_CLASS)
        .astype(int)
    )

    if dataframe["case_id"].eq("").any():
        raise ValueError(
            f"{path.name} contains empty case IDs."
        )

    if dataframe["roi_id"].eq("").any():
        raise ValueError(
            f"{path.name} contains empty ROI IDs."
        )

    if dataframe["roi_id"].duplicated().any():
        duplicates = (
            dataframe.loc[
                dataframe["roi_id"].duplicated(),
                "roi_id",
            ]
            .astype(str)
            .tolist()
        )

        raise ValueError(
            f"{path.name} contains duplicate ROI IDs:\n"
            + ", ".join(duplicates[:20])
        )

    return dataframe


def audit_frozen_splits(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """
    Verify zero case and ROI overlap before processing.
    """

    train_cases = set(
        train_df["case_id"].astype(str)
    )

    validation_cases = set(
        validation_df["case_id"].astype(str)
    )

    test_cases = set(
        test_df["case_id"].astype(str)
    )

    train_rois = set(
        train_df["roi_id"].astype(str)
    )

    validation_rois = set(
        validation_df["roi_id"].astype(str)
    )

    test_rois = set(
        test_df["roi_id"].astype(str)
    )

    case_overlaps = {
        "train_validation": (
            train_cases
            & validation_cases
        ),
        "train_test": (
            train_cases
            & test_cases
        ),
        "validation_test": (
            validation_cases
            & test_cases
        ),
    }

    roi_overlaps = {
        "train_validation": (
            train_rois
            & validation_rois
        ),
        "train_test": (
            train_rois
            & test_rois
        ),
        "validation_test": (
            validation_rois
            & test_rois
        ),
    }

    if any(case_overlaps.values()):
        raise RuntimeError(
            "Case leakage detected in frozen Phase 12 manifests."
        )

    if any(roi_overlaps.values()):
        raise RuntimeError(
            "ROI leakage detected in frozen Phase 12 manifests."
        )


def calculate_margin_bbox(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """
    Add a controlled margin around the original lesion bounding box.

    Coordinates are clipped to the image dimensions.
    """

    bbox_width = x2 - x1
    bbox_height = y2 - y1

    margin_x = int(
        round(
            bbox_width
            * ROI_MARGIN_FRACTION
        )
    )

    margin_y = int(
        round(
            bbox_height
            * ROI_MARGIN_FRACTION
        )
    )

    expanded_x1 = max(
        0,
        x1 - margin_x,
    )

    expanded_y1 = max(
        0,
        y1 - margin_y,
    )

    expanded_x2 = min(
        image_width,
        x2 + margin_x,
    )

    expanded_y2 = min(
        image_height,
        y2 + margin_y,
    )

    return (
        expanded_x1,
        expanded_y1,
        expanded_x2,
        expanded_y2,
    )


def pad_to_square(
    image: np.ndarray,
) -> np.ndarray:
    """
    Pad an image to a square shape without stretching it.

    Padding value is based on the median intensity of the ROI.
    """

    height, width = image.shape[:2]

    if height <= 0 or width <= 0:
        raise ValueError(
            "Cannot pad an empty ROI."
        )

    target_size = max(
        height,
        width,
    )

    vertical_padding = (
        target_size
        - height
    )

    horizontal_padding = (
        target_size
        - width
    )

    top = vertical_padding // 2
    bottom = (
        vertical_padding
        - top
    )

    left = horizontal_padding // 2
    right = (
        horizontal_padding
        - left
    )

    padding_value = int(
        np.median(image)
    )

    squared = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_CONSTANT,
        value=padding_value,
    )

    return squared


def process_single_roi(
    row: pd.Series,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Load, crop, square-pad, resize and normalize one ROI.
    """

    image_path = Path(
        str(row["image_path"])
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE,
    )

    if image is None:
        raise ValueError(
            f"OpenCV could not read image: {image_path}"
        )

    actual_height, actual_width = image.shape[:2]

    x1 = int(
        round(
            float(row["bbox_x1"])
        )
    )

    y1 = int(
        round(
            float(row["bbox_y1"])
        )
    )

    x2 = int(
        round(
            float(row["bbox_x2"])
        )
    )

    y2 = int(
        round(
            float(row["bbox_y2"])
        )
    )

    x1 = max(
        0,
        min(
            x1,
            actual_width,
        ),
    )

    y1 = max(
        0,
        min(
            y1,
            actual_height,
        ),
    )

    x2 = max(
        0,
        min(
            x2,
            actual_width,
        ),
    )

    y2 = max(
        0,
        min(
            y2,
            actual_height,
        ),
    )

    if x2 <= x1:
        raise ValueError(
            f"Invalid horizontal bounding box: "
            f"x1={x1}, x2={x2}"
        )

    if y2 <= y1:
        raise ValueError(
            f"Invalid vertical bounding box: "
            f"y1={y1}, y2={y2}"
        )

    (
        expanded_x1,
        expanded_y1,
        expanded_x2,
        expanded_y2,
    ) = calculate_margin_bbox(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        image_width=actual_width,
        image_height=actual_height,
    )

    roi = image[
        expanded_y1:expanded_y2,
        expanded_x1:expanded_x2,
    ]

    if roi.size == 0:
        raise ValueError(
            "Extracted ROI is empty."
        )

    original_roi_height, original_roi_width = (
        roi.shape[:2]
    )

    squared_roi = pad_to_square(
        roi
    )

    resized_roi = cv2.resize(
        squared_roi,
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        interpolation=cv2.INTER_AREA,
    )

    normalized_roi = (
        resized_roi.astype(
            np.float32
        )
        / 255.0
    )

    # Add one grayscale channel:
    # shape becomes (64, 64, 1)
    normalized_roi = np.expand_dims(
        normalized_roi,
        axis=-1,
    )

    metadata = {
        "roi_id": str(row["roi_id"]),
        "case_id": normalize_case_id(
            row["case_id"]
        ),
        "split": str(row["split"]),
        "tirads": normalize_tirads(
            row["tirads"]
        ),
        "tirads_class": int(
            row["tirads_class"]
        ),
        "image_name": str(
            row["image_name"]
        ),
        "image_path": str(
            image_path
        ),
        "actual_image_width": int(
            actual_width
        ),
        "actual_image_height": int(
            actual_height
        ),
        "original_bbox_x1": int(x1),
        "original_bbox_y1": int(y1),
        "original_bbox_x2": int(x2),
        "original_bbox_y2": int(y2),
        "expanded_bbox_x1": int(
            expanded_x1
        ),
        "expanded_bbox_y1": int(
            expanded_y1
        ),
        "expanded_bbox_x2": int(
            expanded_x2
        ),
        "expanded_bbox_y2": int(
            expanded_y2
        ),
        "extracted_roi_width": int(
            original_roi_width
        ),
        "extracted_roi_height": int(
            original_roi_height
        ),
        "output_width": IMAGE_SIZE,
        "output_height": IMAGE_SIZE,
        "output_channels": 1,
        "minimum_pixel_value": float(
            normalized_roi.min()
        ),
        "maximum_pixel_value": float(
            normalized_roi.max()
        ),
        "mean_pixel_value": float(
            normalized_roi.mean()
        ),
        "processing_success": True,
        "failure_reason": "",
    }

    return (
        normalized_roi,
        metadata,
    )


def process_split(
    dataframe: pd.DataFrame,
    split_name: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Process every ROI belonging to one split.
    """

    images: list[np.ndarray] = []
    labels: list[int] = []
    case_ids: list[str] = []
    roi_ids: list[str] = []

    success_records: list[
        dict[str, Any]
    ] = []

    failed_records: list[
        dict[str, Any]
    ] = []

    total_rows = len(dataframe)

    for position, (_, row) in enumerate(
        dataframe.iterrows(),
        start=1,
    ):
        try:
            (
                processed_roi,
                metadata,
            ) = process_single_roi(row)

            images.append(
                processed_roi
            )

            labels.append(
                int(
                    row["tirads_class"]
                )
            )

            case_ids.append(
                normalize_case_id(
                    row["case_id"]
                )
            )

            roi_ids.append(
                str(
                    row["roi_id"]
                )
            )

            success_records.append(
                metadata
            )

        except Exception as error:
            failed_records.append(
                {
                    "roi_id": str(
                        row.get(
                            "roi_id",
                            "",
                        )
                    ),
                    "case_id": normalize_case_id(
                        row.get(
                            "case_id",
                            "",
                        )
                    ),
                    "split": split_name,
                    "image_name": str(
                        row.get(
                            "image_name",
                            "",
                        )
                    ),
                    "image_path": str(
                        row.get(
                            "image_path",
                            "",
                        )
                    ),
                    "tirads": normalize_tirads(
                        row.get(
                            "tirads",
                            "",
                        )
                    ),
                    "tirads_class": row.get(
                        "tirads_class",
                        "",
                    ),
                    "processing_success": False,
                    "failure_reason": str(error),
                }
            )

        if (
            position % 25 == 0
            or position == total_rows
        ):
            print(
                f"{split_name.capitalize():10s} "
                f"{position:4d}/{total_rows:4d}"
            )

    if not images:
        raise RuntimeError(
            f"No ROI images were successfully processed "
            f"for split '{split_name}'."
        )

    X = np.stack(
        images,
        axis=0,
    ).astype(
        np.float32
    )

    y = np.asarray(
        labels,
        dtype=np.int64,
    )

    case_id_array = np.asarray(
        case_ids,
        dtype=str,
    )

    roi_id_array = np.asarray(
        roi_ids,
        dtype=str,
    )

    success_df = pd.DataFrame(
        success_records
    )

    failed_df = pd.DataFrame(
        failed_records
    )

    return (
        X,
        y,
        case_id_array,
        roi_id_array,
        success_df,
        failed_df,
    )


def save_numpy_outputs(
    split_name: str,
    X: np.ndarray,
    y: np.ndarray,
    case_ids: np.ndarray,
    roi_ids: np.ndarray,
    metadata_df: pd.DataFrame,
) -> None:
    """
    Save arrays and metadata for one split.
    """

    if split_name == "train":
        np.save(
            TRAIN_X_PATH,
            X,
        )

        np.save(
            TRAIN_Y_PATH,
            y,
        )

        np.save(
            TRAIN_CASE_IDS_PATH,
            case_ids,
        )

        np.save(
            TRAIN_ROI_IDS_PATH,
            roi_ids,
        )

        metadata_df.to_csv(
            TRAIN_METADATA_PATH,
            index=False,
        )

    elif split_name == "validation":
        np.save(
            VALIDATION_X_PATH,
            X,
        )

        np.save(
            VALIDATION_Y_PATH,
            y,
        )

        np.save(
            VALIDATION_CASE_IDS_PATH,
            case_ids,
        )

        np.save(
            VALIDATION_ROI_IDS_PATH,
            roi_ids,
        )

        metadata_df.to_csv(
            VALIDATION_METADATA_PATH,
            index=False,
        )

    elif split_name == "test":
        np.save(
            TEST_X_PATH,
            X,
        )

        np.save(
            TEST_Y_PATH,
            y,
        )

        np.save(
            TEST_CASE_IDS_PATH,
            case_ids,
        )

        np.save(
            TEST_ROI_IDS_PATH,
            roi_ids,
        )

        metadata_df.to_csv(
            TEST_METADATA_PATH,
            index=False,
        )

    else:
        raise ValueError(
            f"Unsupported split name: {split_name}"
        )


def create_preview(
    X: np.ndarray,
    y: np.ndarray,
    case_ids: np.ndarray,
    roi_ids: np.ndarray,
    split_name: str,
    output_path: Path,
) -> None:
    """
    Create a deterministic ROI preview grid.
    """

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    selected_indices: list[int] = []

    # Select examples class by class to improve preview coverage.
    for class_index in range(
        len(EXPECTED_TIRADS_ORDER)
    ):
        class_indices = np.where(
            y == class_index
        )[0]

        if len(class_indices) == 0:
            continue

        samples_per_class = max(
            1,
            PREVIEW_IMAGES_PER_SPLIT
            // len(EXPECTED_TIRADS_ORDER),
        )

        selected = rng.choice(
            class_indices,
            size=min(
                samples_per_class,
                len(class_indices),
            ),
            replace=False,
        )

        selected_indices.extend(
            selected.tolist()
        )

    if len(selected_indices) < PREVIEW_IMAGES_PER_SPLIT:
        remaining_indices = [
            index
            for index in range(len(X))
            if index not in selected_indices
        ]

        additional_count = min(
            PREVIEW_IMAGES_PER_SPLIT
            - len(selected_indices),
            len(remaining_indices),
        )

        if additional_count > 0:
            selected_indices.extend(
                rng.choice(
                    remaining_indices,
                    size=additional_count,
                    replace=False,
                ).tolist()
            )

    selected_indices = selected_indices[
        :PREVIEW_IMAGES_PER_SPLIT
    ]

    columns = 6
    rows = int(
        np.ceil(
            len(selected_indices)
            / columns
        )
    )

    figure = plt.figure(
        figsize=(
            15,
            3.1 * rows,
        )
    )

    for plot_number, array_index in enumerate(
        selected_indices,
        start=1,
    ):
        axis = figure.add_subplot(
            rows,
            columns,
            plot_number,
        )

        axis.imshow(
            X[array_index, :, :, 0],
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )

        label_name = EXPECTED_TIRADS_ORDER[
            int(
                y[array_index]
            )
        ]

        axis.set_title(
            (
                f"TI-RADS {label_name}\n"
                f"Case {case_ids[array_index]}\n"
                f"ROI {roi_ids[array_index]}"
            ),
            fontsize=8,
        )

        axis.axis("off")

    figure.suptitle(
        (
            f"Phase 13 ROI Preview - "
            f"{split_name.capitalize()}"
        ),
        fontsize=15,
    )

    figure.tight_layout(
        rect=[
            0,
            0,
            1,
            0.96,
        ]
    )

    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def class_distribution(
    y: np.ndarray,
) -> dict[str, int]:
    """
    Return TI-RADS class counts.
    """

    return {
        label: int(
            np.sum(
                y
                == class_index
            )
        )
        for class_index, label in enumerate(
            EXPECTED_TIRADS_ORDER
        )
    }


def verify_processed_outputs(
    split_results: dict[
        str,
        dict[str, Any],
    ],
) -> dict[str, Any]:
    """
    Verify shapes, labels, pixel ranges and leakage after processing.
    """

    train_cases = set(
        split_results["train"]["case_ids"]
        .astype(str)
    )

    validation_cases = set(
        split_results["validation"]["case_ids"]
        .astype(str)
    )

    test_cases = set(
        split_results["test"]["case_ids"]
        .astype(str)
    )

    train_rois = set(
        split_results["train"]["roi_ids"]
        .astype(str)
    )

    validation_rois = set(
        split_results["validation"]["roi_ids"]
        .astype(str)
    )

    test_rois = set(
        split_results["test"]["roi_ids"]
        .astype(str)
    )

    case_overlap_counts = {
        "train_validation": len(
            train_cases
            & validation_cases
        ),
        "train_test": len(
            train_cases
            & test_cases
        ),
        "validation_test": len(
            validation_cases
            & test_cases
        ),
    }

    roi_overlap_counts = {
        "train_validation": len(
            train_rois
            & validation_rois
        ),
        "train_test": len(
            train_rois
            & test_rois
        ),
        "validation_test": len(
            validation_rois
            & test_rois
        ),
    }

    if any(case_overlap_counts.values()):
        raise RuntimeError(
            "Case leakage detected after ROI processing."
        )

    if any(roi_overlap_counts.values()):
        raise RuntimeError(
            "ROI leakage detected after ROI processing."
        )

    split_audit = {}

    for split_name in EXPECTED_SPLITS:
        X = split_results[
            split_name
        ]["X"]

        y = split_results[
            split_name
        ]["y"]

        case_ids = split_results[
            split_name
        ]["case_ids"]

        roi_ids = split_results[
            split_name
        ]["roi_ids"]

        if X.shape[0] != len(y):
            raise RuntimeError(
                f"{split_name}: X and y lengths differ."
            )

        if X.shape[0] != len(case_ids):
            raise RuntimeError(
                f"{split_name}: X and case ID lengths differ."
            )

        if X.shape[0] != len(roi_ids):
            raise RuntimeError(
                f"{split_name}: X and ROI ID lengths differ."
            )

        expected_shape = (
            X.shape[0],
            IMAGE_SIZE,
            IMAGE_SIZE,
            1,
        )

        if X.shape != expected_shape:
            raise RuntimeError(
                f"{split_name}: unexpected X shape {X.shape}."
            )

        if not np.isfinite(X).all():
            raise RuntimeError(
                f"{split_name}: non-finite pixel values found."
            )

        minimum_pixel = float(
            X.min()
        )

        maximum_pixel = float(
            X.max()
        )

        if minimum_pixel < 0.0:
            raise RuntimeError(
                f"{split_name}: pixel values below zero."
            )

        if maximum_pixel > 1.0:
            raise RuntimeError(
                f"{split_name}: pixel values above one."
            )

        unique_labels = sorted(
            np.unique(y)
            .astype(int)
            .tolist()
        )

        split_audit[split_name] = {
            "sample_count": int(
                len(X)
            ),
            "unique_case_count": int(
                len(
                    set(
                        case_ids.astype(str)
                    )
                )
            ),
            "unique_roi_count": int(
                len(
                    set(
                        roi_ids.astype(str)
                    )
                )
            ),
            "shape": list(
                X.shape
            ),
            "dtype": str(
                X.dtype
            ),
            "minimum_pixel": minimum_pixel,
            "maximum_pixel": maximum_pixel,
            "mean_pixel": float(
                X.mean()
            ),
            "unique_labels": unique_labels,
            "class_distribution": (
                class_distribution(y)
            ),
        }

    return {
        "split_audit": split_audit,
        "case_overlap_counts": (
            case_overlap_counts
        ),
        "roi_overlap_counts": (
            roi_overlap_counts
        ),
    }


def save_audit_csv(
    all_metadata_df: pd.DataFrame,
) -> None:
    """
    Save the combined successful ROI processing audit.
    """

    ordered_columns = [
        "roi_id",
        "case_id",
        "split",
        "tirads",
        "tirads_class",
        "image_name",
        "image_path",
        "actual_image_width",
        "actual_image_height",
        "original_bbox_x1",
        "original_bbox_y1",
        "original_bbox_x2",
        "original_bbox_y2",
        "expanded_bbox_x1",
        "expanded_bbox_y1",
        "expanded_bbox_x2",
        "expanded_bbox_y2",
        "extracted_roi_width",
        "extracted_roi_height",
        "output_width",
        "output_height",
        "output_channels",
        "minimum_pixel_value",
        "maximum_pixel_value",
        "mean_pixel_value",
        "processing_success",
        "failure_reason",
    ]

    available_columns = [
        column
        for column in ordered_columns
        if column in all_metadata_df.columns
    ]

    all_metadata_df[
        available_columns
    ].to_csv(
        DATASET_AUDIT_PATH,
        index=False,
    )


def save_summary(
    audit: dict[str, Any],
    failed_df: pd.DataFrame,
) -> None:
    """
    Save JSON distribution and text summary.
    """

    distribution_payload = {
        "configuration": {
            "image_size": IMAGE_SIZE,
            "channels": 1,
            "color_mode": "grayscale",
            "normalization": "[0, 1]",
            "roi_margin_fraction": (
                ROI_MARGIN_FRACTION
            ),
            "square_padding": True,
            "augmentation_applied": False,
            "smote_applied": False,
            "random_seed": RANDOM_SEED,
        },
        "processed_dataset": audit,
        "failed_roi_count": int(
            len(failed_df)
        ),
    }

    DATASET_DISTRIBUTION_PATH.write_text(
        json.dumps(
            distribution_payload,
            indent=4,
        )
        + "\n",
        encoding="utf-8",
    )

    split_audit = audit[
        "split_audit"
    ]

    lines = [
        "PHASE 13 - ROI DATASET PREPARATION SUMMARY",
        "=" * 72,
        "",
        "Configuration",
        "-" * 72,
        f"Output image size              : {IMAGE_SIZE} x {IMAGE_SIZE}",
        "Image channels                 : 1",
        "Color mode                     : Grayscale",
        "Pixel normalization            : [0, 1]",
        f"ROI margin fraction            : {ROI_MARGIN_FRACTION:.2f}",
        "Square padding                 : Enabled",
        "Augmentation                   : Not applied",
        "SMOTE                          : Not applied",
        "Model training                 : Not performed",
        "",
        "Processed split statistics",
        "-" * 72,
    ]

    for split_name in EXPECTED_SPLITS:
        current = split_audit[
            split_name
        ]

        lines.extend(
            [
                f"{split_name.capitalize()} samples"
                f"{' ' * max(1, 25 - len(split_name))}: "
                f"{current['sample_count']}",
                f"{split_name.capitalize()} unique cases"
                f"{' ' * max(1, 20 - len(split_name))}: "
                f"{current['unique_case_count']}",
                f"{split_name.capitalize()} array shape"
                f"{' ' * max(1, 24 - len(split_name))}: "
                f"{current['shape']}",
                f"{split_name.capitalize()} class counts"
                f"{' ' * max(1, 22 - len(split_name))}: "
                + ", ".join(
                    f"{label}="
                    f"{current['class_distribution'][label]}"
                    for label in EXPECTED_TIRADS_ORDER
                ),
                "",
            ]
        )

    lines.extend(
        [
            "Leakage checks",
            "-" * 72,
            (
                "Train/validation case overlap : "
                f"{audit['case_overlap_counts']['train_validation']}"
            ),
            (
                "Train/test case overlap       : "
                f"{audit['case_overlap_counts']['train_test']}"
            ),
            (
                "Validation/test case overlap  : "
                f"{audit['case_overlap_counts']['validation_test']}"
            ),
            (
                "Train/validation ROI overlap  : "
                f"{audit['roi_overlap_counts']['train_validation']}"
            ),
            (
                "Train/test ROI overlap        : "
                f"{audit['roi_overlap_counts']['train_test']}"
            ),
            (
                "Validation/test ROI overlap   : "
                f"{audit['roi_overlap_counts']['validation_test']}"
            ),
            "",
            "Processing failures",
            "-" * 72,
            f"Failed ROI records             : {len(failed_df)}",
            "",
            "Decision",
            "-" * 72,
            (
                "Phase 13 is valid only when all case-overlap and "
                "ROI-overlap counts are zero."
            ),
            (
                "The saved train, validation and test arrays must remain "
                "frozen for all subsequent baseline and ordinal models."
            ),
            (
                "Only the training set may receive augmentation or class "
                "balancing during later training phases."
            ),
            (
                "Validation and test data must never be augmented, "
                "oversampled or used for fitting preprocessing parameters."
            ),
        ]
    )

    DATASET_SUMMARY_PATH.write_text(
        "\n".join(lines)
        + "\n",
        encoding="utf-8",
    )


# =====================================================================
# Main execution
# =====================================================================

def main() -> None:
    """
    Execute Phase 13.
    """

    print("=" * 78)
    print("PHASE 13 : PREPARE ROI DATASET")
    print("=" * 78)

    validate_required_files()

    print("\nLoading frozen Phase 12 manifests...")

    train_df = load_split_manifest(
        TRAIN_MANIFEST_PATH,
        "train",
    )

    validation_df = load_split_manifest(
        VALIDATION_MANIFEST_PATH,
        "validation",
    )

    test_df = load_split_manifest(
        TEST_MANIFEST_PATH,
        "test",
    )

    print(
        f"Train ROI records      : {len(train_df)}"
    )

    print(
        f"Validation ROI records : {len(validation_df)}"
    )

    print(
        f"Test ROI records       : {len(test_df)}"
    )

    print(
        f"Total ROI records      : "
        f"{len(train_df) + len(validation_df) + len(test_df)}"
    )

    print("\nAuditing frozen split integrity...")

    audit_frozen_splits(
        train_df,
        validation_df,
        test_df,
    )

    print(
        "Frozen Phase 12 split integrity: PASSED"
    )

    split_dataframes = {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }

    split_results: dict[
        str,
        dict[str, Any],
    ] = {}

    all_success_metadata = []
    all_failed_metadata = []

    print("\nExtracting and processing ROI images...")

    for split_name in EXPECTED_SPLITS:
        print(
            "\n"
            + "-" * 78
        )

        print(
            f"Processing {split_name.upper()} split"
        )

        print(
            "-" * 78
        )

        (
            X,
            y,
            case_ids,
            roi_ids,
            success_df,
            failed_df,
        ) = process_split(
            split_dataframes[
                split_name
            ],
            split_name,
        )

        split_results[
            split_name
        ] = {
            "X": X,
            "y": y,
            "case_ids": case_ids,
            "roi_ids": roi_ids,
            "metadata": success_df,
            "failed": failed_df,
        }

        all_success_metadata.append(
            success_df
        )

        if not failed_df.empty:
            all_failed_metadata.append(
                failed_df
            )

        save_numpy_outputs(
            split_name=split_name,
            X=X,
            y=y,
            case_ids=case_ids,
            roi_ids=roi_ids,
            metadata_df=success_df,
        )

    if all_success_metadata:
        combined_success_df = pd.concat(
            all_success_metadata,
            ignore_index=True,
        )
    else:
        combined_success_df = pd.DataFrame()

    if all_failed_metadata:
        combined_failed_df = pd.concat(
            all_failed_metadata,
            ignore_index=True,
        )
    else:
        combined_failed_df = pd.DataFrame(
            columns=[
                "roi_id",
                "case_id",
                "split",
                "image_name",
                "image_path",
                "tirads",
                "tirads_class",
                "processing_success",
                "failure_reason",
            ]
        )

    save_audit_csv(
        combined_success_df
    )

    combined_failed_df.to_csv(
        FAILED_RECORDS_PATH,
        index=False,
    )

    print("\nVerifying processed dataset...")

    audit = verify_processed_outputs(
        split_results
    )

    print(
        "Processed dataset verification: PASSED"
    )

    print("\nCreating ROI preview grids...")

    create_preview(
        X=split_results["train"]["X"],
        y=split_results["train"]["y"],
        case_ids=split_results["train"]["case_ids"],
        roi_ids=split_results["train"]["roi_ids"],
        split_name="train",
        output_path=TRAIN_PREVIEW_PATH,
    )

    create_preview(
        X=split_results["validation"]["X"],
        y=split_results["validation"]["y"],
        case_ids=split_results["validation"]["case_ids"],
        roi_ids=split_results["validation"]["roi_ids"],
        split_name="validation",
        output_path=VALIDATION_PREVIEW_PATH,
    )

    create_preview(
        X=split_results["test"]["X"],
        y=split_results["test"]["y"],
        case_ids=split_results["test"]["case_ids"],
        roi_ids=split_results["test"]["roi_ids"],
        split_name="test",
        output_path=TEST_PREVIEW_PATH,
    )

    save_summary(
        audit=audit,
        failed_df=combined_failed_df,
    )

    print("\n" + "=" * 78)
    print("PHASE 13 ROI DATASET PREPARATION COMPLETE")
    print("=" * 78)

    print("\nPrepared arrays")
    print("-" * 78)

    for split_name in EXPECTED_SPLITS:
        current = audit[
            "split_audit"
        ][split_name]

        print(
            f"{split_name.capitalize():10s} "
            f"shape : {tuple(current['shape'])}"
        )

        print(
            f"{split_name.capitalize():10s} "
            f"cases : {current['unique_case_count']}"
        )

        print(
            f"{split_name.capitalize():10s} "
            f"class distribution : "
            + ", ".join(
                f"{label}="
                f"{current['class_distribution'][label]}"
                for label in EXPECTED_TIRADS_ORDER
            )
        )

    print("\nLeakage checks")
    print("-" * 78)

    print(
        "Train/validation case overlap : "
        f"{audit['case_overlap_counts']['train_validation']}"
    )

    print(
        "Train/test case overlap       : "
        f"{audit['case_overlap_counts']['train_test']}"
    )

    print(
        "Validation/test case overlap  : "
        f"{audit['case_overlap_counts']['validation_test']}"
    )

    print(
        "Train/validation ROI overlap  : "
        f"{audit['roi_overlap_counts']['train_validation']}"
    )

    print(
        "Train/test ROI overlap        : "
        f"{audit['roi_overlap_counts']['train_test']}"
    )

    print(
        "Validation/test ROI overlap   : "
        f"{audit['roi_overlap_counts']['validation_test']}"
    )

    print("\nProcessing failures")
    print("-" * 78)

    print(
        f"Failed ROI records             : "
        f"{len(combined_failed_df)}"
    )

    print("\nProcessed dataset directory")
    print("-" * 78)

    print(
        PROCESSED_DIR
    )

    print("\nResult directory")
    print("-" * 78)

    print(
        RESULT_DIR
    )

    print("\nImportant")
    print("-" * 78)

    print(
        "No augmentation was applied."
    )

    print(
        "No SMOTE was applied."
    )

    print(
        "No model was trained."
    )

    print(
        "The Phase 12 case-level split was not modified."
    )

    print(
        "Freeze these arrays for all subsequent experiments."
    )

    print("=" * 78)


if __name__ == "__main__":
    main()