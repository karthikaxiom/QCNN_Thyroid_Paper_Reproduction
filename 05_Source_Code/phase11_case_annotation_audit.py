"""
Phase 11 - Case and Annotation Audit

Purpose
-------
Audit the public DDTI dataset before any further model training.

This script:
1. Parses all XML files and matches them to images using the verified case rule.
2. Builds a case-level image manifest.
3. Parses freehand lesion annotations from each <mark>.
4. Validates image numbers, polygon points, bounds and image dimensions.
5. Audits the previous image-level train/test split for case leakage.
6. Audits the Phase 9 train/validation split for case leakage.
7. Creates a manifest of valid ROI candidates.
8. Generates a preview grid of lesion crops.
9. Saves all audit reports.

No model is trained.
No source image or annotation file is modified.
"""

from __future__ import annotations

import ast
import csv
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIR.parent

RAW_ROOT = PROJECT_ROOT / "02_Dataset_Raw" / "archive"
PROCESSED_DIR = PROJECT_ROOT / "03_Dataset_Processed"
MODEL_DIR = PROJECT_ROOT / "06_Models"

MASTER_DATASET_PATH = PROCESSED_DIR / "master_dataset.csv"
FINAL_DATASET_PATH = PROCESSED_DIR / "final_dataset.csv"

PHASE9_RESULT_DIR = (
    PROJECT_ROOT
    / "07_Results"
    / "phase9_paper_tirads_class_weighted"
)

TRAIN_INDICES_PATH = PHASE9_RESULT_DIR / "train_indices.npy"
VALIDATION_INDICES_PATH = PHASE9_RESULT_DIR / "validation_indices.npy"

RESULT_DIR = (
    PROJECT_ROOT
    / "07_Results"
    / "phase11_case_annotation_audit"
)

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Output files
# ---------------------------------------------------------------------

CASE_IMAGE_MAPPING_PATH = RESULT_DIR / "case_image_mapping.csv"
CASE_DISTRIBUTION_PATH = RESULT_DIR / "case_distribution.json"

ANNOTATION_INVENTORY_PATH = RESULT_DIR / "annotation_inventory.csv"
ANNOTATION_COVERAGE_REPORT_PATH = RESULT_DIR / "annotation_coverage_report.txt"

PREVIOUS_SPLIT_LEAKAGE_REPORT_PATH = (
    RESULT_DIR / "previous_split_leakage_report.txt"
)

INVALID_ANNOTATION_RECORDS_PATH = (
    RESULT_DIR / "invalid_annotation_records.csv"
)

ROI_CANDIDATE_MANIFEST_PATH = (
    RESULT_DIR / "roi_candidate_manifest.csv"
)

ROI_PREVIEW_GRID_PATH = RESULT_DIR / "roi_preview_grid.png"
AUDIT_SUMMARY_PATH = RESULT_DIR / "phase11_audit_summary.txt"

PREVIOUS_SPLIT_IMAGE_ASSIGNMENTS_PATH = (
    RESULT_DIR / "previous_split_image_assignments.csv"
)

LEAKING_CASE_DETAILS_PATH = (
    RESULT_DIR / "leaking_case_details.csv"
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

VALID_TIRADS = {
    "2",
    "3",
    "4a",
    "4b",
    "4c",
    "5",
}

TIRADS_TO_CLASS = {
    "2": 0,
    "3": 1,
    "4a": 2,
    "4b": 3,
    "4c": 4,
    "5": 5,
}

RANDOM_SEED = 42
TEST_SIZE = 0.20
ROI_MARGIN_RATIO = 0.15
PREVIEW_COUNT = 12


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def require_paths(paths: list[Path]) -> None:
    """Verify required files and folders."""

    missing = [str(path) for path in paths if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Required paths were not found:\n"
            + "\n".join(missing)
        )


def normalize_case_id(value: Any) -> str:
    """Convert a case ID to a clean string."""

    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.endswith(".0"):
        text = text[:-2]

    return text


def extract_case_id_from_xml_path(xml_path: Path) -> str:
    """Use the XML filename as the authoritative case ID."""

    return normalize_case_id(xml_path.stem)


def extract_image_number(
    image_name: str,
    case_id: str,
) -> int | None:
    """
    Extract image number from names such as:
    10_1.jpg, 10_2.jpg, or 10.jpg.
    """

    stem = Path(image_name).stem

    if stem == case_id:
        return 1

    pattern = rf"^{re.escape(case_id)}_(\d+)$"
    match = re.match(pattern, stem)

    if match:
        return int(match.group(1))

    return None


def find_images_for_case(
    case_id: str,
    all_images: list[Path],
) -> list[Path]:
    """
    Match images using the verified rule:
    image stem equals case_id or starts with case_id + "_".
    """

    matched: list[Path] = []

    for image_path in all_images:
        stem = image_path.stem

        if stem == case_id or stem.startswith(case_id + "_"):
            matched.append(image_path)

    def sort_key(path: Path) -> tuple[int, str]:
        number = extract_image_number(path.name, case_id)
        return (
            number if number is not None else 10**9,
            path.name.lower(),
        )

    return sorted(matched, key=sort_key)


def safe_text(element: ET.Element, tag: str) -> str:
    """Read an XML child value safely."""

    child = element.find(tag)

    if child is None or child.text is None:
        return ""

    return child.text.strip()


def parse_svg_payload(svg_text: str) -> list[dict[str, Any]]:
    """
    Parse the JSON-like SVG annotation payload.

    The dataset usually stores valid JSON. ast.literal_eval is used only
    as a fallback for minor formatting differences.
    """

    cleaned = svg_text.strip()

    if not cleaned:
        return []

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        payload = ast.literal_eval(cleaned)

    if isinstance(payload, dict):
        payload = [payload]

    if not isinstance(payload, list):
        raise ValueError(
            f"Unexpected SVG payload type: {type(payload).__name__}"
        )

    return payload


def validate_points(
    points: Any,
    width: int,
    height: int,
) -> tuple[
    bool,
    list[tuple[float, float]],
    list[str],
]:
    """Validate polygon points and image bounds."""

    issues: list[str] = []
    cleaned_points: list[tuple[float, float]] = []

    if not isinstance(points, list):
        return False, cleaned_points, ["points_not_a_list"]

    for index, point in enumerate(points):
        if not isinstance(point, dict):
            issues.append(f"point_{index}_not_a_dict")
            continue

        if "x" not in point or "y" not in point:
            issues.append(f"point_{index}_missing_xy")
            continue

        try:
            x = float(point["x"])
            y = float(point["y"])
        except (TypeError, ValueError):
            issues.append(f"point_{index}_non_numeric")
            continue

        if not math.isfinite(x) or not math.isfinite(y):
            issues.append(f"point_{index}_not_finite")
            continue

        cleaned_points.append((x, y))

        if x < 0 or x >= width or y < 0 or y >= height:
            issues.append(f"point_{index}_out_of_bounds")

    if len(cleaned_points) < 3:
        issues.append("fewer_than_three_valid_points")

    unique_points = set(cleaned_points)

    if len(unique_points) < 3:
        issues.append("fewer_than_three_unique_points")

    return len(issues) == 0, cleaned_points, issues


def bounding_box_from_points(
    points: list[tuple[float, float]],
    width: int,
    height: int,
    margin_ratio: float = ROI_MARGIN_RATIO,
) -> tuple[int, int, int, int]:
    """Build a clipped ROI bounding box with contextual margin."""

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    box_width = max(max_x - min_x, 1.0)
    box_height = max(max_y - min_y, 1.0)

    margin_x = box_width * margin_ratio
    margin_y = box_height * margin_ratio

    x1 = max(0, int(math.floor(min_x - margin_x)))
    y1 = max(0, int(math.floor(min_y - margin_y)))
    x2 = min(width, int(math.ceil(max_x + margin_x + 1)))
    y2 = min(height, int(math.ceil(max_y + margin_y + 1)))

    if x2 <= x1:
        x2 = min(width, x1 + 1)

    if y2 <= y1:
        y2 = min(height, y1 + 1)

    return x1, y1, x2, y2


def dataframe_to_csv(
    rows: list[dict[str, Any]],
    output_path: Path,
    columns: list[str],
) -> pd.DataFrame:
    """Save rows with stable columns, including when there are no rows."""

    dataframe = pd.DataFrame(rows, columns=columns)
    dataframe.to_csv(output_path, index=False)
    return dataframe


# ---------------------------------------------------------------------
# XML, image and annotation audit
# ---------------------------------------------------------------------

def build_case_and_annotation_audit() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Parse all cases, images and lesion annotations.

    Returns:
        case_image_df
        annotation_df
        invalid_df
        roi_df
    """

    print("\nDiscovering raw XML and image files...")

    xml_files = sorted(
        RAW_ROOT.rglob("*.xml"),
        key=lambda path: path.name.lower(),
    )

    all_images = sorted(
        [
            path
            for path in RAW_ROOT.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ],
        key=lambda path: path.name.lower(),
    )

    print(f"XML files found   : {len(xml_files)}")
    print(f"Image files found : {len(all_images)}")

    image_path_lookup = {
        str(path.resolve()).lower(): path
        for path in all_images
    }

    image_name_lookup: dict[str, list[Path]] = defaultdict(list)

    for image_path in all_images:
        image_name_lookup[image_path.name.lower()].append(image_path)

    case_image_rows: list[dict[str, Any]] = []
    annotation_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    roi_rows: list[dict[str, Any]] = []

    for xml_index, xml_path in enumerate(xml_files, start=1):
        case_id = extract_case_id_from_xml_path(xml_path)

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception as error:
            invalid_rows.append(
                {
                    "case_id": case_id,
                    "xml_file": xml_path.name,
                    "image_name": "",
                    "image_number": "",
                    "mark_index": "",
                    "region_index": "",
                    "error_type": "xml_parse_error",
                    "details": str(error),
                }
            )
            continue

        xml_case_number = safe_text(root, "number")
        age = safe_text(root, "age")
        sex = safe_text(root, "sex")
        composition = safe_text(root, "composition")
        echogenicity = safe_text(root, "echogenicity")
        margins = safe_text(root, "margins")
        calcifications = safe_text(root, "calcifications")
        tirads = safe_text(root, "tirads").lower()

        matched_images = find_images_for_case(
            case_id,
            all_images,
        )

        image_number_map: dict[int, Path] = {}

        for image_path in matched_images:
            image_number = extract_image_number(
                image_path.name,
                case_id,
            )

            width = ""
            height = ""
            image_readable = False
            image_error = ""

            try:
                with Image.open(image_path) as image:
                    width, height = image.size
                    image_readable = True
            except Exception as error:
                image_error = str(error)

            if image_number is not None:
                if image_number in image_number_map:
                    invalid_rows.append(
                        {
                            "case_id": case_id,
                            "xml_file": xml_path.name,
                            "image_name": image_path.name,
                            "image_number": image_number,
                            "mark_index": "",
                            "region_index": "",
                            "error_type": "duplicate_image_number",
                            "details": (
                                f"Another image is already mapped to "
                                f"number {image_number}"
                            ),
                        }
                    )
                else:
                    image_number_map[image_number] = image_path

            case_image_rows.append(
                {
                    "case_id": case_id,
                    "xml_case_number": xml_case_number,
                    "xml_file": xml_path.name,
                    "image_name": image_path.name,
                    "image_number": image_number,
                    "image_path": str(image_path),
                    "image_width": width,
                    "image_height": height,
                    "image_readable": image_readable,
                    "image_error": image_error,
                    "age": age,
                    "sex": sex,
                    "composition": composition,
                    "echogenicity": echogenicity,
                    "margins": margins,
                    "calcifications": calcifications,
                    "tirads": tirads,
                    "tirads_valid": tirads in VALID_TIRADS,
                    "case_image_count": len(matched_images),
                }
            )

        if not matched_images:
            invalid_rows.append(
                {
                    "case_id": case_id,
                    "xml_file": xml_path.name,
                    "image_name": "",
                    "image_number": "",
                    "mark_index": "",
                    "region_index": "",
                    "error_type": "no_images_for_case",
                    "details": (
                        "No image matched the verified case naming rule."
                    ),
                }
            )

        marks = root.findall("mark")

        if not marks:
            invalid_rows.append(
                {
                    "case_id": case_id,
                    "xml_file": xml_path.name,
                    "image_name": "",
                    "image_number": "",
                    "mark_index": "",
                    "region_index": "",
                    "error_type": "no_mark_elements",
                    "details": "XML contains no <mark> element.",
                }
            )

        for mark_index, mark in enumerate(marks, start=1):
            image_number_text = safe_text(mark, "image")
            svg_text = safe_text(mark, "svg")

            try:
                image_number = int(image_number_text)
            except (TypeError, ValueError):
                invalid_rows.append(
                    {
                        "case_id": case_id,
                        "xml_file": xml_path.name,
                        "image_name": "",
                        "image_number": image_number_text,
                        "mark_index": mark_index,
                        "region_index": "",
                        "error_type": "invalid_mark_image_number",
                        "details": (
                            f"Cannot convert image number "
                            f"{image_number_text!r} to integer."
                        ),
                    }
                )
                continue

            image_path = image_number_map.get(image_number)

            if image_path is None:
                invalid_rows.append(
                    {
                        "case_id": case_id,
                        "xml_file": xml_path.name,
                        "image_name": "",
                        "image_number": image_number,
                        "mark_index": mark_index,
                        "region_index": "",
                        "error_type": "mark_image_not_found",
                        "details": (
                            "No case image is mapped to this mark image "
                            "number."
                        ),
                    }
                )
                continue

            try:
                with Image.open(image_path) as image:
                    width, height = image.size
            except Exception as error:
                invalid_rows.append(
                    {
                        "case_id": case_id,
                        "xml_file": xml_path.name,
                        "image_name": image_path.name,
                        "image_number": image_number,
                        "mark_index": mark_index,
                        "region_index": "",
                        "error_type": "image_open_error",
                        "details": str(error),
                    }
                )
                continue

            try:
                regions = parse_svg_payload(svg_text)
            except Exception as error:
                invalid_rows.append(
                    {
                        "case_id": case_id,
                        "xml_file": xml_path.name,
                        "image_name": image_path.name,
                        "image_number": image_number,
                        "mark_index": mark_index,
                        "region_index": "",
                        "error_type": "svg_parse_error",
                        "details": str(error),
                    }
                )
                continue

            if not regions:
                invalid_rows.append(
                    {
                        "case_id": case_id,
                        "xml_file": xml_path.name,
                        "image_name": image_path.name,
                        "image_number": image_number,
                        "mark_index": mark_index,
                        "region_index": "",
                        "error_type": "empty_svg_regions",
                        "details": "No annotation region found.",
                    }
                )

            for region_index, region in enumerate(
                regions,
                start=1,
            ):
                region_type = str(
                    region.get("regionType", "")
                ).strip()

                points_value = region.get("points")

                (
                    points_valid,
                    cleaned_points,
                    point_issues,
                ) = validate_points(
                    points_value,
                    width,
                    height,
                )

                point_count = len(cleaned_points)
                unique_point_count = len(set(cleaned_points))

                if cleaned_points:
                    xs = [point[0] for point in cleaned_points]
                    ys = [point[1] for point in cleaned_points]

                    min_x = min(xs)
                    max_x = max(xs)
                    min_y = min(ys)
                    max_y = max(ys)

                    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = (
                        bounding_box_from_points(
                            cleaned_points,
                            width,
                            height,
                        )
                    )

                    bbox_width = bbox_x2 - bbox_x1
                    bbox_height = bbox_y2 - bbox_y1
                    bbox_area = bbox_width * bbox_height
                    image_area = width * height
                    bbox_area_ratio = (
                        bbox_area / image_area
                        if image_area > 0
                        else 0.0
                    )
                else:
                    min_x = max_x = min_y = max_y = ""
                    bbox_x1 = bbox_y1 = bbox_x2 = bbox_y2 = ""
                    bbox_width = bbox_height = bbox_area = ""
                    bbox_area_ratio = ""

                annotation_valid = (
                    points_valid
                    and region_type.lower() == "freehand"
                )

                if region_type.lower() != "freehand":
                    point_issues.append(
                        f"unexpected_region_type:{region_type}"
                    )

                annotation_rows.append(
                    {
                        "case_id": case_id,
                        "xml_file": xml_path.name,
                        "image_name": image_path.name,
                        "image_path": str(image_path),
                        "image_number": image_number,
                        "image_width": width,
                        "image_height": height,
                        "mark_index": mark_index,
                        "region_index": region_index,
                        "region_type": region_type,
                        "point_count": point_count,
                        "unique_point_count": unique_point_count,
                        "min_x": min_x,
                        "max_x": max_x,
                        "min_y": min_y,
                        "max_y": max_y,
                        "bbox_x1": bbox_x1,
                        "bbox_y1": bbox_y1,
                        "bbox_x2": bbox_x2,
                        "bbox_y2": bbox_y2,
                        "bbox_width": bbox_width,
                        "bbox_height": bbox_height,
                        "bbox_area": bbox_area,
                        "bbox_area_ratio": bbox_area_ratio,
                        "tirads": tirads,
                        "tirads_class": TIRADS_TO_CLASS.get(
                            tirads,
                            "",
                        ),
                        "tirads_valid": tirads in VALID_TIRADS,
                        "annotation_valid": annotation_valid,
                        "validation_issues": ";".join(
                            sorted(set(point_issues))
                        ),
                    }
                )

                if not annotation_valid:
                    invalid_rows.append(
                        {
                            "case_id": case_id,
                            "xml_file": xml_path.name,
                            "image_name": image_path.name,
                            "image_number": image_number,
                            "mark_index": mark_index,
                            "region_index": region_index,
                            "error_type": "invalid_annotation_region",
                            "details": ";".join(
                                sorted(set(point_issues))
                            ),
                        }
                    )
                elif tirads in VALID_TIRADS:
                    roi_rows.append(
                        {
                            "roi_id": (
                                f"{case_id}_{image_number}_"
                                f"{mark_index}_{region_index}"
                            ),
                            "case_id": case_id,
                            "xml_file": xml_path.name,
                            "image_name": image_path.name,
                            "image_path": str(image_path),
                            "image_number": image_number,
                            "mark_index": mark_index,
                            "region_index": region_index,
                            "region_type": region_type,
                            "point_count": point_count,
                            "image_width": width,
                            "image_height": height,
                            "bbox_x1": bbox_x1,
                            "bbox_y1": bbox_y1,
                            "bbox_x2": bbox_x2,
                            "bbox_y2": bbox_y2,
                            "bbox_width": bbox_width,
                            "bbox_height": bbox_height,
                            "bbox_area_ratio": bbox_area_ratio,
                            "tirads": tirads,
                            "tirads_class": TIRADS_TO_CLASS[tirads],
                            "points_json": json.dumps(
                                [
                                    {"x": x, "y": y}
                                    for x, y in cleaned_points
                                ],
                                separators=(",", ":"),
                            ),
                            "roi_valid": True,
                        }
                    )

        if xml_index % 50 == 0 or xml_index == len(xml_files):
            print(
                f"Processed XML files: "
                f"{xml_index}/{len(xml_files)}"
            )

    case_image_columns = [
        "case_id",
        "xml_case_number",
        "xml_file",
        "image_name",
        "image_number",
        "image_path",
        "image_width",
        "image_height",
        "image_readable",
        "image_error",
        "age",
        "sex",
        "composition",
        "echogenicity",
        "margins",
        "calcifications",
        "tirads",
        "tirads_valid",
        "case_image_count",
    ]

    annotation_columns = [
        "case_id",
        "xml_file",
        "image_name",
        "image_path",
        "image_number",
        "image_width",
        "image_height",
        "mark_index",
        "region_index",
        "region_type",
        "point_count",
        "unique_point_count",
        "min_x",
        "max_x",
        "min_y",
        "max_y",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "bbox_width",
        "bbox_height",
        "bbox_area",
        "bbox_area_ratio",
        "tirads",
        "tirads_class",
        "tirads_valid",
        "annotation_valid",
        "validation_issues",
    ]

    invalid_columns = [
        "case_id",
        "xml_file",
        "image_name",
        "image_number",
        "mark_index",
        "region_index",
        "error_type",
        "details",
    ]

    roi_columns = [
        "roi_id",
        "case_id",
        "xml_file",
        "image_name",
        "image_path",
        "image_number",
        "mark_index",
        "region_index",
        "region_type",
        "point_count",
        "image_width",
        "image_height",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "bbox_width",
        "bbox_height",
        "bbox_area_ratio",
        "tirads",
        "tirads_class",
        "points_json",
        "roi_valid",
    ]

    case_image_df = dataframe_to_csv(
        case_image_rows,
        CASE_IMAGE_MAPPING_PATH,
        case_image_columns,
    )

    annotation_df = dataframe_to_csv(
        annotation_rows,
        ANNOTATION_INVENTORY_PATH,
        annotation_columns,
    )

    invalid_df = dataframe_to_csv(
        invalid_rows,
        INVALID_ANNOTATION_RECORDS_PATH,
        invalid_columns,
    )

    # Deduplicate exact ROI geometries while preserving an audit trail.
    geometry_columns = [
        "case_id",
        "image_name",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "points_json",
    ]

    unique_roi_rows = []
    duplicate_roi_rows = []
    retained_roi_by_geometry = {}

    for row in roi_rows:
        geometry_key = tuple(
            row[column] for column in geometry_columns
        )

        if geometry_key in retained_roi_by_geometry:
            duplicate_row = dict(row)
            duplicate_row["retained_roi_id"] = (
                retained_roi_by_geometry[geometry_key]
            )
            duplicate_roi_rows.append(duplicate_row)
            continue

        retained_roi_by_geometry[geometry_key] = row["roi_id"]
        unique_roi_rows.append(row)

    duplicate_geometry_path = (
        ROI_CANDIDATE_MANIFEST_PATH.parent
        / "duplicate_roi_geometry_records.csv"
    )

    pd.DataFrame(
        duplicate_roi_rows,
        columns=roi_columns + ["retained_roi_id"],
    ).to_csv(
        duplicate_geometry_path,
        index=False,
    )

    print(
        "Exact duplicate ROI geometries removed: "
        f"{len(duplicate_roi_rows)}"
    )
    print(
        "Unique ROI geometries retained: "
        f"{len(unique_roi_rows)}"
    )

    roi_df = dataframe_to_csv(
        unique_roi_rows,
        ROI_CANDIDATE_MANIFEST_PATH,
        roi_columns,
    )

    return (
        case_image_df,
        annotation_df,
        invalid_df,
        roi_df,
    )


# ---------------------------------------------------------------------
# Case distribution
# ---------------------------------------------------------------------

def save_case_distribution(
    case_image_df: pd.DataFrame,
    roi_df: pd.DataFrame,
) -> dict[str, Any]:
    """Summarise images and annotations at case level."""

    unique_cases = (
        case_image_df["case_id"]
        .astype(str)
        .nunique()
        if not case_image_df.empty
        else 0
    )

    image_counts = (
        case_image_df.groupby("case_id")["image_name"]
        .nunique()
        if not case_image_df.empty
        else pd.Series(dtype=int)
    )

    labelled_rows = case_image_df[
        case_image_df["tirads_valid"] == True
    ]

    labelled_cases = (
        labelled_rows["case_id"]
        .astype(str)
        .nunique()
        if not labelled_rows.empty
        else 0
    )

    roi_case_counts = (
        roi_df.groupby("case_id")["roi_id"]
        .nunique()
        if not roi_df.empty
        else pd.Series(dtype=int)
    )

    tirads_image_distribution = (
        labelled_rows["tirads"]
        .value_counts()
        .reindex(
            ["2", "3", "4a", "4b", "4c", "5"],
            fill_value=0,
        )
        .astype(int)
        .to_dict()
    )

    tirads_case_distribution = {}

    if not labelled_rows.empty:
        one_label_per_case = (
            labelled_rows.sort_values(
                ["case_id", "image_number"]
            )
            .drop_duplicates("case_id")
        )

        tirads_case_distribution = (
            one_label_per_case["tirads"]
            .value_counts()
            .reindex(
                ["2", "3", "4a", "4b", "4c", "5"],
                fill_value=0,
            )
            .astype(int)
            .to_dict()
        )
    else:
        tirads_case_distribution = {
            label: 0
            for label in ["2", "3", "4a", "4b", "4c", "5"]
        }

    distribution = {
        "total_case_image_rows": int(len(case_image_df)),
        "unique_cases_with_images": int(unique_cases),
        "cases_with_valid_tirads": int(labelled_cases),
        "cases_with_multiple_images": int(
            (image_counts > 1).sum()
        ),
        "maximum_images_in_one_case": int(
            image_counts.max()
            if not image_counts.empty
            else 0
        ),
        "total_valid_roi_regions": int(len(roi_df)),
        "cases_with_valid_roi_regions": int(
            roi_df["case_id"].astype(str).nunique()
            if not roi_df.empty
            else 0
        ),
        "maximum_roi_regions_in_one_case": int(
            roi_case_counts.max()
            if not roi_case_counts.empty
            else 0
        ),
        "image_count_frequency": {
            str(int(image_count)): int(case_count)
            for image_count, case_count
            in image_counts.value_counts()
            .sort_index()
            .items()
        },
        "tirads_image_distribution": (
            tirads_image_distribution
        ),
        "tirads_case_distribution": (
            tirads_case_distribution
        ),
    }

    CASE_DISTRIBUTION_PATH.write_text(
        json.dumps(
            distribution,
            indent=4,
        )
        + "\n",
        encoding="utf-8",
    )

    return distribution


# ---------------------------------------------------------------------
# Previous split leakage audit
# ---------------------------------------------------------------------

def reconstruct_previous_test_split(
    final_df: pd.DataFrame,
) -> tuple[
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
]:
    """
    Reconstruct the Phase 8 image-level 80:20 split.

    The same dataset filtering and random_state=42 protocol are used.
    """

    labelled_df = final_df[
        final_df["tirads"]
        .astype(str)
        .str.lower()
        .isin(VALID_TIRADS)
    ].copy()

    labelled_df["tirads"] = (
        labelled_df["tirads"]
        .astype(str)
        .str.lower()
    )

    labelled_df["tirads_class"] = (
        labelled_df["tirads"]
        .map(TIRADS_TO_CLASS)
        .astype(int)
    )

    original_row_indices = labelled_df.index.to_numpy()
    labels = labelled_df["tirads_class"].to_numpy()

    train_rows, test_rows = train_test_split(
        original_row_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    assignment = labelled_df[
        [
            "case_id",
            "image_name",
            "image_number",
            "tirads",
            "tirads_class",
        ]
    ].copy()

    assignment["original_final_dataset_row"] = (
        assignment.index
    )

    assignment["previous_train_test_split"] = np.where(
        assignment.index.isin(test_rows),
        "test",
        "train",
    )

    return train_rows, test_rows, assignment


def audit_previous_splits(
    final_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Audit:
    1. Previous Phase 8 train/test image split.
    2. Previous Phase 9 train/validation split within the old train set.
    """

    (
        old_train_rows,
        old_test_rows,
        assignment,
    ) = reconstruct_previous_test_split(final_df)

    assignment["case_id"] = (
        assignment["case_id"]
        .map(normalize_case_id)
    )

    train_cases = set(
        assignment.loc[
            assignment["previous_train_test_split"] == "train",
            "case_id",
        ]
    )

    test_cases = set(
        assignment.loc[
            assignment["previous_train_test_split"] == "test",
            "case_id",
        ]
    )

    train_test_leaking_cases = sorted(
        train_cases & test_cases,
        key=lambda value: (
            int(value)
            if value.isdigit()
            else value
        ),
    )

    # Preserve old training order exactly as train_test_split returned it.
    labelled_df = final_df[
        final_df["tirads"]
        .astype(str)
        .str.lower()
        .isin(VALID_TIRADS)
    ].copy()

    old_train_df = labelled_df.loc[
        old_train_rows
    ].copy()

    old_train_df = old_train_df.reset_index().rename(
        columns={
            "index": "original_final_dataset_row"
        }
    )

    phase9_available = (
        TRAIN_INDICES_PATH.exists()
        and VALIDATION_INDICES_PATH.exists()
    )

    train_validation_leaking_cases: list[str] = []
    phase9_train_cases: set[str] = set()
    phase9_validation_cases: set[str] = set()

    if phase9_available:
        train_indices = np.load(
            TRAIN_INDICES_PATH
        ).astype(int)

        validation_indices = np.load(
            VALIDATION_INDICES_PATH
        ).astype(int)

        expected_index_set = set(
            range(len(old_train_df))
        )

        combined_index_set = set(
            np.concatenate(
                [
                    train_indices,
                    validation_indices,
                ]
            ).tolist()
        )

        if combined_index_set != expected_index_set:
            raise ValueError(
                "Phase 9 train/validation indices do not match "
                "the reconstructed 279-sample old training set."
            )

        old_train_df["phase9_split"] = ""

        old_train_df.loc[
            train_indices,
            "phase9_split",
        ] = "train"

        old_train_df.loc[
            validation_indices,
            "phase9_split",
        ] = "validation"

        old_train_df["case_id"] = (
            old_train_df["case_id"]
            .map(normalize_case_id)
        )

        phase9_train_cases = set(
            old_train_df.loc[
                old_train_df["phase9_split"] == "train",
                "case_id",
            ]
        )

        phase9_validation_cases = set(
            old_train_df.loc[
                old_train_df["phase9_split"] == "validation",
                "case_id",
            ]
        )

        train_validation_leaking_cases = sorted(
            phase9_train_cases
            & phase9_validation_cases,
            key=lambda value: (
                int(value)
                if value.isdigit()
                else value
            ),
        )

        assignment = assignment.merge(
            old_train_df[
                [
                    "original_final_dataset_row",
                    "phase9_split",
                ]
            ],
            on="original_final_dataset_row",
            how="left",
        )
    else:
        assignment["phase9_split"] = ""

    assignment.to_csv(
        PREVIOUS_SPLIT_IMAGE_ASSIGNMENTS_PATH,
        index=False,
    )

    detail_rows: list[dict[str, Any]] = []

    for case_id in train_test_leaking_cases:
        case_rows = assignment[
            assignment["case_id"] == case_id
        ]

        detail_rows.append(
            {
                "audit_type": "old_train_test",
                "case_id": case_id,
                "train_images": ";".join(
                    case_rows.loc[
                        case_rows[
                            "previous_train_test_split"
                        ]
                        == "train",
                        "image_name",
                    ].astype(str)
                ),
                "test_or_validation_images": ";".join(
                    case_rows.loc[
                        case_rows[
                            "previous_train_test_split"
                        ]
                        == "test",
                        "image_name",
                    ].astype(str)
                ),
            }
        )

    if phase9_available:
        for case_id in train_validation_leaking_cases:
            case_rows = old_train_df[
                old_train_df["case_id"] == case_id
            ]

            detail_rows.append(
                {
                    "audit_type": "phase9_train_validation",
                    "case_id": case_id,
                    "train_images": ";".join(
                        case_rows.loc[
                            case_rows["phase9_split"]
                            == "train",
                            "image_name",
                        ].astype(str)
                    ),
                    "test_or_validation_images": ";".join(
                        case_rows.loc[
                            case_rows["phase9_split"]
                            == "validation",
                            "image_name",
                        ].astype(str)
                    ),
                }
            )

    dataframe_to_csv(
        detail_rows,
        LEAKING_CASE_DETAILS_PATH,
        [
            "audit_type",
            "case_id",
            "train_images",
            "test_or_validation_images",
        ],
    )

    leakage_summary = {
        "old_train_images": int(
            (
                assignment[
                    "previous_train_test_split"
                ]
                == "train"
            ).sum()
        ),
        "old_test_images": int(
            (
                assignment[
                    "previous_train_test_split"
                ]
                == "test"
            ).sum()
        ),
        "old_train_cases": int(len(train_cases)),
        "old_test_cases": int(len(test_cases)),
        "old_train_test_leaking_case_count": int(
            len(train_test_leaking_cases)
        ),
        "old_train_test_leaking_cases": (
            train_test_leaking_cases
        ),
        "phase9_indices_available": phase9_available,
        "phase9_train_case_count": int(
            len(phase9_train_cases)
        ),
        "phase9_validation_case_count": int(
            len(phase9_validation_cases)
        ),
        "phase9_train_validation_leaking_case_count": int(
            len(train_validation_leaking_cases)
        ),
        "phase9_train_validation_leaking_cases": (
            train_validation_leaking_cases
        ),
    }

    lines = [
        "PHASE 11 - PREVIOUS SPLIT LEAKAGE AUDIT",
        "=" * 54,
        "",
        "Previous image-level train/test split",
        "-" * 54,
        f"Training images              : "
        f"{leakage_summary['old_train_images']}",
        f"Test images                  : "
        f"{leakage_summary['old_test_images']}",
        f"Training cases               : "
        f"{leakage_summary['old_train_cases']}",
        f"Test cases                   : "
        f"{leakage_summary['old_test_cases']}",
        f"Cases present in both        : "
        f"{leakage_summary['old_train_test_leaking_case_count']}",
        "",
        "Leaking train/test case IDs:",
        ", ".join(train_test_leaking_cases)
        if train_test_leaking_cases
        else "None",
        "",
        "Phase 9 train/validation split",
        "-" * 54,
        f"Saved indices available      : "
        f"{phase9_available}",
        f"Training cases               : "
        f"{leakage_summary['phase9_train_case_count']}",
        f"Validation cases             : "
        f"{leakage_summary['phase9_validation_case_count']}",
        f"Cases present in both        : "
        f"{leakage_summary['phase9_train_validation_leaking_case_count']}",
        "",
        "Leaking train/validation case IDs:",
        ", ".join(train_validation_leaking_cases)
        if train_validation_leaking_cases
        else "None",
        "",
        "Interpretation",
        "-" * 54,
        (
            "The previous experiments used image-level splitting. "
            "When a case contains multiple images, related images can "
            "appear in different subsets. These experiments remain part "
            "of the reproduction record, but the final proposed-method "
            "evaluation must use case-level splitting."
        ),
    ]

    PREVIOUS_SPLIT_LEAKAGE_REPORT_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return leakage_summary


# ---------------------------------------------------------------------
# Reports and preview
# ---------------------------------------------------------------------

def save_annotation_coverage_report(
    case_image_df: pd.DataFrame,
    annotation_df: pd.DataFrame,
    invalid_df: pd.DataFrame,
    roi_df: pd.DataFrame,
) -> dict[str, Any]:
    """Save annotation coverage and validity statistics."""

    total_images = int(
        case_image_df["image_name"].nunique()
        if not case_image_df.empty
        else 0
    )

    annotated_images = int(
        annotation_df["image_name"].nunique()
        if not annotation_df.empty
        else 0
    )

    valid_annotation_regions = int(
        (
            annotation_df["annotation_valid"]
            == True
        ).sum()
        if not annotation_df.empty
        else 0
    )

    invalid_annotation_regions = int(
        (
            annotation_df["annotation_valid"]
            != True
        ).sum()
        if not annotation_df.empty
        else 0
    )

    coverage = {
        "total_images_matched_to_xml_cases": total_images,
        "images_with_at_least_one_annotation": (
            annotated_images
        ),
        "annotation_image_coverage_percent": (
            100.0 * annotated_images / total_images
            if total_images > 0
            else 0.0
        ),
        "total_annotation_regions": int(
            len(annotation_df)
        ),
        "valid_annotation_regions": (
            valid_annotation_regions
        ),
        "invalid_annotation_regions": (
            invalid_annotation_regions
        ),
        "valid_tirads_roi_regions": int(len(roi_df)),
        "valid_tirads_roi_images": int(
            roi_df["image_name"].nunique()
            if not roi_df.empty
            else 0
        ),
        "valid_tirads_roi_cases": int(
            roi_df["case_id"].astype(str).nunique()
            if not roi_df.empty
            else 0
        ),
        "invalid_record_count": int(len(invalid_df)),
        "region_type_distribution": (
            annotation_df["region_type"]
            .fillna("")
            .value_counts()
            .to_dict()
            if not annotation_df.empty
            else {}
        ),
        "regions_per_image_distribution": (
            annotation_df.groupby("image_name")
            .size()
            .value_counts()
            .sort_index()
            .astype(int)
            .to_dict()
            if not annotation_df.empty
            else {}
        ),
    }

    lines = [
        "PHASE 11 - ANNOTATION COVERAGE REPORT",
        "=" * 50,
        "",
        f"Images matched to XML cases        : "
        f"{coverage['total_images_matched_to_xml_cases']}",
        f"Images with annotations            : "
        f"{coverage['images_with_at_least_one_annotation']}",
        f"Annotation image coverage          : "
        f"{coverage['annotation_image_coverage_percent']:.2f}%",
        f"Total annotation regions           : "
        f"{coverage['total_annotation_regions']}",
        f"Valid annotation regions           : "
        f"{coverage['valid_annotation_regions']}",
        f"Invalid annotation regions         : "
        f"{coverage['invalid_annotation_regions']}",
        f"Valid TI-RADS ROI regions          : "
        f"{coverage['valid_tirads_roi_regions']}",
        f"Valid TI-RADS ROI images           : "
        f"{coverage['valid_tirads_roi_images']}",
        f"Valid TI-RADS ROI cases            : "
        f"{coverage['valid_tirads_roi_cases']}",
        f"Invalid audit records              : "
        f"{coverage['invalid_record_count']}",
        "",
        "Region type distribution",
        "-" * 50,
    ]

    for key, value in coverage[
        "region_type_distribution"
    ].items():
        lines.append(f"{key or '<blank>'}: {value}")

    lines.extend(
        [
            "",
            "Regions per annotated image",
            "-" * 50,
        ]
    )

    for region_count, image_count in coverage[
        "regions_per_image_distribution"
    ].items():
        lines.append(
            f"{region_count} region(s): {image_count} image(s)"
        )

    ANNOTATION_COVERAGE_REPORT_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return coverage


def create_roi_preview_grid(
    roi_df: pd.DataFrame,
) -> None:
    """Generate a representative grid of annotated ROI crops."""

    if roi_df.empty:
        print("No valid ROI candidates available for preview.")
        return

    selected_rows: list[pd.Series] = []

    for tirads in ["2", "3", "4a", "4b", "4c", "5"]:
        class_rows = roi_df[
            roi_df["tirads"] == tirads
        ]

        if not class_rows.empty:
            selected_rows.append(
                class_rows.iloc[0]
            )

    remaining_count = PREVIEW_COUNT - len(selected_rows)

    if remaining_count > 0:
        selected_ids = {
            row["roi_id"]
            for row in selected_rows
        }

        remaining_rows = roi_df[
            ~roi_df["roi_id"].isin(selected_ids)
        ]

        for _, row in remaining_rows.head(
            remaining_count
        ).iterrows():
            selected_rows.append(row)

    selected_rows = selected_rows[:PREVIEW_COUNT]

    if not selected_rows:
        return

    columns = 3
    rows = math.ceil(len(selected_rows) / columns)

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(12, 4 * rows),
    )

    axes_array = np.atleast_1d(axes).ravel()

    for axis in axes_array:
        axis.axis("off")

    for axis, row in zip(
        axes_array,
        selected_rows,
    ):
        image_path = Path(str(row["image_path"]))

        try:
            with Image.open(image_path) as image:
                image_rgb = image.convert("RGB")

                points = json.loads(
                    row["points_json"]
                )

                point_tuples = [
                    (
                        float(point["x"]),
                        float(point["y"]),
                    )
                    for point in points
                ]

                overlay = image_rgb.copy()
                draw = ImageDraw.Draw(overlay)

                draw.line(
                    point_tuples
                    + [point_tuples[0]],
                    fill="white",
                    width=2,
                )

                x1 = int(row["bbox_x1"])
                y1 = int(row["bbox_y1"])
                x2 = int(row["bbox_x2"])
                y2 = int(row["bbox_y2"])

                crop = overlay.crop(
                    (x1, y1, x2, y2)
                )

                axis.imshow(crop)
                axis.set_title(
                    f"Case {row['case_id']} | "
                    f"{row['image_name']} | "
                    f"TI-RADS {row['tirads']}"
                )
                axis.axis("off")
        except Exception as error:
            axis.text(
                0.5,
                0.5,
                f"Preview error:\n{error}",
                ha="center",
                va="center",
                wrap=True,
            )
            axis.axis("off")

    figure.suptitle(
        "Phase 11: Annotation-Guided ROI Preview",
        fontsize=14,
    )

    figure.tight_layout(
        rect=[0, 0, 1, 0.97]
    )

    figure.savefig(
        ROI_PREVIEW_GRID_PATH,
        dpi=250,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_final_summary(
    distribution: dict[str, Any],
    coverage: dict[str, Any],
    leakage: dict[str, Any],
    invalid_df: pd.DataFrame,
) -> None:
    """Save the consolidated Phase 11 audit summary."""

    lines = [
        "PHASE 11 - CASE AND ANNOTATION AUDIT SUMMARY",
        "=" * 58,
        "",
        "Dataset structure",
        "-" * 58,
        f"Unique cases with images         : "
        f"{distribution['unique_cases_with_images']}",
        f"Cases with valid TI-RADS         : "
        f"{distribution['cases_with_valid_tirads']}",
        f"Cases with multiple images       : "
        f"{distribution['cases_with_multiple_images']}",
        f"Maximum images in one case       : "
        f"{distribution['maximum_images_in_one_case']}",
        "",
        "Annotation structure",
        "-" * 58,
        f"Images with annotation           : "
        f"{coverage['images_with_at_least_one_annotation']}",
        f"Total annotation regions         : "
        f"{coverage['total_annotation_regions']}",
        f"Valid annotation regions         : "
        f"{coverage['valid_annotation_regions']}",
        f"Valid TI-RADS ROI regions        : "
        f"{coverage['valid_tirads_roi_regions']}",
        f"Valid TI-RADS ROI cases          : "
        f"{coverage['valid_tirads_roi_cases']}",
        f"Invalid audit records            : "
        f"{len(invalid_df)}",
        "",
        "Previous split leakage",
        "-" * 58,
        f"Old train/test shared cases      : "
        f"{leakage['old_train_test_leaking_case_count']}",
        f"Phase 9 train/validation shared  : "
        f"{leakage['phase9_train_validation_leaking_case_count']}",
        "",
        "Research decision",
        "-" * 58,
        (
            "The previous image-level experiments are retained as the "
            "reproduction history. The final proposed-method experiments "
            "must use a fixed case-level split. Valid freehand annotations "
            "will be used to create nodule-centred ROI inputs."
        ),
        "",
        "Next authorised step",
        "-" * 58,
        (
            "Create one leakage-free case-level train/validation/test "
            "manifest. Do not train a new model until that manifest and "
            "the ROI preview have been reviewed."
        ),
    ]

    AUDIT_SUMMARY_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    """Run the complete Phase 11 audit."""

    print("=" * 78)
    print("PHASE 11 : CASE AND ANNOTATION AUDIT")
    print("=" * 78)

    require_paths(
        [
            RAW_ROOT,
            MASTER_DATASET_PATH,
            FINAL_DATASET_PATH,
        ]
    )

    final_df = pd.read_csv(
        FINAL_DATASET_PATH
    )

    (
        case_image_df,
        annotation_df,
        invalid_df,
        roi_df,
    ) = build_case_and_annotation_audit()

    distribution = save_case_distribution(
        case_image_df,
        roi_df,
    )

    coverage = save_annotation_coverage_report(
        case_image_df,
        annotation_df,
        invalid_df,
        roi_df,
    )

    leakage = audit_previous_splits(
        final_df,
    )

    create_roi_preview_grid(
        roi_df,
    )

    save_final_summary(
        distribution,
        coverage,
        leakage,
        invalid_df,
    )

    print("\n" + "=" * 78)
    print("PHASE 11 AUDIT COMPLETE")
    print("=" * 78)

    print("\nDataset")
    print("-" * 78)
    print(
        "Unique cases with images        : "
        f"{distribution['unique_cases_with_images']}"
    )
    print(
        "Cases with valid TI-RADS        : "
        f"{distribution['cases_with_valid_tirads']}"
    )
    print(
        "Cases with multiple images      : "
        f"{distribution['cases_with_multiple_images']}"
    )

    print("\nAnnotations")
    print("-" * 78)
    print(
        "Total annotation regions        : "
        f"{coverage['total_annotation_regions']}"
    )
    print(
        "Valid annotation regions        : "
        f"{coverage['valid_annotation_regions']}"
    )
    print(
        "Valid TI-RADS ROI regions       : "
        f"{coverage['valid_tirads_roi_regions']}"
    )
    print(
        "Valid TI-RADS ROI cases         : "
        f"{coverage['valid_tirads_roi_cases']}"
    )

    print("\nPrevious split leakage")
    print("-" * 78)
    print(
        "Old train/test shared cases     : "
        f"{leakage['old_train_test_leaking_case_count']}"
    )
    print(
        "Phase 9 train/validation shared : "
        f"{leakage['phase9_train_validation_leaking_case_count']}"
    )

    print("\nGenerated files")
    print("-" * 78)

    for output_path in [
        CASE_IMAGE_MAPPING_PATH,
        CASE_DISTRIBUTION_PATH,
        ANNOTATION_INVENTORY_PATH,
        ANNOTATION_COVERAGE_REPORT_PATH,
        PREVIOUS_SPLIT_LEAKAGE_REPORT_PATH,
        INVALID_ANNOTATION_RECORDS_PATH,
        ROI_CANDIDATE_MANIFEST_PATH,
        ROI_PREVIEW_GRID_PATH,
        AUDIT_SUMMARY_PATH,
        PREVIOUS_SPLIT_IMAGE_ASSIGNMENTS_PATH,
        LEAKING_CASE_DETAILS_PATH,
    ]:
        print(output_path)

    print("\nImportant:")
    print("No model was trained.")
    print("No original image or XML file was modified.")
    print(
        "Review the ROI preview and audit summary before "
        "creating the final case-level split."
    )

    print("=" * 78)


if __name__ == "__main__":
    main()
