from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SOURCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIR.parent

PHASE11_DIR = PROJECT_ROOT / "07_Results" / "phase11_case_annotation_audit"
ROI_MANIFEST_PATH = PHASE11_DIR / "roi_candidate_manifest.csv"

RESULT_DIR = PROJECT_ROOT / "07_Results" / "phase12_case_level_split"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

CASE_LEVEL_MANIFEST_PATH = RESULT_DIR / "case_level_manifest.csv"
TRAIN_CASES_PATH = RESULT_DIR / "train_cases.csv"
VALIDATION_CASES_PATH = RESULT_DIR / "validation_cases.csv"
TEST_CASES_PATH = RESULT_DIR / "test_cases.csv"
TRAIN_ROI_MANIFEST_PATH = RESULT_DIR / "train_roi_manifest.csv"
VALIDATION_ROI_MANIFEST_PATH = RESULT_DIR / "validation_roi_manifest.csv"
TEST_ROI_MANIFEST_PATH = RESULT_DIR / "test_roi_manifest.csv"
SPLIT_DISTRIBUTION_PATH = RESULT_DIR / "split_distribution.json"
SPLIT_AUDIT_REPORT_PATH = RESULT_DIR / "split_audit_report.txt"
SPLIT_CASE_ASSIGNMENTS_PATH = RESULT_DIR / "split_case_assignments.csv"

RANDOM_SEED = 42
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15

EXPECTED_TIRADS_ORDER = ["2", "3", "4a", "4b", "4c", "5"]
TIRADS_TO_CLASS = {"2": 0, "3": 1, "4a": 2, "4b": 3, "4c": 4, "5": 5}


def normalize_case_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return text[:-2] if text.endswith(".0") else text


def normalize_tirads(value: Any) -> str:
    return "" if pd.isna(value) else str(value).strip().lower()


def verify_configuration() -> None:
    total = TRAIN_FRACTION + VALIDATION_FRACTION + TEST_FRACTION
    if not math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Split fractions must total 1.0.")
    if not ROI_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Phase 11 ROI manifest was not found:\n{ROI_MANIFEST_PATH}"
        )


def load_roi_manifest() -> pd.DataFrame:
    roi_df = pd.read_csv(ROI_MANIFEST_PATH)

    required = {
        "roi_id",
        "case_id",
        "image_name",
        "image_path",
        "tirads",
        "tirads_class",
        "roi_valid",
    }
    missing = required - set(roi_df.columns)
    if missing:
        raise ValueError(
            "Missing ROI-manifest columns: " + ", ".join(sorted(missing))
        )

    roi_df["case_id"] = roi_df["case_id"].map(normalize_case_id)
    roi_df["tirads"] = roi_df["tirads"].map(normalize_tirads)

    valid_flag = (
        roi_df["roi_valid"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )

    roi_df = roi_df[
        valid_flag
        & roi_df["tirads"].isin(EXPECTED_TIRADS_ORDER)
        & roi_df["case_id"].ne("")
    ].copy()

    roi_df["tirads_class"] = (
        roi_df["tirads"].map(TIRADS_TO_CLASS).astype(int)
    )

    if roi_df.empty:
        raise ValueError("No valid TI-RADS ROI records remain.")

    duplicates = roi_df["roi_id"].astype(str).duplicated()
    if duplicates.any():
        ids = roi_df.loc[duplicates, "roi_id"].astype(str).tolist()
        raise ValueError("Duplicate roi_id values: " + ", ".join(ids[:20]))

    return roi_df


def build_case_manifest(roi_df: pd.DataFrame) -> pd.DataFrame:
    label_counts = roi_df.groupby("case_id")["tirads"].nunique()
    conflicts = label_counts[label_counts > 1].index.astype(str).tolist()
    if conflicts:
        raise ValueError(
            "Cases with conflicting TI-RADS labels: " + ", ".join(conflicts)
        )

    case_df = (
        roi_df.groupby("case_id", as_index=False)
        .agg(
            tirads=("tirads", "first"),
            tirads_class=("tirads_class", "first"),
            roi_count=("roi_id", "nunique"),
            image_count=("image_name", "nunique"),
            first_image_name=("image_name", "first"),
        )
    )

    case_df["case_id"] = case_df["case_id"].map(normalize_case_id)
    case_df = case_df.sort_values(
        ["tirads_class", "case_id"], kind="stable"
    ).reset_index(drop=True)
    return case_df


def create_split(case_df: pd.DataFrame) -> pd.DataFrame:
    indices = np.arange(len(case_df))
    labels = case_df["tirads_class"].to_numpy()

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=VALIDATION_FRACTION + TEST_FRACTION,
        random_state=RANDOM_SEED,
        stratify=labels,
        shuffle=True,
    )

    temp_labels = labels[temp_idx]
    validation_share = VALIDATION_FRACTION / (
        VALIDATION_FRACTION + TEST_FRACTION
    )

    validation_idx, test_idx = train_test_split(
        temp_idx,
        train_size=validation_share,
        random_state=RANDOM_SEED,
        stratify=temp_labels,
        shuffle=True,
    )

    split_df = case_df.copy()
    split_df["split"] = ""
    split_df.loc[train_idx, "split"] = "train"
    split_df.loc[validation_idx, "split"] = "validation"
    split_df.loc[test_idx, "split"] = "test"

    if split_df["split"].eq("").any():
        raise RuntimeError("At least one case was not assigned.")

    return split_df


def assign_rois(
    roi_df: pd.DataFrame,
    split_case_df: pd.DataFrame,
) -> pd.DataFrame:
    assigned = roi_df.merge(
        split_case_df[["case_id", "split"]],
        on="case_id",
        how="left",
        validate="many_to_one",
    )
    if assigned["split"].isna().any():
        missing_cases = (
            assigned.loc[assigned["split"].isna(), "case_id"]
            .astype(str)
            .unique()
            .tolist()
        )
        raise RuntimeError(
            "Unassigned ROI cases: " + ", ".join(missing_cases)
        )
    if len(assigned) != len(roi_df):
        raise RuntimeError("ROI row count changed during assignment.")
    return assigned


def sorted_overlap(first: set[str], second: set[str]) -> list[str]:
    return sorted(
        first & second,
        key=lambda x: (0, int(x)) if x.isdigit() else (1, x),
    )


def audit_split(
    case_df: pd.DataFrame,
    roi_df: pd.DataFrame,
) -> dict[str, Any]:
    sets = {
        name: set(
            case_df.loc[case_df["split"] == name, "case_id"].astype(str)
        )
        for name in ["train", "validation", "test"]
    }

    overlaps = {
        "train_validation": sorted_overlap(
            sets["train"], sets["validation"]
        ),
        "train_test": sorted_overlap(sets["train"], sets["test"]),
        "validation_test": sorted_overlap(
            sets["validation"], sets["test"]
        ),
    }

    if any(overlaps.values()):
        raise RuntimeError("Case leakage was detected.")

    union = sets["train"] | sets["validation"] | sets["test"]
    if len(union) != len(case_df):
        raise RuntimeError("Case assignments are incomplete.")

    duplicated_rois = int(
        (roi_df["roi_id"].astype(str).value_counts() > 1).sum()
    )
    if duplicated_rois:
        raise RuntimeError("At least one ROI was assigned more than once.")

    class_presence = {}
    for name in ["train", "validation", "test"]:
        present = set(
            case_df.loc[case_df["split"] == name, "tirads"].astype(str)
        )
        class_presence[name] = {
            "present": [
                label for label in EXPECTED_TIRADS_ORDER if label in present
            ],
            "missing": [
                label for label in EXPECTED_TIRADS_ORDER if label not in present
            ],
        }

    return {
        "total_cases": int(len(case_df)),
        "total_rois": int(len(roi_df)),
        "train_cases": int(len(sets["train"])),
        "validation_cases": int(len(sets["validation"])),
        "test_cases": int(len(sets["test"])),
        "train_validation_overlap_count": len(
            overlaps["train_validation"]
        ),
        "train_test_overlap_count": len(overlaps["train_test"]),
        "validation_test_overlap_count": len(
            overlaps["validation_test"]
        ),
        "duplicated_roi_assignments": duplicated_rois,
        "class_presence": class_presence,
    }


def build_distribution(
    dataframe: pd.DataFrame,
    unit_name: str,
) -> dict[str, Any]:
    result = {}
    for name in ["train", "validation", "test"]:
        subset = dataframe[dataframe["split"] == name]
        counts = (
            subset["tirads"]
            .astype(str)
            .value_counts()
            .reindex(EXPECTED_TIRADS_ORDER, fill_value=0)
            .astype(int)
            .to_dict()
        )
        total = int(len(subset))
        result[name] = {
            f"{unit_name}_count": total,
            "class_counts": counts,
            "class_percentages": {
                label: (
                    round(100.0 * count / total, 4) if total else 0.0
                )
                for label, count in counts.items()
            },
        }
    return result


def save_outputs(
    case_df: pd.DataFrame,
    roi_df: pd.DataFrame,
    audit: dict[str, Any],
) -> dict[str, Any]:
    case_df = case_df.sort_values(
        ["split", "tirads_class", "case_id"], kind="stable"
    ).reset_index(drop=True)

    roi_df = roi_df.sort_values(
        ["split", "tirads_class", "case_id", "image_name", "roi_id"],
        kind="stable",
    ).reset_index(drop=True)

    case_df.to_csv(CASE_LEVEL_MANIFEST_PATH, index=False)
    case_df.to_csv(SPLIT_CASE_ASSIGNMENTS_PATH, index=False)

    case_paths = {
        "train": TRAIN_CASES_PATH,
        "validation": VALIDATION_CASES_PATH,
        "test": TEST_CASES_PATH,
    }
    roi_paths = {
        "train": TRAIN_ROI_MANIFEST_PATH,
        "validation": VALIDATION_ROI_MANIFEST_PATH,
        "test": TEST_ROI_MANIFEST_PATH,
    }

    for name, path in case_paths.items():
        case_df[case_df["split"] == name].to_csv(path, index=False)

    for name, path in roi_paths.items():
        roi_df[roi_df["split"] == name].to_csv(path, index=False)

    case_distribution = build_distribution(case_df, "case")
    roi_distribution = build_distribution(roi_df, "roi")

    payload = {
        "random_seed": RANDOM_SEED,
        "requested_fractions": {
            "train": TRAIN_FRACTION,
            "validation": VALIDATION_FRACTION,
            "test": TEST_FRACTION,
        },
        "case_distribution": case_distribution,
        "roi_distribution": roi_distribution,
        "audit": audit,
    }

    SPLIT_DISTRIBUTION_PATH.write_text(
        json.dumps(payload, indent=4) + "\n",
        encoding="utf-8",
    )

    lines = [
        "PHASE 12 - CASE-LEVEL SPLIT AUDIT REPORT",
        "=" * 62,
        "",
        "Configuration",
        "-" * 62,
        f"Random seed                    : {RANDOM_SEED}",
        f"Requested train fraction       : {TRAIN_FRACTION:.2f}",
        f"Requested validation fraction  : {VALIDATION_FRACTION:.2f}",
        f"Requested test fraction        : {TEST_FRACTION:.2f}",
        "",
        "Final case counts",
        "-" * 62,
        f"Total cases                    : {audit['total_cases']}",
        f"Train cases                    : {audit['train_cases']}",
        f"Validation cases               : {audit['validation_cases']}",
        f"Test cases                     : {audit['test_cases']}",
        "",
        "Final ROI counts",
        "-" * 62,
        f"Total ROI records              : {audit['total_rois']}",
        f"Train ROI records              : {roi_distribution['train']['roi_count']}",
        f"Validation ROI records         : {roi_distribution['validation']['roi_count']}",
        f"Test ROI records               : {roi_distribution['test']['roi_count']}",
        "",
        "Leakage checks",
        "-" * 62,
        f"Train/validation overlap       : {audit['train_validation_overlap_count']}",
        f"Train/test overlap             : {audit['train_test_overlap_count']}",
        f"Validation/test overlap        : {audit['validation_test_overlap_count']}",
        f"Duplicated ROI assignments     : {audit['duplicated_roi_assignments']}",
        "",
        "Case class distributions",
        "-" * 62,
    ]

    for name in ["train", "validation", "test"]:
        counts = case_distribution[name]["class_counts"]
        lines.append(
            f"{name.capitalize()}: "
            + ", ".join(
                f"{label}={counts[label]}"
                for label in EXPECTED_TIRADS_ORDER
            )
        )

    lines.extend(
        [
            "",
            "Class-presence checks",
            "-" * 62,
        ]
    )

    for name in ["train", "validation", "test"]:
        missing = audit["class_presence"][name]["missing"]
        lines.append(
            f"{name.capitalize()} missing classes: "
            + (", ".join(missing) if missing else "None")
        )

    lines.extend(
        [
            "",
            "Decision",
            "-" * 62,
            (
                "The split is valid only when all overlap counts and "
                "duplicated ROI assignments are zero."
            ),
            (
                "Reuse this exact case-level split for all final ROI "
                "categorical, ordinal and optional quantum experiments."
            ),
            "No model was trained and no source data was modified.",
        ]
    )

    SPLIT_AUDIT_REPORT_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    return payload


def main() -> None:
    print("=" * 78)
    print("PHASE 12 : CREATE LEAKAGE-FREE CASE-LEVEL SPLIT")
    print("=" * 78)

    verify_configuration()

    print("\nLoading Phase 11 ROI manifest...")
    roi_df = load_roi_manifest()
    print(f"Valid ROI records loaded : {len(roi_df)}")
    print(f"Unique ROI cases loaded : {roi_df['case_id'].nunique()}")

    print("\nBuilding case manifest...")
    case_df = build_case_manifest(roi_df)
    print(f"Eligible cases           : {len(case_df)}")

    print("\nCreating deterministic stratified case split...")
    split_case_df = create_split(case_df)

    print("\nAssigning ROI records by case...")
    assigned_roi_df = assign_rois(roi_df, split_case_df)

    print("\nRunning leakage checks...")
    audit = audit_split(split_case_df, assigned_roi_df)
    distribution = save_outputs(
        split_case_df,
        assigned_roi_df,
        audit,
    )

    case_dist = distribution["case_distribution"]
    roi_dist = distribution["roi_distribution"]

    print("\n" + "=" * 78)
    print("PHASE 12 SPLIT COMPLETE")
    print("=" * 78)

    print("\nCase counts")
    print("-" * 78)
    print(f"Train      : {case_dist['train']['case_count']}")
    print(
        f"Validation : {case_dist['validation']['case_count']}"
    )
    print(f"Test       : {case_dist['test']['case_count']}")
    print(f"Total      : {audit['total_cases']}")

    print("\nROI counts")
    print("-" * 78)
    print(f"Train      : {roi_dist['train']['roi_count']}")
    print(
        f"Validation : {roi_dist['validation']['roi_count']}"
    )
    print(f"Test       : {roi_dist['test']['roi_count']}")
    print(f"Total      : {audit['total_rois']}")

    print("\nLeakage checks")
    print("-" * 78)
    print(
        "Train/validation overlap : "
        f"{audit['train_validation_overlap_count']}"
    )
    print(
        "Train/test overlap       : "
        f"{audit['train_test_overlap_count']}"
    )
    print(
        "Validation/test overlap  : "
        f"{audit['validation_test_overlap_count']}"
    )
    print(
        "Duplicated ROI records   : "
        f"{audit['duplicated_roi_assignments']}"
    )

    print("\nCase class distributions")
    print("-" * 78)
    for name in ["train", "validation", "test"]:
        counts = case_dist[name]["class_counts"]
        print(
            f"{name.capitalize():10s}: "
            + ", ".join(
                f"{label}={counts[label]}"
                for label in EXPECTED_TIRADS_ORDER
            )
        )

    print("\nGenerated files")
    print("-" * 78)
    for path in [
        CASE_LEVEL_MANIFEST_PATH,
        TRAIN_CASES_PATH,
        VALIDATION_CASES_PATH,
        TEST_CASES_PATH,
        TRAIN_ROI_MANIFEST_PATH,
        VALIDATION_ROI_MANIFEST_PATH,
        TEST_ROI_MANIFEST_PATH,
        SPLIT_DISTRIBUTION_PATH,
        SPLIT_AUDIT_REPORT_PATH,
        SPLIT_CASE_ASSIGNMENTS_PATH,
    ]:
        print(path)

    print("\nImportant:")
    print("No model was trained.")
    print("No dataset or Phase 11 result was modified.")
    print("Freeze and reuse this exact split for all final experiments.")
    print("=" * 78)


if __name__ == "__main__":
    main()