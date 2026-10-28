"""Read-only manifest check for frozen Phase 12/13 DDTI case-ID partition.

python 05_Source_Code/verify_frozen_case_splits.py

Verifies saved CSV metadata without reading raw clinical images, annotation
XML, NumPy arrays or patient identifiers. Case-ID separation does NOT prove
that different IDs represent different individuals.
"""
from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "07_Results/phase12_case_level_split/split_case_assignments.csv"
ROI_DIR = ROOT / "03_Dataset_Processed/phase13_roi_dataset"
SPLITS = {"train": (203, 311), "validation": (44, 58), "test": (44, 71)}
LABELS = {"2": 0, "3": 1, "4a": 2, "4b": 3, "4c": 4, "5": 5}


def require(ok: bool, msg: str) -> None:
    if not ok:
        raise AssertionError(msg)


def read(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"Missing frozen metadata: {path}")
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def valid_bbox(r: dict[str, str], prefix: str) -> None:
    width = int(r["actual_image_width"])
    height = int(r["actual_image_height"])
    x1, x2 = int(r[prefix + "_x1"]), int(r[prefix + "_x2"])
    y1, y2 = int(r[prefix + "_y1"]), int(r[prefix + "_y2"])
    require(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height,
            f"Invalid {prefix} bounds on {r['roi_id']}")


def main() -> None:
    assignments = read(MANIFEST)
    require(len(assignments) == 291, "Wrong number of frozen case IDs")
    cases: dict[str, dict[str, str]] = {}
    for item in assignments:
        cid, split = item["case_id"], item["split"]
        require(cid not in cases, f"Duplicate case assignment: {cid}")
        require(split in SPLITS, f"Invalid split: {cid}")
        require(item["tirads"] in LABELS and
                int(item["tirads_class"]) == LABELS[item["tirads"]],
                f"Case category/encoding mismatch: {cid}")
        require(int(item["roi_count"]) >= 1 and int(item["image_count"]) >= 1,
                f"Invalid source counts for case: {cid}")
        cases[cid] = item
    rois: set[str] = set()
    seen_case_split: dict[str, str] = {}
    counts: Counter[str] = Counter()
    by_case: Counter[str] = Counter()
    class_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for split, (expected_cases, expected_rois) in SPLITS.items():
        records = read(ROI_DIR / f"{split}_roi_metadata.csv")
        require(len(records) == expected_rois, f"Incorrect {split} ROI count")
        split_cases: set[str] = set()
        for item in records:
            cid, roi_id = item["case_id"], item["roi_id"]
            require(cid in cases and cases[cid]["split"] == split,
                    f"ROI assigned to wrong case split: {roi_id}")
            require(item["split"] == split,
                    f"ROI metadata split mismatch: {roi_id}")
            require(roi_id not in rois, f"Duplicate ROI across splits: {roi_id}")
            rois.add(roi_id)
            split_cases.add(cid)
            if cid in seen_case_split:
                require(seen_case_split[cid] == split,
                        f"Case ID overlaps partitions: {cid}")
            seen_case_split[cid] = split
            assignment = cases[cid]
            require(item["tirads"] == assignment["tirads"] and
                    item["tirads_class"] == assignment["tirads_class"],
                    f"ROI/case category mismatch: {roi_id}")
            require(int(item["output_width"]) == 64 and
                    int(item["output_height"]) == 64 and
                    int(item["output_channels"]) == 1,
                    f"ROI tensor shape metadata differs: {roi_id}")
            require(item["processing_success"].lower() == "true" and
                    not item["failure_reason"],
                    f"ROI recorded a processing failure: {roi_id}")
            valid_bbox(item, "original_bbox")
            valid_bbox(item, "expanded_bbox")
            for dimension in ("x", "y"):
                require(int(item[f"expanded_bbox_{dimension}1"]) <=
                        int(item[f"original_bbox_{dimension}1"]) and
                        int(item[f"expanded_bbox_{dimension}2"]) >=
                        int(item[f"original_bbox_{dimension}2"]),
                        f"Expanded ROI excludes original bounds: {roi_id}")
            require(int(item["extracted_roi_width"]) ==
                    int(item["expanded_bbox_x2"]) - int(item["expanded_bbox_x1"]) and
                    int(item["extracted_roi_height"]) ==
                    int(item["expanded_bbox_y2"]) - int(item["expanded_bbox_y1"]),
                    f"ROI recorded crop geometry differs: {roi_id}")
            pixels = [float(item[k]) for k in
                      ("minimum_pixel_value", "mean_pixel_value",
                       "maximum_pixel_value")]
            require(all(math.isfinite(x) and 0 <= x <= 1 for x in pixels) and
                    pixels[0] <= pixels[1] <= pixels[2],
                    f"Invalid pixel summary metadata: {roi_id}")
            counts[split] += 1
            by_case[cid] += 1
            class_counts[split][item["tirads"]] += 1
        require(len(split_cases) == expected_cases,
                f"Wrong distinct recorded cases in {split}")

    require(len(seen_case_split) == len(cases) == 291,
            "Not all frozen cases have processed ROI metadata")
    require(len(rois) == 440 and sum(counts.values()) == 440,
            "Frozen ROI inventory inconsistent")
    for cid, item in cases.items():
        require(by_case[cid] == int(item["roi_count"]),
                f"Manifest versus processed ROI count mismatch: {cid}")
    for split in SPLITS:
        require(set(class_counts[split]) == set(LABELS),
                f"Six-category support missing from split {split}")

    print("PASS: 291 case IDs, 440 ROI metadata records, no case-ID/ROI overlap.")
    print("Frozen partitions: train 203/311, validation 44/58, test 44/71.")
    print("Category mapping, ROI totals, bounding geometry and pixel summaries checked.")
    print("Scope: recorded identifiers and metadata only; no patient identity,")
    print("       source-image contents, pathology diagnosis or consent verification.")


if __name__ == "__main__":
    main()
