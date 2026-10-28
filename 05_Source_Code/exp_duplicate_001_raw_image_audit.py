"""EXT-DUPLICATE-001: exact raw ultrasound-file duplication audit.

Scans every raw JPG/PNG file as bytes and groups *exactly identical* content
by SHA-256. This is an integrity audit, not an image-classification experiment.

The script intentionally reads all raw image bytes, including files belonging
to test-assigned case IDs, solely for duplicate detection. It does not decode
images, inspect visual content, load images as model inputs, change split
assignments, train models, or read predictions. Use the saved fixed split file
only to identify whether any exact-byte duplicate crosses assigned partitions.

Exact-byte checks cannot detect visually similar images recompressed differently.
A duplicated raw image is not proof that two case IDs represent one patient.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "02_Dataset_Raw" / "archive"
SPLITS = ROOT / "07_Results" / "phase12_case_level_split" / "split_case_assignments.csv"
OUT = ROOT / "07_Results" / "exp_duplicate_001"
FILE_PATTERN = re.compile(r"^(?P<case>\d+)_(?P<image>\d+)\.(?:jpg|jpeg|png)$", re.I)
ALLOWED_SPLITS = {"train", "validation", "test"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_assignments() -> dict[str, str]:
    if not SPLITS.is_file():
        raise FileNotFoundError(SPLITS)
    assignment = {}
    with SPLITS.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not {"case_id", "split"}.issubset(reader.fieldnames or []):
            raise ValueError("Split manifest is missing case_id or split")
        for row in reader:
            case_id = row["case_id"].strip()
            split = row["split"].strip()
            if not case_id or not case_id.isdigit() or split not in ALLOWED_SPLITS:
                raise ValueError("Invalid split-manifest row")
            if case_id in assignment:
                raise ValueError(f"Duplicate case assignment: {case_id}")
            assignment[case_id] = split
    if len(assignment) != 291:
        raise ValueError(f"Expected 291 case assignments, found {len(assignment)}")
    return assignment


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite prior audit: {OUT}")
    assignments = load_assignments()
    if not RAW.is_dir():
        raise FileNotFoundError(RAW)

    candidates = sorted(
        (p for p in RAW.iterdir()
         if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}),
        key=lambda p: p.name.lower(),
    )
    if len(candidates) != 480:
        raise ValueError(f"Expected 480 raw image files, found {len(candidates)}")

    by_digest: dict[str, list[dict]] = defaultdict(list)
    for path in candidates:
        match = FILE_PATTERN.fullmatch(path.name)
        if not match:
            raise ValueError(f"Unexpected raw image filename: {path.name}")
        case_id = match.group("case")
        split = assignments.get(case_id, "outside_strict_split")
        by_digest[sha256_file(path)].append({
            "image_path": path.relative_to(ROOT).as_posix(),
            "case_id": case_id,
            "split": split,
            "size_bytes": path.stat().st_size,
        })

    groups = []
    for digest, files in sorted(by_digest.items()):
        if len(files) < 2:
            continue
        files.sort(key=lambda d: d["image_path"])
        cases = sorted({f["case_id"] for f in files})
        assigned = sorted({f["split"] for f in files if f["split"] in ALLOWED_SPLITS})
        groups.append({
            "sha256": digest,
            "file_count": len(files),
            "case_count": len(cases),
            "cross_case": len(cases) > 1,
            "cross_assigned_split": len(assigned) > 1,
            "assigned_splits": assigned,
            "files": files,
        })

    group_counts = {
        "raw_image_files": len(candidates),
        "distinct_file_sha256": len(by_digest),
        "duplicate_groups": len(groups),
        "files_in_duplicate_groups": sum(g["file_count"] for g in groups),
        "redundant_files": sum(g["file_count"] - 1 for g in groups),
        "cross_case_duplicate_groups": sum(g["cross_case"] for g in groups),
        "cross_assigned_split_duplicate_groups": sum(
            g["cross_assigned_split"] for g in groups),
    }
    if sum(group_counts[k] != expected for k, expected in [
        ("raw_image_files", 480),
        ("distinct_file_sha256", 471),
        ("duplicate_groups", 6),
        ("cross_case_duplicate_groups", 2),
        ("cross_assigned_split_duplicate_groups", 0),
    ]):
        raise RuntimeError(f"Raw dataset differs from initial Git tree audit: {group_counts}")

    report = {
        "experiment": "EXT-DUPLICATE-001",
        "method": "SHA-256 over raw image file bytes",
        "scope": "480 recovered raw DDTI ultrasound images",
        "split_assignment_source": SPLITS.relative_to(ROOT).as_posix(),
        "split_assignment_sha256": sha256_file(SPLITS),
        "original_files_modified": False,
        "image_files_decoded_or_displayed": False,
        "raw_test_images_hashed_for_identity": True,
        "test_predictions_accessed": False,
        "model_selection_or_training_performed": False,
        "cannot_detect": [
            "Near-duplicate images with different encodings or crops",
            "Distinct files depicting the same nodule or patient",
            "False patient identity separation across case IDs",
        ],
        "counts": group_counts,
        "duplicate_groups": groups,
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "exact_raw_image_duplicates.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (OUT / "duplicate_files.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sha256", "image_path", "case_id", "split",
                         "size_bytes", "cross_case", "cross_assigned_split"])
        for group in groups:
            for item in group["files"]:
                writer.writerow([
                    group["sha256"], item["image_path"], item["case_id"],
                    item["split"], item["size_bytes"], group["cross_case"],
                    group["cross_assigned_split"],
                ])
    print("\nEXT-DUPLICATE-001 COMPLETE")
    print(json.dumps(group_counts, indent=2))
    for group in groups:
        if group["cross_case"]:
            print("CROSS-CASE GROUP:", [f["image_path"] for f in group["files"]])
    print("Saved:", OUT)
    print("Original files and frozen split were not modified.")


if __name__ == "__main__":
    main()
