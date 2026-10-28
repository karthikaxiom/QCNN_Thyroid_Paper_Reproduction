"""EXT-NEARDUP-001: independent perceptual near-duplicate candidate audit.

Audit only; no training, test prediction reading, model selection, image
redistribution, dataset or split modification. Uses all original raw image
files (including test-case files) solely for dataset-integrity verification.

Candidate pairs are flagged by pHash and dHash Hamming-distance thresholds
and an optional normalized-pixel correlation check. A perceptual-hash match
is NOT proof of duplicate acquisition, identical patient, or split leakage.
Re-encoded, rotated, shifted or differently cropped images can be missed.

Predefined screening thresholds:
 - pHash distance <= 8 AND dHash distance <= 12, OR
 - pHash distance <= 4 with normalized-pixel correlation >= 0.95.
No experiment scores are tuned using the outcome of this screening.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "02_Dataset_Raw" / "archive"
SPLITS = ROOT / "07_Results" / "phase12_case_level_split" / "split_case_assignments.csv"
OUT = ROOT / "07_Results" / "exp_neardup_001"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_SPLITS = {"train", "validation", "test"}
PHASH_THRESHOLD = 8
DHASH_THRESHOLD = 12
STRONG_PHASH_THRESHOLD = 4
CORRELATION_THRESHOLD = 0.95


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for part in iter(lambda: src.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def phash(image: np.ndarray) -> int:
    small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(small.astype(np.float32))
    low = dct[:8, :8].reshape(-1)
    values = low[1:]  # omit the image mean / DC term
    median = float(np.median(values))
    bits = low > median
    bits[0] = False
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


def dhash(image: np.ndarray) -> int:
    small = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).ravel()
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


def standardized_grayscale(image: np.ndarray) -> np.ndarray:
    image96 = cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA)
    vec = image96.astype(np.float32).ravel()
    vec = vec - vec.mean()
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-7 else np.zeros_like(vec)


def load_splits() -> dict[str, str]:
    table = {}
    with SPLITS.open(encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        for row in rows:
            case = str(row["case_id"]).strip()
            split = str(row["split"]).strip().lower()
            if not case.isdigit() or split not in ALLOWED_SPLITS or case in table:
                raise ValueError("Invalid Phase 12 case-split manifest")
            table[case] = split
    if len(table) != 291:
        raise ValueError(f"Unexpected case assignment count: {len(table)}")
    return table


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite prior audit: {OUT}")
    if not RAW_DIR.is_dir() or not SPLITS.is_file():
        raise FileNotFoundError("Raw data or Phase12 split manifest missing")
    assignments = load_splits()
    files = sorted(
        (p for p in RAW_DIR.iterdir()
         if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
        key=lambda p: p.name.lower())
    if len(files) != 480:
        raise ValueError(f"Expected 480 raw images, got {len(files)}")

    records = []
    for path in files:
        stem = path.stem.split("_")
        if len(stem) != 2 or not all(part.isdigit() for part in stem):
            raise ValueError(f"Invalid expected raw image file: {path.name}")
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None or image.ndim != 2 or min(image.shape) < 16:
            raise ValueError(f"Unreadable or unusually small image: {path.name}")
        case_id = stem[0]
        records.append({
            "image_path": path.relative_to(ROOT).as_posix(),
            "case_id": case_id,
            "split": assignments.get(case_id, "outside_strict_split"),
            "sha256": digest_file(path),
            "phash": phash(image),
            "dhash": dhash(image),
            "normalized": standardized_grayscale(image),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
        })

    candidates = []
    nearest_by_case = defaultdict(lambda: 65)
    pairs_checked = 0
    cross_assigned_split_candidates = 0
    for idx, a in enumerate(records):
        for b in records[idx + 1:]:
            if a["case_id"] == b["case_id"]:
                continue
            pairs_checked += 1
            pdist = (a["phash"] ^ b["phash"]).bit_count()
            ddist = (a["dhash"] ^ b["dhash"]).bit_count()
            nearest_by_case[a["case_id"]] = min(nearest_by_case[a["case_id"]], pdist)
            nearest_by_case[b["case_id"]] = min(nearest_by_case[b["case_id"]], pdist)

            if not (pdist <= PHASH_THRESHOLD and ddist <= DHASH_THRESHOLD):
                if pdist > STRONG_PHASH_THRESHOLD:
                    continue
            correlation = float(np.dot(a["normalized"], b["normalized"]))
            # Both-hash agreement is the main criterion; the alternative is
            # a very close pHash with strong pixel correlation.
            match = (
                (pdist <= PHASH_THRESHOLD and ddist <= DHASH_THRESHOLD)
                or (pdist <= STRONG_PHASH_THRESHOLD and correlation >= CORRELATION_THRESHOLD)
            )
            if not match:
                continue
            cross_assigned = (
                a["split"] in ALLOWED_SPLITS
                and b["split"] in ALLOWED_SPLITS
                and a["split"] != b["split"]
            )
            cross_assigned_split_candidates += int(cross_assigned)
            candidates.append({
                "image_a": a["image_path"], "case_a": a["case_id"],
                "split_a": a["split"],
                "image_b": b["image_path"], "case_b": b["case_id"],
                "split_b": b["split"],
                "exact_same_file_bytes": a["sha256"] == b["sha256"],
                "phash_hamming": pdist,
                "dhash_hamming": ddist,
                "normalized_pixel_correlation": correlation,
                "cross_assigned_split_candidate": cross_assigned,
            })

    candidates.sort(key=lambda x: (
        not x["cross_assigned_split_candidate"],
        not x["exact_same_file_bytes"],
        x["phash_hamming"], x["dhash_hamming"],
        -x["normalized_pixel_correlation"],
        x["image_a"], x["image_b"]))
    sha = {r["image_path"]: r["sha256"] for r in records}
    counts = {
        "files_scanned": len(records),
        "unique_image_sha256": len(set(sha.values())),
        "different_case_file_pairs_checked": pairs_checked,
        "candidate_different_case_pairs": len(candidates),
        "candidate_cross_assigned_split_pairs": cross_assigned_split_candidates,
        "exact_byte_identical_different_case_pairs": sum(
            x["exact_same_file_bytes"] for x in candidates),
    }
    if counts["files_scanned"] != 480 or counts["unique_image_sha256"] != 471:
        raise RuntimeError("Unexpected source dataset; stop and investigate")
    report = {
        "experiment": "EXT-NEARDUP-001",
        "purpose": "Data-provenance and split-integrity candidate screening",
        "thresholds": {
            "phash_and_dhash_max_hamming": [
                PHASH_THRESHOLD, DHASH_THRESHOLD],
            "strong_phash_max_hamming": STRONG_PHASH_THRESHOLD,
            "strong_phash_min_correlation": CORRELATION_THRESHOLD,
        },
        "input_split_manifest_sha256": digest_file(SPLITS),
        "raw_image_sha256": sha,
        "counts": counts,
        "candidates": candidates,
        "nearest_cross_case_phash_summary": {
            "min": min(nearest_by_case.values()),
            "median": float(np.median(list(nearest_by_case.values()))),
            "max": max(nearest_by_case.values()),
        },
        "test_case_images_decoded_for_data_integrity": True,
        "validation_test_predictions_accessed": False,
        "model_training_or_selection_performed": False,
        "original_files_modified": False,
        "limitations": [
            "Candidates are not confirmed duplicates; manual source-image review is needed.",
            "Neither perceptual hash establishes patient identity.",
            "Some genuine near-duplicates may be missed by orientation, compression, crop or scale.",
            "Simple image structural similarity may yield false positives.",
            "The screen includes test-assigned images for integrity only, not model tuning.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "near_duplicate_candidate_audit.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with (OUT / "candidate_pairs.csv").open("w", newline="", encoding="utf-8") as f:
        cols = [
            "image_a", "case_a", "split_a", "image_b", "case_b", "split_b",
            "exact_same_file_bytes", "phash_hamming", "dhash_hamming",
            "normalized_pixel_correlation",
            "cross_assigned_split_candidate",
        ]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(candidates)
    print("\nEXT-NEARDUP-001 COMPLETE")
    print(json.dumps(counts, indent=2))
    print("Saved:", OUT)
    print("Potential near-duplicates require human review; no original files changed.")


if __name__ == "__main__":
    main()
