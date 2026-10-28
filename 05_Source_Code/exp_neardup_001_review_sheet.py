"""Generate a LOCAL-ONLY visual review sheet for three near-duplicate candidates.

The candidates were selected by a fixed automated screening threshold. This
script visualizes original source images, not case-level model predictions.
Never modify or redistribute images without confirming dataset permissions.
Do not commit the review contact sheet to a public branch without verifying
there are no identifying/burned-in annotations.

The output is a human-review aid, not an automatic duplicate determination.
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "07_Results" / "exp_neardup_001" / "candidate_pairs.csv"
OUT = ROOT / "07_Results" / "exp_neardup_001" / "review_local"
EXPECTED = [
    ("277_1.jpg", "348_1.jpg"),
    ("177_1.jpg", "313_1.jpg"),
    ("305_1.jpg", "34_1.jpg"),
]


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite earlier review sheets: {OUT}")
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)
    with SOURCE.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    selected = [r for r in rows if r["cross_assigned_split_candidate"] == "True"]
    if len(selected) != 3 or {
            (Path(x["image_a"]).name, Path(x["image_b"]).name)
            for x in selected} != set(EXPECTED):
        raise ValueError("Cross-split candidates differ from the frozen screening audit")
    selected.sort(key=lambda r: EXPECTED.index(
        (Path(r["image_a"]).name, Path(r["image_b"]).name)))
    fig, axes = plt.subplots(3, 3, figsize=(12, 11))
    fig.suptitle(
        "Raw ultrasound near-duplicate candidate review — NOT confirmed leakage",
        fontsize=13, weight="bold",
    )
    for i, entry in enumerate(selected):
        images = []
        for side in ("a", "b"):
            file = ROOT / entry["image_" + side]
            if not file.is_file():
                raise FileNotFoundError(file)
            gray = cv2.imread(str(file), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise ValueError(f"Unreadable raw ultrasound: {file}")
            images.append(gray)
            ax = axes[i, 0 if side == "a" else 1]
            ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
            ax.set_title(f"{file.name} | {entry['split_' + side]}", fontsize=11)
            ax.axis("off")
        aa = cv2.resize(images[0], (192, 192), interpolation=cv2.INTER_AREA)
        bb = cv2.resize(images[1], (192, 192), interpolation=cv2.INTER_AREA)
        dif = cv2.absdiff(aa, bb)
        ax = axes[i, 2]
        ax.imshow(dif, cmap="gray", vmin=0, vmax=255)
        ax.set_title(
            f"Absolute difference (resized)\npHash {entry['phash_hamming']} · "
            f"dHash {entry['dhash_hamming']} · corr {float(entry['normalized_pixel_correlation']):.3f}",
            fontsize=10,
        )
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, .96))
    OUT.mkdir(parents=True, exist_ok=False)
    path = OUT / "cross_split_candidate_review.png"
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)
    print("EXT-NEARDUP-001 REVIEW SHEET COMPLETE")
    print("Local image:", path)
    print("Review the pairs visually; screening alone does not verify duplicates.")
    print("Do NOT commit this review PNG without checking for identifying marks.")


if __name__ == "__main__":
    main()
