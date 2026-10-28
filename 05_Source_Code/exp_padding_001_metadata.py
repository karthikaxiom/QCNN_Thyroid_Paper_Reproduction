"""EXT-PADDING-001: training-only ROI padding-geometry integrity audit.

Phase 13 pads each expanded ROI to a square with its *median pixel intensity*
before resizing to 64x64. Thus the proportion of zero-valued pixels is NOT a
measure of the padding introduced by the preprocessing pipeline.

This script derives the exact theoretical padded area *before resizing* from
the saved expanded ROI dimensions. Actual output pixels at pad boundaries may
be mixtures due to image resampling; the calculation is geometrical only.

Inputs: Phase 13 training metadata, no images, no validation/test arrays.
Outputs: source-hashed JSON, per-class CSV and grayscale figure files.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "03_Dataset_Processed" / "phase13_roi_dataset" / "train_roi_metadata.csv"
OUT = ROOT / "07_Results" / "exp_padding_001" / "local"
LABELS = ["2", "3", "4a", "4b", "4c", "5"]


def summarize(values: np.ndarray) -> dict:
    v = np.asarray(values, dtype=float)
    return {
        "n": int(v.size),
        "median": float(np.median(v)),
        "mean": float(v.mean()),
        "min": float(v.min()),
        "max": float(v.max()),
        "over_25_percent": int(np.sum(v > 0.25)),
        "over_50_percent": int(np.sum(v > 0.50)),
        "strictly_positive": int(np.sum(v > 0.0)),
    }


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite previous report: {OUT}")
    if not SOURCE.is_file():
        raise FileNotFoundError(SOURCE)

    df = pd.read_csv(SOURCE, dtype={"roi_id": str, "case_id": str, "tirads": str})
    required = {"roi_id", "case_id", "tirads", "tirads_class", "split",
                "extracted_roi_width", "extracted_roi_height",
                "expanded_bbox_x1", "expanded_bbox_x2",
                "expanded_bbox_y1", "expanded_bbox_y2"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(df.columns))}")
    if len(df) != 311 or df["case_id"].nunique() != 203:
        raise ValueError("Expected exactly 311 training ROIs across 203 case IDs")
    if df["roi_id"].duplicated().any():
        raise ValueError("ROI ID duplicates")
    if not df["split"].eq("train").all():
        raise ValueError("This audit is restricted to the training split")
    if set(df["tirads"]) != set(LABELS):
        raise ValueError("Unexpected TI-RADS category")
    mapping = {x: k for k, x in enumerate(LABELS)}
    if not (df["tirads_class"].astype(int) == df["tirads"].map(mapping)).all():
        raise ValueError("TI-RADS mapping mismatch")
    if (df.groupby("case_id")["tirads_class"].nunique() > 1).any():
        raise ValueError("Case label contradiction")

    w = df["extracted_roi_width"].to_numpy(dtype=float)
    h = df["extracted_roi_height"].to_numpy(dtype=float)
    if not (np.isfinite(w).all() and np.isfinite(h).all()) or np.any((w <= 0) | (h <= 0)):
        raise ValueError("Nonpositive or nonfinite ROI dimensions")
    if not np.array_equal(w, (
        df["expanded_bbox_x2"] - df["expanded_bbox_x1"]).to_numpy(dtype=float)):
        raise ValueError("Expanded x bounds disagree with ROI width")
    if not np.array_equal(h, (
        df["expanded_bbox_y2"] - df["expanded_bbox_y1"]).to_numpy(dtype=float)):
        raise ValueError("Expanded y bounds disagree with ROI height")

    # Square area=max(width,height)^2. Original area=width*height.
    # Padded proportion = 1 - min(width,height)/max(width,height).
    fraction = 1.0 - np.minimum(w, h) / np.maximum(w, h)
    if (fraction < 0).any() or (fraction >= 1).any():
        raise ValueError("Invalid calculated padded proportions")

    df["estimated_padding_fraction_pre_resize"] = fraction
    class_rows = []
    for label in LABELS:
        group = df.loc[df["tirads"] == label]
        stats = summarize(group["estimated_padding_fraction_pre_resize"].to_numpy())
        class_rows.append({
            "tirads": label,
            "n_cases": int(group["case_id"].nunique()),
            "n_rois": stats["n"],
            "median_padding_fraction": stats["median"],
            "mean_padding_fraction": stats["mean"],
            "over_25_percent": stats["over_25_percent"],
            "over_50_percent": stats["over_50_percent"],
        })
    overall = summarize(fraction)
    # An independently computed repository-wide inventory found these counts.
    if (overall["n"] != 311 or overall["strictly_positive"] != 309 or
            overall["over_25_percent"] != 157 or overall["over_50_percent"] != 16 or
            not np.isclose(overall["median"], 0.256, atol=1e-12)):
        raise RuntimeError(f"Source metadata changed relative to precheck: {overall}")

    report = {
        "experiment": "EXT-PADDING-001",
        "method": "Padding area from expanded ROI width/height before 64x64 resampling",
        "formula": "1 - min(roi_width, roi_height)/max(roi_width, roi_height)",
        "padding_fill": "Per-ROI median grayscale intensity; NOT zero-valued padding",
        "input_path": SOURCE.relative_to(ROOT).as_posix(),
        "input_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "training_only": True,
        "images_loaded": False,
        "validation_or_test_accessed": False,
        "original_files_modified": False,
        "overall": overall,
        "by_tirads": class_rows,
        "limitations": [
            "This is pre-resize theoretical padding area, not a direct count of pixels.",
            "Interpolation after resizing may blend original and padded regions.",
            "Different padding proportions do not prove a cause of prediction errors.",
            "Case label grouping does not establish patient identity independence.",
        ],
    }

    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "padding_geometry_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(class_rows).to_csv(OUT / "padding_by_tirads.csv", index=False)
    df[["roi_id", "case_id", "tirads", "extracted_roi_width",
        "extracted_roi_height", "estimated_padding_fraction_pre_resize"]].to_csv(
            OUT / "training_roi_padding_geometry.csv", index=False)

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 10,
    })

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.hist(100 * fraction, bins=np.arange(0, 76, 5), edgecolor="white", color="0.35")
    ax.axvline(25, color="0.1", ls="--", lw=1.3, label="25% reference")
    ax.axvline(100 * overall["median"], color="0.6", ls="-.", lw=1.3,
               label=f'Median {100 * overall["median"]:.1f}%')
    ax.set_xlabel("Theoretical square-padding area before resizing (%)")
    ax.set_ylabel("Number of training ROIs")
    ax.set_title("Phase 13 training ROI padding geometry (n=311)")
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"padding_fraction_distribution.{ext}",
                    dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    vals = [100 * df.loc[df.tirads == label,
                         "estimated_padding_fraction_pre_resize"].to_numpy()
            for label in LABELS]
    ax.boxplot(vals, tick_labels=LABELS, showfliers=True,
               boxprops={"color": "0.3"}, medianprops={"color": "0.0"})
    ax.axhline(25, ls="--", color="0.5", lw=1.0)
    ax.set_xlabel("Assigned TI-RADS category")
    ax.set_ylabel("Theoretical square-padding area (%)")
    ax.set_title("Training ROI padding across categories")
    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"padding_fraction_by_tirads.{ext}",
                    dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("\nEXT-PADDING-001 COMPLETE")
    print("Overall:", overall)
    for rec in class_rows:
        print(f"TI-RADS {rec['tirads']:2s}: n={rec['n_rois']:3d}, "
              f"median={100*rec['median_padding_fraction']:.2f}%")
    print("Figures:", OUT)
    print("Training metadata only; no original files modified.")


if __name__ == "__main__":
    main()
