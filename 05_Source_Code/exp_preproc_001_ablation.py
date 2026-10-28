"""EXT-PREPROC-001: prespecified training-only preprocessing sensitivity study.

Compares saved Phase 13 median-filled square ROIs with (a) reflection-padded
square ROIs and (b) directly stretched rectangular crops. All variants use
the identical 203-case, fivefold stratified CV protocol, fixed grayscale
features, and fixed logistic-regression parameters of EXT-FUSION-001.

Scientific safeguards:
- The original 311 Phase 13 training ROIs are the reference inputs.
- Reconstruct every median-padded ROI from its raw source image and Phase 13
  crop metadata; abort if reconstruction drifts by > one pixel-intensity step.
- Independently reproduce the original EXT-FUSION-001 image-only OOF scores.
- No Phase 13 validation or test image arrays or historical test predictions
  are used for model selection, training, or evaluation.
- Differences are exploratory and cannot establish a causal source of errors.
- Never overwrite previous results or modify original data.

Direct resizing deliberately changes aspect ratio. It is an ablation control,
not an assertion of a clinically appropriate deployment preprocessing method.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from exp_fusion_001 import (
    ROOT, SEED, CLASSES, CLASS_NAMES, load_training_data,
    image_features, metrics, file_sha256,
)

OUT = ROOT / "07_Results" / "exp_preproc_001"
REFERENCE_PRED = ROOT / "07_Results" / "exp_fusion_001" / "oof_predictions.csv"
VARIANTS = ("median_original", "reflection_padding", "direct_stretch")
BOOTSTRAP_SEED = 20261028
BOOTSTRAP_REPS = 5000


def read_raw_training_crops(metadata: pd.DataFrame):
    """Only load raw images referenced by the 311 *training* ROI records."""
    crops = []
    file_hashes = {}
    cached = {}
    for _, row in metadata.iterrows():
        raw = Path(str(row["image_path"]))
        if not raw.is_absolute():
            raw = ROOT / raw
        if not raw.is_file():
            raise FileNotFoundError(raw)
        key = str(raw.resolve())
        if key not in cached:
            image = cv2.imread(str(raw), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Cannot decode training source: {raw}")
            cached[key] = image
            file_hashes[raw.relative_to(ROOT).as_posix()] = file_sha256(raw)
        image = cached[key]
        if (image.shape[1] != int(row["actual_image_width"]) or
                image.shape[0] != int(row["actual_image_height"])):
            raise ValueError(f"Original image dimensions changed: {raw}")
        x1 = int(row["expanded_bbox_x1"])
        x2 = int(row["expanded_bbox_x2"])
        y1 = int(row["expanded_bbox_y1"])
        y2 = int(row["expanded_bbox_y2"])
        if not (0 <= x1 < x2 <= image.shape[1] and
                0 <= y1 < y2 <= image.shape[0]):
            raise ValueError("Expanded ROI bounds invalid")
        crop = image[y1:y2, x1:x2]
        if (crop.shape != (int(row["extracted_roi_height"]),
                           int(row["extracted_roi_width"]))):
            raise ValueError("Crop shape differs from Phase 13 metadata")
        crops.append(crop)
    return crops, file_hashes


def to_64(crop: np.ndarray, variant: str) -> np.ndarray:
    h, w = crop.shape
    if min(h, w) <= 1:
        raise ValueError("ROI is too narrow for reflection padding")
    if variant == "direct_stretch":
        resized = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
        return (resized.astype(np.float32) / 255.0)[..., None]
    side = max(h, w)
    dh = side - h
    dw = side - w
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    if variant == "median_original":
        padded = cv2.copyMakeBorder(
            crop, top, bottom, left, right,
            borderType=cv2.BORDER_CONSTANT, value=int(np.median(crop)),
        )
    elif variant == "reflection_padding":
        padded = cv2.copyMakeBorder(
            crop, top, bottom, left, right, borderType=cv2.BORDER_REFLECT_101,
        )
    else:
        raise ValueError(f"Unexpected variant: {variant}")
    resized = cv2.resize(padded, (64, 64), interpolation=cv2.INTER_AREA)
    return (resized.astype(np.float32) / 255.0)[..., None]


def fit_oof(features, y, case_ids, cases, case_y):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    probabilities = np.full((len(y), 6), np.nan)
    fold_records = []
    for fold, (train_case_idx, held_case_idx) in enumerate(
            folds.split(cases, case_y), 1):
        training_cases = cases[train_case_idx]
        held_cases = cases[held_case_idx]
        train_idx = np.flatnonzero(np.isin(case_ids, training_cases))
        held_idx = np.flatnonzero(np.isin(case_ids, held_cases))
        if set(case_ids[train_idx]) & set(case_ids[held_idx]):
            raise RuntimeError("Case leakage detected")
        if set(y[train_idx]) != set(CLASSES) or set(y[held_idx]) != set(CLASSES):
            raise RuntimeError("Missing class within fold")
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1, class_weight="balanced", max_iter=1500,
                random_state=SEED,
            ),
        )
        clf.fit(features[train_idx], y[train_idx])
        if list(clf.classes_) != CLASSES:
            raise ValueError("Unexpected class ordering")
        p = clf.predict_proba(features[held_idx])
        probabilities[held_idx] = p
        fold_records.append({
            "fold": fold,
            "training_cases": len(training_cases),
            "held_cases": len(held_cases),
            "held_rois": len(held_idx),
            "held_roi_accuracy": metrics(y[held_idx], p.argmax(axis=1))["accuracy"],
        })
    if not np.isfinite(probabilities).all() or not np.allclose(
            probabilities.sum(axis=1), 1, atol=1e-6):
        raise RuntimeError("Invalid OOF predictions")
    case_predictions = np.array([
        int(probabilities[case_ids == case].mean(axis=0).argmax())
        for case in cases
    ])
    return probabilities, case_predictions, fold_records


def paired(truth, left, right, indices):
    lc = left == truth
    rc = right == truth
    left_only = int(np.sum(lc & ~rc))
    right_only = int(np.sum(~lc & rc))
    n_discordant = left_only + right_only
    p_value = (float(binomtest(min(left_only, right_only),
                              n_discordant, 0.5).pvalue)
               if n_discordant else 1.0)
    difference = rc.astype(float)[indices].mean(axis=1) - (
        lc.astype(float)[indices].mean(axis=1))
    return {
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "both_correct": int(np.sum(lc & rc)),
        "both_wrong": int(np.sum(~lc & ~rc)),
        "difference_right_minus_left": float(rc.mean() - lc.mean()),
        "case_bootstrap_95_percentile_ci":
            np.percentile(difference, [2.5, 97.5]).astype(float).tolist(),
        "exact_mcnemar_p_two_sided": p_value,
    }


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing results: {OUT}")
    X_original, y, case_ids, roi_ids, metadata, cases, case_y = (
        load_training_data()
    )
    if not REFERENCE_PRED.is_file():
        raise FileNotFoundError(
            f"Missing matched image-only CV reference: {REFERENCE_PRED}")

    crops, file_hashes = read_raw_training_crops(metadata)
    median_rebuilt = np.stack([to_64(c, "median_original") for c in crops])
    max_absolute_reconstruction_error = float(np.max(
        np.abs(median_rebuilt - X_original)))
    differing_pixels = int(np.count_nonzero(median_rebuilt != X_original))
    if max_absolute_reconstruction_error > (1 / 255 + 1e-6):
        raise RuntimeError(
            "Original Phase 13 array reconstruction differs by >1 intensity step")

    raw_variants = {
        "median_original": X_original,
        "reflection_padding": np.stack([
            to_64(c, "reflection_padding") for c in crops]),
        "direct_stretch": np.stack([
            to_64(c, "direct_stretch") for c in crops]),
    }
    reconstructed_metrics = {}
    for name, X in raw_variants.items():
        if X.shape != (311, 64, 64, 1):
            raise ValueError("Wrong generated array shape")
        if not (np.isfinite(X).all() and X.min() >= 0 and X.max() <= 1):
            raise ValueError("Invalid generated image values")
        reconstructed_metrics[name] = {
            "mean_absolute_pixel_difference_from_reference": float(
                np.mean(np.abs(X - X_original))),
            "fraction_of_pixels_changed_from_reference": float(
                np.mean(X != X_original)),
        }

    reference = pd.read_csv(
        REFERENCE_PRED, dtype={"roi_id": str, "case_id": str})
    reference = reference[reference["model"] == "image"]
    if len(reference) != 311 or reference["roi_id"].duplicated().any():
        raise ValueError("Invalid image-only reference OOF table")
    reference = reference.set_index("roi_id").loc[roi_ids]
    if (not np.array_equal(reference["case_id"].astype(str).to_numpy(), case_ids)
            or not np.array_equal(reference["true_class"].to_numpy(int), y)):
        raise ValueError("Reference OOF samples do not align")
    previous_probs = reference[
        ["p2", "p3", "p4a", "p4b", "p4c", "p5"]].to_numpy(float)

    records = {}
    case_preds = {}
    predictions = []
    for name in VARIANTS:
        feats = image_features(raw_variants[name])
        if feats.shape != (311, 128):
            raise ValueError("Feature count changed")
        probs, cp, fold_details = fit_oof(
            feats, y, case_ids, cases, case_y)
        if name == "median_original" and not np.allclose(
                probs, previous_probs, rtol=1e-6, atol=1e-7):
            raise RuntimeError(
                "Median-original OOF predictions do not replicate EXT-FUSION-001")
        case_preds[name] = cp
        record = {
            "pooled_roi": metrics(y, probs.argmax(axis=1)),
            "pooled_case": metrics(case_y, cp),
            "folds": fold_details,
            "representation_difference": reconstructed_metrics[name],
            "predicted_case_counts": np.bincount(cp, minlength=6).tolist(),
        }
        records[name] = record
        for i in range(311):
            predictions.append([
                name, str(roi_ids[i]), str(case_ids[i]), int(y[i]),
                int(probs[i].argmax()), *probs[i].astype(float).tolist(),
            ])
        print(f"{name:20s} CASE accuracy="
              f"{record['pooled_case']['accuracy']:.4f} "
              f"balanced={record['pooled_case']['balanced_accuracy']:.4f} "
              f"macroF1={record['pooled_case']['macro_f1']:.4f}")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(
        0, len(cases), size=(BOOTSTRAP_REPS, len(cases)))
    comparisons = {
        f"median_original_to_{name}": paired(
            case_y, case_preds["median_original"], case_preds[name], indices)
        for name in ("reflection_padding", "direct_stretch")
    }
    report = {
        "experiment": "EXT-PREPROC-001",
        "status": "Exploratory matched development-case preprocessing ablation",
        "variants": list(VARIANTS),
        "training_cases": len(cases),
        "training_rois": len(y),
        "case_folds": 5,
        "cv_seed": SEED,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_repetitions": BOOTSTRAP_REPS,
        "median_reference_reconstructed": True,
        "max_absolute_reconstruction_error":
            max_absolute_reconstruction_error,
        "reconstructed_pixel_value_mismatches": differing_pixels,
        "median_original_oof_matches_prior_fusion_run": True,
        "input_hashes": {
            "X_roi_train.npy": file_sha256(
                ROOT / "03_Dataset_Processed/phase13_roi_dataset/X_roi_train.npy"),
            "train_roi_metadata.csv": file_sha256(
                ROOT / "03_Dataset_Processed/phase13_roi_dataset/train_roi_metadata.csv"),
            "reference_oof_predictions.csv": file_sha256(REFERENCE_PRED),
            "training_raw_images": file_hashes,
        },
        "cv_settings": {
            "feature_representation": "8x8 mean grayscale + horizontal-gradient pooling",
            "standard_scaler": "Fitted in each training fold",
            "classifier": "LogisticRegression",
            "C": 0.1, "class_weight": "balanced", "max_iter": 1500,
            "case_prediction": "Argmax of case mean ROI class probabilities",
        },
        "versions": {"opencv": cv2.__version__, "sklearn": sklearn.__version__},
        "validation_accessed": False,
        "test_accessed": False,
        "original_files_modified": False,
        "models": records,
        "paired_case_comparisons": comparisons,
        "limitations": [
            "Training-set OOF development evidence only; no untouched test evaluation.",
            "Median-vs-reflection-versus-stretch sensitivity is not causal attribution.",
            "Stretch ablation changes aspect ratios and should not be deployed by default.",
            "Case IDs are not independently verified unique patient identities.",
            "Repeated exploratory model contrasts lack multiplicity correction.",
        ],
    }

    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (OUT / "oof_predictions.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["model", "roi_id", "case_id", "true_class", "predicted_class",
             "p2", "p3", "p4a", "p4b", "p4c", "p5"])
        writer.writerows(predictions)

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8, 4.3))
    xs = np.arange(len(VARIANTS))
    for k, label in enumerate(("accuracy", "balanced_accuracy", "macro_f1")):
        ax.bar(xs + (k - 1) * 0.23,
               [100 * records[v]["pooled_case"][label] for v in VARIANTS],
               width=0.23, color=["0.18", "0.42", "0.68"][k],
               label=label.replace("_", " "))
    ax.set_xticks(xs, ["Median fill\n(reference)", "Reflection\npadding",
                       "Direct stretch\n(sensitivity)"])
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Case-level training cross-validation | preprocessing sensitivity")
    ax.legend(ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    for fmt in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"preprocessing_case_metrics.{fmt}",
                    bbox_inches="tight", dpi=220)
    plt.close(fig)

    print("\nEXT-PREPROC-001 COMPLETE")
    print("Reconstruction max error:", max_absolute_reconstruction_error)
    print("Paired comparisons:", json.dumps(comparisons, indent=2))
    print("Saved:", OUT)
    print("No original dataset, validation/test arrays or model checkpoints modified.")


if __name__ == "__main__":
    main()
