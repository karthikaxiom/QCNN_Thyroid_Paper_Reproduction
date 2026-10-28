"""EXT-FUSION-001: paired ROI-image, annotation-geometry and fusion baselines.

Protocol: five-fold stratified cross-validation at case level, using Phase 13
training records only. The original validation/test splits are not loaded.

Prespecified comparisons:
    * Majority-class predictor (no image/geometry features)
    * 8x8 pooled grayscale and horizontal-gradient features
    * Four normalized annotation-geometry features
    * Concatenation of image and annotation-geometry features

The study is exploratory. Cross-validated measurements on the original
training partition are NOT independent test results or clinical validation.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "03_Dataset_Processed" / "phase13_roi_dataset"
RESULTS = ROOT / "07_Results" / "exp_fusion_001"
SEED = 42
CLASSES = list(range(6))
CLASS_NAMES = ["2", "3", "4a", "4b", "4c", "5"]

SOURCES = {
    "images": DATA / "X_roi_train.npy",
    "labels": DATA / "y_roi_train.npy",
    "case_ids": DATA / "case_ids_roi_train.npy",
    "roi_ids": DATA / "roi_ids_train.npy",
    "metadata": DATA / "train_roi_metadata.csv",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_training_data():
    for path in SOURCES.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    X = np.load(SOURCES["images"], allow_pickle=False).astype(np.float32)
    y = np.load(SOURCES["labels"], allow_pickle=False).astype(int).ravel()
    case_ids = np.load(SOURCES["case_ids"], allow_pickle=False).astype(str)
    roi_ids = np.load(SOURCES["roi_ids"], allow_pickle=False).astype(str)

    if X.shape != (311, 64, 64, 1):
        raise ValueError(f"Expected 311 cropped ROIs; found {X.shape}")
    if len(y) != 311 or len(case_ids) != 311 or len(roi_ids) != 311:
        raise ValueError("Image/label/case/ROI identifier lengths differ")
    if not np.isfinite(X).all() or X.min() < 0 or X.max() > 1:
        raise ValueError("Invalid normalized ultrasound image values")
    if not np.isin(y, CLASSES).all():
        raise ValueError("Unexpected class label")
    if len(set(roi_ids)) != len(roi_ids):
        raise ValueError("Duplicate ROI IDs")

    metadata = pd.read_csv(SOURCES["metadata"], dtype={"roi_id": str})
    if metadata["roi_id"].isna().any() or not metadata["roi_id"].is_unique:
        raise ValueError("Metadata ROI IDs must be unique")
    metadata = metadata.set_index("roi_id").loc[roi_ids]

    if not np.array_equal(metadata["case_id"].astype(str).to_numpy(), case_ids):
        raise ValueError("Case IDs differ between saved arrays and metadata")
    if not np.array_equal(metadata["tirads_class"].to_numpy(dtype=int), y):
        raise ValueError("ROI classes differ between saved arrays and metadata")

    unique_cases, indices = np.unique(case_ids, return_index=True)
    case_labels = y[indices]
    if len(unique_cases) != 203:
        raise ValueError(f"Expected 203 cases; found {len(unique_cases)}")
    for case, label in zip(unique_cases, case_labels):
        if not np.all(y[case_ids == case] == label):
            raise ValueError(f"Case {case} has inconsistent labels")
    return X, y, case_ids, roi_ids, metadata, unique_cases, case_labels


def image_features(X: np.ndarray) -> np.ndarray:
    pixels = X[..., 0]
    gradient = np.abs(
        np.diff(pixels, axis=2, append=pixels[:, :, -1:])
    )

    def pooled_eight(a):
        return a.reshape(-1, 8, 8, 8, 8).mean(axis=(2, 4)).reshape(len(a), -1)

    return np.column_stack((pooled_eight(pixels), pooled_eight(gradient)))


def geometry_features(metadata: pd.DataFrame) -> np.ndarray:
    W = metadata["actual_image_width"].to_numpy(dtype=float)
    H = metadata["actual_image_height"].to_numpy(dtype=float)
    x1 = metadata["original_bbox_x1"].to_numpy(dtype=float)
    x2 = metadata["original_bbox_x2"].to_numpy(dtype=float)
    y1 = metadata["original_bbox_y1"].to_numpy(dtype=float)
    y2 = metadata["original_bbox_y2"].to_numpy(dtype=float)
    width = x2 - x1
    height = y2 - y1
    if not np.all((W > 0) & (H > 0) & (width > 0) & (height > 0)):
        raise ValueError("Invalid original-image or bounding-box dimensions")

    geometry = np.column_stack((
        np.log((width * height) / (W * H)),
        np.log((width / W) / (height / H)),
        ((x1 + x2) / 2) / W,
        ((y1 + y2) / 2) / H,
    ))
    if not np.isfinite(geometry).all():
        raise ValueError("Non-finite geometry feature detected")
    return geometry


def metrics(truth: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(truth, pred)),
        "balanced_accuracy": float(recall_score(
            truth, pred, labels=CLASSES, average="macro", zero_division=0
        )),
        "macro_f1": float(f1_score(
            truth, pred, labels=CLASSES, average="macro", zero_division=0
        )),
        "confusion_matrix": confusion_matrix(
            truth, pred, labels=CLASSES
        ).tolist(),
    }


def paired_case_comparison(truth, left, right):
    left_correct = left == truth
    right_correct = right == truth
    return {
        "left_only_correct": int(np.sum(left_correct & ~right_correct)),
        "right_only_correct": int(np.sum(~left_correct & right_correct)),
        "both_correct": int(np.sum(left_correct & right_correct)),
        "both_incorrect": int(np.sum(~left_correct & ~right_correct)),
        "delta_accuracy_right_minus_left": float(
            np.mean(right_correct) - np.mean(left_correct)
        ),
    }


def main():
    if RESULTS.exists():
        raise FileExistsError(
            f"EXT-FUSION-001 results already exist; refusing to overwrite {RESULTS}"
        )
    X, y, case_ids, roi_ids, metadata, cases, case_y = load_training_data()
    img = image_features(X)
    geo = geometry_features(metadata)
    if not (img.shape == (311, 128) and geo.shape == (311, 4)):
        raise ValueError("Unexpected feature dimensions")

    inputs = {
        "majority": img,
        "image": img,
        "geometry": geo,
        "fusion": np.column_stack((img, geo)),
    }
    folds = list(StratifiedKFold(
        n_splits=5, shuffle=True, random_state=SEED
    ).split(cases, case_y))
    report = {}
    all_predictions = []
    case_predictions = {}

    for name, features in inputs.items():
        probabilities = np.full((len(y), 6), np.nan)
        fold_records = []
        for fold_number, (train_case_idx, hold_case_idx) in enumerate(folds, 1):
            train_cases = cases[train_case_idx]
            held_cases = cases[hold_case_idx]
            train_idx = np.flatnonzero(np.isin(case_ids, train_cases))
            held_idx = np.flatnonzero(np.isin(case_ids, held_cases))
            if set(case_ids[train_idx]) & set(case_ids[held_idx]):
                raise RuntimeError("CV case leakage detected")
            if set(y[train_idx]) != set(CLASSES) or set(y[held_idx]) != set(CLASSES):
                raise RuntimeError("All six TI-RADS labels must occur in each fold")

            if name == "majority":
                model = DummyClassifier(strategy="most_frequent")
            else:
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=0.1, class_weight="balanced", max_iter=1500,
                        random_state=SEED,
                    ),
                )
            model.fit(features[train_idx], y[train_idx])
            if list(model.classes_) != CLASSES:
                raise RuntimeError("Classifier has unexpected class order")
            p = model.predict_proba(features[held_idx])
            probabilities[held_idx] = p
            fold_records.append({
                "fold": fold_number,
                "train_cases": int(len(train_cases)),
                "held_cases": int(len(held_cases)),
                "held_rois": int(len(held_idx)),
                "held_roi_metrics": metrics(y[held_idx], p.argmax(axis=1)),
            })

        if not np.isfinite(probabilities).all():
            raise RuntimeError("Missing out-of-fold probabilities")
        roi_pred = probabilities.argmax(axis=1)
        case_pred = np.array([
            int(probabilities[case_ids == case].mean(axis=0).argmax())
            for case in cases
        ])
        case_predictions[name] = case_pred
        report[name] = {
            "folds": fold_records,
            "pooled_roi": metrics(y, roi_pred),
            "pooled_case": metrics(case_y, case_pred),
            "predicted_roi_counts": np.bincount(roi_pred, minlength=6).tolist(),
            "predicted_case_counts": np.bincount(case_pred, minlength=6).tolist(),
        }
        for i in range(len(y)):
            all_predictions.append([
                name, str(roi_ids[i]), str(case_ids[i]), int(y[i]),
                int(roi_pred[i]), *map(float, probabilities[i]),
            ])
        print(
            f"{name:9s} ROI accuracy={report[name]['pooled_roi']['accuracy']:.4f} "
            f"balanced={report[name]['pooled_roi']['balanced_accuracy']:.4f} "
            f"macroF1={report[name]['pooled_roi']['macro_f1']:.4f} | "
            f"CASE accuracy={report[name]['pooled_case']['accuracy']:.4f} "
            f"balanced={report[name]['pooled_case']['balanced_accuracy']:.4f} "
            f"macroF1={report[name]['pooled_case']['macro_f1']:.4f}"
        )

    pairwise = {
        "image_to_fusion": paired_case_comparison(
            case_y, case_predictions["image"], case_predictions["fusion"]
        ),
        "geometry_to_fusion": paired_case_comparison(
            case_y, case_predictions["geometry"], case_predictions["fusion"]
        ),
        "majority_to_fusion": paired_case_comparison(
            case_y, case_predictions["majority"], case_predictions["fusion"]
        ),
    }
    audit = {
        "experiment": "EXT-FUSION-001",
        "protocol": "5-fold case-stratified CV on Phase 13 training partition only",
        "exploratory": True,
        "seed": SEED,
        "n_cases": len(cases),
        "n_rois": len(y),
        "label_order": CLASS_NAMES,
        "hyperparameters_fixed_a_priori": {
            "logistic_C": 0.1,
            "logistic_class_weight": "balanced",
            "logistic_max_iter": 1500,
            "feature_scaler": "StandardScaler (fitted within CV training folds)",
        },
        "feature_dimensions": {name: f.shape[1] for name, f in inputs.items()},
        "validation_accessed": False,
        "test_accessed": False,
        "source_sha256": {key: file_sha256(p) for key, p in SOURCES.items()},
        "sklearn_version": sklearn.__version__,
        "models": report,
        "paired_case_comparisons": pairwise,
        "limitation": (
            "Cross-validated development-set results; case separation does not "
            "by itself establish independence of patient identities or clinical validity."
        ),
    }
    RESULTS.mkdir(parents=True, exist_ok=False)
    (RESULTS / "metrics.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    with (RESULTS / "oof_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "roi_id", "case_id", "true_class", "predicted_class",
            "p2", "p3", "p4a", "p4b", "p4c", "p5",
        ])
        writer.writerows(all_predictions)

    print("\nEXT-FUSION-001 COMPLETE")
    print("Paired case-level comparisons:", json.dumps(pairwise, indent=2))
    print("Results saved to:", RESULTS)
    print("Original validation and test partitions not accessed.")
    print("Frozen datasets and model checkpoints unchanged.")


if __name__ == "__main__":
    main()