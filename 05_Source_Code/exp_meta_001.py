from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "03_Dataset_Processed" / "phase13_roi_dataset"
OUT = ROOT / "07_Results" / "exp_meta_001"

paths = {
    "metadata": DATA / "train_roi_metadata.csv",
    "labels": DATA / "y_roi_train.npy",
    "case_ids": DATA / "case_ids_roi_train.npy",
    "roi_ids": DATA / "roi_ids_train.npy",
}

if OUT.exists():
    raise FileExistsError("EXT-META-001 already exists. Refusing overwrite.")

for path in paths.values():
    if not path.is_file():
        raise FileNotFoundError(path)

y = np.load(paths["labels"], allow_pickle=False).astype(int).ravel()
cases = np.load(paths["case_ids"], allow_pickle=False).astype(str)
roi_ids = np.load(paths["roi_ids"], allow_pickle=False).astype(str)

meta = pd.read_csv(paths["metadata"], dtype={"roi_id": str})
assert meta["roi_id"].is_unique

meta = meta.set_index("roi_id").loc[roi_ids]

assert len(meta) == len(y) == len(cases) == 311
assert np.array_equal(meta["case_id"].astype(str).to_numpy(), cases)
assert np.array_equal(meta["tirads_class"].to_numpy(dtype=int), y)

W = meta["actual_image_width"].to_numpy(dtype=float)
H = meta["actual_image_height"].to_numpy(dtype=float)
x1 = meta["original_bbox_x1"].to_numpy(dtype=float)
x2 = meta["original_bbox_x2"].to_numpy(dtype=float)
y1 = meta["original_bbox_y1"].to_numpy(dtype=float)
y2 = meta["original_bbox_y2"].to_numpy(dtype=float)

w = x2 - x1
h = y2 - y1

assert np.all((W > 0) & (H > 0) & (w > 0) & (h > 0))

area = (w * h) / (W * H)
aspect = (w / W) / (h / H)
center_x = ((x1 + x2) / 2) / W
center_y = ((y1 + y2) / 2) / H

features = np.column_stack([
    np.log(area),
    np.log(aspect),
    center_x,
    center_y,
])

assert np.isfinite(features).all()

case_ids, first = np.unique(cases, return_index=True)
case_labels = y[first]

assert len(case_ids) == 203

for cid, label in zip(case_ids, case_labels):
    assert np.all(y[cases == cid] == label)

def metrics(true, predicted):
    return {
        "accuracy": float(accuracy_score(true, predicted)),
        "balanced_accuracy": float(recall_score(
            true, predicted, labels=list(range(6)),
            average="macro", zero_division=0
        )),
        "macro_f1": float(f1_score(
            true, predicted, labels=list(range(6)),
            average="macro", zero_division=0
        )),
    }

oof = np.full((len(y), 6), np.nan)
fold_info = []

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (tr_cases, va_cases) in enumerate(
    cv.split(case_ids, case_labels), 1
):
    tr = np.flatnonzero(np.isin(cases, case_ids[tr_cases]))
    va = np.flatnonzero(np.isin(cases, case_ids[va_cases]))

    assert not (set(cases[tr]) & set(cases[va]))

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=1500,
            random_state=42,
        ),
    )

    model.fit(features[tr], y[tr])
    assert list(model.classes_) == list(range(6))

    p = model.predict_proba(features[va])
    oof[va] = p

    fold_info.append({
        "fold": fold,
        "held_cases": len(va_cases),
        "held_rois": len(va),
        "roi_metrics": metrics(y[va], p.argmax(axis=1)),
    })

assert np.isfinite(oof).all()

case_predictions = np.array([
    int(oof[cases == cid].mean(axis=0).argmax())
    for cid in case_ids
])

roi_result = metrics(y, oof.argmax(axis=1))
case_result = metrics(case_labels, case_predictions)

OUT.mkdir(parents=True, exist_ok=False)

report = {
    "experiment": "EXT-META-001",
    "protocol": "5-fold case-stratified training-only CV",
    "features": [
        "log_bbox_area_fraction",
        "log_relative_aspect_ratio",
        "normalized_center_x",
        "normalized_center_y",
    ],
    "train_rois": 311,
    "train_cases": 203,
    "validation_accessed": False,
    "test_accessed": False,
    "folds": fold_info,
    "roi_metrics": roi_result,
    "case_metrics": case_result,
    "input_hashes": {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    },
}

(OUT / "metrics.json").write_text(
    json.dumps(report, indent=2) + "\n",
    encoding="utf-8",
)

pd.DataFrame({
    "roi_id": roi_ids,
    "case_id": cases,
    "true_class": y,
    "predicted_class": oof.argmax(axis=1),
    **{
        f"probability_{k}": oof[:, k]
        for k in range(6)
    },
}).to_csv(OUT / "oof_predictions.csv", index=False)

print("\nEXT-META-001 COMPLETE")
print("Geometry-only ROI metrics:", roi_result)
print("Geometry-only case metrics:", case_result)
print("Saved:", OUT)
print("No validation or test data accessed.")
