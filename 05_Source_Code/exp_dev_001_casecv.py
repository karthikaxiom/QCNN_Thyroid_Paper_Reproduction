from pathlib import Path
import csv
import json
import hashlib
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "03_Dataset_Processed" / "phase13_roi_dataset"
OUT = ROOT / "07_Results" / "exp_dev_001_casecv"

paths = {
    "images": DATA / "X_roi_train.npy",
    "labels": DATA / "y_roi_train.npy",
    "cases": DATA / "case_ids_roi_train.npy",
}

if OUT.exists():
    raise FileExistsError("EXT-DEV-001 results already exist. Refusing overwrite.")

X = np.load(paths["images"], allow_pickle=False).astype(np.float32)
y = np.load(paths["labels"], allow_pickle=False).astype(int).ravel()
groups = np.load(paths["cases"], allow_pickle=False).astype(str)

assert X.shape == (311, 64, 64, 1)
assert len(y) == len(groups) == 311
assert np.isfinite(X).all()
assert X.min() >= 0 and X.max() <= 1
assert np.isin(y, range(6)).all()

case_ids, first = np.unique(groups, return_index=True)
case_y = y[first]

assert len(case_ids) == 203
for case, label in zip(case_ids, case_y):
    assert np.all(y[groups == case] == label)

# Fixed intensity and horizontal-gradient features.
pixels = X[..., 0]
grad = np.abs(np.diff(pixels, axis=2, append=pixels[:, :, -1:]))

def pool8(a):
    return a.reshape(-1, 8, 8, 8, 8).mean(axis=(2, 4)).reshape(len(a), -1)

features = np.column_stack([pool8(pixels), pool8(grad)])

def metrics(truth, predictions):
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(recall_score(
            truth, predictions, labels=list(range(6)),
            average="macro", zero_division=0
        )),
        "macro_f1": float(f1_score(
            truth, predictions, labels=list(range(6)),
            average="macro", zero_division=0
        )),
    }

splitter = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=42
)

results = {}
prediction_rows = []

for name in ["majority", "pooled_logistic"]:
    oof = np.full((311, 6), np.nan)
    fold_info = []

    for fold, (train_c, held_c) in enumerate(
        splitter.split(case_ids, case_y), start=1
    ):
        train = np.where(np.isin(groups, case_ids[train_c]))[0]
        held = np.where(np.isin(groups, case_ids[held_c]))[0]

        assert not (set(groups[train]) & set(groups[held]))
        assert set(y[train]) == set(range(6))
        assert set(y[held]) == set(range(6))

        if name == "majority":
            model = DummyClassifier(strategy="most_frequent")
        else:
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=0.1,
                    class_weight="balanced",
                    max_iter=1500,
                    random_state=42
                )
            )

        model.fit(features[train], y[train])
        assert list(model.classes_) == list(range(6))

        probs = model.predict_proba(features[held])
        oof[held] = probs

        fold_info.append({
            "fold": fold,
            "training_cases": len(train_c),
            "held_cases": len(held_c),
            "roi_metrics": metrics(y[held], probs.argmax(axis=1))
        })

    assert np.isfinite(oof).all()

    case_predictions = [
        int(oof[groups == c].mean(axis=0).argmax())
        for c in case_ids
    ]

    results[name] = {
        "roi": metrics(y, oof.argmax(axis=1)),
        "case": metrics(case_y, np.array(case_predictions)),
        "folds": fold_info
    }

    for i in range(len(y)):
        prediction_rows.append([
            name, i, groups[i], int(y[i]),
            int(oof[i].argmax()), *map(float, oof[i])
        ])

OUT.mkdir(parents=True, exist_ok=False)

record = {
    "experiment": "EXT-DEV-001",
    "protocol": "5-fold case-stratified cross-validation",
    "training_cases": 203,
    "training_rois": 311,
    "validation_accessed": False,
    "test_accessed": False,
    "seed": 42,
    "input_hashes": {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    },
    "results": results
}

(OUT / "metrics.json").write_text(
    json.dumps(record, indent=2) + "\n",
    encoding="utf-8"
)

with (OUT / "oof_predictions.csv").open(
    "w", newline="", encoding="utf-8"
) as f:
    writer = csv.writer(f)
    writer.writerow([
        "model", "roi_index", "case_id", "true_class",
        "predicted_class", "p2", "p3", "p4a", "p4b", "p4c", "p5"
    ])
    writer.writerows(prediction_rows)

print("\nEXT-DEV-001 COMPLETE")
for name, result in results.items():
    print("\n", name)
    print("ROI metrics:", result["roi"])
    print("CASE metrics:", result["case"])

print("\nSaved:", OUT)
print("Original validation/test partitions not accessed.")
