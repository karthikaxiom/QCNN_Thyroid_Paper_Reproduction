"""EXT-MODELS-001: controlled multi-classifier comparison on identical case folds.

This is an exploratory follow-up motivated by the single-class collapse of
the historical Phase 14 CNN. It does NOT replace or repair historical scores.

Two matched 64-feature representations (classical pooled intensities and
four-qubit fixed PennyLane expectation features) are each evaluated using
the same four *predeclared* classifiers:
    linear logistic regression, RBF support-vector machine,
    random forest, distance-weighted k-nearest neighbors.

Data: 311 Phase 13 TRAIN ROIs belonging to 203 recorded case IDs.
Protocol: identical 5 case-stratified held-out folds; no validation or test
input is read, no class is removed, and no classifier is tuned on held-out
folds. Outcomes are case-level OOF predictions averaged from ROI probabilities.

SVC(probability=True) uses scikit-learn's internal probability calibration
on the *training partition only* and does not independently group cases
inside that calibration; outer held-out case separation remains intact.
The four classifiers are fixed sensitivity baselines, not a proof of
outperformance or a preregistered independent test of superiority.

Run from repo root:
    python .\05_Source_Code\exp_models_001_matched_classifiers.py
Outputs written ONLY to 07_Results/exp_models_001/ (must not exist).
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from exp_fusion_001 import (
    ROOT, SEED, CLASSES, CLASS_NAMES, load_training_data, file_sha256, metrics,
)
from exp_qmap_001_fixed_feature_compare import quantum_features

OUT = ROOT / "07_Results" / "exp_models_001"
QMAP_OUT = ROOT / "07_Results" / "exp_qmap_001"
CONFIG = {
    "protocol": "exploratory, fivefold case-stratified OOF using Phase13 TRAIN only",
    "case_fold_seed": SEED,
    "classifiers": {
        "linear_logistic": {"C": 0.1, "class_weight": "balanced", "max_iter": 1500},
        "rbf_svm": {"C": 1.0, "gamma": "scale", "class_weight": "balanced",
                    "probability": True, "random_state": SEED},
        "random_forest": {"n_estimators": 300, "min_samples_leaf": 2,
                          "max_features": "sqrt", "class_weight": "balanced_subsample",
                          "random_state": SEED, "n_jobs": -1},
        "knn_7": {"n_neighbors": 7, "weights": "distance"},
    },
    "paired_bootstrap_resamples": 5000,
    "paired_bootstrap_seed": 20261028,
    "no_hyperparameter_search": True,
}
MODELS = tuple(CONFIG["classifiers"])
FEATURES = ("classical_64", "fixed_quantum_64")
PALETTE = {"classical_64": "#2474BC", "fixed_quantum_64": "#7753C8"}
METRIC_KEYS = ("accuracy", "balanced_accuracy", "macro_f1")


def model_for(name):
    if name == "linear_logistic":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(**CONFIG["classifiers"][name], random_state=SEED),
        )
    if name == "rbf_svm":
        return make_pipeline(StandardScaler(), SVC(**CONFIG["classifiers"][name]))
    if name == "random_forest":
        return RandomForestClassifier(**CONFIG["classifiers"][name])
    if name == "knn_7":
        return make_pipeline(StandardScaler(), KNeighborsClassifier(
            **CONFIG["classifiers"][name]))
    raise ValueError(f"Unregistered model: {name}")


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_inputs_and_reference():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite previous results: {OUT}")
    ref_metrics = json.loads((QMAP_OUT / "metrics.json").read_text(encoding="utf-8"))
    ref_csv = QMAP_OUT / "oof_predictions.csv"
    if not ref_csv.is_file() or ref_metrics["experiment"] != "EXT-QMAP-001":
        raise RuntimeError("Missing frozen matched-map reference results")
    recorded = ref_metrics["input_sha256"]
    names = {
        "training_images": "X_roi_train.npy",
        "training_labels": "y_roi_train.npy",
        "training_case_ids": "case_ids_roi_train.npy",
        "training_roi_ids": "roi_ids_train.npy",
        "training_metadata": "train_roi_metadata.csv",
    }
    folder = ROOT / "03_Dataset_Processed" / "phase13_roi_dataset"
    for label, name in names.items():
        if file_sha256(folder / name) != recorded[label]:
            raise RuntimeError(f"Frozen input hash mismatch: {name}")
    return ref_metrics, ref_csv


def folds_for_cases(cases, case_y, case_ids):
    folds = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for k, (itrain, itest) in enumerate(splitter.split(cases, case_y), 1):
        tr = np.flatnonzero(np.isin(case_ids, cases[itrain]))
        te = np.flatnonzero(np.isin(case_ids, cases[itest]))
        if np.intersect1d(case_ids[tr], case_ids[te]).size:
            raise RuntimeError("Case leaked between training and held-out fold")
        if set(np.unique(tr).tolist()) & set(np.unique(te).tolist()):
            raise RuntimeError("ROI leaked")
        if set(np.unique(case_y[itrain]).tolist()) != set(CLASSES):
            raise RuntimeError("Missing class in fold's training set")
        folds.append((k, tr, te, cases[itrain], cases[itest]))
    return folds


def reference_linear_check(reference_path, roi_ids, case_ids, truths, probs_by_rep):
    found = {}
    with reference_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["model"], row["roi_id"])
            if key in found:
                raise RuntimeError("Duplicate frozen reference prediction")
            found[key] = row
    for rep in FEATURES:
        probs = probs_by_rep[rep]
        if probs.shape != (len(roi_ids), 6):
            raise RuntimeError("Reference-check input dimensions wrong")
        for i, roi_id in enumerate(roi_ids):
            row = found.get((rep, str(roi_id)))
            if (row is None or str(case_ids[i]) != row["case_id"]
                    or truths[i] != int(row["true_class"])):
                raise RuntimeError(f"Reference alignment mismatch: {rep}, {roi_id}")
            expected = np.asarray([float(row[k]) for k in
                                   ["p2", "p3", "p4a", "p4b", "p4c", "p5"]])
            if not np.allclose(expected, probs[i], rtol=0, atol=1e-8):
                raise RuntimeError(f"Frozen LR prediction mismatch: {rep}, {roi_id}")
    if len(found) != 2 * len(roi_ids):
        raise RuntimeError("Reference row count differs from expected")


def paired(truth, left_pred, right_pred):
    a = left_pred == truth
    b = right_pred == truth
    left_only = int(np.sum(a & ~b))
    right_only = int(np.sum(~a & b))
    discordant = left_only + right_only
    rng = np.random.default_rng(CONFIG["paired_bootstrap_seed"])
    indices = rng.integers(0, len(truth), size=(5000, len(truth)))
    delta = b[indices].mean(axis=1) - a[indices].mean(axis=1)
    return {
        "both_correct": int(np.sum(a & b)),
        "both_wrong": int(np.sum(~a & ~b)),
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "delta_accuracy_right_minus_left": float(b.mean() - a.mean()),
        "paired_case_bootstrap_95_percentile_ci":
            np.percentile(delta, [2.5, 97.5]).tolist(),
        "exact_mcnemar_p": float(binomtest(
            min(left_only, right_only), discordant, 0.5).pvalue)
            if discordant else 1.0,
    }


def main():
    ref_metrics, ref_csv = validate_inputs_and_reference()
    X, y, case_ids, roi_ids, _, cases, case_y = load_training_data()
    if X.shape != (311, 64, 64, 1) or len(cases) != 203:
        raise RuntimeError("Unexpected frozen development input")
    pooled = X[..., 0].reshape(len(X), 8, 8, 8, 8).mean(axis=(2, 4))
    quantum_map, qinfo = quantum_features(pooled)
    feat = {
        "classical_64": pooled.reshape(len(X), 64).astype(float),
        "fixed_quantum_64": quantum_map,
    }
    if not np.isfinite(quantum_map).all() or quantum_map.shape != (311, 64):
        raise RuntimeError("Invalid fixed quantum features")
    folds = folds_for_cases(cases, case_y, case_ids)
    results, oof_rows, case_rows, case_predictions = {}, [], [], {}
    probs_check = {}
    for rep in FEATURES:
        values = feat[rep]
        for name in MODELS:
            key = f"{rep}__{name}"
            oof = np.full((len(X), 6), np.nan, dtype=float)
            fold_info = []
            for fold, tr, te, train_cases, held_cases in folds:
                estimator = model_for(name)
                estimator.fit(values[tr], y[tr])
                if list(estimator.classes_) != CLASSES:
                    raise RuntimeError(f"Class order mismatch: {key}")
                p = estimator.predict_proba(values[te])
                if p.shape != (len(te), 6) or not np.isfinite(p).all():
                    raise RuntimeError(f"Invalid fold probabilities: {key}")
                oof[te] = p
                fold_info.append({
                    "fold": fold, "train_cases": len(train_cases),
                    "held_cases": len(held_cases), "held_rois": len(te),
                    "held_class_support":
                        np.bincount(y[te], minlength=6).tolist(),
                })
            if (not np.isfinite(oof).all()
                    or not np.allclose(oof.sum(axis=1), 1, atol=1e-6)):
                raise RuntimeError(f"Missing/invalid OOF probabilities: {key}")
            cp = np.asarray([
                int(oof[case_ids == cid].mean(axis=0).argmax())
                for cid in cases], dtype=int)
            cp_probs = np.vstack([
                oof[case_ids == cid].mean(axis=0) for cid in cases])
            metric = {
                "roi": metrics(y, oof.argmax(axis=1)),
                "case": metrics(case_y, cp),
                "predicted_case_counts": np.bincount(cp, minlength=6).tolist(),
                "true_case_counts": np.bincount(case_y, minlength=6).tolist(),
                "predicted_class_diversity": int(len(set(cp))),
                "folds": fold_info,
            }
            results[key] = metric
            case_predictions[key] = cp
            if name == "linear_logistic":
                probs_check[rep] = oof
                original = ref_metrics["models"][rep]["pooled_case"]
                for mk in METRIC_KEYS:
                    if abs(metric["case"][mk] - original[mk]) > 1e-9:
                        raise RuntimeError(
                            f"Existing matched-map LR not reproduced: {rep}/{mk}")
            for i in range(len(y)):
                oof_rows.append([
                    key, roi_ids[i], case_ids[i], int(y[i]),
                    int(oof[i].argmax()), *[float(v) for v in oof[i]]])
            for i, cid in enumerate(cases):
                case_rows.append([
                    key, str(cid), int(case_y[i]), int(cp[i]),
                    *[float(v) for v in cp_probs[i]]])
            print(f"{key:35s} accuracy={metric['case']['accuracy']:.4f} "
                  f"balanced={metric['case']['balanced_accuracy']:.4f} "
                  f"macroF1={metric['case']['macro_f1']:.4f} "
                  f"classes predicted={metric['predicted_class_diversity']}/6")

    reference_linear_check(ref_csv, roi_ids, case_ids, y, probs_check)
    comparisons = {}
    for model_name in MODELS:
        comparisons["quantum_vs_classical__" + model_name] = paired(
            case_y, case_predictions[f"classical_64__{model_name}"],
            case_predictions[f"fixed_quantum_64__{model_name}"])
    for rep in FEATURES:
        for name in MODELS[1:]:
            comparisons[f"{rep}__{name}_vs_linear_logistic"] = paired(
                case_y, case_predictions[f"{rep}__linear_logistic"],
                case_predictions[f"{rep}__{name}"])

    report = {
        "experiment": "EXT-MODELS-001",
        "protocol": CONFIG,
        "evidence_level": "Exploratory train-only out-of-fold cross-validation",
        "training_rois": len(X), "training_cases": len(cases),
        "case_class_labels": CLASS_NAMES,
        "dataset": "Phase13 frozen TRAIN only",
        "original_validation_accessed": False, "original_test_accessed": False,
        "original_models_or_results_modified": False,
        "reference_lr_roi_predictions_verified": True,
        "quantum_reference_discrepancy":
            qinfo["reference_single_patch_check_max_abs_error"],
        "input_sha256": ref_metrics["input_sha256"],
        "reference_metrics_sha256":
            file_digest(QMAP_OUT / "metrics.json"),
        "reference_predictions_sha256": file_digest(ref_csv),
        "versions": {
            "numpy": np.__version__, "sklearn": sklearn.__version__,
        },
        "models": results,
        "paired_exploratory_comparisons": comparisons,
        "limitations": [
            "Exploratory, not independent patient-unique test evaluation.",
            "Eight configurations and multiple comparisons without multiplicity correction.",
            "Model hyperparameters are fixed engineering choices, not tuned for highest score.",
            "The historical collapsed CNN is not retrained or directly matched in parameter count.",
            "The four-qubit circuit is simulated and fixed, not end-to-end trained.",
            "SVC internal probability calibration is not independently case-grouped, "
            "but uses only the outer fold's training data.",
            "The source labels are TI-RADS categories, not histopathology-confirmed malignancy.",
            "Paired bootstrap intervals omit retraining and fold-assignment uncertainty.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    cols = ["model", "roi_id", "case_id", "true_class", "predicted_class",
            "p2", "p3", "p4a", "p4b", "p4c", "p5"]
    with (OUT / "oof_predictions.csv").open(
        "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(cols)
        writer.writerows(oof_rows)
    with (OUT / "case_predictions.csv").open(
        "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["model", "case_id", "true_class", "predicted_class",
                         "p2", "p3", "p4a", "p4b", "p4c", "p5"])
        writer.writerows(case_rows)
    # Full six-class confusions, including zeros, retained in the JSON.
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12.5, 5.1), sharey=True)
    axnames = ("Accuracy", "Macro F1")
    for ax, metric in zip((a0, a1), ("accuracy", "macro_f1")):
        ys = np.arange(len(MODELS))
        for i, rep in enumerate(FEATURES):
            xs = [100 * results[f"{rep}__{name}"]["case"][metric]
                  for name in MODELS]
            yoff = ys + (i - 0.5) * 0.18
            ax.scatter(xs, yoff, s=100, label=rep.replace("_", " "),
                       color=PALETTE[rep], zorder=4)
            for val, yy in zip(xs, yoff):
                ax.annotate(f"{val:.1f}%", (val, yy),
                            xytext=(5, 0), textcoords="offset points",
                            va="center", fontsize=8)
        ax.set_xlim(0, 100)
        ax.set_yticks(ys, [name.replace("_", " ").upper() for name in MODELS])
        ax.invert_yaxis()
        ax.set_title("Case-level " + metric.replace("_", " "), fontsize=12)
        ax.set_xlabel("Percentage (absolute scale)")
        ax.grid(axis="x", alpha=.25)
    a0.legend(loc="lower right", fontsize=8)
    fig.suptitle("Matched classifier families on the same 203 training cases",
                 fontsize=14)
    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"matched_classifier_comparison.{ext}",
                    dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("\nEXT-MODELS-001 COMPLETE")
    print("Matched original logistic OOF probabilities: verified")
    print("No historical model, image, validation, or test set modified.")
    print("Results:", OUT)


if __name__ == "__main__":
    main()
