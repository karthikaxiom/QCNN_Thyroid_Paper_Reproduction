"""EXT-INTEGRITY-001: Recompute the historical prediction metrics independently.

Read-only inputs: saved Phase 9, Phase 14, and Phase 16 prediction tables
and their corresponding historical metrics. No model inference is performed,
no experiment is retrained, and no existing result is modified.

Historical 'test' filenames retain their original names, but this verification
is not a new untouched independent test or a new model evaluation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "07_Results"
OUTPUT = RESULTS / "exp_integrity_001"
SCOPES = [
    ("phase14_roi", "phase14_roi_categorical_baseline/test_roi_predictions.csv",
     "phase14_roi_categorical_baseline/evaluation_metrics.json", "true_class", "predicted_class", 6, "roi_level"),
    ("phase14_case", "phase14_roi_categorical_baseline/test_case_predictions.csv",
     "phase14_roi_categorical_baseline/evaluation_metrics.json", "true_class", "predicted_class", 6, "case_level"),
    ("phase16_binary_proxy", "phase16_binary_qucnet_evaluation/binary_test_predictions.csv",
     "phase16_binary_qucnet_evaluation/binary_evaluation_metrics.json", "true_class", "predicted_class", 2, None),
    ("phase9_quantum_sixclass", "phase9_paper_tirads_class_weighted/test_predictions.csv",
     "phase9_paper_tirads_class_weighted/test_metrics.json", "true_class_id", "predicted_class_id", 6, None),
]


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(b)
    return digest.hexdigest()


def confusion(true, pred, k):
    result = np.zeros((k, k), dtype=int)
    np.add.at(result, (true, pred), 1)
    return result


def calculate(cm):
    k = len(cm)
    totals = cm.sum(axis=1)
    positive_predictions = cm.sum(axis=0)
    tp = np.diag(cm).astype(float)
    recall = np.divide(tp, totals, out=np.zeros(k), where=totals > 0)
    precision = np.divide(tp, positive_predictions, out=np.zeros(k),
                          where=positive_predictions > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(k), where=precision + recall > 0)
    return {
        "total": int(cm.sum()),
        "correct": int(tp.sum()),
        "accuracy": float(tp.sum() / cm.sum()),
        "balanced_accuracy": float(np.mean(recall)),
        "macro_precision": float(np.mean(precision)),
        "macro_f1": float(np.mean(f1)),
        "class_support": totals.astype(int).tolist(),
        "predicted_class_counts": positive_predictions.astype(int).tolist(),
        "confusion_matrix": cm.tolist(),
    }


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit: {OUTPUT}")
    records = {}
    all_verified = True
    inputs = {}
    for id_, csv_file, json_file, true_key, pred_key, k, section in SCOPES:
        source = RESULTS / csv_file
        summary = RESULTS / json_file
        if not source.is_file() or not summary.is_file():
            raise FileNotFoundError(f"Missing historical source: {source} or {summary}")
        frame = pd.read_csv(source)
        y = frame[true_key].to_numpy(dtype=int)
        p = frame[pred_key].to_numpy(dtype=int)
        if len(y) == 0 or np.any((y < 0) | (y >= k) | (p < 0) | (p >= k)):
            raise ValueError(f"Invalid labels: {id_}")
        cm = confusion(y, p, k)
        actual = calculate(cm)
        original = json.loads(summary.read_text(encoding="utf-8"))
        documented = original[section] if section else original
        checks = {}
        for field in ("accuracy", "balanced_accuracy", "macro_f1"):
            if field in documented:
                checks[field] = bool(np.isclose(
                    actual[field], float(documented[field]), rtol=0, atol=1e-10))
        if id_ == "phase16_binary_proxy":
            counts = documented["confusion_matrix"]
            checks["true_negative"] = int(cm[0, 0]) == int(counts["true_negative"])
            checks["false_positive"] = int(cm[0, 1]) == int(counts["false_positive"])
            checks["false_negative"] = int(cm[1, 0]) == int(counts["false_negative"])
            checks["true_positive"] = int(cm[1, 1]) == int(counts["true_positive"])
            tp, tn, fp, fn = cm[1, 1], cm[0, 0], cm[0, 1], cm[1, 0]
            checks["sensitivity"] = bool(np.isclose(
                tp / (tp + fn), documented["sensitivity_recall_malignant"], atol=1e-10))
            checks["specificity"] = bool(np.isclose(
                tn / (tn + fp), documented["specificity_benign"], atol=1e-10))
            checks["positive_f1"] = bool(np.isclose(
                2 * tp / (2 * tp + fp + fn), documented["f1_score_malignant"], atol=1e-10))
        elif "macro_precision" in documented:
            checks["macro_precision"] = bool(np.isclose(
                actual["macro_precision"], documented["macro_precision"], atol=1e-10))
        # Sanity check against independent sklearn computations.
        checks["sklearn_accuracy"] = bool(np.isclose(actual["accuracy"], accuracy_score(y, p), atol=1e-10))
        checks["sklearn_balanced"] = bool(np.isclose(actual["balanced_accuracy"],
            balanced_accuracy_score(y, p), atol=1e-10))
        checks["sklearn_macro_f1"] = bool(np.isclose(actual["macro_f1"],
            f1_score(y, p, labels=list(range(k)), average="macro", zero_division=0), atol=1e-10))
        records[id_] = {
            "prediction_file": str(source.relative_to(ROOT)),
            "metric_file": str(summary.relative_to(ROOT)),
            "measured": actual,
            "checks": checks,
            "pass": all(checks.values()),
        }
        all_verified &= records[id_]["pass"]
        for path in (source, summary):
            inputs[str(path.relative_to(ROOT))] = file_sha256(path)
        print(f"{id_:24s} {'PASS' if records[id_]['pass'] else 'FAIL'} "
              f"correct={actual['correct']}/{actual['total']} "
              f"accuracy={actual['accuracy']:.6f}")
    if not all_verified:
        raise RuntimeError("Metric discrepancy discovered; inspect reports before writing audit")
    OUTPUT.mkdir(parents=True, exist_ok=False)
    (OUTPUT / "historical_metrics_audit.json").write_text(
        json.dumps({
            "experiment": "EXT-INTEGRITY-001",
            "scope": "Read-only re-derivation of historical stored prediction metrics",
            "original_results_modified": False,
            "historical_test_status": "Previously analyzed; not an untouched holdout for extension selection",
            "input_sha256": inputs,
            "verified_all": True,
            "evaluations": records,
        }, indent=2) + "\n", encoding="utf-8")
    print("EXT-INTEGRITY-001 COMPLETE")
    print("Saved:", OUTPUT / "historical_metrics_audit.json")
    print("No original result files modified.")


if __name__ == "__main__":
    main()
