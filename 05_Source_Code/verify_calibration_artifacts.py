"""Audit eight saved case-level classifiers and probability calibration, without training.

python 05_Source_Code/verify_calibration_artifacts.py

Uses only Python stdlib and committed prediction / metric records.
Independent recomputation of *stored evidence*, not independent training.
"""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "07_Results" / "exp_models_001"
CAL_DIR = ROOT / "07_Results" / "exp_calib_001"
COLUMNS = ("p2", "p3", "p4a", "p4b", "p4c", "p5")
CLASSES = range(6)


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def near(actual: float, expected: float, where: str, tolerance: float = 1e-10) -> None:
    ensure(math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=tolerance),
           f"{where}: actual={actual}, archived={expected}")


def csvrows(path: Path) -> list[dict[str, str]]:
    ensure(path.is_file(), f"Missing file: {path}")
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    old = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))
    cal = json.loads((CAL_DIR / "calibration_metrics.json").read_text(encoding="utf-8"))
    ensure(old["experiment"] == "EXT-MODELS-001", "Wrong classifier experiment")
    ensure(cal["experiment"] == "EXT-CALIB-001", "Wrong reliability experiment")
    ensure(old["training_cases"] == 203, "Unexpected archive population")

    saved = csvrows(MODEL_DIR / "case_predictions.csv")
    bin_table = csvrows(CAL_DIR / "calibration_bins.csv")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    truth_map: dict[str, int] = {}
    for row in saved:
        name = row["model"]
        t, predicted = int(row["true_class"]), int(row["predicted_class"])
        ensure(t in CLASSES and predicted in CLASSES, "Invalid class integer")
        p = [float(row[k]) for k in COLUMNS]
        ensure(all(math.isfinite(z) and -1e-12 <= z <= 1 + 1e-12 for z in p),
               f"Nonfinite or invalid probability: {name}/{row['case_id']}")
        near(sum(p), 1, f"Probability normalization {name}", 1e-9)
        ensure(predicted == max(CLASSES, key=lambda i: p[i]),
               f"Prediction not probability argmax: {name}/{row['case_id']}")
        case_id = row["case_id"]
        if case_id in truth_map:
            ensure(truth_map[case_id] == t, f"Inconsistent label: {case_id}")
        truth_map[case_id] = t
        grouped[name].append(row)

    ensure(len(saved) == 1624 and len(truth_map) == 203,
           "Wrong model prediction size or distinct case IDs")
    ensure(set(grouped) == set(cal["models"]) == set(old["models"]) and len(grouped) == 8,
           "Eight model identities not aligned")
    ensure(len(bin_table) == 80, "Missing calibration-bin records")
    table_by_key = {}
    for row in bin_table:
        key = (row["model"], float(row["bin_low"]))
        ensure(key not in table_by_key, f"Duplicate calibration bin {key}")
        table_by_key[key] = row

    for model, group in grouped.items():
        ensure(len(group) == 203 and {r["case_id"] for r in group} == set(truth_map),
               f"Missing or repeated cases: {model}")
        cm = [[0] * 6 for _ in CLASSES]
        brier_sum = nll_sum = entropy_sum = confidence_sum = 0.0
        high = high_wrong = 0
        by_bin: list[list[tuple[float, int]]] = [[] for _ in range(10)]
        for row in group:
            t = int(row["true_class"])
            pred = int(row["predicted_class"])
            p = [float(row[k]) for k in COLUMNS]
            cm[t][pred] += 1
            brier_sum += sum((p[k] - (1 if k == t else 0)) ** 2 for k in CLASSES)
            nll_sum -= math.log(max(p[t], 1e-15))
            entropy_sum -= sum(z * math.log(z) for z in p if z > 0)
            confidence = max(p)
            confidence_sum += confidence
            if confidence >= 0.5:
                high += 1
                if pred != t:
                    high_wrong += 1
            by_bin[min(int(confidence * 10), 9)].append((confidence, int(pred == t)))

        support = [sum(cm[k]) for k in CLASSES]
        pred_counts = [sum(cm[i][k] for i in CLASSES) for k in CLASSES]
        require_support = all(x > 0 for x in support)
        ensure(require_support, f"Missing reference class support: {model}")
        correct = sum(cm[k][k] for k in CLASSES)
        case_accuracy = correct / 203
        balanced = sum(cm[k][k] / support[k] for k in CLASSES) / 6
        macro_f1 = sum(2 * cm[k][k] / (support[k] + pred_counts[k])
                       if support[k] + pred_counts[k] else 0 for k in CLASSES) / 6
        ref = old["models"][model]["case"]
        ensure(cm == ref["confusion_matrix"], f"Confusion matrix mismatch: {model}")
        for field, val in (("accuracy", case_accuracy), ("balanced_accuracy", balanced),
                           ("macro_f1", macro_f1)):
            near(val, ref[field], f"{model} saved classifier {field}")

        record = cal["models"][model]
        expected = {
            "accuracy": case_accuracy,
            "multiclass_brier_score": brier_sum / 203,
            "mean_negative_log_likelihood": nll_sum / 203,
            "mean_predictive_entropy": entropy_sum / 203,
            "mean_max_class_probability": confidence_sum / 203,
        }
        for field, value in expected.items():
            near(value, record[field], f"{model} {field}")
        ensure(record["cases"] == 203, f"Incorrect calibration case count: {model}")
        ensure(high == record["high_confidence_ge_0_5_cases"], f"High confidence count: {model}")
        ensure(high_wrong == record["incorrect_high_confidence_ge_0_5_cases"],
               f"Incorrect high confidence count: {model}")

        ece = 0.0
        for index, entries in enumerate(by_bin):
            lower = index / 10
            csvrow = table_by_key.get((model, lower))
            ensure(csvrow is not None, f"Missing bin {index} for {model}")
            b = record["bins"][index]
            near(b["range"][0], lower, f"JSON bin start: {model}/{index}")
            near(b["range"][1], (index + 1) / 10, f"JSON bin end: {model}/{index}")
            near(float(csvrow["bin_high"]), (index + 1) / 10,
                 f"CSV bin end: {model}/{index}")
            ensure(int(csvrow["n"]) == len(entries) == b["n"],
                   f"Bin support mismatch: {model}/{index}")
            if not entries:
                ensure(all(b[k] is None for k in
                           ("avg_confidence", "accuracy", "abs_calibration_gap")),
                       f"Nonempty stats for empty bin: {model}/{index}")
                ensure(all(csvrow[k] == "" for k in
                           ("mean_confidence", "accuracy", "abs_calibration_gap")),
                       f"Nonempty CSV empty-bin stats: {model}/{index}")
                continue
            avg_conf = sum(x[0] for x in entries) / len(entries)
            bin_acc = sum(x[1] for x in entries) / len(entries)
            gap = abs(avg_conf - bin_acc)
            ece += gap * len(entries) / 203
            for field, value in (("avg_confidence", avg_conf),
                                 ("accuracy", bin_acc),
                                 ("abs_calibration_gap", gap)):
                near(value, b[field], f"JSON {model}/{index}/{field}")
            for field, value in (("mean_confidence", avg_conf), ("accuracy", bin_acc),
                                 ("abs_calibration_gap", gap)):
                near(value, csvrow[field], f"CSV {model}/{index}/{field}", 5.1e-9)
        near(ece, record["expected_calibration_error_10_equal_width_bins"],
             f"ECE {model}")
    print("PASS: 1,624 saved case predictions, eight six-class confusion matrices,")
    print("      Brier, NLL, entropy, confidence/error counts, ECE and 80 bins.")
    print("Scope: independent stored-data recomputation only, not model retraining,")
    print("       pathology validation or patient-identity verification.")


if __name__ == "__main__":
    main()
