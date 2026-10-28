"""Verify the 21-row master quantitative results inventory from its source confusion matrices.

python 05_Source_Code/verify_master_inventory.py

Read-only and stdlib-only. Works on the repository's committed, derived
evidence without third-party libraries, original ultrasound images or model
training. This confirms traceability of archived metrics, NOT independence of
the patient population, validity of raw annotations, or model reproducibility.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INVENTORY = ROOT / "Research_Extension" / "MASTER_VERIFIED_RESULTS.csv"
PREFIX = "07_Results/"
STUDIES = {
    "EXT-FUSION-001": "07_Results/exp_fusion_001/metrics.json",
    "EXT-PREPROC-001": "07_Results/exp_preproc_001/metrics.json",
    "EXT-QMAP-001": "07_Results/exp_qmap_001/metrics.json",
    "EXT-INTEGRITY-001": "07_Results/exp_integrity_001/historical_metrics_audit.json",
    "EXT-MODELS-001": "07_Results/exp_models_001/metrics.json",
}
EXPECTED_COUNTS = {
    "EXT-FUSION-001": 4,
    "EXT-PREPROC-001": 3,
    "EXT-QMAP-001": 2,
    "EXT-INTEGRITY-001": 4,
    "EXT-MODELS-001": 8,
}


def demand(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def near(value: float, saved: float, label: str, tol: float = 1.1e-9) -> None:
    demand(math.isclose(float(value), float(saved), abs_tol=tol, rel_tol=0),
           f"{label}: calculated={value}, inventory={saved}")


def measures(cm: list[list[int]]) -> dict[str, float | int]:
    k = len(cm)
    demand(k in (2, 6) and all(len(row) == k for row in cm),
           "Expected a square two-class or six-class confusion matrix")
    demand(all(type(x) is int and x >= 0 for row in cm for x in row),
           "Invalid matrix entries")
    n = sum(map(sum, cm))
    correct = sum(cm[i][i] for i in range(k))
    support = [sum(cm[i]) for i in range(k)]
    demand(n > 0 and all(x > 0 for x in support),
           "Missing true-class support")
    predicted = [sum(cm[j][i] for j in range(k)) for i in range(k)]
    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n,
        "balanced_accuracy": sum(cm[i][i] / support[i] for i in range(k)) / k,
        "macro_f1": sum(
            2 * cm[i][i] / (support[i] + predicted[i])
            if support[i] + predicted[i] else 0.0
            for i in range(k)
        ) / k,
    }


def main() -> None:
    demand(INVENTORY.is_file(), "Missing master quantitative inventory")
    with INVENTORY.open(newline="", encoding="utf-8-sig") as handle:
        table = list(csv.DictReader(handle))
    demand(len(table) == sum(EXPECTED_COUNTS.values()), "Unexpected inventory size")

    saved = {}
    for study, source in STUDIES.items():
        path = ROOT / source
        demand(path.is_file(), f"Missing source result: {source}")
        saved[study] = json.loads(path.read_text(encoding="utf-8"))
        demand(saved[study]["experiment"] == study, "Wrong experiment ID " + study)
    keys = set()
    counts = {study: 0 for study in STUDIES}

    for row in table:
        study, model, unit = row["study"], row["model"], row["unit"]
        demand(study in STUDIES, f"Unknown study {study}")
        demand(unit in ("case", "roi", "image"), "Unknown observation unit")
        key = (study, model, unit)
        demand(key not in keys, f"Duplicate metric entry: {key}")
        keys.add(key)
        counts[study] += 1
        demand(row["source_result"] == STUDIES[study],
               f"Source path mismatch: {key}")
        o = saved[study]
        if study == "EXT-INTEGRITY-001":
            measured = o["evaluations"][model]["measured"]
            demand(o["evaluations"][model]["pass"] is True,
                   f"Historical integrity audit did not pass: {model}")
            demand(unit == ("roi" if model == "phase14_roi"
                            else "image" if model in
                            ("phase16_binary_proxy", "phase9_quantum_sixclass")
                            else "case"), "Historical unit mismatch")
        elif study == "EXT-MODELS-001":
            measured = o["models"][model]["case"]
            demand(unit == "case", "Model comparison is case-level")
        else:
            measured = o["models"][model]["pooled_case"]
            demand(unit == "case", "Development inventory is case-level")
        cm = measured["confusion_matrix"]
        check = measures(cm)
        for name in ("n", "correct"):
            demand(check[name] == int(row[name]), f"{key}: {name} mismatch")
        for name in ("accuracy", "balanced_accuracy", "macro_f1"):
            near(check[name], row[name], f"{key}: {name}")
            near(check[name], measured[name], f"{key}: original {name}", 1e-12)

        if study != "EXT-INTEGRITY-001":
            demand(check["n"] == 203, f"{key}: incorrect development population")
            demand("training-only" in row["evidence_type"],
                   f"{key}: missing exploratory development evidence qualifier")
        else:
            demand("historical" in row["evidence_type"] and
                   "not new untouched test" in row["evidence_type"],
                   f"{key}: historical test evidence must be qualified")
    demand(counts == EXPECTED_COUNTS,
           f"Unexpected per-study rows: {counts}")
    print("PASS: all 21 master metrics and 5 underlying source JSON files.")
    print("Verified accuracy, class-balanced accuracy, macro-F1, correct counts,")
    print("confusion matrices, source paths, unit, denominator and evidence level.")
    print("Scope: stored-data recomputation only; no fresh training or clinical test.")


if __name__ == "__main__":
    main()
