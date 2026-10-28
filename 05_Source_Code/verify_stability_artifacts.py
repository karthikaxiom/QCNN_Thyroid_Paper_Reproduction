"""Read-only, stdlib-only verification of saved EXT-STABILITY-001 evidence.

Run from any directory: python 05_Source_Code/verify_stability_artifacts.py
No raw DDTI images, ML packages, retraining or networking are needed.
This independently recomputes stored metrics, not the underlying model fits.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "07_Results" / "exp_models_001" / "case_predictions.csv"
OUT = ROOT / "07_Results" / "exp_stability_001"
SEEDS = (42, 101, 202, 303, 404)
MODELS = tuple(f"{rep}__{clf}"
               for rep in ("classical_64", "fixed_quantum_64")
               for clf in ("linear_logistic", "random_forest"))
CLASSES = tuple(range(6))
PROBS = ("p2", "p3", "p4a", "p4b", "p4c", "p5")
FIELDS = ("accuracy", "balanced_accuracy", "macro_f1")
TOL = 1e-10


def require(ok: bool, detail: str) -> None:
    if not ok:
        raise AssertionError(detail)


def rows(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"Missing evidence: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def near(observed: float, expected: float, label: str, tol=TOL) -> None:
    require(math.isclose(float(observed), float(expected), abs_tol=tol, rel_tol=0),
            f"{label}: calculated {observed}, archived {expected}")


def row_probs(row: dict[str, str], label: str) -> list[float]:
    p = [float(row[k]) for k in PROBS]
    require(all(math.isfinite(x) and -1e-12 <= x <= 1 + 1e-12 for x in p),
            f"Invalid probabilities: {label}")
    near(sum(p), 1.0, f"Probability sum {label}", tol=1e-9)
    require(int(row["predicted_class"]) == max(range(6), key=p.__getitem__),
            f"Predicted label is not probability argmax: {label}")
    require(int(row["true_class"]) in CLASSES, f"Invalid true class: {label}")
    return p


def get_scores(group: list[dict[str, str]], label: str) -> tuple[dict, list[list[int]]]:
    cm = [[0] * 6 for _ in CLASSES]
    for row in group:
        cm[int(row["true_class"])][int(row["predicted_class"])] += 1
    require(sum(map(sum, cm)) == 203, f"Wrong case count: {label}")
    correct = sum(cm[k][k] for k in CLASSES)
    support = [sum(cm[k]) for k in CLASSES]
    predicted = [sum(cm[i][k] for i in CLASSES) for k in CLASSES]
    require(all(x > 0 for x in support), f"Missing true label support: {label}")
    metrics = {
        "accuracy": correct / 203,
        "balanced_accuracy": sum(cm[k][k] / support[k] for k in CLASSES) / 6,
        "macro_f1": sum((2 * cm[k][k] / (support[k] + predicted[k]))
                        if (support[k] + predicted[k]) else 0 for k in CLASSES) / 6,
        "correct_cases": correct,
        "predicted_classes": sum(x > 0 for x in predicted),
    }
    return metrics, cm


def verify() -> None:
    report_path = OUT / "metrics.json"
    require(report_path.is_file(), "Missing stability metrics JSON")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    require(report["experiment"] == "EXT-STABILITY-001", "Wrong experiment")
    require(report["seeds"] == list(SEEDS), "Wrong or missing CV seeds")
    require(report["fold_count"] == 5 and report["training_only_cases"] == 203,
            "Unexpected CV design")
    require(report["training_only_rois"] == 311, "Unexpected source ROI count")
    require(report["seed42_reference_probabilities_match"] is True,
            "Archived seed 42 check not affirmed")
    # The original checksum was recorded on Windows. Git may check out a
    # text CSV with LF rather than CRLF bytes on Linux. Accept only the saved
    # exact hash or its *line-ending-only* CRLF equivalent; every seed-42
    # numerical probability and case ID is also compared below.
    raw_ref = REF.read_bytes()
    canonical_lf = raw_ref.replace(b"\r\n", b"\n")
    possible_hashes = {
        hashlib.sha256(raw_ref).hexdigest(),
        hashlib.sha256(canonical_lf.replace(b"\n", b"\r\n")).hexdigest(),
    }
    require(report["reference_case_predictions_sha256"] in possible_hashes,
            "Frozen reference differs beyond permitted LF/CRLF normalization")

    baseline = rows(REF)
    require(len(baseline) == 8 * 203, "Expected eight archived 203-case classifier arms")
    reference = {}
    truths = {}
    for row in baseline:
        key = (row["model"], row["case_id"])
        require(key not in reference, f"Duplicate baseline reference: {key}")
        reference[key] = row_probs(row, str(key))
        t = int(row["true_class"])
        require(row["case_id"] not in truths or truths[row["case_id"]] == t,
                f"Inconsistent baseline case label: {row['case_id']}")
        truths[row["case_id"]] = t
    require(len(truths) == 203, "Baseline does not contain 203 unique case IDs")

    oof = rows(OUT / "case_oof_predictions.csv")
    saved = rows(OUT / "seed_metrics.csv")
    require(len(oof) == len(SEEDS) * len(MODELS) * 203,
            "Unexpected number of OOF records")
    require(len(saved) == len(SEEDS) * len(MODELS),
            "Unexpected number of saved metrics rows")
    index = {}
    for row in saved:
        key = (int(row["fold_seed"]), row["model"])
        require(key not in index, f"Duplicate saved metrics group: {key}")
        index[key] = row

    groups = defaultdict(list)
    folds = {}
    for row in oof:
        seed, model, case_id = int(row["fold_seed"]), row["model"], row["case_id"]
        require(seed in SEEDS and model in MODELS, "Unexpected seed or model")
        require(case_id in truths and int(row["true_class"]) == truths[case_id],
                f"Unknown case or inconsistent case label: {case_id}")
        fold = int(row["fold"])
        require(1 <= fold <= 5, f"Invalid case fold: {seed}, {case_id}")
        fold_key = (seed, case_id)
        require(fold_key not in folds or folds[fold_key] == fold,
                f"Case appears in two held-out folds: {fold_key}")
        folds[fold_key] = fold
        probs = row_probs(row, f"{seed}/{model}/{case_id}")
        if seed == 42:
            expected = reference.get((model, case_id))
            require(expected is not None, f"Missing frozen case: {model}/{case_id}")
            for a, b in zip(probs, expected):
                near(a, b, f"Seed 42 frozen probability: {model}/{case_id}", tol=1e-8)
        groups[(seed, model)].append(row)

    expected_keys = {(s, m) for s in SEEDS for m in MODELS}
    require(set(groups) == expected_keys and set(index) == expected_keys,
            "Missing/extra model-by-seed group")
    require(len(folds) == 203 * 5, "Incomplete case-to-fold assignments")
    scores = {}
    for key, group in groups.items():
        seed, model = key
        require(len(group) == 203 and {x["case_id"] for x in group} == set(truths),
                f"Missing or repeated OOF case: {key}")
        metric, cm = get_scores(group, str(key))
        row = index[key]
        for field in FIELDS:
            near(metric[field], float(row[field]), f"{key} CSV {field}")
        require(metric["correct_cases"] == int(row["correct_cases"]),
                f"{key} wrong correct count")
        require(metric["predicted_classes"] == int(row["predicted_classes"]),
                f"{key} wrong predicted class count")
        j = next((x["case_metrics"] for x in report["individual_per_seed"][model]
                  if x["seed"] == seed), None)
        require(j is not None and cm == j["confusion_matrix"],
                f"{key} JSON confusion mismatch")
        for field in FIELDS:
            near(metric[field], j[field], f"{key} JSON {field}")
        scores[key] = metric

    for model in MODELS:
        for field in FIELDS:
            v = [scores[(seed, model)][field] for seed in SEEDS]
            s = report["across_seed_summary"][model][field]
            for idx, value in enumerate(v):
                near(value, s["per_seed"][idx], f"{model} {field} seed index {idx}")
            near(mean(v), s["mean"], f"{model} {field} mean")
            near(stdev(v), s["std_across_seeds_ddof_1"], f"{model} {field} sample SD")
            near(min(v), s["min"], f"{model} {field} min")
            near(max(v), s["max"], f"{model} {field} max")

    for clf in ("linear_logistic", "random_forest"):
        diffs = [scores[(s, "fixed_quantum_64__" + clf)]["accuracy"]
                 - scores[(s, "classical_64__" + clf)]["accuracy"] for s in SEEDS]
        recorded = report["paired_representation_difference"][clf]
        for i, d in enumerate(diffs):
            near(d, recorded["quantum_minus_classical_accuracy_per_seed"][i],
                 f"Paired {clf} seed {SEEDS[i]}")
        near(mean(diffs), recorded["mean_difference"], f"Paired {clf} mean")
        require(all(d > 0 for d in diffs) == recorded["all_seeds_positive"],
                f"Incorrect positivity assertion: {clf}")
        require(all(d < 0 for d in diffs) == recorded["all_seeds_negative"],
                f"Incorrect negativity assertion: {clf}")

    print("PASS: 4,060 stability case rows, 20 confusion matrices and metric sets,")
    print("      all five-seed summaries, paired deltas, and 812 frozen seed-42 vectors.")
    print("Scope: stored artifacts only; NOT an independent training replication,")
    print("      patient-identity audit, or clinical validation.")


if __name__ == "__main__":
    verify()
