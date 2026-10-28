r"""EXT-STABILITY-001: case-fold stability of existing fixed feature maps.

Controlled, exploratory follow-up. Five *declared* case-stratified
fivefold partitions. Four methods: two existing fixed 64-feature maps
(classical pooled image, PennyLane fixed four-qubit expectations)
crossed with two already published classifier families (balanced
logistic regression and 300-tree random forest).

Only the 311 frozen Phase13 TRAIN ROIs / 203 case identifiers are read.
No hyperparameter optimization, validation/test access, label changes,
trainable quantum circuits, or historical result replacement.

The first repeated seed (42) MUST reproduce all four saved
EXT-MODELS-001 case probability tables to within 1e-8 before output
is committed. The same fixed quantum filter is computed once per run
and reused for all seeds; reference-circuit probes are checked.

Run from the repository root, in its existing virtual environment:
    python .\05_Source_Code\exp_stability_001_repeated_casecv.py

Creates 07_Results/exp_stability_001 only after every evaluation and
reference check passes. Refuses overwrite.
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
from sklearn.model_selection import StratifiedKFold

from exp_fusion_001 import ROOT, CLASSES, load_training_data, file_sha256, metrics
from exp_models_001_matched_classifiers import model_for
from exp_qmap_001_fixed_feature_compare import quantum_features

OUT = ROOT / "07_Results" / "exp_stability_001"
BASELINE_DIR = ROOT / "07_Results" / "exp_models_001"
QMAP = ROOT / "07_Results" / "exp_qmap_001"
SPLIT_SEEDS = (42, 101, 202, 303, 404)
FEATURES = ("classical_64", "fixed_quantum_64")
CLASSIFIERS = ("linear_logistic", "random_forest")
CLASS_LABELS = ("2", "3", "4a", "4b", "4c", "5")
METRICS = ("accuracy", "balanced_accuracy", "macro_f1")
COLOURS = {"classical_64": "#2474BC", "fixed_quantum_64": "#8250D9"}


def hash_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def guard_input_hashes(reference):
    folder = ROOT / "03_Dataset_Processed" / "phase13_roi_dataset"
    paths = {
        "training_images": folder / "X_roi_train.npy",
        "training_labels": folder / "y_roi_train.npy",
        "training_case_ids": folder / "case_ids_roi_train.npy",
        "training_roi_ids": folder / "roi_ids_train.npy",
        "training_metadata": folder / "train_roi_metadata.csv",
    }
    for key, path in paths.items():
        if not path.is_file() or file_sha256(path) != reference[key]:
            raise RuntimeError(f"Frozen Phase13 input differs from reference: {key}")


def reference_probabilities():
    path = BASELINE_DIR / "case_predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    data = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            model = row["model"]
            if model not in {
                f"{r}__{m}" for r in FEATURES for m in CLASSIFIERS
            }:
                continue
            key = (model, row["case_id"])
            if key in data:
                raise RuntimeError(f"Duplicate archived case: {key}")
            data[key] = np.array([float(row["p" + label])
                                  for label in CLASS_LABELS])
    if len(data) != 4 * 203:
        raise RuntimeError("Missing reference case predictions")
    return data, hash_path(path)


def one_partition(seed, y, case_ids, cases, case_y):
    if len(cases) != 203 or len(y) != 311:
        raise RuntimeError("Unexpected original train sizes")
    splitter = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=int(seed))
    split = []
    case_positions = np.full(len(cases), -1, dtype=int)
    for fold, (train_cases, held_cases) in enumerate(
            splitter.split(cases, case_y), 1):
        case_positions[held_cases] = fold
        itr = np.flatnonzero(np.isin(case_ids, cases[train_cases]))
        iheld = np.flatnonzero(np.isin(case_ids, cases[held_cases]))
        if not len(iheld) or np.intersect1d(
                case_ids[itr], case_ids[iheld]).size:
            raise RuntimeError("Case overlap or empty fold")
        if set(np.unique(y[itr])) != set(CLASSES):
            raise RuntimeError("Training fold does not contain all six classes")
        split.append((fold, itr, iheld))
    if (case_positions < 0).any():
        raise RuntimeError("Unassigned case")
    return split, case_positions


def eval_arm(feature_array, model_name, split, y, case_ids, cases, case_y):
    oof = np.full((len(y), 6), np.nan, dtype=np.float64)
    for fold, train_idx, held_idx in split:
        model = model_for(model_name)
        model.fit(feature_array[train_idx], y[train_idx])
        if list(model.classes_) != CLASSES:
            raise RuntimeError(f"Invalid model class order: {model_name}")
        probabilities = model.predict_proba(feature_array[held_idx])
        if (probabilities.shape != (len(held_idx), 6)
                or not np.isfinite(probabilities).all()
                or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6)):
            raise RuntimeError("Invalid held-out probabilities")
        if np.isfinite(oof[held_idx]).any():
            raise RuntimeError("Repeated held-out ROI")
        oof[held_idx] = probabilities
    if not np.isfinite(oof).all():
        raise RuntimeError("Incomplete OOF probabilities")
    case_probs = np.stack([
        oof[case_ids == cid].mean(axis=0) for cid in cases])
    if not np.allclose(case_probs.sum(axis=1), 1, atol=1e-6):
        raise RuntimeError("Invalid aggregated case probabilities")
    pred = case_probs.argmax(axis=1)
    result = metrics(case_y, pred)
    result["correct_cases"] = int((case_y == pred).sum())
    result["predicted_case_support"] = np.bincount(
        pred, minlength=6).astype(int).tolist()
    return case_probs, result


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUT}")
    ref_metrics = json.loads(
        (QMAP / "metrics.json").read_text(encoding="utf-8"))
    if ref_metrics["experiment"] != "EXT-QMAP-001":
        raise RuntimeError("Reference experiment does not match")
    guard_input_hashes(ref_metrics["input_sha256"])
    archived, archived_sha = reference_probabilities()
    X, y, case_ids, roi_ids, _, cases, case_y = load_training_data()
    if X.shape != (311, 64, 64, 1) or len(cases) != 203:
        raise RuntimeError("Unexpected Phase13 development input")
    pooled = X[..., 0].reshape(311, 8, 8, 8, 8).mean(
        axis=(2, 4))
    classical = pooled.reshape(311, 64).astype(np.float64)
    print("Computing fixed quantum representation once for all fold seeds.")
    quantum, filter_check = quantum_features(pooled)
    features = {"classical_64": classical,
                "fixed_quantum_64": quantum}
    if quantum.shape != (311, 64) or not np.isfinite(quantum).all():
        raise RuntimeError("Quantum features invalid")

    seed_rows = []
    oof_rows = []
    individual = {}
    predictions = {}
    for seed in SPLIT_SEEDS:
        split, held_fold = one_partition(
            seed, y, case_ids, cases, case_y)
        for rep in FEATURES:
            for model_name in CLASSIFIERS:
                key = f"{rep}__{model_name}"
                case_probs, met = eval_arm(
                    features[rep], model_name, split,
                    y, case_ids, cases, case_y)
                if seed == 42:
                    for i, cid in enumerate(cases):
                        expected = archived[(key, str(cid))]
                        if not np.allclose(
                                expected, case_probs[i], rtol=0, atol=1e-8):
                            raise RuntimeError(
                                f"Seed 42 differs from frozen reference: {key}, {cid}")
                individual.setdefault(key, []).append({
                    "seed": int(seed),
                    "case_metrics": met,
                })
                seed_rows.append([
                    int(seed), key, met["correct_cases"],
                    *[float(met[x]) for x in METRICS],
                    int(np.count_nonzero(met["predicted_case_support"]))])
                predictions[(int(seed), key)] = (
                    case_probs.argmax(axis=1) == case_y)
                for i, cid in enumerate(cases):
                    oof_rows.append([
                        int(seed), key, str(cid), int(case_y[i]),
                        int(case_probs[i].argmax()), int(held_fold[i]),
                        *case_probs[i].tolist()])
                print(
                    f"seed={seed} {key:35s} "
                    f"accuracy={met['accuracy']:.4f} "
                    f"bAcc={met['balanced_accuracy']:.4f} "
                    f"F1={met['macro_f1']:.4f}")
    summary = {}
    for name, records in individual.items():
        summary[name] = {}
        for field in METRICS:
            vals = np.array([
                rec["case_metrics"][field] for rec in records])
            summary[name][field] = {
                "mean": float(vals.mean()),
                "std_across_seeds_ddof_1": float(vals.std(ddof=1)),
                "min": float(vals.min()),
                "max": float(vals.max()),
                "per_seed": vals.astype(float).tolist(),
            }
    paired = {}
    for clf in CLASSIFIERS:
        key = clf
        differences = []
        discordance = []
        for seed in SPLIT_SEEDS:
            classical_correct = predictions[
                (seed, "classical_64__" + clf)]
            quantum_correct = predictions[
                (seed, "fixed_quantum_64__" + clf)]
            differences.append(float(
                quantum_correct.mean() - classical_correct.mean()))
            discordance.append({
                "seed": int(seed),
                "classical_only_correct": int(
                    (classical_correct & ~quantum_correct).sum()),
                "quantum_only_correct": int(
                    (~classical_correct & quantum_correct).sum()),
            })
        paired[key] = {
            "quantum_minus_classical_accuracy_per_seed":
                differences,
            "mean_difference": float(np.mean(differences)),
            "all_seeds_positive": all(x > 0 for x in differences),
            "all_seeds_negative": all(x < 0 for x in differences),
            "discordance": discordance,
        }
    report = {
        "experiment": "EXT-STABILITY-001",
        "status": "Completed after seed-42 exact-reference checks",
        "scientific_role": "Case-fold assignment sensitivity (exploratory)",
        "seeds": list(SPLIT_SEEDS), "fold_count": 5,
        "training_only_cases": len(cases),
        "training_only_rois": len(y),
        "comparison": "2 fixed 64-feature maps x 2 classifiers",
        "one_quantum_feature_calculation_for_all_seeds": True,
        "quantum_reference_check": filter_check,
        "model_hyperparameters": {
            "logistic": {
                "C": 0.1, "class_weight": "balanced",
                "max_iter": 1500, "random_state": 42},
            "random_forest": {
                "n_estimators": 300, "min_samples_leaf": 2,
                "max_features": "sqrt",
                "class_weight": "balanced_subsample",
                "random_state": 42}},
        "frozen_input_sha256": ref_metrics["input_sha256"],
        "reference_case_predictions_sha256": archived_sha,
        "seed42_reference_probabilities_match": True,
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "original_test_or_validation_accessed": False,
        "original_models_or_saved_results_modified": False,
        "individual_per_seed": individual,
        "across_seed_summary": summary,
        "paired_representation_difference": paired,
        "limitations": [
            "Repeated fivefold partitions of the SAME cases, not five independent patient cohorts.",
            "Only fold assignment is varied: model RNG and quantum circuit remain fixed.",
            "Reported ranges/SDs measure development-split variability and are not confidence intervals.",
            "No multiplicity-corrected confirmatory test or external validation.",
            "Case identifiers are not independently validated as distinct patients.",
            "TI-RADS categories are radiological labels, not pathology outcomes.",
            "Quantum filter is fixed and simulated, not an end-to-end trainable quantum network.",
        ],
    }

    # Write only after validation of all seeds and the archived case results.
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (OUT / "seed_metrics.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "fold_seed", "model", "correct_cases",
            "accuracy", "balanced_accuracy", "macro_f1",
            "predicted_classes"])
        writer.writerows(seed_rows)
    with (OUT / "case_oof_predictions.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "fold_seed", "model", "case_id", "true_class",
            "predicted_class", "fold",
            "p2", "p3", "p4a", "p4b", "p4c", "p5"])
        writer.writerows(oof_rows)

    fig, ax = plt.subplots(figsize=(10.2, 4.7))
    for i, model_name in enumerate(CLASSIFIERS):
        for j, rep in enumerate(FEATURES):
            key = rep + "__" + model_name
            vals = 100 * np.array(
                summary[key]["balanced_accuracy"]["per_seed"])
            xpos = np.arange(len(SPLIT_SEEDS))
            ax.plot(
                xpos, vals, marker="o", markersize=6.5,
                linewidth=2, color=COLOURS[rep],
                linestyle="-" if model_name == "linear_logistic" else "--",
                label=rep.replace("_", " ") + " / " +
                      model_name.replace("_", " "))
    ax.set_xticks(np.arange(len(SPLIT_SEEDS)),
                  [str(seed) for seed in SPLIT_SEEDS])
    ax.set_xlabel("Case-fold random seed (all results shown)")
    ax.set_ylabel("Case-level balanced accuracy (%)")
    ax.set_title("Fold-assignment sensitivity on the same 203 cases")
    ax.grid(alpha=.23)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"case_fold_stability.{suffix}",
                    dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("\nEXT-STABILITY-001 COMPLETE")
    print("Existing seed-42 case predictions reproduced exactly.")
    print("All five seeds recorded; no best-seed selection.")
    print("No original validation/test data accessed or historical files modified.")
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
