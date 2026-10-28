"""EXT-QMAP-001: matched fixed quantum-feature versus classical-feature audit.

Development study on *only* the 311 Phase13 training ROIs (203 case IDs).
Uses the same fivefold case-stratified splits and fixed multiclass logistic
classifier for both feature maps. The two arms have 64 features each:

  1) 8x8 mean-pooled grayscale image intensities (64 classical features).
  2) Four Pauli-Z expectation outputs from the existing fixed four-qubit,
     two-layer simulated quantum filter on each of sixteen 2x2 patches of
     that SAME pooled 8x8 image (16 patches x 4 channels = 64 features).

This is a comparison of *fixed feature mappings*, not of trained quantum
circuits, full QuCNet architectures, quantum computational advantage, cancer
diagnosis, or clinical utility. The original Phase9 test protocol is not
reused. No validation/test arrays are loaded. All probabilities are OOF.
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
import pennylane as qml
import sklearn
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from exp_fusion_001 import (
    ROOT, SEED, CLASSES, CLASS_NAMES, load_training_data, file_sha256, metrics,
)

OUT = ROOT / "07_Results" / "exp_qmap_001"
REFERENCE_CODE = ROOT / "05_Source_Code" / "utilities" / "quantum_utils.py"
CONFIG = {
    "n_qubits": 4,
    "layers": 2,
    "encoding": "RX(pixel_intensity_radians) on four wires",
    "ansatz": "Fixed seeded RY rotations and CNOT ring 0-1, 1-2, 2-3, 3-0 each layer",
    "measurements": "Expectation value of Pauli-Z on each of four wires",
    "quantum_weights_seed": 42,
    "classifier_seed": SEED,
    "cv_folds": 5,
    "C": 0.1,
    "class_weight": "balanced",
    "max_iter": 1500,
    "quantum_batch_size": 128,
    "bootstrap_seed": 20261028,
    "bootstrap_case_resamples": 5000,
}

# The existing utilities.quantum_utils.py seeds NumPy with 42 and samples
# (2, 4) RY angles via uniform(0, 2*pi). A local MT19937 RandomState gives
# exactly the same weight draw without modifying process-global RNG state.
WEIGHTS = np.random.RandomState(42).uniform(
    0, 2 * np.pi, size=(2, 4)).astype(float)
DEVICE = qml.device("default.qubit", wires=4, shots=None)


@qml.qnode(DEVICE)
def fixed_four_qubit_filter(batched_patches):
    for wire in range(4):
        qml.RX(batched_patches[:, wire], wires=wire)
    for layer in range(2):
        for wire in range(4):
            qml.RY(float(WEIGHTS[layer, wire]), wires=wire)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[3, 0])
    return [qml.expval(qml.PauliZ(i)) for i in range(4)]


def quantum_features(pooled: np.ndarray) -> tuple[np.ndarray, dict]:
    n = len(pooled)
    patches = (pooled.reshape(n, 4, 2, 4, 2)
               .transpose(0, 1, 3, 2, 4)
               .reshape(n * 16, 4)
               .astype(float))
    if patches.shape != (311 * 16, 4) or not np.isfinite(patches).all():
        raise ValueError("Unexpected image patch shape or nonfinite pixels")

    # Verify equivalence against the original standalone single-patch
    # PennyLane implementation before accepting batched quantum features.
    from utilities.quantum_utils import quantum_filter_fast
    probe_indices = [0, len(patches) // 2, len(patches) - 1]
    ref = np.stack([
        np.asarray(quantum_filter_fast(patches[i]), dtype=float)
        for i in probe_indices
    ])
    batched_probes = np.stack(
        [patches[i] for i in probe_indices]).astype(float)
    probe_out = np.asarray(fixed_four_qubit_filter(batched_probes), dtype=float)
    if probe_out.shape != (4, len(probe_indices)):
        raise RuntimeError(
            f"PennyLane broadcast output shape differs from expected: {probe_out.shape}")
    max_probe_error = float(np.max(np.abs(probe_out.T - ref)))
    if max_probe_error > 1e-6:
        raise RuntimeError(
            f"Batched quantum mapping does not reproduce original reference: {max_probe_error}")

    outputs = []
    step = int(CONFIG["quantum_batch_size"])
    for start in range(0, len(patches), step):
        chunk = patches[start:start + step]
        raw = np.asarray(fixed_four_qubit_filter(chunk), dtype=float)
        if raw.shape != (4, len(chunk)):
            raise RuntimeError(
                f"Unexpected simulated filter output: {raw.shape} for {len(chunk)} patches")
        outputs.append(raw.T)
        print(f"Quantum patches: {min(start + step, len(patches))}/{len(patches)}")
    quantum_map = np.concatenate(outputs, axis=0)
    if quantum_map.shape != (311 * 16, 4):
        raise RuntimeError("Incorrect quantum output shape")
    if (not np.isfinite(quantum_map).all()
            or np.max(np.abs(quantum_map)) > 1.000001):
        raise RuntimeError("Pauli-Z expectation values outside allowed range")
    return quantum_map.reshape(n, 64), {
        "n_circuit_evaluations": len(patches),
        "n_unique_patches": int(len(patches)),
        "reference_single_patch_check_max_abs_error": max_probe_error,
        "quantum_weights": WEIGHTS.tolist(),
        "quantum_feature_std_min": float(quantum_map.std(axis=0).min()),
        "quantum_feature_std_max": float(quantum_map.std(axis=0).max()),
    }


def fit_case_cv(feats, y, case_ids, cases, case_y):
    folds = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=SEED).split(cases, case_y)
    probs = np.full((len(y), 6), np.nan, dtype=float)
    fold_reports = []
    for fold_number, (ic_train, ic_held) in enumerate(folds, 1):
        trcases = cases[ic_train]
        vacases = cases[ic_held]
        tr = np.flatnonzero(np.isin(case_ids, trcases))
        held = np.flatnonzero(np.isin(case_ids, vacases))
        if (set(case_ids[tr]) & set(case_ids[held])
                or set(y[tr]) != set(CLASSES)
                or set(y[held]) != set(CLASSES)):
            raise RuntimeError("Case-fold leakage or missing classes")
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1, class_weight="balanced", max_iter=1500,
                random_state=SEED),
        )
        model.fit(feats[tr], y[tr])
        if list(model.classes_) != CLASSES:
            raise RuntimeError("Unexpected fitted class order")
        output = model.predict_proba(feats[held])
        probs[held] = output
        fold_reports.append({
            "fold": fold_number,
            "train_cases": int(len(trcases)),
            "held_cases": int(len(vacases)),
            "held_rois": int(len(held)),
            "held_roi_metrics": metrics(y[held], output.argmax(axis=1)),
        })

    if not np.isfinite(probs).all() or not np.allclose(
            probs.sum(axis=1), 1, atol=1e-6):
        raise RuntimeError("Invalid or missing OOF class probabilities")
    pred_case = np.asarray([
        int(probs[case_ids == cid].mean(axis=0).argmax())
        for cid in cases], dtype=int)
    return probs, pred_case, fold_reports


def paired_comparison(truth, classical, quantum):
    a = classical == truth
    b = quantum == truth
    classical_only = int(np.sum(a & ~b))
    quantum_only = int(np.sum(b & ~a))
    discordant = classical_only + quantum_only
    pvalue = float(binomtest(
        min(classical_only, quantum_only),
        discordant, 0.5).pvalue) if discordant else 1.0

    rng = np.random.default_rng(int(CONFIG["bootstrap_seed"]))
    indices = rng.integers(
        0, len(truth),
        size=(int(CONFIG["bootstrap_case_resamples"]), len(truth)))
    diffs = b[indices].mean(axis=1) - a[indices].mean(axis=1)
    return {
        "both_correct": int(np.sum(a & b)),
        "both_wrong": int(np.sum(~a & ~b)),
        "classical_only_correct": classical_only,
        "quantum_only_correct": quantum_only,
        "delta_accuracy_quantum_minus_classical":
            float(b.mean() - a.mean()),
        "paired_case_bootstrap_95_percentile_ci":
            np.percentile(diffs, [2.5, 97.5]).astype(float).tolist(),
        "exact_mcnemar_two_sided_p": pvalue,
    }


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite prior analysis: {OUT}")
    X, y, case_ids, roi_ids, metadata, cases, case_y = load_training_data()
    if X.shape != (311, 64, 64, 1) or len(cases) != 203:
        raise ValueError("Unexpected training-partition shape")
    pooled = X[..., 0].reshape(311, 8, 8, 8, 8).mean(axis=(2, 4))
    classical_map = pooled.reshape(311, 64).astype(float)
    quantum_map, qinfo = quantum_features(pooled)
    feature_maps = {"classical_64": classical_map, "fixed_quantum_64": quantum_map}
    for name, values in feature_maps.items():
        if values.shape != (311, 64) or not np.isfinite(values).all():
            raise ValueError(f"{name} has wrong feature shape")

    records = {}
    predictions = []
    case_pred = {}
    for name, feats in feature_maps.items():
        oof, cp, folds = fit_case_cv(feats, y, case_ids, cases, case_y)
        case_pred[name] = cp
        record = {
            "pooled_roi": metrics(y, oof.argmax(axis=1)),
            "pooled_case": metrics(case_y, cp),
            "folds": folds,
            "feature_dimension": feats.shape[1],
            "feature_variability": {
                "min_feature_std": float(feats.std(axis=0).min()),
                "median_feature_std": float(np.median(feats.std(axis=0))),
                "max_feature_std": float(feats.std(axis=0).max()),
            },
            "predicted_case_counts": np.bincount(cp, minlength=6).tolist(),
        }
        records[name] = record
        for i in range(len(y)):
            predictions.append([
                name, str(roi_ids[i]), str(case_ids[i]),
                int(y[i]), int(oof[i].argmax()), *oof[i].tolist(),
            ])
        print(f"{name:18s}: CASE accuracy="
              f"{record['pooled_case']['accuracy']:.4f}, "
              f"balanced={record['pooled_case']['balanced_accuracy']:.4f}, "
              f"macroF1={record['pooled_case']['macro_f1']:.4f}")

    paired = paired_comparison(
        case_y, case_pred["classical_64"], case_pred["fixed_quantum_64"])
    report = {
        "experiment": "EXT-QMAP-001",
        "task": "Frozen feature-map comparison, six-class TI-RADS categories",
        "protocol": "Fivefold training-only case-stratified out-of-fold logistic regression",
        "evidence_level": "Exploratory fixed mapping, not full QuCNet replication",
        "training_roi_count": len(y),
        "training_case_count": len(cases),
        "feature_dimensions": {"classical": 64, "fixed_quantum": 64},
        "filter_configuration": CONFIG,
        "quantum_backend": "PennyLane default.qubit (shots=None)",
        "quantum_filter_crosscheck": qinfo,
        "dependencies": {
            "pennylane": qml.__version__,
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "input_sha256": {
            "training_images": file_sha256(
                ROOT / "03_Dataset_Processed/phase13_roi_dataset/X_roi_train.npy"),
            "training_labels": file_sha256(
                ROOT / "03_Dataset_Processed/phase13_roi_dataset/y_roi_train.npy"),
            "training_case_ids": file_sha256(
                ROOT / "03_Dataset_Processed/phase13_roi_dataset/case_ids_roi_train.npy"),
            "training_roi_ids": file_sha256(
                ROOT / "03_Dataset_Processed/phase13_roi_dataset/roi_ids_train.npy"),
            "training_metadata": file_sha256(
                ROOT / "03_Dataset_Processed/phase13_roi_dataset/train_roi_metadata.csv"),
            "reference_quantum_utils": file_sha256(REFERENCE_CODE),
        },
        "validation_accessed": False,
        "test_accessed": False,
        "original_checkpoints_modified": False,
        "models": records,
        "paired_case_comparison": paired,
        "limitations": [
            "Feature-map experiment; no trainable quantum-circuit parameters or full QuCNet CNN.",
            "Quantum filter is a fixed random RY ansatz, not optimized by classification labels.",
            "The 8x8 image-pooling approach differs from the 64x64 original quantum map.",
            "The 64 classical intensities and 64 quantum expectations are equally sized but differ in nonlinear expressivity.",
            "The sample is small, case IDs are not confirmed patient-unique, and categories are not pathology labels.",
            "Case-bootstrap uncertainty does not include model refitting and CV-split variation.",
            "Previously inspected test predictions are not used for model selection or new evaluation.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (OUT / "oof_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "model", "roi_id", "case_id", "true_class", "predicted_class",
            "p2", "p3", "p4a", "p4b", "p4c", "p5"])
        writer.writerows(predictions)

    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    keys = ["accuracy", "balanced_accuracy", "macro_f1"]
    x = np.arange(len(feature_maps))
    for i, key in enumerate(keys):
        ax.bar(x + (i - 1) * 0.23,
               [100 * records[m]["pooled_case"][key] for m in feature_maps],
               width=0.23, color=["0.15", "0.42", "0.68"][i],
               label=key.replace("_", " "))
    ax.set_xticks(x, ["Classical: pooled intensities", "Quantum: fixed 4-qubit map"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Case-level CV metric (%)")
    ax.set_title("Training-only fixed quantum and classical feature comparison")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"fixed_feature_map_case_comparison.{ext}",
                    bbox_inches="tight", dpi=220)
    plt.close(fig)

    print("\nEXT-QMAP-001 COMPLETE")
    print("Paired cases:", json.dumps(paired, indent=2))
    print("Reference filter agreement:",
          qinfo["reference_single_patch_check_max_abs_error"])
    print("Results:", OUT)
    print("No validation/test inputs accessed or historical models modified.")


if __name__ == "__main__":
    main()
