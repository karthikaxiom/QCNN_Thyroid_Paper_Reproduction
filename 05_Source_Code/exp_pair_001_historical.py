"""EXT-PAIR-001: paired historical Phase 9 and Phase 10 six-class evaluation.

Re-derives paired classification metrics from previously saved predictions.
This is NOT a new experiment or an independently held-out model evaluation:
the original image-level partitions can share case IDs across splits.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "07_Results"
OUT = RESULTS / "exp_pair_001"
PATHS = {
    "quantum": RESULTS / "phase9_paper_tirads_class_weighted" / "test_predictions.csv",
    "classical": RESULTS / "phase10_classical_tirads_class_weighted" / "test_predictions.csv",
}
LABELS = ["2", "3", "4a", "4b", "4c", "5"]
PROB_COLS = [f"probability_tirads_{name}" for name in LABELS]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def model_statistics(frame):
    true = frame.true_class_id.to_numpy(dtype=int)
    pred = frame.predicted_class_id.to_numpy(dtype=int)
    proba = frame[PROB_COLS].to_numpy(dtype=float)
    if not np.isfinite(proba).all() or (proba < 0).any():
        raise ValueError("Invalid probability table")
    if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Probability rows do not sum to 1")
    if not np.array_equal(np.argmax(proba, axis=1), pred):
        raise ValueError("Argmax differs from stored prediction")
    entropy = -np.sum(proba * np.log(np.clip(proba, 1e-12, 1.0)), axis=1)
    return {
        "correct": int(np.sum(pred == true)),
        "accuracy": float(accuracy_score(true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
        "macro_f1": float(f1_score(true, pred, labels=range(6),
                                   average="macro", zero_division=0)),
        "predicted_counts": np.bincount(pred, minlength=6).tolist(),
        "mean_max_softmax_probability": float(proba.max(axis=1).mean()),
        "mean_softmax_entropy_nats": float(entropy.mean()),
        "confusion_matrix": confusion_matrix(true, pred, labels=range(6)).tolist(),
    }


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite prior paired audit: {OUT}")
    frames = {}
    for name, path in PATHS.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path).sort_values("sample_index").reset_index(drop=True)
        if len(frame) != 70 or not np.array_equal(frame.sample_index.to_numpy(), np.arange(70)):
            raise ValueError(f"Unexpected sample indices for {name}")
        if not frame.true_class_id.between(0, 5).all():
            raise ValueError(f"Invalid true class for {name}")
        if not frame.predicted_class_id.between(0, 5).all():
            raise ValueError(f"Invalid predicted class for {name}")
        frames[name] = frame

    a, b = frames["quantum"], frames["classical"]
    if not a[["sample_index", "true_class_id"]].equals(
            b[["sample_index", "true_class_id"]]):
        raise ValueError("Models are not aligned on identical samples/labels")

    y = a.true_class_id.to_numpy(dtype=int)
    qa = a.predicted_class_id.to_numpy(dtype=int) == y
    cb = b.predicted_class_id.to_numpy(dtype=int) == y
    q_only = int(np.sum(qa & ~cb))
    c_only = int(np.sum(~qa & cb))
    discordant = q_only + c_only
    pvalue = float(binomtest(min(q_only, c_only), discordant, 0.5).pvalue) \
        if discordant else 1.0
    paired = {
        "both_correct": int(np.sum(qa & cb)),
        "both_incorrect": int(np.sum(~qa & ~cb)),
        "quantum_only_correct": q_only,
        "classical_only_correct": c_only,
        "discordant": discordant,
        "classical_minus_quantum_accuracy": float(cb.mean() - qa.mean()),
        "mcnemar_exact_two_sided_p": pvalue,
    }
    assert sum((paired["both_correct"], paired["both_incorrect"],
                q_only, c_only)) == 70

    models = {name: model_statistics(frame) for name, frame in frames.items()}
    if models["quantum"]["correct"] != 13 or models["classical"]["correct"] != 17:
        raise ValueError("Historical checkpoint predictions differ from frozen records")

    report = {
        "experiment": "EXT-PAIR-001",
        "comparison": "Previously evaluated Phase 9 quantum and Phase 10 classical",
        "evaluation_unit": "Image",
        "paired_images": 70,
        "historical_evaluation_only": True,
        "independent_case_or_patient_holdout": False,
        "clinical_interpretation_allowed": False,
        "uniform_six_class_softmax_entropy_nats": float(np.log(6)),
        "limitations": [
            "Case IDs overlapped historical image-level train/held-out partitions.",
            "No independence-adjusted uncertainty interval is claimed.",
            "Exact McNemar p-value is exploratory; no superiority conclusion is warranted.",
            "Softmax confidence is not evidence of calibrated probability.",
            "Reconstruction is not the original authors' undisclosed implementation.",
        ],
        "source_sha256": {name: digest(path) for name, path in PATHS.items()},
        "models": models,
        "paired_correctness": paired,
    }
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / "historical_paired_audit.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    names = ["Both correct", "Quantum only", "Classical only", "Both wrong"]
    counts = [paired["both_correct"], q_only, c_only, paired["both_incorrect"]]
    bars = ax.bar(range(4), counts, color=["0.25", "0.48", "0.64", "0.83"])
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, count + .5, str(count),
                ha="center", va="bottom")
    ax.set_xticks(range(4), names)
    ax.set_ylabel("Images (n=70)")
    ax.set_ylim(0, max(counts) * 1.2)
    ax.set_title("Historical paired correctness (Phase 9 vs Phase 10)")
    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"historical_paired_correctness.{ext}",
                    dpi=220, bbox_inches="tight")
    plt.close(fig)

    print("\nEXT-PAIR-001 COMPLETE")
    print("Quantum:", models["quantum"]["correct"], "/70",
          "| Classical:", models["classical"]["correct"], "/70")
    print("Paired:", paired)
    print("Mean quantum softmax entropy:",
          models["quantum"]["mean_softmax_entropy_nats"])
    print("Mean classical softmax entropy:",
          models["classical"]["mean_softmax_entropy_nats"])
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
