"""EXT-PADLINK-001: exploratory association of ROI padding and CV prediction changes.

Uses ONLY already generated training-set case-stratified out-of-fold results
from EXT-PREPROC-001 and Phase 13 train ROI metadata. No training, new
hyperparameter search, raw image access, validation/test evaluation, or edits
to the original datasets. This is a descriptive post-hoc mechanism check.

Question: across the 203 training cases, does more *geometrically estimated*
pre-resize median-fill padding coincide with larger improvements in true-label
probability or case classification under alternative preprocessing?

This cannot establish that padding caused a performance change. Preprocessing
also changes image representation, and ROI/case structure and TI-RADS category
can confound observed correlations.
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
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
TRAIN_META = ROOT / "03_Dataset_Processed/phase13_roi_dataset/train_roi_metadata.csv"
INPUT = ROOT / "07_Results/exp_preproc_001"
SOURCE_METRICS = INPUT / "metrics.json"
SOURCE_PRED = INPUT / "oof_predictions.csv"
OUT = ROOT / "07_Results/exp_padlink_001"
VARIANTS = ("median_original", "reflection_padding", "direct_stretch")
ALTERNATIVES = ("reflection_padding", "direct_stretch")
PROBABILITY_COLUMNS = ("p2", "p3", "p4a", "p4b", "p4c", "p5")
CLASS_NAMES = ("2", "3", "4a", "4b", "4c", "5")
SEED = 20261028
N_BOOTSTRAP = 5000
EPS = 1e-12


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def safe_number(x: float) -> float | None:
    value = float(x)
    return value if np.isfinite(value) else None


def scores(y, p) -> dict:
    return {
        "accuracy": float(accuracy_score(y, p)),
        "balanced_accuracy": float(balanced_accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, labels=range(6),
                                  average="macro", zero_division=0)),
    }


def load_data():
    for path in (TRAIN_META, SOURCE_METRICS, SOURCE_PRED):
        if not path.is_file():
            raise FileNotFoundError(path)
    audit = json.loads(SOURCE_METRICS.read_text(encoding="utf-8"))
    if audit.get("experiment") != "EXT-PREPROC-001":
        raise ValueError("Unexpected source experiment")
    if audit.get("validation_accessed") is not False or audit.get("test_accessed") is not False:
        raise ValueError("Source experiment was not restricted to training CV")
    if audit.get("training_cases") != 203 or audit.get("training_rois") != 311:
        raise ValueError("Unexpected source case/ROI counts")
    if audit.get("median_original_oof_matches_prior_fusion_run") is not True:
        raise ValueError("Original OOF replication prerequisite not met")
    if set(audit.get("models", {})) != set(VARIANTS):
        raise ValueError("Incorrect reported preprocessing methods")

    metadata = pd.read_csv(
        TRAIN_META, dtype={"roi_id": str, "case_id": str, "tirads": str})
    if (len(metadata) != 311 or metadata.case_id.nunique() != 203 or
            metadata.roi_id.duplicated().any() or
            not metadata.split.eq("train").all()):
        raise ValueError("Invalid train metadata")
    if set(metadata.tirads) != set(CLASS_NAMES):
        raise ValueError("Unexpected TI-RADS labels")

    w = metadata.extracted_roi_width.to_numpy(dtype=float)
    h = metadata.extracted_roi_height.to_numpy(dtype=float)
    if (not np.isfinite(w).all() or not np.isfinite(h).all() or
            np.any(w <= 0) or np.any(h <= 0)):
        raise ValueError("Bad ROI dimensions")
    metadata["padding_fraction"] = 1 - np.minimum(w, h) / np.maximum(w, h)
    if not np.all((metadata.padding_fraction >= 0) & (metadata.padding_fraction < 1)):
        raise ValueError("Invalid theoretical padding fraction")
    if (metadata.groupby("case_id")["tirads_class"].nunique() != 1).any():
        raise ValueError("Conflicting labels within a case")

    predictions = pd.read_csv(
        SOURCE_PRED, dtype={"roi_id": str, "case_id": str, "model": str})
    if len(predictions) != 3 * 311 or set(predictions.model) != set(VARIANTS):
        raise ValueError("Expected three variants times 311 ROI OOF predictions")
    expected = metadata.set_index("roi_id")
    collected = {}
    aligned = None
    for variant in VARIANTS:
        rows = predictions.loc[predictions.model == variant].copy()
        if len(rows) != 311 or rows.roi_id.duplicated().any():
            raise ValueError(f"Invalid unique ROI predictions for {variant}")
        if set(rows.roi_id) != set(expected.index):
            raise ValueError("Prediction and train-metadata ROI membership differ")
        rows = rows.set_index("roi_id").loc[expected.index]
        if not np.array_equal(rows.case_id.to_numpy(), expected.case_id.to_numpy()):
            raise ValueError("Mismatched metadata case IDs")
        if not np.array_equal(rows.true_class.to_numpy(dtype=int),
                              expected.tirads_class.to_numpy(dtype=int)):
            raise ValueError("Mismatched metadata class labels")
        values = rows.loc[:, PROBABILITY_COLUMNS].to_numpy(dtype=float)
        if (not np.isfinite(values).all() or (values < 0).any()
                or not np.allclose(values.sum(axis=1), 1, atol=1e-5)
                or not np.array_equal(
                    values.argmax(axis=1), rows.predicted_class.to_numpy(dtype=int))):
            raise ValueError("Corrupt OOF probabilities")
        if aligned is None:
            aligned = rows.loc[:, ["case_id", "true_class"]].copy()
        elif not rows.loc[:, ["case_id", "true_class"]].equals(aligned):
            raise ValueError("Models have non-identical ROI alignment")

        by_case = rows.groupby("case_id", sort=True)
        case_probs = by_case[list(PROBABILITY_COLUMNS)].mean()
        case_y = by_case.true_class.first().to_numpy(dtype=int)
        if (by_case.true_class.nunique() != 1).any() or len(case_y) != 203:
            raise ValueError("Unusable case aggregation")
        case_pred = case_probs.to_numpy().argmax(axis=1)
        obs_scores = scores(case_y, case_pred)
        reported = audit["models"][variant]["pooled_case"]
        for field in obs_scores:
            if not np.isclose(obs_scores[field], reported[field], atol=1e-10):
                raise ValueError(f"Out-of-fold metric disagreement: {variant} {field}")
        if not np.isin(case_y, range(6)).all():
            raise ValueError("Unexpected case labels")
        case_ptrue = case_probs.to_numpy()[np.arange(203), case_y]
        collected[variant] = {
            "cases": case_probs.index.to_numpy(dtype=str),
            "y": case_y,
            "pred": case_pred,
            "ptrue": case_ptrue,
            "log_loss_per_case": -np.log(np.clip(case_ptrue, EPS, 1)),
            "scores": obs_scores,
        }
    cases_ref = collected["median_original"]["cases"]
    labels_ref = collected["median_original"]["y"]
    for variant in VARIANTS:
        if not (np.array_equal(collected[variant]["cases"], cases_ref) and
                np.array_equal(collected[variant]["y"], labels_ref)):
            raise ValueError("Case aggregation differs by variant")

    by_case_pad = metadata.groupby("case_id")["padding_fraction"]
    median_pad = by_case_pad.median().reindex(cases_ref)
    mean_pad = by_case_pad.mean().reindex(cases_ref)
    roi_count = by_case_pad.size().reindex(cases_ref)
    if median_pad.isna().any() or roi_count.isna().any():
        raise ValueError("Missing per-case padding data")
    if not np.isclose(float(median_pad.median()), 0.26597906403940885, atol=1e-12):
        raise ValueError("Unexpected reference case-median padding")
    return audit, collected, median_pad.to_numpy(), mean_pad.to_numpy(), roi_count.to_numpy()


def summarize(audit, collected, median_pad, mean_pad, roi_count):
    reference = collected["median_original"]
    case_ids = reference["cases"]
    truth = reference["y"]
    quartiles = pd.qcut(
        median_pad, q=4, labels=False, duplicates="raise").astype(int) + 1
    if set(quartiles) != {1, 2, 3, 4}:
        raise ValueError("Unexpected padding quantiles")
    case_table = pd.DataFrame({
        "case_id": case_ids,
        "true_class": truth,
        "true_tirads": [CLASS_NAMES[x] for x in truth],
        "roi_count": roi_count,
        "median_padding_fraction": median_pad,
        "mean_padding_fraction": mean_pad,
        "padding_quartile": quartiles,
    })
    for variant in VARIANTS:
        obj = collected[variant]
        case_table[f"pred_{variant}"] = obj["pred"]
        case_table[f"correct_{variant}"] = (obj["pred"] == truth).astype(int)
        case_table[f"true_class_probability_{variant}"] = obj["ptrue"]
        case_table[f"log_loss_{variant}"] = obj["log_loss_per_case"]

    rng = np.random.default_rng(SEED)
    bootstrap_indices = rng.integers(
        0, len(case_ids), size=(N_BOOTSTRAP, len(case_ids)))
    report = {
        "experiment": "EXT-PADLINK-001",
        "evidence_level": "Exploratory post-hoc training-case OOF analysis",
        "cases": len(case_ids),
        "rois": int(np.sum(roi_count)),
        "padding_reference": "Pre-resize Phase13 median-intensity square padding",
        "padding_formula": "1 - min(expanded ROI width, height) / max(width, height)",
        "case_padding_summary": "Median padding fraction across each case's ROIs",
        "quartiles": "Four pooled quantiles defined using case-median padding",
        "bootstrap_seed": SEED,
        "bootstrap_resamples": N_BOOTSTRAP,
        "sources_sha256": {
            str(x.relative_to(ROOT)): sha256(x)
            for x in (TRAIN_META, SOURCE_METRICS, SOURCE_PRED)
        },
        "validation_or_test_accessed": False,
        "model_training_or_selection": False,
        "original_data_modified": False,
        "variants": {
            v: {
                "case_metrics": collected[v]["scores"],
                "mean_case_log_loss": float(np.mean(collected[v]["log_loss_per_case"]))
            } for v in VARIANTS
        },
        "comparisons": {},
        "interpretation_limits": [
            "Post-hoc correlation is not proof that square padding caused model errors.",
            "Direct stretching changes anatomical aspect ratio and other image properties.",
            "Case IDs are not verified independent patient IDs.",
            "Quartile composition may vary by TI-RADS category, case ROI count and acquisition.",
            "Exploratory p-values and intervals are not adjusted for multiple comparisons.",
            "Case bootstrap does not reflect model refitting or alternative fold assignment.",
            "No independently untouched test set was used for the extension.",
        ],
    }
    for alt in ALTERNATIVES:
        left = collected["median_original"]
        right = collected[alt]
        correctness_change = ((right["pred"] == truth).astype(int) -
                              (left["pred"] == truth).astype(int))
        loss_gain = left["log_loss_per_case"] - right["log_loss_per_case"]
        prob_gain = right["ptrue"] - left["ptrue"]
        case_table[f"delta_correct_{alt}"] = correctness_change
        case_table[f"delta_log_loss_improvement_{alt}"] = loss_gain
        case_table[f"delta_true_probability_{alt}"] = prob_gain

        rho_loss, p_loss = spearmanr(median_pad, loss_gain)
        rho_correct, p_correct = spearmanr(median_pad, correctness_change)
        boot_mean = loss_gain[bootstrap_indices].mean(axis=1)
        by_quartile = []
        for q in range(1, 5):
            sel = quartiles == q
            by_quartile.append({
                "quartile": q,
                "cases": int(np.sum(sel)),
                "padding_min": float(np.min(median_pad[sel])),
                "padding_max": float(np.max(median_pad[sel])),
                "padding_median": float(np.median(median_pad[sel])),
                "TI_RADS_counts": {
                    c: int(np.sum(truth[sel] == k))
                    for k, c in enumerate(CLASS_NAMES)
                },
                "median_method_accuracy": float(np.mean(left["pred"][sel] == truth[sel])),
                "alternative_accuracy": float(np.mean(right["pred"][sel] == truth[sel])),
                "delta_case_accuracy": float(np.mean(correctness_change[sel])),
                "delta_case_log_loss_improvement": float(np.mean(loss_gain[sel])),
            })

        report["comparisons"][f"median_to_{alt}"] = {
            "mean_case_log_loss_gain": float(np.mean(loss_gain)),
            "case_bootstrap_95_CI_for_mean_log_loss_gain":
                np.percentile(boot_mean, [2.5, 97.5]).tolist(),
            "mean_true_class_probability_gain": float(np.mean(prob_gain)),
            "spearman_case_median_padding_vs_log_loss_gain": {
                "rho": safe_number(rho_loss),
                "exploratory_two_sided_p": safe_number(p_loss),
            },
            "spearman_case_median_padding_vs_correctness_change": {
                "rho": safe_number(rho_correct),
                "exploratory_two_sided_p": safe_number(p_correct),
            },
            "by_padding_quartile": by_quartile,
        }
    return report, case_table


def plot_results(report: dict, case_table: pd.DataFrame):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False})
    names = ["Reflection padding", "Direct stretch"]
    keys = [f"median_to_{v}" for v in ALTERNATIVES]
    x = np.arange(1, 5)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, key in enumerate(keys):
        vals = [
            100 * z["delta_case_accuracy"]
            for z in report["comparisons"][key]["by_padding_quartile"]]
        ax.plot(x, vals, marker=("o" if k == 0 else "s"),
                color=("0.25" if k == 0 else "0.60"),
                label=names[k])
    ax.axhline(0, ls="--", linewidth=1, color="0.5")
    ax.set_xticks(x, [f"Q{i}" for i in x])
    ax.set_xlabel("Quartile of median ROI padding fraction per case")
    ax.set_ylabel("Change in case accuracy (percentage points)")
    ax.set_title("Exploratory accuracy change by original padding amount")
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"accuracy_delta_by_padding_quartile.{ext}",
                    dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, alt in enumerate(ALTERNATIVES):
        ax.scatter(100 * case_table.median_padding_fraction,
                   case_table[f"delta_log_loss_improvement_{alt}"],
                   s=19, alpha=0.55, linewidths=0,
                   color=("0.20" if k == 0 else "0.60"), label=names[k])
    ax.axhline(0, color="0.50", ls="--", lw=1)
    ax.set_xlabel("Median original ROI padding per case (%)")
    ax.set_ylabel("Change in true-class log loss (positive = improvement)")
    ax.set_title("Exploratory padding versus predictive-loss change")
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        fig.savefig(OUT / f"case_padding_vs_log_loss_gain.{ext}",
                    dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite audit: {OUT}")
    audit, collected, median_pad, mean_pad, roi_count = load_data()
    report, case_table = summarize(
        audit, collected, median_pad, mean_pad, roi_count)
    OUT.mkdir(parents=True, exist_ok=False)
    case_table.to_csv(OUT / "case_preprocessing_association.csv", index=False)
    (OUT / "association_metrics.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plot_results(report, case_table)
    print("\nEXT-PADLINK-001 COMPLETE")
    for name, comparison in report["comparisons"].items():
        print(name)
        print("  mean log-loss gain:",
              comparison["mean_case_log_loss_gain"])
        print("  bootstrap interval:",
              comparison["case_bootstrap_95_CI_for_mean_log_loss_gain"])
        print("  Spearman padding vs loss gain:",
              comparison["spearman_case_median_padding_vs_log_loss_gain"])
        print("  quartile deltas:", [
            round(q["delta_case_accuracy"], 5)
            for q in comparison["by_padding_quartile"]])
    print("Saved:", OUT)
    print("Development predictions and training metadata only; no models retrained.")


if __name__ == "__main__":
    main()
