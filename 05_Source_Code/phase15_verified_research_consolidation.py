from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SOURCE_DIR = Path(__file__).resolve().parent
ROOT = SOURCE_DIR.parent
RESULTS = ROOT / "07_Results"
OUT = RESULTS / "phase15_final_research_evaluation"
OUT.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_binary_metrics(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="ignore")

    def value(label: str) -> float:
        match = re.search(rf"{label}\s*:\s*([0-9.]+)", text, flags=re.I)
        return float(match.group(1)) if match else np.nan

    return {
        "accuracy": value("Accuracy"),
        "precision": value("Precision"),
        "recall": value("Recall"),
        "f1_score": value("F1 Score"),
    }


def build_verified_summary() -> pd.DataFrame:
    rows = []

    # 1. Binary QuCNet TI-RADS-derived proxy classifier
    binary = read_binary_metrics(RESULTS / "metrics.txt")
    rows.append({
        "experiment": "QuCNet Proxy-Label Classifier",
        "phase": "Binary baseline",
        "task": "TI-RADS-derived low- vs high-suspicion proxy",
        "evaluation_unit": "Image",
        "input": "64x64x4 quantum-inspired feature maps",
        "split_protocol": "Historical validation-selected holdout protocol",
        "accuracy": binary["accuracy"],
        "balanced_accuracy": np.nan,
        "precision": binary["precision"],
        "recall": binary["recall"],
        "f1_score": binary["f1_score"],
        "status": "Completed binary experiment",
        "interpretation": (
            "Best overall accuracy among completed experiments, with "
            "high-suspicion-proxy recall but weak low-suspicion-proxy specificity."
        ),
    })

    # 2. Phase 8 TI-RADS
    phase8 = read_json(RESULTS / "phase8_tirads" / "metrics.json")
    rows.append({
        "experiment": "Phase 8 TI-RADS Baseline",
        "phase": "Phase 8",
        "task": "Six-class TI-RADS classification",
        "evaluation_unit": "Image",
        "input": "Prepared TI-RADS image features",
        "split_protocol": "Earlier image-level evaluation",
        "accuracy": phase8.get("accuracy", np.nan),
        "balanced_accuracy": phase8.get("balanced_accuracy", np.nan),
        "precision": phase8.get("macro_precision", np.nan),
        "recall": phase8.get("macro_recall", np.nan),
        "f1_score": phase8.get("macro_f1", np.nan),
        "status": "Completed multiclass baseline",
        "interpretation": "Earlier six-class TI-RADS baseline.",
    })

    # 3. Phase 9 quantum TI-RADS
    phase9 = read_json(
        RESULTS
        / "phase9_paper_tirads_class_weighted"
        / "test_metrics.json"
    )
    rows.append({
        "experiment": "Phase 9 Quantum TI-RADS",
        "phase": "Phase 9",
        "task": "Class-weighted six-class TI-RADS classification",
        "evaluation_unit": "Image",
        "input": "Quantum-inspired feature representation",
        "split_protocol": "Earlier image-level evaluation",
        "accuracy": phase9.get("accuracy", np.nan),
        "balanced_accuracy": phase9.get("balanced_accuracy", np.nan),
        "precision": phase9.get("macro_precision", np.nan),
        "recall": phase9.get("macro_recall", np.nan),
        "f1_score": phase9.get("macro_f1", np.nan),
        "status": "Completed quantum multiclass experiment",
        "interpretation": (
            "Quantum-inspired TI-RADS model under class weighting."
        ),
    })

    # 4. Phase 10 classical TI-RADS
    phase10 = read_json(
        RESULTS
        / "phase10_classical_tirads_class_weighted"
        / "test_metrics.json"
    )
    rows.append({
        "experiment": "Phase 10 Classical TI-RADS",
        "phase": "Phase 10",
        "task": "Class-weighted six-class TI-RADS classification",
        "evaluation_unit": "Image",
        "input": "Classical image representation",
        "split_protocol": "Earlier image-level evaluation",
        "accuracy": phase10.get("accuracy", np.nan),
        "balanced_accuracy": phase10.get("balanced_accuracy", np.nan),
        "precision": phase10.get("macro_precision", np.nan),
        "recall": phase10.get("macro_recall", np.nan),
        "f1_score": phase10.get("macro_f1", np.nan),
        "status": "Completed classical multiclass experiment",
        "interpretation": (
            "Outperformed the Phase 9 quantum-inspired TI-RADS model."
        ),
    })

    # 5-6. Strict ROI/case-level experiment
    phase14 = read_json(
        RESULTS
        / "phase14_roi_categorical_baseline"
        / "evaluation_metrics.json"
    )

    roi = phase14["roi_level"]
    rows.append({
        "experiment": "Phase 14 Strict ROI Baseline",
        "phase": "Phase 14",
        "task": "Six-class TI-RADS classification",
        "evaluation_unit": "ROI",
        "input": "64x64 grayscale ROI crops",
        "split_protocol": "Leakage-free case-level train/validation/test split",
        "accuracy": roi.get("accuracy", np.nan),
        "balanced_accuracy": roi.get("balanced_accuracy", np.nan),
        "precision": roi.get("macro_precision", np.nan),
        "recall": roi.get("macro_recall", np.nan),
        "f1_score": roi.get("macro_f1", np.nan),
        "status": "Failed baseline: single-class collapse",
        "interpretation": (
            "Scientifically important negative result under the strictest "
            "evaluation protocol."
        ),
    })

    case = phase14["case_level"]
    rows.append({
        "experiment": "Phase 14 Strict Case Aggregation",
        "phase": "Phase 14",
        "task": "Six-class TI-RADS classification",
        "evaluation_unit": "Case",
        "input": "Mean probability across case ROIs",
        "split_protocol": "Leakage-free case-level train/validation/test split",
        "accuracy": case.get("accuracy", np.nan),
        "balanced_accuracy": case.get("balanced_accuracy", np.nan),
        "precision": case.get("macro_precision", np.nan),
        "recall": case.get("macro_recall", np.nan),
        "f1_score": case.get("macro_f1", np.nan),
        "status": "Failed baseline: single-class collapse",
        "interpretation": (
            "Case-level aggregation did not recover performance after model "
            "collapse."
        ),
    })

    df = pd.DataFrame(rows)

    percentage_columns = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1_score",
    ]

    for column in percentage_columns:
        df[f"{column}_percent"] = df[column] * 100

    df.to_csv(OUT / "verified_performance_summary.csv", index=False)
    return df


def create_accuracy_figure(df: pd.DataFrame) -> None:
    plot_df = df.sort_values("accuracy", ascending=True)

    plt.figure(figsize=(11, 6.5))
    values = plot_df["accuracy_percent"]

    plt.barh(plot_df["experiment"], values)
    plt.xlabel("Accuracy (%)")
    plt.title("Verified Accuracy Across Distinct Research Experiments")
    plt.xlim(0, 60)

    for i, value in enumerate(values):
        plt.text(value + 0.8, i, f"{value:.2f}%", va="center")

    plt.tight_layout()
    plt.savefig(
        OUT / "verified_accuracy_comparison.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close()


def create_table_figure(df: pd.DataFrame) -> None:
    table_df = df[
        [
            "experiment",
            "task",
            "evaluation_unit",
            "accuracy_percent",
            "balanced_accuracy_percent",
            "precision_percent",
            "recall_percent",
            "f1_score_percent",
            "status",
        ]
    ].copy()

    rename_map = {
        "experiment": "Experiment",
        "task": "Task",
        "evaluation_unit": "Unit",
        "accuracy_percent": "Accuracy",
        "balanced_accuracy_percent": "Balanced Acc.",
        "precision_percent": "Precision",
        "recall_percent": "Recall",
        "f1_score_percent": "F1",
        "status": "Status",
    }

    table_df = table_df.rename(columns=rename_map)

    for column in [
        "Accuracy",
        "Balanced Acc.",
        "Precision",
        "Recall",
        "F1",
    ]:
        table_df[column] = table_df[column].apply(
            lambda x: f"{x:.2f}%" if pd.notna(x) else "—"
        )

    fig, ax = plt.subplots(figsize=(18, 6.5))
    ax.axis("off")

    table = ax.table(
        cellText=table_df.values,
        colLabels=table_df.columns,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.7)

    ax.set_title(
        "Verified Performance Summary by Task and Evaluation Protocol",
        fontsize=15,
        pad=20,
    )

    plt.tight_layout()
    plt.savefig(
        OUT / "verified_performance_summary_table.png",
        dpi=220,
        bbox_inches="tight",
    )
    plt.close()


def write_research_interpretation(df: pd.DataFrame) -> None:
    binary = df.iloc[0]
    phase9 = df[df["experiment"] == "Phase 9 Quantum TI-RADS"].iloc[0]
    phase10 = df[df["experiment"] == "Phase 10 Classical TI-RADS"].iloc[0]
    roi = df[df["experiment"] == "Phase 14 Strict ROI Baseline"].iloc[0]
    case = df[
        df["experiment"] == "Phase 14 Strict Case Aggregation"
    ].iloc[0]

    lines = [
        "FINAL VERIFIED RESEARCH PERFORMANCE INTERPRETATION",
        "=" * 78,
        "",
        "Important reporting rule",
        "-" * 78,
        (
            "The experiments below solve different tasks and use different "
            "evaluation units. Their accuracy values must not be presented as "
            "directly interchangeable."
        ),
        "",
        "1. Binary QuCNet TI-RADS-derived proxy classification",
        "-" * 78,
        f"Accuracy               : {binary['accuracy_percent']:.2f}%",
        f"Precision              : {binary['precision_percent']:.2f}%",
        f"Recall                 : {binary['recall_percent']:.2f}%",
        f"F1 score               : {binary['f1_score_percent']:.2f}%",
        (
            "Interpretation          : The binary experiment achieved the "
            "highest accuracy and strong high-suspicion-proxy recall, but produced many "
            "false-positive high-suspicion-proxy predictions."
        ),
        "",
        "2. Six-class TI-RADS comparison",
        "-" * 78,
        (
            f"Phase 9 quantum model  : "
            f"{phase9['accuracy_percent']:.2f}% accuracy"
        ),
        (
            f"Phase 10 classical     : "
            f"{phase10['accuracy_percent']:.2f}% accuracy"
        ),
        (
            "Interpretation          : Under the matched class-weighted "
            "TI-RADS comparison, the classical model outperformed the "
            "quantum-inspired model."
        ),
        "",
        "3. Strict leakage-free ROI and case evaluation",
        "-" * 78,
        (
            f"ROI-level accuracy     : "
            f"{roi['accuracy_percent']:.2f}%"
        ),
        (
            f"Case-level accuracy    : "
            f"{case['accuracy_percent']:.2f}%"
        ),
        (
            "Interpretation          : The six-class CNN collapsed to a "
            "single class under the strict case-separated ROI protocol. This "
            "is retained as an honest negative result rather than hidden."
        ),
        "",
        "Final scientific conclusion",
        "-" * 78,
        (
            "The reproduction does not support one universal project "
            "accuracy. It supports a task-specific set of results: 54.29% "
            "for TI-RADS-derived binary proxy classification, 18.57% versus 24.29% in the "
            "matched quantum-versus-classical TI-RADS comparison, and "
            "12.50% ROI / 11.36% case accuracy for the strict leakage-free "
            "six-class baseline."
        ),
        (
            "The central research contribution is the discovery that stricter "
            "case-level and ROI-level evaluation substantially reduces the "
            "optimistic performance observed in earlier image-level "
            "experiments."
        ),
    ]

    (OUT / "verified_research_interpretation.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    print("=" * 78)
    print("PHASE 15 V2 : VERIFIED RESEARCH CONSOLIDATION")
    print("=" * 78)

    df = build_verified_summary()
    create_accuracy_figure(df)
    create_table_figure(df)
    write_research_interpretation(df)

    print("Verified summary completed.")
    print(f"Outputs saved in: {OUT}")


if __name__ == "__main__":
    main()