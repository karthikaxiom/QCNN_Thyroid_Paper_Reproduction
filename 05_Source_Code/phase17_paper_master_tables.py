from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SOURCE_DIR = Path(__file__).resolve().parent
ROOT = SOURCE_DIR.parent

RESULTS = ROOT / "07_Results"
PROCESSED = ROOT / "03_Dataset_Processed"
MODELS = ROOT / "06_Models"

OUT = RESULTS / "phase17_paper_master_tables"
OUT.mkdir(parents=True, exist_ok=True)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_table_image(
    df: pd.DataFrame,
    output_path: Path,
    title: str,
    width: float = 16,
) -> None:
    figure_height = max(4.5, 0.55 * (len(df) + 2))

    fig, ax = plt.subplots(figsize=(width, figure_height))
    ax.axis("off")

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.5)

    ax.set_title(title, fontsize=15, pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def create_dataset_flow_table() -> pd.DataFrame:
    rows = [
        {
            "Stage": "Raw metadata inventory",
            "Unit": "Metadata rows",
            "Count": 480,
            "Purpose": "Initial DDTI-derived metadata inventory",
        },
        {
            "Stage": "Diagnosed records",
            "Unit": "Images",
            "Count": 349,
            "Purpose": "Records with benign or malignant diagnosis",
        },
        {
            "Stage": "Benign records",
            "Unit": "Images",
            "Count": 157,
            "Purpose": "Binary class 0",
        },
        {
            "Stage": "Malignant records",
            "Unit": "Images",
            "Count": 192,
            "Purpose": "Binary class 1",
        },
        {
            "Stage": "Diagnosis-missing records",
            "Unit": "Rows",
            "Count": 131,
            "Purpose": "Excluded from binary classification",
        },
        {
            "Stage": "Binary training set",
            "Unit": "Images",
            "Count": 279,
            "Purpose": "Original binary training split",
        },
        {
            "Stage": "SMOTE-balanced binary training set",
            "Unit": "Images",
            "Count": 306,
            "Purpose": "153 benign and 153 malignant",
        },
        {
            "Stage": "Binary test set",
            "Unit": "Images",
            "Count": 70,
            "Purpose": "31 benign and 39 malignant",
        },
        {
            "Stage": "Strict ROI test set",
            "Unit": "ROIs",
            "Count": 72,
            "Purpose": "Leakage-free Phase 14 ROI evaluation",
        },
        {
            "Stage": "Strict case-level test set",
            "Unit": "Cases",
            "Count": 44,
            "Purpose": "Case-aggregated Phase 14 evaluation",
        },
    ]

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "dataset_flow_table.csv", index=False)

    save_table_image(
        df,
        OUT / "dataset_flow_table.png",
        "Dataset Flow and Evaluation Units",
        width=15,
    )

    return df


def create_experiment_configuration_table() -> pd.DataFrame:
    phase14_config = read_json(
        RESULTS
        / "phase14_roi_categorical_baseline"
        / "training_configuration.json"
    )

    rows = [
        {
            "Experiment": "Binary QuCNet proxy-label experiment",
            "Task": "TI-RADS-derived low- vs high-suspicion proxy",
            "Input": "64x64x4 quantum-inspired maps",
            "Balancing": "SMOTE",
            "Classes": 2,
            "Evaluation unit": "Image",
            "Split protocol": "Historical validation-selected holdout",
            "Model": "Flatten + Dense 128 + Dense 64 + Softmax",
        },
        {
            "Experiment": "Phase 8 TI-RADS",
            "Task": "Six-class TI-RADS",
            "Input": "Prepared TI-RADS image features",
            "Balancing": "None / baseline",
            "Classes": 6,
            "Evaluation unit": "Image",
            "Split protocol": "Earlier image-level split",
            "Model": "TI-RADS baseline model",
        },
        {
            "Experiment": "Phase 9 Quantum TI-RADS",
            "Task": "Weighted six-class TI-RADS",
            "Input": "Quantum-inspired representation",
            "Balancing": "Class weights",
            "Classes": 6,
            "Evaluation unit": "Image",
            "Split protocol": "Earlier image-level split",
            "Model": "Quantum-inspired TI-RADS model",
        },
        {
            "Experiment": "Phase 10 Classical TI-RADS",
            "Task": "Weighted six-class TI-RADS",
            "Input": "Classical representation",
            "Balancing": "Class weights",
            "Classes": 6,
            "Evaluation unit": "Image",
            "Split protocol": "Earlier image-level split",
            "Model": "Classical CNN",
        },
        {
            "Experiment": "Phase 14 Strict ROI",
            "Task": "Six-class TI-RADS",
            "Input": "64x64x1 ROI crops",
            "Balancing": "Class weights",
            "Classes": phase14_config.get("number_of_classes", 6),
            "Evaluation unit": "ROI",
            "Split protocol": "Leakage-free case-level split",
            "Model": "Compact categorical CNN",
        },
        {
            "Experiment": "Phase 14 Case Aggregation",
            "Task": "Six-class TI-RADS",
            "Input": "Mean ROI probabilities per case",
            "Balancing": "Inherited from ROI model",
            "Classes": 6,
            "Evaluation unit": "Case",
            "Split protocol": "Leakage-free case-level split",
            "Model": "Probability aggregation",
        },
    ]

    df = pd.DataFrame(rows)
    df.to_csv(
        OUT / "experiment_configuration_table.csv",
        index=False,
    )

    save_table_image(
        df,
        OUT / "experiment_configuration_table.png",
        "Experiment Configuration Summary",
        width=20,
    )

    return df


def create_master_results_table() -> pd.DataFrame:
    verified_path = (
        RESULTS
        / "phase15_final_research_evaluation"
        / "verified_performance_summary.csv"
    )

    if not verified_path.exists():
        raise FileNotFoundError(
            f"Verified performance file not found: {verified_path}"
        )

    verified = pd.read_csv(verified_path)

    binary_metrics = read_json(
        RESULTS
        / "phase16_binary_qucnet_evaluation"
        / "binary_evaluation_metrics.json"
    )

    rows = []

    for _, row in verified.iterrows():
        experiment = str(row["experiment"])

        result = {
            "Experiment": experiment,
            "Task": row["task"],
            "Unit": row["evaluation_unit"],
            "Accuracy": row["accuracy"],
            "Balanced accuracy": row["balanced_accuracy"],
            "Precision": row["precision"],
            "Recall / Sensitivity": row["recall"],
            "Specificity": np.nan,
            "F1 score": row["f1_score"],
            "ROC-AUC": np.nan,
            "Average precision": np.nan,
            "MCC": np.nan,
            "Status": row["status"],
        }

        if experiment == "QuCNet Tumor Classifier":
            result["Accuracy"] = binary_metrics.get("accuracy", np.nan)
            result["Precision"] = binary_metrics.get(
                "precision_positive_malignant",
                np.nan,
            )
            result["Recall / Sensitivity"] = binary_metrics.get(
                "sensitivity_recall_malignant",
                np.nan,
            )
            result["Specificity"] = binary_metrics.get(
                "specificity_benign",
                np.nan,
            )
            result["F1 score"] = binary_metrics.get(
                "f1_score_malignant",
                np.nan,
            )
            result["ROC-AUC"] = binary_metrics.get(
                "roc_auc",
                np.nan,
            )
            result["Average precision"] = binary_metrics.get(
                "average_precision",
                np.nan,
            )
            result["MCC"] = binary_metrics.get(
                "matthews_correlation_coefficient",
                np.nan,
            )

        rows.append(result)

    df = pd.DataFrame(rows)

    df.to_csv(
        OUT / "master_results_table.csv",
        index=False,
    )

    display_df = df.copy()

    metric_columns = [
        "Accuracy",
        "Balanced accuracy",
        "Precision",
        "Recall / Sensitivity",
        "Specificity",
        "F1 score",
        "ROC-AUC",
        "Average precision",
        "MCC",
    ]

    for column in metric_columns:
        display_df[column] = display_df[column].apply(
            lambda value: (
                f"{100 * float(value):.2f}%"
                if pd.notna(value)
                and column not in ["MCC"]
                else (
                    f"{float(value):.3f}"
                    if pd.notna(value)
                    else "—"
                )
            )
        )

    save_table_image(
        display_df,
        OUT / "master_results_table.png",
        "Master Results Table",
        width=22,
    )

    return df


def create_paper_artifact_checklist() -> None:
    checklist = pd.DataFrame(
        [
            ["Dataset flow table", "Complete"],
            ["Experiment configuration table", "Complete"],
            ["Master results table", "Complete"],
            ["Binary ROC curve", "Complete"],
            ["Binary precision-recall curve", "Complete"],
            ["Binary confusion matrices", "Complete"],
            ["Binary classification report", "Complete"],
            ["TI-RADS confusion matrices", "Complete"],
            ["Quantum vs classical comparison", "Complete"],
            ["Strict ROI and case evaluation", "Complete"],
            ["Error-analysis gallery", "Pending"],
            ["Final architecture diagram", "Pending"],
            ["Grad-CAM explainability", "Optional / Pending"],
        ],
        columns=["Paper artifact", "Status"],
    )

    checklist.to_csv(
        OUT / "paper_artifact_checklist.csv",
        index=False,
    )


def main() -> None:
    print("=" * 78)
    print("PHASE 17 : PAPER MASTER TABLES")
    print("=" * 78)

    print("[1/4] Creating dataset-flow table...")
    create_dataset_flow_table()

    print("[2/4] Creating experiment-configuration table...")
    create_experiment_configuration_table()

    print("[3/4] Creating master results table...")
    create_master_results_table()

    print("[4/4] Creating artifact checklist...")
    create_paper_artifact_checklist()

    print("\nPhase 17 completed successfully.")
    print(f"Outputs saved in: {OUT}")


if __name__ == "__main__":
    main()