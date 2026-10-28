"""Generate two publication-layout, source-verified QuCNet evidence figures.

No retraining, raw clinical data, internet access, or historical file changes.
Inputs: archived EXT-MODELS-001 JSON and EXT-STABILITY-001 CSV/JSON.
Outputs: new SVG, vector PDF, and 300-dpi PNG plus reproducibility manifest
under a separate, caller-selected directory.

Run from repo:
    python 05_Source_Code/build_ieee_evidence_figures.py --outdir paper_figures

The default destination is deliberately outside the original result folders.
Never compare original QuCNet authors' test scores directly to these case-level
train-only evaluations. This plots fixed simulated four-qubit features only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
MODELS_FILE = ROOT / "07_Results" / "exp_models_001" / "metrics.json"
STABILITY_JSON = ROOT / "07_Results" / "exp_stability_001" / "metrics.json"
STABILITY_CSV = ROOT / "07_Results" / "exp_stability_001" / "seed_metrics.csv"
SEEDS = (42, 101, 202, 303, 404)
FAMILIES = (
    ("linear_logistic", "Logistic"),
    ("rbf_svm", "RBF SVM"),
    ("random_forest", "Random forest"),
    ("knn_7", "7-nearest neighbors"),
)
MAPS = ("classical_64", "fixed_quantum_64")
METRICS = (
    ("accuracy", "Accuracy"),
    ("balanced_accuracy", "Balanced accuracy"),
    ("macro_f1", "Macro F1"),
)
MAJORITY_N = 57
N_CASES = 203
TOL = 1e-12


def ensure(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_data() -> tuple[dict, dict, list[dict]]:
    models = json.loads(MODELS_FILE.read_text(encoding="utf-8"))
    stability = json.loads(STABILITY_JSON.read_text(encoding="utf-8"))
    with STABILITY_CSV.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ensure(models["experiment"] == "EXT-MODELS-001" and
           stability["experiment"] == "EXT-STABILITY-001",
           "Incorrect source experiment")
    ensure(models["training_cases"] == N_CASES and
           stability["training_only_cases"] == N_CASES,
           "Case denominator changed")
    ensure(tuple(stability["seeds"]) == SEEDS, "Fold seeds unexpectedly changed")
    ensure(len(rows) == 20, "Expected 4 saved arms across 5 fold seeds")
    expected = {
        f"{m}__{clf}" for m in MAPS for clf in
        ("linear_logistic", "random_forest")
    }
    seen = set()
    for row in rows:
        seed, model = int(row["fold_seed"]), row["model"]
        ensure(seed in SEEDS and model in expected,
               f"Unexpected seed/model row: {seed}, {model}")
        ensure((seed, model) not in seen, "Duplicated seed/model row")
        seen.add((seed, model))
        correct = int(row["correct_cases"])
        ensure(math.isclose(correct / N_CASES, float(row["accuracy"]),
                            rel_tol=0, abs_tol=TOL), "Incorrect accuracy denominator")
        for key, _ in METRICS:
            metric = float(row[key])
            ensure(math.isfinite(metric) and 0 <= metric <= 1,
                   "Non-finite or impossible saved metric")
        summary = stability["individual_per_seed"][model]
        ensure(len(summary) == len(SEEDS) and
               tuple(entry["seed"] for entry in summary) == SEEDS,
               "Unexpected per-seed JSON archive order")
        saved = summary[SEEDS.index(seed)]
        # Saved JSON uses an ordered list with a case_metrics member.
        for key, _ in METRICS:
            target = saved["case_metrics"][key]
            ensure(math.isclose(float(row[key]), float(target),
                                rel_tol=0, abs_tol=TOL),
                   f"CSV/JSON metric mismatch: {model}, {seed}, {key}")
    ensure(len(seen) == len(SEEDS) * len(expected), "Missing a model/seed")
    for model in expected:
        original = models["models"][model]["case"]
        reference = next(x for x in rows if int(x["fold_seed"]) == 42 and
                         x["model"] == model)
        for key, _ in METRICS:
            ensure(math.isclose(float(reference[key]), float(original[key]),
                                rel_tol=0, abs_tol=TOL),
                   "Seed-42 stability results do not match original model archive")
    # Validate original source totals and class count, not just reported metric fields.
    for model in (f"{m}__{clf}" for m in MAPS for clf, _ in FAMILIES):
        cm = models["models"][model]["case"]["confusion_matrix"]
        ensure(len(cm) == 6 and all(len(row) == 6 for row in cm),
               f"Missing six-class confusion matrix: {model}")
        ensure(sum(map(sum, cm)) == N_CASES, "Wrong source confusion denominator")
        ensure(math.isclose(sum(cm[i][i] for i in range(6))/N_CASES,
                            models["models"][model]["case"]["accuracy"],
                            abs_tol=TOL, rel_tol=0), "Stored confusion/accuracy mismatch")
    return models, stability, rows


def style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.3,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "figure.titlesize": 10.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })


def matched_comparison(models: dict):
    # Exactly 7.16 inches wide: near 516 pt IEEE two-column planning width.
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.36))
    fig.subplots_adjust(left=.185, right=.983, top=.76, bottom=.20, wspace=.17)
    fig.suptitle("Matched six-class classifier comparison", y=.965,
                 fontsize=10.5, fontweight="semibold")
    ys = list(range(3, -1, -1))
    for j, (metric, title) in enumerate(METRICS):
        ax = axes[j]
        for i, (clf, label) in enumerate(FAMILIES):
            y = ys[i]
            classical = models["models"][f"classical_64__{clf}"]["case"][metric] * 100
            quantum = models["models"][f"fixed_quantum_64__{clf}"]["case"][metric] * 100
            ax.plot(classical, y+.13, "o", color="#161616", ms=5.0,
                    markeredgewidth=.9, zorder=3)
            ax.plot(quantum, y-.13, "D", color="white", markeredgecolor="#161616",
                    ms=5.0, markeredgewidth=1.2, zorder=3)
        ax.set_xlim(0, 40)
        ax.set_xticks([0, 10, 20, 30, 40])
        ax.set_ylim(-.50, 3.52)
        ax.set_title(title, pad=9, fontweight="medium")
        ax.set_xlabel("Percent", labelpad=4)
        ax.grid(axis="x", color="#e2e2e2", linewidth=.55, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", which="both", length=0)
        ax.spines[["top","right","left"]].set_visible(False)
        if j == 0:
            ax.set_yticks(ys, [label for _,label in FAMILIES])
        else:
            ax.set_yticks(ys, [""]*4)
        if j == 0:
            ax.axvline(MAJORITY_N/N_CASES*100, color="#777777", linestyle=(0,(3,3)),
                       linewidth=.9, zorder=1)
    legend = [
        Line2D([],[],color="#161616",marker="o",linestyle="None",markersize=5,
               label="Classical (pooled pixels)"),
        Line2D([],[],color="#161616",marker="D",markerfacecolor="white",
               linestyle="None",markersize=5,label="Fixed simulated quantum"),
    ]
    fig.legend(handles=legend, loc="upper center", ncol=2,
               bbox_to_anchor=(.50,.875), frameon=False, columnspacing=1.1)
    fig.text(.50,.055,
             "Training-only case-grouped OOF (n=203); seed 42. Dashed line: majority baseline on accuracy.",
             ha="center", va="center", fontsize=7.5)
    return fig


def fold_stability(rows: list[dict]):
    by = {(int(r["fold_seed"]),r["model"]):r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.25), sharey=True)
    fig.subplots_adjust(left=.095,right=.982,top=.75,bottom=.23,wspace=.08)
    fig.suptitle("Sensitivity to recorded case-fold assignment", y=.962,
                 fontsize=10.5, fontweight="semibold")
    for j, (clf, title) in enumerate(
        (("linear_logistic", "Logistic regression"),("random_forest", "Random forest"))
    ):
        ax = axes[j]
        for map_name, label, fmt, line in [
            ("classical_64","Classical", "o", "-"),
            ("fixed_quantum_64","Fixed simulated quantum", "D", "--"),
        ]:
            model = f"{map_name}__{clf}"
            values = [float(by[(seed,model)]["accuracy"])*100 for seed in SEEDS]
            ax.plot(SEEDS, values, marker=fmt, markersize=5.2,
                    markerfacecolor=("#161616" if map_name=="classical_64" else "white"),
                    markeredgecolor="#161616", markeredgewidth=1.1, linestyle=line,
                    color="#161616", linewidth=1.2, zorder=3, label=label)
        ax.axhline(MAJORITY_N/N_CASES*100, linestyle=(0,(3,3)), linewidth=.9,
                   color="#777777", zorder=1)
        ax.set_xticks(SEEDS)
        ax.set_xlim(28,418)
        ax.set_ylim(18,35)
        ax.set_yticks([20,25,30,35])
        ax.set_title(title,pad=6, fontweight="medium")
        ax.set_xlabel("Case-fold seed",labelpad=5)
        ax.grid(axis="y", linewidth=.55, color="#e2e2e2", zorder=0)
        ax.tick_params(axis="both",length=3)
        ax.spines[["top","right"]].set_visible(False)
        if j == 0:
            ax.set_ylabel("Case accuracy (%)",labelpad=5)
    legend=[
        Line2D([],[],color="#161616",linestyle="-",marker="o",
               markerfacecolor="#161616",markersize=5,label="Classical"),
        Line2D([],[],color="#161616",linestyle="--",marker="D",
               markerfacecolor="white",markersize=5,label="Fixed simulated quantum"),
        Line2D([],[],color="#777777",linestyle=(0,(3,3)),label="Majority reference"),
    ]
    fig.legend(handles=legend,loc="upper center",bbox_to_anchor=(.5,.86),
               ncol=3,frameon=False,columnspacing=1.35)
    fig.text(.5,.054,"Same 203 recorded training case IDs in all five repeats; NOT five independent cohorts.",
             ha="center",va="center",fontsize=7.6)
    return fig


def export(fig, outdir: Path, stem: str) -> list[Path]:
    outputs=[]
    for ext in ("svg","pdf","png"):
        path=outdir/(stem+"."+ext)
        fig.savefig(path,dpi=300,facecolor="white")
        ensure(path.is_file() and path.stat().st_size>2500,
               f"Empty publication export: {path}")
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir",default="paper_figures",
                   help="Separate output directory (must not be source Results)")
    args=p.parse_args()
    outdir=Path(args.outdir).resolve()
    forbidden=(ROOT/"07_Results").resolve()
    ensure(outdir != forbidden and forbidden not in outdir.parents,
           "Refusing to overwrite historical results directory")
    ensure(outdir != ROOT.resolve(), "Choose a dedicated output directory")
    ensure(not any(outdir.glob("qucnet_ieee_*.pdf")),
           "Publication export exists; choose a clean output directory")
    models,stability,rows=read_data()
    outdir.mkdir(parents=True,exist_ok=True)
    style()
    files=export(matched_comparison(models),outdir,"qucnet_ieee_matched_classifiers")
    files+=export(fold_stability(rows),outdir,"qucnet_ieee_casefold_stability")
    manifest={
        "purpose":"Derived publication-figure candidates; no source training or image checkout",
        "source_artifacts":{str(p.relative_to(ROOT)):file_hash(p)
                            for p in (MODELS_FILE,STABILITY_JSON,STABILITY_CSV)},
        "figures":{p.name:{"sha256":file_hash(p),"bytes":p.stat().st_size}
                   for p in files},
        "case_count":N_CASES,
        "fold_assignment_seeds":list(SEEDS),
        "majority_reference_correct":MAJORITY_N,
        "figure_width_inches":7.16,
        "not_patient_level_or_clinical_validation":True,
    }
    (outdir/"figure_manifest.json").write_text(
        json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print("PASS: 2 source-derived IEEE double-column figure candidates.")
    print("PASS: 6 outputs (SVG, vector PDF, 300 dpi PNG) plus SHA256 manifest.")
    print("PASS: 8 seed-42 classifiers and 20 repeated-fold rows checked.")
    print("No original research results, source figures or clinical data altered.")
    print("Pending: actual compiled IEEE column-size visual proof.")


if __name__=="__main__":
    main()
