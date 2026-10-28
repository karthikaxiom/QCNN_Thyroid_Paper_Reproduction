"""Read-only verification of existing publication candidate SVG figure assets.

Run: python 05_Source_Code/verify_svg_figure_assets.py

Checks 15 experiment-plot SVGs and nine README figure SVGs, as well as
the original PNG/PDF companions for experimental plots. This uses no
source ultrasound images, no plotting packages, and no internet access.

Passing means the archived SVGs are structurally usable and have matching
source exports. It does *not* prove visually readable typography at final
IEEE column size or correctness of every data point.
"""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = {
    "exp_fusion_001_statistics": (
        "case_confusion_fusion", "case_confusion_geometry",
        "case_confusion_image", "case_confusion_majority",
        "case_cv_metrics", "paired_accuracy_differences",
    ),
    "exp_models_001": ("matched_classifier_comparison",),
    "exp_padding_001/local": (
        "padding_fraction_by_tirads", "padding_fraction_distribution",
    ),
    "exp_padlink_001": (
        "accuracy_delta_by_padding_quartile", "case_padding_vs_log_loss_gain",
    ),
    "exp_pair_001": ("historical_paired_correctness",),
    "exp_preproc_001": ("preprocessing_case_metrics",),
    "exp_qmap_001": ("fixed_feature_map_case_comparison",),
    "exp_stability_001": ("case_fold_stability",),
}
EDITORIAL = (
    "fig_04_extension_fixed_maps_color",
    "fig_05_extension_preprocessing_color",
    "fig_06_extension_padding_logloss_color",
    "fig_07_extension_fusion_ci_color",
    "fig_08_matched_classifier_families_color",
    "fig_09_sixclass_ovr_roc_color",
    "fig_10_classwise_sensitivity_color",
    "fig_11_model_pipeline_comparison_color",
    "fig_12_multiclass_calibration_color",
)
SVG_NS = "{http://www.w3.org/2000/svg}"
HREF_SUFFIX = "}href"
UNIT = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*(?:pt|px)?\s*$")


def require(condition: bool, msg: str) -> None:
    if not condition:
        raise AssertionError(msg)


def dimension(text: str | None, path: str) -> float:
    require(text is not None, f"Missing dimension: {path}")
    m = UNIT.fullmatch(text)
    require(m is not None, f"Unsupported SVG length: {path}: {text}")
    n = float(m.group(1))
    require(math.isfinite(n) and n > 0, f"Invalid SVG dimension: {path}")
    return n


def check_svg(path: Path) -> tuple[float, float]:
    require(path.is_file(), f"Missing SVG: {path}")
    require(path.stat().st_size > 900, f"Suspiciously empty SVG: {path}")
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        raise AssertionError(f"SVG parsing failed: {path}: {e}") from e
    require(root.tag == SVG_NS + "svg", f"Invalid SVG root: {path}")
    raw_width = dimension(root.attrib.get("width"), str(path))
    raw_height = dimension(root.attrib.get("height"), str(path))
    vb = root.attrib.get("viewBox")
    require(vb is not None, f"Missing SVG viewBox: {path}")
    parts = vb.replace(",", " ").split()
    require(len(parts) == 4, f"Invalid SVG viewBox: {path}")
    _, _, w, h = (float(part) for part in parts)
    require(all(math.isfinite(float(v)) for v in parts) and w > 0 and h > 0,
            f"Invalid SVG viewBox dimensions: {path}")
    require(0.01 < (w / h) < 50, f"Unexpected figure aspect ratio: {path}")
    element_count = 0
    drawn = False
    for elem in root.iter():
        element_count += 1
        tag = elem.tag.rsplit("}", 1)[-1]
        if tag in ("path", "text", "rect", "circle", "ellipse", "line", "polygon", "polyline"):
            drawn = True
        if tag == "script":
            raise AssertionError(f"Unexpected executable SVG script: {path}")
        for name, val in elem.attrib.items():
            if name == "href" or name.endswith(HREF_SUFFIX):
                require(val.startswith("#") or val.startswith("data:"),
                        f"Externally linked SVG element: {path}")
    require(drawn and element_count > 5, f"Empty or incomplete SVG drawing: {path}")
    return raw_width, raw_height


def main() -> None:
    checked = []
    paired = 0
    for folder, stems in RESULTS.items():
        for stem in stems:
            basis = ROOT / "07_Results" / folder / stem
            width, height = check_svg(basis.with_suffix(".svg"))
            require(basis.with_suffix(".png").is_file(), f"Missing PNG: {basis}")
            require(basis.with_suffix(".pdf").is_file(), f"Missing PDF: {basis}")
            checked.append((basis, width, height))
            paired += 1
    for stem in EDITORIAL:
        fig = ROOT / "09_Output_images" / "README" / (stem + ".svg")
        width, height = check_svg(fig)
        checked.append((fig, width, height))
    require(len(checked) == 24 and paired == 15, "Incomplete predefined figure set")
    # This source export is especially wide; at 516 pt IEEE textwidth its
    # scale factor is 516/891 ~ 0.579. Do NOT call it print proof.
    wide = next(w for p, w, _ in checked
                if p.stem == "matched_classifier_comparison")
    require(wide > 850, "Expected wide classifier figure changed; re-review sizing")
    print("PASS: 24 structurally parsed SVGs, including 15 experimental figures.")
    print("      All 15 experimental PNG/PDF companion exports exist.")
    print("      No SVG executable scripts or externally linked assets were found.")
    print("FLAG: matched-classifier SVG is very wide; paper-width visual redesign")
    print("      or source-plot re-export is needed before final manuscript placement.")
    print("Scope: structural/vector integrity ONLY; no print-size text reading,")
    print("       plot-data point audit, or inspection of original clinical images.")


if __name__ == "__main__":
    main()
