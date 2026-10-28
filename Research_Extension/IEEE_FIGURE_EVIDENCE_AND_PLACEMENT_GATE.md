# Gate 4 — Research Figures, Tables and IEEE-Scale Production Review

**Date:** 28 October 2026  
**Stage:** Publication production planning, **not manuscript writing**  
**Result:** Existing figure files are structurally verified and their source data are preserved. **Final IEEE print-size readability is NOT yet certified.**

## 1. Verification performed

The research branch contains **15 versioned experimental plots** with matching **SVG, PNG and PDF** files and nine additional README editorial/scientific SVG figures. The read-only `05_Source_Code/verify_svg_figure_assets.py` uses the Python standard library to verify all **24 SVG files** are parseable, have positive dimensions and a viewBox, include vector drawing, and contain no SVG script or external linked assets. The same script verifies the **15 experimental PNG/PDF companion file paths**.

GitHub Actions **PASS**: [research evidence and figure audit](https://github.com/karthikaxiom/QCNN_Thyroid_Paper_Reproduction/actions/runs/40666066419). Its sparse checkout includes derived plots, saved predictions and metadata, **not original DDTI JPEG/XML medical records**.

This verification does **not** mean that plotted point values were independently reverse-engineered from the artwork; numerical checks remain tied to the saved JSON, CSV and prediction files. Nor does it mean anyone has inspected the exported figures at the final IEEE column width in a compiled article. The original SVGs, PNGs, PDFs and model results have not been altered.

## 2. Measured experimental plot geometry

Indicative IEEE widths used for planning are **252 pt (single column)** and **516 pt (two columns)**. Final dimensions must be taken from the specific journal template. Effective SVG font sizes shrink with the export scaling factor; estimated labels assume 10 pt base Matplotlib text and are **not** measured minimum glyph sizes.

| Archived plot | Source SVG width | Scale at 252 pt | Scale at 516 pt | Print decision |
| --- | ---: | ---: | ---: | --- |
| `exp_qmap_001/fixed_feature_map_case_comparison.svg` | 532.13 pt | 0.474 | 0.970 | Candidate double column; still check fine labels |
| `exp_preproc_001/preprocessing_case_metrics.svg` | 568.13 pt | 0.444 | 0.908 | Candidate double column |
| `exp_padlink_001/case_padding_vs_log_loss_gain.svg` | 568.10 pt | 0.444 | 0.908 | Supplementary, double column |
| `exp_fusion_001_statistics/paired_accuracy_differences.svg` | 582.54 pt | 0.433 | 0.886 | Supplementary, double column |
| `exp_fusion_001_statistics/case_confusion_fusion.svg` | 333.03 pt | 0.757 | 1.549 | Candidate single column after proof |
| `exp_models_001/matched_classifier_comparison.svg` | **891.34 pt** | **0.283** | **0.579** | **NOT print-ready as-is at either journal column width; redesign / regenerate for 516 pt or separate panels** |
| `exp_stability_001/case_fold_stability.svg` | **726.64 pt** | **0.347** | **0.710** | **Flag for label-size / line-visibility redesign before final placement** |

The two last flags are scientifically significant: the multi-classifier comparison and repeated-seed stability experiment are likely central to the study, so their primary figures must be as legible as the historical dataset tables. Redesign should preserve *every plotted estimate* and show six-category balanced accuracy alongside raw accuracy where relevant. Do not hide adverse results.

The README's newer illustrations (ROC, sensitivity, classifier-family, pipeline comparison, calibration) use **unitless pixel-like SVG widths from 1,050 to 1,310**, so dividing those width numbers by IEEE point dimensions would mix units. Measure their *actual* text geometry in the final PDF before approving their use. The SVG structure checker does not certify typography or accessibility contrast.

## 3. Compact figure and table plan — scientific evidence first

**Main figures (candidate order, not final numbering):**

1. **Dataset and evaluation flow:** 480 source JPEGs / 390 source case IDs → 349 recoverably labeled JPEGs / 298 case IDs → 440 strict ROIs / 291 recorded case IDs → frozen 311/58/71 ROI train/validation/test, representing 203/44/44 case IDs. State explicitly that case IDs are not verified unique patients. Source: split manifests and `verify_frozen_case_splits.py`. Existing dataset-flow image `09_Output_images/README/fig_02_audited_dataset_split.png` is raster-only; **prepare a print-appropriate vector version during manuscript figure production**, not by quietly changing counts.
2. **Architecture and reproducibility differences:** separate a diagram showing the *reported* QuCNet approach from our implementer's fixed four-qubit PennyLane mapping, ROI preprocessing and matched 64-dimensional controls. Source: original-paper fidelity audit, source code, quantum crosscheck. Existing `fig_11_model_pipeline_comparison_color.svg` is a starting design, **not source-exact published architecture proof**.
3. **Matched classical versus simulated fixed-quantum classifiers:** the eight-model comparison of accuracy, balanced accuracy, macro-F1, with the majority baseline and transparent TI-RADS 3 weakness. Source: `07_Results/exp_models_001/metrics.json` and `MASTER_VERIFIED_RESULTS.csv`. Existing wide SVG must be redesigned for double-column print. Do not call this an end-to-end trainable QCNN comparison.
4. **Repeated case-fold sensitivity:** seeds 42, 101, 202, 303 and 404 for two classifier families across both fixed maps, including model-ranking reversal, unchanged case pool and weak macro/classwise metrics. Source: `07_Results/exp_stability_001/metrics.json`, `case_oof_predictions.csv`. Existing figure is wide and requires true-size proof; label five **fold assignment** seeds, not independent trials or cohorts.
5. **Failure-mode and reliability:** show **zero TI-RADS 3 sensitivity (0/10) for both random-forest arms on each of five seeds**, compared with the logistic alternatives, *plus* a separate class-balanced reliability or probability-quality comparison if space permits. Sources: `exp_stability_001` and `exp_calib_001`. Do not pool the same 10 cases over seeds as if 50 people had been tested. The existing editorial sensitivity / calibration SVGs need caption-level source review and size proof.
6. **Historical evaluation limits:** preserve the observed Phase 14 collapse and its correct denominator; explain that 4/71 ROI and 2/44 recorded cases are **historical, previously used evaluations** with distinct preprocessing. A separate source-backed vector plot may be needed for print if the present PNG is too small. Do not visually imply a fair one-to-one comparison to the original paper's reported 97.63%/93.87%.

**Main tables:** (a) original-paper versus reconstruction protocol with what is recoverable; (b) DDTI case/image/ROI counts and class distributions; (c) full eight-model case metrics, no selective best-model-only line; (d) five-seed means with sample SD and category-specific failure; (e) clear provenance, threat-to-validity and nonclinical inference summary. The dataset-source citation is **Pedraza et al., Proc. SPIE 9287, 92870W (2015)**, DOI `10.1117/12.2073532`; the base-paper citation is **Swathi et al., IEEE Access 12, 27829–27842 (2024)**, DOI `10.1109/ACCESS.2024.3367806`.

**Supplementary / secondary figures:** matched preprocessing ablation, image/geometry fusion, padding association, near-duplicate screen *without raw patient images*, cache-key audit, and source-supported ROC/AUC (the latter is training-case one-versus-rest OOF, not clinical discrimination validation). Place only figures that substantively help a reader, not every exploratory output.

## 4. Figure-by-figure scientific release criteria

Before any illustration is allowed into the article:

- **Claim fidelity:** every number, label order (2/3/4a/4b/4c/5), denominator, case/ROI unit and evaluation population matches an immutable source file. Distinguish a fold mean from a pooled OOF metric. Identify simulated fixed quantum, not physical/hardware quantum.
- **Visual truthfulness:** zero-recall classes remain visible; avoid cropped y-axes that exaggerate small differences, unsupported significance marks, manual point editing, decorative pseudo-data, or omitting a strong majority-class reference.
- **Statistical honesty:** case-bootstrap intervals and exploratory McNemar p-values do **not** include full model-refitting and sampling uncertainty; no across-seed clinical confidence interval or significance inference. Avoid using the highest seed as the sole reported result.
- **Production:** editable figure-generation source; vector PDF from Matplotlib for the actual paper; check axis labels, legends, math text, captions, colorblind/grayscale distinguishability, minimum print glyph size, line weights and clipping on the *compiled* template. Figures are black-and-white compatible when printed, even if online SVG uses color.
- **Legal/privacy:** use synthetic schematic graphics and approved aggregate data. No raw ultrasound scans or XML-derived identifying details in figures until the separate Gate 3 rights/privacy conditions are resolved.
- **Student-author voice:** descriptions of the work should be specific and straightforward, e.g. "I checked the case assignments against the ROI metadata and found no repeated case ID across partitions." Do not claim personal access to pathology records, actual patient IDs, independent clinical tests or original authors' missing parameters.

## 5. Gate outcome and remaining work

**Gate 4 evidence and figure *structure* audit: PASS.** All predefined figures are present and checked automatically. **Gate 4 final printable layout: OPEN** (needs a real IEEE template with chosen figure widths, corrected primary wide charts, and visual true-size proof). Do not mark article figures "submission ready" until this is done.

No manuscript is authorized yet. The next reviewer-ready package should contain the selected figure/table register with the exact source paths and finalized caption assertions, after which manuscript composition and actual final-scale proof can proceed. The real local XML privacy scan also remains open until its aggregate terminal output is available.
