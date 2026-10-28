# Source-backed IEEE double-column figure production — verified build

**Recorded:** 28 October 2026  
**Stage:** Pre-manuscript figure preparation; article writing remains deferred.  
**Source branch:** `research/bn-recalibration-audit`  
**Figure generator:** `05_Source_Code/build_ieee_evidence_figures.py`  
**GitHub Actions:** [successful run 40669696948](https://github.com/karthikaxiom/QCNN_Thyroid_Paper_Reproduction/actions/runs/40669696948)  
**Downloadable Actions artifact:** `qucnet-ieee-double-column-figure-candidates` (six figure files and `figure_manifest.json`). GitHub's 30-day artifact expiry means this ZIP is a convenience, **not permanent archival storage**; the code and numeric source data remain tracked for regeneration.

## What was actually produced

A dedicated reproducible figure builder now reads three **saved, frozen, nonclinical numerical inputs only**:

- `07_Results/exp_models_001/metrics.json`: eight matched classical/quantum classifier outputs for **203 recorded training case IDs** in case-stratified seed-42 OOF evaluation.
- `07_Results/exp_stability_001/metrics.json`: five prespecified case-fold seeds and archive metadata/summary.
- `07_Results/exp_stability_001/seed_metrics.csv`: 20 seed-by-model metrics (five seeds × two fixed representations × two classifiers).

The generator checks the **203-case denominator**, all eight saved six-class confusion matrices, all 20 saved fold-seed result rows, agreement of the CSV with the stored per-seed JSON records, and exact matching of the four seed-42 repeated rows to the earlier matched classifiers. The plot code recomputes percentages from the *saved actual values*, not manually entered heights.

| Candidate figure | Evidence displayed | Exported files | Principal limits |
| --- | --- | --- | --- |
| `qucnet_ieee_matched_classifiers` | Four classifier families: logistic regression, RBF SVM, random forest and 7-nearest neighbors; one pair of matched classical / fixed-quantum 64-feature arms; **accuracy, six-class balanced accuracy, macro F1**; majority reference on accuracy only | SVG, vector PDF, 300-dpi PNG | **Seed 42**, training-only case OOF; not a test-set or source-paper reproduction score |
| `qucnet_ieee_casefold_stability` | Separate logistic and random-forest subpanels, classical and fixed-quantum accuracy for fold-assignment seeds **42, 101, 202, 303, 404**, majority-class reference **57/203** | SVG, vector PDF, 300-dpi PNG | The same **203 case IDs** are regrouped five times, **not five new cohorts**; differences are descriptive, not significance estimates |

**Print design choices:** Figure width **7.16 in** (approximately a 516-pt IEEE full-width planning span), high-contrast black/white symbols distinguish feature maps for color-blind and grayscale printing, axis labels and explicit legend, fixed common metric axis for fair comparison, a majority reference wherever appropriate, plus an explicit footer identifying training-case-only scope. The source code uses a consistent DejaVu Sans font and PDF TrueType embedding options. Final venue-specific column widths must still be checked in the eventual template.

## Confirmed independent execution

The GitHub workflow `.github/workflows/ieee-source-figure-build.yml` runs in a new Linux environment and **sparsely checks out only the figure code and three saved numeric sources**. It installs Matplotlib **3.10.7**, generates figures, checks that **all six exports** exist and their SHA-256 hashes/byte lengths agree with `figure_manifest.json`, and uploads the derived ZIP.

Successful build log (28 October 2026, run 40669696948) confirmed:

- **PASS:** two source-derived double-column figure candidates.
- **PASS:** six SVG/PDF/PNG files and a SHA-256 manifest.
- **PASS:** eight seed-42 models and 20 repeated-fold cases-by-model checks.
- **PASS:** output files match recorded hashes.
- **PASS:** CI artifact upload (approx. **284 KB**, artifact ID **12807013414**). Retention until **27 November 2026**, unless the repository removes it earlier.

**Neither raw DDTI image/XML nor historical results, original figure files, saved models or patient records were modified.** Only added code and workflow create these publication candidates.

## Essential scientific caption and interpretation controls

1. **Always call the quantum representation a fixed, simulated four-qubit feature map.** It is not a trainable quantum CNN, a Qiskit hardware run, or the precise original circuit.
2. **State 'training-only, recorded case-ID grouped OOF, n = 203'**. Do not claim unique patients or unseen external clinical test cases.
3. On the matched classifier figure, **seed 42 is an exploratory snapshot** among multiple model families; the highest quantum RF accuracy is **63/203 = 31.03%**, and no controlled quantum superiority was established (quantum RF versus classical RF paired exact McNemar p≈0.1755, exploratory).
4. On stability, the quantum RF mean accuracy is **29.36%** versus classical RF **27.49%**, but the ranking reverses at seed **101** and quantum RF falls below the majority reference (**55/203 vs 57/203**) at seed **202**. **Both RF variants miss every one of the ten recorded TI-RADS 3 cases in every fold seed**; that weakness must be reported alongside figures rather than hidden.
5. Some displayed differences are only a few cases and their uncertainty is not a demonstrated generalization gain. Five seeds are correlated repeated partitions of the same data.
6. No plotting artifact should use the 97.63%/93.87% QuCNet *source paper* metrics as a directly matched competing row: dataset, label provenance, circuit and evaluation denominators differ.
7. A graph rendered correctly is not a clinical, patient-identity, publication rights, or human ethics validation.

## How to regenerate without modifying original evidence

From the branch and local Python environment (Matplotlib installed):

```powershell
& .\.venv\Scripts\python.exe .\05_Source_Code\build_ieee_evidence_figures.py --outdir paper_figures
```

Use a new empty output directory or remove only the **newly generated publication artifact directory** after safely backing it up; the script refuses to write into `07_Results` and refuses to overwrite an existing candidate PDF. The original research images and results do not need to be downloaded by the paper-figure workflow.

The current GitHub Actions artifact also supplies the exact outputs already built; if it expires, rerun the workflow or local generator from source.

## Next check — no premature claims

- **Read figures at the true 516-pt placed width** in a two-column IEEE PDF, checking every axis, 7–9 pt annotations and grayscale contrast. The present source and CI checks verify dimensions and generated exports, **not final human visual readability in the article PDF**.
- If needed, revise figure-generation source and rebuild without changing original empirical evidence. Document figure revisions in the manifest.
- Choose article main figures/table captions and supplementary studies with guide input; the two new exports do not exhaust all necessary research illustrations.
- The source dataset rights/privacy decisions, patient identity limitation, external clinical validation and exact venue fit remain separately governed.
- **IEEE manuscript writing is still reserved for last**, consistent with the researcher's direction.

**Gate disposition:** Primary wide-chart *regeneration* demonstrated, independently rerunnable and available as artifact. **True-size final template proof remains open.**


## Visual proof of corrected candidate exports — 28 October 2026

The initial generated PNGs were downloaded and visually inspected. **They showed an overlap between the legend and subplot headings**; passing numeric/structural CI alone was not sufficient. The figure code was therefore revised (commit `cdf4b808`) to give the legends and subplot titles separate vertical bands, preserving the same evidence and 7.16-inch export width. **Corrected build [40669696948](https://github.com/karthikaxiom/QCNN_Thyroid_Paper_Reproduction/actions/runs/40669696948) passed**, and the corrected ZIP was inspected directly. Its two PNGs now show unobstructed headings, legends, plot markers, axes and warning footers at the exported preview size. All six final output files matched the SHA-256 manifest in the downloaded ZIP. The corrected artifact's ID is `12807882024` (not the earlier `12807013414`).

**Independent PDF-render inspection:** I additionally rendered **both corrected vector PDF exports to PNG at 180 dpi** and inspected those rendered pages. The figure titles, legends, subplot headings, axes, data markers, case-denominator annotation, and warnings were legible without clipping or overlap at the intrinsic exported size. This confirms the PDF export is consistent with the preview rather than relying on the PNG outputs alone. It still does not substitute for a physically printed or article-embedded IEEE double-column proof.

**Scientific and production boundary:** This checks exported visual layout and numerical integrity. It is **not a printed proof in the final compiled IEEE template**, which remains the final step before article submission. The original research figures and input result files were not overwritten.
