# QuCNet Revisited — experimental and mathematical closeout matrix

**Status (2026-10-28): EMPIRICAL EXPERIMENTS COMPLETED; final manuscript readiness checks still open.**

This file is a research verification artifact, **not a manuscript draft**. It supplements — and does not replace — the established `README.md`, `Project_Log.txt`, historical phases or model files. No further exploratory model training is planned unless a demonstrable correctness error requires it.

## Evidence hierarchy and evaluation scope

- **Source supervision:** six originally assigned TI-RADS categories (2, 3, 4a, 4b, 4c, 5), **not pathology-confirmed benign/malignant**. Historical binary outcomes are suspicion-proxy labels derived from these categories.
- **Source pool:** 480 archived raw images; 349 with source TI-RADS labels and 131 lacking them, with 390 case IDs in raw inventory (298 labeled case IDs). The stricter Phase13 dataset contains 440 ROIs from 291 cases, divided 311 train ROIs / 203 train cases, 58 validation ROIs / 44 validation cases, and 71 test ROIs / 44 test cases.
- **Main development evidence:** fivefold case-stratified out-of-fold predictions exclusively on the original 203 Phase13 training cases; standardization fitted inside each fold; any ROIs of the same case held together.
- **Historical evaluations:** preserve Phase9/10, Phase14 and Phase16 reports exactly. Earlier image-level splits had case-ID contamination; historical test cases have already been examined and cannot serve as a freshly untouched test for newly explored representations.
- **Patient uniqueness:** case ID is not a confirmed longitudinal patient identifier. There is no demonstrated patient-independent external validation.

## Cross-checked quantitative evidence

| Study and unit | Classical/reference | Quantum/alternative | Verification or inference |
| --- | --- | --- | --- |
| Historical Phase9/10 image-level six class, 70 images | Classical 17/70 (24.29%) | Quantum 13/70 (18.57%) | Paired exact McNemar p=0.5235; no superiority claim; earlier case-overlap limitations |
| Strict Phase14 test ROI/case | 4/71 (5.63%) ROI | 2/44 (4.55%) case | Saved predictions and metrics independently recomputed; predicts one class |
| Historical binary suspicion proxy, 70 images | 38/70 (54.29%) quantum | — | Recomputed; no histopathology label |
| EXT-FUSION-001 case CV | Image 45/203 (22.17%) | Fusion 46/203 (22.66%) | Difference +1/203, paired bootstrap interval −2.46 to +3.94 percentage points, McNemar p=1.00 |
| EXT-PREPROC-001 case CV | Median-filled 45/203 (22.17%) | Reflection 51/203 (25.12%); direct stretch 57/203 (28.08%) | Exact reference reconstruction (max pixel error 0); neither paired accuracy interval excludes zero |
| EXT-PADDING-001 geometry | 311 training ROIs | Median theoretical square-padding area 25.6%; 157/311 above 25% | Median-filled pixels are *not* zero padding; pre-resize area calculation |
| EXT-PADLINK-001 case CV | Original padding | Reflection: log-loss gain −0.0594; direct stretch: +0.0122 | Both mean-gain bootstrap CIs include zero; padding-amount Spearman ρ=0.0716 and −0.0107, respectively |
| EXT-QMAP-001 **matched fixed feature maps**, case CV | Classical 64: **54/203 (26.60%)** | Fixed quantum 64: **43/203 (21.18%)** | Independently recomputed full confusions, bAcc and macro-F1 from 622 ROI OOF rows. Difference quantum−classical −11/203 (−5.42 points); 95% case-bootstrap CI [−10.84, 0.00] points; exact McNemar p=0.07076 |
| EXT-DUPLICATE-001 | 480 raw images, 471 unique byte contents | Six exact-duplicate groups, two across case IDs | No exact-byte groups cross two assigned splits |
| EXT-NEARDUP-001 | 114,847 different-case file pairs screened | 23 perceptual candidates, three assigned cross-split | Three raw-image pairs visually inspected, no identical frames confirmed; cannot rule out same patient across distinct images |

**Data backed by the repository:** `07_Results/exp_fusion_001`, `exp_fusion_001_statistics`, `exp_preproc_001`, `exp_padding_001`, `exp_padlink_001`, `exp_duplicate_001`, `exp_neardup_001`, `exp_qmap_001`, and `Research_Extension/verification`. Original source files have not been overwritten.

## Mathematical definitions and dimensional checks

For 6×6 confusion matrix C with rows as true TI-RADS categories and columns as predicted categories:

- N = sum of all C entries; number correct = trace(C); accuracy = trace(C)/N.
- Recall of category i = C[i,i] / sum_j C[i,j], with undefined zero-support terms handled explicitly.
- Balanced accuracy = (1/6) sum_i Recall_i.
- Precision of category i = C[i,i] / sum_j C[j,i] (zero if no predicted samples).
- Macro-F1 = (1/6) sum_i 2 Precision_i Recall_i/(Precision_i + Recall_i), with zero denominator assigned zero.
- Paired accuracy difference (model B minus A) = (1/n) sum_case (1[B correct] − 1[A correct]).
- Exact McNemar uses only case-discordance counts. EXT-QMAP-001: classical-only=21, quantum-only=10, n_discordant=31. Two-sided exact binomial p=0.07075554598122835. Both-correct=33; both-wrong=139; total=203.
- Paired nonparametric bootstrap draws complete cases with replacement, with the same drawn indices for both models (5,000 repeats). Percentile intervals are exploratory, **not** an independent replication or an adjustment for selection of preprocessing/hyperparameters.

**Phase13 ROI preprocessing:** Original expanded rectangular ROI has width w and height h; square area = max(w,h)^2; original retained area = w*h; theoretical padded area fraction = 1−min(w,h)/max(w,h). Square padding uses median ROI intensity, then OpenCV INTER_AREA resizing to 64×64×1. The original median-filled ROIs were reconstructed without any pixel discrepancy. Direct anisotropic stretching changes lesion aspect ratio.

**EXT-QMAP-001 feature geometry:**
1. Each Phase13 64×64×1 ROI is mean-pooled into 8×8 intensities; the 64 classical features are those 8×8 values.
2. Reshape 8×8 into a 4×4 grid of disjoint 2×2 patches: 16 patches × 4 input intensities per ROI.
3. The *fixed simulated* four-wire circuit has RX encoding on each wire, two layers of four fixed RY rotations and four CNOT-ring gates (eight RY angles and eight CNOTs per patch total). The output is four Pauli-Z expectation values per patch. Thus 16 × 4 = 64 quantum features per ROI.
4. Across 311 training ROIs, 311 × 16 = **4,976 fixed-patch simulation evaluations**. This is not 4,976 independent patients, training examples or learned quantum circuits.
5. Cross-check against `utilities/quantum_utils.py` gave max absolute expectation-value discrepancy **1.2571526930571508×10⁻⁸**, below the 10⁻⁶ acceptance threshold. Circuit weights are fixed from seed 42 and *not* gradient-trained.
6. This four-qubit simulation is not evidence of quantum computational speedup; features and network families differ from a full QuCNet classifier.

## Image-integrity visual review

The three cross-assigned-split candidates were inspected from their original raw frames through connected GitHub, not only via 64-bit perceptual hashes. They were **not visually identical frames**:

- Validation 277_1 versus train 348_1: differing anatomy, lesion position and direction markings.
- Train 177_1 versus test 313_1: distinct anatomy and framing. The full raw 313_1 frame contains burned-in "TI-RADS 4C" text **below** the recorded Phase13 ROI: recorded crop ends at y=190 in a height-360 raw image.
- Test 305_1 versus train 34_1: different lesion appearance and placement.

These observations reduce the case for exact source-frame duplication among those *flagged pairs*, not proof of different patients or absence of any unflagged duplicates. Do not publish review contact sheets before permissions and potentially identifying markings have been checked. Source review is logged in `Research_Extension/verification/near_duplicate_visual_review.json`.

## Figure quality control

Visually reviewed the stored PNG versions of the four plots below, with the same SVG/PDF source export families available for publication. Text, metric units, grayscale contrasts, legends and category labels are present; no bar-chart axis exceeds its stated units.

- `07_Results/exp_qmap_001/fixed_feature_map_case_comparison.png`
- `07_Results/exp_preproc_001/preprocessing_case_metrics.png`
- `07_Results/exp_padlink_001/case_padding_vs_log_loss_gain.png`
- `07_Results/exp_fusion_001_statistics/paired_accuracy_differences.png`

**Open:** All figures have not yet been individually verified at final IEEE one-column printed size. PNG plots with y-axis 0–100 appropriately show absolute percentage scale but visually compress modest between-model differences; pairwise interval figures must accompany any discussion of superiority. PNG inspection does not independently prove vector PDF/SVG rendering quality.

## Remaining closeout gates — NOT READY FOR MANUSCRIPT YET

1. **Full paper methods and equations checked:** the locally available 14-page original IEEE PDF was inspected directly; its §III–VI preprocessing, four-channel QFT, quantum circuit, architectures, callbacks, mathematical definitions and authors' distinct test denominators have been compared against the repository code. The precise source/implementation matrix is recorded in ORIGINAL_PAPER_FULLTEXT_FIDELITY_AUDIT.md. The source gives broad pipeline details but not all circuit or CNN hyperparameters, and even its RX angle descriptions differ between sections. This is not a gate-for-gate identical reproduction.
2. **Dataset provenance located but rights unresolved:** original CIM@LAB listing and Kaggle DDTI mirror verified; mirror license is 'Other (specified in description)', not an explicit unrestricted redistribution grant. Confirm permissions and screen raw images/XML and publication figures for identifying/burned-in material before reusing them.
3. **Selected figures audited:** five vector SVGs have measured physical widths and reproducible CSV/JSON sources. Four wide plots require double-column IEEE placement; single-column downscaling would reduce nominal label sizes to ~4.3–4.7 pt. The figure audit is in IEEE_FIGURE_PREFLIGHT.md. Actual final printed template proof remains an editorial/typesetting step.
4. Final consistency review of the experiment table, frozen-split counts, optional patient-ID uncertainty, publication ethics, negative findings and explicit limitations. Confirm manuscript outline only after these checks.
5. The public research branch and the historical result records should be reconciled into the README by a **narrow, additive, reviewed update** only when approved for final handoff. Do not overwrite its existing historical text.

**Scientific bottom line:** Fixed quantum image features did not outperform equally dimensioned classical inputs here; no result in this repository establishes clinical TI-RADS accuracy, pathology-confirmed malignancy classification, quantum advantage, or a definitive superiority comparison. This is valuable negative / reproducibility evidence when accurately scoped.

## External source verification

Bibliographic identity and reported source-paper abstract metrics were checked against DBLP and the authors' institutional publication record. DDTI attribution and the unclear redistribution language were checked against the original CIM@LAB listing and the Kaggle mirror metadata. Links, caveats and proposed responsible handling are saved in `Research_Extension/SOURCE_AND_DATA_RIGHTS_AUDIT.md`.

A machine-readable **13-row consolidated results inventory** was generated directly from the repository's verified metrics JSON, distinguishing unit/partition/evidence level: `Research_Extension/MASTER_VERIFIED_RESULTS.csv`.

**No new empirical trials are needed by default.** The full-paper method comparison and selected figure-format audit are now documented. The remaining substantive unresolved item is raw DDTI image/XML redistribution permission; final compiled IEEE print proof is a manuscript production step, not a new scientific experiment. Do not claim unrestricted image rights.

## Final method and figure verification update — 2026-10-28

I independently inspected the complete 14-page original Main_Paper.pdf (not only its metadata/abstract) and reconciled the paper-specific training/testing claims with the source code. The paper states a 128×128 full-image quantum input, four 64×64 channels from 2×2 quantum patches, source 80/20 image split, mean/std scaling, augmentation and SMOTE. The more recent 64×64 ROI case-separated experiment changes the observation unit and preprocessing; it is therefore an independent methodological audit, **not exact numerical reproduction**. Details, page anchors and specific unresolved circuit settings are in ORIGINAL_PAPER_FULLTEXT_FIDELITY_AUDIT.md.

**Source-article numerical notation note:** IEEE printed equation (19) depicts the softmax denominator with lower index starting at j=i. That text is not the standard all-category normalization. The independent implementation uses the standard sum over all classes; do not copy the apparent printed notation anomaly into our own equations as valid software mathematics.

Selected vector figures and accompanying data are accounted for. Existing wide SVGs must be placed as double-column figures; they are too small when blindly scaled into a single IEEE column. See IEEE_FIGURE_PREFLIGHT.md.

**Remaining decision external to machine learning experiments:** obtain explicit documented DDTI raw image/XML re-publication authority or refrain from distributing raw medical images and annotate the dataset source and limits. The publicly tracked raw files are an existing repository-release concern; do not rewrite or delete original Git history without the owner's express instructions. No image copy is required to present the numeric experiments in the article.

## Final source-implementation integrity checkpoint — 2026-10-28

The historical reproduction was reviewed using six explicit checkpoints (genuine quantum simulation, dataset/split fidelity, architecture, optimization settings, metric comparability and approximation disclosure). The **judgment-based original-protocol fidelity score is 55/100**, documented in `Research_Extension/ORIGINAL_PAPER_FULLTEXT_FIDELITY_AUDIT.md`. This percentage is not model accuracy or an indicator of the rigor of the subsequent case-separated audit.

The original quantum transformer caches expectation values under 8-bit quantized patch keys. `EXT-CACHE-001` checked four local training input arrays and confirmed **zero collisions in original-only binary and TI-RADS arrays**, but **17,535 colliding keys / 28,766 subsequent mismatched patch occurrences** in the binary SMOTE array (2.30% of 1,253,376 patches) and **84,657 keys / 219,533 occurrences** in the TI-RADS SMOTE array (11.60% of 1,892,352 patches). This proves the memoization was lossy for synthetic float-valued patches; it does not measure the quantum expectation-value deviations or causal performance changes. Phase 9's verified original-image prefix (279/462) limits direct exposure of its original-only extracted maps, conditional on a clean, sequential cache pass. Detailed evidence is in `ORIGINAL_PAPER_FULLTEXT_FIDELITY_AUDIT.md`. No historical inputs, models, checkpoints or result files were modified.

The recovered DDTI image inventory, labels, and original reported test denominators are not identical; without the authors' exact source-image IDs and actual pathology labels, complete one-to-one reconstruction cannot be certified. These are data provenance limits, not matters that arbitrary further tuning can resolve. The methodologically distinct case-stratified study remains complete and valid as an exploratory audit.

**User-controlled manuscript boundary:** No IEEE manuscript or Overleaf authoring is authorized yet. Preparation may proceed through audit notes and numeric checks only; paper drafting starts only upon express instruction.
