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

## EXT-MODELS-001 eight-classifier verification — 2026-10-28

**Guide-motivated follow-up, separate from the historical strict Phase14 CNN:** Four fixed classifier families (logistic regression, RBF SVM, random forest and 7-neighbor kNN) were applied to **each of two fixed 64-feature maps** (pooled grayscale and PennyLane four-qubit expectations), all on the same 203 Phase13 **training** cases grouped into five out-of-fold partitions. The earlier collapsed CNN, its test cases and its confusion matrices were not changed or refitted.

**Independent numerical checks:** reconstructed 203 case predictions for each of the eight arms from 311 ROI OOF probability records; all **1,624 case rows** and **2,488 ROI rows** validated for unique identities, six normalized class probabilities, case-level averaging, predicted labels and matching confusion matrices/metrics. The script independently checked that both logistic OOF arrays match the earlier EXT-QMAP-001 reference probabilities.

**Key results:** classical logistic 54/203, balanced accuracy .2522, macro-F1 .2426; classical RBF SVM 57/203, .1809/.1379; classical RF 54/203, .1892/.1786; classical kNN 58/203, .1928/.1763. Fixed quantum logistic 43/203, .2191/.2068; fixed quantum SVM 54/203, .1729/.1319; quantum RF **63/203**, .2354/.2335; quantum kNN 59/203, .2042/.1907.

Quantum RF compared with matched classical RF: paired delta **+9/203 (+4.43 percentage points)**, case-bootstrap exploratory 95% CI **[−1.00, +9.85]** points, exact McNemar **p=.1755**. Quantum RF compared with case-majority reference (57/203): +6/203, two-sided McNemar **p=.5504**; baseline-only correct 32, quantum-RF-only correct 38. The highest observed balanced accuracy and macro-F1 among eight configurations were attained by **classical logistic regression**, despite quantum RF's highest raw accuracy.

**Crucial class-level finding:** quantum RF outputs include all six predicted labels, but its TI-RADS 3 confusion-matrix diagonal is **0/10** (it predicts TI-RADS 3 for only one case). Both SVM variants predict only four categories. Retaining all zero confusion cells and class-level counts is scientifically necessary; a chart that simply removes zeros would misrepresent the data.

**Inference and limitations:** model-family comparisons were chosen for exploratory follow-up after guide feedback, without multiplicity adjustment or a new untouched holdout. The within-quantum RF-versus-logistic comparison has unadjusted p=.0091, but this does not demonstrate general superiority after multiple comparisons and does not establish a quantum advantage. Internal SVM(probability=True) calibration was performed inside outer training folds, though its internal folds were not case-grouped; scikit-learn 1.9 reports future deprecation. No clinical, patient-level, or pathology-ground-truth inference follows.

Committed source: `05_Source_Code/exp_models_001_matched_classifiers.py`. Saved results: `07_Results/exp_models_001/`. Public colour figure is source-backed in `09_Output_images/README/fig_08_matched_classifier_families_color.svg`. The `MASTER_VERIFIED_RESULTS.csv` inventory now includes all eight rows. All original models, historical charts and data remain untouched.

**Readiness:** This closes the guide-requested bounded comparison. Further unconstrained classifier tuning would weaken the inferential value of these already-inspected development predictions. Manuscript work remains deliberately deferred until requested.

## EXT-STABILITY-001 independently verified — 2026-10-28

**Evidence:** The predeclared five-fold assignment seeds (42, 101, 202, 303, 404) were executed on only the original Phase 13 training records (311 ROIs, 203 case IDs), for two frozen 64-dimensional feature representations crossed with balanced logistic regression and random forest. The source and results reside in `05_Source_Code/exp_stability_001_repeated_casecv.py` and `07_Results/exp_stability_001/`; the source result commit is `cb46ed0`.

**Independent calculation:** All 4,060 archived case-level out-of-fold records were inspected: 20 unique model/seed groups, each containing 203 uniquely identified case IDs. Six-class probabilities sum to one within floating-point tolerance, predicted labels equal the probability argmax, case truths and assigned folds are consistent, and all 20 confusion matrices, accuracies, balanced accuracies and macro-F1 values agree with the saved JSON and CSV. Every one of 812 seed-42 model/case probability vectors matches the earlier EXT-MODELS-001 file within maximum absolute error `1.11e-16`. No output discrepancies found.

**Across-seed mean (sample SD) case-level accuracy:** classical logistic `24.53% (2.42 pp)`; fixed-quantum logistic `23.55% (1.61 pp)`; classical RF `27.49% (2.71 pp)`; fixed-quantum RF `29.36% (1.58 pp)`. Mean balanced accuracies were `23.23%, 23.64%, 20.42%, 22.50%`, respectively; macro-F1 values were `22.07%, 22.08%, 19.62%, 22.30%`. RF quantum-minus-classical accuracy differences by seed were `+4.43, -1.48, +2.46, +2.96, +0.99 pp`; relative ranking reverses at seed 101. The descriptive mean RF difference is `+1.87 pp`, not a confirmatory superiority estimate. Logistic relative ranking also changes sign.

**Methodological boundary:** Case-fold assignments vary; model RNG, features, quantum circuit and hyperparameters do not. Repeated measurements on the same cases are correlated and are neither new independent cohorts nor inferential confidence intervals. This does not retest the original full QuCNet architecture under an untouched case-separated cohort, prove patient distinctness, validate pathology labels, or demonstrate a clinical or quantum advantage.

**Empirical research gate:** Bounded guide-directed comparative and sensitivity experiments are now complete. The dataset-licensing/republication decision, independent patient identity and external validation remain unresolved external constraints; source/figure/citation and manuscript production review must still be performed prior to submission. No arbitrary new model search is justified by these observations. The IEEE manuscript and Overleaf remain unauthorized until the researcher explicitly requests writing.

## Targeted related-work and contribution audit — 2026-10-28

**Gate 1 update:** The newer, more comprehensive [scientific contribution and novelty review](SCIENTIFIC_CONTRIBUTION_AND_NOVELTY_REVIEW.md) also examines **two directly relevant 2026 quantum-inspired thyroid-ultrasound studies** (Behuria et al., DOI 10.36922/AIH026050007; Raghaw et al., DOI 10.1002/ima.70428), plus close 2025–2026 medical QML reproducibility papers. It records a precise claim/evidence matrix and reviewer objections. Our novelty is a **QuCNet/DDTI-specific reproducibility audit**, not first-of-kind quantum thyroid classification or an original QCNN. Suitability for a particular journal remains a guide/editor judgment.



This is a **research-positioning note, not manuscript prose**. The following publicly published works were checked against the proposed contribution; they do not validate our dataset or outcomes.

| Prior work (verified source) | Existing contribution | What is / is not distinctive here |
| --- | --- | --- |
| Swathi, Altalbe and Kumar, *QuCNet* (IEEE Access 2024), DOI [10.1109/ACCESS.2024.3367806](https://doi.org/10.1109/ACCESS.2024.3367806) | Original thyroid-ultrasound quantum-inspired pipeline and reported classification results | **Direct reproduction target**. Our contribution is a traceable *method-fidelity and evaluation audit*, not the invention of quantum convolution or a direct numerical replication of their test set. |
| Eslami et al., *Reproducibility report: quantum machine learning methods in fundus analysis* (*Eye*, published 2 August 2025), DOI [10.1038/s41433-025-03916-w](https://doi.org/10.1038/s41433-025-03916-w) | Medical-imaging QML reproducibility report with open-source quantum/classical comparisons | Reproducibility reporting and classical comparison **are not novel by themselves**. Our narrower setting is QuCNet/DDTI thyroid imaging and case/ROI metadata provenance, not ophthalmic fundus imaging. |
| Hashmi et al., *Do Quantum Kernels Improve Medical Image Classification?* (*J. Imaging Informatics in Medicine*, published 21 September 2026), DOI [10.1007/s10278-026-02270-x](https://doi.org/10.1007/s10278-026-02270-x) | Leakage-controlled quantum-kernel/classical RBF-SVM comparisons on MRI and BreastMNIST with matched features and multiple seeds | Controlled comparisons, source separation and multi-seed sensitivity **have close precedent**. Our methods use a fixed four-qubit patch-expectation map, not the same quantum kernel, and examine thyroid labels and archival implementation fidelity. |
| Duong et al., *Does the Quantum Layer Help? A Leakage-Aware Benchmark of Hybrid Quantum–Classical Models for Brain Tumor MRI Classification* (*Diagnostics*, 2026), DOI [10.3390/diagnostics16213485](https://doi.org/10.3390/diagnostics16213485) | Matched classical/quantum controls, leakage and patient-source auditing, ablations and uncertainty tests in brain MRI | Another strong **methodological predecessor**. Do not describe leakage-awareness or questioning quantum advantage as an unprecedented finding. Our archival QuCNet/TI-RADS replication question and exact code/data provenance differ; our patient identity remains unverified. |
| *A systematic literature review of quantum machine learning for medical: trends, datasets, topics, and methods* (*IJ Cognitive Computing in Engineering*, 2026), DOI [10.1016/j.ijcce.2026.05.002](https://doi.org/10.1016/j.ijcce.2026.05.002) | Survey identifies widespread simulated QML, limited external clinical validation and benchmarking limitations | Supports the **importance** of careful benchmarking, not priority or novelty of any particular audit. |

### Candidate defensible scholarly contribution

A **QuCNet-specific, data-provenance-aware independent reproduction and methodological audit** combining (a) exact source-versus-implementation disclosure; (b) case-ID grouped, ROI-specific re-evaluation rather than silently treating source-paper image scores as comparable; (c) matched 64-feature classical versus *fixed simulated* quantum maps across four classifier families; (d) seed-wise sensitivity and probability-quality auditing; and (e) identification of image/annotation duplicates, label-provenance limits, and loss-prone quantum cache-key behavior. The novelty, if judged sufficient by the eventual venue, lies in this **specific combined evidence about the QuCNet/DDTI reproduction**, not in generic use of QML, random forests, calibration, or repeated cross-validation.

### Reviewer challenges not resolved by more tuning

1. **Specificity versus novelty:** several 2025–2026 studies already implement stringent leakage-aware quantum/classical comparisons. Any novelty statement must name the thyroid/QuCNet-specific evidence, be qualified by the completeness of related-work search and resist generic 'first' language.
2. **Comparator ceiling:** 203 case-ID development examples with labels derived from TI-RADS source records and weak macro-F1 cannot support clinical utility. Five repeated splits on the *same cases* quantify fold sensitivity only.
3. **Identity and ground truth:** patient-level uniqueness and pathology verification are not established; both remain essential for stronger clinical inference.
4. **Deployment and quantum claims:** no hardware experiment or demonstrated speedup; fixed simulated circuit weights prevent attribution to trainable-quantum advantage.
5. **Audit transparency:** the qualitative source-fidelity 55/100 rubric is judgment-based; never present it as accuracy, a formal quality score, or proof that the published authors' method is wrong.
6. **Release governance:** third-party DDTI JPG/XML are publicly tracked despite old repository wording; legal/consent and image/XML identifying-material review need documented resolution before unqualified public redistribution claims.

**Decision:** This body of work supports a possible carefully scoped *methodological reproducibility study*. It does **not** currently justify claims of novel quantum architecture, statistically established quantum advantage, clinically reliable TI-RADS grading, or completely leakage-free patient evaluation. No new unfocused training is recommended. Literature positioning, authorship/guide review, source-data rights, actual independent validation (if feasible), and final IEEE figure/layout checks remain separate gates.

## Classwise and calibration audit closure — 2026-10-28

The original EXT-STABILITY-001 archive was checked again using the stdlib-only `05_Source_Code/verify_stability_artifacts.py`, with an additional **classwise failure-mode assertion**. Across all five prespecified case-fold seeds (42, 101, 202, 303, 404), classical and fixed-quantum **random forests both achieved 0/10 TI-RADS 3 recall in every repeat**. This represents **the same ten recorded case IDs** reassigned to development folds; it is *not* 50 independently sampled patients. Fixed-quantum RF was below the 57/203 majority-label reference at seed 202 (55/203); its average accuracy advantage over the matched classical RF therefore does not establish category-level adequacy or a robust clinical gain.

The independent `05_Source_Code/verify_calibration_artifacts.py` now recalculates **all eight** six-class classifier confusion matrices and metrics plus **1,624** saved case-probability records, multi-class Brier score, negative log-likelihood, predictive entropy, max-class confidence, count of high-confidence errors, **ten-bin expected calibration error**, and **80 per-bin calibration records**. The same official GitHub Actions workflow checks calibration and repeated case-fold stability with **sparse checkout excluding all raw DDTI JPG/XML**, and successfully completed the integrated run: [GitHub CI evidence](https://github.com/karthikaxiom/QCNN_Thyroid_Paper_Reproduction/actions/runs/40662104959).

**Limits:** The automated check independently reproduces *stored predictions and derived summaries*, not the quantum feature simulations, original trained classifiers, provenance or actual patient uniqueness. Weak TI-RADS 3 sensitivity is descriptive, not a proven mechanistic consequence of specific circuit encoding or features. The Kaggle listing itself calls DDTI open-access for scientific use but uses the license category "Other (specified in description)." Research access and unrestricted third-party republication should remain separate documentation questions. The latter does not obstruct continuing ethical analysis or independently checked publication of derived aggregate findings.

## Independent thyroid validation feasibility — public sources checked 2026-10-28

**Gate 2 decision — current six-class task: NO-GO for direct external testing on screened cohorts.** See the [full external-validation compatibility review](EXTERNAL_VALIDATION_FEASIBILITY_GATE.md) for primary-source comparisons across ThyroidXL, Stanford cine-clips, TN3K and TN-SCUI2020, access/governance distinctions, and a separate conditional prospective study protocol. The main scientific obstacle is incompatible target taxonomies: source labels 2/3/4a/4b/4c/5 are **not directly interchangeable** with ACR TR1–TR5 or pathology-confirmed binary diagnosis. The public ThyroidXL dataset card also contains internally contradictory patient/image counts, which must be reconciled from an authorized version before any new experiment. **No external data were downloaded, no model externally evaluated, and no external accuracy is claimed.**

This is **source screening only**. No data were downloaded, no model trained, no outside ground truth was inspected, and no external validation result can yet be reported.

| Potential data source | Verified resource and benefit | Critical compatibility/rights gate |
| --- | --- | --- |
| **ThyroidXL** (MICCAI 2025), official [project](https://github.com/VNPT-AI-Official/ThyroidXL) and [published paper](https://doi.org/10.1007/978-3-032-05182-0_60) | Official project describes **11,635 images from 4,093 patients**, benign/malignant diagnoses supported by clinical cytology/pathology, patient-level benchmark partition and ACR TI-RADS scoring. This is a much stronger candidate for independent patient-grouped medical evaluation than repeated DDTI tuning. | The **binary pathology endpoint and ACR TI-RADS scale are not the six source labels 2, 3, 4a, 4b, 4c, 5** used here. A prospective, clinically justified endpoint/harmonization and model adaptation would be necessary; **cannot directly claim external six-class transfer**. Verify dataset card, licenses and permitted use. An already-inspected test set cannot be repurposed for model selection. |
| **Stanford thyroid ultrasound cine-clip** ([institutional data page](https://aimi.stanford.edu/datasets/thyroid-ultrasound-cine-clip), DOI [10.71718/7m5n-rh16](https://doi.org/10.71718/7m5n-rh16)) | **167 patients, 192 biopsy-confirmed nodules**, with TI-RADS descriptors, segmentation, clinical information and histopathology. Offers documented patients and clinical labels. | Acquisition is *cine-clip videos*, not the same processed JPG ROIs; explicit access/consent/reuse conditions must be checked. The paper's six-category labels cannot be presumed compatible. Substantial domain and protocol changes needed. |
| **TN3K** ([original release](https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation)) | Large separate ultrasound image source, commonly used for nodule masks and some benign/malignant experiments. | Dataset versions and patient grouping, pathology status, actual image-class metadata and distribution rights must be audited before any comparison. Masks or binary categories alone do not supply the six QuCNet-style TI-RADS labels. |

**Scientific decision:** ThyroidXL is the best *candidate* for a distinct future pathology-verified clinical research direction, **not a plug-in test dataset for the existing six-class model**. A standalone independent study would require owner-approved access, dataset-to-target mapping, de-identification, patient-level leakage audit, prespecified protocol and sufficiently strong classical comparators. Do not reframe existing DDTI development scores as external validation. Neither availability of these datasets nor their advertised results guarantees compatible licensing or an evaluable endpoint.

## Final archived-evidence verification gate — 2026-10-28

**Status:** Four independent, read-only, Python-standard-library verifiers have been committed and run successfully through GitHub Actions on Linux; the checkout includes derived numerical CSV/JSON and ROI/case *metadata only*, not raw medical JPG/XML. Source [workflow](../.github/workflows/saved-evidence-audit.yml); the [verified passing run](https://github.com/karthikaxiom/QCNN_Thyroid_Paper_Reproduction/actions/runs/40663165948).

| Verification gate | Source-backed work | Result | What this does **not** prove |
| --- | --- | --- | --- |
| `verify_stability_artifacts.py` | 4,060 saved case-level OOF records, 20 confusion matrices, five fold seeds, 812 seed-42 archived probability comparisons, paired differences, across-seed means/SDs; TI-RADS 3 recall and majority-baseline assertions | **PASS** | Independent re-training, fold-independent patient cohorts, quantum superiority |
| `verify_calibration_artifacts.py` | 1,624 case-level probabilities across eight classifiers, eight confusion matrices, multiclass Brier/NLL, entropy, confidence errors, ten-bin ECE and 80 calibration-bin rows | **PASS** | Clinical calibration or independent prospective diagnosis |
| `verify_master_inventory.py` | All 21 rows of `MASTER_VERIFIED_RESULTS.csv` compared with five original result JSON archives and underlying confusion-matrix denominators, accuracy, balanced accuracy, macro-F1, evidence classification | **PASS** | Source-label correctness, leakage-free patient identity, test generalization |
| `verify_frozen_case_splits.py` | Phase 12 case assignments against all 440 Phase 13 ROI metadata records: 203/311 train, 44/58 validation, 44/71 historical test; unique ROIs, case-ID partition exclusivity, class mapping, geometry and pixel-summary validity | **PASS** | Different case IDs are different patients; source JPG/XML inspection or clinical annotation accuracy |

**Evidentiary boundary:** These four verifiers check *archived data consistency* with independent simple computations and reject corrupted claims. They do not regenerate initial quantum features, retrain CNNs, independently prove the labeling/consent provenance or supply uninspected clinical test evidence. The historical Phase 14 failure, mismatched source-paper protocol, fixed simulated quantum mapping, tiny TI-RADS 3 support, and weak model metrics remain visible and unaltered.

### Scientific research freeze and review decision

**Completed within the agreed scope:** Original implementation versus source-paper fidelity analysis, frozen case-ID and ROI inventory, exploratory historical evaluations, feature/preprocessing/deduplication/cache audits, matched eight-classifier comparison, classwise ROC/sensitivity analyses, probability reliability, five-seed fold-assignment sensitivity, independently checked figures/source numerical data, and automated results regression tests.

**Still external or manuscript-stage:** Documented dataset image/XML redistribution terms and privacy checks; evidence establishing distinct patient identities; genuinely compatible, independently acquired clinical validation; institutional guide input on novelty and contribution; final authorship/disclosure/citation decisions; print-size figure proof and compilation. These are **not** silently marked completed and no unverifiable success is claimed. In particular, an external thyroid ultrasound dataset with different TI-RADS/pathology targets cannot simply be added as a comparable untouched six-category test.

**Recommendation:** Freeze these empirical results. Present a narrowly framed independent QuCNet/DDTI methodological reproducibility study for guide review. Do **not** resume open-ended architecture tuning or claim publication acceptance/medal entitlement. **No IEEE manuscript or Overleaf work is authorized yet**; this is an evidence-and-gates closeout record only.


## Gate 3 — Clinical source-data governance (28 October 2026)

The full [DDTI data rights, privacy and publication gate](DDTI_DATA_RIGHTS_AND_PRIVACY_GATE.md) is committed and linked from the project README. **Scientific research access: supported by original university listing and Kaggle. Raw-file redistribution right: not established by a separately stated license.** Kaggle calls the mirror open-access for science but assigns **Other (specified in description)** rather than an explicit CC0/CC-BY release. The original DDTI article is Pedraza et al., Proc. SPIE 9287, 92870W (2015), DOI [10.1117/12.2073532](https://doi.org/10.1117/12.2073532); cite it separately from the 2024 QuCNet source.

The public repository already tracks 480 ultrasound JPGs and 390 XML annotations. A **four-file XML tag-name sample** showed age and sex and clinical report fields; no statement about actual patient identifiability is supported. **Subsequent researcher-reported local whole-XML structural screen:** the committed scanner parsed **390/390** files, with **0 parse errors**, **0 oversize files**, `age` and `sex` each occurring **390** times, **0 identifier-like field names flagged**, **21 nonempty narrative fields**, and **0 email/phone pattern matches**. See the [dated local XML result record](DDTI_XML_LOCAL_PRIVACY_SCREEN_RESULT.md). This is reported local output, **not independent CI execution on the real XMLs**. All 21 populated narrative fields still require a local human privacy review; identifying content embedded in JPGs and patient consent remain outside the scanner's scope. The added `audit_ddti_xml_fields_local.py` is a non-disclosing aggregate-only scanner for the researcher's locally held XML archive. Its **synthetic CI test passed** ([workflow run](https://github.com/karthikaxiom/QCNN_Thyroid_Paper_Reproduction/actions/runs/40665210131)), the real 390-XML structural run is now researcher-reported as complete, while narrative-field and image-by-image privacy checks have not yet been completed. No raw images, XML or historic Git objects were deleted, overwritten, migrated or re-licensed.

**Supplementary independent GitHub Actions XML check (28 October 2026):** The [automated XML triage report](DDTI_XML_AUTOMATED_TRIAGE_RESULT.md), verified in [run 40668545283](https://github.com/karthikaxiom/QCNN_Thyroid_Paper_Reproduction/actions/runs/40668545283), replaces the manual 21-answer prompt. The rule-based scanner parsed all **390** archived XMLs and screened **21** nonempty clinical narrative fields, with **2** conservative follow-up flags (one date/year, one long entry) and **19** nonmatching fields. No actual source text or individual case IDs appeared in the output. This is **first-pass heuristic detection only**; a nonmatching narrative is not certified de-identified, and human image privacy checks, consent and raw-file rights remain unverified.

**Gate 3 disposition: documentary evidence and safety tooling prepared; unrestricted third-party raw-data release clearance and comprehensive privacy review remain open.** Derived statistical research and approved code can continue. The IEEE article and full figure production are still scheduled last; any AI-generation disclosure should follow actual use and then-current IEEE rules, rather than being concealed or fabricated.

## Gate 4 — Publication figure evidence and layout (28 October 2026)

The [figure evidence, selection and IEEE placement gate](IEEE_FIGURE_EVIDENCE_AND_PLACEMENT_GATE.md) inventories the candidate main/supplementary figures and tables. The `05_Source_Code/verify_svg_figure_assets.py` script checks **24 SVGs** (15 experimental, 9 README explanatory) and verifies the **15 experimental SVG/PNG/PDF export triplets** by structural XML parse and file existence. It passed in a sparse [GitHub Actions run](https://github.com/karthikaxiom/QCNN_Thyroid_Paper_Reproduction/actions/runs/40666066419), without reading original ultrasound images. All historical plots and numerical outputs remain untouched.

**Print blockers identified:** `exp_models_001/matched_classifier_comparison.svg` is **891.34 pt wide** and `exp_stability_001/case_fold_stability.svg` is **726.64 pt wide**; simply scaling them to an indicative 516 pt IEEE double column yields relative sizes **0.579** and **0.710**, respectively. They require legibility redesign/re-export and true-size print proof. Source asset existence **does not** prove that every figure point/caption value matches its metrics, that colors work in grayscale, or that text passes print-size review. Existing SVGs should not be called final paper figures. Original source files must be preserved; any revised publication exports should be derived separately.

**Gate 4 disposition:** structural/vector/archive verification **PASS**; exact IEEE manuscript-layout print preflight **OPEN until the article stage**. Research figure selection, source mapping and nonclaims are documented. Actual manuscript writing remains reserved for last.

## Print-scale figure creation follow-up — 28 October 2026

Two **new source-generated 7.16-in full-width figure candidates** were produced by [`05_Source_Code/build_ieee_evidence_figures.py`](../05_Source_Code/build_ieee_evidence_figures.py), rather than replacing historical publication plots. The figures cover (i) all eight matched seed-42 classical/fixed-quantum classifiers, showing accuracy, six-class balanced accuracy and macro-F1, and (ii) per-seed accuracy of the logistic and random-forest arms over seeds 42, 101, 202, 303, 404, with the majority-case reference. The generator checks 203-case denominators, eight original confusion matrices, 20 per-seed results and the seed-42 source agreement before exporting SVG/vector PDF/300-dpi PNG.

[GitHub Actions figure build 40669696948](https://github.com/karthikaxiom/QCNN_Thyroid_Paper_Reproduction/actions/runs/40669696948) **passed** and uploaded a SHA-256 manifest and six derived files. The permanent, reproducible [production record](IEEE_PRINT_FIGURE_GENERATION_RECORD.md) defines exact inputs, interpretation, regeneration and remaining review. **Full-size journal-template visual proof remains open**, as do the separate dataset privacy/release and clinical-evidence limitations. No manuscript was written.
