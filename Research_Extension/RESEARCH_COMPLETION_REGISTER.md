# Research extension — execution and verification register

**Status (2026-10-28):** Prespecified empirical experiments complete; independent scientific, figure and citation closeout ongoing. Manuscript preparation remains deferred.

This register complements the existing `README.md` and `Project_Log.txt`. It does not replace or revise the historical Phase 0–18 record.

## Evidence boundaries

- Preserve all historical datasets, source scripts, checkpoints, results, and split assignments.
- The six labels are source-recorded TI-RADS 2, 3, 4a, 4b, 4c and 5; the historical binary labels are TI-RADS-derived suspicion proxies, not verified pathology diagnoses.
- Case-ID separation is documented; unique patient identity across case IDs has not been independently established.
- Restrict feature and model exploration to the original Phase 13 training cases. The original validation set is development evidence; previously inspected test results are historical findings, not an untouched test for newly selected models.
- Never report synthetic script tests as real dataset results. Every real observation must have a reproducible source artifact.

## Recorded experiments

| ID | Research question | Evidence and status |
| --- | --- | --- |
| EXT-BN-001 | Do saved Batch Normalization statistics explain Phase 14 inference collapse? | Completed; source, metrics and predictions on extension branch. Recalibration changes validation accuracy from 2/58 to 11/58, still below the majority baseline. |
| EXT-DEV-001 | Can simple ultrasound intensity/gradient features classify six TI-RADS categories? | Completed; training-only fivefold case-stratified CV. Pooled case accuracy 22.17%; majority baseline 28.08%. |
| EXT-DATA-001 | What is the provenance of binary labels and the case-ID integrity? | Completed locally; 349 TI-RADS-labeled images and 131 unlabeled; zero case-ID overlap across frozen partitions; binary labels are derived proxies. Check whether its local files were committed before attempting synchronization. |
| EXT-META-001 | How much categorical information resides in ROI annotation geometry? | Completed; training-only CV; pooled case accuracy 19.21% and balanced accuracy 22.70%. |
| EXT-FUSION-001 | Do image and annotation-geometry features provide complementary information? | Script committed; **real-data run pending**. |

## Remaining research gates

1. Execute EXT-FUSION-001 on the actual local Phase 13 training arrays. Preserve the script version, environment, metadata, out-of-fold predictions, confusion matrices and paired-case counts.
2. Independently recompute all new metrics from prediction tables. Produce class-level uncertainty and paired comparisons with clearly exploratory interpretations.
3. Audit ROI preprocessing, annotation validity, cross-case image duplication and dataset/license provenance; document limitations before proposing a new architecture.
4. Make one prespecified, resource-matched classical/quantum extension only if supported by development-set evidence. Preserve failures and use multiple seeds if feasible.
5. Review all equations, denominators, tensor shapes, model parameter counts, original-versus-extension tables, figures, citations and statistical claims.
6. Only after research gates pass, improve the existing README by a **narrowly appended and cross-referenced** extension section and begin final IEEE manuscript preparation.

## Quantitative reporting requirements

For each model report sample and case counts, accuracy, six-class balanced accuracy, macro-F1, full confusion matrix, predicted-class counts, confidence measures, random seeds and source hashes. Maintain image-level and case-level results as separate estimands.

Exploratory McNemar comparisons operate on paired case correctness, and case-bootstrap confidence intervals resample entire cases. Neither establishes clinical performance or captures uncertainty from alternate model fits and CV partitions.

## Continuation procedure

The extension study is maintained on the research branch. Each result is backed by committed scripts, input hashes, prediction tables and verification artifacts; no experiment is assumed to have run unless its output is recorded.

## EXT-DUPLICATE-001 — exact raw image duplication audit

The raw image Git-object audit found **480 files but 471 unique byte contents**. Six exact-duplicate groups contain 15 files (nine redundant copies); two groups span different case IDs. No exact-byte duplicate group spans two of the assigned train/validation/test partitions.

- Five identical files: cases 150, 233, 323, 326, 349; only case 233 belongs to the strict Phase 12 training split.
- Two identical files: cases 353 and 396; only case 353 belongs to the strict Phase 12 training split.
- The source CSV provides TI-RADS for the assigned training cases, while the non-assigned copies in these groups lack TI-RADS values. This is a provenance issue, not a demonstrated conflicting class-label pair.

Evidence: `07_Results/exp_duplicate_001/remote_git_blob_audit.json`. A separate, independently reproducible local SHA-256 audit script is `05_Source_Code/exp_duplicate_001_raw_image_audit.py`, with outputs under `07_Results/exp_duplicate_001/local_sha256/`.

**Limitation:** Git blob hashes establish exact file-byte equality. They do not detect near-duplicates after re-encoding and do not establish patient identity. The raw image hashes of test-assigned cases were checked for integrity only; no raw images were decoded, test predictions consulted, or models selected using test data.

**Synchronized:** The real-data EXT-FUSION-001 CV, case-level paired statistical results, EXT-INTEGRITY-001, local SHA-256 duplicate audit and paired historical figures were committed and pushed as `3c615cb`. All 29 evidence files are repository-backed. Original Phase 0–18 records remain unchanged.

## EXT-PAIR-001 — historical quantum/classical matched predictions

Read-only recomputation of stored Phase 9 quantum and Phase 10 classical six-class image-level predictions (70 aligned samples) found 13/70 and 17/70 correct respectively. Both were correct for 4 samples, quantum-only for 9, classical-only for 13 and both wrong for 44. The paired accuracy difference was +5.71 percentage points for classical, with exploratory exact McNemar two-sided p = 0.5235. There is **no supported claim of superiority**.

The quantum softmax entropy averaged 1.791524 nats versus the uniform-six-class maximum of ln(6) = 1.791759; classical averaged 1.747455 nats. These scores do not establish calibrated uncertainty.

Sources: `Research_Extension/verification/phase9_phase10_paired_historical.json` (remote result) and `05_Source_Code/exp_pair_001_historical.py` (independent locally executable figure/report script). The historical image-level protocol retained case overlap and does not support a new case- or patient-independent evaluation claim.

## EXT-FUSION-001 — verified development comparison

Real 5-fold case-separated CV on 203 training cases (311 ROIs) gave the following **case-level** results:

| Model | Accuracy | Balanced accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| Majority | 0.28079 | 0.16667 | 0.07308 |
| Image-feature logistic regression | 0.22167 | 0.20102 | 0.19748 |
| Geometry-only logistic regression | 0.19212 | 0.22703 | 0.17845 |
| Image + geometry fusion | 0.22660 | 0.21169 | 0.20411 |

For fusion versus image-only, paired case accuracy improved by only 1/203 = 0.49 percentage points; the exploratory 95% paired-case bootstrap interval was −2.46 to +3.94 percentage points, and exact McNemar p = 1.00. No reliable fusion advantage is established. Full source: `07_Results/exp_fusion_001/metrics.json` and `07_Results/exp_fusion_001_statistics/statistical_audit.json`. These are training-only development comparisons, not untouched test results.

The independently computed historical metrics check was passed by all four available prediction tables: `07_Results/exp_integrity_001/historical_metrics_audit.json`.

## EXT-PADDING-001 — training ROI square-padding geometry

The Phase 13 source `05_Source_Code/phase13_prepare_roi_dataset.py` squares each rectangular ROI using a **median-intensity constant fill**, then downsamples to 64 × 64. A zero-pixel count is therefore **not** a padding-area estimate.

A read-only, source-metadata calculation for 311 training ROIs (203 cases) found:

- Median theoretical pre-resize padding fraction: **25.6%**; mean **25.58%**.
- **309/311 ROIs** require nonzero square padding; **157/311** have >25% padded area and **16/311** have >50%.
- Case-level median-of-ROI padding was **26.60%** across the 203 cases.
- TI-RADS category median ROI padding fractions range from **22.73%** (5) to **29.38%** (4a).

These quantities follow `1-min(width,height)/max(width,height)` and describe geometric area **before** resizing; they do not identify exact interpolated pixels or prove causal effects on classifier performance. Evidence: `07_Results/exp_padding_001/remote_metadata_precheck.json`. The independent local statistics-and-figure script is `05_Source_Code/exp_padding_001_metadata.py`; execution on local training metadata is pending.

**Research decision:** Do not infer that padding was absent from zero-valued-pixel counts. Prioritize one controlled, training-only preprocessing comparison before considering a further expensive quantum architecture run.

## EXT-PREPROC-001 — prespecified sensitivity experiment (execution pending)

Code: `05_Source_Code/exp_preproc_001_ablation.py`.

This development-only comparison tests three representations of the same 311 Phase 13 training ROIs, using identical case-stratified fivefold partitions, an unchanged 128-feature pooled image representation, and the same logistic-regression pipeline from EXT-FUSION-001:

1. Reference: original median-intensity square padding and `cv2.INTER_AREA` resizing.
2. Alternative: `cv2.BORDER_REFLECT_101` square padding.
3. Sensitivity-only control: direct anisotropic resize without square padding (changes aspect ratios).

Before fitting, the script reconstructs every reference ROI from its raw training source and metadata, checks reconstruction drift, and requires the original image-only OOF predictions to replicate the committed `EXT-FUSION-001` evidence. It saves case and ROI metrics, paired case-bootstrap intervals, exact McNemar results, full OOF probabilities, source hashes and vector/raster charts.

**Interpretation guardrail:** Even a difference in these exploratory comparisons would not prove that padding caused the Phase 14 CNN collapse; the architecture, representation and training objective differ. No new test set is available for confirmatory claims.

**Pending local commands:** Run `exp_padding_001_metadata.py` (if its outputs are absent), then `exp_preproc_001_ablation.py`. Preserve results as separate committed experiment evidence. The intentional two-file supporting-paper cleanup should be committed separately from research experiments.

## EXT-PREPROC-001 — initial real-data execution (pending result synchronization)

On 203 training cases, fivefold case-separated out-of-fold results reported from the local experiment:

| Preprocessing | Case correct / 203 | Accuracy | Balanced accuracy | Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Phase 13 median constant fill | 45 | 0.22167 | 0.20102 | 0.19748 |
| Reflection padding | 51 | 0.25123 | 0.2276 | 0.2230 |
| Direct anisotropic stretching | 57 | 0.28079 | 0.2753 | 0.2683 |

All 311 original padded ROIs reconstructed **exactly** (max pixel error 0.0); the original image-only OOF predictions also matched the committed prior experiment.

Relative to median padding, reflection changed case accuracy by +6/203 = +2.96 percentage points (exploratory case-bootstrap 95% CI −2.46 to +8.37, exact McNemar p = 0.392). Direct stretch changed accuracy by +12/203 = +5.91 percentage points (exploratory 95% CI −0.49 to +12.32, exact McNemar p = 0.104). Neither comparison establishes a statistically reliable advantage. The 57/203 overall accuracy for direct stretching equals the training-CV majority-class accuracy but exceeds it on balanced accuracy and macro-F1.

**Evidence status:** The complete results, out-of-fold prediction tables and vector figures were committed in `180917e` and independently verified against the uploaded JSON on GitHub. These are exploratory development CV findings, not results from an untouched test partition. Direct stretching changes aspect ratios and is not an automatically safe medical imaging method.

## EXT-PADLINK-001 — controlled descriptive follow-up (completed)

Code: `05_Source_Code/exp_padlink_001_association.py`. This performs **no training**. It joins existing EXT-PREPROC-001 out-of-fold case predictions with Phase 13 training-only ROI metadata, computes median square-padding fraction per case, then examines paired true-label log-loss change and accuracy change for reflection padding and direct stretch versus the original median fill. The analysis includes case-bootstrap intervals, exploratory Spearman associations and TI-RADS counts in padding quartiles, with source-hashed outputs and vector figures.

The purpose is to determine whether greater original padding is associated with a greater benefit from alternative preprocessing. Any observed association could still be confounded by lesion morphology, case composition, ROI sizes, acquisition and annotation practices. This is a hypothesis-generating analysis, not proof of a mechanism. No test or validation inputs are accessed.

**Evidence status:** Complete association outputs and figures committed in `180917e`; the intentional supporting-paper duplicates were removed in a distinct commit `a0baf20`, preserving the remaining base and supporting paper.

## EXT-PADLINK-001 — association results and interpretation (verified)

Real training-only case-level out-of-fold association results were verified from `07_Results/exp_padlink_001/association_metrics.json` (203 cases). Median original-padding amount did **not** show a detectable monotonic relationship with log-loss improvement:

| Alternative preprocessing | Mean case log-loss gain (positive = better) | Exploratory case-bootstrap 95% CI | Spearman ρ: padding vs gain | p-value |
| --- | ---: | ---: | ---: | ---: |
| Reflection | −0.05944 | [−0.14323, +0.02674] | +0.07163 | 0.30983 |
| Direct stretch | +0.01221 | [−0.08941, +0.11624] | −0.01070 | 0.87960 |

Mean case-level log losses were median original 1.88514, reflection 1.94458, direct stretch 1.87293. Although alternative preprocessing changed case accuracy in the original fivefold OOF experiment, neither the paired accuracy confidence intervals nor the log-loss-gain intervals exclude zero. Reflection had *worse* mean case log loss despite improved classification accuracy.

No evidence currently demonstrates that the **amount** of median-filled padding caused the classification failure. Multiple exploratory tests were conducted without multiplicity correction. These claims are about training-partition OOF predictions, not about an untouched patient-independent test set.

## Bounded remaining empirical work

Only two targeted gates remain before manuscript readiness assessment:

1. **Dataset integrity:** Candidate near-duplicate scan across raw files using deterministic perceptual hashes and independent comparison statistics. Treat matches as flagged candidates for manual review, not proof of leakage/patient overlap. Preserve all frozen train/validation/test partitions.
2. **Matched quantum-vs-classical study:** A prespecified, affordable training-only case-stratified comparison of the *existing fixed, simulated four-qubit quantum feature mapping* against appropriately matched classical features on identical cases/folds. Explicitly distinguish this feature-map study from full original QuCNet reproduction. Do not choose a model via previously inspected test data.

After these, close the experimental register, finish mathematical/figure/limitations checks, improve the existing README only where verified, and prepare the manuscript. Avoid open-ended model searches.

## Independent rederivation of EXT-PREPROC-001

The uploaded `07_Results/exp_preproc_001/oof_predictions.csv` contains 311 out-of-fold ROI predictions **per model**, aligned to 203 unique case IDs. The verification re-aggregated the six predicted probabilities by case, constructed the 6×6 confusion matrices and recalculated accuracy, balanced accuracy and macro-F1. All nine resulting scores match `07_Results/exp_preproc_001/metrics.json` to numerical precision:

- Median-original: 45/203, accuracy 0.221675, balanced 0.201021, macro-F1 0.197484.
- Reflection: 51/203, accuracy 0.251232, balanced 0.227623, macro-F1 0.222953.
- Direct stretch: 57/203, accuracy 0.280788, balanced 0.275300, macro-F1 0.268313.

This confirms metric/data consistency, not improvement significance or external validity.

## EXT-NEARDUP-001 — prespecified perceptual candidate screen (pending local run)

Code: `05_Source_Code/exp_neardup_001_integrity.py`, committed after result synchronization. It checks 480 raw files using deterministic 64-bit pHash and dHash measurements with frozen candidate thresholds and reports candidate pairs, exact same-byte indicators and whether candidate files span different assigned Phase 12 splits. This scan **decodes raw images from all splits, including test, solely for data-integrity screening**. It does not access predictions or train models. Do not interpret a flagged candidate as confirmed patient duplication, and manually review any cross-split candidates before drawing a leakage conclusion. Resizing, crop changes and shadows can cause perceptual false positives/negatives.

After its outputs are preserved, the only planned new model investigation is a bounded, matched feature-map audit involving the existing simulated four-qubit filter and a classical comparator, on the same training-only case folds. The final paper remains deferred until the methods, figures and results are verified.

## Final bounded experiment — EXT-QMAP-001 (completed; independently verified)

Code: `05_Source_Code/exp_qmap_001_fixed_feature_compare.py`.

A fixed feature-map comparison reproduces the **existing reference** four-qubit, two-layer, seeded RX/RY/CNOT-ring/Pauli-Z circuit from `utilities/quantum_utils.py`. The circuit is simulated on `default.qubit`, with no learned quantum parameters or hardware claims. For speed, it is applied to sixteen 2×2 patches of the **same 8×8 grayscale-pooled** training ROI used by the classical input, yielding 64 fixed quantum expectation values per ROI; the classical arm uses the 64 corresponding input intensities. The script cross-checks batched PennyLane outputs against three calls of the original single-patch filter and aborts if they differ.

Both arms receive the exact 311 ROI records, 203 cases, identical fivefold case-stratified partitions and logistic regression with `C=0.1`, `class_weight=balanced`, with scaling fitted per fold. Source hashes, full OOF class probabilities, six-class confusion matrices, paired bootstrap interval, exact McNemar test and vector/raster comparison charts are saved.

**Scope:** A matched *fixed mapping* audit, not a claim of optimized quantum learning, quantum advantage, reproduction of the original high-resolution QuCNet feature map, or patient-independent clinical validation. Historical validation and test sets remain untouched. Results must be judged against a no-information baseline and their paired uncertainty, not only overall accuracy. The classical and quantum representations have equal feature dimension but different nonlinear transformations, so fair dimension matching does not eliminate all representational confounds.

## Visual review of unresolved near-duplicates

From `07_Results/exp_neardup_001/candidate_pairs.csv`, 23 cross-case pairs were flagged, including **three** crossing different assigned splits (none exactly identical):

- `277_1.jpg` [validation] versus `348_1.jpg` [train]: pHash 8, dHash 9, normalized correlation 0.8625.
- `177_1.jpg` [train] versus `313_1.jpg` [test]: pHash 8, dHash 12, normalized correlation 0.8228.
- `305_1.jpg` [test] versus `34_1.jpg` [train]: pHash 8, dHash 12, normalized correlation 0.6880.

All three satisfy the inclusive **boundary** of the primary pHash criterion (distance 8). The evidence does not confirm duplicates, leakage or common patient identity. Stored verification file: `Research_Extension/verification/near_duplicate_cross_split_candidates.json`.

Local-only figure generator: `05_Source_Code/exp_neardup_001_review_sheet.py`; it creates a three-row contact sheet under `07_Results/exp_neardup_001/review_local/`, to be reviewed visually before any confirmation. Avoid uploading raw review images until identifying annotations and redistribution terms have been checked.

The quantum/classical feature-map audit was completed and the three flagged cross-split raw-frame pairs were visually inspected. This is not a patient-identity audit. Continue final mathematical, methodological, figure, citation and publication-readiness review; manuscript work remains deferred.

## EXPERIMENTAL CLOSEOUT — EXT-QMAP-001 and visual review

`EXT-QMAP-001` real training-only paired study was completed and pushed as commit `0868955`. The independent verification reconstructed all 203 case predictions by averaging six class probabilities from the 311 ROI out-of-fold rows per model. All stored case-level confusion matrices, accuracies, balanced accuracies and macro-F1 values matched.

- Equal-dimensional 64-feature classical pooled image: **54/203 cases (26.60099%)**; balanced accuracy 25.22328%; macro-F1 24.26322%.
- Existing fixed four-qubit quantum simulation on sixteen pooled 2×2 patches: **43/203 cases (21.18227%)**; balanced accuracy 21.90883%; macro-F1 20.67882%.
- Quantum-minus-classical case accuracy = **−11/203 (−5.4187 percentage points)**; classical-only correct 21, quantum-only correct 10, both correct 33, both wrong 139; exact two-sided McNemar p = **0.0707555**. Exploratory paired case-bootstrap interval [−10.84, 0.00] percentage points; no confirmatory superiority claim.
- Fixed quantum circuit checked against original reference `utilities/quantum_utils.py` (maximum expectation-value disagreement 1.2572×10⁻⁸). This is **not** training a quantum circuit and cannot demonstrate quantum speedup.

Evidence `07_Results/exp_qmap_001/metrics.json`, `07_Results/exp_qmap_001/oof_predictions.csv`, and independent read-only verification `Research_Extension/verification/exp_qmap_001_independent_metrics.json`.

The three pre-flagged cross-split raw ultrasound pairs were reviewed directly. None appeared to be an identical raw frame. The case/patient identity question cannot be resolved from imagery alone; maintain this limitation. One full raw test image (`313_1.jpg`) has visible TI-RADS 4C text in its bottom border, outside the saved cropped test ROI y-range 12–190. Raw-overlay effects cannot be generalized from this one inspection. Evidence `Research_Extension/verification/near_duplicate_visual_review.json`. No raw ultrasound image has been republished in the research register.

**Figure and arithmetic checkpoint:** Four stored PNG figure exports were visually checked for labels, axes and legibility at native display size (fixed-feature comparison, preprocessing sensitivity, padding/log-loss relationship, and paired fusion intervals); all have corresponding vector-source exports. IEEE final printed-column rendering, permissions/citations and source-paper equation correspondence remain to be checked.

**Consolidated closeout document:** `Research_Extension/FINAL_RESEARCH_VERIFICATION_MATRIX.md`.

**Current decision: empirical experiment gate CLOSED; manuscript gate NOT YET OPEN** until the final publication-readiness checks documented in that matrix pass. The historical README, original checkpoints, and prior result records remain unmodified.

## EXT-CACHE-001 — completed numerical cache-key audit

The local read-only audit completed successfully on 28 October 2026 against four training inputs, using the historical 8-bit patch-key operation in `utilities/dataset_transformer.py`.

| Source | Images | Distinct colliding cache keys | Later patches differing from first input / total |
| --- | ---: | ---: | ---: |
| Original binary training | 279 | 0 | 0 / 1,142,784 (0%) |
| SMOTE-expanded binary training | 306 | 17,535 | 28,766 / 1,253,376 (2.30%) |
| Original TI-RADS training | 279 | 0 | 0 / 1,142,784 (0%) |
| SMOTE-expanded TI-RADS training | 462 | 84,657 | 219,533 / 1,892,352 (11.60%) |

The source report is committed as `07_Results/exp_cache_001/quantum_cache_key_audit.json` in research commit `571ebcb` and its four dataset totals have been independently checked against the uploaded JSON. Numeric collisions are confirmed for synthetic patches, but neither a per-circuit expectation-value error nor a model-accuracy effect has been measured. The 279 original TI-RADS images form the verified prefix of the historical 462-image SMOTE tensor (`07_Results/phase9_reproduction_audit/audit_report.txt`); no original-only collisions were detected. Under a fresh sequential cache pass, synthetic collisions after that prefix would not retroactively affect its original images. Avoid asserting exact saved-output equivalence without a corresponding hash/feature comparison.

This finding changes the interpretation of the early SMOTE pipelines, not their frozen stored results. Further full training, source-data changes and manuscript preparation remain outside the current scope. The cache audit is covered in `Research_Extension/ORIGINAL_PAPER_FULLTEXT_FIDELITY_AUDIT.md` and `FINAL_RESEARCH_VERIFICATION_MATRIX.md`.

## EXT-MODELS-001 — matched classifier-family follow-up (source committed, results pending)

Motivation: the historical Phase 14 CNN collapsed to TI-RADS 3 predictions across its 71 ROI / 44 case historical test sets. The zero-heavy confusion matrix documents that failure and **must not be edited or suppressed**. An additional class-diversity and fair model-family comparison is warranted as a separate exploratory study, not as a correction of those historical outputs.

**Predeclared experiment:** `05_Source_Code/exp_models_001_matched_classifiers.py` compares classical 64 pooled-intensity features with the existing 64 fixed simulated four-qubit expectation features. Each arm uses the same **four** classifiers and fixed settings: balanced linear logistic (C=0.1), balanced RBF SVM (C=1.0, gamma=scale, probability=True), random forest (300 trees, min leaf 2, balanced_subsample), and seven-neighbor distance-weighted kNN. Models are not tuned after inspecting held-out predictions.

**Fair scope:** 311 frozen Phase13 training ROIs, 203 case IDs, identical 5 case-stratified folds (seed 42), fold-local standardization where appropriate, ROI probabilities averaged to cases. The script imports the already cross-checked fixed PennyLane filter; before accepting new outputs, it checks saved training file hashes and requires the two newly produced logistic ROI OOF probability arrays to match all **622** archived EXT-QMAP-001 reference rows to tolerance 1e−8. This guardrails inadvertent experiment drift.

**Evidence to retain:** full six-class case and ROI confusion matrices (including zero cells), class-support and predicted-class diversity, accuracy/balanced accuracy/macro-F1, complete ROI and case OOF probabilities, paired whole-case bootstrap CIs (5,000 replicates), exact McNemar disagreements, model definitions and source hashes; colour and vector figures saved alongside. A majority-class case reference remains essential to interpret raw accuracy. RBF SVC uses training-fold-only internal probability calibration; that inner calibration is not independently case grouped, even though the outer held-out case folds are.

**Limitations:** training-only exploratory multi-comparison; at most eight configuration arms, no correction for multiple contrasts, no new unseen external holdout, no verified patient linkage, no pathology-confirmed tumor endpoint, no optimized quantum circuit. No predicted performance improvement is promised. Results will be assessed regardless of whether any classifier surpasses the 28.08% majority-class accuracy reference.

**Status:** EXT-MODELS-001 completed locally and committed as `0313b44`; all 1,624 case prediction rows and 2,488 ROI prediction rows were independently reaggregated and their eight six-class confusion matrices, accuracies, balanced accuracies, macro-F1 scores, and class coverage matched the saved `metrics.json`. Historical Phase 14 results and images remain unchanged.

### EXT-MODELS-001 numerical closure — eight matched configurations

The four classifiers were applied to **each** of two equal-dimensional fixed representations (64 classical pooled-intensity features and 64 fixed four-qubit expectation features), using the same five case-stratified out-of-fold partitions from the 203 original training case IDs. The output is exploratory and does not reclassify previously inspected test cases.

| Model | Correct cases / 203 | Accuracy | Balanced accuracy | Macro-F1 | Predicted categories |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classical logistic regression | 54 | 26.60% | 25.22% | 24.26% | 6/6 |
| Classical RBF SVM | 57 | 28.08% | 18.09% | 13.79% | 4/6 |
| Classical random forest | 54 | 26.60% | 18.92% | 17.86% | 6/6 |
| Classical kNN (k=7) | 58 | 28.57% | 19.28% | 17.63% | 6/6 |
| Fixed quantum logistic regression | 43 | 21.18% | 21.91% | 20.68% | 6/6 |
| Fixed quantum RBF SVM | 54 | 26.60% | 17.29% | 13.19% | 4/6 |
| Fixed quantum random forest | **63** | **31.03%** | 23.54% | 23.35% | 6/6 |
| Fixed quantum kNN (k=7) | 59 | 29.06% | 20.42% | 19.07% | 6/6 |

**Primary interpretation:** The fixed quantum random forest achieved the largest observed case accuracy, 63/203, but only **six cases above** the 57/203 majority-class accuracy reference (+2.96 percentage points). Its paired correctness versus the majority predictor was: 25 both correct; 32 majority-only correct; 38 random-forest-only correct; 108 both wrong. The two-sided exact McNemar p-value computed independently for that descriptive contrast is **0.5504**, with no evidence for a confirmatory advantage.

The quantum random forest exceeded the matched classical random forest by 9/203 cases (+4.43 percentage points), but the **saved paired 95% case-bootstrap interval [−1.00, +9.85] percentage points includes zero**, and exact paired McNemar p = **0.1755**. The fixed-quantum random forest versus its own fixed-quantum logistic counterpart differs by +20/203 (+9.85 points), exploratory interval [3.45, 16.75] points and unadjusted p = 0.0091. This last finding is an **exploratory within-representation model-family contrast**, not proof of quantum advantage or a multiplicity-adjusted confirmatory result.

**Class coverage is not the same as class success.** The quantum random forest predicted all six labels but predicted TI-RADS 3 for only **one case**, and correctly classified **zero of the ten genuine TI-RADS 3 cases** (recall 0%). The RBF SVM outputs predicted just four of six labels. Across models, balanced accuracy and macro-F1 remain weak; the highest observed balanced accuracy and macro-F1 were achieved by **classical logistic regression** (25.22%, 24.26%). The historical Phase 14 near-single-class confusion matrices must be shown as original observations, alongside the newer balanced-feature comparison, rather than being cosmetically altered.

**Independent record verification:** all 1,624 case-level rows, 2,488 ROI-level rows, the expected case identifiers, six probability columns, case-averaging identity and eight full 6×6 confusion matrices agreed with the pushed metrics JSON. The logistic-regression rows also matched the frozen EXT-QMAP-001 reference OOF predictions at experiment runtime. Source: `07_Results/exp_models_001/metrics.json`, `case_predictions.csv`, `oof_predictions.csv`, `matched_classifier_comparison.{png,pdf,svg}` (commit `0313b44`). No observed test/validation partition was accessed and no original experiments were modified.

**SVM technical note:** scikit-learn 1.9 emitted a FutureWarning that SVC(probability=True) is deprecated and will be removed in version 1.11. The warnings are not failed experiments. Replacing its internal probability calculation with CalibratedClassifierCV would change the model definition; any future such change must be separately versioned rather than quietly retroactively altering these saved results.

**Decision:** Do not launch an unbounded accuracy search or treat the maximum of eight exploratory models as a new deployment/clinical benchmark. The remaining tasks are presentation of the full model-family comparison, candid class-level analysis, and guide review—not repeat training.

## Guide-method alignment and hyperparameter closure

This checkpoint addresses the guide's notes: dataset counts and class grouping, three distinct architectures, separate architecture diagrams, paper-versus-implementation differences, learning rate and optimizer settings, model size, multiple comparative baselines, sensitivity/specificity/ROC, software and hardware provenance, and a technically justified discussion of failures. **It is a research pre-submission audit, not a manuscript draft.**

### Requested evidence and completion

| Guide expectation | Verified status |
| --- | --- |
| Problem statement and objectives | Study objective is reproducibility and sensitivity of QuCNet-style TI-RADS classification; not confirmed tumor diagnosis |
| Dataset size, source and classes | 480 raw images, 349 with recovered TI-RADS labels; Phase13 has 440 ROIs/291 case IDs, partitioned as 311/58/71 ROI and 203/44/44 case IDs |
| Architecture diagram for each model | New colour figure distinguishes historic Phase9 QFT-CNN, strict Phase14 ROI CNN, and eight matched development configurations; the pathways are not equivalent |
| Base paper versus reconstructed code | Full-text fidelity audit documents 128x128 full-image source, 64x64x4 four-wire features, processing and test-set mismatches |
| Hyperparameters and model sizes | See the source-backed table below; parameters and data partitions are not estimated |
| Fair comparison and adverse results | Eight same-fold development configurations; original collapsed Phase14 result preserved; majority-class comparison and class-coverage failure disclosed |
| Sensitivity, specificity, ROC/AUC | Eight models × six TI-RADS one-vs-rest categories independently calculated from saved case probabilities, 48 class-model records; colour ROC and recall figures |
| Software, hardware, and computational cost | Software/library versions and historical parameter counts available; local CPU/GPU, measured runtime, GFLOPs and energy consumption remain unverified and will not be invented |
| Discussion and future work | Evidence and limitations documented; final paper writing deferred until authorized |

### Source-backed model and hyperparameter table

| Model family | Architecture and actual input | Learning method and settings | Parameters |
| --- | --- | --- | ---: |
| Original QuCNet article | 128x128 original input; four quantum feature channels at 64x64; Conv/Pool and dense classifier | Adam; 80/20 image split, mean/std scaling, augmentation and SMOTE described | Exact widths, counts and random filter gate parameters not fully recoverable |
| Phase6 binary suspicion proxy | 64x64x4 QFT → Flatten → Dense128 → Dense64 → softmax2 | Adam LR 0.001; batch16; max50 epochs; EarlyStopping patience3; LR plateau patience4 | **2,105,666** |
| Phase9 paper-structured six-class | 64x64x4 QFT → Conv16(3x3) → Pool2 → Conv32(3x3) → Pool2 → Flatten → Dense128 → Dense64 → softmax6 | Adam LR **1e-4**; batch16; max120 epochs; early stop15; LR plateau patience6; class weights/no SMOTE | **1,062,582** |
| Phase14 strict classical ROI CNN | 64x64x1 median-padded lesion ROI → Conv32/64/64/96+BatchNorm/ReLU → global average pooling → Dense64 → softmax6 | Adam LR **0.001**; batch16; max100 epochs; early stop18; LR plateau patience7, factor0.5, min1e-6; class weights/no SMOTE | **134,742** |
| Development logistic regression, either feature map | Each of 64 pooled intensities or 64 fixed four-qubit expectations → fold-local StandardScaler → logistic6 | Balanced, C=0.1, max_iter=1500, seed42, five case-stratified training folds | Not a CNN; no epoch/kernel |
| Development RBF SVM, either feature map | 64 features → fold-local StandardScaler → RBF SVM | C=1, gamma=scale, balanced, probability=True; same case folds | Support-vector count not saved |
| Development random forest, either feature map | 64 features → 300-tree ensemble | Min samples leaf=2, max_features=sqrt, balanced_subsample, seed42 | Tree/node count not saved |
| Development kNN, either feature map | 64 features → fold-local StandardScaler → distance-weighted kNN | k=7, distance weighting; same case folds | No trainable CNN weights |

Phase9 configuration records 223 training and 56 validation images from its 279 original input-side images. The Phase14 frozen partitions are 311/58/71 ROIs and 203/44/44 case IDs. Across the later model-family study, the **same 203 case IDs** provide all out-of-fold evaluations.

**Software provenance:** PennyLane 0.45.1, scikit-learn 1.9.0 and NumPy 2.5.0 in fixed-feature audit records; TensorFlow 2.21.0 and NumPy 2.5.0 in Phase14 training configuration. The quantum model uses the classically simulated PennyLane default.qubit. Local hardware model, reproducible wall time, GFLOPs and energy are not established by saved records.

### Why proposed automatic fixes must not be applied untested

1. **No cross-dataset domain-generalization result exists.** Both source paper and reproduction are associated with DDTI, albeit different or incompletely recovered source image inventories. No second external cohort was used.
2. **The hypothesis that 53% accuracy proves quantum saturation is unsupported.** Historical binary suspicion-proxy accuracy was 38/70 = 54.29% (balanced accuracy 50.70%), while the six-category tasks have different chance/majority baselines. Neither result identifies its own failure mechanism.
3. **RX input angle scaling is a model choice, not an automatic repair.** The saved quantum filter uses input intensities in [0,1] radians. Expanding to [0,pi] changes the circuit and would require a new predeclared ablation. It cannot be asserted to improve results. Pauli-Z output expectations, not gate input angles, lie within [-1,1].
4. **Do not rewrite historic splits.** Phase12 assigns disjoint case IDs before Phase13 ROI extraction. Division by the constant 255 does not estimate train/test population statistics. Later StandardScaler fitting is inside each CV fold. Cross-ID patient identity remains unknown. Earlier historical test/validation reuse and SMOTE pipeline limitations are disclosed.
5. **Cache collision evidence is numerical, not a causal accuracy result.** Distinct synthetic float patches reused 8-bit cache keys for 2.30% of binary SMOTE and 11.60% of TI-RADS SMOTE patch occurrences; original-array training inputs showed no collisions. Quantum output/accuracy effects were not independently measured.
6. **The highest development accuracy is not a proven advantage.** Quantum random forest obtained 63/203 (31.03%), compared with classical random forest 54/203 (26.60%) and majority-class 57/203 (28.08%). Quantum RF minus classical RF paired CI includes zero, p=.1755; its TI-RADS 3 sensitivity remains 0/10.
7. **Guide-specific diagnostics added without retraining.** The eight-model macro one-vs-rest AUCs range from 0.5353 to 0.6069, and full sensitivity, specificity, precision, NPV and ROC data are saved under exp_eval_001. These are descriptive training-only TI-RADS ROC curves, not clinical binary cancer diagnostic ROC curves.

**Scope decision:** The bounded experiments and methodological audits support research discussion with the guide. Any new data processing, gate encoding or model architecture requires a specific testable question and a separately versioned experiment; indefinite accuracy searching is not justified. **Manuscript and Overleaf drafting remain on hold until explicitly requested.**

## Guide-directed research strengthening sequence — locked before manuscript

**Purpose:** Address the guide's handwritten review of the whole research project. The guide expects results from genuinely comparable model families; a model-by-model architecture figure with diagrams; dataset category/split counts; hyperparameters and model complexity; comparison with the published article; sensitivity/specificity/F1/ROC; an explanation of failures; and candid discussion/future work. The public README presentation is retained, with new colour figures added additively.

### Research questions and evidence status

| Research question | Present evidence | Scientific decision |
| --- | --- | --- |
| Does a genuine four-qubit fixed filter match the historical implementation? | PennyLane QNode and tensor channel checks; single-patch reference agreement approximately 1.26e−8; cache-key collisions quantified for synthetic SMOTE patches | **Answered for fixed simulated filter.** Cannot claim author-exact Qiskit gates or quantum speedup. |
| Does a quantum feature representation beat comparable classical features? | Eight models = four classifiers × two 64-dimensional inputs, same 203 case IDs and folds, full probabilities/confusions, paired intervals | **Exploratory; no established quantum superiority.** Quantum RF is highest observed raw accuracy (63/203), but low class-level recall and uncertain paired gain. |
| Why did the strict ROI CNN collapse into one predicted class? | Original metrics; BN recalibration; padding/resize/fusion controls; per-class diagnostic analyses | **Failure documented; cause not proven.** Maintain its zero-heavy confusion matrix. Do not fix charts or assign a cause without an isolation experiment. |
| Which representation choices change predictions and are they robust? | Original/reflection/direct-stretch folds and padding association, all with case-level paired tests | **Exploratory, limited evidence.** Stretch changes shape; accuracy changes do not prove a preprocessing improvement. |
| Can we report clinical diagnostic performance or cross-dataset generalization? | No pathology ground truth; no verified patient linkage; no separate independent dataset or new untouched clinical test | **Not yet answerable**. These are required inputs, not code defects that model tuning can repair. |

### Limited further empirical work, if pursued before writing

1. **Priority A — reproducibility and stability, not a model-selection contest:** perform a single predeclared repeatability analysis of the *same fixed classical/quantum feature comparisons* under multiple case-stratified fold seeds, with every repeat reported rather than choosing a best seed. Report distribution and rank instability of balanced accuracy and macro-F1, whole-case grouping, and no use of historical validation/test partitions. This adds uncertainty about fold assignment but remains development-only.
2. **Priority B — circuit-encoding ablation only if needed to answer a precise question:** contrast the existing RX(pixel in [0,1]) with one expressly named RX(pi·pixel) variant under the same 203 training cases, fixed circuit weights, same classifier(s), identical folds and bootstrap procedure. A different encoding is a *hypothesis*, never an automatic correction. Report all chosen variants, computational cost and adverse findings.
3. **Priority C — genuinely independent validation only if a legitimate external dataset becomes available:** recover appropriate TI-RADS labels, verify patient identifiers and consent/terms, define an untouched test protocol before loading outcomes, and do not merge data sources with incompatible targets. This would materially strengthen claims; it cannot be simulated using the already-inspected 44 historical test cases.
4. **Presentation/engineering check:** retain exact historical architecture, parameters, model-size counts and hyperparameters; record actual hardware/library versions and measured runtime only where logs exist; show source-paper numbers as *non-comparable author reports* and explain why.

These are **candidate future tasks, not completed experiments or evidence of improved accuracy**. Do not launch all exploratory arms in a search for a favorable p-value. If time is limited, prioritize fully checked interpretation, external validation feasibility and a concise, reproducible comparative matrix over accumulating weakly identified models.

**Writing gate:** IEEE/Overleaf manuscript preparation begins only on the researcher's explicit instruction. The eventual paper should be an independent methodological audit with precise equations, paper-versus-code deviations, all adverse results, and reviewer-auditable files. No result is represented as a new clinical benchmark or quantum computational advantage.

## Final research extensions — calibration and fold stability

The guide's requirements for multiple matched model families, classwise sensitivity/specificity and ROC curves, visible architecture paths, hyperparameter comparisons and failure analysis have been addressed using the previously saved model probabilities. The original case-separated Phase14 failure and historical test records remain preserved. The updated full-colour README includes three model-path diagrams, all eight development model comparisons, ROC/class-recall figures and one probability-reliability figure.

### EXT-CALIB-001 — completed, source-backed reliability analysis

Eight sets of 203 saved **case-level out-of-fold** probabilities were verified for class order, normalization and case count. Without rerunning models, the audit computed the multiclass Brier score, negative log likelihood, mean maximum prediction probability, predictive entropy and ten-bin expected calibration error (ECE). Exact records: `07_Results/exp_calib_001/calibration_metrics.json` and `calibration_bins.csv`, derived from `exp_models_001/case_predictions.csv`.

| Model | Multiclass Brier (lower better) | Negative log likelihood (lower better) | Ten-bin ECE |
| --- | ---: | ---: | ---: |
| Classical logistic regression | 0.831 | 1.766 | 0.109 |
| Classical RBF SVM | 0.784 | 1.628 | 0.034 |
| Classical random forest | 0.786 | 1.637 | 0.057 |
| Classical kNN | 0.847 | 6.764 | 0.163 |
| Fixed quantum logistic regression | 0.836 | 1.777 | 0.148 |
| Fixed quantum RBF SVM | 0.785 | 1.635 | 0.051 |
| Fixed quantum random forest | **0.783** | 1.636 | 0.081 |
| Fixed quantum kNN | 0.856 | 6.322 | 0.166 |

The relatively large kNN negative log likelihood arises because some held-out cases receive extremely small probability for their true category. At a maximum-predicted-class-confidence threshold of 0.50, classical kNN incorrectly classified **40/63** high-confidence cases and quantum kNN **34/46**. These counts do not support patient-facing confidence claims. Equal-width ECE is descriptive and sensitive to the small sample and binning convention. No calibration model was trained or selected using held-out case outcomes.

### EXT-STABILITY-001 — completed and independently checked (2026-10-28)

The predeclared script `05_Source_Code/exp_stability_001_repeated_casecv.py` was executed locally and its complete evidence pushed to the research branch at commit `cb46ed0`. It varies **only the case-stratified fivefold partition seed** (42, 101, 202, 303, 404) for the same 311 Phase 13 training ROIs / 203 recorded case IDs. The two existing fixed 64-feature maps (pooled grayscale; simulated four-qubit Pauli-Z expectations) were crossed with balanced logistic regression (C=0.1) and a 300-tree random forest with unchanged classifier RNG and hyperparameters. Twenty model/seed configurations (100 fitted folds) were retained. The fixed quantum features were computed once and reused.

Independent read-back verification from GitHub recalculated **all 4,060 case-level OOF prediction rows**: 203 unique case IDs in each of 20 model/seed configurations, six finite normalized class probabilities per case, consistent true labels/fold assignments, and predicted classes. All 20 accuracy, six-class balanced accuracy, macro-F1 and confusion matrices agreed with both `seed_metrics.csv` and `metrics.json`, with **zero discrepancies**. The 812 seed-42 case-probability vectors were compared to frozen `exp_models_001/case_predictions.csv`: maximum absolute difference **1.11 × 10^-16**, below the 1e-8 tolerance. These checks verify the uploaded outputs, not independent re-execution of the training pipeline or independent confirmation of patient identity.

| Feature map / classifier | Five-seed mean accuracy ± SD | Mean balanced accuracy | Mean macro-F1 |
| --- | ---: | ---: | ---: |
| Classical / logistic | 24.53% ± 2.42 pp | 23.23% | 22.07% |
| Fixed quantum / logistic | 23.55% ± 1.61 pp | 23.64% | 22.08% |
| Classical / random forest | 27.49% ± 2.71 pp | 20.42% | 19.62% |
| Fixed quantum / random forest | **29.36% ± 1.58 pp** | 22.50% | 22.30% |

For quantum-minus-classical **random-forest** accuracy, the five seed-wise differences (percentage points) were **+4.43, −1.48, +2.46, +2.96, +0.99** (mean **+1.87 pp**); the classical forest outperformed on seed 101. Logistic differences were **−5.42, −2.46, +2.96, +1.97, −1.97 pp** (mean **−0.99 pp**), also changing sign. The cross-seed mean is descriptive; no stable model-superiority or quantum-advantage claim is supported.

Source evidence: `07_Results/exp_stability_001/metrics.json`, `seed_metrics.csv`, `case_oof_predictions.csv`, and the colour `case_fold_stability.svg` (also PNG/PDF). Source-code and frozen-input SHA256 checks were performed by the run; stored output and reference probabilities were separately audited.

**Statistical boundaries:** The five folds are reassigned across the **same** recorded cases: these are five correlated development-set evaluations, **not five independent cohorts**. Across-seed SD/ranges are not confidence intervals or confirmatory significance tests. Classifier RNG is held fixed, so this measures **fold-assignment sensitivity only**. No original validation/test data were accessed by the script. Case identifiers are not verified patient identities, the six TI-RADS labels are not pathology outcomes, the circuit is fixed and simulated, and independent external validation remains absent. No best seed was selected.

**Remaining external dependency:** clinically independent or genuine cross-dataset conclusions cannot be made without a separate appropriately labeled dataset, patient-linkage information and prospective/untouched evaluation design. No such external dataset has been adopted or evaluated. Do not advertise cross-dataset generalization.

**Writing gate:** no manuscript or Overleaf work starts until expressly authorized. After the stability output is checked, the guide-directed bounded empirical research can be frozen for paper preparation without running further speculative models.
