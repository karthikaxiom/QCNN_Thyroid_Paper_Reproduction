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

At each live session verify the branch head, current commits, saved experimental results and outstanding gates. The next actionable task is to execute the already committed `05_Source_Code/exp_fusion_001.py` on the user's local data. No unscheduled background work or later notification is implied.

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

The uploaded `07_Results/exp_preproc_001/oof_predictions.csv` contains 311 out-of-fold ROI predictions **per model**, aligned to 203 unique case IDs. I independently re-aggregated the six predicted probabilities to cases by mean, obtained argmax labels, constructed each 6×6 confusion matrix and recalculated accuracy, balanced accuracy and macro-F1. All nine resulting scores match `07_Results/exp_preproc_001/metrics.json` to numerical precision:

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

`EXT-QMAP-001` real training-only paired study was completed and pushed as commit `0868955`. I independently reconstructed the 203 case predictions by averaging six class probabilities from each of the 311 ROI out-of-fold rows per model; all stored case-level confusion matrices, accuracies, balanced accuracies and macro-F1 values matched.

- Equal-dimensional 64-feature classical pooled image: **54/203 cases (26.60099%)**; balanced accuracy 25.22328%; macro-F1 24.26322%.
- Existing fixed four-qubit quantum simulation on sixteen pooled 2×2 patches: **43/203 cases (21.18227%)**; balanced accuracy 21.90883%; macro-F1 20.67882%.
- Quantum-minus-classical case accuracy = **−11/203 (−5.4187 percentage points)**; classical-only correct 21, quantum-only correct 10, both correct 33, both wrong 139; exact two-sided McNemar p = **0.0707555**. Exploratory paired case-bootstrap interval [−10.84, 0.00] percentage points; no confirmatory superiority claim.
- Fixed quantum circuit checked against original reference `utilities/quantum_utils.py` (maximum expectation-value disagreement 1.2572×10⁻⁸). This is **not** training a quantum circuit and cannot demonstrate quantum speedup.

Evidence `07_Results/exp_qmap_001/metrics.json`, `07_Results/exp_qmap_001/oof_predictions.csv`, and independent read-only verification `Research_Extension/verification/exp_qmap_001_independent_metrics.json`.

I reviewed the three pre-flagged cross-assigned-split raw ultrasound pairs directly. None appeared to be an identical raw frame. The case/patient identity question cannot be resolved from imagery alone; maintain this limitation. One full raw test image (`313_1.jpg`) has visible TI-RADS 4C text in its bottom border, outside the saved cropped test ROI y-range 12–190. Raw-overlay effects cannot be generalized from this one inspection. Evidence `Research_Extension/verification/near_duplicate_visual_review.json`. No raw ultrasound image has been republished in the research register.

**Figure and arithmetic checkpoint:** Four stored PNG figure exports were visually checked for labels, axes and legibility at native display size (fixed-feature comparison, preprocessing sensitivity, padding/log-loss relationship, and paired fusion intervals); all have corresponding vector-source exports. IEEE final printed-column rendering, permissions/citations and source-paper equation correspondence remain to be checked.

**Consolidated closeout document:** `Research_Extension/FINAL_RESEARCH_VERIFICATION_MATRIX.md`.

**Current decision: empirical experiment gate CLOSED; manuscript gate NOT YET OPEN** until the final publication-readiness checks documented in that matrix pass. The historical README, original checkpoints, and prior result records remain unmodified.
