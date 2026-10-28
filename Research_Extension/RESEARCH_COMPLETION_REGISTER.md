# Research extension — execution and verification register

**Status:** Development experiments in progress; the final manuscript is deferred.

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

**Pending:** Real-data `EXT-FUSION-001`, paired statistics and historical integrity audit outputs were executed locally and reported in terminal logs, but the generated files have not yet been pushed to GitHub. Preserve and synchronize these outputs before treating them as repository-backed research artifacts.
