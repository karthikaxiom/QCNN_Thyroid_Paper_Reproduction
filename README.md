<div align="center">

# Reproduction and Methodological Audit of QuCNet for Thyroid Nodule Classification

### A Systematic Reproduction Study of Quantum-Inspired Deep Learning for Thyroid Ultrasound Classification Using the Public DDTI Dataset

---

![Research](https://img.shields.io/badge/Research-Reproducibility%20Study-success)
![Status](https://img.shields.io/badge/Status-Completed-blue)
![Framework](https://img.shields.io/badge/Framework-TensorFlow%202.21-orange)
![Language](https://img.shields.io/badge/Python-3.13-yellow)
![Dataset](https://img.shields.io/badge/Dataset-DDTI-red)

</div>

---

# Research Summary

Artificial Intelligence has demonstrated remarkable success in medical image classification. Recently, Quantum Machine Learning (QML) has emerged as a promising research direction by combining concepts from quantum computing with deep learning architectures. One such work is **QuCNet**, a Quantum-Inspired Convolutional Neural Network proposed for thyroid nodule classification using ultrasound images.

The original paper reported exceptionally high classification performance using quantum-inspired feature extraction on the Digital Database of Thyroid Ultrasound Images (DDTI). However, reproducing published research is often challenging because many implementation details, preprocessing decisions, dataset versions and experimental settings are not fully disclosed.

This repository presents a **complete independent reproduction and methodological audit** of the QuCNet framework using the publicly available DDTI dataset.

Unlike a conventional implementation project, the primary objective of this work was **not to maximize classification accuracy**, but to answer a more fundamental scientific question:

> **Can the performance reported in the original QuCNet paper be independently reproduced using the publicly recoverable dataset and the implementation details described in the publication?**

To answer this question, the complete pipeline was reconstructed from scratch, including dataset verification, XML parsing, metadata extraction, preprocessing, quantum-inspired feature generation, classical and quantum baselines, annotation auditing, patient-level leakage analysis, lesion-level ROI extraction, and a final leakage-free evaluation protocol.

The study successfully reconstructed the entire experimental workflow and identified several important reproducibility challenges, including missing pathology labels, incomplete methodological specifications, annotation-level ambiguities, and patient-level data leakage risks.

Although the complete implementation pipeline was successfully executed, the reported performance of the original publication could **not** be independently reproduced using the publicly available resources.

Rather than treating this as a failure, the project preserves these findings as a transparent and scientifically valuable reproducibility study.

---

# Why This Research Matters

Reproducibility is one of the foundations of scientific research.

A published model can only be considered reliable when independent researchers are able to reconstruct the methodology, obtain comparable experimental conditions, and verify the reported findings.

Medical AI research becomes particularly important because model performance may directly influence future clinical decision-support systems. Therefore, identifying missing implementation details, hidden assumptions, undocumented preprocessing steps, or dataset inconsistencies contributes to improving transparency and research quality.

This repository aims to provide a carefully documented reference implementation that future researchers can build upon while understanding the practical challenges involved in reproducing quantum-inspired medical image classification systems.

---

# Reference Paper

**Title**

> QuCNet: Quantum-Inspired CNN for Optimized Thyroid Nodule Classification

**Journal**

IEEE Access (2024)

**DOI**

10.1109/ACCESS.2024.3367806

---

> **Important Note**
>
> This repository is an **independent academic reproduction** of the published methodology.
> It is **not** an official implementation released by the original authors.

---

# Research Objectives

The objectives of this work were:

- Reconstruct the complete QuCNet pipeline using only publicly available resources.
- Study the DDTI dataset in detail by analysing XML annotations, metadata and image relationships.
- Develop a reproducible preprocessing pipeline for thyroid ultrasound images.
- Reconstruct the quantum-inspired feature generation procedure described in the paper.
- Compare quantum-inspired features against conventional grayscale image representations.
- Evaluate both binary and multi-class TI-RADS classification tasks.
- Investigate the effect of annotation quality, dataset structure and patient-level data leakage.
- Build a leakage-free lesion-level evaluation protocol using expert-marked ROI annotations.
- Document every phase of the reproduction process in chronological order.
- Preserve both successful implementations and negative experimental findings for future reproducibility research.

---

# Research Contributions

This repository provides:

- A complete phase-wise reconstruction of the QuCNet workflow.
- Automated XML parsing and metadata extraction for the DDTI dataset.
- Dataset verification and consistency auditing.
- Quantum-inspired feature generation pipeline.
- Classical CNN and Quantum-inspired CNN implementations.
- Controlled experimental comparisons.
- Annotation-level and patient-level dataset auditing.
- Leakage-free case-level train-validation-test splitting.
- Lesion ROI dataset generation using expert annotations.
- ROI-level and Case-level performance evaluation.
- Complete research log documenting every development stage.
- Reproducible implementation scripts, reports and evaluation outputs.

---

# Current Project Status

| Component | Status |
|------------|--------|
| Dataset Verification | ✅ Completed |
| XML Metadata Parsing | ✅ Completed |
| Image Preprocessing | ✅ Completed |
| Binary Classification Reproduction | ✅ Completed |
| Quantum Feature Generation | ✅ Completed |
| Classical Baseline | ✅ Completed |
| Multi-Class TI-RADS Evaluation | ✅ Completed |
| Annotation Audit | ✅ Completed |
| Case-Level Leakage Audit | ✅ Completed |
| Frozen Dataset Split | ✅ Completed |
| ROI Dataset Generation | ✅ Completed |
| Final Leakage-Free Evaluation | ✅ Completed |
| Research Documentation | ✅ Ongoing |

---

> **Project Outcome**
>
> The complete QuCNet methodology was successfully reconstructed and evaluated.
> However, the high classification performance reported in the original publication could not be independently reproduced using the publicly available DDTI dataset and the methodological details disclosed in the paper.
>
> The repository therefore serves as a **complete reproducibility study**, documenting both the implementation process and the scientific challenges encountered during independent verification.

---

# Experimental Pipeline

The complete reproduction was performed through a structured multi-phase workflow. Every phase was independently verified before progressing to the next stage, ensuring that the final implementation remained reproducible, transparent and scientifically defensible.

```text
                     PUBLIC DDTI DATASET
                              │
                              ▼
                 Dataset Verification & Audit
                              │
                              ▼
                 XML Parsing & Metadata Extraction
                              │
                              ▼
               Master Dataset Generation (CSV)
                              │
                              ▼
              Image Inspection & Preprocessing
                              │
                              ▼
            Binary Classification Reproduction
                              │
                              ▼
         Quantum Feature Generation (QuCNet)
                              │
                              ▼
      Classical CNN & Quantum CNN Comparison
                              │
                              ▼
      Multi-Class TI-RADS Classification Study
                              │
                              ▼
         Complete Annotation & Case-Level Audit
                              │
                              ▼
        Leakage-Free Patient-Level Dataset Split
                              │
                              ▼
            Lesion ROI Dataset Construction
                              │
                              ▼
        Final ROI-Level & Case-Level Evaluation
                              │
                              ▼
        Scientific Analysis & Reproducibility Report
```

---

# Repository Structure

```text
QCNN_Thyroid_Paper_Reproduction
│
├── 01_Paper
│   ├── Reference paper
│   └── Supporting literature
│
├── 02_Dataset_Raw
│   └── Original DDTI dataset
│
├── 03_Dataset_Processed
│   └── Generated datasets, NumPy arrays and processed outputs
│
├── 04_Notebooks
│   └── Experimental notebooks and exploratory analysis
│
├── 05_Source_Code
│   ├── Phase-wise implementation
│   ├── Utility modules
│   └── Reproduction scripts
│
├── 06_Models
│   └── Saved TensorFlow model checkpoints
│
├── 07_Results
│   ├── Metrics
│   ├── Reports
│   ├── Confusion matrices
│   ├── Prediction files
│   └── Experimental outputs
│
├── 08_Presentation
│   └── Presentation material
│
├── 09_Output_images
│   └── Pipeline visualizations
│
├── Project_Log.txt
│
├── README.md
│
├── requirements.txt
│
└── .gitignore
```

---

# Dataset Overview

This reproduction uses the **Digital Database of Thyroid Ultrasound Images (DDTI)**, a publicly available thyroid ultrasound dataset containing expert-annotated ultrasound scans together with XML metadata describing thyroid nodules.

Unlike many benchmark datasets, the DDTI dataset contains multiple ultrasound images for several patients, XML annotation files, SVG-based lesion boundaries and TI-RADS information, making the preprocessing pipeline considerably more complex than conventional image classification datasets.

During the reproduction process, the complete dataset was independently verified before any model training was performed.

---

## Original Dataset Statistics

| Component | Count |
|-----------|------:|
| XML Files | 390 |
| Ultrasound Images | 480 |
| Images with TI-RADS Labels | 349 |
| Images without TI-RADS Labels | 131 |

---

## Audited Dataset Statistics

Following the complete annotation and patient-level audit performed during the later stages of this research, the validated dataset consisted of:

| Component | Count |
|-----------|------:|
| Valid Patient Cases | 291 |
| Valid Lesion ROIs | 449 |
| Invalid Annotation Records | Documented Separately |

---

# Research Timeline

The reproduction study was carried out as a sequence of progressively refined experimental phases.

Rather than directly implementing the published model, the project focused on understanding every stage of the original methodology, verifying each intermediate result and correcting potential reproducibility issues whenever they were identified.

| Phase | Description | Status |
|------|-----------------------------------------------|:------:|
| Phase 0 | Dataset Verification | ✅ |
| Phase 1 | XML Structure Analysis | ✅ |
| Phase 2 | Master Dataset Generation | ✅ |
| Phase 3 | Image Inspection | ✅ |
| Phase 4 | Image Preprocessing | ✅ |
| Phase 5 | Binary Dataset Preparation | ✅ |
| Phase 6 | Initial QuCNet Baseline | ✅ |
| Phase 7 | Classical CNN Baseline | ✅ |
| Phase 8 | Multi-Class TI-RADS Study | ✅ |
| Phase 9 | Controlled Experimental Comparison | ✅ |
| Phase 10 | Evaluation & Analysis | ✅ |
| Phase 11 | Annotation Audit | ✅ |
| Phase 12 | Leakage-Free Case Split | ✅ |
| Phase 13 | ROI Dataset Generation | ✅ |
| Phase 14 | Final ROI-Level Evaluation | ✅ |

---

# Research Philosophy

Throughout this project, the primary objective was **scientific reproducibility rather than model optimisation**.

Every experimental decision was guided by the following principles:

- Preserve reproducibility over reported accuracy.
- Never modify the testing dataset after observing results.
- Freeze validated dataset splits before training.
- Preserve unsuccessful experiments as part of the research record.
- Maintain complete transparency in preprocessing and evaluation.
- Avoid undocumented modifications to the original methodology.
- Document every important experimental decision chronologically.

These principles ensured that the final conclusions were based on a scientifically rigorous experimental protocol rather than repeated optimisation attempts.

---

# Key Challenges Encountered

Reproducing the QuCNet paper required overcoming several practical challenges that were not fully described in the original publication.

The major challenges included:

- Understanding the XML annotation structure.
- Mapping XML files to corresponding ultrasound images.
- Reconstructing the complete preprocessing pipeline.
- Absence of publicly available pathology-confirmed benign/malignant labels.
- Recovering the quantum-inspired feature generation procedure.
- Reconstructing the reported model architecture.
- Managing severe class imbalance across TI-RADS categories.
- Detecting and eliminating patient-level data leakage.
- Designing a clinically meaningful ROI-level evaluation protocol.
- Preserving complete reproducibility throughout every phase of development.

These challenges significantly influenced the final experimental design and motivated the later annotation audit and leakage-free evaluation protocol.

---
# Methodology

The reproduction was carried out using a structured, phase-wise methodology. Rather than implementing the complete QuCNet architecture directly, every stage of the published workflow was reconstructed independently and validated before progressing to the next phase.

This approach ensured that implementation errors, dataset inconsistencies and methodological assumptions could be identified early without affecting later experimental stages.

The complete workflow evolved through fourteen major phases, beginning with dataset verification and concluding with a leakage-free patient-level evaluation protocol.

---

# Phase 0 — Dataset Verification

The first stage focused on understanding the publicly available DDTI dataset.

The following tasks were performed:

- Downloaded and organized the DDTI dataset.
- Verified XML annotation files.
- Verified ultrasound image availability.
- Confirmed folder organization.
- Checked XML-image correspondence.
- Verified dataset integrity.

### Outcome

The downloaded dataset contained:

| Component | Count |
|-----------|------:|
| XML Files | 390 |
| Ultrasound Images | 480 |

The dataset was successfully verified and considered suitable for further analysis.

---

# Phase 1 — XML Structure Analysis

The XML annotation files were studied in detail to understand how clinical metadata and lesion annotations were represented.

The analysis included:

- Patient metadata
- Age and gender
- Composition
- Echogenicity
- Margins
- Calcifications
- TI-RADS category
- SVG lesion annotations

### Outcome

The XML files provided rich clinical metadata together with expert-marked lesion boundaries.

This phase established the foundation for every later preprocessing step.

---

# Phase 2 — Master Dataset Generation

A complete XML parser was developed to automatically reconstruct the dataset.

The parser extracted:

- Case ID
- XML filename
- Image filename
- Image number
- Patient metadata
- TI-RADS category
- Image path
- Annotation availability

The extracted information was consolidated into structured datasets.

Generated outputs included:

- master_dataset.csv
- master_dataset.xlsx

This dataset became the primary source of metadata throughout the remaining phases.

---

# Phase 3 — Image Inspection

Before preprocessing, every ultrasound image was independently inspected.

The inspection pipeline verified:

- Image readability
- Image dimensions
- Channel consistency
- Pixel intensity distribution
- Corrupted image detection

### Outcome

All 480 ultrasound images were successfully read.

No corrupted images were detected.

The dataset showed highly consistent image dimensions, making it suitable for a unified preprocessing pipeline.

---

# Phase 4 — Image Preprocessing

A standardized preprocessing pipeline was implemented to prepare the images for model training.

The preprocessing operations included:

- Grayscale conversion
- Image resizing
- Pixel normalization
- Dataset verification
- NumPy dataset generation

Only images containing valid TI-RADS labels were retained for supervised learning.

### Final Processed Dataset

| Item | Count |
|------|------:|
| Total Images | 480 |
| Images with TI-RADS | 349 |
| Images without TI-RADS | 131 |

### Important Observation

The publicly available DDTI dataset does not contain pathology-confirmed benign and malignant labels used in the original publication.

This became one of the most important reproducibility challenges encountered during the study.

---

# Phase 5 — Binary Classification Dataset Preparation

To investigate the binary classification task described in the original paper, a provisional binary dataset was constructed.

The pipeline included:

- Binary label generation
- Stratified train-test split
- Dataset serialization
- Class distribution verification

Training data imbalance was handled using SMOTE only on the training partition, preserving an untouched testing dataset.

### Important Clarification

The binary labels generated during this phase were **proxy labels inferred from TI-RADS categories** and not pathology-confirmed diagnoses.

Consequently, the binary experiments should be interpreted as exploratory reproduction experiments rather than exact replications of the original pathology-based classification task.

---

# Phase 6 — Initial QuCNet Reproduction

The first implementation of the published QuCNet methodology was reconstructed.

The implementation included:

- Quantum-inspired feature generation
- Four-channel feature representation
- Dense classifier
- Early stopping
- Learning-rate scheduling
- Model checkpointing

The objective of this phase was to determine whether the reconstructed methodology could achieve performance comparable to that reported in the original publication.

Although the complete pipeline executed successfully, the reported classification accuracy was not reproduced.

---

# Phase 7 — Classical CNN Baseline

A conventional CNN baseline was implemented using the same reconstructed dataset.

The purpose of this comparison was to determine whether the observed performance limitations originated from:

- Quantum-inspired feature generation,
- Dataset quality,
- Label construction,
- or implementation differences.

The classical baseline achieved slightly better performance than the reconstructed QuCNet implementation, suggesting that the reproduction gap could not be explained solely by the quantum feature transformation.

---

# Phase 8 — Multi-Class TI-RADS Classification

The reproduction was extended from binary classification to six-class TI-RADS prediction.

The classification categories included:

- TI-RADS 2
- TI-RADS 3
- TI-RADS 4a
- TI-RADS 4b
- TI-RADS 4c
- TI-RADS 5

Both quantum-inspired and classical models were evaluated under controlled experimental conditions.

---

# Phase 9 — Controlled Comparative Evaluation

To ensure a fair comparison, both models were trained using identical experimental settings.

The following parameters remained unchanged:

- Dataset
- Train-validation split
- Optimizer
- Learning rate
- Batch size
- Number of epochs
- Early stopping strategy
- Evaluation protocol

Only the input representation differed between the two models.

This allowed the influence of the quantum-inspired feature transformation to be isolated from other implementation variables.

---

# Phase 10 — Experimental Evaluation

Detailed evaluation reports were generated, including:

- Classification reports
- Confusion matrices
- Prediction files
- Learning curves
- Performance metrics

The experiments consistently indicated that the reported paper-level performance could not be reproduced using the available public dataset.

Rather than repeatedly tuning the model, the focus shifted toward understanding why the reproduction gap existed.

---

# Transition to Scientific Audit

At this stage, it became evident that architecture optimisation alone could not explain the discrepancy between the reproduced results and those reported in the original publication.

Instead of continuing to modify the model, the research direction shifted toward auditing the dataset itself.

This decision marked the beginning of the second half of the project, which focused on annotation quality, patient-level data leakage and reproducibility analysis.

---

# Phase 11 — Case and Annotation Audit

After completing the initial reproduction and the controlled comparison between quantum-transformed features and original grayscale images, the complete experimental pipeline was reviewed again.

This review revealed an important methodological issue.

The DDTI dataset contains:

- Multiple ultrasound images for several patient cases
- Multiple annotation regions within certain images
- Multiple lesion-level samples originating from the same case

Therefore, an image-level train-test split alone could not guarantee patient-level independence.

If different images or lesion regions from the same case appear in both training and testing partitions, the evaluation may contain hidden data leakage. This can produce overestimated performance because the model may indirectly encounter patient-specific information during training.

For this reason, the earlier image-level experiments were preserved as exploratory reproduction experiments, while a stricter case-level audit was introduced for the final evaluation.

---

## Audit Objectives

The Phase 11 audit was designed to verify:

- Case-to-image relationships
- XML-to-image mappings
- Multiple images belonging to the same case
- Multiple annotation regions belonging to the same image
- Annotation coordinate validity
- Bounding-box quality
- Missing or incomplete annotation records
- Previous train-test assignments
- Case-level overlap across dataset partitions
- ROI-level overlap across dataset partitions

---

## Audit Implementation

The complete audit was performed using:

```text
05_Source_Code/phase11_case_annotation_audit.py
```

The script reconstructed the dataset at three levels:

```text
Patient Case
     │
     ├── One or More Ultrasound Images
     │
     └── One or More Annotated Lesion Regions
```

For every annotation record, the script extracted and verified:

- Case ID
- Image filename
- Image number
- TI-RADS class
- Annotation points
- Bounding-box coordinates
- ROI dimensions
- Coordinate validity
- Previous split assignment

---

## Final Audit Results

| Audit Component | Count |
|---|---:|
| Valid Patient Cases | 291 |
| Valid Lesion ROIs | 449 |
| Invalid or Incomplete Records | Preserved Separately |

Invalid records were not silently removed.

They were documented in dedicated reports to maintain transparency and allow future researchers to inspect the exact reasons for exclusion.

---

## Major Finding

The audit demonstrated that the earlier image-level split could not be treated as the final patient-independent evaluation.

Multiple images or regions originating from the same case could potentially be assigned to different dataset partitions.

The earlier experiments remain scientifically useful because they document the initial reproduction attempt and the controlled comparison between quantum and classical feature representations.

However, they are not used as the final leakage-free evidence.

---

## Phase 11 Generated Outputs

The audit produced:

```text
07_Results/phase11_case_annotation_audit/
```

including:

- `annotation_coverage_report.txt`
- `annotation_inventory.csv`
- `case_distribution.json`
- `case_image_mapping.csv`
- `invalid_annotation_records.csv`
- `leaking_case_details.csv`
- `phase11_audit_summary.txt`
- `previous_split_image_assignments.csv`
- `previous_split_leakage_report.txt`
- `roi_candidate_manifest.csv`
- `roi_preview_grid.png`

---

> **Research Decision**
>
> I decided not to delete, modify or hide the earlier image-level results.
>
> Instead, I preserved them as part of the complete research history and introduced a stricter case-level protocol for the final reproduction stage.

---

# Phase 12 — Leakage-Free Case-Level Split

The objective of Phase 12 was to create a patient-independent train-validation-test split.

Instead of splitting individual images or ROI samples, the dataset was divided using unique case IDs.

All images and lesion regions belonging to the same case were assigned to only one partition.

This ensured that no patient information was shared between training, validation and testing datasets.

---

## Split Strategy

The split was created using the audited case and ROI manifests generated during Phase 11.

The following principles were followed:

- Every case appears in exactly one partition
- Every ROI inherits the split assignment of its parent case
- Validation data is used only for model selection
- Test data remains untouched until final evaluation
- Case assignments are frozen before model training
- No case is moved after observing performance
- Patient-level independence is prioritized over perfect class balance

---

## Final Case Distribution

| Split | Cases |
|---|---:|
| Training | 203 |
| Validation | 44 |
| Testing | 44 |
| **Total** | **291** |

---

## Final ROI Distribution

| Split | ROIs |
|---|---:|
| Training | 319 |
| Validation | 58 |
| Testing | 72 |
| **Total** | **449** |

---

## Leakage Verification

| Comparison | Case Overlap | ROI Overlap |
|---|---:|---:|
| Train vs Validation | 0 | 0 |
| Train vs Test | 0 | 0 |
| Validation vs Test | 0 | 0 |

The final split achieved complete separation at both patient and lesion-region levels.

---

## Phase 12 Generated Outputs

The frozen split files are stored under:

```text
07_Results/phase12_case_level_split/
```

including:

- `case_level_manifest.csv`
- `split_case_assignments.csv`
- `train_cases.csv`
- `validation_cases.csv`
- `test_cases.csv`
- `train_roi_manifest.csv`
- `validation_roi_manifest.csv`
- `test_roi_manifest.csv`
- `split_distribution.json`
- `split_audit_report.txt`

---

> **Frozen Split Policy**
>
> The Phase 12 split is treated as a permanent experimental artifact.
>
> It is not reshuffled, rebalanced or modified to obtain better performance.

---

# Phase 13 — Lesion ROI Dataset Construction

The earlier experiments used complete ultrasound images.

However, the DDTI XML files contain expert-marked lesion boundaries. These annotations make it possible to construct a more focused dataset containing only the nodule region and a small amount of surrounding context.

The objective of Phase 13 was to generate a lesion-level dataset using the validated annotation records and the frozen Phase 12 case assignments.

---

## ROI Processing Procedure

For every valid annotation region, the following operations were performed:

1. Load the original ultrasound image
2. Read the expert annotation coordinates
3. Calculate the lesion bounding box
4. Add approximately 10% contextual margin
5. Crop the lesion region
6. Convert the crop to grayscale
7. Apply square padding
8. Resize the image to `64 × 64`
9. Normalize pixel values to the range `[0,1]`
10. Preserve the case ID
11. Preserve the ROI ID
12. Preserve the TI-RADS label
13. Save the sample under its frozen split

---

## Why Square Padding Was Used

Directly resizing a rectangular lesion crop may distort its geometry.

Square padding was therefore applied before resizing so that the lesion shape remained as consistent as possible while being converted into a fixed-size model input.

---

## Final ROI Dataset Shapes

| Split | Array Shape |
|---|---|
| Training | `(319, 64, 64, 1)` |
| Validation | `(58, 64, 64, 1)` |
| Testing | `(72, 64, 64, 1)` |

---

## Associated Case Counts

| Split | Cases |
|---|---:|
| Training | 203 |
| Validation | 44 |
| Testing | 44 |

---

## ROI Processing Audit

| Audit Item | Result |
|---|---:|
| Failed ROI Records | 0 |
| Case Leakage | 0 |
| ROI Leakage | 0 |

All 449 validated ROI regions were processed successfully.

---

## Generated NumPy Arrays

The final arrays include:

### Training

- `X_roi_train.npy`
- `y_roi_train.npy`
- `case_ids_roi_train.npy`
- `roi_ids_train.npy`

### Validation

- `X_roi_validation.npy`
- `y_roi_validation.npy`
- `case_ids_roi_validation.npy`
- `roi_ids_validation.npy`

### Testing

- `X_roi_test.npy`
- `y_roi_test.npy`
- `case_ids_roi_test.npy`
- `roi_ids_test.npy`

---

## Phase 13 Audit Outputs

The corresponding reports are stored under:

```text
07_Results/phase13_roi_dataset/
```

including:

- `roi_dataset_summary.txt`
- `roi_dataset_distribution.json`
- `roi_dataset_audit.csv`
- `failed_roi_records.csv`
- `train_roi_preview.png`
- `validation_roi_preview.png`
- `test_roi_preview.png`

---

## Data Integrity Decisions

The following controls were applied:

- No augmentation was stored in the frozen arrays
- No SMOTE was applied before splitting
- No synthetic samples were added to validation or testing
- No test image was modified based on model performance
- All augmentation was restricted to the training pipeline
- The frozen Phase 13 arrays were preserved unchanged after validation

---

# Phase 14 — Final Leakage-Free Categorical CNN Baseline

The objective of Phase 14 was to perform one final controlled six-class TI-RADS experiment using the audited ROI dataset.

This experiment was intentionally designed as a compact and transparent baseline rather than an aggressive architecture-search exercise.

The main question was:

> **Can a conventional CNN learn a useful six-class TI-RADS decision boundary from the audited lesion-level dataset under a strict patient-independent evaluation protocol?**

---

## Classification Classes

| Encoded Class | TI-RADS Category |
|---:|---|
| 0 | TI-RADS 2 |
| 1 | TI-RADS 3 |
| 2 | TI-RADS 4a |
| 3 | TI-RADS 4b |
| 4 | TI-RADS 4c |
| 5 | TI-RADS 5 |

---

## Experimental Controls

The final experiment used:

- Frozen Phase 12 case-level split
- Frozen Phase 13 ROI arrays
- Compact CNN architecture
- Training-only augmentation
- Class weights calculated from training labels only
- No SMOTE
- No test-set tuning
- Early stopping
- Learning-rate reduction
- Validation-based checkpoint selection
- Untouched final test evaluation
- ROI-level and case-level metrics

---

## Case-Level Evaluation

Some patient cases contain multiple ROI samples.

Therefore, evaluation was performed at two levels.

### ROI-Level Evaluation

Each lesion region was treated as an independent prediction sample.

### Case-Level Evaluation

For every patient case:

1. Collect all ROI prediction probabilities
2. Average the probability vectors
3. Select the highest-probability TI-RADS class
4. Compare the aggregated prediction with the case label

This produced a more clinically meaningful patient-level result.

---

## Training Behaviour

| Training Item | Result |
|---|---:|
| Completed Epochs | 19 |
| Best Validation Epoch | 1 |

Training accuracy continued to increase after the first epoch, while validation performance declined.

This pattern indicated early overfitting and weak generalization to unseen cases.

The checkpoint saved at Epoch 1 was therefore used for final testing.

---

# Final Leakage-Free Results

## ROI-Level Test Performance

| Metric | Result |
|---|---:|
| Accuracy | 12.50% |
| Balanced Accuracy | 16.67% |
| Macro Precision | 2.08% |
| Macro Recall | 16.67% |
| Macro F1-Score | 3.70% |
| Weighted F1-Score | 2.78% |
| Cohen's Kappa | 0.000 |
| Quadratic Weighted Kappa | 0.000 |
| Mean Absolute Class Error | 2.319 |
| Test Loss | 1.803905 |

---

## Case-Level Test Performance

| Metric | Result |
|---|---:|
| Accuracy | 11.36% |
| Balanced Accuracy | 16.67% |
| Macro Precision | 1.89% |
| Macro Recall | 16.67% |
| Macro F1-Score | 3.40% |
| Weighted F1-Score | 2.32% |
| Cohen's Kappa | 0.000 |
| Quadratic Weighted Kappa | 0.000 |
| Mean Absolute Class Error | 2.273 |

---

## Final Model Observation

The final CNN failed to learn a reliable six-class TI-RADS decision boundary.

Predictions largely collapsed toward a single class, resulting in balanced accuracy close to random-choice performance for six classes.

The growing difference between training and validation performance showed that the model was memorizing limited training patterns rather than learning features that generalized to unseen patient cases.

---

# Controlled Quantum vs Classical Comparison

Before the final case-level audit, a controlled six-class experiment was performed using the earlier image-level dataset.

Both models used:

- The same training samples
- The same validation indices
- The same test set
- The same optimizer
- The same learning rate
- The same batch size
- The same class weights
- The same evaluation protocol

Only the input representation differed.

---

## Comparison Results

| Model | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|
| Quantum-Feature QuCNet | 18.57% | 15.16% | 6.72% | 8.74% |
| Classical Grayscale CNN | 24.29% | 28.09% | 23.18% | 22.76% |
| Majority-Class Baseline | 27.14% | — | — | — |

---

## Interpretation

The classical model outperformed the reconstructed quantum-feature model across balanced accuracy, Macro F1 and Weighted F1.

The quantum model showed near-prediction collapse and used only a limited subset of the six output classes.

The classical model predicted all six classes and demonstrated better class-wise behaviour, although its overall accuracy remained below the majority-class baseline.

Under the reconstructed and controlled conditions, the quantum-transformed features did not demonstrate an advantage over original grayscale ultrasound images.

---

# Scientific Interpretation of the Final Result

The final result should not be interpreted as a software implementation failure.

The complete pipeline executed successfully:

- Dataset parsing succeeded
- XML extraction succeeded
- Annotation validation succeeded
- Case mapping succeeded
- Leakage detection succeeded
- Case-level splitting succeeded
- ROI extraction succeeded
- Dataset serialization succeeded
- Model training succeeded
- Checkpoint selection succeeded
- Test evaluation succeeded
- All required metrics and reports were generated

The low final performance is therefore preserved as a valid experimental finding produced under a strict and reproducible evaluation protocol.

---

# Why the Reported Performance Could Not Be Reproduced

The reproduction gap is likely associated with a combination of several factors.

## 1. Missing Pathology-Confirmed Labels

The public DDTI dataset did not provide the exact benign/malignant pathology mapping required for the binary task described in the reference paper.

The earlier binary labels were therefore provisional TI-RADS-derived proxy labels and cannot be treated as equivalent to pathology-confirmed ground truth.

---

## 2. Possible Dataset Version or Subset Difference

The publicly recovered dataset contained:

- 480 ultrasound images
- 390 XML case files

These values differed from the dataset description reported in the paper.

This suggests that the authors may have used:

- A different dataset version
- A filtered subset
- Additional clinical records
- Private institutional mappings
- Different sample-selection rules

---

## 3. Incomplete Sample-Selection Details

The publication did not fully explain:

- Which exact cases were selected
- Which images were excluded
- How multiple images from one case were handled
- How multiple nodules were handled
- How missing labels were treated
- How invalid annotation regions were handled

These decisions can significantly affect the final dataset.

---

## 4. Unspecified Patient-Level Split

The paper did not provide enough information to reconstruct the exact patient-level train-test split.

An image-level split may introduce case leakage when several images belong to one patient.

The final reproduction therefore used a stricter patient-independent split, which may be more difficult but is methodologically safer.

---

## 5. Incomplete Quantum-Circuit Specification

The publication described the quantum-inspired transformation conceptually but did not provide every implementation detail required for exact reconstruction.

Unspecified details included:

- Exact circuit construction
- Gate ordering
- Parameter initialization
- Random-circuit generation
- Number of circuit repetitions
- Seed values
- Feature aggregation
- Measurement handling

The implemented quantum transformation is therefore a faithful reconstruction attempt, but not a guaranteed byte-for-byte reproduction of the authors' private implementation.

---

## 6. Incomplete Model and Training Details

Several training details required interpretation, including:

- Exact CNN layer configuration
- Initialization strategy
- Regularization
- Optimizer settings
- Augmentation procedure
- Checkpoint criteria
- Early-stopping configuration
- Random seeds
- Hyperparameter-selection process

Small differences in these settings may substantially affect results on a small medical imaging dataset.

---

## 7. Severe Class Imbalance

The six TI-RADS categories were not equally represented.

Minority classes contained limited training samples, making it difficult for the model to learn stable and generalizable patterns.

---

## 8. Visual Similarity Between Adjacent TI-RADS Categories

Adjacent TI-RADS categories may contain highly similar ultrasound characteristics.

For example:

- TI-RADS 4a and 4b
- TI-RADS 4b and 4c
- TI-RADS 4c and 5

Distinguishing these categories may require richer clinical context, larger datasets or expert-engineered features beyond the available image information.

---

# Final Reproduction Finding

The QuCNet methodology was systematically reconstructed and evaluated using the publicly recoverable DDTI dataset.

The research successfully produced:

- A complete dataset parser
- A validated metadata pipeline
- A quantum-inspired feature-generation workflow
- Classical and quantum model comparisons
- A complete annotation audit
- A leakage-free case-level split
- An expert-annotation-based ROI dataset
- ROI-level and case-level evaluation
- Complete reports, predictions and metrics

However, the high classification performance reported in the original publication could not be independently reproduced.

---

> **Final Research Conclusion**
>
> The available public dataset and the methodological details disclosed in the original paper were insufficient to reproduce the reported QuCNet performance.
>
> Under controlled comparison, quantum-transformed features did not outperform original grayscale ultrasound images.
>
> Under the final leakage-free patient-level protocol, the six-class model failed to generalize to unseen cases.
>
> These results are preserved as a transparent negative reproduction finding rather than hidden through repeated model tuning.

---

# Reproduction Closure Decision

After Phase 14, the original-paper reproduction was formally closed.

I decided not to continue adding architectures merely to search for a higher accuracy value.

Doing so would change the objective from scientific reproduction to performance optimization and would weaken the clarity of the study.

Therefore:

- The original-paper reproduction is considered complete
- The negative findings are preserved
- The leakage-free result is treated as the final reproduction evidence
- Earlier image-level experiments remain part of the historical research record
- Future improvements will be presented separately as extension work
- No further model will be presented as part of the original QuCNet reproduction

---
