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
