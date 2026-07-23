# Reproduction and Audit of QuCNet for Thyroid Nodule Classification

A systematic reproduction study of the quantum-inspired CNN proposed for
thyroid ultrasound classification using the publicly recoverable DDTI dataset.

## Research Summary

This repository presents a systematic reproduction and methodological audit of
the paper **“QuCNet: Quantum-Inspired CNN for Optimized Thyroid Nodule
Classification.”**

The study reconstructs the public DDTI data pipeline, XML annotation parser,
TI-RADS classification workflow, quantum-inspired four-channel transformation,
classical CNN baselines, case-level leakage audit, lesion ROI extraction, and
patient-independent evaluation.

The high classification performance reported in the reference paper could not
be independently reproduced using the publicly recoverable dataset and the
implementation details disclosed in the publication.

This repository preserves both successful implementation outputs and negative
experimental findings as part of a transparent reproducibility study.

## Main Findings

- The public dataset contained 480 ultrasound images and 390 XML case files.
- Only 349 images had usable TI-RADS labels.
- Public pathology-confirmed benign/malignant labels were not available.
- The earlier binary labels were therefore treated only as provisional proxies.
- A complete annotation audit identified 291 usable cases and 449 valid ROIs.
- The final case-level split contained zero case and ROI overlap.
- The classical grayscale baseline outperformed the reconstructed quantum-feature model in the controlled image-level experiment.
- The final leakage-free ROI CNN achieved 12.50% ROI accuracy and 11.36% case accuracy.
- The original paper’s reported performance could not be reproduced under publicly reconstructable conditions.

## Reference Paper

**Title:** QuCNet: Quantum-Inspired CNN for Optimized Thyroid Nodule Classification  
**Journal:** IEEE Access  
**Year:** 2024  
**DOI:** 10.1109/ACCESS.2024.3367806

This repository is an independent academic reproduction. It is not an official
implementation released by the original authors.


## Research Objectives

1. Reconstruct the publicly reproducible QuCNet workflow.
2. Audit the available DDTI data and XML annotations.
3. Compare quantum-transformed and original grayscale inputs.
4. identify data leakage and label-reconstruction risks.
5. Build a patient-independent lesion-level evaluation pipeline.
6. Document the gap between reported and independently reproduced results.

## Experimental Pipeline

DDTI Dataset
      ↓
XML and Image Audit
      ↓
Master Dataset Generation
      ↓
Image Preprocessing
      ↓
Initial Binary Proxy Experiment
      ↓
Quantum Feature Generation
      ↓
QuCNet and Classical CNN Comparison
      ↓
Case and Annotation Audit
      ↓
Leakage-Free Case Split
      ↓
Lesion ROI Dataset
      ↓
Final ROI and Case-Level Evaluation



Later, we can replace this with a polished workflow image.

---

### 6. Dataset summary

## Dataset Summary

| Item | Count |
|---|---:|
| Ultrasound images | 480 |


## Research Phases

| Phase | Description | Status |
|---|---|---|
| 0–4 | Dataset inspection, XML parsing and preprocessing | Completed |
| 5 | Dataset split, class balancing and feature preparation | Completed |
| 6 | Initial QuCNet binary baseline | Completed |
| 7 | Classical binary baseline | Completed |
| 8–10 | Six-class TI-RADS controlled comparison | Completed |
| 11 | Case and annotation audit | Completed and frozen |
| 12 | Leakage-free case-level split | Completed and frozen |
| 13 | Lesion ROI dataset preparation | Completed and frozen |
| 14 | Final ROI categorical CNN baseline | Completed |
| XML case files | 390 |
| Images with TI-RADS labels | 349 |
| Images without TI-RADS labels | 131 |
| Audited valid cases | 291 |
| Audited valid ROI regions | 449 |

## Initial Controlled TI-RADS Comparison

| Model | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| Quantum-feature QuCNet | 18.57% | 15.16% | 6.72% |
| Classical grayscale CNN | 24.29% | 28.09% | 23.18% |
| Majority-class baseline | 27.14% | — | — |


Under matched data partitions and training conditions, the quantum-transformed
features did not demonstrate an advantage over original grayscale ultrasound
images.

## Final Leakage-Free ROI Evaluation

### Dataset split

| Split | Cases | ROIs |
|---|---:|---:|
| Train | 203 | 319 |
| Validation | 44 | 58 |
| Test | 44 | 72 |

All case-level and ROI-level overlap checks were zero.

### Test performance

| Evaluation level | Accuracy | Balanced Accuracy | Macro F1 |
|---|---:|---:|---:|
| ROI level | 12.50% | 16.67% | 3.70% |
| Case level | 11.36% | 16.67% | 3.40% |

## Scientific Interpretation

The final result will  not be interpreted as a software execution failure.
The complete pipeline successfully performed dataset parsing, annotation
validation, ROI extraction, leakage-free partitioning, training, checkpoint
selection and untouched test evaluation.

The low test performance is therefore preserved as a valid negative
reproduction finding.

Possible causes of the reproduction gap include:

- unavailable pathology-confirmed labels;
- possible use of a private or different dataset subset;
- incomplete sample-selection information;
- unspecified patient-level partitioning;
- incomplete quantum-circuit details;
- incomplete model and optimization specifications;
- severe class imbalance;
- small minority-class sample sizes;
- similarity between adjacent TI-RADS categories.


## Repository Structure

| Directory | Purpose |
|---|---|
| `05_Source_Code/` | Phase-wise implementation scripts |
| `07_Results/` | Metrics, reports, predictions and figures |
| `Project_Log.txt` | Chronological research record |
| `03_Dataset_Processed/` | Locally generated arrays, excluded from Git |
| `06_Models/` | Locally generated checkpoints, excluded from Git |


## Environment Setup

bash
python -m venv .venv


.venv\Scripts\activate
pip install -r requirements.txt


python 05_Source_Code/phase11_case_annotation_audit.py
python 05_Source_Code/phase12_create_case_level_split.py
python 05_Source_Code/phase13_prepare_roi_dataset.py
python 05_Source_Code/phase14_train_roi_categorical_baseline.py



Before publishing these commands, confirm that each script actually works from the root with that syntax.



### 13. Limitations and ethics

Add:

markdown
## Limitations and Responsible Use

This repository is intended for academic research and reproducibility analysis.

- It is not a clinical diagnostic system.
- Predictions must not be used for patient-care decisions.
- The public data did not contain the original pathology mapping described for the binary task.
- Reproduced labels, outputs and conclusions are limited to the available dataset and disclosed methodology.

- ## Researcher

**Karthik B. R.**  
M.Tech Information Technology  
University Visvesvaraya College of Engineering, Bengaluru

Research interests: quantum machine learning, medical image analysis,
reproducible AI and trustworthy evaluation.
- 
