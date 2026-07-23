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
| XML case files | 390 |
| Images with TI-RADS labels | 349 |
| Images without TI-RADS labels | 131 |
| Audited valid cases | 291 |
| Audited valid ROI regions | 449 |
