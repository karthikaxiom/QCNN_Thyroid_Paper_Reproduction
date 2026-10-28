# Full-text QuCNet method-fidelity audit — final experimental closeout

**Status: COMPLETE at the level supported by the original published article and saved implementation.** Completed 28 October 2026 using the actual 14-page `01_Paper/Main_Paper.pdf` (IEEE Access, 2024, pp. 27829–27842) and the versioned Python source. This document is a verification artifact, **not a manuscript draft**.

## Precise source

Gunti Swathi, Ali Altalbe, and R. Prasanna Kumar, *QuCNet: Quantum-Inspired Convolutional Neural Networks for Optimized Thyroid Nodule Classification*, IEEE Access, vol. 12, pp. 27829–27842, 2024. DOI: 10.1109/ACCESS.2024.3367806. PDF on the research branch: `01_Paper/Main_Paper.pdf`. The publication's PDF copyright notice indicates **CC BY-NC-ND 4.0** for this paper; this does not independently grant unrestricted redistribution of the underlying DDTI medical dataset.

## Exact source-to-implementation comparison

| Aspect | What the published article says | What this repository demonstrates | Fidelity judgment |
| --- | --- | --- | --- |
| Original DDTI inventory | **478 images / 399 medical cases** reported in §III (printed p. 27831); **460** said to have a binary benign/malignant categorization | This archived snapshot contains **480 JPG images** and 390 distinct case IDs; the locally audited six-class records have **349 labeled image records** | Different inventory/snapshot and independently recoverable labels; cannot assert identical evaluation inputs |
| Endpoints | Benign/malignant *tumor classification* and a six-category TI-RADS *suspicious level* classification (§§III,V,VII, pp. 27831–27839) | Six source TI-RADS categories recovered; historical binary labels are **derived from TI-RADS suspicion grouping**, not verified pathology | **Different target provenance**, especially the binary task |
| Input resizing | 128×128 images, then quantum transformation into four 64×64 channels (§§IV–V, pp. 27832–27834) | Historical Phase9 uses stored 64×64×4 quantum maps; strict Phase13 uses **64×64×1 ROI crops with median-intensity square padding** | Phase13 is a **separate case-stratified ROI experiment**, not the paper's full-image input protocol |
| Normalization | Mean subtraction plus standard deviation scaling (§IV, p. 27832) | Phase13 ROI tensors are float32 /255; previous image preprocessing code did not establish identical paper standardization | Not an exact preprocessing reproduction |
| Sampling and augmentation | 80/20 **image** train/test split; training zoom **0.99–1.01**, brightness **0.8–1.2**, horizontal flip (§IV, p. 27832) | Historical image-level held-out records have case-overlap limitations; Phase12/13 uses separately frozen train/validation/test **case** splits; strict ROI augmentation differs | Protocols differ; no fair direct performance equivalence |
| Class balancing | **SMOTE** in source preprocessing (§IV, p. 27832) | Original binary branches used a documented SMOTE pipeline; corrected matched Phase9/10 and strict Phase14 experiments use class weights rather than source-equivalent SMOTE | Implementation differences must be explicit |
| Quantum filter | Random simulated quantum circuit (§V); 2×2 patch, **two quantum layers**, four expectation outputs per patch, arranged into 64×64×4 after 128×128 input (§§V–VI, pp. 27833–27835) | PennyLane `default.qubit`, four RX-encoded wires, two layers of seeded fixed RY gates and CNOT ring, four Pauli-Z means. EXT-QMAP uses this reference circuit at **8×8 pooled resolution**, 16 2×2 patches, yielding 64 features per ROI | Only **structural** and fixed-filter consistency; not gate-for-gate reproducibility of the paper's unreported exact random circuit |
| Quantum simulator | The original reports **Qiskit**, Tesla P100 and 25GB RAM (§VI, p. 27835) | Current fixed circuit uses **PennyLane**, local CPU environment and analytic simulation; no quantum hardware | Backend and environment differ |
| Encoding scale | §V encoding text calls an `Rx(2)` rotation; §VI describes local rotations **scaled by π** (pp. 27833 and 27835). Exact mapping is not fully specified | Saved implementation uses `RX(float(pixel))` without multiplying by π | **Source has incomplete/ambiguous rotation scaling**; cannot claim exact encoding equivalence |
| Binary classifier | Source Fig. 5 plus narrative: quantum four-channel output, flattening and dense classifier (§V, p. 27834); binary cross-entropy (§VI, p. 27835) | Reconstructed binary dense network includes specific layer widths chosen by implementer; earlier binary evaluation 38/70 quantum with test-selection caveats | Broad pipeline correspondence only |
| Six-category classifier | Source Fig. 6 / equations 12–15: two Conv2D/MaxPool pairs, flatten, two dense layers and softmax (§V, pp. 27834–27835); categorical CE (§VI, p. 27835) | Phase9 model defines Conv2D 16/32, Dense 128/64, learning rate 1e-4 — **implementation decisions** | Layer family matched; weights, filters/dense widths and learning rate not uniquely recoverable |
| Optimizer and callbacks | Adam; ReduceLROnPlateau patience **4**, factor **0.5**, min learning rate **1e−6**; EarlyStopping patience **3**; checkpointing (§VI, pp. 27835–27836) | Historical Phase9 training config has `EPOCHS=120` and `VALIDATION_SIZE=0.20`; strict Phase14 uses early stopping patience **18** and ReduceLR patience **7** | Not identical hyperparameter schedule |
| Published training | Original tumor task trained up to 50 epochs (best at epoch 26); suspicious-level task up to 100 epochs (best at 73); run times approx 8.15 and 16 hours (§VI, p. 27836) | Different local experiment durations/seeds/environments; no matched original hardware benchmarking | Timings and stopping curves cannot be replicated by claim |
| Test denominators | Abstract and §VII report **127 test images** and **97.63%** accuracy for tumor classification; **98 test images** and **93.87%** for suspicious level | Earlier local Phase9/10 **70 images**; strict Phase14 **71 ROIs / 44 cases** (already inspected) | **Not comparable estimates**, separate sample unit, inventory, labels and holdout protocols |

## Paper equation scrutiny

The paper's equations (1)–(7) define generic encoding–random circuit–measurement–channel assembly and are not a fully reproducible hardware/circuit specification. Its Conv2D/maxpool and dense equations (8)–(15) define broad layer operations, not precise layer widths or initializers. Equations (16)–(18) provide standard binary cross-entropy, categorical cross-entropy and ReLU. Equation (19), **as printed on p. 27836**, typesets the softmax denominator with a lower summation bound shown as `j=i`, rather than a conventional sum over all classes. **Use the mathematically valid softmax** (p_i=\exp(z_i)/\sum_{j=1}^{K}\exp(z_j)) when describing our own implementation, and do not silently reproduce the apparent source-notation anomaly as correct.

The independent metric audit uses:

- `accuracy = trace(C) / sum(C)`;
- `balanced accuracy = (1/K) Σ_i C_ii / sum_j C_ij`;
- `macro-F1 = (1/K) Σ_i 2 TP_i / (2 TP_i + FP_i + FN_i)` with declared zero division;
- paired McNemar two-sided exact binomial test on discordant **case** correctness;
- 5,000 paired whole-case bootstrap resamples for exploratory intervals, without refitting models.

These were directly recomputed from saved predictions on the research branch. Different experimental protocols are **not** collapsed into a synthetic single accuracy claim.

## Critical discussion boundaries

1. Do **not** frame our strict ROI classifier as a failed exact replication of the authors' 97.63% / 93.87% test performance. It is an explicitly more restrictive but different protocol.
2. Do **not** claim the old binary proxy is pathology-verified benign/malignant.
3. Do **not** attribute the original Phase14 collapse *causally* to square padding or case separation. Exploratory ablations varied several representation properties and their paired intervals include zero.
4. Do **not** interpret a fixed, 64-feature, four-qubit circuit as an optimized, trained quantum CNN or a full quantum-advantage test.
5. No new untouched test or independent patient-level external validation exists after exploratory analysis of previous held-out splits.
6. The source's ambiguity/incomplete specification must be presented as a reproduction limit, not a claim that the authors' entire model is incorrect.

**Fidelity gate decision:** The full paper-to-code methodology comparison is now documented and reconciled. It supports an **independent reproduction and methodological audit**, not a numerical claim of exact original implementation equivalence. The only manuscript-readiness concerns remaining are dataset permission/disclosure and figure presentation, together with normal author/guide review.

## Independent six-checkpoint replication scorecard

**Definition:** The score below measures fidelity of the historical paper-structured code to the original publication's identifiable implementation details. It does not measure the quality of this research, model accuracy, or the value of the later case-separated methodological audit. The rubric is necessarily judgment-based because the original exact quantum circuit and sampling inventory are not fully recoverable.

| Checkpoint | Maximum | Score | Evidence and deviation |
| --- | ---: | ---: | --- |
| 1. Quantum filter and channel geometry | 20 | **15** | Genuine PennyLane QNode on four qubits; RX encoding, two seeded fixed RY/CNOT-ring layers, four Pauli-Z outputs. Phase 5 emits 64x64x4 feature maps for 128x128 images. Backend (PennyLane versus reported Qiskit) and angle scaling are not source-exact. |
| 2. Dataset processing and splits | 20 | **4** | Recovered 480 images versus article's 478; different case counts and test denominators. Code scales pixels to [0,1], not the described mean subtraction/std scaling. Image-level split originally permits case overlap; Phase 12 case split is safer but intentionally differs from the article's 80/20 image split. |
| 3. Classical architecture | 20 | **13** | Binary Phase 6 follows flatten + dense + two outputs; early six-class Phase 8 was dense-only, not the source-described two-convolution classifier. Corrected Phase 9 uses Conv2D(16,3x3), pool(2x2), Conv2D(32,3x3), pool(2x2), flatten, dense(128), dense(64), six-way softmax. Widths/kernels are not uniquely recoverable from article. |
| 4. Training configuration | 15 | **8** | Adam and appropriate cross-entropy objectives are present. Binary Phase 6: LR 0.001, batch 16, 50 epochs, EarlyStopping patience 3 and ReduceLR patience 4. Six-class Phase 9: LR 0.0001, batch 16, 120 epochs, EarlyStopping patience 15 and ReduceLR patience 6, differing from the source suspicious-level experiment (100 epochs and original callbacks). |
| 5. Metrics and baseline comparisons | 15 | **7** | Historic metrics re-derived correctly. Local six-class quantum 13/70 = 18.57% and historical binary suspicion-proxy 38/70 = 54.29%, whereas source reports 93.87% on 98 six-category samples and 97.63% on 127 binary samples. Different labels, cases, splits, and test exposure prohibit a causal interpretation of this gap. |
| 6. Shortcuts and transparent audits | 10 | **8** | Split limitations, TI-RADS proxy provenance, normalization and model choices are now disclosed. However, original quantum caching hashes patches via 8-bit quantized values, potentially reusing a quantum result for a different floating-point patch, especially with SMOTE-generated images; collision frequency is not verified. |
| **Total** | **100** | **55** | **Approximate original-protocol fidelity: 55/100**, not a measured model-performance metric. |

### Concrete code-level verification

1. **Not a mock quantum filter.** File: 05_Source_Code/utilities/quantum_utils.py. Device: PennyLane default.qubit (four wires). The function decorated with qml.qnode executes RX encoding, two RY/CNOT-ring layers, and Pauli-Z expectation measurements; its NumPy conversion is a wrapper around genuine simulated quantum evaluations. It does not use classical Sobel/Gabor operators instead of a quantum circuit. The circuit weights are *fixed random parameters*, not trainable parameters.
2. **Correct historical channel count, but different later feature map.** 05_Source_Code/utilities/dataset_transformer.py explicitly allocates (64,64,4) outputs from 2x2 patches of 128x128 input. EXT-QMAP-001 instead pools to 8x8 and creates 64 total fixed quantum features; it must not be described as the published full-resolution quanvolutional architecture.
3. **Potential quantum-cache approximation.** In dataset_transformer.py, each cache key is formed from (patch*255).astype(uint8).tobytes(). Distinct floating-point values can share a key. This does not prove corrupt historical predictions, but a no-cache verification or exact-float-key comparison is needed before claiming bitwise fidelity. The concern is strongest for synthetic SMOTE-interpolated training patches.
4. **Data normalization not exact.** image_preprocessor.py and phase5_prepare_dataset.py divide grayscale pixels by 255. This differs from documented source mean subtraction and standard deviation normalization.
5. **Historical holdout contamination.** phase6_train.py explicitly passes X_test/y_test as validation_data to fit with EarlyStopping and ModelCheckpoint. That holdout cannot also be reported as a previously untouched test. The corrected strict experiments and later case-fold comparisons keep their different evaluation evidence levels separate.
6. **Six-class architecture must be labeled by phase.** phase8_qucnet_model.py uses Flatten/Dense/Dropout; the intended paper-structured two-Conv2D/two-MaxPooling architecture is phase9_paper_tirads_model.py. The historical Phase 9 uses sparse categorical cross-entropy with integer labels, a valid counterpart to categorical CE with one-hot labels.
7. **Published source targets confirmed externally.** The authors' institutional publication record reports tumor accuracy/precision/recall 97.63/97.72/97.64% on 127 images and suspicious-level 93.87/94.58/93.88% on 98 images. These are source claims, not the recovered dataset's estimates; both precision and recall targets must not be presented as achieved in this repository.

### Priority modifications before any claim of exact reproduction

- **P0: Target provenance.** Establish source-verified benign/malignant outcomes distinct from TI-RADS risk categories. Otherwise retain the explicit term *binary suspicion proxy*, never pathology-confirmed tumor detection.
- **P0: Matched original image inventory.** Identify precisely the author's 127 and 98 test samples and source train/test membership. Without these IDs or a verified equivalent subset, no code adjustment can establish one-to-one numerical reproduction.
- **P0: Quantum cache collision audit.** Measure distinct floating-patch values that map to identical 8-bit cache keys; if collisions exist, create a new versioned no-cache reference test rather than overwriting historical outputs.
- **P0: Independent test protocol.** Do not describe Phase 6 test-as-validation or historically inspected held-out scores as untouched independent test results. Preserve the strict case-separated experiments as a separate methodological audit.
- **P1: Source-compatible processing.** Document actual mean/std normalization, augmentation (zoom 0.99–1.01, brightness 0.8–1.2, horizontal flip), train-only SMOTE and 128x128 full-image pipeline. Do not impose these choices on the case-separated ROI audit simply to raise apparent fidelity.
- **P1: Circuit and classifier parameters.** State the uncertainty in quantum rotation scaling (source descriptions differ), Qiskit/PennyLane backend differences, source-unspecified CNN widths, learning rate and optimizer conventions. Use Phase 9, not the earlier Phase 8 flatten-only network, as the historical paper-structured TI-RADS architecture.
- **P2: Metric completeness.** Preserve confusion matrices and report true sample counts, precision, recall, F1 and the actual class target at ROI, image and case levels as appropriate. Separate source claims from local observations.
- **P2: Reproducibility metadata.** Preserve actual environment, random seeds, source hashes, fold assignments, training/validation selection, original results and all adverse findings.

**Decision:** The existing evidence supports a submission centered on an *independent reproduction and methodological audit*. It does not support an exact-match QuCNet reproduction, clinical diagnosis claims, a proven source-paper error, or a claim of quantum advantage. This technical review is not authorization to create a manuscript; paper drafting and Overleaf remain explicitly deferred until requested.
