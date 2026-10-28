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
