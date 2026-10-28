# QuCNet reproducibility study — research extension protocol

**Status:** Experimental investigation in progress. The final submission manuscript is deliberately deferred.

This protocol complements the existing README, Project Log, and extension completion register. It does not replace or rewrite the historical Phase 0–18 research record.

## Research invariants

1. Preserve the default branch, the historical README, Project Log, source code, Phase 0–18 results, model checkpoints, and the fixed Phase 12/13 split.
2. Add extension studies to the existing source-code and results folders. Do not alter historical metrics to make a result look better.
3. Use the Phase 13 **training cases only** for exploratory feature comparisons and model choices. Previously viewed validation outcomes remain development evidence; the historical test set is not an untouched test for new extensions.
4. Maintain the distinction between ROI, image, case, and confirmed patient. Case-ID separation is documented; patient-ID independence has not been demonstrated.
5. Preserve the original TI-RADS labels (2, 3, 4a, 4b, 4c, 5). Prior binary labels are deterministic **TI-RADS-derived suspicion proxies**, not independently verified pathology outcomes.
6. Record the environment, seeds, sample counts, input source hashes, fold assignments, preprocessing, model configuration, class distributions, and failure cases.
7. Figures must have regenerable source data, appropriate denominators, unobscured axes, and clear units. Synthetic software tests must never be presented as empirical research results.

## Extension status

| ID | Research question | Current evidence |
| --- | --- | --- |
| EXT-BN-001 | Effect of Batch Normalization running-statistic mismatch on Phase 14 | Completed, training-only recalibration. Validation changed from 2/58 to 11/58 correct, still below the 17/58 majority baseline. |
| EXT-DEV-001 | Do pooled image intensity/gradient features carry TI-RADS information? | Completed, case-grouped training CV; case accuracy 22.17%, below the 28.08% majority baseline. |
| EXT-DATA-001 | Binary-label provenance and case split | Completed, 349 images with TI-RADS labels, 131 without; derived binary proxies, no case-ID overlap in frozen partitions. |
| EXT-META-001 | ROI annotation geometry alone as a predictor | Completed, training CV; case balanced accuracy 22.70% and case macro-F1 17.84%. |
| EXT-FUSION-001 | Image, geometry, and concatenated features | Source committed. **Real-data execution pending**. |
| EXT-FUSION-001-STATISTICS | Paired case statistics and publication-ready source figures | Source committed. Run only after real EXT-FUSION-001 output exists. |
| EXT-INTEGRITY-001 | Independent recomputation of historical saved prediction metrics | Source committed. Real-data execution pending; prior direct audit compared major historical metrics. |

## Completion gates

### Gate A — paired development evidence

- Execute fusion CV on the original *training* arrays, retaining all out-of-fold probabilities, true labels, case IDs, configuration and hashes.
- Run the statistical audit after the fusion outputs exist; verify reported measures by regenerating them from the saved predictions.
- Inspect full case-level confusions, class counts, paired-correctness disagreements, and data-quality limitations.
- Treat case-bootstrap intervals and exact McNemar tests as exploratory evidence only. Do not infer independent clinical validation.

### Gate B — dataset and preprocessing integrity

- Reconcile annotation candidates, split manifests, and Phase 13 ROI metadata without changing frozen data.
- Analyze ROI resampling, field-of-view distortion, ultrasound overlays, crop context, and imaging artifacts. Use predefined ablations, not repeated unconstrained tuning.
- Where feasible, search for exact and near-duplicate image content across case IDs. Zero case-ID overlap alone cannot exclude cross-ID duplicated images.
- Verify the origin, permitted use, and redistribution rights of dataset images and reference-paper PDFs. Until independently established, pathology-label claims remain excluded.

### Gate C — controlled model investigation

- Select further classical and simulated quantum experiments based on verified Gate A/B evidence.
- If a new quantum/classical comparison is warranted, match case folds, training data, augmentation, feature information, optimization budget and evaluation units.
- Document circuit data encoding, qubit count, layers, parameter counts, measurement/readout, analytic or shot-based simulation, runtime and hardware used.
- Freeze a prespecified protocol, use multiple seeds where practical, and preserve every result including unsuccessful runs.

### Gate D — mathematics and figures

For a \(K\)-class confusion matrix \(C\), independently calculate:

\[
N=\sum_{i=1}^{K}\sum_{j=1}^{K}C_{ij},\qquad
\operatorname{Accuracy}=\frac{\sum_{i=1}^{K}C_{ii}}{N}.
\]

\[
\operatorname{BalancedAccuracy}
=\frac1K\sum_{i=1}^{K}\frac{C_{ii}}{\sum_j C_{ij}},\qquad
\operatorname{MacroF1}=\frac1K\sum_{i=1}^{K}\frac{2P_iR_i}{P_i+R_i},
\]

with zero-division handling declared. Confirm all ROI-versus-case denominators, model parameter counts, quantum circuit dimensions, tensor shapes, split counts, seeds, and preprocessing math.

For paired case-level predictions of models \(A,B\), let \(c_{im}\) indicate model \(m\) is correct on case \(i\). The paired accuracy difference is:

\[
\Delta_{A\rightarrow B}=\frac1n\sum_{i=1}^{n}(c_{iB}-c_{iA}).
\]

Exact McNemar testing uses only discordant case-correctness pairs. Bootstrap uncertainty resamples *entire cases* in the paired data. This does not capture model-refitting uncertainty and is not confirmatory evidence under repeated exploratory comparisons.

Prepare source-backed visuals covering dataset flow, label provenance, train/validation learning curves where meaningful, ROI and case confusion matrices, model output distributions, cropped-ultrasound examples, paired differences, and reproducible training/quantum pipelines. Export high-resolution PNG and vector SVG/PDF; validate readability at IEEE column width before the manuscript stage.

### Gate E — research-to-manuscript handoff

- Verify all primary metrics and support every conclusion with source evidence.
- Reconcile historical and extension protocols explicitly; do not conflate proxy binary labels with cancer diagnoses or compare unmatched evaluation setups as though controlled.
- Resolve citation and licensing issues, establish novelty, and obtain the guide's feedback.
- **Only after Gates A–D and unresolved blockers are documented**, update the existing README with a narrow extension appendix rather than wholesale rewrites. Prepare an IEEE/Overleaf manuscript only after research readiness is declared.

## Working procedure

At the end of each active session record the repository branch/commit, last verified results, outputs still to run, and the single next executable action. No experiment is performed asynchronously while the chat is inactive.

**Next local execution:** synchronize the research branch, then run the fusion CV. After it creates real prediction files, run the fusion statistical audit and historical metrics verification scripts.
