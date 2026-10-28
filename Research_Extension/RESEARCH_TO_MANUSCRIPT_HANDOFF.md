# QuCNet research closeout and manuscript handoff

**Prepared:** 28 October 2026  
**Study:** *QuCNet Revisited: An Independent Reproduction and Methodological Audit of Thyroid Ultrasound Classification*  
**Publication stage:** **Empirical research/evidence-engineering phase complete — ready to draft, NOT cleared for submission.**  
**Source of truth:** GitHub versioned code, CSV/JSON evidence and this document, not an external proposed manuscript.

## 1. Study identity, not an invented domain-shift experiment

This is an **independent partial reconstruction and methodological audit** of Gunti Swathi, Ali Altalbe and R. Prasanna Kumar, “QuCNet: Quantum-Inspired Convolutional Neural Networks for Optimized Thyroid Nodule Classification,” *IEEE Access* **12** (2024), 27829–27842, DOI [10.1109/ACCESS.2024.3367806](https://doi.org/10.1109/ACCESS.2024.3367806).

**The scientific question:** Given the original report and public DDTI snapshot, what can be reconstructed, what prevents like-for-like replication, and what do case-grouped, matched classical and fixed-simulated-quantum feature comparisons reveal about the reliability of the resulting six-class thyroid-ultrasound evaluation?

This is not the author's original QuCNet training run, a new optimized neural architecture, clinical validation, scanner/institutional domain adaptation, a hardware quantum demonstration, or a claim of quantum speedup. No distinct external-clinic dataset was evaluated. **Do not invent “53.12% external accuracy,” “state saturation caused the drop,” fabricated optimizer or GPU settings, or an unverified patient count.**

## 2. Research evidence status — no more routine training required

| Research task | Archive / reproducibility evidence | Current disposition |
| --- | --- | --- |
| Original paper-to-code fidelity | [ORIGINAL_PAPER_FULLTEXT_FIDELITY_AUDIT.md](ORIGINAL_PAPER_FULLTEXT_FIDELITY_AUDIT.md) | **Complete, at documented source fidelity**; not exact source-paper replication |
| DDTI inventory, ROI and split accounting | [FINAL_RESEARCH_VERIFICATION_MATRIX.md](FINAL_RESEARCH_VERIFICATION_MATRIX.md), `verify_frozen_case_splits.py` | **Complete for recorded case IDs**; distinct patient identities not established |
| Historical Phase 0–18 preservation | `Project_Log.txt`, `07_Results/`, [RESEARCH_COMPLETION_REGISTER.md](RESEARCH_COMPLETION_REGISTER.md) | **Preserved**, not rewritten or reinterpreted as fresh test evidence |
| Mathematical and numeric re-computation | [MASTER_VERIFIED_RESULTS.csv](MASTER_VERIFIED_RESULTS.csv), `verify_master_inventory.py`, `verify_calibration_artifacts.py` | **Completed and independently script-checked** |
| Matched feature maps, classifiers and repeated case-fold stability | `07_Results/exp_qmap_001`, `exp_models_001`, `exp_stability_001`; `verify_stability_artifacts.py` | **Completed**; exploratory, train-only OOF |
| Preprocessing, padding, annotation, fusion, cache and duplicate audits | `07_Results/exp_*`, scientific verification matrix | **Completed at recorded scope**; causal explanations not proven |
| Calibration and TI-RADS class reliability | `07_Results/exp_calib_001`, six-category confusion matrices | **Completed for archived evidence** |
| Novelty scan and thesis | [SCIENTIFIC_CONTRIBUTION_AND_NOVELTY_REVIEW.md](SCIENTIFIC_CONTRIBUTION_AND_NOVELTY_REVIEW.md) | **Internal screening complete**, guide/peer acceptance still open |
| Independent external validation feasibility | [EXTERNAL_VALIDATION_FEASIBILITY_GATE.md](EXTERNAL_VALIDATION_FEASIBILITY_GATE.md) | **Screened: no suitable like-for-like external six-class validation established** |
| Manuscript candidate visualizations | [IEEE_PRINT_FIGURE_GENERATION_RECORD.md](IEEE_PRINT_FIGURE_GENERATION_RECORD.md), workflow [40669696948](https://github.com/karthikaxiom/QCNN_Thyroid_Paper_Reproduction/actions/runs/40669696948) | **Two source-derived SVG/PDF/PNG candidate figures verified and visually inspected**; final typeset template placement open |
| DDTI XML structural/pattern screening | [DDTI_XML_AUTOMATED_TRIAGE_RESULT.md](DDTI_XML_AUTOMATED_TRIAGE_RESULT.md), workflow [40668545283](https://github.com/karthikaxiom/QCNN_Thyroid_Paper_Reproduction/actions/runs/40668545283) | **390/390 parsed; 2 of 21 narrative fields flagged for optional qualified follow-up**; no privacy clearance claim |
| Dataset release terms, images and consent | [DDTI_DATA_RIGHTS_AND_PRIVACY_GATE.md](DDTI_DATA_RIGHTS_AND_PRIVACY_GATE.md) | **OPEN**; original JPG/XML already publicly tracked, but unrestricted onward rights not independently established |

**Frozen experimental scope.** The original archive contains 480 JPG files and 390 XML files, with 349 images carrying recoverable six-class TI-RADS labels. The strict Phase 13 set has **440 ROIs from 291 recorded case IDs**, allocated **311 train ROIs/203 cases, 58 validation ROIs/44 cases, 71 historical test ROIs/44 cases**. The labels are **2, 3, 4a, 4b, 4c, 5**, not pathology-confirmed outcomes. No case ID repeats across those recorded partitions, but patient identity uniqueness is unknown.

**Interpretive anchors.**
- Historical Phase 14 case result: **2/44 = 4.55%**, with single-class collapse. It is a previously accessed test, **not a fresh untouched test for exploratory extensions**.
- Main development experiments: **203 recorded training case IDs**, case-stratified OOF; not verified 203 separate patients.
- Classical 64-dimensional logistic **54/203 = 26.60%** versus fixed simulated four-qubit-feature logistic **43/203 = 21.18%** in the matched seed-42 analysis.
- Fixed quantum forest **63/203 = 31.03%**, classical forest **54/203 = 26.60%** at seed 42. Paired exploratory p≈0.176; superiority not established.
- Five-fold-assignment-seed RF average accuracies: quantum **29.36%** versus classical **27.49%**, with a ranking reversal and weak minority-class performance. Both RF arms show **0/10 TI-RADS 3 recall at each seed** on the *same ten recorded cases*.
- The majority-class baseline is **57/203 = 28.08%**. A slightly higher single-fold accuracy is not clinical benefit or robust superiority.
- The source authors' **97.63% tumor** and **93.87% suspicious-level** scores use other denominators and endpoints. They must **not** be plotted as directly matched test baselines. The **55/100 approximate protocol-fidelity score** in the fidelity audit is **not a model accuracy**.

## 3. Citation, figures and IEEE writing order

- Original QuCNet paper: DOI [10.1109/ACCESS.2024.3367806](https://doi.org/10.1109/ACCESS.2024.3367806), pp. **27829–27842**.
- DDTI primary source: Pedraza, Vargas, Narváez, Durán, Muñoz and Romero, *An open access thyroid ultrasound image database*, *Proc. SPIE* **9287**, 92870W (2015), DOI [10.1117/12.2073532](https://doi.org/10.1117/12.2073532).
- Original-reference and contemporary-reproducibility citations should be copied from verified primary publisher records, not guessed from another AI-generated draft.
- Main figures should include: audited dataset/split provenance; reported-versus-reconstructed protocol; matched model-family comparison; repeated-fold sensitivity; classwise and calibration failure patterns. The first two redesigned double-column vector candidates are available through GitHub Actions run **40669696948**, with a source generator and SHA-256 manifest. Article-specific print proof is not done.
- Draft in this sequence: **document scaffold and BibTeX → exact methods/data provenance → auditable results/tables/figures → related work/limitations → introduction/discussion/conclusion → abstract/keywords/title polish → compiled PDF checking**. Editing order need not match published section order.
- Suggested article sections: **I Introduction; II Related Work; III QuCNet Fidelity and Reproduction Scope; IV Dataset and Evaluation Protocol; V Experimental Results; VI Discussion and Threats to Validity; VII Conclusion.** This is an authoring plan, not a mandatory seven-section IEEE rule.

## 4. Submission-stage obligations — do not falsely mark them complete

These do **not** prevent drafting, but matter before any journal submission:
1. **Guide/coauthor review:** Validate contribution, novelty, statistical interpretation, intended venue and author order/roles. No acceptance or endorsement has been recorded.
2. **Raw-medical-data rights/privacy:** Seek source-specific release clarification or institutional decision; avoid publishing raw medical image examples or redistributing source XML as supplementary files until authorized. No Git history has been erased. The existing public raw-file history needs an informed owner/institutional rights review.
3. **Clinical generalization:** The existing work provides **no independent pathology-grounded or patient-verified multicenter validation**. This is a declared scientific limitation, not a task that can be ticked as completed by assertion.
4. **Article production:** Exact venue/template selection; true-size IEEE figure placement and captions; bibliography/link checks; LaTeX clean compile; source/PDF agreement; authors, affiliations, ORCID and conflicts/funding/data/ethics statements where applicable.
5. **Disclosure:** Describe any AI-generated article content according to the target journal's current editorial rules. Never invent personal research involvement or obscure use by changing fictional hardware settings.

## 5. Repository governance and clean handoff

**Empirical research is frozen.** Do not rerun, rename, delete, overwrite or force-push historic result files. A reproducible correction to an objectively identified mathematical/code error can be reviewed separately, but **no new exploratory model training** is part of the normal writing workflow. Paper-specific writing and derived publication graphics should live in a clearly separate manuscript/Overleaf workspace, with source figure scripts and evidence cross-referenced to the repository. The active research branch remains available as an audit trail.

The current repository has a public history containing source DDTI JPG/XML that pre-dates this closeout. Merging the research branch **does not erase existing raw files** or confer permission to redistribute them. It merely makes verified results and corrected governance notes visible on the default branch.

**Chronological audit completion:** [Project_Log.txt](../Project_Log.txt) has been extended with a **28 October 2026** addendum after the unchanged June-July entries. It records the source-fidelity and EXT-* evidence, objective corrected results and denominators, full XML screening limits, corrected visual exports, final GitHub merge, and the Overleaf screenshot verification. The official IEEE Access `access.tex` example has been imported and compiled into an eight-page sample in the researcher's Overleaf workspace, but the actual manuscript is **not started**. No claim of warning-free final compilation or submission is made.

**Next user-facing action:** Keep the imported official IEEE Access Overleaf project intact, clear the example content only when drafting starts, and author **one locked evidence-backed section at a time**. Do not request another 21-entry privacy review or treat old worklog “pending” headings as current research tasks.
