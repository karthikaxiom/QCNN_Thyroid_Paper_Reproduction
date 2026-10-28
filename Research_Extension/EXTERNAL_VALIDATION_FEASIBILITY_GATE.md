# Gate 2 — Independent External Validation: Clinical and Dataset Compatibility Review

**Research record · 28 October 2026 · QuCNet/thyroid-ultrasound reproduction**  
**Decision:** **NO-GO for directly reporting external six-category validation with the datasets screened below.** A *new, explicitly separate* pathology-grounded or ACR TI-RADS study may be feasible after access, protocol, target and clinical-label review. **No external data were acquired; no model was tested; no clinical performance result exists.**

## Why I examined this

The QuCNet reproduction uses source-assigned **six-category TI-RADS labels 2, 3, 4a, 4b, 4c and 5**. The primary repeatability analysis reuses 203 recorded Phase 13 *training case IDs*. Case-ID separation has been verified; patient identity has not. To be genuine external validation, a candidate cohort needs independently sourced examples, defensible patient grouping and **the same evaluation target or a clinically justified, preregistered mapping**. Extra images or pathology labels alone do not make results comparable.

I screened current primary dataset records and the literature rather than downloading material and choosing an attractive-looking test set. The decision below concerns **task compatibility**, not whether these datasets are valuable for other research.

## Candidate-by-candidate assessment

| Candidate (primary record) | Evidence available as of 28 October 2026 | Six-category QuCNet target compatibility | Access and decision |
| --- | --- | --- | --- |
| **ThyroidXL, MICCAI 2025** · [MICCAI paper](https://papers.miccai.org/miccai-2025/0928-Paper2024.html), [authors' project](https://github.com/VNPT-AI-Official/ThyroidXL), [primary HF dataset](https://huggingface.co/datasets/hunglc007/ThyroidXL) | Authors' project describes **11,635 images / 4,093 patients**, pathology-supported benign/malignant outcomes, radiologist segmentation, patient information, and **ACR TI-RADS**. The MICCAI record links the original dataset at `hunglc007/ThyroidXL`. | **Not directly compatible.** ACR TR1–TR5 groups do not coincide with archival levels 2,3,4a,4b,4c,5. Histopathology-supported binary labels cannot be substituted for the historical **TI-RADS-derived suspicion proxy**. A separate endpoint definition and appropriate supervision would be required. | The **primary dataset page is gated**; files cannot be read without accepting access conditions and sharing account contact information. No access was attempted. **NO-GO for the existing six-class external test; conditional candidate for a separate appropriately designed pathology study**. |
| **Stanford AIMI thyroid cine-clips** · [Stanford source](https://aimi.stanford.edu/data), [RSNA data card](https://atlas.rsna.org/cards/be1e0f02-880f-49be-8d2f-618369334728), [original clinical paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC9152684/) | **167 distinct patients / 192 biopsy-confirmed nodules**, ultrasound cine-clip frames, ACR TI-RADS structured reports, strong source-level clinical provenance, **175 benign / 17 malignant nodules** in the published cohort. | **Not directly compatible** with the six historical risk categories. Video frames are correlated within patient/nodule and cannot be treated as independent still-image test cases. The strong pathology label does not replace the QuCNet grading target. | Stanford datasets are offered for **non-commercial research subject to dataset-specific data-use terms and account access**. **NO-GO for a same-target external six-class test; conditional separately authorized clinical study**. |
| **TN3K** · [primary authors' repository](https://github.com/haifangong/TRFE-Net-for-thyroid-nodule-segmentation) | Authors report **3,493 ultrasound images** for **nodule segmentation** with annotated masks and multiple device views. Classification labels are referenced through a **separate related ACL publication**, not an official six-category label correspondence here. | **Not demonstrated.** This segmentation release does not establish six harmonized historical TI-RADS outcomes or patient-linked external grading labels. | Read-only public repository inspected; the stated **MIT license refers to code**, not necessarily unrestricted image/data reuse. **NO-GO for current six-class test**. |
| **TN-SCUI2020** · [challenge's own data policy](https://tn-scui2020.grand-challenge.org/Dataset/) | Organizers describe **3,644 training images / 3,644 patients**, nodule masks, binary benign/malignant annotations. | **Not compatible** with archival six-category labels; binary classes differ in supervision and meaning. | Crucially, **organizers restrict image publication rights to challenge purposes**. This is an explicit counterexample to assuming accessible/public medical imagery may be republished in a new study. **Exclude from our non-challenge publication plan unless new direct permission is obtained.** |

### Important ThyroidXL source-data consistency issue

The **authors' GitHub project** and the published paper describe 11,635 images / 4,093 patients, with a 9,541-image train / 2,094-image test partition (these image totals add to 11,635). The **accessible public Hugging Face dataset card** has *internally conflicting text*: it mentions **11,545** total images, **3,275** training patients, and **739** test patients while listing 2,477 benign + 877 malignant for train (**3,354**) and 386 benign + 453 malignant for test (**839**). Its stated train/test image counts still add to 11,635. **Do not silently choose an inconsistent patient/class denominator.** Before any future analysis obtain authorized access, verify actual metadata and the dataset version/hash, resolve these counts with the provider, and distinguish a publisher claim from locally verified records.

Sources: [authors' project](https://github.com/VNPT-AI-Official/ThyroidXL), [MICCAI 2025 paper](https://papers.miccai.org/miccai-2025/0928-Paper2024.html), [HF dataset card](https://huggingface.co/datasets/hunglc007/ThyroidXL). No claim of error in the underlying data itself follows from conflicting public-facing documentation.

## Label-system caution: no automatic equivalence

The source study's classes **2, 3, 4a, 4b, 4c, 5** and the *ACR* point-based **TR1–TR5** are different grading schemes. Under ACR, **TR4** covers several score values but is not a ready-made direct mapping to *4a/4b/4c*. Published comparisons distinguish ACR, Kwak and EU TI-RADS systems and warn against treating their cutoffs as synonymous. See [comparative thyroid classification analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC7019412/).

**Not admissible:** (a) dividing ACR TR4 into QuCNet classes 4a–4c arbitrarily; (b) treating pathology-based benign/malignant as the old TI-RADS risk-proxy labels; (c) computing six-class accuracy against unavailable or mismatched labels; (d) pooling images of the same patient across train/test; (e) retroactively declaring the old 44 held-out case IDs to be a newly untouched test; (f) choosing remapping/thresholds using the proposed test labels.

## Scientifically defensible options

**Option A — Preserve this study's original scientific question (recommended).**  
Report the current DDTI case-ID-grouped, exploratory **methodological reproduction and reliability audit** with its limits. It includes valid negative and mixed findings, genuine simulated quantum features and reproducible derived metrics. State **no compatible independent six-class clinical cohort was validated**. This is a *clear limitation*, not a coding error or a hidden missing experiment.

**Option B — Separate future pathology-grounded clinical research (conditional).**  
If there is a strong, guide-approved need to evaluate generalization, design a *different* prospective question using ThyroidXL or Stanford, such as benign/malignant classification using **their verified clinical ground truth**. This **cannot be sold as direct external validation of the existing six-class QuCNet model** and requires a fresh approved protocol and training strategy, balanced classical comparators, appropriate ethical/data terms, patient-disjoint partitions, and strict untouched test discipline. Explain that this is *new research* rather than a retrospectively repaired source reproduction.

**Option C — Truly matching six-category external grading (currently unestablished).**  
Requires a genuinely separate cohort explicitly labeled under the same six-category convention, with independently reliable annotation, confirmed patient IDs, usage rights and locked testing. If only ACR annotations are available, an expert-defined and clinically justified target mapping or fresh specialist re-annotation must be approved *before* accessing test outcomes. We have found **no verified plug-and-play dataset meeting these conditions** in the present targeted screen; this is not proof none exists globally.

## If Option B or C is formally approved: preregistration gate

1. **Define endpoint first**: exact clinical target and taxonomy, study question, data source, license/access terms, and publication consent. Preserve source labels unchanged. For binary pathology, do not back-fill DDTI TI-RADS proxy cases as pathology labels.
2. **Document source version**: official URI, accepted terms/date, directory/file counts and SHA-256, label dictionary, missingness and records of excluded samples. If counts disagree, stop until reconciled.
3. **Verify grouping**: actual patient IDs (not filename surrogates), nodule ID relationships, images/frames per patient, source-hospital identifiers and duplicate/near-duplicate scans. No patient in more than one partition.
4. **Freeze protocol before outcomes**: prespecified eligibility and primary outcome, primary metric and secondary metrics, exactly which historical model/feature extractor is transferable, class mapping, uncertainty/statistics, batch effects and handling of indeterminate cytology. Keep an untouched test set.
5. **Matched controls**: same patient splits, same available information, clear choice of baseline classical models, fixed hyperparameters/selection rule within development data only. Separate supervised clinical task from fixed-four-qubit-feature methodological audit.
6. **Evaluation**: per-patient n, classwise confusion matrix/sensitivity/specificity, balanced accuracy, F1, confidence intervals with patient grouping, Brier/log-loss where justified, missingness and failure cases. Distinguish patient-wise and image-wise denominators.
7. **Ethical release**: do not commit raw medical files, original identity data or gated credentials. Archive only approved de-identified derived summaries and scripts. Comply with dataset-specific citation and access conditions.
8. **No interpretation ahead of evidence**: until access, protocol and independent execution are complete, the result is 'not evaluated', never 'externally validated'.

**No data download, training, model selection or clinical accuracy calculation is authorized or recorded by this screening document.**

## Gate 2 conclusion and next step

**Decision for the current research:** **Closed as an evidence-based no-go for direct external six-class evaluation with currently screened datasets.** The originally planned reproduction remains scientifically meaningful as a *methodological audit*, but **patient-independent external validation is not established**. It would be a scientific error to call Gate 2 'externally validated'.

**Next pre-publication gate:** dataset rights, privacy and publication governance (including the already publicly tracked DDTI files); then figure/submission engineering. Public download access alone should not be described as an unrestricted license to redistribute source medical JPG/XML in Git history or journal supplements. No IEEE manuscript writing has begun.

**Research writing policy:** Use precise, straightforward first-person descriptions of what was actually examined or computed; avoid generic claims of 'gold standard', 'state of the art', 'first ever', 'perfect' or unobserved clinical improvements. Every result needs a source, denominator, evaluation unit, and limitation. The researcher should be able to defend every sentence at guide review.
