# Gate 3 — DDTI Data Rights, Privacy, and Publication Governance

**Review date:** 28 October 2026  
**Project:** Independent QuCNet reproduction and methodological audit  
**Stage:** Pre-publication research governance, **not manuscript text or legal advice**  
**Scientific status:** Research use and derived-statistics analysis can continue. **The rights to redistribute original JPEG/XML files remain unverified, and full patient-data/image privacy review is not complete.**

**Local XML screen update (researcher report, 28 October 2026):** The researcher ran the committed scanner locally after synchronizing to `b68f58b`. It parsed **390/390 XML files**, with **0 parser errors**, **0 oversize files**, age and sex each present in **390 field occurrences**, **0 identifier-like XML field names flagged**, **21 nonempty narrative fields**, and **0 email/phone pattern matches**. Exact aggregate evidence and limits are preserved in [the local XML screening record](DDTI_XML_LOCAL_PRIVACY_SCREEN_RESULT.md). **This was not run against the actual XMLs in CI**, nor does it establish de-identification, image privacy or redistribution rights.

## 1. What I verified from primary and secondary sources

1. **Original producer:** Universidad Nacional de Colombia's CIM@LAB lists [DDTI as an open-access thyroid ultrasound image project](https://cimalab.unal.edu.co/projects/detail/20/) and lists the [Thyroid Digital Image Database](https://cimalab.unal.edu.co/software/detail/2) with a download route. The site's public listing, as retrieved for this review, did **not** supply an explicit Creative Commons, public-domain, or comparable unrestricted source-image/XML redistribution grant.
2. **Kaggle mirror:** [DDTI: Thyroid Ultrasound Images](https://www.kaggle.com/datasets/dasmehdixtr/ddti-thyroid-ultrasound-images/metadata), publisher `dasmehdixtr`, is publicly accessible and describes an **open access resource for the scientific community**. The displayed license category is **"Other (specified in description)"**, *not* CC0, CC BY or another explicit blanket redistribution permission. It also expressly states that an XML file may contain expert annotations and patient information.
3. **Kaggle license semantics:** [Kaggle's dataset metadata documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets_metadata.md) distinguishes `other` from specific CC0, CC-BY, CC-BY-NC and database licenses. [Kaggle dataset publishing documentation](https://www.kaggle.com/docs/datasets) advises providers selecting Other to specify the particular license terms in their description. I did not locate a separate written grant permitting arbitrary redistribution of DDTI JPG/XML in an independent public Git repository.
4. **Original academic source:** Pedraza, L., Vargas, C., Narváez, F., Durán, O., Muñoz, E., and Romero, E., *An open access thyroid ultrasound image database*, Proc. SPIE **9287**, 92870W (2015), DOI [10.1117/12.2073532](https://doi.org/10.1117/12.2073532). Cite the original authors **as well as** the exact Kaggle source and access/version; the name of the mirror uploader does not replace dataset authorship.
5. **Publication expectations:** [IEEE Access preparation guidance](https://ieeeaccess.ieee.org/authors/preparing-your-article/) requires accurate methods, truthful data and references, and disclosure of AI-generated **article content** where its stated policy applies. [IEEE publication ethics](https://journals.ieeeauthorcenter.ieee.org/become-an-ieee-journal-author/publishing-ethics/ethical-requirements/) requires authors to report results and images accurately. This pre-publication audit does not grant third-party data rights or decide the University's ethics policy.

**Conclusion of source review:** Public scientific access is real and supports the already-conducted academic analysis. **Open-to-download ≠ explicit permission to republish unmodified or derived medical images/XML everywhere.** This is an evidence distinction, not an allegation that the original download was improper.

## 2. What is actually in this research repository

I checked the complete tracked branch file tree on 28 October 2026. The currently public repository contains:

| Material | Tracked amount | Release assessment |
| --- | ---: | --- |
| `02_Dataset_Raw/archive/*.jpg` | **480** | Original ultrasound images publicly tracked; copyright/source conditions and identifying overlays not fully reviewed |
| `02_Dataset_Raw/archive/*.xml` | **390** | Source expert annotations and per-case clinical metadata publicly tracked |
| Historical experimental code, results and figure assets | Preserved in Git | Continue citing original authors and displaying source-aware derived statistics, subject to normal ethics and publication decisions |
| `LICENSE.txt` | MIT license for the project's software | **Not a license from the DDTI rightsholders** for the JPG/XML originals |

The raw data are **already in reachable Git history**; a `.gitignore` rule cannot remove previously committed objects. Making this branch private, removing files from its latest tree, or rewriting history would each have different impacts and may require institutional guidance, user authorization and backups. **No deletion, force-push, history rewrite or visibility change was performed.**

The historical reference article `01_Paper/Main_Paper.pdf` has its own copyright/licensing conditions, separate again from raw DDTI and software. Do not silently bundle original article pages or raw ultrasound figures into new journal supplementary material without checking scope.

## 3. Focused clinical-annotation privacy sample

I inspected the **XML field names only** from four publicly tracked source records (selected examples in the repository). All four contained `age` and `sex` fields and qualitative ultrasound/diagnostic annotation fields, including free-text-style report fields. This substantiates the **presence of demographic and health-related metadata**, not a finding of direct names, contact details or reidentification risk. **The other 386 XML files have not been individually checked for identifying content in this remote inspection.**

Risk boundaries:
- Age and sex alone do not necessarily identify someone, but their combinations with clinical records or date/context require care.
- Text in radiology reports, source-file paths, annotations and image overlays could contain additional information; **absence of direct-identification tags in a few samples does not clear the archive**.
- The public images have **not** undergone a documented complete image-by-image human privacy inspection in this gate. OCR is not a substitute for clinical/privacy review.
- I created `05_Source_Code/audit_ddti_xml_fields_local.py` for a local, **non-disclosing aggregate screen** of all XMLs. It does not print original field values, person identifiers, individual case IDs or images and has a synthetic-data `--self-test`. Its real-data structural run has now been **completed locally and reported by the researcher**: all 390 XMLs parsed without errors, with 21 populated narrative fields requiring focused local review. See the [aggregate result](DDTI_XML_LOCAL_PRIVACY_SCREEN_RESULT.md). This does not independently check source images or actual identity risk. Even a clean scan is not a declaration of anonymity or consent.

## 4. Scientific publication and release decisions by content type

| Output | Decision now | Conditions |
| --- | --- | --- |
| Repository source code written for the study | **May remain available** | Maintain correct authorship, dependencies, relevant third-party code licenses and scientific claims |
| Aggregated accuracy, confusion, reliability and repeated-CV statistical results | **May be used for scientific reporting** | Correct source attribution, no raw identifying detail, unit/denominator and limitations |
| Case-level prediction CSV and ROI summary metadata | **Continue using for verification** | Avoid additional patient-sensitive information; check data-sharing agreement if data become linkable |
| Original DDTI JPG images and all original XML files | **Redistribution terms unverified; no new mirroring approved** | Confirm source owner's terms or obtain written scope-specific permission. Conduct privacy checks and choose a controlled release approach |
| Cropped/modified source-ultrasound examples in article figures | **Hold until reviewed** | Confirm figure reuse rights, check burned-in identifiers/context, attribute source and any modifications |
| Fully synthetic diagrams and aggregate-statistic plots | **Available for manuscript preparation** | Ensure faithful numbers and no fabricated clinical data |
| QuCNet source article PDF | **Separate rights review** | Provide DOI/link and lawful citation; no automatic sublicensing under our software MIT license |

**Preferred reproducibility architecture, if raw-file republication stays ambiguous:** publish code, derived aggregate metrics, and a frozen dataset acquisition-and-preprocessing protocol; instruct authorized readers to obtain the original source from the university/Kaggle. For an already-public historical raw-data Git history, obtain the researcher's approval and appropriate institutional advice before any visibility or history decision. Never claim that a newly added notice retroactively cures existing source rights.

## 5. Decision branches to complete before submission

**A — Explicit image/XML redistribution right found.** Keep dated written evidence, the rightsholder, version, allowed media (GitHub / paper / supplements), attribution and license. Confirm patient confidentiality separately. Only then mark raw-data redistribution permission as verified.

**B — Research use confirmed; no clear republication right.** Continue reproducibility/statistical reporting, avoid adding raw-source medical images to articles/supplements, and decide on an owner-approved separate authorized acquisition route. Determine institutional treatment of already-public Git history without destructive unilateral changes.

**C — Potential identifiers detected or permission explicitly restricted.** Do not publicize examples further; seek university ethics/data-governance advice and a controlled mitigation/backup plan. Record facts without disclosing a patient's identity in Git commits, issues, chats or reports.

**D — No response or contact impossible.** Preserve "redistribution not established" as the honest status. Lack of reply is **not** permission. The scientific contribution can still be reported with a transparent provenance/data-availability statement and approved derived evidence.

## 6. Specific steps left open

1. **Local XML screen: COMPLETE (researcher-reported).** All 390 XMLs parsed with no parser errors; no identifier-like field names or email/phone patterns flagged. The 21 nonempty free-text field occurrences require a separate human local review. The CI runs only a synthetic fixture and cannot certify the actual medical files.
2. **Human image privacy pass** over source frames that would be redistributed or used in figures, including any burned-in identifiers, accession numbers, dates or recognizable annotations. Mark reviewed counts and reviewer role; do not copy screenshots/contact sheets into public Git.
3. **Record legal/ethical basis** of academic use versus raw-file redistribution. The prepared [provider clarification request](DDTI_PERMISSION_REQUEST_TEMPLATE.md) can be used if no license text is identified. Keep replies private unless explicitly shareable.
4. **Record data provenance** in the publication package: original Pedraza et al. citation and DOI, primary source project, exact Kaggle mirror, dataset version/download date if known, original archive size and checksums already preserved, and deviations from source paper.
5. **Review contributor/AI-use disclosures** at manuscript stage against actual support used and publisher/institutional policies. The human researcher must inspect, understand, independently defend, and take responsibility for every method and result. Natural writing does not replace correct disclosure; do not misrepresent authorship or hide material use when disclosure is required.
6. **Do not merge/force-push/main rewrite** as a substitute for scientific or rights clearance; such changes need a separately approved governance decision.

## 7. Status and handoff

**Gate 3 — Evidence review completed; unconditional raw-file redistribution clearance NOT obtained.** Scientifically, this does not stop the existing reproduction audit or publication planning. It does stop claiming "unrestricted image/XML republication is licensed" until substantiated.

**Next research milestone:** Run the local-only XML field screen and review the narrow figure set intended for publication; secure dataset-steward clarification or documented institutional decision if raw originals will remain publicly downloadable. Then proceed to final figure layout/submission engineering, leaving IEEE manuscript writing for last.

This document does **not** assert a new clinical result, a finding of identifiable patient information, or permission that the source has not explicitly provided.
