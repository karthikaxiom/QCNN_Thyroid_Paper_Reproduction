# Automated DDTI XML Privacy Triage — Independent GitHub Actions Run

**Recorded 28 October 2026.**  
**Repository branch:** `research/bn-recalibration-audit`  
**Code:** `05_Source_Code/auto_screen_ddti_xml.py`  
**Verified successful run:** [GitHub Actions run 40668545283](https://github.com/karthikaxiom/QCNN_Thyroid_Paper_Reproduction/actions/runs/40668545283)  
**Mode:** Non-interactive read-only structural/pattern review; no patient XML text, filenames or case IDs printed into runner logs or saved.

## Why this replaces the interactive exercise

The earlier local `review_ddti_xml_narratives_local.py` asked the researcher to manually classify 21 short clinical entries one at a time. It was burdensome and could encourage unnecessary sharing of medical text. This new scanner examines **all 390 XML source files automatically** using configured patterns, including **clinical dates/years** and **unusually long free-text**. It is appropriate as a *first-pass triage*, not as specialist privacy certification.

The GitHub job uses sparse checkout restricted to the scanner file and `02_Dataset_Raw/archive/*.xml`, with no JPG images. It executes synthetic unit examples before the real scan. Logs contain only aggregate counters. The underlying source XML was already tracked in the existing public repository; the job does not create copies in Git history, release artifacts or published datasets.

## Actual verified aggregate results

| Measure | GitHub Actions result |
| --- | ---: |
| Source XML files found | **390** |
| Successfully parsed | **390** |
| Malformed, oversized or unreadable files | **0** |
| Nonempty narrative field occurrences | **21** |
| XML records containing populated narratives | **21** |
| Narrative fields automatically flagged by at least one rule | **2** |
| Narrative fields without any configured flag | **19** |
| Narratives with potential date/year | **1** |
| Narratives flagged for length | **1** |
| Narratives containing identifier-related phrases | **0** |
| Narratives containing email patterns | **0** |
| Narratives containing phone patterns | **0** |
| Identifier-like XML element/attribute names | **0** |
| Email or phone patterns anywhere in parsed XML text/attributes | **0** |

**Pattern definitions:** The scanner looks for month/year or numeric dates, four-digit years, common identifiers (patient ID, medical record, dates of birth, phone, addresses, clinician designation), email and phone formats, and unexpectedly long narrative text (length greater than 180 characters). Rule flags are deliberately conservative. They are not a validated clinical de-identification system or a guarantee of detecting actual names, places, identifiers, contextual clues or dates in an unfamiliar format.

**Interpretation:** The automatic rules marked **two of 21 nonempty clinical narratives for manual inspection**: one because of a possible date/year, and one because of its length. The remaining **19 merely produced no match to the configured patterns**; they have not been independently certified as anonymous. Aggregate flag categories are not proof of a privacy violation and must not be described as such.

## What remains before public medical-source redistribution

- **Two flagged narrative fields:** Require confidential local review by a person authorized to handle DDTI medical data. No individual text or filename should be added to GitHub documentation, chat or commits.
- **Nineteen unflagged narrative fields:** Automated non-match does not independently certify they are safe; an authorized human spot-check or formal privacy policy may still be appropriate, particularly for unforeseen identifiers.
- **All original JPGs:** This scanner does not inspect the 480 ultrasound images, burned-in overlays or embedded metadata. The image privacy check remains open.
- **Source rights/consent:** Publicly accessible DDTI and Kaggle scientific-use statements do not themselves independently establish blanket license permission to mirror raw JPGs/XML. This remains the separate Gate 3 governance decision.
- **Clinical validity:** Nothing in this screening establishes distinct patient identities for 291 Phase 13 case IDs, pathology-confirmed outcomes, or clinical model validation.

**Decision:** The automated XML screening task is **COMPLETED AND VERIFIED**; full data privacy and redistribution clearance remain **OPEN**. No medical raw files, model checkpoints, training results or historical phase files were modified.

## Working procedure

Researchers need not answer 21 interactive `c/f/q` prompts. For any authorized local rerun, the one-command non-interactive scan is:

```powershell
& .\.venv\Scripts\python.exe .\05_Source_Code\auto_screen_ddti_xml.py
```

This command prints only aggregate screening results. The successful linked GitHub run already provides evidence; no repeated execution is needed merely to reproduce the same counters.

Any future human review is separate and confidential, with only a final aggregate disposition shared publicly. The IEEE manuscript remains deferred until scientific and publication gates are approved.
