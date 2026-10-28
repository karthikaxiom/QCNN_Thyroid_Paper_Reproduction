# DDTI publication and repository-release decision protocol

**Status after Gate 3 review (28 October 2026): scientific research access confirmed; unrestricted raw JPG/XML redistribution permission NOT independently established.** Full primary-source analysis, actual public-archive counts, sampled XML field names and publication decision tree are in [DDTI_DATA_RIGHTS_AND_PRIVACY_GATE.md](DDTI_DATA_RIGHTS_AND_PRIVACY_GATE.md). An offline XML field scanner exists at `05_Source_Code/audit_ddti_xml_fields_local.py`; only its **synthetic** self-test has been executed in CI. A real whole-archive structural scan and human source-image privacy review remain open. This is research-governance material, not a paper draft or legal opinion.

## Why this decision is needed

The original thyroid-ultrasound dataset is attributed to CIM@LAB / DDTI. Its availability for academic download does **not** by itself grant permission to republish raw ultrasound images, patient-annotation XML, or all derivative crops in an independently public GitHub repository or an IEEE paper. A third-party dataset index describes its license as "Other (specified in description)." The original project's express terms and any relevant institutional consent restrictions require direct verification.

This repository's historical Git objects contain raw images; the .gitignore rule does **not** retroactively make already-tracked files private. Merely deleting files in the latest commit would not remove them from accessible prior commits. No automated deletion, history rewrite, visibility change or public release should occur without explicit owner authorization and an appropriate backup/recovery plan.

## Permission questions to send to the source maintainer

1. May this dataset be used for independent academic reproduction and methodological analysis, with appropriate original-source attribution?
2. May selected raw ultrasound JPEGs, their segmentation/annotation regions, and any XML metadata be publicly redistributed via GitHub, including preserved historical commits? If yes, under which precise license and attribution requirements?
3. May cropped, resized, de-identified examples be reproduced in an open-access conference/journal paper or its supplementary files?
4. Do original scan images, annotations, XML or burned-in text contain patient identifiers or restrictions from patient consent that require removing/obscuring them?
5. Is there a preferred data-access URL, citation, notice or access-upon-request process that allows the repository to remain scientifically reproducible without mirroring the underlying dataset?

Keep any written permission evidence and record its scope, date, contact, and approved media. Do not share private correspondence or identities publicly without consent.

## Proposed decision tree

- **Explicit redistributable license confirmed:** Add a narrowly scoped dataset NOTICE with source/version, link, credited authors, allowed purposes, and relevant license text or official URL. Verify de-identification separately; permission to redistribute does not remove privacy duties.
- **Research access permitted but public redistribution unclear or prohibited:** Keep experiments/results/scripts accessible but do not publish source images or XML anew. Discuss repository visibility and a history-remediation strategy with the repository owner before taking action; historical Git blobs require deliberate handling.
- **No response:** Record “rights unverified”; avoid raw-image figures and supplementary redistribution, and arrange a clean reproducibility protocol where authorized researchers obtain the source independently. Do not imply lack of response is consent.
- **Identifying material discovered:** Stop plans for public promotion of such files and seek appropriate institutional guidance. Do not use publication-only anonymization as a substitute for resolving previously public Git history.

## Proposed article data-availability language — conditional only

> The study analyzes a locally acquired copy of the public-facing DDTI thyroid ultrasound archive. Source image and annotation redistribution is subject to the original dataset provider's terms and applicable privacy restrictions. Reproducible preprocessing code, case-level experimental protocols, evaluation summaries, and derived metrics are maintained separately from any redistribution of source medical images. No pathology-confirmed binary ground truth was independently established in this reproduction.

The text above is **not yet approved for publication**. Revise after source permission is confirmed, including whether any code/results can remain in a public repository.

## Repository hygiene (no destructive actions)

- Preserve `README.md`, `Project_Log.txt`, all historical experiments and original `Main_Paper.pdf`.
- Never run `git filter-repo`, `git push --force`, visibility changes, or bulk deletes solely to make a clean paper; these have downstream impacts.
- Continue excluding local raw-ultrasound review-sheet composites and avoid staging `07_Results/exp_neardup_001/review_local/`.
- This governance gate is separate from experimentally completed research; the open rights decision should be disclosed as a restriction on public release rather than hidden.
