# Source-paper identity and dataset reuse rights — verification checkpoint

**Date:** 2026-10-28. **Status:** bibliography and full-paper methods verified in separate artifacts; raw-image/XML redistribution rights are still unconfirmed. This is research metadata, not manuscript copy.

## Original QuCNet source — verified bibliography

- Title: *QuCNet: Quantum-Inspired Convolutional Neural Networks for Optimized Thyroid Nodule Classification*.
- Authors: Gunti Swathi, Ali Altalbe, and R. Prasanna Kumar.
- Venue: IEEE Access, **volume 12 (2024), pages 27829–27842**.
- DOI: **10.1109/ACCESS.2024.3367806**.
- Independent bibliographic record: https://dblp.dagstuhl.de/rec/journals/access/SwathiAK24.html
- Institutional publication record: https://psau.sa.elsevierpure.com/en/publications/qucnet-quantum-inspired-convolutional-neural-networks-for-optimiz/

The institutional abstract reports **97.63%** tumor-classification accuracy on **127 test images** and **93.87%** suspicious-level accuracy on **98 test images**. Those are **authors' reported results**, not independently replicated in our experiment. Our historical 70-image six-class evaluation and strict Phase13 case-separated 44-case evaluation are **different datasets/partitions/protocols**; do not use the source-paper figures as directly comparable estimates.

**Full-paper methods independently inspected:** The available 14-page `01_Paper/Main_Paper.pdf` was reviewed against stored code. Details, page anchors, reported test denominators and unrecoverable exact hyperparameters are documented in `Research_Extension/ORIGINAL_PAPER_FULLTEXT_FIDELITY_AUDIT.md`; the remaining uncertainty is not a lack of reading, but incomplete or ambiguous reproduction parameters in the source and differences between protocols.

## DDTI source and public copies

Original project website: https://cimalab.unal.edu.co/software/detail/2 (University of Colombia CIM@LAB TDID/thyroid image database listing).

A mirror/index description is at https://www.kaggle.com/dasmehdixtr/ddti-thyroid-ultrasound-images/metadata. It describes DDTI as a freely accessible scientific database and records its license as **"Other (specified in description)"** rather than an unambiguous permissive redistribution license. It also notes expert annotation and patient information in XML. The index page's description of historical holdings and other derivative mirrors' 637-image inventories should **not** be conflated with the actual 480-image file collection verified in this repository.

**Research-data use and permission are distinct from unrestricted republication.** The fact that data can be downloaded for academic work does not, by itself, establish permission to redistribute unaltered clinical ultrasound images or XML files, or to place them in a public repository/published figures. The present branch has historically stored such files; this audit does not delete or move them because preserving original research artifacts is essential and further user approval is needed.

**Action before publication/public release:** Verify original DDTI image/XML usage and redistribution terms from the original publisher/maintainer or an explicit current license, review the repository's existing raw images and XML for identifying material, and decide whether a permission notice, restrictive distribution policy, or separate private data acquisition process is warranted. Preserve Git history and user-approved provenance records. Do not include source ultrasound contact sheets in the paper until these questions are settled.

## Labels and clinical wording

Our machine-readable original source fields contain six TI-RADS risk categories. The binary low-versus-high suspicion task is derived from these categories; the project has **not** independently authenticated pathology-confirmed benign/malignant outcomes. Do not paraphrase the original QuCNet paper's claimed tumor-classification endpoint as though this reproduction verified histopathology labels.

## No further tuning from these external results

These third-party sources are used for scientific context, identity and permissions only. They were not used to select classification models or train/test partitions. The previously inspected held-out splits remain historical evidence, not a new independent holdout.

**Status:** Citation and full-text method comparisons documented; raw-image/XML rights and potentially identifying overlays remain unresolved for public-release decisions. A non-destructive permission/notice checklist is in `Research_Extension/DATA_RELEASE_PERMISSION_DECISION.md`. The selected scientific figures use derived statistical evidence and do not require raw source images.


## Gate 3 follow-up (28 October 2026)

The [source-data rights and privacy gate](DDTI_DATA_RIGHTS_AND_PRIVACY_GATE.md) verifies the primary university listing, Kaggle ‘Other (specified in description)’ license category, dataset-original Pedraza et al. citation (DOI 10.1117/12.2073532), and 480 tracked JPG + 390 tracked XML files. Four sampled original XMLs included age and sex field names; **this does not establish that any person is identified**, nor clear the other 386 XMLs or the raw image overlays. The local aggregate-only XML scanner was synthetic-tested on GitHub Actions; full-source scan remains open. Source research access is supported while blanket GitHub medical-data redistribution permission is still unconfirmed.

## Local XML screen status — 28 October 2026

The researcher reports running the committed non-disclosing structural scanner locally after fast-forwarding the research branch. All **390/390 DDTI XML files parsed**, with zero parsing or oversize errors; `age` and `sex` occurred 390 times each; zero identifier-like field names and zero email/phone regex matches were flagged; and **21 populated narrative field occurrences** remain for manual local inspection. See [DDTI_XML_LOCAL_PRIVACY_SCREEN_RESULT.md](DDTI_XML_LOCAL_PRIVACY_SCREEN_RESULT.md) for the precise report and limits. This is **researcher-reported execution**, distinct from the synthetic GitHub Actions test. It is not confirmation of de-identification, consent, or an unrestricted right to mirror medical JPG/XML. No raw medical content or local field values were uploaded as part of this update.
