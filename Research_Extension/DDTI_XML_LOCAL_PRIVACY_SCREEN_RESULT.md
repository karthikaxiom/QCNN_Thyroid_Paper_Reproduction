# DDTI annotation XML — Local structural privacy screen

**Recorded:** 28 October 2026  
**Study:** Independent QuCNet thyroid-ultrasound reproduction  
**Evidence source:** Researcher's own local PowerShell output, copied into the research conversation after executing the previously committed `05_Source_Code/audit_ddti_xml_fields_local.py`. This is **researcher-reported local execution**; no assistant/CI direct access to the underlying local medical XML files is implied.

## Command and source scope

The researcher reported a successful fast-forward synchronization of local `research/bn-recalibration-audit` to commit `b68f58b`, then executed the following in the activated virtual environment:

```powershell
& .\.venv\Scripts\python.exe .\05_Source_Code\audit_ddti_xml_fields_local.py
```

The scanner's documented scope is `02_Dataset_Raw/archive/*.xml`, producing **aggregate field/pattern counts only**. It does not transmit original case IDs, XML values or ultrasound pixels.

## Exact aggregate results reported

| Automated screen | Researcher-reported output |
| --- | ---: |
| XML files found | **390** |
| XML files successfully parsed | **390** |
| XML parsing errors | **0** |
| Oversize XML files skipped | **0** |
| `age` field occurrences | **390** |
| `sex` field occurrences | **390** |
| Identifier-like field *names* flagged by the scanner | **None** |
| Non-empty free-text clinical field occurrences | **21** |
| Email-pattern field matches | **0** |
| Phone-pattern field matches | **0** |

**Technical interpretation:** The automatic structural pass processed all 390 locally present XML files without parsing/size failures. No email/phone pattern or identifier-like XML field name was flagged by this *specific heuristic*. The 21 nonempty narrative fields remain possible locations for identifying details that the heuristic might miss, and the 390 age/sex **field occurrences** are not independent patient-count estimates. No source values were retained in this record.

**Not checked by this screen:** Actual reidentification risk from combinations of attributes; clinician/hospital names or dates in free text; identifiers overlaid/burned into any of the 480 ultrasound JPGs; image metadata; source-data consent; legal authorization to redistribute XML/images; patient identity uniqueness across recorded case IDs. No claim of HIPAA/GDPR compliance, anonymization, or medical privacy clearance is made.

## Follow-up review record and publication boundary

1. **Machine parsing step:** Complete as researcher-reported; source version described above. CI's separate synthetic fixture passed, but CI is **not** evidence of a full 390-file medical XML audit.
2. **Narrative text:** Manual local review of the **21 populated free-text field occurrences** for names, institutions, contact details, dates, IDs and unusually identifying clinical narratives is still **pending**. A field occurrence is not necessarily a unique case. Avoid copying contents or clinical narratives into GitHub, email or chat. Record only number reviewed, number requiring action, review date and reviewer role.
3. **Images:** Human privacy review for potential burned-in overlays/identifiers in the **480 source JPEGs** remains **pending** before any new raw-image reuse in figures/supplements. Do not upload contact sheets containing medical scans into GitHub.
4. **Rights:** The producer's publicly accessible research dataset and the Kaggle mirror do not, by themselves, independently establish broad onward redistribution rights. Original-image/XML republication scope remains subject to an explicit source license, written clarification, or documented institutional decision.
5. **Scientific work:** Stored predictions, numerical results, reproducibility checks and non-identifying aggregate charts remain suitable for continued *methodological* work. Keep pathology-confirmed clinical claims, fully patient-independent validation and unrestricted public release claims out of the manuscript until evidenced.

**Status of this sub-gate:** **390/390 XML structures screened; 0 parser errors; no flagged identifier field names; 21 narrative fields still need a human review.** This is a useful screening milestone, **not** a clean privacy certificate.

**Release rule:** Do not mark the overarching Gate 3 raw medical data permission/privacy decision as cleared on the strength of this automated scan. The original Git history was not altered or purged.
