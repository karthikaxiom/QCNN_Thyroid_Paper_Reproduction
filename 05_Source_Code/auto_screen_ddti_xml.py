"""Automated, non-disclosing screening of DDTI clinical annotation XML.

    python 05_Source_Code/auto_screen_ddti_xml.py
    python 05_Source_Code/auto_screen_ddti_xml.py --self-test

Reads XML from the existing local repository, writes no files, and prints
aggregate counts only: NEVER XML values, case IDs, filenames, or private text.
The rule-based screen is a TRIAGE, not a declaration of anonymization, legal
permission, consent, or clinical privacy clearance.

It deliberately flags possible examination dates even when no name is present.
No optical scan of image overlays is performed.
"""
from __future__ import annotations

import argparse
import collections
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "02_Dataset_Raw" / "archive"
FIELDS = frozenset({"reportbacaf", "reporteco", "history", "comments", "notes", "report", "observations"})
MAX_BYTES = 2 * 1024 * 1024

EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
PHONE = re.compile(r"(?<!\w)(?:\+\d{1,3}[-. ]?)?(?:\(\d{2,4}\)[-. ]?)?\d{3}[-. ]\d{3}[-. ]\d{3,4}(?!\w)")
MONTH = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
    r"jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\.?\s*(?:\d{1,2}(?:st|nd|rd|th)?\s*,?\s*)?"
    r"(?:19|20)\d{2}\b", re.I,
)
YEAR = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
NUMERIC_DATE = re.compile(r"\b(?:0?[1-9]|[12]\d|3[01])[-/.](?:0?[1-9]|1[0-2])[-/.](?:19|20)?\d{2}\b|"
                          r"\b(?:19|20)\d{2}[-/.](?:0?[1-9]|1[0-2])[-/.](?:0?[1-9]|[12]\d|3[01])\b")
IDENTIFIER = re.compile(
    r"\b(?:patient\s*(?:name|id|number)|full\s*name|date\s*of\s*birth|"
    r"born\s+on|dob|m\.?r\.?n\.?|medical\s*record|passport|"
    r"hospital\s*(?:id|number)|national\s*id|phone|telephone|"
    r"address|e-?mail|accession\s*(?:no|number)|"
    r"dr\.?\s+[a-z]{2,})\b", re.I,
)
SENSITIVE_FIELD = re.compile(
    r"(?:patient.?name|first.?name|last.?name|full.?name|surname|"
    r"date.?of.?birth|dob|patient.?id|national.?id|"
    r"medical.?record|passport|contact|address|phone|telephone|email)",
    re.I,
)
DEMOGRAPHIC_FIELDS = {"age", "sex", "gender"}


def short_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].casefold().strip()


def signal_categories(text: str) -> set[str]:
    signals: set[str] = set()
    if EMAIL.search(text):
        signals.add("email_pattern")
    if PHONE.search(text):
        signals.add("phone_pattern")
    if MONTH.search(text) or YEAR.search(text) or NUMERIC_DATE.search(text):
        signals.add("potential_date_or_year")
    if IDENTIFIER.search(text):
        signals.add("identifier_phrase")
    if len(text) > 180:
        signals.add("length_review")
    return signals


def scan(xml_bytes: bytes, tally: collections.Counter[str]) -> None:
    if len(xml_bytes) > MAX_BYTES:
        tally["oversize_xml"] += 1
        return
    try:
        root = ET.fromstring(xml_bytes)
    except (ET.ParseError, UnicodeError, ValueError):
        tally["xml_parse_error"] += 1
        return
    tally["parsed_xml"] += 1
    had_narrative = False
    for el in root.iter():
        tag = short_tag(el.tag)
        if tag in DEMOGRAPHIC_FIELDS:
            tally["demographic_" + tag + "_fields"] += 1
        if SENSITIVE_FIELD.search(tag):
            tally["identifier_like_field_names"] += 1
        for attr_name, attr_value in el.attrib.items():
            if SENSITIVE_FIELD.search(short_tag(attr_name)):
                tally["identifier_like_field_names"] += 1
            if EMAIL.search(attr_value):
                tally["email_patterns_anywhere"] += 1
            if PHONE.search(attr_value):
                tally["phone_patterns_anywhere"] += 1
        value = (el.text or "").strip()
        if value and EMAIL.search(value):
            tally["email_patterns_anywhere"] += 1
        if value and PHONE.search(value):
            tally["phone_patterns_anywhere"] += 1
        if tag in FIELDS and value:
            had_narrative = True
            tally["narrative_fields"] += 1
            signals = signal_categories(value)
            if signals:
                tally["narrative_fields_rule_flagged"] += 1
                for category in signals:
                    tally["signal_" + category] += 1
            else:
                tally["narrative_fields_no_pattern_flag"] += 1
    if had_narrative:
        tally["xml_containing_narrative"] += 1


def self_test() -> None:
    assert signal_categories("Nodule 2") == set()
    assert signal_categories("First FNA on feb. 2014 Bethesda 3") == {"potential_date_or_year"}
    assert "potential_date_or_year" in signal_categories("06/02/2014 visit")
    assert "identifier_phrase" in signal_categories("Patient ID recorded")
    assert "email_pattern" in signal_categories("test@example.org")
    tally: collections.Counter[str] = collections.Counter()
    scan(b"<case><age>42</age><reporteco>Nodule 2</reporteco>"
         b"<reportbacaf>FNA in feb. 2014</reportbacaf></case>", tally)
    assert tally["parsed_xml"] == 1
    assert tally["narrative_fields"] == 2
    assert tally["narrative_fields_rule_flagged"] == 1
    assert tally["narrative_fields_no_pattern_flag"] == 1
    assert tally["demographic_age_fields"] == 1
    print("PASS: synthetic automated XML privacy triage, no medical data accessed.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not DATA_DIR.is_dir():
        print("ERROR: source XML directory unavailable; nothing examined.", file=sys.stderr)
        return 2
    paths = sorted(DATA_DIR.glob("*.xml"))
    if not paths:
        print("ERROR: no source XML records found.", file=sys.stderr)
        return 2

    tally: collections.Counter[str] = collections.Counter()
    for path in paths:
        try:
            scan(path.read_bytes(), tally)
        except OSError:
            tally["xml_read_error"] += 1
    print("AUTOMATED DDTI XML PRIVACY TRIAGE — AGGREGATES ONLY")
    print(f"XML files found: {len(paths)}")
    print(f"Parsed: {tally['parsed_xml']}; malformed: {tally['xml_parse_error']}; "
          f"oversize: {tally['oversize_xml']}; unreadable: {tally['xml_read_error']}")
    print(f"Clinical narrative field occurrences: {tally['narrative_fields']}")
    print(f"XML records containing narrative fields: {tally['xml_containing_narrative']}")
    print(f"Narrative fields automatically flagged for review: {tally['narrative_fields_rule_flagged']}")
    print(f"Narrative fields with no configured pattern match: {tally['narrative_fields_no_pattern_flag']}")
    print(f"Potential dates/years in narratives: {tally['signal_potential_date_or_year']}")
    print(f"Identifier-related phrases in narratives: {tally['signal_identifier_phrase']}")
    print(f"Email-pattern narrative fields: {tally['signal_email_pattern']}")
    print(f"Phone-pattern narrative fields: {tally['signal_phone_pattern']}")
    print(f"Long narrative fields flagged: {tally['signal_length_review']}")
    print(f"Identifier-like XML field name occurrences: {tally['identifier_like_field_names']}")
    print(f"Email-pattern occurrences in XML text/attrs: {tally['email_patterns_anywhere']}")
    print(f"Phone-pattern occurrences in XML text/attrs: {tally['phone_patterns_anywhere']}")
    print("No source values, paths, or case identifiers were printed or saved.")
    print("NO-PATTERN-FLAG does NOT mean verified de-identification.")
    print("JPG overlays, linkage/reidentification, patient consent, and")
    print("raw-data redistribution permissions remain outside this scan.")
    if tally["parsed_xml"] != len(paths):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
