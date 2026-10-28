"""Local-only, non-disclosing structural/privacy-screen for DDTI annotation XML.

Run on an authorized local copy:
    python 05_Source_Code/audit_ddti_xml_fields_local.py

This tool reads all XML in 02_Dataset_Raw/archive without making network
requests or writing to disk. It prints ONLY aggregate tag/field counts.
It never prints original XML values, case numbers, paths, or image pixels.
A successful scan is NOT de-identification or an authorization to redistribute.
Use --self-test to check the parser without accessing any medical data.
"""
from __future__ import annotations

import argparse
import collections
import io
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "02_Dataset_Raw" / "archive"
ATTR_PAT = re.compile(
    r"(?:patient.?name|first.?name|last.?name|full.?name|surname|"
    r"birth|date.?of.?birth|dob|national.?id|patient.?id|"
    r"document.?id|medical.?record|address|telephone|phone|"
    r"e.?mail|email|passport|identification)",
    re.I,
)
EMAIL_PAT = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
# Stronger rather than broad numeric heuristics; findings must be manually checked.
PHONE_HINT = re.compile(r"(?:\+\d{1,3}[-. ]?)?(?:\(\d{2,4}\)[-. ]?)?\d{3}[-. ]\d{3}[-. ]\d{3,4}")
FREE_TEXT = {"reportbacaf", "reporteco", "history", "comments", "notes", "report", "observations"}
DEMOGRAPHIC = {"age", "sex", "gender"}
MAX_FILE_SIZE = 2 * 1024 * 1024


def tagname(value: str) -> str:
    return value.rsplit("}", 1)[-1].casefold().strip()


def inspect_bytes(data: bytes, aggregate: dict) -> None:
    if len(data) > MAX_FILE_SIZE:
        aggregate["oversize"] += 1
        return
    try:
        root = ET.fromstring(data)
    except (ET.ParseError, UnicodeError, ValueError):
        aggregate["parse_errors"] += 1
        return
    aggregate["parsed"] += 1
    sensitive_seen = set()
    tag_counts = aggregate["tag_count"]
    for el in root.iter():
        key = tagname(el.tag)
        tag_counts[key] += 1
        if key in DEMOGRAPHIC:
            aggregate["demographic_counts"][key] += 1
        if key in FREE_TEXT and el.text and el.text.strip():
            aggregate["nonempty_narrative_fields"] += 1
        if ATTR_PAT.search(key):
            sensitive_seen.add(key)
        for name, value in el.attrib.items():
            aname = tagname(name)
            if ATTR_PAT.search(aname):
                sensitive_seen.add("attribute:" + aname)
            if EMAIL_PAT.search(value):
                aggregate["possible_email_fields"] += 1
            if PHONE_HINT.search(value):
                aggregate["possible_phone_fields"] += 1
        if el.text:
            if EMAIL_PAT.search(el.text):
                aggregate["possible_email_fields"] += 1
            if PHONE_HINT.search(el.text):
                aggregate["possible_phone_fields"] += 1
    aggregate["sensitive_field_names"].update(sensitive_seen)


def fresh_aggregate() -> dict:
    return {
        "parsed": 0,
        "parse_errors": 0,
        "oversize": 0,
        "tag_count": collections.Counter(),
        "demographic_counts": collections.Counter(),
        "nonempty_narrative_fields": 0,
        "possible_email_fields": 0,
        "possible_phone_fields": 0,
        "sensitive_field_names": set(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true",
                        help="Run synthetic fixture only, no medical files")
    args = parser.parse_args()
    aggregate = fresh_aggregate()
    if args.self_test:
        # Fictional fixture, not sourced from DDTI.
        fake = b"<case><age>50</age><sex>F</sex><patient_name>Example</patient_name><reporteco>Test note</reporteco></case>"
        inspect_bytes(fake, aggregate)
        assert aggregate["parsed"] == 1
        assert aggregate["demographic_counts"] == {"age": 1, "sex": 1}
        assert "patient_name" in aggregate["sensitive_field_names"]
        assert aggregate["nonempty_narrative_fields"] == 1
        print("PASS: synthetic XML scanner test; no source medical data accessed.")
        return 0

    if not SOURCE.is_dir():
        print("ERROR: no local DDTI XML directory present. No scan performed.",
              file=sys.stderr)
        return 2
    paths = sorted(SOURCE.glob("*.xml"))
    if not paths:
        print("ERROR: no local DDTI XML files present. No scan performed.",
              file=sys.stderr)
        return 2
    for path in paths:
        try:
            inspect_bytes(path.read_bytes(), aggregate)
        except OSError:
            aggregate["parse_errors"] += 1
    print("DDTI XML structural field screening (non-disclosing aggregate only)")
    print(f"Files found: {len(paths)}")
    print(f"Parsed: {aggregate['parsed']}; parse errors: {aggregate['parse_errors']};"
          f" oversized: {aggregate['oversize']}")
    print("Demographic field occurrences: " +
          ", ".join(f"{k}={v}" for k, v in
                    sorted(aggregate["demographic_counts"].items())))
    print("Potential identifier field *names* only: " +
          (", ".join(sorted(aggregate["sensitive_field_names"])) or "none flagged"))
    print(f"Non-empty free-text clinical fields: {aggregate['nonempty_narrative_fields']}")
    print(f"Possible email-pattern field matches: {aggregate['possible_email_fields']}")
    print(f"Possible phone-pattern field matches: {aggregate['possible_phone_fields']}")
    print("All counts are preliminary screening signals, not verified identifiers.")
    print("IMPORTANT: image overlays, patient consent and medical-data release rights")
    print("are NOT checked. Any publishable examples need human privacy review.")
    return 1 if aggregate["parse_errors"] or aggregate["oversize"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
