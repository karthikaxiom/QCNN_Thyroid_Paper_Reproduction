"""Read-only, local-only triage of populated DDTI clinical-text XML elements.

This tool intentionally DISPLAYS source narrative text ONLY on your own local
computer to permit review. Never paste its running output into chats, emails,
issues, Git commits or a public report. Only the final aggregate tally may be
shared. This is a preliminary human privacy screen, NOT a legal/clinical
de-identification certificate or authorization to republish source files.

Run on an authorized local copy:
    python 05_Source_Code/review_ddti_xml_narratives_local.py

Synthetic self-test (accesses no DDTI):
    python 05_Source_Code/review_ddti_xml_narratives_local.py --self-test

No network requests, data file writes, logs, or file modifications are made.
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "02_Dataset_Raw" / "archive"
FIELDS = frozenset({"reportbacaf", "reporteco", "history", "comments", "notes", "report", "observations"})
MAX_BYTES = 2 * 1024 * 1024


def tagname(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].casefold().strip()


def narratives(path: Path) -> list[tuple[str, str]]:
    raw = path.read_bytes()
    if len(raw) > MAX_BYTES:
        raise ValueError("XML file exceeds permitted screening size")
    root = ET.fromstring(raw)
    return [(tagname(el.tag), el.text.strip()) for el in root.iter()
            if tagname(el.tag) in FIELDS and el.text and el.text.strip()]


def self_test() -> None:
    # The self-test does not open the actual dataset or create any file.
    xml = "<case><reporteco>Fictional example</reporteco><reportbacaf> </reportbacaf><age>50</age></case>"
    root = ET.fromstring(xml)
    found = [(tagname(el.tag), el.text.strip()) for el in root.iter()
             if tagname(el.tag) in FIELDS and el.text and el.text.strip()]
    assert found == [("reporteco", "Fictional example")]
    print("PASS: synthetic narrative field detector; no medical source data read.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not SOURCE.is_dir():
        print("No local DDTI archive directory found.", file=sys.stderr)
        return 2
    paths = sorted(SOURCE.glob("*.xml"))
    if not paths:
        print("No source XMLs found.", file=sys.stderr)
        return 2
    try:
        records = [(path, field, value) for path in paths
                   for field, value in narratives(path)]
    except (OSError, ValueError, ET.ParseError) as exc:
        print(f"Cannot complete local review: {type(exc).__name__}.", file=sys.stderr)
        return 2
    print("LOCAL-ONLY CLINICAL TEXT REVIEW")
    print(f"XMLs inspected for narrative fields: {len(paths)}")
    print(f"Nonempty narrative field occurrences: {len(records)}")
    print("Sensitive source text will appear BELOW on this computer.")
    print("Do not share screenshots or paste the individual review output.")
    print("Mark 'c' only for no obvious identifier in this specific field;")
    print("'f' flags for qualified privacy review; 'q' stops. This is not")
    print("a complete privacy or publication-rights certification.")
    inspected = clear = flagged = 0
    for n, (path, field, value) in enumerate(records, 1):
        print(f"\n--- Field {n} of {len(records)} | local file: {path.name} | {field} ---")
        print(value)
        while True:
            try:
                response = input("[c] no obvious identifier, [f] flag, [q] stop: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                response = "q"
                print()
            if response in ("c", "f", "q"):
                break
            print("Please enter c, f or q.")
        if response == "q":
            break
        inspected += 1
        if response == "c":
            clear += 1
        else:
            flagged += 1
    print("\nSAFE AGGREGATE SUMMARY (share this section only)")
    print(f"XML files structurally checked for free-text fields: {len(paths)}")
    print(f"Nonempty narrative fields found: {len(records)}")
    print(f"Narrative fields triaged locally: {inspected}")
    print(f"No obvious identifiers on preliminary visual triage: {clear}")
    print(f"Flagged for qualified privacy review: {flagged}")
    print(f"Not reviewed: {len(records) - inspected}")
    print("This screen does not examine the image JPGs or grant raw-data")
    print("redistribution rights. Review by qualified personnel may still be needed.")
    return 0 if inspected == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
