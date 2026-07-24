from pathlib import Path
import os
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = str(PROJECT_ROOT / "02_Dataset_Raw" / "archive")

blank_count = 0
filled_count = 0

print("=" * 60)
print("VERIFYING TI-RADS VALUES")
print("=" * 60)

for file in sorted(os.listdir(DATASET_DIR)):

    if not file.endswith(".xml"):
        continue

    path = os.path.join(DATASET_DIR, file)

    tree = ET.parse(path)
    root = tree.getroot()

    tirads = root.findtext("tirads", default="").strip()

    if tirads == "":
        blank_count += 1
        print(f"{file}  --->  BLANK")

    else:
        filled_count += 1

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"XML with TI-RADS : {filled_count}")
print(f"XML without TI-RADS : {blank_count}")