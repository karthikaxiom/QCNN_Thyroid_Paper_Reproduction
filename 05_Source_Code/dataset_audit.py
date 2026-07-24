from pathlib import Path
import os
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = str(PROJECT_ROOT / "02_Dataset_Raw" / "archive")

all_files = os.listdir(DATASET_DIR)

xml_files = sorted([f for f in all_files if f.endswith(".xml")])
jpg_files = sorted([f for f in all_files if f.lower().endswith(".jpg")])

print("=" * 60)
print("DDTI DATASET AUDIT")
print("=" * 60)

print(f"XML Files : {len(xml_files)}")
print(f"JPG Images: {len(jpg_files)}")

# ---------------------------------------------------
# Count images belonging to each case
# ---------------------------------------------------

case_images = defaultdict(list)

for image in jpg_files:
    case_id = image.split("_")[0]
    case_images[case_id].append(image)

print("\nCases with multiple images:\n")

multiple = 0

for case, images in sorted(case_images.items(), key=lambda x: int(x[0])):
    if len(images) > 1:
        multiple += 1
        print(f"Case {case} --> {len(images)} images")

print("\n" + "=" * 60)
print(f"Cases having multiple images : {multiple}")
print("=" * 60)

# ---------------------------------------------------
# Check XML without images
# ---------------------------------------------------

print("\nChecking XML without matching images...")

missing = []

for xml in xml_files:

    case = os.path.splitext(xml)[0]

    if case not in case_images:
        missing.append(case)

if len(missing) == 0:
    print("All XML files have at least one image.")
else:
    print("Missing image cases:")
    print(missing)

print("\nAudit Complete.")