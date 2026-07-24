from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
image_folder = str(PROJECT_ROOT / "02_Dataset_Raw" / "archive")

jpg_count = 0
xml_count = 0

for file in os.listdir(image_folder):

    if file.lower().endswith(".jpg"):
        jpg_count += 1

    elif file.lower().endswith(".xml"):
        xml_count += 1

print("\nDataset Exploration")
print("---------------------")
print("JPG Images :", jpg_count)
print("XML Files  :", xml_count)

print("\nFirst 20 Files:\n")

for file in sorted(os.listdir(image_folder))[:20]:
    print(file)