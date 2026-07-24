from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
dataset_path = str(PROJECT_ROOT / "02_Dataset_Raw" / "archive")

jpg_count = 0
xml_count = 0

for file in os.listdir(dataset_path):

    if file.endswith(".jpg"):
        jpg_count += 1

    elif file.endswith(".xml"):
        xml_count += 1

print("Dataset Inspection Report")
print("-" * 30)
print("JPG Files :", jpg_count)
print("XML Files :", xml_count)
print("Total Files :", jpg_count + xml_count)