from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
folder = str(PROJECT_ROOT / "02_Dataset_Raw" / "archive")

keywords = [
    "benign",
    "malignant",
    "diagnosis",
    "label",
    "histopathology",
    "bacaf"
]

for xml_file in os.listdir(folder):
    if xml_file.endswith(".xml"):
        path = os.path.join(folder, xml_file)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read().lower()

            for key in keywords:
                if key in text:
                    print(xml_file, "->", key)