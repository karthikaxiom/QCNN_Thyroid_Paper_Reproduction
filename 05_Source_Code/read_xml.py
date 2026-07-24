from pathlib import Path
import xml.etree.ElementTree as ET

PROJECT_ROOT = Path(__file__).resolve().parents[1]
xml_file = str(PROJECT_ROOT / "02_Dataset_Raw" / "archive" / "100.xml")

tree = ET.parse(xml_file)
root = tree.getroot()

print("Root Tag:", root.tag)

print("\nAll XML Information:\n")

for elem in root.iter():
    print(elem.tag, ":", elem.text)