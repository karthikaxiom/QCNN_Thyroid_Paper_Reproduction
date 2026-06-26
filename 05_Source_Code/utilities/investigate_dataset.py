import os
import xml.etree.ElementTree as ET

folder = r"D:\QCNN_Thyroid_Paper_Reproduction\02_Dataset_Raw\archive"

bacaf_values = set()
eco_values = set()

for file in os.listdir(folder):
    if file.endswith(".xml"):
        root = ET.parse(os.path.join(folder, file)).getroot()

        for elem in root.iter():

            if elem.tag == "reportbacaf":
                text = (elem.text or "").strip()
                if text != "":
                    bacaf_values.add(text)

            if elem.tag == "reporteco":
                text = (elem.text or "").strip()
                if text != "":
                    eco_values.add(text)

print("\n==============================")
print("reportbacaf values")
print("==============================")
for v in sorted(bacaf_values):
    print(v)

print("\n==============================")
print("reporteco values")
print("==============================")
for v in sorted(eco_values):
    print(v)