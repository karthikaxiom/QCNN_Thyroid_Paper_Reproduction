import os

folder = r"D:\QCNN_Thyroid_Paper_Reproduction\02_Dataset_Raw\archive"

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