import xml.etree.ElementTree as ET

xml_file = r"D:\QCNN_Thyroid_Paper_Reproduction\02_Dataset_Raw\archive\100.xml"

tree = ET.parse(xml_file)
root = tree.getroot()

print("Root Tag:", root.tag)

print("\nAll XML Information:\n")

for elem in root.iter():
    print(elem.tag, ":", elem.text)