import os

dataset_path = r"D:\QCNN_Thyroid_Paper_Reproduction\02_Dataset_Raw\archive"

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