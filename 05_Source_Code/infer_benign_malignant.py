import pandas as pd
import numpy as np

print("=" * 60)
print("INFER BENIGN / MALIGNANT LABELS")
print("=" * 60)

# Load Dataset

csv_path = r"D:\QCNN_Thyroid_Paper_Reproduction\03_Dataset_Processed\master_dataset.csv"

df = pd.read_csv(csv_path)

print(f"Loaded {len(df)} images")

# TI-RADS Mapping

tirads_map = {
    "2": "Benign",
    "3": "Benign",
    "4a": "Benign",
    "4b": "Malignant",
    "4c": "Malignant",
    "5": "Malignant"
}

df["diagnosis"] = ""
# Assign labels from TI-RADS

for i, row in df.iterrows():

    tirads = str(row["tirads"]).strip().lower()

    if tirads in tirads_map:
        df.at[i, "diagnosis"] = tirads_map[tirads]

# Rule-based inference for missing TI-RADS

for i, row in df[df["diagnosis"].isna()].iterrows():

    composition = str(row["composition"]).lower()
    echo = str(row["echogenicity"]).lower()
    margins = str(row["margins"]).lower()
    calc = str(row["calcifications"]).lower()

    malignant = False

    if "spiculated" in margins:
        malignant = True

    if "microcalcification" in calc:
        malignant = True

    if "hypoechogenicity" in echo:
        malignant = True

    if "marked hypoechogenicity" in echo:
        malignant = True

    if malignant:
        df.at[i, "diagnosis"] = "Malignant"
    else:
        df.at[i, "diagnosis"] = "Benign"

# Save

output = r"D:\QCNN_Thyroid_Paper_Reproduction\03_Dataset_Processed\final_dataset.csv"

df.to_csv(output, index=False)

# Summary

print("\nFINAL DATASET SUMMARY")
print("-" * 40)

print(df["diagnosis"].value_counts())

print("\nSaved to")

print(output)

print("\nPHASE 4 COMPLETE")