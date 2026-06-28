import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

print("=" * 60)
print("PHASE 7 : REPRODUCIBILITY AUDIT")
print("=" * 60)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODEL_FOLDER = os.path.join(PROJECT_ROOT, "06_Models")
DATA_FOLDER = os.path.join(PROJECT_ROOT, "03_Dataset_Processed")
RESULT_FOLDER = os.path.join(PROJECT_ROOT, "07_Results")

os.makedirs(RESULT_FOLDER, exist_ok=True)

files = {
    "X_train": "X_train.npy",
    "X_test": "X_test.npy",
    "X_train_smote": "X_train_smote.npy",
    "X_train_quantum": "X_train_quantum.npy",
    "X_test_quantum": "X_test_quantum.npy",
    "y_train": "y_train.npy",
    "y_train_smote": "y_train_smote.npy",
    "y_test": "y_test.npy",
}

report = []

def add(line=""):
    print(line)
    report.append(line)

add("\nFILE SHAPE CHECK")
add("-" * 40)

for name, file in files.items():
    path = os.path.join(MODEL_FOLDER, file)

    if os.path.exists(path):
        arr = np.load(path)
        add(f"{name:18}: {arr.shape}")
    else:
        add(f"{name:18}: MISSING")

add("\nLABEL DISTRIBUTION")
add("-" * 40)

for label_file in ["y_train.npy", "y_train_smote.npy", "y_test.npy"]:
    path = os.path.join(MODEL_FOLDER, label_file)

    if os.path.exists(path):
        y = np.load(path)
        unique, counts = np.unique(y, return_counts=True)
        add(label_file)

        for u, c in zip(unique, counts):
            cls = "Benign" if u == 0 else "Malignant"
            add(f"  {cls:10}: {c}")

add("\nQUANTUM FEATURE STATISTICS")
add("-" * 40)

for qfile in ["X_train_quantum.npy", "X_test_quantum.npy"]:
    path = os.path.join(MODEL_FOLDER, qfile)

    if os.path.exists(path):
        X = np.load(path)
        add(qfile)
        add(f"  Min  : {X.min():.6f}")
        add(f"  Max  : {X.max():.6f}")
        add(f"  Mean : {X.mean():.6f}")
        add(f"  Std  : {X.std():.6f}")
        add(f"  NaN  : {np.isnan(X).sum()}")
        add(f"  Inf  : {np.isinf(X).sum()}")

add("\nFINAL DATASET CHECK")
add("-" * 40)

csv_path = os.path.join(DATA_FOLDER, "final_dataset.csv")

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    add(f"final_dataset.csv rows: {len(df)}")

    if "diagnosis" in df.columns:
        add(str(df["diagnosis"].value_counts(dropna=False)))

    if "tirads" in df.columns:
        add("\nTI-RADS distribution:")
        add(str(df["tirads"].value_counts(dropna=False)))

report_path = os.path.join(RESULT_FOLDER, "phase7_reproducibility_audit.txt")

with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print("\nSaved audit report:")
print(report_path)

print("=" * 60)
print("PHASE 7 AUDIT COMPLETE")
print("=" * 60)