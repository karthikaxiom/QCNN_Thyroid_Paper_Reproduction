import os
import numpy as np
from imblearn.over_sampling import SMOTE

print("=" * 60)
print("PHASE 5 : STEP 2 - APPLYING SMOTE")
print("=" * 60)

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

MODEL_FOLDER = os.path.join(
    PROJECT_ROOT,
    "06_Models"
)

# Load Training Data

X_train = np.load(os.path.join(MODEL_FOLDER, "X_train.npy"))
y_train = np.load(os.path.join(MODEL_FOLDER, "y_train.npy"))

print("\nOriginal Training Shape")
print("----------------------------")
print("Images :", X_train.shape)
print("Labels :", y_train.shape)

print("\nOriginal Class Distribution")
unique, counts = np.unique(y_train, return_counts=True)

for u, c in zip(unique, counts):
    label = "Benign" if u == 0 else "Malignant"
    print(f"{label:12}: {c}")

# Flatten Images

samples = X_train.shape[0]

X_flat = X_train.reshape(samples, -1)

print("\nFlattened Shape :", X_flat.shape)

# Apply SMOTE

print("\nApplying SMOTE...")

smote = SMOTE(
    random_state=42
)

X_smote, y_smote = smote.fit_resample(
    X_flat,
    y_train
)

# Restore Image Shape

X_smote = X_smote.reshape(
    -1,
    128,
    128,
    1
)

print("\nBalanced Dataset")

print("----------------------------")

print("Images :", X_smote.shape)

print("Labels :", y_smote.shape)

print("\nBalanced Class Distribution")

unique, counts = np.unique(y_smote, return_counts=True)

for u, c in zip(unique, counts):
    label = "Benign" if u == 0 else "Malignant"
    print(f"{label:12}: {c}")

# Save

np.save(
    os.path.join(MODEL_FOLDER, "X_train_smote.npy"),
    X_smote
)

np.save(
    os.path.join(MODEL_FOLDER, "y_train_smote.npy"),
    y_smote
)

print("\nSaved Successfully")

print("X_train_smote.npy")

print("y_train_smote.npy")

print("=" * 60)
print("PHASE 5 STEP 2 COMPLETE")
print("=" * 60)